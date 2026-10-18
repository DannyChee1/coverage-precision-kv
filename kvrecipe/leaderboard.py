import argparse
import json
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).parent))
import mix
from mix import N_SINK

RATIOS = {'kvzip_plus@0.9375': 0.9375, 'kvzip_plus@0.96875': 0.96875}
ARMS = ('full', *RATIOS)


def capture_native_class():
    from kvpress.presses.kvzip_press import KVzipPress

    class CaptureNative(KVzipPress):

        def compress_post(self, model):
            self.captured = self.score_val.clone()  # L, 1, H, T

    return CaptureNative


def kvpress_evicted(score_val, ratio, n_sink=N_SINK):
    scores = score_val.clone()
    scores[..., :n_sink] = scores.amax() + 1.0
    n_layer, bsz, heads, ctx_len = scores.shape
    pruned = torch.topk(-scores.reshape(-1), int(scores.numel() * ratio)).indices
    per_layer = torch.bincount(pruned // (bsz * heads * ctx_len), minlength=n_layer).int()
    out = []
    for layer in range(n_layer):
        indices = torch.topk(-scores[layer].reshape(bsz, -1), int(per_layer[layer]), dim=1).indices.flatten()
        out.append((indices // ctx_len, indices % ctx_len))
    return out


def run_example(pipe, row, capture_cls):
    from transformers import DynamicCache
    model = pipe.model
    inputs = pipe.preprocess(row['context'], [row['question']], row['answer_prefix'],
                             max_context_length=pipe.tokenizer.model_max_length)
    context_ids = inputs['context_ids'].to(model.device)
    question_ids = inputs['questions_ids'][0].to(model.device)
    n_tokens = context_ids.shape[1]
    cache = DynamicCache()
    press = capture_cls(compression_ratio=0.5, kvzip_plus_normalization=True, n_sink=N_SINK)
    t0 = time.time()
    with torch.no_grad():
        with press(model):
            model.model(input_ids=context_ids, past_key_values=cache)
    score_seconds = time.time() - t0
    n_layers = len(cache.layers)
    arm_cache = mix.batched_cache(cache, len(ARMS))
    masks = [[] for _ in range(n_layers)]
    evicted = {}
    for row_idx, arm in enumerate(ARMS):
        if arm == 'full':
            continue
        per_layer = kvpress_evicted(press.captured, RATIOS[arm])
        evicted[arm] = sum(int(h.numel()) for h, _ in per_layer)
        for layer, (heads, seqs) in enumerate(per_layer):
            if heads.numel():
                masks[layer].append((torch.full_like(heads, row_idx), heads, seqs))
    total = n_layers * cache.layers[0].keys.shape[1] * n_tokens
    del cache
    mix.set_masks(model, masks)
    t1 = time.time()
    try:
        with torch.no_grad():
            answers = mix.batched_generate(model, pipe.tokenizer, question_ids, arm_cache, n_tokens,
                                             int(row['max_new_tokens']))
    finally:
        mix.clear_masks(model)
    return {'n_tokens': n_tokens, 'score_seconds': score_seconds, 'decode_seconds': time.time() - t1,
            'evicted_share': {a: evicted[a] / total for a in evicted}, 'answers': dict(zip(ARMS, answers))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--per-task', type=int, default=50)
    parser.add_argument('--offset', type=int, default=0)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--dtype', default='bfloat16')
    parser.add_argument('--max-new-tokens', type=int, default=0, help='CPU tests only')
    parser.add_argument('--max-examples', type=int, default=0, help='CPU tests only')
    args = parser.parse_args()
    import kvpress
    from transformers import pipeline
    attn = 'flash_attention_2' if args.device.startswith('cuda') else 'sdpa'
    pipe = pipeline('kv-press-text-generation', model=args.model_path, device=args.device,
                    model_kwargs={'dtype': getattr(torch, args.dtype), 'attn_implementation': attn})
    rows = mix.load_rows(args.per_task, args.offset)
    if args.max_examples:
        rows = rows[::max(1, len(rows) // args.max_examples)][:args.max_examples]
    done = {json.loads(line)['index'] for line in args.out.open()} if args.out.exists() else set()
    capture_cls = capture_native_class()
    with args.out.open('a') as handle:
        for row in rows:
            if row['index'] in done:
                continue
            if args.max_new_tokens:
                row['max_new_tokens'] = min(int(row['max_new_tokens']), args.max_new_tokens)
            result = run_example(pipe, row, capture_cls)
            handle.write(json.dumps({'index': row['index'], 'task': row['task'], 'answer': list(row['answer']),
                                     'model': pipe.model.config.name_or_path, 'dtype': str(pipe.model.dtype),
                                     'attn_implementation': pipe.model.config._attn_implementation,
                                     **result}) + '\n')
            handle.flush()
            print(json.dumps({'done': row['index'], 'task': row['task'],
                              'seconds': round(result['score_seconds'] + result['decode_seconds'], 2)}), flush=True)


if __name__ == '__main__':
    sys.exit(main())
