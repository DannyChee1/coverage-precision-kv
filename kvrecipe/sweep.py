import argparse
import json
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).parent))
import mix
from mix import N_SINK, layer_budget_bits
from confirm import load_rows
from stage2 import KEPT, allocate_uniform, entry_bits_kv, plan_bits_kv, quantize_parts

BUDGETS = {'1.5625': 0.015625, '3.125': 0.03125, '6.25': 0.0625, '12.5': 0.125}  # share of the full FP16 bytes
SPLITS = {'stack4': (4, 4), 'k4v2': (4, 2)}  # key bits, value bits
ARMS = ('full', *(f'{s}@{b}' for b in BUDGETS for s in SPLITS))


def arm_spec(arm):
    split, budget = arm.split('@')
    return SPLITS[split], BUDGETS[budget]


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
    scores = press.captured[:, 0]  # L, H, T
    score_seconds = time.time() - t0
    n_layers, n_heads, head_dim = len(cache.layers), cache.layers[0].keys.shape[1], cache.layers[0].keys.shape[-1]
    compressed = ARMS[1:]
    plans = {}
    for arm in compressed:
        (kb, vb), budget = arm_spec(arm)
        plans[arm] = [allocate_uniform(scores[i], kb, vb, head_dim, budget) for i in range(n_layers)]
    out = {'n_tokens': n_tokens, 'n_layers': n_layers, 'n_kv_heads': n_heads, 'score_seconds': score_seconds,
           'bits': {}, 'budget_bits': {}, 'over_budget_layers': {}, 'kept': {},
           'full_bits': n_layers * n_heads * n_tokens * entry_bits_kv(16, 16, head_dim)}
    for arm in compressed:
        (kb, vb), budget = arm_spec(arm)
        layer_budget = layer_budget_bits(n_heads, n_tokens, head_dim, budget)
        layer_bits = [plan_bits_kv(p, kb, vb, head_dim) for p in plans[arm]]
        out['bits'][arm] = sum(layer_bits)
        out['budget_bits'][arm] = n_layers * layer_budget
        out['over_budget_layers'][arm] = sum(b > layer_budget for b in layer_bits)
        out['kept'][arm] = int(sum((p == KEPT).sum().item() for p in plans[arm]))
    arm_cache = mix.batched_cache(cache, len(ARMS))
    masks = [[] for _ in range(n_layers)]
    for layer_idx in range(n_layers):
        keys, values = cache.layers[layer_idx].keys[0], cache.layers[layer_idx].values[0]
        kq, vq = quantize_parts(keys, values, (4,), (2, 4))
        for row_idx, arm in enumerate(ARMS):
            if arm == 'full':
                continue
            (kb, vb), _ = arm_spec(arm)
            plan = plans[arm][layer_idx]
            sel = plan == KEPT
            arm_cache.layers[layer_idx].keys[row_idx][sel] = kq[kb][sel]
            arm_cache.layers[layer_idx].values[row_idx][sel] = vq[vb][sel]
            heads, seqs = torch.nonzero(plan == 0, as_tuple=True)
            if heads.numel():
                masks[layer_idx].append((torch.full_like(heads, row_idx), heads, seqs))
    del cache
    mix.set_masks(model, masks)
    t1 = time.time()
    try:
        with torch.no_grad():
            answers = mix.batched_generate(model, pipe.tokenizer, question_ids, arm_cache, n_tokens,
                                             int(row['max_new_tokens']))
    finally:
        mix.clear_masks(model)
    out['decode_seconds'] = time.time() - t1
    out['answers'] = dict(zip(ARMS, answers))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--rows', type=Path, required=True)
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
    rows = load_rows(args.rows, args.per_task, args.offset)
    if args.max_examples:
        rows = rows[::max(1, len(rows) // args.max_examples)][:args.max_examples]
    done = {json.loads(line)['index'] for line in args.out.open()} if args.out.exists() else set()
    capture_cls = mix.capture_press_class()
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
