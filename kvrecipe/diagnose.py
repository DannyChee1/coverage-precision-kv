import argparse
import json
from pathlib import Path
import re
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).parent))
import mix

ROWS = ('stack4', 'prec16', 'cov2x')


def answer_tokens(tokenizer, context, answers, n_tokens):
    separator = '#' * (len(context) + 10)
    if tokenizer.chat_template is None:
        text = (tokenizer.bos_token or '') + context
    else:
        text = tokenizer.apply_chat_template([{'role': 'user', 'content': context + separator}],
                                             add_generation_prompt=True, tokenize=False,
                                             enable_thinking=False).split(separator)[0]
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    if len(enc['input_ids']) != n_tokens:
        return None
    offsets = enc['offset_mapping']
    positions = set()
    for gold in answers:
        for m in re.finditer(re.escape(gold.lower()), text.lower()):
            positions.update(i for i, (a, b) in enumerate(offsets) if a < m.end() and b > m.start())
    return sorted(positions)


def diagnose(pipe, row, capture_cls):
    from transformers import DynamicCache
    model = pipe.model
    inputs = pipe.preprocess(row['context'], [row['question']], row['answer_prefix'],
                             max_context_length=pipe.tokenizer.model_max_length)
    context_ids = inputs['context_ids'].to(model.device)
    question_ids = inputs['questions_ids'][0].to(model.device)
    n_tokens = context_ids.shape[1]
    cache = DynamicCache()
    press = capture_cls(compression_ratio=0.5, kvzip_plus_normalization=True, n_sink=mix.N_SINK)
    t0 = time.time()
    with torch.no_grad():
        with press(model):
            model.model(input_ids=context_ids, past_key_values=cache)
    scores = press.captured[:, 0]
    n_layers, n_heads = len(cache.layers), cache.layers[0].keys.shape[1]
    head_dim = cache.layers[0].keys.shape[-1]
    quantized, errors, kerr, verr = [], [], {2: [], 4: []}, {2: [], 4: []}
    for i in range(n_layers):
        keys, values = cache.layers[i].keys[0], cache.layers[i].values[0]
        q = {b: mix.quantize_entries(keys, values, b) for b in (2, 4)}
        quantized.append(q)
        e = {}
        for b in (2, 4):
            ke, ve = mix.relative_error(keys, q[b][0]), mix.relative_error(values, q[b][1])
            kerr[b].append(ke)
            verr[b].append(ve)
            e[b] = (ke + ve) / 2
        errors.append(e)
    plans = {'stack4': [mix.allocate_stack(scores[i], 4, head_dim) for i in range(n_layers)]}
    plans['prec16'] = [torch.where(p == 4, torch.full_like(p, 16), p) for p in plans['stack4']]
    plans['cov2x'] = [mix.allocate_stack(scores[i], 4, head_dim, scale=2.0) for i in range(n_layers)]
    joint = [mix.allocate_joint(scores[i], errors[i], head_dim) for i in range(n_layers)]
    full_bits = n_layers * n_heads * n_tokens * mix.entry_bits(16, head_dim)
    out = {'n_tokens': n_tokens, 'score_seconds': time.time() - t0,
           'share_of_full_bytes': {r: sum(mix.plan_bits(p, head_dim) for p in plans[r]) / full_bits for r in ROWS}}
    s4 = torch.stack(plans['stack4'])[..., mix.N_SINK:] == 4
    jt = torch.stack(joint)[..., mix.N_SINK:]
    out['joint_on_stack4_entries'] = {str(b): float((jt[s4] == b).float().mean()) for b in mix.JOINT_BITS}
    out['joint_tiers'] = {str(b): int((jt == b).sum()) for b in mix.JOINT_BITS}
    out['quant_error'] = {f'{part}{b}': float(torch.stack(src[b]).mean())
                          for part, src in (('key', kerr), ('value', verr)) for b in (2, 4)}
    kept4 = torch.stack(kerr[4])[..., mix.N_SINK:][s4]
    out['quant_error_on_stack4_entries'] = {'key4': float(kept4.mean()),
                                            'value4': float(torch.stack(verr[4])[..., mix.N_SINK:][s4].mean())}
    positions = answer_tokens(pipe.tokenizer, row['context'], list(row['answer']), n_tokens)
    if positions:
        pos = torch.tensor(positions, device=scores.device)
        ranks = scores.argsort(-1).argsort(-1).float() / (n_tokens - 1)  # 1.0 = highest score in its head
        out['answer_tokens'] = {'tokens': len(positions), 'rank_percentile': float(ranks[..., pos].mean()),
                         **{f'kept_{r}': float((torch.stack(plans[r])[..., pos] > 0).float().mean()) for r in ROWS}}
    else:
        out['answer_tokens'] = {'tokens': 0 if positions == [] else None}
    arm_cache = mix.batched_cache(cache, len(ROWS))
    del cache
    masks = [[] for _ in range(n_layers)]
    for r, name in enumerate(ROWS):
        mix.apply_plan(arm_cache, r, plans[name], quantized, masks)
    del quantized
    mix.set_masks(model, masks)
    t1 = time.time()
    try:
        with torch.no_grad():
            answers = mix.batched_generate(model, pipe.tokenizer, question_ids, arm_cache, n_tokens,
                                             int(row['max_new_tokens']))
    finally:
        mix.clear_masks(model)
    out['decode_seconds'] = time.time() - t1
    out['answers'] = dict(zip(ROWS, answers))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--indices', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--dtype', default='bfloat16')
    parser.add_argument('--max-new-tokens', type=int, default=0, help='CPU tests only')
    args = parser.parse_args()
    import kvpress
    from transformers import pipeline
    attn = 'flash_attention_2' if args.device.startswith('cuda') else 'sdpa'
    pipe = pipeline('kv-press-text-generation', model=args.model_path, device=args.device,
                    model_kwargs={'dtype': getattr(torch, args.dtype), 'attn_implementation': attn})
    wanted = {int(i) for i in args.indices.split(',')}
    rows = [r for r in mix.load_rows(50, 0) if r['index'] in wanted]
    done = {json.loads(line)['index'] for line in args.out.open()} if args.out.exists() else set()
    capture_cls = mix.capture_press_class()
    with args.out.open('a') as handle:
        for row in rows:
            if row['index'] in done:
                continue
            if args.max_new_tokens:
                row['max_new_tokens'] = min(int(row['max_new_tokens']), args.max_new_tokens)
            result = diagnose(pipe, row, capture_cls)
            handle.write(json.dumps({'index': row['index'], 'task': row['task'], 'answer': list(row['answer']),
                                     **result}) + '\n')
            handle.flush()
            print(json.dumps({'done': row['index'], 'task': row['task'],
                              'seconds': round(result['score_seconds'] + result['decode_seconds'], 2)}), flush=True)


if __name__ == '__main__':
    sys.exit(main())
