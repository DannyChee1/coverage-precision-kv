import argparse
import json
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).parent))
import mix
from mix import N_SINK, layer_budget_bits
from stage2 import KEPT, SINK, allocate_uniform, entry_bits_kv, plan_bits_kv, quantize_parts

ARMS = ('full', 'stack4', 'k4v2', 'tiered')
UNIFORM = {'stack4': (4, 4), 'k4v2': (4, 2)}  # key bits, value bits
TOP = 2  # plan code of the tiered arm's upper tier; KEPT (1) is its lower tier
TIERS = {TOP: (8, 4), KEPT: (4, 2)}
TOP_SHARE = 0.2  # share of the tiered arm's kept entries in the upper tier, fixed in advance


def overhead_bits(n_heads, head_dim, stored_tiers):
    return n_heads * (head_dim * 16 + 32 * stored_tiers)


def allocate_tiered(scores, head_dim):
    n_heads, n_tokens = scores.shape
    plan = torch.zeros_like(scores, dtype=torch.int16)
    plan[:, :N_SINK] = SINK
    top_cost, low_cost = entry_bits_kv(*TIERS[TOP], head_dim), entry_bits_kv(*TIERS[KEPT], head_dim)
    left = (layer_budget_bits(n_heads, n_tokens, head_dim)
            - n_heads * min(N_SINK, n_tokens) * entry_bits_kv(16, 16, head_dim)
            - overhead_bits(n_heads, head_dim, 3))
    flat = scores[:, N_SINK:].reshape(-1)
    kept = max(0, int(left // (TOP_SHARE * top_cost + (1 - TOP_SHARE) * low_cost)))
    n_top = min(flat.numel(), round(TOP_SHARE * kept))
    n_low = max(0, min(flat.numel() - n_top, int((left - n_top * top_cost) // low_cost)))
    order = torch.topk(flat, n_top + n_low).indices  # sorted, highest first
    body = plan[:, N_SINK:].reshape(-1)
    body[order[:n_top]] = TOP
    body[order[n_top:]] = KEPT
    plan[:, N_SINK:] = body.view(n_heads, n_tokens - N_SINK)
    return plan


def plan_bits_tiered(plan, head_dim):
    n_heads = plan.shape[0]
    counts = {code: int((plan == code).sum().item()) for code in (TOP, KEPT, SINK)}
    return (counts[TOP] * entry_bits_kv(*TIERS[TOP], head_dim) + counts[KEPT] * entry_bits_kv(*TIERS[KEPT], head_dim)
            + counts[SINK] * entry_bits_kv(16, 16, head_dim) + overhead_bits(n_heads, head_dim, 3))


def plans_for(arm, scores, head_dim):
    if arm == 'tiered':
        return [allocate_tiered(s, head_dim) for s in scores]
    return [allocate_uniform(s, *UNIFORM[arm], head_dim) for s in scores]


def bits_for(arm, plan, head_dim):
    return plan_bits_tiered(plan, head_dim) if arm == 'tiered' else plan_bits_kv(plan, *UNIFORM[arm], head_dim)


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
    compressed = [arm for arm in ARMS if arm != 'full']
    plans = {arm: plans_for(arm, scores, head_dim) for arm in compressed}
    layer_budget = layer_budget_bits(n_heads, n_tokens, head_dim)
    out = {'n_tokens': n_tokens, 'n_layers': n_layers, 'n_kv_heads': n_heads, 'score_seconds': score_seconds,
           'bits': {}, 'over_budget_layers': {}, 'kept': {}, 'budget_bits': n_layers * layer_budget,
           'full_bits': n_layers * n_heads * n_tokens * entry_bits_kv(16, 16, head_dim)}
    for arm in compressed:
        layer_bits = [bits_for(arm, p, head_dim) for p in plans[arm]]
        out['bits'][arm] = sum(layer_bits)
        out['over_budget_layers'][arm] = sum(b > layer_budget for b in layer_bits)
        out['kept'][arm] = int(sum(((p == KEPT) | (p == TOP)).sum().item() for p in plans[arm]))
    out['tiered_top'] = int(sum((p == TOP).sum().item() for p in plans['tiered']))
    arm_cache = mix.batched_cache(cache, len(ARMS))
    masks = [[] for _ in range(n_layers)]
    for layer_idx in range(n_layers):
        keys, values = cache.layers[layer_idx].keys[0], cache.layers[layer_idx].values[0]
        kq, vq = quantize_parts(keys, values, (4, 8), (2, 4))
        for row_idx, arm in enumerate(ARMS):
            if arm == 'full':
                continue
            plan = plans[arm][layer_idx]
            widths = {TOP: TIERS[TOP], KEPT: TIERS[KEPT]} if arm == 'tiered' else {KEPT: UNIFORM[arm]}
            for code, (kb, vb) in widths.items():
                sel = plan == code
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


def load_rows(path, per_task, offset):
    rows = [json.loads(line) for line in open(path)]
    out = []
    for task in sorted({r['task'] for r in rows}):
        out.extend([r for r in rows if r['task'] == task][offset:offset + per_task])
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
