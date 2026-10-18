import argparse
import json
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).parent))
import mix
from mix import N_SINK, fake_quant, layer_budget_bits
from confirm import load_rows
from improve import centered_rotated, part_bits, rel, rotation
from stage2 import KEPT, SINK, quantize_parts

BUDGET = 0.015625
TOKENS = 64  # kept tokens per per-channel group
COUNTS = 32 * 2  # a 32-bit count for each of the 2 stored tiers (sinks, kept
SPEC = {'k4v2': (4, 2, 'separable', 'k4v2'), 'stack4_pc': (4, 4, 'perchannel', 'stack4_pc'),
        'k4v2_pc': (4, 2, 'perchannel', 'k4v2_pc'), 'k3_plain': (3, 2, 'separable', 'k3'),
        'k3_centered': (3, 2, 'centered', 'k3'), 'k3_rotated': (3, 2, 'rotated', 'k3'),
        'rk3v2': (3, 2, 'centered_rotated', 'k3'), 'k3_pc': (3, 2, 'perchannel', 'k3')}
ARMS = ('full', *SPEC)


def fixed_overhead(quantizer, head_dim):
    scales = head_dim * 16 if quantizer in ('separable', 'centered') else 0
    means = head_dim * 16 if quantizer in ('centered', 'centered_rotated') else 0
    return scales + means + COUNTS


def reserved_overhead(quantizer, head_dim):
    pad = (TOKENS - 1) * head_dim * 2 * 16 // TOKENS if quantizer == 'perchannel' else 0
    return fixed_overhead(quantizer, head_dim) + pad


def entry_bits(kbits, vbits, head_dim):
    return part_bits(kbits, head_dim) + part_bits(vbits, head_dim)


def plan_overhead(plan_name, head_dim):
    return max(reserved_overhead(q, head_dim) for _, _, q, p in SPEC.values() if p == plan_name)


def allocate(scores, ebits, overhead, head_dim, budget=BUDGET):
    n_heads, n_tokens = scores.shape
    plan = torch.zeros_like(scores, dtype=torch.int16)
    plan[:, :N_SINK] = SINK
    left = (layer_budget_bits(n_heads, n_tokens, head_dim, budget)
            - n_heads * min(N_SINK, n_tokens) * 2 * part_bits(16, head_dim) - n_heads * overhead)
    flat = scores[:, N_SINK:].reshape(-1)
    k = max(0, min(flat.numel(), int(left // ebits)))
    body = plan[:, N_SINK:].reshape(-1)
    body[torch.topk(flat, k).indices] = KEPT
    plan[:, N_SINK:] = body.view(n_heads, n_tokens - N_SINK)
    return plan


def padding_bits(plan, head_dim):
    kept = (plan == KEPT).sum(dim=1)
    return int(((-kept) % TOKENS).sum().item()) * head_dim * 2 * 16 // TOKENS


def plan_bits(plan, arm, head_dim):
    kbits, vbits, quantizer, _ = SPEC[arm]
    kept, sinks = int((plan == KEPT).sum()), int((plan == SINK).sum())
    pad = padding_bits(plan, head_dim) if quantizer == 'perchannel' else 0
    return (kept * entry_bits(kbits, vbits, head_dim) + sinks * 2 * part_bits(16, head_dim)
            + plan.shape[0] * fixed_overhead(quantizer, head_dim) + pad)


def per_channel(x, bits, tokens=TOKENS):
    n, d = x.shape
    if n == 0:
        return x
    pad = (-n) % tokens
    g = torch.cat([x, x[-1:].expand(pad, d)], 0) if pad else x
    g = g.reshape(-1, tokens, d)
    lo, hi = g.amin(1, keepdim=True), g.amax(1, keepdim=True)
    scale = ((hi - lo) / (2 ** bits - 1)).clamp(min=1e-8)
    q = ((g - lo) / scale).round().clamp(0, 2 ** bits - 1) * scale + lo
    return q.reshape(-1, d)[:n]


def quantized_keys(keys, quantizer, bits, rot, kept=None):
    k = keys.float()
    if quantizer == 'separable':
        out = quantize_parts(keys, keys, (bits,), (bits,))[0][bits].float()
    elif quantizer == 'centered':
        mean = k.mean(dim=1, keepdim=True)
        out = quantize_parts(k - mean, k - mean, (bits,), (bits,))[0][bits].float() + mean
    elif quantizer == 'rotated':
        out = fake_quant(k @ rot.T, bits) @ rot
    elif quantizer == 'centered_rotated':
        out = centered_rotated(keys, bits, rot).float()
    else:
        out = k.clone()
        for h in range(k.shape[0]):
            idx = torch.nonzero(kept[h], as_tuple=True)[0]
            out[h, idx] = per_channel(k[h, idx], bits)
    return out.to(keys.dtype)


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
    rot = rotation(head_dim, scores.device)
    plan_names = dict.fromkeys(p for *_, p in SPEC.values())
    plan_cost = {p: entry_bits(*SPEC[next(a for a in SPEC if SPEC[a][3] == p)][:2], head_dim) for p in plan_names}
    plans = {p: [allocate(scores[i], plan_cost[p], plan_overhead(p, head_dim), head_dim) for i in range(n_layers)]
             for p in plan_names}
    layer_budget = layer_budget_bits(n_heads, n_tokens, head_dim, BUDGET)
    out = {'n_tokens': n_tokens, 'n_layers': n_layers, 'n_kv_heads': n_heads, 'score_seconds': score_seconds,
           'bits': {}, 'budget_bits': {}, 'over_budget_layers': {}, 'kept': {}, 'key_error': {}, 'value_error': {},
           'full_bits': n_layers * n_heads * n_tokens * 2 * part_bits(16, head_dim)}
    for arm, (_, _, _, p) in SPEC.items():
        layer_bits = [plan_bits(plan, arm, head_dim) for plan in plans[p]]
        out['bits'][arm] = sum(layer_bits)
        out['budget_bits'][arm] = n_layers * layer_budget
        out['over_budget_layers'][arm] = sum(b > layer_budget for b in layer_bits)
        out['kept'][arm] = int(sum((plan == KEPT).sum().item() for plan in plans[p]))
    arm_cache = mix.batched_cache(cache, len(ARMS))
    masks = [[] for _ in range(n_layers)]
    errors = {arm: ([], []) for arm in SPEC}
    for layer_idx in range(n_layers):
        keys, values = cache.layers[layer_idx].keys[0], cache.layers[layer_idx].values[0]
        vq = quantize_parts(keys, values, (4,), (2, 4))[1]
        shared = {}
        for row_idx, arm in enumerate(ARMS):
            if arm == 'full':
                continue
            kbits, vbits, quantizer, p = SPEC[arm]
            plan = plans[p][layer_idx]
            sel = plan == KEPT
            key = (quantizer, kbits, p if quantizer == 'perchannel' else None)
            if key not in shared:
                shared[key] = quantized_keys(keys, quantizer, kbits, rot, sel)
            kq = shared[key]
            arm_cache.layers[layer_idx].keys[row_idx][sel] = kq[sel]
            arm_cache.layers[layer_idx].values[row_idx][sel] = vq[vbits][sel]
            errors[arm][0].append(rel(kq[sel], keys[sel]))
            errors[arm][1].append(rel(vq[vbits][sel], values[sel]))
            heads, seqs = torch.nonzero(plan == 0, as_tuple=True)
            if heads.numel():
                masks[layer_idx].append((torch.full_like(heads, row_idx), heads, seqs))
    for arm, (ke, ve) in errors.items():
        out['key_error'][arm], out['value_error'][arm] = sum(ke) / len(ke), sum(ve) / len(ve)
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
    args = parser.parse_args()
    import kvpress
    from transformers import pipeline
    attn = 'flash_attention_2' if args.device.startswith('cuda') else 'sdpa'
    pipe = pipeline('kv-press-text-generation', model=args.model_path, device=args.device,
                    model_kwargs={'dtype': getattr(torch, args.dtype), 'attn_implementation': attn})
    rows = load_rows(args.rows, args.per_task, args.offset)
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
