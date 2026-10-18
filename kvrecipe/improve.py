import argparse
import json
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).parent))
import mix
from mix import GROUP, META, N_SINK, fake_quant, layer_budget_bits
from confirm import load_rows
from stage2 import KEPT, SINK, quantize_parts

BUDGETS = {'1.5625': 0.015625, '3.125': 0.03125}
SPLITS = ('k4v2', 'k4v1', 'rk3v2')
ARMS = ('full', *(f'{s}@{b}' for b in BUDGETS for s in SPLITS))


def part_bits(bits, head_dim):
    return head_dim * 16 if bits == 16 else head_dim * bits + (head_dim // GROUP) * META


def entry_bits(split, head_dim):
    kb, vb = {'k4v2': (4, 2), 'k4v1': (4, 1), 'rk3v2': (3, 2)}[split]
    return part_bits(kb, head_dim) + part_bits(vb, head_dim)


def head_overhead_bits(split, head_dim):
    return head_dim * 16 * (2 if split == 'k4v1' else 1) + 32 * 2


def rotation(head_dim, device, seed=0):
    h = torch.ones(1, 1)
    while h.shape[0] < head_dim:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    signs = torch.where(torch.rand(head_dim, generator=torch.Generator().manual_seed(seed)) < 0.5, -1.0, 1.0)
    return (h / head_dim ** 0.5 * signs[None, :]).to(device)


def one_bit(x):
    g = x.reshape(*x.shape[:-1], -1, GROUP)
    mu = g.mean(-1, keepdim=True)
    s = (g - mu).abs().mean(-1, keepdim=True)
    return (mu + s * torch.where(g >= mu, 1.0, -1.0)).reshape(x.shape)


def centered_rotated(x, bits, rot):
    x = x.float()
    mean = x.mean(dim=1, keepdim=True)  # H, 1, D), stored at FP16
    r = (x - mean) @ rot.T
    q = one_bit(r) if bits == 1 else fake_quant(r, bits)
    return q @ rot + mean


def allocate(scores, split, head_dim, budget):
    n_heads, n_tokens = scores.shape
    plan = torch.zeros_like(scores, dtype=torch.int16)
    plan[:, :N_SINK] = SINK
    left = (layer_budget_bits(n_heads, n_tokens, head_dim, budget)
            - n_heads * min(N_SINK, n_tokens) * 2 * part_bits(16, head_dim)
            - n_heads * head_overhead_bits(split, head_dim))
    flat = scores[:, N_SINK:].reshape(-1)
    k = max(0, min(flat.numel(), int(left // entry_bits(split, head_dim))))
    body = plan[:, N_SINK:].reshape(-1)
    body[torch.topk(flat, k).indices] = KEPT
    plan[:, N_SINK:] = body.view(n_heads, n_tokens - N_SINK)
    return plan


def plan_bits(plan, split, head_dim):
    kept, sinks = int((plan == KEPT).sum()), int((plan == SINK).sum())
    return (kept * entry_bits(split, head_dim) + sinks * 2 * part_bits(16, head_dim)
            + plan.shape[0] * head_overhead_bits(split, head_dim))


def quantized_layer(keys, values, rot):
    kq, vq = quantize_parts(keys, values, (4,), (2,))
    k4, v2 = kq[4], vq[2]
    return {'k4v2': (k4, v2),
            'k4v1': (k4, centered_rotated(values, 1, rot).to(values.dtype)),
            'rk3v2': (centered_rotated(keys, 3, rot).to(keys.dtype), v2)}


def rel(xq, x):
    return float(((xq.float() - x.float()).norm(dim=-1) / x.float().norm(dim=-1).clamp(min=1e-9)).mean())


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
    compressed = ARMS[1:]
    plans = {arm: [allocate(scores[i], arm.split('@')[0], head_dim, BUDGETS[arm.split('@')[1]])
                   for i in range(n_layers)] for arm in compressed}
    out = {'n_tokens': n_tokens, 'n_layers': n_layers, 'n_kv_heads': n_heads, 'score_seconds': score_seconds,
           'bits': {}, 'budget_bits': {}, 'over_budget_layers': {}, 'kept': {}, 'key_error': {}, 'value_error': {},
           'full_bits': n_layers * n_heads * n_tokens * 2 * part_bits(16, head_dim)}
    for arm in compressed:
        split, budget = arm.split('@')[0], BUDGETS[arm.split('@')[1]]
        layer_budget = layer_budget_bits(n_heads, n_tokens, head_dim, budget)
        layer_bits = [plan_bits(p, split, head_dim) for p in plans[arm]]
        out['bits'][arm] = sum(layer_bits)
        out['budget_bits'][arm] = n_layers * layer_budget
        out['over_budget_layers'][arm] = sum(b > layer_budget for b in layer_bits)
        out['kept'][arm] = int(sum((p == KEPT).sum().item() for p in plans[arm]))
    arm_cache = mix.batched_cache(cache, len(ARMS))
    masks = [[] for _ in range(n_layers)]
    errors = {arm: ([], []) for arm in compressed}
    for layer_idx in range(n_layers):
        keys, values = cache.layers[layer_idx].keys[0], cache.layers[layer_idx].values[0]
        quant = quantized_layer(keys, values, rot)
        for row_idx, arm in enumerate(ARMS):
            if arm == 'full':
                continue
            kq, vq = quant[arm.split('@')[0]]
            plan = plans[arm][layer_idx]
            sel = plan == KEPT
            arm_cache.layers[layer_idx].keys[row_idx][sel] = kq[sel]
            arm_cache.layers[layer_idx].values[row_idx][sel] = vq[sel]
            errors[arm][0].append(rel(kq[sel], keys[sel]))
            errors[arm][1].append(rel(vq[sel], values[sel]))
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
