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
import keys_ablation as ka
from stage2 import KEPT, SINK

BUDGET = 0.015625
SPEC = {'k3v2_pc': (3, 2, 'perchannel', 'k3v2'), 'k2v2_pc': (2, 2, 'perchannel', 'e640'),
        'k2v2_rk': (2, 2, 'centered_rotated', 'e640'), 'k3v1_pc': (3, 1, 'perchannel', 'e640')}
ROWS = ('full', *SPEC)


def fixed_overhead(row, head_dim):
    kb, vb, quantizer, _ = SPEC[row]
    return ka.COUNTS + (head_dim * 16 if quantizer == 'centered_rotated' else 0) + (head_dim * 16 if vb == 1 else 0)


def reserved_overhead(row, head_dim):
    pad = (ka.TOKENS - 1) * head_dim * 2 * 16 // ka.TOKENS if SPEC[row][2] == 'perchannel' else 0
    return fixed_overhead(row, head_dim) + pad


def plan_overhead(plan_name, head_dim):
    return max(reserved_overhead(r, head_dim) for r, s in SPEC.items() if s[3] == plan_name)


def entry_bits(row, head_dim):
    kb, vb, _, _ = SPEC[row]
    return part_bits(kb, head_dim) + part_bits(vb, head_dim)


def row_bits(plan, row, head_dim):
    kept, sinks = int((plan == KEPT).sum()), int((plan == SINK).sum())
    pad = ka.padding_bits(plan, head_dim) if SPEC[row][2] == 'perchannel' else 0
    return (kept * entry_bits(row, head_dim) + sinks * 2 * part_bits(16, head_dim)
            + plan.shape[0] * fixed_overhead(row, head_dim) + pad)


def quantized_values(values, bits, rot):
    return (centered_rotated(values, 1, rot) if bits == 1 else fake_quant(values.float(), bits)).to(values.dtype)


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
    plan_rows = {s[3]: r for r, s in SPEC.items()}  # any row of each plan sets its entry cost
    plans = {p: [ka.allocate(scores[i], entry_bits(r, head_dim), plan_overhead(p, head_dim), head_dim, BUDGET)
                 for i in range(n_layers)] for p, r in plan_rows.items()}
    layer_budget = layer_budget_bits(n_heads, n_tokens, head_dim, BUDGET)
    out = {'n_tokens': n_tokens, 'n_layers': n_layers, 'n_kv_heads': n_heads, 'score_seconds': score_seconds,
           'bits': {}, 'budget_bits': {}, 'over_budget_layers': {}, 'kept': {}, 'key_error': {}, 'value_error': {},
           'full_bits': n_layers * n_heads * n_tokens * 2 * part_bits(16, head_dim)}
    for r, (_, _, _, p) in SPEC.items():
        layer_bits = [row_bits(plan, r, head_dim) for plan in plans[p]]
        out['bits'][r] = sum(layer_bits)
        out['budget_bits'][r] = n_layers * layer_budget
        out['over_budget_layers'][r] = sum(b > layer_budget for b in layer_bits)
        out['kept'][r] = int(sum((plan == KEPT).sum().item() for plan in plans[p]))
    arm_cache = mix.batched_cache(cache, len(ROWS))
    masks = [[] for _ in range(n_layers)]
    errors = {r: ([], []) for r in SPEC}
    for li in range(n_layers):
        keys, values = cache.layers[li].keys[0], cache.layers[li].values[0]
        kcache, vcache = {}, {}
        for ri, r in enumerate(ROWS):
            if r == 'full':
                continue
            kb, vb, quantizer, p = SPEC[r]
            plan = plans[p][li]
            sel = plan == KEPT
            if (quantizer, kb, p) not in kcache:
                kcache[(quantizer, kb, p)] = ka.quantized_keys(keys, quantizer, kb, rot, sel)
            if vb not in vcache:
                vcache[vb] = quantized_values(values, vb, rot)
            kq, vq = kcache[(quantizer, kb, p)], vcache[vb]
            arm_cache.layers[li].keys[ri][sel] = kq[sel]
            arm_cache.layers[li].values[ri][sel] = vq[sel]
            errors[r][0].append(rel(kq[sel], keys[sel]))
            errors[r][1].append(rel(vq[sel], values[sel]))
            heads, seqs = torch.nonzero(plan == 0, as_tuple=True)
            if heads.numel():
                masks[li].append((torch.full_like(heads, ri), heads, seqs))
    for r, (ke, ve) in errors.items():
        out['key_error'][r], out['value_error'][r] = sum(ke) / len(ke), sum(ve) / len(ve)
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
    out['answers'] = dict(zip(ROWS, answers))
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
