import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import mix
from mix import GROUP, META, N_SINK, fake_quant, layer_budget_bits

ARMS = {'stack4': (4, 4), 'k3v3': (3, 3), 'k4v2': (4, 2)}  # key bits, value bits
KEPT, SINK = 1, 16  # plan codes: kept at the arm's precision, or a 16-bit sink


def part_bits(bits, head_dim):
    if bits == 16:
        return head_dim * 16
    return head_dim * bits + (head_dim // GROUP) * META


def entry_bits_kv(kbits, vbits, head_dim):
    return part_bits(kbits, head_dim) + part_bits(vbits, head_dim)


def allocate_uniform(scores, kbits, vbits, head_dim, budget=mix.BUDGET):
    n_heads, n_tokens = scores.shape
    plan = torch.zeros_like(scores, dtype=torch.int16)
    plan[:, :N_SINK] = SINK
    left = (layer_budget_bits(n_heads, n_tokens, head_dim, budget)
            - n_heads * min(N_SINK, n_tokens) * entry_bits_kv(16, 16, head_dim)
            - n_heads * (head_dim * 16 + 32 * 2))
    flat = scores[:, N_SINK:].reshape(-1)
    k = max(0, min(flat.numel(), int(left // entry_bits_kv(kbits, vbits, head_dim))))
    body = plan[:, N_SINK:].reshape(-1)
    body[torch.topk(flat, k).indices] = KEPT
    plan[:, N_SINK:] = body.view(n_heads, n_tokens - N_SINK)
    return plan


def plan_bits_kv(plan, kbits, vbits, head_dim):
    n_heads = plan.shape[0]
    kept = int((plan == KEPT).sum().item())
    sinks = int((plan == SINK).sum().item())
    return (kept * entry_bits_kv(kbits, vbits, head_dim) + sinks * entry_bits_kv(16, 16, head_dim)
            + n_heads * (head_dim * 16 + 32 * 2))


def quantize_parts(keys, values, key_widths, value_widths):
    channel = keys.float().abs().amax(dim=1, keepdim=True).clamp(min=1e-6)
    k = {b: (fake_quant(keys.float() / channel, b) * channel).to(keys.dtype) for b in key_widths}
    v = {b: fake_quant(values.float(), b).to(values.dtype) for b in value_widths}
    return k, v


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
    plans = {arm: [allocate_uniform(scores[i], kb, vb, head_dim) for i in range(n_layers)]
             for arm, (kb, vb) in ARMS.items()}
    layer_budget = layer_budget_bits(n_heads, n_tokens, head_dim)
    out = {'n_tokens': n_tokens, 'score_seconds': score_seconds, 'bits': {}, 'over_budget_layers': {}, 'kept': {},
           'budget_bits': n_layers * layer_budget,
           'full_bits': n_layers * n_heads * n_tokens * entry_bits_kv(16, 16, head_dim)}
    for arm, (kb, vb) in ARMS.items():
        layer_bits = [plan_bits_kv(p, kb, vb, head_dim) for p in plans[arm]]
        out['bits'][arm] = sum(layer_bits)
        out['over_budget_layers'][arm] = sum(b > layer_budget for b in layer_bits)
        out['kept'][arm] = int(sum((p == KEPT).sum().item() for p in plans[arm]))
    ranks = scores.argsort(-1).argsort(-1).float() / max(1, n_tokens - 1)  # L, H, T), 1.0 = top of its head
    tokens = {'ids': context_ids[0].int().cpu().numpy(),
              'rank': ranks.mean((0, 1)).half().cpu().numpy(),
              'kept_stack4': torch.stack([p > 0 for p in plans['stack4']]).float().mean((0, 1)).half().cpu().numpy(),
              'kept_variant': torch.stack([p > 0 for p in plans['k3v3']]).float().mean((0, 1)).half().cpu().numpy()}
    arm_cache = mix.batched_cache(cache, len(ARMS))
    masks = [[] for _ in range(n_layers)]
    for layer_idx in range(n_layers):
        keys, values = cache.layers[layer_idx].keys[0], cache.layers[layer_idx].values[0]
        kq, vq = quantize_parts(keys, values, (3, 4), (2, 3, 4))
        for row_idx, (arm, (kb, vb)) in enumerate(ARMS.items()):
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
    return out, tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--tokens-dir', type=Path, required=True)
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
    args.tokens_dir.mkdir(parents=True, exist_ok=True)
    done = {json.loads(line)['index'] for line in args.out.open()} if args.out.exists() else set()
    capture_cls = mix.capture_press_class()
    with args.out.open('a') as handle:
        for row in rows:
            if row['index'] in done:
                continue
            if args.max_new_tokens:
                row['max_new_tokens'] = min(int(row['max_new_tokens']), args.max_new_tokens)
            result, tokens = run_example(pipe, row, capture_cls)
            np.savez_compressed(args.tokens_dir / f"{row['index']}.npz", **tokens)
            handle.write(json.dumps({'index': row['index'], 'task': row['task'], 'answer': list(row['answer']),
                                     'dtype': str(pipe.model.dtype),
                                     'attn_implementation': pipe.model.config._attn_implementation,
                                     **result}) + '\n')
            handle.flush()
            print(json.dumps({'done': row['index'], 'task': row['task'],
                              'seconds': round(result['score_seconds'] + result['decode_seconds'], 2)}), flush=True)


if __name__ == '__main__':
    sys.exit(main())
