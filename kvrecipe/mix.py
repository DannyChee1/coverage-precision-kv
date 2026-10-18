import argparse
import copy
import json
import sys
import time
from pathlib import Path

import torch

BUDGET = 0.03125  # bytes of the stored context cache, as a share of the full FP16 cache
GROUP = 64  # channels per quantization group, with an FP16 scale and an FP16 zero point
META = 32  # bits per group (scale + zero point
N_SINK = 4  # first context tokens, kept at 16 bits in every compressed arm
ARMS = ('full', 'evict16', 'stack4', 'stack2', 'joint')
STACK_BITS = {'evict16': 16, 'stack4': 4, 'stack2': 2}
JOINT_BITS = (0, 2, 4, 16)


def entry_bits(bits, head_dim):
    if bits == 0:
        return 0
    if bits == 16:
        return 2 * head_dim * 16
    return 2 * (head_dim * bits + (head_dim // GROUP) * META)


def head_overhead_bits(head_dim, tiers):
    return head_dim * 16 + 32 * tiers


def layer_budget_bits(n_heads, n_tokens, head_dim, budget=BUDGET):
    return budget * n_heads * n_tokens * entry_bits(16, head_dim)


def fake_quant(x, bits):
    shape = x.shape
    g = x.float().reshape(*shape[:-1], shape[-1] // GROUP, GROUP)
    lo, hi = g.amin(-1, keepdim=True), g.amax(-1, keepdim=True)
    scale = ((hi - lo) / (2 ** bits - 1)).clamp(min=1e-8)
    q = ((g - lo) / scale).round().clamp(0, 2 ** bits - 1)
    return (q * scale + lo).reshape(shape)


def quantize_entries(keys, values, bits):
    channel = keys.float().abs().amax(dim=1, keepdim=True).clamp(min=1e-6)  # H, 1, D), stored in FP16
    k = fake_quant(keys.float() / channel, bits) * channel
    v = fake_quant(values.float(), bits)
    return k.to(keys.dtype), v.to(values.dtype)


def relative_error(x, xq):
    return ((x.float() - xq.float()).pow(2).sum(-1) / x.float().pow(2).sum(-1).clamp(min=1e-12))


def allocate_stack(scores, bits, head_dim, scale=1.0):
    n_heads, n_tokens = scores.shape
    plan = torch.zeros_like(scores, dtype=torch.int16)
    plan[:, :N_SINK] = 16
    left = (scale * layer_budget_bits(n_heads, n_tokens, head_dim)
            - n_heads * min(N_SINK, n_tokens) * entry_bits(16, head_dim))
    if bits < 16:
        left -= n_heads * head_overhead_bits(head_dim, tiers=2)
    else:
        left -= n_heads * 32 * 1
    flat = scores[:, N_SINK:].reshape(-1)
    k = max(0, min(flat.numel(), int(left // entry_bits(bits, head_dim))))
    keep = torch.topk(flat, k).indices
    body = plan[:, N_SINK:].reshape(-1)
    body[keep] = bits
    plan[:, N_SINK:] = body.view(n_heads, n_tokens - N_SINK)
    return plan


def allocate_joint(scores, errors, head_dim, iters=60):
    n_heads, n_tokens = scores.shape
    w = scores[:, N_SINK:].float().clamp(min=0)
    err = {0: torch.ones_like(w), 16: torch.zeros_like(w)}
    err.update({b: errors[b][:, N_SINK:].float() for b in (2, 4)})
    cost = {b: float(entry_bits(b, head_dim)) for b in JOINT_BITS}
    left = (layer_budget_bits(n_heads, n_tokens, head_dim)
            - n_heads * min(N_SINK, n_tokens) * entry_bits(16, head_dim)
            - n_heads * head_overhead_bits(head_dim, tiers=4))

    distortion = torch.stack([w * err[b] for b in JOINT_BITS])  # 4, H, T'
    costs = torch.tensor([cost[b] for b in JOINT_BITS], device=w.device, dtype=torch.float64)
    widths = torch.tensor(JOINT_BITS, device=w.device)

    def choose(lam):
        pick = (distortion + lam * costs.view(-1, 1, 1).float()).argmin(0)
        return widths[pick], costs[pick].sum().item()

    lo, hi = 0.0, float(w.max().item() + 1e-6) / min(c for c in cost.values() if c > 0)
    while choose(hi)[1] > left:  # make sure the upper end is feasible
        hi *= 2
    best, _ = choose(hi)
    for _ in range(iters):
        mid = (lo + hi) / 2
        bits, used = choose(mid)
        if used <= left:
            hi, best = mid, bits
        else:
            lo = mid
    plan = torch.zeros((n_heads, n_tokens), dtype=torch.int16, device=w.device)
    plan[:, :N_SINK] = 16
    plan[:, N_SINK:] = best.to(torch.int16)
    return plan


def tier_counts(plan):
    widths = torch.tensor(JOINT_BITS, device=plan.device, dtype=plan.dtype).view(-1, 1, 1)
    return (plan.unsqueeze(0) == widths).sum(-1).cpu()


def plan_bits(plan, head_dim):
    counts = tier_counts(plan).tolist()
    total = 0
    for h in range(plan.shape[0]):
        stored = [b for i, b in enumerate(JOINT_BITS) if b > 0 and counts[i][h] > 0]
        total += (head_dim * 16 if any(b < 16 for b in stored) else 0) + 32 * len(stored)
        total += sum(counts[i][h] * entry_bits(b, head_dim) for i, b in enumerate(JOINT_BITS))
    return total


def apply_plan(cache, row, plans, quantized, masks):
    for layer_idx, plan in enumerate(plans):
        keys, values = cache.layers[layer_idx].keys, cache.layers[layer_idx].values  # rows, H, T, D
        for b in (2, 4):
            sel = plan == b
            if sel.any():
                kq, vq = quantized[layer_idx][b]
                keys[row][sel] = kq[sel]
                values[row][sel] = vq[sel]
        heads, seqs = torch.nonzero(plan == 0, as_tuple=True)
        if heads.numel():
            masks[layer_idx].append((torch.full_like(heads, row), heads, seqs))


EXACT = 'mix_exact'
MASK_ELEMENTS = 2 ** 28  # most mask entries one attention call builds (256 MiB as bool


def exact_mask(module, query, key):
    rows, heads, q_len, _ = query.shape
    k_len = key.shape[-2]
    allow = torch.ones(q_len, k_len, dtype=torch.bool, device=query.device).tril(k_len - q_len)[None, None]
    blocked = getattr(module, 'mix_blocked', None)
    if blocked is not None:
        keep = torch.ones(rows, key.shape[1], k_len, dtype=torch.bool, device=query.device)
        keep[:blocked.shape[0], :, :blocked.shape[-1]] = ~blocked  # rows after the last masked one keep everything
        allow = allow & keep.repeat_interleave(heads // key.shape[1], dim=1)[:, :, None, :]
    return allow.expand(rows, heads, q_len, k_len)


def keep_mask(module, rows, kv_heads, k_len, device):
    blocked = getattr(module, 'mix_blocked', None)
    if blocked is None:
        return None
    keep = torch.ones(rows, kv_heads, k_len, dtype=torch.bool, device=device)
    keep[:blocked.shape[0], :, :blocked.shape[-1]] = ~blocked  # rows after the last masked one keep everything
    return keep


def chunk_mask(keep, rows, kv_heads, groups, q_len, k_len, start, stop, device):
    causal = torch.ones(stop - start, k_len, dtype=torch.bool, device=device).tril(k_len - q_len + start)
    mask = causal if keep is None else causal & keep[:, :, None, None, :]
    return mask.expand(rows, kv_heads, groups, stop - start, k_len).reshape(rows, kv_heads, -1, k_len)


def exact_attention(module, query, key, value, attention_mask, dropout=0.0, scaling=None, **kwargs):
    rows, heads, q_len, dim = query.shape
    kv_heads, k_len = key.shape[1], key.shape[-2]
    groups = heads // kv_heads
    keep = keep_mask(module, rows, kv_heads, k_len, query.device)
    grouped = query.reshape(rows, kv_heads, groups, q_len, dim)
    key, value = key.contiguous(), value.contiguous()
    step = max(1, MASK_ELEMENTS // (rows * heads * k_len))
    parts = []
    for start in range(0, q_len, step):
        stop = min(q_len, start + step)
        out = torch.nn.functional.scaled_dot_product_attention(
            grouped[:, :, :, start:stop].reshape(rows, kv_heads, -1, dim), key, value,
            attn_mask=chunk_mask(keep, rows, kv_heads, groups, q_len, k_len, start, stop, query.device),
            dropout_p=dropout, scale=scaling)
        parts.append(out.reshape(rows, kv_heads, groups, stop - start, dim))
    out = torch.cat(parts, dim=3).reshape(rows, heads, q_len, dim)
    return out.transpose(1, 2).contiguous(), None


def set_masks(model, masks):
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    ALL_ATTENTION_FUNCTIONS[EXACT] = exact_attention
    rows = 1 + max((int(p[0].max()) for parts in masks for p in parts if p[0].numel()), default=0)
    kv_heads = model.config.num_key_value_heads
    for layer_idx, layer in enumerate(model.model.layers):
        parts = masks[layer_idx]
        layer.self_attn.masked_key_indices = None  # kvpress's fake-key path stays off
        if not parts:
            layer.self_attn.mix_blocked = None
            continue
        b, h, s = (torch.cat([p[i] for p in parts]) for i in range(3))
        blocked = torch.zeros(rows, kv_heads, int(s.max()) + 1, dtype=torch.bool, device=s.device)
        blocked[b, h, s] = True
        layer.self_attn.mix_blocked = blocked
    model.mix_attn_original = getattr(model, 'mix_attn_original', model.config._attn_implementation)
    model.config._attn_implementation = EXACT


def clear_masks(model):
    for layer in model.model.layers:
        layer.self_attn.masked_key_indices = None
        layer.self_attn.mix_blocked = None
    if hasattr(model, 'mix_attn_original'):
        model.config._attn_implementation = model.mix_attn_original
        del model.mix_attn_original


def batched_cache(cache, rows):
    batched = copy.deepcopy(cache)
    for layer in batched.layers:
        layer.keys = layer.keys.repeat(rows, 1, 1, 1).contiguous()
        layer.values = layer.values.repeat(rows, 1, 1, 1).contiguous()
    return batched


def batched_generate(model, tokenizer, question_ids, cache, context_length, max_new_tokens, resolved=None, info=None):
    rows = cache.layers[0].keys.shape[0]
    stop = model.generation_config.eos_token_id
    stop = stop if isinstance(stop, list) else [stop]
    stop_t = torch.tensor(stop, device=model.device)
    position = torch.arange(context_length, context_length + question_ids.shape[1], device=model.device)
    position = position.unsqueeze(0).repeat(rows, 1)
    out = model(input_ids=question_ids.repeat(rows, 1), past_key_values=cache, position_ids=position, logits_to_keep=1)
    token = out.logits[:, -1].argmax(-1)
    generated, stopped = [token], torch.isin(token, stop_t)

    def finished():
        if resolved is None:
            return stopped
        texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in torch.stack(generated, 1).tolist()]
        return stopped | torch.tensor([resolved(t) for t in texts], device=stopped.device)
    done = finished()
    position = position[:, -1:] + 1
    for i in range(max_new_tokens - 1):
        if bool(done.all()):
            break
        out = model(input_ids=token.unsqueeze(1), past_key_values=cache, position_ids=position + i)
        token = out.logits[:, -1].argmax(-1)
        generated.append(token)
        stopped = stopped | torch.isin(token, stop_t)
        done = finished()
    if info is not None:
        info.update(steps=len(generated), stopped=stopped.tolist())
    answers = []
    for ids in torch.stack(generated, 1).tolist():
        cut = next((j for j, t in enumerate(ids) if t in stop), len(ids) - 1)
        answers.append(str(tokenizer.decode(ids[:cut + 1], skip_special_tokens=True)))
    return answers


def capture_press_class():
    from kvpress.presses.kvzip_press import KVzipPress

    class CaptureKVzip(KVzipPress):

        def compress_post(self, model):
            self.captured = self.score_val.float().clone()  # L, 1, H, T

    return CaptureKVzip


def run_example(pipe, row, capture_cls):
    model = pipe.model
    from transformers import DynamicCache
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
    head_dim = cache.layers[0].keys.shape[-1]
    n_heads = cache.layers[0].keys.shape[1]
    n_layers = len(cache.layers)
    quantized, errors = [], []
    for layer_idx in range(n_layers):
        keys, values = cache.layers[layer_idx].keys[0], cache.layers[layer_idx].values[0]
        q = {b: quantize_entries(keys, values, b) for b in (2, 4)}
        quantized.append(q)
        errors.append({b: (relative_error(keys, q[b][0]) + relative_error(values, q[b][1])) / 2 for b in (2, 4)})
    full_bits = n_layers * n_heads * n_tokens * entry_bits(16, head_dim)
    layer_budget = layer_budget_bits(n_heads, n_tokens, head_dim)
    out = {'n_tokens': n_tokens, 'score_seconds': score_seconds, 'answers': {}, 'bits': {'full': full_bits},
           'tiers': {}, 'over_budget_layers': {}, 'budget_bits': n_layers * layer_budget, 'full_bits': full_bits}
    arm_cache = batched_cache(cache, len(ARMS))
    del cache
    masks = [[] for _ in range(n_layers)]
    for row_idx, arm in enumerate(ARMS):
        if arm == 'full':
            continue
        plans = [allocate_joint(scores[i], errors[i], head_dim) if arm == 'joint'
                 else allocate_stack(scores[i], STACK_BITS[arm], head_dim) for i in range(n_layers)]
        apply_plan(arm_cache, row_idx, plans, quantized, masks)
        layer_bits = [plan_bits(p, head_dim) for p in plans]
        out['bits'][arm] = sum(layer_bits)
        out['over_budget_layers'][arm] = sum(b > layer_budget for b in layer_bits)
        totals = sum(tier_counts(p).sum(-1) for p in plans).tolist()
        out['tiers'][arm] = {str(b): int(c) for b, c in zip(JOINT_BITS, totals)}
    del quantized
    set_masks(model, masks)
    t1 = time.time()
    try:
        with torch.no_grad():
            answers = batched_generate(model, pipe.tokenizer, question_ids, arm_cache, n_tokens,
                                       int(row['max_new_tokens']))
    finally:
        clear_masks(model)
    out['decode_seconds'] = time.time() - t1
    out['answers'] = dict(zip(ARMS, answers))
    return out


def load_rows(per_task, offset):
    from datasets import load_dataset
    df = load_dataset('simonjegou/ruler', data_dir='4096', split='test', revision=RULER_REVISION).to_pandas()
    rows = []
    for task, group in df.groupby('task', sort=True):
        for index, row in group.iloc[offset:offset + per_task].iterrows():
            rows.append(dict(row) | {'index': int(index)})
    return rows


RULER_REVISION = '24adceac8a0e6532936e8d721cd9e9084d2e4686'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--per-task', type=int, default=50)
    parser.add_argument('--offset', type=int, default=0)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--max-new-tokens', type=int, default=0, help='CPU tests only: cap generation length')
    parser.add_argument('--max-examples', type=int, default=0, help='CPU tests only: stop after N examples')
    parser.add_argument('--dtype', default='bfloat16', help='CPU tests only: float32 is faster on CPU')
    args = parser.parse_args()
    import kvpress  # registers the pipeline and patches attention
    from transformers import pipeline
    attn = 'flash_attention_2' if args.device.startswith('cuda') else 'sdpa'
    pipe = pipeline('kv-press-text-generation', model=args.model_path, device=args.device,
                    model_kwargs={'dtype': getattr(torch, args.dtype), 'attn_implementation': attn})
    capture_cls = capture_press_class()
    rows = load_rows(args.per_task, args.offset)
    if args.max_examples:
        rows = rows[::max(1, len(rows) // args.max_examples)][:args.max_examples]
    done = set()
    if args.out.exists():
        done = {json.loads(line)['index'] for line in args.out.open()}
    with args.out.open('a') as handle:
        for row in rows:
            if row['index'] in done:
                continue
            if args.max_new_tokens:
                row['max_new_tokens'] = min(int(row['max_new_tokens']), args.max_new_tokens)
            result = run_example(pipe, row, capture_cls)
            record = {'index': row['index'], 'task': row['task'], 'answer': list(row['answer']),
                      'dtype': str(pipe.model.dtype), 'attn_implementation': pipe.model.config._attn_implementation,
                      **result}
            handle.write(json.dumps(record) + '\n')
            handle.flush()
            print(json.dumps({'done': row['index'], 'task': row['task'], 'n_tokens': result['n_tokens'],
                              'seconds': round(result['score_seconds'] + result['decode_seconds'], 2)}),
                  flush=True)


if __name__ == '__main__':
    sys.exit(main())
