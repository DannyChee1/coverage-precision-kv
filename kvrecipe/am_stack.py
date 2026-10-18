import argparse
import copy
import json
import math
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'third_party'))
import mix
from attention_matching.highest_attention_keys import HighestAttentionKeysCompaction
from mix import N_SINK, fake_quant, layer_budget_bits
from confirm import load_rows
from improve import part_bits, rel, rotation
import keys_ablation as ka
from stage2 import KEPT

BUDGET = 0.015625
ROWS = ('full', 'am16', 'am_stack4', 'am_k3v2', 'kvzip_k3pc')
AM_ROWS = {'am16': (16, 16), 'am_stack4': (4, 4), 'am_k3v2': (3, 2)}  # key bits, value bits
BETA_BITS, COUNTS, TOKENS = 16, 64, ka.TOKENS
REPEAT_PROMPT = 'Repeat the previous context verbatim.'
MAX_QUERIES = 50_000  # per KV head, as in AM's repeat config
ALG = HighestAttentionKeysCompaction(score_method='rms', nnls_iters=2, nnls_lower_bound=math.exp(-3),
                                     nnls_upper_bound=math.exp(3), c2_method='lsq')
CAPTURE, BIASED = 'mix_am_capture', 'mix_exact_bias'


def am_entry_bits(row, head_dim):
    kb, vb = AM_ROWS[row]
    return part_bits(kb, head_dim) + part_bits(vb, head_dim) + BETA_BITS


def am_overhead(row, head_dim):
    pad = (TOKENS - 1) * head_dim * 2 * 16 // TOKENS if AM_ROWS[row][0] < 16 else 0
    return COUNTS + pad


def am_bits(kept_per_head, row, head_dim):
    kept = torch.as_tensor(kept_per_head)
    pad = int(((-kept) % TOKENS).sum()) * head_dim * 2 * 16 // TOKENS if AM_ROWS[row][0] < 16 else 0
    return int(kept.sum()) * am_entry_bits(row, head_dim) + len(kept) * COUNTS + pad


def pooled_counts(scores, row, head_dim, budget=BUDGET):
    n_heads, n_tokens = scores.shape
    left = layer_budget_bits(n_heads, n_tokens, head_dim, budget) - n_heads * am_overhead(row, head_dim)
    k = max(0, min(scores.numel(), int(left // am_entry_bits(row, head_dim))))
    flat = torch.zeros(scores.numel(), dtype=torch.long, device=scores.device)
    flat[torch.topk(scores.reshape(-1), k).indices] = 1
    return flat.view(n_heads, -1).sum(dim=1).tolist()


def key_stats(Q, K):
    scores32 = (Q @ K.T).to(torch.float32) * (1.0 / Q.shape[1]) ** 0.5
    max_scores = scores32.max(dim=1, keepdim=True)[0]
    exp_scores = torch.exp(scores32 - max_scores)
    attention = exp_scores / exp_scores.sum(dim=1, keepdim=True)
    return torch.sqrt((attention ** 2).mean(dim=0)), exp_scores, max_scores, attention


def compact_head(Q, K, V, t, kbits, vbits, stats):
    rms, exp_scores, max_scores, attention = stats
    idx = torch.topk(rms, t, largest=True).indices
    C1 = K[idx]
    if kbits == 16:
        M = exp_scores[:, idx]
    else:
        idx = idx.sort().values
        C1 = ka.per_channel(K[idx].float(), kbits).to(K.dtype)
        M = torch.exp((Q @ C1.T).to(torch.float32) * (1.0 / Q.shape[1]) ** 0.5 - max_scores)
    B = ALG._nnls_pg(M, exp_scores.sum(dim=1), ALG.nnls_iters, ALG.nnls_lower_bound, ALG.nnls_upper_bound)
    beta = torch.log(B).to(K.dtype)
    C2 = ALG._compute_C2_with_method(C1, beta, K, V, Q, method='lsq', indices=idx.cpu().tolist(),
                                     ridge_lambda=ALG.c2_ridge_lambda, solver=ALG.c2_solver,
                                     ridge_scale=ALG.c2_ridge_scale)
    if vbits < 16:
        C2 = fake_quant(C2.float(), vbits).to(V.dtype)
    ref = attention @ V.float()
    approx = torch.softmax((Q @ C1.T).float() * (1.0 / Q.shape[1]) ** 0.5 + beta.float(), dim=-1) @ C2.float()
    error = float(((approx - ref).norm(dim=-1) / ref.norm(dim=-1).clamp(min=1e-9)).mean())
    return idx, C1, beta, C2, error


def kv_queries(query, n_heads, seed=0):
    groups, n, d = query.shape[0] // n_heads, query.shape[1], query.shape[2]
    if groups * n > MAX_QUERIES:
        keep = torch.randperm(n, generator=torch.Generator().manual_seed(seed))[:MAX_QUERIES // groups].sort().values
        query = query[:, keep.to(query.device)]
    return query.reshape(n_heads, -1, d)


def fit_layer(Q, K, V):
    n_heads, head_dim = K.shape[0], K.shape[-1]
    rms = torch.stack([key_stats(Q[h], K[h])[0] for h in range(n_heads)])
    counts = {row: pooled_counts(rms, row, head_dim) for row in AM_ROWS}
    out, errors = {row: [None] * n_heads for row in AM_ROWS}, {row: [] for row in AM_ROWS}
    for h in range(n_heads):
        stats = key_stats(Q[h], K[h])
        for row, (kb, vb) in AM_ROWS.items():
            if counts[row][h]:
                *entry, err = compact_head(Q[h], K[h], V[h], counts[row][h], kb, vb, stats)
                out[row][h] = tuple(entry)
                errors[row].append(err)
        del stats
    return out, {row: sum(e) / max(len(e), 1) for row, e in errors.items()}


ACTIVE = None  # context length, {layer: fit_layer output}) while the repeat pass runs


def capture_attention(module, query, key, value, attention_mask, dropout=0.0, scaling=None, **kwargs):
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    n_ctx, fits = ACTIVE
    n_heads = key.shape[1]
    fits[module.layer_idx] = fit_layer(kv_queries(query[0], n_heads), key[0, :, :n_ctx], value[0, :, :n_ctx])
    q_len, k_len = query.shape[2], key.shape[-2]
    allow = torch.ones(q_len, k_len, dtype=torch.bool, device=query.device).tril(k_len - q_len)[None, None]
    return sdpa_attention_forward(module, query, key, value, allow, dropout=dropout, scaling=scaling,
                                  is_causal=False, **kwargs)


def biased_attention(module, query, key, value, attention_mask, dropout=0.0, scaling=None, **kwargs):
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    allow = mix.exact_mask(module, query, key)
    bias = torch.zeros(allow.shape[0], key.shape[1], key.shape[-2], dtype=query.dtype, device=query.device)
    stored = getattr(module, 'mix_bias', None)
    if stored is not None:
        bias[:stored.shape[0], :, :stored.shape[-1]] = stored
    bias = bias.repeat_interleave(query.shape[1] // key.shape[1], dim=1)[:, :, None, :]
    mask = torch.where(allow, bias, torch.tensor(float('-inf'), dtype=query.dtype, device=query.device))
    return sdpa_attention_forward(module, query, key, value, mask, dropout=dropout, scaling=scaling,
                                  is_causal=False, **kwargs)


def repeat_pass(pipe, cache, context, n_ctx):
    global ACTIVE
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    ALL_ATTENTION_FUNCTIONS[CAPTURE] = capture_attention
    model, tok = pipe.model, pipe.tokenizer
    prompt = pipe.preprocess(context, [REPEAT_PROMPT], '', max_context_length=tok.model_max_length)
    article = tok.encode(context, add_special_tokens=False, return_tensors='pt')
    ids = torch.cat([prompt['questions_ids'][0], article], 1).to(model.device)
    fits, original = {}, model.config._attn_implementation
    torch.manual_seed(0)  # AM's NNLS starts its power iteration from torch.randn
    ACTIVE, model.config._attn_implementation = (n_ctx, fits), CAPTURE
    try:
        with torch.no_grad():
            model.model(input_ids=ids, past_key_values=copy.deepcopy(cache))
    finally:
        ACTIVE, model.config._attn_implementation = None, original
    return fits


def set_masks_bias(model, masks, biases):
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    ALL_ATTENTION_FUNCTIONS[BIASED] = biased_attention
    mix.set_masks(model, masks)
    for layer, bias in zip(model.model.layers, biases):
        layer.self_attn.mix_bias = bias
    model.config._attn_implementation = BIASED


def clear_masks_bias(model):
    for layer in model.model.layers:
        layer.self_attn.mix_bias = None
    mix.clear_masks(model)


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
    t1 = time.time()
    fits = repeat_pass(pipe, cache, row['context'], n_tokens)
    am_seconds = time.time() - t1
    rot = rotation(head_dim, scores.device)
    kz_plans = [ka.allocate(scores[i], ka.entry_bits(3, 2, head_dim), ka.plan_overhead('k3', head_dim), head_dim)
                for i in range(n_layers)]
    budget = layer_budget_bits(n_heads, n_tokens, head_dim, BUDGET)
    out = {'n_tokens': n_tokens, 'n_layers': n_layers, 'n_kv_heads': n_heads, 'score_seconds': score_seconds,
           'am_seconds': am_seconds, 'bits': {}, 'budget_bits': {}, 'over_budget_layers': {}, 'kept': {},
           'key_error': {}, 'value_error': {}, 'am_output_error': {},
           'full_bits': n_layers * n_heads * n_tokens * 2 * part_bits(16, head_dim)}
    arm_cache = mix.batched_cache(cache, len(ROWS))
    dtype = cache.layers[0].keys.dtype
    masks = [[] for _ in range(n_layers)]
    biases = [torch.zeros(len(ROWS), n_heads, n_tokens, dtype=dtype, device=scores.device) for _ in range(n_layers)]
    layer_bits = {r: [] for r in ROWS[1:]}
    errors = {r: ([], []) for r in ROWS[1:]}
    for li in range(n_layers):
        keys, values = cache.layers[li].keys[0], cache.layers[li].values[0]
        row_keys, row_values = arm_cache.layers[li].keys, arm_cache.layers[li].values
        for ri, r in enumerate(ROWS):
            if r == 'full':
                continue
            if r == 'kvzip_k3pc':
                plan = kz_plans[li]
                sel = plan == KEPT
                row_keys[ri][sel] = ka.quantized_keys(keys, 'perchannel', 3, rot, sel)[sel]
                row_values[ri][sel] = fake_quant(values.float(), 2).to(values.dtype)[sel]
                keep = plan != 0
                layer_bits[r].append(ka.plan_bits(plan, 'k3_pc', head_dim))
            else:
                sel = torch.zeros(n_heads, n_tokens, dtype=torch.bool, device=keys.device)
                for h, entry in enumerate(fits[li][0][r]):
                    if entry is not None:
                        idx, C1, beta, C2 = entry
                        row_keys[ri][h, idx], row_values[ri][h, idx] = C1, C2
                        biases[li][ri, h, idx] = beta
                        sel[h, idx] = True
                keep = sel
                layer_bits[r].append(am_bits(sel.sum(dim=1), r, head_dim))
            errors[r][0].append(rel(row_keys[ri][sel], keys[sel]))
            errors[r][1].append(rel(row_values[ri][sel], values[sel]))
            out['kept'][r] = out['kept'].get(r, 0) + int(sel.sum())
            heads, seqs = torch.nonzero(~keep, as_tuple=True)
            if heads.numel():
                masks[li].append((torch.full_like(heads, ri), heads, seqs))
    for r in ROWS[1:]:
        out['bits'][r] = sum(layer_bits[r])
        out['budget_bits'][r] = n_layers * budget
        out['over_budget_layers'][r] = sum(b > budget for b in layer_bits[r])
        out['key_error'][r] = sum(errors[r][0]) / len(errors[r][0])
        out['value_error'][r] = sum(errors[r][1]) / len(errors[r][1])
    for r in AM_ROWS:
        out['am_output_error'][r] = sum(fits[li][1][r] for li in range(n_layers)) / n_layers
    del cache, fits
    set_masks_bias(model, masks, biases)
    t2 = time.time()
    try:
        with torch.no_grad():
            answers = mix.batched_generate(model, pipe.tokenizer, question_ids, arm_cache, n_tokens,
                                             int(row['max_new_tokens']))
    finally:
        clear_masks_bias(model)
    out['decode_seconds'] = time.time() - t2
    out['answers'] = dict(zip(ROWS, answers))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--rows', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--per-task', type=int, default=26)
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
            seconds = result['score_seconds'] + result['am_seconds'] + result['decode_seconds']
            print(json.dumps({'done': row['index'], 'task': row['task'], 'seconds': round(seconds, 2)}), flush=True)


if __name__ == '__main__':
    sys.exit(main())
