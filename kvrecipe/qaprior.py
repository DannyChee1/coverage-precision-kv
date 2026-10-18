import json
import re
from pathlib import Path

import numpy as np
import torch

from mix import N_SINK, fake_quant, layer_budget_bits
from improve import part_bits
import keys_ablation as ka
from stage2 import KEPT, SINK

TOP, TIER_COUNTS = 2, 32 * 3
ORACLE = {'o16': (16, 16), 'o8': (8, 8), 'o4': (4, 4), 'o42': (4, 2)}
REMAINDER = (3, 2)


def prompt_text(tokenizer, context):
    separator = '#' * (len(context) + 10)
    if tokenizer.chat_template is None:
        return (tokenizer.bos_token or '') + context
    return tokenizer.apply_chat_template([{'role': 'user', 'content': context + separator}], add_generation_prompt=True,
                                         tokenize=False, enable_thinking=False).split(separator)[0]


def number_entity(text, offsets):
    flags = np.zeros(len(offsets))
    for m in re.finditer(r'\S+', text):
        word = m.group().strip('.,;:!?()"\'')
        before = text[:m.start()].rstrip()
        sentence_start = not before or before[-1] in '.!?\n'
        if any(c.isdigit() for c in word) or (word[:1].isupper() and not sentence_start):
            for i, (a, b) in enumerate(offsets):
                if a < m.end() and b > m.start():
                    flags[i] = 1
    return flags


def load_idf(path):
    table = json.loads(Path(path).read_text())
    return {int(k): v for k, v in table['idf'].items()}, table['default']


def token_prior(tokenizer, context, n_tokens, idf, kind):
    text = prompt_text(tokenizer, context)
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    if len(enc['input_ids']) != n_tokens:
        return None
    table, default = idf
    values = np.array([table.get(t, default) for t in enc['input_ids']])
    if kind == 'ne_idf':
        values = number_entity(text, enc['offset_mapping']) * 1e3 + values
    return values


def percentile(values):
    from scipy.stats import rankdata
    v = np.asarray(values)
    return torch.tensor((rankdata(v, method='average') - 1) / max(len(v) - 1, 1), dtype=torch.float32)


def layer_percentile(z):
    flat = z.float().reshape(z.shape[0], -1)
    ranks = flat.argsort(dim=1).argsort(dim=1).float() / max(flat.shape[1] - 1, 1)
    return ranks.view_as(z)


def blend(z, prior_pct, lam):
    return layer_percentile(z) + lam * prior_pct.to(z.device).view(1, 1, -1)


def prior_only(z, prior_pct):
    return prior_pct.to(z.device).view(1, 1, -1) + layer_percentile(z) / 1e6


def protected_positions(gold, m, n_tokens):
    gold = sorted({p for p in gold if N_SINK <= p < n_tokens})
    if m <= len(gold):
        return gold[:m]
    chosen, runs = set(gold), []
    for p in gold:  # contiguous runs of gold positions are occurrences
        if runs and p == runs[-1][1] + 1:
            runs[-1][1] = p
        else:
            runs.append([p, p])
    step = 1
    while len(chosen) < m and step < n_tokens:
        grew = False
        for lo, hi in runs:
            for p in (lo - step, hi + step):
                if len(chosen) < m and N_SINK <= p < n_tokens and p not in chosen:
                    chosen.add(p)
                    grew = True
        if not grew and all(lo - step < N_SINK and hi + step >= n_tokens for lo, hi in runs):
            break
        step += 1
    return sorted(chosen)


def protected_bits(kbits, vbits, head_dim):
    return 2 * part_bits(16, head_dim) if kbits == 16 else part_bits(kbits, head_dim) + part_bits(vbits, head_dim)


def oracle_plan(z, gold, arm, remainder, head_dim, budget):
    n_heads, n_tokens = z.shape
    plan = torch.zeros_like(z, dtype=torch.int16)
    plan[:, :N_SINK] = SINK
    limit = layer_budget_bits(n_heads, n_tokens, head_dim, budget)
    fixed = n_heads * min(N_SINK, n_tokens) * 2 * part_bits(16, head_dim) + n_heads * TIER_COUNTS
    n_gold = len({p for p in gold if N_SINK <= p < n_tokens})
    allowance = min(n_gold * protected_bits(16, 16, head_dim), (limit - fixed) // n_heads)  # per head
    m = int(allowance // protected_bits(*ORACLE[arm], head_dim))
    pos = protected_positions(gold, m, n_tokens)
    if pos:
        plan[:, pos] = TOP
    if remainder:
        left = limit - fixed - n_heads * allowance - n_heads * ka.reserved_overhead('perchannel', head_dim)
        free = (plan == 0)[:, N_SINK:].reshape(-1)
        flat = z[:, N_SINK:].reshape(-1).float().masked_fill(~free, float('-inf'))
        k = max(0, min(int(free.sum()), int(left // ka.entry_bits(*REMAINDER, head_dim))))
        body = plan[:, N_SINK:].reshape(-1)
        body[torch.topk(flat, k).indices] = KEPT
        plan[:, N_SINK:] = body.view(n_heads, n_tokens - N_SINK)
    return plan, {'protected': len(pos), 'allowance_bits': int(allowance), 'bits': oracle_bits(plan, arm, head_dim)}


def oracle_bits(plan, arm, head_dim):
    n_heads = plan.shape[0]
    top, kept, sinks = (int((plan == code).sum()) for code in (TOP, KEPT, SINK))
    return (top * protected_bits(*ORACLE[arm], head_dim) + kept * ka.entry_bits(*REMAINDER, head_dim)
            + ka.padding_bits(plan, head_dim) + sinks * 2 * part_bits(16, head_dim) + n_heads * TIER_COUNTS)


def oracle_fill(arm_cache, masks, cache, row, plans, arm):
    kbits, vbits = ORACLE[arm]
    for li, plan in enumerate(plans):
        keys, values = cache.layers[li].keys[0], cache.layers[li].values[0]
        top, kept = plan == TOP, plan == KEPT
        if top.any() and kbits < 16:  # per token, groups of 64 channels: no per-channel scales to store
            arm_cache.layers[li].keys[row][top] = fake_quant(keys, kbits).to(keys.dtype)[top]
            arm_cache.layers[li].values[row][top] = fake_quant(values, vbits).to(values.dtype)[top]
        if kept.any():
            kq = ka.quantized_keys(keys, 'perchannel', REMAINDER[0], None, kept)
            vq = fake_quant(values, REMAINDER[1]).to(values.dtype)  # as the recipe's values
            arm_cache.layers[li].keys[row][kept] = kq[kept]
            arm_cache.layers[li].values[row][kept] = vq[kept]
        heads, seqs = torch.nonzero(plan == 0, as_tuple=True)
        if heads.numel():
            masks[li].append((torch.full_like(heads, row), heads, seqs))
