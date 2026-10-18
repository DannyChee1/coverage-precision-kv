import argparse
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).parent))
import mix
import controls as kc
from mix import N_SINK, layer_budget_bits
from diagnose import answer_tokens
from improve import part_bits
import keys_ablation as ka
from qwen14b import hidden_scores
from stage2 import KEPT, SINK, quantize_parts
import mix_turboquant as tq

BASE = ('evict16', 'stack4pc', 'k3v2pc')
TIERS = {'tiera': ((4, 4), (2, 2)), 'tierb': ((4, 4), (3, 2))}  # upper, lower) as (key bits, value bits
TQ = {'tqk4v4': (4, 4), 'tqk3v2': (3, 2), 'tqk2v2': (2, 2)}
SPLITS = BASE + tuple(TIERS) + tuple(TQ)
PREC16 = 'Z.prec16@1.5625'
GATE = {'TQ.k4v4': (4, 4, 'mse'), 'TQ.k3v3': (3, 3, 'mse'), 'TQ.k3v3ip': (3, 3, 'ip')}
TOP, TOP_SHARE, TIER_COUNTS = 2, 0.2, 32 * 3
STATE = {'tq_keys': 'mse', 'tca': None, 'gate': False, 'diag': {}, 'masks': None, 'scores': None, 'row_arms': None}
BASE_BUILD, BASE_EXAMPLE, BASE_DOCUMENT = kc.build_rows, kc.run_example, kc.run_document


def split_of(arm):
    split, budget = arm.split('.', 1)[1].split('@')
    return split, budget


def arms(benchmark, budget=None):
    if STATE['gate']:
        return list(GATE)
    out = []
    for b in [budget] if budget else list(kc.BUDGETS):
        out += [f'Z.{s}@{b}' for s in SPLITS + (('tca',) if STATE['tca'] else ())]
        out += [PREC16] if b == '1.5625' else []
    return out


def padding(mask, head_dim):
    return int(((-mask.sum(dim=1)) % ka.TOKENS).sum().item()) * head_dim * 2 * 16 // ka.TOKENS


def tier_overhead(head_dim):
    return TIER_COUNTS + 2 * (ka.TOKENS - 1) * head_dim * 2 * 16 // ka.TOKENS


def allocate_tiers(scores, split, head_dim, budget):
    upper, lower = (ka.entry_bits(*b, head_dim) for b in TIERS[split])
    n_heads, n_tokens = scores.shape
    plan = torch.zeros_like(scores, dtype=torch.int16)
    plan[:, :N_SINK] = SINK
    left = (layer_budget_bits(n_heads, n_tokens, head_dim, budget)
            - n_heads * min(N_SINK, n_tokens) * 2 * part_bits(16, head_dim) - n_heads * tier_overhead(head_dim))
    flat = scores[:, N_SINK:].reshape(-1)
    kept = max(0, int(left // (TOP_SHARE * upper + (1 - TOP_SHARE) * lower)))
    n_top = min(flat.numel(), round(TOP_SHARE * kept))
    n_low = max(0, min(flat.numel() - n_top, int((left - n_top * upper) // lower)))
    order = torch.topk(flat, n_top + n_low).indices
    body = plan[:, N_SINK:].reshape(-1)
    body[order[:n_top]] = TOP
    body[order[n_top:]] = KEPT
    plan[:, N_SINK:] = body.view(n_heads, n_tokens - N_SINK)
    return plan


def tier_bits(plan, split, head_dim):
    upper, lower = (ka.entry_bits(*b, head_dim) for b in TIERS[split])
    top, low, sinks = (int((plan == code).sum()) for code in (TOP, KEPT, SINK))
    return (top * upper + low * lower + sinks * 2 * part_bits(16, head_dim) + plan.shape[0] * TIER_COUNTS
            + padding(plan == TOP, head_dim) + padding(plan == KEPT, head_dim))


def tq_entry_bits(split, head_dim):
    return tq.entry_bits(*TQ[split], STATE['tq_keys'], head_dim)


def tq_plan_bits(plan, ebits, head_dim):
    return (int((plan == KEPT).sum()) * ebits + int((plan == SINK).sum()) * 2 * part_bits(16, head_dim)
            + plan.shape[0] * ka.COUNTS)


def custom_plan(scores, split, head_dim, budget):
    if split in TIERS:
        return allocate_tiers(scores, split, head_dim, budget)
    if split in TQ:
        return ka.allocate(scores, tq_entry_bits(split, head_dim), ka.COUNTS, head_dim, budget)
    return kc.allocate(scores, 'stack4pc', head_dim, kc.BUDGETS['1.5625'])  # prec16: stack4pc's entries


def fill(arm_cache, masks, cache, row, split, plans):
    for li, plan in enumerate(plans):
        keys, values = cache.layers[li].keys[0], cache.layers[li].values[0]
        if split in TIERS:
            vq = quantize_parts(keys, values, (4,), (2, 3, 4))[1]
            for code, (kbits, vbits) in zip((TOP, KEPT), TIERS[split]):
                sel = plan == code
                kq = ka.quantized_keys(keys, 'perchannel', kbits, None, sel)
                arm_cache.layers[li].keys[row][sel] = kq[sel]
                arm_cache.layers[li].values[row][sel] = vq[vbits][sel]
        elif split in TQ:
            sel = plan == KEPT
            kbits, vbits = TQ[split]
            k, v = tq.quantize(keys, values, sel, kbits, vbits, STATE['tq_keys'], li)
            arm_cache.layers[li].keys[row][sel], arm_cache.layers[li].values[row][sel] = k, v
        heads, seqs = torch.nonzero(plan == 0, as_tuple=True)
        if heads.numel():
            masks[li].append((torch.full_like(heads, row), heads, seqs))


def tca_rows(cache, scores, row_arms, arm_cache, masks, out):
    import tca as tca
    import tca_run as run
    rows = [(row, a) for row, a in enumerate(row_arms) if a != 'full' and split_of(a)[0] == 'tca']
    if not rows:
        return
    n_layers = len(cache.layers)
    n_heads, n_tokens, head_dim = cache.layers[0].keys.shape[1:]
    sink, dev = min(N_SINK, n_tokens), cache.layers[0].keys.device
    eps_v, eps_k = (torch.tensor(run.STATE['tables'][k], device=dev) for k in ('eps_v', 'eps_k'))
    keys = torch.stack([layer.keys[0] for layer in cache.layers])
    ctx = tca.Context(keys, scores['Z'].to(dev), scores['QS'].to(dev), eps_v, eps_k, run.STATE['flip'], sink)
    del keys
    setting = run.STATE['plan']['settings'][STATE['tca']]
    for row, arm in rows:
        limit = layer_budget_bits(n_heads, n_tokens, head_dim, kc.BUDGETS[split_of(arm)[1]])
        p = tca.plan(ctx, [setting], limit, cache={})[0]
        run.apply(arm_cache, masks, cache, row, p['vbits'], p['kbits'], sink)
        used = tca.plan_bits(p['vbits'], p['kbits'], sink, head_dim)
        out['bits'][arm], out['budget_bits'][arm] = float(used.sum()), n_layers * limit
        out['over_budget_layers'][arm] = int((used > limit).sum())
        out['kept'][arm] = int((p['vbits'][..., sink:] > 0).sum())


def build_rows(cache, scores, row_arms):
    arm_cache, masks, out = BASE_BUILD(cache, scores, [a if a == 'full' or split_of(a)[0] in BASE else 'full'
                                                       for a in row_arms])
    n_layers = len(cache.layers)
    n_heads, n_tokens, head_dim = cache.layers[0].keys.shape[1:]
    for row, arm in enumerate(row_arms):
        if arm == 'full' or split_of(arm)[0] in BASE + ('tca',):
            continue
        split, budget = split_of(arm)
        limit = layer_budget_bits(n_heads, n_tokens, head_dim, kc.BUDGETS[budget])
        plans = [custom_plan(scores['Z'][li], split, head_dim, kc.BUDGETS[budget]) for li in range(n_layers)]
        fill(arm_cache, masks, cache, row, split, plans)
        if split == 'prec16':
            layer_bits = [tq_plan_bits(p, ka.entry_bits(16, 16, head_dim), head_dim) for p in plans]
            STATE['diag'][arm] = sum(layer_bits)
        else:
            layer_bits = [tier_bits(p, split, head_dim) if split in TIERS
                          else tq_plan_bits(p, tq_entry_bits(split, head_dim), head_dim) for p in plans]
            out['bits'][arm], out['budget_bits'][arm] = sum(layer_bits), n_layers * limit
            out['over_budget_layers'][arm] = sum(b > limit for b in layer_bits)
        out['kept'][arm] = int(sum(((p == KEPT) | (p == TOP)).sum().item() for p in plans))
    if STATE['tca']:
        tca_rows(cache, scores, row_arms, arm_cache, masks, out)
    STATE.update(masks=masks, scores=scores, row_arms=row_arms)
    return arm_cache, masks, out


def gate_scores(pipe, context_ids, question_ids, capture_kvzip, capture_scorers, visible):
    from transformers import DynamicCache
    cache = DynamicCache()
    with torch.no_grad():
        pipe.model.model(input_ids=context_ids, past_key_values=cache)
    return cache, {}


def gate_rows(cache, scores, row_arms):
    n_layers = len(cache.layers)
    n_heads, n_tokens, head_dim = cache.layers[0].keys.shape[1:]
    sink = min(N_SINK, n_tokens)
    full_bits = n_layers * n_heads * n_tokens * 2 * part_bits(16, head_dim)
    out = {'bits': {}, 'budget_bits': {}, 'over_budget_layers': {}, 'kept': {}, 'full_bits': full_bits}
    arm_cache = mix.batched_cache(cache, len(row_arms))
    sel = torch.ones(n_heads, n_tokens, dtype=torch.bool, device=cache.layers[0].keys.device)
    sel[:, :sink] = False
    for row, arm in enumerate(row_arms):
        if arm == 'full':
            continue
        kbits, vbits, kind = GATE[arm]
        for li in range(n_layers):
            keys, values = cache.layers[li].keys[0], cache.layers[li].values[0]
            k, v = tq.quantize(keys, values, sel, kbits, vbits, kind, li)
            arm_cache.layers[li].keys[row][sel], arm_cache.layers[li].values[row][sel] = k, v
        body = int(sel.sum()) * n_layers
        out['bits'][arm] = (body * tq.entry_bits(kbits, vbits, kind, head_dim)
                            + n_layers * n_heads * sink * 2 * part_bits(16, head_dim))
        out['budget_bits'][arm], out['over_budget_layers'][arm], out['kept'][arm] = full_bits, 0, body
    return arm_cache, [[] for _ in range(n_layers)], out


def tca_scores(pipe, context_ids, question_ids, capture_kvzip, capture_scorers, visible):
    import tca_run as run
    from transformers import DynamicCache
    run.tca_tables(pipe)
    cache, sums, queries = DynamicCache(), {}, {}
    press = capture_kvzip(compression_ratio=0.5, kvzip_plus_normalization=True, n_sink=N_SINK)
    with torch.no_grad(), press(pipe.model):
        handles = run.hooks(pipe.model, sums, queries)
        try:
            pipe.model.model(input_ids=context_ids, past_key_values=cache)
        finally:
            for h in handles:
                h.remove()
    return cache, {'Z': press.captured[:, 0], 'QS': torch.stack([queries[i] for i in range(len(cache.layers))])}


def peak_gib():
    return torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else 0.0


def gold_entries(tokenizer, row, n_tokens):
    positions = answer_tokens(tokenizer, row['context'], row['answer'], n_tokens)
    if not positions:
        return {'positions': 0}
    z, masks, row_arms = STATE['scores']['Z'].float(), STATE['masks'], STATE['row_arms']
    n_layers, n_heads, n_t = z.shape
    p = torch.tensor(positions, device=z.device)
    rank = z.argsort(-1).argsort(-1).float() / max(n_t - 1, 1)
    evicted = {a: 0 for a in row_arms if a != 'full'}
    for layer in masks:
        for rows, _, seqs in layer:
            evicted[row_arms[int(rows[0])]] += int(torch.isin(seqs, p.to(seqs.device)).sum())
    total = n_layers * n_heads * len(positions)
    return {'positions': len(positions), 'kept_share': {a: 1 - e / total for a, e in evicted.items()},
            'rank_percentile': 100 * float(rank[..., p].mean())}


def run_example(pipe, row, capture_kvzip, capture_scorers):
    STATE.update(diag={}, masks=None, scores=None, row_arms=None)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    out = BASE_EXAMPLE(pipe, row, capture_kvzip, capture_scorers)
    out.update(tq_keys=STATE['tq_keys'], diagnostic_bits=STATE['diag'], peak_gib=peak_gib())
    if not STATE['gate']:
        out['gold_entries'] = gold_entries(pipe.tokenizer, row, out['n_tokens'])
    STATE.update(masks=None, scores=None)
    return out


def run_document(pipe, doc, capture_kvzip, capture_scorers, cap=kc.QUALITY_CAP):
    STATE.update(diag={}, masks=None, scores=None, row_arms=None)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    out = BASE_DOCUMENT(pipe, doc, capture_kvzip, capture_scorers, cap)
    out.update(tq_keys=STATE['tq_keys'], diagnostic_bits=STATE['diag'], peak_gib=peak_gib())
    STATE.update(masks=None, scores=None)
    return out


def configure(gate=False, tq_keys='mse', tca=None):
    STATE.update(gate=gate, tq_keys=tq_keys, tca=tca)
    kc.GROUPS = {'Z': SPLITS}
    kc.arms = arms
    kc.score_selectors = gate_scores if gate else tca_scores if tca else hidden_scores
    kc.build_rows = gate_rows if gate else build_rows
    kc.run_example, kc.run_document = run_example, run_document


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--gate', action='store_true')
    parser.add_argument('--tq-keys', choices=tq.KINDS, default='mse')
    parser.add_argument('--tca-plan', type=Path)
    parser.add_argument('--tca-setting', default='tca')
    parser.add_argument('--tca-calib', type=Path)
    parser.add_argument('--tca-calib-rows', type=Path)
    parser.add_argument('--model-key', choices=('llama', 'qwen3'))
    args, rest = parser.parse_known_args()
    if args.gate and args.tca_plan:
        raise SystemExit('the gate has no TCA row')
    if args.tca_plan:
        import tca_run as run
        plan = json.loads(args.tca_plan.read_text())
        if args.tca_setting not in plan['settings'] or not (args.tca_calib and args.tca_calib_rows and args.model_key):
            raise SystemExit('--tca-plan needs a setting in it, --tca-calib, --tca-calib-rows and --model-key')
        run.STATE.update(plan=plan, flip=tuple(plan['flip'][args.model_key]), tca_calib=args.tca_calib,
                         tca_calib_rows=args.tca_calib_rows)
    configure(args.gate, args.tq_keys, args.tca_setting if args.tca_plan else None)
    sys.argv = [sys.argv[0], *rest]
    return kc.main()


if __name__ == '__main__':
    sys.exit(main())
