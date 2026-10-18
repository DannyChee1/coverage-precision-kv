import argparse
import json
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).parent))
import controls as kc
import rdkv_hidden as rh
import tca as tca
from mix import N_SINK, layer_budget_bits
from stage2 import KEPT, SINK

MODES = {'calib': ('k3v2pc', 'rdkv', 'tca1', 'tca2', 'tca3', 'tca4', 'tca5', 'tca6'),
         'check': ('tcafit',),
         'ablate': ('k3v2pc', 'rdkvpc', 'tca2', 'tca2a', 'tca2b', 'tca2c', 'tca2d', 'tca2e'),
         'test': ('k3v2pc', 'rdkv', 'rdkvpc', 'tcafit', 'tca2')}
CALIB_PER_TASK = 3
STATE = {'plan': None, 'flip': None, 'tables': None, 'tca_calib': None, 'tca_calib_rows': None, 'probe': False,
         'rec': {}, 'probe_seconds': 0.0, 'routing_terms': True}


def split(arm):
    return arm.split('.', 1)[1].split('@')


def is_ours(arm):
    return arm != 'full' and (split(arm)[0].startswith('tca') or split(arm)[0] == 'rdkvpc')


def hooks(model, sums, queries):
    def hook(module, args, kwargs):
        hidden = kwargs['hidden_states']
        cos, sin = kwargs['position_embeddings']
        q = module.q_proj(hidden).view(*hidden.shape[:-1], -1, module.head_dim)
        if hasattr(module, 'q_norm'):
            q = module.q_norm(q)
        q = q.transpose(1, 2)
        q, _ = sys.modules[type(module).__module__].apply_rotary_pos_emb(q, q, cos, sin)
        sums[module.layer_idx] = (q[0].float() ** 2).sum(1)
        pos = tca.sample_positions(q.shape[2], N_SINK).to(q.device)
        queries[module.layer_idx] = q[0, :, pos].float()
    return [layer.self_attn.register_forward_pre_hook(hook, with_kwargs=True) for layer in model.model.layers]


def tca_tables(pipe):
    if STATE['tables'] is not None:
        return STATE['tables']
    path = STATE['tca_calib']
    if path.exists():
        STATE['tables'] = json.loads(path.read_text())
        return STATE['tables']
    from transformers import DynamicCache
    t0, per, rows, sums = time.time(), {}, [], None
    for row in map(json.loads, open(STATE['tca_calib_rows'])):
        if per.get(row['task'], 0) < CALIB_PER_TASK:
            per[row['task']] = per.get(row['task'], 0) + 1
            rows.append(row)
    for row in rows:
        inputs = pipe.preprocess(row['context'], [row['question']], row['answer_prefix'],
                                 max_context_length=pipe.tokenizer.model_max_length)
        cache = DynamicCache()
        with torch.no_grad():
            pipe.model.model(input_ids=inputs['context_ids'].to(pipe.model.device), past_key_values=cache)
        parts = [tca.eps_sums(layer.keys[0], layer.values[0]) for layer in cache.layers]
        parts = [torch.stack([p[i] for p in parts]) for i in range(4)]
        sums = parts if sums is None else [s + p for s, p in zip(sums, parts)]
    ev, ek = zip(*[tca.eps_tables(sums[0][li], sums[1][li], sums[2][li], sums[3][li]) for li in range(len(sums[0]))])
    STATE['tables'] = {'eps_v': torch.stack(ev).tolist(), 'eps_k': torch.stack(ek).tolist(),
                       'bits_v': list(tca.BITS_V), 'bits_k': list(tca.BITS_K), 'contexts': len(rows),
                       'indices': [r['index'] for r in rows], 'seconds': time.time() - t0}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(STATE['tables']) + '\n')
    return STATE['tables']


def scores_hidden(pipe, context_ids, question_ids, capture_kvzip, capture_scorers, visible):
    from transformers import DynamicCache
    model = pipe.model
    rh.tables(pipe)
    tca_tables(pipe)
    cache, sums, queries = DynamicCache(), {}, {}
    press = capture_kvzip(compression_ratio=0.5, kvzip_plus_normalization=True, n_sink=N_SINK)
    with torch.no_grad(), press(model):
        handles = hooks(model, sums, queries)
        try:
            model.model(input_ids=context_ids, past_key_values=cache)
        finally:
            for h in handles:
                h.remove()
    n_layers, hkv = len(cache.layers), cache.layers[0].keys.shape[1]
    qn = torch.stack([sums[i].reshape(hkv, -1, sums[i].shape[-1]).sum(1).sqrt() for i in range(n_layers)])
    return cache, {'Z': press.captured[:, 0], 'QN': qn, 'QS': torch.stack([queries[i] for i in range(n_layers)])}


def apply(arm_cache, masks, cache, row, vbits, kbits, sink):
    for li in range(len(cache.layers)):
        keys, values = cache.layers[li].keys[0], cache.layers[li].values[0]
        body = vbits[li] > 0
        body[:, :sink] = False
        arm_cache.layers[li].keys[row] = tca.quant_keys(keys, body, kbits[li])
        arm_cache.layers[li].values[row] = tca.quant_values(values, vbits[li])
        heads, seqs = torch.nonzero(vbits[li] == 0, as_tuple=True)
        if heads.numel():
            masks[li].append((torch.full_like(heads, row), heads, seqs))


def record(ctx, vbits, kbits, sink, terms=None, shares=None):
    C, V, S, R = ctx.terms(vbits, kbits, routing=STATE['routing_terms']) if terms is None else terms
    out = {'C': float(C.mean()), 'V': float(V.mean()), 'S': float(S.mean()), 'R': float(R.mean()),
           'hist': tca.histogram(vbits, kbits, sink)}
    if shares is not None:
        out['shares'] = [round(float(s), 2) for s in shares]
    return out


def probe(ctx, layer_bits, cache):
    pl = STATE['plan']['probe']
    base = {'alpha': 1.0, 'nu': 1.0, 'rho': 0.0}
    kap = [float(o['shares'].median())
           for o in tca.plan(ctx, [{**base, 'kappa': k} for k in pl['kappa']], layer_bits, cache=cache)]
    k0 = pl['kappa'][min(range(len(kap)), key=lambda i: abs(kap[i] - pl['target_share']))]
    rho = [float(o['shares'].median())
           for o in tca.plan(ctx, [{**base, 'kappa': k0, 'rho': r} for r in pl['rho']], layer_bits, cache=cache)]
    return {'kappa_shares': kap, 'kappa0': k0, 'rho_shares': rho}


def build_rows(cache, scores, row_arms):
    arm_cache, masks, out = rh.build_rows(cache, scores, ['full' if is_ours(a) else a for a in row_arms])
    budgets = sorted({split(a)[1] for a in row_arms if a != 'full'})
    n_layers = len(cache.layers)
    n_heads, n_tokens, head_dim = cache.layers[0].keys.shape[1:]
    sink = min(N_SINK, n_tokens)
    tables = STATE['tables']
    dev = cache.layers[0].keys.device
    eps_v, eps_k = (torch.tensor(tables[k], device=dev) for k in ('eps_v', 'eps_k'))
    keys = torch.stack([layer.keys[0] for layer in cache.layers])
    ctx = tca.Context(keys, scores['Z'].to(dev), scores['QS'].to(dev), eps_v, eps_k, STATE['flip'], sink,
                      qn=scores['QN'].to(dev))
    del keys
    settings = STATE['plan']['settings']
    for b in budgets:
        limit = layer_budget_bits(n_heads, n_tokens, head_dim, kc.BUDGETS[b])
        mine = [(row, a) for row, a in enumerate(row_arms) if a != 'full' and split(a)[1] == b and is_ours(a)]
        tca_arms = [(row, a) for row, a in mine if split(a)[0].startswith('tca')]
        vcache = {}
        plans = tca.plan(ctx, [settings[split(a)[0]] for _, a in tca_arms], limit, cache=vcache) if tca_arms else []
        done = [(row, a, p['vbits'], p['kbits'], p['terms'], p['shares']) for (row, a), p in zip(tca_arms, plans)]
        for row, a in mine:
            if split(a)[0] == 'rdkvpc':
                w_t, w_c = rh.rdkv_weights(cache, scores)
                vbits, kbits = tca.rdkv_pc_plan(w_t, w_c, eps_v, eps_k, limit / n_heads, sink, head_dim)
                done.append((row, a, vbits, kbits, None, None))
        for row, a, vbits, kbits, terms, shares in done:
            apply(arm_cache, masks, cache, row, vbits, kbits, sink)
            used = tca.plan_bits(vbits, kbits, sink, head_dim)
            out['bits'][a], out['budget_bits'][a] = float(used.sum()), n_layers * limit
            out['over_budget_layers'][a] = int((used > limit).sum())
            out['kept'][a] = int((vbits[..., sink:] > 0).sum())
            STATE['rec'][a] = record(ctx, vbits, kbits, sink, terms, shares)
        k3 = f'Z.k3v2pc@{b}'
        if k3 in row_arms:
            plan = torch.stack([kc.allocate(scores['Z'][i], 'k3v2pc', head_dim, kc.BUDGETS[b])
                                for i in range(n_layers)])
            vbits = torch.where(plan == SINK, 16, torch.where(plan == KEPT, 2, 0)).to(dev)
            kbits = torch.where((plan == KEPT).any(-1, keepdim=True), 3, 0).expand(-1, -1, head_dim).to(dev)
            STATE['rec'][k3] = record(ctx, vbits, kbits, sink)
        if STATE['probe']:
            t0 = time.time()
            STATE['rec'].setdefault('probe', {})[b] = probe(ctx, limit, vcache)
            STATE['probe_seconds'] += time.time() - t0
    return arm_cache, masks, out


def run_example(pipe, row, capture_kvzip, capture_scorers):
    STATE['rec'], STATE['probe_seconds'], rh.STATE['hist'] = {}, 0.0, {}
    out = rh.BASE_EXAMPLE(pipe, row, capture_kvzip, capture_scorers)
    out.update(tca=STATE['rec'], rdkv=rh.STATE['hist'], probe_seconds=STATE['probe_seconds'])
    return out


def run_document(pipe, doc, capture_kvzip, capture_scorers, cap=kc.QUALITY_CAP):
    STATE['rec'], STATE['probe_seconds'], rh.STATE['hist'] = {}, 0.0, {}
    out = rh.BASE_DOCUMENT(pipe, doc, capture_kvzip, capture_scorers, cap)
    out.update(tca=STATE['rec'], rdkv=rh.STATE['hist'], probe_seconds=STATE['probe_seconds'])
    return out


def configure(mode):
    kc.GROUPS = {'Z': MODES[mode]}
    if mode == 'ablate':  # one budget; in place, so every module holding the dict sees it
        kc.BUDGETS.pop('3.125', None)
        STATE['routing_terms'] = False
    kc.score_selectors = scores_hidden
    kc.build_rows = build_rows
    kc.run_example, kc.run_document = run_example, run_document


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--mode', choices=tuple(MODES), required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--model-key', choices=('llama', 'qwen3'), required=True)
    parser.add_argument('--tca-calib', type=Path, required=True)
    parser.add_argument('--tca-calib-rows', type=Path, required=True)
    parser.add_argument('--calib', type=Path, required=True, help="RDKV-hidden's distortion tables")
    parser.add_argument('--calib-rows', type=Path, required=True)
    parser.add_argument('--probe', action='store_true')
    args, rest = parser.parse_known_args()
    STATE['plan'] = json.loads(args.plan.read_text())
    STATE.update(flip=tuple(STATE['plan']['flip'][args.model_key]), tca_calib=args.tca_calib,
                 tca_calib_rows=args.tca_calib_rows, probe=args.probe)
    rh.STATE.update(calib=args.calib, calib_rows=args.calib_rows)
    configure(args.mode)
    sys.argv = [sys.argv[0], *rest]
    return kc.main()


if __name__ == '__main__':
    sys.exit(main())
