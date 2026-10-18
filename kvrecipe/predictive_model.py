import argparse
import glob
import json
import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from mix import N_SINK, layer_budget_bits
import controls as kc
import keys_ablation as ka
from analyze import score
import short_window_check as swc

ROOT = Path(__file__).parents[1]
Q = ROOT / 'results'
RECORDS = Path(__file__).resolve().parents[1] / 'records'
MODELS = {'llama': {'layers': 32, 'heads': 8}, 'qwen3': {'layers': 36, 'heads': 8}}
HEAD_DIM = 128
CACHES = {'k3v2pc': 3, 'stack4pc': 4}  # cache: key bits
BUDGETS = ('1.5625', '3.125')
KS = np.exp(np.linspace(math.log(1), math.log(400), 4001))


def flip_curve(model):
    z = np.load(Q / f'routing/{model}.npz')
    curve = {}
    for bi, bits in enumerate(z['bits']):
        n = z['n'][..., bi, 0].sum(axis=(0, 1)) / z['visits'][..., bi, 0].sum(axis=(0, 1))
        flips = z['flip1'][..., bi, 0].sum(axis=(0, 1)) / z['rows'][..., bi, 0].sum(axis=(0, 1))
        curve[int(bits)] = (np.log2(n), flips)
    return curve


def flip(curve, bits, n):
    x, y = curve[bits]
    v = math.log2(max(n, 1.0))
    if v >= x[-1]:
        return float(y[-1] + (v - x[-1]) * (y[-1] - y[-2]) / (x[-1] - x[-2]))
    return float(np.interp(v, x, y))


def kept_per_layer(cache, budget, n_tokens, heads=8):
    left = (layer_budget_bits(heads, n_tokens, HEAD_DIM, kc.BUDGETS[budget])
            - heads * min(N_SINK, n_tokens) * 2 * ka.part_bits(16, HEAD_DIM) - heads * kc.overhead(cache, HEAD_DIM))
    kbits, vbits = kc.SPLITS[cache]
    return max(0, min(heads * (n_tokens - N_SINK), int(left // ka.entry_bits(kbits, vbits, HEAD_DIM))))


def predict(full, q, n_per_head, bits, k, curve, routing=True):
    a = 1 - (1 - min(1.0, q)) ** k
    r = 1 - flip(curve, bits, n_per_head) if routing else 1.0
    return full * a * r


def ruler_cells(model):
    recs = [json.loads(line) for f in sorted(glob.glob(str(RECORDS / f'review_4k_{model}_ruler_[0-9].jsonl')))
            for line in open(f)]
    shape = MODELS[model]
    cells = []
    for task in sorted({r['task'] for r in recs}):
        rows = [r for r in recs if r['task'] == task and r['gold_entries'].get('positions', 0) > 0]
        full = 100 * np.mean([r['scores']['full'] / 100 for r in rows])
        for cache, bits in CACHES.items():
            for budget in BUDGETS:
                arm = f'Z.{cache}@{budget}'
                f = np.mean([r['kept'][arm] / (shape['layers'] * shape['heads'] * (r['n_tokens'] - N_SINK)) for r in rows])
                g = np.mean([r['gold_entries']['kept_share'][arm] for r in rows])
                s = np.array([r['scores'][arm] for r in rows])
                other = np.array([r['scores'][f"Z.{'stack4pc' if cache == 'k3v2pc' else 'k3v2pc'}@{budget}"] for r in rows])
                cells.append({'task': task, 'cache': cache, 'bits': bits, 'budget': budget, 'n': len(rows),
                              'full': float(full), 'acc': float(s.mean()), 'var': float(s.var(ddof=1) / len(rows)),
                              'gap_var': float((s - other).var(ddof=1) / len(rows)),
                              'keep': float(f), 'lift': float(g / f),
                              'kept_per_head': float(np.mean([r['kept'][arm] / (shape['layers'] * shape['heads'])
                                                              for r in rows]))})
    return cells


def fit_k(cells, curve, routing=True):
    w = np.array([c['n'] for c in cells], float)
    obs = np.array([c['acc'] for c in cells])
    best = min(KS, key=lambda k: float(np.sum(w * (obs - np.array(
        [predict(c['full'], c['keep'] * c['lift'], c['kept_per_head'], c['bits'], k, curve, routing)
         for c in cells])) ** 2)))
    return float(best)


def gaps(cells, values):
    by = {(c['task'], c['cache'], c['budget']): v for c, v in zip(cells, values)}
    return {(t, b): by[(t, 'k3v2pc', b)] - by[(t, 'stack4pc', b)] for t, _, b in by}


def calibrate(model, routing=True):
    curve, cells = flip_curve(model), ruler_cells(model)
    k = fit_k(cells, curve, routing)
    pred = [predict(c['full'], c['keep'] * c['lift'], c['kept_per_head'], c['bits'], k, curve, routing) for c in cells]
    obs = [c['acc'] for c in cells]
    cv = []
    for task in sorted({c['task'] for c in cells}):
        train = [c for c in cells if c['task'] != task]
        kt = fit_k(train, curve, routing)
        cv += [(c, predict(c['full'], c['keep'] * c['lift'], c['kept_per_head'], c['bits'], kt, curve, routing))
               for c in cells if c['task'] == task]
    cv_cells, cv_pred = [c for c, _ in cv], [p for _, p in cv]
    level_err = np.array(cv_pred) - np.array([c['acc'] for c in cv_cells])
    g_pred, g_obs = gaps(cv_cells, cv_pred), gaps(cv_cells, [c['acc'] for c in cv_cells])
    gap_err = np.array([g_pred[key] - g_obs[key] for key in g_obs])
    sign_hits = sum((g_pred[key] > 0) == (g_obs[key] > 0) for key in g_obs)
    qa = [i for i, c in enumerate(cells) if c['task'] in ('qa_1', 'qa_2')]
    gap_var = {(c['task'], c['budget']): c['gap_var'] for c in cv_cells if c['cache'] == 'k3v2pc'}
    return {'k': k, 'cells': [c | {'pred': p} for c, p in zip(cells, pred)],
            'sigma_model_level': float(np.sqrt(max(0.0, np.mean(level_err ** 2 - np.array([c['var'] for c in cv_cells]))))),
            'sigma_model_gap': float(np.sqrt(max(0.0, np.mean(gap_err ** 2 - np.array([gap_var[key] for key in g_obs]))))),
            'naive_ratio': naive(cells),
            'in_sample_rmse': float(np.sqrt(np.mean((np.array(pred) - np.array(obs)) ** 2))),
            'cv_level_rmse': float(np.sqrt(np.mean(level_err ** 2))),
            'cv_gap_rmse': float(np.sqrt(np.mean(gap_err ** 2))),
            'cv_gap_sign_hits': f'{sign_hits} of {len(g_obs)}',
            'qa_cells_in_sample_mae': float(np.mean([abs(pred[i] - obs[i]) for i in qa])),
            'pooled': pooled(cells, pred)}


def naive(cells):
    out = {}
    for name, tasks in (('all_tasks', None), ('qa_tasks', ('qa_1', 'qa_2'))):
        for cache in CACHES:
            for budget in BUDGETS:
                sel = [c for c in cells if c['cache'] == cache and c['budget'] == budget and (not tasks or c['task'] in tasks)]
                out[f'{name} {cache}@{budget}'] = float(sum(c['acc'] * c['n'] for c in sel) / sum(c['full'] * c['n'] for c in sel))
    return out


def pooled(cells, pred):
    out = {}
    for cache in CACHES:
        for budget in BUDGETS:
            sel = [i for i, c in enumerate(cells) if c['cache'] == cache and c['budget'] == budget]
            out[f'{cache}@{budget}'] = {'obs': float(np.mean([cells[i]['acc'] for i in sel])),
                                        'pred': float(np.mean([pred[i] for i in sel]))}
    return out


def e1_lifts():
    e1 = json.loads((Q / 'answerblind/e1_result.json').read_text())
    return {m: {(src, b): e1['models'][m]['cells'][f'{src} kvzip@{b}']['lift'] for src in ('squad', 'nq')
                for b in BUDGETS} for m in MODELS}


def e1_tokens():
    out = {}
    for m in MODELS:
        out[m] = {json.loads(line)['id']: json.loads(line)['n_tokens'] for line in open(RECORDS / f'abl_e1_{m}_run.jsonl')}
    return out


def test_set(tokens):
    floor = swc.floor_tokens('1.5625')
    rows = []
    for source, picks in (('squad', lambda i, doc: doc['questions'][2:4]), ('nq', lambda i, doc: doc['questions'] if 100 <= i < 200 else [])):
        for i, doc in enumerate(map(json.loads, open(ROOT / f'data/answerblind/{source}.jsonl'))):
            if min(tokens[m][doc['id']] for m in MODELS) < floor:
                continue
            rows += [{'source': source, 'doc_id': doc['id'], 'qid': q['qid']} for q in picks(i, doc)]
    return rows


def doc_cells(model, rows, tokens, lifts):
    shape, out = MODELS[model], []
    for source in ('squad', 'nq'):
        docs = [r['doc_id'] for r in rows if r['source'] == source]
        for cache, bits in CACHES.items():
            for budget in BUDGETS:
                kept = [kept_per_layer(cache, budget, tokens[model][d]) for d in docs]
                keep = float(np.mean([k / (shape['heads'] * (tokens[model][d] - N_SINK)) for k, d in zip(kept, docs)]))
                out.append({'source': source, 'cache': cache, 'bits': bits, 'budget': budget, 'n': len(docs),
                            'keep': keep, 'lift': lifts[model][(source, budget)],
                            'kept_per_head': float(np.mean(kept)) / shape['heads']})
    return out


def check_keep_arithmetic(model):
    recs = [json.loads(line) for f in sorted(glob.glob(str(RECORDS / f'review_4k_{model}_ruler_[0-9].jsonl')))
            for line in open(f)]
    worst = 0
    for r in recs:
        for cache in CACHES:
            for budget in BUDGETS:
                expect = kept_per_layer(cache, budget, r['n_tokens']) * MODELS[model]['layers']
                worst = max(worst, abs(r['kept'][f'Z.{cache}@{budget}'] - expect))
    return worst


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    tokens, lifts = e1_tokens(), e1_lifts()
    rows = test_set(tokens)
    cal, preds = {}, {'test_set': {'squad': sum(r['source'] == 'squad' for r in rows),
                                   'nq': sum(r['source'] == 'nq' for r in rows), 'rows': rows}, 'models': {}}
    for m in MODELS:
        cal[m] = calibrate(m) | {'keep_arithmetic_max_abs_error': check_keep_arithmetic(m),
                                 'without_routing': {k: v for k, v in calibrate(m, routing=False).items()
                                                     if k in ('k', 'in_sample_rmse', 'cv_level_rmse', 'cv_gap_rmse',
                                                              'cv_gap_sign_hits')}}
        curve, k = flip_curve(m), cal[m]['k']
        cells = doc_cells(m, rows, tokens, lifts)
        for c in cells:
            c['ratio'] = predict(1.0, c['keep'] * c['lift'], c['kept_per_head'], c['bits'], k, curve)
            c['ratio_without_routing'] = predict(1.0, c['keep'] * c['lift'], c['kept_per_head'], c['bits'], k, curve,
                                                 routing=False)
            c['flip'] = flip(curve, c['bits'], c['kept_per_head'])
        preds['models'][m] = {'k': k, 'cells': cells}
        print(m, 'k', round(k, 2), 'cv level rmse', round(cal[m]['cv_level_rmse'], 2), 'cv gap rmse',
              round(cal[m]['cv_gap_rmse'], 2), cal[m]['cv_gap_sign_hits'], 'keep arithmetic error',
              cal[m]['keep_arithmetic_max_abs_error'])
        for c in cells:
            print('  ', c['source'], c['cache'], c['budget'], 'keep %.4f lift %.3f q %.4f n/head %.0f flip %.4f ratio %.3f'
                  % (c['keep'], c['lift'], c['keep'] * c['lift'], c['kept_per_head'], c['flip'], c['ratio']))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / 'calibration.json').write_text(json.dumps(cal, indent=1) + '\n')
    (args.out_dir / 'predictions.json').write_text(json.dumps(preds, indent=1) + '\n')


if __name__ == '__main__':
    main()
