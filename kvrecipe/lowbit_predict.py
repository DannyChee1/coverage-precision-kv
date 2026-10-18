import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import allocation_model as am
import lowbit as lb

ROOT = Path(__file__).parents[1] / 'results/allocation'
ARMS = {'k3v2_pc': (3, 'pc', 2), 'k2v2_pc': (2, 'pc', 2), 'k2v2_rk': (2, 'rk', 2), 'k3v1_pc': (3, 'pc', 1)}
BUDGET, FULL, RHO, BOOT = 1.5625, 96.0, 4.0, 200


def plan_coverage(arm, n_tokens=am.T_REF, head_dim=128):
    per_head = BUDGET / 100 * n_tokens * 2 * am.part_bits(16)
    left = per_head - am.SINKS * 2 * am.part_bits(16) - lb.plan_overhead(lb.SPEC[arm][3], head_dim)
    return (int(left // lb.entry_bits(arm, head_dim)) + am.SINKS) / n_tokens


def arm_rows(model):
    return {arm: dict(model=model, bench='ruler', f=plan_coverage(arm), kb=kb, kq=kq, vb=vb, chance=0.0, full=FULL)
            for arm, (kb, kq, vb) in ARMS.items()}


def predict(rows):
    (p, _), model = am.fit_all(rows, 'mult', RHO)
    out = {}
    for m in am.MODELS:
        ar = arm_rows(m)
        pred = dict(zip(ar, model.predict(p, list(ar.values()))))
        out[m] = {a: {'f': round(ar[a]['f'], 4), 'score': float(pred[a]), 'delta_vs_k3v2_pc': float(pred[a] - pred['k3v2_pc'])}
                  for a in ar}
    return out, p, model


def bootstrap(rows):
    runs = sorted({r['run'] for r in rows})
    rng = np.random.default_rng(0)
    deltas = {(m, a): [] for m in am.MODELS for a in ARMS}
    for _ in range(BOOT):
        pick = rng.choice(runs, size=len(runs), replace=True)
        boot = [dict(r, run=f'{run}#{i}') for i, run in enumerate(pick) for r in rows if r['run'] == run]
        pred, _, _ = predict(boot)
        for m in am.MODELS:
            for a in ARMS:
                deltas[(m, a)].append(pred[m][a]['delta_vs_k3v2_pc'])
    return {f'{m}.{a}': np.percentile(v, [2.5, 50, 97.5]).round(2).tolist() for (m, a), v in deltas.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = {'budget': BUDGET, 'full_assumed': FULL, 'rho': RHO, 'models': {}}
    for name, table in (('A_committed', 'table.json'), ('B_with_controls', 'table_v2.json')):
        rows = am.usable(json.loads((ROOT / table).read_text()))
        pred, p, model = predict(rows)
        result['models'][name] = {'rows': len(rows), 'params': dict(zip(model.names(), map(float, p))),
                                  'predictions': pred, 'delta_interval_95': bootstrap(rows)}
    rows_a = am.usable(json.loads((ROOT / 'table.json').read_text()))
    (p, _), model = am.fit_all(rows_a, 'mult', RHO)
    ctrl = [r for r in am.usable(json.loads((ROOT / 'table_v2.json').read_text())) if r['run'] == 'controls']
    for r, y in zip(ctrl, model.predict(p, ctrl)):
        r['pred'] = float(y)
        r['in_support'] = am.support(rows_a, r)
    result['model_A_on_controls'] = {
        'all': am.metrics(ctrl), 'in_support': am.metrics([r for r in ctrl if r['in_support']]),
        'rows': [{k: r[k] for k in ('model', 'bench', 'arm', 'f', 'score', 'pred', 'in_support')} for r in ctrl]}
    args.out.write_text(json.dumps(result, indent=1) + '\n')
    for name, res in result['models'].items():
        for m in am.MODELS:
            print(name, m, {a: (round(v['score'], 2), round(v['delta_vs_k3v2_pc'], 2), res['delta_interval_95'][f'{m}.{a}'])
                            for a, v in res['predictions'][m].items()})
    print('model A on controls:', json.dumps(result['model_A_on_controls']['all']))
    print('  in support:', json.dumps(result['model_A_on_controls']['in_support']))


if __name__ == '__main__':
    main()
