import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import controls_analyze as ca
import tca_run as run

ALPHA, BUDGETS, BENCHES = 0.05, ('1.5625', '3.125'), ('ruler', 'quality')
TERMS = ('C', 'V', 'S', 'R')


def pooled(cells, a, b, sided):
    delta = float(np.mean([acc[f'Z.{a}@{bud}'] - acc[f'Z.{b}@{bud}'] for acc, _, bud in cells]))
    d = np.mean([dr[f'Z.{a}@{bud}'] - dr[f'Z.{b}@{bud}'] for _, dr, bud in cells], axis=0)
    low, high = float(np.mean(d <= 0)), float(np.mean(d >= 0))
    return {'delta': delta, 'p': low if sided == 1 else min(1.0, 2 * min(low, high)), 'sided': sided,
            'ci95': np.percentile(d, [2.5, 97.5]).tolist()}


def holm(tests, alpha):
    still = True
    for rank, (key, t) in enumerate(sorted(tests.items(), key=lambda kv: kv[1]['p'])):
        threshold = alpha / (len(tests) - rank)
        passed = still and t['p'] <= threshold
        still = passed
        t.update(holm_threshold=threshold, holm_pass=bool(passed), passes=bool(passed and t['delta'] >= ca.MARGIN))
    return tests


def predicted(records, weights, budget, arm):
    w = np.array([weights[k] for k in ('alpha', 'nu', 'kappa', 'rho')])
    d = [np.array([r['tca'][f'Z.k3v2pc@{budget}'][t] - r['tca'][f'Z.{arm}@{budget}'][t] for t in TERMS])
         for r in records]
    return 100 * float(np.mean([w @ x for x in d]))


def allocation(records, budget):
    out = {}
    for arm in ('tcafit', 'tca2', 'rdkvpc'):
        a = f'Z.{arm}@{budget}'
        v = np.sum([r['tca'][a]['hist']['v'] for r in records], axis=0)
        k = np.sum([r['tca'][a]['hist']['k'] for r in records], axis=0)
        out[arm] = {'value_bits_share': (v / v.sum()).round(4).tolist(),
                    'key_bits_share': (k / k.sum()).round(4).tolist(),
                    'terms': {t: float(np.mean([r['tca'][a][t] for r in records])) for t in TERMS}}
        if 'shares' in records[0]['tca'][a]:
            out[arm]['key_share_mean'] = float(np.mean([np.mean(r['tca'][a]['shares']) for r in records]))
    out['k3v2pc_terms'] = {t: float(np.mean([r['tca'][f'Z.k3v2pc@{budget}'][t] for r in records])) for t in TERMS}
    return out


def leave_one_out(cells, keys, a, b):
    out = {}
    for i, key in enumerate(keys):
        t = pooled([c for j, c in enumerate(cells) if j != i], a, b, 1)
        out[key] = {'delta': t['delta'], 'p': t['p'], 'ci95': t['ci95']}
    return out


def analyze(models, weights, primary, ruler_expected=650):
    run.configure('test')
    ca.EXPECTED['ruler'] = (ruler_expected, ruler_expected)
    cells, keys, per, valid_all, per_cell, harm, extras = [], [], {}, True, {}, [], {}
    for m, recs in models.items():
        per[m] = {}
        for bench in BENCHES:
            rs = [r for r in recs if r['benchmark'] == bench]
            if not rs:
                valid_all = False
                continue
            out = ca.summary(rs, bench)
            acc, draws = out.pop('_acc_draws')
            valid_all &= out['valid_all']
            per[m][bench] = out
            for b in BUDGETS:
                cells.append((acc, draws, b))
                key = f'{m} {bench} @{b}'
                keys.append(key)
                for comp in ('rdkv', 'k3v2pc'):
                    terms = [(f'Z.{primary}@{b}', 1), (f'Z.{comp}@{b}', -1)]
                    per_cell[f'{key} {primary}-{comp}'] = ca.contrast(acc, draws, terms, 1)
                two = ca.contrast(acc, draws, [(f'Z.{primary}@{b}', 1), (f'Z.k3v2pc@{b}', -1)], 2)
                if two['ci95'][1] < -ca.MARGIN:
                    harm.append(key)
                extras[key] = {
                    f'{primary}-k3v2pc two-sided': two,
                    'tcafit-tca2 two-sided': ca.contrast(acc, draws, [(f'Z.tcafit@{b}', 1), (f'Z.tca2@{b}', -1)], 2),
                    'rdkvpc-rdkv': ca.contrast(acc, draws, [(f'Z.rdkvpc@{b}', 1), (f'Z.rdkv@{b}', -1)], 2),
                    'rdkvpc-k3v2pc': ca.contrast(acc, draws, [(f'Z.rdkvpc@{b}', 1), (f'Z.k3v2pc@{b}', -1)], 2),
                    f'predicted {primary}-k3v2pc': predicted(rs, weights, b, primary),
                    'allocation': allocation(rs, b)}
    tests = holm({f'D1 {primary}-rdkv': pooled(cells, primary, 'rdkv', 1),
                  f'D2 {primary}-k3v2pc': pooled(cells, primary, 'k3v2pc', 1)}, ALPHA)
    passed = all(t['passes'] for t in tests.values()) and valid_all and len(cells) == 8 and not harm
    loo_d2 = leave_one_out(cells, keys, primary, 'k3v2pc')
    concentrated = [k for k, t in loo_d2.items() if t['delta'] < ca.MARGIN or t['p'] > 0.05]
    return {'call': 'PASS' if passed else 'KILL', 'primary_arm': primary, 'primary': tests,
            'valid_all': bool(valid_all), 'harm_cells': harm,
            'secondary': {'per_cell_holm16': holm(per_cell, ALPHA),
                          'D3 tcafit-tca2 (routing)': pooled(cells, 'tcafit', 'tca2', 2),
                          f'{primary}-rdkvpc': pooled(cells, primary, 'rdkvpc', 1),
                          'leave_one_out_D1': leave_one_out(cells, keys, primary, 'rdkv'),
                          'leave_one_out_D2': loo_d2, 'concentrated_in': concentrated,
                          'rdkvpc-rdkv': pooled(cells, 'rdkvpc', 'rdkv', 2),
                          'rdkvpc-k3v2pc': pooled(cells, 'rdkvpc', 'k3v2pc', 2),
                          'cells': extras},
            'per_model': per, 'weights': weights,
            'rule': {'alpha': ALPHA, 'margin': ca.MARGIN, 'bootstrap': {'B': ca.B, 'seed': ca.SEED},
                     'ruler_expected': ruler_expected}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', nargs='+', action='append', required=True, metavar=('NAME', 'FILE'))
    parser.add_argument('--fit', type=Path, required=True)
    parser.add_argument('--primary', type=Path, required=True, help='tca/primary.json from tca_check.py')
    parser.add_argument('--ruler-expected', type=int, default=650)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = analyze({name: ca.load(files) for name, *files in args.model}, json.loads(args.fit.read_text())['weights'],
                  json.loads(args.primary.read_text())['primary'], args.ruler_expected)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps({k: out[k] for k in ('call', 'primary_arm', 'primary', 'valid_all', 'harm_cells')}, indent=1))


if __name__ == '__main__':
    main()
