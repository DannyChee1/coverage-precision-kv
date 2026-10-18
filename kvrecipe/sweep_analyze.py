import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from analyze import score
from sweep import ARMS, BUDGETS

B, SEED, ALPHA, FULL_FLOOR, EXPECTED = 10_000, 0, 0.05, 85.0, 650


def load(paths):
    rows = {}
    for p in paths:
        for line in open(p):
            r = json.loads(line)
            rows[r['index']] = r
    return sorted(rows.values(), key=lambda r: (r['task'], r['index']))


def crossover(deltas):
    xs = [math.log2(BUDGETS[b]) for b in BUDGETS]
    for i in range(len(deltas) - 1):
        if deltas[i] > 0 >= deltas[i + 1]:
            x = xs[i] + (xs[i + 1] - xs[i]) * deltas[i] / (deltas[i] - deltas[i + 1])
            return 100 * 2 ** x
    return None


def model_summary(rows):
    tasks = sorted({r['task'] for r in rows})
    s = {a: {t: np.array([r['scores'][a] for r in rows if r['task'] == t]) for t in tasks}
         for a in ARMS}
    means = {a: float(np.mean([s[a][t].mean() for t in tasks])) for a in ARMS}
    rng = np.random.default_rng(SEED)
    draws = {a: np.zeros(B) for a in ARMS}
    for t in tasks:
        idx = rng.integers(0, len(s['full'][t]), size=(B, len(s['full'][t])))
        for a in ARMS:
            draws[a] += s[a][t][idx].mean(axis=1) / len(tasks)
    cells, curves = {}, []
    for b in BUDGETS:
        d = draws[f'k4v2@{b}'] - draws[f'stack4@{b}']
        curves.append(d)
        cells[b] = {'delta': means[f'k4v2@{b}'] - means[f'stack4@{b}'],
                    'p_two_sided': float(min(1.0, 2 * min(np.mean(d <= 0), np.mean(d >= 0)))),
                    'ci95': np.percentile(d, [2.5, 97.5]).tolist(),
                    'stack4_gap_to_full': means['full'] - means[f'stack4@{b}'],
                    'k4v2_gap_to_full': means['full'] - means[f'k4v2@{b}']}
    point = crossover([cells[b]['delta'] for b in BUDGETS])
    boot = [crossover([c[i] for c in curves]) for i in range(B)]
    found = [x for x in boot if x is not None]
    over = [(r['index'], a) for r in rows for a in ARMS[1:]
            if r['bits'][a] > r['budget_bits'][a] or r['over_budget_layers'][a]]
    return {'cells': cells, 'means': means, 'per_task': {t: {a: float(s[a][t].mean()) for a in ARMS} for t in tasks},
            'crossover_percent_of_bytes': point,
            'crossover_bootstrap': {'share_with_crossover': len(found) / B,
                                    'ci95': np.percentile(found, [2.5, 97.5]).tolist() if found else None},
            'budget': {'pass': not over, 'over': over[:20],
                       'kept_k4v2_vs_stack4': {b: float(np.mean([r['kept'][f'k4v2@{b}'] / r['kept'][f'stack4@{b}']
                                                                 for r in rows])) for b in BUDGETS},
                       'share_of_full_bytes': {a: float(np.mean([r['bits'][a] / r['full_bits'] for r in rows]))
                                               for a in ARMS[1:]}},
            'examples': len(rows), 'models': sorted({r['model'] for r in rows}),
            'seconds_per_example': float(np.mean([r['score_seconds'] + r['decode_seconds'] for r in rows]))}


def pattern(cells):
    order = list(BUDGETS)
    better = [b for b in order if cells[b]['verdict'] == 'k4v2 better']
    worse = [b for b in order if cells[b]['verdict'] == '4-bit stack better']
    if not better and not worse:
        return 'FLAT'
    if any(order.index(w) < order.index(b) for w in worse for b in better):
        return 'REVERSED'
    if better and worse:
        return 'CROSSOVER'
    return 'NO CROSSOVER IN RANGE' if better else 'STACK BETTER ONLY'


def analyze(models, expected=EXPECTED):
    out = {name: model_summary(rows) for name, rows in models.items()}
    tests = sorted(((out[m]['cells'][b]['p_two_sided'], m, b) for m in out for b in BUDGETS))
    still = True
    for rank, (p, m, b) in enumerate(tests):
        threshold = ALPHA / (len(tests) - rank)
        passed = still and p <= threshold
        still = passed
        cell = out[m]['cells'][b]
        cell['holm_threshold'], cell['holm_pass'] = threshold, bool(passed)
        cell['verdict'] = ('no detectable difference' if not passed else
                           'k4v2 better' if cell['delta'] > 0 else '4-bit stack better')
    for m, summary in out.items():
        summary['valid'] = bool(summary['budget']['pass'] and summary['means']['full'] >= FULL_FLOOR
                                and summary['examples'] == expected)
        summary['call'] = pattern(summary['cells']) if summary['valid'] else 'NOT VALID'
    return {'per_model': out, 'calls': {m: out[m]['call'] for m in out}, 'bootstrap': {'B': B, 'seed': SEED},
            'holm': {'alpha': ALPHA, 'tests': len(tests)}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', nargs='+', action='append', required=True, metavar=('NAME', 'FILE'))
    parser.add_argument('--out')
    args = parser.parse_args()
    out = analyze({name: load(files) for name, *files in args.model})
    text = json.dumps(out, indent=1)
    if args.out:
        Path(args.out).write_text(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
