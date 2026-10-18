import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from analyze import score
from keys_ablation_analyze import load, two_sided
from lowbit import ROWS, SPEC

B, SEED, ALPHA, MARGIN, FULL_FLOOR, EXPECTED = 10_000, 0, 0.025, 1.0, 85.0, 650
REF = 'k3v2_pc'
ARMS = tuple(r for r in SPEC if r != REF)
SHARED = tuple(r for r, s in SPEC.items() if s[3] == 'e640')


def model_summary(rows):
    tasks = sorted({r['task'] for r in rows})
    s = {a: {t: np.array([r['scores'][a] for r in rows if r['task'] == t]) for t in tasks}
         for a in ROWS}
    means = {a: float(np.mean([s[a][t].mean() for t in tasks])) for a in ROWS}
    rng = np.random.default_rng(SEED)
    draws = {a: np.zeros(B) for a in ROWS}
    for t in tasks:
        idx = rng.integers(0, len(s['full'][t]), size=(B, len(s['full'][t])))
        for a in ROWS:
            draws[a] += s[a][t][idx].mean(axis=1) / len(tasks)
    tests = {}
    for a in ARMS:
        d = draws[a] - draws[REF]
        tests[a] = {'delta': means[a] - means[REF], 'p_one_sided': float(np.mean(d <= 0)), 'p_two_sided': two_sided(d),
                    'ci95': np.percentile(d, [2.5, 97.5]).tolist()}
    over = [(r['index'], a) for r in rows for a in ROWS[1:]
            if r['bits'][a] > r['budget_bits'][a] or r['over_budget_layers'][a]]
    same = all(len({r['kept'][a] for a in SHARED}) == 1 for r in rows)
    mean_of = lambda key: {a: float(np.mean([r[key][a] for r in rows])) for a in ROWS[1:]}
    return {'tests': tests, 'means': means, 'per_task': {t: {a: float(s[a][t].mean()) for a in ROWS} for t in tasks},
            'budget': {'pass': not over, 'over': over[:20], 'shared_plan_identical_entries': same,
                       'kept_vs_ref': {a: float(np.mean([r['kept'][a] / r['kept'][REF] for r in rows])) for a in ARMS},
                       'kept_fraction': {a: float(np.mean([(r['kept'][a] + 4 * r['n_layers'] * r['n_kv_heads'])
                                                           / (r['n_layers'] * r['n_kv_heads'] * r['n_tokens'])
                                                           for r in rows])) for a in SPEC},
                       'share_of_full_bytes': {a: float(np.mean([r['bits'][a] / r['full_bits'] for r in rows]))
                                               for a in ROWS[1:]}},
            'key_error': mean_of('key_error'), 'value_error': mean_of('value_error'),
            'examples': len(rows), 'models': sorted({r['model'] for r in rows}),
            'seconds_per_example': float(np.mean([r['score_seconds'] + r['decode_seconds'] for r in rows]))}


def holm(out):
    tests = sorted((out[m]['tests'][a]['p_one_sided'], m, a) for m in out for a in out[m]['tests'])
    still = True
    for rank, (p, m, a) in enumerate(tests):
        threshold = ALPHA / (len(tests) - rank)
        passed = still and p <= threshold
        still = passed
        t = out[m]['tests'][a]
        t['holm_threshold'], t['holm_pass'] = threshold, bool(passed)
        t['shown_better'] = bool(passed and t['delta'] >= MARGIN and out[m]['valid'])


def score_predictions(out, predictions):
    scored = {}
    for name, pm in predictions['models'].items():
        rows = []
        for m in out:
            for a in ARMS:
                pred = pm['predictions'][m][a]['delta_vs_k3v2_pc']
                lo, _, hi = pm['delta_interval_95'][f'{m}.{a}']
                obs = out[m]['tests'][a]['delta']
                rows.append({'model': m, 'arm': a, 'predicted': pred, 'observed': obs, 'error': pred - obs,
                             'interval95': [lo, hi], 'observed_in_interval': bool(lo <= obs <= hi),
                             'sign_agrees': bool(np.sign(pred) == np.sign(obs)),
                             'predicted_beats_by_1': bool(pred >= MARGIN),
                             'shown_better': out[m]['tests'][a]['shown_better']})
        err = np.array([r['error'] for r in rows])
        scored[name] = {'rows': rows, 'mae': float(np.abs(err).mean()), 'bias': float(err.mean()),
                        'sign_agreement': float(np.mean([r['sign_agrees'] for r in rows])),
                        'in_interval': float(np.mean([r['observed_in_interval'] for r in rows])),
                        'call_agreement': float(np.mean([r['predicted_beats_by_1'] == r['shown_better'] for r in rows]))}
    return scored


def analyze(models, predictions=None, expected=EXPECTED):
    out = {name: model_summary(rows) for name, rows in models.items()}
    for summary in out.values():
        summary['valid'] = bool(summary['budget']['pass'] and summary['budget']['shared_plan_identical_entries']
                                and summary['means']['full'] >= FULL_FLOOR and summary['examples'] == expected)
    holm(out)
    shown = [f'{a} on {m}' for m in out for a in ARMS if out[m]['tests'][a]['shown_better']]
    result = {'call': ('SHOWN: ' + ', '.join(shown)) if shown else 'NOT SHOWN: no arm beats k3v2_pc by >= 1 pt',
              'per_model': out,
              'rules': {'B': B, 'seed': SEED, 'alpha_one_sided': ALPHA, 'margin': MARGIN, 'tests': 2 * len(ARMS),
                        'full_floor': FULL_FLOOR, 'expected_examples': expected}}
    if predictions:
        result['prediction_scoring'] = score_predictions(out, predictions)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', nargs='+', action='append', required=True, metavar=('NAME', 'FILE'))
    parser.add_argument('--predictions', type=Path)
    parser.add_argument('--expected', type=int, default=EXPECTED)
    parser.add_argument('--out')
    args = parser.parse_args()
    predictions = json.loads(args.predictions.read_text()) if args.predictions else None
    result = analyze({m[0]: load(m[1:]) for m in args.model}, predictions, args.expected)
    text = json.dumps(result, indent=1)
    if args.out:
        Path(args.out).write_text(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
