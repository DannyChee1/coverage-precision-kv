import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from am_stack import AM_ROWS, ROWS
from analyze import score
from keys_ablation_analyze import load, two_sided

B, SEED, ALPHA, MARGIN, FULL_FLOOR, EXPECTED = 10_000, 0, 0.025, 1.0, 85.0, 338
PRIMARY = {'P1': ('am_k3v2', 'am_stack4'), 'P2': ('am_k3v2', 'am16')}
SECONDARY = {'S1': ('am_k3v2', 'kvzip_k3pc'), 'S2': ('am16', 'kvzip_k3pc'), 'S3': ('am_stack4', 'am16'),
             'S4': ('am_stack4', 'kvzip_k3pc')}


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

    def contrast(a, b):
        d = draws[a] - draws[b]
        return {'a': a, 'b': b, 'delta': means[a] - means[b], 'p_one_sided': float(np.mean(d <= 0)),
                'p_two_sided': two_sided(d), 'ci95': np.percentile(d, [2.5, 97.5]).tolist()}
    over = [(r['index'], a) for r in rows for a in ROWS[1:]
            if r['bits'][a] > r['budget_bits'][a] or r['over_budget_layers'][a]]
    mean_of = lambda key, arms: {a: float(np.mean([r[key][a] for r in rows])) for a in arms}
    return {'tests': {k: contrast(a, b) for k, (a, b) in PRIMARY.items()},
            'secondary': {k: contrast(a, b) for k, (a, b) in SECONDARY.items()},
            'means': means, 'per_task': {t: {a: float(s[a][t].mean()) for a in ROWS} for t in tasks},
            'budget': {'pass': not over, 'over': over[:20],
                       'kept_vs_am16': {a: float(np.mean([r['kept'][a] / r['kept']['am16'] for r in rows]))
                                        for a in ROWS[1:]},
                       'kept_per_kv_head': {a: float(np.mean([r['kept'][a] / (r['n_layers'] * r['n_kv_heads'])
                                                              for r in rows])) for a in ROWS[1:]},
                       'share_of_full_bytes': {a: float(np.mean([r['bits'][a] / r['full_bits'] for r in rows]))
                                               for a in ROWS[1:]}},
            'key_error': mean_of('key_error', ROWS[1:]), 'value_error': mean_of('value_error', ROWS[1:]),
            'am_output_error': mean_of('am_output_error', AM_ROWS),
            'examples': len(rows), 'models': sorted({r['model'] for r in rows}),
            'seconds_per_example': {k: float(np.mean([r[k] for r in rows]))
                                    for k in ('score_seconds', 'am_seconds', 'decode_seconds')}}


def holm(out):
    tests = sorted((out[m]['tests'][k]['p_one_sided'], m, k) for m in out for k in out[m]['tests'])
    still = True
    for rank, (p, m, k) in enumerate(tests):
        threshold = ALPHA / (len(tests) - rank)
        passed = still and p <= threshold
        still = passed
        t = out[m]['tests'][k]
        t['holm_threshold'], t['holm_pass'] = threshold, bool(passed)
        t['shown'] = bool(passed and t['delta'] >= MARGIN and out[m]['valid'])


def call(out, k):
    shown = [m for m in out if out[m]['tests'][k]['shown']]
    if len(shown) == len(out):
        return 'SHOWN on both models'
    return 'SHOWN on ' + ', '.join(shown) + ' only' if shown else 'NOT SHOWN'


def analyze(models, cpu_equivalence, expected=EXPECTED):
    out = {name: model_summary(rows) for name, rows in models.items()}
    for summary in out.values():
        summary['valid'] = bool(summary['budget']['pass'] and summary['means']['full'] >= FULL_FLOOR
                                and summary['examples'] == expected and cpu_equivalence)
    holm(out)
    return {'calls': {k: call(out, k) for k in PRIMARY}, 'per_model': out,
            'rules': {'B': B, 'seed': SEED, 'alpha_one_sided': ALPHA, 'margin': MARGIN, 'tests': 4,
                      'full_floor': FULL_FLOOR, 'expected_examples': expected,
                      'cpu_equivalence_passed': cpu_equivalence}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', nargs='+', action='append', required=True, metavar=('NAME', 'FILE'))
    parser.add_argument('--cpu-check', type=Path, required=True)
    parser.add_argument('--expected', type=int, default=EXPECTED)
    parser.add_argument('--out')
    args = parser.parse_args()
    cpu = json.loads(args.cpu_check.read_text())
    result = analyze({m[0]: load(m[1:]) for m in args.model}, bool(cpu.get('am16_equals_am_compute_compacted_cache')),
                     args.expected)
    text = json.dumps(result, indent=1)
    if args.out:
        Path(args.out).write_text(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
