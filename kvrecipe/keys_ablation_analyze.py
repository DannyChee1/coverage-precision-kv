import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from analyze import score
from keys_ablation import ARMS

B, SEED, ALPHA, MARGIN, FULL_FLOOR, EXPECTED = 10_000, 0, 0.05, 1.0, 85.0, 650
THREE_BIT = ('k3_plain', 'k3_centered', 'k3_rotated', 'rk3v2', 'k3_pc')
CONTRASTS = {'C1': ('rk3v2', 'k3_rotated'), 'C2': ('rk3v2', 'k3_centered'), 'C3': ('k3_centered', 'k3_plain'),
             'C4': ('k3_rotated', 'k3_plain'), 'C5': ('k3_pc', 'rk3v2'), 'C6': ('k4v2_pc', 'stack4_pc'),
             'C7': ('best_3bit', 'k4v2_pc')}
SECONDARY = {'S1': ('k3_plain', 'k4v2'), 'S2': ('rk3v2', 'k4v2'), 'S3': ('k3_pc', 'k4v2'), 'S4': ('k4v2_pc', 'k4v2'),
             'S5': ('k3_pc', 'k3_plain')}


def load(paths):
    rows = {}
    for p in paths:
        for line in open(p):
            r = json.loads(line)
            rows[r['index']] = r
    return sorted(rows.values(), key=lambda r: (r['task'], r['index']))


def two_sided(d):
    return float(min(1.0, 2 * min(np.mean(d <= 0), np.mean(d >= 0))))


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
    best = max(THREE_BIT, key=lambda a: means[a])
    stacked = np.stack([draws[a] for a in THREE_BIT])
    draws['best_3bit'] = stacked.max(axis=0)  # the best 3-bit arm, chosen again in every replicate
    means['best_3bit'] = means[best]

    def contrast(a, b):
        d = draws[a] - draws[b]
        return {'a': a, 'b': b, 'delta': means[a] - means[b], 'p': two_sided(d),
                'ci95': np.percentile(d, [2.5, 97.5]).tolist()}
    tests = {k: contrast(a, b) for k, (a, b) in CONTRASTS.items()}
    tests['C7']['best_3bit_arm'] = best
    tests['C7']['best_arm_share_of_replicates'] = {a: float(np.mean(stacked.argmax(axis=0) == i))
                                                   for i, a in enumerate(THREE_BIT)}
    inter = (draws['rk3v2'] - draws['k3_rotated']) - (draws['k3_centered'] - draws['k3_plain'])
    interaction = {'delta': tests['C1']['delta'] - tests['C3']['delta'], 'ci95': np.percentile(inter, [2.5, 97.5]).tolist()}
    over = [(r['index'], a) for r in rows for a in ARMS[1:]
            if r['bits'][a] > r['budget_bits'][a] or r['over_budget_layers'][a]]
    same3 = all(len({r['kept'][a] for a in THREE_BIT}) == 1 for r in rows)
    mean_of = lambda key: {a: float(np.mean([r[key][a] for r in rows])) for a in ARMS[1:]}
    return {'tests': tests, 'interaction_C1_minus_C3': interaction,
            'secondary': {k: contrast(a, b) for k, (a, b) in SECONDARY.items()},
            'means': means, 'per_task': {t: {a: float(s[a][t].mean()) for a in ARMS} for t in tasks},
            'budget': {'pass': not over, 'over': over[:20], 'three_bit_identical_entries': same3,
                       'kept_vs_k4v2': {a: float(np.mean([r['kept'][a] / r['kept']['k4v2'] for r in rows]))
                                        for a in ARMS[1:]},
                       'share_of_full_bytes': {a: float(np.mean([r['bits'][a] / r['full_bits'] for r in rows]))
                                               for a in ARMS[1:]}},
            'key_error': mean_of('key_error'), 'value_error': mean_of('value_error'),
            'examples': len(rows), 'models': sorted({r['model'] for r in rows}),
            'seconds_per_example': float(np.mean([r['score_seconds'] + r['decode_seconds'] for r in rows]))}


def holm(out):
    tests = sorted((out[m]['tests'][k]['p'], m, k) for m in out for k in out[m]['tests'])
    still = True
    for rank, (p, m, k) in enumerate(tests):
        threshold = ALPHA / (len(tests) - rank)
        passed = still and p <= threshold
        still = passed
        t = out[m]['tests'][k]
        t['holm_threshold'], t['holm_pass'] = threshold, bool(passed)
        t['shown'] = bool(passed and abs(t['delta']) >= MARGIN and out[m]['valid'])
        t['shown_sign'] = int(np.sign(t['delta'])) if t['shown'] else 0


def readings(out):
    models = list(out)
    sign = lambda m, k: out[m]['tests'][k]['shown_sign']
    both = lambda k, s: all(sign(m, k) == s for m in models)
    either = lambda k, s: any(sign(m, k) == s for m in models)
    held = []
    if both('C3', 1) and not either('C2', 1):
        held.append('centering is the active ingredient')
    if both('C2', 1):
        held.append('rotation adds after centering')
    if all(out[m]['interaction_C1_minus_C3']['ci95'][0] > 0 for m in models):
        held.append('the combination matters (interaction)')
    quantizer = ('C1', 'C2', 'C3', 'C4', 'C5')
    if (not any(sign(m, k) for m in models for k in quantizer)
            and all(out[m]['secondary']['S1']['ci95'][0] > 0 for m in models)):
        held.append('mostly coverage')
    if both('C5', 1) or (not either('C5', -1) and not either('C2', 1)):
        held.append('the per-channel quantizer is enough')

    def call(k, pos, neg):
        if either(k, -1):
            return neg + ' (' + ', '.join(m for m in models if sign(m, k) == -1) + ')'
        if both(k, 1):
            return pos + ' on both models'
        if either(k, 1):
            return pos + ' on ' + ', '.join(m for m in models if sign(m, k) == 1) + ' only'
        return 'NOT SHOWN'
    return {'quantizer_readings': held or ['MIXED'],
            'C6': call('C6', 'SURVIVES', 'REVERSES'), 'C7': call('C7', 'ADD', 'LOSE')}


def analyze(models, expected=EXPECTED):
    out = {name: model_summary(rows) for name, rows in models.items()}
    for summary in out.values():
        summary['valid'] = bool(summary['budget']['pass'] and summary['budget']['three_bit_identical_entries']
                                and summary['means']['full'] >= FULL_FLOOR and summary['examples'] == expected)
    holm(out)
    return {'readings': readings(out), 'per_model': out,
            'rules': {'B': B, 'seed': SEED, 'alpha_two_sided': ALPHA, 'margin': MARGIN, 'tests': 14,
                      'full_floor': FULL_FLOOR, 'expected_examples': expected}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', nargs='+', action='append', required=True, metavar=('NAME', 'FILE'))
    parser.add_argument('--expected', type=int, default=EXPECTED)
    parser.add_argument('--out')
    args = parser.parse_args()
    result = analyze({m[0]: load(m[1:]) for m in args.model}, args.expected)
    text = json.dumps(result, indent=1)
    if args.out:
        Path(args.out).write_text(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
