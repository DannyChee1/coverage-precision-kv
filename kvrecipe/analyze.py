import argparse
import json
from pathlib import Path
import re

import numpy as np

ARMS = ('full', 'evict16', 'stack4', 'stack2', 'joint')
SIMPLE = ('evict16', 'stack4', 'stack2')
B, SEED = 10_000, 0
MARGIN = 1.0
CONTROL = re.compile(r'[\x00-\x1f]')


def score(task, prediction, refs):
    pred = CONTROL.sub('', prediction.strip()).strip().lower()
    hits = [1.0 if r.lower() in pred else 0.0 for r in refs]
    return 100 * (max(hits) if task.split('_')[0] == 'qa' else sum(hits) / len(hits))


def load(paths):
    rows = {}
    for path in paths:
        for line in open(path):
            r = json.loads(line)
            rows[r['index']] = r
    return sorted(rows.values(), key=lambda r: (r['task'], r['index']))


def per_example(rows):
    tasks = sorted({r['task'] for r in rows})
    s = {a: {t: np.array([r['scores'][a] for r in rows if r['task'] == t]) for t in tasks}
         for a in ARMS}
    return tasks, s


def mean(values_by_task, tasks):
    return float(np.mean([values_by_task[t].mean() for t in tasks]))


def bootstrap(s, tasks):
    rng = np.random.default_rng(SEED)
    draws = {a: np.zeros(B) for a in ARMS}
    for t in tasks:
        n = len(s[ARMS[0]][t])
        idx = rng.integers(0, n, size=(B, n))
        for a in ARMS:
            draws[a] += s[a][t][idx].mean(axis=1) / len(tasks)
    return draws


def interval(estimate, draws):
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {'estimate': estimate, 'ci95': [float(lo), float(hi)]}


def budget_check(rows):
    over = [(r['index'], a) for r in rows for a in ARMS[1:] if r['bits'][a] > r['budget_bits']
            or r.get('over_budget_layers', {}).get(a, 0)]
    share = {a: float(np.mean([r['bits'][a] / r['full_bits'] for r in rows])) for a in ARMS}
    tiers = {a: {b: int(sum(r['tiers'][a][b] for r in rows)) for b in rows[0]['tiers'][a]} for a in ARMS[1:]}
    return {'pass': not over, 'over_budget': over[:20], 'mean_share_of_full_bytes': share, 'tier_counts': tiers}


def analyze(rows):
    tasks, s = per_example(rows)
    means = {a: mean(s[a], tasks) for a in ARMS}
    draws = bootstrap(s, tasks)
    best = max(SIMPLE, key=means.get)
    best_draws = np.max([draws[a] for a in SIMPLE], axis=0)
    delta = interval(means['joint'] - means[best], draws['joint'] - best_draws)
    budget = budget_check(rows)
    call = 'CONTINUE' if delta['estimate'] >= MARGIN and delta['ci95'][0] > 0 else 'KILL'
    return {
        'call': call if budget['pass'] else 'NOT VALID (budget)', 'call_if_budget_ignored': call,
        'primary': {'best_simple_stack': best, 'joint_minus_best_simple': delta, 'margin': MARGIN},
        'means': means,
        'per_task': {t: {a: float(s[a][t].mean()) for a in ARMS} for t in tasks},
        'secondary': {
            **{f'joint_minus_{a}': interval(means['joint'] - means[a], draws['joint'] - draws[a]) for a in SIMPLE},
            **{f'{a}_minus_full': interval(means[a] - means['full'], draws[a] - draws['full']) for a in ARMS[1:]}},
        'budget': budget,
        'examples': len(rows), 'examples_per_task': {t: len(s['full'][t]) for t in tasks},
        'dtype': sorted({r['dtype'] for r in rows}),
        'attn_implementation': sorted({r['attn_implementation'] for r in rows}),
        'seconds_per_example': float(np.mean([r['score_seconds'] + r['decode_seconds'] for r in rows])),
        'bootstrap': {'B': B, 'seed': SEED}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('files', nargs='+')
    parser.add_argument('--out')
    args = parser.parse_args()
    out = analyze(load(args.files))
    text = json.dumps(out, indent=1)
    if args.out:
        Path(args.out).write_text(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
