import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import controls_analyze as ca
import review4k_analyze as r4
from longbench_metrics import score_example

ALPHA = 0.025
BUDGETS = r4.BUDGETS
TASKS, PER_TASK = ('gov_report', 'qmsum'), 100
ROWS = ['full', *[f'Z.{s}@{b}' for b in BUDGETS for s in ('evict16', 'stack4pc', 'k3v2pc')]]


def degenerate(text, n=4, times=4):
    words = text.split()
    grams = Counter(tuple(words[i:i + n]) for i in range(len(words) - n + 1))
    return bool(grams) and max(grams.values()) >= times


def task_draws(records):
    s = {a: {t: np.array([100 * r['scores'][a] for r in records if r['task'] == t])
             for t in TASKS} for a in ROWS}
    acc = {a: float(np.mean([s[a][t].mean() for t in TASKS])) for a in ROWS}
    rng = np.random.default_rng(ca.SEED)
    draws = {a: np.zeros(ca.B) for a in ROWS}
    for t in TASKS:
        idx = rng.integers(0, len(s[ROWS[0]][t]), size=(ca.B, len(s[ROWS[0]][t])))
        for a in ROWS:
            draws[a] += s[a][t][idx].mean(axis=1) / len(TASKS)
    return acc, draws, {a: {t: float(s[a][t].mean()) for t in TASKS} for a in ROWS}


def summary(records):
    over = [(r['id'], a) for r in records for a in r['bits']
            if r['bits'][a] > r['budget_bits'][a] or r['over_budget_layers'][a]]
    counts = Counter(r['task'] for r in records)
    valid = {'budget': not over, 'complete': all(counts[t] == PER_TASK for t in TASKS) and len(records) == 2 * PER_TASK,
             'no_think': not any(r['think']['full'] for r in records)}
    acc, draws, per_task = task_draws(records)
    return {'accuracy': acc, 'per_task': per_task, 'valid': valid, 'valid_all': all(valid.values()),
            'records': len(records), 'budget': {'pass': not over, 'over': over[:20]},
            'kept': {a: float(np.mean([r['kept'][a] for r in records])) for a in records[0]['kept']},
            'out_tokens': {a: float(np.mean([r['out_tokens'][a] for r in records])) for a in ROWS},
            'degenerate': {a: float(np.mean([r['degenerate'][a] for r in records])) for a in ROWS},
            'truncated': int(sum(bool(r.get('truncated')) for r in records))}, (acc, draws)


def analyze(models):
    per, cells, secondary = {}, [], {}
    for m, recs in models.items():
        rs = [r for r in recs if r['benchmark'] == 'longbench']
        if not rs:
            continue
        per[m], (acc, d) = summary(rs)
        for b in BUDGETS:
            cells.append((f'{m} C @{b}', ca.contrast(acc, d, [(f'Z.k3v2pc@{b}', 1), (f'Z.stack4pc@{b}', -1)], 1),
                          per[m]['valid_all']))
            for arm in ('k3v2pc', 'stack4pc'):
                secondary[f'{m} {arm}-evict16 @{b}'] = ca.contrast(
                    acc, d, [(f'Z.{arm}@{b}', 1), (f'Z.evict16@{b}', -1)], 2)
    tests = r4.holm(cells, ALPHA)
    calls = {f'C @{b}': 'HOLDS ON GENERATION' if all(f'{m} C @{b}' in tests and tests[f'{m} C @{b}']['passes']
                                                     for m in models) else 'NOT SHOWN' for b in BUDGETS}
    return {'calls': calls, 'tests': tests, 'secondary_vs_evict16': secondary, 'per_model': per,
            'rule': {'alpha': ALPHA, 'margin': ca.MARGIN, 'tests': len(cells), 'sided': 1, 'tasks': TASKS,
                     'per_task': PER_TASK, 'bootstrap': {'B': ca.B, 'seed': ca.SEED}}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', nargs='+', action='append', required=True, metavar=('NAME', 'FILE'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    models = {name: list({r['id']: r for f in files for r in map(json.loads, open(f))}.values())
              for name, *files in args.model}
    out = analyze(models)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps(out['calls'], indent=1))


if __name__ == '__main__':
    main()
