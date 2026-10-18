import argparse
import json
from pathlib import Path

import numpy as np

RECORDS = Path(__file__).parents[1] / 'records'
RUNS = {'stage1': ('fr_{}_run.jsonl', ['Z', 'W10', 'W25', 'W50', 'W100']),
        'confirmation': ('fc_{}_conf.jsonl', ['Z', 'W50', 'R50'])}
MODELS = ('llama', 'qwen3')
B, SEED = 10_000, 0


def summarize(pattern, keys):
    rng = np.random.default_rng(SEED)
    cells = {}
    for m in MODELS:
        records = [json.loads(line) for line in open(RECORDS / pattern.format(m))]
        for task in sorted({r['task'] for r in records}):
            rows = [r['frontier'] for r in records if r['task'] == task and r['frontier']['evm_positions'] > 0]
            if not rows:
                continue
            num = {k: np.array([f['evm_kept'][k] for f in rows]) for k in keys}
            den = {k: np.array([f['keep_rate'][k] for f in rows]) for k in keys}
            idx = rng.integers(0, len(rows), size=(B, len(rows)))
            cells[(m, task)] = {'rows': len(rows), 'lift': {k: float(num[k].mean() / den[k].mean()) for k in keys},
                                'draws': {k: num[k][idx].mean(axis=1) / den[k][idx].mean(axis=1) for k in keys}}
    out = {'cells': {f'{m} {t}': {'rows': c['rows'], 'lift': c['lift']} for (m, t), c in cells.items()}}
    for group in ('pooled', *MODELS):
        members = [c for c in cells if group == 'pooled' or c[0] == group]
        lift = {k: float(np.mean([cells[c]['lift'][k] for c in members])) for k in keys}
        draws = {k: np.mean([cells[c]['draws'][k] for c in members], axis=0) for k in keys}
        out[group] = {'lift': lift, 'minus_Z': {k: {'delta': lift[k] - lift['Z'],
                                                    'ci95': np.percentile(draws[k] - draws['Z'], [2.5, 97.5]).tolist()}
                                                for k in keys if k != 'Z'}}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path,
                        default=Path(__file__).parents[1] / 'results/frontier/evidence_lift.json')
    args = parser.parse_args()
    result = {'note': 'Exploratory, not pre-registered; see the module docstring.', 'bootstrap': {'B': B, 'seed': SEED},
              **{name: summarize(*spec) for name, spec in RUNS.items()}}
    args.out.write_text(json.dumps(result, indent=1) + '\n')
    for name in RUNS:
        print(name, {k: (round(v['delta'], 3), [round(x, 3) for x in v['ci95']])
                     for k, v in result[name]['pooled']['minus_Z'].items()},
              {k: round(v, 3) for k, v in result[name]['pooled']['lift'].items()})


if __name__ == '__main__':
    main()
