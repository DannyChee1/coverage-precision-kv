import argparse
import glob
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
from analyze import score

ROOT = Path(__file__).parents[1] / 'results'
RECORDS = Path(__file__).parents[1] / 'records'
MODELS = ('llama', 'qwen3')
ARMS = {'full': 'full', 'four': 'Z.stack4pc@1.5625', 'recipe': 'Z.k3v2pc@1.5625'}
GROUPS = {'qa': ('qa_1', 'qa_2'), 'niah': tuple(f'niah_{t}' for t in (
    'single_1', 'single_2', 'single_3', 'multikey_1', 'multikey_2', 'multikey_3', 'multiquery', 'multivalue')),
    'aggregation': ('vt', 'cwe', 'fwe')}
MIN_GAP, B, SEED = 5.0, 10_000, 0
SMOKE = {json.loads(line)['index'] for path in glob.glob(str(Path(__file__).parents[1] / 'data/ruler-*/smoke.jsonl'))
         for line in open(path)}  # smoke rows share the record files; the analyses exclude them


def accuracy(m):
    rows = {}
    for pattern in (f'review_4k_{m}_ruler_*.jsonl', f'rdkv_{m}_ruler_*.jsonl'):
        for path in sorted(glob.glob(str(RECORDS / pattern))):
            for line in open(path):
                r = json.loads(line)
                if r['index'] not in SMOKE:
                    rows.setdefault(r['index'], r)
    out = {}
    for r in rows.values():
        out.setdefault(r['task'], []).append([r['scores'][a] for a in ARMS.values()])
    return {t: np.array(v) for t, v in out.items()}


def lift(m):
    out = {}
    for path in (RECORDS / f'fc_{m}_conf.jsonl', RECORDS / f'bd_{m}_run.jsonl'):
        for line in open(path):
            r = json.loads(line)
            f = r.get('frontier') or {}
            if r.get('benchmark') == 'ruler' and f.get('gold_positions', 0) > 0 and r['index'] not in SMOKE:
                out.setdefault(r['task'], []).append((f['gold_kept']['Z'], f['keep_rate']['Z']))
    return {t: {'rows': len(v), 'lift': float(np.mean([a for a, _ in v]) / np.mean([b for _, b in v]))}
            for t, v in out.items()}


def niah_lift(m):
    rows = [r['frontier'] for r in map(json.loads, open(RECORDS / f'fc_{m}_conf.jsonl'))
            if r['task'].startswith('niah') and r['frontier']['gold_positions'] > 0 and r['index'] not in SMOKE]
    return {'rows': len(rows), 'lift': float(np.mean([f['gold_kept']['Z'] for f in rows])
                                            / np.mean([f['keep_rate']['Z'] for f in rows]))}


def recovered(acc, tasks):
    gain = sum(acc[t][:, 2].mean() - acc[t][:, 1].mean() for t in tasks)
    gap = sum(acc[t][:, 0].mean() - acc[t][:, 1].mean() for t in tasks)
    return gain / gap if gap / len(tasks) >= MIN_GAP else None, gain, gap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=ROOT / 'objection1/result.json')
    args = parser.parse_args()
    rng = np.random.default_rng(SEED)
    result = {'note': 'Exploratory; see README.md, "Exploratory notes".', 'bootstrap': {'B': B, 'seed': SEED},
              'min_gap': MIN_GAP, 'models': {}}
    for m in MODELS:
        acc, lifts = accuracy(m), lift(m)
        tasks = {t: {'n': len(a), 'full': float(a[:, 0].mean()), 'four_bit': float(a[:, 1].mean()),
                     'recipe': float(a[:, 2].mean()), 'gain': float(a[:, 2].mean() - a[:, 1].mean()),
                     'gap': float(a[:, 0].mean() - a[:, 1].mean()), 'lift': lifts.get(t, {}).get('lift'),
                     'lift_rows': lifts.get(t, {}).get('rows', 0)} for t, a in sorted(acc.items())}
        for t in tasks.values():
            t['recovered'] = t['gain'] / t['gap'] if t['gap'] >= MIN_GAP else None
        groups = {}
        for g, members in GROUPS.items():
            share, gain, gap = recovered(acc, members)
            groups[g] = {'recovered': share, 'gain': gain / len(members), 'gap': gap / len(members),
                         'lift': float(np.mean([tasks[t]['lift'] for t in members]))}
        draws = {g: [] for g in ('qa', 'niah')}
        for _ in range(B):
            sample = {t: acc[t][rng.integers(0, len(acc[t]), len(acc[t]))] for t in (*GROUPS['qa'], *GROUPS['niah'])}
            for g in draws:
                draws[g].append(recovered(sample, GROUPS[g])[0] or np.nan)
        diff = np.array(draws['niah']) - np.array(draws['qa'])
        share = {g: groups[g]['recovered'] or np.nan for g in ('qa', 'niah')}
        groups['niah_minus_qa'] = {'delta': share['niah'] - share['qa'],
                                   'ci95': np.nanpercentile(diff, [2.5, 97.5]).tolist(),
                                   'share_of_draws_below_zero': float(np.nanmean(diff < 0))}
        ok = [t for t, v in tasks.items() if v['recovered'] is not None and v['lift'] is not None]
        rho = spearmanr([tasks[t]['lift'] for t in ok], [tasks[t]['recovered'] for t in ok])
        result['models'][m] = {'tasks': tasks, 'groups': groups, 'niah_lift_confirmation': niah_lift(m),
                               'spearman_lift_vs_recovered': {'tasks': ok, 'rho': float(rho[0]), 'p': float(rho[1])}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + '\n')
    for m, r in result['models'].items():
        print(m)
        for t, v in r['tasks'].items():
            print(f"  {t:20s} n={v['n']:3d} full {v['full']:5.1f} 4bit {v['four_bit']:5.1f} recipe {v['recipe']:5.1f}"
                  f" gain {v['gain']:+5.1f} gap {v['gap']:5.1f}"
                  f" rec {v['recovered'] if v['recovered'] is None else round(v['recovered'], 2)}"
                  f" lift {v['lift'] if v['lift'] is None else round(v['lift'], 2)}")
        for g, v in r['groups'].items():
            print('  ', g, {k: (round(x, 3) if isinstance(x, float) else x) for k, x in v.items()})
        print('   spearman', r['spearman_lift_vs_recovered'], 'niah lift (confirmation)', r['niah_lift_confirmation'])


if __name__ == '__main__':
    main()
