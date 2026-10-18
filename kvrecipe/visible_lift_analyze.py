import argparse
import json
from pathlib import Path
import random
import statistics

EXPECTED = {'qa_1': 140, 'qa_2': 137, 'qa_squad': 128, 'qa_nq': 100}
SELECTORS, BUDGETS = ('Z', 'Zv', 'SNv'), ('1.5625', '3.125')
B, SEED = 10_000, 0


def lift(rows, arm):
    return statistics.mean(r['gold_kept'][arm] for r in rows) / statistics.mean(r['keep_rate'][arm] for r in rows)


def draws(groups, arms):
    rng = random.Random(SEED)
    out = {a: [] for a in arms}
    for _ in range(B):
        sample = [r for g in (rng.choice(groups) for _ in groups) for r in g]
        for a in arms:
            out[a].append(lift(sample, a))
    return out


def ci(values):
    v = sorted(values)
    return [v[int(0.025 * B)], v[int(0.975 * B) - 1]]


def analyze(models, expect=True):
    out = {'validity': {}, 'cells': {}}
    arms = [f'{s}@{b}' for s in SELECTORS for b in BUDGETS]
    for m, records in models.items():
        good = [r for r in records if r.get('gold_kept') and all(a in r['gold_kept'] for a in arms)]
        for t, n in EXPECTED.items():
            rows = [r for r in good if r['task'] == t]
            out['validity'][f'{m} {t}'] = {'rows': len([r for r in records if r['task'] == t]), 'complete': len(rows),
                                           'expected': n, 'valid': (not expect or len(rows) == n) and bool(rows)}
            if not rows:
                continue
            groups = {}
            for r in rows:
                groups.setdefault(r.get('doc_id') or r['index'], []).append(r)
            d = draws(list(groups.values()), arms)
            cell = {'rows': len(rows)}
            for a in arms:
                cell[a] = {'lift': lift(rows, a), 'ci95': ci(d[a]),
                           'keep_rate': statistics.mean(r['keep_rate'][a] for r in rows)}
            for b in BUDGETS:
                diff = [x - y for x, y in zip(d[f'Zv@{b}'], d[f'Z@{b}'])]
                cell[f'Zv-Z@{b}'] = {'delta': cell[f'Zv@{b}']['lift'] - cell[f'Z@{b}']['lift'], 'ci95': ci(diff)}
            cell['percentile'] = {k: statistics.mean(r['percentile'][k] for r in rows) for k in ('Z', 'Zv')}
            out['cells'][f'{m} {t}'] = cell
    valid = all(v['valid'] for v in out['validity'].values()) and len(out['cells']) == len(models) * len(EXPECTED)
    zv = {k: c['Zv@1.5625'] for k, c in out['cells'].items()}
    high = sorted(k for k, c in zv.items() if c['ci95'][0] > 1.5)
    holds = all(c['lift'] < 1.5 and c['ci95'][1] < 2.0 for c in zv.values())
    call = 'HOLDS' if holds else 'SOFTEN' if high else 'MIXED'
    out['rule'] = {'call': call if valid else 'INVALID', 'valid': valid, 'cells_above': high,
                   'cells_not_below': sorted(k for k, c in zv.items() if not (c['lift'] < 1.5 and c['ci95'][1] < 2.0)),
                   'wording': 'makes' if call == 'HOLDS' else 'accompanies', 'bootstrap': {'B': B, 'seed': SEED}}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', nargs=2, action='append', required=True, metavar=('NAME', 'FILE'))
    parser.add_argument('--no-expect', action='store_true', help='CPU checks: skip the expected counts')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    models = {m: [json.loads(line) for line in open(f)] for m, f in args.model}
    out = analyze(models, not args.no_expect)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps({'rule': out['rule'], 'Zv@1.5625': {k: [round(c['Zv@1.5625']['lift'], 3),
                                                              [round(x, 3) for x in c['Zv@1.5625']['ci95']]]
                                                          for k, c in out['cells'].items()}}, indent=1))


if __name__ == '__main__':
    main()
