import argparse
import json
from pathlib import Path
import random
import statistics

EXPECTED = {'squad': 69, 'nq': 200}
SELECTORS, BUDGETS = ('kvzip', 'kv2', 'rarity'), ('1.5625', '3.125')
B, SEED = 10_000, 0


def lift(records, arm):
    return statistics.mean(r['gold_kept'][arm] for r in records) / statistics.mean(r['keep_rate'][arm] for r in records)


def interval(records, arm):
    rng = random.Random(SEED)
    draws = sorted(lift([rng.choice(records) for _ in records], arm) for _ in range(B))
    return [draws[int(0.025 * B)], draws[int(0.975 * B) - 1]]


def analyze(records):
    out = {'validity': {}, 'cells': {}}
    for source, n in EXPECTED.items():
        rs = [r for r in records if r['source'] == source]
        good = [r for r in rs if r.get('gold_kept') and all(f'{s}@{b}' in r['gold_kept'] for s in SELECTORS
                                                             for b in BUDGETS)]
        out['validity'][source] = {'documents': len(rs), 'expected': n, 'complete': len(good)}
        for s in SELECTORS:
            for b in BUDGETS:
                arm = f'{s}@{b}'
                out['cells'][f'{source} {arm}'] = {
                    'retention': statistics.mean(r['gold_kept'][arm] for r in good),
                    'keep_rate': statistics.mean(r['keep_rate'][arm] for r in good),
                    'lift': lift(good, arm), 'ci95': interval(good, arm)}
        out['cells'][f'{source} kvzip percentile'] = statistics.mean(r['kvzip_gold_percentile'] for r in good)
    out['valid'] = all(v['documents'] == v['expected'] == v['complete'] for v in out['validity'].values())
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', nargs=2, action='append', metavar=('NAME', 'FILE'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = {'rule': 'KVzip+ at 1.5625%: GENERALIZES if every lift < 1.5 with upper < 2.0; RESCOPE if any lift >= 2.0 '
                   'with lower > 1.5; else MIXED', 'models': {}}
    for name, path in args.model:
        out['models'][name] = analyze([json.loads(line) for line in open(path)])
        m = out['models'][name]
        print(name, 'valid' if m['valid'] else 'INVALID', {k: (round(v['lift'], 2), [round(c, 2) for c in v['ci95']])
                                                            for k, v in m['cells'].items() if isinstance(v, dict)})
    key = [m['cells'][f'{s} kvzip@1.5625'] for m in out['models'].values() for s in EXPECTED]
    valid = len(out['models']) == 2 and all(m['valid'] for m in out['models'].values())
    if not valid:
        out['call'] = 'INVALID'
    elif all(c['lift'] < 1.5 and c['ci95'][1] < 2.0 for c in key):
        out['call'] = 'GENERALIZES'
    elif any(c['lift'] >= 2.0 and c['ci95'][0] > 1.5 for c in key):
        out['call'] = 'RESCOPE'
    else:
        out['call'] = 'MIXED'
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1) + '\n')
    print('E1', out['call'])


if __name__ == '__main__':
    main()
