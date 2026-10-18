import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
import controls_analyze as ca
import qwen14b

ALPHA = 0.025
TESTS = {'P': [('Z.k4v2pc', 1), ('Z.stack4pc', -1)], 'S': [('Z.k3v2pc', 1), ('Z.k4v2pc', -1)]}
BUDGETS = ('1.5625', '3.125')


def analyze(records, ruler_expected=650):
    qwen14b.configure()
    ca.EXPECTED['ruler'] = (ruler_expected, ruler_expected)
    per_bench, cells = {}, []
    for bench in ('ruler', 'quality'):
        rs = [r for r in records if r['benchmark'] == bench]
        if not rs:
            continue
        out = ca.summary(rs, bench)
        acc, draws = out.pop('_acc_draws')
        per_bench[bench] = out
        for name, terms in TESTS.items():
            for b in BUDGETS:
                t = ca.contrast(acc, draws, [(f'{a}@{b}', w) for a, w in terms], 1)
                cells.append((f'{bench} {name} @{b}', t, out['valid_all']))
    cells.sort(key=lambda c: c[1]['p'])
    still, tests = True, {}
    for rank, (key, t, valid) in enumerate(cells):
        threshold = ALPHA / (len(cells) - rank)
        passed = still and t['p'] <= threshold
        still = passed
        t.update(holm_threshold=threshold, holm_pass=bool(passed), valid=bool(valid),
                 passes=bool(passed and t['delta'] >= ca.MARGIN and valid))
        tests[key] = t
    holds = all(tests.get(f'{b} P @1.5625', {}).get('passes') for b in ('ruler', 'quality'))
    adds = any(tests.get(f'{b} S @1.5625', {}).get('passes') for b in ('ruler', 'quality'))
    return {'tests': tests, 'calls': {'trade': 'HOLDS' if holds else 'NOT SHOWN', 'three_bit_keys': 'ADD' if adds
                                      else 'NOT SHOWN'},
            'per_benchmark': per_bench, 'rule': {'alpha': ALPHA, 'margin': ca.MARGIN, 'tests': len(cells),
                                                 'bootstrap': {'B': ca.B, 'seed': ca.SEED}}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--records', nargs='+', required=True)
    parser.add_argument('--ruler-expected', type=int, default=650)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(ca.load(args.records), args.ruler_expected)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + '\n')
    print(json.dumps({k: {'delta': round(v['delta'], 2), 'p': v['p'], 'passes': v['passes']}
                      for k, v in result['tests'].items()}, indent=1))
    print(result['calls'])


if __name__ == '__main__':
    main()
