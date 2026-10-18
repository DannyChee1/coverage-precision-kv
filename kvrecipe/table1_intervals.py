import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
import controls_analyze as ca

STEPS = {'k4v2 - 4-bit': (('Z.k4v2pc', 1), ('Z.stack4pc', -1)), 'k3v2 - k4v2': (('Z.k3v2pc', 1), ('Z.k4v2pc', -1))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', nargs='+', action='append', required=True, metavar=('NAME', 'FILE'))
    parser.add_argument('--out', required=True)
    parser.add_argument('--ruler-expected', type=int, default=338)
    args = parser.parse_args()
    ca.EXPECTED['ruler'] = (args.ruler_expected, args.ruler_expected)
    out = {}
    for name, *files in args.model:
        records = ca.load(files)
        for bench in ('ruler', 'quality'):
            acc, draws = ca.summary([r for r in records if r['benchmark'] == bench], bench)['_acc_draws']
            for step, terms in STEPS.items():
                for budget in ca.BUDGETS:
                    at = [(f'{arm}@{budget}', w) for arm, w in terms]
                    out[f'{name} {bench} {step} @{budget}'] = ca.contrast(acc, draws, at, sided=1)
    out = {'tests': out, 'bootstrap': {'B': ca.B, 'seed': ca.SEED}, 'note': 'exploratory, not pre-registered'}
    Path(args.out).write_text(json.dumps(out, indent=1) + '\n')
    for k, v in out['tests'].items():
        print(f"{k:42s} {v['delta']:+6.2f}  ({v['ci95'][0]:+.2f}, {v['ci95'][1]:+.2f})")


if __name__ == '__main__':
    main()
