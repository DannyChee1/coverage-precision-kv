import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
import controls_analyze as ca
import review4k_analyze as r4

ALPHA = 0.025
BUDGETS = r4.BUDGETS
N, FULL_FLAG = 338, 75.0
SPLITS = ('stack4pc', 'k4v2pc', 'k3v2pc', 'evict16')
CONTRASTS = {'P': (('k4v2pc', 1), ('stack4pc', -1)), 'S': (('k3v2pc', 1), ('k4v2pc', -1))}


def analyze(models, four_k=None):
    with_tca = any('Z.tca@1.5625' in r['bits'] for recs in models.values() for r in recs)
    splits = SPLITS + (('tca',) if with_tca else ())
    arms = ['full', *[f'Z.{s}@{b}' for b in BUDGETS for s in splits]]
    per, draws, cells, secondary = {}, {}, [], {}
    for m, recs in models.items():
        rs = [r for r in recs if r['benchmark'] == 'ruler']
        if not rs:
            continue
        with r4.configured(arms, ruler_n=N, full_floor=0.0):
            out = ca.summary(rs, 'ruler')
        acc, d = out.pop('_acc_draws')
        out['full_flagged'] = acc['full'] < FULL_FLAG
        out['peak_gib'] = max((r.get('peak_gib', 0.0) for r in rs), default=None)
        per[m], draws[m] = {'ruler': out}, {'ruler': (acc, d)}
        for name, terms in CONTRASTS.items():
            for b in BUDGETS:
                t = ca.contrast(acc, d, [(f'Z.{a}@{b}', w) for a, w in terms], 1)
                if four_k:
                    a4 = four_k['per_model'][m]['ruler']['accuracy']
                    t['delta_4k'] = sum(w * a4[f'Z.{a}@{b}'] for a, w in terms)
                cells.append((f'{m} {name} @{b}', t, out['valid_all']))
        for b in BUDGETS:
            for arm in ('k3v2pc', 'stack4pc'):
                secondary[f'{m} {arm}-evict16 @{b}'] = ca.contrast(
                    acc, d, [(f'Z.{arm}@{b}', 1), (f'Z.evict16@{b}', -1)], 2)
    tests = r4.holm(cells, ALPHA)
    calls = {}
    for b in BUDGETS:
        for name, label in (('P', 'HOLDS AT 16K'), ('S', 'ADD')):
            keys = [f'{m} {name} @{b}' for m in models]
            ok = all(k in tests and tests[k]['passes'] for k in keys)
            calls[f'{name} @{b}'] = label if ok else 'NOT SHOWN'
    out = {'calls': calls, 'tests': tests, 'secondary_vs_evict16': secondary, 'per_model': per,
           'tca_included': with_tca,
           'rule': {'alpha': ALPHA, 'margin': ca.MARGIN, 'tests': len(cells), 'sided': 1, 'full_flag_below': FULL_FLAG,
                    'bootstrap': {'B': ca.B, 'seed': ca.SEED}, 'examples': N}}
    if with_tca:
        out['tca'] = r4.challenger(per, draws, ('tca',), benches=('ruler',))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', nargs='+', action='append', required=True, metavar=('NAME', 'FILE'))
    parser.add_argument('--four-k', type=Path, help='results/controls/results.json')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = analyze({name: ca.load(files) for name, *files in args.model},
                  json.loads(args.four_k.read_text()) if args.four_k else None)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps(out['calls'], indent=1))


if __name__ == '__main__':
    main()
