import argparse
import json
from pathlib import Path
import random
import statistics
import sys

sys.path.insert(0, str(Path(__file__).parent))
from analyze import score

B1 = '1.5625'
ROWS = ('full', f'Z.k3v2pc@{B1}', *(f'Z.{r}@{B1}' for r in ('ans', 'evm', 'rnda', 'rndb', 'ev')))
TESTS = {'P1': (f'Z.evm@{B1}', f'Z.rnda@{B1}'), 'P2': (f'Z.ans@{B1}', f'Z.rndb@{B1}')}
B, SEED, ALPHA, MARGIN = 10_000, 0, 0.05, 1.0


def scored(records):
    return [{a: r['scores'][a] for a in ROWS} | {'task': r['task']} for r in records]


def pooled(cells, a, b):
    return statistics.mean(statistics.mean(x[a] - x[b] for x in c) for c in cells)


def test(cells, a, b):
    rng = random.Random(SEED)
    delta = pooled(cells, a, b)
    draws = [pooled([[rng.choice(c) for _ in c] for c in cells], a, b) for _ in range(B)]
    draws.sort()
    return {'delta': delta, 'p': sum(d <= 0 for d in draws) / B, 'ci95': [draws[int(0.025 * B)], draws[int(0.975 * B) - 1]]}


def describe(cell):
    acc = {a: statistics.mean(x[a] for x in cell) for a in ROWS}
    gap = acc['full'] - acc[f'Z.k3v2pc@{B1}']
    return {'n': len(cell), 'accuracy': acc,
            'gap_share': {a: (acc[a] - acc[f'Z.k3v2pc@{B1}']) / gap if gap else None for a in ROWS}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', nargs=3, action='append', metavar=('NAME', 'RULER', 'DOCS'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out, ruler_cells, valid = {'models': {}}, [], True
    for name, ruler_path, docs_path in args.model:
        ruler = [json.loads(line) for line in open(ruler_path)]
        docs = [json.loads(line) for line in open(docs_path)]
        v = {'qa_1': sum(r['task'] == 'qa_1' for r in ruler), 'qa_2': sum(r['task'] == 'qa_2' for r in ruler),
             'all_rows': all(set(ROWS) <= set(r['scores']) for r in ruler + docs),
             'sets_found': all(r['e2']['ok'] for r in ruler + docs),
             'within_budget': all(not any(r['over_budget_layers'].values()) for r in ruler + docs)}
        v['valid'] = v['qa_1'] == v['qa_2'] == 100 and v['all_rows'] and v['sets_found'] and v['within_budget']
        valid &= v['valid']
        s_ruler, s_docs = scored(ruler), scored(docs)
        cells = {t: [x for x in s_ruler if x['task'] == t] for t in ('qa_1', 'qa_2')}
        dcells = {t: [x for x in s_docs if x['task'] == t] for t in ('qa_squad', 'qa_nq')}
        ruler_cells += list(cells.values())
        out['models'][name] = {
            'validity': v, 'cells': {t: describe(c) for t, c in {**cells, **dcells}.items() if c},
            'docs_tests': {f'{k} {t}': test([c], *pair) for k, pair in TESTS.items() for t, c in dcells.items() if c},
            'protected_median': {a: statistics.median(r['e2']['protected'][a] for r in ruler if r['e2']['ok'])
                                 for a in ROWS[2:]},
            'budget_share_median': {a: statistics.median(r['e2']['budget_share'][a] for r in ruler if r['e2']['ok'])
                                    for a in ROWS[2:]}}
    tests = {k: test(ruler_cells, *pair) for k, pair in TESTS.items()}
    order = sorted(tests, key=lambda k: tests[k]['p'])
    for rank, k in enumerate(order):
        tests[k]['holm_threshold'] = ALPHA / (len(order) - rank)
    passed = True
    for k in order:  # Holm stops at the first failure
        passed = passed and tests[k]['p'] <= tests[k]['holm_threshold']
        tests[k]['holm_pass'] = passed
        tests[k]['passes'] = passed and tests[k]['delta'] >= MARGIN and valid and len(ruler_cells) == 4
    out.update(tests=tests, valid=valid)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1) + '\n')
    print({k: (round(t['delta'], 2), [round(c, 2) for c in t['ci95']], t['p'], t['passes']) for k, t in tests.items()})


if __name__ == '__main__':
    main()
