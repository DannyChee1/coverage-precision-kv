import argparse
from collections import Counter
from contextlib import contextmanager
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
import controls_analyze as ca
from analyze import score

ALPHA = 0.05
BUDGETS = ('1.5625', '3.125')
BENCHES = ('ruler', 'quality')
SPLITS = ('evict16', 'stack4pc', 'k3v2pc', 'tiera', 'tierb', 'tqk4v4', 'tqk3v2', 'tqk2v2')
PREC16 = 'Z.prec16@1.5625'
RULER_N, QUALITY_N = 338, (115, 2086)
GATE_N, GATE_MARGIN = 169, 1.0
GATE_ROWS = ('full', 'TQ.k4v4', 'TQ.k3v3', 'TQ.k3v3ip')
LABELS = {'PASS': 'TurboQuant', 'FAIL': 'TurboQuant-style'}
DIAG = {'stack4': 'Z.stack4pc@1.5625', 'prec16': PREC16, 'cov2x': 'Z.stack4pc@3.125', 'k3v2pc': 'Z.k3v2pc@1.5625'}
REPRO_ROWS = ('full', *[f'Z.{s}@{b}' for b in BUDGETS for s in ('stack4pc', 'k3v2pc')])


def row_arms(with_tca):
    splits = SPLITS + (('tca',) if with_tca else ())
    return ['full', *[f'Z.{s}@{b}' for b in BUDGETS for s in splits], PREC16]


@contextmanager
def configured(arms, ruler_n=RULER_N, full_floor=85.0):
    saved = ca.arms, dict(ca.EXPECTED), dict(ca.FULL_FLOOR)
    ca.arms = lambda benchmark, budget=None: arms[1:]
    ca.EXPECTED.update(ruler=(ruler_n, ruler_n), quality=QUALITY_N)
    ca.FULL_FLOOR.update(ruler=full_floor, quality=50.0)
    try:
        yield
    finally:
        ca.arms = saved[0]
        ca.EXPECTED.clear()
        ca.EXPECTED.update(saved[1])
        ca.FULL_FLOOR.clear()
        ca.FULL_FLOOR.update(saved[2])


def holm(cells, alpha):
    cells = sorted(cells, key=lambda c: c[1]['p'])
    still, tests = True, {}
    for rank, (key, t, valid) in enumerate(cells):
        threshold = alpha / (len(cells) - rank)
        passed = still and t['p'] <= threshold
        still = passed
        t.update(holm_threshold=threshold, holm_pass=bool(passed), valid=bool(valid),
                 passes=bool(passed and t['delta'] >= ca.MARGIN and valid))
        tests[key] = t
    return tests


def challenger(per, draws, arms, alpha=ALPHA, benches=BENCHES, models=None, extra_valid=None):
    models = models or list(per)
    cells, recipe_minus = [], {}
    for m in models:
        for bench in benches:
            if bench not in per.get(m, {}):
                continue
            acc, d = draws[m][bench]
            valid = per[m][bench]['valid_all'] and (extra_valid or {}).get((m, bench), True)
            for arm in arms:
                for b in BUDGETS:
                    key = f'{m} {bench} {arm} @{b}'
                    cells.append((key, ca.contrast(acc, d, [(f'Z.{arm}@{b}', 1), (f'Z.k3v2pc@{b}', -1)], 1), valid))
                    recipe_minus[key] = ca.contrast(acc, d, [(f'Z.k3v2pc@{b}', 1), (f'Z.{arm}@{b}', -1)], 2)
    tests = holm(cells, alpha)
    expected = len(models) * len(benches) * len(BUDGETS)
    calls = {}
    for arm in arms:
        keys = [k for k in tests if f' {arm} @' in k]
        better = [k for k in keys if tests[k]['passes']]
        holds = (len(keys) == expected and all(tests[k]['valid'] for k in keys)
                 and all(recipe_minus[k]['ci95'][0] > -ca.MARGIN for k in keys))
        calls[arm] = {'call': 'CHALLENGER BETTER' if better else 'RECIPE HOLDS' if holds else 'NOT SEPARATED',
                      'better_cells': better}
    return {'calls': calls, 'tests': tests, 'k3v2pc_minus_arm': recipe_minus,
            'rule': {'alpha': alpha, 'margin': ca.MARGIN, 'tests': len(cells), 'sided': 1}}


def eviction(per, draws):
    cells, complete = {}, True
    for m in per:
        for bench in BENCHES:
            if bench not in per[m]:
                complete = False
                continue
            acc, d = draws[m][bench]
            for b in BUDGETS:
                cells[f'{m} {bench} @{b}'] = {
                    'E': ca.contrast(acc, d, [(f'Z.k3v2pc@{b}', 1), (f'Z.evict16@{b}', -1)], 2),
                    'stack4pc-evict16': ca.contrast(acc, d, [(f'Z.stack4pc@{b}', 1), (f'Z.evict16@{b}', -1)], 2),
                    'valid': per[m][bench]['valid_all']}
    below = (complete and len(cells) == 4 * len(per) and all(c['valid'] for c in cells.values())
             and all(c['E']['ci95'][0] > 0 for c in cells.values()))
    return {'call': 'EVICTION ALONE BELOW THE RECIPE' if below else 'AS MEASURED', 'cells': cells}


def diagnosis(records, bench):
    if bench == 'ruler':
        units = [{k: r['scores'][a] for k, a in DIAG.items()}
                 | {'full': r['scores']['full'], 'record': r} for r in records]
    else:
        units = [{k: 100.0 * bool(q['correct'][a]) for k, a in {**DIAG, 'full': 'full'}.items()}
                 for d in records for q in d['questions']]
    fails = [u for u in units if u['stack4'] < u['full']]
    classes = Counter()
    for u in fails:
        precision, coverage = u['prec16'] >= u['full'], u['cov2x'] >= u['full']
        classes['either' if precision and coverage else 'precision' if precision else
                'coverage' if coverage else 'neither'] += 1
    cov, prec = classes['coverage'], classes['precision']
    call = ('COVERAGE-LIMITED' if cov > 0 and cov >= 2 * prec else
            'PRECISION-LIMITED' if prec > 0 and prec >= 2 * cov else 'NEITHER DOMINATES')
    gap = sum(u['full'] - u['stack4'] for u in fails)
    out = {'call': call, 'failures': len(fails), 'units': len(units),
           'classes': {c: classes[c] for c in ('coverage', 'precision', 'either', 'neither')},
           'gap_recovered': {k: (sum(u[k] - u['stack4'] for u in fails) / gap if gap else None)
                             for k in ('prec16', 'cov2x', 'k3v2pc')}}
    gold = [u['record']['gold_entries'] for u in fails if bench == 'ruler'
            and u['record'].get('gold_entries', {}).get('positions')]
    if gold:
        out['gold_entries'] = {'examples': len(gold),
                               'kept_share': {a: sum(g['kept_share'][a] for g in gold) / len(gold)
                                              for a in gold[0]['kept_share']},
                               'rank_percentile': sum(g['rank_percentile'] for g in gold) / len(gold)}
    return out


def agreement(records, control, bench):
    if bench == 'ruler':
        a, b = ({r['index']: r for r in rs} for rs in (records, control))
        common = sorted(set(a) & set(b))
        same = {row: sum(a[i]['scores'][row]
                         == b[i]['scores'][row] for i in common)
                for row in REPRO_ROWS}
    else:
        a, b = ({q['qid']: q for d in rs for q in d['questions']} for rs in (records, control))
        common = sorted(set(a) & set(b))
        same = {row: sum(bool(a[i]['correct'][row]) == bool(b[i]['correct'][row]) for i in common)
                for row in REPRO_ROWS}
    return {'common': len(common), 'agreement': {row: (same[row] / len(common) if common else None) for row in same}}


def gate(records):
    acc, draws, per_task = ca.ruler_draws(records, list(GATE_ROWS))
    gap = ca.contrast(acc, draws, [('full', 1), ('TQ.k4v4', -1)], 2)
    complete = len(records) == GATE_N
    call = 'INCOMPLETE' if not complete else 'PASS' if gap['delta'] <= GATE_MARGIN else 'FAIL'
    variant = 'ip' if acc['TQ.k3v3ip'] > acc['TQ.k3v3'] else 'mse'
    return {'call': call, 'label': LABELS.get(call), 'key_variant': variant,
            'full_minus_tqk4v4': gap, 'accuracy': acc, 'per_task': per_task, 'examples': len(records),
            'rule': {'margin': GATE_MARGIN, 'examples': GATE_N, 'bootstrap': {'B': ca.B, 'seed': ca.SEED}}}


def analyze(models, gate_out, controls=None):
    with_tca = any('Z.tca@1.5625' in r['bits'] for recs in models.values() for r in recs)
    per, draws, diag, repro, tq_ok = {}, {}, {}, {}, {}
    for m, recs in models.items():
        per[m], draws[m], diag[m] = {}, {}, {}
        for bench in BENCHES:
            rs = [r for r in recs if r['benchmark'] == bench]
            if not rs:
                continue
            with configured(row_arms(with_tca)):
                out = ca.summary(rs, bench)
            draws[m][bench] = out.pop('_acc_draws')
            per[m][bench] = out
            diag[m][bench] = diagnosis(rs, bench)
            tq_ok[(m, bench)] = all(r.get('tq_keys') == gate_out['key_variant'] for r in rs)
            out['tq_keys_match_gate'] = tq_ok[(m, bench)]
            control = [r for r in (controls or {}).get(m, []) if r['benchmark'] == bench]
            if control:
                repro[f'{m} {bench}'] = agreement(rs, control, bench)
    two_tier = challenger(per, draws, ('tiera', 'tierb'))
    tq = challenger(per, draws, ('tqk3v2',), extra_valid=tq_ok)
    out = {'eviction': eviction(per, draws), 'two_tier': two_tier, 'turboquant': tq,
           'turboquant_label': LABELS.get(gate_out['call'], LABELS['FAIL']), 'gate': gate_out['call'],
           'tq_key_variant': gate_out['key_variant'], 'diagnosis': diag, 'reproducibility': repro, 'per_model': per,
           'tca_included': with_tca}
    if with_tca:
        out['tca'] = challenger(per, draws, ('tca',))
    secondary = {}
    for m in per:
        for bench in per[m]:
            acc, d = draws[m][bench]
            for b in BUDGETS:
                for arm in ('tiera', 'tierb', 'tqk4v4', 'tqk3v2', 'tqk2v2') + (('tca',) if with_tca else ()):
                    secondary[f'{m} {bench} {arm}-stack4pc @{b}'] = ca.contrast(
                        acc, d, [(f'Z.{arm}@{b}', 1), (f'Z.stack4pc@{b}', -1)], 2)
                secondary[f'{m} {bench} tqk3v2-tqk4v4 @{b}'] = ca.contrast(
                    acc, d, [('Z.tqk3v2@' + b, 1), ('Z.tqk4v4@' + b, -1)], 2)
                secondary[f'{m} {bench} tqk2v2-tqk3v2 @{b}'] = ca.contrast(
                    acc, d, [('Z.tqk2v2@' + b, 1), ('Z.tqk3v2@' + b, -1)], 2)
    out['secondary'] = secondary
    out['rule'] = {'bootstrap': {'B': ca.B, 'seed': ca.SEED}, 'ruler_expected': RULER_N, 'quality_expected': QUALITY_N}
    return out


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='cmd', required=True)
    g = sub.add_parser('gate')
    g.add_argument('files', nargs='+')
    g.add_argument('--out', type=Path, required=True)
    a = sub.add_parser('main')
    a.add_argument('--model', nargs='+', action='append', required=True, metavar=('NAME', 'FILE'))
    a.add_argument('--gate', type=Path, required=True)
    a.add_argument('--control', nargs='+', action='append', default=[], metavar=('NAME', 'FILE'))
    a.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.cmd == 'gate':
        out = gate(ca.load(args.files))
        brief = {k: out[k] for k in ('call', 'label', 'key_variant')}
    else:
        out = analyze({name: ca.load(files) for name, *files in args.model}, json.loads(args.gate.read_text()),
                      {name: ca.load(files) for name, *files in args.control})
        brief = {'eviction': out['eviction']['call'], 'two_tier': out['two_tier']['calls'],
                 'turboquant': out['turboquant']['calls'], 'label': out['turboquant_label']}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps(brief, indent=1))


if __name__ == '__main__':
    main()
