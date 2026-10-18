import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from analyze import score
from longbench_metrics import score_example

B, SEED = 10_000, 0
Z, W, R = 'Z.k3v2pc@1.5625', 'W50.k3v2pc@1.5625', 'R50.k3v2pc@1.5625'
SELECTORS = {'Z': Z, 'W50': W}
NIAH = ('niah_single_1', 'niah_single_2', 'niah_single_3', 'niah_multikey_1', 'niah_multikey_2', 'niah_multikey_3',
        'niah_multivalue', 'niah_multiquery')
SETS = ('qa_squad', 'qa_1', 'qa_2', 'niah')
EXPECTED = {'qa_squad': 127, 'qa_1': 100, 'qa_2': 100, **{t: 50 for t in NIAH}}
QUALITY_N = (115, 2086)
LB_TASKS, LB_N = ('gov_report', 'qmsum'), 100
MARGIN, ALPHA, POOLED_FLOOR, CELL_FLOOR = 1.0, 0.05, -1.0, -2.0
REVIEW = Path(__file__).resolve().parents[1] / 'results/review'


def ci(d):
    return np.percentile(d, [2.5, 97.5]).tolist()


def set_of(task):
    return 'niah' if task.startswith('niah') else task


def load(paths, key):
    out = {}
    for p in paths:
        for line in open(p):
            r = json.loads(line)
            out[r[key]] = r
    return [out[k] for k in sorted(out, key=str)]


def frontier_ok(r):
    f = r.get('frontier') or {}
    return bool(f.get('tokenization_ok')) and set(SELECTORS) <= set(f.get('keep_rate', {}))


def within(r, rows):
    return all(r['kept'][a] == r['kept'][Z] for a in rows if a != 'full') and not any(
        r['over_budget_layers'].get(a) for a in rows if a != 'full')


def conf_model(records, rows, expect=True):
    counts = defaultdict(int)
    for r in records:
        counts[r['task']] += 1
    valid = {'counts': not expect or dict(counts) == EXPECTED,
             'all_rows': all(set(rows) <= set(r['scores']) for r in records),
             'budget_and_equal_keep': all(within(r, rows) for r in records),
             'tokenized_with_measures': all(frontier_ok(r) for r in records)}
    rng = np.random.default_rng(SEED)
    sets = {}
    for s in SETS:
        tasks = sorted(t for t in counts if set_of(t) == s)
        if not tasks:
            continue
        acc, draws, per_task = dict.fromkeys(rows, 0.0), {a: np.zeros(B) for a in rows}, {}
        for t in tasks:
            sel = [r for r in records if r['task'] == t]
            sc = {a: np.array([r['scores'][a] for r in sel]) for a in rows}
            idx = rng.integers(0, len(sc[rows[0]]), size=(B, len(sc[rows[0]])))
            per_task[t] = {'n': len(sc[rows[0]]), **{a: float(sc[a].mean()) for a in rows}}
            for a in rows:
                acc[a] += sc[a].mean() / len(tasks)
                draws[a] += sc[a][idx].mean(axis=1) / len(tasks)
        sel = [r for r in records if set_of(r['task']) == s]
        sets[s] = {'acc': acc, 'draws': draws, 'per_task': per_task, 'measures': measures(sel)}
    return sets, {**valid, 'valid': all(valid.values())}


def measures(records):
    out = {}
    for k in [*SELECTORS, 'R50'] if all('R50' in r['frontier']['keep_rate'] for r in records) else SELECTORS:
        f = [r['frontier'] for r in records if frontier_ok(r)]
        lift = [x['gold_kept'][k] / x['keep_rate'][k] for x in f if x['gold_kept'].get(k) is not None]
        out[k] = {'keep_rate': float(np.mean([x['keep_rate'][k] for x in f])) if f else None,
                  'plain_mass': float(np.mean([x['plain_mass'][k] for x in f])) if f else None,
                  'lift': float(np.mean(lift)) if lift else None}
    return out


def quality_model(records, rows, expect=True):
    questions = [q for d in records for q in d['questions']]
    valid = {'counts': not expect or (len(records), len(questions)) == QUALITY_N,
             'all_rows': all(set(rows) <= set(q['correct']) for q in questions),
             'budget_and_equal_keep': all(within(d, rows) for d in records),
             'tokenized_with_measures': all(frontier_ok(d) for d in records)}
    sums = np.array([[sum(q['correct'][a] for q in d['questions']) for a in rows] for d in records], float)
    n = np.array([len(d['questions']) for d in records], float)
    idx = np.random.default_rng(SEED).integers(0, len(records), size=(B, len(records)))
    draws = 100 * sums[idx].sum(1) / n[idx].sum(1)[:, None]
    acc = 100 * sums.sum(0) / n.sum()
    return ({'acc': dict(zip(rows, acc.tolist())), 'draws': {a: draws[:, i] for i, a in enumerate(rows)},
             'measures': measures(records)}, {**valid, 'valid': all(valid.values())})


def longbench_model(records, expect=True):
    rows = [Z, W]
    counts = defaultdict(int)
    for r in records:
        counts[r['task']] += 1
    valid = {'counts': not expect or dict(counts) == {t: LB_N for t in LB_TASKS},
             'all_rows': all(set(rows) <= set(r['scores']) for r in records),
             'budget_and_equal_keep': all(within(r, rows) for r in records),
             'tokenized_with_measures': all(frontier_ok(r) for r in records),
             'no_think': not any(r['think'][a] for r in records for a in rows)}
    rng = np.random.default_rng(SEED)
    cells = {}
    for t in LB_TASKS:
        sel = [r for r in records if r['task'] == t]
        if not sel:
            continue
        sc = {a: np.array([100 * r['scores'][a] for r in sel]) for a in rows}
        idx = rng.integers(0, len(sel), size=(B, len(sel)))
        cells[t] = {'acc': {a: float(sc[a].mean()) for a in rows}, 'draws': {a: sc[a][idx].mean(axis=1) for a in rows},
                    'out_tokens': {a: float(np.mean([r['out_tokens'][a] for r in sel])) for a in rows},
                    'measures': measures(sel)}
    return cells, {**valid, 'valid': all(valid.values())}


def summarize(cell, rows):
    d = cell['draws'][W] - cell['draws'][Z]
    return {'n_rows': rows, 'accuracy': {a: cell['acc'][a] for a in rows},
            'ci95': {a: ci(cell['draws'][a]) for a in rows if a in cell['draws']},
            'delta': cell['acc'][W] - cell['acc'][Z], 'delta_ci95': ci(d), 'measures': cell.get('measures')}


def contrast(cells, a, b):
    d = np.mean([c['draws'][a] - c['draws'][b] for c in cells.values()], axis=0)
    p = float(min(1.0, 2 * min(np.mean(d <= 0), np.mean(d >= 0))))
    models = sorted({m for m, _ in cells})
    return {'delta': float(np.mean([c['acc'][a] - c['acc'][b] for c in cells.values()])), 'ci95': ci(d), 'p': p,
            'per_model': {m: float(np.mean([c['acc'][a] - c['acc'][b] for (k, _), c in cells.items() if k == m]))
                          for m in models}}


def references():
    try:
        r1 = json.loads((REVIEW / 'result_4k.json').read_text())['per_model']
        r3 = json.loads((REVIEW / 'result_longbench.json').read_text())['per_model']
    except (OSError, KeyError):
        return {}
    return {m: {'quality': {a: r1[m]['quality']['accuracy'][a] for a in ('full', Z)},
                'longbench': {a: r3[m]['per_task'][a] for a in ('full', Z)}} for m in r1 if m in r3}


def analyze(conf, quality, longbench, expect=True):
    out = {'models': {}, 'validity': {}}
    conf_cells, harm_cells = {}, {}
    for m, recs in conf.items():
        rows = [a for a in ('full', Z, W, R) if all(a in r['scores'] for r in recs)]
        sets, valid = conf_model(recs, rows, expect)
        out['validity'][f'{m} confirmation'] = valid
        for s, cell in sets.items():
            conf_cells[(m, s)] = cell
            out['models'].setdefault(m, {})[s] = {**summarize(cell, rows), 'per_task': cell['per_task']}
    for m, recs in quality.items():
        rows = ['full', Z, W] if all('full' in q['correct'] for d in recs for q in d['questions']) else [Z, W]
        cell, valid = quality_model(recs, rows, expect)
        out['validity'][f'{m} quality'] = valid
        harm_cells[(m, 'quality')] = cell
        out['models'].setdefault(m, {})['quality'] = summarize(cell, rows)
    for m, recs in longbench.items():
        cells, valid = longbench_model(recs, expect)
        out['validity'][f'{m} longbench'] = valid
        for t, cell in cells.items():
            harm_cells[(m, t)] = cell
            out['models'].setdefault(m, {})[t] = {**summarize(cell, [Z, W]), 'out_tokens': cell['out_tokens']}
    part_ok = lambda part, models: all(out['validity'][f'{m} {part}']['valid'] for m in models)
    out['valid'] = {k: v['valid'] for k, v in out['validity'].items()}
    if conf_cells:
        mean = lambda a: np.mean([c['acc'][a] for c in conf_cells.values()])
        d = np.mean([c['draws'][W] - c['draws'][Z] for c in conf_cells.values()], axis=0)
        delta, p = float(mean(W) - mean(Z)), float(np.mean(d <= 0))
        ok = part_ok('confirmation', conf) and len(conf_cells) == 2 * len(SETS)
        out['confirmation'] = {
            'cells': len(conf_cells), 'pooled': {'Z': float(mean(Z)), 'W50': float(mean(W))}, 'delta': delta,
            'ci95': ci(d), 'one_sided_lower95': float(np.percentile(d, 5)), 'p': p,
            'confirmed': bool(ok and delta >= MARGIN and p < ALPHA), 'valid': ok,
            'per_model': {m: float(np.mean([conf_cells[(m, s)]['acc'][W] - conf_cells[(m, s)]['acc'][Z]
                                            for s in SETS if (m, s) in conf_cells])) for m in conf},
            'per_set': {s: float(np.mean([conf_cells[(m, s)]['acc'][W] - conf_cells[(m, s)]['acc'][Z]
                                          for m in conf if (m, s) in conf_cells])) for s in SETS},
            'not_helped': sorted(f'{m} {s}' for (m, s), c in conf_cells.items() if c['acc'][W] - c['acc'][Z] <= 0),
            'niah_tasks_not_helped': sorted(f'{m} {t}' for (m, s), c in conf_cells.items() if s == 'niah'
                                            for t, v in c['per_task'].items() if v[W] - v[Z] <= 0)}
        if all(R in c['acc'] for c in conf_cells.values()):
            out['confirmation']['control_R50'] = {name: contrast(conf_cells, a, b) for name, a, b in
                                                  (('R50 - Z', R, Z), ('W50 - R50', W, R))}
    if harm_cells:
        deltas = {f'{m} {t}': c['acc'][W] - c['acc'][Z] for (m, t), c in harm_cells.items()}
        d = np.mean([c['draws'][W] - c['draws'][Z] for c in harm_cells.values()], axis=0)
        lower = float(np.percentile(d, 2.5))
        ok = part_ok('quality', quality) and part_ok('longbench', longbench) and len(harm_cells) == 6
        out['no_harm'] = {'cells': deltas, 'pooled_delta': float(np.mean(list(deltas.values()))), 'ci95': ci(d),
                          'pooled_lower95': lower, 'worst_cell': min(deltas, key=deltas.get), 'valid': ok,
                          'no_harm': bool(ok and lower > POOLED_FLOOR and min(deltas.values()) >= CELL_FLOOR)}
    out['reference_review_runs'] = references()
    out['rule'] = {'bootstrap': {'B': B, 'seed': SEED}, 'margin': MARGIN, 'alpha': ALPHA,
                   'pooled_floor': POOLED_FLOOR, 'cell_floor': CELL_FLOOR, 'expected': EXPECTED,
                   'quality_expected': QUALITY_N, 'longbench_per_task': LB_N}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', nargs=2, action='append', default=[], metavar=('MODEL', 'FILE'))
    parser.add_argument('--quality', nargs=2, action='append', default=[], metavar=('MODEL', 'FILE'))
    parser.add_argument('--longbench', nargs='+', action='append', default=[], metavar='MODEL FILE')
    parser.add_argument('--no-expect', action='store_true', help='CPU checks: skip the expected counts')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    conf = {m: load([f], 'index') for m, f in args.conf}
    quality = {m: load([f], 'doc_id') for m, f in args.quality}
    longbench = {m: load(fs, 'id') for m, *fs in args.longbench}
    out = analyze(conf, quality, longbench, not args.no_expect)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps({k: out[k] for k in ('valid', 'confirmation', 'no_harm') if k in out}, indent=1))


if __name__ == '__main__':
    main()
