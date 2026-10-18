import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from analyze import score
from controls import BUDGETS, arms

B, SEED, MARGIN = 10_000, 0, 1.0
EXPECTED = {'ruler': (650, 650), 'quality': (115, 2086)}  # records, questions); the lean variant sets RULER to 338
FULL_FLOOR = {'ruler': 85.0, 'quality': 50.0}
PARSE_FLOOR = 0.95
PRIMARY = {
    'A': {'benchmarks': ('ruler',), 'sided': 2, 'alpha': 0.05,
          'contrasts': {'I_A': (('Zv.k4v2pc', 1), ('Zv.stack4pc', -1), ('Z.k4v2pc', -1), ('Z.stack4pc', 1))}},
    'B': {'benchmarks': ('ruler', 'quality'), 'sided': 1, 'alpha': 0.025,
          'contrasts': {'KD k4v2pc-stack4pc': (('KD.k4v2pc', 1), ('KD.stack4pc', -1))}},
    'C': {'benchmarks': ('ruler', 'quality'), 'sided': 1, 'alpha': 0.025,
          'contrasts': {'SN KIVI4-evict16': (('SN.stack4pc', 1), ('SN.evict16', -1))}},
    'D': {'benchmarks': ('ruler', 'quality'), 'sided': 1, 'alpha': 0.025,
          'contrasts': {'Z k4v2pc-k3v3pc': (('Z.k4v2pc', 1), ('Z.k3v3pc', -1)),
                        'Z k4v2pc-k2v4pc': (('Z.k4v2pc', 1), ('Z.k2v4pc', -1))}},
}
SECONDARY = {
    'A2': {'benchmarks': ('ruler',), 'sided': 1, 'alpha': 0.025,
           'contrasts': {'Zv k4v2pc-stack4pc': (('Zv.k4v2pc', 1), ('Zv.stack4pc', -1))}},
    'B2': {'benchmarks': ('ruler', 'quality'), 'sided': 1, 'alpha': 0.025,
           'contrasts': {'KD k3v2pc-k4v2pc': (('KD.k3v2pc', 1), ('KD.k4v2pc', -1))}},
    'C2': {'benchmarks': ('ruler', 'quality'), 'sided': 2, 'alpha': 0.05,
           'contrasts': {'SN KIVI2-KIVI4': (('SN.k2v2pc', 1), ('SN.stack4pc', -1))}},
}


def load(paths):
    records = {}
    for p in paths:
        for line in open(p):
            r = json.loads(line)
            records[r.get('index', r.get('doc_id'))] = r
    return [records[k] for k in sorted(records, key=str)]


def ruler_draws(records, row_arms):
    tasks = sorted({r['task'] for r in records})
    s = {a: {t: np.array([r['scores'][a] for r in records if r['task'] == t]) for t in tasks}
         for a in row_arms}
    acc = {a: float(np.mean([s[a][t].mean() for t in tasks])) for a in row_arms}
    rng = np.random.default_rng(SEED)
    draws = {a: np.zeros(B) for a in row_arms}
    for t in tasks:
        idx = rng.integers(0, len(s[row_arms[0]][t]), size=(B, len(s[row_arms[0]][t])))
        for a in row_arms:
            draws[a] += s[a][t][idx].mean(axis=1) / len(tasks)
    return acc, draws, {a: {t: float(s[a][t].mean()) for t in tasks} for a in row_arms}


def quality_draws(records, row_arms):
    sums = np.array([[sum(q['correct'][a] for q in d['questions']) for a in row_arms] for d in records], float)
    counts = np.array([len(d['questions']) for d in records], float)
    idx = np.random.default_rng(SEED).integers(0, len(records), size=(B, len(records)))
    draws = 100 * sums[idx].sum(1) / counts[idx].sum(1)[:, None]
    acc = 100 * sums.sum(0) / counts.sum()
    return dict(zip(row_arms, acc.tolist())), {a: draws[:, i] for i, a in enumerate(row_arms)}, None


def contrast(acc, draws, terms, sided):
    delta = sum(w * acc[a] for a, w in terms)
    d = sum(w * draws[a] for a, w in terms)
    low, high = float(np.mean(d <= 0)), float(np.mean(d >= 0))
    p = low if sided == 1 else min(1.0, 2 * min(low, high))
    return {'delta': float(delta), 'p': p, 'sided': sided, 'ci95': np.percentile(d, [2.5, 97.5]).tolist()}


def summary(records, bench):
    row_arms = ['full', *arms(bench)]
    acc, draws, per_task = (ruler_draws if bench == 'ruler' else quality_draws)(records, row_arms)
    over = [(r.get('index', r.get('doc_id')), a) for r in records for a in r['bits']
            if r['bits'][a] > r['budget_bits'][a] or r['over_budget_layers'][a]]
    n_records, n_q = EXPECTED[bench]
    questions = len(records) if bench == 'ruler' else sum(len(d['questions']) for d in records)
    valid = {'budget': not over, 'complete': (len(records), questions) == (n_records, n_q),
             'full_accuracy': acc['full'] >= FULL_FLOOR[bench]}
    out = {'accuracy': acc, 'valid': valid, 'records': len(records), 'questions': questions,
           'budget': {'pass': not over, 'over': over[:20],
                      'share_of_full_bytes': {a: float(np.mean([r['bits'][a] / r['full_bits'] for r in records]))
                                              for a in records[0]['bits']}},
           'kept': {a: float(np.mean([r['kept'][a] for r in records])) for a in records[0]['kept']},
           'gap_to_full': {a: float(acc[a] - acc['full']) for a in row_arms if a != 'full'},
           'seconds': {'per_record': float(np.mean([r['score_seconds'] + r['build_seconds'] + r['decode_seconds']
                                                    for r in records]))}}
    if bench == 'quality':
        qs = [q for d in records for q in d['questions']]
        out['parsed'] = {a: float(np.mean([q['choice'][a] is not None for q in qs])) for a in row_arms}
        valid['full_parsed'] = out['parsed']['full'] >= PARSE_FLOOR
    else:
        out['per_task'] = per_task
    out['valid_all'] = all(valid.values())
    out['_acc_draws'] = (acc, draws)
    return out


def family_tests(out, families):
    result = {}
    for name, fam in families.items():
        cells = []
        for m in out:
            for bench in fam['benchmarks']:
                if bench not in out[m]:
                    continue
                acc, draws = out[m][bench]['_acc_draws']
                for label, terms in fam['contrasts'].items():
                    for b in BUDGETS:
                        t = contrast(acc, draws, [(f'{a}@{b}', w) for a, w in terms], fam['sided'])
                        cells.append((t['p'], f'{m} {bench} {label} @{b}', t, out[m][bench]['valid_all']))
        cells.sort(key=lambda c: c[0])
        still, tests = True, {}
        for rank, (p, key, t, valid) in enumerate(cells):
            threshold = fam['alpha'] / (len(cells) - rank)
            passed = still and p <= threshold
            still = passed
            t.update(holm_threshold=threshold, holm_pass=bool(passed), valid=bool(valid),
                     passes=bool(passed and abs(t['delta']) >= MARGIN and valid))
            tests[key] = t
        result[name] = {'tests': tests, 'n': len(cells), 'alpha': fam['alpha'], 'sided': fam['sided']}
    return result


def sensitivity(primary):
    cells = sorted((t['p'], f'{fam} {key}', t) for fam, f in primary.items() for key, t in f['tests'].items())
    still, out = True, {}
    for rank, (p, key, t) in enumerate(cells):
        threshold = 0.025 / (len(cells) - rank)
        passed = still and p <= threshold
        still = passed
        out[key] = {'delta': t['delta'], 'p': p, 'holm_threshold': threshold,
                    'passes': bool(passed and abs(t['delta']) >= MARGIN and t['valid'])}
    return {'tests': len(cells), 'alpha': 0.025, 'results': out}


def analyze(models, ruler_expected=None):
    if ruler_expected:
        EXPECTED['ruler'] = (ruler_expected, ruler_expected)
    out = {m: {bench: summary([r for r in recs if r['benchmark'] == bench], bench)
               for bench in EXPECTED if any(r['benchmark'] == bench for r in recs)} for m, recs in models.items()}
    primary, secondary = family_tests(out, PRIMARY), family_tests(out, SECONDARY)
    for m in out:
        for bench in out[m]:
            out[m][bench].pop('_acc_draws')
    return {'primary': primary, 'secondary': secondary, 'sensitivity_one_family': sensitivity(primary),
            'per_model': out, 'bootstrap': {'B': B, 'seed': SEED}, 'margin': MARGIN}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', nargs='+', action='append', required=True, metavar=('NAME', 'FILE'))
    parser.add_argument('--out')
    parser.add_argument('--ruler-expected', type=int, default=0, help='338 for the lean variant')
    args = parser.parse_args()
    out = analyze({name: load(files) for name, *files in args.model}, args.ruler_expected)
    out['ruler_expected'] = EXPECTED['ruler'][0]
    text = json.dumps(out, indent=1)
    if args.out:
        Path(args.out).write_text(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
