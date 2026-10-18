import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import nnls

sys.path.insert(0, str(Path(__file__).parent))
from analyze import score

TERMS = ('C', 'V', 'S', 'R')
SETTINGS = tuple(f'tca{i}' for i in range(1, 7))
BUDGETS = ('1.5625', '3.125')
A_PER_TASK, MARGIN, B_BOOT, SEED = 10, 1.0, 1000, 0


def split_of(records, manifest):
    out, by_task = {}, defaultdict(list)
    q = {d: s for s in ('A', 'B') for d in manifest['splits'][s]['doc_ids']}
    for r in records:
        if r['benchmark'] == 'quality':
            out[r['doc_id']] = q.get(r['doc_id'])
        else:
            by_task[r['task']].append(r['index'])
    for task, idx in by_task.items():
        for rank, i in enumerate(sorted(idx)):
            out[i] = 'A' if rank < A_PER_TASK else 'B'
    return out


def observations(model, records, splits):
    rows = []
    for r in records:
        key = r.get('index', r.get('doc_id'))
        arms = [f'Z.{s}@{b}' for b in BUDGETS for s in (*SETTINGS, 'k3v2pc')]
        if r['benchmark'] == 'ruler':
            for a in arms:
                rows.append((model, 'ruler', (model, key), key, splits[key], a,
                             r['scores'][a] / 100, [r['tca'][a][t] for t in TERMS]))
        else:
            for qi, q in enumerate(r['questions']):
                for a in arms:
                    rows.append((model, 'quality', (model, key, qi), key, splits[key], a, float(q['correct'][a]),
                                 [r['tca'][a][t] for t in TERMS]))
    return rows


def design(rows):
    groups, cells = defaultdict(list), defaultdict(int)
    for i, r in enumerate(rows):
        groups[r[2]].append(i)
        cells[(r[0], r[1])] += 1
    y = np.array([r[6] for r in rows])
    X = np.array([r[7] for r in rows])
    for idx in groups.values():
        y[idx] -= y[idx].mean()
        X[idx] -= X[idx].mean(0)
    w = np.array([1.0 / cells[(r[0], r[1])] for r in rows])
    return y, X, w


def fit(rows):
    y, X, w = design(rows)
    s = np.sqrt(w)
    beta, _ = nnls(-X * s[:, None], y * s)
    return beta


def cv_error(rows, folds=5):
    examples = sorted({(r[0], r[3]) for r in rows}, key=str)
    order = np.random.default_rng(SEED).permutation(len(examples))
    fold = {examples[i]: k % folds for k, i in enumerate(order)}
    err0 = err1 = 0.0
    for k in range(folds):
        train = [r for r in rows if fold[(r[0], r[3])] != k]
        test = [r for r in rows if fold[(r[0], r[3])] == k]
        beta = fit(train)
        y, X, w = design(test)
        err0 += float((w * y ** 2).sum())
        err1 += float((w * (y + X @ beta) ** 2).sum())
    return {'mse_no_difference': err0, 'mse_model': err1, 'relative_reduction': 1 - err1 / err0 if err0 else 0.0}


def rho_interval(rows):
    by_cell = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_cell[(r[0], r[1])][r[3]].append(r)
    rng, rhos = np.random.default_rng(SEED), []
    for _ in range(B_BOOT):
        sample = []
        for c, ex in by_cell.items():
            keys = list(ex)
            for j, k in enumerate(rng.integers(0, len(keys), len(keys))):
                sample += [(*r[:2], (*r[2], 'b', j), *r[3:]) for r in ex[keys[k]]]
        rhos.append(fit(sample)[3])
    return np.percentile(rhos, [2.5, 97.5]).tolist()


def cell_scores(rows, split):
    acc = defaultdict(list)
    for r in rows:
        if r[4] == split:
            acc[(r[0], r[1], r[5].split('@')[1], r[5].split('.')[1].split('@')[0])].append(r[6])
    return {k: 100 * float(np.mean(v)) for k, v in acc.items()}


def gate(rows, extra, beta):
    cells = sorted({(r[0], r[1], b) for r in rows for b in BUDGETS})
    a, b = cell_scores(rows, 'A'), cell_scores(rows, 'B')
    a_score = {s: float(np.mean([a[(*c, s)] for c in cells])) for s in SETTINGS}
    best = max(SETTINGS, key=lambda s: a_score[s])
    d_k3 = float(np.mean([b[(*c, best)] - b[(*c, 'k3v2pc')] for c in cells]))
    d_rdkv = float(np.mean([b[(*c, best)] - extra[(*c, 'rdkv')] for c in cells]))
    passed = d_k3 >= MARGIN and d_rdkv >= MARGIN and beta[3] > 0
    return {'call': 'PASS' if passed else 'STOP', 'best_setting_on_A': best, 'A_scores': a_score,
            'B_best_minus_k3v2pc': d_k3, 'B_best_minus_rdkv': d_rdkv, 'rho_positive': bool(beta[3] > 0),
            'B_cells': {'|'.join(c): {s: b[(*c, s)] for s in (*SETTINGS, 'k3v2pc')} | {'rdkv': extra[(*c, 'rdkv')]}
                        for c in cells}}


def rdkv_b(model, records, splits):
    acc = defaultdict(list)
    for r in records:
        key = r.get('index', r.get('doc_id'))
        if splits[key] != 'B':
            continue
        for bud in BUDGETS:
            a = f'Z.rdkv@{bud}'
            if r['benchmark'] == 'ruler':
                acc[(model, 'ruler', bud, 'rdkv')].append(r['scores'][a])
            else:
                acc[(model, 'quality', bud, 'rdkv')] += [100 * float(q['correct'][a]) for q in r['questions']]
    return {k: float(np.mean(v)) for k, v in acc.items()}


def analyze(models, manifest):
    rows, extra = [], {}
    for m, recs in models.items():
        splits = split_of(recs, manifest)
        rows += observations(m, recs, splits)
        extra.update(rdkv_b(m, recs, splits))
    train = [r for r in rows if r[4] == 'A']
    beta = fit(train)
    out = {'weights': dict(zip(('alpha', 'nu', 'kappa', 'rho'), map(float, beta))),
           'cv': cv_error(train), 'rho_interval95': rho_interval(train),
           'per_model': {m: dict(zip(('alpha', 'nu', 'kappa', 'rho'), map(float, fit([r for r in train if r[0] == m]))))
                         for m in models},
           'observations': {'A': len(train), 'B': len(rows) - len(train)}}
    out['gate1'] = gate(rows, extra, beta)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', nargs='+', action='append', required=True, metavar=('NAME', 'FILE'))
    parser.add_argument('--manifest', type=Path, default=Path('data/tca/manifest.json'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    models = {name: [json.loads(line) for f in files for line in open(f)] for name, *files in args.model}
    out = analyze(models, json.loads(args.manifest.read_text()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps({k: out[k] for k in ('weights', 'cv', 'rho_interval95')} | {'gate1': {
        k: v for k, v in out['gate1'].items() if k != 'B_cells'}}, indent=1))


if __name__ == '__main__':
    main()
