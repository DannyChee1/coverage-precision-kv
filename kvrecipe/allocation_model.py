import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

QUANTIZERS = ('sep', 'pc', 'rk', 'cen', 'rot')
BENCHES = ('ruler', 'quality', 'longhealth')
TAU = 1.5
D, HEADS, T_REF, SINKS = 128, 8, 3900, 4
FORMS = ('mult', 'logit', 'cov')


def part_bits(b):
    return D * 16 if b == 16 else D * b + (D // 64) * 32


def overhead(kq, kb):
    if kb == 16:
        return 64
    return 64 + {'sep': D * 16, 'cen': 2 * D * 16, 'rot': 0, 'rk': D * 16, 'pc': 63 * D * 2 * 16 // 64}[kq]


def coverage(budget, kb, kq, vb, n_tokens=T_REF):
    per_head = budget / 100 * n_tokens * 2 * part_bits(16)
    left = per_head - SINKS * 2 * part_bits(16) - overhead(kq, kb)
    return (max(0, int(left // (part_bits(kb) + part_bits(vb)))) + SINKS) / n_tokens


MODELS = ('llama', 'qwen3')
SHARED_Q = ('pc', 'rk', 'cen', 'rot')  # key-quantizer multipliers relative to sep, shared by both models
PER_MODEL = ('beta', 'theta', 'off_quality', 'off_longhealth', 'kscale', 'nu')


class Model:

    def __init__(self, form, rho):
        self.form, self.rho = form, rho

    def names(self):
        per = PER_MODEL[:4] if self.form == 'cov' else PER_MODEL
        shared = () if self.form == 'cov' else tuple(f'mult_{q}' for q in SHARED_Q)
        return [f'{m}.{k}' for m in MODELS for k in per] + list(shared)

    def unpack(self, p):
        per = len(PER_MODEL[:4] if self.form == 'cov' else PER_MODEL)
        params = {m: dict(zip(PER_MODEL, p[i * per:(i + 1) * per])) for i, m in enumerate(MODELS)}
        mult = {'sep': 1.0, 'none': 0.0, **dict(zip(SHARED_Q, p[2 * per:]))}
        return params, mult

    def ratio(self, p, rows):
        params, mult = self.unpack(p)
        out = []
        for r in rows:
            q = params[r['model']]
            z = q['beta'] * (np.log(r['f']) - q['theta'] - {'ruler': 0.0, 'quality': q['off_quality'],
                                                             'longhealth': q['off_longhealth']}[r['bench']])
            if self.form == 'cov':
                out.append(1 / (1 + np.exp(-z)))
                continue
            pen = (q['kscale'] * (mult[r['kq']] if r['kb'] < 16 else 0.0) * 2.0 ** (-self.rho * r['kb'])
                   + q['nu'] * 2.0 ** (-self.rho * r['vb']))
            out.append(np.exp(-pen) / (1 + np.exp(-z)) if self.form == 'mult' else 1 / (1 + np.exp(-(z - pen))))
        return np.array(out)

    def predict(self, p, rows):
        r = self.ratio(p, rows)
        return np.array([x['chance'] + (x['full'] - x['chance']) * ri for x, ri in zip(rows, r)])

    def fit(self, rows):
        s = np.array([r['score'] for r in rows])
        sd = np.sqrt(s * (100 - s) / np.array([r['n'] for r in rows]) + TAU ** 2)
        per = 4 if self.form == 'cov' else 6
        x0 = np.array(([2.0, -3.0, 0.0, 0.0] + ([10.0, 1.0] if per == 6 else [])) * 2
                      + ([1.0] * len(SHARED_Q) if per == 6 else []))
        lo = np.array(([0.1, -8, -3, -3] + ([0.0, 0.0] if per == 6 else [])) * 2
                      + ([0.0] * len(SHARED_Q) if per == 6 else []))
        hi = np.array(([20, 2, 3, 3] + ([1e6, 1e6] if per == 6 else [])) * 2
                      + ([50.0] * len(SHARED_Q) if per == 6 else []))
        res = least_squares(lambda p: (self.predict(p, rows) - s) / sd, x0, bounds=(lo, hi))
        p = res.x.copy()
        params, mult = self.unpack(p)
        if self.form != 'cov':
            present_q = {r['kq'] for r in rows if r['kb'] < 16}
            for i, q in enumerate(SHARED_Q):
                if q not in present_q:
                    p[2 * per + i] = 1.0
        for i, m in enumerate(MODELS):
            present_b = {r['bench'] for r in rows if r['model'] == m}
            for j, b in ((2, 'quality'), (3, 'longhealth')):
                if b not in present_b:
                    p[i * per + j] = 0.0
        return p, float(res.cost)


def usable(table):
    return [r for r in table if r['arm'] != 'full' and r['run'] != 'diagnose']


def fit_all(rows, form, rho):
    model = Model(form, rho)
    return model.fit(rows), model


def support(train, r):
    t = [x for x in train if x['model'] == r['model']]
    fs = [x['f'] for x in t]
    return bool(any((x['kb'], x['kq']) == (r['kb'], r['kq']) for x in t) and any(x['vb'] == r['vb'] for x in t)
                and min(fs) * 0.95 <= r['f'] <= max(fs) * 1.05)


def loro(rows, form, rho):
    preds = []
    for run in sorted({r['run'] for r in rows}):
        train = [r for r in rows if r['run'] != run]
        test = [r for r in rows if r['run'] == run]
        (p, _), model = fit_all(train, form, rho)
        for r, y in zip(test, model.predict(p, test)):
            preds.append(dict(r, pred=float(y), in_support=support(train, r)))
    return preds


def same_arm_baseline(rows):
    out = []
    for r in rows:
        others = [x for x in rows if x['run'] != r['run'] and all(x[k] == r[k] for k in
                  ('model', 'bench', 'kb', 'kq', 'vb')) and abs(x['f'] / r['f'] - 1) < 0.05]
        if others:
            ratio = np.mean([(x['score'] - x['chance']) / (x['full'] - x['chance']) for x in others])
            out.append(dict(r, pred=float(r['chance'] + (r['full'] - r['chance']) * ratio)))
    return out


def metrics(preds):
    err = np.array([p['pred'] - p['score'] for p in preds])
    pairs = []
    groups = {}
    for p in preds:
        groups.setdefault((p['run'], p['model'], p['bench'], p['budget']), []).append(p)
    for g in groups.values():
        for a, b in itertools.combinations(g, 2):
            obs, pr = a['score'] - b['score'], a['pred'] - b['pred']
            pairs.append((obs, pr))
    pairs = np.array(pairs) if pairs else np.zeros((0, 2))
    big = pairs[np.abs(pairs[:, 0]) >= 1.0] if len(pairs) else pairs
    return {'configs': len(preds), 'mae': float(np.abs(err).mean()), 'rmse': float(np.sqrt((err ** 2).mean())),
            'bias': float(err.mean()), 'pairs_same_budget': len(pairs),
            'pair_diff_mae': float(np.abs(pairs[:, 0] - pairs[:, 1]).mean()) if len(pairs) else None,
            'pairs_obs_ge_1pt': len(big),
            'pair_sign_agreement_obs_ge_1pt': float((np.sign(big[:, 0]) == np.sign(big[:, 1])).mean()) if len(big) else None}


def by_run(preds):
    out = {}
    for run in sorted({p['run'] for p in preds}):
        e = [p['pred'] - p['score'] for p in preds if p['run'] == run]
        out[run] = {'configs': len(e), 'mae': float(np.mean(np.abs(e))), 'bias': float(np.mean(e))}
    return out


CANDIDATES = [(16, 'none', 16)] + [(kb, q, vb) for kb in (2, 3, 4, 5, 6, 8) for q in ('sep', 'pc', 'rk')
                                    for vb in (1, 2, 3, 4, 8)]


def rule(p, model, tested, budgets=(1.5625, 3.125, 6.25, 12.5), bench='ruler', full=96.0):
    out = {}
    for m in MODELS:
        out[m] = {}
        for b in budgets:
            cands = []
            for kb, q, vb in CANDIDATES:
                f = coverage(b, kb, q, vb)
                if f <= SINKS / T_REF:
                    continue
                row = dict(model=m, f=min(f, 1.0), kb=kb, kq=q, vb=vb, bench=bench, chance=0.0, full=full)
                cands.append(dict(config=f'k{kb}{q}v{vb}' if kb < 16 else 'evict16', f=round(min(f, 1.0), 4),
                                  pred=float(model.predict(p, [row])[0]),
                                  tested=(m, kb, q, vb) in tested))
            cands.sort(key=lambda c: -c['pred'])
            ref = next(c for c in cands if c['config'] == 'k3pcv2')
            out[m][str(b)] = {'top': cands[:6], 'k3pcv2': ref,
                              'best_tested': next(c for c in cands if c['tested'])}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--table', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--rho', type=float, default=2.0)
    args = parser.parse_args()
    rows = usable(json.loads(args.table.read_text()))
    tested = {(r['model'], r['kb'], r['kq'], r['vb']) for r in rows}
    result = {'rows': len(rows), 'rho': args.rho, 'tau': TAU, 'forms': {}}
    for form in FORMS:
        preds = loro(rows, form, args.rho)
        (p, cost), model = fit_all(rows, form, args.rho)
        inside = [x for x in preds if x['in_support']]
        result['forms'][form] = {
            'loro': metrics(preds), 'loro_in_support': metrics(inside),
            'loro_out_of_support': metrics([x for x in preds if not x['in_support']]),
            'loro_by_run': by_run(preds),
            'loro_predictions': [{k: x[k] for k in ('run', 'model', 'bench', 'arm', 'f', 'score', 'pred', 'in_support')}
                                 for x in preds],
            'fit': {'params': dict(zip(model.names(), map(float, p))), 'cost': cost},
            'rule': rule(p, model, tested)}
    base = same_arm_baseline(rows)
    result['same_arm_baseline'] = metrics(base)
    result['same_arm_baseline_by_run'] = by_run(base)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + '\n')
    for form in FORMS:
        f = result['forms'][form]
        print(form, 'all', json.dumps(f['loro']))
        print(form, 'in-support', json.dumps(f['loro_in_support']))
    print('same-arm baseline', json.dumps(result['same_arm_baseline']))


if __name__ == '__main__':
    main()
