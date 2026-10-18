import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import controls_analyze as ca
import tca_run as run

ALPHA, BUDGET, MARGIN = 0.05, '1.5625', 1.0
SWITCHES = {'a': 'key weight (squared to linear)', 'b': 'budget (per layer to per head)',
            'c': 'K/V split (searched to fixed 0.5)', 'd': '3-bit keys (allowed to not)',
            'e': 'value and coverage weighting (raw x6 to pooled x1)'}
TEST_FRESH12 = {'tca2-k3v2pc': {'llama': 82.8 - 77.7, 'qwen3': 79.1 - 68.6},
                'rdkvpc-k3v2pc': {'llama': 37.3 - 77.7, 'qwen3': 41.4 - 68.6}}


def arm(split):
    return f'Z.{split}@{BUDGET}'


def stat(cells, terms):
    delta = float(np.mean([sum(w * acc[arm(a)] for a, w in terms) for acc, _ in cells]))
    d = np.mean([sum(w * dr[arm(a)] for a, w in terms) for _, dr in cells], axis=0)
    return delta, d


def summarize(delta, d):
    return {'delta': delta, 'ci95': np.percentile(d, [2.5, 97.5]).tolist(), 'p_one_sided': float(np.mean(d <= 0))}


def analyze(models, expected=325):
    run.configure('ablate')
    ca.EXPECTED['ruler'] = (expected, expected)
    cells, per, valid = [], {}, True
    for m, recs in models.items():
        out = ca.summary([r for r in recs if r['benchmark'] == 'ruler'], 'ruler')
        acc, draws = out.pop('_acc_draws')
        per[m] = out
        valid &= out['valid_all']
        cells.append((acc, draws))
    T, dT = stat(cells, [('tca2', 1), ('rdkvpc', -1)])
    deltas = {x: stat(cells, [('tca2', 1), (f'tca2{x}', -1)]) for x in SWITCHES}
    I_terms = [('tca2', 1), ('rdkvpc', -1)] + [(t, w) for x in SWITCHES for t, w in (('tca2', -1), (f'tca2{x}', 1))]
    I, dI = stat(cells, I_terms)
    order = sorted(SWITCHES, key=lambda x: float(np.mean(deltas[x][1] <= 0)))
    still, calls = True, {}
    for rank, x in enumerate(order):
        delta, d = deltas[x]
        s = summarize(delta, d)
        threshold = ALPHA / (len(order) - rank)
        holm = still and s['p_one_sided'] <= threshold
        still = holm
        lo, hi = s['ci95']
        carries = holm and delta >= MARGIN
        if carries:
            call = 'DOMINANT' if delta >= 0.5 * T else 'CARRIES PART'
        elif hi < 0:
            call = 'HURTS TCA2'
        elif lo <= 0 <= hi and abs(delta) < 2.0:
            call = 'NO MEASURABLE EFFECT'
        else:
            call = 'UNRESOLVED'
        calls[x] = {**s, 'switch': SWITCHES[x], 'holm_threshold': threshold, 'holm_pass': bool(holm), 'call': call,
                    'share_of_T': delta / T if T else None,
                    'per_model': {m: c[0][arm('tca2')] - c[0][arm(f'tca2{x}')] for m, c in zip(models, cells)}}
    sI = summarize(I, dI)
    interactions = (sI['ci95'][0] > 0 or sI['ci95'][1] < 0) and abs(I) >= 0.25 * abs(T)
    sanity = {k: {'fresh13': {m: c[0][arm(a)] - c[0][arm('k3v2pc')] for m, c in zip(models, cells)}, 'fresh12_test': v}
              for (k, v), a in zip(TEST_FRESH12.items(), ('tca2', 'rdkvpc'))}
    return {'T tca2-rdkvpc': {**summarize(T, dT), 'per_model': {m: c[0][arm('tca2')] - c[0][arm('rdkvpc')]
                                                                for m, c in zip(models, cells)}},
            'switches': calls, 'residual_I': {**sI, 'interactions_matter': bool(interactions)},
            'valid_all': bool(valid), 'sanity': sanity, 'per_model': per,
            'rule': {'alpha': ALPHA, 'margin': MARGIN, 'dominant_share': 0.5, 'interaction_share': 0.25,
                     'no_effect_abs': 2.0, 'bootstrap': {'B': ca.B, 'seed': ca.SEED}, 'expected': expected}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', nargs='+', action='append', required=True, metavar=('NAME', 'FILE'))
    parser.add_argument('--expected', type=int, default=325)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = analyze({name: ca.load(files) for name, *files in args.model}, args.expected)
    for m in out['per_model'].values():
        m.pop('_acc_draws', None)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps({'T': out['T tca2-rdkvpc'], 'calls': {x: (c['delta'], c['call']) for x, c in out['switches'].items()},
                      'I': out['residual_I'], 'valid_all': out['valid_all']}, indent=1))


if __name__ == '__main__':
    main()
