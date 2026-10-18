import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from analyze import score

ROWS = ('k4v4', 'k4v2', 'k2v4', 'k2v2')
B, SEED = 10_000, 0


def model_reading(rows, confirm):
    tasks = sorted({r['task'] for r in rows})
    s = {a: {t: np.array([r['scores'][a] for r in rows if r['task'] == t]) for t in tasks}
         for a in ROWS}
    means = {a: float(np.mean([s[a][t].mean() for t in tasks])) for a in ROWS}
    rng = np.random.default_rng(SEED)
    draws = np.zeros(B)
    for t in tasks:
        idx = rng.integers(0, len(s['k4v2'][t]), size=(B, len(s['k4v2'][t])))
        draws += (s['k4v2'][t][idx].mean(1) - s['k2v4'][t][idx].mean(1)) / len(tasks)
    asym = {'A_S': means['k4v2'] - means['k2v4'], 'ci95': np.percentile(draws, [2.5, 97.5]).tolist(),
            'delta_S_K': means['k4v4'] - means['k2v4'], 'delta_S_V': means['k4v4'] - means['k4v2']}
    metrics = {p: {k: float(np.mean([r['metrics'][p][k] for r in rows])) for k in rows[0]['metrics'][p]}
               for p in ('kept', 'full')}
    m = metrics['kept']
    supported = m['D16_K2'] >= 2 * m['E_V2'] and asym['ci95'][0] > 0
    against = m['D16_K2'] <= m['E_V2'] or asym['ci95'][1] < 0
    reading = 'SUPPORTED' if supported else 'NOT SUPPORTED' if against else 'INCONCLUSIVE'
    diag = {p: {'E_K2_over_E_V2': metrics[p]['E_K2'] / metrics[p]['E_V2'],
                'D16_K2_over_E_V2': metrics[p]['D16_K2'] / metrics[p]['E_V2'],
                'R_V': {b: metrics[p][f'E_V{b}'] / metrics[p][f'e_V{b}'] for b in (2, 3, 4)},
                'R_K': {b: metrics[p][f'E_K{b}'] / metrics[p][f'e_K{b}'] for b in (2, 3, 4)},
                'joint_E_K4V2_below_E_K3V3': metrics[p]['E_K4V2'] < metrics[p]['E_K3V3']} for p in metrics}
    kept = [r['metrics']['kept'] for r in rows]
    leak = {'leak_mean': float(np.mean([k['leak_mean'] for k in kept])),
            'leak_max': float(max(k['leak_max'] for k in kept)),
            'share_examples_leak_max_over_0.01': float(np.mean([k['leak_max'] > 0.01 for k in kept]))}
    leak['flag'] = bool(leak['leak_mean'] > 0.001 or leak['share_examples_leak_max_over_0.01'] > 0.05)
    conf = {r['index']: r for r in confirm}
    agree = [abs(r['scores']['k4v2']
                 - conf[r['index']]['scores']['k4v2']) < 1e-9 for r in rows]
    return {'reading': reading, 'harness_leak_check': leak, 'asymmetry': asym, 'scores': means, 'metrics': metrics,
            'diagnostics': diag,
            'per_task': {t: {a: float(s[a][t].mean()) for a in ROWS} for t in tasks},
            'k4v2_row_matches_confirmation': float(np.mean(agree)), 'examples': len(rows),
            'seconds': {k: float(np.mean([r[k] for r in rows])) for k in ('score_seconds', 'probe_seconds',
                                                                          'decode_seconds')}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', nargs=2, action='append', required=True, metavar=('NAME', 'FILE'))
    parser.add_argument('--confirm', nargs='+', action='append', required=True)
    parser.add_argument('--out')
    args = parser.parse_args()
    out = {}
    for (name, path), conf in zip(args.model, args.confirm):
        rows = [json.loads(line) for line in open(path)]
        out[name] = model_reading(rows, [json.loads(line) for p in conf for line in open(p)])
    readings = {r['reading'] for r in out.values()}
    overall = readings.pop() if readings in ({'SUPPORTED'}, {'NOT SUPPORTED'}) else 'MIXED (' + ', '.join(
        f'{n}: {r["reading"]}' for n, r in out.items()) + ')'
    out = {'reading': overall, 'per_model': out, 'bootstrap': {'B': B, 'seed': SEED}}
    text = json.dumps(out, indent=1)
    if args.out:
        Path(args.out).write_text(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
