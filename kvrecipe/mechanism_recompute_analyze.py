import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from mechanism_analyze import model_reading

METRICS = [f'D{top}_K{b}' for top in (16, 64) for b in (2, 3, 4)]


def overall(readings):
    names = set(readings.values())
    if names in ({'SUPPORTED'}, {'NOT SUPPORTED'}):
        return names.pop()
    return 'MIXED (' + ', '.join(f'{n}: {r}' for n, r in readings.items()) + ')'


def rule(d16, e_v2, ci):
    if d16 >= 2 * e_v2 and ci[0] > 0:
        return 'SUPPORTED'
    if d16 <= e_v2 or ci[1] < 0:
        return 'NOT SUPPORTED'
    return 'INCONCLUSIVE'


def model_recompute(rows, confirm, recompute):
    by_index = {r['index']: r['metrics'] for r in recompute}
    original = model_reading(rows, confirm)
    fixed_rows = copy.deepcopy(rows)
    check = 0.0
    for r in fixed_rows:
        m = by_index[r['index']]
        for key in METRICS:
            check = max(check, abs(m[key]['stage_a'] - r['metrics']['kept'][key]))
            r['metrics']['kept'][key] = m[key]['per_head']
    corrected = model_reading(fixed_rows, confirm)
    e_v2, ci = original['metrics']['kept']['E_V2'], original['asymmetry']['ci95']
    per_layer = {}
    for key in METRICS:
        values = [by_index[r['index']][key]['per_layer'] for r in rows]
        defined = [v for v in values if v is not None]
        per_layer[key] = {'mean': float(np.mean(defined)) if defined else None, 'examples': len(defined)}
    layers = {top: {k: int(sum(by_index[r['index']][f'layers_top{top}'][k] for r in rows))
                    for k in ('all_heads', 'any_head', 'total')} for top in (16, 64)}
    d16 = {'original': original['metrics']['kept']['D16_K2'], 'per_head': corrected['metrics']['kept']['D16_K2'],
           'per_layer': per_layer['D16_K2']['mean']}
    return {
        'reading_original': original['reading'], 'reading_corrected': corrected['reading'],
        'reading_per_layer_variant': rule(d16['per_layer'], e_v2, ci) if d16['per_layer'] is not None else None,
        'D16_K2': d16, 'E_V2': e_v2, 'bar_2x_E_V2': 2 * e_v2,
        'ratio_D16_K2_over_E_V2': {k: v / e_v2 for k, v in d16.items() if v is not None},
        'A_S': original['asymmetry'],
        'kept_D': {key: {'original': original['metrics']['kept'][key], 'per_head': corrected['metrics']['kept'][key],
                         'per_layer': per_layer[key]} for key in METRICS},
        'full_context_D16_K2': original['metrics']['full']['D16_K2'],
        'layers': layers, 'stage_a_recomputed_max_abs_diff': check, 'examples': len(rows),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--recompute', type=Path, required=True)
    parser.add_argument('--model', nargs=2, action='append', required=True, metavar=('NAME', 'FILE'))
    parser.add_argument('--confirm', nargs='+', action='append', required=True)
    parser.add_argument('--out')
    args = parser.parse_args()
    recompute = [json.loads(line) for line in args.recompute.read_text().splitlines()]
    out = {}
    for (name, path), conf in zip(args.model, args.confirm):
        rows = [json.loads(line) for line in open(path)]
        out[name] = model_recompute(rows, [json.loads(line) for p in conf for line in open(p)],
                                    [r for r in recompute if r['model'] == name])
    result = {'reading_original': overall({n: r['reading_original'] for n, r in out.items()}),
              'reading_corrected': overall({n: r['reading_corrected'] for n, r in out.items()}),
              'per_model': out}
    text = json.dumps(result, indent=1)
    if args.out:
        Path(args.out).write_text(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
