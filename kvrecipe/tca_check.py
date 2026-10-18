import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from analyze import score
import tca_fit as fit


def model_of(record):
    return 'llama' if 'llama' in record['model'].lower() else 'qwen3'


def cell_scores(records, splits, arm):
    acc = defaultdict(list)
    for r in records:
        if splits[r.get('index', r.get('doc_id'))] != 'B':
            continue
        for b in fit.BUDGETS:
            a = f'Z.{arm}@{b}'
            key = (model_of(r), r['benchmark'], b)
            if r['benchmark'] == 'ruler':
                acc[key].append(r['scores'][a])
            else:
                acc[key] += [100 * float(q['correct'][a]) for q in r['questions']]
    return {k: float(np.mean(v)) for k, v in acc.items()}, {k: len(v) for k, v in acc.items()}


def choose(check, calib, manifest):
    splits = {}  # from the calibration records only: they hold all 15 examples per task, the check only split B
    for m in ('llama', 'qwen3'):
        splits.update(fit.split_of([r for r in calib if model_of(r) == m], manifest))
    fitted, n_fit = cell_scores(check, splits, 'tcafit')
    grid, n_grid = cell_scores(calib, splits, 'tca2')
    cells = sorted(grid)
    if len(cells) != 8 or sorted(fitted) != cells or any(n_fit[c] != n_grid[c] for c in cells):
        raise SystemExit(f'check and calibration cover different split-B data: {n_fit} vs {n_grid}')
    pooled = {'tcafit': float(np.mean([fitted[c] for c in cells])), 'tca2': float(np.mean([grid[c] for c in cells]))}
    primary = 'tcafit' if pooled['tcafit'] > pooled['tca2'] else 'tca2'
    return {'primary': primary, 'secondary': 'tca2' if primary == 'tcafit' else 'tcafit', 'B_pooled': pooled,
            'B_cells': {'|'.join(c): {'tcafit': fitted[c], 'tca2': grid[c], 'n': n_grid[c]} for c in cells},
            'rule': 'higher equal-weight mean over the 8 split-B cells is primary; a tie goes to tca2'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', type=Path, nargs='+', required=True)
    parser.add_argument('--calib', type=Path, nargs='+', required=True)
    parser.add_argument('--manifest', type=Path, default=Path('data/tca/manifest.json'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    read = lambda paths: [json.loads(line) for p in paths for line in open(p)]
    out = choose(read(args.check), read(args.calib), json.loads(args.manifest.read_text()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
