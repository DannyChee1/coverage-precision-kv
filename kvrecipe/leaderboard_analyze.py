import argparse
import ast
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from analyze import score

TOLERANCE, AGREEMENT = 2.0, 0.90
RUNS = {'full': 'ruler__4096__meta-llama--Llama-3.1-8B-Instruct__no_press__0.00',
        'kvzip_plus@0.9375': 'ruler__4096__meta-llama--Meta-Llama-3.1-8B-Instruct__kvzip_plus__0.94',
        'kvzip_plus@0.96875': 'ruler__4096__meta-llama--Meta-Llama-3.1-8B-Instruct__kvzip_plus__0.97'}
QUOTED = re.compile(r"'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"")


def answers(cell):
    return [ast.literal_eval(item) for item in QUOTED.findall(cell)]


def task_mean(values, tasks):
    tasks = np.array(tasks)
    return float(np.mean([values[tasks == t].mean() for t in sorted(set(tasks))]))


def analyze(rows, leaderboard):
    out, tasks = {}, [r['task'] for r in rows]
    for arm, name in RUNS.items():
        ours, theirs = [], []
        for r in rows:
            if not r['reference_match'][arm]:
                raise ValueError(f"row {r['index']} does not match the leaderboard's row")
            ours.append(r['scores'][arm])
            theirs.append(r['reference_scores'][arm])
        ours, theirs = np.array(ours), np.array(theirs)
        ours_mean, theirs_mean = task_mean(ours, tasks), task_mean(theirs, tasks)
        row = {'ours': ours_mean, 'leaderboard_same_examples': theirs_mean, 'difference': ours_mean - theirs_mean,
               'agreement': float(np.mean(np.abs(ours - theirs) < 1e-9))}
        if arm != 'full':
            row['evicted_share'] = float(np.mean([r['evicted_share'][arm] for r in rows]))
            row['reproduces'] = bool(abs(row['difference']) <= TOLERANCE and row['agreement'] >= AGREEMENT)
        out[arm] = row
    compressed = [a for a in RUNS if a != 'full']
    return {'call': 'REPRODUCED' if all(out[a]['reproduces'] for a in compressed) else 'NOT REPRODUCED',
            'arms': out, 'examples': len(rows), 'tolerance_pts': TOLERANCE, 'agreement_floor': AGREEMENT}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('files', nargs='+')
    parser.add_argument('--leaderboard', default='external/kvpress-leaderboard')
    parser.add_argument('--out')
    args = parser.parse_args()
    rows = {}
    for f in args.files:
        for line in open(f):
            r = json.loads(line)
            rows[r['index']] = r
    out = analyze([rows[k] for k in sorted(rows)], args.leaderboard)
    text = json.dumps(out, indent=1)
    if args.out:
        Path(args.out).write_text(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
