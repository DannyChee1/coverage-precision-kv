import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).parent))
from analyze import score
from leaderboard_analyze import RUNS, answers as leaderboard_answers
from rdkv_fidelity_analyze import score as fidelity_score
from review_longbench_analyze import degenerate
from longbench_metrics import score_example

DROP = {'answer', 'gold', 'docs', 'context', 'question', 'prompt', 'generated', 'text'}
SNAPSHOT = re.compile(r'(?:.*/)?models--([^/]+)--([^/]+)/snapshots/([0-9a-f]{40})/?$')


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def model_id(path):
    m = SNAPSHOT.match(path or '')
    return f'{m.group(1)}/{m.group(2)}@{m.group(3)}' if m else path


def arm_score(record, output):
    if record.get('benchmark') == 'longbench':
        return score_example(record['task'], output, record['gold'])
    if 'all_classes' in record:
        return fidelity_score(record['task'], output, record['gold'], record['all_classes'])
    return score(record['task'], output, record['answer'])


def strip(value, key=None):
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            if k in DROP and not (k == 'gold' and is_numeric(v)):
                continue
            if k == 'answers' and isinstance(v, dict) and all(isinstance(x, str) for x in v.values()):
                out['output_sha'] = {a: sha(x) for a, x in v.items()}
                continue
            out[k] = strip(v, k)
        return out
    if isinstance(value, list):
        return [strip(v, key) for v in value]
    return value


def is_numeric(value):
    return isinstance(value, list) and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in value)


def export(record, leaderboard=None):
    out = strip(record)
    outputs = record.get('answers')
    if isinstance(outputs, dict) and all(isinstance(x, str) for x in outputs.values()):
        out['scores'] = {a: arm_score(record, x) for a, x in outputs.items()}
        if record.get('benchmark') == 'longbench' or 'frontier' in record:
            out['think'] = {a: '<think>' in x for a, x in outputs.items()}
            out['degenerate'] = {a: degenerate(x) for a, x in outputs.items()}
    if 'model' in out:
        out['model'] = model_id(out['model'])
    if leaderboard is not None:
        refs = {arm: frame.iloc[record['index']] for arm, frame in leaderboard.items()}
        out['reference_match'] = {arm: bool(ref['task'] == record['task'] and sorted(
            leaderboard_answers(ref['answer'])) == sorted(record['answer'])) for arm, ref in refs.items()}
        out['reference_scores'] = {arm: score(record['task'], ref['predicted_answer'], record['answer'])
                                   for arm, ref in refs.items()}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('raw', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('files', nargs='+')
    parser.add_argument('--leaderboard', type=Path, help='the published predictions, for the leaderboard records')
    args = parser.parse_args()
    frames = None
    if args.leaderboard:
        import pandas as pd
        frames = {arm: pd.read_csv(args.leaderboard / name / 'predictions.csv', keep_default_na=False)
                  for arm, name in RUNS.items()}
    args.out.mkdir(parents=True, exist_ok=True)
    for name in args.files:
        src = args.raw / name
        if name.endswith('.jsonl'):
            lines = [json.dumps(export(json.loads(line), frames), separators=(',', ':')) for line in open(src)
                     if line.strip()]
            (args.out / name).write_text(''.join(line + '\n' for line in lines))
        else:
            (args.out / name).write_text(json.dumps(export(json.load(open(src))), indent=1) + '\n')
        print(name)


if __name__ == '__main__':
    main()
