import argparse
import json
from pathlib import Path


def correctness(files, row):
    out = {}
    for path in files:
        with open(path) as handle:
            for line in handle:
                for q in json.loads(line).get('questions', []):
                    if row in q.get('correct', {}):
                        out[q['qid']] = bool(q['correct'][row])
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--a', nargs='+', required=True)
    parser.add_argument('--b', nargs='+', required=True)
    parser.add_argument('--row', default='full')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    a, b = correctness(args.a, args.row), correctness(args.b, args.row)
    common = sorted(set(a) & set(b))
    agree = sum(a[q] == b[q] for q in common)
    text = json.dumps({'questions': len(common), 'agree': agree, 'agreement': agree / len(common),
                       'accuracy_a': 100 * sum(a[q] for q in common) / len(common),
                       'accuracy_b': 100 * sum(b[q] for q in common) / len(common)}, indent=1)
    if args.out:
        args.out.write_text(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
