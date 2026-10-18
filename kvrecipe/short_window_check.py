import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from mix import N_SINK
from controls import BUDGETS
import qaprior as qp


def floor_tokens(budget):
    return math.ceil(N_SINK / BUDGETS[budget])


def prompt_tokens(tok, context):
    return len(tok(qp.prompt_text(tok, context), add_special_tokens=False)['input_ids'])


def scan(tok, paths, budget):
    floor, rows, short, shortest = floor_tokens(budget), 0, [], None
    for path in paths:
        for line in open(path):
            row = json.loads(line)
            n = prompt_tokens(tok, row['context'])
            rows += 1
            shortest = n if shortest is None else min(shortest, n)
            if n < floor:
                short.append({'file': Path(path).name, 'index': row.get('index', row.get('doc_id')),
                              'task': row.get('task', row.get('benchmark')), 'tokens': n})
    return {'budget': budget, 'floor_tokens': floor, 'rows': rows, 'shortest': shortest, 'short': short,
            'passes': rows > 0 and not short}


def main():
    from transformers import AutoTokenizer
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer', required=True)
    parser.add_argument('--budget', default='1.5625', choices=sorted(BUDGETS))
    parser.add_argument('rows', nargs='+')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    out = scan(AutoTokenizer.from_pretrained(args.tokenizer), args.rows, args.budget)
    if args.out:
        args.out.write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps({k: v for k, v in out.items() if k != 'short'} | {'short_rows': len(out['short'])}))
    return 0 if out['passes'] else 1


if __name__ == '__main__':
    sys.exit(main())
