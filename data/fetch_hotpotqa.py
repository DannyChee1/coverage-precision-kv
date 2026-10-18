import argparse
import hashlib
import json
from pathlib import Path

REPO, FILE = 'hotpotqa/hotpot_qa', 'distractor/validation-00000-of-00001.parquet'
REVISION = '1908d6afbbead072334abe2965f91bd2709910ab'  # checked: it reproduces SHA256 below
SHA256 = '6906b710fe6793b450844c16d63429b87f5767ffd867aeeeef3c75ac5c2bf489'


def rows(table):
    for r in table.to_pylist():
        yield {'_id': r['id'], 'answer': r['answer'], 'question': r['question'],
               'supporting_facts': [[t, int(s)] for t, s in zip(r['supporting_facts']['title'],
                                                                r['supporting_facts']['sent_id'])],
               'context': [[t, list(s)] for t, s in zip(r['context']['title'], r['context']['sentences'])],
               'type': r['type'], 'level': r['level']}


def main():
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--revision', default=REVISION)
    args = parser.parse_args()
    path = hf_hub_download(REPO, FILE, repo_type='dataset', revision=args.revision)
    text = json.dumps(list(rows(pq.read_table(path))))
    digest = hashlib.sha256(text.encode()).hexdigest()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(args.out, digest, 'matches the manifests' if digest == SHA256 else f'DIFFERS from {SHA256}')
    raise SystemExit(digest != SHA256)


if __name__ == '__main__':
    main()
