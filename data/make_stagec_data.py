import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

QUALITY_COMMIT = 'f84977c40dbfef70c9cab48037b7becfc8e45f73'
LONGHEALTH_COMMIT = '479922279ce8fa528fe2cd87d854c6a2f8365d21'
RAW = 'https://raw.githubusercontent.com'
SOURCES = {
    'QuALITY.v1.0.1.htmlstripped.dev':
        f'{RAW}/nyu-mll/quality/{QUALITY_COMMIT}/data/v1.0.1/QuALITY.v1.0.1.htmlstripped.dev',
    'benchmark_v5.json': f'{RAW}/kbressem/LongHealth/{LONGHEALTH_COMMIT}/data/benchmark_v5.json',
}
LETTERS = 'ABCDE'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def quality(path):
    docs = {}
    for line in open(path):
        r = json.loads(line)
        doc = docs.setdefault(r['article_id'], {'doc_id': f"quality-{r['article_id']}", 'benchmark': 'quality',
                                                'title': r['title'], 'context': r['article'].strip(),
                                                'questions': []})
        if doc['context'] != r['article'].strip():
            raise ValueError(f"article {r['article_id']} differs between its sets")
        for q in r['questions']:
            if len(q['options']) != 4 or q['gold_label'] not in (1, 2, 3, 4):
                raise ValueError(f"question {q['question_unique_id']} is not 4-way with a gold label")
            doc['questions'].append({'qid': q['question_unique_id'], 'question': q['question'],
                                     'options': q['options'], 'gold': [q['gold_label'] - 1],
                                     'difficult': q['difficult']})
    return list(docs.values())


def longhealth(path):
    docs = []
    for pid, p in json.load(open(path)).items():
        context = '\n\n'.join(f'<{k}>\n{v.strip()}\n</{k}>' for k, v in p['texts'].items())
        questions = []
        for q in p['questions']:
            options = [q[f'answer_{c}'] for c in 'abcde']
            gold = [i for i, o in enumerate(options) if o == q['correct']]
            if not gold:
                raise ValueError(f"{pid} question {q['No']}: the correct answer is not an option")
            questions.append({'qid': f"{pid}-{q['No']}", 'question': q['question'], 'options': options,
                              'gold': gold, 'difficult': None})
        docs.append({'doc_id': f'longhealth-{pid}', 'benchmark': 'longhealth', 'title': pid, 'context': context,
                     'questions': questions})
    return docs


def smoke(q_docs, l_docs):
    by_len = sorted(q_docs, key=lambda d: (len(d['context']), d['doc_id']))
    picks = [by_len[0], by_len[len(by_len) // 2], by_len[-1]]
    return picks + [max(l_docs, key=lambda d: (len(d['context']), d['doc_id']))]


def write(path, docs):
    with open(path, 'w') as handle:
        for d in docs:
            handle.write(json.dumps(d) + '\n')
    return sha256(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=Path, help='directory holding the source files (downloaded if missing)')
    parser.add_argument('--out', type=Path, default=Path('data/stagec'))
    args = parser.parse_args()
    src = args.src or args.out / 'src'
    src.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, url in SOURCES.items():
        if not (src / name).exists():
            urllib.request.urlretrieve(url, src / name)
    q_docs = quality(src / 'QuALITY.v1.0.1.htmlstripped.dev')
    l_docs = longhealth(src / 'benchmark_v5.json')
    manifest = {
        'quality': {'commit': QUALITY_COMMIT, 'file': 'data/v1.0.1/QuALITY.v1.0.1.htmlstripped.dev',
                    'documents': len(q_docs), 'questions': sum(len(d['questions']) for d in q_docs),
                    'difficult': sum(q['difficult'] for d in q_docs for q in d['questions']),
                    'jsonl_sha256': write(args.out / 'quality.jsonl', q_docs)},
        'longhealth': {'commit': LONGHEALTH_COMMIT, 'file': 'data/benchmark_v5.json', 'documents': len(l_docs),
                       'questions': sum(len(d['questions']) for d in l_docs),
                       'several_gold': [q['qid'] for d in l_docs for q in d['questions'] if len(q['gold']) > 1],
                       'jsonl_sha256': write(args.out / 'longhealth.jsonl', l_docs)},
        'source_sha256': {name: sha256(src / name) for name in SOURCES},
    }
    picks = smoke(q_docs, l_docs)
    manifest['smoke'] = {'documents': [d['doc_id'] for d in picks],
                         'questions': sum(len(d['questions']) for d in picks),
                         'jsonl_sha256': write(args.out / 'smoke.jsonl', picks)}
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
    print(json.dumps(manifest, indent=1))


if __name__ == '__main__':
    main()
