import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'kvrecipe'))
from answer_blind_data import SQUAD, sentence_span

HOTPOT = Path('external/RULER/scripts/data/synthetic/json/hotpotqa.json')
RULER = Path('data/ruler-fresh19-4096')
DOCS = Path('data/answerblind')


def ruler_rows(path):
    squad = {}
    for article in json.load(open(SQUAD))['data']:
        for p in article['paragraphs']:
            for q in p['qas']:
                if q['answers']:
                    squad[q['question'].strip()] = (p['context'], q['answers'][0])
    hotpot = {d['question'].strip(): d for d in json.load(open(HOTPOT))}
    out = []
    for row in map(json.loads, open(path)):
        if not row['task'].startswith('qa'):
            out.append({**row, 'evidence_spans': []})
            continue
        question = row['question'].rsplit('Question:', 1)[-1].strip()
        context, spans = row['context'], []
        if row['task'] == 'qa_1':
            paragraph, answer = squad[question]
            at = context.index(paragraph)
            s, e = sentence_span(paragraph, answer['answer_start'], answer['answer_start'] + len(answer['text']))
            spans.append([at + s, at + e])
        else:
            item = hotpot[question]
            paragraphs = dict(item['context'])
            for title, k in item['supporting_facts']:
                doc = f':\n{title}\n' + ''.join(paragraphs[title])
                at = context.index(doc) + len(f':\n{title}\n')
                offset = sum(len(s) for s in paragraphs[title][:k])
                spans.append([at + offset, at + offset + len(paragraphs[title][k].rstrip())])
        out.append({**row, 'evidence_spans': spans})
    return out


def doc_rows(smoke):
    rows, base = [], {'squad': 2_300_000, 'nq': 2_400_000}
    for source, limit in (('squad', 2 if smoke else 100), ('nq', 2 if smoke else 100)):
        picked = []
        for doc in map(json.loads, open(DOCS / f'{source}.jsonl')):
            for q in doc['questions'][:2]:
                if len(picked) < limit:
                    picked.append((doc, q))
        for i, (doc, q) in enumerate(picked):
            refs = sorted(set(q['answers']) | {doc['context'][a:b] for a, b in q['spans']})
            rows.append({'index': base[source] + i, 'task': f'qa_{source}', 'doc_id': doc['id'], 'qid': q['qid'],
                         'context': doc['context'], 'question': q['question'], 'answer_prefix': 'Answer:',
                         'answer': refs, 'max_new_tokens': 32, 'answer_spans': q['spans'],
                         'evidence_spans': [q['evidence']]})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    sets = {'ruler_test': ruler_rows(RULER / 'test.jsonl'), 'ruler_smoke': ruler_rows(RULER / 'smoke.jsonl'),
            'docs_test': doc_rows(False), 'docs_smoke': doc_rows(True)}
    sha = lambda b: hashlib.sha256(b).hexdigest()
    manifest = {}
    for name, rows in sets.items():
        path = args.out / f'e2_{name}.jsonl'
        path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
        manifest[name] = {'rows': len(rows), 'tasks': sorted({r['task'] for r in rows}),
                          'file_sha256': sha(path.read_bytes()),
                          'indices': [[r['index'], r.get('doc_id'), r.get('qid')] for r in rows]}
    (args.out / 'e2_manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
    print({k: (v['rows'], v['tasks']) for k, v in manifest.items()})


if __name__ == '__main__':
    main()
