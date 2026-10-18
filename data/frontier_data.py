import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'kvrecipe'))
from answer_blind_e2_data import ruler_rows
import predictive_model as pm

DOCS = Path(__file__).resolve().parents[1] / 'data/answerblind'
FRESH20 = Path(__file__).resolve().parents[1] / 'data/ruler-fresh20-4096'
BASE = {'squad': 2_800_000, 'nq': 2_810_000, 'smoke': 2_890_000}


def row(index, source, doc, q):
    refs = sorted(set(q['answers']) | {doc['context'][a:b] for a, b in q['spans']})
    return {'index': index, 'task': f'qa_{source}', 'doc_id': doc['id'], 'qid': q['qid'], 'context': doc['context'],
            'question': q['question'], 'answer_prefix': 'Answer:', 'answer': refs, 'max_new_tokens': 32,
            'answer_spans': q['spans'], 'evidence_spans': [q['evidence']]}


def main():
    tokens = pm.e1_tokens()
    floor = pm.swc.floor_tokens('1.5625')
    short = lambda d: min(tokens[m][d['id']] for m in pm.MODELS)
    squad = [d for d in map(json.loads, open(DOCS / 'squad.jsonl')) if short(d) >= floor]
    nq = [d for d in map(json.loads, open(DOCS / 'nq.jsonl')) if short(d) >= floor]
    test = [row(BASE['squad'] + i, 'squad', d, q) for i, (d, q) in enumerate((d, q) for d in squad
                                                                              for q in d['questions'][4:6])]
    test += [row(BASE['nq'] + i, 'nq', d, d['questions'][0]) for i, d in enumerate(nq)]
    smoke = []
    for source, docs, pick in (('squad', squad, lambda d: d['questions'][6]), ('nq', nq, lambda d: d['questions'][0])):
        ordered = sorted(docs, key=short)
        smoke += [row(BASE['smoke'] + len(smoke), source, d, pick(d)) for d in (ordered[0], ordered[-1])]
    ruler = {k: ruler_rows(FRESH20 / f'{k}.jsonl') for k in ('test', 'smoke')}
    manifest = {'note': 'IDs, prompt lengths and hashes only; the rows hold SQuAD, NQ and RULER text and stay local.'}
    for name, rows in (('fr_docs', test), ('fr_docs_smoke', smoke), ('fr_ruler', ruler['test']),
                       ('fr_ruler_smoke', ruler['smoke']), ('fr_test', test + ruler['test']),
                       ('fr_smoke', smoke + ruler['smoke'])):
        path = DOCS / f'{name}.jsonl'
        path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
        manifest[name] = {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'rows': len(rows),
                          'ids': [{'index': r['index'], 'task': r['task'], 'doc_id': r.get('doc_id'),
                                   'qid': r.get('qid')} for r in rows]}
    (DOCS / 'fr_manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
    print({k: (v['rows'], v['sha256'][:16]) for k, v in manifest.items() if k != 'note'})


if __name__ == '__main__':
    main()
