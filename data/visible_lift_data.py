import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'kvrecipe'))
from frontier_data import DOCS, row
import predictive_model as pm

FRESH14 = Path(__file__).resolve().parents[1] / 'data/ruler-fresh14-4096'
BASE = {'squad': 2_840_000, 'nq': 2_850_000, 'smoke': 2_897_000}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ruler(kind, per_task=None):
    rows = [json.loads(line) for line in open(FRESH14 / f'{kind}.jsonl')]
    out = []
    for task in ('qa_1', 'qa_2'):
        rs = [r for r in rows if r['task'] == task and any(a.lower() in r['context'].lower() for a in r['answer'])]
        out += rs[:per_task] if per_task else rs
    return out


def main():
    tokens = pm.e1_tokens()
    floor = pm.swc.floor_tokens('1.5625')
    short = lambda d: min(tokens[m][d['id']] for m in pm.MODELS)
    squad = [d for d in map(json.loads, open(DOCS / 'squad.jsonl')) if short(d) >= floor]
    nq = [json.loads(line) for line in open(DOCS / 'nq.jsonl')]
    test = ruler('test')
    test += [row(BASE['squad'] + i, 'squad', d, q) for i, (d, q) in enumerate((d, q) for d in squad
                                                                               for q in d['questions'][:2])]
    test += [row(BASE['nq'] + i, 'nq', d, d['questions'][0]) for i, d in enumerate(nq[:100])]
    ordered = sorted(squad, key=short)
    smoke = ruler('smoke', 2)
    smoke += [row(BASE['smoke'] + i, 'squad', d, d['questions'][2]) for i, d in enumerate((ordered[0], ordered[-1]))]
    smoke += [row(BASE['smoke'] + 2 + i, 'nq', d, d['questions'][0]) for i, d in enumerate(nq[100:102])]
    manifest = {'note': 'IDs and hashes only; the rows hold RULER, SQuAD and NQ text and stay local.',
                'inputs': {'ruler_fresh14_test': sha(FRESH14 / 'test.jsonl'),
                           'ruler_fresh14_smoke': sha(FRESH14 / 'smoke.jsonl'),
                           'squad': sha(DOCS / 'squad.jsonl'), 'nq': sha(DOCS / 'nq.jsonl')}}
    for name, rows in (('vl_rows', test), ('vl_smoke', smoke)):
        path = DOCS / f'{name}.jsonl'
        path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
        manifest[name] = {'sha256': sha(path), 'rows': len(rows),
                          'ids': [{'index': r['index'], 'task': r['task'], 'doc_id': r.get('doc_id'),
                                   'qid': r.get('qid')} for r in rows]}
    (DOCS / 'vl_manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
    print({k: (v['rows'], v['sha256'][:16]) for k, v in manifest.items() if k not in ('note', 'inputs')})


if __name__ == '__main__':
    main()
