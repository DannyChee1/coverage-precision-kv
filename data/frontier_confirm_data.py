import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'kvrecipe'))
from answer_blind_e2_data import ruler_rows
from frontier_data import DOCS, row
import predictive_model as pm

ROOT = Path(__file__).resolve().parents[1]
FRESH21 = ROOT / 'data/ruler-fresh21-4096'
QUALITY = {'test': ROOT / 'data/stagec/quality.jsonl', 'smoke': ROOT / 'data/stagec/controls_smoke.jsonl'}
LB_PLAN = ROOT / 'results/review/longbench_plan.json'
LONGBENCH = ROOT / 'data/longbench'
TOKENIZERS = {'llama': ('meta-llama/Llama-3.1-8B-Instruct', '0e9e39f249a16976918f6564b8830bc894c89659'),
              'qwen3': ('Qwen/Qwen3-8B', 'b968826d9c46dd6066d109eabc6255188de91218')}
BASE = {'squad': 2_820_000, 'smoke': 2_895_000}
NIAH_PER_TASK = 50


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ruler(kind):
    rows = ruler_rows(FRESH21 / f'{kind}.jsonl')
    keep = {'test': NIAH_PER_TASK, 'smoke': 1}[kind]
    out = [r for r in rows if r['task'].startswith('qa')]
    for task in sorted({r['task'] for r in rows if not r['task'].startswith('qa')}):
        out += [r for r in rows if r['task'] == task][:keep]
    return out


def longbench_rows(tok):
    from review_longbench import split_prompt
    plan = json.loads(LB_PLAN.read_text())['tasks']
    prompts = json.loads((LONGBENCH / 'dataset2prompt.json').read_text())
    out = []
    for task, p in plan.items():
        raw = (LONGBENCH / 'data' / f'{task}.jsonl').read_bytes()
        if hashlib.sha256(raw).hexdigest() != p['sha256']:
            raise SystemExit(f'{task}.jsonl does not match the plan hash')
        examples = {r['_id']: r for r in map(json.loads, raw.decode().splitlines())}
        for i in p['test_ids'] + p['smoke_ids']:
            out.append({'index': i, 'task': task, 'context': split_prompt(tok, prompts[task], examples[i])[0]})
    return out


def main():
    from transformers import AutoTokenizer
    tokens = pm.e1_tokens()
    floor = pm.swc.floor_tokens('1.5625')
    short = lambda d: min(tokens[m][d['id']] for m in pm.MODELS)
    squad = [d for d in map(json.loads, open(DOCS / 'squad.jsonl')) if short(d) >= floor]
    test = [row(BASE['squad'] + i, 'squad', d, q) for i, (d, q) in enumerate((d, q) for d in squad
                                                                              for q in d['questions'][7:9])]
    ordered = sorted([d for d in squad if len(d['questions']) >= 10], key=short)
    smoke = [row(BASE['smoke'] + i, 'squad', d, d['questions'][9]) for i, d in enumerate((ordered[0], ordered[-1]))]
    files = {'fc_conf': test + ruler('test'), 'fc_conf_smoke': smoke + ruler('smoke')}
    for model, (name, revision) in TOKENIZERS.items():
        files[f'fc_lb_{model}'] = longbench_rows(AutoTokenizer.from_pretrained(name, revision=revision))
    manifest = {'note': 'IDs and hashes only; the rows hold SQuAD, RULER and LongBench text and stay local.',
                'inputs': {'ruler_fresh21_test': sha(FRESH21 / 'test.jsonl'),
                           'ruler_fresh21_smoke': sha(FRESH21 / 'smoke.jsonl'),
                           'quality_test': sha(QUALITY['test']), 'quality_smoke': sha(QUALITY['smoke']),
                           'longbench_plan': sha(LB_PLAN)}}
    for name, rows in files.items():
        path = DOCS / f'{name}.jsonl'
        path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
        manifest[name] = {'sha256': sha(path), 'rows': len(rows),
                          'ids': [{'index': r['index'], 'task': r['task'], 'doc_id': r.get('doc_id'),
                                   'qid': r.get('qid')} for r in rows]}
    (DOCS / 'fc_manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
    print({k: (v['rows'], v['sha256'][:16]) for k, v in manifest.items() if k not in ('note', 'inputs')})


if __name__ == '__main__':
    main()
