import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'kvrecipe'))
import mix

RULER_COMMIT = '3b8928fa01a2e25fa4cbaf364164f3622eb9487f'
SEED_FIX = ('-    random_seed = 42 + args.chunk_idx', '+    random_seed = args.random_seed + args.chunk_idx')
WONDERWORDS = '2.2.0'
TASKS = ('niah_single_1', 'niah_single_2', 'niah_single_3', 'niah_multikey_1', 'niah_multikey_2', 'niah_multikey_3',
         'niah_multivalue', 'niah_multiquery', 'vt', 'cwe', 'fwe', 'qa_1', 'qa_2')
QUESTION_PATTERNS = {
    'niah': re.compile(r'What (?:is|are all) the special magic'),
    'vt': re.compile(r'Question: Find all variables that are assigned the value'),
    'cwe': re.compile(r'Question: What are the 10 most common words in the above list\?'),
    'fwe': re.compile(r'Question: Do not provide any explanation\.'),
    'qa': re.compile(r'Answer the question based on the given documents\.'),
}
ANSWER_PATTERNS = {'niah': re.compile(r'The special magic'), 'vt': re.compile(r'Answer:'),
                   'cwe': re.compile(r'Answer:'), 'fwe': re.compile(r'Answer:'), 'qa': re.compile(r'Answer:')}
MAX_NEW_TOKENS = {'niah': 128, 'vt': 30, 'cwe': 120, 'fwe': 50, 'qa': 32}
QA_SAMPLES, QA_CHUNKS = 1000, 10
SHIM = '''import json


def write_manifest(output_path, target_manifest, ensure_ascii: bool = True):
    with open(output_path, "w", encoding="utf-8") as outfile:
        for tgt in target_manifest:
            json.dump(tgt, outfile, ensure_ascii=ensure_ascii)
            outfile.write("\\n")


def read_manifest(manifest):
    with open(manifest, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
'''


def split(text, task):
    family = task.split('_')[0]
    idx = list(QUESTION_PATTERNS[family].finditer(text))[-1].start()
    context, qa = text[:idx], text[idx:]
    idx = ANSWER_PATTERNS[family].search(qa).start()
    return context, qa[:idx], qa[idx:]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_ruler(ruler):
    head = subprocess.run(['git', '-C', str(ruler), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    diff = subprocess.run(['git', '-C', str(ruler), 'diff', 'HEAD', '--', 'scripts/data/prepare.py'],
                          capture_output=True, text=True).stdout
    changed = [line for line in diff.splitlines() if line[:1] in '+-' and line[:3] not in ('+++', '---')]
    if head != RULER_COMMIT or changed != list(SEED_FIX):
        raise SystemExit(f'RULER must be {RULER_COMMIT} plus only the seed fix; got {head} with {changed}')
    if importlib.metadata.version('wonderwords') != WONDERWORDS:
        raise SystemExit(f'wonderwords {WONDERWORDS} is required')


def generate(ruler, tokenizer, seed, work, per_task, tasks=TASKS, qa_chunk=5, qa_samples=QA_SAMPLES,
             qa_chunks=QA_CHUNKS, max_seq_length=4096):
    shim = work / 'nemo_shim' / 'nemo' / 'collections' / 'asr' / 'parts' / 'utils'
    shim.mkdir(parents=True, exist_ok=True)
    for parent in [shim, *shim.parents][:5]:
        (parent / '__init__.py').touch()
    (shim / 'manifest_utils.py').write_text(SHIM)
    env = dict(os.environ, PATH=f'{Path(sys.executable).parent}:{os.environ["PATH"]}',
               PYTHONPATH=str(work / 'nemo_shim'))
    for task in tasks:
        result = subprocess.run([sys.executable, 'prepare.py', '--save_dir', str(work / str(seed)),
                                 '--benchmark', 'synthetic', '--task', task, '--tokenizer_path', str(tokenizer),
                                 '--tokenizer_type', 'hf', '--max_seq_length', str(max_seq_length),
                                 '--model_template_type', 'base',
                                 *(['--num_samples', str(qa_samples), '--chunk_amount', str(qa_chunks),
                                    '--chunk_idx', str(qa_chunk)] if task.startswith('qa')
                                   else ['--num_samples', str(per_task)]),
                                 '--random_seed', str(seed)],
                                cwd=ruler / 'scripts' / 'data', env=env, capture_output=True, text=True)
        if 'Traceback' in result.stdout + result.stderr:
            raise SystemExit(f'{task}: RULER failed\n{result.stdout[-2000:]}{result.stderr[-2000:]}')
    return work / str(seed)


def used_rows(parquet, per_task=50):
    df = pd.read_parquet(parquet)
    return [dict(row) | {'index': int(index)} for _, group in df.groupby('task', sort=True)
            for index, row in group.iloc[:per_task].iterrows()]


def converted(path, task):
    for _, record in pd.read_json(path / task / 'validation.jsonl', lines=True).iterrows():
        context, question, answer_prefix = split(record['input'], task)
        yield {'context': context, 'question': question, 'answer_prefix': answer_prefix,
               'answer': list(record['outputs']), 'task': task, 'max_new_tokens': MAX_NEW_TOKENS[task.split('_')[0]]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ruler', type=Path, required=True, help='RULER checkout (see RULER_COMMIT), data downloaded')
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--work', type=Path, required=True, help='scratch directory for RULER output')
    parser.add_argument('--generate', type=int, default=60, help='examples generated per task')
    parser.add_argument('--per-task', type=int, default=50, help='examples kept per task')
    parser.add_argument('--out', type=Path, default=Path('data/ruler-fresh-4096'))
    parser.add_argument('--qa-chunk', type=int, default=5, help='qa questions 100k to 100k + 99 by default')
    parser.add_argument('--qa-samples', type=int, default=QA_SAMPLES, help='qa samples split into chunks')
    parser.add_argument('--qa-chunks', type=int, default=QA_CHUNKS, help='number of qa chunks')
    parser.add_argument('--exclude', nargs='*', default=[], help='earlier fresh sets (jsonl) to exclude as well')
    parser.add_argument('--smoke-per-task', type=int, default=0, help='extra examples per task for a timing smoke')
    parser.add_argument('--index-base', type=int, default=100_000)
    parser.add_argument('--calib', nargs='*', default=[],
                        help='TASK:N more fresh examples of TASK after the kept ones, written to calib.jsonl')
    parser.add_argument('--max-seq-length', type=int, default=4096)
    parser.add_argument('--tasks', nargs='+', default=list(TASKS), choices=TASKS, help='a subset, e.g. qa_1 qa_2')
    parser.add_argument('--tokenizer-name', default='meta-llama/Llama-3.1-8B-Instruct @0e9e39f')
    parser.add_argument('--used-parquet', type=Path,
                        help='local simonjegou/ruler 4096 test parquet for the used 650, in place of datasets')
    args = parser.parse_args()
    check_ruler(args.ruler)
    raw = generate(args.ruler, args.tokenizer, args.seed, args.work, args.generate, tasks=args.tasks,
                   qa_chunk=args.qa_chunk,
                   qa_samples=args.qa_samples, qa_chunks=args.qa_chunks, max_seq_length=args.max_seq_length)
    size = args.qa_samples // args.qa_chunks
    used = (used_rows(args.used_parquet) if args.used_parquet else mix.load_rows(50, 0)) + [
        json.loads(line) for p in args.exclude for line in open(p)]
    used_contexts = {r['context'] for r in used}
    used_questions = {r['question'] for r in used if r['task'] in ('qa_1', 'qa_2')}
    calib_n = {t: int(n) for t, n in (c.split(':') for c in args.calib)}
    rows, smoke, calib, excluded = [], [], [], {}
    for task in args.tasks:
        kept, need = [], args.per_task + args.smoke_per_task + calib_n.get(task, 0)
        for row in converted(raw, task):
            if row['context'] in used_contexts or row['question'] in used_questions:
                excluded[task] = excluded.get(task, 0) + 1
            elif len(kept) < need:
                kept.append(row)
        if len(kept) < need:
            raise SystemExit(f'{task}: only {len(kept)} fresh examples')
        rows.extend(kept[:args.per_task])
        smoke.extend(kept[args.per_task:args.per_task + args.smoke_per_task])
        calib.extend(kept[args.per_task + args.smoke_per_task:])
    for index, row in enumerate(rows + smoke + calib):
        row['index'] = args.index_base + index  # disjoint from the HF dataset's and earlier sets' indices
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / 'test.jsonl'
    path.write_text(''.join(json.dumps(r, sort_keys=True) + '\n' for r in rows))
    if smoke:
        (args.out / 'smoke.jsonl').write_text(''.join(json.dumps(r, sort_keys=True) + '\n' for r in smoke))
    if calib:
        (args.out / 'calib.jsonl').write_text(''.join(json.dumps(r, sort_keys=True) + '\n' for r in calib))
    source = args.ruler / 'scripts' / 'data' / 'synthetic' / 'json'
    manifest = {'ruler_commit': RULER_COMMIT, 'ruler_patch': 'ce84052 (prepare.py honours --random_seed)',
                'wonderwords': WONDERWORDS, 'seed': args.seed, 'generated_per_task': args.generate,
                'qa': f'questions {size * args.qa_chunk}-{size * args.qa_chunk + size - 1} (prepare.py chunk '
                      f'{args.qa_chunk} of {args.qa_chunks} over {args.qa_samples:,}; document seed = seed + '
                      f'{args.qa_chunk})',
                'kept_per_task': args.per_task, 'max_seq_length': args.max_seq_length, 'template': 'base',
                'tokenizer': args.tokenizer_name,
                'tokenizer_json_sha256': sha256(args.tokenizer / 'tokenizer.json'),
                'source_sha256': {name: sha256(source / name) for name in
                                  ('PaulGrahamEssays.json', 'squad.json', 'hotpotqa.json', 'english_words.json')},
                'excluded_as_overlap_with_used_650': excluded, 'examples': len(rows),
                'test_jsonl_sha256': sha256(path)}
    if args.exclude:
        manifest['also_excluded_sets'] = [str(p) for p in args.exclude]
    if args.index_base != 100_000:
        manifest['index_base'] = args.index_base
    if smoke:
        manifest.update(smoke_per_task=args.smoke_per_task, smoke_examples=len(smoke),
                        smoke_jsonl_sha256=sha256(args.out / 'smoke.jsonl'))
    if calib:
        manifest.update(calib_per_task=calib_n, calib_examples=len(calib),
                        calib_jsonl_sha256=sha256(args.out / 'calib.jsonl'))
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
    print(json.dumps(manifest, indent=1))


if __name__ == '__main__':
    main()
