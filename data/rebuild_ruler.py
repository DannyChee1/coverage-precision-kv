import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ROOT / 'data/manifests'
TOKENIZERS = {'meta-llama/Llama-3.1-8B-Instruct': '0e9e39f249a16976918f6564b8830bc894c89659',
              'Qwen/Qwen3-8B': 'b968826d9c46dd6066d109eabc6255188de91218'}
USED_650 = '24adceac8a0e6532936e8d721cd9e9084d2e4686'  # simonjegou/ruler, the 650 examples every set excludes
ALL_TASKS = 13
EQUIVALENT = {'e3da074df24e8369009918aa5cdbdd254dadcde4c63f7569d36afd6f2268caa8':
              '6906b710fe6793b450844c16d63429b87f5767ffd867aeeeef3c75ac5c2bf489'}
NIAH = ['niah_single_1', 'niah_single_2', 'niah_single_3', 'niah_multikey_1', 'niah_multikey_2', 'niah_multikey_3',
        'niah_multivalue', 'niah_multiquery']
SUBSETS = {'ruler-fresh18-4096': ['qa_1', 'qa_2', *NIAH], 'ruler-fresh21-4096': [*NIAH, 'qa_1', 'qa_2']}
QA = re.compile(r'chunk (\d+) of (\d+) over ([\d,]+)')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def manifest(name):
    return json.loads((MANIFESTS / name / 'manifest.json').read_text())


def options(name, m):
    chunk, chunks, samples = QA.search(m['qa']).groups()
    n_tasks = m['examples'] // m['kept_per_task']
    tasks = SUBSETS.get(name) or {2: ['qa_1', 'qa_2'], ALL_TASKS: None}.get(n_tasks, 'unknown')
    if tasks == 'unknown' or (tasks and len(tasks) != n_tasks):
        raise SystemExit(f'{name}: {n_tasks} tasks, not a known subset')
    opts = ['--seed', m['seed'], '--generate', m['generated_per_task'], '--per-task', m['kept_per_task'],
            '--qa-chunk', chunk, '--qa-chunks', chunks, '--qa-samples', samples.replace(',', ''),
            '--max-seq-length', m['max_seq_length'], '--tokenizer-name', m['tokenizer'],
            '--index-base', m.get('index_base', 100_000), '--out', ROOT / 'data' / name]
    if tasks:
        opts += ['--tasks', *tasks]
    if m.get('smoke_per_task'):
        opts += ['--smoke-per-task', m['smoke_per_task']]
    if m.get('calib_per_task'):
        opts += ['--calib', *(f'{t}:{n}' for t, n in m['calib_per_task'].items())]
    if m.get('also_excluded_sets'):
        opts += ['--exclude', *(ROOT / p for p in m['also_excluded_sets'])]
    return [str(o) for o in opts]


def order():
    names = sorted(p.name for p in MANIFESTS.iterdir() if p.name.startswith('ruler-'))
    deps = {n: {Path(p).parent.name for p in manifest(n).get('also_excluded_sets', [])} for n in names}
    done = []
    while len(done) < len(names):
        ready = [n for n in names if n not in done and deps[n] <= set(done)]
        if not ready:
            raise SystemExit('circular exclusions')
        done.append(ready[0])
    return done


def already_built(name, m):
    files = {'test_jsonl_sha256': 'test.jsonl', 'smoke_jsonl_sha256': 'smoke.jsonl', 'calib_jsonl_sha256': 'calib.jsonl'}
    out = ROOT / 'data' / name
    return all((out / f).exists() and sha256(out / f) == m[k] for k, f in files.items() if k in m)


def rebuild(name, ruler, work):
    from huggingface_hub import hf_hub_download, snapshot_download
    m = manifest(name)
    if already_built(name, m):
        print(f"{name}: identical (already built, {m['examples']} examples)")
        return True
    source = ruler / 'scripts/data/synthetic/json'
    for file, digest in m['source_sha256'].items():
        if sha256(source / file) not in (digest, EQUIVALENT.get(digest)):
            print(f'{name}: not built, RULER {file} differs from the manifest ({digest[:8]}...)')
            return False
    tokenizer, short = m['tokenizer'].split(' @')
    path = snapshot_download(tokenizer, revision=TOKENIZERS[tokenizer], allow_patterns=['*.json'])
    assert TOKENIZERS[tokenizer].startswith(short)
    used = hf_hub_download('simonjegou/ruler', '4096/test-00000-of-00001.parquet', repo_type='dataset',
                           revision=USED_650)
    cmd = [sys.executable, str(ROOT / 'data/make_ruler_fresh.py'), '--ruler', str(ruler), '--tokenizer', path,
           '--work', str(work / name), '--used-parquet', used, *options(name, m)]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if proc.returncode:
        raise SystemExit(f'{name}: make_ruler_fresh.py failed\n{proc.stderr[-2000:]}')
    built = json.loads((ROOT / 'data' / name / 'manifest.json').read_text())
    ok = all(built.get(k) == m[k] for k in m if k.endswith('_sha256') and k != 'source_sha256')
    print(f"{name}: {'identical' if ok else 'DIFFERS'} ({m['examples']} examples)")
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('set')
    parser.add_argument('--ruler', type=Path, required=True)
    parser.add_argument('--work', type=Path)
    args = parser.parse_args()
    names = order() if args.set == 'all' else [args.set]
    with tempfile.TemporaryDirectory() as tmp:
        results = [rebuild(n, args.ruler.resolve(), args.work or Path(tmp)) for n in names]
    sys.exit(0 if all(results) else 1)


if __name__ == '__main__':
    main()
