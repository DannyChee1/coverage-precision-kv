"""python run.py list [KEY ...]                       analyses and driver calls in studies.toml
python run.py verify [--tier N] [KEY ...]           rebuild results/ into a scratch folder, compare bytes
python run.py analyze [KEY ...]                     rebuild them in place
python run.py drive KEY [--one] [--match GLOB] [--models DIR] [--dry]   GPU driver calls, writing raw/
python run.py export                                raw/ -> records/ (text replaced by scores and hashes)
python run.py data                                  rebuild data/ from the sources (README.md, Data)
"""
import argparse
import filecmp
import fnmatch
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
import tomllib

ROOT = Path(__file__).resolve().parent
RAW = Path(os.environ.get('RAW_DIR', ROOT / 'raw'))
BRACE = re.compile(r'\{([^{}]*,[^{}]*)\}')
LEADERBOARD = ('leaderboard_llama_0.jsonl', 'leaderboard_llama_1.jsonl')


def registry(kind, keys=()):
    entries = tomllib.loads((ROOT / 'studies.toml').read_text())[kind]
    return [e for e in entries if not keys or e['key'] in keys]


def product(token):
    m = BRACE.search(token)
    if not m:
        return [token]
    return [x for part in m.group(1).split(',') for x in product(token[:m.start()] + part + token[m.end():])]


def pairwise(text):
    groups = [g.split(',') for g in BRACE.findall(text)]
    if not groups:
        return [text]
    out = []
    for i in range(len(groups[0])):
        values = iter(g[i] for g in groups)
        out.append(BRACE.sub(lambda m: next(values), text))
    return out


def analysis_command(entry, out):
    subst = {'@K': str(ROOT / 'records'), '@RAW': str(RAW), '@R': str(ROOT / 'results'), '@OUT': str(out)}
    tokens = [t for raw in shlex.split(entry['args']) for t in product(raw)]
    tokens = [re.sub(r'@K|@RAW|@R|@OUT', lambda m: subst[m.group(0)], t) for t in tokens]
    return [sys.executable, str(ROOT / 'kvrecipe' / tokens[0]), *tokens[1:]]


def analyses(action, keys, tier):
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        for e in registry('analysis', keys):
            if e['tier'] > tier:
                continue
            out = Path(tmp) if action == 'verify' else ROOT if e['produces'] == ['figures'] else ROOT / 'results'
            for p in e['produces']:
                (out / p).parent.mkdir(parents=True, exist_ok=True)
            env = {'PYTHONDONTWRITEBYTECODE': '1', 'HF_HUB_OFFLINE': '1', 'MPLCONFIGDIR': str(Path(tmp) / '.mpl'),
                   **{k: v for k, v in os.environ.items() if k in ('PATH', 'HOME', 'HF_HOME')}}
            proc = subprocess.run(analysis_command(e, out), cwd=ROOT, capture_output=True, text=True, env=env)
            if proc.returncode:
                failures += 1
                print(f"FAIL {e['key']}: {e['args'].split()[0]}\n{proc.stderr[-1500:]}")
                continue
            for p in e['produces']:
                if action == 'analyze':
                    print(f"done {e['key']}: {p}")
                    continue
                if p == 'figures':
                    print(f"BUILT     {e['key']:18s} {p} (not in the repo, nothing to compare)")
                    continue
                ok = filecmp.cmp(Path(tmp) / p, ROOT / 'results' / p, shallow=False)
                failures += not ok
                print(f"{'IDENTICAL' if ok else 'DIFFERS  '} {e['key']:18s} {p}")
    if action == 'verify':
        print('all identical' if not failures else f'{failures} problem(s)')
    return failures


def calls(entry, one):
    if one:
        return [(entry['one_example'], pairwise(entry['out'])[0])] if entry['one_example'] else []
    args, outs = pairwise(entry['args']), pairwise(entry['out'])
    return list(zip(args * len(outs) if len(args) == 1 else args, outs))


def model_path(model, models):
    name, revision = model.split('@')
    if models:
        return str(Path(models) / name.replace('/', '--'))
    from huggingface_hub import snapshot_download
    return snapshot_download(name, revision=revision)


def drive(keys, one, match, models, dry):
    for entry in registry('driver', keys):
        todo = [c for c in calls(entry, one) if not match or fnmatch.fnmatch(c[1], match)]
        if not todo:
            continue
        path = entry['model'] if dry else model_path(entry['model'], models)
        env = {**os.environ, 'TOKENIZERS_PARALLELISM': 'false', **entry.get('env', {})}
        for args, record in todo:
            out = RAW / ('one' if one else '') / record
            cmd = [sys.executable, str(ROOT / 'kvrecipe' / entry['script']), '--model-path', path, '--out', str(out),
                   *shlex.split(args)]
            print('+', shlex.join(cmd[1:]), flush=True)
            if dry:
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            if subprocess.run(cmd, cwd=ROOT, env=env).returncode:
                raise SystemExit(f"{entry['script']} failed")


def export():
    base = [sys.executable, str(ROOT / 'kvrecipe/records.py'), str(RAW), str(ROOT / 'records')]
    names = sorted(p.name for p in RAW.glob('*.jsonl') if p.name not in LEADERBOARD)
    if names:
        subprocess.run([*base, *names], cwd=ROOT, check=True)
    lb = [n for n in LEADERBOARD if (RAW / n).exists()]
    if lb:
        subprocess.run([*base, *lb, '--leaderboard', str(ROOT / 'external/kvpress-leaderboard')], cwd=ROOT, check=True)


SOURCES = ('external/RULER/scripts/data/prepare.py', 'external/RULER/scripts/data/synthetic/json/squad.json',
           'external/RULER/scripts/data/synthetic/json/PaulGrahamEssays.json',
           'external/natural_questions/validation-0000.parquet', 'data/longbench/data/gov_report.jsonl',
           'data/longbench/dataset2prompt.json', 'data/longbench/dataset2maxlen.json')
BUILT = {'data/stagec/manifest.json': 'stagec/manifest.json', 'data/tca/manifest.json': 'tca/manifest.json',
         **{f'data/answerblind/{n}.json': f'answerblind/{n}.json' for n in
            ('manifest', 'e2_manifest', 'fr_manifest', 'fc_manifest', 'vl_manifest')}}


def hashes(o, path=''):
    if isinstance(o, dict):
        return {k: v for key, val in o.items() for k, v in hashes(val, f'{path}.{key}').items()}
    if isinstance(o, str) and re.fullmatch(r'[0-9a-f]{64}', o):
        return {path: o}
    return {}


def data():
    missing = [s for s in SOURCES if not (ROOT / s).exists()]
    if missing:
        raise SystemExit('missing sources (README.md, Data): ' + ', '.join(missing))
    py = lambda script, *args: subprocess.run([sys.executable, str(ROOT / 'data' / script), *args], cwd=ROOT, check=True)
    ruler = 'external/RULER'
    py('fetch_hotpotqa.py', '--out', f'{ruler}/scripts/data/synthetic/json/hotpotqa.json')
    py('rebuild_ruler.py', 'all', '--ruler', ruler)
    py('make_stagec_data.py', '--out', 'data/stagec')
    smoke = (ROOT / 'data/stagec/smoke.jsonl').read_text().splitlines(keepends=True)
    (ROOT / 'data/stagec/controls_smoke.jsonl').write_text(''.join(smoke[:3]))
    py('make_tca_data.py', '--out', 'data/tca')
    tca = ROOT / 'data/tca'
    (tca / 'quality_train_AB.jsonl').write_bytes((tca / 'quality_train_A.jsonl').read_bytes()
                                                 + (tca / 'quality_train_B.jsonl').read_bytes())
    py('answer_blind_data.py', '--nq', 'external/natural_questions/validation-0000.parquet', '--out', 'data/answerblind')
    py('answer_blind_e2_data.py', '--out', 'data/answerblind')
    for script in ('frontier_data.py', 'frontier_confirm_data.py', 'visible_lift_data.py'):
        py(script)
    failures = 0
    for built, ref in BUILT.items():
        mine = hashes(json.loads((ROOT / built).read_text()))
        theirs = hashes(json.loads((ROOT / 'data/manifests' / ref).read_text()))
        ok = mine == theirs
        failures += not ok
        print(f"{'identical' if ok else 'DIFFERS  '} {built}")
    sys.exit(1 if failures else 0)


def main():
    parser = argparse.ArgumentParser(usage=__doc__)
    parser.add_argument('action', choices=['list', 'verify', 'analyze', 'drive', 'export', 'data'])
    parser.add_argument('keys', nargs='*')
    parser.add_argument('--tier', type=int)
    parser.add_argument('--one', action='store_true')
    parser.add_argument('--match')
    parser.add_argument('--models')
    parser.add_argument('--dry', action='store_true')
    args = parser.parse_intermixed_args()
    if args.action == 'list':
        for e in registry('analysis', args.keys):
            print(f"{e['key']:18s} tier {e['tier']}  {e['paper']}")
        for e in registry('driver', args.keys):
            for _, record in calls(e, False):
                print(f"{e['key']:18s} {e['script']:24s} -> raw/{record}")
    elif args.action in ('verify', 'analyze'):
        tier = args.tier or (1 if args.action == 'verify' else 3)
        sys.exit(1 if analyses(args.action, args.keys, tier) else 0)
    elif args.action == 'drive':
        if not args.keys:
            raise SystemExit('name the keys to run (python run.py list)')
        drive(args.keys, args.one, args.match, args.models, args.dry)
    elif args.action == 'export':
        export()
    else:
        data()


if __name__ == '__main__':
    main()
