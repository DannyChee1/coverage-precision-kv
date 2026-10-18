import argparse
import json
from pathlib import Path
import random
import sys
import urllib.request

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'kvrecipe'))
from make_stagec_data import QUALITY_COMMIT, RAW, quality, sha256, write

TRAIN = 'QuALITY.v1.0.1.htmlstripped.train'
URL = f'{RAW}/nyu-mll/quality/{QUALITY_COMMIT}/data/v1.0.1/{TRAIN}'
SEED, SPLITS = 20262102, (('A', 20), ('B', 10), ('smoke', 2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=Path, default=Path('data/stagec/src'))
    parser.add_argument('--dev', type=Path, default=Path('data/stagec/quality.jsonl'))
    parser.add_argument('--out', type=Path, default=Path('data/tca'))
    args = parser.parse_args()
    args.src.mkdir(parents=True, exist_ok=True)
    src = args.src / TRAIN
    if not src.exists():
        urllib.request.urlretrieve(URL, src)
    dev = {json.loads(line)['doc_id'] for line in open(args.dev)}
    docs = sorted((d for d in quality(src) if d['doc_id'] not in dev), key=lambda d: d['doc_id'])
    order = random.Random(SEED).sample(range(len(docs)), sum(n for _, n in SPLITS))
    args.out.mkdir(parents=True, exist_ok=True)
    manifest, start = {'source': URL, 'source_sha256': sha256(src), 'seed': SEED, 'train_articles': len(docs),
                       'dropped_as_in_dev': len(quality(src)) - len(docs), 'splits': {}}, 0
    for name, n in SPLITS:
        picked = [docs[i] for i in order[start:start + n]]
        start += n
        path = args.out / f'quality_train_{name}.jsonl'
        manifest['splits'][name] = {'articles': len(picked), 'questions': sum(len(d['questions']) for d in picked),
                                    'doc_ids': [d['doc_id'] for d in picked], 'jsonl_sha256': write(path, picked)}
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
    print(json.dumps({k: v for k, v in manifest.items() if k != 'splits'}, indent=1),
          {k: (v['articles'], v['questions']) for k, v in manifest['splits'].items()})


if __name__ == '__main__':
    main()
