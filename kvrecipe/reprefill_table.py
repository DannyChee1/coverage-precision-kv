import argparse
import json
from pathlib import Path

import numpy as np

KV_BYTES = {'llama': 131072, 'qwen3': 147456}  # per token: layers x 8 KV heads x 128 x 2 (K and V) x 2 bytes
LENGTHS = (4096, 8192, 16384, 32768)
BUDGETS = {'1.5625': 0.015625, '3.125': 0.03125}
BANDWIDTH_GBPS = {'pcie_host': 25.0, 'nvme': 6.0, '10gbe': 1.25, '100gbe': 12.5}  # nominal, named in the report
TOKEN_ID_BYTES = 4
GPU_USD_PER_HOUR = 2.25  # L40S hourly rate, the rate every cost here uses
STORAGE_USD_PER_GB_MONTH = 0.023  # object storage, standard tier list price
DEQUANT_LENGTH = 16384


def fit(points):
    n = np.array([p[0] for p in points], float) / 1000
    t = np.array([p[1] for p in points], float)
    (a, b), *_ = np.linalg.lstsq(np.stack([n, n * n], 1), t, rcond=None)
    return float(a), float(b)


def seconds(kind, model, n, fits, bench):
    measured = (bench or {}).get(model, {}).get(f'{kind}_seconds', {}).get(str(n))
    if measured:
        return float(np.median(measured)), 'measured'
    a, b = fits[model][kind]
    k = n / 1000
    return a * k + b * k * k, 'fit'


def table(records, accuracy, bench=None):
    fits = {m: {kind: fit([(r['n_tokens'], r[f'{kind}_seconds']) for r in rs]) for kind in ('prefill', 'score')}
            for m, rs in records.items()}
    rows = []
    for m in records:
        bw = dict(BANDWIDTH_GBPS)
        if (bench or {}).get(m, {}).get('h2d_gbps'):
            bw['pcie_host_measured'] = bench[m]['h2d_gbps']
        for n in LENGTHS:
            prefill, prefill_src = seconds('prefill', m, n, fits, bench)
            score, score_src = seconds('score', m, n, fits, bench)
            full = n * KV_BYTES[m]
            for budget, share in BUDGETS.items():
                stored = share * full
                dq = (bench or {}).get(m, {}).get('dequant_seconds', {}).get(budget)
                dequant = dq * n / DEQUANT_LENGTH if dq is not None else None
                prefill_usd = prefill * GPU_USD_PER_HOUR / 3600
                acc = accuracy['per_model'][m]
                rows.append({
                    'model': m, 'tokens': n, 'budget': budget,
                    'bytes': {'token_ids': n * TOKEN_ID_BYTES, 'full': full, 'recipe': stored},
                    'ttft_seconds': {
                        're_prefill': prefill,
                        'load_full': {k: full / (v * 1e9) for k, v in bw.items()},
                        'load_recipe': {k: stored / (v * 1e9) + (dequant or 0.0) for k, v in bw.items()}},
                    'dequant_included': dequant is not None, 'compress_seconds': score,
                    'sources': {'prefill': prefill_src, 'score': score_src},
                    'break_even_reuses_per_month': {
                        'recipe': stored / 1e9 * STORAGE_USD_PER_GB_MONTH / prefill_usd,
                        'full': full / 1e9 * STORAGE_USD_PER_GB_MONTH / prefill_usd},
                    'accuracy': {b: {'full': acc[b]['accuracy']['full'],
                                     'recipe': acc[b]['accuracy'][f'Z.k3v2pc@{budget}']} for b in acc}})
    flips = {}
    for r in rows:
        for k, t in r['ttft_seconds']['load_recipe'].items():
            if t < r['ttft_seconds']['re_prefill']:
                key = f"{r['model']} @{r['budget']} {k}"
                flips[key] = min(flips.get(key, r['tokens']), r['tokens'])
    return {'rows': rows, 'recipe_faster_from_tokens': flips, 'fits': fits,
            'constants': {'kv_bytes_per_token': KV_BYTES, 'bandwidth_gbps': BANDWIDTH_GBPS,
                          'gpu_usd_per_hour': GPU_USD_PER_HOUR, 'storage_usd_per_gb_month': STORAGE_USD_PER_GB_MONTH,
                          'token_id_bytes': TOKEN_ID_BYTES}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--records', nargs='+', action='append', required=True, metavar=('NAME', 'FILE'))
    parser.add_argument('--accuracy', type=Path, required=True)
    parser.add_argument('--bench', nargs=2, action='append', default=[], metavar=('NAME', 'FILE'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    records = {name: [json.loads(line) for f in files for line in open(f)] for name, *files in args.records}
    bench = {name: json.loads(Path(f).read_text()) for name, f in args.bench}
    out = table(records, json.loads(args.accuracy.read_text()), bench)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps(out['recipe_faster_from_tokens'], indent=1))


if __name__ == '__main__':
    main()
