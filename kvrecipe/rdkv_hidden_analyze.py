import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import controls_analyze as ca
import rdkv as rd
import rdkv_hidden as rh
from rdkv_fidelity_analyze import PAPER_EPS

ALPHA = 0.05
BUDGETS = ('1.5625', '3.125')
BENCHES = ('ruler', 'quality')
LABELS = {'PASS': 'RDKV adapted to the question-hidden setting', 'FAIL': 'RDKV-inspired allocator'}


def rdkv_summary(records):
    out = {}
    for b in BUDGETS:
        arm = f'Z.rdkv@{b}'
        v = np.sum([r['rdkv'][arm]['v'] for r in records], axis=0)
        k = np.sum([r['rdkv'][arm]['k'] for r in records], axis=0)
        slots = [sum(r['rdkv'][arm]['v']) for r in records]
        out[arm] = {'value_bits_share': dict(zip(map(str, rd.BITS), (v / v.sum()).tolist())),
                    'key_bits_share': dict(zip(map(str, rd.BITS), (k / k.sum()).tolist())),
                    'value_tokens_evicted': float(v[0] / v.sum()), 'key_channels_dropped': float(k[0] / k.sum()),
                    'kept_share': {a: float(np.mean([r['kept'][a] / n for r, n in zip(records, slots)]))
                                   for a in (f'Z.stack4pc@{b}', f'Z.k3v2pc@{b}', arm)}}
    return out


def holm(cells, alpha):
    cells = sorted(cells, key=lambda c: c[1]['p'])
    still, tests = True, {}
    for rank, (key, t, valid) in enumerate(cells):
        threshold = alpha / (len(cells) - rank)
        passed = still and t['p'] <= threshold
        still = passed
        t.update(holm_threshold=threshold, holm_pass=bool(passed), valid=bool(valid),
                 passes=bool(passed and t['delta'] >= ca.MARGIN and valid))
        tests[key] = t
    return tests


def analyze(models, gate, calibs=None, ruler_expected=650):
    rh.configure()
    ca.EXPECTED['ruler'] = (ruler_expected, ruler_expected)
    per, cells, secondary, vs_stack, valid_all = {}, [], {}, {}, True
    for m, recs in models.items():
        per[m] = {}
        for bench in BENCHES:
            rs = [r for r in recs if r['benchmark'] == bench]
            if not rs:
                valid_all = False
                continue
            out = ca.summary(rs, bench)
            acc, draws = out.pop('_acc_draws')
            out['rdkv'] = rdkv_summary(rs)
            per[m][bench] = out
            valid_all &= out['valid_all']
            for b in BUDGETS:
                key = f'{m} {bench} @{b}'
                cells.append((key, ca.contrast(acc, draws, [(f'Z.rdkv@{b}', 1), (f'Z.k3v2pc@{b}', -1)], 1),
                              out['valid_all']))
                secondary[key] = ca.contrast(acc, draws, [(f'Z.k3v2pc@{b}', 1), (f'Z.rdkv@{b}', -1)], 2)
                for arm in ('k3v2pc', 'rdkv'):
                    vs_stack[f'{m} {bench} {arm}-stack4pc @{b}'] = ca.contrast(
                        acc, draws, [(f'Z.{arm}@{b}', 1), (f'Z.stack4pc@{b}', -1)], 2)
    tests = holm(cells, ALPHA)
    better = [k for k, t in tests.items() if t['passes']]
    holds = valid_all and len(secondary) == 8 and all(s['ci95'][0] > -ca.MARGIN for s in secondary.values())
    call = ('RDKV-HIDDEN BETTER' if better else 'UNIFORM SPLIT HOLDS' if holds else 'NOT SEPARATED')
    eps = {}
    for name, path in (calibs or {}).items():
        t = json.loads(Path(path).read_text())
        ev, ek = np.array(t['eps_v']), np.array(t['eps_k'])
        eps[name] = {'v': ev.mean((0, 1))[1:4].tolist(), 'k': ek.mean((0, 1))[1:4].tolist(), 'contexts': t['contexts']}
    return {'call': call, 'better_cells': better, 'label': LABELS[gate['call']], 'gate': gate['call'],
            'tests': tests, 'secondary_k3v2pc_minus_rdkv': secondary, 'secondary_vs_stack4pc': vs_stack,
            'per_model': per, 'eps_layer_mean': eps, 'paper_eps_llama': PAPER_EPS,
            'rule': {'alpha': ALPHA, 'margin': ca.MARGIN, 'tests': len(cells), 'sided': 1,
                     'bootstrap': {'B': ca.B, 'seed': ca.SEED}, 'ruler_expected': ruler_expected}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', nargs='+', action='append', required=True, metavar=('NAME', 'FILE'))
    parser.add_argument('--gate', type=Path, required=True, help='the fidelity analysis output (RDKV-FIDELITY.md)')
    parser.add_argument('--calib', nargs=2, action='append', default=[], metavar=('NAME', 'FILE'))
    parser.add_argument('--ruler-expected', type=int, default=650)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = analyze({name: ca.load(files) for name, *files in args.model}, json.loads(args.gate.read_text()),
                  dict(args.calib), args.ruler_expected)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps({k: out[k] for k in ('call', 'better_cells', 'label', 'gate')}, indent=1))


if __name__ == '__main__':
    main()
