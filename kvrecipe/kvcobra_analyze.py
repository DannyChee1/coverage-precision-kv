import argparse
import copy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
import controls_analyze as ca
from review4k_analyze import configured, holm

ALPHA, BUDGETS, BENCHES = 0.05, ('1.5625', '3.125'), ('ruler', 'quality')
NOMINAL = 'kvcobranom'
SECONDARY = (('kvcobra', 'stack4pc'), ('kvcobra', 'tca'), (NOMINAL, 'kvcobra'), (NOMINAL, 'k3v2pc'))


def row_arms(with_tca):
    splits = ('stack4pc', 'k3v2pc') + (('tca',) if with_tca else ()) + ('kvcobra', NOMINAL)
    return ['full', *[f'Z.{s}@{b}' for b in BUDGETS for s in splits]]


def budget_view(records):
    out = copy.deepcopy(records)
    for r in out:
        for key in ('bits', 'budget_bits', 'over_budget_layers'):
            for b in BUDGETS:
                r[key].pop(f'Z.{NOMINAL}@{b}', None)
    return out


def hashes_agree(records):
    return all(len({r['kvcobra'][f'Z.{a}@{b}']['table_hash'] for r in records}) == 1
               for a in ('kvcobra', NOMINAL) for b in BUDGETS)


def bits(records, arm):
    shares = [r['bits'][arm] / r['budget_bits'][arm] for r in records]
    meta = records[0]['kvcobra'][arm]
    return {'share_of_budget_min': min(shares), 'share_of_budget_max': max(shares), **meta}


def analyze(models, gate):
    with_tca = any('Z.tca@1.5625' in r['bits'] for recs in models.values() for r in recs)
    arms = row_arms(with_tca)
    per, draws, primary, secondary, storage = {}, {}, [], {}, {}
    for m, recs in models.items():
        per[m], draws[m] = {}, {}
        for bench in BENCHES:
            rs = [r for r in recs if r['benchmark'] == bench]
            if not rs:
                continue
            with configured(arms):
                out = ca.summary(budget_view(rs), bench)
            draws[m][bench] = out.pop('_acc_draws')
            out['kvcobra_tables_agree'] = hashes_agree(rs)
            out['valid_all'] = out['valid_all'] and out['kvcobra_tables_agree']
            out['kvcobra_layers_over'] = {a: max(r.get('kvcobra_layers_over', {}).get(a, 0) for r in rs)
                                          for a in rs[0].get('kvcobra_layers_over', {})}
            per[m][bench] = out
            acc, d = draws[m][bench]
            for b in BUDGETS:
                key = f'{m} {bench} @{b}'
                primary.append((key, ca.contrast(acc, d, [(f'Z.kvcobra@{b}', 1), (f'Z.k3v2pc@{b}', -1)], 2),
                                out['valid_all']))
                for x, y in SECONDARY:
                    if f'Z.{x}@{b}' in acc and f'Z.{y}@{b}' in acc:
                        secondary[f'{key} {x} - {y}'] = ca.contrast(acc, d, [(f'Z.{x}@{b}', 1), (f'Z.{y}@{b}', -1)],
                                                                    2)
                storage[f'{key}'] = {a: bits(rs, f'Z.{a}@{b}') for a in ('kvcobra', NOMINAL)}
    tests = holm([(k, t, v) for k, t, v in primary], ALPHA)
    for t in tests.values():  # holm() marks passes for Delta >= +margin; record both directions explicitly
        t['call'] = ('RECIPE BETTER' if t['holm_pass'] and t['valid'] and t['delta'] <= -ca.MARGIN else
                     'KV-COBRA BETTER' if t['holm_pass'] and t['valid'] and t['delta'] >= ca.MARGIN else
                     'NOT SEPARATED')
    counts = {c: sum(t['call'] == c for t in tests.values()) for c in ('RECIPE BETTER', 'KV-COBRA BETTER',
                                                                     'NOT SEPARATED')}
    for m in per:
        for bench in per[m]:
            per[m][bench].pop('per_task', None)
    return {'label': 'KV-COBRA' if gate.get('call') == 'PASS' else 'KV-COBRA-inspired', 'gate': gate.get('call'),
            'primary': {'tests': tests, 'counts': counts, 'cells': len(tests),
                        'rule': {'alpha': ALPHA, 'margin': ca.MARGIN, 'sided': 2, 'holm_family': len(tests)}},
            'secondary': secondary, 'storage': storage, 'with_tca': with_tca, 'per_model': per,
            'valid_all': all(o['valid_all'] for p in per.values() for o in p.values()),
            'bootstrap': {'B': ca.B, 'seed': ca.SEED}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', nargs='+', action='append', required=True, metavar=('NAME', 'FILE'))
    parser.add_argument('--gate', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = analyze({name: ca.load(files) for name, *files in args.model}, json.loads(args.gate.read_text()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps(out['primary']['counts']), out['label'], 'valid' if out['valid_all'] else 'NOT VALID')


if __name__ == '__main__':
    main()
