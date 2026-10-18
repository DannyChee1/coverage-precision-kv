import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).parent))
import controls as kc
from mix import layer_budget_bits
import mix_kvcobra as kvc
import review4k as r4

BASE = ('stack4pc', 'k3v2pc')
CHECK = {'kvcobra1': 1, 'kvcobra2': 2, 'kvcobra4': 4}  # bits per dimension per side
CHECK_BUDGET = '1.5625'  # a label only: the check rows are not held to a budget
MAIN = ('kvcobra', 'kvcobranom')
COBRA = MAIN + tuple(CHECK)
STATE = {'model': None, 'arch': None, 'comp': {}, 'meta': {}, 'calibration': None, 'layers_over': {}, 'check': False}
R4_EXAMPLE, R4_DOCUMENT, R4_BUILD = r4.run_example, r4.run_document, r4.build_rows


def arms(benchmark, budget=None):
    if STATE['check']:
        return [f'Z.{s}@{CHECK_BUDGET}' for s in CHECK] if budget in (None, CHECK_BUDGET) else []
    out = []
    for b in [budget] if budget else list(kc.BUDGETS):
        out += [f'Z.{s}@{b}' for s in BASE + (('tca',) if r4.STATE['tca'] else ()) + MAIN]
    return out


def calibrate(pipe):
    if STATE['model'] is not None:
        return
    t0 = time.time()
    texts = kvc.calibration_texts()
    calib = kvc.collect(pipe.model, pipe.tokenizer, texts)
    arch = kvc.arch_of(pipe.model)
    rows = []
    if STATE['check']:
        for name, bpd in CHECK.items():
            n = bpd * arch.head_dim
            rows.append((f'Z.{name}@{CHECK_BUDGET}', (n, n), n, kvc.compressor(arch, calib, n)))
    else:
        for budget, fraction in kc.BUDGETS.items():
            sides, comp = kvc.fit(arch, calib, fraction)
            nominal = int(round(fraction * 16 * arch.head_dim))
            rows += [(f'Z.kvcobra@{budget}', sides, nominal, comp),
                     (f'Z.kvcobranom@{budget}', (nominal, nominal), nominal, kvc.compressor(arch, calib, nominal))]
    for arm, b, nominal, c in rows:
        per_token, meta = kvc.stored_bits(c)
        STATE['comp'][arm] = c
        STATE['meta'][arm] = {'budget_k': b[0], 'budget_v': b[1], 'nominal_per_side': nominal,
                              'bits_per_token': per_token, 'meta_bits': meta, 'table_hash': kvc.table_hash(c)}
    STATE.update(model=pipe.model, arch=arch, calibration={'texts': len(texts), 'tokens': calib.n_cal_tokens,
                                                           'seconds': time.time() - t0})
    del calib


def cobra_rows(cache, row_arms, arm_cache, out):
    rows = [(row, a) for row, a in enumerate(row_arms) if a != 'full' and r4.split_of(a)[0] in COBRA]
    if not rows:
        return
    n_layers = len(cache.layers)
    n_heads, n_tokens, head_dim = cache.layers[0].keys.shape[1:]
    device = cache.layers[0].keys.device
    cos, sin = kvc.rope_angles(STATE['model'], n_tokens, device)
    for row, arm in rows:
        comp = STATE['comp'][arm]
        limit = layer_budget_bits(n_heads, n_tokens, head_dim, kc.BUDGETS[r4.split_of(arm)[1]])
        layer_bits = []
        for li in range(n_layers):
            k, v = kvc.compress_layer(cache.layers[li].keys[0], cache.layers[li].values[0], comp, li, cos, sin)
            arm_cache.layers[li].keys[row], arm_cache.layers[li].values[row] = k, v
            layer_bits.append(kvc.layer_bits(comp, li, n_tokens))
        out['bits'][arm], out['budget_bits'][arm] = float(sum(layer_bits)), n_layers * limit
        out['over_budget_layers'][arm] = 0
        out['kept'][arm] = n_layers * n_heads * n_tokens
        STATE['layers_over'][arm] = max(STATE['layers_over'].get(arm, 0), sum(b > limit for b in layer_bits))


def build_rows(cache, scores, row_arms):
    base = ['full' if a != 'full' and r4.split_of(a)[0] in COBRA else a for a in row_arms]
    arm_cache, masks, out = R4_BUILD(cache, scores, base)
    cobra_rows(cache, row_arms, arm_cache, out)
    return arm_cache, masks, out


def run_example(pipe, row, capture_kvzip, capture_scorers):
    calibrate(pipe)
    STATE['layers_over'] = {}
    out = R4_EXAMPLE(pipe, row, capture_kvzip, capture_scorers)
    out.update(kvcobra=STATE['meta'], kvcobra_calibration=STATE['calibration'],
               kvcobra_layers_over=STATE['layers_over'])
    return out


def run_document(pipe, doc, capture_kvzip, capture_scorers, cap=kc.QUALITY_CAP):
    calibrate(pipe)
    STATE['layers_over'] = {}
    out = R4_DOCUMENT(pipe, doc, capture_kvzip, capture_scorers, cap)
    out.update(kvcobra=STATE['meta'], kvcobra_calibration=STATE['calibration'],
               kvcobra_layers_over=STATE['layers_over'])
    return out


def configure(tca=None, check=False):
    r4.configure(False, 'mse', tca)
    STATE['check'] = check
    if check:  # one decode batch per QuALITY article: the check rows carry one budget label
        kc.BUDGETS = {CHECK_BUDGET: kc.BUDGETS[CHECK_BUDGET]}
    kc.arms = arms
    kc.build_rows = build_rows
    kc.run_example, kc.run_document = run_example, run_document


def main():
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--tca-plan', type=Path)
    parser.add_argument('--tca-setting', default='tca')
    parser.add_argument('--tca-calib', type=Path)
    parser.add_argument('--tca-calib-rows', type=Path)
    parser.add_argument('--model-key', choices=('llama', 'qwen3'))
    parser.add_argument('--check', action='store_true')
    args, rest = parser.parse_known_args()
    if args.check and args.tca_plan:
        raise SystemExit('--check runs without the tca row')
    if args.tca_plan:
        import tca_run as run
        plan = json.loads(args.tca_plan.read_text())
        if args.tca_setting not in plan['settings'] or not (args.tca_calib and args.tca_calib_rows and args.model_key):
            raise SystemExit('--tca-plan needs a setting in it, --tca-calib, --tca-calib-rows and --model-key')
        run.STATE.update(plan=plan, flip=tuple(plan['flip'][args.model_key]), tca_calib=args.tca_calib,
                         tca_calib_rows=args.tca_calib_rows)
    configure(args.tca_setting if args.tca_plan else None, args.check)
    sys.argv = [sys.argv[0], *rest]
    return kc.main()


if __name__ == '__main__':
    sys.exit(main())
