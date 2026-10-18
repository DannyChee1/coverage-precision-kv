import argparse
import json
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).parent))
import mix
import controls as kc
from qwen14b import hidden_scores
import review4k as r4

SPLITS = ('stack4pc', 'k4v2pc', 'k3v2pc', 'evict16')
STATE = {'max_batch': 0}


def batches(tca):
    out = []
    for b in kc.BUDGETS:
        rows = (['full'] if b == '1.5625' else []) + [f'Z.{s}@{b}' for s in SPLITS + (('tca',) if tca else ())]
        size = STATE['max_batch'] if STATE['max_batch'] and b == '1.5625' else len(rows)
        out += [rows[i:i + size] for i in range(0, len(rows), size)]
    return out


def build_rows(cache, scores, row_arms):
    arm_cache, masks, out = r4.BASE_BUILD(cache, scores, ['full' if a.startswith('Z.tca@') else a for a in row_arms])
    if any(a.startswith('Z.tca@') for a in row_arms):
        r4.tca_rows(cache, scores, row_arms, arm_cache, masks, out)
    return arm_cache, masks, out


def run_example(pipe, row, capture_kvzip, capture_scorers):
    model = pipe.model
    inputs = pipe.preprocess(row['context'], [row['question']], row['answer_prefix'],
                             max_context_length=pipe.tokenizer.model_max_length)
    context_ids = inputs['context_ids'].to(model.device)
    question_ids = inputs['questions_ids'][0].to(model.device)
    n_tokens = context_ids.shape[1]
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    cache, scores = kc.score_selectors(pipe, context_ids, question_ids, capture_kvzip, capture_scorers, False)
    out = {'n_tokens': n_tokens, 'score_seconds': time.time() - t0, 'build_seconds': 0.0, 'decode_seconds': 0.0,
           'bits': {}, 'budget_bits': {}, 'over_budget_layers': {}, 'kept': {}, 'answers': {}, 'batches': []}
    for rows in batches(r4.STATE['tca']):
        t1 = time.time()
        arm_cache, masks, acct = build_rows(cache, scores, rows)
        for key in ('bits', 'budget_bits', 'over_budget_layers', 'kept'):
            out[key].update(acct[key])
        out['full_bits'] = acct['full_bits']
        out['build_seconds'] += time.time() - t1
        mix.set_masks(model, masks)
        t2 = time.time()
        try:
            with torch.no_grad():
                answers = mix.batched_generate(model, pipe.tokenizer, question_ids, arm_cache, n_tokens,
                                                 int(row['max_new_tokens']))
        finally:
            mix.clear_masks(model)
        out['decode_seconds'] += time.time() - t2
        out['answers'].update(zip(rows, answers))
        out['batches'].append(len(rows))
        del arm_cache, masks
    del cache
    out['peak_gib'] = r4.peak_gib()
    return out


def configure(max_batch=0, tca=None):
    STATE['max_batch'] = max_batch
    r4.STATE.update(tca=tca, gate=False)
    kc.GROUPS = {'Z': SPLITS}
    kc.score_selectors = r4.tca_scores if tca else hidden_scores
    kc.run_example = run_example


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--max-batch', type=int, default=0)
    parser.add_argument('--tca-plan', type=Path)
    parser.add_argument('--tca-setting', default='tca')
    parser.add_argument('--tca-calib', type=Path)
    parser.add_argument('--tca-calib-rows', type=Path)
    parser.add_argument('--model-key', choices=('llama', 'qwen3'))
    args, rest = parser.parse_known_args()
    if args.tca_plan:
        import tca_run as run
        plan = json.loads(args.tca_plan.read_text())
        if args.tca_setting not in plan['settings'] or not (args.tca_calib and args.tca_calib_rows and args.model_key):
            raise SystemExit('--tca-plan needs a setting in it, --tca-calib, --tca-calib-rows and --model-key')
        run.STATE.update(plan=plan, flip=tuple(plan['flip'][args.model_key]), tca_calib=args.tca_calib,
                         tca_calib_rows=args.tca_calib_rows)
    configure(args.max_batch, args.tca_setting if args.tca_plan else None)
    sys.argv = [sys.argv[0], *rest]
    return kc.main()


if __name__ == '__main__':
    sys.exit(main())
