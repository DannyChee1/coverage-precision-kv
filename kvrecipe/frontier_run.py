from pathlib import Path
import sys
import zlib

import torch

sys.path.insert(0, str(Path(__file__).parent))
from answer_blind_e1 import gold_positions
import content_kvzip as ck
import content_kvzip_run as cr
import controls as kc
from mix import N_SINK
from diagnose import answer_tokens
import qaprior as qp
import review4k as r4
from stage2 import KEPT

B1 = '1.5625'
WEIGHTS = {'Z': 0.0, 'W10': 0.1, 'W25': 0.25, 'W50': 0.5, 'W100': 1.0}
RANDOM = {}  # controls (1 - w) P + w R, R: as many random rows per chunk as E uses (the gate's control
EXTRA = ()  # selectors whose scores a wrapping driver adds after scores() (blend_demotion_run.py
STATE = {'row': None, 'tok': None, 'press_cls': None, 'P': None, 'tokenization_ok': None, 'empty': 0,
         'frontier': None}
R4_EXAMPLE, R4_BUILD, R4_DOCUMENT = r4.run_example, r4.build_rows, r4.run_document


def arms(benchmark, budget=None):
    return [f'{k}.k3v2pc@{B1}' for k in (*WEIGHTS, *RANDOM, *EXTRA)]


def scores(pipe, context_ids, question_ids, capture_kvzip, capture_scorers, visible):
    from transformers import DynamicCache
    tok, row, n = pipe.tokenizer, STATE['row'], context_ids.shape[1]
    text = qp.prompt_text(tok, row['context'])
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ok = len(enc['input_ids']) == n
    ctx = row['context'].lstrip()  # Llama 3.1's chat template trims leading whitespace (RULER vt contexts
    c, e = ck.token_tags(ctx, text.index(ctx), enc['offset_mapping'])
    press = STATE['press_cls'](compression_ratio=0.5, kvzip_plus_normalization=True, n_sink=N_SINK)
    press.tags = {'e': e[:n] > 0, 'c': c[:n] > 0} if ok else {'e': torch.zeros(n).bool().numpy(),
                                                              'c': torch.zeros(n).bool().numpy()}
    press.seed, press.kv2_rhos = int(row['index']), {}
    cache = DynamicCache()
    with torch.no_grad(), press(pipe.model):
        pipe.model.model(input_ids=context_ids, past_key_values=cache)
    p, ev, rv = press.captured[:, 0], press.captured_variants['e'][:, 0], press.captured_variants['r'][:, 0]
    STATE.update(P=p, tokenization_ok=ok, empty=press.empty_chunks, prefix_length=press.prefix_length)
    return cache, {**{k: (1 - w) * p + w * ev for k, w in WEIGHTS.items()},
                   **{k: (1 - w) * p + w * rv for k, w in RANDOM.items()}}


def shares(plans, positions):
    kept = torch.stack([p == KEPT for p in plans])
    return float(kept[..., positions].float().mean()) if len(positions) else None


def build_rows(cache, scores, row_arms):
    arm_cache, masks, out = R4_BUILD(cache, scores, row_arms)
    row, tok = STATE['row'], STATE['tok']
    n_heads, n_tokens, head_dim = cache.layers[0].keys.shape[1:]
    if row.get('answer_spans'):
        gold = gold_positions(tok, row['context'], row['answer_spans'], n_tokens) or []
    elif row.get('answer'):
        gold = [p for p in (answer_tokens(tok, row['context'], row['answer'], n_tokens) or []) if p >= N_SINK]
    else:
        gold = []
    spans = row.get('evidence_spans')
    evidence = (gold_positions(tok, row['context'], spans, n_tokens) or []) if spans else []
    evm = sorted(set(evidence) - set(gold))
    body = STATE['P'][..., N_SINK:].float()
    fr = {'tokenization_ok': STATE['tokenization_ok'], 'gold_positions': len(gold), 'evm_positions': len(evm),
          'empty_e_chunks': STATE['empty'], 'keep_rate': {}, 'gold_kept': {}, 'evm_kept': {}, 'plain_mass': {}}
    for k in (*WEIGHTS, *RANDOM, *EXTRA):
        plans = [kc.allocate(scores[k][li].float(), 'k3v2pc', head_dim, kc.BUDGETS[B1])
                 for li in range(len(cache.layers))]
        kept = torch.stack([p == KEPT for p in plans])[..., N_SINK:].float()
        fr['keep_rate'][k] = float(kept.mean())
        fr['gold_kept'][k], fr['evm_kept'][k] = shares(plans, gold), shares(plans, evm)
        fr['plain_mass'][k] = float((body * kept).sum() / body.sum())
    out['frontier'] = STATE['frontier'] = fr
    return arm_cache, masks, out


def run_example(pipe, row, capture_kvzip, capture_scorers):
    STATE.update(row=row, tok=pipe.tokenizer)
    return R4_EXAMPLE(pipe, row, capture_kvzip, capture_scorers)


def run_document(pipe, doc, capture_kvzip, capture_scorers, cap=kc.QUALITY_CAP):
    STATE.update(row={'context': doc['context'], 'index': zlib.crc32(doc['doc_id'].encode())}, tok=pipe.tokenizer)
    out = R4_DOCUMENT(pipe, doc, capture_kvzip, capture_scorers, cap)
    out['frontier'] = STATE['frontier']
    return out


def configure():
    r4.configure(False, 'mse', None)
    kc.BUDGETS = {B1: kc.BUDGETS[B1]}
    kc.GROUPS = {k: ('k3v2pc',) for k in (*WEIGHTS, *RANDOM, *EXTRA)}
    kc.arms, kc.score_selectors, kc.build_rows, kc.run_example = arms, scores, build_rows, run_example
    kc.run_document = run_document
    STATE['press_cls'] = cr.capture_class()


def main():
    configure()
    return kc.main()


if __name__ == '__main__':
    sys.exit(main())
