import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import controls as kc
import cov2 as cv
from mix import N_SINK
import keys_ablation as ka
import qaprior as qp
from stage2 import KEPT, SINK

BUDGETS = {'1.5625': 0.015625, '3.125': 0.03125}
SPLIT = 'k3v2pc'


def gold_positions(tok, context, spans, n_tokens):
    text = qp.prompt_text(tok, context)
    base = text.index(context)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    if len(enc['input_ids']) != n_tokens:
        return None
    starts = np.array([a for a, _ in enc['offset_mapping']])
    ends = np.array([b for _, b in enc['offset_mapping']])
    out = set()
    for a, b in spans:
        lo = int(np.searchsorted(ends, base + a, side='right'))
        hi = int(np.searchsorted(starts, base + b, side='left'))
        out.update(i for i in range(lo, hi) if ends[i] > starts[i] and i >= N_SINK)
    return sorted(out)


def shares(plans, gold):
    kept = torch.stack([(p == KEPT) | (p == SINK) for p in plans])  # L, H, T
    body = kept[..., N_SINK:]
    return float(kept[..., gold].float().mean()), float(body.float().mean())


def run_doc(pipe, doc, press_cls, idf):
    from transformers import DynamicCache
    start = time.time()
    model, tok = pipe.model, pipe.tokenizer
    cfg = model.config
    q = doc['questions'][0]['question']
    inputs = pipe.preprocess(doc['context'], [q], 'Answer:', max_context_length=tok.model_max_length)
    ids = inputs['context_ids'].to(model.device)
    n = ids.shape[1]
    heads = cfg.num_key_value_heads
    head_dim = getattr(cfg, 'head_dim', None) or cfg.hidden_size // cfg.num_attention_heads
    press = press_cls(compression_ratio=0.5, kvzip_plus_normalization=True, n_sink=N_SINK)
    press.kv2_rhos = {b: cv.retained_share(ka.entry_bits(*kc.SPLITS[SPLIT], head_dim), kc.overhead(SPLIT, head_dim),
                                           heads, n, head_dim, f) for b, f in BUDGETS.items()}
    t0 = time.time()
    with torch.no_grad(), press(model):
        model.model(input_ids=ids, past_key_values=DynamicCache())
    seconds = time.time() - t0
    z = press.captured[:, 0]
    gold = gold_positions(tok, doc['context'], [s for qq in doc['questions'] for s in qq['spans']], n)
    out = {'n_tokens': n, 'questions': len(doc['questions']), 'gold_positions': len(gold or []),
           'seconds': seconds, 'kv2_seconds': press.kv2_seconds}
    if not gold:
        out['total_seconds'] = time.time() - start
        return out
    prior = qp.token_prior(tok, doc['context'], n, idf, 'idf')
    scores = {'kvzip': lambda b: z, 'kv2': lambda b: press.captured_kv2[b][:, 0]}
    if prior is not None:
        pct = qp.percentile(prior)
        scores['rarity'] = lambda b: qp.prior_only(z, pct)
    rank = z.float().argsort(-1).argsort(-1).float() / max(n - 1, 1)
    out['kvzip_gold_percentile'] = 100 * float(rank[..., gold].mean())
    for b, f in BUDGETS.items():
        for name, fn in scores.items():
            s = fn(b)
            plans = [kc.allocate(s[li].float(), SPLIT, head_dim, f) for li in range(s.shape[0])]
            g, k = shares(plans, gold)
            out.setdefault('gold_kept', {})[f'{name}@{b}'] = g
            out.setdefault('keep_rate', {})[f'{name}@{b}'] = k
    out['total_seconds'] = time.time() - start
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--docs', type=Path, nargs='+', required=True)
    parser.add_argument('--idf', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--max-docs', type=int, default=0, help='first N per file (smokes, CPU checks)')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--dtype', default='bfloat16')
    args = parser.parse_args()
    import kvpress
    from transformers import pipeline
    attn = 'flash_attention_2' if args.device.startswith('cuda') else 'sdpa'
    pipe = pipeline('kv-press-text-generation', model=args.model_path, device=args.device,
                    model_kwargs={'dtype': getattr(torch, args.dtype), 'attn_implementation': attn})
    press_cls, idf = cv.capture_class(), qp.load_idf(args.idf)
    docs = []
    for path in args.docs:
        rows = [json.loads(line) for line in open(path)]
        docs += rows[:args.max_docs] if args.max_docs else rows
    done = {json.loads(line)['id'] for line in args.out.open()} if args.out.exists() else set()
    with args.out.open('a') as handle:
        for doc in docs:
            if doc['id'] in done:
                continue
            result = run_doc(pipe, doc, press_cls, idf)
            handle.write(json.dumps({'id': doc['id'], 'source': doc['source'], 'model': pipe.model.config.name_or_path,
                                     **result}) + '\n')
            handle.flush()
            print(json.dumps({'done': doc['id'], 'seconds': round(result['total_seconds'], 2)}), flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == '__main__':
    sys.exit(main())
