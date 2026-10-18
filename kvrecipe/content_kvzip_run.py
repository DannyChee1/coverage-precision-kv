import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from answer_blind_e1 import BUDGETS, SPLIT, gold_positions
import content_kvzip as ck
import controls as kc
import cov2 as cv
from mix import N_SINK
from diagnose import answer_tokens
import keys_ablation as ka
import qaprior as qp
from stage2 import KEPT, SINK

VARIANTS = ('e', 'c', 'r')


def capture_class():
    base = cv.capture_class()

    class CaptureContent(base):
        tags, seed = None, 0  # tags: {'e': bool array, 'c': bool array} over the prompt's tokens

        def prepare(self, model, tokenizer, prev_postfix_size=8):
            pairs = super().prepare(model, tokenizer, prev_postfix_size)
            self.variant = {v: torch.zeros_like(self.score_val) for v in VARIANTS}
            self.rngs = {li: np.random.default_rng([self.seed, li]) for li in range(model.config.num_hidden_layers)}
            self.empty_chunks = 0
            return pairs

        def score_kvzip(self, module, hidden_states, keys, values, attentions, kwargs):
            if self.tags is None:  # KV²'s passes stay plain
                return super().score_kvzip(module, hidden_states, keys, values, attentions, kwargs)
            q_len = hidden_states.shape[1]
            sink = min(self.n_sink, self.start_idx)
            ctx_len = self.end_idx - self.start_idx
            attn = self._compute_cross_attention(module, hidden_states, keys, kwargs)
            if self.kvzip_plus_normalization:  # as KVzipPress.score_kvzip
                attn = torch.einsum('b h g t i, b t -> b h g t i', attn, 1 / torch.norm(hidden_states, dim=-1))
                sub = torch.cat([values[:, :, :sink], values[:, :, self.start_idx:self.end_idx], values[:, :, -q_len:]],
                                dim=2)
                attn = torch.einsum('b h g i t, b h g t -> b h g i t', attn,
                                    self._compute_value_output_norm(module, sub))
            attn = attn[..., sink:sink + ctx_len]
            li, s, e = int(module.layer_idx), self.start_idx, self.end_idx
            plain = attn.amax(dim=(-3, -2))
            self.score_val[li][..., s:e] = plain
            first = q_len - ctx_len - 1  # the row that reproduces the chunk's first token
            rows = {v: first + np.nonzero(self.tags[v][s:e])[0] for v in ('e', 'c')}
            rows['r'] = first + np.sort(self.rngs[li].choice(ctx_len, size=len(rows['e']), replace=False))
            if li == 0 and not len(rows['e']):
                self.empty_chunks += 1
            for v in VARIANTS:
                if len(rows[v]):
                    idx = torch.as_tensor(rows[v], device=attn.device)
                    self.variant[v][li][..., s:e] = attn.index_select(-2, idx).amax(dim=(-3, -2))
                else:
                    self.variant[v][li][..., s:e] = plain
            return keys[:, :, :self.context_length], values[:, :, :self.context_length]

        def compress_post(self, model):
            self.captured_variants = {v: t.float().clone() for v, t in self.variant.items()}
            self.tags = None
            super().compress_post(model)

    return CaptureContent


def shares(plans, positions):
    kept = torch.stack([(p == KEPT) | (p == SINK) for p in plans])  # L, H, T
    return float(kept[..., positions].float().mean()) if len(positions) else None


def run_row(pipe, row, press_cls):
    from transformers import DynamicCache
    start = time.time()
    model, tok = pipe.model, pipe.tokenizer
    cfg = model.config
    inputs = pipe.preprocess(row['context'], ['?'], 'Answer:', max_context_length=tok.model_max_length)
    ids = inputs['context_ids'].to(model.device)
    n = ids.shape[1]
    heads = cfg.num_key_value_heads
    head_dim = getattr(cfg, 'head_dim', None) or cfg.hidden_size // cfg.num_attention_heads
    text = qp.prompt_text(tok, row['context'])
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    out = {'n_tokens': n, 'tokenization_ok': len(enc['input_ids']) == n}
    if not out['tokenization_ok']:
        return out | {'total_seconds': time.time() - start}
    c, e = ck.token_tags(row['context'], text.index(row['context']), enc['offset_mapping'])
    press = press_cls(compression_ratio=0.5, kvzip_plus_normalization=True, n_sink=N_SINK)
    press.tags, press.seed = {'e': e > 0, 'c': c > 0}, int(row['index'])
    press.kv2_rhos = {b: cv.retained_share(ka.entry_bits(*kc.SPLITS[SPLIT], head_dim), kc.overhead(SPLIT, head_dim),
                                           heads, n, head_dim, f) for b, f in BUDGETS.items()}
    t0 = time.time()
    with torch.no_grad(), press(model):
        model.model(input_ids=ids, past_key_values=DynamicCache())
    out['seconds'], out['kv2_seconds'] = time.time() - t0, press.kv2_seconds
    z = press.captured[:, 0]
    var = {v: press.captured_variants[v][:, 0] for v in VARIANTS}
    if 'gold_spans' in row:
        gold = gold_positions(tok, row['context'], row['gold_spans'], n) or []
    else:
        gold = [p for p in (answer_tokens(tok, row['context'], row['answer'], n) or []) if p >= N_SINK]
    evidence = gold_positions(tok, row['context'], row['evidence_spans'], n) or []
    evm = sorted(set(evidence) - set(gold))
    body = z[..., N_SINK:].float()
    out.update(gold_positions=len(gold), evm_positions=len(evm), tagged_e=int(e.sum()), tagged_c=int(c.sum()),
               empty_e_chunks=press.empty_chunks,
               checks={f'{v}_le_plain': bool((var[v] <= z + 1e-6).all()) for v in VARIANTS}
               | {'c_ge_e': bool((var['c'] >= var['e'] - 1e-6).all())})
    scores = {'kvzip': lambda b: z, **{f'kvzip_{v}': (lambda b, v=v: var[v]) for v in VARIANTS},
              'kv2': lambda b: press.captured_kv2[b][:, 0]}
    for b, f in BUDGETS.items():
        for name, fn in scores.items():
            s = fn(b)
            plans = [kc.allocate(s[li].float(), SPLIT, head_dim, f) for li in range(s.shape[0])]
            kept = torch.stack([(p == KEPT) for p in plans])[..., N_SINK:].float()
            key = f'{name}@{b}'
            out.setdefault('keep_rate', {})[key] = float(kept.mean())
            out.setdefault('gold_kept', {})[key] = shares(plans, gold)
            out.setdefault('evm_kept', {})[key] = shares(plans, evm)
            out.setdefault('plain_mass', {})[key] = float((body * kept).sum() / body.sum())
    out['total_seconds'] = time.time() - start
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--rows', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--max-rows', type=int, default=0, help='CPU checks only')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--dtype', default='bfloat16')
    args = parser.parse_args()
    import kvpress
    from transformers import pipeline
    attn = 'flash_attention_2' if args.device.startswith('cuda') else 'sdpa'
    pipe = pipeline('kv-press-text-generation', model=args.model_path, device=args.device,
                    model_kwargs={'dtype': getattr(torch, args.dtype), 'attn_implementation': attn})
    press_cls = capture_class()
    rows = [json.loads(line) for line in open(args.rows)]
    rows = rows[:args.max_rows] if args.max_rows else rows
    done = {json.loads(line)['index'] for line in args.out.open()} if args.out.exists() else set()
    with args.out.open('a') as handle:
        for row in rows:
            if row['index'] in done:
                continue
            result = run_row(pipe, row, press_cls)
            handle.write(json.dumps({'index': row['index'], 'task': row['task'], 'doc_id': row['doc_id'],
                                     'model': pipe.model.config.name_or_path, **result}) + '\n')
            handle.flush()
            print(json.dumps({'done': row['index'], 'seconds': round(result['total_seconds'], 2)}), flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == '__main__':
    sys.exit(main())
