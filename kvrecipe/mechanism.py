import argparse
import copy
from collections import defaultdict
import json
from pathlib import Path
import sys
import time

import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

sys.path.insert(0, str(Path(__file__).parent))
import mix
from mix import N_SINK
from confirm import load_rows
from stage2 import KEPT, allocate_uniform, quantize_parts

ROWS = {'k4v4': (4, 4), 'k4v2': (4, 2), 'k2v4': (2, 4), 'k2v2': (2, 2)}  # all at the k4v2 kept set
BITS = (2, 3, 4)
JOINT = ((4, 4), (4, 2), (3, 3), (2, 4), (2, 2))
TOPK = (16, 64)
NAME = 'mix_probe'


class Probe:

    def __init__(self, n_ctx, first_query, keys_q, values_q, blocked):
        self.n_ctx, self.first_query = n_ctx, first_query
        self.keys_q, self.values_q = keys_q, values_q  # layer -> {bits: (KVH, n_ctx, D)}
        self.blocked = blocked  # layer -> bool (KVH, n_ctx), or None for the full context
        self.sums = defaultdict(float)
        self.counted = defaultdict(int)  # layers in which each top-k metric was defined
        self.layers = 0
        self.leak_max = 0.0

    def summary(self):
        out = {k: v / self.counted.get(k, self.layers) for k, v in sorted(self.sums.items())}
        if self.blocked is not None:
            out['leak_max'] = self.leak_max
        return out


ACTIVE = None


def rel(diff, ref):
    return (diff.norm(dim=-1) / ref.norm(dim=-1).clamp(min=1e-12)).mean().item()


def probe_attention(module, query, key, value, attention_mask, dropout=0.0, scaling=None, **kwargs):
    state, layer = ACTIVE, module.layer_idx
    q = query[0].float()  # H, T, D
    groups = q.shape[0] // key.shape[1]
    k, v = key[0].float().repeat_interleave(groups, 0), value[0].float().repeat_interleave(groups, 0)
    n_ctx, t = state.n_ctx, q.shape[1]
    new = k.shape[1] - n_ctx  # teacher-forced tokens in the cache so far (all of them: one forward
    causal = torch.ones(t, new, dtype=torch.bool, device=q.device).tril(new - t)
    ctx_mask = torch.zeros(q.shape[0], 1, n_ctx, dtype=torch.bool, device=q.device)
    if state.blocked is not None:
        ctx_mask = state.blocked[layer].repeat_interleave(groups, 0)[:, None, :]

    def logits_for(k_ctx):
        logits = torch.cat([q @ k_ctx.transpose(-1, -2), q @ k[:, n_ctx:].transpose(-1, -2)], -1) * scaling
        logits[..., :n_ctx] = logits[..., :n_ctx].masked_fill(ctx_mask, float('-inf'))
        logits[..., n_ctx:] = logits[..., n_ctx:].masked_fill(~causal, float('-inf'))
        return logits

    def weights(k_ctx):
        return logits_for(k_ctx).softmax(-1)

    exact = logits_for(k[:, :n_ctx])
    a = exact.softmax(-1)
    if state.blocked is not None:
        leak_check(state, layer, query, exact, groups, scaling)
    live = (~ctx_mask).sum(-1)[:, 0]  # per query head; top-k is defined only within the live entries
    out = a @ v
    sel = slice(state.first_query, t)  # queries that generate answer tokens
    ref, a_s = out[:, sel], a[:, sel]
    s = state.sums
    for b in BITS:
        kq = state.keys_q[layer][b].float().repeat_interleave(groups, 0)
        vq = state.values_q[layer][b].float().repeat_interleave(groups, 0)
        v_mixed = torch.cat([vq, v[:, n_ctx:]], 1)
        aq = weights(kq)[:, sel]
        for top in TOPK:
            valid = live >= top
            if valid.any():
                picks = a_s[valid, :, :n_ctx].topk(top, -1).indices
                picks_q = aq[valid, :, :n_ctx].topk(top, -1).indices
                hit = (picks[..., :, None] == picks_q[..., None, :]).any(-1).float().mean().item()
                s[f'D{top}_K{b}'] += 1 - hit
                state.counted[f'D{top}_K{b}'] += 1
        s[f'TV_K{b}'] += 0.5 * (aq - a_s).abs().sum(-1).mean().item()
        s[f'E_K{b}'] += rel(aq @ v - ref, ref)
        s[f'E_V{b}'] += rel(a_s @ v_mixed - ref, ref)
    for kb, vb in JOINT:
        kq = state.keys_q[layer][kb].float().repeat_interleave(groups, 0)
        v_mixed = torch.cat([state.values_q[layer][vb].float().repeat_interleave(groups, 0), v[:, n_ctx:]], 1)
        s[f'E_K{kb}V{vb}'] += rel(weights(kq)[:, sel] @ v_mixed - ref, ref)
    state.layers += 1
    return out.to(query.dtype).transpose(0, 1)[None].contiguous(), None


ALL_ATTENTION_FUNCTIONS[NAME] = probe_attention


def leak_check(state, layer, query, exact, groups, scaling):
    from kvpress.attention_patch import search_hyperplane
    qlen = state.first_query + 1  # the question and answer prefix
    heads, _, dim = query.shape[1:]
    x = query[0, :, :qlen].reshape(heads // groups, groups * qlen, dim)  # kvpress's grouping, in the run's dtype
    fake = search_hyperplane(x).float().repeat_interleave(groups, 0)  # H, D
    fake_logit = (query[0, :, :qlen].float() @ fake[:, :, None])[..., 0] * scaling  # H, qlen
    masked = state.blocked[layer].sum(-1).repeat_interleave(groups).float().clamp(min=1)[:, None]
    leak = torch.sigmoid(fake_logit + masked.log() - torch.logsumexp(exact[:, :qlen], -1))
    state.sums['leak_mean'] += leak.mean().item()
    state.leak_max = max(state.leak_max, leak.max().item())


def entry_errors(keys, values, kq, vq, sel):
    out = {}
    for b in BITS:
        out[f'e_K{b}'] = rel(kq[b].float()[sel] - keys.float()[sel], keys.float()[sel])
        out[f'e_V{b}'] = rel(vq[b].float()[sel] - values.float()[sel], values.float()[sel])
    return out


def probe_pass(model, cache, probe_ids, n_ctx, first_query, keys_q, values_q, blocked):
    global ACTIVE
    state = Probe(n_ctx, first_query, keys_q, values_q, blocked)
    original = model.config._attn_implementation
    ACTIVE, model.config._attn_implementation = state, NAME
    try:
        with torch.no_grad():
            positions = torch.arange(n_ctx, n_ctx + probe_ids.shape[1], device=model.device)[None]
            model.model(input_ids=probe_ids, past_key_values=copy.deepcopy(cache), position_ids=positions)
    finally:
        ACTIVE, model.config._attn_implementation = None, original
    return state.summary()


def run_example(pipe, row, full_answer, capture_cls):
    from transformers import DynamicCache
    model, tok = pipe.model, pipe.tokenizer
    inputs = pipe.preprocess(row['context'], [row['question']], row['answer_prefix'],
                             max_context_length=tok.model_max_length)
    context_ids = inputs['context_ids'].to(model.device)
    question_ids = inputs['questions_ids'][0].to(model.device)
    n_ctx = context_ids.shape[1]
    cache = DynamicCache()
    press = capture_cls(compression_ratio=0.5, kvzip_plus_normalization=True, n_sink=N_SINK)
    t0 = time.time()
    with torch.no_grad():
        with press(model):
            model.model(input_ids=context_ids, past_key_values=cache)
    scores = press.captured[:, 0]
    n_layers, head_dim = len(cache.layers), cache.layers[0].keys.shape[-1]
    plans = [allocate_uniform(scores[i], 4, 2, head_dim) for i in range(n_layers)]
    keys_q, values_q, kept_q, kept_v, errors_kept, errors_full = {}, {}, {}, {}, [], []
    for i in range(n_layers):
        keys, values = cache.layers[i].keys[0], cache.layers[i].values[0]
        kq, vq = quantize_parts(keys, values, BITS, BITS)
        body = torch.zeros_like(plans[i], dtype=torch.bool)
        body[:, N_SINK:] = True  # sinks stay at 16 bits in every row and pass
        kept = plans[i] == KEPT
        keys_q[i] = {b: torch.where(body[..., None], kq[b], keys) for b in BITS}
        values_q[i] = {b: torch.where(body[..., None], vq[b], values) for b in BITS}
        kept_q[i] = {b: torch.where(kept[..., None], kq[b], keys) for b in BITS}
        kept_v[i] = {b: torch.where(kept[..., None], vq[b], values) for b in BITS}
        errors_kept.append(entry_errors(keys, values, kq, vq, kept))
        errors_full.append(entry_errors(keys, values, kq, vq, body))
    score_seconds = time.time() - t0
    answer_ids = tok.encode(full_answer, add_special_tokens=False, return_tensors='pt')[:, :int(row['max_new_tokens'])]
    probe_ids = torch.cat([question_ids, answer_ids.to(model.device)], 1)
    first_query = question_ids.shape[1] - 1
    t1 = time.time()
    blocked = {i: plans[i] == 0 for i in range(n_layers)}
    metrics = {'kept': probe_pass(model, cache, probe_ids, n_ctx, first_query, kept_q, kept_v, blocked),
               'full': probe_pass(model, cache, probe_ids, n_ctx, first_query, keys_q, values_q, None)}
    for name, errs in (('kept', errors_kept), ('full', errors_full)):
        metrics[name].update({k: sum(e[k] for e in errs) / n_layers for k in errs[0]})
    probe_seconds = time.time() - t1
    arm_cache = mix.batched_cache(cache, len(ROWS))
    masks = [[] for _ in range(n_layers)]
    for i in range(n_layers):
        for r, (kb, vb) in enumerate(ROWS.values()):
            sel = plans[i] == KEPT
            arm_cache.layers[i].keys[r][sel] = kept_q[i][kb][sel]
            arm_cache.layers[i].values[r][sel] = kept_v[i][vb][sel]
            heads, seqs = torch.nonzero(plans[i] == 0, as_tuple=True)
            if heads.numel():
                masks[i].append((torch.full_like(heads, r), heads, seqs))
    del cache, keys_q, values_q, kept_q, kept_v
    mix.set_masks(model, masks)
    t2 = time.time()
    try:
        with torch.no_grad():
            answers = mix.batched_generate(model, tok, question_ids, arm_cache, n_ctx, int(row['max_new_tokens']))
    finally:
        mix.clear_masks(model)
    return {'n_tokens': n_ctx, 'probe_queries': int(probe_ids.shape[1] - first_query), 'metrics': metrics,
            'answers': dict(zip(ROWS, answers)), 'score_seconds': score_seconds, 'probe_seconds': probe_seconds,
            'decode_seconds': time.time() - t2}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--rows', type=Path, required=True)
    parser.add_argument('--confirm', nargs='+', required=True, help="the model's confirmation outputs")
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--per-task', type=int, default=10)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--dtype', default='bfloat16')
    parser.add_argument('--max-new-tokens', type=int, default=0, help='CPU tests only')
    args = parser.parse_args()
    import kvpress
    from transformers import pipeline
    attn = 'flash_attention_2' if args.device.startswith('cuda') else 'sdpa'
    pipe = pipeline('kv-press-text-generation', model=args.model_path, device=args.device,
                    model_kwargs={'dtype': getattr(torch, args.dtype), 'attn_implementation': attn})
    full = {json.loads(line)['index']: json.loads(line)['answers']['full'] for p in args.confirm for line in open(p)}
    rows = load_rows(args.rows, args.per_task, 0)
    done = {json.loads(line)['index'] for line in args.out.open()} if args.out.exists() else set()
    capture_cls = mix.capture_press_class()
    with args.out.open('a') as handle:
        for row in rows:
            if row['index'] in done:
                continue
            if args.max_new_tokens:
                row['max_new_tokens'] = min(int(row['max_new_tokens']), args.max_new_tokens)
            result = run_example(pipe, row, full[row['index']], capture_cls)
            handle.write(json.dumps({'index': row['index'], 'task': row['task'], 'answer': list(row['answer']),
                                     'model': pipe.model.config.name_or_path, 'dtype': str(pipe.model.dtype),
                                     **result}) + '\n')
            handle.flush()
            print(json.dumps({'done': row['index'], 'task': row['task'],
                              'seconds': round(result['score_seconds'] + result['probe_seconds']
                                               + result['decode_seconds'], 2)}), flush=True)


if __name__ == '__main__':
    sys.exit(main())
