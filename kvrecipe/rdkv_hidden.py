import argparse
import json
import math
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).parent))
import controls as kc
import rdkv as rd
from mix import N_SINK, layer_budget_bits

GROUPS = {'Z': ('stack4pc', 'k3v2pc', 'rdkv')}
STATE = {'tables': None, 'calib': None, 'calib_rows': None, 'hist': {}}
BASE_BUILD, BASE_EXAMPLE, BASE_DOCUMENT = kc.build_rows, kc.run_example, kc.run_document


def fixed_bits(head_dim):
    return 4 * 32 + 3 * head_dim


def is_rdkv(arm):
    return arm.startswith('Z.rdkv@')


def query_hooks(model, sums):
    def hook(module, args, kwargs):
        hidden = kwargs['hidden_states']
        cos, sin = kwargs['position_embeddings']
        q = module.q_proj(hidden).view(*hidden.shape[:-1], -1, module.head_dim)
        if hasattr(module, 'q_norm'):
            q = module.q_norm(q)
        q = q.transpose(1, 2)
        q, _ = sys.modules[type(module).__module__].apply_rotary_pos_emb(q, q, cos, sin)
        sums[module.layer_idx] = (q[0].float() ** 2).sum(1)
    return [layer.self_attn.register_forward_pre_hook(hook, with_kwargs=True) for layer in model.model.layers]


def calibrate(pipe, rows_path, out_path):
    from transformers import DynamicCache
    t0, sums = time.time(), None
    rows = [json.loads(line) for line in open(rows_path)]
    for row in rows:
        inputs = pipe.preprocess(row['context'], [row['question']], row['answer_prefix'],
                                 max_context_length=pipe.tokenizer.model_max_length)
        cache = DynamicCache()
        with torch.no_grad():
            pipe.model.model(input_ids=inputs['context_ids'].to(pipe.model.device), past_key_values=cache)
        parts = [rd.calibration_sums(layer.keys[0], layer.values[0]) for layer in cache.layers]
        parts = [torch.stack([p[i] if p[i].dim() == 2 else p[i][:, None] for p in parts]) for i in range(4)]
        sums = parts if sums is None else [s + p for s, p in zip(sums, parts)]
    ev, ek = zip(*[rd.eps_tables(sums[0][li], sums[1][li][:, 0], sums[2][li], sums[3][li][:, 0])
                   for li in range(sums[0].shape[0])])
    tables = {'eps_v': torch.stack(ev).tolist(), 'eps_k': torch.stack(ek).tolist(), 'bits': list(rd.BITS),
              'contexts': len(rows), 'tasks': sorted({r['task'] for r in rows}), 'indices': [r['index'] for r in rows],
              'seconds': time.time() - t0}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(tables) + '\n')
    return tables


def tables(pipe):
    if STATE['tables'] is None:
        path = STATE['calib']
        STATE['tables'] = (json.loads(path.read_text()) if path.exists()
                           else calibrate(pipe, STATE['calib_rows'], path))
    return STATE['tables']


def hidden_scores(pipe, context_ids, question_ids, capture_kvzip, capture_scorers, visible):
    from transformers import DynamicCache
    model = pipe.model
    tables(pipe)
    cache, sums = DynamicCache(), {}
    press = capture_kvzip(compression_ratio=0.5, kvzip_plus_normalization=True, n_sink=N_SINK)
    with torch.no_grad(), press(model):
        hooks = query_hooks(model, sums)
        try:
            model.model(input_ids=context_ids, past_key_values=cache)
        finally:
            for h in hooks:
                h.remove()
    hkv = cache.layers[0].keys.shape[1]
    qn = torch.stack([sums[i].reshape(hkv, -1, sums[i].shape[-1]).sum(1).sqrt() for i in range(len(cache.layers))])
    return cache, {'Z': press.captured[:, 0], 'QN': qn}


def rdkv_weights(cache, scores):
    d = cache.layers[0].keys.shape[-1]
    w_t = rd.pool(scores['Z'].float().clamp(min=0))
    k_norm = torch.stack([layer.keys[0].float().norm(dim=1) for layer in cache.layers])
    return w_t, scores['QN'].to(k_norm.device) * k_norm / math.sqrt(d)


def build_rows(cache, scores, row_arms):
    arm_cache, masks, out = BASE_BUILD(cache, scores, ['full' if is_rdkv(a) else a for a in row_arms])
    rows = [(row, arm) for row, arm in enumerate(row_arms) if is_rdkv(arm)]
    if not rows:
        return arm_cache, masks, out
    n_layers = len(cache.layers)
    n_heads, n_tokens, head_dim = cache.layers[0].keys.shape[1:]
    sink = min(N_SINK, n_tokens)
    w_t, w_c = rdkv_weights(cache, scores)
    eps_v, eps_k = (torch.tensor(STATE['tables'][k], device=w_t.device) for k in ('eps_v', 'eps_k'))
    for row, arm in rows:
        limit = layer_budget_bits(n_heads, n_tokens, head_dim, kc.BUDGETS[arm.split('@')[1]])
        vbits, kbits, _ = rd.allocate_model(w_t, w_c, eps_v, eps_k, limit / n_heads - fixed_bits(head_dim), sink)
        layer_bits = rd.plan_bits(vbits, kbits, sink).sum(-1) + n_heads * fixed_bits(head_dim)
        out['bits'][arm], out['budget_bits'][arm] = float(layer_bits.sum()), n_layers * limit
        out['over_budget_layers'][arm] = int((layer_bits > limit).sum())
        out['kept'][arm] = int((vbits[..., sink:] > 0).sum())
        body = vbits[..., sink:]
        has = body.gt(0).any(-1, keepdim=True)
        STATE['hist'][arm] = {'v': [int((body == b).sum()) for b in rd.BITS],
                              'k': [int(((kbits == b) & has).sum()) for b in rd.BITS],
                              'heads_without_tokens': int((~has).sum())}
        for li in range(n_layers):
            keys, values = cache.layers[li].keys[0], cache.layers[li].values[0]
            arm_cache.layers[li].keys[row], arm_cache.layers[li].values[row] = rd.apply_plan(
                keys, values, vbits[li], kbits[li], sink)
            heads, seqs = torch.nonzero(vbits[li] == 0, as_tuple=True)
            if heads.numel():
                masks[li].append((torch.full_like(heads, row), heads, seqs))
    return arm_cache, masks, out


def run_example(pipe, row, capture_kvzip, capture_scorers):
    STATE['hist'] = {}
    out = BASE_EXAMPLE(pipe, row, capture_kvzip, capture_scorers)
    out['rdkv'] = STATE['hist']
    return out


def run_document(pipe, doc, capture_kvzip, capture_scorers, cap=kc.QUALITY_CAP):
    STATE['hist'] = {}
    out = BASE_DOCUMENT(pipe, doc, capture_kvzip, capture_scorers, cap)
    out['rdkv'] = STATE['hist']
    return out


def configure():
    kc.GROUPS = GROUPS
    kc.score_selectors = hidden_scores
    kc.build_rows = build_rows
    kc.run_example = run_example
    kc.run_document = run_document


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--calib', type=Path, required=True)
    parser.add_argument('--calib-rows', type=Path, required=True)
    args, rest = parser.parse_known_args()
    STATE.update(calib=args.calib, calib_rows=args.calib_rows)
    configure()
    sys.argv = [sys.argv[0], *rest]
    return kc.main()


if __name__ == '__main__':
    sys.exit(main())
