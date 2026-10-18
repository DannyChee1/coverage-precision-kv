import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import mix
import rdkv as rd

TASKS = ('qasper', 'multifieldqa_en', 'trec', 'lcc')
CALIB_TASKS = ('narrativeqa', 'hotpotqa', 'gov_report', 'qmsum')
RAW_TASKS = ('trec', 'triviaqa', 'samsum', 'lsht', 'lcc', 'repobench-p')
BUDGETS = (64, 128)
WINDOW, MAX_LENGTH, CALIB_PER_TASK, CALIB_TOKENS = 32, 127500, 8, 4096
CAPTURE = 'mix_rdkv_window'
STATE = {}  # 'q': {layer: (Hq, WINDOW, d)} while a prefill runs with capture on


def capture_attention(module, query, key, value, attention_mask, dropout=0.0, scaling=None, **kwargs):
    groups = query.shape[1] // key.shape[1]
    k, v = key.repeat_interleave(groups, dim=1), value.repeat_interleave(groups, dim=1)
    out = torch.nn.functional.scaled_dot_product_attention(query, k, v, is_causal=query.shape[2] > 1,
                                                           scale=scaling)
    if 'q' in STATE and query.shape[2] >= WINDOW:
        STATE['q'][module.layer_idx] = query[0, :, -WINDOW:].float()
    return out.transpose(1, 2).contiguous(), None


def prompt_ids(tok, task, obj, prompts):
    prompt = prompts[task].format(**obj)
    ids = tok(prompt, truncation=False, add_special_tokens=False).input_ids
    if len(ids) > MAX_LENGTH:
        half = MAX_LENGTH // 2
        prompt = tok.decode(ids[:half], skip_special_tokens=True) + tok.decode(ids[-half:], skip_special_tokens=True)
    if task in RAW_TASKS:
        return tok(prompt, truncation=False).input_ids
    text = tok.apply_chat_template([{'role': 'user', 'content': prompt}], add_generation_prompt=True, tokenize=False)
    return tok(text, truncation=False, add_special_tokens=False).input_ids


def prefill(model, ids, capture):
    from transformers import DynamicCache
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    ALL_ATTENTION_FUNCTIONS[CAPTURE] = capture_attention
    saved = model.config._attn_implementation
    model.config._attn_implementation = CAPTURE
    cache = DynamicCache()
    if capture:
        STATE['q'] = {}
    try:
        with torch.no_grad():
            model.model(input_ids=ids, past_key_values=cache)
    finally:
        model.config._attn_implementation = saved
    return cache, STATE.pop('q', None)


def calibrate(model, tok, data, prompts, out_path):
    sums = None
    for task in CALIB_TASKS:
        rows = [json.loads(line) for line in open(data / f'{task}.jsonl')][:CALIB_PER_TASK]
        for obj in rows:
            ids = torch.tensor([prompt_ids(tok, task, obj, prompts)[:CALIB_TOKENS]], device=model.device)
            cache, _ = prefill(model, ids, capture=False)
            parts = [rd.calibration_sums(layer.keys[0], layer.values[0]) for layer in cache.layers]
            parts = [torch.stack([p[i] if p[i].dim() == 2 else p[i][:, None] for p in parts]) for i in range(4)]
            sums = parts if sums is None else [s + p for s, p in zip(sums, parts)]
    ev, ek = zip(*[rd.eps_tables(sums[0][li], sums[1][li][:, 0], sums[2][li], sums[3][li][:, 0])
                   for li in range(sums[0].shape[0])])
    tables = {'eps_v': torch.stack(ev).tolist(), 'eps_k': torch.stack(ek).tolist(), 'bits': list(rd.BITS),
              'calib_tasks': list(CALIB_TASKS), 'per_task': CALIB_PER_TASK, 'tokens': CALIB_TOKENS}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(tables) + '\n')
    return tables


def window_weights(q, keys, n_prefix):
    hq, w, d = q.shape
    hkv = keys.shape[0]
    qg = q.reshape(hkv, (hq // hkv) * w, d)
    k = keys.float()
    logits = qg @ k.transpose(1, 2) / math.sqrt(d)
    pos = (n_prefix - w + torch.arange(w, device=q.device)).repeat(hq // hkv)
    logits = logits.masked_fill(torch.arange(n_prefix, device=q.device)[None, None, :] > pos[None, :, None],
                                float('-inf'))
    w_t = rd.pool(logits.softmax(-1).sum(1))
    w_c = qg.norm(dim=1) * k.norm(dim=1) / math.sqrt(d)
    return w_t, w_c


def build_rows(cache, queries, eps_v, eps_k, budgets):
    rows = ['full'] + [f'{kind}{n}' for n in budgets for kind in ('snap', 'rdkv')]
    n_layers = len(cache.layers)
    hkv, n_prefix, d = cache.layers[0].keys.shape[1:]
    arm_cache = mix.batched_cache(cache, len(rows))
    masks = [[] for _ in range(n_layers)]
    acct = {r: {'bits': 0, 'budget': 0, 'kept': 0} for r in rows[1:]}
    hist = {r: {'v': [0] * len(rd.BITS), 'k': [0] * len(rd.BITS)} for r in rows if r.startswith('rdkv')}
    w_t, w_c = (torch.stack(w) for w in zip(*[window_weights(queries[li], cache.layers[li].keys[0], n_prefix)
                                               for li in range(n_layers)]))
    device = w_t.device
    plans = {}
    for name in hist:
        n = int(name[4:])
        plans[name] = rd.allocate_model(w_t, w_c, torch.tensor(eps_v, device=device),
                                        torch.tensor(eps_k, device=device), n * 2 * d * 16)
        acct[name]['bits'] = plans[name][2]
        for i, b in enumerate(rd.BITS):
            hist[name]['v'][i] = int((plans[name][0] == b).sum())
            hist[name]['k'][i] = int((plans[name][1] == b).sum())
    for li in range(n_layers):
        keys, values = cache.layers[li].keys[0], cache.layers[li].values[0]
        for row, name in enumerate(rows):
            if name == 'full':
                continue
            n = int(name[4:])
            budget = n * 2 * d * 16
            if name.startswith('snap'):
                keep = torch.zeros(hkv, n_prefix, dtype=torch.bool, device=keys.device)
                keep[:, max(0, n_prefix - WINDOW):] = True
                extra = n - WINDOW
                early = n_prefix - WINDOW
                if early > 0 and extra > 0:
                    top = w_t[li, :, :early].topk(min(extra, early), dim=-1).indices
                    keep.scatter_(1, top, True)
                acct[name]['bits'] += int(keep.sum()) * 2 * d * 16
            else:
                vbits, kbits = plans[name][0][li], plans[name][1][li]
                keep = vbits > 0
                arm_cache.layers[li].keys[row], arm_cache.layers[li].values[row] = rd.apply_plan(
                    keys, values, vbits, kbits)
            acct[name]['budget'] += hkv * budget
            acct[name]['kept'] += int(keep.sum())
            heads, seqs = torch.nonzero(~keep, as_tuple=True)
            if heads.numel():
                masks[li].append((torch.full_like(heads, row), heads, seqs))
    return rows, arm_cache, masks, acct, hist


def sync(model):
    if model.device.type == 'cuda':
        torch.cuda.synchronize()


def run_example(model, tok, task, obj, prompts, maxlen, tables, budgets, max_new=0):
    ids = torch.tensor([prompt_ids(tok, task, obj, prompts)], device=model.device)
    prefix, last = ids[:, :-1], ids[:, -1:]
    t0 = time.time()
    cache, queries = prefill(model, prefix, capture=True)
    sync(model)
    t1 = time.time()
    rows, arm_cache, masks, acct, hist = build_rows(cache, queries, tables['eps_v'], tables['eps_k'], budgets)
    del cache
    sync(model)
    t2 = time.time()
    mix.set_masks(model, masks)
    try:
        with torch.no_grad():
            answers = mix.batched_generate(model, tok, last, arm_cache, prefix.shape[1], max_new or maxlen[task])
    finally:
        mix.clear_masks(model)
    return {'task': task, 'n_tokens': int(ids.shape[1]), 'answers': dict(zip(rows, answers)),
            'gold': obj['answers'], 'all_classes': obj.get('all_classes'), 'length': obj.get('length'),
            'accounting': acct, 'bit_histograms': hist,
            'seconds': {'prefill': t1 - t0, 'allocate': t2 - t1, 'decode': time.time() - t2},
            **({'peak_gib': torch.cuda.max_memory_allocated() / 2 ** 30} if model.device.type == 'cuda' else {})}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--data', type=Path, required=True, help='LongBench data/ directory (*.jsonl)')
    parser.add_argument('--config', type=Path, required=True, help='directory with dataset2prompt.json, dataset2maxlen.json')
    parser.add_argument('--calib', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--tasks', nargs='*', default=list(TASKS))
    parser.add_argument('--budgets', type=int, nargs='*', default=list(BUDGETS))
    parser.add_argument('--limit-per-task', type=int, default=0)
    parser.add_argument('--offset', type=int, default=0)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--dtype', default='bfloat16')
    parser.add_argument('--max-new-tokens', type=int, default=0, help='CPU tests only')
    args = parser.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, dtype=getattr(torch, args.dtype),
                                                 attn_implementation='sdpa').to(args.device).eval()
    prompts = json.loads((args.config / 'dataset2prompt.json').read_text())
    maxlen = json.loads((args.config / 'dataset2maxlen.json').read_text())
    tables = json.loads(args.calib.read_text()) if args.calib.exists() else \
        calibrate(model, tok, args.data, prompts, args.calib)
    done = {(json.loads(l)['task'], json.loads(l)['index']) for l in args.out.open()} if args.out.exists() else set()
    with args.out.open('a') as handle:
        for task in args.tasks:
            rows = [json.loads(line) for line in open(args.data / f'{task}.jsonl')]
            stop = args.offset + args.limit_per_task if args.limit_per_task else len(rows)
            for index in range(args.offset, min(stop, len(rows))):
                if (task, index) in done:
                    continue
                rec = run_example(model, tok, task, rows[index], prompts, maxlen, tables, args.budgets,
                                  args.max_new_tokens)
                handle.write(json.dumps({'index': index, **rec}, default=float) + '\n')
                handle.flush()
                print(json.dumps({'done': f'{task}:{index}'}), flush=True)


if __name__ == '__main__':
    sys.exit(main())
