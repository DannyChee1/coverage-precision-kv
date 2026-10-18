import argparse
import json
from pathlib import Path
import re
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).parent))
import mix
from mix import N_SINK, layer_budget_bits
import improve
import keys_ablation as ka
from stage2 import KEPT, entry_bits_kv, quantize_parts
from stagec import load_docs, parse, prompt

BUDGETS = {'1.5625': 0.015625, '3.125': 0.03125}
SPLITS = {'stack4pc': (4, 4, 'stack4_pc'), 'k4v2pc': (4, 2, 'k4v2_pc'), 'k3v2pc': (3, 2, 'k3_pc'),
          'rk3v2': (3, 2, 'rk3v2')}
COMPRESSED = tuple(f'{s}@{b}' for b in BUDGETS for s in SPLITS)
CAP = {'quality': 8, 'longhealth': 256}
LETTERS = 'ABCDE'


def arms(benchmark):
    return (('full',) if benchmark == 'longhealth' else ()) + COMPRESSED


def n_options(benchmark):
    return 4 if benchmark == 'quality' else 5


def _patterns(n):
    x = LETTERS[:n]
    return (re.compile(rf'\s*\(?([{x}])(?:[.):]|[ \t]*\n)'),  # rule 1, delimited
            re.compile(rf'\s*\(?([{x}])\s*$'),  # rule 1 at the end of the answer
            re.compile(rf'\s*\(?(?:[{x}][ \t]*)?$'),  # rule 1 still possible
            re.compile(rf'(?i:correct\s+(?:answer|option|choice)|\banswer)\s*(?:(?i:is|would be|should be))?'
                       rf'\s*[:\-]?\s*\(?([{x}])\b'))  # rule 2


def letter(text, n):
    lead, end, _, statement = _patterns(n)
    t = text.replace('*', '')
    m = lead.match(t) or end.match(t) or statement.search(t)
    return LETTERS.index(m.group(1)) if m else None


def resolved(text, n):
    lead, _, open_, statement = _patterns(n)
    t = text.replace('*', '')
    if lead.match(t):
        return True
    if open_.fullmatch(t):
        return False
    m = statement.search(t)
    return bool(m) and m.end() < len(t) and not (t[m.end()].isalnum() or t[m.end()] == '_')


def spec(arm):
    split, budget = arm.split('@')
    return split, BUDGETS[budget]


def allocate(scores, split, head_dim, budget):
    if split == 'rk3v2':
        return improve.allocate(scores, 'rk3v2', head_dim, budget)
    kbits, vbits, _ = SPLITS[split]
    return ka.allocate(scores, ka.entry_bits(kbits, vbits, head_dim), ka.reserved_overhead('perchannel', head_dim),
                       head_dim, budget)


def plan_bits(plan, split, head_dim):
    if split == 'rk3v2':
        return improve.plan_bits(plan, 'rk3v2', head_dim)
    return ka.plan_bits(plan, SPLITS[split][2], head_dim)


def build_rows(cache, scores, row_arms):
    n_layers, n_heads, head_dim = len(cache.layers), cache.layers[0].keys.shape[1], cache.layers[0].keys.shape[-1]
    n_tokens = cache.layers[0].keys.shape[2]
    plans = {}
    for arm in (a for a in row_arms if a != 'full'):
        split, budget = spec(arm)
        plans[arm] = [allocate(scores[i], split, head_dim, budget) for i in range(n_layers)]
    out = {'bits': {}, 'budget_bits': {}, 'over_budget_layers': {}, 'kept': {}, 'key_error': {},
           'full_bits': n_layers * n_heads * n_tokens * entry_bits_kv(16, 16, head_dim)}
    for arm, layer_plans in plans.items():
        split, budget = spec(arm)
        layer_budget = layer_budget_bits(n_heads, n_tokens, head_dim, budget)
        layer_bits = [plan_bits(p, split, head_dim) for p in layer_plans]
        out['bits'][arm] = sum(layer_bits)
        out['budget_bits'][arm] = n_layers * layer_budget
        out['over_budget_layers'][arm] = sum(b > layer_budget for b in layer_bits)
        out['kept'][arm] = int(sum((p == KEPT).sum().item() for p in layer_plans))
    arm_cache = mix.batched_cache(cache, len(row_arms))
    masks = [[] for _ in range(n_layers)]
    rot = improve.rotation(head_dim, scores.device)
    errors = {arm: [] for arm in plans}
    for layer_idx in range(n_layers):
        keys, values = cache.layers[layer_idx].keys[0], cache.layers[layer_idx].values[0]
        vq = quantize_parts(keys, values, (4,), (2, 4))[1]
        rk3 = None
        for row_idx, arm in enumerate(row_arms):
            if arm == 'full':
                continue
            split, _ = spec(arm)
            kbits, vbits, _ = SPLITS[split]
            plan = plans[arm][layer_idx]
            sel = plan == KEPT
            if split == 'rk3v2':
                rk3 = improve.centered_rotated(keys, 3, rot).to(keys.dtype) if rk3 is None else rk3
                kq = rk3
            else:
                kq = ka.quantized_keys(keys, 'perchannel', kbits, rot, sel)
            arm_cache.layers[layer_idx].keys[row_idx][sel] = kq[sel]
            arm_cache.layers[layer_idx].values[row_idx][sel] = vq[vbits][sel]
            errors[arm].append(improve.rel(kq[sel], keys[sel]))
            heads, seqs = torch.nonzero(plan == 0, as_tuple=True)
            if heads.numel():
                masks[layer_idx].append((torch.full_like(heads, row_idx), heads, seqs))
    out['key_error'] = {arm: sum(e) / len(e) for arm, e in errors.items()}
    return arm_cache, masks, out


def run_document(pipe, doc, capture_cls, cap=None):
    from transformers import DynamicCache
    model = pipe.model
    bench, row_arms = doc['benchmark'], arms(doc['benchmark'])
    n, cap = n_options(bench), cap or CAP[bench]
    inputs = pipe.preprocess(doc['context'], [prompt(q) for q in doc['questions']], '',
                             max_context_length=pipe.tokenizer.model_max_length)
    context_ids = inputs['context_ids'].to(model.device)
    n_tokens = context_ids.shape[1]
    cache = DynamicCache()
    press = capture_cls(compression_ratio=0.5, kvzip_plus_normalization=True, n_sink=N_SINK)
    t0 = time.time()
    with torch.no_grad():
        with press(model):
            model.model(input_ids=context_ids, past_key_values=cache)
    scores = press.captured[:, 0]  # L, H, T
    score_seconds = time.time() - t0
    t1 = time.time()
    arm_cache, masks, out = build_rows(cache, scores, row_arms)
    del cache
    out.update({'arms': list(row_arms), 'cap': cap, 'n_tokens': n_tokens, 'n_layers': len(arm_cache.layers),
                'score_seconds': score_seconds, 'build_seconds': time.time() - t1, 'questions': []})
    mix.set_masks(model, masks)
    t2 = time.time()
    try:
        for position, (question, ids) in enumerate(zip(doc['questions'], inputs['questions_ids'])):
            t = time.time()
            info = {}
            with torch.no_grad():
                answers = mix.batched_generate(model, pipe.tokenizer, ids.to(model.device), arm_cache, n_tokens,
                                                 cap, resolved=lambda text: resolved(text, n), info=info)
            arm_cache.crop(n_tokens)
            choice = {arm: letter(a, n) for arm, a in zip(row_arms, answers)}
            out['questions'].append({
                'qid': question['qid'], 'position': position, 'gold': question['gold'],
                'difficult': question['difficult'], 'answers': dict(zip(row_arms, answers)), 'choice': choice,
                'correct': {arm: c in question['gold'] for arm, c in choice.items()},
                'old_rule': {arm: parse(a, n) for arm, a in zip(row_arms, answers)},
                'stopped': dict(zip(row_arms, info['stopped'])), 'steps': info['steps'], 'seconds': time.time() - t})
    finally:
        mix.clear_masks(model)
    out['decode_seconds'] = time.time() - t2
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--docs', type=Path, nargs='+', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--part', type=int, default=0)
    parser.add_argument('--parts', type=int, default=1)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--dtype', default='bfloat16')
    parser.add_argument('--max-questions', type=int, default=0, help='CPU tests only')
    args = parser.parse_args()
    import kvpress
    from transformers import pipeline
    attn = 'flash_attention_2' if args.device.startswith('cuda') else 'sdpa'
    pipe = pipeline('kv-press-text-generation', model=args.model_path, device=args.device,
                    model_kwargs={'dtype': getattr(torch, args.dtype), 'attn_implementation': attn})
    docs = load_docs(args.docs, args.part, args.parts)
    done = {json.loads(line)['doc_id'] for line in args.out.open()} if args.out.exists() else set()
    capture_cls = mix.capture_press_class()
    with args.out.open('a') as handle:
        for doc in docs:
            if doc['doc_id'] in done:
                continue
            if args.max_questions:
                doc['questions'] = doc['questions'][:args.max_questions]
            result = run_document(pipe, doc, capture_cls)
            handle.write(json.dumps({'doc_id': doc['doc_id'], 'benchmark': doc['benchmark'],
                                     'model': pipe.model.config.name_or_path, 'dtype': str(pipe.model.dtype),
                                     'attn_implementation': pipe.model.config._attn_implementation,
                                     **result}) + '\n')
            handle.flush()
            seconds = result['score_seconds'] + result['build_seconds'] + result['decode_seconds']
            print(json.dumps({'done': doc['doc_id'], 'questions': len(result['questions']),
                              'seconds': round(seconds, 2)}), flush=True)


if __name__ == '__main__':
    sys.exit(main())
