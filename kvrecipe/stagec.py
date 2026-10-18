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
from stage2 import KEPT, allocate_uniform, entry_bits_kv, plan_bits_kv, quantize_parts

BUDGETS = {'1.5625': 0.015625, '3.125': 0.03125}
SPLITS = {'stack4': (4, 4), 'k4v2': (4, 2), 'rk3v2': (3, 2)}  # key bits, value bits); rk3v2 keys centered-rotated
ARMS = ('full', *(f'{s}@{b}' for b in BUDGETS for s in SPLITS))
LETTERS = 'ABCDE'
INSTRUCTION = 'Answer with the letter of the correct option.'
MAX_NEW_TOKENS = 8


def prompt(question):
    options = '\n'.join(f'{LETTERS[i]}. {o}' for i, o in enumerate(question['options']))
    return f"\n\nQuestion: {question['question']}\n\n{options}\n\n{INSTRUCTION}"


def parse(text, n_options):
    found = re.search(rf'\b([{LETTERS[:n_options]}])\b', text)
    return LETTERS.index(found.group(1)) if found else None


def load_docs(paths, part=0, parts=1):
    docs = [json.loads(line) for path in paths for line in open(path)]
    return docs[part::parts]


def arm_spec(arm):
    split, budget = arm.split('@')
    return split, BUDGETS[budget]


def allocate(scores, split, head_dim, budget):
    if split == 'rk3v2':
        return improve.allocate(scores, split, head_dim, budget)
    return allocate_uniform(scores, *SPLITS[split], head_dim, budget)


def plan_bits(plan, split, head_dim):
    if split == 'rk3v2':
        return improve.plan_bits(plan, split, head_dim)
    return plan_bits_kv(plan, *SPLITS[split], head_dim)


def build_rows(cache, scores):
    n_layers, n_heads, head_dim = len(cache.layers), cache.layers[0].keys.shape[1], cache.layers[0].keys.shape[-1]
    n_tokens = cache.layers[0].keys.shape[2]
    plans = {}
    for arm in ARMS[1:]:
        split, budget = arm_spec(arm)
        plans[arm] = [allocate(scores[i], split, head_dim, budget) for i in range(n_layers)]
    out = {'bits': {}, 'budget_bits': {}, 'over_budget_layers': {}, 'kept': {},
           'full_bits': n_layers * n_heads * n_tokens * entry_bits_kv(16, 16, head_dim)}
    for arm in ARMS[1:]:
        split, budget = arm_spec(arm)
        layer_budget = layer_budget_bits(n_heads, n_tokens, head_dim, budget)
        layer_bits = [plan_bits(p, split, head_dim) for p in plans[arm]]
        out['bits'][arm] = sum(layer_bits)
        out['budget_bits'][arm] = n_layers * layer_budget
        out['over_budget_layers'][arm] = sum(b > layer_budget for b in layer_bits)
        out['kept'][arm] = int(sum((p == KEPT).sum().item() for p in plans[arm]))
    arm_cache = mix.batched_cache(cache, len(ARMS))
    masks = [[] for _ in range(n_layers)]
    rot = improve.rotation(head_dim, scores.device)
    for layer_idx in range(n_layers):
        keys, values = cache.layers[layer_idx].keys[0], cache.layers[layer_idx].values[0]
        kq, vq = quantize_parts(keys, values, (4,), (2, 4))
        parts = {'stack4': (kq[4], vq[4]), 'k4v2': (kq[4], vq[2]),
                 'rk3v2': (improve.centered_rotated(keys, 3, rot).to(keys.dtype), vq[2])}
        for row_idx, arm in enumerate(ARMS):
            if arm == 'full':
                continue
            k_part, v_part = parts[arm_spec(arm)[0]]
            plan = plans[arm][layer_idx]
            sel = plan == KEPT
            arm_cache.layers[layer_idx].keys[row_idx][sel] = k_part[sel]
            arm_cache.layers[layer_idx].values[row_idx][sel] = v_part[sel]
            heads, seqs = torch.nonzero(plan == 0, as_tuple=True)
            if heads.numel():
                masks[layer_idx].append((torch.full_like(heads, row_idx), heads, seqs))
    return arm_cache, masks, out


def run_document(pipe, doc, capture_cls, max_new_tokens=MAX_NEW_TOKENS):
    from transformers import DynamicCache
    model = pipe.model
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
    arm_cache, masks, out = build_rows(cache, scores)
    del cache
    out.update({'n_tokens': n_tokens, 'n_layers': len(arm_cache.layers), 'score_seconds': score_seconds,
                'build_seconds': time.time() - t1, 'questions': []})
    mix.set_masks(model, masks)
    t2 = time.time()
    try:
        for position, (question, ids) in enumerate(zip(doc['questions'], inputs['questions_ids'])):
            t = time.time()
            with torch.no_grad():
                answers = mix.batched_generate(model, pipe.tokenizer, ids.to(model.device), arm_cache, n_tokens,
                                                 max_new_tokens)
            arm_cache.crop(n_tokens)
            choice = {arm: parse(a, len(question['options'])) for arm, a in zip(ARMS, answers)}
            out['questions'].append({'qid': question['qid'], 'position': position, 'gold': question['gold'],
                                     'difficult': question['difficult'], 'answers': dict(zip(ARMS, answers)),
                                     'choice': choice,
                                     'correct': {arm: c in question['gold'] for arm, c in choice.items()},
                                     'seconds': time.time() - t})
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
