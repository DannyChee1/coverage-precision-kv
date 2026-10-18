import argparse
import json
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).parent))
import mix
from mix import N_SINK, layer_budget_bits
import keys_ablation as ka
from confirm import load_rows
from improve import part_bits, rotation
from rebaseline import letter, resolved
from stage2 import KEPT, SINK, quantize_parts
from stagec import load_docs, prompt

BUDGETS = {'1.5625': 0.015625, '3.125': 0.03125}
SPLITS = {'stack4pc': (4, 4), 'k4v2pc': (4, 2), 'k3v2pc': (3, 2), 'k3v3pc': (3, 3), 'k2v4pc': (2, 4),
          'k2v2pc': (2, 2), 'evict16': (16, 16)}  # key bits, value bits); keys per-channel below 16 bits
FIXED = {'k3v3pc': 'k4v2pc', 'k2v4pc': 'k4v2pc'}  # rows that reuse another row's kept set (same entries, same bytes
GROUPS = {'Z': ('stack4pc', 'k4v2pc', 'k3v2pc', 'k3v3pc', 'k2v4pc'), 'Zv': ('stack4pc', 'k4v2pc', 'k3v2pc'),
          'KD': ('stack4pc', 'k4v2pc', 'k3v2pc'), 'SN': ('evict16', 'stack4pc', 'k2v2pc')}
QUALITY_CAP = 8
SNAP_WINDOW = 64


def arms(benchmark, budget=None):
    budgets = [budget] if budget else list(BUDGETS)
    groups = [g for g in GROUPS if benchmark == 'ruler' or g != 'Zv']
    return [f'{g}.{s}@{b}' for b in budgets for g in groups for s in GROUPS[g]]


def parse_arm(arm):
    selector, rest = arm.split('.', 1)
    split, budget = rest.split('@')
    return selector, split, BUDGETS[budget]


def overhead(split, head_dim):
    return ka.COUNTS if split == 'evict16' else ka.reserved_overhead('perchannel', head_dim)


def allocate(scores, split, head_dim, budget):
    kbits, vbits = SPLITS[split]
    return ka.allocate(scores, ka.entry_bits(kbits, vbits, head_dim), overhead(split, head_dim), head_dim, budget)


def plan_bits(plan, split, head_dim):
    kbits, vbits = SPLITS[split]
    kept, sinks = int((plan == KEPT).sum()), int((plan == SINK).sum())
    pad = 0 if split == 'evict16' else ka.padding_bits(plan, head_dim)
    return (kept * ka.entry_bits(kbits, vbits, head_dim) + sinks * 2 * part_bits(16, head_dim)
            + plan.shape[0] * ka.COUNTS + pad)


def capture_scorers_class():
    from kvpress.presses.keydiff_press import KeyDiffPress
    from kvpress.presses.scorer_press import ScorerPress
    from kvpress.presses.snapkv_press import SnapKVPress

    class CaptureScorers(ScorerPress):

        def __post_init__(self):
            super().__post_init__()
            self.keydiff, self.snapkv, self.captured = KeyDiffPress(), SnapKVPress(window_size=SNAP_WINDOW), {}

        def compress(self, module, hidden_states, keys, values, attentions, kwargs):
            layer = int(module.layer_idx)
            self.captured[layer] = {
                'KD': self.keydiff.score(module, hidden_states, keys, values, attentions, kwargs)[0].float(),
                'SN': self.snapkv.score(module, hidden_states, keys, values, attentions, kwargs)[0].float()}
            return keys, values

    return CaptureScorers


def score_selectors(pipe, context_ids, question_ids, capture_kvzip, capture_scorers, visible):
    from transformers import DynamicCache
    model, n_tokens = pipe.model, context_ids.shape[1]
    cache = DynamicCache()
    press = capture_kvzip(compression_ratio=0.5, kvzip_plus_normalization=True, n_sink=N_SINK)
    with torch.no_grad(), press(model):
        model.model(input_ids=context_ids, past_key_values=cache)
    scores = {'Z': press.captured[:, 0]}
    if visible:
        press = capture_kvzip(compression_ratio=0.5, kvzip_plus_normalization=True, n_sink=N_SINK)
        with torch.no_grad(), press(model):
            model.model(input_ids=torch.cat([context_ids, question_ids], dim=1), past_key_values=DynamicCache())
        scores['Zv'] = press.captured[:, 0, :, :n_tokens]
    scorer = capture_scorers(compression_ratio=0.5)
    with torch.no_grad(), scorer(model):
        model.model(input_ids=context_ids, past_key_values=DynamicCache())
    for name in ('KD', 'SN'):
        scores[name] = torch.stack([scorer.captured[i][name] for i in range(len(cache.layers))])
    return cache, scores


def build_rows(cache, scores, row_arms):
    n_layers, n_heads, head_dim = len(cache.layers), cache.layers[0].keys.shape[1], cache.layers[0].keys.shape[-1]
    n_tokens = cache.layers[0].keys.shape[2]
    plans = {}
    for arm in row_arms:
        if arm == 'full':
            continue
        selector, split, budget = parse_arm(arm)
        base = FIXED.get(split, split)
        key = (selector, base, budget)
        if key not in plans:
            plans[key] = [allocate(scores[selector][i], base, head_dim, budget) for i in range(n_layers)]
    out = {'bits': {}, 'budget_bits': {}, 'over_budget_layers': {}, 'kept': {},
           'full_bits': n_layers * n_heads * n_tokens * 2 * part_bits(16, head_dim)}
    for arm in (a for a in row_arms if a != 'full'):
        selector, split, budget = parse_arm(arm)
        layer_plans = plans[(selector, FIXED.get(split, split), budget)]
        limit = layer_budget_bits(n_heads, n_tokens, head_dim, budget)
        layer_bits = [plan_bits(p, split, head_dim) for p in layer_plans]
        out['bits'][arm], out['budget_bits'][arm] = sum(layer_bits), n_layers * limit
        out['over_budget_layers'][arm] = sum(b > limit for b in layer_bits)
        out['kept'][arm] = int(sum((p == KEPT).sum().item() for p in layer_plans))
    arm_cache = mix.batched_cache(cache, len(row_arms))
    masks = [[] for _ in range(n_layers)]
    rot = rotation(head_dim, cache.layers[0].keys.device)
    for layer in range(n_layers):
        keys, values = cache.layers[layer].keys[0], cache.layers[layer].values[0]
        vq = quantize_parts(keys, values, (4,), (2, 3, 4))[1]
        vq[16] = values
        quantized = {}
        for row, arm in enumerate(row_arms):
            if arm == 'full':
                continue
            selector, split, budget = parse_arm(arm)
            plan_key = (selector, FIXED.get(split, split), budget)
            plan = plans[plan_key][layer]
            kbits, vbits = SPLITS[split]
            sel = plan == KEPT
            if kbits == 16:
                kq = keys
            else:
                if (plan_key, kbits) not in quantized:
                    quantized[(plan_key, kbits)] = ka.quantized_keys(keys, 'perchannel', kbits, rot, sel)
                kq = quantized[(plan_key, kbits)]
            arm_cache.layers[layer].keys[row][sel] = kq[sel]
            arm_cache.layers[layer].values[row][sel] = vq[vbits][sel]
            heads, seqs = torch.nonzero(plan == 0, as_tuple=True)
            if heads.numel():
                masks[layer].append((torch.full_like(heads, row), heads, seqs))
    return arm_cache, masks, out


def run_example(pipe, row, capture_kvzip, capture_scorers):
    model = pipe.model
    inputs = pipe.preprocess(row['context'], [row['question']], row['answer_prefix'],
                             max_context_length=pipe.tokenizer.model_max_length)
    context_ids = inputs['context_ids'].to(model.device)
    question_ids = inputs['questions_ids'][0].to(model.device)
    n_tokens = context_ids.shape[1]
    t0 = time.time()
    cache, scores = score_selectors(pipe, context_ids, question_ids, capture_kvzip, capture_scorers, True)
    t1 = time.time()
    row_arms = ['full', *arms('ruler')]
    arm_cache, masks, out = build_rows(cache, scores, row_arms)
    del cache
    out.update(n_tokens=n_tokens, score_seconds=t1 - t0, build_seconds=time.time() - t1)
    mix.set_masks(model, masks)
    t2 = time.time()
    try:
        with torch.no_grad():
            answers = mix.batched_generate(model, pipe.tokenizer, question_ids, arm_cache, n_tokens,
                                             int(row['max_new_tokens']))
    finally:
        mix.clear_masks(model)
    out['decode_seconds'] = time.time() - t2
    out['answers'] = dict(zip(row_arms, answers))
    return out


def run_document(pipe, doc, capture_kvzip, capture_scorers, cap=QUALITY_CAP):
    model, n = pipe.model, 4
    inputs = pipe.preprocess(doc['context'], [prompt(q) for q in doc['questions']], '',
                             max_context_length=pipe.tokenizer.model_max_length)
    context_ids = inputs['context_ids'].to(model.device)
    n_tokens = context_ids.shape[1]
    t0 = time.time()
    cache, scores = score_selectors(pipe, context_ids, None, capture_kvzip, capture_scorers, False)
    questions = [{'qid': q['qid'], 'position': i, 'gold': q['gold'], 'difficult': q['difficult'], 'answers': {},
                  'choice': {}, 'correct': {}, 'stopped': {}, 'steps': {}, 'seconds': 0.0}
                 for i, q in enumerate(doc['questions'])]
    out = {'n_tokens': n_tokens, 'cap': cap, 'score_seconds': time.time() - t0, 'build_seconds': 0.0,
           'decode_seconds': 0.0, 'arms': ['full', *arms('quality')], 'bits': {}, 'budget_bits': {},
           'over_budget_layers': {}, 'kept': {}, 'questions': questions}
    for b_index, budget in enumerate(BUDGETS):
        row_arms = (['full'] if b_index == 0 else []) + arms('quality', budget)
        t1 = time.time()
        arm_cache, masks, acct = build_rows(cache, scores, row_arms)
        for key in ('bits', 'budget_bits', 'over_budget_layers', 'kept'):
            out[key].update(acct[key])
        out['full_bits'] = acct['full_bits']
        out['build_seconds'] += time.time() - t1
        mix.set_masks(model, masks)
        t2 = time.time()
        try:
            for record, question, ids in zip(out['questions'], doc['questions'], inputs['questions_ids']):
                t = time.time()
                info = {}
                with torch.no_grad():
                    answers = mix.batched_generate(model, pipe.tokenizer, ids.to(model.device), arm_cache,
                                                     n_tokens, cap, resolved=lambda text: resolved(text, n),
                                                     info=info)
                arm_cache.crop(n_tokens)
                for arm, answer, stop in zip(row_arms, answers, info['stopped']):
                    choice = letter(answer, n)
                    record['answers'][arm], record['choice'][arm] = answer, choice
                    record['correct'][arm], record['stopped'][arm] = choice in question['gold'], stop
                    record['steps'][arm] = info['steps']
                record['seconds'] += time.time() - t
        finally:
            mix.clear_masks(model)
        out['decode_seconds'] += time.time() - t2
        del arm_cache
    del cache
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--rows', type=Path)
    parser.add_argument('--docs', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--per-task', type=int, default=50)
    parser.add_argument('--offset', type=int, default=0)
    parser.add_argument('--part', type=int, default=0)
    parser.add_argument('--parts', type=int, default=1)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--dtype', default='bfloat16')
    parser.add_argument('--max-new-tokens', type=int, default=0, help='CPU tests only')
    parser.add_argument('--max-items', type=int, default=0, help='CPU tests only')
    parser.add_argument('--max-questions', type=int, default=0, help='CPU tests only')
    args = parser.parse_args()
    if (args.rows is None) == (args.docs is None):
        raise SystemExit('give exactly one of --rows (RULER) or --docs (QuALITY)')
    import kvpress
    from transformers import pipeline
    attn = 'flash_attention_2' if args.device.startswith('cuda') else 'sdpa'
    pipe = pipeline('kv-press-text-generation', model=args.model_path, device=args.device,
                    model_kwargs={'dtype': getattr(torch, args.dtype), 'attn_implementation': attn})
    capture_kvzip, capture_scorers = mix.capture_press_class(), capture_scorers_class()
    items = (load_rows(args.rows, args.per_task, args.offset) if args.rows
             else load_docs([args.docs], args.part, args.parts))
    if args.max_items:
        items = items[::max(1, len(items) // args.max_items)][:args.max_items]
    key = 'index' if args.rows else 'doc_id'
    done = {json.loads(line)[key] for line in args.out.open()} if args.out.exists() else set()
    with args.out.open('a') as handle:
        for item in items:
            if item[key] in done:
                continue
            if args.rows:
                if args.max_new_tokens:
                    item['max_new_tokens'] = min(int(item['max_new_tokens']), args.max_new_tokens)
                result = run_example(pipe, item, capture_kvzip, capture_scorers)
                record = {'index': item['index'], 'task': item['task'], 'answer': list(item['answer']),
                          'benchmark': 'ruler'}
            else:
                if args.max_questions:
                    item['questions'] = item['questions'][:args.max_questions]
                result = run_document(pipe, item, capture_kvzip, capture_scorers)
                record = {'doc_id': item['doc_id'], 'benchmark': 'quality'}
            handle.write(json.dumps({**record, 'model': pipe.model.config.name_or_path,
                                     'dtype': str(pipe.model.dtype),
                                     'attn_implementation': pipe.model.config._attn_implementation,
                                     **result}) + '\n')
            handle.flush()
            seconds = result['score_seconds'] + result['build_seconds'] + result['decode_seconds']
            print(json.dumps({'done': item[key], 'seconds': round(seconds, 2)}), flush=True)


if __name__ == '__main__':
    sys.exit(main())
