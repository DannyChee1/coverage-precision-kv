import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).parent))
import mix
import controls as kc
from qwen14b import hidden_scores
from review4k import peak_gib

SPLITS = ('evict16', 'stack4pc', 'k3v2pc')
ROWS = ['full', *[f'Z.{s}@{b}' for b in kc.BUDGETS for s in SPLITS]]
MAX_LENGTH = 16000


def split_prompt(tokenizer, template, obj, max_length=MAX_LENGTH):
    tail = template.split('{context}')[1]
    prompt, suffix = template.format(**obj), tail.format(**obj)
    ids = tokenizer(prompt, truncation=False, add_special_tokens=False).input_ids
    truncated = len(ids) > max_length
    if truncated:
        half = max_length // 2
        prompt = (tokenizer.decode(ids[:half], skip_special_tokens=True)
                  + tokenizer.decode(ids[-half:], skip_special_tokens=True))
    if not prompt.endswith(suffix):
        raise ValueError('the truncated prompt no longer ends with the question part')
    return prompt[:len(prompt) - len(suffix)], suffix, len(ids), truncated


def run_example(pipe, task, obj, prompts, max_new, capture_kvzip):
    model, tok = pipe.model, pipe.tokenizer
    context, question, prompt_tokens, truncated = split_prompt(tok, prompts[task], obj)
    inputs = pipe.preprocess(context, [question], '', max_context_length=tok.model_max_length)
    context_ids = inputs['context_ids'].to(model.device)
    question_ids = inputs['questions_ids'][0].to(model.device)
    n_tokens = context_ids.shape[1]
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    cache, scores = hidden_scores(pipe, context_ids, question_ids, capture_kvzip, None, False)
    t1 = time.time()
    arm_cache, masks, out = kc.build_rows(cache, scores, ROWS)
    del cache
    out.update(n_tokens=n_tokens, prompt_tokens=prompt_tokens, truncated=truncated, score_seconds=t1 - t0,
               build_seconds=time.time() - t1)
    mix.set_masks(model, masks)
    t2 = time.time()
    try:
        with torch.no_grad():
            answers = mix.batched_generate(model, tok, question_ids, arm_cache, n_tokens, max_new)
    finally:
        mix.clear_masks(model)
    out['decode_seconds'] = time.time() - t2
    out['answers'] = dict(zip(ROWS, answers))
    out['out_tokens'] = {a: len(tok(text, add_special_tokens=False).input_ids) for a, text in zip(ROWS, answers)}
    out['peak_gib'] = peak_gib()
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True, help='LongBench data/ with the task files')
    parser.add_argument('--config', type=Path, required=True,
                        help='directory with dataset2prompt.json and dataset2maxlen.json')
    parser.add_argument('--task', choices=('gov_report', 'qmsum'), required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--dtype', default='bfloat16')
    parser.add_argument('--max-new-tokens', type=int, default=0, help='CPU tests only')
    parser.add_argument('--max-items', type=int, default=0, help='CPU tests only')
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())['tasks'][args.task]
    raw = (args.data / f'{args.task}.jsonl').read_bytes()
    if hashlib.sha256(raw).hexdigest() != plan['sha256']:
        raise SystemExit(f'{args.task}.jsonl does not match the plan hash')
    rows = {r['_id']: r for r in map(json.loads, raw.decode().splitlines())}
    ids = plan['smoke_ids' if args.smoke else 'test_ids'][:args.max_items or None]
    prompts = json.loads((args.config / 'dataset2prompt.json').read_text())
    max_new = args.max_new_tokens or json.loads((args.config / 'dataset2maxlen.json').read_text())[args.task]
    import kvpress
    from transformers import pipeline
    attn = 'flash_attention_2' if args.device.startswith('cuda') else 'sdpa'
    pipe = pipeline('kv-press-text-generation', model=args.model_path, device=args.device,
                    model_kwargs={'dtype': getattr(torch, args.dtype), 'attn_implementation': attn})
    capture_kvzip = mix.capture_press_class()
    done = {json.loads(line)['id'] for line in args.out.open()} if args.out.exists() else set()
    with args.out.open('a') as handle:
        for i in ids:
            if i in done:
                continue
            result = run_example(pipe, args.task, rows[i], prompts, max_new, capture_kvzip)
            handle.write(json.dumps({'benchmark': 'longbench', 'task': args.task, 'id': i, 'gold': rows[i]['answers'],
                                     'model': pipe.model.config.name_or_path, 'dtype': str(pipe.model.dtype),
                                     'attn_implementation': pipe.model.config._attn_implementation,
                                     **result}) + '\n')
            handle.flush()
            seconds = result['score_seconds'] + result['build_seconds'] + result['decode_seconds']
            print(json.dumps({'done': i, 'seconds': round(seconds, 2)}), flush=True)


if __name__ == '__main__':
    sys.exit(main())
