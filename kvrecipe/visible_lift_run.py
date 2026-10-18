import argparse
import json
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).parent))
from answer_blind_e1 import BUDGETS, SPLIT, gold_positions, shares
import mix
from mix import N_SINK
import controls as kc
from diagnose import answer_tokens

STATE = {'doc': None, 'z': None}


def percentile(score, gold):
    n = score.shape[-1]
    rank = score.float().argsort(-1).argsort(-1).float() / max(n - 1, 1)
    return 100 * float(rank[..., gold].mean())


def run_row(pipe, row, capture_kvzip, capture_scorers):
    from transformers import DynamicCache
    start = time.time()
    model, tok = pipe.model, pipe.tokenizer
    head_dim = getattr(model.config, 'head_dim', None) or model.config.hidden_size // model.config.num_attention_heads
    inputs = pipe.preprocess(row['context'], [row['question']], row['answer_prefix'],
                             max_context_length=tok.model_max_length)
    ids = inputs['context_ids'].to(model.device)
    q_ids = inputs['questions_ids'][0].to(model.device)
    n = ids.shape[1]
    if row.get('answer_spans'):
        gold = gold_positions(tok, row['context'], row['answer_spans'], n)
    else:
        gold = [p for p in (answer_tokens(tok, row['context'], row['answer'], n) or []) if p >= N_SINK]
    out = {'n_tokens': n, 'question_tokens': int(q_ids.shape[1]), 'gold_positions': len(gold or [])}
    if not gold:
        return out | {'seconds': time.time() - start}
    key = row.get('doc_id') or row['index']
    if STATE['doc'] != key:  # the hidden cache depends only on the context
        press = capture_kvzip(compression_ratio=0.5, kvzip_plus_normalization=True, n_sink=N_SINK)
        with torch.no_grad(), press(model):
            model.model(input_ids=ids, past_key_values=DynamicCache())
        STATE.update(doc=key, z=press.captured[:, 0])
    both = torch.cat([ids, q_ids], dim=1)
    press = capture_kvzip(compression_ratio=0.5, kvzip_plus_normalization=True, n_sink=N_SINK)
    with torch.no_grad(), press(model):
        model.model(input_ids=both, past_key_values=DynamicCache())
    zv = press.captured[:, 0, :, :n]
    scorer = capture_scorers(compression_ratio=0.5)
    with torch.no_grad(), scorer(model):
        model.model(input_ids=both, past_key_values=DynamicCache())
    snv = torch.stack([scorer.captured[i]['SN'][..., :n] for i in range(len(scorer.captured))])
    scores = {'Z': STATE['z'], 'Zv': zv, 'SNv': snv}
    out.update(gold_kept={}, keep_rate={}, percentile={k: percentile(scores[k], gold) for k in ('Z', 'Zv')})
    for b, f in BUDGETS.items():
        for name, s in scores.items():
            plans = [kc.allocate(s[li].float(), SPLIT, head_dim, f) for li in range(s.shape[0])]
            out['gold_kept'][f'{name}@{b}'], out['keep_rate'][f'{name}@{b}'] = shares(plans, gold)
    return out | {'seconds': time.time() - start}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--rows', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--max-rows', type=int, default=0, help='CPU tests only')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--dtype', default='bfloat16')
    args = parser.parse_args()
    import kvpress
    from transformers import pipeline
    attn = 'flash_attention_2' if args.device.startswith('cuda') else 'sdpa'
    pipe = pipeline('kv-press-text-generation', model=args.model_path, device=args.device,
                    model_kwargs={'dtype': getattr(torch, args.dtype), 'attn_implementation': attn})
    capture_kvzip, capture_scorers = mix.capture_press_class(), kc.capture_scorers_class()
    rows = [json.loads(line) for line in open(args.rows)][:args.max_rows or None]
    done = {json.loads(line)['index'] for line in args.out.open()} if args.out.exists() else set()
    with args.out.open('a') as handle:
        for row in rows:
            if row['index'] in done:
                continue
            result = run_row(pipe, row, capture_kvzip, capture_scorers)
            handle.write(json.dumps({'index': row['index'], 'task': row['task'], 'doc_id': row.get('doc_id'),
                                     'model': pipe.model.config.name_or_path, **result}) + '\n')
            handle.flush()
            print(json.dumps({'done': row['index'], 'seconds': round(result['seconds'], 2)}), flush=True)


if __name__ == '__main__':
    sys.exit(main())
