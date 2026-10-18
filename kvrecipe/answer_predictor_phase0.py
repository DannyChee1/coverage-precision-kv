import argparse
import json
from pathlib import Path
import re
import statistics
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from mix import N_SINK
from analyze import score
from diagnose import answer_tokens
import qaprior as qp

K = Path('records')
IDF = Path('results/qaprior')
TOKENIZERS = {'llama': ('meta-llama/Llama-3.1-8B-Instruct', '0e9e39f249a16976918f6564b8830bc894c89659'),
              'qwen3': ('Qwen/Qwen3-8B', 'b968826d9c46dd6066d109eabc6255188de91218')}
RESERVES = (0.001, 0.0025, 0.005)


def occurrences(gold, n_tokens):
    runs = []
    for p in sorted({p for p in gold if N_SINK <= p < n_tokens}):
        if runs and p == runs[-1][-1] + 1:
            runs[-1].append(p)
        else:
            runs.append([p])
    return runs


SQUAD = Path('external/RULER/scripts/data/synthetic/json/squad.json')
CANDIDATE = re.compile(r"(?:[A-Z][\w'.-]*(?:\s+(?:of|the|de|and|von|van)?\s*[A-Z][\w'.-]*)*)|\b\d[\d,.]*\b")


def competing_answers(rows):
    paragraphs = {}
    for d in json.loads(SQUAD.read_text())['data']:
        for p in d['paragraphs']:
            paragraphs[p['context']] = {a['text'].strip().lower() for q in p['qas'] if not q['is_impossible']
                                        for a in q['answers']}
    out = {}
    for task in ('qa_1', 'qa_2'):
        rs = [r for r in rows.values() if r['task'] == task]
        docs = [[d.strip() for d in re.split(r'(?:^|\n\n)Document \d+:\n', r['context']) if d.strip()] for r in rs]
        out[task] = {'documents_median': statistics.median(len(d) for d in docs),
                     'candidate_spans_median': statistics.median(len({m.group().lower() for m in
                                                                      CANDIDATE.finditer(r['context'])}) for r in rs)}
        if task == 'qa_1':
            answers = [len(set().union(*[paragraphs[x] for x in d if x in paragraphs] or [set()])) for d in docs]
            out[task].update(paragraphs_found=statistics.mean(sum(x in paragraphs for x in d) / len(d) for d in docs),
                             annotated_answers_median=statistics.median(answers),
                             annotated_answers_min=min(answers))
    return out


def capture(values, gold, n_top):
    order = np.lexsort((np.random.default_rng(0).random(len(values)), -np.asarray(values)))
    top = set(order[:n_top].tolist())
    gold = [p for p in gold if N_SINK <= p < len(values)]
    runs = occurrences(gold, len(values))
    return {'token_recall': sum(p in top for p in gold) / len(gold), 'answer_hit': any(all(p in top for p in r)
                                                                                    for r in runs),
            'any_hit': any(p in top for p in gold)}


def main():
    from transformers import AutoTokenizer
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    rows = {json.loads(line)['index']: json.loads(line) for line in open('data/ruler-fresh14-4096/test.jsonl')}
    out = {'label': 'EXPLORATORY', 'models': {}}
    flips_all = {'up': 0, 'down': 0, 'n': 0}
    for model, (name, revision) in TOKENIZERS.items():
        tok = AutoTokenizer.from_pretrained(name, revision=revision)
        idf = qp.load_idf(IDF / f'idf_{model}.json')
        recs = [json.loads(line) for s in (0, 1) for line in open(K / f'qaprior_{model}_test_{s}.jsonl')]
        per, bins = [], {}
        for r in recs:
            rec = r['scores']['Z.k3v2pc@1.5625']
            orc = r['scores']['Z.o16r@1.5625']
            ks = r.get('gold_entries', {}).get('kept_share', {}).get('Z.k3v2pc@1.5625')
            e = {'task': r['task'], 'n_tokens': r['n_tokens'], 'gold_positions': r.get('oracle_gold_positions', 0),
                 'recipe': rec, 'oracle': orc, 'recipe_gold_kept_share': ks}
            gold = answer_tokens(tok, rows[r['index']]['context'], rows[r['index']]['answer'], r['n_tokens']) or []
            if e['gold_positions'] > 0 and gold:
                e['answer_tokens_per_occurrence'] = statistics.median(len(o) for o in occurrences(gold, r['n_tokens']))
                for kind in ('idf', 'ne_idf'):
                    values = qp.token_prior(tok, rows[r['index']]['context'], r['n_tokens'], idf, kind)
                    for frac in RESERVES:
                        n_top = max(1, int(round(frac * r['n_tokens'])))
                        e[f'{kind}@{frac}'] = capture(values, gold, n_top)
            per.append(e)
            if ks is not None:
                b = '0' if ks == 0 else '<25%' if ks < 0.25 else '25-50%' if ks < 0.5 else '>=50%'
                bins.setdefault(b, []).append(rec)
        withgold = [e for e in per if e['gold_positions'] > 0 and 'idf@0.001' in e]
        up = sum(e['oracle'] > e['recipe'] for e in per)
        down = sum(e['oracle'] < e['recipe'] for e in per)
        flips_all['up'] += up
        flips_all['down'] += down
        flips_all['n'] += len(per)
        flip_up = [e for e in withgold if e['oracle'] > e['recipe']]
        summary = {
            'n_tokens_median': statistics.median(e['n_tokens'] for e in per),
            'gold_positions_median': statistics.median(e['gold_positions'] for e in withgold),
            'answer_tokens_per_occurrence_median': statistics.median(e['answer_tokens_per_occurrence']
                                                                     for e in withgold),
            'reserve_tokens_median': {str(f): statistics.median(max(1, round(f * e['n_tokens'])) for e in per)
                                      for f in RESERVES},
            'oracle_flips': {'up': up, 'down': down, 'n': len(per)},
            'recipe_accuracy_by_gold_kept_share': {b: {'n': len(v), 'accuracy': sum(v) / len(v)}
                                                   for b, v in sorted(bins.items())},
            'recipe_gold_kept_share_mean': statistics.mean(e['recipe_gold_kept_share'] for e in per
                                                           if e['recipe_gold_kept_share'] is not None),
            'capture': {f'{kind}@{frac}': {m: statistics.mean(float(e[f'{kind}@{frac}'][m]) for e in withgold)
                                           for m in ('token_recall', 'answer_hit', 'any_hit')}
                        for kind in ('idf', 'ne_idf') for frac in RESERVES},
            'capture_on_flip_up': {f'{kind}@{frac}': statistics.mean(float(e[f'{kind}@{frac}']['answer_hit'])
                                                                     for e in flip_up)
                                   for kind in ('idf', 'ne_idf') for frac in RESERVES},
            'flip_up_with_gold': len(flip_up)}
        out['models'][model] = {'summary': summary, 'examples': per}
        print(model, json.dumps(summary, indent=None)[:1800], flush=True)
    out['flips'] = flips_all
    out['competing_answers'] = competing_answers(rows)
    print(json.dumps(out['competing_answers']))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1) + '\n')


if __name__ == '__main__':
    main()
