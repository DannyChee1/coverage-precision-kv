import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'kvrecipe'))
import qaprior as qp

SQUAD = Path('external/RULER/scripts/data/synthetic/json/squad.json')
TOKENIZERS = {'llama': ('meta-llama/Llama-3.1-8B-Instruct', '0e9e39f249a16976918f6564b8830bc894c89659'),
              'qwen3': ('Qwen/Qwen3-8B', 'b968826d9c46dd6066d109eabc6255188de91218')}
HEADER = ('Answer the question based on the given documents. Only give me the answer and do not output any other '
          'words.\n\nThe following are given documents.\n\nDocument 1:\n')
QUESTION = ('Answer the question based on the given documents. Only give me the answer and do not output any other '
            'words.\n\nQuestion: {q}')
LIMIT = 3968  # prompt tokens for the context, leaving room for the question and a 32-token answer within 4,096
SENTENCE = re.compile(r'[^.!?]*(?:[.!?]+["\')\]]*|$)')


def sentence_span(text, start, end):
    for m in SENTENCE.finditer(text):
        if m.start() <= start < max(m.end(), m.start() + 1) and m.end() > m.start():
            return m.start(), max(m.end(), end)
    return start, end


def fits(tokenizers, context):
    return all(len(tok(qp.prompt_text(tok, context), add_special_tokens=False)['input_ids']) <= LIMIT
               for tok in tokenizers.values())


def squad_docs(tokenizers):
    data = json.load(open(SQUAD))['data']
    docs = []
    for ai, article in enumerate(data):
        title = article['title'].replace('_', ' ')
        paras, i = article['paragraphs'], 0
        while i < len(paras):
            j = i + 1
            while j < len(paras) and fits(tokenizers, HEADER + title + '\n'
                                          + '\n'.join(p['context'] for p in paras[i:j + 1])):
                j += 1
            body, questions, at = title + '\n', [], len(HEADER) + len(title) + 1
            for p in paras[i:j]:
                for q in p['qas']:
                    if q.get('is_impossible') or not q['answers']:
                        continue
                    spans = sorted({(at + a['answer_start'], at + a['answer_start'] + len(a['text']))
                                    for a in q['answers']})
                    first = q['answers'][0]
                    ev = sentence_span(p['context'], first['answer_start'], first['answer_start'] + len(first['text']))
                    questions.append({'qid': q['id'], 'question': QUESTION.format(q=q['question'].strip()),
                                      'answers': sorted({a['text'] for a in q['answers']}), 'spans': spans,
                                      'evidence': [at + ev[0], at + ev[1]]})
                body += p['context'] + '\n'
                at += len(p['context']) + 1
            context = HEADER + body.rstrip('\n')
            if not fits(tokenizers, context):
                raise SystemExit(f'squad article {ai} paragraph {i} does not fit alone')
            for q in questions:
                for a, b in q['spans']:
                    assert context[a:b] in q['answers'], (q['qid'], context[a:b])
            docs.append({'id': f'squad-{ai:02d}-{i:03d}', 'source': 'squad', 'context': context,
                         'questions': questions})
            i = j
    return docs


def nq_docs(path, tokenizers, count):
    import pyarrow.parquet as pq
    table = pq.ParquetFile(path)
    docs = []
    for group in range(table.num_row_groups):
        for ex in table.read_row_group(group).to_pylist():
            if len(docs) == count:
                return docs
            tokens = ex['document']['tokens']
            words, starts, at = [], {}, 0
            title = ex['document']['title']
            prefix = HEADER + title + '\n'
            for k, (w, html) in enumerate(zip(tokens['token'], tokens['is_html'])):
                if html:
                    continue
                starts[k] = len(prefix) + at
                words.append(w)
                at += len(w) + 1
            spans, answers = set(), set()
            for sa in ex['annotations']['short_answers']:
                for s, e, t in zip(sa['start_token'], sa['end_token'], sa['text']):
                    inside = [k for k in range(s, e) if k in starts]
                    if inside:
                        a = starts[inside[0]]
                        last = inside[-1]
                        spans.add((a, starts[last] + len(tokens['token'][last])))
                        answers.add(t)
            if not spans:
                continue
            lo, hi = 1, len(words)  # the most words whose context fits
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if fits(tokenizers, prefix + ' '.join(words[:mid])):
                    lo = mid
                else:
                    hi = mid - 1
            context = prefix + ' '.join(words[:lo])
            spans = sorted((a, b) for a, b in spans if b <= len(context))
            if not spans:
                continue
            body = context[len(prefix):]
            ev = sentence_span(body, spans[0][0] - len(prefix), spans[0][1] - len(prefix))
            docs.append({'id': f"nq-{ex['id']}", 'source': 'nq', 'context': context,
                         'questions': [{'qid': ex['id'],
                                        'question': QUESTION.format(q=ex['question']['text'].strip() + '?'),
                                        'answers': sorted(answers), 'spans': spans,
                                        'evidence': [len(prefix) + ev[0], len(prefix) + ev[1]]}]})
    return docs


def main():
    from transformers import AutoTokenizer
    parser = argparse.ArgumentParser()
    parser.add_argument('--nq', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--nq-count', type=int, default=200)
    args = parser.parse_args()
    tokenizers = {m: AutoTokenizer.from_pretrained(name, revision=rev) for m, (name, rev) in TOKENIZERS.items()}
    sets = {'squad': squad_docs(tokenizers), 'nq': nq_docs(args.nq, tokenizers, args.nq_count)}
    args.out.mkdir(parents=True, exist_ok=True)
    sha = lambda b: hashlib.sha256(b).hexdigest()
    manifest = {'limit_prompt_tokens': LIMIT, 'squad_source_sha256': sha(SQUAD.read_bytes()),
                'nq_source': str(args.nq), 'nq_source_sha256': sha(args.nq.read_bytes()), 'sets': {}}
    for name, docs in sets.items():
        path = args.out / f'{name}.jsonl'
        path.write_text(''.join(json.dumps(d) + '\n' for d in docs))
        manifest['sets'][name] = {'documents': len(docs), 'questions': sum(len(d['questions']) for d in docs),
                                  'file_sha256': sha(path.read_bytes()),
                                  'ids': [[d['id'], sha(d['context'].encode())[:16], [q['qid'] for q in d['questions']]]
                                          for d in docs]}
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
    print({k: (v['documents'], v['questions']) for k, v in manifest['sets'].items()})


if __name__ == '__main__':
    main()
