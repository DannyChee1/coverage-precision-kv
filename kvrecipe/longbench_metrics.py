import difflib
import re
import string
from collections import Counter

TASK_METRIC = {
    'narrativeqa': 'qa_f1', 'qasper': 'qa_f1', 'multifieldqa_en': 'qa_f1', 'hotpotqa': 'qa_f1', '2wikimqa': 'qa_f1',
    'musique': 'qa_f1', 'gov_report': 'rouge', 'qmsum': 'rouge', 'multi_news': 'rouge', 'trec': 'classification',
    'triviaqa': 'qa_f1', 'samsum': 'rouge', 'passage_retrieval_en': 'retrieval', 'passage_count': 'count',
    'lcc': 'code_sim', 'repobench-p': 'code_sim',
}
FIRST_LINE_TASKS = {'trec', 'triviaqa', 'samsum'}
PUNCTUATION = set(string.punctuation)
RECURSION_LIMIT = 1000
ROUGE_CALL_DEPTH = 9  # frames above rouge's recursive LCS backtrack in LongBench's eval.py call chain


def normalize_answer(s):
    s = s.lower()
    s = ''.join(ch for ch in s if ch not in PUNCTUATION)
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    return ' '.join(s.split())


def qa_f1(prediction, ground_truth, **_):
    pred, gold = normalize_answer(prediction).split(), normalize_answer(ground_truth).split()
    same = sum((Counter(pred) & Counter(gold)).values())
    if same == 0:
        return 0
    p, r = same / len(pred), same / len(gold)
    return 2 * p * r / (p + r)


def _lcs_words(x, y):
    table = [[0] * (len(y) + 1) for _ in range(len(x) + 1)]
    for i in range(1, len(x) + 1):
        for j in range(1, len(y) + 1):
            table[i][j] = table[i - 1][j - 1] + 1 if x[i - 1] == y[j - 1] else max(table[i - 1][j], table[i][j - 1])
    words, i, j, depth = set(), len(x), len(y), 1
    while i > 0 and j > 0:
        depth += 1
        if x[i - 1] == y[j - 1]:
            words.add(x[i - 1])
            i, j = i - 1, j - 1
        elif table[i - 1][j] > table[i][j - 1]:
            i -= 1
        else:
            j -= 1
    return words, depth


def rouge_l(prediction, ground_truth, **_):
    hyp = [' '.join(s.split()) for s in prediction.split('.') if len(s) > 0]
    ref = [' '.join(s.split()) for s in ground_truth.split('.') if len(s) > 0]
    if not hyp or not ref:
        return 0.0
    m = len({w for s in ref for w in s.split(' ')})
    n = len({w for s in hyp for w in s.split(' ')})
    union = set()
    for r in ref:
        for h in hyp:
            words, depth = _lcs_words(r.split(' '), h.split(' '))
            if depth + ROUGE_CALL_DEPTH > RECURSION_LIMIT:
                return 0.0
            union |= words
    rec, prec = len(union) / m, len(union) / n
    return 2.0 * ((prec * rec) / (prec + rec + 1e-8))


def classification(prediction, ground_truth, all_classes=None, **_):
    matches = [c for c in all_classes if c in prediction]
    for term in matches:  # mutates while iterating, as the original does
        if term in ground_truth and term != ground_truth:
            matches.remove(term)
    return 1.0 / len(matches) if ground_truth in matches else 0.0


def retrieval(prediction, ground_truth, **_):
    gold = re.findall(r'Paragraph (\d+)', ground_truth)[0]
    numbers = re.findall(r'\d+', prediction)
    return 0.0 if not numbers else sum(n == gold for n in numbers) / len(numbers)


def count(prediction, ground_truth, **_):
    numbers = re.findall(r'\d+', prediction)
    return 0.0 if not numbers else sum(n == str(ground_truth) for n in numbers) / len(numbers)


def _lev_ratio(a, b):
    prev = [0] * (len(b) + 1)
    for ca in a:
        cur = [0]
        for j, cb in enumerate(b):
            cur.append(prev[j] + 1 if ca == cb else max(prev[j + 1], cur[j]))
        prev = cur
    return 2 * prev[-1] / (len(a) + len(b))


def fuzz_ratio(a, b, variant='lev'):
    if a == b:
        return 100
    if not a or not b:
        return 0
    r = _lev_ratio(a, b) if variant == 'lev' else difflib.SequenceMatcher(None, a, b).ratio()
    return int(round(100 * r))


def code_sim(prediction, ground_truth, fuzz='lev', **_):
    line = next((ln for ln in prediction.lstrip('\n').split('\n') if '`' not in ln and '#' not in ln and '//' not in ln), '')
    return fuzz_ratio(line, ground_truth, fuzz) / 100


METRICS = {'qa_f1': qa_f1, 'rouge': rouge_l, 'classification': classification, 'retrieval': retrieval,
           'count': count, 'code_sim': code_sim}


def score_example(task, prediction, answers, all_classes=None, fuzz='lev'):
    if task in FIRST_LINE_TASKS:
        prediction = prediction.lstrip('\n').split('\n')[0]
    metric = METRICS[TASK_METRIC[task]]
    return max([0.0] + [metric(prediction, gt, all_classes=all_classes, fuzz=fuzz) for gt in answers])
