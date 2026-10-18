import argparse
from collections import Counter
import json
from pathlib import Path
import re
import string

import numpy as np

TASKS = ('qasper', 'multifieldqa_en', 'trec', 'lcc')
EXPECTED = {'qasper': 200, 'multifieldqa_en': 150, 'trec': 200, 'lcc': 500}
PAPER = {'full': {'qasper': 44.82, 'multifieldqa_en': 55.46, 'trec': 73.00, 'lcc': 65.24},
         'snap64': {'qasper': 21.05, 'multifieldqa_en': 39.35, 'trec': 39.00, 'lcc': 54.74},
         'rdkv64': {'qasper': 34.47, 'multifieldqa_en': 54.16, 'trec': 59.50, 'lcc': 61.70},
         'snap128': {'qasper': 24.19, 'multifieldqa_en': 48.91, 'trec': 47.00, 'lcc': 59.14},
         'rdkv128': {'qasper': 41.34, 'multifieldqa_en': 55.55, 'trec': 65.00, 'lcc': 62.68}}
PAPER_EPS = {'k': (0.149, 0.0062, 2.2e-5), 'v': (0.313, 0.0140, 4.9e-5)}
TOL = 3.0


def normalize_answer(s):
    s = s.lower()
    s = ''.join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    return ' '.join(s.split())


def qa_f1(prediction, truth, **_):
    p, t = normalize_answer(prediction).split(), normalize_answer(truth).split()
    same = sum((Counter(p) & Counter(t)).values())
    if same == 0:
        return 0.0
    precision, recall = same / len(p), same / len(t)
    return 2 * precision * recall / (precision + recall)


def classification(prediction, truth, all_classes=(), **_):
    matches = [c for c in all_classes if c in prediction]
    for term in matches:  # LongBench's loop, removing from the list it iterates, kept as is
        if term in truth and term != truth:
            matches.remove(term)
    return 1.0 / len(matches) if truth in matches else 0.0


def lcs(a, b):
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for ca in a:
        cur = [0]
        for j, cb in enumerate(b):
            cur.append(prev[j] + 1 if ca == cb else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]


def code_sim(prediction, truth, **_):
    line = ''
    for candidate in prediction.lstrip('\n').split('\n'):
        if '`' not in candidate and '#' not in candidate and '//' not in candidate:
            line = candidate
            break
    if not line or not truth:
        return 0.0
    return round(100 * 2 * lcs(line, truth) / (len(line) + len(truth))) / 100


METRIC = {'qasper': qa_f1, 'multifieldqa_en': qa_f1, 'trec': classification, 'lcc': code_sim}


def score(task, prediction, golds, all_classes):
    if task in ('trec', 'triviaqa', 'samsum', 'lsht'):
        prediction = prediction.lstrip('\n').split('\n')[0]
    return max(METRIC[task](prediction, g, all_classes=all_classes or []) for g in golds)


def analyze(records, calib):
    by_task = {t: [r for r in records if r['task'] == t] for t in TASKS}
    rows = sorted(records[0]['scores'])
    scores = {row: {t: 100 * float(np.mean([r['scores'][row]
                                            for r in rs])) for t, rs in by_task.items() if rs}
              for row in rows}
    subset = {row: float(np.mean(list(s.values()))) for row, s in scores.items()}
    paper_subset = {row: float(np.mean(list(v.values()))) for row, v in PAPER.items()}
    gap_full = subset['rdkv64'] - subset['full']
    adv_snap = subset['rdkv64'] - subset['snap64']
    paper_gap_full = paper_subset['rdkv64'] - paper_subset['full']
    paper_adv = paper_subset['rdkv64'] - paper_subset['snap64']
    crit = {'(i) |RDKV64 - FullKV - paper| <= 3.0': abs(gap_full - paper_gap_full) <= TOL,
            '(ii) RDKV64 - SnapKV64 >= paper advantage / 2': adv_snap >= paper_adv / 2}
    complete = {t: len(by_task[t]) == EXPECTED[t] for t in TASKS}
    passed = all(crit.values()) and all(complete.values())
    eps_v, eps_k = np.array(calib['eps_v']), np.array(calib['eps_k'])
    acct = {row: {'bits_over_budget': float(np.mean([r['accounting'][row]['bits'] / r['accounting'][row]['budget']
                                                     for r in records])),
                  'kept_entries_mean': float(np.mean([r['accounting'][row]['kept'] for r in records]))}
            for row in rows if row != 'full'}
    return {'call': 'PASS' if passed else 'FAIL',
            'label': 'RDKV adapted to the question-hidden setting' if passed else 'RDKV-inspired allocator',
            'criteria': crit, 'complete': complete,
            'ours': {'subset': subset, 'per_task': scores},
            'paper': {'subset': paper_subset, 'per_task': PAPER},
            'contrasts': {'rdkv64 - full': [gap_full, paper_gap_full], 'rdkv64 - snap64': [adv_snap, paper_adv],
                          'rdkv128 - full': [subset['rdkv128'] - subset['full'],
                                             paper_subset['rdkv128'] - paper_subset['full']],
                          'rdkv128 - snap128': [subset['rdkv128'] - subset['snap128'],
                                                paper_subset['rdkv128'] - paper_subset['snap128']]},
            'pipeline_checks': {'full - paper': subset['full'] - paper_subset['full'],
                                'snap64 - paper': subset['snap64'] - paper_subset['snap64']},
            'eps_layer_mean': {'k': eps_k.mean((0, 1))[1:4].tolist(), 'v': eps_v.mean((0, 1))[1:4].tolist(),
                               'paper': PAPER_EPS},
            'accounting': acct, 'tolerance': TOL}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--records', type=Path, nargs='+', required=True)
    parser.add_argument('--calib', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    recs = {}
    for p in args.records:
        for line in open(p):
            r = json.loads(line)
            recs[(r['task'], r['index'])] = r
    result = analyze(list(recs.values()), json.loads(args.calib.read_text()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + '\n')
    print(json.dumps({k: result[k] for k in ('call', 'label', 'criteria', 'contrasts', 'pipeline_checks')}, indent=1))


if __name__ == '__main__':
    main()
