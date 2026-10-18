from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from answer_blind_e1 import gold_positions
import controls as kc
from mix import N_SINK, layer_budget_bits
from diagnose import answer_tokens
import qaprior as qp
import review4k as r4

B1 = '1.5625'
ROWS = ('ans', 'evm', 'rnda', 'rndb', 'ev')
STATE = {'row': None, 'tokenizer': None}
R4_EXAMPLE, R4_BUILD = r4.run_example, r4.build_rows


def arms(benchmark, budget=None):
    return [f'Z.k3v2pc@{B1}', *(f'Z.{r}@{B1}' for r in ROWS)]


def position_sets(row, tok, n_tokens):
    if row.get('answer_spans'):
        answer = gold_positions(tok, row['context'], row['answer_spans'], n_tokens)
    else:
        answer = answer_tokens(tok, row['context'], row['answer'], n_tokens)
    evidence = gold_positions(tok, row['context'], row['evidence_spans'], n_tokens)
    if answer is None or evidence is None:
        return None
    answer = {p for p in answer if N_SINK <= p < n_tokens}
    evidence = {p for p in evidence if N_SINK <= p < n_tokens}
    evm = sorted(evidence - answer)
    rng = np.random.default_rng([0, int(row['index'])])
    pool = np.array(sorted(set(range(N_SINK, n_tokens)) - evidence - answer))
    pick = lambda k: sorted(rng.choice(pool, size=min(k, len(pool)), replace=False).tolist())
    return {'ans': sorted(answer), 'evm': evm, 'rnda': pick(len(evm)), 'rndb': pick(len(answer)),
            'ev': sorted(evidence | answer)}


def build_rows(cache, scores, row_arms):
    base = ['full' if a != 'full' and r4.split_of(a)[0] in ROWS else a for a in row_arms]
    arm_cache, masks, out = R4_BUILD(cache, scores, base)
    r4.STATE['row_arms'] = row_arms
    n_layers = len(cache.layers)
    n_heads, n_tokens, head_dim = cache.layers[0].keys.shape[1:]
    sets = position_sets(STATE['row'], STATE['tokenizer'], n_tokens)
    out['e2'] = {'ok': sets is not None}
    if sets is None:
        return arm_cache, masks, out
    out['e2'].update(answer_tokens=len(sets['ans']), evidence_tokens=len(sets['ev']), protected={}, budget_share={})
    z = scores['Z']
    for index, arm in enumerate(row_arms):
        if arm == 'full' or r4.split_of(arm)[0] not in ROWS:
            continue
        name, budget = r4.split_of(arm)
        built = [qp.oracle_plan(z[li], sets[name], 'o16', True, head_dim, kc.BUDGETS[budget]) for li in range(n_layers)]
        qp.oracle_fill(arm_cache, masks, cache, index, [p for p, _ in built], 'o16')
        limit = layer_budget_bits(n_heads, n_tokens, head_dim, kc.BUDGETS[budget])
        bits = [info['bits'] for _, info in built]
        out['bits'][arm], out['budget_bits'][arm] = float(sum(bits)), n_layers * limit
        out['over_budget_layers'][arm] = sum(b > limit for b in bits)
        out['kept'][arm] = int(sum(((p == qp.TOP) | (p == qp.KEPT)).sum().item() for p, _ in built))
        out['e2']['protected'][arm] = built[0][1]['protected']
        out['e2']['budget_share'][arm] = built[0][1]['allowance_bits'] / (limit / n_heads)
    return arm_cache, masks, out


def run_example(pipe, row, capture_kvzip, capture_scorers):
    STATE.update(row=row, tokenizer=pipe.tokenizer)
    return R4_EXAMPLE(pipe, row, capture_kvzip, capture_scorers)


def configure():
    r4.configure(False, 'mse', None)
    kc.BUDGETS = {B1: kc.BUDGETS[B1]}
    kc.arms = arms
    kc.build_rows = build_rows
    kc.run_example = run_example


def main():
    configure()
    return kc.main()


if __name__ == '__main__':
    sys.exit(main())
