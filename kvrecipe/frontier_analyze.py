import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
from analyze import score

WEIGHTS = {'Z': 0.0, 'W10': 0.1, 'W25': 0.25, 'W50': 0.5, 'W100': 1.0}
ARMS = {k: f'{k}.k3v2pc@1.5625' for k in WEIGHTS}
SETS = {'qa_squad': 128, 'qa_nq': 200, 'qa_1': 100, 'qa_2': 100}
B, SEED, ALPHA, MARGIN = 10_000, 0, 0.05, 1.0


def ratio(num, den):
    num = [x for x in num if x is not None]
    return float(np.mean(num) / np.mean(den)) if num else None


def analyze(records):
    rng = np.random.default_rng(SEED)
    validity, cells = {}, {}
    for m, recs in records.items():
        v = {t: sum(r['task'] == t for r in recs) for t in SETS}
        v['all_rows'] = all({'full', *ARMS.values()} <= set(r['scores']) for r in recs)
        v['within_budget'] = all(not any(r['over_budget_layers'].values()) for r in recs)
        v['equal_keep'] = all(len({round(r['frontier']['keep_rate'][k], 9) for k in WEIGHTS}) == 1 for r in recs)
        v['tokenized'] = all(r['frontier']['tokenization_ok'] for r in recs)
        v['valid'] = all(v[t] == n for t, n in SETS.items()) and all(v[k] for k in ('all_rows', 'within_budget',
                                                                                     'equal_keep', 'tokenized'))
        validity[m] = v
        for t in SETS:
            rows = [r for r in recs if r['task'] == t]
            if not rows:
                continue
            acc = {k: np.array([r['scores'][a] for r in rows]) for k, a in ARMS.items()}
            acc['full'] = np.array([r['scores']['full'] for r in rows])
            idx = rng.integers(0, len(rows), size=(B, len(rows)))
            fr = [r['frontier'] for r in rows]
            gold_rows = [f for f in fr if f['gold_positions'] > 0]
            evm_rows = [f for f in fr if f['evm_positions'] > 0]
            cells[(m, t)] = {
                'n': len(rows), 'acc': acc, 'draws': {k: a[idx].mean(axis=1) for k, a in acc.items()},
                'lift': {k: ratio([f['gold_kept'][k] for f in gold_rows], [f['keep_rate'][k] for f in gold_rows])
                         for k in WEIGHTS},
                'evidence_lift': {k: ratio([f['evm_kept'][k] for f in evm_rows], [f['keep_rate'][k] for f in evm_rows])
                                  for k in WEIGHTS},
                'mass': {k: float(np.mean([f['plain_mass'][k] for f in fr])) for k in WEIGHTS}}
    keys = list(cells)
    pooled_draws = {k: np.mean([cells[c]['draws'][k] for c in keys], axis=0) for k in [*WEIGHTS, 'full']}
    pooled = {k: float(np.mean([cells[c]['acc'][k].mean() for c in keys])) for k in [*WEIGHTS, 'full']}
    tests = {}
    for k in list(WEIGHTS)[1:]:
        d = pooled_draws[k] - pooled_draws['Z']
        tests[k] = {'w': WEIGHTS[k], 'delta': pooled[k] - pooled['Z'], 'ci95': np.percentile(d, [2.5, 97.5]).tolist(),
                    'p': float(np.mean(d <= 0))}
    order = sorted(tests, key=lambda k: tests[k]['p'])
    passed = True
    for rank, k in enumerate(order):
        tests[k]['holm_threshold'] = ALPHA / (len(order) - rank)
        passed = passed and tests[k]['p'] <= tests[k]['holm_threshold']
        tests[k]['holm_pass'] = passed
    valid = all(v['valid'] for v in validity.values()) and len(validity) == 2
    for k in tests:
        tests[k]['beats_plain'] = valid and tests[k]['holm_pass'] and tests[k]['delta'] >= MARGIN
    winners = [k for k in tests if tests[k]['beats_plain']]
    winner = max(winners, key=lambda k: tests[k]['delta']) if winners else None

    def peak_share(draws):
        stack = np.stack([draws[k] for k in WEIGHTS])
        best = stack.argmax(axis=0)
        return {k: float(np.mean(best == i)) for i, k in enumerate(WEIGHTS)}

    groups = {'pooled': keys, **{m: [c for c in keys if c[0] == m] for m in records},
              **{t: [c for c in keys if c[1] == t] for t in SETS}}
    shape = {}
    for name, members in groups.items():
        draws = {k: np.mean([cells[c]['draws'][k] for c in members], axis=0) for k in WEIGHTS}
        point = {k: float(np.mean([cells[c]['acc'][k].mean() for c in members])) for k in WEIGHTS}
        shape[name] = {'accuracy': point, 'ci95': {k: np.percentile(draws[k], [2.5, 97.5]).tolist() for k in WEIGHTS},
                       'peak': max(point, key=point.get), 'peak_share': peak_share(draws)}
    placement = {'pooled': {
        'lift': {k: float(np.mean([cells[c]['lift'][k] for c in keys if cells[c]['lift'][k] is not None]))
                 for k in WEIGHTS},
        'evidence_lift': {k: float(np.mean([cells[c]['evidence_lift'][k] for c in keys
                                            if cells[c]['evidence_lift'][k] is not None])) for k in WEIGHTS},
        'mass': {k: float(np.mean([cells[c]['mass'][k] for c in keys])) for k in WEIGHTS}}}
    acc_curve = [pooled[k] for k in WEIGHTS]
    placement['spearman_accuracy_vs_lift'] = float(spearmanr(acc_curve, list(placement['pooled']['lift'].values()))[0])
    placement['spearman_accuracy_vs_mass'] = float(spearmanr(acc_curve, list(placement['pooled']['mass'].values()))[0])
    out_cells = {f'{m} {t}': {'n': c['n'], 'accuracy': {k: float(c['acc'][k].mean()) for k in [*WEIGHTS, 'full']},
                              'ci95': {k: np.percentile(c['draws'][k], [2.5, 97.5]).tolist() for k in WEIGHTS},
                              'lift': c['lift'], 'evidence_lift': c['evidence_lift'], 'mass': c['mass']}
                 for (m, t), c in cells.items()}
    return {'validity': validity, 'valid': valid, 'pooled': pooled, 'tests': tests, 'winner': winner,
            'winner_w': WEIGHTS[winner] if winner else None, 'shape': shape, 'placement': placement,
            'cells': out_cells}


def figure(result, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    surface, ink, muted, grid = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
    colors = {'qa_squad': '#2a78d6', 'qa_nq': '#eb6834', 'qa_1': '#1baf7a', 'qa_2': '#eda100'}
    markers = {'qa_squad': 'o', 'qa_nq': 's', 'qa_1': '^', 'qa_2': 'D'}
    names = {'qa_squad': 'SQuAD articles', 'qa_nq': 'NQ pages', 'qa_1': 'RULER qa_1', 'qa_2': 'RULER qa_2'}
    models = sorted({k.split()[0] for k in result['cells']})
    xs = list(WEIGHTS.values())
    rows = (('accuracy', 'Accuracy (%)'), ('lift', 'Answer lift (x chance)'), ('mass', "Share of plain KVzip+'s mass"))
    fig, axes = plt.subplots(len(rows), len(models), figsize=(5.2 * len(models), 9.2), sharex=True,
                             facecolor=surface, squeeze=False)
    for ci, m in enumerate(models):
        for ri, (key, label) in enumerate(rows):
            ax = axes[ri][ci]
            ax.set_facecolor(surface)
            ends = []
            for t in SETS:
                c = result['cells'].get(f'{m} {t}')
                if not c:
                    continue
                ys = [c[key][k] for k in WEIGHTS]
                if key == 'accuracy':
                    lo = [y - c['ci95'][k][0] for y, k in zip(ys, WEIGHTS)]
                    hi = [c['ci95'][k][1] - y for y, k in zip(ys, WEIGHTS)]
                    shift = 0.012 * (list(SETS).index(t) - 1.5)  # small offsets so the intervals don't overlap
                    ax.errorbar([x + shift for x in xs], ys, yerr=[lo, hi], color=colors[t], lw=2, marker=markers[t],
                                ms=6, capsize=0, elinewidth=1, label=names[t])
                else:
                    ax.plot(xs, ys, color=colors[t], lw=2, marker=markers[t], ms=6, label=names[t])
                ends.append((ys[-1], t))
            if ci == len(models) - 1 and ends:  # direct labels at the line ends, spread so they don't overlap
                lo, hi = ax.get_ylim()
                gap, placed = 0.06 * (hi - lo), []
                for y, t in sorted(ends):
                    placed.append((max(y, placed[-1][0] + gap) if placed else y, t))
                for y, t in placed:
                    ax.annotate(names[t], (xs[-1], y), xytext=(8, 0), textcoords='offset points', fontsize=8,
                                color=muted, va='center', annotation_clip=False)
            ax.grid(True, color=grid, lw=0.8)
            for side in ('top', 'right'):
                ax.spines[side].set_visible(False)
            for side in ('left', 'bottom'):
                ax.spines[side].set_color(grid)
            ax.tick_params(colors=muted, labelsize=8)
            if ci == 0:
                ax.set_ylabel(label, color=ink, fontsize=9)
            if ri == 0:
                ax.set_title({'llama': 'Llama-3.1-8B', 'qwen3': 'Qwen3-8B'}.get(m, m), color=ink, fontsize=10)
            if ri == len(rows) - 1:
                ax.set_xlabel('Weight w on the content-weighted score', color=ink, fontsize=9)
            ax.set_xticks(xs)
            ax.set_xticklabels(['0', '0.1', '0.25', '0.5', '1'])
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(SETS), frameon=False, fontsize=8, labelcolor=ink)
    fig.suptitle('Answer-context frontier at 1.5625%: accuracy (95% CI), answer lift and context mass by blend weight',
                 color=ink, fontsize=10)
    fig.tight_layout(rect=(0, 0.03, 0.97, 0.97))
    fig.savefig(path, dpi=160, facecolor=surface)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--records', nargs=2, action='append', metavar=('MODEL', 'FILE'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--figure', type=Path)
    args = parser.parse_args()
    records = {m: [json.loads(line) for line in open(f)] for m, f in args.records}
    result = analyze(records)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + '\n')
    if args.figure:
        figure(result, args.figure)
    print(json.dumps({'valid': result['valid'], 'winner': result['winner'], 'pooled': result['pooled'],
                      'tests': {k: {x: (round(y, 4) if isinstance(y, float) else y) for x, y in t.items()}
                                for k, t in result['tests'].items()},
                      'peak': result['shape']['pooled']['peak'],
                      'peak_share': result['shape']['pooled']['peak_share']}, indent=1))


if __name__ == '__main__':
    main()
