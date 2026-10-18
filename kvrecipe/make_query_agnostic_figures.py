import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).parents[1] / 'results'
THEMES = {
    'light': dict(surface='#fcfcfb', ink='#0b0b0b', secondary='#52514e', muted='#898781', grid='#e1e0d9',
                  axis='#c3c2b7', s1='#2a78d6', s2='#eb6834', s3='#1baf7a', s4='#8a5cd1', s5='#d1344a',
                  s6='#b8860b'),
    'dark': dict(surface='#1a1a19', ink='#ffffff', secondary='#c3c2b7', muted='#898781', grid='#2c2c2a',
                 axis='#383835', s1='#3987e5', s2='#d95926', s3='#199e70', s4='#9b6fe0', s5='#e04a5f',
                 s6='#d6a92b'),
}
MODELS = (('llama', 'Llama-3.1-8B'), ('qwen3', 'Qwen3-8B'))


def save(fig, out, name, theme, t):
    fig.savefig(out / f'{name}-{theme}.png', facecolor=t['surface'])
    fig.savefig(out / f'{name}-{theme}.pdf', facecolor=t['surface'], metadata={'CreationDate': None})


def style(ax, t, grid='y'):
    ax.set_facecolor(t['surface'])
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('bottom', 'left'):
        ax.spines[side].set_color(t['axis'])
    ax.tick_params(colors=t['muted'], labelcolor=t['secondary'], length=0, labelsize=8)
    ax.grid(axis=grid, color=t['grid'], linewidth=1)
    ax.set_axisbelow(True)


def label(ax, t, text, **kw):
    ax.set_xlabel(text, fontsize=8.5, color=t['secondary']) if kw.get('x', True) else ax.set_ylabel(
        text, fontsize=8.5, color=t['secondary'])


def table():
    return json.loads((ROOT / 'allocation/table.json').read_text())


def lowbit_points():
    path = ROOT / 'lowbit/results.json'
    if not path.exists():
        return {}
    res = json.loads(path.read_text())['per_model']
    out = {}
    for m, s in res.items():
        kept = s['budget']['kept_fraction']
        out[m] = [(a, 100 * kept[a], s['means'][a]) for a in ('k3v2_pc', 'k2v2_pc', 'k2v2_rk', 'k3v1_pc')]
    return out


def selectors(out, theme):
    t = THEMES[theme]
    ctrl = json.loads((ROOT / 'controls/results.json').read_text())['per_model']
    am = json.loads((ROOT / 'am-stack/results.json').read_text())['per_model']
    groups = (('KVzip+', 'Z', ('stack4pc', 'k4v2pc', 'k3v2pc'), 'k4v4  k4v2  k3v2'),
              ('KVzip+, question\nvisible', 'Zv', ('stack4pc', 'k4v2pc', 'k3v2pc'), 'k4v4  k4v2  k3v2'),
              ('KeyDiff', 'KD', ('stack4pc', 'k4v2pc', 'k3v2pc'), 'k4v4  k4v2  k3v2'),
              ('Attention\nMatching', 'AM', ('am16', 'am_stack4', 'am_k3v2'), '16-bit  k4v4  k3v2'),
              ('SnapKV', 'SN', ('evict16', 'stack4pc', 'k2v2pc'), '16-bit  k4v4  k2v2'))
    roles = (('most bits per entry', t['s2'], 's'), ('middle', t['s3'], '^'), ('fewest bits, most entries', t['s1'], 'o'))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), dpi=200, facecolor=t['surface'], sharey=True)
    for ax, (m, title) in zip(axes, MODELS):
        style(ax, t)
        acc = ctrl[m]['ruler']['accuracy']
        for gi, (name, key, arms, _) in enumerate(groups):
            ys = ([am[m]['means'][a] for a in arms] if key == 'AM' else [acc[f'{key}.{a}@1.5625'] for a in arms])
            xs = [gi - 0.22, gi, gi + 0.22]
            ax.plot(xs, ys, color=t['axis'], linewidth=1.5, zorder=1)
            for (label_, color, marker), x, y in zip(roles, xs, ys):
                ax.plot(x, y, marker, color=color, markersize=6, markeredgecolor=t['surface'], markeredgewidth=1.2,
                        zorder=2, label=label_ if gi == 0 else None)
        full = acc['full']
        ax.axhline(full, color=t['muted'], linewidth=1, linestyle='--')
        ax.text(-0.45, full + 1.2, f'full cache {full:.1f}', fontsize=7.5, color=t['secondary'])
        ax.set_xticks(range(len(groups)), [f'{g[0]}\n{g[3]}' for g in groups], fontsize=6.6)
        ax.set_ylim(0, 100)
        ax.set_title(title, loc='left', fontsize=10, color=t['ink'])
    label(axes[0], t, 'RULER 4k score', x=False)
    legend = axes[0].legend(loc='lower left', frameon=False, fontsize=7, bbox_to_anchor=(0.0, 0.04))
    for text in legend.get_texts():
        text.set_color(t['secondary'])
    fig.tight_layout()
    save(fig, out, 'fig1-selectors', theme, t)
    plt.close(fig)


def coverage(out, theme):
    t = THEMES[theme]
    rows = table()
    low = lowbit_points()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), dpi=200, facecolor=t['surface'], sharey=True)
    names = {'stack4_pc': 'k4v4', 'k4v2_pc': 'k4v2', 'k3_pc': 'k3v2', 'k3v2_pc': 'k3v2', 'k2v2_pc': 'k2v2',
             'k2v2_rk': 'k2v2 rot', 'k3v1_pc': 'k3v1'}
    for ax, (m, title) in zip(axes, MODELS):
        style(ax, t)
        keys = sorted((r for r in rows if r['run'] == 'keys' and r['model'] == m and r['arm'] in
                       ('stack4_pc', 'k4v2_pc', 'k3_pc')), key=lambda r: r['f'])
        full = next(r['score'] for r in rows if r['run'] == 'keys' and r['model'] == m and r['arm'] == 'full')
        x, y = [100 * r['f'] for r in keys], [r['score'] for r in keys]
        ax.plot(x, y, '-o', color=t['s1'], linewidth=2, markersize=6, markeredgecolor=t['surface'],
                markeredgewidth=1.5, label='key ablation (fresh-4)')
        for r in keys:
            ax.annotate(names[r['arm']], (100 * r['f'], r['score']), xytext=(-30, 4) if r['arm'] == 'k3_pc' else (4, -11),
                        textcoords='offset points', fontsize=7.5, color=t['secondary'])
        if m in low:
            pts = low[m]
            line = [p for p in pts if p[0] in ('k3v2_pc', 'k2v2_pc')]
            ax.plot([p[1] for p in line], [p[2] for p in line], '-s', color=t['s2'], linewidth=2, markersize=6,
                    markeredgecolor=t['surface'], markeredgewidth=1.5, label='low-bit run (fresh-7)')
            others = [p for p in pts if p[0] in ('k2v2_rk', 'k3v1_pc')]
            ax.plot([p[1] for p in others], [p[2] for p in others], '^', linestyle='none', color=t['s3'], markersize=7,
                    markeredgecolor=t['surface'], markeredgewidth=1.5, label='low-bit run, other 640-bit arms')
            stacked = sorted((p for p in pts if p[0] != 'k3v2_pc'), key=lambda p: p[2])
            ys = [stacked[0][2]]
            for p in stacked[1:]:
                ys.append(max(p[2], ys[-1] + 2.0))
            for (a, kx, ky), ly in zip(stacked, ys):
                ax.annotate(names[a], (kx, ky), xytext=(kx + 0.15, ly), textcoords='data', fontsize=7.5,
                            color=t['secondary'], va='center')
        ax.set_xlim(4.8, 10.2)
        ax.axhline(full, color=t['muted'], linewidth=1, linestyle='--')
        ax.text(ax.get_xlim()[0] + 0.1, full + 0.6, f'full cache {full:.1f}', fontsize=7.5, color=t['secondary'])
        ax.set_title(title, loc='left', fontsize=10, color=t['ink'])
        label(ax, t, 'entries kept (%)')
    label(axes[0], t, 'RULER 4k score', x=False)
    legend = axes[1].legend(loc='lower right', frameon=False, fontsize=7.5)
    for text in legend.get_texts():
        text.set_color(t['secondary'])
    fig.tight_layout()
    save(fig, out, 'fig2-coverage', theme, t)
    plt.close(fig)


def sweep(out, theme):
    t = THEMES[theme]
    rows = [r for r in table() if r['run'] == 'stageb']
    budgets = (1.5625, 3.125, 6.25, 12.5)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), dpi=200, facecolor=t['surface'], sharey=True)
    for ax, (m, title) in zip(axes, MODELS):
        style(ax, t)
        full = next(r['score'] for r in rows if r['model'] == m and r['arm'] == 'full')
        for arm, color, marker, name in (('stack4', t['s2'], 's', '4-bit stack'), ('k4v2', t['s1'], 'o', 'k4v2')):
            ys = [next(r['score'] for r in rows if r['model'] == m and r['arm'] == f'{arm}@{b}') for b in budgets]
            ax.plot(range(4), ys, '-' + marker, color=color, linewidth=2, markersize=6, markeredgecolor=t['surface'],
                    markeredgewidth=1.5, label=name)
            ax.annotate(f'{name} {ys[0]:.1f}', (0, ys[0]), xytext=(-8, 0), textcoords='offset points',
                        fontsize=7.5, color=t['secondary'], va='center', ha='right')
        ax.axhline(full, color=t['muted'], linewidth=1, linestyle='--')
        ax.text(2.2, full + 0.8, f'full cache {full:.1f}', fontsize=7.5, color=t['secondary'])
        ax.set_xticks(range(4), [f'{b:g}%' for b in budgets])
        ax.set_xlim(-1.1, 3.25)
        ax.set_title(title, loc='left', fontsize=10, color=t['ink'])
        label(ax, t, 'bytes kept, share of the FP16 cache')
    label(axes[0], t, 'RULER 4k score', x=False)
    legend = axes[1].legend(loc='lower right', frameon=False, fontsize=7.5)
    for text in legend.get_texts():
        text.set_color(t['secondary'])
    fig.tight_layout()
    save(fig, out, 'fig3-sweep', theme, t)
    plt.close(fig)


def headline_points(m, bench):
    review = json.loads((ROOT / 'review/result_4k.json').read_text())['per_model'][m][bench]
    ctrl = json.loads((ROOT / 'controls/results.json').read_text())['per_model'][m][bench]

    def pts(res, arm, budgets=(1.5625, 3.125)):
        share = res['budget']['share_of_full_bytes']
        return [(100 * share[f'{arm}@{b}'], res['accuracy'][f'{arm}@{b}']) for b in budgets]

    per_entry = review['budget']['share_of_full_bytes']['Z.evict16@1.5625'] / review['kept']['Z.evict16@1.5625']
    prec16 = (100 * per_entry * review['kept']['Z.prec16@1.5625'], review['accuracy']['Z.prec16@1.5625'])
    return review['accuracy']['full'], {
        'SnapKV, 16-bit eviction': pts(ctrl, 'SN.evict16'),
        'SnapKV, KIVI-2': pts(ctrl, 'SN.k2v2pc'),
        'KeyDiff, k3v2': pts(ctrl, 'KD.k3v2pc'),
        'KVzip+, 16-bit eviction': pts(review, 'Z.evict16') + [prec16],
        'KVzip+, 4-bit': pts(review, 'Z.stack4pc'),
        'KVzip+, k4v2': pts(ctrl, 'Z.k4v2pc'),
        'KVzip+, k3v2 (recipe)': pts(review, 'Z.k3v2pc'),
        'Two tiers, 4-bit over 2-bit': pts(review, 'Z.tiera'),
        'Two tiers, 4-bit over k3v2': pts(review, 'Z.tierb'),
        'TurboQuant k4v4': pts(review, 'Z.tqk4v4'),
        'TurboQuant k3v2': pts(review, 'Z.tqk3v2'),
        'TurboQuant k2v2': pts(review, 'Z.tqk2v2'),
        'tca2 allocator': pts(review, 'Z.tca'),
    }


def headline(out, theme):
    t = THEMES[theme]
    styles = {
        'SnapKV, 16-bit eviction': (t['muted'], 'x'), 'SnapKV, KIVI-2': (t['muted'], 'v'),
        'KeyDiff, k3v2': (t['s6'], 'D'), 'KVzip+, 16-bit eviction': (t['secondary'], 'o'),
        'KVzip+, 4-bit': (t['s2'], 's'), 'KVzip+, k4v2': (t['s2'], '^'), 'KVzip+, k3v2 (recipe)': (t['s1'], 'o'),
        'Two tiers, 4-bit over 2-bit': (t['s4'], '<'), 'Two tiers, 4-bit over k3v2': (t['s4'], '>'),
        'TurboQuant k4v4': (t['s3'], 's'), 'TurboQuant k3v2': (t['s3'], 'o'), 'TurboQuant k2v2': (t['s3'], 'v'),
        'tca2 allocator': (t['s5'], '*'),
    }
    names = list(styles)
    offset = {k: 10 ** (0.2 * (i / (len(names) - 1) - 0.5)) for i, k in enumerate(names)}
    budgets = (1.5625, 3.125)
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.6), dpi=200, facecolor=t['surface'], sharex=True)
    for row, (bench, name, ylim) in enumerate((('ruler', 'RULER-4K', (0, 101)), ('quality', 'QuALITY', (20, 81)))):
        for col, (m, title) in enumerate(MODELS):
            ax = axes[row][col]
            style(ax, t)
            full, series = headline_points(m, bench)
            best = [max(v[i][1] for v in series.values()) for i in range(2)]
            lo, hi = budgets[0] * 10 ** -0.11, budgets[1] * 10 ** -0.11
            ax.plot([lo, hi, hi, 7.6], [best[0], best[0], best[1], best[1]], color=t['ink'], alpha=0.13,
                    linewidth=8, solid_capstyle='round', solid_joinstyle='round', zorder=0,
                    label='best-at-size envelope: best score at this size or less')
            for k, v in series.items():
                color, marker = styles[k]
                xs = [b * offset[k] for b in budgets] + [p[0] for p in v[2:]]
                ys = [p[1] for p in v]
                if k.startswith('KVzip+, 16'):
                    ax.plot(xs, ys, '-', color=color, linewidth=1.2, zorder=1)
                ax.plot(xs, ys, marker, linestyle='none', color=color, label=k, zorder=2,
                        markersize=11 if marker == '*' else 7.5 if k.endswith('(recipe)') else 6,
                        markeredgecolor=color if marker == 'x' else t['surface'],
                        markeredgewidth=1.4 if marker == 'x' else 0.8)
            ax.axhline(full, color=t['muted'], linewidth=1, linestyle='--')
            ax.text(1.12, full + 1.2, f'full cache {full:.1f}', fontsize=7, color=t['secondary'])
            ax.set_xscale('log')
            ax.set_xlim(1.1, 7.6)
            ax.set_ylim(*ylim)
            ax.set_xticks([1.5625, 3.125, 6.25], ['1.56% (64x)', '3.13% (32x)', '6.25% (16x)'])
            ax.minorticks_off()
            ax.set_title(f'{title}, {name}', loc='left', fontsize=9.5, color=t['ink'])
            if row == 1:
                label(ax, t, 'real bytes, share of the FP16 cache (log scale)')
            if col == 0:
                label(ax, t, f'{name} score', x=False)
    handles, labels = axes[0][0].get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc='lower center', ncol=4, frameon=False, fontsize=7,
                        bbox_to_anchor=(0.5, 0.0))
    for text in legend.get_texts():
        text.set_color(t['secondary'])
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    save(fig, out, 'fig-headline', theme, t)
    plt.close(fig)


def frontier(out, theme):
    t = THEMES[theme]
    r = json.loads((ROOT / 'frontier/result.json').read_text())
    keys, ws = ('Z', 'W10', 'W25', 'W50', 'W100'), (0.0, 0.1, 0.25, 0.5, 1.0)
    fig, axes = plt.subplots(1, 4, figsize=(9, 1.95), dpi=200, facecolor=t['surface'])
    panels = (('Accuracy, 4 QA sets (95% CI)', 'accuracy'), ('Answer lift (x chance)', 'lift'),
              ('Plain-mass share, vs w = 0', 'mass'), ('Evidence lift (x chance)', 'evidence_lift'))
    lifts = [sum(c[kind][k] for c in cs) / len(cs) for cs in ([c for n, c in r['cells'].items() if n.startswith(m)]
                                                              for m, _ in MODELS)
             for kind in ('lift', 'evidence_lift') for k in keys]
    pad = 0.06 * (max(lifts) - min(lifts))
    for ax, (title, kind) in zip(axes, panels):
        style(ax, t)
        ends = []
        for (m, name), color, dx in zip(MODELS, (t['s1'], t['s2']), (-0.008, 0.008)):
            cells = [c for k, c in r['cells'].items() if k.startswith(m)]
            if kind == 'accuracy':
                y = [r['shape'][m]['accuracy'][k] for k in keys]
                lo = [y[i] - r['shape'][m]['ci95'][k][0] for i, k in enumerate(keys)]
                hi = [r['shape'][m]['ci95'][k][1] - y[i] for i, k in enumerate(keys)]
                ax.errorbar([w + dx for w in ws], y, yerr=[lo, hi], color=color, linewidth=2, marker='o',
                            markersize=4, capsize=0, elinewidth=1, label=name)
            else:
                y = [sum(c[kind][k] / (c[kind]['Z'] if kind == 'mass' else 1) for c in cells) / len(cells)
                     for k in keys]
                ax.plot(ws, y, color=color, linewidth=2, marker='o', markersize=4, label=name)
            ends.append((y[-1], name.split('-')[0]))
        if kind in ('lift', 'evidence_lift'):  # one scale for both lifts, so their slopes compare
            ax.set_ylim(min(lifts) - pad, max(lifts) + pad)
        lo, hi = ax.get_ylim()
        gap = (hi - lo) * 0.14
        (a, an), (b, bn) = sorted(ends)
        if b - a < gap:  # spread end labels that would collide, keeping both above the axis
            mid = max((a + b) / 2, lo + gap)
            a, b = mid - gap / 2, mid + gap / 2
        for v, n in ((a, an), (b, bn)):
            ax.annotate(n, (1.0, v), xytext=(6, 0), textcoords='offset points', fontsize=7, color=t['secondary'],
                        va='center')
        ax.set_xticks(ws, ['0', '.1', '.25', '.5', '1'], fontsize=7.5)
        ax.set_xlim(-0.05, 1.2)
        ax.set_title(title, loc='left', fontsize=8.5, color=t['ink'])
    fig.supxlabel('blend weight w: 0 = plain KVzip+ (P), 1 = content-weighted KVzip+ (E)', fontsize=8,
                  color=t['secondary'], y=0.13)
    handles, labels = axes[0].get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc='lower center', ncol=2, frameon=False, fontsize=7.5,
                        bbox_to_anchor=(0.5, -0.02))
    for text in legend.get_texts():
        text.set_color(t['secondary'])
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    save(fig, out, 'fig-frontier', theme, t)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=Path('docs/figures/paper'))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for theme in THEMES:
        selectors(args.out, theme)
        coverage(args.out, theme)
        sweep(args.out, theme)
        headline(args.out, theme)
        frontier(args.out, theme)
    print(f'wrote {sorted(p.name for p in args.out.glob("fig*"))}')


if __name__ == '__main__':
    main()
