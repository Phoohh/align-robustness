#!/usr/bin/env python3
"""Validate the 42 recorded seeds and regenerate the rho curve and size table."""
import argparse
import csv
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
RHOS = (0., .0005, .001, .002, .003, .005, .01)
FRACTIONS = (100, 50, 20, 10)
SEEDS = (0, 1, 2)


def load_results(path):
    groups = {}
    with Path(path).open(newline='') as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        method, percent, seed = row['method'], int(row['percent']), int(row['seed'])
        rho = float(row['rho']) if row['rho'] else None
        clean, aa = float(row['clean']), float(row['aa'])
        if not all(math.isfinite(v) and 0 <= v <= 100 for v in (clean, aa)):
            raise ValueError('Accuracies must be finite percentages')
        if aa > clean:
            raise ValueError('Full AutoAttack accuracy cannot exceed clean accuracy')
        if seed not in SEEDS:
            raise ValueError(f'Unexpected seed: {seed}')
        key = (method, percent, rho)
        bucket = groups.setdefault(key, {})
        if seed in bucket:
            raise ValueError(f'Duplicate seed: {key}, {seed}')
        bucket[seed] = (clean, aa)
    expected = {('align', 100, rho) for rho in RHOS}
    expected |= {('align', fraction, .002) for fraction in FRACTIONS}
    expected |= {('raat', fraction, None) for fraction in FRACTIONS}
    if set(groups) != expected:
        raise ValueError(f'Unexpected/missing groups: {set(groups) ^ expected}')
    for key, seeds in groups.items():
        if set(seeds) != set(SEEDS):
            raise ValueError(f'Incomplete three-seed result: {key}')
    return groups


def summarize(groups):
    summaries = {}
    for key, seeds in groups.items():
        clean, aa = zip(*(seeds[s] for s in SEEDS))
        means = [(c + r) / 2 for c, r in zip(clean, aa)]
        geometric = [math.sqrt(c * r) for c, r in zip(clean, aa)]
        summaries[key] = {
            'clean_mean': statistics.mean(clean), 'clean_std': statistics.pstdev(clean),
            'aa_mean': statistics.mean(aa), 'aa_std': statistics.pstdev(aa),
            'mean_mean': statistics.mean(means), 'mean_std': statistics.pstdev(means),
            'G_of_mean_accuracies': math.sqrt(statistics.mean(clean) * statistics.mean(aa)),
            'per_seed_G_mean': statistics.mean(geometric),
            'per_seed_G_std': statistics.pstdev(geometric),
        }
    return summaries


def write_csv(path, rows):
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def cell(row, metric):
    return f'{row[metric + "_mean"]:.2f} ± {row[metric + "_std"]:.2f}'


def write_summaries(summaries, output):
    rho_rows = [dict(rho=rho, **summaries['align', 100, rho]) for rho in RHOS]
    size_rows = [dict(method=method, percent=p, images=p * 500,
                      **summaries[method, p, .002 if method == 'align' else None])
                 for p in FRACTIONS for method in ('raat', 'align')]
    write_csv(output / 'impact_of_rho.csv', rho_rows)
    write_csv(output / 'impact_of_dataset_size.csv', size_rows)
    header = ('Training data', 'Images', 'Method', 'Clean', 'Robust (AA)', 'Mean', 'G')
    data = []
    for p in FRACTIONS:
        for method, label, rho in (('raat', 'RAAT', None), ('align', 'Ours', .002)):
            row = summaries[method, p, rho]
            data.append([f'{p}%', f'{p * 500:,}', label, cell(row, 'clean'), cell(row, 'aa'),
                         cell(row, 'mean'), f'{row["G_of_mean_accuracies"]:.2f}'])
    md = ['| ' + ' | '.join(header) + ' |', '|' + '|'.join(['---'] * len(header)) + '|']
    md += ['| ' + ' | '.join(row) + ' |' for row in data]
    md += ['', 'CIFAR-10 / ResNet-18 / L∞ 8/255. Accuracy (%), mean ± population SD',
           'over seeds 0, 1, 2; full standard AutoAttack on 10,000 test examples.',
           'Ours: Align-only without selection, ρ = 0.002. Fixed 110 epochs at every',
           'fraction; smaller subsets receive fewer optimizer updates.',
           'Mean = (C + R)/2, computed per seed then summarized. G = sqrt(mean(C) × mean(R));',
           'G is a single aggregate, not the mean of per-seed geometric means.', '']
    (output / 'impact_of_dataset_size.md').write_text('\n'.join(md))
    return rho_rows, data, header


def plot(rho_rows, data, header, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 11, 'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'svg.fonttype': 'none', 'svg.hashsalt': 'align-study'})
    fig, left = plt.subplots(figsize=(8.4, 5.1))
    right = left.twinx()
    xs = [r['rho'] * 1000 for r in rho_rows]
    handles = []
    for ax, metric, label, color, marker, limits in (
        (left, 'clean', 'Clean accuracy', '#1674B2', 'o', (84, 87)),
        (right, 'aa', 'Robust accuracy (AA)', '#DF6732', 's', (47, 51)),
    ):
        handle = ax.errorbar(xs, [r[metric + '_mean'] for r in rho_rows],
                    yerr=[r[metric + '_std'] for r in rho_rows],
                    label=label, color=color, marker=marker, linewidth=2,
                    markersize=5, capsize=3)
        handles.append(handle)
        ax.set_ylim(*limits)
        ax.set_ylabel(label + ' (%)', color=color)
        ax.tick_params(axis='y', colors=color)
        ax.spines['top'].set_visible(False)
    left.set(title=r'Impact of $\rho$', xlabel=r'$\rho$ ($\times 10^{-3}$)')
    left.set_xticks([0, 1, 2, 3, 5, 10])
    left.grid(axis='y', alpha=.2)
    left.spines['left'].set_color('#1674B2')
    right.spines['right'].set_color('#DF6732')
    left.legend(handles=handles, loc='upper left', frameon=False)
    fig.text(.5, .02, r'CIFAR-10 · ResNet-18 · $\ell_\infty$ 8/255 · 3 seeds; error bars: population SD'
             '\nLeft and right y-axes use different scales.', ha='center', fontsize=9)
    fig.tight_layout(rect=[0, .09, 1, 1])
    save(fig, output / 'impact_of_rho')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11.8, 5.3))
    ax.axis('off')
    ax.set_title('Impact of training dataset size', fontsize=16, pad=16)
    labels = ['Training\ndata', 'Images', 'Method', 'Clean', 'Robust (AA)', 'Mean', 'G']
    table = ax.table(cellText=data, colLabels=labels, cellLoc='center', loc='upper center',
                     colWidths=[.10, .10, .10, .18, .18, .18, .10], bbox=[0, .30, 1, .65])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    for (row, col), c in table.get_celld().items():
        c.set_edgecolor('#d8e0e9')
        c.set_linewidth(.6)
        if row == 0:
            c.set_facecolor('#e8eef5')
            c.get_text().set_weight('bold')
        elif (row - 1) // 2 % 2 == 0:
            c.set_facecolor('#f7f9fc')
    for start in range(0, len(data), 2):
        for col in (3, 4, 5, 6):
            vals = [float(data[row][col].split(' ± ')[0]) for row in (start, start + 1)]
            for row, val in zip((start, start + 1), vals):
                if val == max(vals):
                    table[row + 1, col].get_text().set_weight('bold')
    ax.text(.5, .21, r'CIFAR-10 · ResNet-18 · $\ell_\infty$ 8/255 · mean ± population SD (3 seeds)',
            ha='center', fontsize=10, transform=ax.transAxes)
    ax.text(.5, .13, r'Ours: Align-only without selection, $\rho=0.002$; full standard AA on 10,000 test images.'
            '\nFixed 110 epochs at every fraction; smaller subsets have fewer optimizer updates.',
            ha='center', fontsize=9, transform=ax.transAxes)
    ax.text(.5, .035, r'Mean = $(C+R)/2$; $G=\sqrt{\overline{C}\,\overline{R}}$ (no SD assigned to this aggregate).'
            '\nBold: larger displayed mean for each data fraction and metric.',
            ha='center', fontsize=9, transform=ax.transAxes)
    fig.tight_layout()
    save(fig, output / 'impact_of_dataset_size')
    plt.close(fig)


def save(fig, stem):
    for fmt in ('png', 'pdf', 'svg'):
        metadata = {'Creator': 'Recorded robustness study'}
        if fmt == 'pdf':
            metadata.update(CreationDate=None, ModDate=None)
        elif fmt == 'svg':
            metadata.update(Date=None)
        fig.savefig(stem.with_suffix('.' + fmt), dpi=240, bbox_inches='tight', metadata=metadata)
        if fmt == 'svg':
            path = stem.with_suffix('.svg')
            path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'results/cifar10_resnet18/per_seed.csv')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results/cifar10_resnet18')
    parser.add_argument('--no-plots', action='store_true', help='Validate and write summaries without Matplotlib')
    args = parser.parse_args()
    groups = load_results(args.input)
    summaries = summarize(groups)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rho_rows, data, header = write_summaries(summaries, args.output_dir)
    if not args.no_plots:
        plot(rho_rows, data, header, args.output_dir)
    print('VALID_STUDY_SEEDS=42/42 RHO_POINTS=7/7 DATA_SIZE_ROWS=8/8')
    print(f'OUTPUT_DIR={args.output_dir}')


if __name__ == '__main__':
    main()
