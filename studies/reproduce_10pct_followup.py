#!/usr/bin/env python3
"""Summarize the separate, paired three-seed 10% rho follow-up."""
import argparse
import csv
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from studies.reproduce_figures import cell, summarize, write_csv

GROUPS = (('raat', 10, None), ('align', 10, .002), ('align', 10, .001))


def load_followup(path):
    groups, subset_hashes = {}, {}
    with Path(path).open(newline='') as handle:
        for row in csv.DictReader(handle):
            key = row['method'], int(row['percent']), float(row['rho']) if row['rho'] else None
            seed = int(row['seed'])
            c, r = float(row['clean']), float(row['aa'])
            if key not in GROUPS or seed not in (0, 1, 2):
                raise ValueError('Unexpected follow-up configuration or seed')
            if not math.isfinite(c) or not math.isfinite(r) or not 0 <= r <= c <= 100:
                raise ValueError('Invalid clean/AA percentage')
            if int(row['test_size']) != 10000 or row['autoattack'] != 'standard_full_APGD-CE_APGD-T_FAB-T_SQUARE':
                raise ValueError('Full standard AA on 10,000 test examples is required')
            digest = row['subset_indices_sha256']
            if len(digest) != 64 or any(x not in '0123456789abcdef' for x in digest):
                raise ValueError('Invalid subset checksum')
            if subset_hashes.setdefault(seed, digest) != digest:
                raise ValueError('The compared methods used different training subsets')
            bucket = groups.setdefault(key, {})
            if seed in bucket:
                raise ValueError('Duplicate follow-up seed')
            bucket[seed] = c, r
    if set(groups) != set(GROUPS) or any(set(v) != {0, 1, 2} for v in groups.values()):
        raise ValueError('All three seeds are required for each follow-up configuration')
    return groups


def write_summary(groups, output):
    summary = summarize(groups)
    output.mkdir(parents=True, exist_ok=True)
    rows = [dict(method=k[0], percent=k[1], rho='' if k[2] is None else k[2], **summary[k]) for k in GROUPS]
    write_csv(output / 'comparison.csv', rows)
    lines = ['# Follow-up: rho at 10% training data', '',
             '| Method | rho | Seeds | Clean | AA | Mean | G |',
             '|---|---|---|---|---|---|---|']
    for key, row in zip(GROUPS, rows):
        label = 'RAAT' if key[0] == 'raat' else 'Ours (without selection)'
        cells = [label, str(key[2]) if key[2] is not None else 'N/A', '0, 1, 2']
        cells += [cell(row, metric) for metric in ('clean', 'aa', 'mean')]
        cells += [f'{row["G_of_mean_accuracies"]:.2f}']
        lines.append('| ' + ' | '.join(cells) + ' |')
    lines += ['', 'CIFAR-10 / ResNet-18 / L∞ 8/255. Each run uses the same 5,000-image',
              'class-balanced subset as the other methods at its seed, 110 epochs, and',
              'full standard AutoAttack on 10,000 test images. All other parameters',
              'match the original dataset-size study. Values are percentages.', '',
              'Clean, AA, and paired-seed Mean use mean ± population SD. G is',
              '`sqrt(mean(Clean) * mean(AA))`, with no assigned SD. The separate',
              'per-seed geometric summary is included in `comparison.csv`.', '',
              'The rho=0.001 runs were added after inspecting the original rho=0.002',
              'results. This is a follow-up, not an independent confirmation of a',
              'preselected hyperparameter. The original fixed-rho dataset-size table',
              'and full-data rho curve are retained unchanged. RAAT and rho=0.002',
              'reuse their original completed seeds; only rho=0.001 adds training.', '',
              'Lowering rho slightly increases the observed Clean and Mean averages,',
              'but lowers AA and G relative to rho=0.002. Both tested settings remain',
              'below RAAT in Mean and G. Three-seed differences are descriptive and',
              'do not establish statistical significance.', '']
    (output / 'comparison.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    folder = ROOT / 'results/cifar10_resnet18_10pct_followup'
    parser.add_argument('--input', type=Path, default=folder / 'per_seed.csv')
    parser.add_argument('--output-dir', type=Path, default=folder)
    args = parser.parse_args()
    write_summary(load_followup(args.input), args.output_dir)
    print('VALID_FOLLOWUP_SEEDS=9/9 NEW_SEEDS=3 REUSED_SEEDS=6')


if __name__ == '__main__':
    main()
