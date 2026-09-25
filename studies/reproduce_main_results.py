#!/usr/bin/env python3
"""Recompute the main comparison from recovered per-seed measurements."""
import argparse
import csv
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from studies.reproduce_figures import cell, summarize, write_csv

METHODS = ('PGD-AT', 'TRADES', 'MART', 'Cons-AT', 'RAAT', 'RAAT++', 'Ours')
EVIDENCE = {'archived_per_seed_table', 'native_result_record',
            'training_and_full_aa_checked', 'full_aa_log_checked'}


def configurations(path):
    with Path(path).open(newline='') as handle:
        return {(r['arch'], r['norm'], r['dataset']): r['candidate']
                for r in csv.DictReader(handle, delimiter='\t')}


def load_main(path, index=ROOT / 'configs/index.tsv'):
    configs = configurations(index)
    groups, evidence = {}, {}
    with Path(path).open(newline='') as handle:
        for row in csv.DictReader(handle):
            setting = row['arch'], row['norm'], row['dataset']
            method, seed = row['method'], int(row['seed'])
            if setting not in configs or method not in METHODS or seed not in (0, 1, 2):
                raise ValueError('Unexpected main setting, method, or seed')
            expected = configs[setting] if method == 'Ours' else 'recorded_baseline'
            if row['candidate'] != expected:
                raise ValueError('Candidate does not match the released configuration')
            clean, aa = float(row['clean']), float(row['aa'])
            if not math.isfinite(clean) or not math.isfinite(aa) or not 0 <= aa <= clean <= 100:
                raise ValueError('Invalid accuracy percentage')
            digest = row['source_sha256']
            if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
                raise ValueError('Missing source checksum')
            if row['evidence'] not in EVIDENCE:
                raise ValueError('Unknown source evidence type')
            key = (*setting, method)
            bucket = groups.setdefault(key, {})
            if seed in bucket:
                raise ValueError('Duplicate main-comparison seed')
            bucket[seed] = clean, aa
            evidence.setdefault(key, set()).add(row['evidence'])
    for key, seeds in groups.items():
        if set(seeds) != {0, 1, 2}:
            raise ValueError(f'Incomplete three-seed group: {key}')
    for setting in configs:
        if (*setting, 'Ours') not in groups:
            raise ValueError(f'Missing released Ours setting: {setting}')
    return configs, groups, evidence


def write_main(configs, groups, evidence, output):
    summaries = summarize(groups)
    rows, coverage = [], []
    lines = ['# Main comparison from recovered seed records', '',
             'Accuracy (%). Clean, AA, and paired Mean use population SD over seeds',
             '0, 1, 2. G is `sqrt(mean(Clean) * mean(AA))`, computed before rounding.',
             'See [source coverage](README.md) for the evidence levels and limitations.', '']
    for arch in ('r18', 'wrn'):
        for norm in ('Linf', 'L2'):
            lines += [f'## {arch} / {norm}', '',
                      '| Dataset | Method | Candidate | Clean | AA | Mean | G |',
                      '|---|---|---|---|---|---|---|']
            for dataset in ('cifar10', 'cifar100', 'tinyimagenet'):
                setting = arch, norm, dataset
                for method in METHODS:
                    key = (*setting, method)
                    candidate = configs[setting] if method == 'Ours' else 'recorded_baseline'
                    identity = dict(arch=arch, norm=norm, dataset=dataset, method=method, candidate=candidate)
                    coverage.append(dict(**identity, available_seeds='0,1,2' if key in groups else '',
                                         record_count=len(groups.get(key, {})),
                                         status='complete_records' if key in groups else 'missing_records',
                                         evidence=';'.join(sorted(evidence.get(key, set())))))
                    cells = [dataset, method, candidate if method == 'Ours' else 'recorded']
                    if key in summaries:
                        s = summaries[key]
                        rows.append(dict(**identity, **s))
                        cells += [cell(s, metric) for metric in ('clean', 'aa', 'mean')]
                        cells += [f'{s["G_of_mean_accuracies"]:.2f}']
                    else:
                        cells += ['N/A'] * 4
                    lines.append('| ' + ' | '.join(cells) + ' |')
            lines.append('')
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / 'summary.csv', rows)
    write_csv(output / 'coverage.csv', coverage)
    (output / 'comparison.md').write_text('\n'.join(lines))


def main():
    folder = ROOT / 'results/main_without_selection'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=folder / 'per_seed.csv')
    parser.add_argument('--output-dir', type=Path, default=folder)
    args = parser.parse_args()
    configs, groups, evidence = load_main(args.input)
    write_main(configs, groups, evidence, args.output_dir)
    ours = sum(len(v) for k, v in groups.items() if k[-1] == 'Ours')
    baseline = sum(len(v) for k, v in groups.items() if k[-1] != 'Ours')
    print(f'OURS_SEEDS={ours}/36 BASELINE_SEEDS={baseline}/216 COMPLETE_GROUPS={len(groups)}/84')


if __name__ == '__main__':
    main()
