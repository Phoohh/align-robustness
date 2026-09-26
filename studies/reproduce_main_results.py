#!/usr/bin/env python3
"""Recompute seed summaries and display separately labeled archived aggregates."""
import argparse
import csv
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from studies.reproduce_figures import cell, summarize, write_csv

METHODS = ('PGD-AT', 'TRADES', 'MART', 'Cons-AT', 'RAAT', 'RAAT++', 'Ours')
NORMS = ('Linf', 'L2')
DATASETS = ('cifar10', 'cifar100', 'tinyimagenet')
REPORTED_AGGREGATES = ROOT / 'results/main_without_selection/wrn_baseline_reported_aggregates.csv'
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


def load_reported_aggregates(path):
    """Validate all 36 archived WRN baseline groups without reconstructing seeds."""
    expected = {('wrn', norm, dataset, method)
                for norm in NORMS for dataset in DATASETS for method in METHODS[:-1]}
    metrics = tuple(f'{metric}_{stat}' for metric in ('clean', 'aa', 'mean')
                    for stat in ('mean', 'std'))
    required = {'arch', 'norm', 'dataset', 'method', 'precision', 'reported_G'}
    required.update(f'reported_{metric}' for metric in metrics)
    aggregates = {}
    with Path(path).open(newline='') as handle:
        reader = csv.DictReader(handle)
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError('Missing archived aggregate columns')
        for row in reader:
            key = tuple(row[field] for field in ('arch', 'norm', 'dataset', 'method'))
            if key not in expected:
                raise ValueError(f'Unexpected archived WRN baseline group: {key}')
            if key in aggregates:
                raise ValueError(f'Duplicate archived aggregate group: {key}')
            if row['precision'] != 'rounded_aggregate_only':
                raise ValueError('Unknown archived aggregate precision')
            try:
                values = {metric: float(row[f'reported_{metric}']) for metric in metrics}
                reported_g = float(row['reported_G'])
            except (TypeError, ValueError) as error:
                raise ValueError('Invalid archived aggregate number') from error
            if not all(math.isfinite(value) for value in (*values.values(), reported_g)):
                raise ValueError('Archived aggregates must be finite')
            if not 0 <= values['aa_mean'] <= values['clean_mean'] <= 100:
                raise ValueError('Invalid archived accuracy percentage')
            if not 0 <= values['mean_mean'] <= 100 or not 0 <= reported_g <= 100:
                raise ValueError('Invalid archived Mean or G percentage')
            if any(values[f'{metric}_std'] < 0 for metric in ('clean', 'aa', 'mean')):
                raise ValueError('Archived standard deviations must be nonnegative')
            # Keep the recorded text: this is not a G recomputation from rounded means.
            aggregates[key] = dict(**values, reported_G=row['reported_G'])
    if set(aggregates) != expected:
        raise ValueError('Archived aggregates must cover all 36 WRN baseline groups')
    return aggregates


def write_main(configs, groups, evidence, output, reported_aggregates=REPORTED_AGGREGATES):
    archived = load_reported_aggregates(reported_aggregates)
    summaries = summarize(groups)
    rows, coverage = [], []
    lines = ['# Main comparison from seed records and archived aggregates', '',
             'Accuracy (%). For rows marked **Recomputed from seeds**, Clean, AA, and paired Mean',
             'use population SD over seeds 0, 1, 2. Only these rows compute G as',
             '`sqrt(mean(Clean) * mean(AA))` before rounding.',
             'Rows marked **Archived rounded aggregate** preserve the reported Clean, AA, and Mean',
             'means and SDs; G uses the recorded `reported_G`, rounded only for display. These rows do not',
             'reconstruct seeds or SDs, or claim a G computed from unrounded seed measurements.',
             'Seed records take precedence when both sources are available. `summary.csv` contains',
             'only seed-based summaries; `coverage.csv` continues to report missing seed records.',
             'See [source coverage](README.md) for the evidence levels and limitations.', '']
    for arch in ('r18', 'wrn'):
        for norm in NORMS:
            lines += [f'## {arch} / {norm}', '',
                      '| Dataset | Method | Candidate | Clean | AA | Mean | G | Source |',
                      '|---|---|---|---|---|---|---|---|']
            for dataset in DATASETS:
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
                        cells += [f'{s["G_of_mean_accuracies"]:.2f}', 'Recomputed from seeds']
                    elif key in archived:
                        s = archived[key]
                        cells += [cell(s, metric) for metric in ('clean', 'aa', 'mean')]
                        cells += [f'{float(s["reported_G"]):.2f}', 'Archived rounded aggregate']
                    else:
                        cells += ['N/A'] * 4 + ['Missing records']
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
    parser.add_argument('--reported-aggregates', type=Path, default=REPORTED_AGGREGATES,
                        help='Archived WRN baseline aggregates used only in the comparison table')
    parser.add_argument('--output-dir', type=Path, default=folder)
    args = parser.parse_args()
    configs, groups, evidence = load_main(args.input)
    write_main(configs, groups, evidence, args.output_dir, args.reported_aggregates)
    ours = sum(len(v) for k, v in groups.items() if k[-1] == 'Ours')
    baseline = sum(len(v) for k, v in groups.items() if k[-1] != 'Ours')
    print(f'OURS_SEEDS={ours}/36 BASELINE_SEEDS={baseline}/216 COMPLETE_GROUPS={len(groups)}/84')


if __name__ == '__main__':
    main()
