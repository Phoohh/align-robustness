# Main results and source coverage

[Main comparison](comparison.md) · [144 per-seed measurements](per_seed.csv) ·
[Summaries recomputed from seeds](summary.csv) · [Seed-record coverage](coverage.csv)

The released Ours configurations now have **36/36 measured seed records** across
all 12 settings. Ours always means Align-only without alignment-sample selection.
The six ResNet-18 baselines have **108/108 records**. The six WRN baselines have
36 existing [reported aggregate rows](wrn_baseline_reported_aggregates.csv),
although their **108 per-seed records** have not been recovered for this artifact.
This is a gap in the available records, not a claim that those experiments
were never run.

The main comparison displays all **84 method/setting rows**. Its Source column
distinguishes the 48 groups recomputed from seeds from the 36 WRN baseline
groups shown using their archived rounded aggregates. No training or evaluation
was added to fill this display. `summary.csv` retains only seed-derived statistics;
`coverage.csv` continues to identify the missing WRN baseline seed records.

Recompute the comparison, coverage, and statistics without a GPU or dependencies:

```bash
python studies/reproduce_main_results.py --output-dir ./runs/recomputed-main
```

Expected output:
`OURS_SEEDS=36/36 BASELINE_SEEDS=108/216 COMPLETE_GROUPS=48/84`.
The loader rejects duplicate seeds, incomplete seed groups, invalid percentages,
and Ours candidates inconsistent with the released configuration index.

## Evidence levels

| Records | Count | Evidence |
|---|---:|---|
| Ours, nine tuned settings | 27 | Native training completion at epoch 110, final learning rate, and all four standard AA stages checked |
| Ours, ResNet-18 CIFAR-10 under both norms | 6 | Recovered authoritative per-seed table; manifest checksum and duplicate backup agree |
| Ours, WRN L2 CIFAR-10 | 3 | Recovered native result records with full-AA protocol metadata |
| ResNet-18 baselines | 108 | Same recovered authoritative per-seed table |

The [sanitized provenance](provenance.json) preserves source checksums and each
record's evidence level. Recovered table entries and native result metadata do
not constitute a fresh check of the original full logs. Raw logs, checkpoints,
personal paths, and cluster identifiers are not distributed here.
None of these evidence levels claims a new 110-epoch accuracy reproduction
using the portable release environment.

The original ResNet-18 table merged 35 repaired cells with 91 original cells.
It contained 126 records: six baselines and an earlier Ours setting for each
dataset/norm pair. This release reuses its 108 baseline records and the six
CIFAR-10 Ours reference records. Other Ours rows use the later selected candidates.
The original `NRR` column was a harmonic score. It is not renamed to G or used
in the geometric calculations.

## Statistics and selection

Clean, AA, and paired Mean use mean ± population SD over seeds 0, 1, 2.
Mean is `(Clean + AA) / 2` within each seed. Main G is
`sqrt(mean(Clean) * mean(AA))`, evaluated before presentation rounding, without
an invented SD. The separately named per-seed geometric mean and SD are also
included in `summary.csv`. Precision is limited by the recorded measurements.

The WRN baseline archive preserves the reported rounded Clean/AA/Mean and SD,
plus `reported_G`. The comparison displays these existing values without
reconstructing seeds or recomputing unrounded baseline G. Comparisons to these
archived thresholds retain that precision limitation. A recovered seed group
takes precedence over an archived aggregate in the display.

The [WRN Tiny-ImageNet follow-up](../wrn_tinyimagenet_linf_followup/README.md)
selects `e15wt03`. The [ResNet-18 Tiny-ImageNet follow-up](../r18_tinyimagenet_linf_followup/README.md)
selects `g19rt03`, whose Mean and G improve on `e15rt05` but whose G still falls
below the fixed baseline target. Hyperparameters were selected using historical
results; these are not independently preselected confirmation trials.

Completeness of seed records does not imply statistical significance, superiority
on every metric, or completion of every experiment in a future manuscript.
