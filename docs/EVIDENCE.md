# Historical evidence update

The configuration files describe previously evaluated hyperparameters. They do
not certify a new GPU reproduction using the portable release environment.

The following three entries previously listed only seed 0. The completed-run
summary now records seeds 0, 1, and 2 for the **same candidate** in each entry.
In that original update, only evidence metadata changed. The later WRN L∞
follow-ups below replace selected candidates after their missing seeds completed.

| Architecture | Norm | Dataset | Candidate | Seeds | Mean | G | Both above fixed baseline maxima? |
|---|---|---|---|---|---:|---:|---|
| ResNet-18 | L∞ | Tiny-ImageNet | e15rt05 | 0,1,2 | 32.44000 | 27.91259 | No |
| WRN-28-10 | L∞ | Tiny-ImageNet | e15wt10 | 0,1,2 | 36.39833 | 32.42240 | No |
| WRN-28-10 | L2 | Tiny-ImageNet | e15w207 | 0,1,2 | 56.73667 | 55.86508 | Yes |

Here G = sqrt(mean(Clean) × mean(AA)), using unrounded seed means in the original
summary. The numbers above are the reported rounded summary, not reconstructed
individual seeds. The original update did not bundle these individual records;
the later recovery below adds the current selected records. The original evidence source was the completed geometric-result summary
provided with the experiment logs.

All 12 configuration entries now have recorded three-seed evidence. This count
measures completed evaluations; it is not a count of settings that beat all
baselines. The CIFAR-10 sensitivity studies and the WRN follow-up below include
their complete per-seed measurements in this repository.

## Completed WRN L∞ follow-up

The existing `e15wt03` candidate now has all three seeds. Seed 0 was reused;
seeds 1 and 2 completed 110 epochs and full standard AA. Their three-seed
Clean / AA / Mean / G are 55.76667 / 19.51667 / 37.64167 / 32.99060.
The current WRN L∞ Tiny-ImageNet configuration now selects this candidate;
the previous `e15wt10` configuration and the historical summary above remain
available. See [raw seeds, parameters, and checked evidence](../results/wrn_tinyimagenet_linf_followup/README.md).

Mean exceeds the archived fixed threshold 36.61, and G exceeds the separately
defined fixed threshold 32.75643. These reference thresholds can come from
different baselines. AA remains below the cited baseline AA of 20.69;
the aggregate improvement does not imply superiority on every metric or
statistical significance. No new parameter combination was added for this
completion. The subsequent ResNet-18 completion is described below.

## Completed ResNet-18 L∞ follow-up and recovered main records

The existing `g19rt03` candidate now has all three seeds, reusing seed 0 and
completing seeds 1 and 2 with 110 epochs and full standard AA. Clean / AA /
Mean / G are 49.58333 / 15.93667 / 32.76000 / 28.11037. Mean and G improve on
`e15rt05`, so the released configuration now selects `g19rt03`. G still falls
below the fixed target 28.19006. The previous configuration and both candidates'
measurements remain in the [follow-up](../results/r18_tinyimagenet_linf_followup/README.md).

The [main catalog](../results/main_without_selection/README.md) now contains
36/36 Ours and 108/216 baseline seed measurements. The missing baseline records
are all WRN; their existing reported aggregates are displayed in the comparison
with a source label, without conversion into seed measurements. Unrecovered
records do not imply unperformed experiments. The recovered ResNet-18 reference table includes the
previously missing L2 CIFAR-10 Ours seeds. Source checksums and evidence levels
are recorded without personal paths. Full training logs were checked for 27 of
the selected Ours records; six are supported by the authoritative archived table
and three by native result metadata.

Changing evidence metadata changes the configuration hash used by the portable
driver. To resume a portable run created with an older release, retain its exact
original configuration; do not edit that run's configuration to bypass the hash
check.
