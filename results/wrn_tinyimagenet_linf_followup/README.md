# WRN-28-10 / Tiny-ImageNet / L∞ follow-up

The existing without-selection candidate `e15wt03` now has seeds 0, 1, and 2.
Seed 0 is reused; only seeds 1 and 2 add training. Each run completed 110 epochs
with final learning rate 0.001 and full standard AutoAttack (APGD-CE, APGD-T,
FAB-T, Square) at epsilon 8/255 on the 10,000-image validation set.
The checkpoint is selected by the training pipeline's PGD evaluation.

| Candidate | Seeds | Clean | AA | Mean | G |
|---|---|---|---|---|---|
| e15wt03 | 0, 1, 2 | 55.77 ± 0.30 | 19.52 ± 0.12 | 37.64 ± 0.13 | 32.99 |

[Raw seeds](per_seed.csv) and [unrounded statistics](summary.csv) accompany the
[sanitized evidence](provenance.json). Clean, AA, and paired Mean use population
SD. Main-table G is `sqrt(mean(Clean) * mean(AA))`, without an assigned SD.
The separately named per-seed geometric statistics are also retained.

The archived fixed Mean threshold is 36.61. The fixed G threshold is 32.75643,
derived from the archived baseline aggregate Clean 51.86 and AA 20.69.
These two maxima come from different baseline comparisons; they are not a
single reconstructed baseline row. The completed candidate exceeds both fixed
aggregate thresholds. Its AA of 19.52 remains below the cited baseline AA of
20.69. This does not establish statistical significance or dominance on every
metric, and the baseline thresholds have only their recorded precision.

The released configuration now selects `e15wt03`: rho 0.0025, AWP start 10,
lambda 1.25, boundary range 0.1, Beta alpha 0.85, initial LR 0.1, weight decay
0.00055. The [previous e15wt10 configuration](historical_e15wt10_config.json)
is retained, along with its historical summary in [the evidence notes](../../docs/EVIDENCE.md).
This is a history-informed follow-up, not an independently preselected trial.
Use the original configuration when resuming an older portable run; do not
change a run's configuration to bypass its checksum guard.
