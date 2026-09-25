# ResNet-18 / Tiny-ImageNet / L∞ follow-up

The existing candidate `g19rt03` now has seeds 0, 1, and 2. Seed 0 was reused;
seeds 1 and 2 completed 110 epochs with final learning rate 0.00105 and full
standard AutoAttack (APGD-CE, APGD-T, FAB-T, Square) at epsilon 8/255 on all
10,000 validation images. The evaluated checkpoint is selected by PGD accuracy.

| Candidate | Seeds | Clean | AA | Mean | G |
|---|---|---|---|---|---|
| Previous e15rt05 | 0, 1, 2 | 48.97 ± 0.35 | 15.91 ± 0.12 | 32.44 ± 0.22 | 27.91 |
| Completed g19rt03 | 0, 1, 2 | 49.58 ± 0.24 | 15.94 ± 0.09 | 32.76 ± 0.09 | 28.11 |

See [all six raw records](per_seed.csv), [unrounded statistics](summary.csv),
and [checked evidence](provenance.json). Clean, AA, and paired Mean use population
SD. Main G is `sqrt(mean(Clean) * mean(AA))`, with no assigned SD. The separate
per-seed geometric statistics are retained in the CSV.

The completed candidate improves both Mean and G relative to `e15rt05`.
Mean exceeds the archived fixed threshold of 32.16, but **G remains below the
fixed target of 28.19006** derived from baseline Clean 47.19 and AA 16.84.
Its AA remains below that baseline AA. These maxima concern different baseline
comparisons and are not one reconstructed baseline row. The restored baseline
seed records permit exact recomputation in the [main comparison](../main_without_selection/comparison.md);
the historical fixed target retains its originally recorded precision.

The logs show completed training, the expected final learning rate, and all
four AA stages. The target miss is a measured outcome, not an incomplete job.
The new candidate mainly improves Clean; its AA gain over `e15rt05` is only
0.02667 percentage points, leaving the geometric score below the target.
Three-seed descriptive differences do not establish statistical significance.

The released configuration now selects `g19rt03`: rho 0.0026, AWP start 0,
lambda 1.35, boundary range 0.15, Beta alpha 0.85, initial LR 0.105, and
weight decay 0.0005. The [previous configuration](historical_e15rt05_config.json)
is preserved. Keep a run's original configuration when resuming it.
This history-informed completion adds no new parameter combination.
