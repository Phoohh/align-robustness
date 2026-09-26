# Follow-up: rho at 10% training data

| Method | rho | Seeds | Clean | AA | Mean | G |
|---|---|---|---|---|---|---|
| RAAT | N/A | 0, 1, 2 | 63.48 ± 3.87 | 27.91 ± 0.45 | 45.70 ± 1.79 | 42.09 |
| Ours (without selection) | 0.002 | 0, 1, 2 | 61.54 ± 3.09 | 28.41 ± 0.76 | 44.97 ± 1.85 | 41.81 |
| Ours (without selection) | 0.001 | 0, 1, 2 | 61.96 ± 2.30 | 28.07 ± 0.95 | 45.01 ± 1.20 | 41.70 |

CIFAR-10 / ResNet-18 / L∞ 8/255. Each run uses the same 5,000-image
class-balanced subset as the other methods at its seed, 110 epochs, and
full standard AutoAttack on 10,000 test images. All other parameters
match the original dataset-size study. Values are percentages.

Clean, AA, and paired-seed Mean use mean ± population SD. G is
`sqrt(mean(Clean) * mean(AA))`, with no assigned SD. The separate
per-seed geometric summary is included in `comparison.csv`.

The rho=0.001 runs were added after inspecting the original rho=0.002
results. This is a follow-up, not an independent confirmation of a
preselected hyperparameter. The original fixed-rho dataset-size table
and full-data rho curve are retained unchanged. RAAT and rho=0.002
reuse their original completed seeds; only rho=0.001 adds training.

Lowering rho slightly increases the observed Clean and Mean averages,
but lowers AA and G relative to rho=0.002. Both tested settings remain
below RAAT in Mean and G. Three-seed differences are descriptive and
do not establish statistical significance.
