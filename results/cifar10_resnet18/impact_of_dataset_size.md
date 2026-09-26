| Training data | Images | Method | Clean | Robust (AA) | Mean | G |
|---|---|---|---|---|---|---|
| 100% | 50,000 | RAAT | 85.82 ± 0.12 | 47.74 ± 0.10 | 66.78 ± 0.10 | 64.01 |
| 100% | 50,000 | Ours | 85.57 ± 0.33 | 49.02 ± 0.13 | 67.30 ± 0.18 | 64.77 |
| 50% | 25,000 | RAAT | 81.21 ± 0.08 | 41.99 ± 0.21 | 61.60 ± 0.09 | 58.40 |
| 50% | 25,000 | Ours | 81.56 ± 0.52 | 43.31 ± 0.15 | 62.43 ± 0.26 | 59.43 |
| 20% | 10,000 | RAAT | 68.47 ± 0.75 | 31.37 ± 0.22 | 49.92 ± 0.47 | 46.35 |
| 20% | 10,000 | Ours | 70.43 ± 1.85 | 33.52 ± 0.08 | 51.98 ± 0.93 | 48.59 |
| 10% | 5,000 | RAAT | 63.48 ± 3.87 | 27.91 ± 0.45 | 45.70 ± 1.79 | 42.09 |
| 10% | 5,000 | Ours | 61.54 ± 3.09 | 28.41 ± 0.76 | 44.97 ± 1.85 | 41.81 |

CIFAR-10 / ResNet-18 / L∞ 8/255. Accuracy (%), mean ± population SD
over seeds 0, 1, 2; full standard AutoAttack on 10,000 test examples.
Ours: Align-only without selection, ρ = 0.002. Fixed 110 epochs at every
fraction; smaller subsets receive fewer optimizer updates.
Mean = (C + R)/2, computed per seed then summarized. G = sqrt(mean(C) × mean(R));
G is a single aggregate, not the mean of per-seed geometric means.
