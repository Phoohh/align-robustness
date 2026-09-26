# Main comparison from seed records and archived aggregates

Accuracy (%). For rows marked **Recomputed from seeds**, Clean, AA, and paired Mean
use population SD over seeds 0, 1, 2. Only these rows compute G as
`sqrt(mean(Clean) * mean(AA))` before rounding.
Rows marked **Archived rounded aggregate** preserve the reported Clean, AA, and Mean
means and SDs; G uses the recorded `reported_G`, rounded only for display. These rows do not
reconstruct seeds or SDs, or claim a G computed from unrounded seed measurements.
Seed records take precedence when both sources are available. `summary.csv` contains
only seed-based summaries; `coverage.csv` continues to report missing seed records.
See [source coverage](README.md) for the evidence levels and limitations.

## r18 / Linf

| Dataset | Method | Candidate | Clean | AA | Mean | G | Source |
|---|---|---|---|---|---|---|---|
| cifar10 | PGD-AT | recorded | 84.11 ± 0.17 | 48.28 ± 0.24 | 66.19 ± 0.10 | 63.72 | Recomputed from seeds |
| cifar10 | TRADES | recorded | 81.30 ± 0.07 | 49.62 ± 0.06 | 65.46 ± 0.06 | 63.52 | Recomputed from seeds |
| cifar10 | MART | recorded | 78.47 ± 0.44 | 47.98 ± 0.04 | 63.22 ± 0.23 | 61.36 | Recomputed from seeds |
| cifar10 | Cons-AT | recorded | 84.25 ± 0.71 | 48.50 ± 0.17 | 66.38 ± 0.28 | 63.92 | Recomputed from seeds |
| cifar10 | RAAT | recorded | 85.82 ± 0.12 | 47.74 ± 0.10 | 66.78 ± 0.10 | 64.01 | Recomputed from seeds |
| cifar10 | RAAT++ | recorded | 79.49 ± 0.39 | 48.49 ± 0.11 | 63.99 ± 0.17 | 62.08 | Recomputed from seeds |
| cifar10 | Ours | reference | 85.57 ± 0.33 | 49.02 ± 0.13 | 67.30 ± 0.18 | 64.77 | Recomputed from seeds |
| cifar100 | PGD-AT | recorded | 58.91 ± 0.56 | 24.97 ± 0.32 | 41.94 ± 0.44 | 38.36 | Recomputed from seeds |
| cifar100 | TRADES | recorded | 57.76 ± 0.38 | 25.20 ± 0.19 | 41.48 ± 0.11 | 38.15 | Recomputed from seeds |
| cifar100 | MART | recorded | 52.88 ± 0.26 | 25.94 ± 0.06 | 39.41 ± 0.14 | 37.04 | Recomputed from seeds |
| cifar100 | Cons-AT | recorded | 59.05 ± 0.25 | 25.29 ± 0.17 | 42.17 ± 0.08 | 38.65 | Recomputed from seeds |
| cifar100 | RAAT | recorded | 60.58 ± 0.12 | 23.38 ± 0.30 | 41.98 ± 0.13 | 37.63 | Recomputed from seeds |
| cifar100 | RAAT++ | recorded | 53.96 ± 0.22 | 25.57 ± 0.11 | 39.76 ± 0.10 | 37.14 | Recomputed from seeds |
| cifar100 | Ours | e15ri09 | 59.70 ± 0.16 | 25.35 ± 0.06 | 42.52 ± 0.11 | 38.90 | Recomputed from seeds |
| tinyimagenet | PGD-AT | recorded | 46.81 ± 0.12 | 16.50 ± 0.07 | 31.66 ± 0.08 | 27.80 | Recomputed from seeds |
| tinyimagenet | TRADES | recorded | 47.62 ± 0.09 | 15.40 ± 0.07 | 31.51 ± 0.08 | 27.08 | Recomputed from seeds |
| tinyimagenet | MART | recorded | 40.72 ± 0.21 | 16.84 ± 0.12 | 28.78 ± 0.09 | 26.18 | Recomputed from seeds |
| tinyimagenet | Cons-AT | recorded | 47.19 ± 0.93 | 16.84 ± 0.32 | 32.02 ± 0.61 | 28.19 | Recomputed from seeds |
| tinyimagenet | RAAT | recorded | 49.02 ± 0.62 | 15.31 ± 0.21 | 32.16 ± 0.40 | 27.39 | Recomputed from seeds |
| tinyimagenet | RAAT++ | recorded | 40.76 ± 0.49 | 16.42 ± 0.07 | 28.59 ± 0.21 | 25.87 | Recomputed from seeds |
| tinyimagenet | Ours | g19rt03 | 49.58 ± 0.24 | 15.94 ± 0.09 | 32.76 ± 0.09 | 28.11 | Recomputed from seeds |

## r18 / L2

| Dataset | Method | Candidate | Clean | AA | Mean | G | Source |
|---|---|---|---|---|---|---|---|
| cifar10 | PGD-AT | recorded | 89.56 ± 0.05 | 68.52 ± 0.17 | 79.04 ± 0.11 | 78.33 | Recomputed from seeds |
| cifar10 | TRADES | recorded | 86.54 ± 0.19 | 69.52 ± 0.11 | 78.03 ± 0.10 | 77.56 | Recomputed from seeds |
| cifar10 | MART | recorded | 86.70 ± 0.32 | 68.46 ± 0.09 | 77.58 ± 0.20 | 77.04 | Recomputed from seeds |
| cifar10 | Cons-AT | recorded | 89.69 ± 0.61 | 69.25 ± 0.09 | 79.47 ± 0.27 | 78.81 | Recomputed from seeds |
| cifar10 | RAAT | recorded | 90.15 ± 0.01 | 68.33 ± 0.16 | 79.24 ± 0.07 | 78.49 | Recomputed from seeds |
| cifar10 | RAAT++ | recorded | 87.10 ± 0.17 | 68.98 ± 0.02 | 78.04 ± 0.07 | 77.51 | Recomputed from seeds |
| cifar10 | Ours | reference | 89.98 ± 0.16 | 69.61 ± 0.19 | 79.80 ± 0.10 | 79.14 | Recomputed from seeds |
| cifar100 | PGD-AT | recorded | 65.80 ± 0.28 | 39.48 ± 0.13 | 52.64 ± 0.10 | 50.97 | Recomputed from seeds |
| cifar100 | TRADES | recorded | 62.22 ± 0.14 | 40.75 ± 0.22 | 51.49 ± 0.05 | 50.36 | Recomputed from seeds |
| cifar100 | MART | recorded | 61.54 ± 0.17 | 41.04 ± 0.23 | 51.29 ± 0.04 | 50.25 | Recomputed from seeds |
| cifar100 | Cons-AT | recorded | 66.12 ± 0.64 | 40.25 ± 0.16 | 53.19 ± 0.33 | 51.59 | Recomputed from seeds |
| cifar100 | RAAT | recorded | 66.82 ± 0.03 | 38.61 ± 0.13 | 52.71 ± 0.08 | 50.79 | Recomputed from seeds |
| cifar100 | RAAT++ | recorded | 60.77 ± 0.68 | 40.74 ± 0.17 | 50.76 ± 0.26 | 49.76 | Recomputed from seeds |
| cifar100 | Ours | 2108 | 67.82 ± 0.21 | 40.60 ± 0.32 | 54.21 ± 0.25 | 52.48 | Recomputed from seeds |
| tinyimagenet | PGD-AT | recorded | 58.53 ± 0.34 | 39.85 ± 0.29 | 49.19 ± 0.27 | 48.30 | Recomputed from seeds |
| tinyimagenet | TRADES | recorded | 56.31 ± 0.33 | 40.61 ± 0.06 | 48.46 ± 0.19 | 47.82 | Recomputed from seeds |
| tinyimagenet | MART | recorded | 56.45 ± 0.36 | 41.07 ± 0.16 | 48.76 ± 0.19 | 48.15 | Recomputed from seeds |
| tinyimagenet | Cons-AT | recorded | 59.40 ± 0.32 | 40.91 ± 0.24 | 50.15 ± 0.24 | 49.29 | Recomputed from seeds |
| tinyimagenet | RAAT | recorded | 59.02 ± 0.51 | 38.93 ± 0.13 | 48.98 ± 0.31 | 47.94 | Recomputed from seeds |
| tinyimagenet | RAAT++ | recorded | 56.23 ± 0.07 | 41.21 ± 0.22 | 48.72 ± 0.14 | 48.14 | Recomputed from seeds |
| tinyimagenet | Ours | e15r211 | 60.68 ± 0.17 | 40.23 ± 0.36 | 50.45 ± 0.18 | 49.41 | Recomputed from seeds |

## wrn / Linf

| Dataset | Method | Candidate | Clean | AA | Mean | G | Source |
|---|---|---|---|---|---|---|---|
| cifar10 | PGD-AT | recorded | 86.96 ± 0.11 | 51.48 ± 0.22 | 69.22 ± 0.10 | 66.91 | Archived rounded aggregate |
| cifar10 | TRADES | recorded | 85.11 ± 0.12 | 53.37 ± 0.14 | 69.24 ± 0.10 | 67.40 | Archived rounded aggregate |
| cifar10 | MART | recorded | 83.18 ± 0.25 | 51.88 ± 0.06 | 67.53 ± 0.10 | 65.69 | Archived rounded aggregate |
| cifar10 | Cons-AT | recorded | 86.96 ± 0.14 | 51.93 ± 0.07 | 69.45 ± 0.09 | 67.20 | Archived rounded aggregate |
| cifar10 | RAAT | recorded | 87.84 ± 0.12 | 50.60 ± 0.23 | 69.22 ± 0.18 | 66.67 | Archived rounded aggregate |
| cifar10 | RAAT++ | recorded | 83.49 ± 0.72 | 51.89 ± 0.21 | 67.69 ± 0.45 | 65.82 | Archived rounded aggregate |
| cifar10 | Ours | w2c05 | 87.81 ± 0.48 | 52.27 ± 0.50 | 70.04 ± 0.47 | 67.75 | Recomputed from seeds |
| cifar100 | PGD-AT | recorded | 62.16 ± 0.09 | 27.62 ± 0.36 | 44.89 ± 0.14 | 41.43 | Archived rounded aggregate |
| cifar100 | TRADES | recorded | 61.27 ± 0.27 | 27.88 ± 0.29 | 44.58 ± 0.09 | 41.33 | Archived rounded aggregate |
| cifar100 | MART | recorded | 57.86 ± 0.20 | 28.39 ± 0.18 | 43.13 ± 0.11 | 40.53 | Archived rounded aggregate |
| cifar100 | Cons-AT | recorded | 62.93 ± 0.13 | 28.09 ± 0.04 | 45.51 ± 0.07 | 42.04 | Archived rounded aggregate |
| cifar100 | RAAT | recorded | 65.07 ± 0.18 | 26.41 ± 0.39 | 45.74 ± 0.28 | 41.45 | Archived rounded aggregate |
| cifar100 | RAAT++ | recorded | 56.60 ± 1.93 | 27.92 ± 0.16 | 42.26 ± 0.89 | 39.75 | Archived rounded aggregate |
| cifar100 | Ours | e15wi08 | 64.52 ± 0.58 | 27.53 ± 0.43 | 46.02 ± 0.37 | 42.14 | Recomputed from seeds |
| tinyimagenet | PGD-AT | recorded | 52.46 ± 1.10 | 20.05 ± 0.34 | 36.26 ± 0.72 | 32.43 | Archived rounded aggregate |
| tinyimagenet | TRADES | recorded | 53.14 ± 0.05 | 20.08 ± 0.25 | 36.61 ± 0.15 | 32.67 | Archived rounded aggregate |
| tinyimagenet | MART | recorded | 47.23 ± 0.15 | 21.03 ± 0.13 | 34.13 ± 0.10 | 31.52 | Archived rounded aggregate |
| tinyimagenet | Cons-AT | recorded | 51.86 ± 0.29 | 20.69 ± 0.28 | 36.28 ± 0.16 | 32.76 | Archived rounded aggregate |
| tinyimagenet | RAAT | recorded | 54.01 ± 0.27 | 18.74 ± 0.45 | 36.38 ± 0.36 | 31.81 | Archived rounded aggregate |
| tinyimagenet | RAAT++ | recorded | 46.60 ± 0.15 | 20.51 ± 0.12 | 33.56 ± 0.06 | 30.92 | Archived rounded aggregate |
| tinyimagenet | Ours | e15wt03 | 55.77 ± 0.30 | 19.52 ± 0.12 | 37.64 ± 0.13 | 32.99 | Recomputed from seeds |

## wrn / L2

| Dataset | Method | Candidate | Clean | AA | Mean | G | Source |
|---|---|---|---|---|---|---|---|
| cifar10 | PGD-AT | recorded | 91.26 ± 0.03 | 70.48 ± 0.23 | 80.87 ± 0.12 | 80.20 | Archived rounded aggregate |
| cifar10 | TRADES | recorded | 89.10 ± 0.19 | 71.89 ± 0.10 | 80.50 ± 0.05 | 80.03 | Archived rounded aggregate |
| cifar10 | MART | recorded | 89.69 ± 0.07 | 71.84 ± 0.05 | 80.76 ± 0.02 | 80.27 | Archived rounded aggregate |
| cifar10 | Cons-AT | recorded | 91.69 ± 0.24 | 71.20 ± 0.17 | 81.45 ± 0.10 | 80.80 | Archived rounded aggregate |
| cifar10 | RAAT | recorded | 91.50 ± 0.20 | 70.23 ± 0.17 | 80.86 ± 0.14 | 80.16 | Archived rounded aggregate |
| cifar10 | RAAT++ | recorded | 88.86 ± 0.17 | 71.23 ± 0.34 | 80.05 ± 0.26 | 79.56 | Archived rounded aggregate |
| cifar10 | Ours | reference | 91.67 ± 0.07 | 71.63 ± 0.02 | 81.65 ± 0.03 | 81.03 | Recomputed from seeds |
| cifar100 | PGD-AT | recorded | 69.13 ± 0.32 | 41.99 ± 0.16 | 55.56 ± 0.09 | 53.88 | Archived rounded aggregate |
| cifar100 | TRADES | recorded | 65.12 ± 0.12 | 43.02 ± 0.13 | 54.07 ± 0.02 | 52.93 | Archived rounded aggregate |
| cifar100 | MART | recorded | 64.42 ± 0.07 | 43.51 ± 0.20 | 53.97 ± 0.10 | 52.94 | Archived rounded aggregate |
| cifar100 | Cons-AT | recorded | 69.33 ± 0.36 | 42.43 ± 0.24 | 55.88 ± 0.13 | 54.24 | Archived rounded aggregate |
| cifar100 | RAAT | recorded | 70.69 ± 0.21 | 41.14 ± 0.21 | 55.92 ± 0.16 | 53.93 | Archived rounded aggregate |
| cifar100 | RAAT++ | recorded | 63.38 ± 0.57 | 43.26 ± 0.27 | 53.32 ± 0.39 | 52.36 | Archived rounded aggregate |
| cifar100 | Ours | n03 | 70.75 ± 0.41 | 42.23 ± 0.16 | 56.49 ± 0.13 | 54.66 | Recomputed from seeds |
| tinyimagenet | PGD-AT | recorded | 63.79 ± 0.25 | 45.24 ± 0.27 | 54.51 ± 0.11 | 53.72 | Archived rounded aggregate |
| tinyimagenet | TRADES | recorded | 62.18 ± 0.32 | 45.99 ± 0.04 | 54.08 ± 0.16 | 53.48 | Archived rounded aggregate |
| tinyimagenet | MART | recorded | 62.20 ± 0.07 | 46.50 ± 0.08 | 54.35 ± 0.07 | 53.78 | Archived rounded aggregate |
| tinyimagenet | Cons-AT | recorded | 64.90 ± 0.26 | 46.40 ± 0.18 | 55.65 ± 0.14 | 54.88 | Archived rounded aggregate |
| tinyimagenet | RAAT | recorded | 64.97 ± 0.26 | 44.15 ± 0.30 | 54.56 ± 0.08 | 53.56 | Archived rounded aggregate |
| tinyimagenet | RAAT++ | recorded | 61.63 ± 0.37 | 46.27 ± 0.46 | 53.95 ± 0.42 | 53.40 | Archived rounded aggregate |
| tinyimagenet | Ours | e15w207 | 66.64 ± 0.16 | 46.83 ± 0.28 | 56.74 ± 0.19 | 55.87 | Recomputed from seeds |
