# Main comparison from recovered seed records

Accuracy (%). Clean, AA, and paired Mean use population SD over seeds
0, 1, 2. G is `sqrt(mean(Clean) * mean(AA))`, computed before rounding.
See [source coverage](README.md) for the evidence levels and limitations.

## r18 / Linf

| Dataset | Method | Candidate | Clean | AA | Mean | G |
|---|---|---|---|---|---|---|
| cifar10 | PGD-AT | recorded | 84.11 ± 0.17 | 48.28 ± 0.24 | 66.19 ± 0.10 | 63.72 |
| cifar10 | TRADES | recorded | 81.30 ± 0.07 | 49.62 ± 0.06 | 65.46 ± 0.06 | 63.52 |
| cifar10 | MART | recorded | 78.47 ± 0.44 | 47.98 ± 0.04 | 63.22 ± 0.23 | 61.36 |
| cifar10 | Cons-AT | recorded | 84.25 ± 0.71 | 48.50 ± 0.17 | 66.38 ± 0.28 | 63.92 |
| cifar10 | RAAT | recorded | 85.82 ± 0.12 | 47.74 ± 0.10 | 66.78 ± 0.10 | 64.01 |
| cifar10 | RAAT++ | recorded | 79.49 ± 0.39 | 48.49 ± 0.11 | 63.99 ± 0.17 | 62.08 |
| cifar10 | Ours | reference | 85.57 ± 0.33 | 49.02 ± 0.13 | 67.30 ± 0.18 | 64.77 |
| cifar100 | PGD-AT | recorded | 58.91 ± 0.56 | 24.97 ± 0.32 | 41.94 ± 0.44 | 38.36 |
| cifar100 | TRADES | recorded | 57.76 ± 0.38 | 25.20 ± 0.19 | 41.48 ± 0.11 | 38.15 |
| cifar100 | MART | recorded | 52.88 ± 0.26 | 25.94 ± 0.06 | 39.41 ± 0.14 | 37.04 |
| cifar100 | Cons-AT | recorded | 59.05 ± 0.25 | 25.29 ± 0.17 | 42.17 ± 0.08 | 38.65 |
| cifar100 | RAAT | recorded | 60.58 ± 0.12 | 23.38 ± 0.30 | 41.98 ± 0.13 | 37.63 |
| cifar100 | RAAT++ | recorded | 53.96 ± 0.22 | 25.57 ± 0.11 | 39.76 ± 0.10 | 37.14 |
| cifar100 | Ours | e15ri09 | 59.70 ± 0.16 | 25.35 ± 0.06 | 42.52 ± 0.11 | 38.90 |
| tinyimagenet | PGD-AT | recorded | 46.81 ± 0.12 | 16.50 ± 0.07 | 31.66 ± 0.08 | 27.80 |
| tinyimagenet | TRADES | recorded | 47.62 ± 0.09 | 15.40 ± 0.07 | 31.51 ± 0.08 | 27.08 |
| tinyimagenet | MART | recorded | 40.72 ± 0.21 | 16.84 ± 0.12 | 28.78 ± 0.09 | 26.18 |
| tinyimagenet | Cons-AT | recorded | 47.19 ± 0.93 | 16.84 ± 0.32 | 32.02 ± 0.61 | 28.19 |
| tinyimagenet | RAAT | recorded | 49.02 ± 0.62 | 15.31 ± 0.21 | 32.16 ± 0.40 | 27.39 |
| tinyimagenet | RAAT++ | recorded | 40.76 ± 0.49 | 16.42 ± 0.07 | 28.59 ± 0.21 | 25.87 |
| tinyimagenet | Ours | g19rt03 | 49.58 ± 0.24 | 15.94 ± 0.09 | 32.76 ± 0.09 | 28.11 |

## r18 / L2

| Dataset | Method | Candidate | Clean | AA | Mean | G |
|---|---|---|---|---|---|---|
| cifar10 | PGD-AT | recorded | 89.56 ± 0.05 | 68.52 ± 0.17 | 79.04 ± 0.11 | 78.33 |
| cifar10 | TRADES | recorded | 86.54 ± 0.19 | 69.52 ± 0.11 | 78.03 ± 0.10 | 77.56 |
| cifar10 | MART | recorded | 86.70 ± 0.32 | 68.46 ± 0.09 | 77.58 ± 0.20 | 77.04 |
| cifar10 | Cons-AT | recorded | 89.69 ± 0.61 | 69.25 ± 0.09 | 79.47 ± 0.27 | 78.81 |
| cifar10 | RAAT | recorded | 90.15 ± 0.01 | 68.33 ± 0.16 | 79.24 ± 0.07 | 78.49 |
| cifar10 | RAAT++ | recorded | 87.10 ± 0.17 | 68.98 ± 0.02 | 78.04 ± 0.07 | 77.51 |
| cifar10 | Ours | reference | 89.98 ± 0.16 | 69.61 ± 0.19 | 79.80 ± 0.10 | 79.14 |
| cifar100 | PGD-AT | recorded | 65.80 ± 0.28 | 39.48 ± 0.13 | 52.64 ± 0.10 | 50.97 |
| cifar100 | TRADES | recorded | 62.22 ± 0.14 | 40.75 ± 0.22 | 51.49 ± 0.05 | 50.36 |
| cifar100 | MART | recorded | 61.54 ± 0.17 | 41.04 ± 0.23 | 51.29 ± 0.04 | 50.25 |
| cifar100 | Cons-AT | recorded | 66.12 ± 0.64 | 40.25 ± 0.16 | 53.19 ± 0.33 | 51.59 |
| cifar100 | RAAT | recorded | 66.82 ± 0.03 | 38.61 ± 0.13 | 52.71 ± 0.08 | 50.79 |
| cifar100 | RAAT++ | recorded | 60.77 ± 0.68 | 40.74 ± 0.17 | 50.76 ± 0.26 | 49.76 |
| cifar100 | Ours | 2108 | 67.82 ± 0.21 | 40.60 ± 0.32 | 54.21 ± 0.25 | 52.48 |
| tinyimagenet | PGD-AT | recorded | 58.53 ± 0.34 | 39.85 ± 0.29 | 49.19 ± 0.27 | 48.30 |
| tinyimagenet | TRADES | recorded | 56.31 ± 0.33 | 40.61 ± 0.06 | 48.46 ± 0.19 | 47.82 |
| tinyimagenet | MART | recorded | 56.45 ± 0.36 | 41.07 ± 0.16 | 48.76 ± 0.19 | 48.15 |
| tinyimagenet | Cons-AT | recorded | 59.40 ± 0.32 | 40.91 ± 0.24 | 50.15 ± 0.24 | 49.29 |
| tinyimagenet | RAAT | recorded | 59.02 ± 0.51 | 38.93 ± 0.13 | 48.98 ± 0.31 | 47.94 |
| tinyimagenet | RAAT++ | recorded | 56.23 ± 0.07 | 41.21 ± 0.22 | 48.72 ± 0.14 | 48.14 |
| tinyimagenet | Ours | e15r211 | 60.68 ± 0.17 | 40.23 ± 0.36 | 50.45 ± 0.18 | 49.41 |

## wrn / Linf

| Dataset | Method | Candidate | Clean | AA | Mean | G |
|---|---|---|---|---|---|---|
| cifar10 | PGD-AT | recorded | N/A | N/A | N/A | N/A |
| cifar10 | TRADES | recorded | N/A | N/A | N/A | N/A |
| cifar10 | MART | recorded | N/A | N/A | N/A | N/A |
| cifar10 | Cons-AT | recorded | N/A | N/A | N/A | N/A |
| cifar10 | RAAT | recorded | N/A | N/A | N/A | N/A |
| cifar10 | RAAT++ | recorded | N/A | N/A | N/A | N/A |
| cifar10 | Ours | w2c05 | 87.81 ± 0.48 | 52.27 ± 0.50 | 70.04 ± 0.47 | 67.75 |
| cifar100 | PGD-AT | recorded | N/A | N/A | N/A | N/A |
| cifar100 | TRADES | recorded | N/A | N/A | N/A | N/A |
| cifar100 | MART | recorded | N/A | N/A | N/A | N/A |
| cifar100 | Cons-AT | recorded | N/A | N/A | N/A | N/A |
| cifar100 | RAAT | recorded | N/A | N/A | N/A | N/A |
| cifar100 | RAAT++ | recorded | N/A | N/A | N/A | N/A |
| cifar100 | Ours | e15wi08 | 64.52 ± 0.58 | 27.53 ± 0.43 | 46.02 ± 0.37 | 42.14 |
| tinyimagenet | PGD-AT | recorded | N/A | N/A | N/A | N/A |
| tinyimagenet | TRADES | recorded | N/A | N/A | N/A | N/A |
| tinyimagenet | MART | recorded | N/A | N/A | N/A | N/A |
| tinyimagenet | Cons-AT | recorded | N/A | N/A | N/A | N/A |
| tinyimagenet | RAAT | recorded | N/A | N/A | N/A | N/A |
| tinyimagenet | RAAT++ | recorded | N/A | N/A | N/A | N/A |
| tinyimagenet | Ours | e15wt03 | 55.77 ± 0.30 | 19.52 ± 0.12 | 37.64 ± 0.13 | 32.99 |

## wrn / L2

| Dataset | Method | Candidate | Clean | AA | Mean | G |
|---|---|---|---|---|---|---|
| cifar10 | PGD-AT | recorded | N/A | N/A | N/A | N/A |
| cifar10 | TRADES | recorded | N/A | N/A | N/A | N/A |
| cifar10 | MART | recorded | N/A | N/A | N/A | N/A |
| cifar10 | Cons-AT | recorded | N/A | N/A | N/A | N/A |
| cifar10 | RAAT | recorded | N/A | N/A | N/A | N/A |
| cifar10 | RAAT++ | recorded | N/A | N/A | N/A | N/A |
| cifar10 | Ours | reference | 91.67 ± 0.07 | 71.63 ± 0.02 | 81.65 ± 0.03 | 81.03 |
| cifar100 | PGD-AT | recorded | N/A | N/A | N/A | N/A |
| cifar100 | TRADES | recorded | N/A | N/A | N/A | N/A |
| cifar100 | MART | recorded | N/A | N/A | N/A | N/A |
| cifar100 | Cons-AT | recorded | N/A | N/A | N/A | N/A |
| cifar100 | RAAT | recorded | N/A | N/A | N/A | N/A |
| cifar100 | RAAT++ | recorded | N/A | N/A | N/A | N/A |
| cifar100 | Ours | n03 | 70.75 ± 0.41 | 42.23 ± 0.16 | 56.49 ± 0.13 | 54.66 |
| tinyimagenet | PGD-AT | recorded | N/A | N/A | N/A | N/A |
| tinyimagenet | TRADES | recorded | N/A | N/A | N/A | N/A |
| tinyimagenet | MART | recorded | N/A | N/A | N/A | N/A |
| tinyimagenet | Cons-AT | recorded | N/A | N/A | N/A | N/A |
| tinyimagenet | RAAT | recorded | N/A | N/A | N/A | N/A |
| tinyimagenet | RAAT++ | recorded | N/A | N/A | N/A | N/A |
| tinyimagenet | Ours | e15w207 | 66.64 ± 0.16 | 46.83 ± 0.28 | 56.74 ± 0.19 | 55.87 |
