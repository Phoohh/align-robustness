# Historical evidence update

The configuration files describe previously evaluated hyperparameters. They do
not certify a new GPU reproduction using the portable release environment.

The following three entries previously listed only seed 0. The completed-run
summary now records seeds 0, 1, and 2 for the **same candidate** in each entry.
Only evidence metadata was updated; no hyperparameter or candidate was replaced.

| Architecture | Norm | Dataset | Candidate | Seeds | Mean | G | Both above fixed baseline maxima? |
|---|---|---|---|---|---:|---:|---|
| ResNet-18 | L∞ | Tiny-ImageNet | e15rt05 | 0,1,2 | 32.44000 | 27.91259 | No |
| WRN-28-10 | L∞ | Tiny-ImageNet | e15wt10 | 0,1,2 | 36.39833 | 32.42240 | No |
| WRN-28-10 | L2 | Tiny-ImageNet | e15w207 | 0,1,2 | 56.73667 | 55.86508 | Yes |

Here G = sqrt(mean(Clean) × mean(AA)), using unrounded seed means in the original
summary. The numbers above are the reported rounded summary, not reconstructed
individual seeds. Per-seed raw records for these three entries are not bundled
in this update. The evidence source is the completed geometric-result summary
provided with the experiment logs.

All 12 configuration entries now have recorded three-seed evidence. This count
measures completed evaluations; it is not a count of settings that beat all
baselines. Only the separate CIFAR-10 sensitivity studies include their complete
per-seed measurements in this repository.

Changing evidence metadata changes the configuration hash used by the portable
driver. To resume a portable run created with an older release, retain its exact
original configuration; do not edit that run's configuration to bypass the hash
check.
