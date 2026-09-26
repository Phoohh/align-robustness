# Recorded RAAT baseline

The `raat_recorded` study option targets the native baseline snapshot used in
the supplied CIFAR-10 / ResNet-18 / L∞ measurements. It is separate from Ours,
which remains Align-only without alignment-sample selection. The comparison
tables retain their historical RAAT label.

The baseline's source manifest was compared with the snapshot recorded at
study creation: all 38 original Python source checksums matched. The study
snapshot adds the class-balanced subset hook. Its model definitions, optimizer,
input attacks, augmentation, and training loss come from those recorded sources.

The native baseline loss file has SHA-256
`6cea7f1734d2f943bc2e4a6e5bf6781fd6a24b8918ba41e14454113d0628c662`.
The shared packaged loss file has SHA-256
`032bb2980a8bde896c190b5129dede0f196663c4d1222c88893e51a35e5c9cc9`.
Their only source difference is the switch selecting every alignment sample
and an explicit integer dtype for the resulting indices. Disabling that
switch retains the native baseline's correctness-based alignment selection.
The classification boundary treatment is preserved in both methods.

The actual three-seed baseline training records use `AWP_TRAIN=0`,
`AWP_OBJECTIVE=full`, and `ALIGN_ALL_SAMPLES=0`. The new entry point sets these
explicitly, enables the inherited classification boundary treatment, and
disables exploratory flags. Its configuration is restricted to the recorded
110-epoch CIFAR-10 / ResNet-18 / L∞ protocol, with learning rate 0.1, weight
decay 0.0005, lambda 1, boundary range 0.1, and boundary alpha 0.75. The chosen
data fraction and seed determine the same nested subsets used by Ours.

The recorded snapshot reuses the first augmented view's detached input
perturbation on the second view. This inherited path is retained. The source
audit identifies the measured snapshot; it does not establish byte identity
with every upstream RAAT revision or claim a fresh 110-epoch reproduction.
Upstream licensing and attribution remain in
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

```bash
python studies/run_study.py --method raat_recorded --percent 10 --seeds 0 1 2 --dry-run
```

Ours with rho=0 follows its all-sample alignment path and must not be relabeled
as RAAT. Results and configurations carry distinct method and selection labels.

The [validation record](validation_reproduction_20260923.json) compares this
unmodified native kernel against the portable path using identical weights,
inputs, and random states. With all, some, or none of the samples eligible for
alignment, one optimizer step gave identical parameters and gradients. This
small numerical check does not establish end-to-end bitwise reproducibility.
