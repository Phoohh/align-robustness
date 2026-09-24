# Alignment-driven adversarial weight perturbation

## Scope

This report describes the released Align-only **without data selection** method
and two completed sensitivity studies on CIFAR-10 with ResNet-18. The repository
also contains setting-specific training configurations for CIFAR-100,
Tiny-ImageNet, and WRN-28-10 under L∞ and L2 attacks. Their configuration evidence
records completed historical evaluations; it does not imply that every setting
beats every baseline.

## Method and implementation

The training batch provides two independently augmented views of each image.
The inherited input-attack path generates a PGD perturbation on the first view
and transfers its detached perturbation to the second view with clipping.
The alignment term compares the prediction on an interpolation of adversarial
views with the interpolated clean-view predictions using the inherited
Jensen–Shannon loss. Classification-term boundary treatment is retained.

Align-only specifies the loss used to **construct the adversarial weight
perturbation**, not the entire optimizer objective. With alignment gradient
g_l for a trainable parameter tensor W_l, the implementation adds

    delta W_l = rho × ||W_l||₂ × g_l / (||g_l||₂ + 1e-12).

This perturbation is applied only to tensors with more than one dimension.
At the perturbed weights, the code recomputes the classification-plus-alignment
loss, backpropagates, restores the original weights, and performs the optimizer
step. The configured start epoch determines when this AWP branch becomes active.

Without selection means that the alignment term includes all samples rather
than using its correctness-based selection mask. The driver fixes
`AWP_OBJECTIVE=align_only` and `ALIGN_ALL_SAMPLES=1`; the inherited
classification-term boundary treatment remains enabled. The rho study changes
only the tensor-relative perturbation magnitude.

## Experimental protocol

Both studies use CIFAR-10, ResNet-18, L∞ epsilon 8/255, 110 training epochs,
and seeds 0, 1, and 2. The best checkpoint is selected by the training pipeline's
PGD evaluation and evaluated with full standard AutoAttack on all 10,000 test
images. Robust accuracy here always refers to this AutoAttack evaluation.
The full parameter settings and subset construction are given in
[the study protocol](../studies/README.md).

The data-size comparison uses the original RAAT baseline with its original
selection and no AWP. Ours is Align-only without selection with rho = 0.002.
The subsets are class balanced, nested within each seed, and matched across
methods. All fractions use 110 epochs; smaller fractions therefore have fewer
optimizer updates. Six full-data reference seeds are reused from the completed
reference experiments, and 36 new seeds complete these studies.

## Results

### Impact of rho

![Impact of rho](../results/cifar10_resnet18/impact_of_rho.png)

Across the tested rho values, mean robust accuracy rises from 48.05% at rho = 0
to 49.82% at rho = 0.01, an increase of 1.77 percentage points. Mean clean
accuracy remains between 85.06% and 85.73%. Rho = 0.01 has the highest observed
robust accuracy in this grid, while rho = 0.005 has the highest observed clean
accuracy. These observations do not establish an optimum outside the tested
range. The plot uses separate, labeled y-axis scales for the two accuracies.

### Impact of training dataset size

![Impact of dataset size](../results/cifar10_resnet18/impact_of_dataset_size.png)

The [full table](../results/cifar10_resnet18/impact_of_dataset_size.md) reports
Clean, Robust (AA), arithmetic Mean, and geometric G for both methods.
Relative to RAAT, Ours has higher mean robust accuracy at all four fractions:
approximately +1.28, +1.32, +2.15, and +0.50 percentage points at 100%, 50%,
20%, and 10%, respectively. The largest observed gain occurs at 20%.

The advantage is not uniform across metrics. At 10%, Ours has lower mean
clean accuracy (61.54% versus 63.48%), despite its higher mean robust accuracy.
Ours has higher arithmetic Mean and geometric G at 100%, 50%, and 20%; RAAT
has higher values for both aggregates at 10%.
The clean-accuracy spread is also larger at this fraction. These descriptive
results from three seeds do not by themselves establish statistical significance.

### Follow-up at 10% training data

After inspecting the 10% result, three additional runs tested rho = 0.001 with
the same seeds, matched subsets, and all other parameters. They give Clean
61.96 ± 2.30, AA 28.07 ± 0.95, Mean 45.01 ± 1.20, and G 41.70. Relative to
rho = 0.002, Clean and Mean increase slightly, while AA and G decrease. Mean
and G remain below RAAT. This follow-up does not replace a row in the original
fixed-rho study. Its [nine-row comparison and raw seeds](../results/cifar10_resnet18_10pct_followup/comparison.md)
include six reused reference records and three new records. All three new
runs completed 110 epochs and full standard AA; their test set and subset
identity were checked against the original paired runs.

### WRN / Tiny-ImageNet / L∞ follow-up

The historical `e15wt03` candidate now has three completed seeds after reusing
seed 0 and training seeds 1 and 2. Clean is 55.77 ± 0.30, AA 19.52 ± 0.12,
Mean 37.64 ± 0.13, and G 32.99. Both aggregates exceed their respective fixed
archived baseline targets; AA remains below the cited baseline AA of 20.69.
This is a descriptive, history-informed comparison, not a significance claim.
The [complete seeds and evidence](../results/wrn_tinyimagenet_linf_followup/README.md)
record 110-epoch completion, final LR 0.001, and all four standard AA stages.
The released setting now uses `e15wt03`; its previous `e15wt10` configuration
and summary are retained. The original CIFAR-10 studies are unchanged.

### Metric definitions

For clean accuracy C and robust accuracy R, Mean = (C + R)/2. We compute this
within each seed and report its mean and population SD, along with Clean and R.
The displayed geometric aggregate is G = sqrt(mean(C) × mean(R)), matching the
main results table. It is distinct from the mean of sqrt(C_seed × R_seed),
which the numerical CSV and main summary script provide separately.

## Reproducibility and limits

The repository includes the original model/training kernels, portable training
and full-AA entry points, 12 without-selection configurations, the 42 study
measurements, and a script to regenerate both visuals and numerical tables.
A portable subset-training entry point now includes a separate recorded-RAAT
baseline option, based on a source and environment audit of the native snapshot.
See [the baseline audit](BASELINE.md) for its scope and inherited input-attack path.

The supplied measurements were collected with the native experimental snapshots.
The complete historical environment lockfile, datasets, raw cluster logs, and
trained checkpoints are not bundled. Packaging tests check configuration handling,
metrics, and the recorded-result analysis; they do not repeat 110-epoch GPU
training or certify an exact numerical reproduction in a new environment.
The original training and loss kernels were retained in this report update.

This report covers the two sensitivity studies. It does not claim complete
baseline dominance across all 12 settings or replace the main comparison table
with a selection of sensitivity-study results.
