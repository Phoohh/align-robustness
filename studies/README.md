# CIFAR-10 / ResNet-18 sensitivity studies

This directory reproduces the **analysis and figures from recorded measurements**.
It does not launch the historical cluster jobs or retrain RAAT. The portable
training driver also does not implement the class-balanced data-fraction
selection used in this study: its dataset loaders train on the full training
set. Changing rho in a full-data configuration does not recreate a 10%, 20%,
or 50% experiment. The subset construction below documents the recorded
protocol; it is not a packaged subset-training entry point.

See the [reproduction coverage table](../docs/REPRODUCIBILITY.md) before choosing
between validating recorded results and starting new training.

## Regenerate the curve and table

From the repository root:

```bash
python -m pip install -r studies/requirements.txt
python studies/reproduce_figures.py
```

The command validates all 42 unique seed records, then writes CSV summaries,
the dataset-size Markdown table, and PNG/PDF/SVG versions of both visuals under
`results/cifar10_resnet18/`. Use `--output-dir /path/to/output` to write elsewhere.
For a dependency-free numerical check, use:

```bash
python studies/reproduce_figures.py --no-plots --output-dir ./runs/study-summary
```

The rho plot uses **Clean accuracy on the left y-axis** and **Robust accuracy
(AutoAttack) on the right y-axis**. The axes have different scales. Points are
three-seed means and error bars are population standard deviations.

## Shared protocol

| Item | Value |
|---|---|
| Dataset and architecture | CIFAR-10, ResNet-18 BasicBlock [2,2,2,2] |
| Threat model | L∞, epsilon = 8/255 |
| Training attack | PGD-10, step size = 2/255 |
| Optimization | SGD, momentum 0.9, initial LR 0.1, weight decay 0.0005 |
| Duration and schedule | 110 epochs, LR drops at epochs 100 and 105 |
| Batch size | 128 |
| Alignment | lambda = 1, temperature = 0.5 |
| Boundary treatment | range = 0.1, Beta-distribution alpha = 0.75 |
| AWP start | Epoch 10 |
| Seeds | 0, 1, 2 |
| Checkpoint | PGD-best checkpoint from a completed run |
| Evaluation | Clean and full standard AutoAttack on all 10,000 test images |

Standard AutoAttack comprises APGD-CE, APGD-T, FAB-T, and SQUARE. These are
historical measurements from the native experiment snapshots; the complete
historical environment lockfile was not supplied. The portable main-training
entry point has separate validation and should not be described as having
reproduced these accuracies in its current environment.

## Impact of rho

Ours is **Align-only without selection**. Rho is the parameter called `gamma`
in the public JSON configuration, and `AWP_GAMMA` in the training environment.
The values are 0, 0.0005, 0.001, 0.002, 0.003, 0.005, and 0.01; all other
hyperparameters are fixed and all 50,000 training images are used.

Rho = 0 means zero weight-perturbation magnitude in the same Align-only code
path. It does not stand for the separately run RAAT baseline.

## Impact of dataset size

Both methods use 100%, 50%, 20%, or 10% of the training images, corresponding
to 50,000, 25,000, 10,000, or 5,000 images. The test set remains unchanged.

- Ours: Align-only without selection, rho = 0.002.
- RAAT: the original RAAT implementation with its original sample selection
  and AWP disabled. The portable training configurations in this release are
  only for Ours; the separate native RAAT training harness is not included.

Subsets are class balanced and nested within each seed. The same subset is used
by both methods at a given seed and fraction. The original construction groups
the 50,000 training indices by class, shuffles each class with a local
`random.Random(seed)`, takes the first `50 × percent` indices from each class,
then sorts the combined indices. At 100%, the original full order is retained.
This selection does not consume the training RNG state.

Each fraction is trained for **110 epochs**, so smaller subsets receive fewer
optimizer updates. This is a fixed-epoch data-size study, not a fixed-update
or fixed-compute comparison.

## Aggregation and provenance

Clean, Robust (AA), and Mean are computed per seed, then summarized as mean ±
population SD (`ddof=0`). Mean = (Clean + AA)/2. Its SD is calculated from the
paired seed values, not by averaging the two component SDs.

The displayed **G** is sqrt(mean(Clean) × mean(AA)), matching the main table.
No SD is attached to this single aggregate. The CSV also provides the mean and
SD of per-seed geometric means in separate, explicitly named columns.

The 42 records contain 36 new study seeds and six archived 100% reference seeds.
The three Ours reference seeds at rho = 0.002 belong to both studies and are
counted once. `per_seed.csv` preserves every recorded accuracy and its source
category. Private log paths were removed; raw logs and checkpoints are not
included. `provenance.json` records the input archive checksum and the removed
field. `protocol.json` preserves the recorded protocol and defines aggregation.

The script rejects duplicate, missing, nonfinite, and out-of-range results.
It never fills missing runs with zeros or generates synthetic observations.
