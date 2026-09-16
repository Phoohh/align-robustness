# Alignment-driven adversarial weight perturbation

Code for Align-only **without data selection** on CIFAR-10, CIFAR-100, and Tiny-ImageNet
with ResNet-18 and WideResNet-28-10 under L∞ and L2 attacks.

This release contains 12 setting-specific without-selection configurations:
2 architectures × 3 datasets × 2 attack norms, each run with seeds 0, 1, and 2.
Each run trains a model, selects its checkpoint by PGD accuracy, and then runs
clean evaluation plus **full standard AutoAttack on all 10,000 test examples**.

## Install

Use a separate Python 3.11 or 3.12 environment. Install matching PyTorch and
torchvision builds for your CUDA driver. The CPU smoke checks for this release
use PyTorch 2.5.1 and torchvision 0.20.1; this is not a recovered lockfile of the
historical cluster environment.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Example for a CUDA 12.1 compatible system:
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

See [PyTorch's version-specific installation commands](https://pytorch.org/get-started/previous-versions/)
for other CUDA versions or a CPU-only environment. Full experiments need a GPU;
select one visible GPU with `CUDA_VISIBLE_DEVICES=0` when several are installed.

## Data

Pass your dataset directory with `--data-root`. CIFAR-10/100 download through
torchvision on first use. For Tiny-ImageNet, extract the original
`tiny-imagenet-200` directory inside that root. The validation loader accepts
both the original `val/images` + `val_annotations.txt` layout and a validation
tree already grouped by class. It reads the data without moving images.

The input range is [0, 1]. The experiment pipeline uses random crop with padding
4 and horizontal flip for training, with two independently augmented views;
evaluation uses only conversion to a tensor.

## Run

First print the exact commands and flags without starting training:

```bash
python run_experiment.py --config configs/without_selection/r18_Linf_cifar10.json --seeds 0 --dry-run
```

Run one configuration with seeds 0, 1, and 2:

```bash
CUDA_VISIBLE_DEVICES=0 python run_experiment.py \
  --config configs/without_selection/r18_Linf_cifar10.json \
  --seeds 0 1 2 --data-root ./data --output-root ./runs
```

Seeds execute sequentially. For independent GPU allocations, run separate
commands with different `--seeds` values. Choose another dataset, architecture,
or norm from `configs/without_selection/`, for example:

```bash
CUDA_VISIBLE_DEVICES=0 python run_experiment.py \
  --config configs/without_selection/wrn_L2_cifar100.json \
  --seeds 0 1 2 --data-root ./data --output-root ./runs
```

Training completion automatically launches standard AutoAttack
(`apgd-ce`, `apgd-t`, `fab-t`, `square`). Evaluation has no test-subset option.
Each seed writes checkpoints, logs, configuration metadata, and `result.json`
under `runs/<experiment_id>/seed<N>/`.

Continue an interrupted run by repeating its command with `--resume`.
The configuration must match the existing directory. The driver retains the
best checkpoint in that directory, restores the global learning-rate schedule,
and continues from the next epoch. Completed evaluations are skipped.
For this release, exact bitwise equivalence across an interruption is not claimed.

## Configurations and method semantics

[configs/index.tsv](configs/index.tsv) lists all 12 settings and their seven
hyperparameters. The JSON files specify the same values. These without-selection
configurations preserve the recorded per-setting hyperparameters. Nine have
recorded three-seed evaluations; R18/L∞/Tiny-ImageNet, WRN/L∞/Tiny-ImageNet,
and WRN/L2/Tiny-ImageNet currently have only seed-0 evidence in this release.
Their additional seeds are still being completed. These evidence labels describe the existing experiments, not a
new reproduction with this packaged environment, and do not imply that every
setting outperforms all baselines.

| Parameter | Role |
|---|---|
| `gamma` | Weight-perturbation magnitude, scaled by each parameter tensor's norm |
| `awp_start` | First epoch using adversarial weight perturbation |
| `lam` | Alignment-loss coefficient |
| `bd_range` | Fraction of the input perturbation used for boundary treatment |
| `bd_alpha` | Symmetric Beta-distribution shape used to sample the mixing coefficient |
| `lr` | Initial SGD learning rate |
| `weight_decay` | SGD weight decay |

The published configurations use 110 epochs, batch size 128, PGD-10 training,
momentum 0.9, and learning-rate drops by 0.1 after epochs 100 and 105.
L∞ uses epsilon 8/255 and step size 2/255; L2 uses epsilon 128/255 and
step size 32/255. Alignment temperature is 0.5.

**Align-only names the objective used to construct the weight perturbation.**
The optimizer still trains with both classification and alignment losses.
Alignment includes every sample, without a correctness-based selection mask.
The driver fixes `ALIGN_ALL_SAMPLES=1` and `RAAT_DATA_SELECTION=1` and only
accepts `selection: "no"`. Here the selection label refers to alignment-sample
selection; `RAAT_DATA_SELECTION=1` preserves the inherited classification-term
boundary treatment. Boundary treatment in the classification term
remains active.

The inherited input-attack path attacks the first augmented view and transfers
its detached perturbation to the second view with clipping. That computational
path is retained. Older exploratory branches remain in the inherited trainer;
the configuration driver explicitly disables them, preventing inherited shell
variables from changing an experiment.

The actual architectures are ResNet-18 with `BasicBlock` and WRN-28-10. Historical
names `pre_resnet18` and `wrn3410` remain accepted aliases for checkpoint/source
compatibility. Tiny-ImageNet uses first-convolution stride 2 in both models.

## Summarize

```bash
python summarize.py ./runs
```

For clean accuracy C and AutoAttack accuracy R (both in percent):

- Mean = (C + R) / 2
- Geometric mean = sqrt(CR)

The summary computes every metric **per seed first**, then reports its mean
and population standard deviation (`ddof=0`). Missing seeds remain labeled
`partial`. Computing sqrt(mean(C) × mean(R)) from an existing table is a
different aggregation and need not equal the average of per-seed geometric means.

## Source and validation

Model definitions and loss/training kernels are retained from the provided
experiment snapshots. Packaging changes are described in
[docs/CHANGES.md](docs/CHANGES.md), with source checksums in
[docs/source_manifest.json](docs/source_manifest.json).
The entry points, data paths, and result collection are portable; cluster
accounts, personal paths, job IDs, old result dumps, and backup scripts are not
part of this repository.

```bash
python -m unittest discover -s tests -v
```

These are configuration, CPU numerical, and small synthetic-data workflow
checks. They do not repeat the 110-epoch GPU experiments or reproduce their
reported accuracies. Datasets, trained weights, and full experiment logs are
not bundled.

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for source attribution and
the license status of the supplied snapshot.
