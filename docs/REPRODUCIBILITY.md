# Reviewer reproduction guide

This snapshot contains Align-only **without alignment-sample selection**.
It provides runnable full-data training configurations and analysis of recorded
CIFAR-10 sensitivity measurements. The coverage below distinguishes those two
uses so that a plotted measurement is not mistaken for a newly reproduced run.

## Obtain and inspect the snapshot

Download the source ZIP supplied with this artifact or from its anonymous
repository page, extract it, and open a terminal in the directory containing
`README.md`, `run_experiment.py`, and `configs/`. Git and access to an author
account are not required. No credentials are needed by the training code.

Keep the original ZIP as a record of the version being reviewed. Run commands
from this extracted directory and write new results under `runs/`.

## Quick checks without a GPU

Use Python 3.11 or 3.12. These checks use only the Python standard library;
they do not download datasets or run training:

```bash
python run_experiment.py --config configs/without_selection/r18_Linf_cifar10.json --seeds 0 --dry-run
python studies/reproduce_figures.py --no-plots --output-dir ./runs/recomputed-study
python studies/reproduce_main_results.py --output-dir ./runs/recomputed-main
python -m unittest discover -s tests -p 'test_report_results.py' -v
```

The study command should print
`VALID_STUDY_SEEDS=42/42 RHO_POINTS=7/7 DATA_SIZE_ROWS=8/8`.
The recorded-result test file contains five tests. The two CSV summaries and
the Markdown table in `runs/recomputed-study/` can be compared with their
counterparts in `results/cifar10_resnet18/`. These checks validate aggregation
and data consistency, not the measured accuracies of a newly trained model.
The main-comparison command should print
`OURS_SEEDS=36/36 BASELINE_SEEDS=108/216 COMPLETE_GROUPS=48/84`.
Its coverage CSV leaves missing WRN baseline seed groups explicit.

To render both figures as well, install the plotting dependencies:

```bash
python -m pip install -r studies/requirements.txt
python studies/reproduce_figures.py --output-dir ./runs/recomputed-study
```

This produces PNG, PDF, and SVG files. The rho plot has Clean on the left and
Robust (full standard AutoAttack) on the right. The dataset-size table includes
both methods' Clean, AA, arithmetic Mean, and geometric G.

## Reproduction coverage

| Artifact or experiment | Included capability | Remaining limitation |
|---|---|---|
| CIFAR-10 rho plot and dataset-size table | All 42 unique measured seed records, recorded protocol, CSV/Markdown aggregation, and figure generation | Historical measurements, not newly certified training results in the packaged environment |
| Ours: 12 architecture/dataset/norm settings | Full-data training, resume, PGD-best checkpoint selection, full standard AA, and all 36 recorded seeds | Native evidence varies from recovered tables to checked training logs; no new full training reproduction with the portable package |
| Main baselines | 108 ResNet-18 seed records and separately archived WRN reported aggregates | 108 WRN baseline seed records remain unavailable |
| Dataset-size training at 10%, 20%, and 50% | `studies/run_study.py`, the native nested subset algorithm, index checksums, and full test-set evaluation | No new 110-epoch accuracy reproduction of the portable entry point is claimed |
| RAAT comparison | Recorded results and a distinct `raat_recorded` training option using the audited baseline path | Targets the measured native snapshot; no claim of identity with every upstream RAAT revision |
| 10% rho follow-up | Three new measured seeds at rho=0.001, compared with six reused reference seeds | Follow-up after observing the original result; Mean and G still trail RAAT |
| Tiny-ImageNet L∞ follow-ups | Missing seeds completed for existing WRN and ResNet-18 candidates, with all raw measurements | WRN exceeds both fixed aggregate targets; ResNet-18 still misses G; comparisons informed by previous results |
| Trained checkpoints, raw logs, and historical environment | Source checksums, configuration evidence, and validation records | Weights, complete raw logs, and an exact historical environment lockfile are not included |

The two CIFAR-10 sensitivity studies and the separate 10% follow-up include their complete per-seed
measurements in this snapshot. Configuration evidence for the 12 main settings
records historical completion; it does not mean that all settings beat their
baselines. See [EVIDENCE.md](EVIDENCE.md). This snapshot does not claim coverage
of every experiment that may appear in a manuscript.
The [main data catalog](../results/main_without_selection/README.md) lists source
coverage and differentiates archival measurements from freshly checked logs.

## Train and evaluate Ours

Full experiments are intended for a Linux system with an NVIDIA GPU and a
compatible CUDA/PyTorch installation. Follow the environment and dataset
instructions in the [main README](../README.md). The suggested package versions
are a portable setup, not a recovered lockfile for the historical experiments.

```bash
CUDA_VISIBLE_DEVICES=0 python run_experiment.py \
  --config configs/without_selection/r18_Linf_cifar10.json \
  --seeds 0 1 2 --data-root ./data --output-root ./runs
```

Each seed runs 110 training epochs and then evaluates all 10,000 test images
with Clean and standard AutoAttack (`apgd-ce`, `apgd-t`, `fab-t`, `square`).
The checkpoint is chosen by PGD accuracy. The driver uses the same configuration
and seed to gate evaluation after training completion. Append `--resume` to
continue an interrupted run; retain that run's original configuration.

To evaluate seed 0's checkpoint separately after that run completes:

```bash
CUDA_VISIBLE_DEVICES=0 python eval.py \
  --config configs/without_selection/r18_Linf_cifar10.json \
  --checkpoint ./runs/r18_Linf_cifar10_without_selection_reference/seed0/best.model \
  --seed 0 --data-root ./data \
  --output ./runs/recheck-seed0/result.json --require-training-complete
```

Use a new output path: the evaluator refuses to overwrite an existing result.
For a different setting, use its matching configuration, seed, and checkpoint.
`python summarize.py ./runs` summarizes completed result files; avoid placing
duplicate evaluations of the same seed in the directory being aggregated.

For the data-size study, the following paired commands use identical subsets
at each seed. Both retain all 10,000 test images and automatically run standard AA:

```bash
python studies/run_study.py --percent 10 --rho 0.002 --seeds 0 1 2 --data-root ./data
python studies/run_study.py --method raat_recorded --percent 10 --seeds 0 1 2 --data-root ./data
```

Change `--percent` to 20, 50, or 100 for the other fractions; use 100 for the
full-data rho sweep. Add `--dry-run` to inspect without starting training.
See [study instructions](../studies/README.md) and the
[recorded baseline audit](BASELINE.md) for provenance and method labels.

## Validation status and interpretation

With the full dependencies installed, run:

```bash
python -m unittest discover -s tests -v
```

The complete-dependency check of the earlier source passed all 20 tests then
present with no skips. The main-data update adds three standard-library tests;
its local check passed 16 tests and skipped seven for missing PyTorch, as recorded
in [validation_main_20260925.json](validation_main_20260925.json). On an
A100 GPU, three native-baseline single-step cases produced identical parameters
and gradients; the 12 fraction/seed subsets matched their native records (the
100% cases retain the complete original order). A real ResNet-18 completed two
small training epochs, including resume, with four examples per epoch and a
finite saved checkpoint. This is a workflow check, not an accuracy reproduction.
Exact versions, source checksums, and limitations are recorded in
[validation_reproduction_20260923.json](validation_reproduction_20260923.json).

The suite contains configuration checks, recorded-result checks, numerical
checks, and short synthetic-data workflow checks. In the documented Python
3.12 local integration environment for the earlier report update, 7 tests passed and 7 were skipped because
PyTorch was absent; all 36 configuration/seed dry runs passed. A partially
installed training environment can instead fail imports. Skips do not certify
the corresponding numerical or training workflows. Exact records and the
initial incomplete-dependency run are retained in
[validation_local_20260922.json](validation_local_20260922.json).

Neither the quick checks nor the synthetic tests reproduce a 110-epoch GPU
experiment or certify its reported accuracy. Clean, AA, and paired-seed Mean
use population SD. Main-table G is `sqrt(mean(Clean) * mean(AA))`, a single
aggregate without an assigned SD; the per-seed geometric summary is separate.

Upstream licensing and source attributions are retained in
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) and [LICENSE](../LICENSE).
