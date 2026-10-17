# Selection-free SAAR: validation-controlled revision

This protocol is separate from the historical measurements in `results/` and
the schema-1 configurations in `configs/without_selection/`. Those measurements
retain classification-side boundary treatment and a test-selected checkpoint;
they are not results for this revision. Historical records are preserved.

## Method

The SAAR kernel uses every sample for both classification and alignment. It
does not identify boundary samples, apply correctness masks, or scale input
perturbations according to predictions. Both adversarial views contribute
full-budget mean CE. The alignment loss, temperature 0.5, folded Beta sampling,
relative per-tensor weight perturbation, and BatchNorm behavior are retained.

The main configuration uses independent PGD perturbations for both augmented
views: concatenate the views into a 2B batch and optimize one perturbation per
input. The separately labeled shared ablation attacks the first view and
transfers its detached perturbation to the second. Neither is a weight
perturbation; input attacks and SAAR weight perturbations are separate steps.

## Development and final reporting

1. Split the original training set into 90% training and 10% validation,
   stratified by class with fixed split seed 2027. The split is shared across
   candidate configurations and training seeds. Validation uses only ToTensor.
   Save exact indices, class counts, and hashes. Never construct the official
   test dataset during this stage (Tiny-ImageNet's official validation is the
   final evaluation set).
2. Prespecify lambda=1, SGD momentum=0.9, LR=0.1, weight decay=0.0005,
   Beta shape=0.75, AWP start=10, batch size=128, and 110 epochs. Compare
   rho in {0.001, 0.002, 0.005}, initially using training seed 0. Select by
   validation PGD accuracy; exact ties choose the smaller rho. These are new
   candidates, not historically selected per-setting configurations.
3. Write an immutable validation selection lock. The final configuration must
   match its complete training specification, view construction, and split
   definition. Do not use test Clean/AA/Mean/G to change the lock.
4. For the selected configuration, initialize new models for seeds 0,1,2 and
   train on the complete original training split for exactly 110 epochs. Do not
   construct validation or test loaders during final training. Select the
   fixed last epoch, not the best test or validation epoch.
5. Evaluate that checkpoint with Clean and full standard four-attack AutoAttack
   on all 10,000 official evaluation images. Use the existing definitions of
   Mean and geometric G. Report every selected seed, including unfavorable
   outcomes. Any subsequent test-informed change is a new exploratory study,
   not a replacement confirmatory run.

Linf uses epsilon 8/255 and step 2/255; SAAR L2 uses epsilon 128/255 and step
32/255. Both use PGD-10 with random initialization. LR decays by 0.1 after
epochs 100 and 105. Development checkpoints are chosen only on held-out PGD;
their scores are not final AutoAttack measurements.

## Commands

Run from the repository root, with the dependencies in `requirements.txt`:

```bash
python -m protocols.selection_free_v2.prepare
python run_experiment.py --config protocols/selection_free_v2/configs/v2_r18_Linf_cifar10_independent_rho0p001_dev.json --seeds 0 --data-root data --output-root runs/v2
```

Repeat for the three rho candidates. Development runs never automatically
launch final-test evaluation. After all candidates finish:

```bash
python -m protocols.selection_free_v2.select --configs protocols/selection_free_v2/configs/v2_r18_Linf_cifar10_independent_rho0p001_dev.json protocols/selection_free_v2/configs/v2_r18_Linf_cifar10_independent_rho0p002_dev.json protocols/selection_free_v2/configs/v2_r18_Linf_cifar10_independent_rho0p005_dev.json --run-root runs/v2 --seeds 0 --output runs/v2/selection_lock.json --final-config runs/v2/selected_final.json
python run_experiment.py --config runs/v2/selected_final.json --selection-lock runs/v2/selection_lock.json --seeds 0 1 2 --data-root data --output-root runs/v2
```

Both stages resume only within their own configuration-bound output directories.
Archived checkpoints cannot be reused as new-protocol training runs. The final
driver binds the validation lock to training completion and final evaluation.

## Comparisons and remaining experiments

The required ablation crosses weight perturbation off/on with input attacks
shared/independent, keeping all sample handling disabled and all other choices
matched. Independent+on is the main method. Report compute cost as well as
accuracy because input attack construction changes attack computation.

The official RAAT comparison retains the pinned upstream objective, including
its boundary treatment and alignment selection, and uses independent attacks.
Only the outer validation/final-checkpoint protocol is unified. The audited
historical shared-perturbation snapshot is a recorded variant, not proof that
the official code shared perturbations. Upstream RAAT's L2 default step is
15/255; this is preserved and disclosed for that baseline rather than silently
changed to SAAR's 32/255. See `docs/OFFICIAL_RAAT_AUDIT.md`.

The previous rho and training-size figures remain historical. New-protocol
training-size subsets must be specified separately; the driver rejects the
legacy subset hook for this protocol. Pending results must not be filled with
old measurements or zeros. The manuscript working copy labels legacy tables
and adds pending revised comparisons until real results become available.
