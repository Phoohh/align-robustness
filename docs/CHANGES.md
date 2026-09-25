# Packaging changes

## Recovered main data and completed ResNet-18 follow-up

- Added all 36 current Ours seed records, 108 ResNet-18 baseline records,
  unrounded summaries, and explicit source/coverage metadata. Kept the missing
  108 WRN baseline seed records unavailable and archived reported aggregates
  separately; historical harmonic NRR values are never relabeled as G.
- Completed existing `g19rt03` seeds 1 and 2, reusing seed 0. Selected this
  improved ResNet-18 L∞ Tiny-ImageNet candidate and retained `e15rt05`'s config
  and measurements. G still misses its fixed baseline target.
- Added a dependency-free main-table reproduction command and three checks for
  source consistency, missing coverage, duplicate/incomplete seeds, and candidate
  identity. Original study figures/data and training kernels remain unchanged.

## Completed WRN follow-up

- Reused `e15wt03` seed 0 and completed its missing seeds 1 and 2 with full AA.
  Published the three raw records, unrounded statistics, and sanitized provenance.
- Updated only the WRN L∞ Tiny-ImageNet selected configuration from `e15wt10`
  to `e15wt03`, retaining the previous configuration and historical summary.
  Mean and G exceed their fixed archived targets; AA remains below the cited
  baseline AA. No new parameter combination was introduced.
- Kept Python source and both original CIFAR-10 study figures/data unchanged.

## Study reproduction completion

- Added an explicit CIFAR-10 study launcher with the native nested class-balanced
  subset algorithm. Configurations, resume checks, and results bind the selected
  fraction and seed to an index checksum; test evaluation stays at 10,000 images.
- Added a separate fixed-protocol `raat_recorded` option after comparing the
  actual native baseline source manifest and runtime flags. The shared loss
  kernel is unchanged; Ours still accepts only without-selection semantics.
- Added the complete three-seed rho=0.001 follow-up at 10% data, reusing six
  existing reference measurements. Kept both original study figures and their
  fixed-rho table unchanged. The new rho does not improve Mean or G over RAAT.

## Report update

- Moved the project from a nested directory to the repository root so the full
  README and documented commands work from the landing page.
- Added the completed CIFAR-10/ResNet-18 studies: 42 unique per-seed measurements,
  protocol, provenance, a technical report, and a script regenerating CSV,
  Markdown, PNG, PDF, and SVG outputs.
- The rho curve uses a left Clean-accuracy axis and a right AutoAttack-accuracy
  axis, with three-seed population-SD error bars. The dataset-size table now
  includes arithmetic Mean and the main-table geometric aggregate G.
- Updated three Tiny-ImageNet evidence labels from one to three completed seeds.
  Candidate identities and all training hyperparameters are unchanged.
- Main summaries distinguish `G_of_mean_accuracies` from `per_seed_G_mean_std`.
  Existing per-seed result JSON metrics retain their original meanings.
- Kept the original model, attack, training, evaluation, and resume code unchanged.
  No new GPU training was performed for this repository update.

## Historical packaging revisions

- Revision 3 retains only the 12 without-selection configurations (36 seed runs).
  The public configuration driver rejects selection-enabled configurations.
  Reported metrics are Clean, AutoAttack, arithmetic mean, and geometric mean.
- Consolidated four snapshots into one core. The only Python-source difference
  between snapshots was the corrected resume schedule in `common/train.py`.
- Retained model definitions, classification/alignment kernels, PGD settings,
  sample selection, input perturbation reuse, and AWP computation.
- Added truthful `resnet18`/`wrn2810` aliases; legacy aliases are supported.
- Added explicit data/output paths and worker counts. Replaced the destructive
  validation-data rearrangement script with a read-only Tiny-ImageNet loader.
- Added a configuration driver, deterministic AutoAttack seed (0), full-dataset
  evaluation, finite-checkpoint checks, and per-seed JSON results. This pins a
  new evaluation seed explicitly; historical runs did not all record that seed.
- Resume now stays in the original output directory, preserves inherited best
  weights, and starts at the next epoch after the saved completed epoch. The
  original trainer could repeat its saved epoch. Exact RNG continuation across
  process restarts is not implemented in this packaging revision.
- Added import compatibility for AdverTorch 0.2.3 with modern PyTorch.
- Always save the first finite PGD-selected checkpoint, including an initial
  0%-accuracy epoch; reject nonfinite PGD errors before marking a run complete.
- Removed result dumps, scheduler wrappers tied to one cluster, backups,
  personal path prefixes, and links to external log directories.
- Replaced an old machine-specific Conda export with a minimal dependency list.
  The original cluster's complete environment lockfile was not in the archive.
- Restored the MIT license from the upstream RAAT repository, retaining its
  existing copyright notice; no author identities were added.

The seven hyperparameter values in each configuration were recovered from the
experiment submission definitions, not inferred from accuracy numbers.

## Without-selection release

- Retain only the 12 without-selection configurations, including their setting-specific hyperparameters and evidence labels.
- Restrict the configuration driver to alignment on all samples (`ALIGN_ALL_SAMPLES=1`). Classification-term boundary treatment is unchanged.
- Keep Clean, AA, arithmetic mean, and geometric mean as reported metrics.
- At revision 3, nine configurations had three-seed historical evidence and
  three Tiny-ImageNet configurations had only seed-0 evidence. The report update
  above supersedes those three evidence labels.
