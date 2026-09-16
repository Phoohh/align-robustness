# Packaging changes

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
- Nine configurations have three-seed historical evidence; three Tiny-ImageNet configurations remain seed-0-only in the supplied records.
