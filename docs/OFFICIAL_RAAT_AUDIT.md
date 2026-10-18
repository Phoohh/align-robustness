# Official RAAT baseline audit

`raat_official` is the upstream RAAT training objective, separately named from
the archived `raat_recorded` variant. The archived variant transfers the first
view's perturbation to the second; it must not be presented as this baseline.

## Pinned source and attribution

Source repository: [FlaAI/RAAT](https://github.com/FlaAI/RAAT), commit
`61de35b7e2db141e2904eee59460a6d98964525d`, the
`RAAT_Code_(Sec_5.1&5.2)` implementation. The upstream MIT license is retained
in [LICENSE](../LICENSE) and attribution in
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

| Upstream file | SHA-256 |
|---|---|
| [training/adv_train/adv_train_bd.py](https://github.com/FlaAI/RAAT/blob/61de35b7e2db141e2904eee59460a6d98964525d/RAAT_Code_%28Sec_5.1%265.2%29/training/adv_train/adv_train_bd.py) | `b38de0879c4c018493946f4cde9b2b320861ede1dd99e0575ceca49bf388e091` |
| [training/__init__.py](https://github.com/FlaAI/RAAT/blob/61de35b7e2db141e2904eee59460a6d98964525d/RAAT_Code_%28Sec_5.1%265.2%29/training/__init__.py) | `a168b4148d6e91a78a4a66a5973ed59109c90ee853234b850e8613a21d39ce91` |

The verbatim sources are embedded as test fixtures in
[test_official_raat.py](../tests/test_official_raat.py). Fixture checksums are
verified before numerical comparison; tests require no network or private
experiment archive.

## Preserved objective and attack construction

[raat_official.py](../training/adv_train/raat_official.py) retains the following
upstream behavior:

- Concatenate the two independently augmented views into a `2B` batch and call
  PGD on that batch. Every row has its own input perturbation; no first-view
  perturbation is transferred to the second view. A single batched call is
  independent per-view optimization, rather than a shared perturbation.
- In evaluation mode, identify clean-misclassified, boundary, and non-boundary
  views. A boundary view is clean-correct, full-adversarial-wrong, and
  reduced-adversarial-wrong. The reduced input is
  `x + BD_boundary_range * (x_adv - x)`; it is not a separately optimized attack.
- Construct classification inputs in the original order of groups:
  non-boundary full attacks, boundary reduced attacks, and clean-misclassified
  full attacks. Original labels and ordinary mean cross-entropy are retained.
- Include an original sample in alignment only when both augmented clean
  views are classified correctly. This is the upstream correctness selection;
  it remains active regardless of the SAAR all-sample environment setting.
- Draw one Beta coefficient per minibatch and fold it as `max(beta, 1-beta)`.
  Compare the prediction on the interpolated adversarial input against the
  interpolated clean **logits**, with temperature applied afterward. Both
  prediction branches remain differentiable.
- Preserve the upstream JS objective: ordinary `softmax(logits/T)`, the arithmetic
  probability mixture, the upstream `1e-8` lower bound only on that mixture before
  its log, and the mean of the two KL divergences. Compute each target entropy
  using `log_softmax(logits/T)` to avoid a nonfinite derivative when softmax
  underflows a target probability to zero. This is an algebraically equivalent
  numerical evaluation, rather than a new probability/logit clamp. The helper
  is local to this baseline.
  The archived package's `_safe_logits`, logit clipping, and `nan_to_num`
  substitutions are not used because they would change the objective.
- Perform one ordinary loss backward/update and the original scheduler step.
  There is no adversarial weight perturbation, AWP recomputation, or SAAR loss.
  The original training/evaluation modes and BatchNorm update behavior remain.

The original `loss = loss_ce; loss += loss_con` expression is also preserved.
Consequently, the upstream `train/loss_cls` log contains the combined objective
when alignment is present; this historical logging behavior is not a new
definition of the cross-entropy component or a model-selection metric.

## Local adaptations and explicit pipeline differences

The local kernel changes dependencies, device/interface handling, failure
checks, and the numerical evaluation of target entropy relative to the pinned
implementation, while preserving its objective:

1. Import `torch` explicitly and use the local mathematically equivalent JS
   helper instead of importing the modified archived helper. Continue using
   the package's unchanged `AverageMeter` utility. The verbatim upstream helper
   remains embedded in the numerical test fixture.
2. Use the model parameter's device instead of a module-global CUDA choice.
   This supports CPU validation on hosts that also have a GPU.
3. Reject nonzero weight perturbation or shared-attack requests, invalid finite
   scalar parameters, empty loaders, and nonfinite attack inputs, screening
   logits, losses, or gradients. These checks stop a failed run; they do not
   clip/sanitize its logits or replace its loss. For valid finite inputs they
   do not alter the update.
4. Add an identifying log line and explanatory comments. Preserve the existing
   training function signature and the original loss and metric calculations.

### Zero-probability target derivative

The pinned implementation differentiates `F.kl_div(log_mixture, softmax(z/T))`
through both probability targets. In finite precision, a sufficiently small
softmax output becomes exactly zero. The probability-target entropy derivative
then involves `log(0)` and can produce a finite forward loss but NaN gradients.
On CPU PyTorch 2.8.0, float32 logits `[100, 0, -100]` versus `[0, 100, -100]`
at temperature 0.5 reproduce this: the upstream loss is approximately 0.69314718,
while both logit gradients contain NaNs. Float64 reproduces the same issue with
logit magnitude 1000.

The local helper evaluates each KL contribution as
`sum(p * (log_softmax(z/T) - log_mixture)) / batch_size`. For finite scaled
logits, `log_softmax` remains finite even when `p` underflows to zero. This avoids
the undefined intermediate target derivative and retains the finite composed
logit-space derivative. It keeps the upstream probabilities, temperature,
mixture floor, differentiability of both branches, and reduction. It adds no
logit or target-probability clipping, epsilon smoothing, detachment, or
`nan_to_num`. Floating-point operation ordering can change the final few bits;
ordinary-input values, gradients, and updates are therefore checked with tight
numeric tolerances, rather than claimed to be bit-identical.

The data/checkpoint wrapper is deliberately **not** claimed to be byte-identical
to the original upstream experiment pipeline. For the corrected comparison it
uses the same training-only validation protocol as SAAR: development uses the
fixed stratified training holdout, and final full-training-set refits use a
configuration locked before final evaluation and the fixed last epoch. Public
test data, including Tiny-ImageNet's official validation set used for final
reporting, do not select hyperparameters or checkpoints in this protocol.
The shared wrapper name `selection_free_v2` identifies this corrected workflow;
it does not mean that the RAAT objective itself has had selection removed.

The baseline driver retains the upstream L2 training attack step `15/255`;
SAAR retains its previously recorded explicit `32/255` step. This is an explicit
attack-step difference, not an assertion that those training attacks are
matched in step size. The common L2 budget is `128/255` and both use 10 PGD
steps. For Linf, both use budget `8/255` and step `2/255`.

Historical `raat_recorded` weights/results are retained as historical evidence;
they are not relabeled as newly trained `raat_official` results. Numerical
kernel equivalence is not an end-to-end accuracy reproduction.

## Validation

On a CPU host with PyTorch 2.8.0, nine tests passed:

- Verify both embedded upstream source hashes.
- Compare one optimizer step against the verbatim upstream kernel for all,
  some, and no alignment-eligible samples. Parameters and gradients are
  equal to tight float64 tolerances in each case; scheduler states agree.
- Verify a single `2B` attack input and repeated labels, with no AWP forward or
  recomputation pass.
- Check the exact boundary-reduced classification inputs and the exclusion of
  clean-wrong samples from alignment.
- Compare JS values and gradients with the upstream helper, including large
  finite logits whose behavior differs from clipped logits.
- Compare ordinary random-logit values and gradients for float32 and float64.
- Reproduce the upstream finite-loss/NaN-gradient failure on underflowing float32
  and float64 targets, and verify finite gradients from the local helper.
- Complete a finite optimizer update with extreme float32 logits.
- Reject AWP/shared-attack requests and nonfinite attacks before weight updates.

Run with `python -m unittest discover -s tests -p 'test_official_raat.py' -v`.
This audit did not launch a training campaign or measure new benchmark accuracy.
