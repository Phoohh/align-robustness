# Source attribution and licensing

The supplied snapshot identifies its original project as
**Robust Alignment: Harmonizing Accuracy and Robustness in Adversarial Training**
(RAAT). This repository packages and extends that experimental code. Existing
source references in the code are retained, including the logger reference and
the DARTS cutout reference.

External dependencies are installed separately, rather than copied into this
repository:

- [AutoAttack](https://github.com/fra31/auto-attack), pinned in `requirements.txt`.
- [AdverTorch](https://github.com/BorealisAI/advertorch).
- [PyTorch and torchvision](https://pytorch.org/).

No LICENSE or COPYING file was present in the supplied snapshots. The upstream
[RAAT repository](https://github.com/FlaAI/RAAT), at commit
`61de35b7e2db141e2904eee59460a6d98964525d`, supplies an MIT license with
copyright `(c) 2026 Y. Wang`. That file is reproduced unchanged as [LICENSE](LICENSE).
The original README title and source layout match the supplied project's
upstream identity. Packaging additions use the same MIT license. Required
upstream attribution is retained in the anonymized repository.

Installing external packages does not change their licenses. In particular,
AdverTorch is separately distributed under its own LGPL/GPL terms, and its
implementation is not vendored here.
