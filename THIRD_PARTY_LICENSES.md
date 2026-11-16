# Third-Party Licenses

This file lists licenses for code adapted or referenced from third-party repositories.
Full license texts are added here as code is imported in subsequent tasks.

---

## TS-TCC
- Repository: https://github.com/emadeldeen24/TS-TCC
- License: MIT
- Used in: `contrastive_framework/models/timeseries_models.py`
- License text: *(to be added when code is imported)*

---

## SubTab
- Repository: https://github.com/AstraZeneca/SubTab
- License: Apache-2.0
- Used in: `contrastive_framework/models/tabular_models.py`
- License text: *(to be added when code is imported)*

---

## MoCo
- Repository: https://github.com/facebookresearch/moco
- Branch used: KaimingHe-patch-2 (moco/builder.py)
- License: CC BY-NC 4.0
- Used in: `contrastive_framework/models/image_models.py`, `contrastive_framework/models/base.py` (SplitBatchNorm2d inspired by MoCo CIFAR notebook)
- Copyright: Copyright (c) Facebook, Inc. and its affiliates.

Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)
https://creativecommons.org/licenses/by-nc/4.0/

You are free to:
  Share — copy and redistribute the material in any medium or format
  Adapt — remix, transform, and build upon the material

Under the following terms:
  Attribution — You must give appropriate credit, provide a link to the license,
                and indicate if changes were made.
  NonCommercial — You may not use the material for commercial purposes.

This framework is used for academic research (engineering thesis) only — non-commercial use.

---

## BYOL (deepmind-research)
- Repository: https://github.com/google-deepmind/deepmind-research (directory: `byol/`)
- License: Apache-2.0
- Used in: `contrastive_framework/models/image_models.py`
- Note: Original is JAX; reimplemented in PyTorch following the paper.
- License text: *(to be added when code is imported)*

---

## SCARF
- Paper: arXiv:2106.15147
- No official code repository.
- Implemented from scratch following the paper.
