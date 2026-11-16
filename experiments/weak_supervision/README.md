# V1 — GeleNet Weak-Supervision Baseline

## Purpose

This branch keeps the author's GeleNet architecture unchanged, including the original Knowledge Transfer Module (KTM), and prepares a scribble-supervised training baseline.

- Upstream model implementation: `model/GeleNet_models.py`
- Original training entry point: `train_GeleNet.py`
- Original data pipeline: `data.py`
- Original prediction head: one saliency logit map (plus its sigmoid probability)

## Supervision contract

The weak-supervision experiment must train from sparse foreground/background scribbles only.

1. The training loss may read class labels only at annotated scribble pixels.
2. Unannotated pixels must be excluded from the partial classification loss.
3. Dense ground-truth masks, when available, are reserved for validation/testing and must not enter the training loss.
4. The original GeleNet KTM is retained without architectural changes in V1.

**Annotation encoding is intentionally not assumed yet.** Before implementing the dataset loader, confirm the actual scribble file format and how foreground, background, and unlabeled pixels are encoded. The loader must return an image, a binary scribble target, and a valid-pixel mask.

## Fair comparison with V2

V1 is the control condition for V2. Keep the following identical between branches:

- training/validation split and scribble annotations;
- input resolution and image normalization;
- optimizer, learning-rate schedule, batch size, epoch count, and random seed;
- weak-supervision loss terms and their weights, except for a specifically isolated KTM-related change;
- evaluation scripts and dense test ground truth.

Do not change the decoder, backbone, data augmentation, or loss while claiming a KTM-only ablation.

## Current status

- [x] Original GeleNet source retained from the upstream fork.
- [x] V1 branch created from the untouched upstream snapshot.
- [ ] Scribble annotation encoding confirmed.
- [ ] Partial-label dataset loader and masked loss implemented.
- [ ] Training and evaluation smoke tests completed.

## Upstream notes

The upstream repository's original training script uses dense masks with BCE + IoU. That objective is **not** a weak-supervision loss and must not be reused unchanged for scribble training. The original script is retained as a reference.
