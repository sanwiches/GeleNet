# V2 — GeleNet Weak Supervision with NAMLab-aware KTM

This branch is based on `v1-weak-supervision`. It retains the same weak-supervision objective and data protocol, and modifies KTM by adding an optional NAMLab region-membership prior to its attention logits.

## KTM modification

Let the original KTM content affinity be (E=QK). Given a NAMLab region-ID map (R), define

[
B_{ij}=\mathbb{1}[R_i=R_j],\qquad
A'=\operatorname{softmax}(E+\operatorname{softplus}(\alpha)B).
]

The learnable scalar `region_strength` is initialized to -2.0, so the starting point remains close to original KTM. Region IDs are resized to the KTM feature grid using nearest-neighbor interpolation. The prior is optional for backward compatibility: when `region_ids=None`, KTM follows the original attention computation.

The GeleNet forward method accepts `region_ids=None` and passes it to `KTM_23`.

## NAMLab source

The upstream `NAMLab.zip`, upstream README, and MIT LICENSE are stored under `third_party/NAMLab/`. The source archive is retained unchanged. Retain the original copyright/license notice when redistributing or modifying its source.

Upstream: https://github.com/YunpingZheng/NAMLab

## Implementation boundaries

- This commit adds a **same-region attention-logit bias**. It does not yet implement multi-level hierarchy fusion; region-level selection/fusion must be specified and ablated separately.
- The data pipeline still needs to load NAMLab region-ID maps and call `model(images, region_ids=...)`.
- Scribble annotation encoding must be confirmed before implementing the partial-label loader and loss.
- Dense ground-truth masks must not be used in training loss.
- V1 and V2 must share identical splits, scribbles, augmentation, optimizer, schedule, seed, and loss weights.

## Status

- [x] Original GeleNet source retained on `main`.
- [x] V1 weak-supervision branch created.
- [x] NAMLab source archive, upstream README, and MIT LICENSE vendored.
- [x] Optional NAMLab same-region bias added to KTM.
- [ ] Implement and validate scribble-only training in V1.
- [ ] Add synchronized image/scribble/region-map transforms in V2.
- [ ] Validate region-map alignment and KTM tensor shapes.
- [ ] Run controlled V1/V2 comparison and ablations.

## Citation

Y. Zheng, B. Yang, and M. Sarem, “Hierarchical Image Segmentation Based on Nonsymmetry and Anti-packing Pattern Representation Model,” IEEE Transactions on Image Processing, 2021.
