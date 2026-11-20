# V3 adaptive NAMLab scale pilot

**数据划分与结果解释必读：** [为什么从1400全量训练改为1260训练加140内部验证，以及旧结果的比较边界](DATA_SPLIT_AND_COMPARISON_PROTOCOL.md)。此前已查看测试集结果；新增验证划分不能消除这一历史。

This is an experimental extension of fixed-level V3, inspired by learning scale weights in Attention to Scale (Chen et al., CVPR2016), https://openaccess.thecvf.com/content_cvpr_2016/html/Chen_Attention_to_Scale_CVPR_2016_paper.html . It mixes NAMLab membership priors rather than image-pyramid predictions. No performance improvement is claimed.

## Three matched modes

- fixed20: only candidate20 contributes; min(20, available_levels) on short hierarchies.
- uniform: equal weights over distinct candidate memberships10,20,40.
- adaptive: global average pooled KTM x2+x3 features ->32-to16-to3 MLP ->masked softmax; weights depend on each image. Final gate layer starts at zero, giving uniform initial weights.

All modes construct the same modules in the same order. ScaleKTM inherits the existing KTM, retaining attention/value/residual operations. ScaleGeleNet initializes from the existing ImageNet backbone and copies the just-initialized base KTM weights. It does not warm-start from previously trained V3 saliency weights. Region strength remains softplus(-2) initially. The added gate parameters are present but unused in fixed20/uniform, keeping other initialization and random-number consumption matched. This is an image-level soft mixture, not pixel-level attention or hard layer selection.

Candidate maps are canonicalized and nearest-resized from native resolution to352 during preparation, then nearest-projected to the22x22 KTM grid. Membership-equivalent candidates are deduplicated on that actual grid, including label permutations. Representative priority20,10,40 avoids extra votes for fallback duplicates. Diagnostics record requested order10,20,40, actual source levels, active candidates and per-image weights. Available-layer fallback is independent of GT. Canonical relabeling preserves memberships; IDs are integers and checked before float32 interpolation.

## Matched pilot protocol

Use tools/prepare_scale_experiment.py with an existing decoded scribble manifest and the original training MAT directory. Seed42 creates a deterministic1260/140 train/internal-validation split from1400 images. No dense GT is read. Validation is held-out scribble BCE averaged over annotated pixels, not dense saliency metrics. Empty annotation samples remain; an entirely unannotated validation split fails.

Train all three modes on the SAME prepared directory with the same seed42, fresh pretrained backbone, batch8, size352, Adam lr1e-4, decay0.1 at epoch30, gradient element clamp0.5, num_workers0, Partial BCE only, no RC or extra augmentation. --epochs44 means44 actual epochs. Saves epochs4,9,...44 and best_val.pth selected by held-out scribble BCE. Logs config/source and input-manifest hashes, epoch losses, validation image weights and state. --max-steps3 with --epochs1 is a smoke run and marked smoke_completed; it must not enter the result table. No resume path or automatic recovery is supplied.

Explicitly seeds Python, NumPy and torch/CUDA; deterministic cuDNN flags set. This is not a guarantee of bitwise reproducibility across versions/devices. Keep environment and backbone weight hashes matched. Training and output paths must be fresh; original experiments remain unchanged.

Old1400-image full-training runs are historical references, not controlled comparators for this1260-image pilot. Determine a fixed recipe on the internal validation split; do not choose layers, seed or epochs by the600-image test set. Formal full-data training and multi-seed repeats remain subsequent work.

## Commands from the code root

```
python tools/prepare_scale_experiment.py --manifest EXISTING_MANIFEST --regions TRAIN_MAT_DIR --output NEW_PREPARED_DIR
python train_scale_GeleNet.py --prepared PREPARED_DIR --output NEW_RUN_DIR --mode adaptive --epochs1 --max-steps3
python train_scale_GeleNet.py --prepared PREPARED_DIR --output NEW_RUN_DIR --mode fixed20
python train_scale_GeleNet.py --prepared PREPARED_DIR --output NEW_RUN_DIR --mode uniform
python train_scale_GeleNet.py --prepared PREPARED_DIR --output NEW_RUN_DIR --mode adaptive
```

model/pvt_v2_b2.pth is required by the unchanged base constructor, relative to current working directory. Existing timm/torch/torchvision/Pillow/NumPy/SciPy dependencies are needed. Data, backbone weights, prepared maps and completed checkpoints are not in Git. Preparation and run output directories must not already exist. Launch long runs under nohup only after smoke passes.

Inference-map preparation needs no GT: tools/prepare_scale_experiment.py --images IMAGE_DIR --regions MAT_DIR --output NEW_MAP_DIR. tools/predict_scale_GeleNet.py reads the training config to reconstruct mode and strictly loads weights. It exports native-size grayscale PNGs with existing resize-logits ->sigmoid ->minmax ->round policy and scale_weights.json. Use the existing tools/score_sod_pngs.py in the pinned scoring environment with GT>127. This experimental predictor rejects smoke-run configs.

## Verification and limits

Eight local CPU tests passed under torch2.8.0, NumPy/SciPy/Pillow and torchvision: direct original-KTM fixed20 equivalence with nonzero residual gamma, finite nonzero adaptive gradient, normalized weights and initial uniform equivalence, label-permuted duplicates, post-projection duplicates, image-dependent learned weights, invalid inputs, synthetic MAT fallback/split and real loader tensor shape/dtype. Sixteen packaged Python files parsed successfully. Tests isolate KTM without constructing the PVT backbone. The local environment lacks timm; no full-model initialization/forward, full CUDA training or real-server data smoke has run. Server torch1.9.0 compatibility remains pending its smoke check. No new training metrics or adaptive-layer optimum has been established.

## New files

model/scale_prior.py; model/adaptive_GeleNet.py; scale_data.py; train_scale_GeleNet.py; tools/prepare_scale_experiment.py; tools/predict_scale_GeleNet.py; tests/test_scale_prior.py; tests/test_scale_integration.py. Fixed-level source and its archive remain unchanged.
