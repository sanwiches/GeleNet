# S-EOR matched scale pilot: reported server training completion

Evidence: user-pasted terminal output received on 2026-11-24 (Asia/Shanghai).
This records server-reported artifacts; the checkpoints and complete logs have
not been downloaded or independently inspected. Dense saliency test scoring for
this pilot has not yet been reported.

Run: `/home/data2/XintongHu/GeleNet/experiments/v3_scale_seor_train_mPOzDLop`.
The batch status is `COMPLETED`; each mode status is `completed`, with an
epoch-44 record. The three modes ran sequentially on GPU 0. The code snapshot
is `<run>/code`; the shared prepared split is
`/home/data2/XintongHu/GeleNet/code/GeleNet_v3_scale_iRgdyYCF/prepared_seor`.

Recipe: 1260 training / 140 held-out scribble-validation images, seed 42,
44 actual epochs, size 352, batch 8, Adam lr 1e-4, decay 0.1 at epoch 30,
Partial BCE, no RC. See DATA_SPLIT_AND_COMPARISON_PROTOCOL.md for the change
from historical full-1400 training and prior exposure to the 600-image test set.

| Mode | Best validation epoch | Best validation BCE | Epoch-44 training BCE | Epoch-44 validation BCE |
|---|---:|---:|---:|---:|
| fixed20 | 13 | 0.029523423855271097 | 0.0005794906354350144 | 0.049669903720589326 |
| uniform | 20 | 0.030774278508790578 | 0.0009124239400022552 | 0.06522727350183016 |
| adaptive | 13 | 0.044570642255355855 | 0.003029160543519538 | 0.08337139801501024 |

Validation mean weights are ordered by requested levels [10, 20, 40]:

| Mode | At best validation epoch | At epoch 44 |
|---|---|---|
| fixed20 | [0, 1, 0] | [0, 1, 0] |
| uniform | [0.3333333433, 0.3333333433, 0.3333333433] | same |
| adaptive | [0.3349814115, 0.3319941915, 0.3330243994] | [0.3298741145, 0.3339276433, 0.3361982420] |

The reported held-out BCE currently favors fixed20. Adaptive has not established
an improvement. Means close to one third do not establish that all images use
the same weights: per-image distributions must be inspected, excluding the
weight changes forced by deduplicated/fallback candidates. Higher final than
best BCE motivates preserving both checkpoints; it does not directly measure
dense-mask saliency quality.

## Frozen evaluation plan

`tools/evaluate_scale_experiment.py` uses the completed run's code snapshot,
checks source/config/manifest hashes, and writes a fresh evaluation directory.
It first evaluates best_val.pth for the three modes (epochs 13, 20, 13), then
the three epoch-44 weights. With `--all-checkpoints`, it subsequently evaluates
epochs 4, 9, 14, 19, 24, 29, 34, 39 for each mode: 30 jobs total.

Test inputs are the existing 600 EORSSD images and NAMLab MAT files under
`experiments/namlab_eorrsd_test_vuCvT51X/regions`. Inference preparation selects
min(requested level, available layers) for candidates 10/20/40 and records
actual levels. This uses no GT. The original short hierarchy, including 0754,
is retained; the model deduplicates equal memberships at the KTM grid.

Inference uses the original scale predictor: native-size bilinear logits
(align_corners=False), sigmoid, per-image min-max, rounded uint8 grayscale PNG.
Scoring snapshots `experiments/score_sod_pngs_gt127.py` and uses the existing
`envs/sod_eval_162` environment (PySODMetrics 1.6.2). GT is converted in memory
using >127 foreground, <=127 background. S, weighted F, max/mean/adaptive F/E,
and MAE remain the established project protocol. Preflight requires the scorer
hash to equal `evaluation_runs/v3_both_epoch44_local_XHsQg5ad/score_used.py`,
the snapshot used in the completed historical V3 evaluation.

Outputs include all_checkpoints.csv, primary_results.json, per-job inputs/hashes,
PNG predictions, curves/protocol/pair hashes from the scorer, and per-image
scale weights. gate_diagnostics.json includes per-level mean/std/min/max for
images with all three candidates active and counts fallback/deduplicated cases.
late_window_stability.json summarizes epochs 24/29/34/39/44 by mean and
population standard deviation only when all five are available. This is
single-run checkpoint variation, not multi-seed robustness. Test trajectories
do not replace validation-based checkpoint selection.

The runner requires a new output directory and does not implement resume.
No completed benchmark metrics or full-1400 retraining are claimed here.

Four focused local standard-library tests cover primary-first job scheduling,
best-validation selection and source/smoke checks, fallback-aware weight
statistics, and exclusion of best_val from the five-checkpoint stability window.
No new GPU inference or server scoring has been performed locally.
