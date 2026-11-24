# S-EOR matched scale pilot: completed evaluation report

## Evidence and scope

Received on 2026-11-24 (Asia/Shanghai): the server reported state=completed,
completed=30 and ALL_COMPLETED. The user supplied a terminal summary of all
30 jobs, late-window statistics and 12 gate-statistic groups.

Training: `/home/data2/XintongHu/GeleNet/experiments/v3_scale_seor_train_mPOzDLop`.

Evaluation: `/home/data2/XintongHu/GeleNet/evaluation_runs/v3_scale_seor_eval_nwJedHZA/results`.

All modes share the 1260/140 split, seed 42 and 44 actual epochs. Scoring uses
600 EORSSD test images, native-size saved grayscale PNGs, GT >127 and
PySODMetrics 1.6.2. These are within-pilot comparisons. Historical full-1400
runs and their different initialization histories are not matched controls.

[Terminal report](../experiments/scale_seor_pilot_20261124/terminal_report.txt)
and [structured transcription](../experiments/scale_seor_pilot_20261124/reported_summary.json)
preserve the supplied evidence. Metric and stability values were printed to
five decimal places; gate values to six. Do not interpret trailing zeros as
recovered full precision. The attachment SHA256 is
`81782e20d3a6341e7313dad9bed35306bcd2906638f2246de4699a59ec0e54c9`.
Archived terminal text normalizes line endings; this hash identifies the
original supplied attachment rather than the normalized archive file.
The original full-precision CSV/JSON, weights, predictions and pair hashes
remain on the server and were not independently downloaded.

## Checkpoints selected by internal validation

Selection uses minimum held-out scribble BCE, with epochs 13, 20 and 13
for fixed20, uniform and adaptive respectively. S/wF/F/E are higher-is-better;
MAE is lower-is-better.

| Mode | Epoch | S | wF | maxF | maxE | MAE |
|---|---:|---:|---:|---:|---:|---:|
| fixed20 | 13 | 0.49559 | 0.18061 | 0.26632 | 0.51089 | 0.15831 |
| uniform | 20 | 0.51106 | 0.19462 | 0.27167 | 0.51513 | 0.14091 |
| adaptive | 13 | 0.51246 | 0.18792 | 0.27047 | 0.53521 | 0.14295 |

Both mixtures improve all five displayed metrics over fixed20 at their
validation-selected checkpoints. Adaptive versus uniform is mixed:
adaptive improves S by 0.00140 and maxE by 0.02008, while wF, maxF and MAE
favor uniform. Adaptive's worse validation BCE therefore does not imply
worse dense saliency scores; sparse BCE and whole-image saliency scores
show different rankings in this pilot. This observation does not justify
retrospectively replacing the validation selection rule with test selection.

## Fixed final checkpoint

| Mode | Epoch | S | wF | maxF | maxE | MAE |
|---|---:|---:|---:|---:|---:|---:|
| fixed20 | 44 | 0.51845 | 0.20324 | 0.26881 | 0.50851 | 0.13445 |
| uniform | 44 | 0.52623 | 0.20595 | 0.27384 | 0.52019 | 0.12335 |
| adaptive | 44 | 0.53104 | 0.20836 | 0.28754 | 0.54243 | 0.12165 |

At epoch 44 adaptive leads all five displayed metrics. Relative to fixed20:
S +0.01259, wF +0.00512, maxF +0.01873, maxE +0.03392 and MAE -0.01280
(about 9.52% lower). Relative to uniform: S +0.00481, wF +0.00241,
maxF +0.01370, maxE +0.02224 and MAE -0.00170 (about 1.38% lower).
These are observations from one seed, not statistical-significance claims.

## Late-window behavior

Epochs 24/29/34/39/44, mean +/- population standard deviation:

| Mode | S | wF | maxF | maxE | MAE |
|---|---:|---:|---:|---:|---:|
| fixed20 | 0.52015 +/- 0.00227 | 0.20014 +/- 0.00505 | 0.26372 +/- 0.01219 | 0.51374 +/- 0.00571 | 0.13145 +/- 0.00424 |
| uniform | 0.52518 +/- 0.00580 | 0.20272 +/- 0.00812 | 0.27252 +/- 0.00897 | 0.52524 +/- 0.00370 | 0.12343 +/- 0.00393 |
| adaptive | 0.52462 +/- 0.00643 | 0.19971 +/- 0.00627 | 0.27823 +/- 0.00723 | 0.53835 +/- 0.01368 | 0.12810 +/- 0.00522 |

Uniform improves all five late-window means over fixed20. Adaptive versus
uniform improves mean maxF/maxE, while mean S/wF/MAE favor uniform. In
particular, adaptive MAE is 0.12810 versus uniform 0.12343. Adaptive does
not have an across-the-board late-window advantage. It has higher S/maxE/MAE
variation than uniform, although lower wF/maxF variation. These are
checkpoint fluctuations within one run, not multi-seed stability estimates.

The fixed-final advantage is stronger than the overall trajectory evidence.
All 30 rows are retained in the structured transcription rather than
publishing just a favorable checkpoint.

## What did the gate learn?

| Checkpoint | Test mean weights [10,20,40] | Across-image standard deviations |
|---|---|---|
| best_val | [0.334973, 0.332015, 0.333011] | [0.000539, 0.000159, 0.000486] |
| epoch_44 | [0.329871, 0.333977, 0.336152] | [0.000519, 0.000514, 0.000364] |

Statistics above use the 598 test images with all three distinct candidates
active. All 140 validation images also have three active candidates.
Two test images have deduplicated candidate memberships and are excluded
from these variation estimates; this prevents fallback constraints from
being mistaken for learned image-dependent selection.

The gate has nonzero but very small image-dependent variations. At epoch 44,
per-level standard deviations are 0.000519, 0.000514 and 0.000364, about
0.052, 0.051 and 0.036 percentage points. Mean weights are still close to
one third. This is evidence for an almost constant soft mixture in the
reported distributions, not strong selection of different NAMLab levels
for different image properties. Small variations could still matter
downstream; these aggregate statistics do not measure their causal effect.

The final adaptive score gain cannot yet be attributed specifically to
image-dependent gating rather than a nearly constant mixture and differences
in training trajectories.

## Next diagnostic, not yet performed

Use the existing adaptive epoch-44 checkpoint for three inference conditions:

1. Its original learned gate (existing predictions can be reused).
2. Force equal weights over active distinct memberships.
3. Force its validation-set mean weights
   [0.3298741144793374, 0.3339276432991028, 0.3361982420086861],
   masking inactive candidates and renormalizing per image.

Keep checkpoint, image transforms, candidate maps, membership deduplication
and scorer fixed. Only two extra inference passes are required. The constant
weights must come from the internal validation set, not test averages.
Compare paired predictions and scoring to test whether image-dependent
weights contribute at inference. This is an intervention on one trained
checkpoint; it does not replace training ablations or multi-seed repeats.

No such intervention, full-1400 retraining or new multi-seed experiment
has been performed in this report. The test set has already informed this
research; follow the previously documented exposure limitation in
[the split protocol](DATA_SPLIT_AND_COMPARISON_PROTOCOL.md).
