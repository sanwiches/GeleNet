# Short NAMLab test hierarchy

Server output on 2026-11-20 reports 600 test MAT files. Strict level-20 validation failed only for 0754, with 10 available levels. Inspection reports all ten maps have shape (416, 503), with unique region counts 1,2,3,4,5,6,7,8,9,12. This is not evidence that a 20th level exists.

For both V3/EORSSD-S and V3/S-EOR evaluation, the explicit --allow-last-level option selects min(20, available_levels). Empty hierarchies and invalid map shapes/dtypes still fail. No test images are excluded and no raw MAT files are modified. Each completed prediction run saves predictions_region_selection.json beside predictions, documenting every selected level and fallback. Selection depends only on hierarchy availability, not GT values or evaluation scores.

The new tools/predict_v3_regions_fallback.py preserves the strict predictor separately. Syntax checked locally; GPU execution and full 600-image scoring remain pending server execution. Existing strict validation.json/status.txt remain historical failure evidence and must not be relabeled as passing.

V3/S-EOR training completion and existence of GeleNet.pth.44 (98 MB) were reported by the server. Existing runs completed 44 epochs. This change requires no retraining. Final-epoch comparisons use checkpoint 44; checkpoint searches on test scores are exploratory and should be reported separately.
