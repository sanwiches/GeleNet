# Evaluation protocol and evidence, 2026-11-20

Scripts: tools/predict_v3_regions.py requires --code, --regions and --level (20). Missing levels fail explicitly; no silent fallback. Region resize matches training: PIL nearest to 352, existing KTM torch nearest to feature grid. GeleNet author-style logit resize -> sigmoid -> per-image min/max -> rounded uint8 PNG. Rounding is explicit; exact legacy imageio float conversion is not claimed.

Score saved PNGs with tools/score_sod_pngs.py and pysodmetrics==1.6.2; GT >127 ->255, otherwise 0, in memory only. Preserve library default metric preprocessing; F beta parameter 0.3, weighted F beta 1. Main result is final epoch44; all nine checkpoints are exploratory and best is selected by S, never spliced per metric.

Server reported successful level20 membership reproduction on training images 0001,1029,1034 using archived run_dataset sha256 c7b4779cde664c91fec1fb097ce9bb07d0ce671e3158d22e5e592742465d16c9. Archived program/Linux source is in experiments/namlab_reproduce_check_3aIPoO1o/program on server. Reported source k=1.8,u=0.9,var=2.5,size=18800 with branch150. Server source/binary have NOT been uploaded here; source-build identity and upstream diff remain pending. Three sample matches are evidence, not full 1400-image reconstruction.

Training: V3 EORSSD-S epoch44 completed, and V3 S-EOR started. Server loader/trainer integration still needs exact source import into Git; do not treat these evaluation-only additions as a complete training integration commit. Test region generation and V3 scoring pending. Scripts passed local AST syntax check only; runtime verification requires server execution.
