# Fixed-level V3 server source import

Imported 2026-11-20 from user-provided v3_previous_work_sources.tar.gz.
Archive SHA256: e783e8a978da3abfcfa3589ef6ae4ec6e16588c7b58270e33b1f7ff759944a46.

## What is preserved

All 71 exported files are archived without source edits beneath experiments/server_archive_20261120/, using their server-relative paths. export_manifest.json records byte counts and SHA256. Both run snapshots, initial preparation snapshot, configs/status, patched NAMLab Linux source, intermediate .before backups and three-image comparison record are included. Snapshots are evidence; absolute paths remain historical server locations.

Root region_data.py and train_v3_GeleNet.py are exact copies shared by both completed runs. Root model/GeleNet_models.py, model/pvtv2.py, weak_data.py and weak_loss.py already match both exported run snapshots by Git blob hash. Root utils.py remains unchanged; the archived run utility additionally prints learning rates. Use the complete archived run code for exact source replay.

## Actual training path

region_data.RegionDataset reads regions_l20/<image-stem>.npy beside its own source file; checks native shape; nearest-resizes labels to training size and returns int64 region_ids. train_v3_GeleNet.py moves labels to CUDA and calls model(images, region_ids=region_ids). The model forwards to KTM_23 and adds softplus(region_strength) times same-region membership before softmax.

The original train_weak_GeleNet.py remains the V2 trainer and omits region_ids. Use the newly imported train_v3_GeleNet.py for fixed-level region-aware training. Neither entry point implements adaptive scale fusion.

Runtime requirements not shipped: ImageNet PVT weights at ./model/pvt_v2_b2.pth, dataset manifest triplets, decoded target/valid masks, native-resolution regions_l20/*.npy and Python/CUDA dependencies. Set working directory to the relevant code root. Copied source alone is not a runnable dataset bundle.

Default --epoch 45 executes range(1,45), i.e.44 epochs, and saves4,9,...44. Configs report seed42, but the exported entry point does not set any random seed; no launch wrapper is supplied here. Treat seed42 as reported metadata, not verified deterministic setup.

## NAMLab changes and evidence

ComputeDBC/ComputeThreshold/scanning and other archived files contain diagnostics; .before files preserve intermediate server states, not an independently identified upstream baseline. docs/NAMLAB_SERVER_BACKUP_DIFFS_20261120.patch compares each available backup to its current file and is NOT a single cumulative upstream patch.

removeRemnant.cpp contains an algorithm-affecting conditional: skip remnant merging when NumSeg > saveHir and NumSeg <150. run_dataset.cpp and its backups retain parameter experiments. Diagnostic prints and algorithm changes must not be conflated.

comparison.json reports the archived executable SHA256 c7b4779cde664c91fec1fb097ce9bb07d0ce671e3158d22e5e592742465d16c9 and identical level20 memberships for0001,1029,1034 (60,60,41 available levels). The executable is not in this export. No compiler rerun or source-to-binary equivalence was established here; three sample matches do not establish a full1400-image reproduction.

## Checks and remaining limitations

Verified all71 files against the supplied manifest; parsed24 Python files; checked both run snapshot core sources for equality; reviewed trainer/loader forwarding and diagnostic/algorithm separation. No credential patterns found by targeted inspection. No CUDA training, MATLAB/OpenCV C++ build, or fresh inference was run locally. Generation/export scripts, complete dataset manifests, dependency lockfiles, launch wrappers and weight hashes are not contained in this package. Previously recorded evaluation results remain separate evidence.
