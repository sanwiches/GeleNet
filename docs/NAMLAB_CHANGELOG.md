# V3 NAMLab modification and evidence record

Audit date: 2026-11-19. V1=original GeleNet; V2=scribble Partial BCE; V3=V2 plus NAMLab membership prior in KTM. V3 remains incomplete as a region-aware training system.

## Committed source and lineage

- cb6fcdc: vendors third_party/NAMLab/NAMLab.zip, README.upstream.md and MIT LICENSE. Declared source: https://github.com/YunpingZheng/NAMLab. The archive was described as unchanged in historical documentation; this audit did not independently compare its bytes to an upstream commit. No upstream revision/hash should be inferred.
- 1f7d4ed0083cb8ada0bf1ce366d707b56e6e6193: adds region_strength initialized to -2, optional region_ids in GeleNet.forward and KTM.forward, and passes IDs to KTM_23. IDs accept BxHxW or Bx1xHxW; batch/rank checks and nearest-neighbor resizing exist. Adds softplus(region_strength) * indicator(R_i == R_j) before softmax. Omitting IDs skips the bias. This is a single map membership prior, not hierarchical fusion or RC loss.
- 4885b19cb521474bb164fa6901cf2c7cdc96109f: historical NAMLab KTM documentation. Its old V1/V2 labels are superseded.
- V2 candidate commit: bb5ededc6a47fa8464045a4409aaa410726c2ce4. V3 combines that candidate's data/loss/trainer with the existing NAMLab model and archive. Both histories are parents of the integration commit; the old branch tips remain available.

## What this audit verified

Source inspection confirms optional model-to-KTM forwarding and the bias implementation. The V2 candidate model matched original main ignoring line endings. Eight local Python source files parsed successfully. Runtime test discovery failed on missing PIL; no GPU, dataset, NAMLab binary or end-to-end training test was run here. Historical server test results in README_V2_SCRIBBLE.md are reported evidence only, with no raw logs/hashes attached to this audit.

## Server-only temporary C++ diagnostics

The referenced discussion mentions temporary server diagnostics. No standalone patch, server checkout, source hash, compiler command, executable hash or complete diagnostic log was available in this audit. The inspected committed tree has only the vendored NAMLab archive/README/license under third_party/NAMLab. Therefore diagnostic instrumentation is NOT established as a committed fix or a verified algorithm change. Do not infer details of the patch from discussion summaries.

## Abnormal-image handling

No independently traceable NAMLab abnormal-image/threshold repair patch or before/after regression output was established. Status: evidence pending, not verified and not imported. Obtain the exact C++ diff against an identified upstream revision, affected image IDs/checksums, parameters, failure logs and rerun output before marking any repair complete.

prepare_eorrsd_s.py does contain separate EORSSD-S annotation validation: exact white/black handling, ambiguous-pixel auditing and explicit --exclude-ambiguous conversion. This concerns scribble annotation adaptation, not a NAMLab segmentation crash/threshold fix.

## Incomplete region-data training loop

The imported weak loader emits image/target/valid_mask only. train_weak_GeleNet.py calls model(images) without region_ids, so the NAMLab bias is inactive in this trainer even on V3. The repository does not yet supply the complete NAMLab generation -> exported region-ID map -> manifest pairing -> synchronized transforms -> device transfer -> model(images, region_ids=...) loop. V3 must not be reported as a completed NAMLab training experiment.

Next implementation checks: region-map format/level selection and stable IDs; alignment through resize/augmentation; device/dtype and large-ID precision; ignored/invalid region policy; missing-map fail-fast behavior; checkpoint compatibility with the added parameter; memory and gradient tests; matched V2/V3 training/ablation. RC and multilevel fusion are not implemented by the current KTM change.

## Safe Git organization

V1 points exactly to original main. V2 is based on V1; V3 retains both V2 and the old NAMLab history. No old refs are deleted or force-updated. Default main remains unchanged. Before creation, no V1/V2/V3 branches or tags existed; all four legacy branches reported protected=false and rulesets returned []. Repository metadata reported push/admin permission; the protection-detail endpoint returned 403 (integration access limitation), so detailed protection policy could not be independently inspected.
