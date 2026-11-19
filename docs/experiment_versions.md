# Canonical experiment versions

| Branch | Definition | Status |
|---|---|---|
| V1 | Original GeleNet, original KTM and dense trainer | Exact main snapshot 859e9049b833e65f383cc016b9c1b126f8a7191a |
| V2 | V1 plus EORSSD-S scribble Partial BCE baseline; no NAMLab or RC | Local candidate selected by user on 2026-11-19; server identity pending |
| V3 | V2 plus optional NAMLab membership bias in KTM | Research implementation; region-data training loop incomplete |

Legacy main, v1-weak-supervision, v2-weak-supervision-ktm and v3-namlab are retained as historical references. Their old version numbers are superseded by this table. main stays the default branch. V1 is deliberately unchanged; consult this document on V2/V3 for version definitions.

V2 imports weak_data.py, weak_loss.py, train_weak_GeleNet.py, evaluate_weak_GeleNet.py, prepare_eorrsd_s.py and existing tests from the local v2-scribble-partial-bce worktree. No checkpoint, dataset or temporary zip is imported. The model matches V1 ignoring line endings. Training calls model(images), uses target/valid_mask and Partial BCE only. No RC, CRF or region IDs are used. The original dense trainer remains available but is not the V2 entry point.

Validation: source inspection completed. Unit test discovery was attempted from the repository root and failed importing PIL; no runtime tests passed in this audit. Existing documents report server unit/synthetic/real-batch checks on 2026-11-18; those are historical claims, not independently reproduced results. Server script hashes, manifests and raw run logs are needed to tie this candidate to the reported 44-epoch run.

V3 retains the old NAMLab history and incorporates the V2 candidate as an additional parent. This records both lines without rewriting either. See docs/NAMLAB_CHANGELOG.md for evidence and unfinished work.
