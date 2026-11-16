# GeleNet experiment branches

The repository's `main` branch is the untouched fork of the author's GeleNet implementation.

| Branch | Role | Network | Supervision |
|---|---|---|---|
| `main` | Upstream reference | Original GeleNet + original KTM | Original dense-mask training code |
| `v1-weak-supervision` | Weak-supervision control | Original GeleNet + original KTM | Sparse scribble labels |
| `v2-weak-supervision-ktm` | KTM research variant | GeleNet with an explicitly documented KTM modification | Same sparse scribble labels and training protocol as V1 |

## Experimental rule

V1 and V2 must differ only in the KTM intervention being tested. Any NAMLab preprocessing, region-map format, region-to-feature alignment, and KTM attention modification must be documented in V2. The exact algorithm should be finalized before changing the model implementation.

The GeleNet upstream repository currently has no `LICENSE` file. Keep the upstream attribution and original source history intact; do not add a license that purports to relicense the upstream GeleNet code.
