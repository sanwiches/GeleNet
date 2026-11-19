# Version 2: GeleNet with scribble supervision

This local branch keeps the original GeleNet model, KTM, and forward return
`(logits, sigmoid(logits))`. The original dense-mask trainer
`train_GeleNet.py` and `data.py` are unchanged. Version 1 remains on its own
branch. The previously documented NAMLab KTM modification is outside this
scribble-only version and is not used here.

## Data contract

`weak_data.py` returns a dict with `image` (normalized RGB, `3xHxW`),
`target` (binary foreground label, `1xHxW`) and `valid_mask` (binary annotated
pixel indicator, `1xHxW`). Image and both masks are resized together; masks
use nearest-neighbor interpolation. No random spatial augmentation is enabled.

The generic loader accepts an explicit JSON manifest containing path triplets:

```json
[
  ["images/example.jpg", "targets/example.png", "valid/example.png"]
]
```

Relative paths are resolved from the manifest; absolute paths also work. Both masks must already be single-channel
images at the original image resolution with **only 0 and 255** pixel values.
The `target` is foreground at 255 and background at 0. The `valid_mask` is
annotated at 255; target values outside annotated pixels are ignored. This is
an intermediate, decoded format. Dense ground truth is never read by this training path.

The PAFR EORRSD-S archive has been observed to contain 1400 `img`, `gt`, and
`mask` triplets. `prepare_eorrsd_s.py` validates every triplet and converts
its `gt` and `mask` files into single-channel `target` and `valid_mask` files.
The archive contains some colored or intermediate-valued pixels, so an
`--audit-only` pass reports their counts. The 2026-11-18 audit found 1400
matched triplets, 2,307,007 exact-white foreground pixels, 15,935,066
exact-white valid pixels, and ambiguous pixels in 14 samples (4,469 in `gt`,
9,526 in `mask`; about 0.06% of the confirmed valid count). Conversion requires an explicit
`--exclude-ambiguous` choice, under which only exact RGB white is labeled and
all other colors remain unannotated. It rejects missing pairs, mismatched
sizes, and foreground outside the valid mask. The full archive passed this
audit. The same 1400 triplets were converted successfully on 2026-11-18 and
`manifest.json` was generated under the remote
`datasets/weak_prepared/EORRSD-S/` directory. A real-batch GPU smoke test
passed on GPU 7 with two samples, 9,127 valid pixels, and Partial BCE loss
0.819756. S-EOR and ORSI-4199 require separate format inspection.

## Loss and training

`weak_loss.partial_bce_with_logits` calculates BCE on the unchanged saliency
logits and divides by the count of valid pixels. An entirely unannotated batch
has zero loss and zero gradient. There is no IoU, RC, or region-consistency
term. Run from the repository root with a CUDA PyTorch environment:

```bash
python train_weak_GeleNet.py --manifest /path/to/manifest.json --output /path/to/v2_checkpoints
```

The epoch, optimizer, learning rate schedule, clip value, batch size, and input
size defaults match the original trainer. Use a separate output directory from
Version 1. The original test script can still use the model's unchanged output.

## Validation

On 2026-11-18, the remote Ubuntu environment (PyTorch 1.9.0+cu111) passed all
five unit tests. A synthetic one-batch forward, Partial BCE, backward, and
optimizer step passed on an RTX 3090 with `CUDA_VISIBLE_DEVICES=5`; the reported
loss was 0.323207 and peak allocated memory was 0.59 GiB. This checks the code
path, not compatibility with raw scribble dataset files. Real-data validation
still requires confirming the annotation encoding and file layout.


## Provenance clarification (2026-11-19)
This is the local Partial BCE candidate selected by the user for V2. Server test and dataset audit numbers above are historical reports, not rerun results from this audit. The closing statement that real-data validation is still required predates the real-batch report above. No server file hashes or raw logs were available to establish byte-for-byte identity with the 44-epoch run. Local test discovery on 2026-11-19 was blocked by missing Pillow (PIL); no tests passed in this audit.
