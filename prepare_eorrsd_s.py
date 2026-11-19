"""Validate PAFR EORRSD-S and create the generic Version 2 manifest.

The raw archive has img/, gt/, mask/ triplets. We interpret gt as positive
scribbles and mask as all valid scribble pixels only after strict checks.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def binary_channel(path, size):
    with Image.open(path) as image:
        if image.size != size:
            raise ValueError("size mismatch: {}".format(path))
        pixels = np.asarray(image)
    if pixels.ndim == 3 and pixels.shape[2] == 3:
        black = np.all(pixels == 0, axis=2)
        white = np.all(pixels == 255, axis=2)
    elif pixels.ndim != 2:
        raise ValueError("expected grayscale or RGB mask: {}".format(path))
    else:
        black = pixels == 0
        white = pixels == 255
    ambiguous = int(np.count_nonzero(~(black | white)))
    return (white.astype(np.uint8) * 255), ambiguous


def prepare(root, output, audit_only=False, exclude_ambiguous=False):
    root, output = Path(root).resolve(), Path(output).resolve()
    if root == output or root in output.parents:
        raise ValueError("output must be separate from raw image/label directories")
    image_dir, gt_dir, mask_dir = (root / part for part in ("img", "gt", "mask"))
    images = {}
    for path in image_dir.iterdir():
        if path.is_file() and path.suffix.lower() in (".jpg", ".jpeg", ".png"):
            if path.stem in images:
                raise ValueError("duplicate image stem: {}".format(path.stem))
            images[path.stem] = path
    gt_files = {path.stem: path for path in gt_dir.glob("*.png")}
    mask_files = {path.stem: path for path in mask_dir.glob("*.png")}
    if not images or images.keys() != gt_files.keys() or images.keys() != mask_files.keys():
        raise ValueError("img, gt and mask stems must match exactly")

    positive = valid = 0
    ambiguous_gt = ambiguous_mask = 0
    ambiguous_samples = []
    for index, stem in enumerate(sorted(images), 1):
        with Image.open(images[stem]) as image:
            size = image.size
        gt, gt_ambiguous = binary_channel(gt_files[stem], size)
        mask, mask_ambiguous = binary_channel(mask_files[stem], size)
        if np.any((gt == 255) & (mask == 0)):
            raise ValueError("gt foreground lies outside valid mask: {}".format(stem))
        positive += int(np.count_nonzero(gt))
        valid += int(np.count_nonzero(mask))
        ambiguous_gt += gt_ambiguous
        ambiguous_mask += mask_ambiguous
        if gt_ambiguous or mask_ambiguous:
            ambiguous_samples.append((stem, gt_ambiguous, mask_ambiguous))
        if index % 200 == 0:
            print("Validated {}/{}".format(index, len(images)), flush=True)
    if positive == 0 or valid == positive:
        raise ValueError("dataset needs foreground and background scribbles")
    print("Validated {} triplets; foreground pixels: {}; valid pixels: {}; background pixels: {}".format(
        len(images), positive, valid, valid - positive))
    print("Ambiguous pixels: gt={}, mask={}, affected samples={}".format(
        ambiguous_gt, ambiguous_mask, len(ambiguous_samples)))
    print("Largest affected samples (stem, gt, mask): {}".format(
        sorted(ambiguous_samples, key=lambda row: row[1] + row[2], reverse=True)[:10]))
    if audit_only:
        return
    if ambiguous_samples and not exclude_ambiguous:
        raise ValueError("ambiguous pixels found; review audit and pass --exclude-ambiguous to leave them unlabeled")

    target_dir, valid_dir = output / "target", output / "valid_mask"
    target_dir.mkdir(parents=True, exist_ok=True)
    valid_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for index, stem in enumerate(sorted(images), 1):
        with Image.open(images[stem]) as image:
            size = image.size
        target_path, valid_path = target_dir / (stem + ".png"), valid_dir / (stem + ".png")
        Image.fromarray(binary_channel(gt_files[stem], size)[0], mode="L").save(target_path)
        Image.fromarray(binary_channel(mask_files[stem], size)[0], mode="L").save(valid_path)
        manifest.append([str(images[stem]), str(target_path), str(valid_path)])
        if index % 200 == 0:
            print("Converted {}/{}".format(index, len(images)), flush=True)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Manifest: {}".format(output / "manifest.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="extracted EORRSD-S directory")
    parser.add_argument("--output", required=True, help="separate directory for decoded masks")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--exclude-ambiguous", action="store_true",
                        help="treat all pixels except exact white as unlabeled")
    args = parser.parse_args()
    prepare(args.root, args.output, args.audit_only, args.exclude_ambiguous)
