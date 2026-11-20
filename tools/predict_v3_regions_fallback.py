"""Save GeleNet saliency maps and summarize dense-test behavior.

By default this is a raw-sigmoid diagnostic. --author-normalize applies the
per-image min/max step from the original GeleNet test_GeleNet.py before saving.
The printed summary is not the paper's MATLAB benchmark result.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms

import os
import sys
from scipy.io import loadmat
bootstrap = argparse.ArgumentParser(add_help=False)
bootstrap.add_argument('--code', required=True)
location, _ = bootstrap.parse_known_args()
code = Path(location.code).resolve()
os.chdir(code)
sys.path.insert(0, str(code))
from model.GeleNet_models import GeleNet


def paired_files(image_root, gt_root):
    images = {}
    for path in Path(image_root).iterdir():
        if path.is_file() and path.suffix.lower() in (".jpg", ".jpeg", ".png"):
            if path.stem in images:
                raise ValueError("duplicate image stem: {}".format(path.stem))
            images[path.stem] = path
    gts = {}
    for path in Path(gt_root).iterdir():
        if path.is_file() and path.suffix.lower() in (".png", ".jpg", ".jpeg"):
            if path.stem in gts:
                raise ValueError("duplicate GT stem: {}".format(path.stem))
            gts[path.stem] = path
    if not images or images.keys() != gts.keys():
        raise ValueError("test images and dense GT must have identical stems")
    return [(stem, images[stem], gts[stem]) for stem in sorted(images)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--code", required=True)
    parser.add_argument("--regions", required=True)
    parser.add_argument("--level", type=int, default=20)
    parser.add_argument("--allow-last-level", action="store_true",
                        help="use the final available level when requested level is absent")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--gt-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--testsize", type=int, default=352)
    parser.add_argument("--limit", type=int, default=0, help="0 evaluates all images")
    parser.add_argument("--author-normalize", action="store_true",
                        help="match original test_GeleNet.py per-image min/max output")
    args = parser.parse_args()

    if args.level < 1:
        raise ValueError("level must be positive")
    selections = []
    pairs = paired_files(args.image_root, args.gt_root)
    if args.limit:
        pairs = pairs[:args.limit]
    output = Path(args.output).resolve()
    if output == Path(args.image_root).resolve() or output == Path(args.gt_root).resolve():
        raise ValueError("output must be separate from test data")
    output.mkdir(parents=True, exist_ok=True)
    transform = transforms.Compose([
        transforms.Resize((args.testsize, args.testsize)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    model = GeleNet().cuda()
    model.load_state_dict(torch.load(args.weights, map_location="cpu"))
    model.eval()

    maes, pred_fractions, gt_fractions, pred_means = [], [], [], []
    with torch.no_grad():
        for index, (stem, image_path, gt_path) in enumerate(pairs, 1):
            with Image.open(image_path) as source:
                image = transform(source.convert("RGB")).unsqueeze(0).cuda()
            with Image.open(gt_path) as source:
                gt = np.asarray(source.convert("L"), dtype=np.float32) / 255.0
            region_file = Path(args.regions) / (stem + ".mat")
            segs = loadmat(region_file)["segs"].ravel()
            if not len(segs) or (len(segs) < args.level and not args.allow_last_level):
                raise ValueError("Insufficient region levels: " + stem)
            selected = min(args.level, len(segs))
            region = np.asarray(segs[selected - 1])
            selections.append(dict(image=stem, available_levels=len(segs),
                                   requested_level=args.level, selected_level=selected,
                                   fallback=selected != args.level))
            if selected != args.level:
                print("LEVEL_FALLBACK {} requested={} selected={}".format(
                    stem, args.level, selected), flush=True)
            if region.shape != gt.shape or not np.issubdtype(region.dtype, np.integer):
                raise ValueError("Invalid region map: " + stem)
            region = np.array(Image.fromarray(region.astype(np.int32)).resize(
                (args.testsize, args.testsize), Image.NEAREST), dtype=np.int64)
            region = torch.from_numpy(region).unsqueeze(0).cuda()
            logits, _ = model(image, region_ids=region)
            logits = F.interpolate(logits, size=gt.shape, mode="bilinear", align_corners=False)
            probability = logits.sigmoid().cpu().numpy()[0, 0]
            if args.author_normalize:
                probability = (probability - probability.min()) / (
                    probability.max() - probability.min() + 1e-8)
            Image.fromarray(np.uint8(np.clip(probability * 255 + 0.5, 0, 255)), mode="L").save(
                output / (stem + ".png"))
            maes.append(float(np.mean(np.abs(probability - gt))))
            pred_fractions.append(float(np.mean(probability >= 0.5)))
            gt_fractions.append(float(np.mean(gt >= 0.5)))
            pred_means.append(float(np.mean(probability)))
            if index % 100 == 0 or index == len(pairs):
                print("Predicted {}/{}".format(index, len(pairs)), flush=True)
    print("diagnostic_only=True author_normalize={} images={} MAE={:.6f} pred_mean={:.6f} pred_fg_fraction={:.6f} gt_fg_fraction={:.6f}".format(
        args.author_normalize, len(pairs), np.mean(maes), np.mean(pred_means),
        np.mean(pred_fractions), np.mean(gt_fractions)))
    print("near_empty_predictions={} near_full_predictions={}".format(
        sum(v < 0.01 for v in pred_fractions), sum(v > 0.99 for v in pred_fractions)))
    (output.parent / (output.name + "_region_selection.json")).write_text(
        json.dumps(selections, indent=2), encoding="utf-8")
    print("prediction_dir={}".format(output))


if __name__ == "__main__":
    main()

