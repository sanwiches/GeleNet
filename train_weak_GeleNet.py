"""Version 2 scribble training; the original dense trainer remains untouched."""

import argparse
import json
from pathlib import Path

import torch

from model.GeleNet_models import GeleNet
from utils import adjust_lr, clip_gradient
from weak_data import get_weak_loader
from weak_loss import partial_bce_with_logits


def read_samples(manifest):
    """JSON list of explicit [image, target, valid_mask] path triplets."""
    manifest = Path(manifest)
    entries = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(entries, list) or not entries:
        raise ValueError("manifest must be a nonempty JSON list")
    samples = []
    for entry in entries:
        if not isinstance(entry, list) or len(entry) != 3 or any(
                not isinstance(path, str) for path in entry):
            raise ValueError("each manifest entry must be three path strings")
        samples.append(tuple((manifest.parent / path).resolve() for path in entry))
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epoch", type=int, default=45)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batchsize", type=int, default=8)
    parser.add_argument("--trainsize", type=int, default=352)
    parser.add_argument("--clip", type=float, default=0.5)
    parser.add_argument("--decay_rate", type=float, default=0.1)
    parser.add_argument("--decay_epoch", type=int, default=30)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()
    if args.epoch <= 1:
        parser.error("epoch must be greater than 1 (matches original range(1, epoch))")
    if not torch.cuda.is_available():
        parser.error("CUDA is required for this training entry point")

    loader = get_weak_loader(read_samples(args.manifest), args.batchsize,
                             args.trainsize, num_workers=args.num_workers,
                             pin_memory=True)
    model = GeleNet().cuda()
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epoch):
        adjust_lr(optimizer, args.lr, epoch, args.decay_rate, args.decay_epoch)
        model.train()
        for step, batch in enumerate(loader, 1):
            images = batch["image"].cuda(non_blocking=True)
            target = batch["target"].cuda(non_blocking=True)
            valid_mask = batch["valid_mask"].cuda(non_blocking=True)
            optimizer.zero_grad()
            logits, _ = model(images)
            loss = partial_bce_with_logits(logits, target, valid_mask)
            loss.backward()
            clip_gradient(optimizer, args.clip)
            optimizer.step()
            if step % 20 == 0 or step == len(loader):
                print("Epoch {}/{}, step {}/{}, Partial BCE {:.4f}".format(
                    epoch, args.epoch - 1, step, len(loader), loss.item()))
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), output / "GeleNet.pth.{}".format(epoch),
                       _use_new_zipfile_serialization=False)


if __name__ == "__main__":
    main()
