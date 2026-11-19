"""One real scribble batch on the selected CUDA device."""

import argparse

import torch

from model.GeleNet_models import GeleNet
from train_weak_GeleNet import read_samples
from weak_data import get_weak_loader
from weak_loss import partial_bce_with_logits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    samples = read_samples(args.manifest)
    batch = next(iter(get_weak_loader(samples, batchsize=2, trainsize=352,
                                      shuffle=False, num_workers=0)))
    target = batch["target"].cuda()
    valid = batch["valid_mask"].cuda()
    if valid.sum().item() == 0 or (target * valid).sum().item() == 0:
        raise ValueError("first batch lacks valid foreground labels")
    model = GeleNet().cuda().train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    optimizer.zero_grad()
    logits, probability = model(batch["image"].cuda())
    if logits.shape != target.shape or probability.shape != target.shape:
        raise ValueError("model output does not match target shape")
    loss = partial_bce_with_logits(logits, target, valid)
    if not torch.isfinite(loss):
        raise ValueError("non-finite loss")
    loss.backward()
    optimizer.step()
    print("Real batch passed: {} samples, {} valid pixels, loss {:.6f}".format(
        len(batch["image"]), int(valid.sum().item()), loss.item()))


if __name__ == "__main__":
    main()
