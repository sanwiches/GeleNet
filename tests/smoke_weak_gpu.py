"""One synthetic forward/backward step; run with CUDA_VISIBLE_DEVICES set."""

import torch

from model.GeleNet_models import GeleNet
from weak_loss import partial_bce_with_logits


def main():
    assert torch.cuda.is_available(), "CUDA is required"
    torch.manual_seed(0)
    model = GeleNet().cuda().train()
    images = torch.randn(1, 3, 352, 352, device="cuda")
    target = torch.zeros(1, 1, 352, 352, device="cuda")
    valid_mask = torch.zeros_like(target)
    target[:, :, 50:60, 50:60] = 1
    valid_mask[:, :, 50:60, 50:60] = 1
    valid_mask[:, :, 200:210, 200:210] = 1
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    optimizer.zero_grad()
    logits, probability = model(images)
    assert logits.shape == target.shape
    assert probability.shape == target.shape
    loss = partial_bce_with_logits(logits, target, valid_mask)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())
    optimizer.step()
    print("GPU smoke test passed; loss={:.6f}; peak memory={:.2f} GiB".format(
        loss.item(), torch.cuda.max_memory_allocated() / 1024**3))


if __name__ == "__main__":
    main()
