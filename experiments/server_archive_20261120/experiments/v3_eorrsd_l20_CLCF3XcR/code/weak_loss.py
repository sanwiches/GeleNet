"""Sparse supervision loss for GeleNet's unchanged saliency logits."""

import torch
import torch.nn.functional as F


def partial_bce_with_logits(logits, target, valid_mask):
    """Mean BCE over annotated pixels, with a zero loss for an empty mask."""
    if logits.shape != target.shape or logits.shape != valid_mask.shape:
        raise ValueError("logits, target and valid_mask must have identical shapes")
    if not torch.all((valid_mask == 0) | (valid_mask == 1)):
        raise ValueError("valid_mask must be binary")
    if not torch.all((target[valid_mask.bool()] == 0) |
                     (target[valid_mask.bool()] == 1)):
        raise ValueError("target at valid pixels must be binary")
    # Replace ignored values before BCE too: ignored targets may be undefined.
    safe_target = torch.where(valid_mask.bool(), target, torch.zeros_like(target))
    per_pixel = F.binary_cross_entropy_with_logits(logits, safe_target, reduction="none")
    return (per_pixel * valid_mask).sum() / valid_mask.sum().clamp_min(1)
