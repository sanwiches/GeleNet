"""Image-level mixtures of NAMLab membership priors, not image-pyramid attention."""
import torch
from torch import nn
from torch.nn import functional as F


class ScalePrior(nn.Module):
    def __init__(self, channels=32, levels=(10, 20, 40), mode='adaptive'):
        super().__init__()
        self.levels = tuple(levels)
        if self.levels != (10, 20, 40):
            raise ValueError('This experiment fixes candidate order to 10,20,40')
        if mode not in ('fixed20', 'uniform', 'adaptive'):
            raise ValueError('Unknown scale mode')
        self.mode = mode
        self.gate = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Linear(channels, 16), nn.ReLU(), nn.Linear(16, 3))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)
        self.last_weights = None
        self.last_active = None

    def forward(self, features, region_ids):
        b, _, h, w = features.shape
        if region_ids.ndim != 4 or tuple(region_ids.shape[:2]) != (b, 3):
            raise ValueError('region_ids must be Bx3xHxW, ordered 10,20,40')
        if region_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError('Region IDs must be integers')
        if region_ids.device != features.device:
            raise ValueError('Features and regions must share a device')
        if torch.any(region_ids < 0) or torch.any(region_ids > 2**24):
            raise ValueError('Region IDs outside exact float32 interpolation range')
        regions = F.interpolate(region_ids.float(), (h, w), mode='nearest').long()
        flat = regions.flatten(2)
        relations = flat.unsqueeze(-1).eq(flat.unsqueeze(-2))
        # Deduplicate actual memberships AFTER projection to the KTM grid.
        # Prefer level20 as representative, then10, then40. Labels may differ.
        active = torch.zeros((b, 3), dtype=torch.bool, device=features.device)
        kept = []
        for i in (1, 0, 2):
            valid = torch.ones(b, dtype=torch.bool, device=features.device)
            for j in kept:
                same = (relations[:, i] == relations[:, j]).flatten(1).all(1)
                valid = valid & ~(active[:, j] & same)
            active[:, i] = valid
            kept.append(i)
        if self.mode == 'fixed20':
            weights = torch.zeros((b, 3), device=features.device, dtype=features.dtype)
            weights[:, 1] = 1
        elif self.mode == 'uniform':
            weights = active.to(features.dtype)
            weights = weights / weights.sum(1, keepdim=True)
        else:
            logits = self.gate(features)
            weights = logits.masked_fill(~active, float('-inf')).softmax(1)
        self.last_weights = weights.detach()
        self.last_active = active.detach()
        return (relations.to(features.dtype) * weights[:, :, None, None]).sum(1)
