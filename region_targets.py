"""Conservative NAMLab scribble propagation; no dense GT or random targets."""
import numpy as np


def quartile_levels(count):
    if count < 1:
        raise ValueError('Empty hierarchy')
    return [max(count//4, 1), max(count//2, 1), max(3*count//4, 1)]


def propagate(regions, target, valid):
    if regions.ndim != 2 or regions.shape != target.shape or target.shape != valid.shape:
        raise ValueError('Native region and scribble shapes must match')
    if not np.issubdtype(regions.dtype, np.integer):
        raise ValueError('Region IDs must be integers')
    if not np.isin(valid,[0,1]).all() or not np.isin(target[valid.astype(bool)],[0,1]).all():
        raise ValueError('Expected binary valid mask and observed labels')
    _, inverse = np.unique(regions.ravel(), return_inverse=True)
    inverse=inverse.ravel()
    ids=inverse.reshape(regions.shape)
    fg=np.bincount(inverse,weights=((target==1)&(valid==1)).ravel())
    bg=np.bincount(inverse,weights=((target==0)&(valid==1)).ravel(),minlength=len(fg))
    accepted=(fg>0)^(bg>0)
    labels=(fg>0)[ids].astype(np.uint8)
    # Original observations remain exclusively in the original scribble loss.
    mask=(accepted[ids] & (valid==0)).astype(np.uint8)
    labels[mask==0]=0
    stats=dict(regions=len(fg),foreground_only=int(((fg>0)&(bg==0)).sum()),
        background_only=int(((bg>0)&(fg==0)).sum()),
        conflict=int(((fg>0)&(bg>0)).sum()),empty=int(((fg==0)&(bg==0)).sum()),
        expanded_pixels=int(mask.sum()))
    return labels, mask, stats
