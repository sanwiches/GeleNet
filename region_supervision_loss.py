"""One model prediction, four independently normalized masked BCE terms."""
import torch
from weak_loss import partial_bce_with_logits
from region_weight_schedule import FINAL_WEIGHTS,validate_weights

WEIGHTS=FINAL_WEIGHTS


def region_supervision_loss(logits, target, valid_mask, region_targets, region_masks,weights=WEIGHTS):
    weights=validate_weights(weights)
    expected=(logits.shape[0],3,*logits.shape[-2:])
    if tuple(region_targets.shape)!=expected or tuple(region_masks.shape)!=expected:
        raise ValueError('Expected Bx3xHxW coarse/middle/fine supervision')
    terms=[partial_bce_with_logits(logits,target,valid_mask)]
    for i in range(3):
        terms.append(partial_bce_with_logits(logits,region_targets[:,i:i+1],region_masks[:,i:i+1]))
    total=sum(w*term for w,term in zip(weights,terms))
    return total, torch.stack(terms)
