"""Frozen epoch schedule, independent of predictions and test scores."""
import math

FINAL_WEIGHTS=(0.50,0.10,0.15,0.25)
INITIAL_WEIGHTS=(0.90,0.02,0.03,0.05)


def validate_weights(weights):
    if len(weights)!=4 or any(not math.isfinite(w) or w<0 for w in weights):
        raise ValueError('Four finite nonnegative weights required')
    if abs(sum(weights)-1)>1e-6:
        raise ValueError('Weights must sum to one')
    return tuple(weights)


def epoch_weights(epoch, schedule='staged', warmup_epochs=10, ramp_epochs=10):
    if epoch<1 or warmup_epochs<0 or ramp_epochs<1:
        raise ValueError('Invalid epoch schedule counts')
    if schedule=='constant': return FINAL_WEIGHTS
    if schedule!='staged': raise ValueError('Unknown schedule')
    progress=min(max((epoch-warmup_epochs)/ramp_epochs,0.),1.)
    return validate_weights(tuple(a+(b-a)*progress for a,b in zip(INITIAL_WEIGHTS,FINAL_WEIGHTS)))
