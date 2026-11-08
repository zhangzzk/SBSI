"""Small symmetric-proposal pseudo-marginal MH building block.

The likelihood estimate must be nonnegative and unbiased on the likelihood
scale. Its log need not be unbiased. Keep the current estimate AND its
auxiliary payload after rejection; refreshing it unconditionally is invalid.
"""
import numpy as np


def metropolis_accept(log_current, log_proposed, uniforms):
    old, new, u = (np.asarray(x, float) for x in (log_current, log_proposed, uniforms))
    if (old.shape != new.shape or old.shape != u.shape or np.isnan(old).any() or np.isnan(new).any()
            or np.isposinf(old).any() or np.isposinf(new).any() or not np.isfinite(u).all()
            or np.any((u <= 0) | (u >= 1))):
        raise ValueError('aligned log densities and uniforms strictly inside (0,1) required')
    with np.errstate(invalid='ignore'):
        return (np.log(u) < new-old) & np.isfinite(new)


def retained(current, proposed, accepted):
    """Keep rejected rows of a state, estimate or auxiliary-array payload."""
    old, new, use = np.asarray(current), np.asarray(proposed), np.asarray(accepted)
    if old.shape != new.shape or use.shape != old.shape[:1] or use.dtype != bool:
        raise ValueError('aligned arrays and one boolean acceptance per row required')
    return np.where(use.reshape(use.shape+(1,)*(old.ndim-1)), new, old)
