"""Prior-proposal Metropolis on fixed summaries, including mixed measures.

The supplied proposal stream must consist of fresh independent draws from the
intended prior. For T(scene) and likelihood K(T), the target is K dP_T / Z.
The unknown pushforward prior cancels from the independence-MH ratio. This
does not justify local random walks in T without its density, nor does it
make a narrow posterior easy to explore. Rejections must remain in the chain.
"""
import numpy as np


def prior_metropolis(log_likelihood, uniforms):
    """Return proposal indices and acceptances for independent parallel chains.

    Arrays have shapes (chains, draws) and (chains, draws-1). The first draw
    initializes each chain without reference to generating truth or likelihood
    ranking. Each later draw is used once as a fresh prior proposal. Indices
    are into that stream, not a reusable finite catalogue. Burn-in and mixing
    diagnostics are the caller's responsibility; no samples are thinned away.
    -inf is allowed (zero likelihood); NaN and +inf are numerical failures.
    """
    values, u = np.asarray(log_likelihood, float), np.asarray(uniforms, float)
    if (values.ndim != 2 or min(values.shape) < 1
            or u.shape != (values.shape[0], values.shape[1]-1)
            or np.isnan(values).any() or np.isposinf(values).any()
            or not np.isfinite(u).all() or np.any((u <= 0) | (u >= 1))):
        raise ValueError('aligned log likelihoods and uniforms strictly inside (0,1) required')
    rows = np.arange(len(values))
    history = np.zeros(values.shape, dtype=np.int64)
    accepted = np.zeros(u.shape, dtype=bool)
    current = np.zeros(len(values), dtype=np.int64)
    for step in range(1, values.shape[1]):
        old, new = values[rows, current], values[:, step]
        # Both -inf => reject; finite proposals escape zero-density initials.
        with np.errstate(invalid='ignore'):
            take = (np.log(u[:, step-1]) < new-old) & np.isfinite(new)
        current = np.where(take, step, current)
        history[:, step] = current
        accepted[:, step-1] = take
    return history, accepted
