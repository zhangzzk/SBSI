"""Finite-prior complement proposal for an experimental exact-centre pilot."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TailDistribution:
    cdf: np.ndarray
    log_center_mass: float
    prior_mass: float
    maximum_cdf_probability_error: float

    def inverse(self, uniforms):
        u = np.asarray(uniforms, dtype=np.float64)
        if u.ndim != 1 or not np.isfinite(u).all() or np.any((u < 0) | (u >= 1)):
            raise ValueError('uniforms must be a finite vector in [0, 1)')
        atoms = np.searchsorted(self.cdf, u, side='right')
        previous = np.zeros(len(atoms), dtype=np.float64)
        positive = atoms > 0
        previous[positive] = self.cdf[atoms[positive]-1]
        probability = self.cdf[atoms]-previous
        if not np.all(probability > 0):
            raise ValueError('inverse CDF returned a zero-probability atom')
        return atoms, probability


def build_tail_distribution(log_contribution, prior_weights, candidates, *, delta):
    """Mix normalized exact-centre tail mass with normalized prior tail mass.

    Candidates are summed separately and receive exactly zero probability.
    The defensive prior term covers every remaining positive-prior atom, even
    when its centre likelihood is zero. Return probabilities of the actual
    floating-point CDF bins rather than unrounded intended probabilities.
    """
    logs = np.array(log_contribution, dtype=np.float64, copy=True)
    prior = np.array(prior_weights, dtype=np.float64, copy=True)
    selected = np.asarray(candidates, dtype=np.int64)
    if (logs.ndim != 1 or prior.shape != logs.shape
            or np.isnan(logs).any() or np.isposinf(logs).any()
            or not np.isfinite(prior).all() or np.any(prior < 0)):
        raise ValueError('aligned valid log contributions and prior weights required')
    if (selected.ndim != 1 or len(np.unique(selected)) != len(selected)
            or np.any((selected < 0) | (selected >= len(logs)))):
        raise ValueError('candidate indices must be unique and in range')
    if not 0 < delta < 1:
        raise ValueError('defensive fraction must lie strictly between zero and one')
    if np.any(np.isfinite(logs) & (prior == 0)):
        raise ValueError('finite contributions require positive prior support')
    logs[selected], prior[selected] = -np.inf, 0.
    prior_mass = float(prior.sum(dtype=np.float64))
    peak = float(logs.max())
    if prior_mass <= 0 or not np.isfinite(peak):
        raise ValueError('complement must have positive centre and prior mass')
    likelihood = np.exp(logs-peak)
    center_mass = float(likelihood.sum(dtype=np.float64))
    q = (1-delta)*(likelihood/center_mass) + delta*(prior/prior_mass)
    cdf = np.cumsum(q, dtype=np.float64)
    total = float(cdf[-1])
    cdf /= total
    widths = np.diff(cdf, prepend=0.)
    if not np.all(widths[prior > 0] > 0) or np.any(widths[prior == 0] != 0):
        raise ValueError('CDF rounding changed prior support')
    error = float(np.max(np.abs(widths-q/total)))
    return TailDistribution(cdf, peak+float(np.log(center_mass)), prior_mass, error)


def draw_tail(distribution, *, n_draws, seed, object_id):
    """Use the same per-object uniform column and nested prefixes as v1.2."""
    if n_draws <= 0 or object_id < 0:
        raise ValueError('positive draw count and non-negative object id required')
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(object_id)]))
    return distribution.inverse(rng.random((n_draws, 3))[:, 1])
