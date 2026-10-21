"""Uncapped lag-one coupled independence MH on complete population summaries.

The fixed proposal callback draws independent (log Zhat, population observable)
pairs from Q. The invariant summary law is proportional to Zhat * Q. This is
not automatically the desired scientific target: that identification requires
E_Q[Zhat * observable] / E_Q[Zhat] to equal the intended expectation.

Indexing follows Middleton et al. (2019), Algorithm 3 and equation (2): the
first proposal initializes Y_0 while proposing X_1; both chains subsequently
use the same proposal and uniform. The first common acceptance defines tau.
The estimator includes X_m and corrections only through tau-1. Moment and
meeting-tail assumptions are the caller's responsibility, not established for
SBSI by this module. See doc/COUPLED_POPULATION_IMH.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


def _log_normalizer(value):
    array = np.asarray(value)
    if array.ndim != 0 or not np.issubdtype(array.dtype, np.number):
        raise ValueError('log normalizer must be a real scalar')
    if np.iscomplexobj(array):
        raise ValueError('log normalizer must be real')
    result = float(array)
    if np.isnan(result) or np.isposinf(result):
        raise ValueError('log normalizer must be finite or -inf')
    return result


@dataclass(frozen=True, eq=False)
class PopulationSummary:
    """One complete proposal population, compressed to the needed statistics.

    A known extinct population has log_normalizer=-inf. Its finite observable
    placeholder is canonicalized to zero, without dropping or redrawing it.
    NaN/inf observables are errors even for extinction; callers must explicitly
    distinguish known extinction from an unknown failed calculation.
    """
    log_normalizer: float
    observable: np.ndarray

    def __post_init__(self):
        logz = _log_normalizer(self.log_normalizer)
        raw = np.asarray(self.observable)
        if np.iscomplexobj(raw):
            raise ValueError('observable must be real')
        value = np.array(raw, dtype=np.float64, copy=True)
        if value.ndim != 1 or not value.size or not np.isfinite(value).all():
            raise ValueError('observable must be a nonempty finite vector')
        if np.isneginf(logz):
            value.fill(0.)
        value.flags.writeable = False
        object.__setattr__(self, 'log_normalizer', logz)
        object.__setattr__(self, 'observable', value)


def log_acceptance(current_logz, proposed_logz):
    """Independence-MH acceptance with explicit zero-weight conventions.

    From weight zero accept every proposal, including another zero (0/0=1).
    A positive-weight chain rejects a zero-weight proposal, even at uniform=0.
    The target gives zero weight to extinct states; this extension specifies
    their transient behavior without conditioning Q on positive weight.
    """
    current = _log_normalizer(current_logz)
    proposed = _log_normalizer(proposed_logz)
    if np.isneginf(current) or proposed >= current:
        return 0.
    if np.isneginf(proposed):
        return -np.inf
    # Finite extremes may overflow their negative difference. That represents
    # a known vanishing acceptance, not an invalid proposal normalizer.
    return proposed - current


def _accept(log_alpha, uniform):
    if np.isneginf(log_alpha):
        return False
    return uniform == 0 or np.log(uniform) < log_alpha


@dataclass(frozen=True)
class CoupledTransition:
    x: PopulationSummary
    y: PopulationSummary
    accepted_x: bool
    accepted_y: bool

    @property
    def common_acceptance(self):
        return self.accepted_x and self.accepted_y


def coupled_independence_step(x, y, proposal, uniform):
    """Update X_{n-1},Y_{n-2}; y=None means the initial Y_0 draw.

    Shared accepted proposals retain object identity. Equality of log Z or of
    observables is never used to declare a meeting of complete populations.
    Strict uniform < acceptance avoids accepting known zero probability at
    the representable RNG endpoint zero; it is the same continuous-uniform law.
    """
    for state in (x, proposal) if y is None else (x, y, proposal):
        if not isinstance(state, PopulationSummary):
            raise TypeError('states must be PopulationSummary instances')
        if state.observable.shape != x.observable.shape:
            raise ValueError('observable dimension changed')
    u = np.asarray(uniform)
    if u.ndim != 0 or np.iscomplexobj(u) or not np.isfinite(u) or not 0 <= u < 1:
        raise ValueError('uniform must be a finite scalar in [0,1)')
    accepted_x = _accept(log_acceptance(x.log_normalizer, proposal.log_normalizer), float(u))
    accepted_y = y is None or _accept(log_acceptance(y.log_normalizer, proposal.log_normalizer), float(u))
    return CoupledTransition(proposal if accepted_x else x,
                             proposal if accepted_y else y,
                             bool(accepted_x), bool(accepted_y))


class EstimationCancelled(RuntimeError):
    """An unfinished estimator, with no usable estimate attached.

    Cancellation is not a completed replicate. Averaging only replicates that
    finish before a time/meeting cutoff can introduce selection bias.
    """
    def __init__(self, iterations, proposal_calls):
        super().__init__('coupled estimator cancelled before completion; no estimate returned')
        self.iterations = iterations
        self.proposal_calls = proposal_calls


@dataclass(frozen=True)
class CoupledEstimate:
    estimate: np.ndarray
    mcmc_average: np.ndarray
    bias_correction: np.ndarray
    meeting_time: int
    iterations: int
    proposal_calls: int
    k: int
    m: int


def coupled_population_estimate(
    propose: Callable, rng: np.random.Generator, *, k=0, m=0,
    cancelled: Callable[[], bool] | None = None,
):
    """Return H_{k:m} only after meeting and simulation through X_m.

    ``propose(rng)`` must draw a fresh, independent complete population from
    the same fixed Q on every call, including the initial call. No real SMC
    adapter is provided here. Its exception, or an invalid summary, propagates.
    There is deliberately no maximum meeting time or partial-result option.
    ``cancelled()`` is checked before each proposal and raises if true.
    """
    if (isinstance(k, (bool, np.bool_)) or isinstance(m, (bool, np.bool_))
            or not isinstance(k, (int, np.integer))
            or not isinstance(m, (int, np.integer)) or not 0 <= k <= m):
        raise ValueError('require integer 0 <= k <= m')
    k, m = int(k), int(m)
    calls = 0

    def draw(iteration):
        nonlocal calls
        if cancelled is not None and cancelled():
            raise EstimationCancelled(iteration, calls)
        state = propose(rng)
        calls += 1
        if not isinstance(state, PopulationSummary):
            raise TypeError('proposal callback must return PopulationSummary')
        return state

    x = draw(0)
    y = None
    base = x.observable.copy() if k == 0 else np.zeros_like(x.observable)
    correction = np.zeros_like(x.observable)
    n, tau = 0, None
    width = m - k + 1
    # This explicit condition includes X_m, as required by equation (2), and
    # freezes the *first* meeting time even when m requires further updates.
    while tau is None or n < m:
        proposal = draw(n)
        transition = coupled_independence_step(x, y, proposal, rng.random())
        x, y = transition.x, transition.y
        n += 1
        if tau is None and transition.common_acceptance:
            tau = n
        with np.errstate(over='raise', invalid='raise'):
            if k <= n <= m:
                base += x.observable
            if n > k and tau is None:
                correction += min(1., (n-k)/width) * (x.observable-y.observable)
    with np.errstate(over='raise', invalid='raise'):
        average = base / width
        estimate = average + correction
    return CoupledEstimate(estimate, average, correction, tau, n, calls, k, m)
