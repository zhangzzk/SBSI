"""Fixed-schedule annealed importance particles and resampling bookkeeping.

The path is ``gamma_beta(x) = q0(x) * exp(beta * log_potential(x))``.
Here q0 is a *normalized* proposal and log_potential is log p + log K - log
q0. The initial population must be drawn from the full q0, including samples
with zero target mass; filtering those samples loses a normalizing factor.

For a schedule and invariant mutation kernels fixed independently of the
production population, unbiased resampling gives an unbiased estimate of Z,
provided the initial draws have law q0 and q0 covers the target support. This
does not make log Z or normalized posterior expectations unbiased. Population
adaptation is not covered by this guarantee. A pilot may define a schedule and
mutation kernels that are then frozen for fresh independent production draws.

Weight ESS and ancestral ESS diagnose weight concentration and family collapse.
Neither is a count of independent posterior samples, nor a convergence test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


def _log_vector(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or not array.size:
        raise ValueError(f"{name} must be a nonempty one-dimensional array")
    if np.isnan(array).any() or np.isposinf(array).any():
        raise ValueError(f"{name} may contain finite values and -inf only")
    return array


def normalized_log_weights(log_weights: np.ndarray) -> np.ndarray:
    """Normalize finite/-inf log weights, retaining zero-mass particles.

    An all-minus-infinity input returns all minus infinity: this is an extinct
    population with total mass zero, not a uniform probability distribution.
    """
    weights = _log_vector(log_weights, "log_weights")
    largest = np.max(weights)
    if np.isneginf(largest):
        return weights.copy()
    shifted = weights - largest
    return shifted - np.log(np.exp(shifted).sum())


def effective_sample_size(log_weights: np.ndarray) -> float:
    """Return weight ESS, or zero for an extinct population."""
    weights = np.exp(normalized_log_weights(log_weights))
    squared = np.dot(weights, weights)
    return float(1.0 / squared) if squared else 0.0


def reweight(
    log_weights: np.ndarray, log_potential: np.ndarray, delta: float
) -> tuple[np.ndarray, float]:
    """Return normalized new weights and the log normalizer increment.

    The input weights are normalized internally. At delta=0, -inf potentials
    leave their weights unchanged, implementing the beta=0 proposal correctly.
    At positive delta they contribute zero, while remaining in the original
    proposal denominator. Extinction returns (all -inf, -inf); do not discard
    or rerun extinct replicates when estimating a mean normalizing constant.
    """
    weights = normalized_log_weights(log_weights)
    potential = _log_vector(log_potential, "log_potential")
    if weights.shape != potential.shape:
        raise ValueError("weights and potential must have identical shapes")
    if not np.isfinite(delta) or delta < 0:
        raise ValueError("delta must be finite and nonnegative")
    if np.isneginf(weights).all():
        return weights, float("-inf")
    if delta == 0:
        return weights, 0.0
    with np.errstate(over="ignore"):
        updated = weights + delta * potential
    if np.isposinf(updated).any() or np.isnan(updated).any():
        raise ValueError("reweighted log mass overflowed")
    largest = np.max(updated)
    if np.isneginf(largest):
        return updated, float("-inf")
    log_increment = largest + np.log(np.exp(updated - largest).sum())
    return normalized_log_weights(updated), float(log_increment)


def stratified_resample(
    log_weights: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Return N stratified ancestor indices with unbiased offspring counts."""
    weights = np.exp(normalized_log_weights(log_weights))
    if not weights.sum():
        raise ValueError("cannot resample an extinct population")
    cumulative = np.cumsum(weights)
    # Roundoff must not select beyond N. side='right' skips exact zero masses,
    # including a possible exact zero uniform variate at the left endpoint.
    cumulative /= cumulative[-1]
    cumulative[-1] = 1.0
    positions = (np.arange(len(weights)) + rng.random(len(weights))) / len(weights)
    positions = np.minimum(positions, np.nextafter(1.0, 0.0))
    return np.searchsorted(cumulative, positions, side="right")


def effective_ancestors(
    ancestors: np.ndarray, log_weights: np.ndarray | None = None
) -> float:
    """Inverse concentration of initial-ancestor mass, not posterior ESS.

    Duplicate labels are pooled before calculating 1 / sum(family_mass**2).
    Uniform current particle weights are used when log_weights is omitted.
    """
    labels = np.asarray(ancestors)
    if labels.ndim != 1 or not labels.size or labels.dtype.kind not in "iu":
        raise ValueError("ancestors must be a nonempty integer vector")
    if np.any(labels < 0):
        raise ValueError("ancestor labels must be nonnegative")
    if log_weights is None:
        weights = np.full(len(labels), 1.0 / len(labels))
    else:
        weights = np.exp(normalized_log_weights(log_weights))
        if weights.shape != labels.shape:
            raise ValueError("ancestors and weights must have identical shapes")
    _, inverse = np.unique(labels, return_inverse=True)
    masses = np.bincount(inverse, weights=weights)
    squared = np.dot(masses, masses)
    return float(1.0 / squared) if squared else 0.0


@dataclass(frozen=True)
class ParticleStage:
    beta: float
    log_increment: float
    log_normalizer: float
    weight_ess: float
    unique_ancestors: int
    ancestral_ess: float


@dataclass(frozen=True)
class ParticleResult:
    particles: np.ndarray
    log_weights: np.ndarray
    log_normalizer: float
    ancestors: np.ndarray
    stages: tuple[ParticleStage, ...]
    extinct: bool


def annealed_smc(
    initial_particles: np.ndarray,
    log_potential: Callable[[np.ndarray], np.ndarray],
    betas: np.ndarray,
    rng: np.random.Generator,
    *,
    mutation: Callable[[np.ndarray, float, np.random.Generator], np.ndarray] | None = None,
    mutation_steps: int = 1,
) -> ParticleResult:
    """Run a fixed schedule with reweight/resample/mutate at every stage.

    Each callback mutation must preserve q0 * exp(beta * log_potential), with
    any tuning fixed independently of these production particles. The callback
    receives the whole population for vectorization; population-dependent
    proposals require their own invariance proof and are not assumed valid.
    ``None`` is the identity kernel. Final particles have equal weights unless
    extinct. Initial ancestry is retained across every resampling operation.

    This deliberately small driver has no adaptation or convergence stopping.
    Compare fresh independent runs for posterior and normalizer uncertainty.
    """
    particles = np.array(initial_particles, copy=True)
    if particles.ndim < 1 or not len(particles) or not np.isfinite(particles).all():
        raise ValueError("initial_particles must be nonempty and finite")
    schedule = np.asarray(betas, dtype=np.float64)
    if (
        schedule.ndim != 1 or len(schedule) < 2
        or not np.isfinite(schedule).all()
        or schedule[0] != 0 or schedule[-1] != 1
        or not np.all(np.diff(schedule) > 0)
    ):
        raise ValueError("betas must strictly increase from exactly zero to one")
    if (
        isinstance(mutation_steps, (bool, np.bool_))
        or not isinstance(mutation_steps, (int, np.integer))
        or mutation_steps < 0
    ):
        raise ValueError("mutation_steps must be a nonnegative integer")
    count = len(particles)
    weights = np.full(count, -np.log(count))
    ancestors = np.arange(count)
    potential = _log_vector(log_potential(particles), "log_potential output")
    if potential.shape != weights.shape:
        raise ValueError("log_potential must return one value per particle")
    total = 0.0
    stages = []
    for previous, beta in zip(schedule[:-1], schedule[1:]):
        weights, increment = reweight(weights, potential, beta - previous)
        total += increment
        weight_ess = effective_sample_size(weights)
        if weight_ess == 0:
            stages.append(ParticleStage(float(beta), increment, total, 0.0, 0, 0.0))
            return ParticleResult(particles, weights, total, ancestors, tuple(stages), True)
        indices = stratified_resample(weights, rng)
        particles = particles[indices]
        potential = potential[indices]
        ancestors = ancestors[indices]
        weights = np.full(count, -np.log(count))
        if mutation is not None and mutation_steps:
            for _ in range(mutation_steps):
                moved = np.asarray(mutation(particles, float(beta), rng))
                if moved.shape != particles.shape or not np.isfinite(moved).all():
                    raise ValueError("mutation must return finite particles with unchanged shape")
                particles = moved
            potential = _log_vector(log_potential(particles), "log_potential output")
            if potential.shape != weights.shape:
                raise ValueError("log_potential must return one value per particle")
            if np.isneginf(potential).any():
                raise ValueError("mutation returned particles outside positive target support")
        stages.append(ParticleStage(
            float(beta), increment, total, weight_ess,
            int(len(np.unique(ancestors))), effective_ancestors(ancestors),
        ))
    return ParticleResult(particles, weights, total, ancestors, tuple(stages), False)
