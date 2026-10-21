"""Fixed-schedule waste-free SMC, isolated from the native scene driver.

Resample M=N/P ancestors and retain each ancestor plus P-1 invariant iterates.
See doc/WASTE_FREE_SMC_PROTOTYPE.md for the finite-population normalizer and
weighted-observable identities. The callback is one fixed composite kernel;
population-fitted proposals and convergence-based stopping are not supported.
"""
from __future__ import annotations

import numpy as np

from .annealed_particles import (ParticleResult, ParticleStage, _log_vector,
    effective_ancestors, effective_sample_size, normalized_log_weights, reweight)


def stratified_ancestors(log_weights, count, rng):
    """Unbiased stratified resampling of count ancestors from N weighted rows."""
    if isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer)) or count < 1:
        raise ValueError('ancestor count must be a positive integer')
    weights = np.exp(normalized_log_weights(log_weights))
    if not weights.sum():
        raise ValueError('cannot resample an extinct population')
    cumulative = np.cumsum(weights)
    cumulative /= cumulative[-1]
    cumulative[-1] = 1.
    positions = np.minimum((np.arange(count)+rng.random(count))/count, np.nextafter(1., 0.))
    return np.searchsorted(cumulative, positions, side='right')


def waste_free_smc(initial_particles, log_potential, betas, rng, *, path_length,
                   mutation=None):
    """Return N retained states after every weight/resample/path stage.

    The normalized initial proposal must supply all N draws, including zero
    target mass. N must be divisible by path_length. Each mutation invocation
    acts on M=N/path_length rows and must preserve the current bridge for each
    row. It may be a fixed composition of invariant blocks. Every retained
    state is a copy; an in-place callback cannot overwrite an earlier point.

    Final weights are uniform unless extinct. Weight/ancestral ESS do not count
    independent posterior draws. Replicate complete independent populations.
    """
    particles = np.array(initial_particles, copy=True)
    if particles.ndim < 1 or not len(particles) or not np.isfinite(particles).all():
        raise ValueError('initial_particles must be nonempty and finite')
    if (isinstance(path_length, (bool, np.bool_))
            or not isinstance(path_length, (int, np.integer))
            or path_length < 1 or len(particles) % path_length):
        raise ValueError('path_length must be a positive integer dividing N')
    schedule = np.asarray(betas, float)
    if (schedule.ndim != 1 or len(schedule) < 2 or not np.isfinite(schedule).all()
            or schedule[0] != 0 or schedule[-1] != 1 or not np.all(np.diff(schedule) > 0)):
        raise ValueError('betas must strictly increase from zero to one')
    n, p = len(particles), int(path_length)
    m = n//p
    weights = np.full(n, -np.log(n))
    ancestors = np.arange(n)
    potential = _log_vector(log_potential(particles), 'log_potential output')
    if potential.shape != (n,):
        raise ValueError('log_potential must return one value per particle')
    total, stages = 0., []
    for previous, beta in zip(schedule[:-1], schedule[1:]):
        weights, increment = reweight(weights, potential, beta-previous)
        total += increment
        ess = effective_sample_size(weights)
        if ess == 0:
            stages.append(ParticleStage(float(beta), increment, total, 0., 0, 0.))
            return ParticleResult(particles, weights, total, ancestors, tuple(stages), True)
        indices = stratified_ancestors(weights, m, rng)
        current = particles[indices].copy()
        paths = [current.copy()]
        for _ in range(p-1):
            if mutation is not None:
                moved = np.asarray(mutation(current.copy(), float(beta), rng))
                if moved.shape != current.shape or not np.isfinite(moved).all():
                    raise ValueError('mutation must preserve finite particle shape')
                current = moved.copy()
            paths.append(current.copy())
        particles = np.concatenate(paths, axis=0)
        ancestors = np.tile(ancestors[indices], p)
        weights = np.full(n, -np.log(n))
        if mutation is None or p == 1:
            potential = np.tile(potential[indices], p)
        else:
            potential = _log_vector(log_potential(particles), 'log_potential output')
            if potential.shape != (n,):
                raise ValueError('log_potential must return one value per particle')
            if np.isneginf(potential).any():
                raise ValueError('mutation left positive target support')
        stages.append(ParticleStage(float(beta), increment, total, ess,
            len(np.unique(ancestors)), effective_ancestors(ancestors)))
    return ParticleResult(particles, weights, total, ancestors, tuple(stages), False)
