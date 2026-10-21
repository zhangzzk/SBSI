"""Absorbing zero-mass populations without dropping seeds or other objects."""
from copy import copy

import numpy as np

from .annealed_particles import reweight, effective_sample_size, stratified_resample


EXTINCTION_SCHEMA = 'per-observation-zero-mass-v1'


def masked_rows(values, alive):
    """JSON rows with null entries exactly for observations without an estimate."""
    values, alive = np.asarray(values), np.asarray(alive, bool)
    if values.ndim < 1 or values.shape[0] != len(alive):
        raise ValueError('values must align with the observation mask')
    use = alive.reshape(alive.shape+(1,)*(values.ndim-1))
    return np.where(use, values.astype(object), None).tolist()


def subset_target(target, rows):
    """Keep original observation/proposal identities while selecting row slots."""
    selected = copy(target)
    selected.owner = target.owner[rows].copy()
    selected.observed = target.observed.iloc[rows].reset_index(drop=True)
    selected.auxiliary = target.auxiliary[rows].copy()
    # The audit dictionary is deliberately shared: all actual evaluations count.
    return selected


def replace_particles(state, rows, replacement):
    """Scatter complete particle and ragged-scene payloads without aliasing."""
    rows = np.asarray(rows)
    n = len(state.x)
    if (rows.ndim != 1 or rows.dtype.kind not in 'iu' or not len(rows)
            or np.any(rows >= n) or len(np.unique(rows)) != len(rows)
            or len(replacement.x) != len(rows)):
        raise ValueError('unique in-range particle rows matching replacement required')
    if rows.dtype.kind == 'i' and np.any(rows < 0):
        raise ValueError('particle rows must be nonnegative')
    old, new = state.scene, replacement.scene
    if old.targets.keys() != new.targets.keys() or old.neighbours.keys() != new.neighbours.keys():
        raise ValueError('replacement native scene schemas differ')
    combined = type(old)(
        {k:np.concatenate((v, new.targets[k])) for k,v in old.targets.items()},
        {k:np.concatenate((v, new.neighbours[k])) for k,v in old.neighbours.items()},
        np.r_[old.indptr, new.indptr[1:]+old.indptr[-1]],
        np.concatenate((old.offset_arcsec, new.offset_arcsec)))
    selected = np.arange(n)
    selected[rows] = n+np.arange(len(rows))
    fields = {}
    for name in ('x', 'log_prior', 'log_kernel', 'log_proposal', 'type_probability', 'summaries'):
        value, updated = getattr(state, name), getattr(replacement, name)
        if updated.shape != (len(rows),)+value.shape[1:]:
            raise ValueError('replacement particle payload shapes differ')
        fields[name] = value.copy()
        fields[name][rows] = updated
    return type(state)(scene=combined.take(selected), **fields)


class ParticlePopulations:
    """Advance fixed independent observation blocks, with absorbing extinction.

    Only an exactly all-zero incremental weight vector declares extinction.
    Errors, NaNs and unknown failures still raise. An extinct population retains
    Zhat=0 and its seed forever; it is never redrawn or normalized uniformly.
    The all-surviving path preserves the original mutation calls and RNG order.
    """
    def __init__(self, target, observations, particles):
        self.target, self.particles = target, particles
        expected = np.repeat(np.arange(observations), particles)
        if not np.array_equal(target.owner, expected):
            raise ValueError('contiguous original observation particle blocks required')
        self.alive = np.ones(observations, dtype=bool)
        self.extinction_stage = np.full(observations, -1, dtype=int)
        self.logz = np.zeros(observations)
        self.labels = np.tile(np.arange(particles), (observations, 1))

    def reweight_resample(self, state, delta, stage, rng):
        count = self.particles
        potential = state.potential.reshape(len(self.alive), count)
        ess, rows = [], []
        for i in range(len(self.alive)):
            if self.alive[i]:
                weights, increment = reweight(np.full(count, -np.log(count)), potential[i], delta)
                if np.isneginf(increment):
                    self.alive[i] = False
                    self.extinction_stage[i] = stage
                    self.logz[i] = -np.inf
                else:
                    if not np.isfinite(increment):
                        raise ValueError('unknown nonfinite particle normalizer')
                    self.logz[i] += increment
                    if not np.isfinite(self.logz[i]):
                        raise ValueError('normalizer accumulation overflow is not particle extinction')
                    ess.append(float(effective_sample_size(weights)))
                    chosen = stratified_resample(weights, rng)
                    self.labels[i] = self.labels[i, chosen]
                    rows.extend(i*count+chosen)
                    continue
            ess.append(0.)
            rows.extend(i*count+np.arange(count))
        return state.take(np.asarray(rows, int)), ess

    def mutate(self, state, beta, rng, **kwargs):
        if self.alive.all():
            return self.target.mutate(state, beta, rng, **kwargs)
        decisions = np.zeros(len(state.x), bool)
        active = decisions.copy()
        rows = np.flatnonzero(np.repeat(self.alive, self.particles))
        if not len(rows):
            return state, decisions, active
        target = subset_target(self.target, rows)
        moved, decisions[rows], active[rows] = target.mutate(state.take(rows), beta, rng, **kwargs)
        return replace_particles(state, rows, moved), decisions, active

    def receipt(self):
        return dict(extinction_schema=EXTINCTION_SCHEMA, extinct=(~self.alive).tolist(),
            extinction_stage=[None if v < 0 else int(v) for v in self.extinction_stage],
            log_normalizer=masked_rows(self.logz, self.alive))


def survivor_diagnostics(target, state, alive, particles, likelihood, score_function):
    """Evaluate replay/scores only for surviving populations; zero placeholders otherwise.

    Placeholder arrays have an explicit observation mask in their saved artifact;
    they do not represent posterior estimates. No model diagnostics run if all
    populations are extinct. score_function is called once on surviving rows.
    """
    from .galsbi_summary import SummaryBatch
    rows = np.flatnonzero(np.repeat(alive, particles))
    full, score = np.zeros((len(state.x), 9)), np.zeros((len(state.x), 2))
    if not len(rows):
        return full, score, None
    all_rows = len(rows) == len(state.x)
    selected_target = target if all_rows else subset_target(target, rows)
    selected_state = state if all_rows else state.take(rows)
    selected_full = selected_target.full_summaries(selected_state)
    replay = likelihood.log_kernel(SummaryBatch(selected_full, selected_target.auxiliary),
        selected_target.observed, *selected_target.shear)
    if not np.isfinite(replay).all() or not np.isfinite(selected_state.log_kernel).all():
        raise ValueError('surviving population final replay has nonfinite likelihood')
    error = float(np.max(np.abs(replay-selected_state.log_kernel)))
    if error > 1e-6:
        raise ValueError(f'final-state full replay differs: {error}')
    selected_score = np.asarray(score_function(selected_full, selected_target.observed), float)
    if selected_score.shape != (len(rows), 2) or not np.isfinite(selected_score).all():
        raise ValueError('surviving population scores must be aligned and finite')
    full[rows], score[rows] = selected_full, selected_score
    return full, score, error
