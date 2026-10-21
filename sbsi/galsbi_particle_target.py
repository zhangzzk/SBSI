"""Exact scene bookkeeping for experimental annealing from an evaluable q0.

The bridge is q_environment(E) q0(x)**(1-beta) [p(x) K(y|x,E)]**beta.
p is the analytic type-marginalized native intensity in transformed target
coordinates; q_environment is the unchanged numerical Poisson scene law.
The normalized initial q0 is a proposal, never a replacement population prior.
"""
from dataclasses import dataclass

import numpy as np

from .catalogue_disk_response import CatalogueDiskResponse
from .galsbi_scene_prior import GalSBISceneBatch
from .galsbi_summary import SummaryBatch, compress_scenes, scene_model_inputs
from .pseudo_marginal import metropolis_accept, retained
from .scene_exchange import propose_native_exchange


def bridge_log_density(log_prior, log_kernel, log_proposal, beta):
    """Reduced bridge density, with endpoint arithmetic safe for -inf support."""
    if not np.isfinite(beta) or not 0 <= beta <= 1:
        raise ValueError('bridge beta must be in [0,1]')
    lp, lk, lq = np.broadcast_arrays(log_prior, log_kernel, log_proposal)
    if np.isnan(lp).any() or np.isnan(lk).any() or not np.isfinite(lq).all():
        raise ValueError('valid prior/kernel and finite full-support proposal required')
    if np.isposinf(lp).any() or np.isposinf(lk).any():
        raise ValueError('positive infinite target density')
    if beta == 0:
        return lq.copy()
    return beta*(lp+lk)+(1-beta)*lq


@dataclass
class ParticleScenes:
    x: np.ndarray
    scene: GalSBISceneBatch
    log_prior: np.ndarray
    log_kernel: np.ndarray
    log_proposal: np.ndarray
    type_probability: np.ndarray
    summaries: np.ndarray

    @property
    def potential(self):
        return self.log_prior+self.log_kernel-self.log_proposal

    def take(self, rows):
        """Copy complete particle payloads, including repeated ancestors.

        Observation ownership belongs to GalSBIParticleTarget, not this payload.
        Resampling must stay within each observation and preserve its row slots.
        """
        return ParticleScenes(self.x[rows].copy(), self.scene.take(rows),
            self.log_prior[rows].copy(), self.log_kernel[rows].copy(),
            self.log_proposal[rows].copy(), self.type_probability[rows].copy(),
            self.summaries[rows].copy())


class GalSBIParticleTarget:
    def __init__(self, prior, coordinates, likelihood, response_model,
                 angular_distance, conditions, observations, proposals, owner, shear):
        self.prior, self.coordinates, self.likelihood = prior, coordinates, likelihood
        self.response_model, self.angular_distance = response_model, angular_distance
        self.conditions, self.observations = conditions, observations
        self.proposals, self.owner = proposals, np.asarray(owner)
        self.observed = observations.iloc[self.owner].reset_index(drop=True)
        self.auxiliary = self.observed.iloc[:, 2:4].to_numpy()
        self.shear = np.asarray(shear, float)
        self.lazy_response = bool(np.all(self.shear == 0))
        self.audit = dict(likelihood_calls=0, likelihood_rows=0, nuisance_replay_max=0.,
                          summary_replay_max=0., nuisance_checks=0)

    def log_proposal(self, x):
        result = np.empty(len(x))
        for i, proposal in enumerate(self.proposals):
            selected = self.owner == i
            result[selected] = proposal.log_prob(x[selected])
        return result

    def native_prior(self, x):
        """Evaluate p without calling external photometry outside its z domain.

        A full-support q0 has legitimate draws beyond every population z_max.
        Their original coordinates and q0 density remain unchanged. Finite
        native placeholders only allow the ragged scene container to hold
        these zero-weight draws; they never enter a likelihood evaluation.
        """
        x = np.asarray(x, float)
        if x.ndim != 2 or x.shape[1] != 10 or not np.isfinite(x).all():
            raise ValueError('finite ten-coordinate particle matrix required')
        zmax = max(self.prior.galaxy.z_max.values())
        with np.errstate(over='ignore', under='ignore'):
            z = np.exp(x[:, 0])
        in_range = (z > 0) & (z <= zmax)
        safe = x
        if not in_range.all():
            safe = x.copy()
            safe[~in_range] = 0.
            safe[~in_range, 0] = np.log(zmax/2)
            safe[~in_range, 1] = 20.
        lp, data, probabilities = self.coordinates.log_mixture_prior(safe)
        lp[~in_range] = -np.inf
        probabilities[~in_range] = 0.
        return lp, data, probabilities

    def summarize(self, scene, rows, *, full_response=False):
        response = full_response or not self.lazy_response
        zero, ptr, features = scene_model_inputs(scene, self.prior, self.angular_distance,
            self.conditions, response_radius_arcsec=10., response_neighbours=20,
            response=response)
        if response:
            return compress_scenes(zero, CatalogueDiskResponse(ptr, features, self.response_model),
                                   self.auxiliary[rows])
        return SummaryBatch(np.column_stack((zero.to_numpy(), np.zeros(len(zero)))),
                            self.auxiliary[rows])

    def evaluate(self, x, scene, *, reuse=None):
        lp, data, probabilities = self.native_prior(x)
        scene = GalSBISceneBatch(data, scene.neighbours, scene.indptr, scene.offset_arcsec)
        lq = self.log_proposal(x)
        lk = np.full(len(x), -np.inf)
        summaries = np.zeros((len(x), 9))
        valid = np.flatnonzero(np.isfinite(lp))
        if len(valid):
            if reuse is not None:
                # An off-prior initial q0 draw has never had its likelihood
                # evaluated. Entering support requires a fresh kernel even
                # when the physical features themselves have not changed.
                cached = valid[np.isfinite(reuse.log_prior[valid])]
                lk[cached], summaries[cached] = reuse.log_kernel[cached], reuse.summaries[cached]
                valid = valid[~np.isfinite(reuse.log_prior[valid])]
            if len(valid):
                summary = self.summarize(scene.take(valid), valid)
                lk[valid] = self.likelihood.log_kernel(summary, self.observed.iloc[valid], *self.shear)
                summaries[valid] = summary.values
                self.audit['likelihood_calls'] += 1
                self.audit['likelihood_rows'] += len(valid)
        return ParticleScenes(x, scene, lp, lk, lq, probabilities, summaries)

    def initialize(self, rng):
        x = np.empty((len(self.owner), 10))
        for i, proposal in enumerate(self.proposals):
            selected = self.owner == i
            x[selected] = proposal.sample(int(selected.sum()), rng)
        _, data, _ = self.native_prior(x)
        scene, _ = self.prior.sample_given_targets(data, int(rng.integers(0, 2**63)))
        # Invalid prior draws are retained with zero target weight, not retried.
        return self.evaluate(x, scene)

    def mutate(self, state, beta, rng, *, block, scale=1., check_reuse=False):
        """One fixed MH kernel; no covariance/scale fit to these particles.

        Native exchanges use the original p-conditional auxiliary type law.
        Their proposal-density correction is independent of beta, so the
        original native ratio minus delta log p is added to the bridge ratio.
        """
        x, scene = state.x.copy(), state.scene
        hastings = np.zeros(len(x))
        active = np.ones(len(x), dtype=bool)
        temperature_scale = min(4., 1/np.sqrt(max(beta, .01)))
        if block == 'physical':
            columns, widths = [1, 6, 7, 8, 9], np.array([.08, .08, .2, .2, .2])
            x[:, columns] += rng.normal(size=(len(x), 5))*widths*scale*temperature_scale
            x[:, 9] %= 2*np.pi
        elif block == 'nuisance':
            columns = [0, 2, 3, 4, 5]
            x[:, columns] += rng.normal(size=(len(x), 5))*.2*scale*temperature_scale
        elif block == 'sed':
            # SED moves hold z fixed. Outside its native domain a null move
            # is reversible; evaluating redshift-dependent Dirichlet parameters
            # there can overflow even though the particle has zero p weight.
            with np.errstate(over='ignore', under='ignore'):
                z = np.exp(x[:, 0])
            active = (z > 0) & (z <= max(self.prior.galaxy.z_max.values()))
            if active.any():
                x[active], hastings[active] = self.coordinates.propose_sed(x[active], rng)
        elif block == 'environment':
            fraction = 1. if rng.random() < .1 else .1
            scene = self.prior.refresh(scene, fraction, int(rng.integers(0, 2**63)))
        elif block == 'positions':
            scene = self.prior.move_positions(scene, .2*scale, int(rng.integers(0, 2**63)))
        elif block == 'exchange':
            # p(type|x) is undefined outside p support, which q0 can visit at
            # beta=0. Use a null exchange there; the symmetric support gate
            # leaves the bridge invariant and other kernels can enter support.
            rows = np.flatnonzero(np.isfinite(state.log_prior))
            active = np.zeros(len(x), dtype=bool)
            native_ratio = np.zeros(len(x))
            if len(rows):
                swapped, native_ratio[rows], active[rows] = propose_native_exchange(
                    self.prior.galaxy, scene.take(rows), state.type_probability[rows],
                    rng, radius_scale=1.)
                targets = {k: v.copy() for k, v in scene.targets.items()}
                neighbours = {k: v.copy() for k, v in scene.neighbours.items()}
                selected_neighbours = np.isin(scene.owner, rows)
                for key in targets:
                    targets[key][rows] = swapped.targets[key]
                    neighbours[key][selected_neighbours] = swapped.neighbours[key]
                scene = GalSBISceneBatch(targets, neighbours, scene.indptr, scene.offset_arcsec)
                x[rows] = self.coordinates.encode(swapped.targets)
        elif block == 'independence':
            for i, proposal in enumerate(self.proposals):
                selected = self.owner == i
                x[selected] = proposal.sample(int(selected.sum()), rng)
            hastings = state.log_proposal-self.log_proposal(x)
        else:
            raise ValueError('unknown particle mutation block')
        reuse = state if block in ('nuisance', 'sed') else None
        proposed = self.evaluate(x, scene, reuse=reuse)
        if block == 'exchange':
            hastings[:] = -np.inf
            valid = np.isfinite(proposed.log_prior) & np.isfinite(native_ratio)
            hastings[valid] = native_ratio[valid]-(proposed.log_prior[valid]-state.log_prior[valid])
        if reuse is not None and check_reuse:
            actual = self.evaluate(x, scene)
            if not np.array_equal(np.isfinite(proposed.log_kernel), np.isfinite(actual.log_kernel)):
                raise ValueError('particle nuisance cache changes likelihood support')
            valid = np.isfinite(proposed.log_kernel) & np.isfinite(actual.log_kernel)
            if valid.any():
                kernel_error = np.max(np.abs(proposed.log_kernel[valid]-actual.log_kernel[valid]))
                summary_error = np.max(np.abs(proposed.summaries[valid]-actual.summaries[valid]))
                self.audit['nuisance_replay_max'] = max(self.audit['nuisance_replay_max'], float(kernel_error))
                self.audit['summary_replay_max'] = max(self.audit['summary_replay_max'], float(summary_error))
                self.audit['nuisance_checks'] += int(valid.sum())
                if kernel_error > 1e-6 or summary_error > 1e-10:
                    raise ValueError('particle nuisance cache fails full replay')
        old_density = bridge_log_density(state.log_prior, state.log_kernel, state.log_proposal, beta)
        new_density = bridge_log_density(proposed.log_prior, proposed.log_kernel, proposed.log_proposal, beta)
        accepted = metropolis_accept(old_density, new_density+hastings,
            rng.uniform(np.finfo(float).tiny, 1., len(x))) & active
        if block in ('environment', 'positions', 'exchange'):
            new_scene = state.scene.retained(proposed.scene, accepted)
        else:
            new_scene = GalSBISceneBatch(
                {k: retained(v, proposed.scene.targets[k], accepted) for k, v in state.scene.targets.items()},
                state.scene.neighbours, state.scene.indptr, state.scene.offset_arcsec)
        updated = ParticleScenes(retained(state.x, proposed.x, accepted), new_scene,
            *[retained(old, new, accepted) for old, new in [
                (state.log_prior, proposed.log_prior), (state.log_kernel, proposed.log_kernel),
                (state.log_proposal, proposed.log_proposal),
                (state.type_probability, proposed.type_probability), (state.summaries, proposed.summaries)]])
        return updated, accepted, active

    def full_summaries(self, state):
        """Restore exact response coefficients for final direct-score diagnostics."""
        rows = np.arange(len(state.x))
        return self.summarize(state.scene, rows, full_response=True).values
