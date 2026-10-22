"""Exact scene bookkeeping for experimental annealing from an evaluable q0.

The bridge is q_environment(E) q0(x)**(1-beta) [p(x) K(y|x,E)]**beta.
p is the selected type-marginalized native intensity in transformed target
coordinates (analytic by default); q_environment is the unchanged numerical
Poisson scene law. The optional numerical target uses its continuous density.
The normalized initial q0 is a proposal, never a replacement population prior.
"""
from dataclasses import dataclass

import numpy as np

from .catalogue_disk_response import CatalogueDiskResponse
from .galsbi_scene_prior import GalSBISceneBatch
from .galsbi_summary import SummaryBatch, compress_scenes, scene_model_inputs
from .pseudo_marginal import metropolis_accept, retained
from .scene_exchange import propose_native_exchange


def bridge_log_density(log_prior, log_kernel, log_proposal, beta, log_environment_tilt=None):
    """Reduced bridge density, with endpoint arithmetic safe for -inf support."""
    if not np.isfinite(beta) or not 0 <= beta <= 1:
        raise ValueError('bridge beta must be in [0,1]')
    lp, lk, lq = np.broadcast_arrays(log_prior, log_kernel, log_proposal)
    if np.isnan(lp).any() or np.isnan(lk).any() or not np.isfinite(lq).all():
        raise ValueError('valid prior/kernel and finite full-support proposal required')
    if np.isposinf(lp).any() or np.isposinf(lk).any():
        raise ValueError('positive infinite target density')
    if beta == 0:
        result = lq.copy()
    else:
        result = beta*(lp+lk)+(1-beta)*lq
    if log_environment_tilt is not None:
        tilt = np.asarray(log_environment_tilt)
        if tilt.shape != result.shape or not np.isfinite(tilt).all():
            raise ValueError('aligned finite environment log tilt required')
        result = result+(1-beta)*tilt
    return result


@dataclass
class ParticleScenes:
    x: np.ndarray
    scene: GalSBISceneBatch
    log_prior: np.ndarray
    log_kernel: np.ndarray
    log_proposal: np.ndarray
    type_probability: np.ndarray
    summaries: np.ndarray
    environment_payload: object = None

    @property
    def potential(self):
        result = self.log_prior+self.log_kernel-self.log_proposal
        if self.environment_payload is not None:
            result = result-self.environment_payload.log_tilt
        return result

    def take(self, rows):
        """Copy complete particle payloads, including repeated ancestors.

        Observation ownership belongs to GalSBIParticleTarget, not this payload.
        Resampling must stay within each observation and preserve its row slots.
        """
        return ParticleScenes(self.x[rows].copy(), self.scene.take(rows),
            self.log_prior[rows].copy(), self.log_kernel[rows].copy(),
            self.log_proposal[rows].copy(), self.type_probability[rows].copy(),
            self.summaries[rows].copy(),
            None if self.environment_payload is None else self.environment_payload.take(rows))


class GalSBIParticleTarget:
    def __init__(self, prior, coordinates, likelihood, response_model,
                 angular_distance, conditions, observations, proposals, owner, shear,
                 physical_geometry=None, informed_environment=None):
        self.prior, self.coordinates, self.likelihood = prior, coordinates, likelihood
        self.response_model, self.angular_distance = response_model, angular_distance
        self.conditions, self.observations = conditions, observations
        self.proposals, self.owner = proposals, np.asarray(owner)
        self.observed = observations.iloc[self.owner].reset_index(drop=True)
        self.auxiliary = self.observed.iloc[:, 2:4].to_numpy()
        self.shear = np.asarray(shear, float)
        self.physical_geometry = physical_geometry
        self.informed_environment = informed_environment
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

    def evaluate(self, x, scene, *, reuse=None, environment_payload=None):
        lp, data, probabilities = self.native_prior(x)
        scene = GalSBISceneBatch(data, scene.neighbours, scene.indptr, scene.offset_arcsec)
        lq = self.log_proposal(x)
        lk = np.full(len(x), -np.inf)
        summaries = np.zeros((len(x), 9))
        eligible = np.isfinite(lp)
        if self.informed_environment is not None:
            if environment_payload is None:
                from .informed_environment import EnvironmentPayload
                environment_payload = EnvironmentPayload.reference(
                    self.informed_environment.log_tilt(scene, self.owner))
            if environment_payload.valid.shape != (len(x),):
                raise ValueError('environment validity must align with particle rows')
            eligible &= environment_payload.valid
        elif environment_payload is not None:
            raise ValueError('environment payload requires the optional informed law')
        valid = np.flatnonzero(eligible)
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
        return ParticleScenes(x, scene, lp, lk, lq, probabilities, summaries, environment_payload)

    def initialize(self, rng):
        x = np.empty((len(self.owner), 10))
        for i, proposal in enumerate(self.proposals):
            selected = self.owner == i
            x[selected] = proposal.sample(int(selected.sum()), rng)
        _, data, _ = self.native_prior(x)
        scene, _ = self.prior.sample_given_targets(data, int(rng.integers(0, 2**63)))
        if self.informed_environment is not None:
            scene, payload = self.informed_environment.augment(scene, self.owner, rng)
            return self.evaluate(x, scene, environment_payload=payload)
        # Invalid prior draws are retained with zero target weight, not retried.
        return self.evaluate(x, scene)

    def mutate(self, state, beta, rng, *, block, scale=1., check_reuse=False):
        """One fixed MH kernel; no covariance/scale fit to these particles.

        Native exchanges use the selected p-conditional auxiliary type law.
        Their proposal-density correction is independent of beta, so the
        original native ratio minus delta log p is added to the bridge ratio.
        """
        if self.informed_environment is not None:
            if (beta <= 0 or state.environment_payload is None
                    or not state.environment_payload.valid.all()):
                raise ValueError('informed initial environments require positive-beta weighting/resampling before mutation')
        x, scene = state.x.copy(), state.scene
        hastings = np.zeros(len(x))
        active = np.ones(len(x), dtype=bool)
        temperature_scale = min(4., 1/np.sqrt(max(beta, .01)))
        if block == 'physical':
            columns, widths = [1, 6, 7, 8, 9], np.array([.08, .08, .2, .2, .2])
            if self.physical_geometry is None:
                x[:, columns] += rng.normal(size=(len(x), 5))*widths*scale*temperature_scale
            else:
                x[:, columns] += self.physical_geometry.increments(self.owner, beta, rng)*scale
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
        elif block in ('exchange', 'exchange_relocate'):
            # p(type|x) is undefined outside p support, which q0 can visit at
            # beta=0. Use a null exchange there; the symmetric support gate
            # leaves the bridge invariant and other kernels can enter support.
            rows = np.flatnonzero(np.isfinite(state.log_prior))
            active = np.zeros(len(x), dtype=bool)
            native_ratio = np.zeros(len(x))
            if len(rows):
                relocation = (dict(relocation_scale=.1, aperture_arcsec=self.prior.aperture_arcsec,
                                   central_weight=.9) if block == 'exchange_relocate' else {})
                if getattr(self.coordinates, 'target_prior', 'analytic') == 'numerical':
                    relocation['target_prior'] = 'numerical'
                swapped, native_ratio[rows], active[rows] = propose_native_exchange(
                    self.prior.galaxy, scene.take(rows), state.type_probability[rows],
                    rng, radius_scale=1., **relocation)
                targets = {k: v.copy() for k, v in scene.targets.items()}
                neighbours = {k: v.copy() for k, v in scene.neighbours.items()}
                selected_neighbours = np.isin(scene.owner, rows)
                for key in targets:
                    targets[key][rows] = swapped.targets[key]
                    neighbours[key][selected_neighbours] = swapped.neighbours[key]
                offsets = scene.offset_arcsec
                if block == 'exchange_relocate':
                    offsets = offsets.copy()
                    offsets[selected_neighbours] = swapped.offset_arcsec
                scene = GalSBISceneBatch(targets, neighbours, scene.indptr, offsets)
                x[rows] = self.coordinates.encode(swapped.targets)
        elif block == 'independence':
            for i, proposal in enumerate(self.proposals):
                selected = self.owner == i
                x[selected] = proposal.sample(int(selected.sum()), rng)
            hastings = state.log_proposal-self.log_proposal(x)
        else:
            raise ValueError('unknown particle mutation block')
        reuse = state if block in ('nuisance', 'sed') else None
        payload = state.environment_payload
        if payload is not None and block in ('exchange', 'exchange_relocate'):
            # The existing native correction proves these swaps off support.
            # Retain safe old payloads for their certain rejections; never
            # evaluate a likelihood or finite tilt on a zero-base scene.
            supported = np.isfinite(native_ratio)
            if not supported.all():
                scene = state.scene.retained(scene, supported)
                x[~supported] = state.x[~supported]
            continuous = self.informed_environment.continuous_exchange_support(scene)
            if not continuous.all():
                native_ratio[~continuous] = -np.inf
                scene = state.scene.retained(scene, continuous)
                x[~continuous] = state.x[~continuous]
        if payload is not None and block in ('environment', 'positions', 'exchange', 'exchange_relocate'):
            payload = payload.with_tilt(self.informed_environment.log_tilt(scene, self.owner))
        proposed = self.evaluate(x, scene, reuse=reuse, environment_payload=payload)
        if block in ('exchange', 'exchange_relocate'):
            hastings[:] = -np.inf
            valid = np.isfinite(proposed.log_prior) & np.isfinite(native_ratio)
            hastings[valid] = native_ratio[valid]-(proposed.log_prior[valid]-state.log_prior[valid])
        if reuse is not None and check_reuse:
            actual = self.evaluate(x, scene, environment_payload=payload)
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
        old_density = bridge_log_density(state.log_prior, state.log_kernel, state.log_proposal, beta,
            None if state.environment_payload is None else state.environment_payload.log_tilt)
        new_density = bridge_log_density(proposed.log_prior, proposed.log_kernel, proposed.log_proposal, beta,
            None if proposed.environment_payload is None else proposed.environment_payload.log_tilt)
        accepted = metropolis_accept(old_density, new_density+hastings,
            rng.uniform(np.finfo(float).tiny, 1., len(x))) & active
        if block in ('environment', 'positions', 'exchange', 'exchange_relocate'):
            new_scene = state.scene.retained(proposed.scene, accepted)
        else:
            new_scene = GalSBISceneBatch(
                {k: retained(v, proposed.scene.targets[k], accepted) for k, v in state.scene.targets.items()},
                state.scene.neighbours, state.scene.indptr, state.scene.offset_arcsec)
        updated = ParticleScenes(retained(state.x, proposed.x, accepted), new_scene,
            *[retained(old, new, accepted) for old, new in [
                (state.log_prior, proposed.log_prior), (state.log_kernel, proposed.log_kernel),
                (state.log_proposal, proposed.log_proposal),
                (state.type_probability, proposed.type_probability), (state.summaries, proposed.summaries)]],
            environment_payload=(None if state.environment_payload is None else
                state.environment_payload.retained(proposed.environment_payload, accepted)))
        return updated, accepted, active

    def full_summaries(self, state):
        """Restore exact response coefficients for final direct-score diagnostics."""
        rows = np.arange(len(state.x))
        return self.summarize(state.scene, rows, full_response=True).values
