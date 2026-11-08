"""Optional fixed, known-continuous-density initial Poisson augmentation.

See doc/INFORMED_ENVIRONMENT_PROPOSAL.md for the predeclared law and numerical
qualifications. Extended-space invalid draws are retained once, never retried.
Unknown numerical errors and finite-table generator failures always propagate.
"""
from copy import deepcopy
from dataclasses import dataclass, fields

import numpy as np
from scipy.special import expit, logsumexp

from .galsbi_particle_proposal import MeasurementParticleProposal
from .galsbi_scene_prior import ARCSEC2_PER_SR, GalSBISceneBatch
from .numerical_native_intensity import (luminosity_table_coverage,
    numerical_luminosity_log_density, numerical_native_log_intensity,
    numerical_redshift_log_density)
from .poisson_augmentation import point_log_augmentation_ratio, poisson_augmentation_log_tilt
from .scene_exchange import draw_relocation_positions, relocation_position_log_density


@dataclass
class EnvironmentPayload:
    """Tilt plus immutable-origin raw draw evidence, aligned with particle rows."""
    log_tilt: np.ndarray
    valid: np.ndarray
    augmented: np.ndarray
    raw_coordinates: np.ndarray
    raw_types: np.ndarray
    raw_offsets: np.ndarray
    insertion_index: np.ndarray
    classification: np.ndarray

    @classmethod
    def reference(cls, log_tilt):
        log_tilt = np.asarray(log_tilt, float)
        n = len(log_tilt)
        return cls(log_tilt.copy(), np.ones(n, bool), np.zeros(n, bool),
            np.zeros((n, 10)), np.full(n, -1, int), np.zeros((n, 2)),
            np.full(n, -1, int), np.full(n, 'base', dtype='<U40'))

    def take(self, rows):
        return type(self)(**{f.name: getattr(self, f.name)[rows].copy() for f in fields(self)})

    def with_tilt(self, log_tilt):
        result = self.take(np.arange(len(self.valid)))
        result.log_tilt[:] = log_tilt
        return result

    def scatter(self, rows, replacement):
        result = self.take(np.arange(len(self.valid)))
        for f in fields(self):
            old, new = getattr(result, f.name), getattr(replacement, f.name)
            if new.shape != (len(rows),)+old.shape[1:]:
                raise ValueError('environment payload shapes differ')
            old[rows] = new
        return result

    def retained(self, proposed, accepted):
        rows = np.flatnonzero(accepted)
        return self.scatter(rows, proposed.take(rows))


class InformedEnvironmentProposal:
    """Fixed observation-specific h and optional one-point Poisson mixture.

    The source proposals must be frozen Student-t laws. Phase is replaced by
    uniform, independent of the component and type. Metadata is reconstructed
    from source identity rather than copying stale nonuniform phase fields.
    """
    def __init__(self, prior, coordinates, frozen_proposal, source_sha256, *,
                 mixture_probability=.5, position_scale=.1, central_weight=.9):
        if (not isinstance(source_sha256, str) or len(source_sha256) != 64
                or any(c not in '0123456789abcdef' for c in source_sha256)):
            raise ValueError('source proposal artifact SHA256 required')
        if not np.isfinite(mixture_probability) or not 0 < mixture_probability < 1:
            raise ValueError('augmentation probability must lie strictly between zero and one')
        self.prior, self.coordinates = prior, coordinates
        self.mixture_probability = float(mixture_probability)
        self.position_parameters = dict(scale=position_scale,
            aperture_arcsec=prior.aperture_arcsec, central_weight=central_weight)
        # Validate normalized position-law parameters without drawing anything.
        relocation_position_log_density(np.empty((0, 2)), **self.position_parameters)
        proposals = []
        for index, proposal in enumerate(frozen_proposal.proposals):
            if not isinstance(proposal, MeasurementParticleProposal) or proposal.component_family != 'student_t':
                raise ValueError('frozen Student-t coordinate proposals required')
            proposals.append(MeasurementParticleProposal(proposal.means, proposal.scales, proposal.weights,
                component_family='student_t', student_df=proposal.student_df,
                metadata=dict(construction='frozen source Cartesian mixture with explicit uniform phase override',
                    source_artifact_sha256=source_sha256, source_object_index=index,
                    phase_override='independent uniform on [0,2pi), all components',
                    uses_generating_truth=False)))
        if not proposals:
            raise ValueError('nonempty frozen proposal list required')
        self.proposals = tuple(proposals)
        self._metadata = dict(source_artifact_sha256=source_sha256,
            observations=deepcopy(frozen_proposal.metadata['observations']),
            pilot_seeds=deepcopy(frozen_proposal.metadata['pilot_seeds']),
            mixture_probability=self.mixture_probability,
            law='(1-a) qEnv + a (qEnv plus one h point at a uniformly drawn ordered slot)',
            phase_override='independent uniform on [0,2pi), all components',
            type_law='independent uniform counting measure over prior.types',
            native_types=list(prior.galaxy.types), position=self.position_parameters.copy(),
            point_intensity_units='native-mark/type density per arcsec squared',
            retry_filter_or_refill=False,
            density_scope='continuous numerical interpolation component; clamp atoms and finite-RNG exactness not implemented',
            proposals=[p.metadata for p in proposals])

    @property
    def metadata(self):
        return deepcopy(self._metadata)

    def _owners(self, owner, size):
        owner = np.asarray(owner)
        if (owner.shape != (size,) or owner.dtype.kind not in 'iu'
                or np.any(owner < 0) or np.any(owner >= len(self.proposals))):
            raise ValueError('aligned original observation owners required')
        return owner

    def _decode_once(self, x, kinds):
        """Classify known support failures; numerical representation errors raise."""
        galaxy = self.prior.galaxy
        reasons = np.full(len(x), 'valid', dtype='<U40')
        zmax = np.array([galaxy.z_max[galaxy.types[k]] for k in kinds])
        outside = x[:, 0] > np.log(zmax)
        reasons[outside] = 'type_redshift_support'
        rows = np.flatnonzero(~outside)
        decoded = None
        if not len(rows):
            return rows, decoded, reasons
        v, p = x[rows], galaxy.p
        logits = np.column_stack((v[:, 2:6], np.zeros(len(v))))
        c = np.exp(logits-logsumexp(logits, axis=1, keepdims=True))
        z = np.exp(v[:, 0])
        nf, af = expit(v[:, 7]), expit(v[:, 8])
        if (np.any(z <= 0) or np.any(c <= 0) or np.any((nf <= 0) | (nf >= 1))
                or np.any((af <= 0) | (af >= 1))):
            raise FloatingPointError('raw environment decode lost representable interior support; no retry or zero-weight substitution')
        with np.errstate(over='raise', divide='raise', invalid='raise', under='ignore'):
            data, logj = self.coordinates.decode(v, kinds[rows])
        if (not np.isfinite(logj).all() or any(not np.isfinite(value).all() for value in data.values())
                or np.any(data['R_kpc'] <= 0) or np.any(data['n'] <= p['sersic_n_min'])
                or np.any(data['n'] >= p['sersic_n_max']) or np.any(data['a'] <= 0)
                or np.any(data['a'] >= p['ell_beta_emax'])):
            raise FloatingPointError('environment decoder returned nonrepresentable native values; no zero-weight substitution')
        local = np.full(len(rows), 'valid', dtype='<U40')
        cut = (data['m_i'] < p['gals_mag_min']) | (data['m_i'] > p['gals_mag_max'])
        local[cut] = 'intrinsic_photometric_cut'
        for kind, name in enumerate(galaxy.types):
            use = np.flatnonzero((kinds[rows] == kind) & ~cut)
            if not len(use):
                continue
            # Coverage validates the exact cap normalization. A failed cap
            # evaluation is an error, not a numerical-support classification.
            coverage = luminosity_table_coverage(galaxy, name, data['z'][use])
            atom = (coverage['clamp_atom_mass'] > 0) & (data['M_B'][use] == coverage['clamp_magnitude'])
            if atom.any():
                raise FloatingPointError('numerical luminosity clamp atom is outside the implemented continuous proposal measure')
            rz = numerical_redshift_log_density(galaxy, name, data['z'][use])
            rm = numerical_luminosity_log_density(galaxy, name, data['z'][use], data['M_B'][use])
            local[use[np.isneginf(rz)]] = 'numerical_redshift_support'
            local[use[np.isfinite(rz) & np.isneginf(rm)]] = 'numerical_luminosity_support'
        reasons[rows] = local
        kept = local == 'valid'
        decoded = {k: v[kept] for k, v in data.items()}
        if not np.isfinite(numerical_native_log_intensity(galaxy, decoded)).all():
            raise FloatingPointError('unexplained numerical native density failure on declared support')
        return rows[kept], decoded, reasons

    def log_tilt(self, scene, owner):
        owner = self._owners(owner, len(scene.counts))
        galaxy, data = self.prior.galaxy, scene.neighbours
        log_lambda = numerical_native_log_intensity(galaxy, data)-np.log(ARCSEC2_PER_SR)
        if not np.isfinite(log_lambda).all():
            raise FloatingPointError('existing environment has nonpositive/unknown numerical density; cannot label sampler errors as zero mass')
        if len(log_lambda):
            for kind, name in enumerate(galaxy.types):
                use = data['type'] == kind
                if use.any():
                    coverage = luminosity_table_coverage(galaxy, name, data['z'][use])
                    if np.any((coverage['clamp_atom_mass'] > 0)
                              & (data['M_B'][use] == coverage['clamp_magnitude'])):
                        raise FloatingPointError('existing environment contains an unsupported numerical clamp atom')
            x = self.coordinates.encode(data)
            if not np.isfinite(x).all():
                raise FloatingPointError('existing numerical mark is not representable in informed coordinates')
            decoded, logj = self.coordinates.decode(x, data['type'])
            if not np.isfinite(logj).all():
                raise FloatingPointError('nonfinite native-coordinate Jacobian')
            log_h = np.empty(len(x))
            point_owner = owner[scene.owner]
            for index, proposal in enumerate(self.proposals):
                use = point_owner == index
                log_h[use] = proposal.log_prob(x[use])
            log_h += (relocation_position_log_density(scene.offset_arcsec, **self.position_parameters)
                      -np.log(len(galaxy.types))-logj)
            if not np.isfinite(log_h).all():
                raise FloatingPointError('informed point density failed inside declared environment support')
        else:
            log_h = np.empty(0)
        ratios = point_log_augmentation_ratio(log_h, log_lambda)
        return poisson_augmentation_log_tilt(ratios, scene.indptr, self.mixture_probability)

    def continuous_exchange_support(self, scene):
        """Known continuous-support gate after the older native swap correction.

        The native q/p helper now enforces the U>=float.tiny restriction for
        every exchange. This additional optional-law check also rejects any
        continuous-zero neighbour before evaluating its undefined finite tilt.
        Cap-normalization failures
        and clamp atoms still raise; remaining density failures are checked by
        log_tilt and must never be converted to support zeros.
        """
        galaxy, data = self.prior.galaxy, scene.neighbours
        valid_point = np.ones(len(data['z']), bool)
        for kind, name in enumerate(galaxy.types):
            use = np.flatnonzero(data['type'] == kind)
            if not len(use):
                continue
            coverage = luminosity_table_coverage(galaxy, name, data['z'][use])
            if np.any((coverage['clamp_atom_mass'] > 0)
                      & (data['M_B'][use] == coverage['clamp_magnitude'])):
                raise FloatingPointError('exchange encountered an unsupported numerical clamp atom')
            z_density = numerical_redshift_log_density(galaxy, name, data['z'][use])
            m_density = numerical_luminosity_log_density(galaxy, name, data['z'][use], data['M_B'][use])
            if (np.isnan(z_density).any() or np.isnan(m_density).any()
                    or np.isposinf(z_density).any() or np.isposinf(m_density).any()):
                raise FloatingPointError('unknown conditional density failure in exchange support check')
            valid_point[use] = np.isfinite(z_density) & np.isfinite(m_density)
        supported = np.ones(len(scene.counts), bool)
        supported[np.unique(scene.owner[~valid_point])] = False
        return supported

    def augment(self, scene, owner, rng):
        """One raw augmentation draw per selected scene; invalid rows stay flagged."""
        owner = self._owners(owner, len(scene.counts))
        n = len(owner)
        payload = EnvironmentPayload.reference(np.zeros(n))
        payload.augmented = rng.random(n) < self.mixture_probability
        rows = np.flatnonzero(payload.augmented)
        if len(rows):
            for index, proposal in enumerate(self.proposals):
                use = rows[owner[rows] == index]
                payload.raw_coordinates[use] = proposal.sample(len(use), rng)
            payload.raw_types[rows] = rng.integers(len(self.prior.galaxy.types), size=len(rows))
            payload.raw_offsets[rows] = draw_relocation_positions(len(rows), rng, **self.position_parameters)
            payload.insertion_index[rows] = rng.integers(scene.counts[rows]+1)
            kept, data, reasons = self._decode_once(payload.raw_coordinates[rows], payload.raw_types[rows])
            payload.classification[rows] = reasons
            payload.valid[rows] = reasons == 'valid'
            inserted = rows[kept]
            if len(inserted):
                if data.keys() != scene.neighbours.keys():
                    raise ValueError('informed and numerical generator native schemas differ')
                counts = scene.counts.copy()
                counts[inserted] += 1
                pointers = np.r_[0, np.cumsum(counts)]
                new_at = pointers[inserted]+payload.insertion_index[inserted]
                old_at = np.setdiff1d(np.arange(pointers[-1]), new_at, assume_unique=True)
                neighbours = {}
                for key, value in scene.neighbours.items():
                    neighbours[key] = np.empty((pointers[-1],)+value.shape[1:], dtype=value.dtype)
                    neighbours[key][old_at], neighbours[key][new_at] = value, data[key]
                offsets = np.empty((pointers[-1], 2))
                offsets[old_at], offsets[new_at] = scene.offset_arcsec, payload.raw_offsets[inserted]
                scene = GalSBISceneBatch(scene.targets, neighbours, pointers, offsets)
        valid = np.flatnonzero(payload.valid)
        if len(valid):
            payload.log_tilt[valid] = self.log_tilt(scene.take(valid), owner[valid])
        return scene, payload
