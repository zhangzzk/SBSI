"""Normalized object-specific initial laws frozen from independent pilots.

These are proposal heuristics, not estimates or corrections of the posterior.
See doc/PILOT_INITIAL_PROPOSAL.md for the predeclared n13 fitting rule.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.special import i0e, i1e

from .galsbi_particle_proposal import MeasurementParticleProposal


SCHEMA = 'sbsi-frozen-particle-initial-proposal-v2'
NATIVE_TARGET_SOURCES = ('galsbi_target_coordinates.py', 'galsbi_summary.py',
                         'summary_gradients.py', 'models_v3_9_like.json')
IDENTITY_KEYS = ('observations', 'fixed_shear', 'reference_protocol_sha256',
                 'observation_sha256', 'density_batch_size', 'bridge', 'selection')
FIT_RULE = dict(pilots=4, broad_mass=.2, student_df=5., sd_inflation=1.5,
                broad_sd_floor_fraction=.05, variance_ddof=0, phase_kappa_cap=1e4,
                phase_resultant_power=1.5**2,
                floor_applied_before_t_conversion=True,
                floor_reference='original measurement coordinate scale (Gaussian SD; retained broad t shape SD)')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_fingerprint(protocol):
    """Retain every recorded source, with snapshot-root-independent names."""
    raw = protocol.get('implementation_sha256')
    if not isinstance(raw, dict) or not raw:
        raise ValueError('recorded implementation source hashes required')
    result = {}
    for path, value in raw.items():
        name = Path(path).name
        if (name in result or not isinstance(value, str) or len(value) != 64
                or any(c not in '0123456789abcdef' for c in value)):
            raise ValueError('unique source basenames and valid SHA256 hashes required')
        result[name] = value
    if any(name not in result for name in NATIVE_TARGET_SOURCES):
        raise ValueError('recorded native target source hashes incomplete')
    return result


def target_identity(protocol):
    result = {key: deepcopy(protocol[key]) for key in IDENTITY_KEYS}
    sources = source_fingerprint(protocol)
    result['native_target_source_sha256'] = {name: sources[name] for name in NATIVE_TARGET_SOURCES}
    return result


def _validate_metadata(metadata, objects):
    observations, seeds = metadata.get('observations'), metadata.get('pilot_seeds')
    if (not isinstance(observations, list) or len(observations) != objects
            or any(type(v) is not int or v < 0 for v in observations)
            or len(set(observations)) != objects):
        raise ValueError('unique ordered nonnegative observation IDs required')
    if (not isinstance(seeds, list) or len(seeds) != FIT_RULE['pilots']
            or any(type(v) is not int or v < 0 for v in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError('four distinct nonnegative pilot seeds required')
    shear = np.asarray(metadata.get('fixed_shear'), float)
    if shear.shape != (2,) or not np.isfinite(shear).all() or np.hypot(*shear) >= 1:
        raise ValueError('physical fixed shear required')
    for key in IDENTITY_KEYS:
        if key not in metadata:
            raise ValueError(f'missing target identity: {key}')
    sources = metadata.get('native_target_source_sha256', {})
    source_fingerprint({'implementation_sha256': sources})
    for key in ('reference_protocol_sha256', 'observation_sha256'):
        value = metadata[key]
        if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('valid reference and observation hashes required')
    if type(metadata['density_batch_size']) is not int or metadata['density_batch_size'] <= 0:
        raise ValueError('positive numerical density batch required')


class FrozenParticleProposal:
    """A frozen list of normalized proposals with independent-pilot provenance."""

    def __init__(self, proposals, metadata):
        if not proposals or any(not isinstance(p, MeasurementParticleProposal) for p in proposals):
            raise ValueError('nonempty list of normalized measurement proposals required')
        _validate_metadata(metadata, len(proposals))
        self._metadata = deepcopy(metadata)
        self.proposals = tuple(proposals)

    @property
    def metadata(self):
        return deepcopy(self._metadata)

    def validate_production(self, seed, protocol):
        if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
            raise ValueError('production seed must be a nonnegative integer')
        if int(seed) in self._metadata['pilot_seeds']:
            raise ValueError('production seed reuses an independent pilot seed')
        identity = target_identity(protocol)
        for key, value in identity.items():
            if value != self._metadata[key]:
                raise ValueError(f'production target identity differs from pilot: {key}')

    def save(self, path):
        payload = dict(schema=SCHEMA, coordinate_names=list(MeasurementParticleProposal.coordinate_names),
                       proposals=[p.metadata for p in self.proposals], metadata=self.metadata)
        with Path(path).open('x') as handle:
            json.dump(payload, handle, indent=2, allow_nan=False)
            handle.write('\n')

    @classmethod
    def load(cls, path):
        payload = json.loads(Path(path).read_text())
        if (payload.get('schema') != SCHEMA
                or payload.get('coordinate_names') != list(MeasurementParticleProposal.coordinate_names)
                or payload['metadata'].get('fit_rule') != FIT_RULE):
            raise ValueError('incompatible frozen initial proposal schema, coordinates, or fit rule')
        proposals = []
        for p in payload['proposals']:
            if (p.get('component_family') != 'student_t' or p.get('student_df') != 5.
                    or len(p['means']) != 6 or p.get('retry_or_clip_draws') is not False):
                raise ValueError('artifact must retain six unclipped Student-t(df=5) components')
            q = MeasurementParticleProposal(p['means'], p['scales'], p['weights'],
                metadata=p, component_family='student_t', student_df=5.,
                phase_centres=p.get('phase_centres'), phase_concentrations=p.get('phase_concentrations'))
            if (not np.isclose(q.weights[:2].sum(), .2, atol=1e-14, rtol=0)
                    or not np.allclose(q.weights[2:], .2, atol=1e-14, rtol=0)
                    or np.any(q.phase_concentrations[:2] != 0)):
                raise ValueError('artifact must retain 20% broad uniform-phase mass and four equal pilots')
            proposals.append(q)
        return cls(proposals, payload['metadata'])


def phase_moment_parameters(phase):
    """Circular mean and capped kappa after fixed circular-SD inflation."""
    circular = np.exp(1j*np.asarray(phase)).mean()
    resultant = min(1., max(0., float(abs(circular))))
    if resultant <= 64*np.finfo(float).eps:
        return 0., 0., resultant
    centre = float(np.angle(circular) % (2*np.pi))
    desired = resultant**FIT_RULE['phase_resultant_power']
    cap = FIT_RULE['phase_kappa_cap']
    kappa = (cap if desired >= i1e(cap)/i0e(cap) else
             brentq(lambda k: i1e(k)/i0e(k)-desired, 0., cap, xtol=1e-12))
    return centre, float(kappa), resultant


def fit_initial_proposal(runs, broad_proposals, metadata):
    """Fit four separate moment components; no pooling across pilot modes.

    Only coordinates enter the fit. Scores, normalizers, truths, and later
    populations do not. Correlated particles supply empirical moments rather
    than independent-sample uncertainty estimates.
    """
    if len(runs) != FIT_RULE['pilots']:
        raise ValueError('the fixed rule requires exactly four pilot runs')
    objects = len(broad_proposals)
    _validate_metadata(metadata, objects)
    arrays = [np.asarray(run, float) for run in runs]
    if any(run.ndim != 3 or run.shape[0] != objects or run.shape[1] < 2
           or run.shape[2] != 10 or not np.isfinite(run).all() for run in arrays):
        raise ValueError('finite pilot arrays must have shape (objects, >=2 particles, 10)')
    proposals, phase_resultants = [], []
    for obj, broad in enumerate(broad_proposals):
        if (not isinstance(broad, MeasurementParticleProposal) or len(broad.means) != 2
                or not broad._uniform_phase):
            raise ValueError('exactly two original broad uniform-phase components required')
        broad = MeasurementParticleProposal(broad.means, broad.scales, broad.weights,
            component_family='student_t', student_df=5., metadata=dict(broad.metadata,
                construction='original measured means/scales with explicit Student-t(df=5) law'))
        means = np.vstack([broad.means, *[run[obj, :, :9].mean(0) for run in arrays]])
        empirical_sd_floor = FIT_RULE['broad_sd_floor_fraction']*broad.scales.max(0)
        scales = np.vstack([broad.scales, *[
            np.sqrt(3/5)*np.maximum(FIT_RULE['sd_inflation']*np.sqrt(run[obj, :, :9].var(0, ddof=0)),
                                   empirical_sd_floor) for run in arrays]])
        phase_fit = [phase_moment_parameters(run[obj, :, 9]) for run in arrays]
        phase_resultants.append([fit[2] for fit in phase_fit])
        proposals.append(MeasurementParticleProposal(means, scales,
            np.r_[FIT_RULE['broad_mass']*broad.weights, np.full(4, .2)],
            component_family='student_t', student_df=5.,
            phase_centres=[0., 0., *[fit[0] for fit in phase_fit]],
            phase_concentrations=[0., 0., *[fit[1] for fit in phase_fit]],
            metadata=dict(construction='frozen four-pilot moment mixture with broad measurement components',
                uses_generating_truth=False, observation=metadata['observations'][obj],
                broad_measurement_proposal=broad.metadata,
                component_roles=['broad_measured', 'broad_faint', 'pilot0', 'pilot1', 'pilot2', 'pilot3'])))
    details = deepcopy(metadata)
    details.update(fit_rule=deepcopy(FIT_RULE), pilot_particle_counts=[a.shape[1] for a in arrays],
                   phase_resultants=phase_resultants, uses_scores_or_normalizers=False,
                   wrapper_identity_policy='Original hashes retained; mutation-only wrapper differences allowed; native target sources checked.')
    return FrozenParticleProposal(proposals, details)


def fit_initial_proposal_from_runs(paths):
    """Verify immutable completed receipts and return a frozen four-pilot law."""
    paths = [Path(path).resolve() for path in paths]
    if len(paths) != 4 or len(set(paths)) != 4:
        raise ValueError('exactly four distinct pilot run directories required')
    protocols, arrays, sources, identities, fingerprints = [], [], [], [], []
    for path in paths:
        protocol = json.loads((path/'protocol.json').read_text())
        result = json.loads((path/'result.json').read_text())
        protocol_hash, particle_hash = digest(path/'protocol.json'), digest(path/'particles.npz')
        if (result.get('status') != 'completed'
                or result.get('protocol_sha256') != protocol_hash
                or result.get('particles_sha256') != particle_hash
                or result.get('observations') != protocol['observations']
                or result.get('particles') != protocol['particles']
                or result.get('final_stage', {}).get('stage') != protocol['stages']
                or result.get('final_stage', {}).get('beta') != 1.
                or np.asarray(result.get('log_normalizer'), float).shape != (len(protocol['observations']),)
                or not np.isfinite(np.asarray(result.get('log_normalizer'), float)).all()
                or any(result.get('extinct', []))):
            raise ValueError(f'incomplete, extinct, or inconsistent pilot receipt: {path}')
        identity, fingerprint = target_identity(protocol), source_fingerprint(protocol)
        if protocols:
            keys = ('particles', 'stages', 'moves', 'betas', 'scale', 'blocks', 'algorithm',
                    'proposals', 'physical_geometry', 'initial_proposal', 'extinction_schema')
            if (identity != identities[0] or fingerprint != fingerprints[0]
                    or any(protocol.get(k) != protocols[0].get(k) for k in keys)):
                raise ValueError('pilot protocol configurations or recorded implementations differ')
        with np.load(path/'particles.npz', allow_pickle=False) as saved:
            coordinates = saved['coordinates']
            if 'extinct' in saved and np.any(saved['extinct']):
                raise ValueError('extinct particles cannot supply fitted pilot moments')
        if coordinates.shape != (len(protocol['observations']), protocol['particles'], 10):
            raise ValueError('pilot particle dimensions differ from protocol')
        protocols.append(protocol); arrays.append(coordinates)
        identities.append(identity); fingerprints.append(fingerprint)
        sources.append(dict(path=str(path), seed=protocol['seed'], protocol_sha256=protocol_hash,
                            particles_sha256=particle_hash, result_sha256=digest(path/'result.json'),
                            implementation_sha256=fingerprint))
    broad = []
    for p in protocols[0]['proposals']:
        if p.get('coordinate_names') != list(MeasurementParticleProposal.coordinate_names):
            raise ValueError('pilot coordinate conventions differ')
        # Explicit family override is essential: dev_v1 pilots used Gaussians.
        broad.append(MeasurementParticleProposal(p['means'], p['scales'], p['weights'],
            component_family='student_t', student_df=5., metadata=dict(
                construction='original measured means/scales with explicit Student-t(df=5) law',
                measurement=p['measurement'], shear=p['shear'], uses_generating_truth=False)))
    metadata = dict(identities[0], pilot_seeds=[p['seed'] for p in protocols], pilot_runs=sources,
                    production_contract='Freeze before production; reject all pilot seeds and changed target identity.')
    return fit_initial_proposal(arrays, broad, metadata)
