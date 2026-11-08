"""Experimental periodic-phase control variate with independent frozen fitting.

The identity concerns the intended continuous posterior at beta=1. It does
not certify the float32 sampler target, posterior equilibration, finite-SMC
unbiasedness, or variance reduction for correlated production populations.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from .particle_diagnostics import combine_populations


SCHEMA = 'sbsi-frozen-phase-control-v1'
FIT_RULE = ('equal-run pooled empirical covariance(S,T)/variance(T); '
            'zero coefficient if empirical control variance is exactly zero; no intercept correction')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_digest(array):
    array = np.ascontiguousarray(array, dtype='<f8')
    result = hashlib.sha256(json.dumps(list(array.shape)).encode())
    result.update(array.tobytes())
    return result.hexdigest()


def phase_stein_control(values, context_gradient, *, beta=1.):
    """Compute d log K / d psi from unsheared intrinsic ellipticity context.

    Other features and response b must be invariant to target phase. This
    holds for the retained GalSBI summary/response mapping. A nonuniform q0
    invalidates using T alone at intermediate annealing temperatures.
    """
    values, gradient = np.asarray(values, float), np.asarray(context_gradient, float)
    if not np.isscalar(beta) or not np.isfinite(beta) or beta != 1.:
        raise ValueError('this control is restricted to the beta=1 posterior endpoint')
    if (values.ndim != 2 or values.shape[1] != 9 or gradient.shape != (len(values), 8)
            or not np.isfinite(values).all() or not np.isfinite(gradient).all()
            or np.any(np.hypot(values[:, 0], values[:, 1]) >= 1)):
        raise ValueError('finite aligned intrinsic nine-column summaries and eight-column gradients required')
    result = -values[:, 1]*gradient[:, 0]+values[:, 0]*gradient[:, 1]
    if not np.isfinite(result).all():
        raise FloatingPointError('phase control is outside finite numerical support')
    return result


def rotate_summary_phase(values, angle):
    """Rotate only intrinsic ellipticity, preserving all phase-invariant fields."""
    values = np.asarray(values, float)
    if (values.ndim != 2 or values.shape[1] != 9 or not np.isfinite(values).all()
            or not np.isscalar(angle) or not np.isfinite(angle)):
        raise ValueError('finite nine-column summaries and scalar rotation required')
    result = values.copy()
    cosine, sine = np.cos(angle), np.sin(angle)
    result[:, 0] = cosine*values[:, 0]-sine*values[:, 1]
    result[:, 1] = sine*values[:, 0]+cosine*values[:, 1]
    return result


def _metadata_check(metadata, objects):
    observations, seeds = metadata.get('observations'), metadata.get('pilot_seeds')
    if (not isinstance(observations, list) or len(observations) != objects
            or any(type(v) is not int or v < 0 for v in observations)
            or len(set(observations)) != objects):
        raise ValueError('unique ordered nonnegative observations required')
    if (not isinstance(seeds, list) or len(seeds) < 2
            or any(type(v) is not int or v < 0 for v in seeds) or len(set(seeds)) != len(seeds)):
        raise ValueError('at least two distinct nonnegative independent pilot seeds required')
    identity = metadata.get('target_identity')
    if not isinstance(identity, dict) or not identity:
        raise ValueError('explicit target/derivative identity required')
    # Preserve a JSON value, not arbitrary mutable objects or NaN sentinels.
    json.dumps(metadata, allow_nan=False)


def _aligned(scores, control, objects=None, dimensions=None):
    scores, control = np.asarray(scores, float), np.asarray(control, float)
    if (scores.ndim != 3 or not scores.shape[0] or not scores.shape[1] or not scores.shape[2]
            or control.shape != scores.shape[:2] or not np.isfinite(scores).all()
            or not np.isfinite(control).all()
            or (objects is not None and scores.shape[0] != objects)
            or (dimensions is not None and scores.shape[2] != dimensions)):
        raise ValueError('finite aligned scores (objects,particles,dimensions) and controls required')
    return scores, control


class FrozenPhaseControl:
    """Per-observation fixed score coefficients with mandatory seed validation."""

    def __init__(self, coefficients, metadata):
        coefficients = np.array(coefficients, float, copy=True)
        if coefficients.ndim != 2 or 0 in coefficients.shape or not np.isfinite(coefficients).all():
            raise ValueError('finite nonempty coefficient matrix required')
        _metadata_check(metadata, len(coefficients))
        self.coefficients = coefficients
        self.coefficients.flags.writeable = False
        self._metadata = deepcopy(metadata)

    @property
    def metadata(self):
        return deepcopy(self._metadata)

    def validate_production(self, seed, observations, target_identity):
        if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
            raise ValueError('nonnegative integer production seed required')
        if int(seed) in self._metadata['pilot_seeds']:
            raise ValueError('production seed overlaps frozen-coefficient pilot seeds')
        if list(observations) != self._metadata['observations']:
            raise ValueError('production observations or ordering differ from frozen coefficients')
        if target_identity != self._metadata['target_identity']:
            raise ValueError('production target/derivative identity differs from frozen coefficients')

    def apply(self, scores, control, *, seed, observations, target_identity):
        """Return both raw and corrected scores; never recenter production T."""
        self.validate_production(seed, observations, target_identity)
        scores, control = _aligned(scores, control, *self.coefficients.shape)
        corrected = scores-self.coefficients[:, None, :]*control[:, :, None]
        if not np.isfinite(corrected).all():
            raise FloatingPointError('corrected scores are outside finite numerical support')
        return dict(raw_scores=scores.copy(), phase_control=control.copy(), corrected_scores=corrected)

    def save(self, path):
        with Path(path).open('x') as handle:
            json.dump(dict(schema=SCHEMA, coefficients=self.coefficients.tolist(), metadata=self.metadata),
                      handle, indent=2, allow_nan=False)
            handle.write('\n')

    @classmethod
    def load(cls, path):
        payload = json.loads(Path(path).read_text())
        if payload.get('schema') != SCHEMA or payload['metadata'].get('fit_rule') != FIT_RULE:
            raise ValueError('incompatible frozen phase-control schema or fitting rule')
        return cls(payload['coefficients'], payload['metadata'])


def fit_frozen_phase_control(score_runs, control_runs, metadata):
    """Fit a fixed empirical regression heuristic from independent pilots.

    Each run has equal total weight and each of its particles equal within-run
    weight. Correlation does not make this covariance an uncertainty estimate.
    The coefficient may be suboptimal, but is fixed before production. Raw T
    is subtracted in application: pilot or production means are never offsets.
    """
    if len(score_runs) < 2 or len(score_runs) != len(control_runs):
        raise ValueError('at least two aligned independent pilot runs required')
    pairs = [_aligned(s, t) for s, t in zip(score_runs, control_runs)]
    objects, _, dimensions = pairs[0][0].shape
    _metadata_check(metadata, objects)
    if len(metadata['pilot_seeds']) != len(pairs):
        raise ValueError('one independent seed per pilot run required')
    if any(s.shape[0] != objects or s.shape[2] != dimensions or s.shape[1] < 2 for s, _ in pairs):
        raise ValueError('same objects/dimensions and at least two particles per pilot required')
    score_mean = np.mean([s.mean(1) for s, _ in pairs], axis=0)
    control_mean = np.mean([t.mean(1) for _, t in pairs], axis=0)
    variance = np.mean([np.square(t-control_mean[:, None]).mean(1) for _, t in pairs], axis=0)
    covariance = np.mean([((s-score_mean[:, None, :])*(t-control_mean[:, None])[:, :, None]).mean(1)
                          for s, t in pairs], axis=0)
    coefficients = np.divide(covariance, variance[:, None], out=np.zeros_like(covariance),
                             where=variance[:, None] > 0)
    if not np.isfinite(variance).all() or not np.isfinite(coefficients).all():
        raise FloatingPointError('pilot covariance or coefficients are outside finite support')
    details = deepcopy(metadata)
    details.update(fit_rule=FIT_RULE, pilot_particles=[s.shape[1] for s, _ in pairs],
        pilot_array_sha256=[dict(scores=array_digest(s), control=array_digest(t)) for s, t in pairs],
        implementation_sha256=digest(__file__), pilot_control_mean=control_mean.tolist(),
        pilot_control_variance=variance.tolist(), zero_variance=(variance == 0).tolist(),
        uses_normalizer_weights=False, production_control_centering=False,
        scientific_status='Frozen estimator prototype; no sampler or posterior-equilibration certification')
    return FrozenPhaseControl(coefficients, details)


def summarize_phase_populations(log_normalizers, controls, raw_scores, deltas, coefficients=None):
    """Independent-population diagnostics under raw and explicit F64 measures.

    Every input list entry is a complete independent population; particle
    counts may differ. Precision reweighting changes the diagnostic measure
    only: Z64hat=Z32hat*mean(exp(delta)). It never updates production files.
    """
    logz = np.asarray(log_normalizers, float)
    if (logz.ndim != 2 or len(logz) < 2 or not np.isfinite(logz).all()
            or len(controls) != len(logz) or len(raw_scores) != len(logz) or len(deltas) != len(logz)):
        raise ValueError('aligned complete finite independent populations required; no dropping failures')
    moments32, moments64, adjusted_logz = [], [], []
    for i, (control, scores, delta) in enumerate(zip(controls, raw_scores, deltas)):
        scores, control = _aligned(scores, control, objects=logz.shape[1])
        delta = np.asarray(delta, float)
        if delta.shape != control.shape or not np.isfinite(delta).all():
            raise ValueError('finite pointwise F64-minus-saved-F32 log-kernel differences required')
        values = np.concatenate((control[:, :, None], scores), axis=2)
        if coefficients is not None:
            c = np.asarray(coefficients, float)
            if c.shape != (logz.shape[1], scores.shape[2]) or not np.isfinite(c).all():
                raise ValueError('aligned finite frozen coefficients required')
            values = np.concatenate((values, scores-c[:, None, :]*control[:, :, None]), axis=2)
        logsum = logsumexp(delta, axis=1)
        normalized = np.exp(delta-logsum[:, None])
        moments32.append(values.mean(1))
        moments64.append(np.sum(values*normalized[:, :, None], axis=1))
        adjusted_logz.append(logz[i]+logsum-np.log(delta.shape[1]))
    moments32, moments64, adjusted_logz = map(np.asarray, (moments32, moments64, adjusted_logz))
    results = []
    for obj in range(logz.shape[1]):
        row = {}
        for name, z, moments in [('raw_float32_sampled', logz, moments32),
                                 ('precision_reweighted_float64', adjusted_logz, moments64)]:
            summary = combine_populations(z[:, obj], moments[:, obj])
            mean, se = summary['mean'][0], summary['mcse'][0]
            continuous = name == 'precision_reweighted_float64'
            row[name] = dict(summary, control_known_expectation=0. if continuous else None,
                control_expectation_status=('intended_continuous_float64_periodic_identity' if continuous
                    else 'unestablished_for_rounded_float32_target'),
                control_reference_value=0.,
                control_standardized_residual=float(mean/se) if se > 0 else None,
                control_nonzero_with_zero_estimated_mcse=bool(se == 0 and mean != 0))
        results.append(row)
    return results


def load_phase_diagnostic(path):
    """Load an unchanged successfully replayed diagnostic, never a failure."""
    path = Path(path)
    report = json.loads(path.read_text())
    arrays_path = path.with_suffix('.npz')
    if (report.get('schema') != 'sbsi-phase-diagnostic-v1'
            or report.get('numerical_checks_pass') is not True
            or report.get('arrays_sha256') != digest(arrays_path)):
        raise ValueError('unchanged successfully replayed phase diagnostic required')
    if report.get('stride') != 1:
        raise ValueError('phase estimator requires the complete population with stride=1; deterministic thinning is unsupported')
    sources = report.get('source_hashes')
    if not isinstance(sources, dict) or not sources or any(digest(p) != h for p, h in sources.items()):
        raise ValueError('phase diagnostic original input receipts changed or missing')
    with np.load(arrays_path, allow_pickle=False) as stored:
        arrays = {name: stored[name].copy() for name in ('raw_scores', 'phase_control', 'delta_log_kernel')}
    _aligned(arrays['raw_scores'], arrays['phase_control'], objects=len(report['observations']))
    if report.get('retained_particles') != arrays['phase_control'].shape[1]:
        raise ValueError('phase diagnostic retained particle count differs from complete arrays')
    if (arrays['delta_log_kernel'].shape != arrays['phase_control'].shape
            or not np.isfinite(arrays['delta_log_kernel']).all()):
        raise ValueError('finite aligned diagnostic precision differences required')
    return report, arrays
