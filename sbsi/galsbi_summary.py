"""Experimental fixed-summary boundary for the retained V3.9 likelihood.

Compression is exact for a fixed observed radius/flux: eight F8/C8 context
values and one pair-summed disk coefficient. It supplies prior *draws*, not
an evaluable nine-dimensional density. No model is retrained or smoothed.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .catalogue_disk_likelihood import DiskCatalogueLikelihood
from .forward_catalogue import rescale_emulator_pairs
from .output_conditioned_response import TRUTH_FEATURES
from .v39_inference import FEATURES, V39CatalogueCache


def native_shapes(data, angular_diameter_distance):
    """Convert GalSBI/UFig distortion and native radius to SBSI geometry.

    UFig rendering_util.compute_ellip_matrices and draw_photon_gal give
    semiaxes r50*sqrt(1+a), r50*sqrt(1-a), not an area-preserving stretch.
    Thus circularized radius is r50*(1-a*a)**(1/4). Distortion a maps to
    epsilon=a/(1+sqrt(1-a*a)). This explicitly adopts the UFig-rendered
    ellipse convention (zero flexion), not r50 == circularized radius.
    """
    a, phase = np.asarray(data['a'], float), np.asarray(data['psi'], float)
    distance = np.asarray(angular_diameter_distance(data['z']), float)
    radius = np.asarray(data['R_kpc'], float)/(1e3*distance)*(180/np.pi*3600)
    if (a.shape != phase.shape or a.shape != radius.shape
            or not np.isfinite(np.column_stack((a, phase, radius))).all()
            or np.any((a < 0) | (a >= 1)) or np.any(radius <= 0)):
        raise ValueError('finite physical radius, phase and distortion in [0,1) required')
    # This threshold is in the installed UFig matrix routine.
    rendered_a = np.where(a > 1e-7, a, 0.)
    eps = rendered_a/(1+np.sqrt(1-rendered_a**2))
    return dict(e1=eps*np.cos(phase), e2=eps*np.sin(phase),
                semimajor=radius*np.sqrt(1+rendered_a),
                circularized=radius*(1-rendered_a**2)**.25)


def _check_response_contract(prior, response_radius_arcsec, response_neighbours):
    if (not np.isfinite(response_radius_arcsec) or response_radius_arcsec <= 0
            or response_radius_arcsec > prior.aperture_arcsec
            or not isinstance(response_neighbours, (int, np.integer)) or response_neighbours < 1):
        raise ValueError('covered response aperture and positive integer neighbour count required')


def scene_model_inputs(scenes, prior, angular_diameter_distance, conditions, *,
                       response_radius_arcsec, response_neighbours, response=True):
    """Map native scenes to F8/C8 features and the retained response pair CSR.

    All neighbours within 10 arcsec enter crowding. Only the response uses
    its explicitly supplied nearest-k/aperture contract. No truth cuts are
    applied. Input photometry must belong to the current native marks.
    response=False returns (features, None, None) without building pairs.
    """
    _check_response_contract(prior, response_radius_arcsec, response_neighbours)
    target = native_shapes(scenes.targets, angular_diameter_distance)
    zero = pd.DataFrame(np.column_stack((target['e1'], target['e2'], scenes.targets['n'],
        scenes.targets['m_r'], target['circularized'], prior.v39_crowding(scenes))), columns=FEATURES)
    if not np.isfinite(zero.to_numpy()).all():
        raise ValueError('nonfinite mapped model inputs')
    if not response:
        return zero, None, None
    return (zero, *response_pairs(scenes, prior, angular_diameter_distance, conditions,
        response_radius_arcsec=response_radius_arcsec, response_neighbours=response_neighbours, target=target))


def response_pairs(scenes, prior, angular_diameter_distance, conditions, *,
                   response_radius_arcsec, response_neighbours, target=None):
    """The response pair CSR (indptr, features) of scene_model_inputs alone."""
    _check_response_contract(prior, response_radius_arcsec, response_neighbours)
    if target is None:
        target = native_shapes(scenes.targets, angular_diameter_distance)
    neighbour = native_shapes(scenes.neighbours, angular_diameter_distance)
    owner, distance = scenes.owner, np.linalg.norm(scenes.offset_arcsec, axis=1)
    rows = np.flatnonzero(distance < response_radius_arcsec)
    rows = rows[np.lexsort((rows, distance[rows], owner[rows]))]
    _, starts, counts = np.unique(owner[rows], return_index=True, return_counts=True)
    ranks = np.arange(len(rows))-np.repeat(starts, counts)
    rows = rows[ranks < response_neighbours]
    primary = owner[rows]
    pairs = pd.DataFrame(dict(Re_input_p=target['semimajor'][primary],
        Re_input_s=neighbour['semimajor'][rows], r_input_p=scenes.targets['m_r'][primary],
        r_input_s=scenes.neighbours['m_r'][rows], sersic_n_input_p=scenes.targets['n'][primary],
        sersic_n_input_s=scenes.neighbours['n'][rows], distance=distance[rows]))
    features = rescale_emulator_pairs(pairs, conditions).loc[:, TRUTH_FEATURES].to_numpy(float)
    indptr = np.r_[0, np.cumsum(np.bincount(primary, minlength=len(scenes.counts)))]
    if not np.isfinite(features).all():
        raise ValueError('nonfinite mapped model inputs')
    return indptr, features


@dataclass
class SummaryBatch:
    """Nine values per scene, with the observed auxiliaries that define b."""
    values: np.ndarray
    auxiliary: np.ndarray

    def __post_init__(self):
        self.values = np.asarray(self.values, float)
        self.auxiliary = np.asarray(self.auxiliary, float)
        if (self.values.ndim != 2 or self.values.shape[1] != 9 or not len(self.values)
                or self.auxiliary.shape != (len(self.values), 2)
                or not np.isfinite(self.values).all() or not np.isfinite(self.auxiliary).all()
                or np.any(self.auxiliary <= 0)):
            raise ValueError('finite nine-column summaries and positive aligned auxiliaries required')


def compress_scenes(zero, response, auxiliary):
    """Discard the pair list after computing b at the supplied observation(s)."""
    if tuple(zero.columns) != FEATURES or response.n_atoms != len(zero):
        raise ValueError('aligned F8/C8 features and response scenes required')
    aux = np.broadcast_to(np.asarray(auxiliary, float), (len(zero), 2)).copy()
    b, _ = response.predict(np.arange(len(zero)), aux[:, 0], aux[:, 1])
    return SummaryBatch(np.column_stack((zero.to_numpy(), b)), aux)


class _SummedResponse:
    def __init__(self, summary):
        self.summary, self.n_atoms = summary, len(summary.values)

    def predict(self, atoms, radius, flux):
        if not np.array_equal(np.column_stack((radius, flux)), self.summary.auxiliary[atoms]):
            raise ValueError('response summary belongs to different observed radius/flux')
        return self.summary.values[atoms, 8], {}


class SummaryLikelihood:
    """Thin adapter to the unchanged catalogue density and shear conventions.

    Temporary row indices only address the supplied fresh summaries; they do
    not restrict the prior to an existing catalogue. Fixed padded density
    batches avoid the previously demonstrated variable-batch numerical drift.
    Returns log(C8 * disk density) in the existing standardized-output units.
    No empirical prior weight or measured-selection normalizer is included.
    """
    def __init__(self, flow, detector, *, density_batch_size=4096):
        if (tuple(flow.condition_preprocessor.feature_names) != FEATURES
                or tuple(detector.preprocessor.feature_names) != FEATURES
                or not isinstance(density_batch_size, int) or density_batch_size < 1):
            raise ValueError('retained F8/C8 feature contract and positive density batch required')
        self.flow, self.detector = flow, detector
        self.batch_size = density_batch_size

    def log_kernel(self, summary, observed, g1, g2):
        observed = observed.reset_index(drop=True)
        n = len(summary.values)
        if len(observed) == 1:
            observed = observed.iloc[np.zeros(n, dtype=int)].reset_index(drop=True)
        if len(observed) != n:
            raise ValueError('one observation or one per summary required')
        names = self.flow.target_transform.target_names
        if not np.array_equal(observed[list(names[2:])].to_numpy(), summary.auxiliary):
            raise ValueError('response summary belongs to different observed radius/flux')
        result = np.empty(n)
        for start in range(0, n, self.batch_size):
            stop = min(start+self.batch_size, n)
            rows = np.minimum(np.arange(start, start+self.batch_size), stop-1)
            block = SummaryBatch(summary.values[rows], summary.auxiliary[rows])
            zero = pd.DataFrame(block.values[:, :8], columns=FEATURES)
            cache = V39CatalogueCache(zero, np.ones(len(zero)), {}, detector=self.detector)
            backend = DiskCatalogueLikelihood(self.flow, cache, disk_response=_SummedResponse(block))
            values = backend.conditional_log_likelihood(observed.iloc[rows], g1, g2,
                atom_indices=np.arange(len(rows)), object_chunk=self.batch_size)
            probability = cache.get(g1, g2).detection_probability
            if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
                raise ValueError('invalid usability probability')
            with np.errstate(divide='ignore'):
                result[start:stop] = (values+np.log(probability))[:stop-start]
        if np.isnan(result).any() or np.isposinf(result).any():
            raise FloatingPointError('nonfinite summary likelihood evaluation')
        return result
