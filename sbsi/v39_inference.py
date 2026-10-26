"""Explicit physical-crowding and cache contract for V3.9-like inference."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .catalogue_likelihood import CatalogueModelCache, CatalogueModelView, predict_detection_probability
from .shear_map import apply_shear_to_ellipticity

FEATURES = ('e1_input_p', 'e2_input_p', 'sersic_n_input_p', 'r_input_p',
            'circularized_Re_input_p', 'nbr_close_050', 'nbr_close_100', 'nbr_close_200')


def physical_close_crowding(scene, primary_rows=None, *, row_chunk=8192, workers=1):
    """Sum all nonself neighbours within 10 arcsec in one independent scene.

    Matches the training projection: RA*cos(mean DEC), DEC, in arcseconds.
    Restricting primary_rows never removes potential secondary neighbours.
    Return float64 sums after log1p; cast to float32 for the model contract.
    """
    values = scene[['RA', 'DEC', 'r']].to_numpy(float)
    if not len(values) or not np.isfinite(values).all() or row_chunk < 1:
        raise ValueError('nonempty finite scene and positive chunk required')
    if 'case' in scene and scene['case'].nunique() != 1:
        raise ValueError('one independent scene required; do not mix overlapping cases')
    rows = np.arange(len(scene)) if primary_rows is None else np.asarray(primary_rows)
    if rows.ndim != 1 or not np.issubdtype(rows.dtype, np.integer) or np.any((rows < 0) | (rows >= len(scene))):
        raise ValueError('valid integer primary rows required')
    xy = values[:, :2].copy()
    xy[:, 0] *= np.cos(np.deg2rad(values[:, 1].mean()))
    xy *= 3600.
    tree = cKDTree(xy)
    result = np.empty((len(rows), 3))
    for lo in range(0, len(rows), row_chunk):
        primary = rows[lo:lo+row_chunk]
        neighbours = tree.query_ball_point(xy[primary], 10., workers=workers, return_sorted=True)
        counts = np.array([len(x) for x in neighbours], dtype=np.int64)
        secondary = np.concatenate(neighbours).astype(np.int64)
        owner = np.repeat(np.arange(len(primary)), counts)
        nonself = secondary != primary[owner]
        secondary, owner = secondary[nonself], owner[nonself]
        distance2 = np.square(xy[primary[owner]]-xy[secondary]).sum(1)
        ratio = 10.**(-.4*(values[secondary, 2]-values[primary[owner], 2]))
        for column, scale in enumerate((.5, 1., 2.)):
            result[lo:lo+len(primary), column] = np.log1p(np.bincount(owner,
                weights=ratio*np.exp(-distance2/(2*scale**2)), minlength=len(primary)))
    if not np.isfinite(result).all():
        raise ValueError('nonfinite physical crowding')
    return result


class V39CatalogueCache(CatalogueModelCache):
    """Eight physical inputs for both F8 and C8, with optional frozen p(U)."""
    def __init__(self, zero, weights, conditions, *, detector=None, probabilities=None):
        if tuple(zero.columns) != FEATURES or not len(zero) or not np.isfinite(zero).all().all():
            raise ValueError('finite ordered V3.9 eight-input context required')
        weights = np.asarray(weights, float)
        if weights.shape != (len(zero),) or not np.isfinite(weights).all() or np.any(weights < 0) or weights.sum() <= 0:
            raise ValueError('finite nonnegative aligned prior masses required')
        if detector is not None and tuple(detector.preprocessor.feature_names) != FEATURES:
            raise ValueError('V3.9 C8 classifier required')
        zero = zero.copy(deep=False)
        zero.index = pd.RangeIndex(len(zero), name='primary_row')
        super().__init__(SimpleNamespace(galaxies=zero, weights=weights), detector=detector,
            conditions=conditions, detection_radius_arcsec=10., flow_neighbour_radius_arcsec=10.,
            crowding_radii_arcsec=(.5, 1., 2.), flow_features=FEATURES, detection_features=FEATURES)
        self.zero_flow = zero
        self.probabilities = None
        if probabilities is not None:
            self.probabilities = {}
            for point, probability in probabilities.items():
                key = self._key(*point)
                values = np.asarray(probability)
                if key in self.probabilities or values.shape != weights.shape or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
                    raise ValueError('unique nodes and aligned probabilities in [0,1] required')
                self.probabilities[key] = values
        elif detector is None:
            raise ValueError('classifier or frozen probabilities required')

    def get(self, g1, g2):
        key = self._key(g1, g2)
        if key not in self._views:
            flow = self.zero_flow.copy(deep=False)
            flow['e1_input_p'], flow['e2_input_p'] = apply_shear_to_ellipticity(
                self.zero_flow.e1_input_p.to_numpy(), self.zero_flow.e2_input_p.to_numpy(), *key)
            if self.probabilities is not None:
                if key not in self.probabilities:
                    raise ValueError(f'unprepared V3.9 shear node {key}')
                probability = self.probabilities[key]
            else:
                probability = predict_detection_probability(self.detector, flow)
            self._views[key] = CatalogueModelView(*key, flow, pd.DataFrame(index=flow.index),
                probability, np.broadcast_to(np.zeros(2), (len(flow), 2)))
        return self._views[key]

    def validate_model_features(self, flow_model):
        if tuple(flow_model.condition_preprocessor.feature_names) != FEATURES:
            raise ValueError('V3.9 F8 flow required')

    def save(self, *args, **kwargs):
        raise NotImplementedError('persist V3.9 feature and model identities explicitly')

    def with_prior(self, *args, **kwargs):
        raise NotImplementedError('rebuild V3.9 cache explicitly when changing support')
