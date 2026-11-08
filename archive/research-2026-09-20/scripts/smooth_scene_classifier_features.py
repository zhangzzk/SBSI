"""Four fixed smooth radial flux summaries with the audited state coordinates."""
import numpy as np

from sbsi.fixed_g0_domain import FLOW_FEATURES
from scripts.audit_continuous_crowding_residuals import SCALES, smooth_crowding
from scripts.scene_classifier_features import GEOMETRY_FEATURES, geometry_features
from scripts.scene_geometry_coordinates import transport_major_distance

SMOOTH_FEATURES = ('nbr_gaussian_050', 'nbr_gaussian_100', 'nbr_gaussian_200', 'nbr_gaussian_400')
FEATURES = (*FLOW_FEATURES, *GEOMETRY_FEATURES, *SMOOTH_FEATURES)
FORMAT = 'scene_classifier_state_major_smooth_v1'


def augment_state(ids, context, state, pairs, psf_radius):
    if state.shape != (len(ids), 13):
        raise ValueError('aligned thirteen-input state required')
    np.testing.assert_array_equal(state[:, 2:8], context[:, 2:8])
    moved = transport_major_distance(ids, context, state[:, :2], pairs, psf_radius)
    geometry, _ = geometry_features(ids, moved)
    np.testing.assert_array_equal(state[:, 8:13], geometry)
    return np.column_stack((state, smooth_crowding(ids, moved))).astype(np.float32)


def augment_prepared(data, pairs, psf_radius):
    result = {key: np.array(value, copy=True) for key, value in data.items()}
    np.testing.assert_array_equal(data['x0'][:, :8], data['base_context'])
    for leg in (0, 1):
        result[f'x{leg}'] = augment_state(data['ids'], data['base_context'], data[f'x{leg}'], pairs, psf_radius)
    return result


assert len(SCALES) == len(SMOOTH_FEATURES)
