import numpy as np
import pandas as pd
import pytest

from scripts.prepare_transported_geometry import transport_prepared
from scripts.scene_classifier_features import geometry_features
from scripts.scene_geometry_coordinates import major_convolved_scale
from scripts.smooth_scene_classifier_features import SCALES, augment_prepared


def test_smooth_training_features_transport_and_preserve_labels():
    ids = np.array([9])
    context = np.array([[.6, 0, 1, 25, .4, 0, 0, 0]], dtype=np.float32)
    scale = major_convolved_scale(context[:, 4], context[:, :2], .5)
    pairs = pd.DataFrame({'index_input_p': ids, 'index_input_s': [2], 'r_input_p_scaled': [25.],
                         'r_input_s_scaled': [25.], 'distance_scaled': [.49],
                         'Re_input_p_scaled': np.sqrt(1-(.5/scale)**2)})
    geometry, _ = geometry_features(ids, pairs)
    x0 = np.column_stack((context, geometry))
    x1 = x0.copy()
    x1[:, :2] = 0
    raw = {'ids': ids, 'base_context': context, 'geometry': geometry, 'x0': x0, 'x1': x1,
           'u0': np.array([True]), 'u1': np.array([False])}
    data, _ = transport_prepared(raw, pairs, .5)
    output = augment_prepared(data, pairs, .5)
    for leg in (0, 1):
        np.testing.assert_array_equal(output[f'x{leg}'][:, :13], data[f'x{leg}'])
        np.testing.assert_array_equal(output[f'u{leg}'], data[f'u{leg}'])
        target = major_convolved_scale(context[:, 4], data[f'x{leg}'][:, :2], .5)
        expected = np.log1p(np.exp(-.5*(.49*scale/target/np.array(SCALES))**2))
        np.testing.assert_allclose(output[f'x{leg}'][0, 13:], expected, rtol=1e-6)
    assert np.all(output['x1'][:, 13:] < output['x0'][:, 13:])
    assert data['x0'].shape[1] == 13
    with pytest.raises(AssertionError):
        augment_prepared(raw, pairs, .5)  # Reject stale static coordinates in sheared leg.
