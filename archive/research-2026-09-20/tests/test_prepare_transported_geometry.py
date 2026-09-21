import numpy as np
import pandas as pd

from scripts.prepare_transported_geometry import transport_prepared
from scripts.scene_classifier_features import geometry_features
from scripts.scene_geometry_coordinates import major_convolved_scale


def test_training_transform_preserves_labels_and_zero_leg_without_mutating_source():
    ids = np.array([9])
    context = np.array([[.6, 0., 1., 25., .4, 0., 0., 0.]], dtype=np.float32)
    scale = major_convolved_scale(context[:, 4], context[:, :2], .5)
    pairs = pd.DataFrame({'index_input_p': ids, 'index_input_s': [2], 'r_input_p_scaled': [25.],
                          'r_input_s_scaled': [25.], 'distance_scaled': [.49],
                          'Re_input_p_scaled': np.sqrt(1-(.5/scale)**2)})
    geometry, _ = geometry_features(ids, pairs)
    x0 = np.column_stack((context, geometry))
    x1 = x0.copy()
    x1[:, :2] = 0.
    data = {'ids': ids, 'base_context': context, 'geometry': geometry, 'x0': x0, 'x1': x1,
            'u0': np.array([True]), 'u1': np.array([False]), 'd0': np.array([True]), 'd1': np.array([False])}
    original = {k: v.copy() for k, v in data.items()}
    result, changes = transport_prepared(data, pairs, .5)
    assert changes == {'g0': 0, 'g1': 1}
    np.testing.assert_array_equal(result['x0'], original['x0'])
    np.testing.assert_array_equal(result['x1'][:, :8], original['x1'][:, :8])
    np.testing.assert_allclose(np.expm1(result['x1'][:, 8:]), [[0., 1., 0., 0., 0.]])
    for key, expected in original.items():
        np.testing.assert_array_equal(data[key], expected)
        if key not in ('x1',):
            np.testing.assert_array_equal(result[key], expected)
