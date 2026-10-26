import numpy as np
import pandas as pd
import pytest

from sbsi.crowding import major_convolved_scale, transport_major_distance


def fixture():
    ids = np.array([9, 3])
    context = np.zeros((2, 8))
    context[:, 0] = [.6, 0.]
    context[:, 4] = [.4, .3]
    scale = major_convolved_scale(context[:, 4], context[:, :2], .5)
    pairs = pd.DataFrame({'index_input_p': [3, 9], 'distance_scaled': [2., 3.],
                          'Re_input_p_scaled': np.sqrt(1-(.5/scale[::-1])**2)})
    return ids, context, pairs


def test_scale_matches_major_axis_radius_and_preserves_physical_separation():
    ids, context, pairs = fixture()
    np.testing.assert_allclose(major_convolved_scale([.4], [[.6, 0.]], .5), np.hypot(.8, .5))
    target_shape = context[:, :2]*.5
    changed = transport_major_distance(ids, context, target_shape, pairs, .5)
    original_scale = major_convolved_scale(context[:, 4], context[:, :2], .5)[::-1]
    target_scale = major_convolved_scale(context[:, 4], target_shape, .5)[::-1]
    np.testing.assert_allclose(changed['distance_scaled']*target_scale, pairs['distance_scaled']*original_scale)
    pd.testing.assert_frame_equal(transport_major_distance(ids, context, context[:, :2], pairs, .5), pairs)


def test_scale_rotation_invariant_and_wrong_cached_radius_rejected():
    ids, context, pairs = fixture()
    rotated = context[:, :2][:, ::-1]*[1, -1]
    np.testing.assert_array_equal(major_convolved_scale(context[:, 4], rotated, .5),
                                  major_convolved_scale(context[:, 4], context[:, :2], .5))
    pairs['Re_input_p_scaled'] = .1
    with pytest.raises(AssertionError):
        transport_major_distance(ids, context, rotated, pairs, .5)


def test_reject_nonphysical_ellipticity():
    with pytest.raises(ValueError, match='subunit'):
        major_convolved_scale([1.], [[1., 0.]], .5)
