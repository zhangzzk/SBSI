import numpy as np

from scripts.build_scene_response_target_v27 import match_snc


def test_match_snc_marks_absent_measurement_nan():
    skey = np.array([10, 20, 40], dtype=np.int64)
    s1 = np.array([1.0, 2.0, 4.0])
    s2 = -s1
    key = np.array([10, 30, 40], dtype=np.int64)
    match, g1, g2 = match_snc(key, skey, s1, s2)
    np.testing.assert_array_equal(match, [True, False, True])
    np.testing.assert_allclose(g1[match], [1.0, 4.0])
    np.testing.assert_allclose(g2[match], [-1.0, -4.0])
    assert np.isnan(g1[1]) and np.isnan(g2[1])
