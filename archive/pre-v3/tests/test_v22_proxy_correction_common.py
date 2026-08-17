import numpy as np

from scripts.v22_proxy_correction_common import (
    PSF_HALF_LIGHT_ARCSEC,
    proxy_conditional_matrix,
    scene_proxy_from_full_arrays,
)


def test_scene_proxy_recovers_arcsec_distance_and_scene_sum():
    re_p = np.array([0.4, 0.4, 0.8])
    distance = np.array([1.0, 2.0, 0.5])
    post = np.sqrt(re_p**2 + PSF_HALF_LIGHT_ARCSEC**2)
    x_scaled = np.zeros((3, 7), dtype=np.float32)
    x_scaled[:, 0] = re_p / post
    x_scaled[:, 6] = distance / post
    x_aux = np.zeros((3, 2), dtype=np.float32)
    x_aux[:, 0] = np.log10([0.5, 2.0, 0.25])
    pair_scene = np.array([0, 0, 1], dtype=np.int32)
    count = np.array([2, 1], dtype=np.int16)

    proxy, log_proxy = scene_proxy_from_full_arrays(
        x_scaled, x_aux, pair_scene, count
    )
    expected = np.array([0.5 / 1.0**2 + 2.0 / 2.0**2, 0.25 / 0.5**2])
    np.testing.assert_allclose(proxy, expected, rtol=2.0e-6)
    np.testing.assert_allclose(
        log_proxy, np.log10(expected), rtol=2.0e-6, atol=2.0e-7
    )


def test_proxy_conditional_matrix_replaces_only_scene_coordinate():
    scaled = np.arange(14, dtype=np.float32).reshape(2, 7) / 10.0
    raw = np.array([
        [0.4, 0.2, 24.0, 25.0, 1.0, 2.0, 1.0],
        [0.8, 0.4, 23.0, 22.0, 3.0, 4.0, 2.0],
    ], dtype=np.float32)
    pair = np.array([0.01, -0.02], dtype=np.float32)
    logq = np.array([-1.0, 0.5], dtype=np.float32)
    count = np.array([16, 8], dtype=np.int16)
    matrix = proxy_conditional_matrix(scaled, raw, pair, logq, count)

    assert matrix.shape == (2, 12)
    np.testing.assert_array_equal(matrix[:, :7], scaled)
    np.testing.assert_array_equal(matrix[:, 7], pair)
    np.testing.assert_array_equal(matrix[:, 10], logq)
    np.testing.assert_allclose(matrix[:, 11], np.log1p(count))
