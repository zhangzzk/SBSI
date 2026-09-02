import numpy as np

from sbsi.flow_coupling_target import _estimate_grid, centered_slope


def test_centered_slope_fits_intercept_and_drops_nonfinite():
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0, np.nan])
    y = 3.5 * x - 7.0
    assert np.isclose(centered_slope(y, x), 3.5)


def test_coupling_grid_recovers_cell_resolved_mag_and_size_slopes():
    import pandas as pd

    rng = np.random.default_rng(9)
    rows = []
    for flux_bin, mag_slope, size_slope in [(0, -0.4, 0.3), (1, 1.2, -0.2)]:
        for _ in range(200):
            predictor = rng.normal(scale=0.01)
            rows.append(
                {
                    "r_input_p": 20.0 + 4.0 * flux_bin,
                    "Re_input_p": 0.8,
                    "r_blend": 0.0,
                    "orientation_shear": predictor,
                    "delta_mag": 2.0 + mag_slope * predictor,
                    "delta_log_size": -1.0 + size_slope * predictor,
                }
            )
    frame = pd.DataFrame(rows)
    result = _estimate_grid(
        frame,
        np.array([18.0, 22.0, 26.0]),
        np.array([0.5, 1.5]),
        np.array([-1.0, 1.0]),
        min_count=100,
    )
    mag, size, counts, fallback, *_ = result
    np.testing.assert_allclose(mag[:, 0, 0], [-0.4, 1.2], atol=1e-12)
    np.testing.assert_allclose(size[:, 0, 0], [0.3, -0.2], atol=1e-12)
    np.testing.assert_array_equal(counts[:, 0, 0], [200, 200])
    assert not fallback.any()
