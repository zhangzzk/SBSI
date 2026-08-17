import numpy as np
import pandas as pd

from scripts.plot_anchor_high_response_stamps import (
    exact_cutout,
    local_detected_centroid,
    select_quantile_case_diverse,
    square_grid_size,
)


def synthetic_anchors() -> pd.DataFrame:
    rows = []
    for case in range(10):
        for local in range(5):
            response = 0.101 + 0.01 * case + 0.001 * local
            rows.append({
                "case": case,
                "input_index": 1000 * case + local,
                "scene_prediction": response,
                "R_blend_truth": 2.0 * response,
                "bias_truth_minus_model": response,
                "primary_mag": 23.0 + 0.1 * local,
                "log10_primary_size": np.log10(0.8),
                "log10_primary_sersic_n": np.log10(1.5),
                "log1p_n_pairs": np.log1p(10 + local),
            })
    return pd.DataFrame(rows)


def test_quantile_sample_is_thresholded_ordered_and_case_diverse():
    anchors = synthetic_anchors()
    got = select_quantile_case_diverse(anchors, n_select=6, threshold=0.1)
    assert len(got) == 6
    assert got.case.nunique() == 6
    assert got.scene_prediction.gt(0.1).all()
    assert np.all(np.diff(got.scene_prediction) > 0.0)
    assert np.all(np.diff(got.target_quantile) > 0.0)
    np.testing.assert_allclose(
        np.log1p(got.n_pairs), got.log1p_n_pairs, atol=1e-12
    )


def test_exact_cutout_replays_one_indexed_convention():
    image = np.arange(10000, dtype=float).reshape(100, 100)
    stamp, local_x, local_y = exact_cutout(
        image, x_1indexed=51.25, y_1indexed=62.75, stamp_size=8
    )
    expected = image[57:65, 46:54]
    np.testing.assert_array_equal(stamp, expected)
    assert np.isclose(local_x, 4.25)
    assert np.isclose(local_y, 4.75)


def test_square_grid_size_accepts_8_by_8_and_rejects_non_square():
    assert square_grid_size(36) == 6
    assert square_grid_size(64) == 8
    with np.testing.assert_raises(ValueError):
        square_grid_size(63)


def test_detected_centroid_preserves_global_offset_in_local_cutout():
    local_x, local_y = local_detected_centroid(
        local_x_truth=24.25,
        local_y_truth=24.75,
        x_truth_1indexed=125.25,
        y_truth_1indexed=233.75,
        x_detect_1indexed=124.90,
        y_detect_1indexed=234.15,
        stamp_size=48,
    )
    assert np.isclose(local_x, 23.90)
    assert np.isclose(local_y, 25.15)
