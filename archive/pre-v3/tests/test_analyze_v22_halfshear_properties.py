import numpy as np

from scripts.analyze_v22_halfshear_properties import (
    blend_axis,
    constant_shape_test,
    edge_axis,
    profile_axis,
)


def test_edge_and_frozen_blend_assignments():
    ordinary = edge_axis(
        "x", np.asarray([-1.0, 0.0, 0.5, 1.0, 2.0]), np.asarray([0.0, 1.0, 2.0]), "{:.1f}",
    )
    assert ordinary.ids.tolist() == [-1, 0, 0, 1, -1]

    blend = blend_axis(
        np.asarray([-0.1, 0.019, 0.02, 0.1, 0.3, 0.8, 5.0]),
        np.asarray([0.02, 0.05, 0.2, 0.5, 1.0]),
        epsilon=0.02,
    )
    assert blend.ids.tolist() == [0, 0, 1, 2, 3, 4, -1]


def test_constant_shape_gls_distinguishes_flat_and_structured_profiles():
    covariance = np.eye(4) * 0.01
    flat = constant_shape_test(np.ones(4), covariance)
    structured = constant_shape_test(np.asarray([0.0, 1.0, 2.0, 3.0]), covariance)

    assert flat["chi2"] == 0.0
    assert flat["p_value"] == 1.0
    assert structured["chi2"] > 100.0
    assert structured["p_value"] < 1e-10


def test_profile_records_empty_edge_bin_instead_of_failing():
    values = np.asarray([0.2, 0.3, 0.4, 1.2, 1.3, 1.4])
    axis = edge_axis("x", values, np.asarray([0.0, 0.1, 1.0, 2.0]), "{:.1f}")
    result = profile_axis(
        axis,
        np.ones(6, dtype=bool),
        np.asarray([1, 2, 3, 1, 2, 3]),
        np.asarray([0.9, 1.0, 1.1, 0.9, 1.0, 1.1]),
        np.asarray([[1.0, 1.0]] * 3 + [[1.1, 1.1]] * 3),
        min_count=2,
    )

    assert result["omitted_bins_below_min_count"] == [
        {"bin": 1, "label": "[0.0,0.1)", "N": 0}
    ]
    assert len(result["bins"]) == 2
