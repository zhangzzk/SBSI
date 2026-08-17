import numpy as np

from scripts.plot_v22_emulator_label_calibration import (
    fit_overlay_curves,
    quantile_edges,
    summarize_calibration,
)


def test_quantile_edges_are_strict_and_cover_values():
    prediction = np.arange(20, dtype=float)
    edges = quantile_edges(prediction, 4)
    assert len(edges) == 5
    assert np.all(np.diff(edges) > 0)
    assert edges[0] == prediction.min()
    assert edges[-1] == prediction.max()


def test_calibration_uses_case_means_and_paired_residual_error():
    case = np.array([1, 1, 1, 1, 2, 2, 2, 2])
    prediction = np.array([0.1, 0.2, 0.7, 0.8, 0.1, 0.2, 0.7, 0.8])
    label = prediction + np.array([0.1, 0.1, -0.2, -0.2, 0.3, 0.3, 0.0, 0.0])
    got = summarize_calibration(
        case, label, prediction, np.array([0.0, 0.5, 1.0]), 1, 2
    )

    assert got["n_pairs"] == 8
    assert got["n_cases"] == 2
    assert len(got["bins"]) == 2
    assert np.isclose(got["bins"][0]["prediction"]["mean"], 0.15)
    assert np.isclose(got["bins"][0]["label_minus_prediction"]["mean"], 0.2)
    assert np.isclose(got["bins"][1]["label_minus_prediction"]["mean"], -0.1)
    assert np.isclose(
        got["global_pair_label_minus_prediction"]["mean"], 0.05
    )
    assert np.isclose(sum(item["pair_fraction"] for item in got["bins"]), 1.0)


def test_frozen_fit_overlay_validates_source_and_evaluates_both_curves():
    calibration = {"n_pairs": 123, "case_window": [40, 199]}
    fit = {
        "calibration_source": {"n_pairs": 123, "case_window": [40, 199]},
        "calibration_fits": {
            "hinge": {"parameters": [0.01, 2.0, 0.5]},
            "saturating_hinge": {"parameters": [0.01, 2.0, 0.5, 0.1]},
            "natural_cubic_spline": {
                "parameters": [
                    -0.2, -0.01, 0.02, 0.3,
                    -0.1, -0.01, 0.03, 0.08,
                ],
            },
        },
    }
    prediction = np.asarray([-0.2, 0.0, 0.3])
    curves = fit_overlay_curves(calibration, fit, prediction)
    np.testing.assert_allclose(curves["hinge"], [-0.39, 0.01, 0.16])
    np.testing.assert_allclose(
        curves["saturating_hinge"],
        [0.01 - 0.4 / 3.0, 0.01, 0.01 + 0.15 / 4.0],
    )
    assert np.isfinite(curves["natural_cubic_spline"]).all()
    np.testing.assert_allclose(
        curves["natural_cubic_spline"][[0, -1]], [-0.1, 0.08],
        atol=1e-15,
    )
