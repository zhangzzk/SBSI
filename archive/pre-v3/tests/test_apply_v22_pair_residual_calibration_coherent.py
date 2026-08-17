import numpy as np
import pandas as pd

from scripts.apply_v22_pair_residual_calibration_coherent import (
    binned_lookup_residual,
    calibration_components,
    fit_hinge_curve,
    hinge_residual,
    linear_interpolation_residual,
    make_binned_lookup,
    make_linear_interpolation,
    natural_cubic_spline_residual,
    saturating_hinge_residual,
)


def test_weighted_hinge_fit_recovers_exact_curve():
    prediction = np.asarray([-0.4, -0.1, -0.02, 0.0, 0.03, 0.2, 0.7])
    expected = np.asarray([0.012, 1.8, 0.35])
    residual = hinge_residual(prediction, expected)
    frame = pd.DataFrame({
        "prediction_mean": prediction,
        "label_minus_prediction_mean": residual,
        "label_minus_prediction_case_sem": np.linspace(0.01, 0.03, len(prediction)),
        "pair_fraction": np.full(len(prediction), 1.0 / len(prediction)),
    })
    fit = fit_hinge_curve(frame)
    np.testing.assert_allclose(fit["parameters"], expected, atol=1e-12)
    assert fit["unweighted_rmse"] < 1e-12


def test_pair_components_close_for_all_models():
    prediction = np.asarray([-0.3, -0.01, 0.0, 0.02, 0.5])
    for model, parameters, function in (
        ("hinge", np.asarray([0.001, 2.0, 0.4]), hinge_residual),
        (
            "saturating_hinge", np.asarray([0.001, 3.0, 1.5, 0.02]),
            saturating_hinge_residual,
        ),
        (
            "natural_cubic_spline",
            np.asarray([-0.3, -0.01, 0.02, 0.5, -0.2, -0.01, 0.03, 0.1]),
            natural_cubic_spline_residual,
        ),
        (
            "binned_lookup",
            np.asarray([-0.5, -0.02, 0.1, 0.7, -0.2, 0.03, 0.1]),
            binned_lookup_residual,
        ),
        (
            "linear_interpolation",
            np.asarray([-0.3, -0.01, 0.02, 0.5, -0.2, -0.01, 0.03, 0.1]),
            linear_interpolation_residual,
        ),
    ):
        components = calibration_components(prediction, model, parameters)
        np.testing.assert_allclose(
            components["total"], function(prediction, parameters), atol=1e-15,
        )
        np.testing.assert_allclose(
            components["intercept"]
            + components["negative_branch"]
            + components["positive_branch"],
            components["total"], atol=1e-15,
        )


def test_natural_cubic_spline_interpolates_and_clamps_endpoints():
    knots = np.asarray([-0.3, -0.01, 0.02, 0.5])
    values = np.asarray([-0.2, -0.01, 0.03, 0.1])
    parameters = np.r_[knots, values]
    np.testing.assert_allclose(
        natural_cubic_spline_residual(knots, parameters), values, atol=1e-15,
    )
    np.testing.assert_allclose(
        natural_cubic_spline_residual(np.asarray([-3.0, 5.0]), parameters),
        [values[0], values[-1]], atol=1e-15,
    )


def test_binned_lookup_uses_original_edges_without_fitting():
    edges = np.asarray([-0.5, -0.02, 0.1, 0.7])
    values = np.asarray([-0.2, 0.03, 0.1])
    parameters = np.r_[edges, values]
    prediction = np.asarray([-2.0, -0.3, -0.02, 0.0, 0.1, 0.4, 2.0])
    np.testing.assert_array_equal(
        binned_lookup_residual(prediction, parameters),
        [-0.2, -0.2, 0.03, 0.03, 0.1, 0.1, 0.1],
    )
    calibration = pd.DataFrame({
        "lower": edges[:-1],
        "upper": edges[1:],
        "prediction_mean": [-0.3, 0.0, 0.4],
        "label_minus_prediction_mean": values,
        "pair_fraction": [0.2, 0.3, 0.5],
    })
    lookup = make_binned_lookup(calibration)
    assert lookup["estimation"] == "direct saved bin means; no fitted parameters"
    np.testing.assert_array_equal(
        lookup["fitted_residual_at_bin_means"], values,
    )


def test_linear_interpolation_uses_original_points_without_fitting():
    knots = np.asarray([-0.5, -0.02, 0.1, 0.7])
    values = np.asarray([-0.2, -0.03, 0.05, 0.1])
    parameters = np.r_[knots, values]
    np.testing.assert_array_equal(
        linear_interpolation_residual(knots, parameters), values,
    )
    np.testing.assert_array_equal(
        linear_interpolation_residual([-2.0, 2.0], parameters),
        [values[0], values[-1]],
    )
    calibration = pd.DataFrame({
        "prediction_mean": knots,
        "label_minus_prediction_mean": values,
        "pair_fraction": np.full(len(knots), 0.25),
    })
    curve = make_linear_interpolation(calibration)
    assert curve["estimation"] == "straight segments through saved bin means; no fit"
    np.testing.assert_array_equal(curve["fitted_residual_at_bin_means"], values)
