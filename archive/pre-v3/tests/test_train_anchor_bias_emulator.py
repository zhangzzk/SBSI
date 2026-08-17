import numpy as np
import pandas as pd

from scripts.train_anchor_bias_emulator import (
    case_balanced_mean,
    case_balanced_weights,
    conditional_curve,
    metric_summary,
    prediction_calibration,
    quantile_edges,
    within_case_permutation,
)


def test_case_balanced_weights_give_equal_total_weight_per_case():
    cases = np.asarray([1, 1, 1, 2, 2, 3])
    weights = case_balanced_weights(cases)
    totals = pd.DataFrame({"case": cases, "weight": weights}).groupby("case").weight.sum()
    assert np.allclose(totals, totals.iloc[0])
    assert np.isclose(weights.mean(), 1.0)


def test_metric_summary_uses_case_balanced_mse():
    frame = pd.DataFrame({
        "case": [1, 1, 1, 2],
        "bias_truth_minus_model": [1.0, 1.0, 1.0, 3.0],
    })
    constant = case_balanced_mean(
        frame.bias_truth_minus_model.to_numpy(), frame.case.to_numpy()
    )
    assert np.isclose(constant, 2.0)
    got = metric_summary(frame, np.asarray([1.0, 1.0, 1.0, 3.0]), constant)
    assert np.isclose(got["case_balanced_mse"], 0.0)
    assert np.isclose(got["mse_skill"], 1.0)


def test_within_case_permutation_never_crosses_case():
    cases = np.asarray([1, 1, 1, 2, 2, 3, 3, 3])
    permutation = within_case_permutation(cases, np.random.default_rng(5))
    assert np.array_equal(cases, cases[permutation])
    assert sorted(permutation.tolist()) == list(range(len(cases)))


def test_conditional_curve_case_balances_unequal_rows():
    development = pd.DataFrame({"x": np.arange(20, dtype=float)})
    test = pd.DataFrame({
        "case": [1, 1, 1, 2],
        "bias_truth_minus_model": [0.0, 0.0, 0.0, 2.0],
        "x": [1.0, 1.1, 1.2, 1.3],
    })
    rows = conditional_curve(
        development, test, np.zeros(len(test)), "x", "all", n_bins=2
    )
    occupied = next(row for row in rows if row["n_rows"] == 4)
    assert np.isclose(occupied["target_mean"], 1.0)
    assert occupied["n_cases"] == 2


def test_conditional_curve_omits_single_case_extreme_bin():
    development = pd.DataFrame({"x": [0.0, 0.0, 0.0, 1.0, 2.0]})
    test = pd.DataFrame({
        "case": [1, 2, 1],
        "bias_truth_minus_model": [0.0, 0.0, 3.0],
        "x": [0.0, 0.0, 2.0],
    })
    rows = conditional_curve(
        development, test, np.zeros(len(test)), "x", "all", n_bins=3
    )
    assert len(rows) == 1
    assert rows[0]["n_cases"] == 2


def test_conditional_curve_returns_empty_for_constant_coordinate():
    development = pd.DataFrame({"x": np.zeros(10)})
    test = pd.DataFrame({
        "case": [1, 2],
        "bias_truth_minus_model": [0.0, 1.0],
        "x": [0.0, 0.0],
    })
    assert conditional_curve(
        development, test, np.zeros(len(test)), "x", "carrier", n_bins=3
    ) == []


def test_quantile_edges_stay_bounded_for_zero_spike_continuous_tail():
    values = np.r_[np.zeros(10_000), np.linspace(0.1, 1.0, 500)]
    edges = quantile_edges(values, n_bins=12)
    assert len(edges) <= 13
    assert edges[0] == -np.inf
    assert edges[-1] == np.inf


def test_prediction_calibration_recovers_ordered_span():
    cases = np.repeat(np.arange(20), 10)
    prediction = np.tile(np.arange(10, dtype=float), 20)
    frame = pd.DataFrame({
        "case": cases,
        "bias_truth_minus_model": prediction * 2.0,
    })
    edges = np.r_[-np.inf, np.arange(0.5, 9.0, 1.0), np.inf]
    got = prediction_calibration(frame, prediction, edges)
    assert np.isclose(got["observed_high_minus_low"]["mean"], 18.0)
    assert np.isclose(got["bin_mean_spearman"], 1.0)
    assert np.isclose(got["bin_mean_calibration_slope"], 2.0)
