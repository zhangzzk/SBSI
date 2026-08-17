import numpy as np

from scripts.plot_v22_grouped_rscene_pair_residual_validation import (
    summarize_shared_bins,
)


def test_shared_bins_keep_x_fixed_and_shift_only_corrected_residual() -> None:
    case = np.asarray([20, 20, 20, 20, 21, 21, 21, 21])
    baseline = np.asarray([-0.2, -0.1, 0.1, 0.2] * 2)
    correction = np.asarray([0.01, 0.02, 0.03, 0.04] * 2)
    label = baseline + np.asarray([0.04, 0.03, 0.02, 0.01] * 2)
    edges = np.asarray([-0.2, 0.0, 0.2])
    got = summarize_shared_bins(
        case, label, baseline, baseline + correction, edges, 20, 21
    )

    assert got["n_pairs"] == 8
    assert got["n_cases"] == 2
    for index in (0, 1):
        raw = got["models"]["V2.2"]["bins"][index]
        corrected = got["models"]["Scene-informed correction"]["bins"][index]
        np.testing.assert_allclose(
            raw["v22_coordinate"]["mean"],
            corrected["v22_coordinate"]["mean"],
        )
        np.testing.assert_allclose(
            corrected["label_minus_prediction"]["mean"],
            raw["label_minus_prediction"]["mean"]
            - corrected["applied_correction"]["mean"],
        )


def test_shared_bins_case_balance_differs_from_pooled_when_counts_differ() -> None:
    case = np.asarray([20, 20, 20, 21])
    baseline = np.asarray([-0.2, -0.1, -0.05, -0.1])
    label = np.asarray([1.0, 1.0, 1.0, 3.0])
    edges = np.asarray([-0.2, 0.0, 0.1])
    # Add one row in the positive bin for each case so both bins are valid.
    case = np.r_[case, [20, 21]]
    baseline = np.r_[baseline, [0.05, 0.05]]
    label = np.r_[label, [0.0, 0.0]]
    got = summarize_shared_bins(
        case, label, baseline, baseline, edges, 20, 21
    )
    first = got["models"]["V2.2"]["bins"][0]
    expected_case_balanced_label = (1.0 + 3.0) / 2.0
    np.testing.assert_allclose(first["label"]["mean"], expected_case_balanced_label)
    assert not np.isclose(first["label"]["mean"], (3.0 + 3.0) / 4.0)
