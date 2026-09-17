import numpy as np

from scripts.diagnose_truth_only_boundary_predictability import (
    case_bin_summary,
    sampled_case,
)


def test_sampled_case_is_disjoint_and_deterministic():
    arrays = {
        "features": np.arange(30, dtype=np.float32).reshape(10, 3),
        "response": np.arange(10, dtype=np.float32),
        "radius": np.arange(10, dtype=np.float32) + 3.01,
    }
    fit, evaluation = sampled_case(arrays, fit_rows=4, evaluation_rows=3, seed=7)
    fit2, evaluation2 = sampled_case(arrays, fit_rows=4, evaluation_rows=3, seed=7)
    np.testing.assert_array_equal(fit["response"], fit2["response"])
    np.testing.assert_array_equal(evaluation["response"], evaluation2["response"])
    assert not set(fit["response"]) & set(evaluation["response"])


def test_case_bin_summary_uses_paired_case_residuals():
    rows = case_bin_summary(
        case=np.array([0, 0, 1, 1]),
        radius=np.array([3.01, 3.02, 3.01, 3.02]),
        measured=np.array([1.0, 3.0, 2.0, 4.0]),
        predictions={"model": np.array([0.0, 2.0, 1.0, 3.0])},
        edges=(3.0, 3.1, np.inf),
    )
    assert rows[0]["measured_response"] == 2.5
    assert rows[0]["predicted_response"] == 1.5
    assert rows[0]["measured_minus_predicted"] == 1.0
    assert rows[0]["cases"] == 2
    assert rows[0]["objects"] == 4
