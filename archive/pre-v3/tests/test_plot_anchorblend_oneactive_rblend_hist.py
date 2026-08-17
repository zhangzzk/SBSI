import numpy as np
import pandas as pd

from scripts.plot_anchorblend_oneactive_rblend_hist import (
    histogram_by_case,
    summarize,
    weighted_quantile,
)


def test_weighted_quantile_and_case_histogram_are_normalized():
    frame = pd.DataFrame({
        "case": [0, 0, 0, 1, 1, 1],
        "R_one_pair": [-2.0, 0.0, 1.0, -1.0, 0.0, 2.0],
        "n_pairs": [1, 2, 3, 3, 2, 1],
    })
    median = weighted_quantile(
        frame.R_one_pair.to_numpy(), frame.n_pairs.to_numpy(), np.asarray([0.5]),
    )
    assert -1.0 <= median[0] <= 1.0
    mean, sem = histogram_by_case(frame, np.asarray([-3.0, -0.5, 0.5, 3.0]), "n_pairs")
    np.testing.assert_allclose(mean.sum(), 1.0, atol=1e-15)
    assert np.isfinite(sem).all()


def test_summary_uses_cases_as_mean_uncertainty_unit():
    frame = pd.DataFrame({
        "case": np.repeat([400, 401], 4),
        "R_one_pair": [-2.0, -0.2, 0.1, 1.0, -1.0, -0.1, 0.2, 2.0],
        "n_pairs": [2, 3, 4, 5, 5, 4, 3, 2],
    })
    result = summarize(frame, 0.001)
    assert result["n_selected_pairs"] == 8
    assert result["n_cases"] == 2
    assert result["case_window"] == [400, 401]
    assert result["ht_pair_population_case_balanced_mean"]["n_cases"] == 2
    assert result["display_limits"][0] == -result["display_limits"][1]
