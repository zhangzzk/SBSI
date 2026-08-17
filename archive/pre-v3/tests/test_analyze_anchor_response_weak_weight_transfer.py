import numpy as np
import pandas as pd

from scripts.analyze_anchor_response_weak_weight_transfer import (
    case_stat,
    subset_mask,
    summarize_slice,
)


def test_case_stat_uses_case_sem():
    result = case_stat(np.asarray([1.0, 3.0, 5.0, 7.0]))
    assert result["mean"] == 4.0
    assert np.isclose(result["case_sem"], np.std([1.0, 3.0, 5.0, 7.0], ddof=1) / 2.0)
    assert result["n_cases"] == 4


def test_tail_selection_is_frozen_baseline_only():
    frame = pd.DataFrame({
        "prediction_baseline": [0.1, 0.10001, -1.0],
        "prediction_model": [9.0, -9.0, 9.0],
        "R_blend_truth": [0.0, 0.0, 0.0],
    })
    assert subset_mask(frame, "tail").tolist() == [False, True, False]
    assert subset_mask(frame, "outside_tail").tolist() == [True, False, True]


def test_summarize_slice_preserves_paired_gap_identity():
    frame = pd.DataFrame({
        "case": [10, 10, 11, 11],
        "R_blend_truth": [0.4, 0.6, 0.8, 1.0],
        "prediction_baseline": [0.2, 0.4, 0.6, 0.8],
        "prediction_model": [0.3, 0.5, 0.7, 0.9],
    })
    summary, by_case = summarize_slice(
        frame,
        "prediction_model",
        10,
        11,
        np.ones(len(frame), dtype=bool),
    )
    assert np.allclose(by_case.truth_minus_baseline, 0.2)
    assert np.allclose(by_case.truth_minus_prediction, 0.1)
    assert np.allclose(by_case.prediction_minus_baseline, 0.1)
    assert np.isclose(summary["fraction_of_baseline_gap_removed"], 0.5)
    assert summary["absolute_gap_improved"]
