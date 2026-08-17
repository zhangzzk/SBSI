import numpy as np
import pandas as pd

from scripts.diag_v22_summed_label_closure import (
    input_frame_distance,
    summarize,
    summarize_component,
)


def test_summarize_uses_case_means_and_primary_denominator():
    case = pd.DataFrame({
        "n_primaries": [10, 20],
        "n_pairs": [30, 80],
        "label_sum_mean": [1.0, 3.0],
        "prediction_sum_mean": [0.5, 2.5],
        "null_sum_mean": [-0.1, 0.1],
    })
    got = summarize(case)
    assert got["n_cases"] == 2
    assert got["n_primaries"] == 30
    assert got["n_pairs"] == 110
    assert np.isclose(got["label_sum_mean"], 2.0)
    assert np.isclose(got["prediction_sum_mean"], 1.5)
    assert np.isclose(got["prediction_minus_label"], -0.5)
    assert np.isclose(got["gap_case_sem"], 0.0)
    assert np.isclose(got["null_sum_mean"], 0.0)


def test_component_summary_preserves_all_primary_denominator():
    case = pd.DataFrame({
        "n_primaries": [10, 20],
        "shell_n_pairs": [5, 10],
        "shell_label_sum_mean": [0.2, 0.4],
        "shell_prediction_sum_mean": [0.1, 0.3],
        "shell_null_sum_mean": [-0.02, 0.02],
    })
    got = summarize_component(case, "shell")
    assert got["n_pairs"] == 15
    assert np.isclose(got["mean_pairs_per_primary"], 0.5)
    assert np.isclose(got["label_sum_mean"], 0.3)
    assert np.isclose(got["prediction_sum_mean"], 0.2)
    assert np.isclose(got["label_minus_prediction"], 0.1)
    assert np.isclose(got["prediction_minus_label"], -0.1)


def test_input_frame_distance_returns_arcseconds():
    frame = pd.DataFrame({
        "RA_input_p": [10.0], "DEC_input_p": [0.0],
        "RA_input_s": [10.0 + 2.5 / 3600.0], "DEC_input_s": [0.0],
    })
    assert np.isclose(input_frame_distance(frame)[0], 2.5, atol=1e-9)
