import numpy as np
import pandas as pd

from scripts.analyze_anchorblend_v22_diagnostic import population_summaries, summary


def test_summary_uses_case_as_independent_unit():
    frame = pd.DataFrame({
        "case": [0, 0, 1],
        "R_blend_truth": [1.0, 1.0, 2.0],
        "prediction": [0.5, 0.5, 1.5],
    })
    row = summary(frame, "prediction")
    assert row["truth_case_mean"] == 1.5
    assert row["prediction_case_mean"] == 1.0
    assert row["prediction_minus_truth"] == -0.5
    assert np.isclose(row["truth_over_prediction_scale"], 1.5)


def test_population_summary_reports_v22_intersection():
    frame = pd.DataFrame({
        "case": [0, 1, 100, 101],
        "R_blend_truth": [1.0, 1.0, 2.0, 2.0],
        "prediction": [0.5, 0.5, 1.5, 1.5],
        "r_input_p_plus": [25.0, 26.0, 25.0, 26.0],
        "Re_input_p_plus": [0.6, 0.6, 0.6, 0.6],
    })
    row = population_summaries(frame, "prediction", 50)
    assert row["intersection_with_v22_primary_box"]["fraction_of_native_objects"] == 0.5
    assert row["intersection_with_v22_primary_box"]["all"]["n_objects"] == 2
