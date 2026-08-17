import numpy as np
import pandas as pd

from scripts.analyze_anchorblend_distance_models import summarize_split


def test_summarize_split_uses_case_level_gaps():
    frame = pd.DataFrame({
        "case": [0, 0, 1, 1],
        "R_blend_truth": [1.0, 1.0, 3.0, 3.0],
        "R_blend_lsst_r_extnbr_v22": [0.0, 0.0, 2.0, 2.0],
        "R_blend_lsst_r_extnbr_indist": [0.5, 0.5, 2.5, 2.5],
        "R_blend_lsst_r_extnbr_indist_wc5": [1.0, 1.0, 3.0, 3.0],
        "R_blend_lsst_r_extnbr_indist_wc20": [1.5, 1.5, 3.5, 3.5],
    })
    got = summarize_split(frame)
    assert got["truth_case_mean"] == 2.0
    assert got["models"]["lsst_r_extnbr_v22"]["prediction_minus_truth"] == -1.0
    assert got["models"]["lsst_r_extnbr_indist_wc5"]["prediction_minus_truth"] == 0.0
    assert got["candidate_vs_baseline"]["fraction_of_baseline_deficit_closed"] == 1.0
    assert np.isclose(got["models"]["lsst_r_extnbr_v22"]["gap_case_sem"], 0.0)
