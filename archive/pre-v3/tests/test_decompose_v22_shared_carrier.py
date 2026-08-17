import numpy as np
import pandas as pd

from scripts.decompose_v22_shared_carrier import rule_masks, summarize


def test_summary_uses_case_weighted_contributions():
    frame = pd.DataFrame({
        "case": [1, 1, 1, 1, 2, 2, 2, 2],
        "value": [2.0, 2.0, 0.0, 0.0, 4.0, 4.0, 0.0, 0.0],
    })
    selected = np.array([True, True, False, False] * 2)
    got = summarize(frame, selected, "value")
    assert got["selected_fraction"]["mean"] == 0.5
    assert got["selected_conditional"]["mean"] == 3.0
    assert got["selected_contribution"]["mean"] == 1.5
    assert got["selected_carrier_share"] == 1.0


def test_fixed_rule_masks_are_nested_as_expected():
    frame = pd.DataFrame({
        "dominant_to_runner_up_abs_response": [4.0, 6.0, 11.0, 21.0],
        "dominant_response": [1.0, 1.0, -1.0, 1.0],
        "top_abs_fraction": [0.6, 0.8, 0.8, 0.9],
    })
    got = rule_masks(frame)
    assert got["pilot_frozen_ratio_gt5_positive"].tolist() == [False, True, False, True]
    assert got["core_ratio_gt10_positive"].tolist() == [False, False, False, True]
    assert got["narrow_ratio_gt20"].tolist() == [False, False, False, True]
