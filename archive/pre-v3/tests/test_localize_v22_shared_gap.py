import numpy as np
import pandas as pd

from scripts.localize_v22_shared_gap import CaseAggregator, candidate_rules


def test_case_aggregator_reports_conditional_and_contribution():
    frame = pd.DataFrame({
        "case": [1, 1, 1, 1, 2, 2, 2, 2],
        "deficit": [1.0, 1.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0],
    })
    mask = np.array([True, True, False, False] * 2)
    got = CaseAggregator(frame, "deficit").summarize(mask)
    assert got is not None
    assert got["selected_fraction"]["mean"] == 0.5
    assert got["selected_conditional_deficit"]["mean"] == 1.5
    assert got["complement_conditional_deficit"]["mean"] == 0.0
    assert got["selected_contribution"]["mean"] == 0.75
    assert got["carrier_share"] == 1.0


def test_candidate_grid_contains_frozen_anchor_controls_and_joint_distance_sign():
    conditions, rules = candidate_rules()
    assert "ratio:>20" in conditions
    assert "top_fraction:>0.775277" in conditions
    assert (
        "ratio:>20", "distance:[3,5)", "dominant_response:positive"
    ) in rules
