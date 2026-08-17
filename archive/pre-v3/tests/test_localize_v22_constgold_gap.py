import numpy as np
import pandas as pd

from scripts.localize_v22_constgold_gap import selected_vs_complement, summarize_bins


def toy_frame():
    rows = []
    for case in range(4):
        for value in range(10):
            model = 1.0
            gap = 0.2 if value >= 5 else 0.0
            rows.append({
                "case": case, "feature": value, "gap": gap,
                "R_sim": model + gap, "R_flow": 0.8,
                "R_blend": 0.2, "R_model": model,
            })
    return pd.DataFrame(rows)


def test_selected_contribution_uses_all_case_rows_as_denominator():
    frame = toy_frame()
    selected = frame.feature.to_numpy() >= 5
    result = selected_vs_complement(frame, selected)
    np.testing.assert_allclose(result["selected_fraction"]["mean"], 0.5)
    np.testing.assert_allclose(
        result["selected_contribution_to_global_gap"]["mean"], 0.1
    )
    np.testing.assert_allclose(
        result["selected_minus_complement_gap"]["mean"], 0.2
    )


def test_bin_summary_preserves_global_gap_as_sum_of_contributions():
    frame = toy_frame()
    result = summarize_bins(frame, "feature", np.array([-np.inf, 5.0, np.inf]))
    contribution = sum(
        row["contribution_to_global_gap"]["mean"] for row in result["bins"]
    )
    np.testing.assert_allclose(contribution, frame.groupby("case").gap.mean().mean())
