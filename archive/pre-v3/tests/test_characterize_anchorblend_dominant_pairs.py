import numpy as np
import pandas as pd

from scripts.characterize_anchorblend_dominant_pairs import (
    conditional_gap_summary,
    select_dominant_pairs,
)


def test_select_dominant_pair_is_stable_and_uses_absolute_response():
    pairs = pd.DataFrame({
        "anchor_index": [1, 1, 1, 2, 2],
        "secondary_index": [12, 11, 10, 20, 21],
        "response": [0.2, -0.5, 0.5, 0.1, 0.3],
        "distance": [1.0, 2.0, 3.0, 1.0, 2.0],
    })
    out = select_dominant_pairs(pairs)
    # Equal absolute responses for anchor 1 use the lower stable secondary ID.
    assert out.set_index("anchor_index").loc[1, "secondary_index"] == 10
    assert out.set_index("anchor_index").loc[2, "secondary_index"] == 21
    np.testing.assert_allclose(out.set_index("anchor_index").loc[1, "R_abs_sum"], 1.2)
    np.testing.assert_allclose(
        out.set_index("anchor_index").loc[1, "top_abs_fraction"], 0.5 / 1.2,
    )


def test_gap_split_uses_paired_cases_and_replays_global_contribution():
    frame = pd.DataFrame({
        "case": [1, 1, 1, 1, 2, 2, 2, 2],
        "gap": [-4.0, -4.0, 2.0, 2.0, -2.0, -2.0, 2.0, 2.0],
    })
    selected = np.array([True, True, False, False] * 2)
    out = conditional_gap_summary(frame, selected)
    np.testing.assert_allclose(out["selected_conditional_gap"]["mean"], -3.0)
    np.testing.assert_allclose(out["outside_conditional_gap"]["mean"], 2.0)
    total = (
        out["selected_contribution_to_global_gap"]["mean"]
        + out["outside_contribution_to_global_gap"]["mean"]
    )
    np.testing.assert_allclose(total, frame.groupby("case").gap.mean().mean())
