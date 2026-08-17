import numpy as np
import pandas as pd

from scripts.diagnose_anchorblend_bright_secondary_cut import summarize_cut


def test_cut_summary_contributions_replay_baseline():
    frame = pd.DataFrame({
        "case": [1, 1, 1, 1, 2, 2, 2, 2],
        "gap": [-4.0, -4.0, 2.0, 2.0, -2.0, -2.0, 2.0, 2.0],
    })
    remove = np.array([True, True, False, False] * 2)
    out = summarize_cut(frame, remove)
    np.testing.assert_allclose(out["kept_gap"]["mean"], 2.0)
    np.testing.assert_allclose(out["removed_gap"]["mean"], -3.0)
    replay = (
        out["kept_contribution_to_baseline"]["mean"]
        + out["removed_contribution_to_baseline"]["mean"]
    )
    np.testing.assert_allclose(replay, out["baseline_gap"]["mean"])
