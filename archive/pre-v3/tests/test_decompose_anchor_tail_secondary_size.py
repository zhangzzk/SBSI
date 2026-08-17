import numpy as np
import pandas as pd

from scripts.decompose_anchor_tail_secondary_size import decompose


def test_tail_size_cells_close_tail_and_global_gaps():
    frame = pd.DataFrame({
        "case": [1, 1, 1, 1, 2, 2, 2, 2],
        "input_index": range(8),
        "R_blend_truth": [1.0, 0.3, 0.2, 0.0, 0.8, 0.4, 0.1, 0.0],
        "scene_prediction": [0.6, 0.5, 0.1, 0.0, 0.6, 0.5, 0.1, 0.0],
        "has_deployed_pair": [True, True, True, False] * 2,
        "log10_dominant_secondary_size": [
            np.log10(0.3), np.log10(0.5), np.log10(0.2), np.nan,
            np.log10(0.3), np.log10(0.5), np.log10(0.2), np.nan,
        ],
    })
    got = decompose(frame, threshold=0.4, size_cut=0.4)
    assert got["cells"]["tail_small"]["n_rows"] == 2
    assert got["cells"]["tail_kept"]["n_rows"] == 2
    assert got["cells"]["outside_small"]["n_rows"] == 2
    assert got["cells"]["outside_kept"]["n_rows"] == 2
    assert got["closure"]["maximum_case_global_additive_error"] < 1e-15
    assert got["closure"]["maximum_case_tail_additive_error"] < 1e-15
    assert np.isclose(
        got["tail"]["before_removal"]["gap"]["mean"],
        0.075,
    )
    assert np.isclose(
        got["tail"]["after_removing_small"]["gap"]["mean"],
        -0.15,
    )


def test_outside_small_can_cancel_tail_small_globally():
    frame = pd.DataFrame({
        "case": [1, 1, 1, 1, 2, 2, 2, 2],
        "input_index": range(8),
        "R_blend_truth": [1.0, 0.5, -0.2, 0.0] * 2,
        "scene_prediction": [0.5, 0.5, 0.0, 0.0] * 2,
        "has_deployed_pair": [True, True, True, False] * 2,
        "log10_dominant_secondary_size": [
            np.log10(0.3), np.log10(0.5), np.log10(0.2), np.nan,
        ] * 2,
    })
    got = decompose(frame, threshold=0.4, size_cut=0.4)
    tail_small = got["cells"]["tail_small"][
        "additive_contribution_to_global_gap"
    ]["mean"]
    outside_small = got["cells"]["outside_small"][
        "additive_contribution_to_global_gap"
    ]["mean"]
    assert tail_small > 0
    assert outside_small < 0
    assert np.isclose(tail_small + outside_small, 0.075)
