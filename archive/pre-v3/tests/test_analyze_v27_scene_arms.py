import numpy as np
import pandas as pd

from scripts.analyze_v27_scene_arms import bins, finite_halfshear_rows, merge_halfshear_frames


def test_same_named_flow_columns_can_be_renamed_before_merge():
    new = pd.DataFrame({"case": [1], "input_index": [2], "r_sim_self": [0.8],
                        "R_flow_s501": [0.81]})
    old = pd.DataFrame({"case": [1], "input_index": [2], "R_flow_s501": [0.79]})
    merged = merge_halfshear_frames(new, old, "R_flow_s501", "R_flow_s501")
    assert merged.loc[0, "variant"] == 0.81
    assert merged.loc[0, "baseline"] == 0.79


def test_quantile_bins_keep_tied_zero_shells_together():
    values = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0])
    group = bins(values, 4)
    assert np.unique(group[values == 0.0]).tolist() == [0]
    assert group.max() + 1 < 4


def test_halfshear_finite_mask_rejects_undefined_truth():
    frame = pd.DataFrame({
        "r_sim_self": [0.8, np.nan],
        "variant": [0.81, 0.82],
        "baseline": [0.79, 0.80],
    })
    assert finite_halfshear_rows(frame).tolist() == [True, False]
