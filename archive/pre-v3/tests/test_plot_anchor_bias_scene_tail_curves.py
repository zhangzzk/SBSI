import numpy as np
import pandas as pd

from scripts.plot_anchor_bias_scene_tail_curves import (
    extreme_contrast,
    frozen_tail_definition,
    tail_mask,
)


def test_frozen_tail_definition_uses_last_all_scene_bin():
    curves = pd.DataFrame({
        "population": ["all", "all", "carrier"],
        "feature": ["scene_prediction", "scene_prediction", "scene_prediction"],
        "bin": [0, 1, 1],
        "lower": [-np.inf, 0.4, 0.2],
        "upper": [0.4, np.inf, np.inf],
        "n_rows": [20, 10, 5],
    })
    got = frozen_tail_definition(curves)
    assert got["threshold"] == 0.4
    assert got["parent_bin"] == 1
    assert got["parent_test_rows"] == 10
    assert got["parent_curve_bins"] == 2


def test_tail_mask_includes_frozen_edge():
    frame = pd.DataFrame({"scene_prediction": [0.39, 0.4, 0.5, np.nan]})
    assert tail_mask(frame, 0.4).tolist() == [False, True, True, False]


def test_extreme_contrast_is_case_paired_high_minus_low():
    development = pd.DataFrame({"x": np.arange(20, dtype=float)})
    test = pd.DataFrame({
        "case": [1, 1, 2, 2],
        "bias_truth_minus_model": [1.0, 3.0, 2.0, 6.0],
        "x": [1.0, 19.0, 1.0, 19.0],
    })
    got = extreme_contrast(
        development, test, np.asarray([0.0, 1.0, 0.0, 2.0]), "x", 2
    )
    assert got is not None
    assert got["n_paired_cases"] == 2
    assert np.isclose(got["target_high_minus_low"]["mean"], 3.0)
    assert np.isclose(got["prediction_high_minus_low"]["mean"], 1.5)
