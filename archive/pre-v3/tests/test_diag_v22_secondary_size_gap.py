import numpy as np
import pandas as pd

from scripts.diag_v22_secondary_size_gap import (
    anchor_removed_mask,
    summarize_anchor_cut,
    summarize_half_shear_accumulators,
)


def test_half_shear_cut_boundary_and_additive_closure():
    edges = np.array([0.0, 0.2, 0.4, 1.0])
    counts = np.array([[2, 1, 3], [1, 2, 1]])
    label = np.array([[0.4, 0.3, 0.6], [0.1, 0.4, 0.3]])
    prediction = np.array([[0.2, 0.2, 0.5], [0.2, 0.2, 0.2]])
    null = np.zeros_like(label)
    size_sum = np.array([[0.2, 0.3, 1.8], [0.1, 0.6, 0.6]])
    n_primary = np.array([2, 4])

    got = summarize_half_shear_accumulators(
        counts, label, prediction, null, size_sum, n_primary, edges, 0.4
    )
    below = got["groups"]["below_cut"]
    above = got["groups"]["at_or_above_cut"]
    assert below["n_pairs"] == 6
    assert above["n_pairs"] == 4
    assert np.isclose(
        below["additive_label_minus_prediction_per_primary"]["mean"],
        np.mean([(0.4 + 0.3 - 0.2 - 0.2) / 2, (0.1 + 0.4 - 0.2 - 0.2) / 4]),
    )
    assert np.isclose(
        got["groups"]["all"][
            "additive_label_minus_prediction_per_primary"
        ]["mean"],
        below["additive_label_minus_prediction_per_primary"]["mean"]
        + above["additive_label_minus_prediction_per_primary"]["mean"],
    )
    assert got["maximum_additive_closure_error"] == 0.0


def test_anchor_removed_mask_keeps_no_pair_and_uses_strict_cut():
    frame = pd.DataFrame({
        "has_deployed_pair": [True, True, False],
        "log10_dominant_secondary_size": [np.log10(0.399), np.log10(0.4), np.nan],
    })
    assert anchor_removed_mask(frame, 0.4).tolist() == [True, False, False]


def test_anchor_cut_uses_case_means_and_closes_additively():
    frame = pd.DataFrame({
        "case": [1, 1, 1, 2, 2, 2],
        "R_blend_truth": [1.0, 0.4, 0.2, 0.8, 0.3, 0.1],
        "scene_prediction": [0.0, 0.2, 0.1, 0.2, 0.2, 0.1],
        "has_deployed_pair": [True, True, False, True, True, False],
        "log10_dominant_secondary_size": [
            np.log10(0.3), np.log10(0.5), np.nan,
            np.log10(0.2), np.log10(0.6), np.nan,
        ],
    })
    got = summarize_anchor_cut(frame, 0.4)
    assert got["populations"]["removed"]["n_rows"] == 2
    assert got["populations"]["kept"]["n_rows"] == 4
    assert np.isclose(got["removed_fraction_pooled"], 1 / 3)
    assert got["additive_gap_decomposition"]["maximum_case_closure_error"] < 1e-15
    original = got["populations"]["all"]["truth_minus_prediction"]["mean"]
    removed = got["additive_gap_decomposition"]["removed_contribution"]["mean"]
    kept = got["additive_gap_decomposition"]["kept_contribution"]["mean"]
    assert np.isclose(original, removed + kept)
