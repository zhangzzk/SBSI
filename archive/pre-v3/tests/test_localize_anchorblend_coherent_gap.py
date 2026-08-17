import numpy as np
import pandas as pd

from scripts.localize_anchorblend_coherent_gap import (
    aggregate_pairs,
    quantile_edges,
    summarize_bins,
)


def test_aggregate_pairs_preserves_signed_sum_and_shells():
    pairs = pd.DataFrame({
        "anchor_index": [1, 1, 1, 2],
        "secondary_index": [10, 11, 12, 20],
        "response": [0.3, -0.1, 0.2, -0.4],
        "distance": [1.0, 3.0, 7.0, 1.5],
    })
    got = aggregate_pairs(pairs)
    np.testing.assert_allclose(got.loc[1, "pair_sum"], 0.4)
    np.testing.assert_allclose(got.loc[1, "R_positive_sum"], 0.5)
    np.testing.assert_allclose(got.loc[1, "R_negative_sum"], -0.1)
    np.testing.assert_allclose(got.loc[1, "R_abs_sum"], 0.6)
    np.testing.assert_allclose(got.loc[1, "R_abs_near_0_2"], 0.3)
    np.testing.assert_allclose(got.loc[1, "R_abs_mid_2_5"], 0.1)
    np.testing.assert_allclose(got.loc[1, "R_abs_far_5_10"], 0.2)
    np.testing.assert_allclose(got.loc[1, "top_abs_fraction"], 0.5)


def test_quantile_edges_are_open_at_both_ends():
    edges = quantile_edges(np.arange(100.0))
    assert np.isneginf(edges[0])
    assert np.isposinf(edges[-1])
    assert len(edges) == 6


def test_bin_contributions_sum_to_global_gap():
    rows = []
    for case in range(4):
        for value in range(10):
            rows.append({
                "case": case, "feature": float(value), "gap": value - 4.5,
                "R_model_sum": value + 1.0, "R_coherent_truth": 5.5,
            })
    frame = pd.DataFrame(rows)
    out = summarize_bins(frame, "feature", quantile_edges(frame.feature.to_numpy()))
    contribution = sum(x["contribution_to_global_gap"]["mean"] for x in out["bins"])
    np.testing.assert_allclose(
        contribution, frame.groupby("case").gap.mean().mean(), atol=1e-14,
    )
