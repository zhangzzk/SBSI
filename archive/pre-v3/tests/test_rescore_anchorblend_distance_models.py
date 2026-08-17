import numpy as np
import pandas as pd

from scripts.rescore_anchorblend_distance_models import aggregate_pairs


def test_aggregate_pairs_fills_isolated_and_shells():
    pairs = pd.DataFrame({
        "index_input_p": [10, 10, 11],
        "distance": [0.25, 1.25, 7.5],
        "R_blend_a": [1.0, 2.0, 4.0],
        "R_blend_b": [-1.0, 3.0, 5.0],
    })
    got = aggregate_pairs(pairs, [10, 11, 12], ["R_blend_a", "R_blend_b"]).set_index(
        "input_index"
    )
    assert got.loc[10, "n_pairs"] == 2
    assert got.loc[12, "n_pairs"] == 0
    assert got.loc[10, "R_blend_a"] == 3.0
    assert got.loc[12, "R_blend_b"] == 0.0
    assert got.loc[10, "n_d0_0p5"] == 1
    assert got.loc[10, "R_blend_a_d1_2"] == 2.0
    assert got.loc[11, "R_blend_b_d7_10"] == 5.0
    assert np.isfinite(got.to_numpy(float)).all()
