import numpy as np
import pandas as pd

from scripts.build_v22_constgold_pair_features import summarize_pair_features


def test_summarize_pair_features_tracks_stable_dominant_pair_and_ranks():
    field = pd.DataFrame({
        "index": [1, 2, 3, 4],
        "r": [24.0, 23.0, 25.0, 22.0],
        "Re": [0.8, 0.6, 0.7, 1.0],
        "sersic_n": [1.0, 2.0, 3.0, 4.0],
    })
    pairs = pd.DataFrame({
        "index_p": [1, 1, 1, 2],
        "index_s": [2, 3, 4, 3],
        "response": [0.2, -0.05, 0.2, -0.1],
        "distance": [4.0, 1.0, 3.0, 2.5],
    })
    got = summarize_pair_features(pairs, field).set_index("input_index")

    # Tied maxima are broken by the smaller secondary index.
    assert got.loc[1, "secondary_index"] == 2
    assert got.loc[1, "dominant_response"] == 0.2
    assert got.loc[1, "runner_up_abs_response"] == 0.2
    assert got.loc[1, "dominant_flux_rank"] == 2
    assert got.loc[1, "dominant_distance_rank"] == 3
    assert got.loc[1, "n_response_pairs_ge_10pct_max"] == 3
    np.testing.assert_allclose(got.loc[1, "R_blend"], 0.35)
    np.testing.assert_allclose(got.loc[1, "R_abs_sum"], 0.45)
    np.testing.assert_allclose(got.loc[1, "top_abs_fraction"], 0.2 / 0.45)
    np.testing.assert_allclose(got.loc[1, "R_abs_near_0_2"], 0.05)
    np.testing.assert_allclose(got.loc[1, "R_abs_mid_2_5"], 0.4)
    np.testing.assert_allclose(got.loc[1, "mean_pair_distance"], 8.0 / 3.0)
    np.testing.assert_allclose(
        got.loc[1, "std_pair_distance"], np.std([4.0, 1.0, 3.0], ddof=1)
    )
    assert got.loc[1, "n_pair_d1_2"] == 1
    assert got.loc[1, "n_pair_d3_5"] == 2
    np.testing.assert_allclose(got.loc[1, "R_pair_d1_2"], -0.05)
    np.testing.assert_allclose(got.loc[1, "R_pair_d3_5"], 0.4)
    np.testing.assert_allclose(got.loc[1, "R_abs_pair_d1_2"], 0.05)
    np.testing.assert_allclose(got.loc[1, "R_abs_pair_d3_5"], 0.4)

    # A one-pair primary has an explicit runner-up flag and zero response.
    assert not got.loc[2, "has_runner_up"]
    assert got.loc[2, "runner_up_abs_response"] == 0.0
    assert np.isnan(got.loc[2, "runner_up_distance"])
    assert got.loc[2, "std_pair_distance"] == 0.0
