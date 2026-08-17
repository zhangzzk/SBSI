import numpy as np
import pandas as pd

from scripts.build_anchor_bias_features import derive_features, pair_scene_summary


def test_pair_scene_summary_closes_fixed_shells():
    pairs = pd.DataFrame({
        "anchor_index": [1, 1, 1, 2],
        "secondary_index": [10, 11, 12, 20],
        "distance": [0.5, 2.5, 8.0, 4.0],
        "response": [0.2, -0.1, 0.4, -0.3],
    })
    got = pair_scene_summary(pairs, pd.Index([1, 2], name="input_index"))
    one = got.set_index("input_index").loc[1]
    assert one.pair_count_replay == 3
    assert np.isclose(one.R_pair_replay, 0.5)
    assert np.isclose(one.R_abs_pair_replay, 0.7)
    assert one.n_pair_d0_1 == 1
    assert one.n_pair_d2_3 == 1
    assert one.n_pair_d7_10 == 1
    assert sum(one[f"n_pair_{name}"] for name in (
        "d0_1", "d1_2", "d2_3", "d3_5", "d5_7", "d7_10"
    )) == 3


def test_derive_features_uses_truth_only_in_target():
    row = {
        "primary_size": 1.0, "primary_sersic_n": 2.0,
        "dominant_secondary_size": 2.0, "dominant_secondary_sersic_n": 3.0,
        "dominant_distance": 3.0, "dominant_flux_ratio_primary": 100.0,
        "dominant_size_ratio_primary": 2.0, "dominant_abs_response": 0.2,
        "dominant_to_runner_up_abs_response": 10.0, "primary_flux": 2.0,
        "neighbour_flux_sum": 20.0, "primary_mag": 23.0,
        "has_deployed_pair": 1,
        "dominant_secondary_mag": 18.0, "dominant_flux_share_neighbours": 0.5,
        "dominant_flux_rank": 1, "dominant_distance_rank": 2, "n_pairs": 3,
        "R_blend_lsst_r_extnbr_v22": 0.4, "response": 0.2,
        "runner_up_abs_response": 0.02, "other_abs_response_sum": 0.1,
        "R_abs_sum": 0.5, "top_abs_fraction": 0.4,
        "n_response_pairs_ge_10pct_max": 2, "R_blend_truth": 0.7,
        "min_pair_distance": 1.0, "mean_pair_distance": 3.0,
        "std_pair_distance": 1.0,
    }
    for name in ("d0_1", "d1_2", "d2_3", "d3_5", "d5_7", "d7_10"):
        row[f"n_pair_{name}"] = 0
        row[f"R_pair_{name}"] = 0.0
        row[f"R_abs_pair_{name}"] = 0.0
    got = derive_features(pd.DataFrame([row])).iloc[0]
    assert np.isclose(got.bias_truth_minus_model, 0.3)
    assert np.isclose(got.log10_flux_ratio, 2.0)
    assert np.isclose(got.log10_overlap_scale, 0.0)
    assert np.isclose(
        got.log10_surface_brightness_ratio,
        2.0 - 2.0 * np.log10(2.0),
    )
    assert np.isclose(got.signed_to_abs_response, 0.8)
    assert np.isclose(got.cancelled_abs_response_fraction, 0.2)
