import numpy as np
import pandas as pd

from scripts.apply_anchor_bias_emulator_constgold import (
    block_summary,
    finite_stat,
    prepare_emulator_features,
)
from scripts.build_anchor_bias_features import FULL_FEATURES
from scripts.build_v22_constgold_pair_features import summarize_pair_features


def small_pair_summary() -> pd.DataFrame:
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
    result = summarize_pair_features(pairs, field)
    result.insert(0, "case", 40)
    return result


def test_prepare_emulator_features_matches_anchor_schema_exactly():
    got = prepare_emulator_features(small_pair_summary()).set_index("input_index")
    assert list(got.columns) == ["case", *FULL_FEATURES]
    np.testing.assert_allclose(got.loc[1, "scene_prediction"], 0.35)
    np.testing.assert_allclose(got.loc[1, "dominant_response"], 0.2)
    np.testing.assert_allclose(got.loc[1, "other_abs_response_sum"], 0.25)
    np.testing.assert_allclose(got.loc[1, "R_pair_d1_2"], -0.05)
    np.testing.assert_allclose(got.loc[1, "R_pair_d3_5"], 0.4)
    np.testing.assert_allclose(got.loc[1, "log10_primary_size"], np.log10(0.8))
    np.testing.assert_allclose(
        got.loc[1, "log10_overlap_scale"], np.log10((0.8 + 0.6) / 4.0)
    )


def test_block_summary_reports_exact_additive_recovery():
    cases = pd.DataFrame({
        "case": [1, 2, 3, 4],
        "n_rows": [10, 10, 10, 10],
        "R_sim": [3.0, 4.0, 5.0, 6.0],
        "R_flow": [1.0, 1.0, 1.0, 1.0],
        "R_blend": [1.0, 1.0, 1.0, 1.0],
        "R_model": [2.0, 2.0, 2.0, 2.0],
        "raw_gap": [1.0, 2.0, 3.0, 4.0],
        "correction": [0.5, 1.0, 1.5, 2.0],
    })
    got = block_summary(
        cases, np.ones(len(cases), dtype=bool), {"toy": "correction"}
    )
    correction = got["corrections"]["toy"]
    np.testing.assert_allclose(correction["recovered_fraction"]["value"], 0.5)
    np.testing.assert_allclose(correction["remaining_fraction"]["value"], 0.5)
    np.testing.assert_allclose(correction["remaining_gap"]["mean"], 1.25)
    np.testing.assert_allclose(
        got["raw_m_percent_case_distribution"]["mean"], 125.0
    )
    expected_corrected_m = np.mean(100.0 * (
        cases.R_sim.to_numpy() / (cases.R_model + cases.correction).to_numpy()
        - 1.0
    ))
    np.testing.assert_allclose(
        correction["corrected_m_percent_case_distribution"]["mean"],
        expected_corrected_m,
    )


def test_finite_stat_keeps_frozen_constant_strict_json_safe():
    got = finite_stat(np.full(10, 0.125))
    assert got["case_sem"] == 0.0
    assert got["t"] is None
    assert got["p"] == 0.0
