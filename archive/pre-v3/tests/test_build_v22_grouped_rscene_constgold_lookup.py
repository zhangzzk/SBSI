import numpy as np

from scripts.build_v22_grouped_rscene_constgold_lookup import (
    aggregate_pair_corrections,
    scene_context,
)


def test_scene_context_matches_training_float32_convention() -> None:
    primary = np.asarray([1, 1, 1, 4, 4, 9], dtype=np.int64)
    prediction = np.asarray(
        [0.1, 0.2, -0.05, 0.4, -0.1, 0.7], dtype=np.float64
    )
    starts, counts, scene_rows, count_rows = scene_context(primary, prediction)
    np.testing.assert_array_equal(starts, [0, 3, 5])
    np.testing.assert_array_equal(counts, [3, 2, 1])
    np.testing.assert_allclose(scene_rows, [0.25, 0.25, 0.25, 0.3, 0.3, 0.7])
    np.testing.assert_array_equal(count_rows, [3, 3, 3, 2, 2, 1])
    assert scene_rows.dtype == np.float32
    assert count_rows.dtype == np.uint8


def test_aggregate_pair_corrections_sums_each_scene_once() -> None:
    got = aggregate_pair_corrections(
        primary=np.asarray([1, 1, 2, 2, 2]),
        secondary=np.asarray([3, 4, 1, 3, 4]),
        pair_prediction=np.asarray([0.1, -0.03, 0.2, 0.04, -0.01]),
        pair_correction=np.asarray([0.01, 0.02, -0.03, 0.04, 0.01]),
    ).set_index("input_index")
    assert got.loc[1, "n_pairs"] == 2
    assert got.loc[2, "n_pairs"] == 3
    np.testing.assert_allclose(got.loc[1, "R_blend_v22"], 0.07)
    np.testing.assert_allclose(got.loc[1, "grouped_pair_correction_sum"], 0.03)
    np.testing.assert_allclose(got.loc[1, "R_blend"], 0.10)
    np.testing.assert_allclose(got.loc[2, "R_blend_v22"], 0.23)
    np.testing.assert_allclose(got.loc[2, "grouped_pair_correction_sum"], 0.02)
    np.testing.assert_allclose(got.loc[2, "R_blend"], 0.25)
