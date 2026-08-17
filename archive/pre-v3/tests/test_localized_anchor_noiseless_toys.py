import numpy as np
import pandas as pd

from scripts.analyze_localized_anchor_noiseless_toys import add_decomposition
from scripts.prepare_localized_anchor_noiseless_toys import select_population
from scripts.run_localized_anchor_noiseless_toy_case import (
    central_column,
    compose_arm_images,
    tangent_offsets,
)


def test_select_population_uses_scene_prediction_and_builds_compact_stratum():
    frame = pd.DataFrame({
        "case": [699, 700, 700, 701], "input_index": [1, 2, 3, 4],
        "scene_prediction": [0.9, 0.39, 0.5, 0.6],
        "bias_truth_minus_model": [99.0, 88.0, -77.0, 66.0],
        "R_blend_truth": [1.0, 1.0, 1.0, 1.0],
        "has_deployed_pair": [1, 1, 1, 1],
        "log10_dominant_secondary_size": np.log10([0.3, 0.3, 0.39, 0.41]),
        "log1p_n_pairs": np.log1p([2, 3, 4, 5]),
    })
    got = select_population(frame, 700, 701, 0.4, 0.4)
    assert got.input_index.tolist() == [3, 4]
    assert got.compact_dominant_secondary.tolist() == [True, False]
    assert got.n_deployed_pairs.tolist() == [4, 5]


def test_tangent_offsets_center_primary_and_replay_small_separation():
    frame = pd.DataFrame({
        "index": [10, 20], "RA": [180.0, 180.0 + 1.0 / 3600.0],
        "DEC": [0.0, 2.0 / 3600.0],
    })
    x, y = tangent_offsets(frame, 10)
    np.testing.assert_allclose([x[0], y[0]], 0.0, atol=1e-14)
    np.testing.assert_allclose([x[1], y[1]], [1.0, 2.0], atol=1e-9)


def test_compose_arm_images_keeps_other_neighbours_unsheared():
    z = lambda value: np.full((2, 2), value, dtype=float)
    sources = [
        {"zero": z(10)},
        {"zero": z(1), "g1_plus": z(2), "g1_minus": z(0),
         "g2_plus": z(3), "g2_minus": z(-1)},
        {"zero": z(4), "g1_plus": z(6), "g1_minus": z(2),
         "g2_plus": z(8), "g2_minus": z(0)},
    ]
    coherent, individual = compose_arm_images(sources)
    np.testing.assert_allclose(coherent["g1_plus"], z(18))
    np.testing.assert_allclose(individual[0]["g1_plus"], z(16))
    np.testing.assert_allclose(individual[1]["g1_plus"], z(17))


def test_central_column_and_decomposition_close():
    np.testing.assert_allclose(
        central_column(np.array([0.3, 0.1]), np.array([0.1, -0.1]), 0.1),
        [1.0, 1.0],
    )
    frame = pd.DataFrame({
        "R_model_sum": [0.1], "R_original_coherent_g1": [0.12],
        "original_truth_minus_model_g1": [0.02],
        "R11_coherent": [0.16], "R11_individual_sum": [0.13],
        "R_trace_coherent": [0.15], "R_trace_individual_sum": [0.14],
    })
    got = add_decomposition(frame)
    np.testing.assert_allclose(got.additivity_gap_g1, 0.03)
    np.testing.assert_allclose(got.individual_truth_minus_model_g1, 0.03)
    np.testing.assert_allclose(got.coherent_truth_minus_model_g1, 0.06)
    np.testing.assert_allclose(got.decomposition_replay_g1, 0.0)
    np.testing.assert_allclose(got.decomposition_replay_trace, 0.0)
