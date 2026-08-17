import numpy as np
import pandas as pd

from scripts.fit_v22_rscene_pair_mean_calibration import (
    aggregate_contiguous_scenes,
    assign_bins,
    crossfit_bin_correction,
    fit_bin_correction,
    make_scene_edges,
    summarize_scenes,
)


def test_bin_correction_exactly_closes_pooled_pair_mean():
    scene_prediction = np.array([-0.1, -0.05, 0.12, 0.30])
    pair_count = np.array([2, 1, 3, 2])
    residual_sum = np.array([0.04, -0.01, 0.09, 0.02])
    edges = np.array([-np.inf, 0.0, np.inf])
    correction, bins = fit_bin_correction(
        scene_prediction, pair_count, residual_sum, edges
    )
    assert np.allclose(correction, [0.01, 0.022])
    for index in range(2):
        keep = bins == index
        assert np.isclose(
            residual_sum[keep].sum() - correction[index] * pair_count[keep].sum(),
            0.0,
        )


def test_scene_edges_are_label_independent_and_include_forced_boundaries():
    prediction = np.linspace(-0.2, 0.4, 101)
    edges = make_scene_edges(prediction, 5, forced_edges=(0.1, 0.2))
    assert np.isneginf(edges[0])
    assert np.isposinf(edges[-1])
    assert 0.1 in edges
    assert 0.2 in edges
    assert len(edges) == 6
    bins = assign_bins(np.array([0.099, 0.1, 0.199, 0.2]), edges)
    assert bins[0] + 1 == bins[1]
    assert bins[2] + 1 == bins[3]


def test_contiguous_scene_aggregation_and_per_pair_scene_shift():
    case = np.array([0, 0, 0, 1])
    input_index = np.array([10, 10, 20, 30])
    label = np.array([0.2, 0.3, -0.1, 0.4])
    prediction = np.array([0.1, 0.15, -0.05, 0.3])
    scenes = aggregate_contiguous_scenes(case, input_index, label, prediction)
    assert np.array_equal(scenes["n_pairs"], [2, 1, 1])
    assert np.allclose(scenes["label_sum"], [0.5, -0.1, 0.4])
    correction = np.array([0.01, 0.02, -0.03])
    summary = summarize_scenes(
        scenes["case"], scenes["label_sum"], scenes["prediction_sum"],
        scenes["n_pairs"], correction, 1.0, np.ones(3, dtype=bool),
    )
    expected_prediction = scenes["prediction_sum"] + scenes["n_pairs"] * correction
    assert np.isclose(summary["prediction"]["mean"], np.mean([
        expected_prediction[:2].mean(), expected_prediction[2]
    ]))


def test_case_crossfit_never_uses_heldout_case_residual():
    scenes = pd.DataFrame({
        "case": [0, 0, 1, 1, 2, 2, 3, 3],
        "scene_prediction": [-1.0, 1.0] * 4,
        "validation_pair_count": [1] * 8,
        "validation_residual_sum": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    })
    edges = np.array([-np.inf, 0.0, np.inf])
    fold_by_case = np.array([0, 1, 2, 3])
    oof, models = crossfit_bin_correction(scenes, edges, fold_by_case)
    assert models.shape == (4, 2)
    # Case zero sees only cases 1--3: negative/positive means are 5 and 6.
    assert np.allclose(oof[:2], [5.0, 6.0])
