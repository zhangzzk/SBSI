import numpy as np

from scripts.evaluate_v22_oof_learning import (
    group_structure,
    reduce_group,
    row_summary,
    scene_summaries,
)
from scripts.prepare_v22_oof_learning_cache import fold_assignment
from scripts.train_v22_oof_learning import (
    capped_mean_one_inverse,
    conditional_matrix,
)


def test_case_folds_are_balanced_and_exclude_external_cases():
    first = fold_assignment()
    second = fold_assignment()
    assert np.array_equal(first, second)
    assert np.all(first[:40] == -1)
    assert np.array_equal(
        np.bincount(first[40:].astype(int), minlength=4),
        np.full(4, 40),
    )


def test_conditional_matrix_adds_prediction_and_pair_ratios():
    scaled = np.arange(21, dtype=np.float32).reshape(3, 7)
    raw = np.asarray(
        [
            [0.5, 0.25, 24.0, 25.0, 1.0, 2.0, 1.5],
            [0.6, 1.2, 25.0, 23.0, 2.0, 1.0, 2.5],
            [0.8, 0.8, 22.0, 22.0, 1.5, 1.5, 3.5],
        ],
        dtype=np.float32,
    )
    prediction = np.asarray([0.1, -0.2, 0.3], dtype=np.float32)
    index = np.asarray([2, 0], dtype=np.int32)
    result = conditional_matrix(scaled, raw, prediction, index)
    assert result.shape == (2, 10)
    assert np.array_equal(result[:, :7], scaled[index])
    assert np.allclose(result[:, 7], prediction[index])
    assert np.allclose(result[:, 8], -0.4 * (raw[index, 3] - raw[index, 2]))
    assert np.allclose(result[:, 9], np.log10(raw[index, 1] / raw[index, 0]))


def test_stabilized_inverse_variance_weights_are_capped_and_mean_one():
    variance = np.geomspace(1.0e-3, 10.0, 10001)
    weights, summary = capped_mean_one_inverse(variance)
    assert np.isfinite(weights).all()
    assert weights.min() >= 0.25
    assert weights.max() <= 4.0
    assert np.isclose(weights.astype(float).mean(), 1.0, atol=2.0e-6)
    assert np.isclose(summary["weight_mean"], 1.0, atol=2.0e-6)


def test_group_reduction_handles_unsorted_pair_rows():
    case = np.asarray([1, 0, 1, 0, 0], dtype=np.int16)
    primary = np.asarray([4, 2, 4, 3, 2], dtype=np.int64)
    value = np.asarray([3.0, 1.0, 5.0, 7.0, 2.0])
    order, starts, group_case = group_structure(case, primary)
    reduced = reduce_group(value, order, starts)
    assert np.array_equal(group_case, [0, 0, 1])
    assert np.allclose(reduced, [3.0, 7.0, 8.0])


def test_row_summary_uses_paired_case_mse_difference():
    case = np.repeat(np.arange(2), 2).astype(np.int16)
    label = np.asarray([0.0, 2.0, 1.0, 3.0])
    baseline = np.zeros(4)
    prediction = np.ones(4)
    result = row_summary(case, label, prediction, baseline, 0, 1)
    # Baseline MSEs are 2 and 5; candidate MSEs are 1 and 2.
    assert np.isclose(result["mse"]["mean"], 1.5)
    assert np.isclose(result["mse_minus_baseline"]["mean"], -2.0)


def test_scene_summary_reconstructs_exact_two_component_response():
    case = np.repeat(np.arange(40, dtype=np.int16), 2)
    primary = np.repeat(np.arange(40, dtype=np.int64) + 100, 2)
    angle = np.tile(np.asarray([0.0, 45.0]), 40)
    # y=(2,3). At spin-2 directions (1,0) and (0,1), the stored
    # parallel/null coordinates are (2,3) and (3,-2).
    label = np.tile(np.asarray([2.0, 3.0]), 40)
    null = np.tile(np.asarray([3.0, -2.0]), 40)
    prediction = np.tile(np.asarray([2.0, 3.0]), 40)
    payload, table = scene_summaries(
        case,
        primary,
        angle,
        label,
        null,
        {"baseline": prediction},
    )
    vector = payload["models"]["baseline"]["vector_closure"]
    assert np.isclose(vector["all_c0_39"]["mean"], 1.0)
    tail = payload["models"]["baseline"]["scene_scalar"][
        "half_scene_gt_0p05"
    ]
    assert np.isclose(tail["full_equivalent_label_minus_prediction"]["mean"], 0.0)
    assert set(table.selection) == {
        "all",
        "half_scene_gt_0p05",
        "half_scene_gt_0p10",
    }
