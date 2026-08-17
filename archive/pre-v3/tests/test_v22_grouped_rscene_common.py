import numpy as np

from scripts.v22_grouped_rscene_common import (
    CONDITIONAL_FEATURES,
    assign_scene_bins,
    conditional_matrix,
    grouped_gradient_hessian,
    grouped_loss_value,
    make_scene_edges,
    scene_edges_from_metadata,
)


def test_grouped_gradient_matches_finite_difference() -> None:
    prediction = np.asarray([0.2, -0.1, 0.4, 0.5, -0.2], dtype=float)
    target = np.asarray([0.0, 0.3, -0.2, 0.1, -0.4], dtype=float)
    bins = np.asarray([0, 0, 1, 1, 1], dtype=np.uint8)
    strength = 3.5
    gradient, hessian = grouped_gradient_hessian(
        prediction, target, bins, strength
    )
    epsilon = 1.0e-6
    numeric = np.empty(len(prediction))
    for index in range(len(prediction)):
        plus = prediction.copy()
        minus = prediction.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        numeric[index] = (
            grouped_loss_value(plus, target, bins, strength)
            - grouped_loss_value(minus, target, bins, strength)
        ) / (2.0 * epsilon)
    # The XGBoost gradient is multiplied by the harmless common factor N.
    np.testing.assert_allclose(gradient / len(prediction), numeric, atol=2.0e-7)
    assert np.all(hessian > 0.0)


def test_group_block_hessian_gives_exact_bin_leaf_newton_step() -> None:
    prediction = np.asarray([0.3, 0.1, -0.2, 0.4], dtype=float)
    target = np.zeros(4)
    bins = np.asarray([0, 0, 1, 1], dtype=np.uint8)
    gradient, hessian = grouped_gradient_hessian(
        prediction, target, bins, strength=10.0
    )
    for bin_index in (0, 1):
        local = bins == bin_index
        step = -float(gradient[local].sum() / hessian[local].sum())
        np.testing.assert_allclose(step, -prediction[local].mean(), atol=1.0e-7)


def test_conditional_matrix_includes_pair_ratios_and_scene_context() -> None:
    scaled = np.arange(21, dtype=np.float32).reshape(3, 7)
    raw = np.asarray([
        [0.5, 0.25, 23.0, 24.0, 1.0, 2.0, 1.5],
        [0.6, 0.90, 24.0, 22.0, 2.0, 3.0, 2.5],
        [1.0, 1.00, 25.0, 25.0, 3.0, 4.0, 3.5],
    ], dtype=np.float32)
    pair = np.asarray([0.01, -0.02, 0.03], dtype=np.float32)
    scene = np.asarray([0.2, 0.2, -0.1], dtype=np.float32)
    count = np.asarray([2, 2, 1], dtype=np.uint8)
    output = conditional_matrix(scaled, raw, pair, scene, count, slice(None))
    assert output.shape == (3, len(CONDITIONAL_FEATURES))
    np.testing.assert_array_equal(output[:, :7], scaled)
    np.testing.assert_allclose(output[:, 7], pair)
    np.testing.assert_allclose(output[:, 8], [-0.4, 0.8, 0.0])
    np.testing.assert_allclose(
        output[:, 9], np.log10([0.5, 1.5, 1.0]), atol=5.0e-8
    )
    np.testing.assert_allclose(output[:, 10], scene)
    np.testing.assert_allclose(output[:, 11], np.log1p(count.astype(float)))


def test_scene_edges_keep_physical_boundaries_without_sliver_bins() -> None:
    values = np.linspace(-0.2, 0.4, 10_001)
    edges = make_scene_edges(values, n_quantile_bins=10, forced_edges=(0.1, 0.2))
    assert len(edges) == 11
    assert 0.1 in edges and 0.2 in edges
    bins = assign_scene_bins(np.asarray([-0.1, 0.1, 0.2, 0.3]), edges)
    assert np.all(np.diff(bins.astype(int)) >= 0)


def test_scene_edges_restore_strict_json_infinity_sentinels() -> None:
    metadata = {
        "scene_bin_definition": {"edges": [None, -0.1, 0.1, None]}
    }
    edges = scene_edges_from_metadata(metadata)
    np.testing.assert_array_equal(edges[1:-1], [-0.1, 0.1])
    assert np.isneginf(edges[0]) and np.isposinf(edges[-1])
