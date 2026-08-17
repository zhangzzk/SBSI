import numpy as np
import torch

from sbs_shear.scene_model import (
    DeepSetsConditioner,
    RADIAL_SCENE_NEIGHBOR_FEATURES,
    SCENE_GEOMETRY_NEIGHBOR_FEATURES,
    SetConditionedMeasurementFlow,
    SetFeatureStandardizer,
)
from sbs_shear.preprocessing import raw_columns_for_selection_features


def test_set_feature_standardizer_pads_and_masks_variable_neighbors():
    values = [
        np.array([[1.0, 2.0], [3.0, np.nan]], dtype=np.float32),
        np.empty((0, 2), dtype=np.float32),
    ]

    transform = SetFeatureStandardizer.fit(values, ["a", "b"])
    padded, mask, raw_counts, clipped_counts = transform.transform_padded(values, max_neighbors=1)

    assert padded.shape == (2, 1, 2)
    np.testing.assert_allclose(mask, [[1.0], [0.0]])
    np.testing.assert_array_equal(raw_counts, [2, 0])
    np.testing.assert_array_equal(clipped_counts, [1, 0])


def test_radial_geometry_mode_keeps_only_scalar_neighbor_geometry():
    radial = set(SCENE_GEOMETRY_NEIGHBOR_FEATURES["radial"])

    assert radial == set(RADIAL_SCENE_NEIGHBOR_FEATURES)
    assert "distance_scaled" in radial
    assert "e_abs_s" in radial
    assert "pair_pframe_cos2" not in radial
    assert "pair_pframe_sin2" not in radial
    assert "e_pframe_parallel_s" not in radial


def test_e_abs_s_requests_secondary_shape_columns():
    columns = raw_columns_for_selection_features(["e_abs_s"])

    assert {"axis_ratio_input_s", "position_angle_input_s"}.issubset(columns)


def test_deepsets_conditioner_ignores_padded_neighbor_values():
    torch.manual_seed(7)
    conditioner = DeepSetsConditioner(
        primary_dim=3,
        neighbor_dim=2,
        context_dim=4,
        hidden_dim=5,
        neighbor_layers=2,
        context_layers=2,
    )
    primary = torch.randn(1, 3)
    neighbors_a = torch.randn(1, 2, 2)
    neighbors_b = neighbors_a.clone()
    neighbors_b[:, 1, :] = torch.tensor([1000.0, -1000.0])
    mask = torch.tensor([[1.0, 0.0]])

    out_a = conditioner(primary, neighbors_a, mask)
    out_b = conditioner(primary, neighbors_b, mask)

    torch.testing.assert_close(out_a, out_b)


def test_set_conditioned_measurement_flow_log_prob_and_sample_shape():
    torch.manual_seed(11)
    model = SetConditionedMeasurementFlow(
        target_dim=4,
        primary_dim=3,
        neighbor_dim=2,
        context_dim=5,
        set_hidden_dim=6,
        flow_hidden_dim=8,
        flow_layers=1,
        n_flows=2,
    )
    target = torch.randn(3, 4)
    primary = torch.randn(3, 3)
    neighbors = torch.randn(3, 2, 2)
    mask = torch.tensor([[1.0, 1.0], [1.0, 0.0], [0.0, 0.0]])

    log_prob = model.log_prob(target, primary, neighbors, mask)
    samples = model.sample(primary, neighbors, mask, n_samples=2)

    assert torch.isfinite(log_prob).all()
    assert samples.shape == (3, 2, 4)
