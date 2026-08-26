from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from sbsi.flow import FlowTrainingConfig
from sbsi.flow_response_target import estimate_response_matrix
from sbsi.flow_response_audit import _forward_model_delta, _mean_model_matrix, _shear_state_masks
from sbsi.flow_training import _fold_catalogue_shear_into_shape
from sbsi.flow_training import epoch_response
from sbsi.shear_map import apply_shear_to_ellipticity
from sbsi.training import split_data


def test_grouped_case_split_keeps_shear_legs_together():
    frame = pd.DataFrame(
        {
            "case": np.repeat(np.arange(20), 4),
            "leg": np.tile([0, 0, 1, 1], 20),
        }
    )
    train, validation = split_data(frame, seed=501, validation_size=0.2, group_column="case")
    assert len(set(train["case"]) & set(validation["case"])) == 0
    assert len(set(train["case"])) == 16
    assert len(set(validation["case"])) == 4
    assert set(train["leg"]) == {0, 1}
    assert set(validation["leg"]) == {0, 1}


def test_catalogue_shear_is_folded_once_and_gamma_zeroed():
    frame = pd.DataFrame(
        {
            "e1_input_rot0_p": [0.1, -0.2],
            "e2_input_rot0_p": [0.2, 0.1],
            "gamma1_input_p": [0.05, 0.0],
            "gamma2_input_p": [0.0, -0.05],
        }
    )
    expected = apply_shear_to_ellipticity(
        frame["e1_input_rot0_p"],
        frame["e2_input_rot0_p"],
        frame["gamma1_input_p"],
        frame["gamma2_input_p"],
    )
    folded = _fold_catalogue_shear_into_shape(frame.copy())
    np.testing.assert_allclose(folded["e1_input_rot0_p"], expected[0])
    np.testing.assert_allclose(folded["e2_input_rot0_p"], expected[1])
    np.testing.assert_array_equal(folded["gamma1_input_p"], 0.0)
    np.testing.assert_array_equal(folded["gamma2_input_p"], 0.0)


def test_full_response_matrix_estimator_recovers_cross_terms():
    angle = np.linspace(0.0, 2.0 * np.pi, 2000, endpoint=False)
    applied = 0.05 * np.column_stack([np.cos(angle), np.sin(angle)])
    truth = np.array([[0.82, 0.03], [-0.02, 0.79]])
    delta = applied @ truth.T
    np.testing.assert_allclose(estimate_response_matrix(delta, applied), truth, atol=1e-12)


def test_audit_model_matrix_uses_measured_by_applied_orientation():
    class ShapePreprocessor:
        def transform_frame(self, frame):
            return frame[["e1_input_p", "e2_input_p"]].to_numpy(np.float32)

    class IdentityMean(torch.nn.Module):
        def _shift(self, context):
            return context

    count = 5
    frame = pd.DataFrame(
        {
            "e1_input_rot0_p": np.zeros(count),
            "e2_input_rot0_p": np.zeros(count),
            "gamma1_input_p": np.zeros(count),
            "gamma2_input_p": np.zeros(count),
            "Re_input_p": np.ones(count),
            "Re_input_s": np.ones(count),
            "r_input_p": np.full(count, 22.0),
            "r_input_s": np.full(count, 23.0),
            "distance": np.ones(count),
        }
    )
    bundle = SimpleNamespace(
        metadata={"condition_features": ["e1_input_p", "e2_input_p"]},
        condition_preprocessor=ShapePreprocessor(),
        target_transform=SimpleNamespace(scales=np.array([2.0, 3.0])),
        model=IdentityMean(),
        device=torch.device("cpu"),
    )
    matrix = _mean_model_matrix(bundle, frame, delta=0.02, batch_size=2)
    np.testing.assert_allclose(matrix, [[2.0, 0.0], [0.0, 3.0]], atol=1e-6)


def test_primary_only_mask_treats_missing_secondary_as_unsheared():
    frame = pd.DataFrame(
        {
            "gamma1_input_p": [0.05, 0.05, 0.0, 0.05],
            "gamma2_input_p": [0.0, 0.0, 0.0, 0.0],
            "gamma1_input_s": [0.0, 0.03, 0.0, np.nan],
            "gamma2_input_s": [0.0, 0.04, 0.0, np.nan],
        }
    )
    primary, secondary = _shear_state_masks(frame)
    np.testing.assert_array_equal(primary, [True, True, False, True])
    np.testing.assert_array_equal(secondary, [False, True, False, False])


def test_forward_model_delta_uses_zero_to_positive_shear():
    class ShapePreprocessor:
        def transform_frame(self, frame):
            return frame[["e1_input_p", "e2_input_p"]].to_numpy(np.float32)

    class IdentityMean(torch.nn.Module):
        def _shift(self, context):
            return context

    count = 8
    frame = pd.DataFrame(
        {
            "e1_input_rot0_p": np.zeros(count),
            "e2_input_rot0_p": np.zeros(count),
            "gamma1_input_p": np.zeros(count),
            "gamma2_input_p": np.zeros(count),
            "Re_input_p": np.ones(count),
            "Re_input_s": np.ones(count),
            "r_input_p": np.full(count, 22.0),
            "r_input_s": np.full(count, 23.0),
            "distance": np.ones(count),
        }
    )
    angle = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    applied = 0.05 * np.column_stack([np.cos(angle), np.sin(angle)])
    bundle = SimpleNamespace(
        metadata={"condition_features": ["e1_input_p", "e2_input_p"]},
        condition_preprocessor=ShapePreprocessor(),
        target_transform=SimpleNamespace(scales=np.array([2.0, 3.0])),
        model=IdentityMean(),
        device=torch.device("cpu"),
    )
    delta = _forward_model_delta(bundle, frame, applied, batch_size=3)
    np.testing.assert_allclose(delta, applied * [2.0, 3.0], atol=1e-6)


def test_matrix_regularizer_reads_diagonals_and_cross_terms():
    class IdentityMean(torch.nn.Module):
        def train(self, mode=True):
            return super().train(mode)

        def log_prob(self, target, context):
            return torch.zeros(len(target), dtype=target.dtype, device=target.device)

        def _shift(self, context, residual=None):
            return context

    delta = 0.02
    response = torch.tensor([[0.82, 0.03], [-0.02, 0.79]], dtype=torch.float32)
    count = 32
    zero = torch.zeros((count, 2))
    plus1 = delta * response[:, 0].repeat(count, 1)
    minus1 = -plus1
    plus2 = delta * response[:, 1].repeat(count, 1)
    minus2 = -plus2
    tensors = (
        torch.zeros((count, 2)),
        zero,
        torch.ones(count),
        zero,
        plus1,
        plus2,
        minus1,
        minus2,
        torch.zeros(count, dtype=torch.long),
    )
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*tensors), batch_size=count)
    _, matching_loss, trace, *_ = epoch_response(
        IdentityMean(),
        loader,
        torch.device("cpu"),
        (1.0, 1.0),
        delta,
        response.reshape(1, 4),
        1.0,
        response_difference="central",
        response_components="matrix",
    )
    assert matching_loss < 1e-12
    assert np.isclose(trace, 0.5 * (response[0, 0] + response[1, 1]))

    wrong_target = response.clone()
    wrong_target[0, 1] = 0.0
    _, cross_loss, *_ = epoch_response(
        IdentityMean(),
        loader,
        torch.device("cpu"),
        (1.0, 1.0),
        delta,
        wrong_target.reshape(1, 4),
        1.0,
        response_difference="central",
        response_components="matrix",
    )
    assert cross_loss > 0.0


def test_mixed_flow_config_renders_balanced_grouped_arguments(tmp_path):
    g0 = tmp_path / "g0.feather"
    g05 = tmp_path / "g05.feather"
    output = tmp_path / "model.pt"
    g0.touch()
    g05.touch()
    config = FlowTrainingConfig(
        catalogue=g0,
        additional_catalogues=(g05,),
        output=output,
        response_weight=0.0,
        coupling_weight=0.0,
        shear_case=None,
        fold_catalogue_shear=True,
        validation_group_column="case",
        validation_size=0.2,
        expected_case_count=200,
    )
    config.validate()
    argv = config.to_argv()
    assert argv[argv.index("--additional-catalogue") + 1] == str(g05)
    assert "--fold-catalogue-shear" in argv
    assert argv[argv.index("--validation-group-column") + 1] == "case"
    assert "--shear-case" not in argv
    assert "--response-weight" not in argv
