import pytest

import numpy as np

from scripts.eval_v21_rblend_scale import (
    isotonic_transform,
    validate_additive_calibration,
    validate_calibration,
)


def calibration(**updates):
    value = {
        "tag": "lsst_r_extnbr_v21",
        "dev_scale": 1.07,
        "test_residual": 0.002,
        "test_residual_se": 0.002,
    }
    value.update(updates)
    return value


def test_calibration_gate_accepts_independent_consistent_scale():
    assert validate_calibration(calibration(), 0.006, 2.0) == 1.07


def test_calibration_gate_uses_post_validation_all_case_refit():
    assert validate_calibration(
        calibration(deployment_scale=1.09), 0.006, 2.0
    ) == 1.09


def test_calibration_gate_accepts_explicit_matching_tag():
    value = calibration(tag="lsst_r_extnbr_v22")
    assert validate_calibration(
        value, 0.006, 2.0, expected_tag="lsst_r_extnbr_v22"
    ) == 1.07


@pytest.mark.parametrize(
    "updates",
    [
        {"tag": "wrong"},
        {"dev_scale": 2.1},
        {"test_residual": 0.007},
        {"test_residual": 0.005, "test_residual_se": 0.002},
    ],
)
def test_calibration_gate_rejects_invalid_or_failed_validation(updates):
    with pytest.raises(RuntimeError):
        validate_calibration(calibration(**updates), 0.006, 2.0)


def test_isotonic_transform_interpolates_and_clips():
    calibration = {
        "x_thresholds": np.array([0.0, 1.0, 2.0]),
        "y_thresholds": np.array([0.0, 2.0, 3.0]),
    }
    np.testing.assert_allclose(
        isotonic_transform([-1.0, 0.5, 3.0], calibration), [0.0, 1.0, 3.0]
    )


def test_additive_calibration_requires_heldout_gate():
    value = {
        "tag": "lsst_r_extnbr_v22",
        "method": "matched_outside10_additive_response",
        "gate_passed": True,
        "deployment_offset": 0.007,
        "test_residual": 0.001,
        "test_residual_se": 0.002,
    }
    assert validate_additive_calibration(
        value, 0.006, 2.0, "lsst_r_extnbr_v22",
    ) == 0.007
    with pytest.raises(RuntimeError, match="failed its held-out gate"):
        validate_additive_calibration(
            {**value, "gate_passed": False}, 0.006, 2.0, "lsst_r_extnbr_v22",
        )
