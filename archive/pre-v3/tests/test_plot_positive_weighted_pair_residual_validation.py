import numpy as np
import pytest

from scripts.plot_positive_weighted_pair_residual_validation import (
    flatten,
    validate_reference,
)


def calibration_summary():
    return {
        "case_window": [20, 39],
        "n_pairs": 100,
        "global_pair_label": {"mean": 0.7},
        "global_pair_prediction": {"mean": 0.6},
        "global_pair_label_minus_prediction": {
            "mean": 0.1,
            "case_sem": 0.02,
        },
    }


def test_reference_guard_closes_all_validation_quantities():
    reference = {
        "models": {
            "amp_0.050": {
                "validation_c20_39": {
                    "case_window": [20, 39],
                    "n_rows": 100,
                    "pooled_label_mean": 0.7,
                    "pooled_prediction_mean": 0.6,
                    "residual_mean": {"mean": 0.1, "case_sem": 0.02},
                }
            }
        }
    }
    checks = validate_reference(
        calibration_summary(), reference, "amp_0.050"
    )
    assert checks and all(checks.values())

    bad = calibration_summary()
    bad["global_pair_label_minus_prediction"]["mean"] = 0.11
    with pytest.raises(RuntimeError, match="reference closure failed"):
        validate_reference(bad, reference, "amp_0.050")


def test_flatten_preserves_model_specific_bins():
    def model(alpha, tag, prediction):
        return {
            "alpha": alpha,
            "tag": tag,
            "bins": [{
                "bin": 0,
                "lower": -1.0,
                "upper": 1.0,
                "n_pairs": 8,
                "pair_fraction": 1.0,
                "prediction": {"mean": prediction, "case_sem": 0.01},
                "label": {"mean": 0.2, "case_sem": 0.02},
                "label_minus_prediction": {
                    "mean": 0.2 - prediction,
                    "case_sem": 0.03,
                },
            }],
        }

    table = flatten([
        model(0.05, "a005", 0.1),
        model(0.10, "a010", 0.3),
    ])
    assert table.tag.tolist() == ["a005", "a010"]
    np.testing.assert_allclose(table.prediction_mean, [0.1, 0.3])
    np.testing.assert_allclose(table.residual_mean, [0.1, -0.1])
