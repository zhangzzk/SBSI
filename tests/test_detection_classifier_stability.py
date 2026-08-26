import numpy as np

from _script_loader import load_script_module


MODULE = load_script_module("evaluate_detection_classifier_shear_stability.py")


def _response(truth, model):
    return {
        "truth_per_case": truth,
        "model_per_case": model,
    }


def test_summarize_response_uses_paired_case_residuals():
    records = [
        {
            "fixed_intrinsic": {"truth": -0.02, "model": -0.01},
            "rendered_per_leg": {"truth": -0.03, "model": -0.04},
        },
        {
            "fixed_intrinsic": {"truth": -0.04, "model": -0.05},
            "rendered_per_leg": {"truth": -0.01, "model": -0.02},
        },
    ]
    result = MODULE.summarize_response(records)
    assert np.isclose(result["fixed_intrinsic"]["truth"], -0.03)
    assert np.isclose(result["fixed_intrinsic"]["model_minus_truth"], 0.0)
    assert result["fixed_intrinsic"]["paired_case_sem"] > 0


def test_amplitude_comparison_preserves_case_pairing():
    results = {
        "0.02": {
            "models": {
                "model": {
                    "response": {
                        "fixed_intrinsic": _response([-0.01, -0.03], [-0.02, -0.02]),
                        "rendered_per_leg": _response([-0.02, -0.04], [-0.03, -0.03]),
                    }
                }
            }
        },
        "0.05": {
            "models": {
                "model": {
                    "response": {
                        "fixed_intrinsic": _response([-0.02, -0.02], [-0.03, -0.01]),
                        "rendered_per_leg": _response([-0.03, -0.03], [-0.04, -0.02]),
                    }
                }
            }
        },
    }
    result = MODULE.paired_amplitude_comparison(results, ["model"])
    fixed = result["models"]["model"]["fixed_intrinsic"]
    assert result["low_shear_label"] == "0.02"
    assert np.isclose(fixed["truth_low_minus_high"], 0.0)
    assert np.isclose(fixed["model_low_minus_high"], 0.0)
