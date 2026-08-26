import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "evaluate_detection_classifier_constgold.py"
)
SPEC = importlib.util.spec_from_file_location(
    "evaluate_detection_classifier_constgold", SCRIPT
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_antithetic_detection_response_is_zero_for_identical_selection():
    shape = np.array([-0.3, -0.1, 0.2, 0.4])
    weights = np.array([0.2, 0.8, 0.4, 0.7])
    response = MODULE.antithetic_detection_response(shape, weights, weights, 0.02)
    assert abs(response) < 1.0e-12


def test_antithetic_detection_response_has_consistent_sign():
    shape = np.array([-0.3, -0.1, 0.2, 0.4])
    minus = np.ones(4)
    plus = np.array([1.0, 1.0, 0.0, 0.0])
    response = MODULE.antithetic_detection_response(shape, minus, plus, 0.02)
    assert response < 0


def test_response_summary_uses_paired_case_errors():
    records = [
        {"truth": -0.01, "model": -0.02},
        {"truth": -0.03, "model": -0.02},
    ]
    result = MODULE.summarize_response(records)
    assert np.isclose(result["truth"], -0.02)
    assert np.isclose(result["model"], -0.02)
    assert np.isclose(result["model_minus_truth"], 0.0)
    assert result["paired_case_sem"] > 0


def test_rendered_decomposition_summary_keeps_detected_minus_parent():
    records = [
        {"parent": 1.0, "detected": 0.98, "detected_minus_parent": -0.02},
        {"parent": 1.01, "detected": 0.99, "detected_minus_parent": -0.02},
    ]
    result = MODULE.rendered_decomposition_summary(records)
    assert np.isclose(result["parent"]["response"], 1.005)
    assert np.isclose(result["detected_minus_parent"]["response"], -0.02)
