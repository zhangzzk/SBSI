"""Regression for scoring the actual Boolean usability flags in saved rows."""
import numpy as np
import importlib.util
import os
import pytest

runner = os.environ.get('SBSI_CLASSIFIER_SCORING_RUNNER')
if not runner:
    pytest.skip('requires the isolated evaluation runner', allow_module_level=True)
spec = importlib.util.spec_from_file_location('fixed_classifier_runner', runner)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
metrics = module.metrics


def test_boolean_labels_have_correct_binary_scores():
    scores = metrics(np.array([.8, .2]), np.array([True, False]))
    np.testing.assert_allclose(scores['bce'], -np.log(.8))
    np.testing.assert_allclose(scores['brier'], .04)
    np.testing.assert_allclose(scores['calibration'], 0, atol=1e-15)


def test_boolean_and_numeric_labels_agree():
    p = np.array([.1, .7, .6, .9])
    u = np.array([False, True, False, True])
    assert metrics(p, u) == metrics(p, u.astype(float))
