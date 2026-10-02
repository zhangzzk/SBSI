import numpy as np
import pytest

from _script_loader import load_script_module


MODULE = load_script_module("audit_inference_uncertainty.py")


def test_case_uncertainty_recognizes_perfect_within_case_correlation():
    # For constant curvature this is exactly a sample mean. Repeating every
    # case's value creates no new independent information.
    values = np.array([[-2., 1.], [-1., -1.], [1., -2.], [2., 2.]])
    repeats = 10
    score = np.repeat(values, repeats, axis=0)
    information = np.broadcast_to(np.eye(2), (len(score), 2, 2))
    s = MODULE.summarize([0, 0], score, information, np.repeat(np.arange(4), repeats))
    np.testing.assert_allclose(s["case_se"], values.std(0, ddof=1) / 2)
    np.testing.assert_allclose(s["row_se"], score.std(0, ddof=1) / np.sqrt(len(score)))
    assert np.all(s["case_se"] > 3 * s["row_se"])


def test_bootstrap_recomputes_random_curvature_ratio():
    score = np.array([[1., 3.], [4., -1.], [-2., 2.]])
    information = np.array([np.eye(2) * x for x in (1., 2., 4.)])
    actual = MODULE.resample_cases([0, 0], score, information, replicates=1000, seed=13)
    weights = np.random.default_rng(13).multinomial(3, [1/3] * 3, size=1000)
    expected = (weights @ score) / (weights @ np.array([1., 2., 4.]))[:, None]
    np.testing.assert_allclose(actual["standard_error"], expected.std(0, ddof=1))
    np.testing.assert_allclose(actual["percentile_95_interval"], np.quantile(expected, [.025, .975], axis=0).T)
    assert actual["invalid_information_replicates"] == 0


def test_case_uncertainty_rejects_single_group():
    with pytest.raises(ValueError, match="at least two cases"):
        MODULE.summarize([0, 0], np.ones((4, 2)), np.broadcast_to(np.eye(2), (4, 2, 2)), np.zeros(4))
