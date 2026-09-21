import numpy as np
import pytest
import torch

from scripts.audit_classifier_measurement_transfer import ensemble_logs, match_ids, response_moments, summarize


def test_stable_ensemble_extreme_logits():
    logits = torch.tensor([[1000., -1000., 2.], [900., -900., -2.]])
    lp, lq = ensemble_logs(logits)
    assert torch.isfinite(lp).all() and torch.isfinite(lq).all()
    torch.testing.assert_close(lp.exp()+lq.exp(), torch.ones(3, dtype=torch.float64))
    assert lq[0] < -899 and lp[1] < -899
    assert lp[2].exp().item() == pytest.approx(.5)


def test_identity_join_reports_missing_and_rejects_duplicates():
    ni, oi, counts = match_ids(np.array([7, 3, 8]), np.array([3, 7, 6, 5]))
    np.testing.assert_array_equal(ni, [1, 0])
    np.testing.assert_array_equal(oi, [0, 1])
    assert counts == {'matched': 2, 'new_without_old': 1, 'old_outside_new': 2}
    with pytest.raises(ValueError, match='unique'):
        match_ids(np.array([1, 1]), np.array([1]))


def test_usability_moment_detected_measurement_failures_and_pairing():
    gamma = np.array([[.05, 0], [0, .05], [-.05, 0], [0, -.05]])
    e = np.array([[.2, .3], [.4, .1], [-.1, .2], [.3, -.2]])
    u0, ug = np.array([1, 0, 1, 1], bool), np.array([0, 1, 1, 0], bool)
    d = np.ones(4, bool)
    row = response_moments(e, gamma, u0.astype(float), ug.astype(float), u0, ug, d, d)
    result = summarize([row, row], 'response')
    np.testing.assert_allclose(result['truth_shape_prediction_minus_usable']['estimate'], 0.)
    expected = sum(np.outer(e[i], gamma[i])*(float(ug[i])-float(u0[i])) for i in range(4))@np.linalg.inv(gamma.T@gamma)
    np.testing.assert_allclose(result['truth_shape']['estimate'][1], expected)
    np.testing.assert_allclose(result['truth_shape']['estimate'][2], 0.)
    np.testing.assert_allclose(result['truth_shape']['case_bootstrap_se'], 0., atol=1e-12)
