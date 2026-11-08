import numpy as np
import pandas as pd
import pytest

from scripts.audit_continuous_crowding_residuals import SCALES, smooth_crowding, partial_associations


def test_kernel_values_isolation_and_no_response_features():
    pairs = pd.DataFrame({'index_input_p': [2, 2], 'index_input_s': [6, 7],
                          'r_input_p_scaled': [0., 0.], 'r_input_s_scaled': [0., 0.],
                          'distance_scaled': [0., .5], 'R_blend': [999., -999.]})
    result = smooth_crowding(np.array([2, 5]), pairs)
    np.testing.assert_allclose(result[0], np.log1p(1+np.exp(-.5*(.5/np.array(SCALES))**2)))
    np.testing.assert_array_equal(result[1], 0)
    np.testing.assert_array_equal(result, smooth_crowding(np.array([2, 5]), pairs.drop(columns='R_blend')))
    with pytest.raises(ValueError, match='outside the parent'):
        smooth_crowding(np.array([5]), pairs)


def test_partial_association_matches_direct_residualization_and_removes_confounding():
    q, _ = np.linalg.qr(np.column_stack((np.ones(200), np.random.default_rng(14).normal(size=(200, 3)))))
    x, u, v = q[:, :2], q[:, 2], q[:, 3]
    g = (3*q[:, 1]+u)[:, None]
    for r in (5*q[:, 1]+v, 5*q[:, 1]+.4*u+v):
        result = partial_associations(x.T@x, x.T@g, g.T@g, x.T@r, g.T@r, r@r, len(r))
        re = r-x@np.linalg.lstsq(x, r, rcond=None)[0]
        ge = g-x@np.linalg.lstsq(x, g, rcond=None)[0]
        np.testing.assert_allclose(result['partial_covariance'], re@ge/len(r), atol=1e-14)
        np.testing.assert_allclose(result['partial_correlation'], (re@ge)/np.sqrt((re@re)*(ge**2).sum(0)), atol=1e-13)
    assert result['partial_correlation'][0] > .3


def test_reject_zero_residual_variance():
    with pytest.raises(ValueError, match='residual variances'):
        partial_associations([[1.]], [[1.]], [[1.]], [1.], [1.], 1., 1.)
