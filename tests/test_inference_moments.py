import numpy as np
import pytest
from numpy.polynomial.hermite import hermgauss

from sbsi.inference_moments import (ShearDerivatives, posterior_derivatives,
    selected_derivatives, bounded_newton_proposal, selection_log_contrasts)
from sbsi.mock_population import last_retained_parent


def test_louis_identity_recovers_exact_correlated_gaussian_marginal():
    A = np.array([[1., .4], [.3, .9]])
    noise = np.diag([.2, .3])
    invnoise = np.linalg.inv(noise)
    posterior_cov = np.linalg.inv(np.eye(2)+A.T@invnoise@A)
    g, y = np.array([.04, -.02]), np.array([.3, -.7])
    mean = posterior_cov@A.T@invnoise@(y-g)
    q, w = hermgauss(5)
    xx, yy = np.meshgrid(q, q)
    latent = mean+np.sqrt(2)*np.column_stack([xx.ravel(), yy.ravel()])@np.linalg.cholesky(posterior_cov).T
    score = (y-g-latent@A.T)@invnoise
    hessian = np.broadcast_to(-invnoise, (len(latent), 2, 2))
    weights = np.outer(w, w).ravel()/np.pi
    result = posterior_derivatives(g, score, hessian, log_weights=np.log(weights))
    exact = np.linalg.inv(noise+A@A.T)
    np.testing.assert_allclose(result.score, exact@(y-g), atol=1e-14)
    np.testing.assert_allclose(result.hessian, -exact, atol=1e-14)
    assert abs(result.hessian[0, 1]) > .1
    # Averaging complete-data curvature alone is decisively wrong in this case.
    assert np.linalg.norm(result.hessian+invnoise) > 1


def test_common_centre_selection_and_bounded_newton_step():
    g = np.array([.01, -.02])
    information = np.array([[4., 1.], [1., 3.]])
    target = np.array([.04, .06])
    item = ShearDerivatives(g, information@(target-g), -information)
    selection = ShearDerivatives(g, [.1, .2], [[.3, .1], [.1, .2]])
    result = selected_derivatives([item, item], selection)
    np.testing.assert_allclose(result.score, 2*(item.score-selection.score))
    np.testing.assert_allclose(result.hessian, 2*(item.hessian-selection.hessian))
    np.testing.assert_allclose(bounded_newton_proposal(item, max_norm=1.), target)
    assert np.isclose(np.linalg.norm(bounded_newton_proposal(item, max_norm=.005)-g), .005)
    with pytest.raises(ValueError, match='same shear'):
        selected_derivatives([item], ShearDerivatives(g+[1e-12, 0], [0, 0], -np.eye(2)))
    with pytest.raises(ValueError, match='positive definite'):
        bounded_newton_proposal(ShearDerivatives(g, [1, 0], [[1, 0], [0, -1]]), max_norm=.01)


def test_selection_covariance_keeps_shared_noise_and_allows_zero_chunks():
    mass = np.array([[0., 0., .2], [1., 2., .4], [2., 4., .7], [3., 6., .6]])
    # First contrast is exactly constant despite noisy masses in both columns.
    contrasts = np.array([[-1., 1., 0.], [0., -1., 1.]])
    result = selection_log_contrasts(mass, contrasts)
    np.testing.assert_allclose(result['value'][0], np.log(2))
    np.testing.assert_allclose(result['covariance'][0], 0, atol=1e-30)
    # Differentiate the estimator under single-replicate upweighting.
    h = 1e-5
    numeric = []
    for row in mass:
        plus = (mass.sum(0)+h*row)/(len(mass)+h)
        minus = (mass.sum(0)-h*row)/(len(mass)-h)
        numeric.append(contrasts@(np.log(plus)-np.log(minus))/(2*h)*len(mass))
    np.testing.assert_allclose(result['influence'], numeric, atol=1e-9)


def test_selection_joint_covariance_matches_independent_bootstrap():
    rng = np.random.default_rng(43)
    a = rng.normal(size=(256, 2))
    mass = np.exp(.15*a@np.array([[1., .8, -.4], [.1, .3, .9]]))
    contrasts = np.array([[-1., 1., 0.], [-1., 0., 1.]])
    result = selection_log_contrasts(mass, contrasts)
    # Independent resampling of whole chunks preserves each row's correlations.
    means = mass[rng.integers(0, len(mass), size=(2500, len(mass)))].mean(1)
    bootstrap_cov = np.cov((np.log(means)@contrasts.T).T)
    np.testing.assert_allclose(result['covariance'], bootstrap_cov, rtol=.08, atol=1e-7)
    assert abs(result['covariance'][0, 1]) > 1e-6


@pytest.mark.parametrize('ids', [[0], [0, 2], []])
def test_complete_mock_chunks_keep_original_unselected_tails(ids):
    ids = np.asarray(ids, dtype=int)
    assert last_retained_parent(ids, len(ids), 4, final_chunk=False) == 3
    assert last_retained_parent(ids, len(ids), 4, final_chunk=True) == 3


def test_partial_final_mock_chunk_and_inconsistent_receipt():
    assert last_retained_parent([1, 3, 5], 2, 8, final_chunk=True) == 3
    with pytest.raises(ValueError, match='only the final'):
        last_retained_parent([1, 3, 5], 2, 8, final_chunk=False)


def test_full_parent_coverage_removes_selection_condition_in_exhaustive_toy():
    selected_sum, parent_sum = 0, 0
    for first in [False, True]:
        for second in [False, True]:
            flags = np.array([first, second])
            ids = np.flatnonzero(flags)
            last = last_retained_parent(ids, len(ids), 2, final_chunk=False)
            selected_sum += flags[:last+1].sum()
            parent_sum += last+1
    assert selected_sum/parent_sum == .5
