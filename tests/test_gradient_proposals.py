import numpy as np
import pytest
from sbsi.gradient_proposals import bounded_langevin_mean, gaussian_log_hastings, ReplicaTransport


def test_bounded_drift_and_exact_forward_reverse_probability_flux():
    rng = np.random.default_rng(321)
    x, y = rng.normal(size=(2, 30, 2))
    factor = np.broadcast_to([[.6, 0], [.2, .9]], (30, 2, 2)).copy()
    # Deliberately wrong gradient: proposal correctness must not rely on it.
    drift = lambda z: np.column_stack((np.sin(4*z[:, 0]), 10+z[:, 1]**2))
    mx = bounded_langevin_mean(x, drift(x), factor, max_drift=.7)
    my = bounded_langevin_mean(y, drift(y), factor, max_drift=.7)
    assert np.max(np.linalg.norm(np.linalg.solve(factor, (mx-x)[..., None])[..., 0], axis=1)) <= .7+1e-14
    h = gaussian_log_hastings(x, y, mx, my, factor)
    np.testing.assert_allclose(h, -gaussian_log_hastings(y, x, my, mx, factor), atol=1e-14)
    logq = lambda delta: -.5*np.linalg.solve(factor, delta[..., None])[..., 0].__pow__(2).sum(1)
    lp = lambda z: -.5*(z*z).sum(1)
    log_forward = lp(x)+logq(y-mx)+np.minimum(0, lp(y)-lp(x)+h)
    log_reverse = lp(y)+logq(x-my)+np.minimum(0, lp(x)-lp(y)-h)
    np.testing.assert_allclose(log_forward, log_reverse, atol=1e-13)


def test_wrong_drift_corrected_sampler_recovers_curved_target():
    rng = np.random.default_rng(772)
    x = rng.normal(size=(48, 2))
    factor = np.broadcast_to([[.65, 0], [0, .4]], (len(x), 2, 2)).copy()
    logp = lambda z: -.5*z[:, 0]**2-.5*((z[:, 1]-.5*(z[:, 0]**2-1))/.4)**2
    # Inexact smooth guide; banana moments are available analytically.
    guide = lambda z: -.25*z
    draws = []
    for t in range(20000):
        mx = bounded_langevin_mean(x, guide(x), factor)
        y = mx+np.einsum('nij,nj->ni', factor, rng.normal(size=x.shape))
        my = bounded_langevin_mean(y, guide(y), factor)
        accept = np.log(rng.random(len(x))) < logp(y)-logp(x)+gaussian_log_hastings(x, y, mx, my, factor)
        x = np.where(accept[:, None], y, x)
        if t >= 4000:
            draws.append(x.copy())
    z = np.concatenate(draws)
    np.testing.assert_allclose(z.mean(0), [0, 0], atol=.06)
    np.testing.assert_allclose(z.var(0), [1, .66], atol=.10)


def test_invalid_cholesky_and_nonfinite_drift_rejected():
    with pytest.raises(ValueError):
        bounded_langevin_mean(np.zeros((1, 2)), np.zeros((1, 2)), np.array([[[1, 1], [0, 1]]]))
    with pytest.raises(ValueError):
        bounded_langevin_mean(np.zeros((1, 2)), np.full((1, 2), np.nan), np.eye(2)[None])


def test_transport_requires_cold_hot_cold_and_records_endpoint_change():
    tracker = ReplicaTransport(3, 1, 1)
    tracker.update(np.arange(3), np.array([[2.], [0.], [0.]]))
    tracker.update(np.array([1, 0, 2]), np.array([[0.], [2.], [0.]]))
    tracker.update(np.array([0, 2, 1]), np.array([[0.], [0.], [3.]]))
    assert tracker.count.tolist() == [0]
    tracker.update(np.array([0, 2, 1]), np.array([[0.], [4.], [0.]]))
    tracker.update(np.array([1, 0, 2]), np.array([[5.], [0.], [0.]]))
    assert tracker.count.tolist() == [1]
    assert tracker.report()['mean_absolute_endpoint_change'] == [[3.]]
    for _ in range(10):
        tracker.update(np.arange(3), np.array([[5.], [0.], [0.]]))
    assert tracker.count.tolist() == [1]
