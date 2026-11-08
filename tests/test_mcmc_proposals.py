"""Correlated-Gaussian posterior closure and degenerate-warmup support."""
import numpy as np

from sbsi.mcmc_proposals import (complementary_stretch_proposal, empirical_block_cholesky,
                                gaussian_block_delta, pooled_block_cholesky)
from sbsi.pseudo_marginal import metropolis_accept, retained


def test_complementary_stretch_is_affine_equivariant_and_preserves_nuisance_coordinates():
    x = np.random.default_rng(801).normal(size=(12, 5))
    factor = np.array([[2., -.3, 0], [.4, 1., .2], [0, -.5, 1.]])
    offset = np.array([3., -4., 1.])
    transformed = x.copy()
    transformed[:, :3] = x[:, :3]@factor.T+offset
    p, h, active = complementary_stretch_proposal(x, [0, 1, 2], 4, np.random.default_rng(802))
    q, j, mask = complementary_stretch_proposal(transformed, [0, 1, 2], 4, np.random.default_rng(802))
    np.testing.assert_array_equal(active.reshape(-1, 4).sum(1), 2)
    np.testing.assert_array_equal(p[~active], x[~active])
    np.testing.assert_array_equal(p[:, 3:], x[:, 3:])
    np.testing.assert_allclose(q[:, :3], p[:, :3]@factor.T+offset, atol=1e-14)
    np.testing.assert_array_equal(h, j)
    np.testing.assert_array_equal(active, mask)


def test_hybrid_stretch_recovers_gaussian_with_walker_specific_conditional_targets():
    # Different nuisance states can give each walker a different conditional
    # density; complementary anchors do not change the target of that walker.
    rng = np.random.default_rng(803)
    ensembles, chains, d = 24, 4, 3
    means = np.tile(np.arange(chains)[:, None]*np.array([.2, -.1, .1]), (ensembles, 1))
    factor = np.array([[.6, 0, 0], [.3, .4, 0], [-.2, .1, .3]])
    covariance = factor@factor.T
    precision = np.linalg.inv(covariance)
    x = means+rng.normal(size=means.shape)

    def density(v):
        delta = v-means
        return -.5*np.einsum('ni,ij,nj->n', delta, precision, delta)

    traces = []
    for step in range(5000):
        if rng.random() < .7:
            proposal, correction, active = complementary_stretch_proposal(x, np.arange(d), chains, rng)
        else:
            proposal = x+.6*rng.normal(size=x.shape)@factor.T
            correction, active = np.zeros(len(x)), np.ones(len(x), dtype=bool)
        accepted = active & metropolis_accept(density(x), density(proposal)+correction,
                                              rng.uniform(1e-15, 1., len(x)))
        x = retained(x, proposal, accepted)
        if step >= 1000:
            traces.append((x-means).reshape(ensembles, chains, d))
    traces = np.array(traces)
    independent_means = traces.mean(axis=(0, 2))
    assert np.all(np.abs(independent_means.mean(0)) < 6*independent_means.std(0, ddof=1)/np.sqrt(ensembles))
    second = np.einsum('teci,tecj->eij', traces, traces)/(len(traces)*chains)
    assert np.all(np.abs(second.mean(0)-covariance) < 6*second.std(0, ddof=1)/np.sqrt(ensembles))


def test_covariance_retains_support_for_stuck_chains_and_isolates_chains():
    history = np.zeros((20, 2, 3))
    history[:, 1] = np.arange(20)[:, None]*np.array([1., -2., 3.])
    factor = empirical_block_cholesky(history, [.1, .2, .3])
    covariance = factor@factor.transpose(0, 2, 1)
    assert np.linalg.eigvalsh(covariance).min() > 0
    np.testing.assert_allclose(covariance[0], .05*np.diag(np.array([.1, .2, .3])**2))
    assert covariance[1, 0, 1] < 0


def test_pooled_covariance_includes_between_chain_variation_but_not_other_observations():
    rng = np.random.default_rng(972)
    history = rng.normal(size=(30, 8, 3))
    history[:, :4] += np.arange(4)[None, :, None]*np.array([1., -2., .5])
    history[:, 4:] += 1000
    factors = pooled_block_cholesky(history, [.1, .2, .3], 4)
    for obs in range(2):
        expected = .95*np.cov(history[:, obs*4:obs*4+4].reshape(-1, 3).T)
        expected += .05*np.diag(np.array([.1, .2, .3])**2)
        for chain in range(obs*4, obs*4+4):
            np.testing.assert_allclose(factors[chain]@factors[chain].T, expected, atol=1e-12)
    assert (factors[0]@factors[0].T)[0, 1] < -1


def test_pooled_frozen_proposal_recovers_correlated_gaussian_from_displaced_chains():
    rng = np.random.default_rng(978)
    mean = np.array([.6, -.8])
    target_factor = np.array([[.4, 0], [-.7, .15]])
    covariance = target_factor@target_factor.T
    precision = np.linalg.inv(covariance)
    history = mean+rng.normal(size=(256, 32, 2))@target_factor.T
    history += np.linspace(-1, 1, 32)[None, :, None]*np.array([.4, -.7])
    factor = pooled_block_cholesky(history, [.2, .2], 4)
    x = history[-1].copy()
    def density(v):
        delta = v-mean
        return -.5*np.einsum('ni,ij,nj->n', delta, precision, delta)
    samples = []
    for step in range(4000):
        proposed = x+gaussian_block_delta(factor, np.full(len(x), 1.4), rng)
        accepted = metropolis_accept(density(x), density(proposed), rng.uniform(1e-15, 1, len(x)))
        x = retained(x, proposed, accepted)
        if step >= 500:
            samples.append(x.copy())
    samples = np.array(samples)
    means = samples.mean(0)
    assert np.all(np.abs(means.mean(0)-mean) < 6*means.std(0, ddof=1)/np.sqrt(len(x)))
    second = np.einsum('tni,tnj->nij', samples-mean, samples-mean)/len(samples)
    assert np.all(np.abs(second.mean(0)-covariance) < 6*second.std(0, ddof=1)/np.sqrt(len(x)))


def test_frozen_correlated_proposal_recovers_gaussian_mean_and_covariance():
    rng = np.random.default_rng(218)
    target_mean = np.array([.6, -.8, 1.2])
    target_factor = np.array([[.4, 0, 0], [-.7, .15, 0], [.3, -.2, .3]])
    covariance = target_factor@target_factor.T
    precision = np.linalg.inv(covariance)
    x = rng.normal(size=(48, 3))
    warmup = target_mean + rng.normal(size=(1024, 48, 3))@target_factor.T
    factor = empirical_block_cholesky(warmup, [.2, .2, .2])
    frozen = factor.copy()

    def density(values):
        d = values-target_mean
        return -.5*np.einsum('ni,ij,nj->n', d, precision, d)

    trace = []
    for step in range(3500):
        proposal = x+gaussian_block_delta(factor, np.full(len(x), 1.3), rng)
        accept = metropolis_accept(density(x), density(proposal), rng.uniform(1e-15, 1, len(x)))
        x = retained(x, proposal, accept)
        if step >= 500:
            trace.append(x.copy())
    trace = np.array(trace)
    moments = trace.mean(0)
    assert np.all(np.abs(moments.mean(0)-target_mean) < 6*moments.std(0, ddof=1)/np.sqrt(len(x)))
    centered = trace-target_mean
    second = np.einsum('tni,tnj->nij', centered, centered)/len(trace)
    assert np.all(np.abs(second.mean(0)-covariance) < 6*second.std(0, ddof=1)/np.sqrt(len(x)))
    np.testing.assert_array_equal(factor, frozen)


def test_circular_covariance_is_invariant_to_phase_origin():
    rng = np.random.default_rng(139)
    latent = rng.normal(size=(1000, 3))
    history = np.stack((latent, (.1+.2*latent+rng.normal(0, .05, latent.shape)) % (2*np.pi)), axis=2)
    original = empirical_block_cholesky(history, [.2, .2], circular_indices=(1,))
    history[:, :, 1] = (history[:, :, 1]+2.7) % (2*np.pi)
    rotated = empirical_block_cholesky(history, [.2, .2], circular_indices=(1,))
    np.testing.assert_allclose(original, rotated, atol=1e-13)
    assert np.all((original@original.transpose(0, 2, 1))[:, 0, 1] > .1)


def test_wrapped_correlated_proposal_recovers_circular_gaussian_moments():
    from scipy.special import logsumexp
    rng = np.random.default_rng(143)
    mean = np.array([.7, 6.2])
    factor0 = np.array([[.5, 0], [.2, .15]])
    covariance = factor0@factor0.T
    precision = np.linalg.inv(covariance)
    warmup = mean+rng.normal(size=(1024, 32, 2))@factor0.T
    warmup[:, :, 1] %= 2*np.pi
    factor = empirical_block_cholesky(warmup, [.2, .2], circular_indices=(1,))
    x = warmup[-1].copy()

    def density(x):
        # Adjacent wraps capture this narrow Gaussian to floating-point precision.
        d = x[:, None, :]-mean
        d = np.broadcast_to(d, (len(x), 3, 2)).copy()
        d[:, :, 1] += 2*np.pi*np.arange(-1, 2)
        return logsumexp(-.5*np.einsum('nki,ij,nkj->nk', d, precision, d), axis=1)

    moments = []
    for step in range(3500):
        proposal = x+gaussian_block_delta(factor, np.full(len(x), 1.5), rng)
        proposal[:, 1] %= 2*np.pi
        accept = metropolis_accept(density(x), density(proposal), rng.uniform(1e-15, 1, len(x)))
        x = retained(x, proposal, accept)
        if step >= 500:
            moments.append(np.column_stack((x[:, 0], np.sin(x[:, 1]), x[:, 0]*np.sin(x[:, 1]))))
    expected = np.array([mean[0], np.sin(mean[1])*np.exp(-covariance[1, 1]/2),
        (mean[0]*np.sin(mean[1])+covariance[0, 1]*np.cos(mean[1]))*np.exp(-covariance[1, 1]/2)])
    means = np.array(moments).mean(0)
    assert np.all(np.abs(means.mean(0)-expected) < 6*means.std(0, ddof=1)/np.sqrt(len(x)))
