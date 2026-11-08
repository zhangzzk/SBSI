"""Analytic checks of SMC bookkeeping, including independent-run uncertainty."""

import json

import numpy as np
import pytest
from scipy.special import logsumexp
from scipy.stats import multivariate_normal, norm

from sbsi.annealed_particles import (
    annealed_smc,
    effective_ancestors,
    effective_sample_size,
    normalized_log_weights,
    reweight,
    stratified_resample,
)


def test_normalization_and_ess_are_stable_with_zero_mass():
    weights = normalized_log_weights(np.array([1000.0, 1000.0, -np.inf]))
    np.testing.assert_allclose(weights[:2], -np.log(2))
    assert np.isneginf(weights[2])
    assert effective_sample_size(weights) == pytest.approx(2)
    assert np.isneginf(normalized_log_weights([-np.inf, -np.inf])).all()
    assert effective_sample_size([-np.inf, -np.inf]) == 0


@pytest.mark.parametrize("bad", [[], [[0.0]], [np.nan], [np.inf]])
def test_bad_log_weights_are_rejected(bad):
    with pytest.raises(ValueError):
        normalized_log_weights(bad)


def test_reweight_retains_the_full_proposal_normalization():
    initial = np.zeros(4)
    potential = np.array([np.log(6), -np.inf, -np.inf, np.log(6)])
    unchanged, zero = reweight(initial, potential, 0)
    np.testing.assert_allclose(unchanged, -np.log(4))
    assert zero == 0
    weights, first = reweight(initial, potential, 0.25)
    weights, second = reweight(weights, potential, 0.75)
    # Half the normalized q0 mass is outside target support. Filtering first
    # would instead give the incorrect normalizing constant 6.
    assert first + second == pytest.approx(np.log(3))
    np.testing.assert_allclose(np.exp(weights), [0.5, 0, 0, 0.5])
    extinct, total = reweight(initial, np.full(4, -np.inf), 0.1)
    assert np.isneginf(extinct).all() and total == -np.inf
    extinct, total = reweight(extinct, np.zeros(4), 0)
    assert np.isneginf(extinct).all() and total == -np.inf


@pytest.mark.parametrize("delta", [-0.1, np.nan, np.inf])
def test_reweight_rejects_invalid_increment(delta):
    with pytest.raises(ValueError):
        reweight([0], [1], delta)


def test_reweight_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        reweight([0, 0], [1], 0.5)


def test_stratified_resampling_zero_mass_and_unbiased_offspring():
    rng = np.random.default_rng(2026102101)
    weights = np.array([0.0, 0.05, 0.25, 0.0, 0.7])
    with np.errstate(divide="ignore"):
        log_weights = np.log(weights)
    counts = np.array([
        np.bincount(stratified_resample(log_weights, rng), minlength=len(weights))
        for _ in range(2000)
    ])
    assert np.all(counts[:, [0, 3]] == 0)
    np.testing.assert_allclose(counts.mean(0), len(weights) * weights, atol=0.04)
    with pytest.raises(ValueError, match="extinct"):
        stratified_resample([-np.inf, -np.inf], rng)


def test_stratified_resampling_skips_zero_mass_at_exact_zero_uniform():
    class ZeroUniform:
        def random(self, n):
            return np.zeros(n)

    indices = stratified_resample([-np.inf, 0, -np.inf], ZeroUniform())
    np.testing.assert_array_equal(indices, [1, 1, 1])


def test_ancestral_ess_pools_duplicates_including_unequal_current_weights():
    ancestors = np.array([10, 10, 20, 30])
    assert effective_ancestors(ancestors) == pytest.approx(8 / 3)
    log_weights = np.log([0.1, 0.2, 0.3, 0.4])
    assert effective_ancestors(ancestors, log_weights) == pytest.approx(1 / 0.34)
    assert effective_ancestors(ancestors, np.full(4, -np.inf)) == 0
    for labels in ([], [0.5], [-1, 0], [[0, 1]]):
        with pytest.raises(ValueError):
            effective_ancestors(labels)
    with pytest.raises(ValueError):
        effective_ancestors(ancestors, [0, 0])


def test_driver_keeps_initial_support_mass_and_records_genealogy():
    initial = np.array([0.1, 0.4, 0.9, 0.2])
    result = annealed_smc(
        initial, lambda x: np.where(x < 0.25, np.log(4), -np.inf),
        [0, 0.1, 0.4, 1], np.random.default_rng(2026102102),
    )
    assert result.log_normalizer == pytest.approx(np.log(2))
    assert np.all(result.particles < 0.25)
    np.testing.assert_array_equal(initial[result.ancestors], result.particles)
    assert not result.extinct
    assert result.stages[0].weight_ess == pytest.approx(2)
    assert result.stages[-1].weight_ess == pytest.approx(4)
    assert result.stages[-1].unique_ancestors == 2
    assert result.stages[-1].ancestral_ess == pytest.approx(2)
    np.testing.assert_allclose(result.log_weights, -np.log(4))
    np.testing.assert_array_equal(initial, [0.1, 0.4, 0.9, 0.2])


def test_extinct_driver_returns_zero_normalizer_without_rerunning():
    def forbidden(*args):
        raise AssertionError("must not mutate an extinct population")

    result = annealed_smc(
        np.arange(3.0), lambda x: np.full(len(x), -np.inf),
        [0, 0.25, 1], np.random.default_rng(2026102103), mutation=forbidden,
    )
    assert result.extinct
    assert result.log_normalizer == -np.inf
    assert len(result.stages) == 1
    assert result.stages[0].unique_ancestors == 0
    assert result.stages[0].ancestral_ess == 0


@pytest.mark.parametrize("schedule", [[0, 0.5], [0.1, 1], [0, 0.5, 0.5, 1],
                                      [0, 0.8, 0.2, 1], [0, np.nan, 1]])
def test_driver_rejects_invalid_schedules(schedule):
    with pytest.raises(ValueError):
        annealed_smc(np.zeros(3), lambda x: np.zeros(len(x)), schedule,
                     np.random.default_rng(1))


@pytest.mark.parametrize("steps", [-1, 1.2, True])
def test_driver_rejects_invalid_mutation_count(steps):
    with pytest.raises(ValueError):
        annealed_smc(np.zeros(3), lambda x: np.zeros(len(x)), [0, 1],
                     np.random.default_rng(1), mutation_steps=steps)


def test_driver_rejects_bad_callback_shapes_and_nonfinite_mutations():
    with pytest.raises(ValueError, match="per particle"):
        annealed_smc(np.zeros(3), lambda x: np.zeros(1), [0, 1],
                     np.random.default_rng(1))
    for mutation in (lambda x, b, r: np.zeros(2), lambda x, b, r: x + np.nan):
        with pytest.raises(ValueError, match="mutation"):
            annealed_smc(np.zeros(3), lambda x: np.zeros(len(x)), [0, 1],
                         np.random.default_rng(1), mutation=mutation)


def test_driver_detects_mutation_leaving_target_support_at_final_stage():
    with pytest.raises(ValueError, match="positive target support"):
        annealed_smc(np.zeros(3), lambda x: np.where(x < 1, 0, -np.inf),
                     [0, 1], np.random.default_rng(1),
                     mutation=lambda x, beta, rng: x + 2)


def _replicate_summary(label, values, truth, absolute_tolerance=0.003):
    """Compare independent population estimates, not individual particle SEs."""
    values = np.asarray(values)
    truth = np.asarray(truth)
    estimate = values.mean(0)
    sem = values.std(0, ddof=1) / np.sqrt(len(values))
    print(json.dumps({"target": label, "replicates": len(values),
                      "estimate": estimate.tolist(), "truth": truth.tolist(),
                      "replicate_mcse": sem.tolist()}))
    assert np.all(np.abs(estimate - truth) < 5 * sem + absolute_tolerance)


def test_replicated_correlated_gaussian_moments_and_normalizer():
    """Analytic 2D posterior, including off-diagonal moment and region mass."""
    covariance = np.array([[0.5, 0.2], [0.2, 0.25]])
    inverse = np.linalg.inv(covariance)
    observation = np.array([0.8, -0.6])
    posterior_cov = np.linalg.inv(np.eye(2) + inverse)
    posterior_mean = posterior_cov @ inverse @ observation
    exact_z = multivariate_normal.pdf(observation, cov=np.eye(2) + covariance)
    log_scale = -np.log(2 * np.pi) - 0.5 * np.linalg.slogdet(covariance)[1]

    def potential(x):
        error = observation - x
        return log_scale - 0.5 * np.einsum("ni,ij,nj->n", error, inverse, error)

    def mutation(x, beta, rng):
        proposal = x + 0.7 * rng.normal(size=x.shape)
        log_ratio = (-0.5 * np.sum(proposal**2 - x**2, axis=1)
                     + beta * (potential(proposal) - potential(x)))
        accepted = np.log(rng.random(len(x))) < log_ratio
        return np.where(accepted[:, None], proposal, x)

    estimates, z_values = [], []
    for seed in range(2026102200, 2026102264):
        rng = np.random.default_rng(seed)
        result = annealed_smc(rng.normal(size=(512, 2)), potential,
                              [0, 0.05, 0.15, 0.35, 0.6, 1], rng,
                              mutation=mutation, mutation_steps=4)
        x = result.particles
        estimates.append([*x.mean(0), *(x**2).mean(0), np.mean(x[:, 0] * x[:, 1]),
                          np.mean(x[:, 0] < 0)])
        z_values.append(np.exp(result.log_normalizer))
        assert 1 <= result.stages[-1].ancestral_ess <= 512
    truth = [*posterior_mean, *(np.diag(posterior_cov) + posterior_mean**2),
             posterior_cov[0, 1] + np.prod(posterior_mean),
             norm.cdf(-posterior_mean[0] / np.sqrt(posterior_cov[0, 0]))]
    _replicate_summary("correlated_gaussian_posterior", estimates, truth)
    _replicate_summary("correlated_gaussian_Z", z_values, exact_z, 0)


def test_replicated_separated_mixture_region_mass_and_normalizer():
    """A sharp unequal-weight two-mode target with known total mass 0.37."""
    exact_z = 0.37

    def log_q(x):
        return norm.logpdf(x, scale=3)

    def log_target(x):
        terms = np.stack([np.log(0.3) + norm.logpdf(x, -3, 0.4),
                          np.log(0.7) + norm.logpdf(x, 3, 0.6)])
        return np.log(exact_z) + logsumexp(terms, axis=0)

    def potential(x):
        return log_target(x) - log_q(x)

    def mutation(x, beta, rng):
        # Both centred and reflected Gaussian proposals are symmetric. The
        # mixture probability and scale are frozen, not fitted to particles.
        center = np.where(rng.random(len(x)) < 0.35, -x, x)
        proposal = center + 0.7 * rng.normal(size=x.shape)
        ratio = (log_q(proposal) - log_q(x)
                 + beta * (potential(proposal) - potential(x)))
        return np.where(np.log(rng.random(len(x))) < ratio, proposal, x)

    region_values, z_values, moments = [], [], []
    for seed in range(2026102300, 2026102364):
        rng = np.random.default_rng(seed)
        result = annealed_smc(3 * rng.normal(size=512), potential,
                              [0, 0.03, 0.08, 0.2, 0.45, 0.7, 1], rng,
                              mutation=mutation, mutation_steps=4)
        region_values.append(np.mean(result.particles < 0))
        moments.append([np.mean(result.particles), np.mean(result.particles**2)])
        z_values.append(np.exp(result.log_normalizer))
    exact_left = 0.3 * norm.cdf(0, -3, 0.4) + 0.7 * norm.cdf(0, 3, 0.6)
    _replicate_summary("separated_mixture_left_mass", region_values, exact_left)
    _replicate_summary("separated_mixture_moments", moments, [1.2, 9.3])
    _replicate_summary("separated_mixture_Z", z_values, exact_z, 0)
