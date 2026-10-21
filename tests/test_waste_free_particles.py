"""Finite-population expectation and retention checks for the isolated driver."""
from fractions import Fraction as F
from itertools import product

import numpy as np
import pytest

from sbsi.annealed_particles import annealed_smc, stratified_resample
from sbsi.waste_free_particles import stratified_ancestors, waste_free_smc


class Uniforms:
    def __init__(self, values):
        self.values = iter(values)

    def random(self, n):
        return np.array([next(self.values) for _ in range(n)])


def midpoint(probabilities, index):
    return float(sum(probabilities[:index])+probabilities[index]/2)


def test_exact_enumeration_including_zero_initial_mass_and_all_extinct():
    g = (F(0), F(1), F(2))
    pi = ((F(0), F(1, 3), F(2, 3)), (F(0), F(1, 5), F(4, 5)))
    kernels = [tuple(tuple(F(int(i == j), 2)+row[j]/2 for j in range(3)) for i in range(3)) for row in pi]
    f = np.array([0., -2., 3.])
    mass = F(0)
    expected_z = expected_zf = 0.
    def potential(x):
        return np.array([-np.inf, 0., np.log(4.)])[np.asarray(x, int)]
    def mutation(x, beta, rng):
        matrix = kernels[0 if beta == .5 else 1]
        return np.array([np.searchsorted(np.cumsum(np.asarray(matrix[int(i)], float)), u, side='right')
            for i, u in zip(x, rng.random(len(x)))])
    for initial in product(range(3), repeat=2):
        total = sum(g[i] for i in initial)
        if not total:
            result = waste_free_smc(initial, potential, [0, .5, 1], Uniforms([]), path_length=2, mutation=mutation)
            assert result.extinct and result.log_normalizer == -np.inf
            mass += F(1, 9)
            continue
        w1 = tuple(g[i]/total for i in initial)
        for a, b in product(range(2), range(3)):
            probability1 = w1[a]*kernels[0][initial[a]][b]
            if not probability1:
                continue
            next_states = (initial[a], b)
            next_total = sum(g[i] for i in next_states)
            w2 = tuple(g[i]/next_total for i in next_states)
            for c, d in product(range(2), range(3)):
                probability = F(1, 9)*probability1*w2[c]*kernels[1][next_states[c]][d]
                if not probability:
                    continue
                uniforms = [midpoint(w1, a), midpoint(kernels[0][initial[a]], b),
                    midpoint(w2, c), midpoint(kernels[1][next_states[c]], d)]
                result = waste_free_smc(initial, potential, [0, .5, 1], Uniforms(uniforms), path_length=2, mutation=mutation)
                np.testing.assert_array_equal(result.particles, [next_states[c], d])
                z = np.exp(result.log_normalizer)
                assert z == pytest.approx(float(total*next_total/4))
                mass += probability
                expected_z += float(probability)*z
                expected_zf += float(probability)*z*f[result.particles].mean()
    assert mass == 1
    assert expected_z == pytest.approx(5/3, abs=2e-14)
    assert expected_zf == pytest.approx(10/3, abs=2e-14)


def test_stratified_smaller_offspring_is_unbiased_and_skips_zeros():
    rng = np.random.default_rng(2026102900)
    weights = np.array([0, .1, .2, 0, .7])
    with np.errstate(divide='ignore'):
        logw = np.log(weights)
    counts = np.array([np.bincount(stratified_ancestors(logw, 2, rng), minlength=5) for _ in range(10000)])
    np.testing.assert_allclose(counts.mean(0), 2*weights, atol=.015)
    assert not counts[:, [0, 3]].any()
    np.testing.assert_array_equal(stratified_ancestors([-np.inf, 0], 1, Uniforms([0])), [1])
    with pytest.raises(ValueError, match='extinct'):
        stratified_ancestors([-np.inf], 1, rng)


def test_p_one_replays_current_generic_identity_mutation():
    initial = np.arange(12.)[:, None]
    potential = lambda x: -.05*x[:, 0]**2
    a = annealed_smc(initial, potential, [0, .2, 1], np.random.default_rng(3), mutation_steps=0)
    b = waste_free_smc(initial, potential, [0, .2, 1], np.random.default_rng(3), path_length=1)
    for name in ('particles', 'log_weights', 'ancestors', 'log_normalizer'):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
    np.testing.assert_array_equal(stratified_ancestors([0, 1, 2], 3, np.random.default_rng(7)),
        stratified_resample([0, 1, 2], np.random.default_rng(7)))


def test_retains_ancestor_and_rejected_iterates_without_aliasing():
    # Deterministic map is only a retention fixture, not a scientific kernel.
    def in_place(x, beta, rng):
        x += 1
        return x
    result = waste_free_smc(np.arange(6), lambda x: np.zeros(len(x)), [0, 1],
        Uniforms([.1, .2]), path_length=3, mutation=in_place)
    np.testing.assert_array_equal(result.particles, [0, 3, 1, 4, 2, 5])
    np.testing.assert_array_equal(result.ancestors, [0, 3, 0, 3, 0, 3])
    identity = waste_free_smc(np.arange(6), lambda x: np.zeros(len(x)), [0, 1],
        Uniforms([.1, .2]), path_length=3, mutation=lambda x, beta, rng: x)
    np.testing.assert_array_equal(identity.particles, [0, 3, 0, 3, 0, 3])


def test_initial_support_denominator_and_absorbing_extinction():
    result = waste_free_smc(np.array([0, 0, 1, 1]), lambda x: np.where(x, np.log(6), -np.inf),
        [0, .2, 1], np.random.default_rng(9), path_length=2)
    assert result.log_normalizer == pytest.approx(np.log(3))
    assert np.all(result.particles == 1)
    def unexpected(*args):
        pytest.fail('mutation after extinction')
    result = waste_free_smc(np.zeros(4), lambda x: np.full(4, -np.inf),
        [0, .2, 1], None, path_length=2, mutation=unexpected)
    assert result.extinct and len(result.stages) == 1 and result.log_normalizer == -np.inf
    assert np.isneginf(result.log_weights).all()


@pytest.mark.parametrize('path_length', [0, -1, True, 1.5, 3])
def test_bad_path_lengths_raise(path_length):
    with pytest.raises(ValueError):
        waste_free_smc(np.zeros(4), lambda x: x, [0, 1], None, path_length=path_length)


@pytest.mark.parametrize('betas', [[0], [0, .5], [0, 1, 1], [0, np.nan, 1]])
def test_bad_schedules_raise(betas):
    with pytest.raises(ValueError):
        waste_free_smc(np.zeros(4), lambda x: x, betas, None, path_length=2)


@pytest.mark.parametrize('potential', [lambda x: np.full(len(x), np.nan),
    lambda x: np.full(len(x), np.inf), lambda x: np.zeros(len(x)+1)])
def test_unknown_densities_raise(potential):
    with pytest.raises(ValueError):
        waste_free_smc(np.zeros(4), potential, [0, 1], None, path_length=2)


def test_mutation_shape_support_and_unknown_failures_raise():
    for mutate in (lambda x, beta, rng: np.zeros(5), lambda x, beta, rng: np.full_like(x, np.nan)):
        with pytest.raises(ValueError):
            waste_free_smc(np.zeros(4), lambda x: x, [0, 1], np.random.default_rng(1), path_length=2, mutation=mutate)
    with pytest.raises(ValueError, match='support'):
        waste_free_smc(np.ones(4), lambda x: np.where(x > 0, 0, -np.inf), [0, 1],
            np.random.default_rng(1), path_length=2, mutation=lambda x, beta, rng: -x)
    def broken(*args):
        raise RuntimeError('unknown failure')
    with pytest.raises(RuntimeError, match='unknown failure'):
        waste_free_smc(np.ones(4), lambda x: x, [0, 1], np.random.default_rng(1), path_length=2, mutation=broken)
