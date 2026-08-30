from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import sbsi.island_annealed_smc as smc_module
from sbsi.island_annealed_smc import (
    UndirectedGraph,
    annealed_mh_transition_matrix,
    run_island_annealed_smc,
)


def _bridge_probability(log_target, q0, beta):
    q0 = np.asarray(q0, dtype=np.float64)
    q0 = q0 / q0.sum(dtype=np.float64)
    log_q0 = np.log(q0)
    if beta == 0.0:
        log_mass = log_q0
    elif beta == 1.0:
        log_mass = np.asarray(log_target, dtype=np.float64)
    else:
        log_mass = (1.0 - beta) * log_q0 + beta * np.asarray(
            log_target, dtype=np.float64
        )
    shift = np.max(log_mass)
    mass = np.exp(log_mass - shift)
    return mass / mass.sum(dtype=np.float64)


def test_independence_and_degree_corrected_graph_kernel_detailed_balance():
    q0 = np.asarray([0.1, 0.2, 0.3, 0.4])
    log_target = np.log(np.asarray([0.7, 0.1, 1.3, 0.4]))
    graph = UndirectedGraph(
        4, np.asarray([[0, 1], [1, 2], [1, 3]], dtype=np.int64)
    )
    beta = 0.63
    transition = annealed_mh_transition_matrix(
        log_target,
        q0,
        beta=beta,
        graph=graph,
        graph_move_probability=0.7,
    )
    probability = _bridge_probability(log_target, q0, beta)
    flux = probability[:, None] * transition

    np.testing.assert_allclose(transition.sum(axis=1), 1.0, atol=2e-16)
    np.testing.assert_allclose(flux, flux.T, rtol=2e-15, atol=2e-17)
    np.testing.assert_allclose(probability @ transition, probability, atol=2e-16)


def test_path_graph_degree_correction_balances_endpoint_and_center():
    q0 = np.full(3, 1.0 / 3.0)
    log_target = np.log(q0)
    graph = UndirectedGraph(3, np.asarray([[0, 1], [1, 2]], dtype=np.int64))
    transition = annealed_mh_transition_matrix(
        log_target,
        q0,
        beta=1.0,
        graph=graph,
        graph_move_probability=0.5,
    )

    # The independence half contributes 1/6 in either direction.  In the
    # graph half, endpoint->center proposes with probability one and accepts
    # with 1/2, while center->endpoint proposes with 1/2 and accepts with one.
    assert transition[0, 1] == pytest.approx(5.0 / 12.0, abs=2e-16)
    assert transition[1, 0] == pytest.approx(5.0 / 12.0, abs=2e-16)


def test_tiny_state_enumeration_and_monte_carlo_are_unbiased():
    q0 = np.asarray([0.15, 0.35, 0.5])
    target = np.asarray([0.9, 0.2, 1.7])
    log_target = np.log(target)
    betas = np.asarray([0.0, 0.2, 0.65, 1.0])
    graph = UndirectedGraph(3, np.asarray([[0, 1], [1, 2]], dtype=np.int64))

    # Exhaust every N=1 Feynman--Kac state path by matrix multiplication.
    # The terminal sum is the exact expectation of one island estimator.
    log_weight = log_target - np.log(q0)
    path_mass = q0.copy()
    for stage in range(1, len(betas)):
        path_mass *= np.exp((betas[stage] - betas[stage - 1]) * log_weight)
        path_mass = path_mass @ annealed_mh_transition_matrix(
            log_target,
            q0,
            beta=float(betas[stage]),
            graph=graph,
            graph_move_probability=0.4,
        )
    assert path_mass.sum(dtype=np.float64) == pytest.approx(
        target.sum(dtype=np.float64), rel=2e-15
    )

    result = run_island_annealed_smc(
        log_target,
        q0,
        betas=betas,
        n_islands=2048,
        particles_per_island=2,
        seed=20260830,
        object_id=25000,
        graph=graph,
        graph_move_probability=0.4,
    )
    exact = target.sum(dtype=np.float64)
    assert abs(result.normalizer - exact) <= 7.0 * result.standard_error


def test_two_particle_enumeration_includes_multinomial_resampling():
    q0 = np.asarray([0.2, 0.3, 0.5])
    target = np.asarray([1.1, 0.4, 1.8])
    log_target = np.log(target)
    beta = 0.4
    log_weight = log_target - np.log(q0)
    first_weight = np.exp(beta * log_weight)
    second_weight = np.exp((1.0 - beta) * log_weight)
    transition = annealed_mh_transition_matrix(
        log_target, q0, beta=beta
    )

    expected = 0.0
    states = range(len(q0))
    # Enumerate the ordered initial particles, both ordered multinomial
    # offspring, and both independent invariant-kernel destinations.
    for x0 in states:
        for x1 in states:
            initial_probability = q0[x0] * q0[x1]
            incremental_mean = 0.5 * (first_weight[x0] + first_weight[x1])
            resampling_probability = np.asarray(
                [first_weight[x0], first_weight[x1]], dtype=np.float64
            )
            resampling_probability /= resampling_probability.sum()
            initial = (x0, x1)
            for parent0 in range(2):
                for parent1 in range(2):
                    parent_probability = (
                        resampling_probability[parent0]
                        * resampling_probability[parent1]
                    )
                    for y0 in states:
                        for y1 in states:
                            mutation_probability = (
                                transition[initial[parent0], y0]
                                * transition[initial[parent1], y1]
                            )
                            final_mean = 0.5 * (
                                second_weight[y0] + second_weight[y1]
                            )
                            expected += (
                                initial_probability
                                * incremental_mean
                                * parent_probability
                                * mutation_probability
                                * final_mean
                            )
    assert expected == pytest.approx(target.sum(dtype=np.float64), rel=3e-15)


def test_constant_weight_is_exact_in_log_domain_beyond_exp_range():
    q0 = np.full(4, 0.25)
    log_target = np.log(q0) + 1000.0
    graph = UndirectedGraph(
        4, np.asarray([[0, 1], [1, 2], [2, 3], [0, 3]], dtype=np.int64)
    )
    result = run_island_annealed_smc(
        log_target,
        q0,
        betas=[0.0, 0.125, 0.5, 0.875, 1.0],
        n_islands=7,
        particles_per_island=11,
        seed=9,
        graph=graph,
        graph_move_probability=0.6,
        moves_per_stage=2,
    )

    np.testing.assert_array_equal(
        result.island_log_normalizer, np.full(7, 1000.0)
    )
    assert result.log_normalizer == 1000.0
    assert np.isinf(result.normalizer)
    assert result.variance == 0.0
    assert result.log_variance == -np.inf
    assert result.standard_error == 0.0
    assert np.isinf(result.island_normalizer).all()


def test_independent_island_variance_matches_direct_sample_variance():
    result = run_island_annealed_smc(
        np.log(np.asarray([0.2, 0.8, 1.4, 0.1])),
        np.asarray([0.4, 0.3, 0.2, 0.1]),
        betas=[0.0, 0.3, 0.7, 1.0],
        n_islands=12,
        particles_per_island=9,
        seed=71,
        object_id=5,
    )
    expected = np.var(result.island_normalizer, ddof=1) / 12.0
    assert result.variance == pytest.approx(expected, rel=2e-15)
    assert result.standard_error == pytest.approx(np.sqrt(expected), rel=2e-15)


def test_terminal_mutation_is_omitted_for_a_normalizer():
    result = run_island_annealed_smc(
        np.log(np.asarray([0.2, 0.8, 1.4, 0.1])),
        np.asarray([0.4, 0.3, 0.2, 0.1]),
        betas=[0.0, 0.25, 0.5, 0.75, 1.0],
        n_islands=4,
        particles_per_island=7,
        seed=73,
        moves_per_stage=3,
    )
    np.testing.assert_array_equal(
        result.stage_move_acceptance_rate[:, -1], np.zeros(4)
    )


def test_isolated_target_mode_retains_zero_islands_and_full_support():
    q0 = np.asarray([0.98, 0.01, 0.01])
    log_target = np.asarray([-np.inf, -np.inf, 0.0])
    graph = UndirectedGraph(3, np.asarray([[0, 1]], dtype=np.int64))
    result = run_island_annealed_smc(
        log_target,
        q0,
        betas=[0.0, 0.5, 1.0],
        n_islands=512,
        particles_per_island=1,
        seed=1234,
        graph=graph,
        graph_move_probability=0.9,
    )

    collapsed = result.collapsed_stage > 0
    assert collapsed.any()
    assert (~collapsed).any()
    assert np.all(result.island_log_normalizer[collapsed] == -np.inf)
    assert np.all(result.island_normalizer[collapsed] == 0.0)
    assert np.isfinite(result.log_normalizer)
    assert result.normalizer > 0.0


def test_all_collapsed_islands_return_an_unmodified_zero_estimate():
    result = run_island_annealed_smc(
        [-np.inf, 0.0],
        [1.0 - 1.0e-12, 1.0e-12],
        betas=[0.0, 0.5, 1.0],
        n_islands=4,
        particles_per_island=1,
        seed=88,
    )
    np.testing.assert_array_equal(result.collapsed_stage, np.ones(4, dtype=int))
    assert result.normalizer == 0.0
    assert result.log_normalizer == -np.inf
    assert result.variance == 0.0
    assert result.standard_error == 0.0


def test_keyed_pcg64_raw_record_is_frozen():
    generator = smc_module._keyed_rng(
        seed=(1 << 40) + 7,
        object_id=(1 << 39) + 11,
        island_id=13,
        stage=3,
        domain=5,
        move=2,
    )
    np.testing.assert_array_equal(
        generator.bit_generator.random_raw(6),
        np.asarray(
            [
                1581441401856123741,
                9205617658804630372,
                2809136180760135715,
                5530072429858218226,
                8375542589134624756,
                14136297808825406615,
            ],
            dtype=np.uint64,
        ),
    )


def test_object_island_order_and_outer_chunk_do_not_change_streams():
    q0 = np.asarray([0.1, 0.2, 0.3, 0.4])
    log_target = np.log(np.asarray([1.0, 0.4, 2.0, 0.3]))
    ids = np.asarray([4, 9, 15, 22, 37, 41], dtype=np.int64)
    graph = UndirectedGraph(
        4, np.asarray([[0, 1], [1, 2], [2, 3]], dtype=np.int64)
    )
    arguments = dict(
        betas=[0.0, 0.25, 0.5, 0.75, 1.0],
        n_islands=len(ids),
        particles_per_island=13,
        seed=(1 << 40) + 77,
        object_id=(1 << 39) + 19,
        graph=graph,
        graph_move_probability=0.55,
        moves_per_stage=2,
    )
    whole = run_island_annealed_smc(
        log_target, q0, island_ids=ids, island_chunk=6, **arguments
    )
    chunked = run_island_annealed_smc(
        log_target, q0, island_ids=ids, island_chunk=1, **arguments
    )
    reverse = run_island_annealed_smc(
        log_target, q0, island_ids=ids[::-1], island_chunk=2, **arguments
    )

    for name in (
        "island_log_normalizer",
        "stage_log_increment",
        "stage_effective_sample_size",
        "stage_move_acceptance_rate",
        "stage_unique_ancestor_count",
        "collapsed_stage",
        "terminal_particles",
    ):
        np.testing.assert_array_equal(getattr(whole, name), getattr(chunked, name))
        np.testing.assert_array_equal(
            getattr(whole, name), getattr(reverse, name)[::-1]
        )
    assert whole.normalizer == chunked.normalizer == reverse.normalizer
    assert whole.variance == chunked.variance == reverse.variance


def test_object_high_word_changes_keyed_stream():
    arguments = dict(
        log_target=np.log(np.asarray([0.3, 0.7, 1.1])),
        q0=np.asarray([0.2, 0.3, 0.5]),
        betas=[0.0, 0.5, 1.0],
        n_islands=4,
        particles_per_island=8,
        seed=3,
    )
    low = run_island_annealed_smc(object_id=17, **arguments)
    high = run_island_annealed_smc(object_id=(1 << 32) + 17, **arguments)
    assert not np.array_equal(low.terminal_particles, high.terminal_particles)


@pytest.mark.parametrize(
    ("log_target", "q0", "match"),
    (
        ([0.0, 1.0], [1.0, 0.0], "strictly positive"),
        ([0.0, 1.0], [1.0, np.nan], "finite and strictly positive"),
        ([0.0, 1.0], [1.0, np.inf], "finite and strictly positive"),
        ([0.0, 1.0], [1.0, -1.0], "finite and strictly positive"),
        ([0.0, np.nan], [0.5, 0.5], "finite values or -inf"),
        ([0.0, np.inf], [0.5, 0.5], "finite values or -inf"),
        ([-np.inf, -np.inf], [0.5, 0.5], "positive total mass"),
    ),
)
def test_invalid_target_or_base_support_fails_closed(log_target, q0, match):
    with pytest.raises(ValueError, match=match):
        run_island_annealed_smc(
            log_target,
            q0,
            betas=[0.0, 1.0],
            n_islands=2,
            particles_per_island=2,
            seed=1,
        )


@pytest.mark.parametrize(
    "betas",
    (
        [0.1, 1.0],
        [0.0, 0.5],
        [0.0, 0.5, 0.5, 1.0],
        [0.0, 0.7, 0.6, 1.0],
        [0.0, np.nan, 1.0],
    ),
)
def test_invalid_fixed_beta_schedule_fails_closed(betas):
    with pytest.raises(ValueError, match="strictly increasing vector"):
        run_island_annealed_smc(
            [0.0, 1.0],
            [0.5, 0.5],
            betas=betas,
            n_islands=2,
            particles_per_island=2,
            seed=1,
        )


def test_invalid_island_and_graph_contracts_fail_closed():
    with pytest.raises(ValueError, match="at least two"):
        run_island_annealed_smc(
            [0.0],
            [1.0],
            betas=[0.0, 1.0],
            n_islands=1,
            particles_per_island=1,
            seed=1,
        )
    with pytest.raises(ValueError, match="unique non-negative"):
        run_island_annealed_smc(
            [0.0],
            [1.0],
            betas=[0.0, 1.0],
            n_islands=2,
            particles_per_island=1,
            seed=1,
            island_ids=[3, 3],
        )
    with pytest.raises(ValueError, match="one-dimensional integer"):
        run_island_annealed_smc(
            [0.0],
            [1.0],
            betas=[0.0, 1.0],
            n_islands=2,
            particles_per_island=1,
            seed=1,
            island_ids=[1.0, 2.0],
        )
    with pytest.raises(ValueError, match="nonempty graph"):
        run_island_annealed_smc(
            [0.0],
            [1.0],
            betas=[0.0, 1.0],
            n_islands=2,
            particles_per_island=1,
            seed=1,
            graph_move_probability=0.5,
        )
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        run_island_annealed_smc(
            [0.0],
            [1.0],
            betas=[0.0, 1.0],
            n_islands=2,
            particles_per_island=1,
            seed=1,
            graph=UndirectedGraph(1, np.empty((0, 2), dtype=np.int64)),
            graph_move_probability=1.0,
        )


@pytest.mark.parametrize(
    ("n_states", "edges", "match"),
    (
        (3, np.asarray([[0.0, 1.0]]), "integer array"),
        (3, np.asarray([[0, 0]]), "self-edges"),
        (3, np.asarray([[0, 1], [1, 0]]), "unique"),
        (3, np.asarray([[0, 3]]), "outside"),
    ),
)
def test_invalid_undirected_graph_fails_closed(n_states, edges, match):
    with pytest.raises(ValueError, match=match):
        UndirectedGraph(n_states, edges)


def test_result_arrays_are_immutable():
    result = run_island_annealed_smc(
        [0.0, 1.0],
        [0.5, 0.5],
        betas=[0.0, 0.5, 1.0],
        n_islands=3,
        particles_per_island=4,
        seed=2,
    )
    assert not result.terminal_particles.flags.writeable
    assert not result.stage_log_increment.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        result.terminal_particles[0, 0] = 1
    with pytest.raises(ValueError, match="result identities"):
        replace(result, normalizer=result.normalizer + 1.0)
