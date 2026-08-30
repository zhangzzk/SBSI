from dataclasses import replace
from fractions import Fraction
from itertools import product

import numpy as np
import pytest

from sbsi.packed_integer_bridge_smc import (
    PACKED_BRIDGE_BETAS,
    PACKED_BRIDGE_LOG_CLIP,
    PACKED_BRIDGE_MOVE_STAGES,
    PACKED_BRIDGE_ROUNDS,
    PACKED_BRIDGE_SCORE_SCALE,
    PACKED_BRIDGE_TICKS,
    _inverse_tick_cdf,
    _quantize_resampling_log_weights,
    _unbiased_randbelow,
    integer_bridge_score,
    run_packed_integer_bridge_smc,
)


def _unique_problem(
    *, islands: int = 3, particles: int = 4, log_ratio: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    atoms = np.arange(
        islands * PACKED_BRIDGE_ROUNDS * particles, dtype=np.int64
    ).reshape(islands, PACKED_BRIDGE_ROUNDS, particles)
    ratios = np.full(atoms.shape, log_ratio, dtype=np.float64)
    return atoms, ratios


def _nonconstant_problem(
    *, islands: int = 3, particles: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    atoms = np.arange(
        islands * PACKED_BRIDGE_ROUNDS * particles, dtype=np.int64
    ).reshape(islands, PACKED_BRIDGE_ROUNDS, particles)
    ratios = ((atoms % 11) - 5).astype(np.float64) / 3.0
    return atoms, ratios


def _result_array_names():
    return (
        "island_ids",
        "island_normalizer",
        "island_log_normalizer",
        "stage_log_normalizer",
        "stage_effective_sample_size",
        "stage_acceptance_rate",
        "stage_movement_rate",
        "stage_unique_lineage_count",
        "stage_unique_state_count",
        "collapsed_stage",
        "stage_atom_ids",
        "stage_log_unnormalized_weight",
        "ancestor_index",
        "resampling_ticks",
        "resampling_u48",
        "resampling_tick_sum",
        "resampling_positive_count",
        "resampling_minimum_positive_tick",
        "resampling_terminal_correction",
        "mh_current_score",
        "mh_proposal_score",
        "mh_randbelow_value",
        "mh_randbelow_attempts",
        "mh_accepted",
        "mh_moved",
        "terminal_atom_ids",
        "terminal_log_ratio",
        "terminal_log_unnormalized_weight",
        "betas",
    )


def test_integer_bridge_score_is_positive_int64_and_zero_only_for_zero_target():
    values = np.array([-np.inf, -1.0e300, -2.0, 0.0, 3.0, 1.0e300])
    base = integer_bridge_score(values, 0)
    assert np.array_equal(base, np.ones(len(values), dtype=np.int64))

    for stage in range(1, PACKED_BRIDGE_MOVE_STAGES + 1):
        score = integer_bridge_score(values, stage)
        assert score.dtype == np.int64
        assert score[0] == 0
        assert np.all(score[1:] >= 1)
        assert np.all(score[1:] <= np.iinfo(np.int64).max)
        assert score[3] == PACKED_BRIDGE_SCORE_SCALE


def test_integer_bridge_score_uses_frozen_floor_clip_mapping():
    values = np.array([-400.0, -1.25, 0.25, 400.0])
    stage = 3
    exponent = np.clip(
        (stage / 8.0) * values,
        -PACKED_BRIDGE_LOG_CLIP,
        PACKED_BRIDGE_LOG_CLIP,
    )
    expected = np.maximum(
        np.floor(PACKED_BRIDGE_SCORE_SCALE * np.exp(exponent)), 1
    ).astype(np.int64)
    assert np.array_equal(integer_bridge_score(values, stage), expected)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_integer_bridge_score_rejects_invalid_logs(bad):
    with pytest.raises(ValueError, match=r"NaN/\+inf"):
        integer_bridge_score(np.array([bad]), 1)


@pytest.mark.parametrize("stage", [-1, 8, 9])
def test_integer_bridge_score_rejects_invalid_stage(stage):
    with pytest.raises(ValueError, match="stage"):
        integer_bridge_score(np.array([0.0]), stage)


def test_resampling_quantizer_reserves_support_and_exact_endpoint():
    log_weight = np.array([np.log(2.0), -np.inf, np.log(7.0), np.log(0.5)])
    law = _quantize_resampling_log_weights(log_weight)
    assert law.ticks.dtype == np.uint64
    assert int(law.ticks.sum(dtype=np.uint64)) == PACKED_BRIDGE_TICKS
    assert law.ticks[1] == 0
    assert np.all(law.ticks[[0, 2, 3]] >= 1)
    assert law.positive_count == 3
    assert law.minimum_positive_tick == int(law.ticks[[0, 2, 3]].min())
    assert law.ticks.flags.writeable is False


def test_resampling_quantizer_preserves_a_tick_after_float_underflow():
    law = _quantize_resampling_log_weights(np.array([0.0, -1.0e300, -np.inf]))
    assert law.positive_count == 2
    assert law.ticks[0] > 0
    assert law.ticks[1] == 1
    assert law.ticks[2] == 0
    assert int(law.ticks.sum(dtype=np.uint64)) == PACKED_BRIDGE_TICKS


def test_equal_power_of_two_particles_receive_equal_ticks():
    law = _quantize_resampling_log_weights(np.zeros(8))
    assert np.array_equal(
        law.ticks,
        np.full(8, PACKED_BRIDGE_TICKS // 8, dtype=np.uint64),
    )
    assert law.terminal_correction == 0


def test_resampling_quantizer_rejects_invalid_or_collapsed_measure():
    for values in (
        np.array([-np.inf, -np.inf]),
        np.array([0.0, np.nan]),
        np.array([0.0, np.inf]),
        np.empty((1, 2)),
    ):
        with pytest.raises(ValueError):
            _quantize_resampling_log_weights(values)


def test_integer_inverse_cdf_uses_right_open_tick_intervals():
    ticks = np.array([2, 0, 3, 1], dtype=np.uint64)
    scale = PACKED_BRIDGE_TICKS // 6
    ticks *= np.uint64(scale)
    ticks[-1] += np.uint64(PACKED_BRIDGE_TICKS - int(ticks.sum()))
    thresholds = np.array(
        [0, 2 * scale - 1, 2 * scale, 5 * scale - 1, 5 * scale],
        dtype=np.uint64,
    )
    selected = _inverse_tick_cdf(ticks, thresholds)
    assert np.array_equal(selected, np.array([0, 0, 2, 2, 3]))
    assert np.all(ticks[selected] > 0)


def test_auxiliary_resampling_correction_is_exact_by_fraction_enumeration():
    # Enumerate every parent vector for a two-child multinomial resample.  The
    # realized ticks need not match normalized [2, 5]; the inverse-r correction
    # still preserves both total mass and an arbitrary test function exactly.
    old_weight = (Fraction(2), Fraction(5))
    test_function = (Fraction(3), Fraction(11))
    law = _quantize_resampling_log_weights(np.log([2.0, 5.0]))
    ticks = tuple(int(value) for value in law.ticks)
    total = PACKED_BRIDGE_TICKS
    particles = 2

    expected_mass = Fraction(0)
    expected_function = Fraction(0)
    for parents in product(range(particles), repeat=particles):
        probability = Fraction(1)
        child_mass = Fraction(0)
        child_function = Fraction(0)
        for parent in parents:
            probability *= Fraction(ticks[parent], total)
            corrected = old_weight[parent] * total / (particles * ticks[parent])
            child_mass += corrected / particles
            child_function += corrected * test_function[parent] / particles
        expected_mass += probability * child_mass
        expected_function += probability * child_function

    assert expected_mass == sum(old_weight) / particles
    assert expected_function == sum(
        weight * function
        for weight, function in zip(old_weight, test_function)
    ) / particles


def test_integer_independence_mh_has_exact_detailed_balance():
    q0 = (Fraction(1, 5), Fraction(4, 5))
    score = (3, 11)
    left_to_right = (
        q0[0]
        * score[0]
        * q0[1]
        * min(Fraction(1), Fraction(score[1], score[0]))
    )
    right_to_left = (
        q0[1]
        * score[1]
        * q0[0]
        * min(Fraction(1), Fraction(score[0], score[1]))
    )
    assert left_to_right == right_to_left


@pytest.mark.parametrize("bound", [1, 2, 3, 17, 2**31 - 1, 2**62 - 57])
def test_unbiased_randbelow_is_keyed_deterministic_and_in_range(bound):
    first = _unbiased_randbelow(
        bound,
        seed=81,
        object_id=19,
        island_id=7,
        stage=4,
        particle=3,
    )
    second = _unbiased_randbelow(
        bound,
        seed=81,
        object_id=19,
        island_id=7,
        stage=4,
        particle=3,
    )
    assert first == second
    assert 0 <= first[0] < bound
    assert first[1] >= 1


def test_constant_ratio_has_exact_structure_zero_variance_and_no_terminal_move():
    atoms, ratios = _unique_problem(log_ratio=np.log(3.0))
    result = run_packed_integer_bridge_smc(atoms, ratios, seed=991, object_id=12)

    assert result.normalizer == pytest.approx(3.0, rel=3e-14)
    assert result.log_normalizer == pytest.approx(np.log(3.0), abs=3e-14)
    assert result.variance == 0.0
    assert result.standard_error == 0.0
    assert np.array_equal(result.collapsed_stage, np.zeros(3, dtype=np.int64))
    assert np.allclose(result.stage_effective_sample_size, 4.0, rtol=0.0, atol=1e-13)
    assert np.all(result.stage_acceptance_rate == 1.0)
    assert np.all(result.stage_movement_rate == 1.0)
    assert np.all(result.resampling_tick_sum == PACKED_BRIDGE_TICKS)
    assert np.all(result.resampling_ticks == PACKED_BRIDGE_TICKS // 4)
    assert np.array_equal(result.stage_atom_ids[:, -1], result.stage_atom_ids[:, -2])
    assert np.array_equal(result.terminal_atom_ids, result.stage_atom_ids[:, -1])
    assert result.proposal_slot_count == 3 * 8 * 4
    assert result.target_finite_slot_count == result.proposal_slot_count
    assert result.target_zero_slot_count == 0
    assert result.target_unique_atom_count == result.proposal_slot_count
    assert result.target_cache_hit_count == 0
    assert result.betas.flags.writeable is False
    assert result.stage_atom_ids.flags.writeable is False
    assert len(result.record_sha256) == 64
    assert len(result.trace_sha256) == 64


def test_acceptance_and_actual_movement_are_recorded_separately():
    atoms = np.full((2, 8, 4), 17, dtype=np.int64)
    ratios = np.zeros(atoms.shape)
    result = run_packed_integer_bridge_smc(atoms, ratios, seed=5)
    assert np.all(result.stage_acceptance_rate == 1.0)
    assert np.all(result.stage_movement_rate == 0.0)
    assert result.target_unique_atom_count == 1
    assert result.target_cache_hit_count == result.proposal_slot_count - 1


def test_true_zero_target_collapses_at_stage_one_without_false_precision_claim():
    atoms, ratios = _unique_problem(log_ratio=-np.inf)
    result = run_packed_integer_bridge_smc(atoms, ratios, seed=16)
    assert result.normalizer == 0.0
    assert np.isneginf(result.log_normalizer)
    assert result.variance == 0.0
    assert result.standard_error == 0.0
    assert np.array_equal(result.collapsed_stage, np.ones(3, dtype=np.int64))
    assert np.all(np.isneginf(result.stage_log_normalizer))
    assert not result.resampling_ticks.any()
    assert np.all(result.ancestor_index == -1)
    assert not result.mh_accepted.any()
    assert result.target_zero_slot_count == result.proposal_slot_count


def test_collapsed_island_is_retained_in_the_linear_mean_and_standard_error():
    atoms, ratios = _unique_problem(islands=2, particles=4)
    ratios[0] = -np.inf
    result = run_packed_integer_bridge_smc(atoms, ratios, seed=16)
    assert np.array_equal(result.collapsed_stage, np.array([1, 0]))
    assert np.allclose(result.island_normalizer, np.array([0.0, 1.0]), atol=1e-14)
    assert result.normalizer == pytest.approx(0.5)
    assert result.standard_error == pytest.approx(0.5)


def test_partial_zero_target_gets_zero_resampling_ticks_and_is_never_selected():
    atoms, ratios = _unique_problem(islands=2, particles=4)
    ratios[:, 0, 1::2] = -np.inf
    result = run_packed_integer_bridge_smc(atoms, ratios, seed=31)
    assert np.all(result.resampling_ticks[:, 0, 1::2] == 0)
    assert np.all(result.resampling_ticks[:, 0, 0::2] >= 1)
    assert np.all(result.resampling_positive_count[:, 0] == 2)
    assert not np.isin(result.ancestor_index[:, 0], np.array([1, 3])).any()


def test_zero_score_independence_proposals_are_rejected():
    atoms, ratios = _unique_problem(islands=2, particles=4)
    ratios[:, 1:, :] = -np.inf
    result = run_packed_integer_bridge_smc(atoms, ratios, seed=9)
    assert not result.mh_accepted.any()
    assert not result.mh_moved.any()
    assert np.all(result.mh_proposal_score == 0)
    assert np.all(result.mh_randbelow_attempts == 0)
    assert np.all(result.collapsed_stage == 0)


@pytest.mark.parametrize("log_ratio", [-100.0, 100.0])
def test_terminal_stage_is_unclipped_after_clipped_intermediate_bridges(log_ratio):
    atoms, ratios = _unique_problem(islands=2, particles=4, log_ratio=log_ratio)
    result = run_packed_integer_bridge_smc(atoms, ratios, seed=67)
    assert result.log_normalizer == pytest.approx(log_ratio, abs=3e-13)
    assert result.normalizer == pytest.approx(np.exp(log_ratio), rel=3e-13)
    assert np.all(result.collapsed_stage == 0)
    if log_ratio > 0:
        score = integer_bridge_score(np.array([log_ratio]), 7)[0]
        clipped = np.floor(
            PACKED_BRIDGE_SCORE_SCALE * np.exp(PACKED_BRIDGE_LOG_CLIP)
        )
        assert score == int(clipped)


def test_duplicate_atom_target_values_must_be_identical():
    atoms, ratios = _unique_problem(islands=2, particles=2)
    atoms[0, 1, 0] = atoms[0, 0, 0]
    ratios[0, 1, 0] = ratios[0, 0, 0] + 1.0
    with pytest.raises(ValueError, match="duplicate atom ids"):
        run_packed_integer_bridge_smc(atoms, ratios, seed=1)


def test_island_input_order_is_canonical_and_bitwise_replayable():
    atoms, ratios = _nonconstant_problem(islands=3, particles=4)
    ids = np.array([19, 3, 11], dtype=np.int64)
    first = run_packed_integer_bridge_smc(
        atoms, ratios, seed=52, object_id=77, island_ids=ids, island_chunk=2
    )
    permutation = np.array([2, 0, 1])
    second = run_packed_integer_bridge_smc(
        atoms[permutation],
        ratios[permutation],
        seed=52,
        object_id=77,
        island_ids=ids[permutation],
        island_chunk=2,
    )
    for name in _result_array_names():
        assert np.array_equal(getattr(first, name), getattr(second, name))
    assert first.record_sha256 == second.record_sha256
    assert first.execution_sha256 == second.execution_sha256


def test_island_chunk_changes_only_execution_identity():
    atoms, ratios = _nonconstant_problem(islands=4, particles=4)
    whole = run_packed_integer_bridge_smc(
        atoms, ratios, seed=127, object_id=8, island_chunk=4
    )
    split = run_packed_integer_bridge_smc(
        atoms, ratios, seed=127, object_id=8, island_chunk=1
    )
    for name in _result_array_names():
        assert np.array_equal(getattr(whole, name), getattr(split, name))
    assert whole.normalizer == split.normalizer
    assert whole.variance == split.variance
    assert whole.trace_sha256 == split.trace_sha256
    assert whole.record_sha256 == split.record_sha256
    assert whole.execution_sha256 != split.execution_sha256


def test_seed_and_absolute_object_id_change_only_random_trace_not_inputs():
    atoms, ratios = _nonconstant_problem(islands=3, particles=5)
    base = run_packed_integer_bridge_smc(atoms, ratios, seed=4, object_id=9)
    seed_changed = run_packed_integer_bridge_smc(
        atoms, ratios, seed=5, object_id=9
    )
    object_changed = run_packed_integer_bridge_smc(
        atoms, ratios, seed=4, object_id=10
    )
    assert base.proposal_atom_ids_sha256 == seed_changed.proposal_atom_ids_sha256
    assert base.target_log_ratio_sha256 == object_changed.target_log_ratio_sha256
    assert base.trace_sha256 != seed_changed.trace_sha256
    assert base.trace_sha256 != object_changed.trace_sha256


def test_result_validation_detects_trace_tampering():
    atoms, ratios = _nonconstant_problem(islands=2, particles=4)
    result = run_packed_integer_bridge_smc(atoms, ratios, seed=13)
    tampered = result.stage_atom_ids.copy()
    tampered[0, 0, 0] += 1
    with pytest.raises(ValueError, match="audit|record hash"):
        replace(result, stage_atom_ids=tampered)


@pytest.mark.parametrize(
    ("atoms", "ratios", "message"),
    [
        (np.zeros((2, 7, 3), dtype=np.int64), np.zeros((2, 7, 3)), "shape"),
        (np.zeros((1, 8, 3), dtype=np.int64), np.zeros((1, 8, 3)), "two islands"),
        (np.zeros((2, 8, 0), dtype=np.int64), np.zeros((2, 8, 0)), "one particle"),
        (np.zeros((2, 8, 3)), np.zeros((2, 8, 3)), "integer"),
        (
            -np.ones((2, 8, 3), dtype=np.int64),
            np.zeros((2, 8, 3)),
            "non-negative",
        ),
    ],
)
def test_run_rejects_invalid_proposal_shapes_and_ids(atoms, ratios, message):
    with pytest.raises(ValueError, match=message):
        run_packed_integer_bridge_smc(atoms, ratios, seed=1)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_run_rejects_nan_or_positive_infinite_target(bad):
    atoms, ratios = _unique_problem(islands=2, particles=2)
    ratios[0, 0, 0] = bad
    with pytest.raises(ValueError, match=r"NaN/\+inf"):
        run_packed_integer_bridge_smc(atoms, ratios, seed=1)


def test_run_rejects_invalid_rng_and_island_identity():
    atoms, ratios = _unique_problem(islands=2, particles=2)
    with pytest.raises(ValueError, match="seed"):
        run_packed_integer_bridge_smc(atoms, ratios, seed=-1)
    with pytest.raises(ValueError, match="object_id"):
        run_packed_integer_bridge_smc(atoms, ratios, seed=1, object_id=-1)
    with pytest.raises(ValueError, match="island_ids"):
        run_packed_integer_bridge_smc(
            atoms, ratios, seed=1, island_ids=np.array([2, 2])
        )
    with pytest.raises(ValueError, match="island_chunk"):
        run_packed_integer_bridge_smc(atoms, ratios, seed=1, island_chunk=0)


def test_beta_and_round_contract_is_exact():
    assert np.array_equal(PACKED_BRIDGE_BETAS, np.arange(9) / 8.0)
    assert PACKED_BRIDGE_BETAS.flags.writeable is False
    assert PACKED_BRIDGE_ROUNDS == 8
    assert PACKED_BRIDGE_MOVE_STAGES == 7
