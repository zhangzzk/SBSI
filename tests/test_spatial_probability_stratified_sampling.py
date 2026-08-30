from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest
import torch

from sbsi.catalogue_sampling import (
    SPATIAL_CELL_COUNT,
    SPATIAL_PS_CELLS_PER_SLAB,
    SPATIAL_PS_DRAWS_PER_CELL,
    SPATIAL_PS_MICROSTRATA,
    SPATIAL_PS_ORDER_CONTRACT,
    SPATIAL_PS_REPLICATES,
    SPATIAL_PS_SEED,
    SPATIAL_PS_SOURCE_COUNTS,
    SPATIAL_PS_TICKS_PER_MICROSTRATUM,
    SPATIAL_PS_VERSION,
    DefensiveLocalProposal,
    ProposalCandidates,
    ProposalCoordinateTable,
    SpatialProbabilityStratifiedMISDraw,
    _spatial_ps_inverse_integer_cdf,
    _spatial_ps_quantize_prior,
    _spatial_ps_quantize_prior_tensor,
    _spatial_ps_quantize_proxy,
    _spatial_ps_quantize_proxy_tensor,
    _spatial_ps_rng_u48,
    _spatial_ps_source_total_ticks,
    _spatial_ps_thresholds,
    reduce_spatial_probability_stratified_mis,
)


def _proposal(
    n_atoms=512,
    *,
    values=None,
    dispersion=None,
    detection=None,
):
    if values is None:
        values = np.zeros((n_atoms, 4), dtype=np.float64)
    if dispersion is None:
        parity = np.where(np.arange(n_atoms) % 2, 1.3, 0.7)
        dispersion = np.repeat(parity[:, None], 4, axis=1)
    if detection is None:
        detection = np.ones(n_atoms, dtype=np.float64)
    coordinates = ProposalCoordinateTable(
        values=np.asarray(values, dtype=np.float64),
        target_names=("x0", "x1", "x2", "x3"),
        center=np.zeros(4),
        scale=np.ones(4),
        dispersion=np.asarray(dispersion, dtype=np.float64),
        statistic="mean",
    )
    return DefensiveLocalProposal(
        coordinates,
        np.ones(n_atoms, dtype=np.float64),
        local_base_weights=np.asarray(detection, dtype=np.float64),
        build_coordinate_tree=False,
    )


def _candidates(rows):
    indices = np.asarray(rows, dtype=np.int64)
    return ProposalCandidates(
        indices=indices,
        distances=np.zeros(indices.shape, dtype=np.float64),
        radius=np.ones(len(indices), dtype=np.float64),
    )


@pytest.fixture(scope="module")
def uniform_draw():
    proposal = _proposal()
    draw = proposal.draw_spatial_probability_stratified_mis(
        _candidates([[0]]),
        np.array([[0.2, -0.1, 0.3, 0.0]], dtype=np.float64),
        object_ids=np.array([41]),
        atom_chunk=8,
        object_chunk=1,
    )
    return proposal, draw


def test_frozen_contract_and_golden_pcg64_u48_record():
    assert SPATIAL_PS_VERSION == "h256_u8_ps2_v1"
    assert SPATIAL_PS_MICROSTRATA == (16, 12, 4)
    assert SPATIAL_PS_SOURCE_COUNTS == (32, 24, 8)
    assert SPATIAL_PS_REPLICATES == 2
    assert SPATIAL_PS_DRAWS_PER_CELL == 64
    assert SPATIAL_PS_CELLS_PER_SLAB == 4
    assert SPATIAL_PS_ORDER_CONTRACT.startswith("ascending_active_atom_id")
    np.testing.assert_array_equal(
        _spatial_ps_rng_u48(SPATIAL_PS_SEED, 41, 7, 0)[:8],
        np.array(
            [
                171466208571840,
                230823947617707,
                129872469319945,
                208658339143896,
                204995807249154,
                273919057553559,
                122331966024552,
                274005793039133,
            ],
            dtype=np.uint64,
        ),
    )
    assert not np.array_equal(
        _spatial_ps_rng_u48(SPATIAL_PS_SEED, 41, 7, 0),
        _spatial_ps_rng_u48(SPATIAL_PS_SEED, 41, 7, 1),
    )


def test_prior_quantizer_defends_tiny_atoms_and_exclusions():
    weights = np.array(
        [1.0e-300, 8.0, 1.0e300, 3.0, 1.0e-200, 2.0],
        dtype=np.float64,
    )
    eligible = np.array([True, False, True, False, True, False])
    ticks, correction = _spatial_ps_quantize_prior(weights, eligible)
    assert int(ticks.sum()) == _spatial_ps_source_total_ticks(2)
    assert np.all(ticks[eligible] >= 1)
    np.testing.assert_array_equal(ticks[~eligible], 0)
    assert correction == 0
    # The trailing excluded atom remains a plateau rather than receiving an
    # endpoint-rounding residual.
    assert ticks[-1] == 0


def test_proxy_quantizer_plateaus_endpoint_and_b12_are_integer_exact():
    eligible = np.array([True, True, False, True, False])
    weights = np.array([0.0, 0.25, 9.0, 0.75, 7.0])
    ticks, correction = _spatial_ps_quantize_proxy(
        weights, eligible, source=1
    )
    total = _spatial_ps_source_total_ticks(1)
    assert int(ticks.sum()) == total
    assert ticks[0] == 0
    assert ticks[2] == ticks[4] == 0
    assert correction == 0
    selected = _spatial_ps_inverse_integer_cdf(
        ticks, np.array([0, total - 1], dtype=np.int64)
    )
    np.testing.assert_array_equal(selected, [1, 3])

    u48 = np.zeros(SPATIAL_PS_SOURCE_COUNTS[1], dtype=np.uint64)
    u48[1::2] = SPATIAL_PS_TICKS_PER_MICROSTRATUM - 1
    thresholds = _spatial_ps_thresholds(1, u48)
    expected = np.repeat(
        np.arange(12, dtype=np.int64)
        * SPATIAL_PS_TICKS_PER_MICROSTRATUM,
        2,
    )
    expected[1::2] += SPATIAL_PS_TICKS_PER_MICROSTRATUM - 1
    np.testing.assert_array_equal(thresholds, expected)
    assert thresholds[-1] == total - 1


def test_integer_inverse_cdf_endpoint_and_boundary_spanning_duplicates():
    tick = SPATIAL_PS_TICKS_PER_MICROSTRATUM
    total = _spatial_ps_source_total_ticks(2)
    ticks = np.array([tick, tick, total - 2 * tick], dtype=np.int64)
    np.testing.assert_array_equal(
        _spatial_ps_inverse_integer_cdf(
            ticks, np.array([tick - 1, tick], dtype=np.int64)
        ),
        [0, 1],
    )

    spanning = np.array([tick // 2, tick, total - 3 * tick // 2])
    u48 = np.zeros(SPATIAL_PS_SOURCE_COUNTS[2], dtype=np.uint64)
    u48[0] = tick - 1
    thresholds = _spatial_ps_thresholds(2, u48)
    selected = _spatial_ps_inverse_integer_cdf(spanning, thresholds)
    # b=0 at T-1 and b=1 at T both land on the atom spanning that boundary.
    assert selected[0] == selected[2] == 1


def test_tensor_quantizers_fail_audit_on_bad_raw_weights():
    eligible = torch.tensor([[True, True, True]])
    bad_proxy = torch.tensor([[0.5, -1.0e-300, 0.5]], dtype=torch.float64)
    *_, valid = _spatial_ps_quantize_proxy_tensor(
        bad_proxy, eligible, source=0
    )
    assert not bool(valid[0])

    bad_prior = torch.tensor([[0.5, 0.0, 0.5]], dtype=torch.float64)
    *_, valid = _spatial_ps_quantize_prior_tensor(bad_prior, eligible)
    assert not bool(valid[0])

    ineligible_nonzero = torch.tensor([[0.5, 0.1, 0.5]], dtype=torch.float64)
    mask = torch.tensor([[True, False, True]])
    *_, valid = _spatial_ps_quantize_proxy_tensor(
        ineligible_nonzero, mask, source=0
    )
    assert not bool(valid[0])


def test_core_draw_layout_exact_exclusion_dyadic_q_and_selected_law(uniform_draw):
    proposal, draw = uniform_draw
    assert isinstance(draw, SpatialProbabilityStratifiedMISDraw)
    assert draw.indices.shape == (1, 16_384)
    assert draw.selected_source_ticks.shape == (1, 16_384, 3)
    assert draw.cells_per_slab == 4
    assert draw.version == SPATIAL_PS_VERSION
    assert draw.order_contract == SPATIAL_PS_ORDER_CONTRACT
    assert not np.isin(draw.indices, [0]).any()
    np.testing.assert_array_equal(
        draw.probability_numerator,
        draw.selected_source_ticks.sum(axis=2, dtype=np.uint64),
    )
    np.testing.assert_array_equal(
        draw.probability,
        np.ldexp(draw.probability_numerator.astype(np.float64), -53),
    )
    np.testing.assert_array_equal(draw.source_tick_sum[:, :, 0], 16 << 48)
    np.testing.assert_array_equal(draw.source_tick_sum[:, :, 1], 12 << 48)
    np.testing.assert_array_equal(draw.source_tick_sum[:, :, 2], 4 << 48)
    assert draw.source_audit_passed.all()
    assert not draw.fallback_scaled_law_mismatch_count.any()
    assert len(proposal._spatial_ps_cell_order) == 4
    assert getattr(proposal, "_spatial_cell_order", None) is None

    for cell in range(SPATIAL_CELL_COUNT):
        block = slice(cell * 64, (cell + 1) * 64)
        np.testing.assert_array_equal(
            np.bincount(draw.component_id[block], minlength=3), [32, 24, 8]
        )
        # Two-atom toy cells necessarily contain duplicates; they must survive.
        assert len(np.unique(draw.indices[0, block])) <= 2


def test_four_cell_slab_limit_is_enforced_before_sampling():
    proposal = _proposal()
    with pytest.raises(ValueError, match="four-cell slab"):
        proposal.draw_spatial_probability_stratified_mis(
            _candidates([[0]]),
            np.zeros((1, 4)),
            atom_chunk=7,
            object_chunk=1,
        )


def test_exclusions_are_bucketed_once_instead_of_rescanned_per_cell(monkeypatch):
    proposal = _proposal()
    partition = proposal.spatial_cell_partition()
    candidates = _candidates([np.arange(0, 128, 2, dtype=np.int64)])
    original = np.searchsorted
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(np, "searchsorted", counted)
    draw = proposal.draw_spatial_probability_stratified_mis(
        candidates,
        np.zeros((1, 4)),
        partition=partition,
        object_ids=np.array([41]),
        atom_chunk=8,
        object_chunk=1,
    )
    # The sole NumPy search is the one-time active-id lookup. Integer-CDF
    # searches are torch kernels; no cell rescans invoke NumPy searchsorted.
    assert calls == 1
    assert not np.isin(draw.indices, candidates.indices[0]).any()


@pytest.mark.parametrize("first", ["archived", "probability_stratified"])
def test_archived_and_probability_stratified_cell_order_caches_are_isolated(first):
    proposal = _proposal()
    candidates = _candidates([[0]])
    observed = np.zeros((1, 4), dtype=np.float64)

    def archived():
        proposal.draw_spatial_mis_reservoirs(
            candidates,
            observed,
            object_ids=np.array([41]),
            atom_chunk=2,
            object_chunk=1,
        )

    def probability_stratified():
        proposal.draw_spatial_probability_stratified_mis(
            candidates,
            observed,
            object_ids=np.array([41]),
            atom_chunk=8,
            object_chunk=1,
        )

    second = (
        probability_stratified if first == "archived" else archived
    )
    (archived if first == "archived" else probability_stratified)()
    first_archived_cache = getattr(proposal, "_spatial_cell_order", None)
    first_ps_cache = getattr(proposal, "_spatial_ps_cell_order", None)
    second()
    assert len(proposal._spatial_cell_order) == 3
    assert len(proposal._spatial_ps_cell_order) == 4
    if first == "archived":
        assert proposal._spatial_cell_order is first_archived_cache
    else:
        assert proposal._spatial_ps_cell_order is first_ps_cache


def test_scaled_prior_fallback_is_full_law_exact():
    with np.errstate(over="ignore", invalid="ignore"):
        proposal = _proposal(
            values=np.full((512, 4), 1.0e200),
            dispersion=np.ones((512, 4)),
        )
        draw = proposal.draw_spatial_probability_stratified_mis(
            _candidates(np.empty((1, 0), dtype=np.int64)),
            np.zeros((1, 4)),
            object_ids=np.array([9]),
            atom_chunk=8,
            object_chunk=1,
        )
    assert draw.proxy_fallback.all()
    assert not draw.fallback_scaled_law_mismatch_count.any()
    np.testing.assert_array_equal(
        draw.selected_source_ticks[:, :, 0],
        4 * draw.selected_source_ticks[:, :, 2],
    )
    np.testing.assert_array_equal(
        draw.selected_source_ticks[:, :, 1],
        3 * draw.selected_source_ticks[:, :, 2],
    )


def test_object_chunking_does_not_change_draws_or_object_hashes():
    proposal = _proposal()
    observed = np.array([[0.1, 0.0, 0.0, 0.0], [-0.2, 0.3, 0.0, 0.1]])
    candidates = _candidates([[0], [1]])
    object_ids = np.array([41, 99])
    first = proposal.draw_spatial_probability_stratified_mis(
        candidates,
        observed,
        object_ids=object_ids,
        atom_chunk=8,
        object_chunk=1,
    )
    second = proposal.draw_spatial_probability_stratified_mis(
        candidates,
        observed,
        object_ids=object_ids,
        atom_chunk=8,
        object_chunk=2,
    )
    for name in (
        "indices",
        "probability",
        "probability_numerator",
        "selected_source_ticks",
        "selected_source_cdf_lower",
        "u48",
        "component_log_normalizer",
        "source_tick_sum",
        "source_terminal_correction",
    ):
        np.testing.assert_array_equal(getattr(first, name), getattr(second, name))
    assert first.object_indices_sha256 == second.object_indices_sha256
    assert first.object_probability_sha256 == second.object_probability_sha256
    assert first.object_selected_source_ticks_sha256 == (
        second.object_selected_source_ticks_sha256
    )
    # Execution chunking is provenance, so whole-call records remain distinct.
    assert first.record_sha256 != second.record_sha256


def test_draw_rejects_probability_mapping_and_order_provenance_mutations(
    uniform_draw,
):
    _, draw = uniform_draw
    numerator = draw.probability_numerator.copy()
    numerator[0, 0] += 1
    with pytest.raises(ValueError, match="invalid values"):
        replace(draw, probability_numerator=numerator)

    lower = draw.selected_source_cdf_lower.copy()
    lower[0, 0] += draw.selected_source_ticks[0, 0, draw.component_id[0]]
    with pytest.raises(ValueError, match="invalid values"):
        replace(draw, selected_source_cdf_lower=lower)

    with pytest.raises(ValueError, match="unsupported"):
        replace(draw, order_contract="atom_order_v0")

    with pytest.raises(ValueError, match="provenance"):
        replace(draw, schedule_sha256="0" * 64)


def test_fraction_identity_proves_integer_realized_law_unbiasedness():
    eligible = np.ones(4, dtype=bool)
    prior, _ = _spatial_ps_quantize_prior(
        np.array([1.0e-200, 1.0, 3.0, 2.0]), eligible
    )
    proxy0, _ = _spatial_ps_quantize_proxy(
        np.array([0.0, 1.0, 2.0, 5.0]), eligible, source=0
    )
    proxy1, _ = _spatial_ps_quantize_proxy(
        np.array([4.0, 0.0, 1.0, 2.0]), eligible, source=1
    )
    laws = (proxy0, proxy1, prior)
    numerator = proxy0 + proxy1 + prior
    target = (Fraction(1, 7), Fraction(2, 5), Fraction(3, 11), Fraction(5, 13))
    expected = Fraction(0)
    tick = SPATIAL_PS_TICKS_PER_MICROSTRATUM
    for source, (law, n_strata) in enumerate(
        zip(laws, SPATIAL_PS_MICROSTRATA)
    ):
        lower = 0
        intervals = []
        for mass in law:
            upper = lower + int(mass)
            intervals.append((lower, upper))
            lower = upper
        for microstratum in range(n_strata):
            stratum_lower = microstratum * tick
            stratum_upper = stratum_lower + tick
            for atom, (atom_lower, atom_upper) in enumerate(intervals):
                overlap = max(
                    0,
                    min(atom_upper, stratum_upper)
                    - max(atom_lower, stratum_lower),
                )
                if overlap:
                    y = target[atom] * Fraction(1 << 53, int(numerator[atom]))
                    expected += Fraction(2, 64) * Fraction(overlap, tick) * y
    assert expected == sum(target, Fraction(0))


def test_paired_reducer_uses_32_within_microstratum_differences(uniform_draw):
    _, draw = uniform_draw
    values = np.empty(draw.indices.shape, dtype=np.float64)
    paired = values.reshape(1, SPATIAL_CELL_COUNT, 32, 2)
    paired[:, :, :, 0] = 1.0
    paired[:, :, :, 1] = 3.0
    reduced = reduce_spatial_probability_stratified_mis(values, draw)
    np.testing.assert_array_equal(reduced.cell_estimate, 2.0)
    np.testing.assert_array_equal(
        reduced.cell_pair_squared_difference_sum, 128.0
    )
    np.testing.assert_array_equal(reduced.cell_variance, 128.0 / 64**2)
    np.testing.assert_allclose(reduced.estimate, 512.0)
    np.testing.assert_allclose(reduced.variance, 8.0)
    with pytest.raises(TypeError, match="requires"):
        reduce_spatial_probability_stratified_mis(values, object())
