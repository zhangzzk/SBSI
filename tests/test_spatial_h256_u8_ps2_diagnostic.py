from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import sbsi.spatial_h256_u8_ps2_diagnostic as diagnostic_module
from sbsi.catalogue_sampling import pareto_tail_index
from sbsi.spatial_h256_u8_ps2_diagnostic import (
    SPATIAL_H256_U8_PS2_BALANCE_TICKS,
    SPATIAL_H256_U8_PS2_CDF_TICKS,
    SPATIAL_H256_U8_PS2_CENTER_GATE_THRESHOLDS,
    SPATIAL_H256_U8_PS2_METHOD_TIME_COMPONENTS,
    SPATIAL_H256_U8_PS2_PROTOCOL,
    SPATIAL_H256_U8_PS2_SOURCE_DRAWS,
    SPATIAL_H256_U8_PS2_SOURCE_MICROSTRATA,
    _integer_contract_consistent,
    _reduce_h256_u8_ps2_log_values,
    _source_isolated_core_crosscheck,
    spatial_h256_u8_ps2_center_summary,
    spatial_h256_u8_ps2_run_manifest_sha256,
)
from sbsi.complement_diagnostic import exact_oracle_object_ids
from sbsi.spatial_neyman_diagnostic import (
    SPATIAL_NEYMAN_CENTER_GATE_THRESHOLDS,
    SPATIAL_NEYMAN_PROTOCOL,
)


def _schedule(cells: int):
    cell_id = []
    source_id = []
    microstratum_id = []
    pair_draw_id = []
    for cell in range(cells):
        for source, count in enumerate(SPATIAL_H256_U8_PS2_SOURCE_MICROSTRATA):
            for microstratum in range(count):
                for draw in range(2):
                    cell_id.append(cell)
                    source_id.append(source)
                    microstratum_id.append(microstratum)
                    pair_draw_id.append(draw)
    return tuple(
        np.asarray(values, dtype=np.int64)
        for values in (cell_id, source_id, microstratum_id, pair_draw_id)
    )


def _digest(label: str) -> str:
    return sha256(label.encode()).hexdigest()


def _passing_diagnostic():
    protocol = deepcopy(SPATIAL_H256_U8_PS2_PROTOCOL)
    object_ids = np.arange(25_000, 25_512, dtype=np.int64)
    selected_ids, _ = exact_oracle_object_ids(25_000, 25_512)
    selected = np.isin(object_ids, selected_ids)
    z = np.zeros(len(object_ids), dtype=np.float64)
    z[selected] = np.resize(np.asarray([-1.0, 1.0]), 64)
    complement = np.ones(len(object_ids), dtype=np.float64)
    standard_error = np.full(len(object_ids), 1.0e-3)
    exact_complement = complement - z * standard_error
    stratum = np.full(len(object_ids), 9.0)
    exact_total = stratum + exact_complement
    columns: dict[str, object] = {
        "object_index": object_ids,
        "stratum_count": np.full(len(object_ids), 16_384, dtype=np.int64),
        "stratum_unique_count": np.full(
            len(object_ids), 16_384, dtype=np.int64
        ),
        "stratum_active_count": np.full(
            len(object_ids), 16_384, dtype=np.int64
        ),
        "production_draws": np.full(len(object_ids), 16_384, dtype=np.int64),
        "production_cells": np.full(len(object_ids), 256, dtype=np.int64),
        "production_microstrata": np.full(
            len(object_ids), 8_192, dtype=np.int64
        ),
        "production_source_proxy_t1": np.full(
            len(object_ids), 8_192, dtype=np.int64
        ),
        "production_source_proxy_t2": np.full(
            len(object_ids), 6_144, dtype=np.int64
        ),
        "production_source_prior": np.full(
            len(object_ids), 2_048, dtype=np.int64
        ),
        "production_draws_in_stratum": np.zeros(
            len(object_ids), dtype=np.int64
        ),
        "source_audit_failed_count": np.zeros(len(object_ids), dtype=np.int64),
        "fallback_scaled_law_mismatch_count": np.zeros(
            len(object_ids), dtype=np.int64
        ),
        "source_excluded_tick_sum": np.zeros(len(object_ids), dtype=np.int64),
        "selected_source_tick_overflow_count": np.zeros(
            len(object_ids), dtype=np.int64
        ),
        "numerator_sum_failure_count": np.zeros(
            len(object_ids), dtype=np.int64
        ),
        "probability_identity_failure_count": np.zeros(
            len(object_ids), dtype=np.int64
        ),
        "cdf_bracket_failure_count": np.zeros(len(object_ids), dtype=np.int64),
        "rng_recomputation_failure_count": np.zeros(
            len(object_ids), dtype=np.int64
        ),
        "partition_cell_membership_failure_count": np.zeros(
            len(object_ids), dtype=np.int64
        ),
        "probability_numerator_minimum": np.ones(
            len(object_ids), dtype=np.int64
        ),
        "probability_numerator_maximum": np.full(
            len(object_ids), 1 << 52, dtype=np.int64
        ),
        "u48_minimum": np.zeros(len(object_ids), dtype=np.int64),
        "u48_maximum": np.full(
            len(object_ids), (1 << 48) - 1, dtype=np.int64
        ),
        "prior_minimum_eligible_tick": np.ones(
            len(object_ids), dtype=np.int64
        ),
        "terminal_correction_maximum": np.ones(
            len(object_ids), dtype=np.int64
        ),
        "terminal_correction_positive_count": np.ones(
            len(object_ids), dtype=np.int64
        ),
        "proxy_t1_prior_fallback_cells": np.zeros(
            len(object_ids), dtype=np.int64
        ),
        "proxy_t2_prior_fallback_cells": np.zeros(
            len(object_ids), dtype=np.int64
        ),
        "canonical_schedule_exact": np.ones(len(object_ids), dtype=bool),
        "integer_source_totals_exact": np.ones(len(object_ids), dtype=bool),
        "integer_prior_floor_satisfied": np.ones(len(object_ids), dtype=bool),
        "integer_proxy_fallback_scaled_exact": np.ones(
            len(object_ids), dtype=bool
        ),
        "integer_balance_probability_exact": np.ones(
            len(object_ids), dtype=bool
        ),
        "integer_cdf_brackets_exact": np.ones(len(object_ids), dtype=bool),
        "partition_cell_membership_exact": np.ones(
            len(object_ids), dtype=bool
        ),
        "core_cell_mass_crosscheck_relative_max": np.zeros(len(object_ids)),
        "core_cell_pair_sum_crosscheck_relative_max": np.zeros(len(object_ids)),
        "log_mass_stratum": np.log(stratum),
        "mass_stratum": stratum,
        "log_mass_complement_spatial_h256_u8_ps2": np.log(complement),
        "mass_complement_spatial_h256_u8_ps2": complement,
        "mass_complement_spatial_h256_u8_ps2_standard_error": standard_error,
        "log_variance_complement_spatial_h256_u8_ps2": np.full(
            len(object_ids), np.log(1.0e-6)
        ),
        "pareto_k_spatial_h256_u8_ps2_complement": np.full(
            len(object_ids), 0.1
        ),
        "exact_oracle_selected": selected,
        "mass_stratum_exact_oracle": stratum,
        "mass_stratum_exact_oracle_relative_error": np.zeros(len(object_ids)),
        "mass_complement_exact_oracle": exact_complement,
        "mass_total_exact_oracle": exact_total,
        "mass_complement_spatial_h256_u8_ps2_minus_exact_z": z,
        "delta_log_total_evidence_spatial_h256_u8_ps2_minus_exact": np.log(
            (stratum + complement) / exact_total
        ),
    }
    for name in diagnostic_module._SPATIAL_H256_U8_PS2_MANIFEST_HASH_COLUMNS:
        columns[name] = [
            _digest(f"{name}:{object_id}") for object_id in object_ids
        ]
    support = [_digest(f"support:{object_id}") for object_id in object_ids]
    columns["stratum_indices_sha256"] = support
    columns["production_exclusion_indices_sha256"] = support.copy()
    frame = pd.DataFrame(columns)
    calls = []
    for index in range(16):
        start = 25_000 + 32 * index
        record_sha256 = _digest(f"record:{index}")
        schedule_sha256 = _digest(f"schedule:{index}")
        cell_order_sha256 = _digest(f"order:{index}")
        calls.append(
            {
                "object_start": start,
                "object_stop": start + 32,
                "objects": 32,
                "seed": protocol["production_seed"],
                "atom_chunk": protocol["spatial_atom_chunk"],
                "object_chunk": protocol["spatial_object_chunk"],
                "cells_per_slab": protocol["cells_per_slab"],
                "partition_sha256": protocol["partition_sha256"],
                "record_sha256": record_sha256,
                "schedule_sha256": schedule_sha256,
                "quantization_sha256": _digest(f"quantization:{index}"),
                "cell_order_sha256": cell_order_sha256,
                "sampler_version": protocol["sampler_version"],
                "rng_contract": protocol["rng_contract"],
                "quantizer_contract": protocol["quantizer_contract"],
                "hash_contract": protocol["sampler_hash_contract"],
                "order_contract": protocol["order_contract"],
            }
        )
        block = frame.index[32 * index : 32 * (index + 1)]
        frame.loc[block, "production_draw_record_sha256"] = record_sha256
        frame.loc[block, "production_schedule_sha256"] = schedule_sha256
        frame.loc[block, "production_cell_order_sha256"] = cell_order_sha256
    spatial = {
        **{
            name: deepcopy(protocol[name])
            for name in (
                "label",
                "initial_strategy",
                "n_global_draws",
                "proposal_seed",
                "production_seed",
                "spatial_atom_chunk",
                "spatial_object_chunk",
                "n_cells",
                "sampler_version",
                "sampler_hash_contract",
                "rng_contract",
                "rng_domain_word",
                "rng_version_word",
                "quantizer_contract",
                "order_contract",
                "cells_per_slab",
                "cycles_per_cell",
                "source_microstrata",
                "source_draws_per_cell",
                "microstrata_per_cell",
                "draws_per_microstratum",
                "draws_per_cell",
                "production_draws",
                "cdf_tick_bits",
                "cdf_ticks_per_microstratum",
                "source_total_ticks",
                "balance_total_ticks",
                "balance_probability_law",
                "prior_minimum_atom_ticks",
                "proxy_fallback",
                "proxy_fallback_tick_scales",
                "integer_target_law",
                "pilot",
                "tail_proposal_recipe",
                "partition_recipe_sha256",
                "partition_sha256",
                "partition_cache_sha256",
                "partition_manifest_sha256",
                "mock_input_sha256",
                "stratum_flow_evaluations",
                "production_flow_evaluations",
            )
        },
        "partition_verified": True,
        "minimum_cell_active_count": 20_000,
        "maximum_cell_active_count": 60_000,
        "maximum_four_cell_slab_active_count": 240_000,
        "proposal_initialization_seconds": 1.0,
        "execution_call_provenance": calls,
    }
    exact = {
        "selection_count": protocol["exact_selection_count"],
        "selected_object_ids": selected_ids.tolist(),
        "selection_manifest_sha256": protocol[
            "exact_selection_manifest_sha256"
        ],
        "atom_chunk": protocol["exact_atom_chunk"],
        "object_chunk": protocol["exact_object_chunk"],
        "active_atoms": protocol["active_atoms"],
        "flow_evaluations": protocol["exact_flow_evaluations"],
    }
    metadata = {
        "observation_start": protocol["universe_start"],
        "observation_stop": protocol["universe_stop"],
        "center": deepcopy(protocol["center"]),
        "n_candidates": protocol["n_candidates"],
        "prefilter_candidates": protocol["prefilter_candidates"],
        "candidate_backend": protocol["candidate_backend"],
        "object_chunk": protocol["diagnostic_object_chunk"],
        "flow_atom_chunk": protocol["flow_atom_chunk"],
        "flow_evaluations": protocol["run_flow_evaluations"],
        "phase_seconds": {
            name: 1.0 for name in SPATIAL_H256_U8_PS2_METHOD_TIME_COMPONENTS
        },
        "spatial_h256_u8_ps2_center": spatial,
        "exact_oracle": exact,
    }
    spatial["run_manifest_sha256"] = spatial_h256_u8_ps2_run_manifest_sha256(
        frame
    )
    return SimpleNamespace(table=frame, metadata=metadata)


def _direct_pair_variance(
    values: np.ndarray,
    cell_id: np.ndarray,
    source_id: np.ndarray,
    microstratum_id: np.ndarray,
    cells: int,
) -> tuple[np.ndarray, np.ndarray]:
    cell_mass = np.empty(cells, dtype=np.float64)
    cell_variance = np.empty(cells, dtype=np.float64)
    for cell in range(cells):
        member = cell_id == cell
        cell_mass[cell] = values[member].mean(dtype=np.float64)
        squared_differences = []
        for source, count in enumerate(SPATIAL_H256_U8_PS2_SOURCE_MICROSTRATA):
            for microstratum in range(count):
                pair = values[
                    member
                    & (source_id == source)
                    & (microstratum_id == microstratum)
                ]
                assert len(pair) == 2
                squared_differences.append(float((pair[0] - pair[1]) ** 2))
        cell_variance[cell] = np.sum(squared_differences) / 4096.0
    return cell_mass, cell_variance


def test_protocol_encodes_exact_integer_tick_law_and_no_pilot_identity():
    protocol = SPATIAL_H256_U8_PS2_PROTOCOL
    ticks = 1 << 48
    assert SPATIAL_H256_U8_PS2_CDF_TICKS == ticks
    assert SPATIAL_H256_U8_PS2_BALANCE_TICKS == 1 << 53
    assert protocol["source_microstrata"] == [16, 12, 4]
    assert protocol["source_draws_per_cell"] == [32, 24, 8]
    assert protocol["source_total_ticks"] == [16 * ticks, 12 * ticks, 4 * ticks]
    assert protocol["balance_total_ticks"] == 32 * ticks == 1 << 53
    assert protocol["prior_minimum_atom_ticks"] == 1
    assert protocol["proxy_fallback"] == "copy_quantized_prior"
    assert protocol["pilot"] == {
        "used": False,
        "draws": 0,
        "target_flow_evaluations": 0,
        "rng_domains": [],
        "allocation": "fixed_uniform_8",
    }
    assert protocol["run_flow_evaluations"] == (
        protocol["stratum_flow_evaluations"]
        + protocol["production_flow_evaluations"]
        + protocol["exact_flow_evaluations"]
    )
    assert _integer_contract_consistent(protocol)


def test_protocol_reuses_only_frozen_artifacts_and_threshold_values():
    new = SPATIAL_H256_U8_PS2_PROTOCOL
    old = SPATIAL_NEYMAN_PROTOCOL
    for name in (
        "partition_recipe_sha256",
        "partition_sha256",
        "partition_cache_sha256",
        "partition_manifest_sha256",
        "exact_selection_manifest_sha256",
        "mock_input_sha256",
    ):
        assert new[name] == old[name]
    assert SPATIAL_H256_U8_PS2_CENTER_GATE_THRESHOLDS == (
        SPATIAL_NEYMAN_CENTER_GATE_THRESHOLDS
    )
    assert SPATIAL_H256_U8_PS2_CENTER_GATE_THRESHOLDS is not (
        SPATIAL_NEYMAN_CENTER_GATE_THRESHOLDS
    )
    changed = deepcopy(new)
    changed["source_total_ticks"][0] -= 1
    assert not _integer_contract_consistent(changed)


def test_run_manifest_is_order_invariant_and_binds_every_hash_column():
    rows = []
    for object_id in (25_001, 25_000):
        row = {"object_index": object_id}
        for name in diagnostic_module._SPATIAL_H256_U8_PS2_MANIFEST_HASH_COLUMNS:
            row[name] = _digest(f"{name}:{object_id}")
        rows.append(row)
    frame = pd.DataFrame(rows)
    expected = spatial_h256_u8_ps2_run_manifest_sha256(frame)
    assert expected == spatial_h256_u8_ps2_run_manifest_sha256(
        frame.iloc[::-1].reset_index(drop=True)
    )
    changed = frame.copy()
    changed.loc[0, "production_u48_sha256"] = _digest("changed")
    assert spatial_h256_u8_ps2_run_manifest_sha256(changed) != expected


def test_log_pair_reduction_matches_direct_normal_regime():
    cells = 3
    schedule = _schedule(cells)
    cell_id, source_id, microstratum_id, _ = schedule
    rng = np.random.default_rng(814)
    values = np.exp(rng.normal(loc=-1.0, scale=0.7, size=len(cell_id)))
    reduced = _reduce_h256_u8_ps2_log_values(
        np.log(values), *schedule, cells=cells
    )
    cell_mass, cell_variance = _direct_pair_variance(
        values, cell_id, source_id, microstratum_id, cells
    )
    expected_variance = float(cell_variance.sum())
    expected_effective = cells * values
    expected_pareto = float(pareto_tail_index(expected_effective[None, :])[0])

    np.testing.assert_allclose(reduced.cell_mass, cell_mass, rtol=2e-15, atol=0.0)
    np.testing.assert_allclose(
        reduced.cell_variance, cell_variance, rtol=8e-15, atol=0.0
    )
    assert reduced.mass == pytest.approx(float(cell_mass.sum()), rel=2e-15)
    assert reduced.variance == pytest.approx(expected_variance, rel=8e-15)
    assert reduced.standard_error == pytest.approx(
        np.sqrt(expected_variance), rel=4e-15
    )
    assert reduced.pareto_k == pytest.approx(expected_pareto, rel=2e-14)
    assert reduced.maximum_effective_weight == pytest.approx(
        float(expected_effective.max()), rel=2e-15
    )
    assert np.array_equal(
        reduced.cell_source_draw_counts,
        np.broadcast_to(
            np.asarray(SPATIAL_H256_U8_PS2_SOURCE_DRAWS), (cells, 3)
        ),
    )
    for array in (
        reduced.cell_mass,
        reduced.cell_variance,
        reduced.cell_log_mass,
        reduced.cell_log_variance,
        reduced.cell_source_draw_counts,
        reduced.scaled_values,
        reduced.cell_source_shift,
        reduced.cell_source_scaled_mass,
        reduced.cell_source_scaled_pair_squared_difference_sum,
    ):
        assert not array.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        reduced.cell_mass[0] = 0.0


def test_pair_variance_preserves_low_varying_source_below_object_dynamic_range():
    schedule = _schedule(1)
    _, source_id, _, _ = schedule
    logs = np.full(len(source_id), 700.0, dtype=np.float64)
    low = source_id == 1
    logs[low] = -50.0 + np.log(np.linspace(1.0, 2.0, low.sum()))
    reduced = _reduce_h256_u8_ps2_log_values(logs, *schedule, cells=1)

    low_values = np.exp(logs[low])
    expected_squared_differences = np.square(
        low_values[0::2] - low_values[1::2]
    )
    expected_log_variance = float(
        np.log(expected_squared_differences.sum()) - np.log(4096.0)
    )
    assert 700.0 - float(logs[low].max()) > np.log(np.finfo(np.float64).max)
    assert reduced.log_variance == pytest.approx(expected_log_variance, abs=2e-14)
    assert reduced.standard_error == pytest.approx(
        np.exp(0.5 * expected_log_variance), rel=2e-14
    )
    assert reduced.standard_error > 0.0
    assert reduced.cell_source_scaled_pair_squared_difference_sum[0, 0] == 0.0
    assert reduced.cell_source_scaled_pair_squared_difference_sum[0, 2] == 0.0
    assert reduced.cell_source_scaled_pair_squared_difference_sum[0, 1] > 0.0


def test_core_crosscheck_uses_three_source_isolated_locally_scaled_calls(
    monkeypatch,
):
    cells = 2
    schedule = _schedule(cells)
    _, source_id, _, _ = schedule
    logs = np.full((3, len(source_id)), -np.inf, dtype=np.float64)
    for row in range(len(logs)):
        for source in range(3):
            selected = source_id == source
            logs[row, selected] = (
                20.0 * source
                + row
                + np.linspace(-0.5, 0.5, selected.sum())
            )
    reductions = [
        _reduce_h256_u8_ps2_log_values(row, *schedule, cells=cells)
        for row in logs
    ]
    calls = []

    def fake_core(values, draw):
        values = np.asarray(values, dtype=np.float64)
        calls.append(values.copy())
        paired = values.reshape(len(values), cells, 32, 2)
        difference = paired[:, :, :, 0] - paired[:, :, :, 1]
        return SimpleNamespace(
            cell_estimate=paired.mean(axis=(2, 3), dtype=np.float64),
            cell_pair_squared_difference_sum=np.square(difference).sum(
                axis=2, dtype=np.float64
            ),
        )

    monkeypatch.setattr(
        diagnostic_module,
        "reduce_spatial_probability_stratified_mis",
        fake_core,
    )
    mass_relative, pair_relative = _source_isolated_core_crosscheck(
        reductions, SimpleNamespace(source_id=source_id)
    )
    assert len(calls) == 3
    for source, values in enumerate(calls):
        assert np.all(values[:, source_id != source] == 0.0)
        assert np.any(values[:, source_id == source] > 0.0)
    np.testing.assert_array_equal(mass_relative, np.zeros(len(logs)))
    assert np.all(pair_relative <= 2.0e-15)


def test_256_cell_log_reconstruction_allows_bounded_roundoff():
    cells = 256
    schedule = _schedule(cells)
    cell_id, source_id, _, _ = schedule
    rng = np.random.default_rng(0)
    source_shift = rng.uniform(-500.0, 350.0, size=(cells, 3))
    logs = source_shift[cell_id, source_id] + rng.normal(
        scale=0.2, size=len(cell_id)
    )

    reduced = _reduce_h256_u8_ps2_log_values(logs, *schedule, cells=cells)
    eps = np.finfo(np.float64).eps
    mass_error_eps = (
        abs(reduced.mass - reduced.cell_mass.sum(dtype=np.float64))
        / reduced.mass
        / eps
    )
    variance_error_eps = (
        abs(reduced.variance - reduced.cell_variance.sum(dtype=np.float64))
        / reduced.variance
        / eps
    )

    assert np.ptp(logs) > np.log(np.finfo(np.float64).max)
    assert max(mass_error_eps, variance_error_eps) > 64.0
    assert mass_error_eps <= 512.0
    assert variance_error_eps <= 512.0


def test_all_negative_infinity_source_is_zero_but_all_zero_mass_fails():
    schedule = _schedule(1)
    _, source_id, _, _ = schedule
    logs = np.linspace(-2.0, 2.0, len(source_id))
    logs[source_id == 2] = -np.inf
    reduced = _reduce_h256_u8_ps2_log_values(logs, *schedule, cells=1)
    assert np.isfinite(reduced.mass)
    assert reduced.mass > 0.0

    with pytest.raises(RuntimeError, match="positive estimated mass"):
        _reduce_h256_u8_ps2_log_values(
            np.full(len(logs), -np.inf), *schedule, cells=1
        )


def test_constant_blocks_have_zero_pair_variance():
    schedule = _schedule(2)
    logs = np.repeat(np.asarray([1.0, -3.0]), 64)
    reduced = _reduce_h256_u8_ps2_log_values(logs, *schedule, cells=2)
    assert reduced.log_variance == -np.inf
    assert reduced.variance == 0.0
    assert reduced.standard_error == 0.0
    assert np.array_equal(reduced.cell_variance, np.zeros(2))


@pytest.mark.parametrize("invalid", [np.nan, np.inf])
def test_invalid_log_value_fails_closed(invalid):
    schedule = _schedule(1)
    logs = np.zeros(64, dtype=np.float64)
    logs[-1] = invalid
    with pytest.raises(ValueError, match=r"free of NaN and \+inf"):
        _reduce_h256_u8_ps2_log_values(logs, *schedule, cells=1)


def test_duplicate_pair_slot_fails_closed():
    schedule = list(_schedule(1))
    schedule[-1] = schedule[-1].copy()
    schedule[-1][1] = 0
    with pytest.raises(ValueError, match="slot is duplicated"):
        _reduce_h256_u8_ps2_log_values(
            np.zeros(64, dtype=np.float64), *schedule, cells=1
        )


def test_passing_summary_preserves_archived_exact_gate_family():
    summary = spatial_h256_u8_ps2_center_summary(_passing_diagnostic())
    assert summary["protocol"]["matched"]
    assert all(
        gate["passed"] for gate in summary["gates"]["integrity"].values()
    )
    archived_exact_keys = {
        "selected_exact_object_count",
        "stratum_relative_crosscheck_max",
        "absolute_delta_log_total_evidence_median",
        "absolute_delta_log_total_evidence_p90_higher",
        "absolute_signed_aggregate_z",
        "exact_z_rms",
        "exact_z_within_1_96_count",
    }
    assert set(summary["gates"]["exact_calibration"]) == archived_exact_keys
    assert summary["gates"]["exact_calibration"][
        "selected_exact_object_count"
    ] == {
        "observed": 64,
        "operator": "==",
        "threshold": 64,
        "passed": True,
    }


def test_summary_fails_integer_identity_tamper():
    diagnostic = _passing_diagnostic()
    diagnostic.table.loc[0, "rng_recomputation_failure_count"] = 1
    summary = spatial_h256_u8_ps2_center_summary(diagnostic)
    assert not summary["gates"]["integrity"]["integer_cdf_law_exact"]["passed"]


def test_summary_rejects_one_undefined_pareto_k():
    diagnostic = _passing_diagnostic()
    diagnostic.table.loc[0, "pareto_k_spatial_h256_u8_ps2_complement"] = np.nan
    summary = spatial_h256_u8_ps2_center_summary(diagnostic)
    gate = summary["gates"]["tail_reliability"][
        "pareto_k_defined_all_objects"
    ]
    assert gate == {
        "observed": 1,
        "operator": "undefined_count ==",
        "threshold": 0,
        "passed": False,
    }


def test_summary_fails_if_pilot_or_allocation_column_appears():
    diagnostic = _passing_diagnostic()
    diagnostic.table["pilot_draws"] = 0
    summary = spatial_h256_u8_ps2_center_summary(diagnostic)
    assert not summary["gates"]["integrity"]["no_pilot_or_allocation"][
        "passed"
    ]


def test_summary_fails_manifest_tamper():
    diagnostic = _passing_diagnostic()
    diagnostic.table.loc[0, "production_u48_sha256"] = _digest("tampered")
    summary = spatial_h256_u8_ps2_center_summary(diagnostic)
    assert not summary["gates"]["integrity"]["run_manifest_recomputes"][
        "passed"
    ]


def test_summary_fails_call_row_hash_mismatch_with_valid_run_manifest():
    diagnostic = _passing_diagnostic()
    diagnostic.table.loc[0, "production_draw_record_sha256"] = _digest(
        "different draw record"
    )
    spatial = diagnostic.metadata["spatial_h256_u8_ps2_center"]
    spatial["run_manifest_sha256"] = spatial_h256_u8_ps2_run_manifest_sha256(
        diagnostic.table
    )

    summary = spatial_h256_u8_ps2_center_summary(diagnostic)
    assert summary["gates"]["integrity"]["run_manifest_recomputes"]["passed"]
    assert not summary["gates"]["integrity"]["core_call_provenance_exact"][
        "passed"
    ]


def test_summary_fails_call_hash_contract_tamper():
    diagnostic = _passing_diagnostic()
    calls = diagnostic.metadata["spatial_h256_u8_ps2_center"][
        "execution_call_provenance"
    ]
    calls[0]["hash_contract"] = "different hash contract"

    summary = spatial_h256_u8_ps2_center_summary(diagnostic)
    assert not summary["gates"]["integrity"]["core_call_provenance_exact"][
        "passed"
    ]


@pytest.mark.parametrize(
    "column",
    (
        "mass_stratum",
        "mass_complement_spatial_h256_u8_ps2",
        "mass_complement_spatial_h256_u8_ps2_standard_error",
    ),
)
def test_summary_fails_physical_log_identity_tamper(column):
    diagnostic = _passing_diagnostic()
    unselected = np.flatnonzero(
        ~diagnostic.table["exact_oracle_selected"].to_numpy(dtype=bool)
    )[0]
    diagnostic.table.loc[unselected, column] *= 2.0

    summary = spatial_h256_u8_ps2_center_summary(diagnostic)
    assert summary["gates"]["integrity"]["finite_positive_estimates"][
        "passed"
    ]
    assert not summary["gates"]["integrity"][
        "mass_and_standard_error_log_identity"
    ]["passed"]


def test_efficiency_fails_closed_when_finite_physical_masses_would_overflow_sum():
    diagnostic = _passing_diagnostic()
    unselected = ~diagnostic.table["exact_oracle_selected"].to_numpy(
        dtype=bool
    )
    huge = np.float64(1.0e308)
    for physical, logged in (
        ("mass_stratum", "log_mass_stratum"),
        (
            "mass_complement_spatial_h256_u8_ps2",
            "log_mass_complement_spatial_h256_u8_ps2",
        ),
    ):
        diagnostic.table.loc[unselected, physical] = huge
        diagnostic.table.loc[unselected, logged] = np.log(huge)

    summary = spatial_h256_u8_ps2_center_summary(diagnostic)
    assert summary["gates"]["integrity"][
        "mass_and_standard_error_log_identity"
    ]["passed"]
    assert not summary["gates"]["efficiency"][
        "mean_relative_variance_time_efficiency"
    ]["passed"]
    assert not summary["gates"]["efficiency"][
        "p90_higher_relative_variance_time_efficiency"
    ]["passed"]


def test_summary_rejects_negative_proposal_initialization_time():
    diagnostic = _passing_diagnostic()
    diagnostic.metadata["spatial_h256_u8_ps2_center"][
        "proposal_initialization_seconds"
    ] = -1.0

    summary = spatial_h256_u8_ps2_center_summary(diagnostic)
    assert not summary["gates"]["integrity"][
        "method_phase_timing_complete"
    ]["passed"]
