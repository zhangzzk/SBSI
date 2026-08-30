from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import sbsi.spatial_neyman_diagnostic as diagnostic_module
from sbsi.complement_diagnostic import exact_oracle_object_ids
from sbsi.spatial_neyman_diagnostic import (
    SPATIAL_NEYMAN_METHOD_TIME_COMPONENTS,
    SPATIAL_NEYMAN_PROTOCOL,
    _higher_percentile,
    _partition_cell_lookup,
    _reduce_spatial_mis_arrays,
    _rescale_log_values,
    _row_arrays,
    _stable_spatial_pilot_variance,
    diagnose_spatial_neyman_complement,
    spatial_neyman_center_summary,
    spatial_neyman_run_manifest_sha256,
)


def _digest(label: str, value: int = 0) -> str:
    return sha256(f"{label}:{value}".encode()).hexdigest()


def _freeze_partition(monkeypatch) -> dict[str, str]:
    identities = {
        "partition_sha256": _digest("partition"),
        "partition_cache_sha256": _digest("partition-cache"),
        "partition_manifest_sha256": _digest("partition-manifest"),
    }
    for name, value in identities.items():
        monkeypatch.setitem(SPATIAL_NEYMAN_PROTOCOL, name, value)
    return identities


def _passing_diagnostic(monkeypatch):
    identities = _freeze_partition(monkeypatch)
    protocol = deepcopy(SPATIAL_NEYMAN_PROTOCOL)
    object_ids = np.arange(25_000, 25_512, dtype=np.int64)
    n_objects = len(object_ids)
    selected_ids, _ = exact_oracle_object_ids(25_000, 25_512)
    exact_selected = np.isin(object_ids, selected_ids)
    exact_z = np.zeros(n_objects, dtype=np.float64)
    exact_z[exact_selected] = np.resize(np.asarray([-1.0, 1.0]), 64)
    complement = np.ones(n_objects, dtype=np.float64)
    standard_error = np.full(n_objects, 1.0e-3, dtype=np.float64)
    exact_complement = complement - exact_z * standard_error
    stratum = np.full(n_objects, 9.0, dtype=np.float64)
    exact_total = stratum + exact_complement

    columns: dict[str, object] = {
        "object_index": object_ids,
        "stratum_count": np.full(n_objects, 16_384, dtype=np.int64),
        "stratum_unique_count": np.full(n_objects, 16_384, dtype=np.int64),
        "stratum_active_count": np.full(n_objects, 16_384, dtype=np.int64),
        "pilot_draws": np.full(n_objects, 4_096, dtype=np.int64),
        "pilot_cells": np.full(n_objects, 256, dtype=np.int64),
        "pilot_source_proxy_t1": np.full(n_objects, 2_048, dtype=np.int64),
        "pilot_source_proxy_t2": np.full(n_objects, 1_536, dtype=np.int64),
        "pilot_source_prior": np.full(n_objects, 512, dtype=np.int64),
        "proxy_t1_prior_fallback_cells": np.zeros(n_objects, dtype=np.int64),
        "proxy_t2_prior_fallback_cells": np.zeros(n_objects, dtype=np.int64),
        "pilot_draws_in_stratum": np.zeros(n_objects, dtype=np.int64),
        "production_reservoir_draws": np.full(n_objects, 131_072, dtype=np.int64),
        "production_selected_draws": np.full(n_objects, 16_384, dtype=np.int64),
        "production_selected_cycles": np.full(n_objects, 2_048, dtype=np.int64),
        "production_cells": np.full(n_objects, 256, dtype=np.int64),
        "production_source_proxy_t1": np.full(n_objects, 8_192, dtype=np.int64),
        "production_source_proxy_t2": np.full(n_objects, 6_144, dtype=np.int64),
        "production_source_prior": np.full(n_objects, 2_048, dtype=np.int64),
        "production_draws_in_stratum": np.zeros(n_objects, dtype=np.int64),
        "allocation_min_cycles": np.full(n_objects, 2, dtype=np.int64),
        "allocation_max_cycles": np.full(n_objects, 64, dtype=np.int64),
        "allocation_floor_cells": np.full(n_objects, 10, dtype=np.int64),
        "allocation_cap_cells": np.zeros(n_objects, dtype=np.int64),
        "allocation_whole_cycle_prefixes": np.ones(n_objects, dtype=bool),
        "allocation_zero_variance_fallback": np.zeros(n_objects, dtype=bool),
        "mass_stratum": stratum,
        "mass_complement_spatial_neyman": complement,
        "mass_complement_spatial_neyman_standard_error": standard_error,
        "pareto_k_spatial_neyman_complement": np.full(n_objects, 0.1),
        "exact_oracle_selected": exact_selected,
        "mass_stratum_exact_oracle": stratum,
        "mass_stratum_exact_oracle_relative_error": np.zeros(n_objects),
        "mass_complement_exact_oracle": exact_complement,
        "mass_total_exact_oracle": exact_total,
        "mass_complement_spatial_minus_exact_z": exact_z,
        "delta_log_total_evidence_spatial_minus_exact": np.log(
            (stratum + complement) / exact_total
        ),
    }
    support = [_digest("support", int(value)) for value in object_ids]
    for name in diagnostic_module._SPATIAL_MANIFEST_HASH_COLUMNS:
        columns[name] = [_digest(name, int(value)) for value in object_ids]
    columns["pilot_exclusion_indices_sha256"] = support
    columns["production_exclusion_indices_sha256"] = support.copy()
    columns["stratum_indices_sha256"] = support.copy()
    normalizers = [_digest("normalizer", int(value)) for value in object_ids]
    columns["pilot_normalizer_sha256"] = normalizers
    columns["production_normalizer_sha256"] = normalizers.copy()
    frame = pd.DataFrame(columns)

    calls = []
    for call_index in range(64):
        start = 25_000 + 8 * call_index
        calls.append(
            {
                "object_start": start,
                "object_stop": start + 8,
                "objects": 8,
                "pilot_seed": 1_008_704,
                "production_seed": 7_008_704,
                "atom_chunk": 262_144,
                "object_chunk": 8,
                "partition_sha256": identities["partition_sha256"],
                "pilot_record_sha256": _digest("pilot-call", call_index),
                "production_record_sha256": _digest("production-call", call_index),
                "allocation_sha256": _digest("allocation-call", call_index),
                "packed_draw_sha256": _digest("packed-call", call_index),
            }
        )
    spatial = {
        "label": protocol["label"],
        "initial_strategy": protocol["initial_strategy"],
        "n_global_draws": protocol["n_global_draws"],
        "pilot_draws": protocol["pilot_draws"],
        "proposal_seed": protocol["proposal_seed"],
        "pilot_seed": protocol["pilot_seed"],
        "production_seed": protocol["production_seed"],
        "spatial_atom_chunk": protocol["spatial_atom_chunk"],
        "spatial_object_chunk": protocol["spatial_object_chunk"],
        "n_cells": protocol["n_cells"],
        "pilot_cycles_per_cell": protocol["pilot_cycles_per_cell"],
        "production_reservoir_cycles_per_cell": protocol[
            "production_reservoir_cycles_per_cell"
        ],
        "allocation_min_cycles": protocol["allocation_min_cycles"],
        "allocation_max_cycles": protocol["allocation_max_cycles"],
        "allocation_total_cycles": protocol["allocation_total_cycles"],
        "tail_proposal_recipe": deepcopy(protocol["tail_proposal_recipe"]),
        "partition_recipe_sha256": protocol["partition_recipe_sha256"],
        **identities,
        "expected_partition_sha256": identities["partition_sha256"],
        "expected_partition_cache_sha256": identities["partition_cache_sha256"],
        "expected_partition_manifest_sha256": identities[
            "partition_manifest_sha256"
        ],
        "partition_verified": True,
        "minimum_cell_active_count": 20_000,
        "maximum_cell_active_count": 60_000,
        "streams_independent": True,
        "mock_input_sha256": deepcopy(protocol["mock_input_sha256"]),
        "stratum_flow_evaluations": protocol["stratum_flow_evaluations"],
        "pilot_flow_evaluations": protocol["pilot_flow_evaluations"],
        "production_flow_evaluations": protocol["production_flow_evaluations"],
        "proposal_initialization_seconds": 1.0,
        "execution_call_provenance": calls,
    }
    exact = {
        "selection_manifest_sha256": protocol[
            "exact_selection_manifest_sha256"
        ],
        "selection_count": protocol["exact_selection_count"],
        "atom_chunk": protocol["exact_atom_chunk"],
        "object_chunk": protocol["exact_object_chunk"],
        "active_atoms": protocol["active_atoms"],
        "flow_evaluations": protocol["exact_flow_evaluations"],
        "selected_object_ids": selected_ids.tolist(),
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
            name: 1.0 for name in SPATIAL_NEYMAN_METHOD_TIME_COMPONENTS
        },
        "spatial_neyman_center": spatial,
        "exact_oracle": exact,
    }
    spatial["run_manifest_sha256"] = spatial_neyman_run_manifest_sha256(frame)
    return SimpleNamespace(table=frame, metadata=metadata)


def test_public_protocol_has_cli_contract_and_starts_unfrozen():
    required = {
        "n_candidates",
        "n_global_draws",
        "proposal_seed",
        "spatial_atom_chunk",
        "spatial_object_chunk",
        "n_cells",
        "allocation_min_cycles",
        "allocation_max_cycles",
        "allocation_total_cycles",
        "partition_sha256",
        "partition_cache_sha256",
        "partition_manifest_sha256",
    }
    assert required <= SPATIAL_NEYMAN_PROTOCOL.keys()
    assert SPATIAL_NEYMAN_PROTOCOL["partition_sha256"] == (
        "0da871a79abfd124360a2d93f02707bb88de42e5281edfda741a819a58b557ab"
    )
    assert SPATIAL_NEYMAN_PROTOCOL["partition_cache_sha256"] == (
        "1a63e4a7e435f79fa319f1a909698ee117bcb6394a257e0258ace9daf0248e3c"
    )
    assert SPATIAL_NEYMAN_PROTOCOL["partition_manifest_sha256"] == (
        "7cd8ae16a65c8488b5ff6c8abfeef5edbdce75b0035dcb37ad83fa51ff29cdef"
    )


def test_science_runner_fails_before_work_while_partition_is_unfrozen(monkeypatch):
    for name in (
        "partition_sha256",
        "partition_cache_sha256",
        "partition_manifest_sha256",
    ):
        monkeypatch.setitem(SPATIAL_NEYMAN_PROTOCOL, name, None)
    with pytest.raises(ValueError, match="disabled until"):
        diagnose_spatial_neyman_complement(
            SimpleNamespace(),
            SimpleNamespace(),
            SimpleNamespace(),
            partition=None,
            partition_cache_sha256=_digest("cache"),
            partition_manifest_sha256=_digest("manifest"),
            partition_load_seconds=1.0,
            proposal_initialization_seconds=1.0,
            confirmation_mock_sha256=deepcopy(
                SPATIAL_NEYMAN_PROTOCOL["mock_input_sha256"]
            ),
        )


def test_science_runner_enforces_cell_size_before_sampler_work(monkeypatch):
    identities = _freeze_partition(monkeypatch)

    class FakePartition:
        partition_sha256 = identities["partition_sha256"]
        recipe_sha256 = SPATIAL_NEYMAN_PROTOCOL["partition_recipe_sha256"]
        cell_id = range(SPATIAL_NEYMAN_PROTOCOL["active_atoms"])
        counts = np.full(256, 16_384, dtype=np.int64)

        @staticmethod
        def verify(proposal):
            return None

    with pytest.raises(RuntimeError, match="more atoms than"):
        diagnose_spatial_neyman_complement(
            SimpleNamespace(selection=None),
            SimpleNamespace(measurements=pd.DataFrame(index=np.arange(512))),
            SimpleNamespace(),
            partition=FakePartition(),
            partition_cache_sha256=identities["partition_cache_sha256"],
            partition_manifest_sha256=identities["partition_manifest_sha256"],
            partition_load_seconds=1.0,
            proposal_initialization_seconds=1.0,
            confirmation_mock_sha256=deepcopy(
                SPATIAL_NEYMAN_PROTOCOL["mock_input_sha256"]
            ),
        )


def test_reduce_spatial_formula_matches_sum_of_means_and_fixed_source_variance():
    schedule = np.asarray([0, 1, 0, 2, 1, 0, 1, 0], dtype=np.int64)
    cell_id = np.concatenate((np.zeros(16, dtype=np.int64), np.ones(24, dtype=np.int64)))
    source_id = np.concatenate((np.tile(schedule, 2), np.tile(schedule, 3)))
    values = np.arange(1.0, 41.0)

    result = _reduce_spatial_mis_arrays(values, cell_id, source_id, cells=2)

    expected_mass = values[:16].mean() + values[16:].mean()
    expected_variance = 0.0
    for cell in range(2):
        member = cell_id == cell
        count = int(member.sum())
        for source in range(3):
            selected = values[member & (source_id == source)]
            expected_variance += len(selected) * selected.var(ddof=1) / count**2
    assert result.mass == pytest.approx(expected_mass)
    assert result.variance == pytest.approx(expected_variance)
    assert result.standard_error == pytest.approx(np.sqrt(expected_variance))
    assert np.array_equal(result.cell_draw_counts, [16, 24])
    assert np.allclose(
        result.effective_weights,
        np.where(cell_id == 0, 40 / 16, 40 / 24) * values,
    )


def test_scaled_float64_reduction_is_invariant_to_large_log_shift():
    log_values = np.log(np.linspace(1.0, 3.0, 40)) + 700.0
    scaled, shift = _rescale_log_values(log_values)
    assert np.isfinite(scaled).all()
    assert scaled.max() == pytest.approx(1.0)
    assert shift == pytest.approx(log_values.max())
    assert np.log(scaled.sum()) + shift == pytest.approx(
        np.log(np.exp(log_values - log_values.max()).sum()) + log_values.max()
    )


def test_production_scaling_fails_if_an_entire_finite_cell_underflows():
    with pytest.raises(RuntimeError, match="underflowed an entire cell"):
        _rescale_log_values(
            np.asarray([0.0, -1_000.0]),
            groups=np.asarray([0, 1]),
        )


def test_pilot_variance_uses_per_cell_scaling_before_global_normalization():
    cells = np.repeat(np.arange(256), 16)
    logs = np.empty((1, len(cells)), dtype=np.float64)
    pattern = np.log(np.linspace(1.0, 2.0, 16))
    for cell in range(256):
        logs[0, cells == cell] = pattern - (1_400.0 if cell else 0.0)
    reservoir = SimpleNamespace(
        indices=np.zeros_like(logs, dtype=np.int64),
        cell_id=cells,
    )

    def reducer(values, _reservoir):
        return values.reshape(1, 256, 16).var(axis=2, ddof=1)

    variance = _stable_spatial_pilot_variance(logs, reservoir, reducer)
    assert variance.shape == (1, 256)
    assert variance[0, 0] == pytest.approx(1.0)
    assert np.all(variance[0, 1:] > 0.0)


def test_higher_percentile_uses_observed_upper_order_statistic():
    values = np.arange(10, dtype=np.float64)
    assert _higher_percentile(values, 90) == 9.0
    assert np.percentile(values, 90) == 8.1


def test_row_array_adapter_handles_shared_reservoir_and_packed_schedules():
    shared = SimpleNamespace(
        indices=np.arange(12).reshape(2, 6),
        probability=np.ones((2, 6)),
        cell_id=np.arange(6),
        component_id=np.resize([0, 1, 2], 6),
    )
    packed = SimpleNamespace(
        indices=np.arange(12).reshape(2, 6),
        probability=np.ones((2, 6)),
        cell_id=np.vstack((np.arange(6), np.arange(6))),
        component_id=np.vstack((np.resize([0, 1, 2], 6),) * 2),
    )
    assert np.array_equal(_row_arrays(shared, 1)[2], np.arange(6))
    assert np.array_equal(_row_arrays(packed, 1)[3], np.resize([0, 1, 2], 6))


def test_partition_lookup_uses_active_aligned_cell_id():
    proposal = SimpleNamespace(
        active_indices=np.asarray([2, 5, 9], dtype=np.int64),
        prior_weights=np.ones(10),
    )
    partition = SimpleNamespace(cell_id=np.asarray([7, 8, 9], dtype=np.int64))
    assert np.array_equal(
        _partition_cell_lookup(partition, proposal, np.asarray([9, 2, 5])),
        [9, 7, 8],
    )


def test_summary_accepts_complete_frozen_protocol(monkeypatch):
    summary = spatial_neyman_center_summary(_passing_diagnostic(monkeypatch))
    assert summary["protocol"]["matched"] is True
    assert summary["gates"]["passed"] is True
    for family in ("integrity", "tail_reliability", "exact_calibration", "efficiency"):
        assert all(item["passed"] for item in summary["gates"][family].values())


def test_null_partition_protocol_fails_closed(monkeypatch):
    diagnostic = _passing_diagnostic(monkeypatch)
    monkeypatch.setitem(SPATIAL_NEYMAN_PROTOCOL, "partition_sha256", None)
    summary = spatial_neyman_center_summary(diagnostic)
    assert summary["protocol"]["matched"] is False
    assert not summary["gates"]["integrity"]["protocol_identity_matched"]["passed"]
    assert not summary["gates"]["integrity"][
        "partition_hash_predeclared_and_verified"
    ]["passed"]


def test_partition_size_guard_is_strict(monkeypatch):
    diagnostic = _passing_diagnostic(monkeypatch)
    diagnostic.metadata["spatial_neyman_center"]["minimum_cell_active_count"] = 16_384
    summary = spatial_neyman_center_summary(diagnostic)
    assert not summary["gates"]["integrity"][
        "partition_hash_predeclared_and_verified"
    ]["passed"]


def test_exact_selection_flag_mutation_fails_identity(monkeypatch):
    diagnostic = _passing_diagnostic(monkeypatch)
    selected = diagnostic.table.index[diagnostic.table["exact_oracle_selected"]]
    unselected = diagnostic.table.index[~diagnostic.table["exact_oracle_selected"]]
    diagnostic.table.loc[selected[0], "exact_oracle_selected"] = False
    diagnostic.table.loc[unselected[0], "exact_oracle_selected"] = True
    summary = spatial_neyman_center_summary(diagnostic)
    assert not summary["gates"]["integrity"]["exact_selection_identity_exact"]["passed"]


def test_float_count_column_fails_integer_integrity(monkeypatch):
    diagnostic = _passing_diagnostic(monkeypatch)
    diagnostic.table["production_selected_cycles"] = diagnostic.table[
        "production_selected_cycles"
    ].astype(np.float64)
    summary = spatial_neyman_center_summary(diagnostic)
    assert not summary["gates"]["integrity"]["integer_typed_count_columns"]["passed"]


def test_rng_alias_with_fresh_manifest_fails_independence(monkeypatch):
    diagnostic = _passing_diagnostic(monkeypatch)
    diagnostic.table.loc[0, "pilot_rng_sha256"] = diagnostic.table.loc[
        0, "production_rng_sha256"
    ]
    diagnostic.metadata["spatial_neyman_center"][
        "run_manifest_sha256"
    ] = spatial_neyman_run_manifest_sha256(diagnostic.table)
    summary = spatial_neyman_center_summary(diagnostic)
    assert not summary["gates"]["integrity"]["independent_rng_streams"]["passed"]
    assert summary["gates"]["integrity"]["run_manifest_recomputes"]["passed"]


def test_support_exclusion_hash_mutation_fails_integrity(monkeypatch):
    diagnostic = _passing_diagnostic(monkeypatch)
    diagnostic.table.loc[0, "pilot_exclusion_indices_sha256"] = _digest("wrong")
    diagnostic.metadata["spatial_neyman_center"][
        "run_manifest_sha256"
    ] = spatial_neyman_run_manifest_sha256(diagnostic.table)
    summary = spatial_neyman_center_summary(diagnostic)
    assert not summary["gates"]["integrity"][
        "support_and_exclusion_hashes_match"
    ]["passed"]


def test_zero_variance_allocation_fallback_fails_gate(monkeypatch):
    diagnostic = _passing_diagnostic(monkeypatch)
    diagnostic.table.loc[0, "allocation_zero_variance_fallback"] = True
    summary = spatial_neyman_center_summary(diagnostic)
    assert not summary["gates"]["integrity"][
        "allocation_bounded_whole_cycle_prefixes"
    ]["passed"]


def test_proxy_fallback_count_outside_cell_range_fails_gate(monkeypatch):
    diagnostic = _passing_diagnostic(monkeypatch)
    diagnostic.table.loc[0, "proxy_t1_prior_fallback_cells"] = 257
    summary = spatial_neyman_center_summary(diagnostic)
    assert not summary["gates"]["integrity"][
        "proxy_prior_fallback_counts_recorded"
    ]["passed"]


def test_pareto_k_at_least_one_fails_reliability(monkeypatch):
    diagnostic = _passing_diagnostic(monkeypatch)
    diagnostic.table.loc[0, "pareto_k_spatial_neyman_complement"] = 1.0
    summary = spatial_neyman_center_summary(diagnostic)
    gate = summary["gates"]["tail_reliability"]["pareto_k_at_least_1_count"]
    assert gate["observed"] == 1
    assert gate["passed"] is False


def test_higher_p90_relative_error_gate_catches_upper_decile(monkeypatch):
    diagnostic = _passing_diagnostic(monkeypatch)
    diagnostic.table.loc[:51, "mass_complement_spatial_neyman_standard_error"] = 0.1
    summary = spatial_neyman_center_summary(diagnostic)
    assert not summary["gates"]["tail_reliability"][
        "complement_relative_error_p90_higher"
    ]["passed"]


def test_exact_stratum_crosscheck_mutation_fails(monkeypatch):
    diagnostic = _passing_diagnostic(monkeypatch)
    selected_row = diagnostic.table.index[diagnostic.table["exact_oracle_selected"]][0]
    diagnostic.table.loc[
        selected_row, "mass_stratum_exact_oracle_relative_error"
    ] = 1.1e-5
    summary = spatial_neyman_center_summary(diagnostic)
    assert not summary["gates"]["exact_calibration"][
        "stratum_relative_crosscheck_max"
    ]["passed"]


def test_zero_exact_z_rms_fails_calibration(monkeypatch):
    diagnostic = _passing_diagnostic(monkeypatch)
    diagnostic.table.loc[
        diagnostic.table["exact_oracle_selected"],
        "mass_complement_spatial_minus_exact_z",
    ] = 0.0
    summary = spatial_neyman_center_summary(diagnostic)
    assert not summary["gates"]["exact_calibration"]["exact_z_rms"]["passed"]


def test_exact_aggregate_bias_fails_signed_z(monkeypatch):
    diagnostic = _passing_diagnostic(monkeypatch)
    selected = diagnostic.table["exact_oracle_selected"]
    diagnostic.table.loc[selected, "mass_complement_exact_oracle"] = 0.99
    summary = spatial_neyman_center_summary(diagnostic)
    assert not summary["gates"]["exact_calibration"][
        "absolute_signed_aggregate_z"
    ]["passed"]


def test_slow_complete_method_fails_efficiency(monkeypatch):
    diagnostic = _passing_diagnostic(monkeypatch)
    diagnostic.metadata["phase_seconds"]["reservoir_mapping"] = 1.0e9
    summary = spatial_neyman_center_summary(diagnostic)
    assert not summary["gates"]["efficiency"][
        "mean_relative_variance_time_efficiency"
    ]["passed"]
    assert not summary["gates"]["efficiency"][
        "p90_higher_relative_variance_time_efficiency"
    ]["passed"]


def test_exact_oracle_time_is_excluded_from_method_time(monkeypatch):
    diagnostic = _passing_diagnostic(monkeypatch)
    first = spatial_neyman_center_summary(diagnostic)["method_seconds"]
    diagnostic.metadata["phase_seconds"]["exact_oracle_flow"] = 1.0e9
    second = spatial_neyman_center_summary(diagnostic)["method_seconds"]
    assert second == first


def test_missing_complete_method_phase_fails_efficiency(monkeypatch):
    diagnostic = _passing_diagnostic(monkeypatch)
    del diagnostic.metadata["phase_seconds"]["allocation"]
    summary = spatial_neyman_center_summary(diagnostic)
    assert np.isnan(summary["method_seconds"])
    assert not summary["gates"]["efficiency"][
        "mean_relative_variance_time_efficiency"
    ]["passed"]
