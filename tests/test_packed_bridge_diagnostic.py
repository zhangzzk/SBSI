from concurrent.futures import ProcessPoolExecutor
import multiprocessing

import numpy as np
import pandas as pd
import pytest

from sbsi.packed_bridge_diagnostic import (
    PACKED_BRIDGE_ISLANDS,
    PACKED_BRIDGE_PARTICLES,
    PACKED_BRIDGE_PROTOCOL,
    PACKED_BRIDGE_ROUNDS,
    PACKED_BRIDGE_SLOTS_PER_OBJECT,
    PackedBridgeDiagnostic,
    PackedTargetPlan,
    _nonnegative_scaled_sum,
    load_packed_bridge_full_trace_replay,
    packed_bridge_run_manifest_sha256,
    run_compact_bridge_batch,
    write_packed_bridge_full_trace_replay,
)
from sbsi.packed_integer_bridge_smc import run_packed_integer_bridge_smc


def _production_slots() -> np.ndarray:
    return (
        np.arange(PACKED_BRIDGE_SLOTS_PER_OBJECT, dtype=np.int64) % 11
    ).reshape(
        PACKED_BRIDGE_ISLANDS,
        PACKED_BRIDGE_ROUNDS,
        PACKED_BRIDGE_PARTICLES,
    )


def test_frozen_protocol_has_one_exact_production_geometry():
    assert PACKED_BRIDGE_PROTOCOL["universe_start"] == 25_000
    assert PACKED_BRIDGE_PROTOCOL["universe_stop"] == 25_512
    assert PACKED_BRIDGE_PROTOCOL["islands"] == 16
    assert PACKED_BRIDGE_PROTOCOL["proposal_rounds"] == 8
    assert PACKED_BRIDGE_PROTOCOL["particles_per_island"] == 256
    assert PACKED_BRIDGE_PROTOCOL["proposal_slots_per_object"] == 32_768
    assert PACKED_BRIDGE_PROTOCOL["new_exact_flow_evaluations"] == 0
    assert PACKED_BRIDGE_PROTOCOL["target_pack_replay_max_abs_delta"] == 1.0e-5


def test_target_plan_scores_unique_atoms_then_masks_stratum_and_scatters():
    slots = _production_slots()
    numerator = np.arange(1, 13, dtype=np.uint64)
    plan = PackedTargetPlan.build(
        slots,
        probability_numerator=numerator,
        exact_stratum=np.array([1, 3, 17], dtype=np.int64),
    )
    log_c = np.linspace(-2.0, 2.0, len(plan.unique_atom_ids))
    ratio, accounting = plan.scatter_log_ratio(log_c)
    expected_unique = log_c - np.log(
        np.ldexp(numerator[plan.unique_atom_ids].astype(np.float64), -51)
    )
    expected_unique[np.isin(plan.unique_atom_ids, [1, 3])] = -np.inf
    expected = expected_unique[plan.inverse].reshape(slots.shape)
    assert np.array_equal(ratio, expected)
    assert accounting["requested_slots"] == 32_768
    assert accounting["unique_atoms"] == 11
    assert accounting["unique_evaluations"] == 11
    assert accounting["complement_unique_atoms"] == 9
    assert accounting["excluded_unique_atoms"] == 2
    assert accounting["cache_hits"] == 32_768 - 11
    assert accounting["finite_slots"] + accounting["zero_slots"] == 32_768


def test_target_plan_rejects_inconsistent_duplicate_target_values_downstream():
    slots = _production_slots()
    plan = PackedTargetPlan.build(
        slots,
        probability_numerator=np.ones(12, dtype=np.uint64),
        exact_stratum=np.array([], dtype=np.int64),
    )
    with pytest.raises(ValueError, match="align"):
        plan.scatter_log_ratio(np.zeros(len(plan.unique_atom_ids) - 1))


def test_nonnegative_scaled_sum_preserves_true_zero_stratum():
    assert _nonnegative_scaled_sum(np.full(5, -np.inf)) == (0.0, -np.inf)
    mass, log_mass = _nonnegative_scaled_sum(np.log(np.array([2.0, 3.0])))
    assert mass == pytest.approx(5.0)
    assert log_mass == pytest.approx(np.log(5.0))


def test_compact_batch_preserves_order_and_reports_process_telemetry():
    atoms = np.stack((_production_slots(), _production_slots() + 100))
    ratios = np.zeros(atoms.shape, dtype=np.float64)
    results, wall, cpu, rss = run_compact_bridge_batch(
        atoms,
        ratios,
        np.array([25_000, 25_001], dtype=np.int64),
        executor=None,
    )
    assert [result.object_id for result in results] == [25_000, 25_001]
    assert all(value >= 0.0 for value in wall)
    assert all(value >= 0.0 for value in cpu)
    assert all(value > 0 for value in rss)
    assert all(not result.island_log_normalizer.flags.writeable for result in results)


def test_spawned_compact_batch_round_trip_is_revalidated():
    atoms = _production_slots()[None, ...]
    ratios = np.zeros(atoms.shape, dtype=np.float64)
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=1, mp_context=context) as executor:
        results, wall, cpu, rss = run_compact_bridge_batch(
            atoms,
            ratios,
            np.array([25_000], dtype=np.int64),
            executor=executor,
        )
    assert results[0].object_id == 25_000
    assert wall[0] >= 0.0
    assert cpu[0] >= 0.0
    assert rss[0] > 0


def test_full_trace_replay_writer_round_trips_without_pickle(tmp_path):
    atoms = np.arange(2 * 8 * 4, dtype=np.int64).reshape(2, 8, 4)
    ratios = ((atoms % 7) - 3).astype(np.float64) / 2.0
    replay = run_packed_integer_bridge_smc(
        atoms,
        ratios,
        seed=7_008_706,
        object_id=25_000,
        island_chunk=1,
    )
    diagnostic = PackedBridgeDiagnostic(
        pd.DataFrame(),
        {"packed_integer_bridge_center": {}},
        replay,
    )
    path = tmp_path / "full_trace_replay.npz"
    identity = write_packed_bridge_full_trace_replay(diagnostic, path)
    restored = load_packed_bridge_full_trace_replay(path)
    assert identity["sha256"]
    assert identity["bytes"] == path.stat().st_size
    assert restored.record_sha256 == replay.record_sha256
    assert restored.trace_sha256 == replay.trace_sha256
    with pytest.raises(FileExistsError):
        write_packed_bridge_full_trace_replay(diagnostic, path)


def test_run_manifest_is_order_invariant_and_tamper_evident():
    hash_columns = {
        name: [f"{index + 1:064x}", f"{index + 3:064x}"]
        for index, name in enumerate(
            (
                "stratum_indices_sha256",
                "stratum_log_target_sha256",
                "exact_reference_stratum_indices_sha256",
                "exact_reference_stratum_log_target_sha256",
                "exact_reference_mass_sha256",
                "base_draw_record_sha256",
                "base_draw_indices_sha256",
                "base_draw_source_sha256",
                "base_draw_source_bits_sha256",
                "base_draw_u48_sha256",
                "base_draw_selected_source_ticks_sha256",
                "base_draw_selected_source_cdf_lower_sha256",
                "base_draw_probability_numerator_sha256",
                "target_unique_atom_ids_sha256",
                "target_unique_probability_numerator_sha256",
                "target_unique_log_c_sha256",
                "target_excluded_unique_sha256",
                "target_excluded_slots_sha256",
                "bridge_proposal_atom_ids_sha256",
                "bridge_target_log_ratio_sha256",
                "bridge_trace_sha256",
                "bridge_record_sha256",
                "bridge_execution_sha256",
                "bridge_island_log_normalizer_sha256",
                "bridge_collapsed_stage_sha256",
                "bridge_stage_log_normalizer_sha256",
                "bridge_stage_effective_sample_size_sha256",
                "bridge_stage_acceptance_rate_sha256",
                "bridge_stage_movement_rate_sha256",
                "bridge_stage_unique_lineage_count_sha256",
                "bridge_stage_unique_state_count_sha256",
            )
        )
    }
    frame = pd.DataFrame({"object_index": [25_001, 25_000], **hash_columns})
    observed = packed_bridge_run_manifest_sha256(frame)
    assert observed == packed_bridge_run_manifest_sha256(frame.iloc[::-1])
    frame.loc[0, "bridge_trace_sha256"] = "f" * 64
    assert observed != packed_bridge_run_manifest_sha256(frame)
