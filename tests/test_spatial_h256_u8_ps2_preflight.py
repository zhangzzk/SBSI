from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from _script_loader import load_script_module


ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT = load_script_module("preflight_spatial_h256_u8_ps2.py")


def test_preflight_workload_and_science_counters_are_frozen():
    assert PREFLIGHT.PREFLIGHT_VERSION == "spatial_h256_u8_ps2_a40_structural_v1"
    assert PREFLIGHT.PREFLIGHT_OBJECT_COUNT == 32
    assert PREFLIGHT.PREFLIGHT_EXCLUSIONS_PER_OBJECT == 16_384
    assert PREFLIGHT.PREFLIGHT_OBJECT_ID_BASE == 1 << 40
    assert PREFLIGHT.PREFLIGHT_OBSERVATION_SEED == 20_260_830
    assert PREFLIGHT.PREFLIGHT_EXCLUSION_SEED == 20_260_831
    assert PREFLIGHT.PREFLIGHT_OBJECT_CHUNK == 32
    assert PREFLIGHT.PREFLIGHT_ATOM_CHUNK == 262_144
    assert PREFLIGHT.PREFLIGHT_CORE_CONTRACT == {
        "sampler_version": "h256_u8_ps2_v1",
        "cells": 256,
        "cells_per_slab": 4,
        "source_microstrata": (16, 12, 4),
        "source_draws_per_cell": (32, 24, 8),
        "replicates": 2,
        "draws_per_cell": 64,
        "ticks_per_microstratum": 1 << 48,
        "sampler_seed": 7_008_704,
    }
    PREFLIGHT._verify_core_contract()
    assert PREFLIGHT.PREFLIGHT_PARTITION_SHA256 == (
        "0da871a79abfd124360a2d93f02707bb88de42e5281edfda741a819a58b557ab"
    )
    assert PREFLIGHT.PREFLIGHT_PARTITION_CACHE_SHA256 == (
        "1a63e4a7e435f79fa319f1a909698ee117bcb6394a257e0258ace9daf0248e3c"
    )
    assert PREFLIGHT.PREFLIGHT_PARTITION_MANIFEST_SHA256 == (
        "7cd8ae16a65c8488b5ff6c8abfeef5edbdce75b0035dcb37ad83fa51ff29cdef"
    )
    assert PREFLIGHT.ZERO_SCIENCE_COUNTERS == {
        "candidate_queries": 0,
        "exact_oracle_evaluations": 0,
        "gate_evaluations": 0,
        "mock_rows_loaded": 0,
        "reducer_evaluations": 0,
        "target_flow_evaluations": 0,
    }


def test_object_keyed_unique_sampler_is_deterministic_and_bounded():
    object_id = PREFLIGHT.PREFLIGHT_OBJECT_ID_BASE + 7
    first = PREFLIGHT._unique_random_positions(
        PREFLIGHT._object_rng(PREFLIGHT.PREFLIGHT_EXCLUSION_SEED, object_id),
        100_000,
        16_384,
    )
    repeated = PREFLIGHT._unique_random_positions(
        PREFLIGHT._object_rng(PREFLIGHT.PREFLIGHT_EXCLUSION_SEED, object_id),
        100_000,
        16_384,
    )
    other = PREFLIGHT._unique_random_positions(
        PREFLIGHT._object_rng(PREFLIGHT.PREFLIGHT_EXCLUSION_SEED, object_id + 1),
        100_000,
        16_384,
    )
    np.testing.assert_array_equal(first, repeated)
    assert len(np.unique(first)) == 16_384
    assert first.min() >= 0 and first.max() < 100_000
    assert not np.array_equal(first, other)


def test_successful_main_prints_exactly_one_json_object(monkeypatch, capsys):
    report = {
        "outcome_free": True,
        "science_counters": dict(PREFLIGHT.ZERO_SCIENCE_COUNTERS),
    }
    monkeypatch.setattr(PREFLIGHT, "run_preflight", lambda args: report)
    result = PREFLIGHT.main(
        [
            "--scene-store",
            "scene",
            "--model-cache",
            "model",
            "--proposal-cache",
            "proposal",
            "--partition-cache",
            "partition",
        ]
    )
    captured = capsys.readouterr()
    assert result == 0
    assert captured.err == ""
    assert captured.out.count("\n") == 1
    assert json.loads(captured.out) == report


def test_script_has_no_science_data_or_evaluation_path():
    source = (ROOT / "scripts" / "preflight_spatial_h256_u8_ps2.py").read_text()
    for forbidden in (
        "MockCatalogue",
        "CatalogueLikelihood",
        "measurement_model",
        "flow_model",
        ".candidates(",
        ".uncertainty_candidates(",
        "reduce_spatial_probability_stratified_mis",
        "exact_oracle_object_ids",
        "diagnose_",
        "pareto_k",
        "acceptance_gate",
    ):
        assert forbidden not in source
    assert "ProposalCandidates(" in source
    assert "draw_spatial_probability_stratified_mis(" in source
    assert "SpatialCellPartition.load(" in source
    assert "build_coordinate_tree=False" in source
    assert "print(json.dumps(" in source


def test_launcher_is_clean_detached_pinned_a40_preflight_only():
    source = (ROOT / "jobs" / "job_preflight_spatial_h256_u8_ps2.sh").read_text()
    for required in (
        "#SBATCH --partition=inter",
        "#SBATCH --gres=gpu:a40:1",
        "#SBATCH --cpus-per-task=8",
        "#SBATCH --mem=48G",
        "#SBATCH --time=00:30:00",
        "SBSI_COMMIT",
        "symbolic-ref -q HEAD",
        "status --porcelain --untracked-files=normal",
        "spatial_partition.npz",
        'exec "$python" "$repo/scripts/preflight_spatial_h256_u8_ps2.py"',
        "--device cuda",
    ):
        assert required in source
    for forbidden in (
        "run_inference.py",
        "--mock-input",
        "--measurement-model",
        "--diagnose-complement",
        "--exact",
        "--gate",
    ):
        assert forbidden not in source
