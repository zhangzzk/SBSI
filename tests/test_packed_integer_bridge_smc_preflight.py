from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "preflight_packed_integer_bridge_smc.py"
SPEC = importlib.util.spec_from_file_location("packed_bridge_preflight", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
preflight = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(preflight)


def test_core_contract_is_frozen_and_outcome_free():
    preflight._verify_core_contract()
    assert preflight.PREFLIGHT_SLOTS_PER_OBJECT == 32_768
    assert preflight.PREFLIGHT_CORE_CONTRACT["objects"] == 32
    assert preflight.PREFLIGHT_CORE_CONTRACT["exclusions_per_object"] == 16_384
    assert all(value == 0 for value in preflight.ZERO_SCIENCE_COUNTERS.values())


def test_unique_random_positions_are_deterministic_and_fail_closed():
    first = preflight._unique_random_positions(np.random.default_rng(19), 40, 11)
    second = preflight._unique_random_positions(np.random.default_rng(19), 40, 11)
    assert np.array_equal(first, second)
    assert len(np.unique(first)) == 11
    assert np.all((first >= 0) & (first < 40))
    for count in (0, 40, 41):
        with pytest.raises(ValueError, match="unique sample"):
            preflight._unique_random_positions(np.random.default_rng(1), 40, count)


def test_synthetic_exclusions_are_object_keyed_sorted_and_on_support():
    active = np.arange(3, 83, 2, dtype=np.int64)
    object_ids = np.asarray([2**40 + 3, 2**40 + 8], dtype=np.uint64)
    first = preflight._synthetic_exclusions(active, object_ids, count=9, seed=71)
    second = preflight._synthetic_exclusions(active, object_ids, count=9, seed=71)
    assert np.array_equal(first, second)
    assert first.shape == (2, 9)
    assert np.all(np.diff(first, axis=1) > 0)
    assert np.isin(first, active).all()
    assert not np.array_equal(first[0], first[1])


def test_synthetic_unique_evaluation_scatter_and_zero_accounting():
    atoms = np.asarray(
        [
            [[4, 7, 4], [11, 7, 19]],
            [[19, 23, 11], [29, 23, 31]],
        ],
        dtype=np.int64,
    )
    exclusions = np.asarray([7, 29], dtype=np.int64)
    base = preflight.IntegerMixtureBase.from_weights(
        np.ones(40, dtype=np.float64),
        np.linspace(1.0, 2.0, 40, dtype=np.float64),
    )
    ratio, accounting = preflight._synthetic_log_ratio(
        atoms,
        exclusions,
        base=base,
        device=torch.device("cpu"),
    )
    assert ratio.shape == atoms.shape
    assert accounting == {
        "requested_slots": atoms.size,
        "unique_evaluations": len(np.unique(atoms)),
        "marginal_probability_lookups": len(np.unique(atoms)),
        "cache_hits": atoms.size - len(np.unique(atoms)),
        "zero_unique_atoms": 2,
        "zero_slots": 3,
    }
    assert np.all(np.isneginf(ratio[atoms == 7]))
    assert np.all(np.isneginf(ratio[atoms == 29]))
    for atom in np.unique(atoms):
        values = ratio[atoms == atom]
        assert np.all(values == values[0])
    assert np.isfinite(ratio[~np.isin(atoms, exclusions)]).all()
    assert np.allclose(
        ratio[(atoms % 2 == 0) & ~np.isin(atoms, exclusions)],
        np.log(0.5),
        atol=5e-15,
    )
    assert np.allclose(
        ratio[(atoms % 2 == 1) & ~np.isin(atoms, exclusions)],
        np.log(1.5),
        atol=5e-15,
    )


def test_known_synthetic_normalizer_uses_the_exact_marginal_mixture():
    base = preflight.IntegerMixtureBase.from_weights(
        np.asarray([1.0, 2.0, 3.0, 4.0]),
        np.asarray([4.0, 3.0, 2.0, 1.0]),
    )
    parity = preflight._parity_probability_numerators(base)
    exclusions = np.asarray([1], dtype=np.int64)
    observed = preflight._known_synthetic_normalizer(
        base, exclusions, parity_numerator=parity
    )
    numerator = base.probability_numerator
    expected = np.ldexp(
        float(numerator[0] + numerator[2] + 3 * numerator[3]), -52
    )
    assert observed == expected


def test_production_shape_mixed_collapse_fixture_retains_zero_islands():
    result = preflight._mixed_collapse_control(object_id=2**40)
    assert np.array_equal(
        result.collapsed_stage,
        np.concatenate((np.ones(8, dtype=np.int64), np.zeros(8, dtype=np.int64))),
    )
    assert result.normalizer == pytest.approx(0.5, abs=5e-13)
    assert result.standard_error == pytest.approx(np.sqrt(1.0 / 60.0), abs=5e-13)


def test_integer_base_cache_is_constructed_then_strictly_loaded(tmp_path):
    raw = np.asarray([0.2, 0.3, 0.0, 0.5])
    detected = np.asarray([0.1, 0.0, 0.0, 0.9])
    cache = tmp_path / "base"
    built, built_mode, built_phases = preflight._load_or_build_base(
        cache, raw, detected
    )
    loaded, loaded_mode, loaded_phases = preflight._load_or_build_base(
        cache, raw, detected
    )
    assert built_mode == "constructed"
    assert loaded_mode == "loaded"
    assert built.law_sha256 == loaded.law_sha256
    assert set(item.name for item in cache.iterdir()) == {
        "arrays.npz",
        "manifest.json",
    }
    assert "integer_base_construct" in built_phases
    assert "integer_base_staged_verify" in built_phases
    assert "integer_base_load" in loaded_phases


def test_source_artifact_verification_binds_merge_report_and_every_payload(
    tmp_path, monkeypatch
):
    expected: dict[str, dict[str, str]] = {}
    roots = {}
    for source, files in {
        "scene_store": ("manifest.json", "galaxies.parquet", "neighbours.npz"),
        "model_cache": (
            "manifest.json",
            "flow_zero.parquet",
            "detection_zero.parquet",
        ),
        "proposal_cache": ("manifest.json", "coordinates.npz"),
    }.items():
        root = tmp_path / source
        root.mkdir()
        roots[source] = root
        expected[source] = {}
        for index, name in enumerate(files):
            path = root / name
            path.write_bytes(f"{source}:{name}:{index}".encode())
            expected[source][name] = preflight._file_sha256(path)
    merge_report = tmp_path / "merge_report.json"
    merge_report.write_text(
        json.dumps(
            {
                "status": "complete",
                "n_positive_prior_atoms": 2,
                "output_sha256": expected,
            }
        )
    )
    monkeypatch.setattr(preflight, "PREFLIGHT_SOURCE_SHA256", expected)
    monkeypatch.setattr(preflight, "PREFLIGHT_PRODUCTION_ATOMS", 2)
    monkeypatch.setattr(
        preflight,
        "PREFLIGHT_MERGE_REPORT_SHA256",
        preflight._file_sha256(merge_report),
    )
    args = SimpleNamespace(
        scene_store=roots["scene_store"],
        model_cache=roots["model_cache"],
        proposal_cache=roots["proposal_cache"],
    )
    result = preflight._verify_source_artifacts(args)
    assert result["merge_report_sha256"] == preflight._file_sha256(merge_report)
    assert result["model_cache"]["files"] == expected["model_cache"]

    (roots["model_cache"] / "detection_zero.parquet").write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="payload identity changed"):
        preflight._verify_source_artifacts(args)


def test_integer_base_cache_lock_and_partial_path_fail_closed(tmp_path):
    raw = np.asarray([0.4, 0.6])
    detected = np.asarray([0.7, 0.3])
    cache = tmp_path / "base"
    lock = tmp_path / ".base.build.lock"
    lock.mkdir()
    with pytest.raises(RuntimeError, match="build lock"):
        preflight._load_or_build_base(cache, raw, detected)
    lock.rmdir()
    cache.mkdir()
    with pytest.raises(RuntimeError, match="not complete"):
        preflight._load_or_build_base(cache, raw, detected)


def test_summary_rejects_nonfinite_or_empty_values():
    assert preflight._summary([3.0, 1.0, 2.0]) == {
        "minimum": 1.0,
        "median": 2.0,
        "maximum": 3.0,
    }
    for values in ([], [1.0, np.inf], [[1.0, 2.0]]):
        with pytest.raises(RuntimeError, match="invalid values"):
            preflight._summary(values)


def test_atomic_report_refuses_overwrite_and_serializes_strict_json(tmp_path):
    path = tmp_path / "nested" / "report.json"
    report = {"finite": 1.25, "identity": "abc"}
    payload = preflight._write_report(path, report)
    assert payload == b'{"finite":1.25,"identity":"abc"}\n'
    assert path.read_bytes() == payload
    with pytest.raises(FileExistsError, match="already exists"):
        preflight._write_report(path, report)


def test_scheduler_wrapper_pins_detached_clean_a40_execution():
    wrapper = (
        ROOT / "jobs" / "job_preflight_packed_integer_bridge_smc.sh"
    ).read_text()
    required = (
        "#SBATCH --gres=gpu:a40:1",
        "SBSI_COMMIT must pin one full lowercase 40-character commit",
        "refusing a branch checkout",
        "refusing a dirty repository checkout",
        "--integer-base-cache",
        "--report",
        "--device cuda",
    )
    assert all(value in wrapper for value in required)
    assert (ROOT / "jobs" / "job_preflight_packed_integer_bridge_smc.sh").stat().st_mode & 0o111


def test_preflight_source_has_no_science_path_import_or_call():
    source = SCRIPT.read_text()
    forbidden = (
        "CatalogueLikelihood",
        "observed_target_tensor",
        "uncertainty_candidates(",
        "log_importance_weights_tensor",
        "mock_catalogue",
        "diagnose_complement_exact_oracle(",
    )
    assert all(token not in source for token in forbidden)
