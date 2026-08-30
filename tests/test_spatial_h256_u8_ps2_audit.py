from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import pytest

from _script_loader import load_script_module
from sbsi.spatial_h256_u8_ps2_diagnostic import (
    SPATIAL_H256_U8_PS2_PROTOCOL,
    SpatialH256U8PS2Diagnostic,
    spatial_h256_u8_ps2_center_summary,
    spatial_h256_u8_ps2_run_manifest_sha256,
)
from test_spatial_h256_u8_ps2_diagnostic import _digest
from test_spatial_h256_u8_ps2_diagnostic import _passing_diagnostic


ROOT = Path(__file__).resolve().parents[1]
AUDITOR = load_script_module("audit_spatial_h256_u8_ps2_result.py")


def _git_blob(path: str) -> bytes:
    return subprocess.run(
        [
            "git",
            "-C",
            str(ROOT),
            "show",
            f"{AUDITOR.AUDITED_IMPLEMENTATION_COMMIT}:{path}",
        ],
        check=True,
        capture_output=True,
    ).stdout


def _git_hashes(paths: tuple[str, ...]) -> dict[str, str]:
    return {path: sha256(_git_blob(path)).hexdigest() for path in paths}


def _runtime_diagnostic() -> SpatialH256U8PS2Diagnostic:
    source = _passing_diagnostic()
    frame = source.table.copy()
    selected = frame["exact_oracle_selected"].to_numpy(dtype=bool)
    frame["maximum_effective_weight_spatial_h256_u8_ps2"] = np.ones(len(frame), dtype=np.float64)
    frame["terminal_correction_maximum"] = np.zeros(len(frame), dtype=np.int64)
    frame["terminal_correction_positive_count"] = np.zeros(len(frame), dtype=np.int64)
    frame.loc[selected, "mass_stratum_exact_oracle_relative_error"] = np.abs(
        frame.loc[selected, "mass_stratum"].to_numpy()
        - frame.loc[selected, "mass_stratum_exact_oracle"].to_numpy()
    ) / np.abs(frame.loc[selected, "mass_stratum_exact_oracle"].to_numpy())
    frame.loc[selected, "mass_complement_spatial_h256_u8_ps2_minus_exact_z"] = (
        frame.loc[selected, "mass_complement_spatial_h256_u8_ps2"].to_numpy()
        - frame.loc[selected, "mass_complement_exact_oracle"].to_numpy()
    ) / frame.loc[selected, "mass_complement_spatial_h256_u8_ps2_standard_error"].to_numpy()
    frame.loc[selected, "delta_log_total_evidence_spatial_h256_u8_ps2_minus_exact"] = np.logaddexp(
        frame.loc[selected, "log_mass_stratum"].to_numpy(),
        frame.loc[selected, "log_mass_complement_spatial_h256_u8_ps2"].to_numpy(),
    ) - np.log(frame.loc[selected, "mass_total_exact_oracle"].to_numpy())
    for name in AUDITOR.EXACT_FLOAT_COLUMNS:
        frame.loc[~selected, name] = np.nan
    frame = frame.loc[:, AUDITOR.EXPECTED_TABLE_COLUMNS]

    protocol = SPATIAL_H256_U8_PS2_PROTOCOL
    spatial = deepcopy(source.metadata["spatial_h256_u8_ps2_center"])
    spatial["proxy_prior_fallback_cells_by_component"] = {
        "proxy_t1": 0,
        "proxy_t2": 0,
    }
    spatial["objects_with_proxy_prior_fallback"] = 0
    spatial["run_manifest_sha256"] = spatial_h256_u8_ps2_run_manifest_sha256(frame)
    spatial = {name: spatial[name] for name in AUDITOR.SPATIAL_METADATA_KEYS}

    exact = deepcopy(source.metadata["exact_oracle"])
    exact["active_atoms_visited_per_selected_object"] = protocol["active_atoms"]
    exact["atom_blocks_per_selected_object"] = (
        protocol["active_atoms"] + protocol["exact_atom_chunk"] - 1
    ) // protocol["exact_atom_chunk"]
    exact["active_indices_sha256"] = _digest("active indices")
    exact = {name: exact[name] for name in AUDITOR.EXACT_METADATA_KEYS}

    phases = deepcopy(source.metadata["phase_seconds"])
    phases["partition_reverification"] = 0.5
    phases["exact_oracle_flow"] = 7.0
    metadata = deepcopy(source.metadata)
    metadata.update(
        {
            "local_observation_start": 0,
            "local_observation_stop": 512,
            "object_id_offset": 25_000,
            "elapsed_seconds": 20.0,
            "phase_seconds": phases,
            "spatial_h256_u8_ps2_center": spatial,
            "exact_oracle": exact,
        }
    )
    metadata = {name: metadata[name] for name in AUDITOR.METADATA_KEYS}
    return SpatialH256U8PS2Diagnostic(frame, metadata)


def _provenance(table_sha256: str) -> dict:
    protocol = SPATIAL_H256_U8_PS2_PROTOCOL
    inference = json.loads(_git_blob("configs/inference.json"))
    likelihood = json.loads(_git_blob("configs/likelihood.json"))
    resolved = deepcopy(inference)
    resolved["proposal"]["candidates"] = protocol["n_candidates"]
    resolved["proposal"]["prefilter_candidates"] = protocol["prefilter_candidates"]
    resolved["proposal"]["seed"] = protocol["proposal_seed"]
    resolved["estimator"]["initial_strategy"] = protocol["initial_strategy"]
    resolved["execution"]["candidate_backend"] = protocol["candidate_backend"]
    resolved["execution"]["object_chunk"] = protocol["diagnostic_object_chunk"]
    resolved["execution"]["atom_chunk"] = protocol["flow_atom_chunk"]
    resolved_hash = sha256(json.dumps(resolved, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    invocation = {
        "diagnose_complement_spatial_h256_u8_ps2": True,
        "diagnose_complement_exact_oracle": True,
        "diagnose_complement_tail_proposal": False,
        "diagnose_complement_pilot_stratum": False,
        "diagnose_complement_spatial_neyman": False,
        "audit_spatial_neyman_partition": False,
        "diagnose_complement_learned_center_checkpoint": None,
        "spatial_neyman_partition_cache": None,
        "observation_start": 25_000,
        "observation_stop": 25_512,
        "proposal_candidates": 16_384,
        "proposal_prefilter_candidates": 131_072,
        "diagnose_complement_global_draws": 16_384,
        "proposal_seed": 8_701,
        "initial_strategy": "mean_observed_shape",
        "candidate_backend": "torch",
        "object_chunk": 32,
        "atom_chunk": 4_096,
        "spatial_h256_u8_ps2_object_chunk": 32,
        "spatial_h256_u8_ps2_atom_chunk": 262_144,
        "cut_abs_ehat": None,
        "cut_bound": [],
        "selection_cache": None,
        "device": "cuda",
        "compile_flow": True,
        "diagnose_complement": "/synthetic/result",
        "spatial_h256_u8_ps2_partition_cache": "/synthetic/partition",
        "mock_input": "/synthetic/mock",
    }
    model = {
        "measurement": likelihood["measurement_model"]["sha256"],
        "emulator": likelihood["emulator"]["model"]["sha256"],
        "emulator_metadata": likelihood["emulator"]["metadata"]["sha256"],
    }
    return {
        "pipeline_release": "diagnostic",
        "pipeline_base_release": inference["release"],
        "pipeline_config_sha256": sha256(_git_blob("configs/inference.json")).hexdigest(),
        "pipeline_resolved_config_sha256": resolved_hash,
        "likelihood_release": likelihood["release"],
        "likelihood_config_sha256": sha256(_git_blob("configs/likelihood.json")).hexdigest(),
        "pipeline_config": resolved,
        "config": invocation,
        "conditions": likelihood["observing_conditions"],
        "selection": None,
        "initial_center": {
            "strategy": protocol["initial_strategy"],
            "center": deepcopy(protocol["center"]),
            "raw_mean_shape": deepcopy(protocol["center"]),
            "target_names": ["measured_ngmix_g1", "measured_ngmix_g2"],
            "n_objects": 125_000,
            "affine_calibration": None,
        },
        "observation_partition": {
            "start": 25_000,
            "stop": 25_512,
            "n_partition": 512,
            "n_total": 125_000,
            "proposal_object_id_offset": 25_000,
        },
        "mock_kind": "likelihood",
        "mock_generated": False,
        "injected_shear": [0.0, 0.0],
        "model_sha256": model,
        "model_cache_sha256": {"manifest.json": _digest("model cache")},
        "scene_sha256": {"manifest.json": _digest("scene")},
        "proposal_cache_sha256": {"manifest.json": _digest("proposal")},
        "mock_input_sha256": deepcopy(protocol["mock_input_sha256"]),
        "spatial_partition_sha256": {
            "partition_sha256": protocol["partition_sha256"],
            "partition_cache_sha256": protocol["partition_cache_sha256"],
            "partition_manifest_sha256": protocol["partition_manifest_sha256"],
        },
        "likelihood_component_sha256": {
            **model,
            "detection": likelihood["emulator"]["detection_model"]["sha256"],
        },
        "pipeline_implementation_sha256": _git_hashes(AUDITOR.PIPELINE_IMPLEMENTATION_PATHS),
        "likelihood_implementation_sha256": _git_hashes(AUDITOR.LIKELIHOOD_IMPLEMENTATION_PATHS),
        "diagnostic_implementation_sha256": _git_hashes(AUDITOR.DIAGNOSTIC_IMPLEMENTATION_PATHS),
        "output_sha256": {"per_object.parquet": table_sha256},
    }


def _write_result(root: Path) -> Path:
    result = root / "result"
    result.mkdir()
    diagnostic = _runtime_diagnostic()
    table_path = result / "per_object.parquet"
    diagnostic.table.to_parquet(table_path)
    summary = diagnostic.summary()
    spatial = spatial_h256_u8_ps2_center_summary(diagnostic)
    summary["spatial_h256_u8_ps2"] = spatial
    summary["gates"] = spatial["gates"]
    payload = {
        "method": AUDITOR.METHOD,
        "metadata": diagnostic.metadata,
        "summary": summary,
        "provenance": _provenance(sha256(table_path.read_bytes()).hexdigest()),
    }
    (result / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    return result


def _load_payload(result: Path) -> dict:
    return json.loads((result / "summary.json").read_text())


def _write_payload(result: Path, payload: dict) -> None:
    (result / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")


@pytest.fixture
def result(tmp_path):
    return _write_result(tmp_path)


def test_audit_accepts_json_roundtrip_and_recomputes_every_gate(result):
    report = AUDITOR.audit_result(result, repository=ROOT)
    assert report["validated"] is True
    assert report["objects"] == 512
    assert report["columns"] == 68
    assert report["audited_implementation_commit"] == ("720a05797602fe2d44831c90443d1264bc630dff")
    assert report["scientific_gates_passed"] is True


def test_audit_rejects_extra_result_file_and_cli_exits_nonzero(result, capsys):
    (result / "unexpected.txt").write_text("unexpected")
    with pytest.raises(AUDITOR.AuditError, match="contain exactly"):
        AUDITOR.audit_result(result, repository=ROOT)
    assert AUDITOR.main([str(result), "--repository", str(ROOT)]) == 1
    assert "artifact audit failed" in capsys.readouterr().err


def test_audit_rejects_summary_metric_tamper(result):
    payload = _load_payload(result)
    payload["summary"]["efficiency_mean"] *= 2.0
    _write_payload(result, payload)
    with pytest.raises(AUDITOR.AuditError, match="metrics or gates"):
        AUDITOR.audit_result(result, repository=ROOT)


def test_audit_rejects_semantically_equal_noncanonical_summary_bytes(result):
    payload = _load_payload(result)
    (result / "summary.json").write_text(json.dumps(payload, separators=(",", ":")))
    with pytest.raises(AUDITOR.AuditError, match="runner encoding"):
        AUDITOR.audit_result(result, repository=ROOT)


def test_audit_rejects_integer_dtype_tamper_even_with_updated_file_hash(result):
    table_path = result / "per_object.parquet"
    frame = pd.read_parquet(table_path)
    frame["stratum_count"] = frame["stratum_count"].astype(np.int32)
    frame.to_parquet(table_path)
    payload = _load_payload(result)
    payload["provenance"]["output_sha256"]["per_object.parquet"] = sha256(table_path.read_bytes()).hexdigest()
    _write_payload(result, payload)
    with pytest.raises(AUDITOR.AuditError, match="not exact int64"):
        AUDITOR.audit_result(result, repository=ROOT)


def test_audit_rejects_exact_derived_column_tamper(result):
    table_path = result / "per_object.parquet"
    frame = pd.read_parquet(table_path)
    selected = np.flatnonzero(frame["exact_oracle_selected"].to_numpy())[0]
    frame.loc[selected, "mass_complement_spatial_h256_u8_ps2_minus_exact_z"] += 0.25
    frame.to_parquet(table_path)
    payload = _load_payload(result)
    payload["provenance"]["output_sha256"]["per_object.parquet"] = sha256(table_path.read_bytes()).hexdigest()
    _write_payload(result, payload)
    with pytest.raises(AUDITOR.AuditError, match="does not recompute"):
        AUDITOR.audit_result(result, repository=ROOT)


@pytest.mark.parametrize(
    "mutate,match",
    (
        (
            lambda payload: payload["metadata"]["spatial_h256_u8_ps2_center"]["execution_call_provenance"][
                0
            ].__setitem__("seed", 1),
            "integrity gates failed",
        ),
        (
            lambda payload: payload["metadata"].__setitem__("center", [0.0, 0.0]),
            "frozen protocol",
        ),
        (
            lambda payload: payload["provenance"]["pipeline_implementation_sha256"].__setitem__(
                "scripts/run_inference.py", "0" * 64
            ),
            "Git blob bytes",
        ),
    ),
)
def test_audit_rejects_provenance_and_protocol_tamper(result, mutate, match):
    payload = _load_payload(result)
    mutate(payload)
    _write_payload(result, payload)
    with pytest.raises(AUDITOR.AuditError, match=match):
        AUDITOR.audit_result(result, repository=ROOT)


def test_optional_scheduler_stream_checks_and_hashes(result, tmp_path):
    stdout = tmp_path / "stdout"
    stderr = tmp_path / "stderr"
    stdout.write_bytes((result / "summary.json").read_bytes())
    stderr.write_text(
        "/tmp/torch.py:10: UserWarning: TensorFloat32 tensor cores for float32 "
        "matrix multiplication available but not enabled. Consider enabling it.\n"
        "  warnings.warn(message)\n"
    )
    report = AUDITOR.audit_result(
        result,
        repository=ROOT,
        scheduler_stdout=stdout,
        scheduler_stderr=stderr,
    )
    assert report["scheduler_stdout_sha256"] == report["summary_json_sha256"]
    assert report["scheduler_stderr_bytes"] == len(stderr.read_bytes())

    stderr.write_text("Traceback (most recent call last):\n")
    with pytest.raises(AUDITOR.AuditError, match="non-warning"):
        AUDITOR.audit_result(result, repository=ROOT, scheduler_stderr=stderr)
