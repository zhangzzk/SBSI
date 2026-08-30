#!/usr/bin/env python3
"""Read-only auditor for one frozen H256-U8-PS2 result directory."""

from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from sbsi.complement_diagnostic import exact_oracle_object_ids  # noqa: E402
from sbsi.spatial_h256_u8_ps2_diagnostic import (  # noqa: E402
    SPATIAL_H256_U8_PS2_METHOD_TIME_COMPONENTS,
    SPATIAL_H256_U8_PS2_PROTOCOL,
    SpatialH256U8PS2Diagnostic,
    _SPATIAL_H256_U8_PS2_MANIFEST_HASH_COLUMNS,
    spatial_h256_u8_ps2_center_summary,
    spatial_h256_u8_ps2_run_manifest_sha256,
)


AUDIT_VERSION = "spatial_h256_u8_ps2_result_audit_v1"
AUDITED_IMPLEMENTATION_COMMIT = "720a05797602fe2d44831c90443d1264bc630dff"

RESULT_FILENAMES = ("per_object.parquet", "summary.json")
METHOD = "held-out complement spatial H256-U8-PS2 diagnostic"

PIPELINE_IMPLEMENTATION_PATHS = (
    "configs/inference.json",
    "scripts/run_inference.py",
    "sbsi/amortized_proposal.py",
    "sbsi/catalogue_null.py",
    "sbsi/catalogue_likelihood.py",
    "sbsi/catalogue_sampling.py",
    "sbsi/learned_retrieval_diagnostic.py",
    "sbsi/spatial_neyman_diagnostic.py",
    "sbsi/spatial_h256_u8_ps2_diagnostic.py",
)
LIKELIHOOD_IMPLEMENTATION_PATHS = (
    "configs/likelihood.json",
    "sbsi/catalogue_likelihood.py",
    "sbsi/catalogue_blend.py",
    "sbsi/catalogue_closure.py",
    "sbsi/scene_prior.py",
    "sbsi/shear_map.py",
    "sbsi/measurement_model.py",
    "sbsi/models.py",
)
DIAGNOSTIC_IMPLEMENTATION_PATHS = (
    "sbsi/complement_diagnostic.py",
    "jobs/job_complement_exact_oracle.sh",
    "sbsi/spatial_h256_u8_ps2_diagnostic.py",
    "jobs/job_complement_spatial_h256_u8_ps2_center.sh",
)

INTEGER_COLUMNS = (
    "object_index",
    "stratum_count",
    "stratum_unique_count",
    "stratum_active_count",
    "production_draws",
    "production_cells",
    "production_microstrata",
    "production_source_proxy_t1",
    "production_source_proxy_t2",
    "production_source_prior",
    "production_draws_in_stratum",
    "source_audit_failed_count",
    "fallback_scaled_law_mismatch_count",
    "source_excluded_tick_sum",
    "selected_source_tick_overflow_count",
    "numerator_sum_failure_count",
    "probability_identity_failure_count",
    "cdf_bracket_failure_count",
    "rng_recomputation_failure_count",
    "partition_cell_membership_failure_count",
    "probability_numerator_minimum",
    "probability_numerator_maximum",
    "u48_minimum",
    "u48_maximum",
    "prior_minimum_eligible_tick",
    "terminal_correction_maximum",
    "terminal_correction_positive_count",
    "proxy_t1_prior_fallback_cells",
    "proxy_t2_prior_fallback_cells",
)
BOOLEAN_COLUMNS = (
    "canonical_schedule_exact",
    "integer_source_totals_exact",
    "integer_prior_floor_satisfied",
    "integer_proxy_fallback_scaled_exact",
    "integer_balance_probability_exact",
    "integer_cdf_brackets_exact",
    "partition_cell_membership_exact",
    "exact_oracle_selected",
)
EXACT_FLOAT_COLUMNS = (
    "mass_stratum_exact_oracle",
    "mass_stratum_exact_oracle_relative_error",
    "mass_complement_exact_oracle",
    "mass_total_exact_oracle",
    "mass_complement_spatial_h256_u8_ps2_minus_exact_z",
    "delta_log_total_evidence_spatial_h256_u8_ps2_minus_exact",
)
FLOAT_COLUMNS = (
    "core_cell_mass_crosscheck_relative_max",
    "core_cell_pair_sum_crosscheck_relative_max",
    "log_mass_stratum",
    "mass_stratum",
    "log_mass_complement_spatial_h256_u8_ps2",
    "mass_complement_spatial_h256_u8_ps2",
    "mass_complement_spatial_h256_u8_ps2_standard_error",
    "log_variance_complement_spatial_h256_u8_ps2",
    "pareto_k_spatial_h256_u8_ps2_complement",
    "maximum_effective_weight_spatial_h256_u8_ps2",
    *EXACT_FLOAT_COLUMNS,
)

# This is the insertion order of the row record in the frozen diagnostic.  A
# different order is a different Parquet schema even if pandas can select the
# same names afterward.
EXPECTED_TABLE_COLUMNS = (
    "object_index",
    "stratum_count",
    "stratum_unique_count",
    "stratum_active_count",
    "stratum_indices_sha256",
    "stratum_log_target_sha256",
    "production_log_ratio_sha256",
    "production_indices_sha256",
    "production_probability_sha256",
    "production_probability_numerator_sha256",
    "production_selected_source_ticks_sha256",
    "production_selected_source_cdf_lower_sha256",
    "production_u48_sha256",
    "production_normalizer_sha256",
    "production_quantization_audit_sha256",
    "production_exclusion_indices_sha256",
    "production_schedule_sha256",
    "production_cell_order_sha256",
    "production_draw_record_sha256",
    "production_draws",
    "production_cells",
    "production_microstrata",
    "production_source_proxy_t1",
    "production_source_proxy_t2",
    "production_source_prior",
    "production_draws_in_stratum",
    "canonical_schedule_exact",
    "integer_source_totals_exact",
    "integer_prior_floor_satisfied",
    "integer_proxy_fallback_scaled_exact",
    "integer_balance_probability_exact",
    "integer_cdf_brackets_exact",
    "source_audit_failed_count",
    "fallback_scaled_law_mismatch_count",
    "source_excluded_tick_sum",
    "selected_source_tick_overflow_count",
    "numerator_sum_failure_count",
    "probability_identity_failure_count",
    "cdf_bracket_failure_count",
    "rng_recomputation_failure_count",
    "partition_cell_membership_failure_count",
    "partition_cell_membership_exact",
    "probability_numerator_minimum",
    "probability_numerator_maximum",
    "u48_minimum",
    "u48_maximum",
    "prior_minimum_eligible_tick",
    "terminal_correction_maximum",
    "terminal_correction_positive_count",
    "proxy_t1_prior_fallback_cells",
    "proxy_t2_prior_fallback_cells",
    "core_cell_mass_crosscheck_relative_max",
    "core_cell_pair_sum_crosscheck_relative_max",
    "log_mass_stratum",
    "mass_stratum",
    "log_mass_complement_spatial_h256_u8_ps2",
    "mass_complement_spatial_h256_u8_ps2",
    "mass_complement_spatial_h256_u8_ps2_standard_error",
    "log_variance_complement_spatial_h256_u8_ps2",
    "pareto_k_spatial_h256_u8_ps2_complement",
    "maximum_effective_weight_spatial_h256_u8_ps2",
    "exact_oracle_selected",
    *EXACT_FLOAT_COLUMNS,
)

METADATA_KEYS = (
    "center",
    "n_candidates",
    "prefilter_candidates",
    "candidate_backend",
    "object_chunk",
    "flow_atom_chunk",
    "observation_start",
    "observation_stop",
    "local_observation_start",
    "local_observation_stop",
    "object_id_offset",
    "flow_evaluations",
    "elapsed_seconds",
    "phase_seconds",
    "spatial_h256_u8_ps2_center",
    "exact_oracle",
)
SPATIAL_METADATA_KEYS = (
    "label",
    "initial_strategy",
    "partition_sha256",
    "partition_cache_sha256",
    "partition_manifest_sha256",
    "partition_verified",
    "partition_recipe_sha256",
    "minimum_cell_active_count",
    "maximum_cell_active_count",
    "maximum_four_cell_slab_active_count",
    "n_cells",
    "spatial_atom_chunk",
    "spatial_object_chunk",
    "n_global_draws",
    "proposal_seed",
    "production_seed",
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
    "proposal_initialization_seconds",
    "mock_input_sha256",
    "stratum_flow_evaluations",
    "production_flow_evaluations",
    "proxy_prior_fallback_cells_by_component",
    "objects_with_proxy_prior_fallback",
    "run_manifest_sha256",
    "execution_call_provenance",
)
EXACT_METADATA_KEYS = (
    "selection_count",
    "selected_object_ids",
    "selection_manifest_sha256",
    "atom_chunk",
    "object_chunk",
    "active_atoms",
    "active_atoms_visited_per_selected_object",
    "atom_blocks_per_selected_object",
    "active_indices_sha256",
    "flow_evaluations",
)
CALL_KEYS = (
    "object_start",
    "object_stop",
    "objects",
    "seed",
    "atom_chunk",
    "object_chunk",
    "cells_per_slab",
    "partition_sha256",
    "record_sha256",
    "schedule_sha256",
    "quantization_sha256",
    "cell_order_sha256",
    "sampler_version",
    "rng_contract",
    "quantizer_contract",
    "hash_contract",
    "order_contract",
)
PHASE_KEYS = (
    *SPATIAL_H256_U8_PS2_METHOD_TIME_COMPONENTS,
    "partition_reverification",
    "exact_oracle_flow",
)
PROVENANCE_KEYS = (
    "pipeline_release",
    "pipeline_base_release",
    "pipeline_config_sha256",
    "pipeline_resolved_config_sha256",
    "likelihood_release",
    "likelihood_config_sha256",
    "pipeline_config",
    "config",
    "conditions",
    "selection",
    "initial_center",
    "observation_partition",
    "mock_kind",
    "mock_generated",
    "injected_shear",
    "model_sha256",
    "model_cache_sha256",
    "scene_sha256",
    "proposal_cache_sha256",
    "mock_input_sha256",
    "spatial_partition_sha256",
    "likelihood_component_sha256",
    "pipeline_implementation_sha256",
    "likelihood_implementation_sha256",
    "diagnostic_implementation_sha256",
    "output_sha256",
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_WARNING_HEADER_RE = re.compile(r"^.+:\d+: UserWarning: (?P<message>.+)$")
_ALLOWED_WARNING_PREFIXES = (
    "TensorFloat32 tensor cores for float32 matrix multiplication available but not enabled.",
    "The given NumPy array is not writable, and PyTorch does not support non-writable tensors.",
)


class AuditError(RuntimeError):
    """Raised when a frozen result identity fails closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def _sha256_bytes(value: bytes) -> str:
    return sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _strict_json_bytes(value: bytes, *, label: str) -> dict[str, Any]:
    def reject_constant(token: str) -> None:
        raise AuditError(f"{label} contains non-standard JSON constant {token}")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise AuditError(f"{label} contains duplicate JSON key {key!r}")
            result[key] = item
        return result

    try:
        parsed = json.loads(
            value.decode("utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=unique_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AuditError(f"{label} is not strict UTF-8 JSON: {error}") from error
    _require(isinstance(parsed, dict), f"{label} must contain one JSON object")
    return parsed


def _canonical_json_sha256(payload: dict[str, Any]) -> str:
    value = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return _sha256_bytes(value)


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _validate_sha_mapping(
    value: Any,
    *,
    label: str,
    expected_keys: tuple[str, ...] | None = None,
) -> dict[str, str]:
    _require(isinstance(value, dict) and value, f"{label} must be a nonempty object")
    if expected_keys is not None:
        _require(tuple(value) == expected_keys, f"{label} keys or order changed")
    _require(
        all(isinstance(key, str) and _is_sha256(item) for key, item in value.items()),
        f"{label} contains a non-SHA-256 identity",
    )
    return value


def _git_blob(repository: Path, path: str) -> bytes:
    result = subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "show",
            f"{AUDITED_IMPLEMENTATION_COMMIT}:{path}",
        ],
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise AuditError(f"cannot read {path} at the audited commit: {detail}")
    return result.stdout


def _verify_commit(repository: Path) -> None:
    result = subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "rev-parse",
            "--verify",
            f"{AUDITED_IMPLEMENTATION_COMMIT}^{{commit}}",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    _require(result.returncode == 0, "audited implementation commit is unavailable")
    _require(
        result.stdout.strip() == AUDITED_IMPLEMENTATION_COMMIT,
        "audited implementation commit resolved ambiguously",
    )


def _git_hashes(repository: Path, paths: tuple[str, ...]) -> dict[str, str]:
    return {path: _sha256_bytes(_git_blob(repository, path)) for path in paths}


def _git_json(repository: Path, path: str) -> dict[str, Any]:
    return _strict_json_bytes(_git_blob(repository, path), label=f"{path} at commit")


def _validate_result_directory(root: Path) -> tuple[Path, Path]:
    _require(root.is_dir() and not root.is_symlink(), "result path is not a real directory")
    entries = tuple(sorted(item.name for item in root.iterdir()))
    _require(
        entries == RESULT_FILENAMES,
        "result directory must contain exactly per_object.parquet and summary.json",
    )
    table = root / "per_object.parquet"
    summary = root / "summary.json"
    for path in (table, summary):
        _require(path.is_file() and not path.is_symlink(), f"{path.name} is not a real file")
    return table, summary


def _validate_table(frame: pd.DataFrame) -> str:
    _require(tuple(frame.columns) == EXPECTED_TABLE_COLUMNS, "Parquet column identity or order changed")
    _require(
        isinstance(frame.index, pd.RangeIndex)
        and frame.index.start == 0
        and frame.index.stop == 512
        and frame.index.step == 1,
        "Parquet row index is not the canonical 0:512 RangeIndex",
    )
    _require(len(frame) == 512, "Parquet must contain exactly 512 rows")
    for name in INTEGER_COLUMNS:
        _require(frame[name].dtype == np.dtype("int64"), f"{name} is not exact int64")
    for name in BOOLEAN_COLUMNS:
        _require(frame[name].dtype == np.dtype("bool"), f"{name} is not exact bool")
    for name in FLOAT_COLUMNS:
        _require(frame[name].dtype == np.dtype("float64"), f"{name} is not exact float64")
    for name in _SPATIAL_H256_U8_PS2_MANIFEST_HASH_COLUMNS:
        _require(frame[name].dtype == np.dtype("O"), f"{name} is not the frozen object/string dtype")
        _require(all(_is_sha256(value) for value in frame[name]), f"{name} contains an invalid SHA-256")

    object_ids = frame["object_index"].to_numpy(dtype=np.int64)
    expected_ids = np.arange(25_000, 25_512, dtype=np.int64)
    _require(np.array_equal(object_ids, expected_ids), "absolute object IDs are not exactly 25000:25512")
    _require(
        all((frame[name].to_numpy(dtype=np.int64) >= 0).all() for name in INTEGER_COLUMNS),
        "an integer audit column contains a negative value",
    )
    selected_ids, _ = exact_oracle_object_ids(25_000, 25_512)
    selected = frame["exact_oracle_selected"].to_numpy(dtype=bool)
    _require(np.array_equal(object_ids[selected], selected_ids), "exact-oracle row mask changed")
    _require(int(selected.sum()) == 64, "exact-oracle row count changed")
    for name in EXACT_FLOAT_COLUMNS:
        values = frame[name].to_numpy(dtype=np.float64)
        _require(np.isfinite(values[selected]).all(), f"{name} is non-finite on an exact row")
        _require(np.isnan(values[~selected]).all(), f"{name} is populated outside the exact subset")

    exact_stratum = frame.loc[selected, "mass_stratum_exact_oracle"].to_numpy(dtype=np.float64)
    exact_complement = frame.loc[selected, "mass_complement_exact_oracle"].to_numpy(dtype=np.float64)
    exact_total = frame.loc[selected, "mass_total_exact_oracle"].to_numpy(dtype=np.float64)
    estimated_stratum = frame.loc[selected, "mass_stratum"].to_numpy(dtype=np.float64)
    estimated_complement = frame.loc[selected, "mass_complement_spatial_h256_u8_ps2"].to_numpy(
        dtype=np.float64
    )
    estimated_se = frame.loc[selected, "mass_complement_spatial_h256_u8_ps2_standard_error"].to_numpy(
        dtype=np.float64
    )
    expected_relative = np.abs(estimated_stratum - exact_stratum) / np.abs(exact_stratum)
    expected_z = (estimated_complement - exact_complement) / estimated_se
    expected_delta = np.logaddexp(
        frame.loc[selected, "log_mass_stratum"].to_numpy(dtype=np.float64),
        frame.loc[selected, "log_mass_complement_spatial_h256_u8_ps2"].to_numpy(dtype=np.float64),
    ) - np.log(exact_total)
    tolerance = 64.0 * np.finfo(np.float64).eps
    _require(
        (exact_stratum > 0.0).all()
        and (exact_complement > 0.0).all()
        and (exact_total > 0.0).all()
        and np.allclose(
            exact_total,
            exact_stratum + exact_complement,
            rtol=tolerance,
            atol=0.0,
        ),
        "exact-oracle physical masses do not close",
    )
    for name, observed, expected in (
        (
            "mass_stratum_exact_oracle_relative_error",
            frame.loc[selected, "mass_stratum_exact_oracle_relative_error"].to_numpy(dtype=np.float64),
            expected_relative,
        ),
        (
            "mass_complement_spatial_h256_u8_ps2_minus_exact_z",
            frame.loc[
                selected,
                "mass_complement_spatial_h256_u8_ps2_minus_exact_z",
            ].to_numpy(dtype=np.float64),
            expected_z,
        ),
        (
            "delta_log_total_evidence_spatial_h256_u8_ps2_minus_exact",
            frame.loc[
                selected,
                "delta_log_total_evidence_spatial_h256_u8_ps2_minus_exact",
            ].to_numpy(dtype=np.float64),
            expected_delta,
        ),
    ):
        _require(
            np.allclose(observed, expected, rtol=tolerance, atol=tolerance),
            f"{name} does not recompute",
        )

    ordinary_float_columns = tuple(
        name
        for name in FLOAT_COLUMNS
        if name not in (*EXACT_FLOAT_COLUMNS, "pareto_k_spatial_h256_u8_ps2_complement")
    )
    _require(
        all(np.isfinite(frame[name].to_numpy(dtype=np.float64)).all() for name in ordinary_float_columns),
        "a production float column contains NaN or infinity",
    )
    _require(
        not np.isinf(frame["pareto_k_spatial_h256_u8_ps2_complement"].to_numpy(dtype=np.float64)).any(),
        "Pareto k contains infinity",
    )
    _require(
        (frame["maximum_effective_weight_spatial_h256_u8_ps2"] > 0.0).all(),
        "maximum effective weights must be positive",
    )
    _require(
        (frame["terminal_correction_maximum"] == 0).all()
        and (frame["terminal_correction_positive_count"] == 0).all(),
        "cumulative integer-CDF terminal correction was unexpectedly nonzero",
    )
    for name in ("proxy_t1_prior_fallback_cells", "proxy_t2_prior_fallback_cells"):
        _require((frame[name] <= 256).all(), f"{name} exceeds the cell count")
    manifest = spatial_h256_u8_ps2_run_manifest_sha256(frame)
    _require(_is_sha256(manifest), "recomputed run manifest is invalid")
    return manifest


def _validate_metadata(metadata: Any, frame: pd.DataFrame, run_manifest: str) -> dict[str, Any]:
    protocol = SPATIAL_H256_U8_PS2_PROTOCOL
    _require(isinstance(metadata, dict), "metadata is not an object")
    _require(tuple(metadata) == METADATA_KEYS, "metadata keys or order changed")
    spatial = metadata["spatial_h256_u8_ps2_center"]
    exact = metadata["exact_oracle"]
    phases = metadata["phase_seconds"]
    _require(
        isinstance(spatial, dict) and tuple(spatial) == SPATIAL_METADATA_KEYS,
        "spatial metadata keys or order changed",
    )
    _require(
        isinstance(exact, dict) and tuple(exact) == EXACT_METADATA_KEYS,
        "exact metadata keys or order changed",
    )
    _require(isinstance(phases, dict) and tuple(phases) == PHASE_KEYS, "phase timing keys or order changed")
    _require(
        all(np.isfinite(float(value)) and float(value) >= 0.0 for value in phases.values()),
        "phase timing contains a negative or non-finite value",
    )
    _require(
        np.isfinite(float(metadata["elapsed_seconds"])) and float(metadata["elapsed_seconds"]) > 0.0,
        "elapsed time is not finite and positive",
    )
    _require(spatial["run_manifest_sha256"] == run_manifest, "run manifest does not recompute")

    fallback = spatial["proxy_prior_fallback_cells_by_component"]
    _require(
        isinstance(fallback, dict) and tuple(fallback) == ("proxy_t1", "proxy_t2"),
        "fallback aggregate identity changed",
    )
    expected_fallback = {
        "proxy_t1": int(frame["proxy_t1_prior_fallback_cells"].sum()),
        "proxy_t2": int(frame["proxy_t2_prior_fallback_cells"].sum()),
    }
    _require(fallback == expected_fallback, "fallback aggregate does not recompute")
    expected_fallback_objects = int(
        ((frame["proxy_t1_prior_fallback_cells"] > 0) | (frame["proxy_t2_prior_fallback_cells"] > 0)).sum()
    )
    _require(
        spatial["objects_with_proxy_prior_fallback"] == expected_fallback_objects,
        "fallback object count does not recompute",
    )

    calls = spatial["execution_call_provenance"]
    _require(isinstance(calls, list) and len(calls) == 16, "execution call count changed")
    _require(
        all(isinstance(call, dict) and tuple(call) == CALL_KEYS for call in calls),
        "execution call provenance keys or order changed",
    )

    expected_ids, expected_selection_manifest = exact_oracle_object_ids(25_000, 25_512)
    _require(exact["selected_object_ids"] == expected_ids.tolist(), "recorded exact IDs changed")
    _require(
        exact["selection_manifest_sha256"] == expected_selection_manifest, "exact selection manifest changed"
    )
    _require(exact["selection_count"] == 64, "exact selection count changed")
    _require(exact["active_atoms"] == protocol["active_atoms"], "exact active-atom count changed")
    _require(
        exact["active_atoms_visited_per_selected_object"] == protocol["active_atoms"],
        "exact scan did not visit every active atom",
    )
    expected_blocks = (protocol["active_atoms"] + protocol["exact_atom_chunk"] - 1) // protocol[
        "exact_atom_chunk"
    ]
    _require(exact["atom_blocks_per_selected_object"] == expected_blocks, "exact atom-block count changed")
    _require(_is_sha256(exact["active_indices_sha256"]), "exact active-index identity is invalid")
    _require(
        exact["flow_evaluations"] == protocol["exact_flow_evaluations"],
        "exact flow count changed",
    )
    return metadata


def _expected_pipeline_config(repository: Path) -> dict[str, Any]:
    protocol = SPATIAL_H256_U8_PS2_PROTOCOL
    expected = deepcopy(_git_json(repository, "configs/inference.json"))
    expected["proposal"]["candidates"] = protocol["n_candidates"]
    expected["proposal"]["prefilter_candidates"] = protocol["prefilter_candidates"]
    expected["proposal"]["seed"] = protocol["proposal_seed"]
    expected["estimator"]["initial_strategy"] = protocol["initial_strategy"]
    expected["execution"]["candidate_backend"] = protocol["candidate_backend"]
    expected["execution"]["object_chunk"] = protocol["diagnostic_object_chunk"]
    expected["execution"]["atom_chunk"] = protocol["flow_atom_chunk"]
    return expected


def _validate_protocol_arguments(config: Any) -> None:
    protocol = SPATIAL_H256_U8_PS2_PROTOCOL
    _require(isinstance(config, dict), "invocation config is not an object")
    expected = {
        "diagnose_complement_spatial_h256_u8_ps2": True,
        "diagnose_complement_exact_oracle": True,
        "diagnose_complement_tail_proposal": False,
        "diagnose_complement_pilot_stratum": False,
        "diagnose_complement_spatial_neyman": False,
        "audit_spatial_neyman_partition": False,
        "diagnose_complement_learned_center_checkpoint": None,
        "spatial_neyman_partition_cache": None,
        "observation_start": protocol["universe_start"],
        "observation_stop": protocol["universe_stop"],
        "proposal_candidates": protocol["n_candidates"],
        "proposal_prefilter_candidates": protocol["prefilter_candidates"],
        "diagnose_complement_global_draws": protocol["n_global_draws"],
        "proposal_seed": protocol["proposal_seed"],
        "initial_strategy": protocol["initial_strategy"],
        "candidate_backend": protocol["candidate_backend"],
        "object_chunk": protocol["diagnostic_object_chunk"],
        "atom_chunk": protocol["flow_atom_chunk"],
        "spatial_h256_u8_ps2_object_chunk": protocol["spatial_object_chunk"],
        "spatial_h256_u8_ps2_atom_chunk": protocol["spatial_atom_chunk"],
        "cut_abs_ehat": None,
        "cut_bound": [],
        "selection_cache": None,
        "device": "cuda",
        "compile_flow": True,
    }
    mismatched = {
        name: {"expected": value, "observed": config.get(name)}
        for name, value in expected.items()
        if config.get(name) != value
    }
    _require(not mismatched, f"invocation protocol mismatch: {mismatched}")
    for name in (
        "diagnose_complement",
        "spatial_h256_u8_ps2_partition_cache",
        "mock_input",
    ):
        _require(isinstance(config.get(name), str) and config[name], f"{name} is missing")


def _validate_provenance(
    provenance: Any,
    *,
    repository: Path,
    table_sha256: str,
) -> dict[str, Any]:
    protocol = SPATIAL_H256_U8_PS2_PROTOCOL
    _require(isinstance(provenance, dict), "provenance is not an object")
    _require(tuple(provenance) == PROVENANCE_KEYS, "provenance keys or order changed")
    inference_config = _git_json(repository, "configs/inference.json")
    likelihood_config = _git_json(repository, "configs/likelihood.json")
    _require(provenance["pipeline_release"] == "diagnostic", "pipeline release is not diagnostic")
    _require(
        provenance["pipeline_base_release"] == inference_config["release"], "pipeline base release changed"
    )
    _require(provenance["likelihood_release"] == likelihood_config["release"], "likelihood release changed")
    _require(
        provenance["pipeline_config_sha256"]
        == _sha256_bytes(_git_blob(repository, "configs/inference.json")),
        "inference config bytes do not match the audited commit",
    )
    _require(
        provenance["likelihood_config_sha256"]
        == _sha256_bytes(_git_blob(repository, "configs/likelihood.json")),
        "likelihood config bytes do not match the audited commit",
    )
    _require(
        provenance["pipeline_config"] == _expected_pipeline_config(repository),
        "resolved pipeline config changed",
    )
    _require(
        provenance["pipeline_resolved_config_sha256"]
        == _canonical_json_sha256(provenance["pipeline_config"]),
        "resolved pipeline config hash does not recompute",
    )
    _validate_protocol_arguments(provenance["config"])

    _require(
        provenance["conditions"] == likelihood_config["observing_conditions"], "observing conditions changed"
    )
    _require(provenance["selection"] is None, "measured selection was enabled")
    _require(provenance["mock_kind"] == "likelihood", "mock kind changed")
    _require(provenance["mock_generated"] is False, "mock was generated in the science invocation")
    shear = np.asarray(provenance["injected_shear"], dtype=np.float64)
    _require(shear.shape == (2,) and np.isfinite(shear).all(), "injected shear is invalid")

    initial = provenance["initial_center"]
    _require(isinstance(initial, dict), "initial-center provenance is invalid")
    _require(initial.get("strategy") == protocol["initial_strategy"], "initial-center strategy changed")
    _require(initial.get("center") == protocol["center"], "initial center changed")
    _require(initial.get("raw_mean_shape") == protocol["center"], "raw initial mean changed")
    _require(initial.get("affine_calibration") is None, "initial-center calibration was enabled")
    expected_initial_targets = ["measured_ngmix_g1", "measured_ngmix_g2"]
    _require(
        all(
            name in likelihood_config["measurement_model"]["target_names"]
            for name in expected_initial_targets
        ),
        "audited likelihood no longer contains the initial shape targets",
    )
    _require(
        initial.get("target_names") == expected_initial_targets,
        "initial target names changed",
    )
    partition = provenance["observation_partition"]
    _require(
        isinstance(partition, dict)
        and partition.get("start") == protocol["universe_start"]
        and partition.get("stop") == protocol["universe_stop"]
        and partition.get("n_partition") == 512
        and partition.get("proposal_object_id_offset") == protocol["universe_start"]
        and isinstance(partition.get("n_total"), int)
        and partition["n_total"] >= protocol["universe_stop"],
        "observation partition identity changed",
    )
    _require(initial.get("n_objects") == partition["n_total"], "initial-center population count changed")

    expected_model = {
        "measurement": likelihood_config["measurement_model"]["sha256"],
        "emulator": likelihood_config["emulator"]["model"]["sha256"],
        "emulator_metadata": likelihood_config["emulator"]["metadata"]["sha256"],
    }
    _require(provenance["model_sha256"] == expected_model, "model identities changed")
    expected_components = {
        **expected_model,
        "detection": likelihood_config["emulator"]["detection_model"]["sha256"],
    }
    _require(
        provenance["likelihood_component_sha256"] == expected_components,
        "likelihood component identities changed",
    )
    for name in ("model_cache_sha256", "scene_sha256", "proposal_cache_sha256"):
        _validate_sha_mapping(provenance[name], label=name)
    _require(
        provenance["mock_input_sha256"] == protocol["mock_input_sha256"], "frozen mock identities changed"
    )
    expected_partition = {
        "partition_sha256": protocol["partition_sha256"],
        "partition_cache_sha256": protocol["partition_cache_sha256"],
        "partition_manifest_sha256": protocol["partition_manifest_sha256"],
    }
    _require(
        provenance["spatial_partition_sha256"] == expected_partition, "spatial partition identities changed"
    )

    implementation_specs = (
        ("pipeline_implementation_sha256", PIPELINE_IMPLEMENTATION_PATHS),
        ("likelihood_implementation_sha256", LIKELIHOOD_IMPLEMENTATION_PATHS),
        ("diagnostic_implementation_sha256", DIAGNOSTIC_IMPLEMENTATION_PATHS),
    )
    for name, paths in implementation_specs:
        recorded = _validate_sha_mapping(provenance[name], label=name, expected_keys=paths)
        _require(
            recorded == _git_hashes(repository, paths),
            f"{name} does not match Git blob bytes at {AUDITED_IMPLEMENTATION_COMMIT}",
        )
    _require(
        provenance["output_sha256"] == {"per_object.parquet": table_sha256},
        "embedded Parquet output identity does not recompute",
    )
    return provenance


def _recomputed_summary(frame: pd.DataFrame, metadata: dict[str, Any]) -> dict[str, Any]:
    # The runner computes its summary before JSON serialization.  Restore the
    # four tuple-valued recipe fields that JSON deliberately writes as arrays.
    runtime_metadata = deepcopy(metadata)
    recipe = runtime_metadata["spatial_h256_u8_ps2_center"]["tail_proposal_recipe"]
    for name in (
        "component_names",
        "component_weights",
        "component_betas",
        "component_cycle",
    ):
        _require(
            isinstance(recipe.get(name), list),
            f"serialized recipe field {name} is not an array",
        )
        recipe[name] = tuple(recipe[name])
    diagnostic = SpatialH256U8PS2Diagnostic(frame, runtime_metadata)
    summary = diagnostic.summary()
    spatial = spatial_h256_u8_ps2_center_summary(diagnostic)
    summary["spatial_h256_u8_ps2"] = spatial
    summary["gates"] = spatial["gates"]
    _require(summary["protocol"]["matched"] is True, "frozen protocol did not recompute")
    integrity = summary["gates"]["integrity"]
    _require(
        isinstance(integrity, dict) and all(item.get("passed") is True for item in integrity.values()),
        "one or more recomputed integrity gates failed",
    )
    # Compare in the JSON domain, where the runtime tuples above intentionally
    # have the same representation as the emitted arrays.
    try:
        return json.loads(json.dumps(summary, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise AuditError(f"recomputed summary is not finite JSON: {error}") from error


def _validate_scheduler_stderr(value: bytes) -> None:
    if value == b"":
        return
    try:
        text = value.decode("utf-8")
    except UnicodeDecodeError as error:
        raise AuditError("scheduler stderr is not UTF-8") from error
    lines = text.splitlines()
    _require(bool(lines), "scheduler stderr contains only line separators")
    seen: set[str] = set()
    index = 0
    while index < len(lines):
        match = _WARNING_HEADER_RE.fullmatch(lines[index])
        _require(match is not None, "scheduler stderr contains non-warning output")
        message = match.group("message")
        matched_prefix = next(
            (prefix for prefix in _ALLOWED_WARNING_PREFIXES if message.startswith(prefix)),
            None,
        )
        _require(matched_prefix is not None, "scheduler stderr contains an unknown warning")
        _require(matched_prefix not in seen, "scheduler stderr repeats a known warning")
        seen.add(matched_prefix)
        index += 1
        if index < len(lines) and lines[index].startswith("  "):
            index += 1


def audit_result(
    result_directory: str | Path,
    *,
    repository: str | Path = REPOSITORY_ROOT,
    scheduler_stdout: str | Path | None = None,
    scheduler_stderr: str | Path | None = None,
) -> dict[str, Any]:
    """Audit one result without modifying it or consulting science inputs."""

    root = Path(result_directory)
    repository_path = Path(repository)
    _verify_commit(repository_path)
    table_path, summary_path = _validate_result_directory(root)
    table_sha256 = _sha256_file(table_path)
    summary_bytes = summary_path.read_bytes()
    payload = _strict_json_bytes(summary_bytes, label="summary.json")
    _require(
        tuple(payload) == ("method", "metadata", "summary", "provenance"),
        "summary.json top-level keys or order changed",
    )
    _require(payload["method"] == METHOD, "diagnostic method identity changed")

    try:
        frame = pd.read_parquet(table_path)
    except Exception as error:
        raise AuditError(f"cannot read per_object.parquet: {error}") from error
    run_manifest = _validate_table(frame)
    metadata = _validate_metadata(payload["metadata"], frame, run_manifest)
    provenance = _validate_provenance(
        payload["provenance"],
        repository=repository_path,
        table_sha256=table_sha256,
    )
    recomputed = _recomputed_summary(frame, metadata)
    _require(payload["summary"] == recomputed, "emitted summary metrics or gates do not recompute")
    expected_payload = {
        "method": METHOD,
        "metadata": metadata,
        "summary": recomputed,
        "provenance": provenance,
    }
    _require(payload == expected_payload, "summary.json semantic content changed")
    expected_bytes = (json.dumps(expected_payload, indent=2) + "\n").encode()
    _require(summary_bytes == expected_bytes, "summary.json bytes are not the frozen runner encoding")

    report: dict[str, Any] = {
        "audit_version": AUDIT_VERSION,
        "validated": True,
        "result_directory": str(root.resolve()),
        "audited_implementation_commit": AUDITED_IMPLEMENTATION_COMMIT,
        "per_object_parquet_sha256": table_sha256,
        "summary_json_sha256": _sha256_bytes(summary_bytes),
        "run_manifest_sha256": run_manifest,
        "objects": 512,
        "columns": len(EXPECTED_TABLE_COLUMNS),
        "scientific_gates_passed": bool(recomputed["gates"]["passed"]),
    }
    if scheduler_stdout is not None:
        stdout_path = Path(scheduler_stdout)
        stdout_bytes = stdout_path.read_bytes()
        _require(stdout_bytes == summary_bytes, "scheduler stdout is not byte-identical to summary.json")
        report["scheduler_stdout_sha256"] = _sha256_bytes(stdout_bytes)
    if scheduler_stderr is not None:
        stderr_path = Path(scheduler_stderr)
        stderr_bytes = stderr_path.read_bytes()
        _validate_scheduler_stderr(stderr_bytes)
        report["scheduler_stderr_sha256"] = _sha256_bytes(stderr_bytes)
        report["scheduler_stderr_bytes"] = len(stderr_bytes)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_directory", type=Path)
    parser.add_argument("--repository", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--scheduler-stdout", type=Path)
    parser.add_argument("--scheduler-stderr", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = audit_result(
            args.result_directory,
            repository=args.repository,
            scheduler_stdout=args.scheduler_stdout,
            scheduler_stderr=args.scheduler_stderr,
        )
    except Exception as error:
        print(f"H256-U8-PS2 artifact audit failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
