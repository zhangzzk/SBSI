"""Frozen centre-development diagnostic for the packed integer bridge.

The exact global base is sampled without conditioning on an observation or
its exact candidate stratum.  Each unique sampled atom is evaluated once with
proposal probability one, producing the numerical ``log c`` carried by the
existing tensor likelihood.  Only on the CPU is the exact dyadic base
probability subtracted and candidate-stratum target mass set to zero.  The
original ``(island, round, particle)`` slot multiplicity is then restored for
the packed bridge.

This module is a development gate, not an inference mode.  It is tied to the
already consumed rows 25,000:25,512, the archived centre and exact-64 subset,
and the immutable outcome-free preflight.  Nothing here authorizes a response
stencil, a fresh confirmation window, or a release/default change.
"""

from __future__ import annotations

from concurrent.futures import Executor, ProcessPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, fields
from hashlib import sha256
import json
import multiprocessing
import os
from pathlib import Path
import resource
import tempfile
import time
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch

from .complement_diagnostic import (
    EXACT_ORACLE_ATOM_CHUNK,
    EXACT_ORACLE_OBJECT_CHUNK,
    exact_oracle_object_ids,
)
from .catalogue_sampling import pareto_tail_index
from .integer_mixture_base import IntegerMixtureBase
from .learned_retrieval_diagnostic import (
    LEARNED_RETRIEVAL_CENTER_BASELINE,
    LEARNED_RETRIEVAL_CENTER_MOCK_SHA256,
)
from .packed_integer_bridge_smc import (
    PACKED_BRIDGE_HASH_CONTRACT,
    PACKED_BRIDGE_INPUT_CONTRACT,
    PACKED_BRIDGE_MOVE_STAGES,
    PACKED_BRIDGE_RESAMPLING_CONTRACT,
    PACKED_BRIDGE_RNG_CONTRACT,
    PACKED_BRIDGE_ROUNDS,
    PACKED_BRIDGE_SCORE_CONTRACT,
    PACKED_BRIDGE_STAGES,
    PACKED_BRIDGE_VERSION,
    PackedIntegerBridgeSMCCompactResult,
    PackedIntegerBridgeSMCResult,
    run_packed_integer_bridge_smc,
    run_packed_integer_bridge_smc_compact,
)
from .spatial_neyman_diagnostic import (
    SPATIAL_NEYMAN_CENTER_GATE_THRESHOLDS,
    _finite_logsumexp,
)


PACKED_BRIDGE_CENTER_LABEL = (
    "SBSI/complement-packed-integer-bridge-center-development/v1"
)
PACKED_BRIDGE_PREFLIGHT_VERSION = "packed_integer_bridge_smc_a40_structural_v1"
PACKED_BRIDGE_PREFLIGHT_REPORT_SHA256 = (
    "2f76e35439e58f26deaad80170d021532b99110e99976fa0b8d1716308594d20"
)
PACKED_BRIDGE_PREFLIGHT_RUN_MANIFEST_SHA256 = (
    "da5d5311a88e76729a004574b370c719406a9835335ecc21d30e15b60ef296e9"
)
PACKED_BRIDGE_PREFLIGHT_COMMIT = (
    "6c3c01815b02d91992781c41fb72ec8e170f6306"
)
PACKED_BRIDGE_EXACT_REFERENCE_TABLE_SHA256 = (
    "bdfd22b66255684aa1b04cf20263c02ed0fc72bef8985f79c2b8ed43a3b2ac7f"
)
PACKED_BRIDGE_EXACT_REFERENCE_SUMMARY_SHA256 = (
    "b0895cf1823c0c29a7a82440449f16e90f33cdbc06816b6ea7aa4d79954b4f63"
)
PACKED_BRIDGE_BASE_LAW_SHA256 = (
    "3e105cc64491ce4972431f8fc16aeb780bc51173adf88d3de95157980117e6af"
)
PACKED_BRIDGE_BASE_AUDIT_SHA256 = (
    "e92ec62a1d9693aa08b83c4c08a4d5e4f227931a3db5d4e268a3dfb6f344bc65"
)
PACKED_BRIDGE_BASE_ARRAYS_SHA256 = (
    "e0ad0e86de019d3fe14d99fc3c04fd1e44aaf8dcd771f42ad4d6ff568c091fca"
)
PACKED_BRIDGE_BASE_MANIFEST_SHA256 = (
    "7b8e820f79392fc250bb1a3515fab8beb71db404e705027277bef6fdc0cd49ef"
)
PACKED_BRIDGE_ACTIVE_ATOMS = 12_760_990
PACKED_BRIDGE_ISLANDS = 16
PACKED_BRIDGE_PARTICLES = 256
PACKED_BRIDGE_SLOTS_PER_OBJECT = (
    PACKED_BRIDGE_ISLANDS * PACKED_BRIDGE_ROUNDS * PACKED_BRIDGE_PARTICLES
)
PACKED_BRIDGE_BASE_SEED = 7_008_705
PACKED_BRIDGE_BASE_DOMAIN = 0x425249444745
PACKED_BRIDGE_KERNEL_SEED = 7_008_706
PACKED_BRIDGE_WORKERS = 7
PACKED_BRIDGE_ISLAND_CHUNK = 16
PACKED_BRIDGE_REPLAY_ISLAND_CHUNK = 5
PACKED_BRIDGE_TARGET_OBJECT_CHUNK = 1
PACKED_BRIDGE_Q_DENOMINATOR_POWER = 51

PACKED_BRIDGE_METHOD_TIME_COMPONENTS = (
    "integer_base_load_and_verify",
    "candidate_query",
    "stratum_exact_flow",
    "integer_base_draw_and_audit",
    "unique_target_flow",
    "bridge_worker_wall",
    "reduction_and_hashing",
    "bridge_pool_shutdown",
    "orchestration",
)

PACKED_BRIDGE_CENTER_GATE_THRESHOLDS = deepcopy(
    SPATIAL_NEYMAN_CENTER_GATE_THRESHOLDS
)

PACKED_BRIDGE_PREFLIGHT_IDENTITY = {
    "report_sha256": PACKED_BRIDGE_PREFLIGHT_REPORT_SHA256,
    "preflight_version": PACKED_BRIDGE_PREFLIGHT_VERSION,
    "run_manifest_sha256": PACKED_BRIDGE_PREFLIGHT_RUN_MANIFEST_SHA256,
    "repository_commit": PACKED_BRIDGE_PREFLIGHT_COMMIT,
    "integer_base": {
        "law_sha256": PACKED_BRIDGE_BASE_LAW_SHA256,
        "audit_sha256": PACKED_BRIDGE_BASE_AUDIT_SHA256,
        "arrays_file_sha256": PACKED_BRIDGE_BASE_ARRAYS_SHA256,
        "manifest_file_sha256": PACKED_BRIDGE_BASE_MANIFEST_SHA256,
        "n_atoms": PACKED_BRIDGE_ACTIVE_ATOMS,
        "active_atoms": PACKED_BRIDGE_ACTIVE_ATOMS,
        "raw_prior_terminal_correction": 0,
        "detected_prior_terminal_correction": 0,
    },
}

PACKED_BRIDGE_PROTOCOL = {
    "label": PACKED_BRIDGE_CENTER_LABEL,
    "universe_start": 25_000,
    "universe_stop": 25_512,
    "center": [0.013610377120038044, 0.002740148300964851],
    "initial_strategy": "mean_observed_shape",
    "n_candidates": 16_384,
    "prefilter_candidates": 131_072,
    "proposal_seed": 8_701,
    "candidate_backend": "torch",
    "diagnostic_object_chunk": 32,
    "target_object_chunk": PACKED_BRIDGE_TARGET_OBJECT_CHUNK,
    "flow_atom_chunk": 4_096,
    "active_atoms": PACKED_BRIDGE_ACTIVE_ATOMS,
    "islands": PACKED_BRIDGE_ISLANDS,
    "particles_per_island": PACKED_BRIDGE_PARTICLES,
    "proposal_rounds": PACKED_BRIDGE_ROUNDS,
    "proposal_slots_per_object": PACKED_BRIDGE_SLOTS_PER_OBJECT,
    "base_seed": PACKED_BRIDGE_BASE_SEED,
    "base_domain": PACKED_BRIDGE_BASE_DOMAIN,
    "bridge_seed": PACKED_BRIDGE_KERNEL_SEED,
    "bridge_workers": PACKED_BRIDGE_WORKERS,
    "slurm_cpus_per_task": 8,
    "cuda_device_name": "NVIDIA A40",
    "worker_thread_environment": {
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    },
    "bridge_island_chunk": PACKED_BRIDGE_ISLAND_CHUNK,
    "bridge_replay_island_chunk": PACKED_BRIDGE_REPLAY_ISLAND_CHUNK,
    "base_version": "integer_mixture_base_v1",
    "base_law_sha256": PACKED_BRIDGE_BASE_LAW_SHA256,
    "bridge_version": PACKED_BRIDGE_VERSION,
    "bridge_rng_contract": PACKED_BRIDGE_RNG_CONTRACT,
    "bridge_score_contract": PACKED_BRIDGE_SCORE_CONTRACT,
    "bridge_resampling_contract": PACKED_BRIDGE_RESAMPLING_CONTRACT,
    "bridge_hash_contract": PACKED_BRIDGE_HASH_CONTRACT,
    "bridge_input_contract": PACKED_BRIDGE_INPUT_CONTRACT,
    "preflight": deepcopy(PACKED_BRIDGE_PREFLIGHT_IDENTITY),
    "exact_selection_manifest_sha256": (
        "6abcc361aec9ada6010f2a91a68aa70440904873518182445c2a712ae69bad6a"
    ),
    "exact_selection_count": 64,
    "exact_atom_chunk": EXACT_ORACLE_ATOM_CHUNK,
    "exact_object_chunk": EXACT_ORACLE_OBJECT_CHUNK,
    "exact_reference_table_sha256": PACKED_BRIDGE_EXACT_REFERENCE_TABLE_SHA256,
    "exact_reference_summary_sha256": PACKED_BRIDGE_EXACT_REFERENCE_SUMMARY_SHA256,
    "stratum_flow_evaluations": 8_388_608,
    "proposal_slots_total": 16_777_216,
    "exact_flow_evaluations": 816_703_360,
    "new_exact_flow_evaluations": 0,
    "mock_input_sha256": deepcopy(LEARNED_RETRIEVAL_CENTER_MOCK_SHA256),
    "all_collapsed_policy": "invalid_precision_no_retry_no_drop",
    "duplicate_policy": "unique_logc_once_scatter_original_slot_multiplicity",
    "stratum_hit_policy": "score_all_unique_q0_atoms_then_mask_exact_stratum",
    "exclusion_policy": "global_unconditioned_draw_then_zero_exact_stratum_target",
    "q0_log_policy": "subtract_exact_dyadic_marginal_in_cpu_float64",
    "target_pack_order": "numpy_unique_ascending_then_inverse_scatter_c_order",
    "target_pack_replay_atom_chunk": 2_048,
    "target_pack_replay_max_abs_delta": 1.0e-5,
    "stratum_hit_crosscheck_max_abs_delta": 1.0e-5,
}


def _is_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(label: str, values, *, dtype: str) -> str:
    canonical = np.ascontiguousarray(values, dtype=np.dtype(dtype))
    shape = np.ascontiguousarray(canonical.shape, dtype=np.dtype(">u8"))
    digest = sha256()
    digest.update(b"SBSI/packed-bridge-diagnostic-array/v1\0")
    digest.update(label.encode("utf-8"))
    digest.update(b"\0")
    digest.update(shape.tobytes())
    digest.update(canonical.tobytes())
    return digest.hexdigest()


def _text_sha256(label: str, value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    digest = sha256()
    digest.update(b"SBSI/packed-bridge-diagnostic-text/v1\0")
    digest.update(label.encode("utf-8"))
    digest.update(b"\0")
    digest.update(payload)
    return digest.hexdigest()


def _record_sha256(label: str, values) -> str:
    digest = sha256()
    digest.update(b"SBSI/packed-bridge-diagnostic-record/v1\0")
    digest.update(label.encode("utf-8"))
    digest.update(b"\0")
    for value in values:
        encoded = str(value).encode("ascii")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _gate(observed: Any, operator: str, threshold: Any, passed: bool) -> dict:
    return {
        "observed": observed,
        "operator": operator,
        "threshold": threshold,
        "passed": bool(passed),
    }


def _higher_percentile(values: np.ndarray, percentile: float) -> float:
    return float(np.percentile(values, percentile, method="higher"))


def _nonnegative_scaled_sum(log_values: np.ndarray) -> tuple[float, float]:
    """Return an exact nonnegative mass, including the all-zero case."""

    log_mass = _finite_logsumexp(np.asarray(log_values, dtype=np.float64))
    if np.isneginf(log_mass):
        return 0.0, log_mass
    with np.errstate(over="ignore"):
        mass = float(np.exp(log_mass))
    if not np.isfinite(mass) or mass <= 0.0:
        raise RuntimeError("packed-bridge exact-stratum reduction overflowed")
    return mass, log_mass


def _exact_int64_vector(values) -> np.ndarray | None:
    try:
        if isinstance(values, pd.Series):
            if not pd.api.types.is_integer_dtype(values.dtype):
                return None
            raw = values.to_numpy()
        else:
            raw = np.asarray(values)
        if raw.ndim != 1 or raw.dtype.kind not in "iu":
            return None
        result = raw.astype(np.int64, copy=False)
        if raw.dtype.kind == "u" and np.any(raw > np.iinfo(np.int64).max):
            return None
        return np.ascontiguousarray(result)
    except (TypeError, ValueError, OverflowError):
        return None


def _verify_preflight_payload(
    payload: dict,
    *,
    report_sha256: str,
    expected: dict = PACKED_BRIDGE_PREFLIGHT_IDENTITY,
) -> dict:
    """Validate the frozen structural result without trusting optional fields."""

    if not isinstance(payload, dict) or report_sha256 != expected["report_sha256"]:
        raise ValueError("packed-bridge preflight report identity changed")
    repository = payload.get("repository")
    if (
        payload.get("preflight_version") != expected["preflight_version"]
        or payload.get("run_manifest_sha256") != expected["run_manifest_sha256"]
        or not isinstance(repository, dict)
        or repository.get("commit") != expected["repository_commit"]
        or repository.get("clean") is not True
        or repository.get("detached") is not True
        or payload.get("outcome_free") is not True
        or payload.get("structural_invariant_failures") != 0
        or payload.get("integer_base") != {
            **expected["integer_base"],
            "mode": payload.get("integer_base", {}).get("mode"),
        }
    ):
        raise ValueError("packed-bridge preflight structural identity changed")
    if payload["integer_base"]["mode"] not in {"constructed", "loaded"}:
        raise ValueError("packed-bridge preflight cache mode is invalid")
    counters = payload.get("science_counters")
    if not isinstance(counters, dict) or not counters or any(counters.values()):
        raise ValueError("packed-bridge preflight was not outcome-free")
    workload = payload.get("workload", {})
    expected_workload = {
        "objects": 32,
        "exclusions_per_object": 16_384,
        "islands": PACKED_BRIDGE_ISLANDS,
        "particles_per_island": PACKED_BRIDGE_PARTICLES,
        "proposal_rounds": PACKED_BRIDGE_ROUNDS,
        "proposal_slots_per_object": PACKED_BRIDGE_SLOTS_PER_OBJECT,
        "proposal_slots_total": 1_048_576,
        "base_seed": PACKED_BRIDGE_BASE_SEED,
        "base_domain": PACKED_BRIDGE_BASE_DOMAIN,
        "bridge_seed": PACKED_BRIDGE_KERNEL_SEED,
        "exclusion_seed": 20_260_832,
        "object_id_base": 1 << 40,
        "object_replay_chunk": 7,
        "island_replay_chunk": PACKED_BRIDGE_REPLAY_ISLAND_CHUNK,
    }
    if workload != expected_workload:
        raise ValueError("packed-bridge preflight workload changed")
    constant = payload.get("constant_control", {})
    collapse = payload.get("mixed_collapse_control", {})
    synthetic = payload.get("synthetic_target", {}).get("known_z_score", {})
    bridge = payload.get("bridge_summary", {})
    if not (
        abs(float(constant.get("normalizer", np.nan)) - 3.0) <= 5.0e-12
        and float(constant.get("standard_error", np.nan)) == 0.0
        and float(constant.get("actual_movement_rate", np.nan)) == 1.0
        and int(collapse.get("collapsed_islands", -1)) == 8
        and abs(float(collapse.get("normalizer", np.nan)) - 0.5) <= 5.0e-12
        and abs(
            float(collapse.get("standard_error", np.nan))
            - float(np.sqrt(1.0 / 60.0))
        )
        <= 5.0e-12
        and float(synthetic.get("rms", np.inf))
        <= float(synthetic.get("rms_gate", -np.inf))
        and max(
            abs(float(synthetic.get("minimum", np.inf))),
            abs(float(synthetic.get("maximum", np.inf))),
        )
        <= float(synthetic.get("maximum_absolute_gate", -np.inf))
        and bridge.get("all_collapsed_objects") == 0
        and bridge.get("independent_island_se_valid_objects") == 32
    ):
        raise ValueError("packed-bridge preflight controls did not pass")
    return deepcopy(expected)


def load_verified_packed_bridge_base(
    cache_path: str | Path,
    preflight_report_path: str | Path,
    *,
    raw_prior,
    detected_prior,
) -> tuple[IntegerMixtureBase, dict]:
    """Load the exact base only after report, file, and live-weight checks."""

    report_path = Path(preflight_report_path)
    report_hash = _file_sha256(report_path)
    try:
        payload = json.loads(report_path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("packed-bridge preflight report is invalid") from error
    identity = _verify_preflight_payload(payload, report_sha256=report_hash)
    root = Path(cache_path)
    observed_files = {
        "arrays_file_sha256": _file_sha256(root / "arrays.npz"),
        "manifest_file_sha256": _file_sha256(root / "manifest.json"),
    }
    expected_files = {
        name: identity["integer_base"][name] for name in observed_files
    }
    if observed_files != expected_files:
        raise ValueError("packed-bridge integer-base cache bytes changed")
    base = IntegerMixtureBase.load(root)
    base.verify_weights(raw_prior, detected_prior)
    if (
        base.n_atoms != PACKED_BRIDGE_ACTIVE_ATOMS
        or int(np.count_nonzero(base.active)) != PACKED_BRIDGE_ACTIVE_ATOMS
        or base.law_sha256 != PACKED_BRIDGE_BASE_LAW_SHA256
        or base.audit_sha256 != PACKED_BRIDGE_BASE_AUDIT_SHA256
        or base.raw_prior_terminal_correction != 0
        or base.detected_prior_terminal_correction != 0
    ):
        raise ValueError("packed-bridge integer-base semantic identity changed")
    return base, {
        **identity,
        "report_path": str(report_path.resolve()),
        "cache_path": str(root.resolve()),
    }


@dataclass(frozen=True)
class PackedBridgeExactReference:
    """Pinned exact-64 values from the independently audited H256 artifact."""

    table: pd.DataFrame
    table_sha256: str
    summary_sha256: str
    selection_manifest_sha256: str
    selected_object_ids: np.ndarray
    active_indices_sha256: str
    source_path: str


def load_packed_bridge_exact_reference(
    path: str | Path,
) -> PackedBridgeExactReference:
    """Load exact values only after byte, integrity, and selection checks."""

    root = Path(path)
    table_path = root / "per_object.parquet"
    summary_path = root / "summary.json"
    table_hash = _file_sha256(table_path)
    summary_hash = _file_sha256(summary_path)
    if (
        table_hash != PACKED_BRIDGE_EXACT_REFERENCE_TABLE_SHA256
        or summary_hash != PACKED_BRIDGE_EXACT_REFERENCE_SUMMARY_SHA256
    ):
        raise ValueError("packed-bridge exact-reference bytes changed")
    try:
        frame = pd.read_parquet(table_path)
        summary = json.loads(summary_path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("packed-bridge exact-reference artifact is invalid") from error
    expected_ids = np.arange(25_000, 25_512, dtype=np.int64)
    object_ids = _exact_int64_vector(frame.get("object_index"))
    selected_ids, selection_manifest = exact_oracle_object_ids(25_000, 25_512)
    required = {
        "object_index",
        "stratum_indices_sha256",
        "stratum_log_target_sha256",
        "exact_oracle_selected",
        "mass_stratum_exact_oracle",
        "mass_complement_exact_oracle",
        "mass_total_exact_oracle",
    }
    exact_metadata = summary.get("metadata", {}).get("exact_oracle", {})
    integrity = (
        summary.get("summary", {})
        .get("spatial_h256_u8_ps2", {})
        .get("gates", {})
        .get("integrity", {})
    )
    selected_mask_valid = bool(
        "exact_oracle_selected" in frame
        and pd.api.types.is_bool_dtype(frame["exact_oracle_selected"].dtype)
    )
    observed_selected = (
        object_ids[frame["exact_oracle_selected"].to_numpy(dtype=bool)]
        if object_ids is not None and selected_mask_valid
        else None
    )
    selected_mask = (
        frame["exact_oracle_selected"].to_numpy(dtype=bool)
        if selected_mask_valid
        else np.zeros(len(frame), dtype=bool)
    )
    finite_exact = (
        frame.loc[
            selected_mask,
            [
                "mass_stratum_exact_oracle",
                "mass_complement_exact_oracle",
                "mass_total_exact_oracle",
            ],
        ].to_numpy(dtype=np.float64)
        if required <= set(frame)
        else np.empty((0, 3), dtype=np.float64)
    )
    if not (
        frame.shape[0] == 512
        and required <= set(frame)
        and object_ids is not None
        and np.array_equal(object_ids, expected_ids)
        and selected_mask_valid
        and observed_selected is not None
        and np.array_equal(observed_selected, selected_ids)
        and selection_manifest == PACKED_BRIDGE_PROTOCOL[
            "exact_selection_manifest_sha256"
        ]
        and exact_metadata.get("selection_manifest_sha256") == selection_manifest
        and exact_metadata.get("selection_count") == 64
        and exact_metadata.get("selected_object_ids") == selected_ids.tolist()
        and exact_metadata.get("active_atoms") == PACKED_BRIDGE_ACTIVE_ATOMS
        and exact_metadata.get("flow_evaluations") == 816_703_360
        and _is_sha256(exact_metadata.get("active_indices_sha256"))
        and isinstance(integrity, dict)
        and integrity
        and all(item.get("passed") is True for item in integrity.values())
        and np.isfinite(finite_exact).all()
        and (finite_exact > 0.0).all()
        and np.allclose(
            finite_exact[:, 0] + finite_exact[:, 1],
            finite_exact[:, 2],
            rtol=64.0 * np.finfo(np.float64).eps,
            atol=0.0,
        )
    ):
        raise ValueError("packed-bridge exact-reference integrity changed")
    retained = frame.loc[:, sorted(required)].copy()
    return PackedBridgeExactReference(
        table=retained,
        table_sha256=table_hash,
        summary_sha256=summary_hash,
        selection_manifest_sha256=selection_manifest,
        selected_object_ids=selected_ids,
        active_indices_sha256=str(exact_metadata["active_indices_sha256"]),
        source_path=str(root.resolve()),
    )


@dataclass(frozen=True)
class PackedTargetPlan:
    """One object's unique atom plan before any target value is inspected."""

    slot_atom_ids: np.ndarray
    unique_atom_ids: np.ndarray
    inverse: np.ndarray
    unique_probability_numerator: np.ndarray
    excluded_unique: np.ndarray
    excluded_slots: np.ndarray

    def __post_init__(self):
        slots = np.ascontiguousarray(self.slot_atom_ids, dtype=np.int64).copy()
        unique = np.ascontiguousarray(self.unique_atom_ids, dtype=np.int64).copy()
        inverse = np.ascontiguousarray(self.inverse, dtype=np.int64).copy()
        numerator = np.ascontiguousarray(
            self.unique_probability_numerator, dtype=np.uint64
        ).copy()
        excluded_unique = np.ascontiguousarray(
            self.excluded_unique, dtype=bool
        ).copy()
        excluded_slots = np.ascontiguousarray(
            self.excluded_slots, dtype=bool
        ).copy()
        if (
            slots.shape
            != (PACKED_BRIDGE_ISLANDS, PACKED_BRIDGE_ROUNDS, PACKED_BRIDGE_PARTICLES)
            or unique.ndim != 1
            or not len(unique)
            or not np.array_equal(unique, np.unique(slots))
            or inverse.shape != (slots.size,)
            or (inverse < 0).any()
            or (inverse >= len(unique)).any()
            or not np.array_equal(unique[inverse], slots.reshape(-1))
            or numerator.shape != unique.shape
            or (numerator == 0).any()
            or excluded_unique.shape != unique.shape
            or excluded_slots.shape != slots.shape
            or not np.array_equal(
                excluded_slots.reshape(-1), excluded_unique[inverse]
            )
        ):
            raise ValueError("packed target plan is inconsistent")
        for name, values in (
            ("slot_atom_ids", slots),
            ("unique_atom_ids", unique),
            ("inverse", inverse),
            ("unique_probability_numerator", numerator),
            ("excluded_unique", excluded_unique),
            ("excluded_slots", excluded_slots),
        ):
            values.setflags(write=False)
            object.__setattr__(self, name, values)

    @classmethod
    def build(
        cls,
        slot_atom_ids,
        *,
        probability_numerator: np.ndarray,
        exact_stratum: np.ndarray,
    ) -> "PackedTargetPlan":
        slots = np.asarray(slot_atom_ids)
        if slots.dtype.kind not in "iu" or (slots < 0).any():
            raise ValueError("packed target slot atom IDs must be nonnegative integers")
        slots = np.ascontiguousarray(slots, dtype=np.int64)
        if slots.shape != (
            PACKED_BRIDGE_ISLANDS,
            PACKED_BRIDGE_ROUNDS,
            PACKED_BRIDGE_PARTICLES,
        ):
            raise ValueError("packed target slots have the wrong production shape")
        q = np.asarray(probability_numerator)
        if q.ndim != 1 or q.dtype.kind not in "iu":
            raise ValueError("base probability numerators must be an integer vector")
        unique, inverse = np.unique(slots.reshape(-1), return_inverse=True)
        if unique[-1] >= len(q):
            raise ValueError("packed target atom ID exceeds the base law")
        numerator = np.ascontiguousarray(q[unique], dtype=np.uint64)
        exact = np.asarray(exact_stratum)
        if (
            exact.ndim != 1
            or exact.dtype.kind not in "iu"
            or (exact < 0).any()
            or len(np.unique(exact)) != len(exact)
        ):
            raise ValueError("exact stratum must contain unique nonnegative atom IDs")
        excluded_unique = np.isin(unique, exact, assume_unique=True)
        return cls(
            slot_atom_ids=slots,
            unique_atom_ids=unique,
            inverse=inverse,
            unique_probability_numerator=numerator,
            excluded_unique=excluded_unique,
            excluded_slots=excluded_unique[inverse].reshape(slots.shape),
        )

    def scatter_log_ratio(self, unique_log_c) -> tuple[np.ndarray, dict]:
        """Subtract exact q0 in float64, zero S, and restore every slot."""

        log_c = np.ascontiguousarray(unique_log_c, dtype=np.float64)
        if (
            log_c.shape != self.unique_atom_ids.shape
            or np.isnan(log_c).any()
            or np.isposinf(log_c).any()
        ):
            raise ValueError("unique log target must be finite or -inf and align")
        probability = np.ldexp(
            self.unique_probability_numerator.astype(np.float64),
            -PACKED_BRIDGE_Q_DENOMINATOR_POWER,
        )
        if not np.isfinite(probability).all() or (probability <= 0.0).any():
            raise RuntimeError("exact dyadic base probability is invalid")
        unique_ratio = log_c - np.log(probability)
        unique_ratio[self.excluded_unique] = -np.inf
        ratio = unique_ratio[self.inverse].reshape(self.slot_atom_ids.shape)
        if np.isnan(ratio).any() or np.isposinf(ratio).any():
            raise RuntimeError("scattered packed target contains NaN or +inf")
        accounting = {
            "requested_slots": int(ratio.size),
            "unique_atoms": int(len(self.unique_atom_ids)),
            "unique_evaluations": int(len(self.unique_atom_ids)),
            "complement_unique_atoms": int(
                np.count_nonzero(~self.excluded_unique)
            ),
            "cache_hits": int(ratio.size - len(self.unique_atom_ids)),
            "excluded_unique_atoms": int(np.count_nonzero(self.excluded_unique)),
            "excluded_slots": int(np.count_nonzero(self.excluded_slots)),
            "finite_slots": int(np.count_nonzero(np.isfinite(ratio))),
            "zero_slots": int(np.count_nonzero(np.isneginf(ratio))),
            "unique_atom_ids_sha256": _array_sha256(
                "target_unique_atom_ids", self.unique_atom_ids, dtype=">i8"
            ),
            "unique_probability_numerator_sha256": _array_sha256(
                "target_unique_probability_numerator",
                self.unique_probability_numerator,
                dtype=">u8",
            ),
            "unique_log_c_sha256": _array_sha256(
                "target_unique_log_c", log_c, dtype=">f8"
            ),
            "excluded_unique_sha256": _array_sha256(
                "target_excluded_unique", self.excluded_unique, dtype=">u1"
            ),
            "excluded_slots_sha256": _array_sha256(
                "target_excluded_slots", self.excluded_slots, dtype=">u1"
            ),
        }
        if accounting["finite_slots"] + accounting["zero_slots"] != ratio.size:
            raise RuntimeError("packed target finite/zero accounting failed")
        return np.ascontiguousarray(ratio, dtype=np.float64), accounting


def _run_compact_bridge_worker(
    proposal_atom_ids: np.ndarray,
    target_log_ratio: np.ndarray,
    seed: int,
    object_id: int,
    island_chunk: int,
) -> tuple[PackedIntegerBridgeSMCCompactResult, float, float, int]:
    wall_started = time.perf_counter()
    cpu_started = time.process_time()
    result = run_packed_integer_bridge_smc_compact(
        proposal_atom_ids,
        target_log_ratio,
        seed=seed,
        object_id=object_id,
        island_chunk=island_chunk,
    )
    return (
        result,
        float(time.perf_counter() - wall_started),
        float(time.process_time() - cpu_started),
        int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    )


def run_compact_bridge_batch(
    proposal_atom_ids: np.ndarray,
    target_log_ratio: np.ndarray,
    object_ids: np.ndarray,
    *,
    executor: Executor | None,
    seed: int = PACKED_BRIDGE_KERNEL_SEED,
    island_chunk: int = PACKED_BRIDGE_ISLAND_CHUNK,
) -> tuple[
    list[PackedIntegerBridgeSMCCompactResult],
    list[float],
    list[float],
    list[int],
]:
    """Run an ordered object batch, propagating every worker failure."""

    atoms = np.asarray(proposal_atom_ids)
    ratios = np.asarray(target_log_ratio, dtype=np.float64)
    ids = np.asarray(object_ids)
    expected_tail = (
        PACKED_BRIDGE_ISLANDS,
        PACKED_BRIDGE_ROUNDS,
        PACKED_BRIDGE_PARTICLES,
    )
    if (
        atoms.ndim != 4
        or atoms.shape[1:] != expected_tail
        or ratios.shape != atoms.shape
        or ids.shape != (len(atoms),)
        or ids.dtype.kind not in "iu"
        or len(np.unique(ids)) != len(ids)
    ):
        raise ValueError("compact bridge batch inputs are invalid")
    arguments = [
        (
            np.ascontiguousarray(atoms[row], dtype=np.int64),
            np.ascontiguousarray(ratios[row], dtype=np.float64),
            int(seed),
            int(ids[row]),
            int(island_chunk),
        )
        for row in range(len(atoms))
    ]
    if executor is None:
        returned = [_run_compact_bridge_worker(*item) for item in arguments]
    else:
        futures = [executor.submit(_run_compact_bridge_worker, *item) for item in arguments]
        try:
            returned = [future.result() for future in futures]
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    results = [item[0].validated_copy() for item in returned]
    wall_seconds = [item[1] for item in returned]
    cpu_seconds = [item[2] for item in returned]
    peak_rss_kib = [item[3] for item in returned]
    if [result.object_id for result in results] != [int(value) for value in ids]:
        raise RuntimeError("compact bridge workers changed object order or identity")
    if any(
        not np.isfinite(value) or value < 0.0
        for value in (*wall_seconds, *cpu_seconds)
    ) or any(value <= 0 for value in peak_rss_kib):
        raise RuntimeError("compact bridge worker timing is invalid")
    return results, wall_seconds, cpu_seconds, peak_rss_kib


def _reference_support_sha256(indices: np.ndarray) -> str:
    array = np.sort(np.asarray(indices, dtype=np.int64))[None, :]
    canonical = np.ascontiguousarray(array, dtype=np.dtype(">i8"))
    shape = np.ascontiguousarray(canonical.shape, dtype=np.dtype(">u8"))
    digest = sha256()
    digest.update(b"SBSI/spatial-probability-stratified-array/v1\0")
    digest.update(b"object_conditioned_exclusion_indices\0")
    digest.update(shape.tobytes())
    digest.update(canonical.tobytes())
    return digest.hexdigest()


def _reference_stratum_log_target_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values, dtype=np.dtype(">f8"))
    digest = sha256()
    digest.update(b"SBSI/spatial-h256-u8-ps2-stratum-log-target/v1\0")
    digest.update(int(array.ndim).to_bytes(8, "big"))
    for width in array.shape:
        digest.update(int(width).to_bytes(8, "big"))
    digest.update(array.tobytes())
    return digest.hexdigest()


_PACKED_BRIDGE_MANIFEST_HASH_COLUMNS = (
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

_PACKED_BRIDGE_COMPACT_ARRAY_SPECS = (
    ("stage_log_normalizer", ">f8", PACKED_BRIDGE_STAGES),
    ("stage_effective_sample_size", ">f8", PACKED_BRIDGE_STAGES),
    ("stage_acceptance_rate", ">f8", PACKED_BRIDGE_MOVE_STAGES),
    ("stage_movement_rate", ">f8", PACKED_BRIDGE_MOVE_STAGES),
    ("stage_unique_lineage_count", ">i8", PACKED_BRIDGE_STAGES),
    ("stage_unique_state_count", ">i8", PACKED_BRIDGE_STAGES),
)


def _compact_stage_column(
    metric: str,
    island: int,
    stage: int,
) -> str:
    return f"bridge_island_{island}_stage_{stage + 1}_{metric}"


def _compact_stage_columns(metric: str, stages: int) -> list[str]:
    return [
        _compact_stage_column(metric, island, stage)
        for island in range(PACKED_BRIDGE_ISLANDS)
        for stage in range(stages)
    ]


def packed_bridge_run_manifest_sha256(frame: pd.DataFrame) -> str:
    """Bind every object to its support, exact draw, target, and bridge trace."""

    required = ("object_index", *_PACKED_BRIDGE_MANIFEST_HASH_COLUMNS)
    if any(name not in frame for name in required):
        raise ValueError("packed-bridge run manifest columns are incomplete")
    object_ids = _exact_int64_vector(frame["object_index"])
    if object_ids is None or len(np.unique(object_ids)) != len(object_ids):
        raise ValueError("packed-bridge run manifest object IDs are invalid")
    ordered = np.argsort(object_ids, kind="stable")
    rows = []
    for row in ordered:
        values = [
            _text_sha256("object_id", int(object_ids[row])),
            *(
                str(frame.iloc[row][name])
                for name in _PACKED_BRIDGE_MANIFEST_HASH_COLUMNS
            ),
        ]
        if any(not _is_sha256(value) for value in values[1:]):
            raise ValueError("packed-bridge run manifest contains an invalid hash")
        rows.append(_record_sha256("object", values))
    return _record_sha256("run", rows)


@dataclass
class PackedBridgeDiagnostic:
    """Per-object bridge estimates and immutable centre protocol identity."""

    table: pd.DataFrame
    metadata: dict
    full_trace_replay: PackedIntegerBridgeSMCResult | None = None
    artifact_root: str | None = None

    def summary(self) -> dict:
        return packed_bridge_center_summary(self)


def write_packed_bridge_full_trace_replay(
    diagnostic: PackedBridgeDiagnostic,
    path: str | Path,
) -> dict:
    """Persist and reload the fixed full trace using an object-free NPZ."""

    result = diagnostic.full_trace_replay
    if not isinstance(result, PackedIntegerBridgeSMCResult):
        raise ValueError("packed-bridge diagnostic lacks its full trace replay")
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite packed trace replay: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {field.name: getattr(result, field.name) for field in fields(result)}
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=target.parent,
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            np.savez_compressed(stream, **payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary_path, target)
        temporary_path.unlink()
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    restored = load_packed_bridge_full_trace_replay(target)
    if (
        restored.record_sha256 != result.record_sha256
        or restored.trace_sha256 != result.trace_sha256
        or restored.execution_sha256 != result.execution_sha256
    ):
        target.unlink(missing_ok=True)
        raise RuntimeError("persisted packed trace replay changed identity")
    identity = {
        "relative_path": target.name,
        "sha256": _file_sha256(target),
        "bytes": int(target.stat().st_size),
        "object_id": result.object_id,
        "record_sha256": result.record_sha256,
        "trace_sha256": result.trace_sha256,
        "execution_sha256": result.execution_sha256,
    }
    diagnostic.metadata["packed_integer_bridge_center"][
        "full_trace_replay_artifact"
    ] = deepcopy(identity)
    diagnostic.artifact_root = str(target.parent.resolve())
    return identity


def load_packed_bridge_full_trace_replay(
    path: str | Path,
) -> PackedIntegerBridgeSMCResult:
    """Reload a full replay and re-run the kernel result's complete audit."""

    target = Path(path)
    names = {field.name for field in fields(PackedIntegerBridgeSMCResult)}
    try:
        with np.load(target, allow_pickle=False) as archive:
            if set(archive.files) != names:
                raise ValueError("packed trace replay fields changed")
            payload = {}
            for field in fields(PackedIntegerBridgeSMCResult):
                value = archive[field.name]
                payload[field.name] = value.item() if value.ndim == 0 else value
    except (OSError, ValueError, KeyError) as error:
        raise ValueError("packed trace replay artifact is invalid") from error
    return PackedIntegerBridgeSMCResult(**payload)


def _scaled_iid_weights(log_ratio: np.ndarray) -> np.ndarray:
    ratios = np.asarray(log_ratio, dtype=np.float64)
    if ratios.ndim != 4:
        raise ValueError("packed bridge Pareto input must be a four-dimensional batch")
    flat = ratios.reshape(len(ratios), -1)
    finite = np.isfinite(flat)
    maximum = np.max(np.where(finite, flat, -np.inf), axis=1)
    result = np.zeros(flat.shape, dtype=np.float64)
    live = np.isfinite(maximum)
    if live.any():
        result[live] = np.where(
            finite[live],
            np.exp(flat[live] - maximum[live, None], dtype=np.float64),
            0.0,
        )
    return result


def _draw_row_hashes(draw, row: int) -> dict[str, str]:
    return {
        "base_draw_indices_sha256": _array_sha256(
            "base_draw_indices", draw.indices[row], dtype=">i8"
        ),
        "base_draw_source_sha256": _array_sha256(
            "base_draw_source", draw.source[row], dtype=">u1"
        ),
        "base_draw_source_bits_sha256": _array_sha256(
            "base_draw_source_bits", draw.source_bits[row], dtype=">u1"
        ),
        "base_draw_u48_sha256": _array_sha256(
            "base_draw_u48", draw.u48[row], dtype=">u8"
        ),
        "base_draw_selected_source_ticks_sha256": _array_sha256(
            "base_draw_selected_source_ticks",
            draw.selected_source_ticks[row],
            dtype=">u8",
        ),
        "base_draw_selected_source_cdf_lower_sha256": _array_sha256(
            "base_draw_selected_source_cdf_lower",
            draw.selected_source_cdf_lower[row],
            dtype=">u8",
        ),
        "base_draw_probability_numerator_sha256": _array_sha256(
            "base_draw_probability_numerator",
            draw.probability_numerator[row],
            dtype=">u8",
        ),
    }


def _draw_rows_equal(left, left_row: int, right, right_row: int) -> bool:
    names = (
        "indices",
        "source",
        "source_bits",
        "u48",
        "selected_source_ticks",
        "selected_source_cdf_lower",
        "probability_numerator",
        "probability",
    )
    return all(
        np.array_equal(getattr(left, name)[left_row], getattr(right, name)[right_row])
        for name in names
    )


def diagnose_packed_integer_bridge_complement(
    likelihood,
    mock,
    proposal,
    *,
    integer_base_cache: str | Path,
    preflight_report: str | Path,
    exact_reference: str | Path,
    proposal_initialization_seconds: float,
    center: tuple[float, float] = (
        0.013610377120038044,
        0.002740148300964851,
    ),
    observation_start: int = 0,
    observation_stop: Optional[int] = None,
    object_id_offset: int = 25_000,
    object_chunk: int = 32,
    target_object_chunk: int = PACKED_BRIDGE_TARGET_OBJECT_CHUNK,
    atom_chunk: int = 4_096,
    candidate_backend: str = "torch",
    n_candidates: int = 16_384,
    prefilter_candidates: int = 131_072,
    proposal_seed: int = 8_701,
    base_seed: int = PACKED_BRIDGE_BASE_SEED,
    base_domain: int = PACKED_BRIDGE_BASE_DOMAIN,
    bridge_seed: int = PACKED_BRIDGE_KERNEL_SEED,
    bridge_workers: int = PACKED_BRIDGE_WORKERS,
    bridge_island_chunk: int = PACKED_BRIDGE_ISLAND_CHUNK,
    confirmation_mock_sha256: Optional[dict[str, str]] = None,
) -> PackedBridgeDiagnostic:
    """Run the one frozen packed-bridge centre-development comparison."""

    protocol = PACKED_BRIDGE_PROTOCOL
    if confirmation_mock_sha256 != LEARNED_RETRIEVAL_CENTER_MOCK_SHA256:
        raise ValueError("packed bridge requires the frozen development mock")
    if getattr(likelihood, "selection", None) is not None:
        raise ValueError("packed bridge requires measured selection to be disabled")
    if candidate_backend != "torch":
        raise ValueError("packed bridge requires Torch candidate search")
    if (
        tuple(center) != tuple(protocol["center"])
        or observation_start != 0
        or (observation_stop is not None and observation_stop != 512)
        or object_id_offset != 25_000
        or object_chunk != protocol["diagnostic_object_chunk"]
        or target_object_chunk != PACKED_BRIDGE_TARGET_OBJECT_CHUNK
        or atom_chunk != protocol["flow_atom_chunk"]
        or n_candidates != protocol["n_candidates"]
        or prefilter_candidates != protocol["prefilter_candidates"]
        or proposal_seed != protocol["proposal_seed"]
        or base_seed != PACKED_BRIDGE_BASE_SEED
        or base_domain != PACKED_BRIDGE_BASE_DOMAIN
        or bridge_seed != PACKED_BRIDGE_KERNEL_SEED
        or bridge_workers != PACKED_BRIDGE_WORKERS
        or bridge_island_chunk != PACKED_BRIDGE_ISLAND_CHUNK
    ):
        raise ValueError("packed-bridge runner only accepts its frozen protocol")
    if (
        not np.isfinite(proposal_initialization_seconds)
        or proposal_initialization_seconds < 0.0
    ):
        raise ValueError("packed-bridge proposal setup time is invalid")
    measurements = mock.measurements
    stop = len(measurements) if observation_stop is None else int(observation_stop)
    if len(measurements) != 512 or stop != 512:
        raise ValueError("mock must be the frozen 512-object development window")
    if len(proposal.active_indices) != PACKED_BRIDGE_ACTIVE_ATOMS:
        raise ValueError("packed bridge requires the frozen full active support")
    runtime_device = torch.device(likelihood.flow_model.device)
    runtime_device_name = (
        torch.cuda.get_device_name(runtime_device)
        if runtime_device.type == "cuda"
        else None
    )
    thread_environment = {
        name: os.environ.get(name)
        for name in protocol["worker_thread_environment"]
    }
    if (
        runtime_device.type != "cuda"
        or runtime_device_name != protocol["cuda_device_name"]
        or os.environ.get("SLURM_CPUS_PER_TASK")
        != str(protocol["slurm_cpus_per_task"])
        or thread_environment != protocol["worker_thread_environment"]
    ):
        raise ValueError("packed bridge requires its frozen A40/8-CPU runtime")

    phase_seconds = {name: 0.0 for name in PACKED_BRIDGE_METHOD_TIME_COMPONENTS}
    phase_seconds["exact_reference_load"] = 0.0
    method_started = time.perf_counter()
    method_cpu_started = time.process_time()
    device = runtime_device
    cuda_memory_enabled = device.type == "cuda"
    cuda_start_allocated = 0
    cuda_start_reserved = 0
    if cuda_memory_enabled:
        torch.cuda.synchronize(device)
        cuda_start_allocated = int(torch.cuda.memory_allocated(device))
        cuda_start_reserved = int(torch.cuda.memory_reserved(device))
        torch.cuda.reset_peak_memory_stats(device)
    tick = time.perf_counter()
    base, preflight_identity = load_verified_packed_bridge_base(
        integer_base_cache,
        preflight_report,
        raw_prior=proposal.prior_weights,
        detected_prior=proposal.local_base_weights,
    )
    phase_seconds["integer_base_load_and_verify"] = time.perf_counter() - tick
    probability_numerator = base.probability_numerator

    tick = time.perf_counter()
    reference = load_packed_bridge_exact_reference(exact_reference)
    phase_seconds["exact_reference_load"] = time.perf_counter() - tick
    reference_rows = reference.table.set_index("object_index", drop=False)
    expected_selected_ids, expected_selection_manifest = exact_oracle_object_ids(
        25_000, 25_512
    )
    if (
        expected_selection_manifest != protocol["exact_selection_manifest_sha256"]
        or not np.array_equal(reference.selected_object_ids, expected_selected_ids)
    ):
        raise RuntimeError("packed-bridge exact selection changed")
    selected_id_set = set(map(int, expected_selected_ids))

    rows: list[dict[str, Any]] = []
    execution_calls: list[dict[str, Any]] = []
    total_target_evaluations = 0
    total_target_replay_evaluations = 0
    total_worker_wall_seconds = 0.0
    total_worker_cpu_seconds = 0.0
    worker_peak_rss_kib = 0
    base_replay_exact = False
    bridge_replay_exact = False
    bridge_replay_record_sha256 = None
    bridge_replay_execution_sha256 = None
    full_trace_replay: PackedIntegerBridgeSMCResult | None = None
    target_pack_replay_finite_mask_exact = False
    target_pack_replay_max_abs_delta = float("nan")
    target_pack_replay_rms_delta = float("nan")
    target_pack_replay_record_sha256 = None
    stratum_hit_crosscheck_max_abs_delta = 0.0
    stratum_hit_crosscheck_count = 0

    context = multiprocessing.get_context("spawn")
    executor = ProcessPoolExecutor(
        max_workers=bridge_workers,
        mp_context=context,
    )
    try:
        for batch_start in range(observation_start, stop, object_chunk):
            batch_stop = min(batch_start + object_chunk, stop)
            observed = measurements.iloc[batch_start:batch_stop].reset_index(drop=True)
            absolute_ids = object_id_offset + np.arange(
                batch_start, batch_stop, dtype=np.int64
            )
            observed_targets = likelihood.observed_target_tensor(observed)

            tick = time.perf_counter()
            candidates = proposal.candidates(
                observed,
                n_candidates=n_candidates,
                prefilter_candidates=prefilter_candidates,
                torch_device=likelihood.flow_model.device,
            )
            active = np.asarray(proposal.active_indices, dtype=np.int64)
            active_positions = np.searchsorted(active, candidates.indices)
            active_valid = active_positions < len(active)
            if active_valid.any():
                active_valid[active_valid] &= (
                    active[active_positions[active_valid]]
                    == candidates.indices[active_valid]
                )
            if (
                candidates.indices.shape != (len(observed), n_candidates)
                or not active_valid.all()
                or any(
                    len(np.unique(row)) != n_candidates
                    for row in candidates.indices
                )
            ):
                raise RuntimeError("packed-bridge exact support is not unique and active")
            phase_seconds["candidate_query"] += time.perf_counter() - tick

            tick = time.perf_counter()
            stratum_log_c = (
                likelihood.log_importance_weights_tensor(
                    observed,
                    center[0],
                    center[1],
                    atom_indices=candidates.indices,
                    proposal_probability=np.ones_like(
                        candidates.indices, dtype=np.float64
                    ),
                    observed_targets=observed_targets,
                    object_chunk=len(observed),
                    atom_chunk=atom_chunk,
                )
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )
            if np.isnan(stratum_log_c).any() or np.isposinf(stratum_log_c).any():
                raise RuntimeError("packed-bridge stratum target contains NaN or +inf")
            phase_seconds["stratum_exact_flow"] += time.perf_counter() - tick

            tick = time.perf_counter()
            draw = base.sample(
                absolute_ids.astype(np.uint64),
                PACKED_BRIDGE_SLOTS_PER_OBJECT,
                seed=base_seed,
                domain=base_domain,
            )
            draw.verify(base)
            if batch_start == observation_start:
                replay_draw = base.sample(
                    np.asarray([absolute_ids[0]], dtype=np.uint64),
                    PACKED_BRIDGE_SLOTS_PER_OBJECT,
                    seed=base_seed,
                    domain=base_domain,
                )
                replay_draw.verify(base)
                base_replay_exact = _draw_rows_equal(draw, 0, replay_draw, 0)
                if not base_replay_exact:
                    raise RuntimeError("packed-bridge base object replay failed")
            phase_seconds["integer_base_draw_and_audit"] += time.perf_counter() - tick
            atoms = draw.indices.reshape(
                len(observed),
                PACKED_BRIDGE_ISLANDS,
                PACKED_BRIDGE_ROUNDS,
                PACKED_BRIDGE_PARTICLES,
            )
            ratios = np.empty(atoms.shape, dtype=np.float64)
            plans: list[PackedTargetPlan] = []
            target_accounting: list[dict[str, Any]] = []

            tick = time.perf_counter()
            for local_row in range(len(observed)):
                plan = PackedTargetPlan.build(
                    atoms[local_row],
                    probability_numerator=probability_numerator,
                    exact_stratum=candidates.indices[local_row],
                )
                selected_atoms = plan.unique_atom_ids
                evaluated = (
                    likelihood.log_importance_weights_tensor(
                        observed.iloc[local_row : local_row + 1],
                        center[0],
                        center[1],
                        atom_indices=selected_atoms[None, :],
                        proposal_probability=np.ones(
                            (1, len(selected_atoms)), dtype=np.float64
                        ),
                        observed_targets=observed_targets[
                            local_row : local_row + 1
                        ],
                        object_chunk=target_object_chunk,
                        atom_chunk=atom_chunk,
                    )
                    .detach()
                    .cpu()
                    .numpy()[0]
                    .astype(np.float64)
                )
                if np.isnan(evaluated).any() or np.isposinf(evaluated).any():
                    raise RuntimeError(
                        "packed-bridge unique target contains NaN or +inf"
                    )
                if plan.excluded_unique.any():
                    stratum_order = np.argsort(
                        candidates.indices[local_row], kind="stable"
                    )
                    sorted_stratum = candidates.indices[local_row, stratum_order]
                    hit_ids = plan.unique_atom_ids[plan.excluded_unique]
                    hit_positions = np.searchsorted(sorted_stratum, hit_ids)
                    if (
                        (hit_positions >= len(sorted_stratum)).any()
                        or not np.array_equal(
                            sorted_stratum[hit_positions], hit_ids
                        )
                    ):
                        raise RuntimeError(
                            "packed-bridge stratum-hit lookup is inconsistent"
                        )
                    separate = stratum_log_c[
                        local_row, stratum_order[hit_positions]
                    ]
                    packed = evaluated[plan.excluded_unique]
                    if not np.array_equal(np.isfinite(separate), np.isfinite(packed)):
                        raise RuntimeError(
                            "packed-bridge stratum-hit finite support changed"
                        )
                    finite_hits = np.isfinite(separate)
                    if finite_hits.any():
                        hit_delta = float(
                            np.max(
                                np.abs(
                                    separate[finite_hits] - packed[finite_hits]
                                )
                            )
                        )
                        stratum_hit_crosscheck_max_abs_delta = max(
                            stratum_hit_crosscheck_max_abs_delta,
                            hit_delta,
                        )
                    stratum_hit_crosscheck_count += len(hit_ids)
                    if (
                        stratum_hit_crosscheck_max_abs_delta
                        > protocol["stratum_hit_crosscheck_max_abs_delta"]
                    ):
                        raise RuntimeError(
                            "packed-bridge stratum-hit target crosscheck changed"
                        )
                ratio, accounting = plan.scatter_log_ratio(evaluated)
                ratios[local_row] = ratio
                plans.append(plan)
                target_accounting.append(accounting)
                total_target_evaluations += accounting["unique_evaluations"]
                if batch_start == observation_start and local_row == 0:
                    replayed = (
                        likelihood.log_importance_weights_tensor(
                            observed.iloc[0:1],
                            center[0],
                            center[1],
                            atom_indices=selected_atoms[None, :],
                            proposal_probability=np.ones(
                                (1, len(selected_atoms)), dtype=np.float64
                            ),
                            observed_targets=observed_targets[0:1],
                            object_chunk=1,
                            atom_chunk=protocol["target_pack_replay_atom_chunk"],
                        )
                        .detach()
                        .cpu()
                        .numpy()[0]
                        .astype(np.float64)
                    )
                    total_target_replay_evaluations += len(selected_atoms)
                    baseline_finite = np.isfinite(evaluated)
                    replay_finite = np.isfinite(replayed)
                    target_pack_replay_finite_mask_exact = bool(
                        np.array_equal(baseline_finite, replay_finite)
                    )
                    delta = evaluated[baseline_finite] - replayed[baseline_finite]
                    target_pack_replay_max_abs_delta = (
                        float(np.max(np.abs(delta))) if len(delta) else 0.0
                    )
                    target_pack_replay_rms_delta = (
                        float(np.sqrt(np.mean(np.square(delta))))
                        if len(delta)
                        else 0.0
                    )
                    target_pack_replay_record_sha256 = _record_sha256(
                        "target_pack_replay",
                        (
                            _array_sha256(
                                "atom_ids", selected_atoms, dtype=">i8"
                            ),
                            _array_sha256(
                                "baseline_log_c", evaluated, dtype=">f8"
                            ),
                            _array_sha256(
                                "replayed_log_c", replayed, dtype=">f8"
                            ),
                        ),
                    )
                    if (
                        not target_pack_replay_finite_mask_exact
                        or not np.isfinite(target_pack_replay_max_abs_delta)
                        or target_pack_replay_max_abs_delta
                        > protocol["target_pack_replay_max_abs_delta"]
                    ):
                        raise RuntimeError(
                            "packed-bridge alternate target pack replay changed values"
                        )
            phase_seconds["unique_target_flow"] += time.perf_counter() - tick

            tick = time.perf_counter()
            (
                bridge_results,
                worker_wall_seconds,
                worker_cpu_seconds,
                worker_rss_kib,
            ) = run_compact_bridge_batch(
                atoms,
                ratios,
                absolute_ids,
                executor=executor,
                seed=bridge_seed,
                island_chunk=bridge_island_chunk,
            )
            phase_seconds["bridge_worker_wall"] += time.perf_counter() - tick
            total_worker_wall_seconds += float(np.sum(worker_wall_seconds))
            total_worker_cpu_seconds += float(np.sum(worker_cpu_seconds))
            worker_peak_rss_kib = max(worker_peak_rss_kib, *worker_rss_kib)
            if batch_start == observation_start:
                reverse = np.arange(PACKED_BRIDGE_ISLANDS - 1, -1, -1)
                replay = run_packed_integer_bridge_smc(
                    atoms[0][reverse],
                    ratios[0][reverse],
                    seed=bridge_seed,
                    object_id=int(absolute_ids[0]),
                    island_ids=reverse,
                    island_chunk=PACKED_BRIDGE_REPLAY_ISLAND_CHUNK,
                )
                bridge_replay_exact = bool(
                    replay.record_sha256 == bridge_results[0].record_sha256
                    and replay.trace_sha256 == bridge_results[0].trace_sha256
                    and replay.execution_sha256
                    != bridge_results[0].execution_sha256
                )
                bridge_replay_record_sha256 = replay.record_sha256
                bridge_replay_execution_sha256 = replay.execution_sha256
                full_trace_replay = replay
                if not bridge_replay_exact:
                    raise RuntimeError("packed-bridge island replay failed")

            tick = time.perf_counter()
            iid_weights = _scaled_iid_weights(ratios)
            pareto = pareto_tail_index(iid_weights)
            support_hashes = [
                _array_sha256(
                    "stratum_indices", candidates.indices[row], dtype=">i8"
                )
                for row in range(len(observed))
            ]
            stratum_hashes = [
                _array_sha256(
                    "stratum_log_target", stratum_log_c[row], dtype=">f8"
                )
                for row in range(len(observed))
            ]
            call_rows = []
            for local_row, object_id in enumerate(absolute_ids):
                result = bridge_results[local_row]
                plan = plans[local_row]
                accounting = target_accounting[local_row]
                if (
                    result.proposal_slot_count != PACKED_BRIDGE_SLOTS_PER_OBJECT
                    or result.target_unique_atom_count != accounting["unique_atoms"]
                    or result.target_cache_hit_count != accounting["cache_hits"]
                    or result.target_finite_slot_count != accounting["finite_slots"]
                    or result.target_zero_slot_count != accounting["zero_slots"]
                ):
                    raise RuntimeError("packed-bridge target/worker accounting differs")
                mass_stratum, log_mass_stratum = _nonnegative_scaled_sum(
                    stratum_log_c[local_row]
                )
                estimated_log_total = float(
                    np.logaddexp(log_mass_stratum, result.log_normalizer)
                )
                collapsed_islands = int(np.count_nonzero(result.collapsed_stage))
                live_islands = PACKED_BRIDGE_ISLANDS - collapsed_islands
                precision_valid = bool(
                    live_islands > 0
                    and np.isfinite(result.standard_error)
                    and result.standard_error > 0.0
                    and np.isfinite(result.log_variance)
                )
                draw_hashes = _draw_row_hashes(draw, local_row)
                record = {
                    "object_index": int(object_id),
                    "stratum_count": n_candidates,
                    "stratum_unique_count": int(
                        len(np.unique(candidates.indices[local_row]))
                    ),
                    "stratum_active_count": n_candidates,
                    "stratum_indices_sha256": support_hashes[local_row],
                    "stratum_log_target_sha256": stratum_hashes[local_row],
                    **draw_hashes,
                    **accounting,
                    "base_draw_source_detected": int(
                        np.count_nonzero(draw.source[local_row] == 0)
                    ),
                    "base_draw_source_prior": int(
                        np.count_nonzero(draw.source[local_row] == 1)
                    ),
                    "base_draw_record_sha256": draw.record_sha256,
                    "bridge_proposal_atom_ids_sha256": (
                        result.proposal_atom_ids_sha256
                    ),
                    "bridge_target_log_ratio_sha256": (
                        result.target_log_ratio_sha256
                    ),
                    "bridge_trace_sha256": result.trace_sha256,
                    "bridge_record_sha256": result.record_sha256,
                    "bridge_execution_sha256": result.execution_sha256,
                    "bridge_island_log_normalizer_sha256": _array_sha256(
                        "bridge_island_log_normalizer",
                        result.island_log_normalizer,
                        dtype=">f8",
                    ),
                    "bridge_collapsed_stage_sha256": _array_sha256(
                        "bridge_collapsed_stage",
                        result.collapsed_stage,
                        dtype=">i8",
                    ),
                    "bridge_worker_wall_seconds": float(
                        worker_wall_seconds[local_row]
                    ),
                    "bridge_worker_cpu_seconds": float(
                        worker_cpu_seconds[local_row]
                    ),
                    "bridge_worker_peak_rss_kib": int(worker_rss_kib[local_row]),
                    "bridge_live_islands": live_islands,
                    "bridge_collapsed_islands": collapsed_islands,
                    "bridge_all_collapsed": bool(live_islands == 0),
                    "bridge_independent_island_se_valid": precision_valid,
                    "log_mass_stratum": log_mass_stratum,
                    "mass_stratum": mass_stratum,
                    "log_mass_complement_packed_bridge": result.log_normalizer,
                    "mass_complement_packed_bridge": result.normalizer,
                    "mass_complement_packed_bridge_standard_error": (
                        result.standard_error
                    ),
                    "log_variance_complement_packed_bridge": result.log_variance,
                    "log_mass_total_packed_bridge": estimated_log_total,
                    "pareto_k_packed_bridge_base_iid": float(pareto[local_row]),
                    "maximum_scaled_packed_bridge_base_iid_weight": float(
                        iid_weights[local_row].max()
                    ),
                    "exact_oracle_selected": int(object_id) in selected_id_set,
                }
                for island in range(PACKED_BRIDGE_ISLANDS):
                    record[f"bridge_island_{island}_log_normalizer"] = float(
                        result.island_log_normalizer[island]
                    )
                    record[f"bridge_island_{island}_collapsed_stage"] = int(
                        result.collapsed_stage[island]
                    )
                for metric, dtype, stages in _PACKED_BRIDGE_COMPACT_ARRAY_SPECS:
                    metric_values = getattr(result, metric)
                    record[f"bridge_{metric}_sha256"] = _array_sha256(
                        f"bridge_{metric}", metric_values, dtype=dtype
                    )
                    for island in range(PACKED_BRIDGE_ISLANDS):
                        for stage in range(stages):
                            value = metric_values[island, stage]
                            record[
                                _compact_stage_column(metric, island, stage)
                            ] = (
                                int(value)
                                if np.dtype(dtype).kind in "iu"
                                else float(value)
                            )
                for stage in range(PACKED_BRIDGE_STAGES):
                    record[f"bridge_stage_{stage + 1}_mean_ess"] = float(
                        result.stage_effective_sample_size[:, stage].mean()
                    )
                    record[f"bridge_stage_{stage + 1}_minimum_lineages"] = int(
                        result.stage_unique_lineage_count[:, stage].min()
                    )
                    record[f"bridge_stage_{stage + 1}_minimum_unique_states"] = int(
                        result.stage_unique_state_count[:, stage].min()
                    )
                for stage in range(PACKED_BRIDGE_MOVE_STAGES):
                    record[f"bridge_stage_{stage + 1}_mean_acceptance"] = float(
                        result.stage_acceptance_rate[:, stage].mean()
                    )
                    record[f"bridge_stage_{stage + 1}_mean_movement"] = float(
                        result.stage_movement_rate[:, stage].mean()
                    )
                exact = reference_rows.loc[int(object_id)]
                reference_support_hash = _reference_support_sha256(
                    candidates.indices[local_row]
                )
                reference_target_hash = _reference_stratum_log_target_sha256(
                    stratum_log_c[local_row]
                )
                if (
                    reference_support_hash != exact["stratum_indices_sha256"]
                    or reference_target_hash != exact["stratum_log_target_sha256"]
                ):
                    raise RuntimeError(
                        "packed bridge does not match exact-reference stratum"
                    )
                record.update(
                    {
                        "exact_reference_stratum_indices_sha256": (
                            reference_support_hash
                        ),
                        "exact_reference_stratum_log_target_sha256": (
                            reference_target_hash
                        ),
                        "exact_reference_stratum_identity_matched": True,
                    }
                )
                if int(object_id) in selected_id_set:
                    exact_stratum = float(exact["mass_stratum_exact_oracle"])
                    exact_complement = float(exact["mass_complement_exact_oracle"])
                    exact_total = float(exact["mass_total_exact_oracle"])
                    difference = result.normalizer - exact_complement
                    record.update(
                        {
                            "mass_stratum_exact_oracle": exact_stratum,
                            "mass_stratum_exact_oracle_relative_error": (
                                abs(mass_stratum - exact_stratum)
                                / abs(exact_stratum)
                            ),
                            "mass_complement_exact_oracle": exact_complement,
                            "mass_total_exact_oracle": exact_total,
                            "mass_complement_packed_bridge_minus_exact_z": (
                                difference / result.standard_error
                                if precision_valid
                                else float("nan")
                            ),
                            "delta_log_total_evidence_packed_bridge_minus_exact": (
                                estimated_log_total - float(np.log(exact_total))
                            ),
                        }
                    )
                    record["exact_reference_mass_sha256"] = _record_sha256(
                        "exact_reference_mass",
                        (
                            _text_sha256("selected", True),
                            _text_sha256("stratum", exact_stratum),
                            _text_sha256("complement", exact_complement),
                            _text_sha256("total", exact_total),
                        ),
                    )
                else:
                    record["exact_reference_mass_sha256"] = _record_sha256(
                        "exact_reference_mass",
                        (_text_sha256("selected", False),),
                    )
                rows.append(record)
                call_rows.append(result.record_sha256)
            execution_calls.append(
                {
                    "object_start": int(absolute_ids[0]),
                    "object_stop": int(absolute_ids[-1]) + 1,
                    "objects": int(len(absolute_ids)),
                    "base_draw_record_sha256": draw.record_sha256,
                    "base_draw_indices_sha256": draw.indices_sha256,
                    "base_draw_probability_sha256": draw.probability_sha256,
                    "bridge_record_sha256": call_rows,
                }
            )
            phase_seconds["reduction_and_hashing"] += time.perf_counter() - tick
    finally:
        tick = time.perf_counter()
        executor.shutdown(wait=True, cancel_futures=True)
        phase_seconds["bridge_pool_shutdown"] += time.perf_counter() - tick

    tick = time.perf_counter()
    frame = pd.DataFrame(rows).sort_values("object_index").reset_index(drop=True)
    run_manifest = packed_bridge_run_manifest_sha256(frame)
    phase_seconds["reduction_and_hashing"] += time.perf_counter() - tick
    elapsed_before_return = time.perf_counter() - method_started
    accounted = sum(
        phase_seconds[name]
        for name in PACKED_BRIDGE_METHOD_TIME_COMPONENTS
        if name != "orchestration"
    )
    unaccounted = elapsed_before_return - phase_seconds["exact_reference_load"] - accounted
    if unaccounted < -max(1.0e-6, 1.0e-9 * elapsed_before_return):
        raise RuntimeError("packed-bridge phase clocks overlap")
    phase_seconds["orchestration"] = max(0.0, float(unaccounted))
    method_seconds = float(
        proposal_initialization_seconds
        + sum(phase_seconds[name] for name in PACKED_BRIDGE_METHOD_TIME_COMPONENTS)
    )
    method_wall_seconds = float(
        proposal_initialization_seconds
        + elapsed_before_return
        - phase_seconds["exact_reference_load"]
    )
    if abs(method_seconds - method_wall_seconds) > max(
        1.0e-6, 1.0e-9 * method_wall_seconds
    ):
        raise RuntimeError("packed-bridge mutually exclusive phase clock does not close")
    if cuda_memory_enabled:
        torch.cuda.synchronize(device)
        cuda_peak_allocated = int(torch.cuda.max_memory_allocated(device))
        cuda_peak_reserved = int(torch.cuda.max_memory_reserved(device))
    else:
        cuda_peak_allocated = 0
        cuda_peak_reserved = 0
    parent_peak_rss_kib = int(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    )
    bridge_metadata = {
        "label": PACKED_BRIDGE_CENTER_LABEL,
        "initial_strategy": protocol["initial_strategy"],
        "proposal_initialization_seconds": float(proposal_initialization_seconds),
        "integer_base": preflight_identity,
        "exact_reference": {
            "source_path": reference.source_path,
            "table_sha256": reference.table_sha256,
            "summary_sha256": reference.summary_sha256,
            "selection_manifest_sha256": reference.selection_manifest_sha256,
            "selected_object_ids": reference.selected_object_ids.tolist(),
            "active_indices_sha256": reference.active_indices_sha256,
            "source_exact_flow_evaluations": 816_703_360,
            "new_exact_flow_evaluations": 0,
        },
        "base_seed": base_seed,
        "base_domain": base_domain,
        "bridge_seed": bridge_seed,
        "bridge_workers": bridge_workers,
        "bridge_process_start_method": "spawn",
        "slurm_cpus_per_task": int(os.environ["SLURM_CPUS_PER_TASK"]),
        "worker_thread_environment": thread_environment,
        "bridge_island_chunk": bridge_island_chunk,
        "bridge_replay_island_chunk": PACKED_BRIDGE_REPLAY_ISLAND_CHUNK,
        "target_object_chunk": target_object_chunk,
        "target_pack_order": protocol["target_pack_order"],
        "target_pack_replay_atom_chunk": protocol[
            "target_pack_replay_atom_chunk"
        ],
        "target_pack_replay_finite_mask_exact": (
            target_pack_replay_finite_mask_exact
        ),
        "target_pack_replay_max_abs_delta": target_pack_replay_max_abs_delta,
        "target_pack_replay_rms_delta": target_pack_replay_rms_delta,
        "target_pack_replay_record_sha256": target_pack_replay_record_sha256,
        "stratum_hit_crosscheck_count": stratum_hit_crosscheck_count,
        "stratum_hit_crosscheck_max_abs_delta": (
            stratum_hit_crosscheck_max_abs_delta
        ),
        "base_replay_exact": base_replay_exact,
        "bridge_replay_exact": bridge_replay_exact,
        "bridge_replay_record_sha256": bridge_replay_record_sha256,
        "bridge_replay_execution_sha256": bridge_replay_execution_sha256,
        "full_trace_replay": {
            "object_id": (
                full_trace_replay.object_id
                if full_trace_replay is not None
                else None
            ),
            "record_sha256": (
                full_trace_replay.record_sha256
                if full_trace_replay is not None
                else None
            ),
            "trace_sha256": (
                full_trace_replay.trace_sha256
                if full_trace_replay is not None
                else None
            ),
            "execution_sha256": (
                full_trace_replay.execution_sha256
                if full_trace_replay is not None
                else None
            ),
            "island_chunk": (
                full_trace_replay.island_chunk
                if full_trace_replay is not None
                else None
            ),
        },
        "bridge_worker_wall_seconds_sum": total_worker_wall_seconds,
        "bridge_worker_cpu_seconds": total_worker_cpu_seconds,
        "bridge_worker_peak_rss_kib": worker_peak_rss_kib,
        "parent_peak_rss_kib": parent_peak_rss_kib,
        "cuda_memory": {
            "enabled": cuda_memory_enabled,
            "device": str(device),
            "device_name": runtime_device_name,
            "start_allocated_bytes": cuda_start_allocated,
            "start_reserved_bytes": cuda_start_reserved,
            "peak_allocated_bytes": cuda_peak_allocated,
            "peak_reserved_bytes": cuda_peak_reserved,
            "peak_increment_allocated_bytes": max(
                0, cuda_peak_allocated - cuda_start_allocated
            ),
            "peak_increment_reserved_bytes": max(
                0, cuda_peak_reserved - cuda_start_reserved
            ),
        },
        "stratum_flow_evaluations": int(len(frame) * n_candidates),
        "target_unique_flow_evaluations": int(total_target_evaluations),
        "target_pack_replay_flow_evaluations": int(
            total_target_replay_evaluations
        ),
        "proposal_slots_total": int(len(frame) * PACKED_BRIDGE_SLOTS_PER_OBJECT),
        "run_manifest_sha256": run_manifest,
        "execution_call_provenance": execution_calls,
        "mock_input_sha256": dict(confirmation_mock_sha256),
        "protocol": deepcopy(protocol),
    }
    metadata = {
        "center": [float(center[0]), float(center[1])],
        "n_candidates": n_candidates,
        "prefilter_candidates": prefilter_candidates,
        "candidate_backend": candidate_backend,
        "object_chunk": object_chunk,
        "target_object_chunk": target_object_chunk,
        "flow_atom_chunk": atom_chunk,
        "observation_start": object_id_offset + observation_start,
        "observation_stop": object_id_offset + stop,
        "local_observation_start": observation_start,
        "local_observation_stop": stop,
        "object_id_offset": object_id_offset,
        "flow_evaluations": int(
            len(frame) * n_candidates
            + total_target_evaluations
            + total_target_replay_evaluations
        ),
        "parent_process_cpu_seconds": float(
            time.process_time() - method_cpu_started
        ),
        "method_seconds": method_seconds,
        "method_wall_seconds": method_wall_seconds,
        "elapsed_seconds": float(time.perf_counter() - method_started),
        "phase_seconds": {name: float(value) for name, value in phase_seconds.items()},
        "packed_integer_bridge_center": bridge_metadata,
    }
    return PackedBridgeDiagnostic(frame, metadata, full_trace_replay)


def _summarize_island_rows(
    logs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(logs, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != PACKED_BRIDGE_ISLANDS:
        raise ValueError("packed-bridge island log matrix has the wrong shape")
    normalizer = np.zeros(len(values), dtype=np.float64)
    log_normalizer = np.full(len(values), -np.inf, dtype=np.float64)
    variance = np.zeros(len(values), dtype=np.float64)
    log_variance = np.full(len(values), -np.inf, dtype=np.float64)
    standard_error = np.zeros(len(values), dtype=np.float64)
    for row in range(len(values)):
        finite = np.isfinite(values[row])
        if not finite.any():
            continue
        shift = float(np.max(values[row, finite]))
        scaled = np.zeros(PACKED_BRIDGE_ISLANDS, dtype=np.float64)
        scaled[finite] = np.exp(values[row, finite] - shift, dtype=np.float64)
        mean = float(scaled.mean(dtype=np.float64))
        log_normalizer[row] = shift + float(np.log(mean))
        centered_sum = float(np.square(scaled - mean).sum(dtype=np.float64))
        if centered_sum > 0.0:
            log_variance[row] = (
                2.0 * shift
                + float(np.log(centered_sum))
                - float(
                    np.log(PACKED_BRIDGE_ISLANDS * (PACKED_BRIDGE_ISLANDS - 1))
                )
            )
        with np.errstate(over="ignore"):
            normalizer[row] = float(np.exp(log_normalizer[row]))
            variance[row] = float(np.exp(log_variance[row]))
            standard_error[row] = float(np.exp(0.5 * log_variance[row]))
    return normalizer, log_normalizer, variance, log_variance, standard_error


def packed_bridge_center_summary(diagnostic: PackedBridgeDiagnostic) -> dict:
    """Evaluate frozen integrity, base-tail, exact, and efficiency gates."""

    frame = diagnostic.table
    metadata = diagnostic.metadata
    bridge = metadata.get("packed_integer_bridge_center")
    if not isinstance(bridge, dict):
        raise ValueError("diagnostic lacks packed-bridge centre metadata")
    expected_protocol = deepcopy(PACKED_BRIDGE_PROTOCOL)
    observed_protocol = bridge.get("protocol")
    protocol_matched = bool(observed_protocol == expected_protocol)
    thresholds = PACKED_BRIDGE_CENTER_GATE_THRESHOLDS
    expected_objects = int(thresholds["objects"])

    def values(name: str) -> np.ndarray:
        if name not in frame:
            return np.full(len(frame), np.nan, dtype=np.float64)
        return frame[name].to_numpy(dtype=np.float64)

    def boolean_all(name: str, expected: bool = True) -> bool:
        return bool(
            name in frame
            and pd.api.types.is_bool_dtype(frame[name].dtype)
            and np.all(frame[name].to_numpy(dtype=bool) == expected)
        )

    object_ids = _exact_int64_vector(frame.get("object_index"))
    expected_object_ids = np.arange(25_000, 25_512, dtype=np.int64)
    ids_exact = bool(
        object_ids is not None and np.array_equal(object_ids, expected_object_ids)
    )
    expected_selected_ids, expected_selection_manifest = exact_oracle_object_ids(
        25_000, 25_512
    )
    exact_mask_valid = bool(
        "exact_oracle_selected" in frame
        and pd.api.types.is_bool_dtype(frame["exact_oracle_selected"].dtype)
    )
    selected = (
        frame["exact_oracle_selected"].to_numpy(dtype=bool)
        if exact_mask_valid
        else np.zeros(len(frame), dtype=bool)
    )
    exact_reference = bridge.get("exact_reference", {})
    archived_reference = None
    if isinstance(exact_reference, dict) and isinstance(
        exact_reference.get("source_path"), str
    ):
        try:
            archived_reference = load_packed_bridge_exact_reference(
                exact_reference["source_path"]
            )
        except (OSError, ValueError):
            archived_reference = None
    archived_table = (
        archived_reference.table.reset_index(drop=True)
        if archived_reference is not None
        else pd.DataFrame()
    )
    archived_support_matches = bool(
        len(archived_table) == expected_objects
        and all(
            observed_name in frame
            and source_name in archived_table
            and frame[observed_name].astype(str).tolist()
            == archived_table[source_name].astype(str).tolist()
            for observed_name, source_name in (
                (
                    "exact_reference_stratum_indices_sha256",
                    "stratum_indices_sha256",
                ),
                (
                    "exact_reference_stratum_log_target_sha256",
                    "stratum_log_target_sha256",
                ),
            )
        )
    )
    exact_selection_ok = bool(
        ids_exact
        and np.array_equal(object_ids[selected], expected_selected_ids)
        and exact_reference.get("selected_object_ids")
        == expected_selected_ids.tolist()
        and exact_reference.get("selection_manifest_sha256")
        == expected_selection_manifest
        and exact_reference.get("table_sha256")
        == PACKED_BRIDGE_EXACT_REFERENCE_TABLE_SHA256
        and exact_reference.get("summary_sha256")
        == PACKED_BRIDGE_EXACT_REFERENCE_SUMMARY_SHA256
        and exact_reference.get("new_exact_flow_evaluations") == 0
        and exact_reference.get("source_exact_flow_evaluations") == 816_703_360
        and _is_sha256(exact_reference.get("active_indices_sha256"))
        and archived_support_matches
        and int(selected.sum()) == 64
        and "exact_reference_stratum_identity_matched" in frame
        and boolean_all("exact_reference_stratum_identity_matched")
        and all(
            name in frame
            and all(_is_sha256(value) for value in frame[name])
            for name in (
                "exact_reference_stratum_indices_sha256",
                "exact_reference_stratum_log_target_sha256",
            )
        )
    )

    integer_names = (
        "stratum_count",
        "stratum_unique_count",
        "stratum_active_count",
        "requested_slots",
        "unique_atoms",
        "unique_evaluations",
        "complement_unique_atoms",
        "cache_hits",
        "excluded_unique_atoms",
        "excluded_slots",
        "finite_slots",
        "zero_slots",
        "base_draw_source_detected",
        "base_draw_source_prior",
        "bridge_live_islands",
        "bridge_collapsed_islands",
    )
    counts = {name: _exact_int64_vector(frame.get(name)) for name in integer_names}
    integer_counts = all(item is not None for item in counts.values())
    support_ok = bool(
        integer_counts
        and np.all(counts["stratum_count"] == 16_384)
        and np.all(counts["stratum_unique_count"] == 16_384)
        and np.all(counts["stratum_active_count"] == 16_384)
    )
    draw_and_pack_ok = bool(
        integer_counts
        and all(np.all(item >= 0) for item in counts.values())
        and np.all(counts["requested_slots"] == PACKED_BRIDGE_SLOTS_PER_OBJECT)
        and np.all(
            counts["base_draw_source_detected"]
            + counts["base_draw_source_prior"]
            == PACKED_BRIDGE_SLOTS_PER_OBJECT
        )
        and np.all(counts["unique_atoms"] + counts["cache_hits"] == counts["requested_slots"])
        and np.all(counts["unique_evaluations"] == counts["unique_atoms"])
        and np.all(
            counts["complement_unique_atoms"] + counts["excluded_unique_atoms"]
            == counts["unique_atoms"]
        )
        and np.all(counts["finite_slots"] + counts["zero_slots"] == counts["requested_slots"])
        and np.all(counts["excluded_unique_atoms"] <= 16_384)
        and np.all(counts["excluded_slots"] <= counts["zero_slots"])
        and np.all(counts["bridge_live_islands"] + counts["bridge_collapsed_islands"] == 16)
    )
    preflight_ok = bool(
        bridge.get("integer_base", {}).get("report_sha256")
        == PACKED_BRIDGE_PREFLIGHT_REPORT_SHA256
        and bridge.get("integer_base", {}).get("run_manifest_sha256")
        == PACKED_BRIDGE_PREFLIGHT_RUN_MANIFEST_SHA256
        and bridge.get("integer_base", {}).get("integer_base")
        == PACKED_BRIDGE_PREFLIGHT_IDENTITY["integer_base"]
    )
    replay_ok = bool(
        integer_counts
        and bridge.get("base_replay_exact") is True
        and bridge.get("bridge_replay_exact") is True
        and _is_sha256(bridge.get("bridge_replay_record_sha256"))
        and _is_sha256(bridge.get("bridge_replay_execution_sha256"))
        and bridge.get("bridge_process_start_method") == "spawn"
        and bridge.get("bridge_workers") == PACKED_BRIDGE_WORKERS
        and bridge.get("bridge_island_chunk") == PACKED_BRIDGE_ISLAND_CHUNK
        and bridge.get("bridge_replay_island_chunk")
        == PACKED_BRIDGE_REPLAY_ISLAND_CHUNK
        and bridge.get("target_pack_order")
        == PACKED_BRIDGE_PROTOCOL["target_pack_order"]
        and bridge.get("target_pack_replay_atom_chunk")
        == PACKED_BRIDGE_PROTOCOL["target_pack_replay_atom_chunk"]
        and bridge.get("target_pack_replay_finite_mask_exact") is True
        and np.isfinite(
            float(bridge.get("target_pack_replay_max_abs_delta", np.nan))
        )
        and float(bridge.get("target_pack_replay_max_abs_delta"))
        <= PACKED_BRIDGE_PROTOCOL["target_pack_replay_max_abs_delta"]
        and np.isfinite(float(bridge.get("target_pack_replay_rms_delta", np.nan)))
        and float(bridge.get("target_pack_replay_rms_delta")) >= 0.0
        and _is_sha256(bridge.get("target_pack_replay_record_sha256"))
        and bridge.get("stratum_hit_crosscheck_count")
        == int(counts["excluded_unique_atoms"].sum())
        and np.isfinite(
            float(bridge.get("stratum_hit_crosscheck_max_abs_delta", np.nan))
        )
        and float(bridge.get("stratum_hit_crosscheck_max_abs_delta"))
        <= PACKED_BRIDGE_PROTOCOL["stratum_hit_crosscheck_max_abs_delta"]
    )
    replay_result = diagnostic.full_trace_replay
    replay_metadata = bridge.get("full_trace_replay", {})
    replay_artifact = bridge.get("full_trace_replay_artifact", {})
    restored_replay = None
    if (
        isinstance(replay_artifact, dict)
        and isinstance(replay_artifact.get("relative_path"), str)
        and Path(replay_artifact["relative_path"]).name
        == replay_artifact["relative_path"]
        and isinstance(diagnostic.artifact_root, str)
        and _is_sha256(replay_artifact.get("sha256"))
    ):
        try:
            replay_path = (
                Path(diagnostic.artifact_root)
                / replay_artifact["relative_path"]
            )
            if _file_sha256(replay_path) == replay_artifact["sha256"]:
                restored_replay = load_packed_bridge_full_trace_replay(replay_path)
        except (OSError, ValueError):
            restored_replay = None
    authoritative_replay = (
        replay_result
        if isinstance(replay_result, PackedIntegerBridgeSMCResult)
        else restored_replay
    )
    in_memory_replay_matches = bool(
        replay_result is None
        or (
            isinstance(replay_result, PackedIntegerBridgeSMCResult)
            and restored_replay is not None
            and replay_result.record_sha256 == restored_replay.record_sha256
            and replay_result.trace_sha256 == restored_replay.trace_sha256
            and replay_result.execution_sha256 == restored_replay.execution_sha256
        )
    )
    full_trace_replay_ok = bool(
        isinstance(authoritative_replay, PackedIntegerBridgeSMCResult)
        and isinstance(replay_metadata, dict)
        and isinstance(replay_artifact, dict)
        and restored_replay is not None
        and in_memory_replay_matches
        and len(frame) == expected_objects
        and authoritative_replay.object_id == 25_000
        and authoritative_replay.island_chunk == PACKED_BRIDGE_REPLAY_ISLAND_CHUNK
        and authoritative_replay.record_sha256
        == frame.iloc[0]["bridge_record_sha256"]
        and authoritative_replay.trace_sha256
        == frame.iloc[0]["bridge_trace_sha256"]
        and authoritative_replay.execution_sha256
        != frame.iloc[0]["bridge_execution_sha256"]
        and authoritative_replay.proposal_atom_ids_sha256
        == frame.iloc[0]["bridge_proposal_atom_ids_sha256"]
        and authoritative_replay.target_log_ratio_sha256
        == frame.iloc[0]["bridge_target_log_ratio_sha256"]
        and replay_metadata.get("object_id") == authoritative_replay.object_id
        and replay_metadata.get("record_sha256")
        == authoritative_replay.record_sha256
        and replay_metadata.get("trace_sha256")
        == authoritative_replay.trace_sha256
        and replay_metadata.get("execution_sha256")
        == authoritative_replay.execution_sha256
        and replay_metadata.get("island_chunk") == authoritative_replay.island_chunk
        and replay_artifact.get("object_id") == authoritative_replay.object_id
        and replay_artifact.get("record_sha256")
        == authoritative_replay.record_sha256
        and replay_artifact.get("trace_sha256")
        == authoritative_replay.trace_sha256
        and replay_artifact.get("execution_sha256")
        == authoritative_replay.execution_sha256
        and isinstance(replay_artifact.get("bytes"), int)
        and replay_artifact["bytes"] > 0
        and restored_replay.record_sha256 == authoritative_replay.record_sha256
        and restored_replay.trace_sha256 == authoritative_replay.trace_sha256
        and restored_replay.execution_sha256
        == authoritative_replay.execution_sha256
    )
    hash_columns_ok = bool(
        len(frame) == expected_objects
        and all(name in frame for name in _PACKED_BRIDGE_MANIFEST_HASH_COLUMNS)
        and all(
            _is_sha256(value)
            for name in _PACKED_BRIDGE_MANIFEST_HASH_COLUMNS
            for value in frame[name]
        )
    )
    try:
        recomputed_manifest = packed_bridge_run_manifest_sha256(frame)
    except (KeyError, TypeError, ValueError):
        recomputed_manifest = None
    manifest_ok = bool(
        _is_sha256(bridge.get("run_manifest_sha256"))
        and bridge.get("run_manifest_sha256") == recomputed_manifest
    )
    calls = bridge.get("execution_call_provenance")
    calls_ok = bool(ids_exact and isinstance(calls, list) and len(calls) == 16)
    if calls_ok:
        for call_index, call in enumerate(calls):
            block = frame.iloc[32 * call_index : 32 * (call_index + 1)]
            expected_start = 25_000 + 32 * call_index
            if not (
                isinstance(call, dict)
                and call.get("object_start") == expected_start
                and call.get("object_stop") == expected_start + 32
                and call.get("objects") == 32
                and _is_sha256(call.get("base_draw_record_sha256"))
                and _is_sha256(call.get("base_draw_indices_sha256"))
                and _is_sha256(call.get("base_draw_probability_sha256"))
                and isinstance(call.get("bridge_record_sha256"), list)
                and call.get("bridge_record_sha256")
                == block["bridge_record_sha256"].astype(str).tolist()
                and block["base_draw_record_sha256"]
                .astype(str)
                .eq(call.get("base_draw_record_sha256"))
                .all()
            ):
                calls_ok = False
                break
    flow_counts_ok = bool(
        bridge.get("stratum_flow_evaluations") == 8_388_608
        and bridge.get("proposal_slots_total") == 16_777_216
        and bridge.get("target_unique_flow_evaluations")
        == int(counts["unique_evaluations"].sum())
        and bridge.get("target_pack_replay_flow_evaluations")
        == int(counts["unique_evaluations"][0])
        and metadata.get("flow_evaluations")
        == 8_388_608
        + int(counts["unique_evaluations"].sum())
        + int(counts["unique_evaluations"][0])
    ) if integer_counts else False
    phases = metadata.get("phase_seconds", {})
    method_seconds = float(metadata.get("method_seconds", np.nan))
    method_wall_seconds = float(metadata.get("method_wall_seconds", np.nan))
    proposal_seconds = float(bridge.get("proposal_initialization_seconds", np.nan))
    worker_wall_rows = values("bridge_worker_wall_seconds")
    worker_cpu_rows = values("bridge_worker_cpu_seconds")
    worker_rss_rows = values("bridge_worker_peak_rss_kib")
    phase_ok = bool(
        isinstance(phases, dict)
        and np.isfinite(method_seconds)
        and method_seconds > 0.0
        and np.isfinite(method_wall_seconds)
        and method_wall_seconds > 0.0
        and np.isfinite(proposal_seconds)
        and proposal_seconds >= 0.0
        and all(
            name in phases
            and np.isfinite(float(phases[name]))
            and float(phases[name]) >= 0.0
            for name in (*PACKED_BRIDGE_METHOD_TIME_COMPONENTS, "exact_reference_load")
        )
        and np.isclose(
            method_seconds,
            proposal_seconds
            + sum(float(phases[name]) for name in PACKED_BRIDGE_METHOD_TIME_COMPONENTS),
            rtol=1.0e-12,
            atol=1.0e-9,
        )
        and np.isclose(method_seconds, method_wall_seconds, rtol=1.0e-12, atol=1.0e-6)
        and np.isfinite(float(bridge.get("bridge_worker_cpu_seconds", np.nan)))
        and float(bridge.get("bridge_worker_cpu_seconds")) >= 0.0
        and np.isfinite(
            float(bridge.get("bridge_worker_wall_seconds_sum", np.nan))
        )
        and float(bridge.get("bridge_worker_wall_seconds_sum")) >= 0.0
        and np.isfinite(worker_wall_rows).all()
        and (worker_wall_rows >= 0.0).all()
        and np.isfinite(worker_cpu_rows).all()
        and (worker_cpu_rows >= 0.0).all()
        and np.isfinite(worker_rss_rows).all()
        and (worker_rss_rows > 0.0).all()
        and np.isclose(
            float(bridge.get("bridge_worker_wall_seconds_sum")),
            float(worker_wall_rows.sum(dtype=np.float64)),
            rtol=64.0 * np.finfo(float).eps,
            atol=1.0e-9,
        )
        and np.isclose(
            float(bridge.get("bridge_worker_cpu_seconds")),
            float(worker_cpu_rows.sum(dtype=np.float64)),
            rtol=64.0 * np.finfo(float).eps,
            atol=1.0e-9,
        )
        and bridge.get("bridge_worker_peak_rss_kib")
        == int(np.max(worker_rss_rows))
    )
    cuda_memory = bridge.get("cuda_memory", {})
    memory_ok = bool(
        isinstance(cuda_memory, dict)
        and cuda_memory.get("enabled") is True
        and isinstance(cuda_memory.get("device"), str)
        and cuda_memory["device"].startswith("cuda")
        and cuda_memory.get("device_name")
        == PACKED_BRIDGE_PROTOCOL["cuda_device_name"]
        and all(
            isinstance(cuda_memory.get(name), int)
            and cuda_memory[name] >= 0
            for name in (
                "start_allocated_bytes",
                "start_reserved_bytes",
                "peak_allocated_bytes",
                "peak_reserved_bytes",
                "peak_increment_allocated_bytes",
                "peak_increment_reserved_bytes",
            )
        )
        and isinstance(bridge.get("parent_peak_rss_kib"), int)
        and bridge["parent_peak_rss_kib"] > 0
        and isinstance(bridge.get("bridge_worker_peak_rss_kib"), int)
        and bridge["bridge_worker_peak_rss_kib"] > 0
        and cuda_memory["peak_allocated_bytes"]
        >= cuda_memory["start_allocated_bytes"]
        and cuda_memory["peak_reserved_bytes"]
        >= cuda_memory["start_reserved_bytes"]
        and cuda_memory["start_allocated_bytes"]
        <= cuda_memory["start_reserved_bytes"]
        and cuda_memory["peak_allocated_bytes"]
        <= cuda_memory["peak_reserved_bytes"]
        and cuda_memory["peak_increment_allocated_bytes"]
        == cuda_memory["peak_allocated_bytes"]
        - cuda_memory["start_allocated_bytes"]
        and cuda_memory["peak_increment_reserved_bytes"]
        == cuda_memory["peak_reserved_bytes"]
        - cuda_memory["start_reserved_bytes"]
        and bridge.get("slurm_cpus_per_task")
        == PACKED_BRIDGE_PROTOCOL["slurm_cpus_per_task"]
        and bridge.get("worker_thread_environment")
        == PACKED_BRIDGE_PROTOCOL["worker_thread_environment"]
        and np.isfinite(float(metadata.get("parent_process_cpu_seconds", np.nan)))
        and float(metadata.get("parent_process_cpu_seconds")) >= 0.0
    )

    island_log_names = [
        f"bridge_island_{island}_log_normalizer"
        for island in range(PACKED_BRIDGE_ISLANDS)
    ]
    collapse_names = [
        f"bridge_island_{island}_collapsed_stage"
        for island in range(PACKED_BRIDGE_ISLANDS)
    ]
    island_arrays_ok = bool(
        all(name in frame for name in (*island_log_names, *collapse_names))
    )
    island_logs = (
        frame[island_log_names].to_numpy(dtype=np.float64)
        if island_arrays_ok
        else np.empty((0, PACKED_BRIDGE_ISLANDS))
    )
    collapse = (
        frame[collapse_names].to_numpy(dtype=np.int64)
        if island_arrays_ok
        else np.empty((0, PACKED_BRIDGE_ISLANDS), dtype=np.int64)
    )
    expected_summary = (
        _summarize_island_rows(island_logs)
        if len(island_logs) == expected_objects
        else tuple(np.full(len(frame), np.nan) for _ in range(5))
    )
    complement_mass = values("mass_complement_packed_bridge")
    complement_log_mass = values("log_mass_complement_packed_bridge")
    complement_variance = np.square(
        values("mass_complement_packed_bridge_standard_error")
    )
    complement_log_variance = values("log_variance_complement_packed_bridge")
    complement_se = values("mass_complement_packed_bridge_standard_error")
    stratum_mass = values("mass_stratum")
    stratum_log_mass = values("log_mass_stratum")
    island_hashes_ok = bool(
        island_arrays_ok
        and "bridge_island_log_normalizer_sha256" in frame
        and "bridge_collapsed_stage_sha256" in frame
        and [
            _array_sha256(
                "bridge_island_log_normalizer", row, dtype=">f8"
            )
            for row in island_logs
        ]
        == frame["bridge_island_log_normalizer_sha256"].astype(str).tolist()
        and [
            _array_sha256("bridge_collapsed_stage", row, dtype=">i8")
            for row in collapse
        ]
        == frame["bridge_collapsed_stage_sha256"].astype(str).tolist()
    )
    island_summary_ok = bool(
        island_arrays_ok
        and island_hashes_ok
        and collapse.shape == (expected_objects, PACKED_BRIDGE_ISLANDS)
        and np.all((collapse >= 0) & (collapse <= PACKED_BRIDGE_STAGES))
        and np.array_equal(
            np.count_nonzero(collapse, axis=1), counts["bridge_collapsed_islands"]
        )
        and np.allclose(complement_mass, expected_summary[0], rtol=64 * np.finfo(float).eps, atol=0.0)
        and np.allclose(complement_log_mass, expected_summary[1], rtol=64 * np.finfo(float).eps, atol=0.0)
        and np.allclose(complement_variance, expected_summary[2], rtol=128 * np.finfo(float).eps, atol=0.0)
        and np.allclose(complement_log_variance, expected_summary[3], rtol=64 * np.finfo(float).eps, atol=0.0)
        and np.allclose(complement_se, expected_summary[4], rtol=64 * np.finfo(float).eps, atol=0.0)
    ) if integer_counts else False
    stage_columns = [
        *(f"bridge_stage_{stage}_mean_ess" for stage in range(1, 9)),
        *(f"bridge_stage_{stage}_minimum_lineages" for stage in range(1, 9)),
        *(f"bridge_stage_{stage}_minimum_unique_states" for stage in range(1, 9)),
        *(f"bridge_stage_{stage}_mean_acceptance" for stage in range(1, 8)),
        *(f"bridge_stage_{stage}_mean_movement" for stage in range(1, 8)),
    ]
    stage_values = (
        frame[stage_columns].to_numpy(dtype=np.float64)
        if all(name in frame for name in stage_columns)
        else np.empty((0, len(stage_columns)))
    )
    compact_arrays: dict[str, np.ndarray] = {}
    compact_arrays_ok = len(frame) == expected_objects
    if compact_arrays_ok:
        for metric, dtype, stages in _PACKED_BRIDGE_COMPACT_ARRAY_SPECS:
            columns = _compact_stage_columns(metric, stages)
            hash_name = f"bridge_{metric}_sha256"
            kind = np.dtype(dtype).kind
            if (
                not all(name in frame for name in columns)
                or hash_name not in frame
                or (
                    kind in "iu"
                    and not all(
                        pd.api.types.is_integer_dtype(frame[name].dtype)
                        for name in columns
                    )
                )
            ):
                compact_arrays_ok = False
                break
            array = frame[columns].to_numpy(
                dtype=np.int64 if kind in "iu" else np.float64
            ).reshape(
                expected_objects,
                PACKED_BRIDGE_ISLANDS,
                stages,
            )
            expected_hashes = [
                _array_sha256(f"bridge_{metric}", row, dtype=dtype)
                for row in array
            ]
            if expected_hashes != frame[hash_name].astype(str).tolist():
                compact_arrays_ok = False
                break
            compact_arrays[metric] = array
    if compact_arrays_ok:
        stage_log = compact_arrays["stage_log_normalizer"]
        stage_ess = compact_arrays["stage_effective_sample_size"]
        stage_acceptance = compact_arrays["stage_acceptance_rate"]
        stage_movement = compact_arrays["stage_movement_rate"]
        stage_lineages = compact_arrays["stage_unique_lineage_count"]
        stage_states = compact_arrays["stage_unique_state_count"]
        collapse_persistence = True
        for row in range(expected_objects):
            for island in range(PACKED_BRIDGE_ISLANDS):
                collapsed = int(collapse[row, island])
                if collapsed and not np.all(
                    np.isneginf(stage_log[row, island, collapsed - 1 :])
                ):
                    collapse_persistence = False
                    break
            if not collapse_persistence:
                break
    else:
        stage_log = np.empty((0, PACKED_BRIDGE_ISLANDS, PACKED_BRIDGE_STAGES))
        stage_ess = np.empty_like(stage_log)
        stage_acceptance = np.empty(
            (0, PACKED_BRIDGE_ISLANDS, PACKED_BRIDGE_MOVE_STAGES)
        )
        stage_movement = np.empty_like(stage_acceptance)
        stage_lineages = np.empty_like(stage_log, dtype=np.int64)
        stage_states = np.empty_like(stage_log, dtype=np.int64)
        collapse_persistence = False
    stage_diagnostics_ok = bool(
        compact_arrays_ok
        and
        stage_values.shape == (expected_objects, len(stage_columns))
        and np.isfinite(stage_values).all()
        and not np.isnan(stage_log).any()
        and not np.isposinf(stage_log).any()
        and np.array_equal(stage_log[:, :, -1], island_logs)
        and collapse_persistence
        and np.isfinite(stage_ess).all()
        and np.all((stage_ess >= 0.0) & (stage_ess <= 256.0 * (1.0 + 1e-12)))
        and np.isfinite(stage_acceptance).all()
        and np.isfinite(stage_movement).all()
        and np.all((stage_acceptance >= 0.0) & (stage_acceptance <= 1.0))
        and np.all((stage_movement >= 0.0) & (stage_movement <= stage_acceptance))
        and np.all((stage_lineages >= 0) & (stage_lineages <= 256))
        and np.all((stage_states >= 0) & (stage_states <= 256))
        and np.allclose(
            frame[
                [f"bridge_stage_{stage}_mean_ess" for stage in range(1, 9)]
            ].to_numpy(dtype=np.float64),
            stage_ess.mean(axis=1),
            rtol=64.0 * np.finfo(float).eps,
            atol=0.0,
        )
        and np.array_equal(
            frame[
                [
                    f"bridge_stage_{stage}_minimum_lineages"
                    for stage in range(1, 9)
                ]
            ].to_numpy(dtype=np.int64),
            stage_lineages.min(axis=1),
        )
        and np.array_equal(
            frame[
                [
                    f"bridge_stage_{stage}_minimum_unique_states"
                    for stage in range(1, 9)
                ]
            ].to_numpy(dtype=np.int64),
            stage_states.min(axis=1),
        )
        and np.allclose(
            frame[
                [
                    f"bridge_stage_{stage}_mean_acceptance"
                    for stage in range(1, 8)
                ]
            ].to_numpy(dtype=np.float64),
            stage_acceptance.mean(axis=1),
            rtol=64.0 * np.finfo(float).eps,
            atol=0.0,
        )
        and np.allclose(
            frame[
                [
                    f"bridge_stage_{stage}_mean_movement"
                    for stage in range(1, 8)
                ]
            ].to_numpy(dtype=np.float64),
            stage_movement.mean(axis=1),
            rtol=64.0 * np.finfo(float).eps,
            atol=0.0,
        )
    )
    stratum_nonnegative = bool(
        np.isfinite(stratum_mass).all()
        and (stratum_mass >= 0.0).all()
        and not np.isnan(stratum_log_mass).any()
        and not np.isposinf(stratum_log_mass).any()
        and np.array_equal(stratum_mass == 0.0, np.isneginf(stratum_log_mass))
        and np.allclose(
            np.log(stratum_mass[stratum_mass > 0.0]),
            stratum_log_mass[stratum_mass > 0.0],
            rtol=64 * np.finfo(float).eps,
            atol=64 * np.finfo(float).eps,
        )
    )
    finite_positive = bool(
        len(frame) == expected_objects
        and stratum_nonnegative
        and np.isfinite(complement_mass).all()
        and np.isfinite(complement_log_mass).all()
        and np.isfinite(complement_se).all()
        and np.isfinite(complement_log_variance).all()
        and (complement_mass > 0.0).all()
        and (complement_se > 0.0).all()
        and boolean_all("bridge_all_collapsed", expected=False)
        and boolean_all("bridge_independent_island_se_valid", expected=True)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        log_identity_ok = bool(
            finite_positive
            and np.allclose(np.log(complement_mass), complement_log_mass, rtol=64 * np.finfo(float).eps, atol=64 * np.finfo(float).eps)
            and np.allclose(2.0 * np.log(complement_se), complement_log_variance, rtol=64 * np.finfo(float).eps, atol=64 * np.finfo(float).eps)
        )
    integrity_gates = {
        "protocol_identity_matched": _gate(protocol_matched, "is", True, protocol_matched),
        "absolute_object_ids_exact": _gate(ids_exact, "is", True, ids_exact),
        "exact_reference_and_selection_identity_exact": _gate(exact_selection_ok, "is", True, exact_selection_ok),
        "preflight_and_integer_base_identity_exact": _gate(preflight_ok, "is", True, preflight_ok),
        "exact_support_unique_active": _gate(support_ok, "is", True, support_ok),
        "unconditioned_draw_and_unique_scatter_accounting_exact": _gate(draw_and_pack_ok, "is", True, draw_and_pack_ok),
        "object_and_island_replay_exact": _gate(replay_ok, "is", True, replay_ok),
        "fixed_full_trace_replay_persisted_and_revalidated": _gate(
            full_trace_replay_ok,
            "is",
            True,
            full_trace_replay_ok,
        ),
        "run_manifest_recomputes": _gate(manifest_ok, "is", True, manifest_ok),
        "manifest_hash_columns_complete": _gate(hash_columns_ok, "is", True, hash_columns_ok),
        "execution_call_provenance_exact": _gate(calls_ok, "is", True, calls_ok),
        "flow_evaluation_counts_exact": _gate(flow_counts_ok, "is", True, flow_counts_ok),
        "method_phase_timing_complete": _gate(phase_ok, "is", True, phase_ok),
        "process_and_cuda_memory_recorded": _gate(memory_ok, "is", True, memory_ok),
        "island_linear_reduction_recomputes": _gate(island_summary_ok, "is", True, island_summary_ok),
        "stage_diagnostics_in_range": _gate(stage_diagnostics_ok, "is", True, stage_diagnostics_ok),
        "all_objects_have_valid_independent_island_precision": _gate(finite_positive, "is", True, finite_positive),
        "mass_and_standard_error_log_identity": _gate(log_identity_ok, "is", True, log_identity_ok),
    }

    pareto = values("pareto_k_packed_bridge_base_iid")
    finite_k = np.isfinite(pareto)
    relative_error = np.divide(
        complement_se,
        np.abs(complement_mass),
        out=np.full(len(frame), np.nan),
        where=np.abs(complement_mass) > 0.0,
    )
    k_below_05 = int(np.sum(finite_k & (pareto < 0.5)))
    k_below_07 = int(np.sum(finite_k & (pareto < 0.7)))
    k_at_least_1 = int(np.sum(finite_k & (pareto >= 1.0)))
    rel_median = (
        float(np.median(relative_error))
        if np.isfinite(relative_error).all()
        else float("nan")
    )
    rel_p90 = (
        _higher_percentile(relative_error, 90)
        if np.isfinite(relative_error).all()
        else float("nan")
    )
    reliability_gates = {
        "base_iid_pareto_k_defined_all_objects": _gate(int((~finite_k).sum()), "undefined_count ==", 0, len(pareto) == expected_objects and finite_k.all()),
        "base_iid_pareto_k_below_0_5_count": _gate(k_below_05, ">=", thresholds["pareto_k_below_0_5_count"], k_below_05 >= thresholds["pareto_k_below_0_5_count"]),
        "base_iid_pareto_k_below_0_7_count": _gate(k_below_07, ">=", thresholds["pareto_k_below_0_7_count"], k_below_07 >= thresholds["pareto_k_below_0_7_count"]),
        "base_iid_pareto_k_at_least_1_count": _gate(k_at_least_1, "==", thresholds["pareto_k_at_least_1_count"], k_at_least_1 == thresholds["pareto_k_at_least_1_count"]),
        "complement_relative_error_median": _gate(rel_median, "<=", thresholds["complement_relative_error_median"], np.isfinite(rel_median) and rel_median <= thresholds["complement_relative_error_median"]),
        "complement_relative_error_p90_higher": _gate(rel_p90, "<=", thresholds["complement_relative_error_p90_higher"], np.isfinite(rel_p90) and rel_p90 <= thresholds["complement_relative_error_p90_higher"]),
    }

    exact_stratum_all = values("mass_stratum_exact_oracle")
    exact_complement_all = values("mass_complement_exact_oracle")
    exact_total_all = values("mass_total_exact_oracle")
    exact_stratum = exact_stratum_all[selected]
    exact_complement = exact_complement_all[selected]
    exact_total = exact_total_all[selected]
    archived_selected = (
        archived_table["exact_oracle_selected"].to_numpy(dtype=bool)
        if "exact_oracle_selected" in archived_table
        else np.zeros(len(archived_table), dtype=bool)
    )
    archived_exact = (
        archived_table.loc[
            archived_selected,
            [
                "mass_stratum_exact_oracle",
                "mass_complement_exact_oracle",
                "mass_total_exact_oracle",
            ],
        ].to_numpy(dtype=np.float64)
        if len(archived_table) == expected_objects
        else np.empty((0, 3), dtype=np.float64)
    )
    expected_mass_hashes = []
    for row, is_selected in enumerate(selected):
        if is_selected:
            expected_mass_hashes.append(
                _record_sha256(
                    "exact_reference_mass",
                    (
                        _text_sha256("selected", True),
                        _text_sha256("stratum", exact_stratum_all[row]),
                        _text_sha256("complement", exact_complement_all[row]),
                        _text_sha256("total", exact_total_all[row]),
                    ),
                )
            )
        else:
            expected_mass_hashes.append(
                _record_sha256(
                    "exact_reference_mass",
                    (_text_sha256("selected", False),),
                )
            )
    exact_mass_identity_ok = bool(
        archived_exact.shape == (64, 3)
        and np.isfinite(archived_exact).all()
        and np.isfinite(exact_stratum).all()
        and np.isfinite(exact_complement).all()
        and np.isfinite(exact_total).all()
        and (exact_stratum > 0.0).all()
        and (exact_complement > 0.0).all()
        and (exact_total > 0.0).all()
        and np.allclose(
            exact_stratum + exact_complement,
            exact_total,
            rtol=64.0 * np.finfo(float).eps,
            atol=0.0,
        )
        and np.array_equal(
            np.column_stack((exact_stratum, exact_complement, exact_total)),
            archived_exact,
        )
        and "exact_reference_mass_sha256" in frame
        and expected_mass_hashes
        == frame["exact_reference_mass_sha256"].astype(str).tolist()
    )
    estimated_total_log = np.logaddexp(stratum_log_mass[selected], complement_log_mass[selected])
    delta_signed = estimated_total_log - np.log(exact_total)
    delta = np.abs(delta_signed)
    exact_z = (complement_mass[selected] - exact_complement) / complement_se[selected]
    stratum_crosscheck = np.abs(stratum_mass[selected] - exact_stratum) / np.abs(
        exact_stratum
    )
    derived_exact_columns_ok = bool(
        np.allclose(
            values("delta_log_total_evidence_packed_bridge_minus_exact")[selected],
            delta_signed,
            rtol=64.0 * np.finfo(float).eps,
            atol=64.0 * np.finfo(float).eps,
        )
        and np.allclose(
            values("mass_complement_packed_bridge_minus_exact_z")[selected],
            exact_z,
            rtol=64.0 * np.finfo(float).eps,
            atol=64.0 * np.finfo(float).eps,
        )
        and np.allclose(
            values("mass_stratum_exact_oracle_relative_error")[selected],
            stratum_crosscheck,
            rtol=64.0 * np.finfo(float).eps,
            atol=64.0 * np.finfo(float).eps,
        )
    )
    valid_exact = bool(
        selected.sum() == 64
        and len(delta) == len(exact_z) == len(stratum_crosscheck) == 64
        and exact_mass_identity_ok
        and derived_exact_columns_ok
        and np.isfinite(delta).all()
        and np.isfinite(exact_z).all()
        and np.isfinite(stratum_crosscheck).all()
    )
    median_delta = float(np.median(delta)) if valid_exact else float("nan")
    p90_delta = _higher_percentile(delta, 90) if valid_exact else float("nan")
    crosscheck_max = float(np.max(stratum_crosscheck)) if valid_exact else float("nan")
    selected_difference = complement_mass[selected] - exact_complement
    aggregate_denominator = float(np.sqrt(np.square(complement_se[selected]).sum()))
    aggregate_z = (
        float(selected_difference.sum() / aggregate_denominator)
        if valid_exact and aggregate_denominator > 0.0
        else float("nan")
    )
    z_rms = (
        float(np.sqrt(np.mean(np.square(exact_z)))) if valid_exact else float("nan")
    )
    within_196 = int(np.sum(np.abs(exact_z) <= 1.96)) if valid_exact else 0
    exact_gates = {
        "selected_exact_object_count": _gate(int(selected.sum()), "==", 64, int(selected.sum()) == 64),
        "absolute_delta_log_total_evidence_median": _gate(median_delta, "<=", thresholds["absolute_delta_log_total_evidence_median"], np.isfinite(median_delta) and median_delta <= thresholds["absolute_delta_log_total_evidence_median"]),
        "absolute_delta_log_total_evidence_p90_higher": _gate(p90_delta, "<=", thresholds["absolute_delta_log_total_evidence_p90_higher"], np.isfinite(p90_delta) and p90_delta <= thresholds["absolute_delta_log_total_evidence_p90_higher"]),
        "stratum_relative_crosscheck_max": _gate(crosscheck_max, "<=", thresholds["stratum_relative_crosscheck_max"], np.isfinite(crosscheck_max) and crosscheck_max <= thresholds["stratum_relative_crosscheck_max"]),
        "absolute_signed_aggregate_z": _gate(abs(aggregate_z) if np.isfinite(aggregate_z) else float("nan"), "<=", thresholds["absolute_signed_aggregate_z"], np.isfinite(aggregate_z) and abs(aggregate_z) <= thresholds["absolute_signed_aggregate_z"]),
        "exact_z_rms": _gate(z_rms, "in_closed_interval", thresholds["exact_z_rms"], np.isfinite(z_rms) and thresholds["exact_z_rms"][0] <= z_rms <= thresholds["exact_z_rms"][1]),
        "exact_z_within_1_96_count": _gate(within_196, ">=", thresholds["exact_z_within_1_96_count"], within_196 >= thresholds["exact_z_within_1_96_count"]),
    }
    integrity_gates["exact_reference_masses_match_archive"] = _gate(
        exact_mass_identity_ok,
        "is",
        True,
        exact_mass_identity_ok,
    )
    integrity_gates["exact_gate_inputs_recompute"] = _gate(
        derived_exact_columns_ok,
        "is",
        True,
        derived_exact_columns_ok,
    )

    total_log_mass = np.logaddexp(stratum_log_mass, complement_log_mass)
    total_log_relative_error_squared = complement_log_variance - 2.0 * total_log_mass
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        total_relative_error_squared = np.exp(total_log_relative_error_squared)
    baseline = LEARNED_RETRIEVAL_CENTER_BASELINE
    finite_efficiency = bool(
        len(total_relative_error_squared) == expected_objects
        and np.isfinite(total_log_mass).all()
        and np.isfinite(total_log_relative_error_squared).all()
        and np.isfinite(total_relative_error_squared).all()
        and (total_relative_error_squared > 0.0).all()
        and np.isfinite(method_seconds)
        and method_seconds > 0.0
    )
    efficiency_mean = (
        float(np.mean(total_relative_error_squared))
        * method_seconds
        / (baseline["mean_total_relative_error_squared"] * baseline["method_seconds"])
        if finite_efficiency
        else float("nan")
    )
    efficiency_p90 = (
        _higher_percentile(total_relative_error_squared, 90)
        * method_seconds
        / (
            baseline["total_relative_error_squared_p90_higher"]
            * baseline["method_seconds"]
        )
        if finite_efficiency
        else float("nan")
    )
    efficiency_gates = {
        "mean_relative_variance_time_efficiency": _gate(efficiency_mean, "<=", thresholds["efficiency_mean"], np.isfinite(efficiency_mean) and efficiency_mean <= thresholds["efficiency_mean"]),
        "p90_higher_relative_variance_time_efficiency": _gate(efficiency_p90, "<=", thresholds["efficiency_p90_higher"], np.isfinite(efficiency_p90) and efficiency_p90 <= thresholds["efficiency_p90_higher"]),
    }
    families = {
        "integrity": integrity_gates,
        "base_tail_reliability": reliability_gates,
        "exact_calibration": exact_gates,
        "efficiency": efficiency_gates,
    }
    passed = all(
        gate["passed"] for family in families.values() for gate in family.values()
    )
    return {
        "protocol": {
            "expected": expected_protocol,
            "observed": observed_protocol,
            "matched": protocol_matched,
        },
        "method_seconds": method_seconds,
        "method_wall_seconds": method_wall_seconds,
        "proposal_initialization_seconds": proposal_seconds,
        "bridge_worker_cpu_seconds": bridge.get("bridge_worker_cpu_seconds"),
        "target_unique_flow_evaluations": bridge.get("target_unique_flow_evaluations"),
        "complement_relative_error_median": rel_median,
        "complement_relative_error_p90_higher": rel_p90,
        "absolute_delta_log_total_evidence_median": median_delta,
        "absolute_delta_log_total_evidence_p90_higher": p90_delta,
        "stratum_relative_crosscheck_max": crosscheck_max,
        "signed_aggregate_z": aggregate_z,
        "exact_z_rms": z_rms,
        "exact_z_within_1_96_count": within_196,
        "efficiency_mean": efficiency_mean,
        "efficiency_p90_higher": efficiency_p90,
        "runtime_caveat": (
            "the frozen 91.04022970038932-second DM3 baseline includes "
            "36.10254 seconds of diagnostic prefilter flow and is not a "
            "production v1.1-infer speed claim"
        ),
        "gates": {
            "predeclared": protocol_matched,
            "thresholds_predeclared": True,
            **families,
            "passed": bool(passed),
            "policy": (
                "all integrity, base-tail, exact-calibration, and fair-efficiency "
                "gates must pass; collapsed islands are retained and never retried"
            ),
        },
    }


__all__ = [
    "PACKED_BRIDGE_ACTIVE_ATOMS",
    "PACKED_BRIDGE_BASE_DOMAIN",
    "PACKED_BRIDGE_BASE_SEED",
    "PACKED_BRIDGE_CENTER_GATE_THRESHOLDS",
    "PACKED_BRIDGE_CENTER_LABEL",
    "PACKED_BRIDGE_EXACT_REFERENCE_SUMMARY_SHA256",
    "PACKED_BRIDGE_EXACT_REFERENCE_TABLE_SHA256",
    "PACKED_BRIDGE_ISLANDS",
    "PACKED_BRIDGE_KERNEL_SEED",
    "PACKED_BRIDGE_METHOD_TIME_COMPONENTS",
    "PACKED_BRIDGE_PARTICLES",
    "PACKED_BRIDGE_PREFLIGHT_IDENTITY",
    "PACKED_BRIDGE_PREFLIGHT_REPORT_SHA256",
    "PACKED_BRIDGE_PROTOCOL",
    "PACKED_BRIDGE_SLOTS_PER_OBJECT",
    "PACKED_BRIDGE_WORKERS",
    "PackedBridgeDiagnostic",
    "PackedBridgeExactReference",
    "PackedTargetPlan",
    "diagnose_packed_integer_bridge_complement",
    "load_packed_bridge_exact_reference",
    "load_verified_packed_bridge_base",
    "packed_bridge_center_summary",
    "packed_bridge_run_manifest_sha256",
    "run_compact_bridge_batch",
]
