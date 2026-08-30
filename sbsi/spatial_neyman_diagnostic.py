"""Frozen centre gate for spatially stratified pilot/Neyman DM3 sampling.

This module is deliberately diagnostic-only.  It sums one deterministic local
support exactly, partitions its complement into outcome-independent spatial
cells, discards a small pilot, and uses an independent fixed-source production
reservoir with whole-cycle Neyman allocation.  Nothing here changes an
inference default.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from hashlib import sha256
import struct
import time
from typing import Any, Optional

import numpy as np
import pandas as pd

from .catalogue_sampling import (
    SPATIAL_PARTITION_RECIPE_SHA256,
    TAIL_PROPOSAL_RECIPE,
    _candidate_exclusion_manifest_sha256,
    pareto_tail_index,
)
from .complement_diagnostic import (
    EXACT_ORACLE_ATOM_CHUNK,
    EXACT_ORACLE_OBJECT_CHUNK,
    _stream_exact_oracle_mass,
    exact_oracle_object_ids,
)
from .learned_retrieval_diagnostic import (
    LEARNED_RETRIEVAL_CENTER_BASELINE,
    LEARNED_RETRIEVAL_CENTER_MOCK_SHA256,
)


SPATIAL_NEYMAN_CENTER_LABEL = "SBSI/complement-spatial-neyman-center-development/v1"
SPATIAL_NEYMAN_CENTER_GATE_THRESHOLDS = {
    "objects": 512,
    "pareto_k_below_0_5_count": 461,
    "pareto_k_below_0_7_count": 507,
    "pareto_k_at_least_1_count": 0,
    "complement_relative_error_median": 0.00882,
    "complement_relative_error_p90_higher": 0.0268,
    "efficiency_mean": 0.90,
    "efficiency_p90_higher": 0.90,
    "absolute_delta_log_total_evidence_median": 0.01,
    "absolute_delta_log_total_evidence_p90_higher": 0.05,
    "stratum_relative_crosscheck_max": 1.0e-5,
    "absolute_signed_aggregate_z": 2.5,
    "exact_z_rms": [0.8, 1.2],
    "exact_z_within_1_96_count": 58,
}
SPATIAL_NEYMAN_METHOD_TIME_COMPONENTS = (
    "partition_load_or_build",
    "candidate_query",
    "stratum_exact_flow",
    "reservoir_construction",
    "reservoir_mapping",
    "pilot_flow",
    "allocation",
    "selected_production_flow",
    "reduction",
)
SPATIAL_NEYMAN_PROTOCOL = {
    "label": SPATIAL_NEYMAN_CENTER_LABEL,
    "universe_start": 25_000,
    "universe_stop": 25_512,
    "center": [0.013610377120038044, 0.002740148300964851],
    "initial_strategy": "mean_observed_shape",
    "n_candidates": 16_384,
    "prefilter_candidates": 131_072,
    "n_global_draws": 16_384,
    "pilot_draws": 4_096,
    "proposal_seed": 8_701,
    "pilot_seed": 1_008_704,
    "production_seed": 7_008_704,
    "candidate_backend": "torch",
    "diagnostic_object_chunk": 8,
    "flow_atom_chunk": 4_096,
    "spatial_atom_chunk": 262_144,
    "spatial_object_chunk": 8,
    "n_cells": 256,
    "pilot_cycles_per_cell": 2,
    "production_reservoir_cycles_per_cell": 64,
    "allocation_min_cycles": 2,
    "allocation_max_cycles": 64,
    "allocation_total_cycles": 2_048,
    "tail_proposal_recipe": asdict(TAIL_PROPOSAL_RECIPE),
    "partition_recipe_sha256": SPATIAL_PARTITION_RECIPE_SHA256,
    # Outcome-blind A40 audit job 16124243 from detached commit b3d8d2e read
    # zero science rows, issued zero candidate queries, and evaluated zero
    # target-flow terms.  Science fails closed on any semantic or byte drift.
    "partition_sha256": (
        "0da871a79abfd124360a2d93f02707bb88de42e5281edfda741a819a58b557ab"
    ),
    "partition_cache_sha256": (
        "1a63e4a7e435f79fa319f1a909698ee117bcb6394a257e0258ace9daf0248e3c"
    ),
    "partition_manifest_sha256": (
        "7cd8ae16a65c8488b5ff6c8abfeef5edbdce75b0035dcb37ad83fa51ff29cdef"
    ),
    "exact_selection_manifest_sha256": (
        "6abcc361aec9ada6010f2a91a68aa70440904873518182445c2a712ae69bad6a"
    ),
    "exact_selection_count": 64,
    "exact_atom_chunk": EXACT_ORACLE_ATOM_CHUNK,
    "exact_object_chunk": EXACT_ORACLE_OBJECT_CHUNK,
    "active_atoms": 12_760_990,
    "stratum_flow_evaluations": 8_388_608,
    "pilot_flow_evaluations": 2_097_152,
    "production_flow_evaluations": 8_388_608,
    "exact_flow_evaluations": 816_703_360,
    "run_flow_evaluations": 835_577_728,
    "mock_input_sha256": dict(LEARNED_RETRIEVAL_CENTER_MOCK_SHA256),
}

# Internal compatibility name used while this diagnostic was being developed.
SPATIAL_NEYMAN_BASE_PROTOCOL = SPATIAL_NEYMAN_PROTOCOL


def spatial_neyman_center_protocol(
    partition_sha256: str,
    partition_cache_sha256: str,
    partition_manifest_sha256: str,
) -> dict:
    """Return the frozen protocol completed by an outcome-blind partition hash."""

    identities = (
        partition_sha256,
        partition_cache_sha256,
        partition_manifest_sha256,
    )
    if not all(_is_sha256(value) for value in identities):
        raise ValueError("all expected partition identities must be lowercase SHA-256")
    return {
        **deepcopy(SPATIAL_NEYMAN_PROTOCOL),
        "partition_sha256": partition_sha256,
        "partition_cache_sha256": partition_cache_sha256,
        "partition_manifest_sha256": partition_manifest_sha256,
    }


def _gate(observed: Any, operator: str, threshold: Any, passed: bool) -> dict:
    return {
        "observed": observed,
        "operator": operator,
        "threshold": threshold,
        "passed": bool(passed),
    }


def _higher_percentile(values: np.ndarray, percentile: float) -> float:
    return float(np.percentile(values, percentile, method="higher"))


def _is_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _exact_int64_vector(values) -> np.ndarray | None:
    try:
        if isinstance(values, pd.Series):
            if not pd.api.types.is_integer_dtype(values.dtype):
                return None
            raw = values.to_numpy()
        else:
            raw = np.asarray(values)
            if raw.dtype.kind not in "iu" or raw.ndim != 1:
                return None
        scalars = raw.tolist()
        if any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or value < 0
            or value > np.iinfo(np.int64).max
            for value in scalars
        ):
            return None
        return np.asarray(scalars, dtype=np.int64)
    except (TypeError, ValueError, OverflowError):
        return None


def _canonical_array_sha256(label: str, values: np.ndarray, dtype: str) -> str:
    """Hash an array with explicit domain, shape, and big-endian dtype."""

    array = np.asarray(values)
    canonical = np.ascontiguousarray(array, dtype=np.dtype(dtype))
    digest = sha256()
    digest.update(label.encode("utf-8") + b"\0")
    digest.update(struct.pack(">Q", canonical.ndim))
    for width in canonical.shape:
        digest.update(struct.pack(">Q", int(width)))
    digest.update(canonical.tobytes())
    return digest.hexdigest()


_SPATIAL_MANIFEST_HASH_COLUMNS = (
    "stratum_indices_sha256",
    "stratum_log_target_sha256",
    "pilot_rng_sha256",
    "production_reservoir_rng_sha256",
    "production_rng_sha256",
    "pilot_exclusion_indices_sha256",
    "production_exclusion_indices_sha256",
    "pilot_cell_ids_sha256",
    "production_reservoir_cell_ids_sha256",
    "selected_production_cell_ids_sha256",
    "pilot_source_ids_sha256",
    "production_reservoir_source_ids_sha256",
    "selected_production_source_ids_sha256",
    "pilot_probability_sha256",
    "pilot_normalizer_sha256",
    "production_reservoir_probability_sha256",
    "production_normalizer_sha256",
    "selected_production_probability_sha256",
    "pilot_log_ratio_sha256",
    "selected_production_log_ratio_sha256",
    "allocation_sha256",
)


def spatial_neyman_run_manifest_sha256(frame: pd.DataFrame) -> str:
    """Hash all per-object support, RNG, cell, source, flow, and allocation ids."""

    required = ("object_index", *_SPATIAL_MANIFEST_HASH_COLUMNS)
    if any(name not in frame for name in required):
        raise ValueError("spatial-Neyman manifest columns are incomplete")
    rows = []
    for record in frame.loc[:, required].to_dict(orient="records"):
        object_id = record["object_index"]
        if (
            isinstance(object_id, (bool, np.bool_))
            or not isinstance(object_id, (int, np.integer))
            or object_id < 0
            or object_id > np.iinfo(np.int64).max
        ):
            raise ValueError("spatial-Neyman object ids must be non-negative int64")
        hashes = tuple(record[name] for name in _SPATIAL_MANIFEST_HASH_COLUMNS)
        if not all(_is_sha256(value) for value in hashes):
            raise ValueError("spatial-Neyman manifest contains an invalid SHA-256")
        rows.append((int(object_id), hashes))
    rows.sort(key=lambda item: item[0])
    if len({object_id for object_id, _ in rows}) != len(rows):
        raise ValueError("spatial-Neyman manifest object ids must be unique")
    digest = sha256()
    digest.update(b"SBSI/spatial-neyman-center-run/v1\0")
    digest.update(struct.pack(">Q", len(rows)))
    for object_id, hashes in rows:
        digest.update(struct.pack(">Q", object_id))
        for value in hashes:
            digest.update(bytes.fromhex(value))
    return digest.hexdigest()


@dataclass(frozen=True)
class _SpatialReduction:
    mass: float
    standard_error: float
    pareto_k: float
    maximum_effective_weight: float
    variance: float
    effective_weights: np.ndarray
    cell_draw_counts: np.ndarray
    cell_source_draw_counts: np.ndarray


def _reduce_spatial_mis_arrays(
    values: np.ndarray,
    cell_id: np.ndarray,
    source_id: np.ndarray,
    *,
    cells: int = 256,
    sources: int = 3,
) -> _SpatialReduction:
    """Reduce fixed-source cell samples using conditional unbiasedness.

    ``values`` are ``c/q_h`` where ``q_h`` is normalized within cell ``h``.
    The complement estimate is the sum of cell means.  Fixed-source variance
    is ``sum_{h,l} n_hl s_hl^2 / n_h^2``.  Pareto diagnostics use the single
    equivalent-mean vector ``M/n_h * c/q_h``.
    """

    values = np.asarray(values, dtype=np.float64)
    cell_id = np.asarray(cell_id, dtype=np.int64)
    source_id = np.asarray(source_id, dtype=np.int64)
    if values.ndim != 1 or cell_id.shape != values.shape or source_id.shape != values.shape:
        raise ValueError("values, cells, and sources must be aligned vectors")
    if not len(values) or not np.isfinite(values).all() or (values < 0.0).any():
        raise ValueError("spatial importance values must be finite, nonnegative, and nonempty")
    if ((cell_id < 0) | (cell_id >= cells)).any():
        raise ValueError("spatial cell ids are out of range")
    if ((source_id < 0) | (source_id >= sources)).any():
        raise ValueError("spatial source ids are out of range")

    cell_counts = np.bincount(cell_id, minlength=cells).astype(np.int64)
    if (cell_counts == 0).any():
        raise ValueError("every spatial cell must receive production draws")
    source_counts = np.zeros((cells, sources), dtype=np.int64)
    mass = 0.0
    variance = 0.0
    effective = np.empty_like(values)
    total_draws = len(values)
    for cell in range(cells):
        member = cell_id == cell
        selected = values[member]
        count = len(selected)
        mass += float(selected.mean(dtype=np.float64))
        effective[member] = (total_draws / count) * selected
        for source in range(sources):
            source_values = values[member & (source_id == source)]
            source_count = len(source_values)
            source_counts[cell, source] = source_count
            if source_count < 2:
                raise ValueError("each spatial cell/source needs two production draws")
            variance += source_count * float(source_values.var(ddof=1)) / count**2
    k = float(pareto_tail_index(effective[None, :])[0])
    return _SpatialReduction(
        mass=float(mass),
        standard_error=float(np.sqrt(variance)),
        pareto_k=k,
        maximum_effective_weight=float(effective.max()),
        variance=float(variance),
        effective_weights=effective,
        cell_draw_counts=cell_counts,
        cell_source_draw_counts=source_counts,
    )


def _rescale_log_values(
    log_values: np.ndarray, *, groups: Optional[np.ndarray] = None
) -> tuple[np.ndarray, float]:
    """Exponentiate after a per-object log shift for stable float64 moments."""

    values = np.asarray(log_values, dtype=np.float64)
    if values.ndim != 1 or np.isnan(values).any() or np.isposinf(values).any():
        raise ValueError("log values must be a vector containing finite values or -inf")
    finite = np.isfinite(values)
    if not finite.any():
        raise ValueError("at least one spatial importance value must be finite")
    shift = float(np.max(values[finite]))
    scaled = np.exp(values - shift, dtype=np.float64)
    if groups is not None:
        groups = np.asarray(groups)
        if groups.shape != values.shape:
            raise ValueError("underflow-check groups must align with log values")
        for group in np.unique(groups):
            member = groups == group
            if np.isfinite(values[member]).any() and not np.any(scaled[member] > 0.0):
                raise RuntimeError("scaled spatial reduction underflowed an entire cell")
    return scaled, shift


def _stable_spatial_pilot_variance(log_values: np.ndarray, reservoir, reducer) -> np.ndarray:
    """Compute relative cell variances without a catalogue-wide exponent shift."""

    logs = np.asarray(log_values, dtype=np.float64)
    if logs.shape != reservoir.indices.shape:
        raise ValueError("pilot log ratios must align with the spatial reservoir")
    cells = np.asarray(reservoir.cell_id, dtype=np.int64)
    if cells.ndim != 1 or cells.shape != (logs.shape[1],):
        raise ValueError("pilot reservoir must expose one shared cell schedule")
    scaled = np.zeros_like(logs)
    shifts = np.full((len(logs), 256), -np.inf, dtype=np.float64)
    for row in range(len(logs)):
        for cell in range(256):
            member = cells == cell
            finite = np.isfinite(logs[row, member])
            if finite.any():
                shift = float(np.max(logs[row, member][finite]))
                shifts[row, cell] = shift
                scaled[row, member] = np.exp(logs[row, member] - shift)
    local_variance = np.asarray(reducer(scaled, reservoir), dtype=np.float64)
    if local_variance.shape != shifts.shape:
        raise RuntimeError("pilot variance reducer returned the wrong cell shape")
    result = np.zeros_like(local_variance)
    positive = local_variance > 0.0
    for row in range(len(logs)):
        usable = positive[row] & np.isfinite(shifts[row])
        if not usable.any():
            continue
        log_variance = np.full(256, -np.inf, dtype=np.float64)
        log_variance[usable] = (
            np.log(local_variance[row, usable]) + 2.0 * shifts[row, usable]
        )
        peak = float(np.max(log_variance[usable]))
        # Preserve every genuinely positive pilot variance.  Values below the
        # normal float64 range are negligible for allocation but remain
        # positive, so they cannot masquerade as an all-zero fallback.
        floor = float(np.log(np.finfo(np.float64).tiny))
        result[row, usable] = np.exp(np.maximum(log_variance[usable] - peak, floor))
    return result


def _scaled_sum(log_values: np.ndarray) -> tuple[float, float]:
    scaled, shift = _rescale_log_values(log_values)
    log_mass = float(shift + np.log(scaled.sum(dtype=np.float64)))
    mass = float(np.exp(log_mass))
    if not np.isfinite(mass) or mass <= 0.0:
        raise RuntimeError("scaled exact-stratum reduction was non-finite")
    return mass, log_mass


@dataclass
class SpatialNeymanDiagnostic:
    """Per-object centre estimates and immutable spatial-sampling identity."""

    table: pd.DataFrame
    metadata: dict

    def summary(self) -> dict:
        return spatial_neyman_center_summary(self)


def _observed_protocol(diagnostic) -> dict | None:
    metadata = diagnostic.metadata
    spatial = metadata.get("spatial_neyman_center")
    exact = metadata.get("exact_oracle")
    if not isinstance(spatial, dict) or not isinstance(exact, dict):
        return None
    try:
        return {
            "label": spatial["label"],
            "universe_start": int(metadata["observation_start"]),
            "universe_stop": int(metadata["observation_stop"]),
            "center": [float(value) for value in metadata["center"]],
            "initial_strategy": spatial["initial_strategy"],
            "n_candidates": int(metadata["n_candidates"]),
            "prefilter_candidates": int(metadata["prefilter_candidates"]),
            "n_global_draws": int(spatial["n_global_draws"]),
            "pilot_draws": int(spatial["pilot_draws"]),
            "proposal_seed": int(spatial["proposal_seed"]),
            "pilot_seed": int(spatial["pilot_seed"]),
            "production_seed": int(spatial["production_seed"]),
            "candidate_backend": metadata["candidate_backend"],
            "diagnostic_object_chunk": int(metadata["object_chunk"]),
            "flow_atom_chunk": int(metadata["flow_atom_chunk"]),
            "spatial_atom_chunk": int(spatial["spatial_atom_chunk"]),
            "spatial_object_chunk": int(spatial["spatial_object_chunk"]),
            "n_cells": int(spatial["n_cells"]),
            "pilot_cycles_per_cell": int(spatial["pilot_cycles_per_cell"]),
            "production_reservoir_cycles_per_cell": int(
                spatial["production_reservoir_cycles_per_cell"]
            ),
            "allocation_min_cycles": int(spatial["allocation_min_cycles"]),
            "allocation_max_cycles": int(spatial["allocation_max_cycles"]),
            "allocation_total_cycles": int(spatial["allocation_total_cycles"]),
            "tail_proposal_recipe": spatial["tail_proposal_recipe"],
            "partition_recipe_sha256": spatial["partition_recipe_sha256"],
            "partition_sha256": spatial["partition_sha256"],
            "partition_cache_sha256": spatial["partition_cache_sha256"],
            "partition_manifest_sha256": spatial["partition_manifest_sha256"],
            "exact_selection_manifest_sha256": exact["selection_manifest_sha256"],
            "exact_selection_count": int(exact["selection_count"]),
            "exact_atom_chunk": int(exact["atom_chunk"]),
            "exact_object_chunk": int(exact["object_chunk"]),
            "active_atoms": int(exact["active_atoms"]),
            "stratum_flow_evaluations": int(spatial["stratum_flow_evaluations"]),
            "pilot_flow_evaluations": int(spatial["pilot_flow_evaluations"]),
            "production_flow_evaluations": int(spatial["production_flow_evaluations"]),
            "exact_flow_evaluations": int(exact["flow_evaluations"]),
            "run_flow_evaluations": int(metadata["flow_evaluations"]),
            "mock_input_sha256": spatial["mock_input_sha256"],
        }
    except (KeyError, TypeError, ValueError):
        return None


def spatial_neyman_center_summary(diagnostic) -> dict:
    """Evaluate every hard gate in the frozen 512-object centre protocol."""

    frame = diagnostic.table
    metadata = diagnostic.metadata
    spatial = metadata.get("spatial_neyman_center")
    if not isinstance(spatial, dict):
        raise ValueError("diagnostic lacks spatial-Neyman centre metadata")
    expected_partition = SPATIAL_NEYMAN_PROTOCOL["partition_sha256"]
    expected_partition_cache = SPATIAL_NEYMAN_PROTOCOL["partition_cache_sha256"]
    expected_partition_manifest = SPATIAL_NEYMAN_PROTOCOL["partition_manifest_sha256"]
    partition_frozen = all(
        _is_sha256(value)
        for value in (
            expected_partition,
            expected_partition_cache,
            expected_partition_manifest,
        )
    )
    expected_protocol = deepcopy(SPATIAL_NEYMAN_PROTOCOL)
    observed_protocol = _observed_protocol(diagnostic)
    protocol_matched = bool(
        partition_frozen and observed_protocol == expected_protocol
    )
    thresholds = SPATIAL_NEYMAN_CENTER_GATE_THRESHOLDS
    baseline = LEARNED_RETRIEVAL_CENTER_BASELINE
    expected_objects = int(thresholds["objects"])

    def values(name: str) -> np.ndarray:
        return frame[name].to_numpy(dtype=np.float64)

    def boolean_column_all(name: str, expected: bool) -> bool:
        return bool(
            name in frame
            and pd.api.types.is_bool_dtype(frame[name].dtype)
            and np.all(frame[name].to_numpy(dtype=bool) == expected)
        )

    object_ids = _exact_int64_vector(frame.get("object_index"))
    ids_exact = bool(
        object_ids is not None
        and np.array_equal(object_ids, np.arange(25_000, 25_512, dtype=np.int64))
    )
    expected_exact_ids, expected_exact_manifest = exact_oracle_object_ids(25_000, 25_512)
    exact_mask_valid = bool(
        "exact_oracle_selected" in frame
        and pd.api.types.is_bool_dtype(frame["exact_oracle_selected"].dtype)
    )
    selected_object_ids = (
        object_ids[frame["exact_oracle_selected"].to_numpy(dtype=bool)]
        if ids_exact and exact_mask_valid
        else None
    )
    recorded_selected_ids = _exact_int64_vector(
        metadata.get("exact_oracle", {}).get("selected_object_ids")
    )
    exact_selection_identity_ok = bool(
        selected_object_ids is not None
        and np.array_equal(selected_object_ids, expected_exact_ids)
        and recorded_selected_ids is not None
        and np.array_equal(recorded_selected_ids, expected_exact_ids)
        and metadata.get("exact_oracle", {}).get("selection_manifest_sha256")
        == expected_exact_manifest
    )
    integer_names = (
        "stratum_count",
        "stratum_unique_count",
        "stratum_active_count",
        "pilot_draws",
        "pilot_cells",
        "pilot_source_proxy_t1",
        "pilot_source_proxy_t2",
        "pilot_source_prior",
        "proxy_t1_prior_fallback_cells",
        "proxy_t2_prior_fallback_cells",
        "pilot_draws_in_stratum",
        "production_reservoir_draws",
        "production_selected_draws",
        "production_selected_cycles",
        "production_cells",
        "production_source_proxy_t1",
        "production_source_proxy_t2",
        "production_source_prior",
        "production_draws_in_stratum",
        "allocation_min_cycles",
        "allocation_max_cycles",
        "allocation_floor_cells",
        "allocation_cap_cells",
    )
    counts = {name: _exact_int64_vector(frame.get(name)) for name in integer_names}
    integer_counts = all(item is not None for item in counts.values())
    support_ok = bool(
        integer_counts
        and np.all(counts["stratum_count"] == 16_384)
        and np.all(counts["stratum_unique_count"] == 16_384)
        and np.all(counts["stratum_active_count"] == 16_384)
    )
    pilot_ok = bool(
        integer_counts
        and np.all(counts["pilot_draws"] == 4_096)
        and np.all(counts["pilot_cells"] == 256)
        and np.all(counts["pilot_source_proxy_t1"] == 2_048)
        and np.all(counts["pilot_source_proxy_t2"] == 1_536)
        and np.all(counts["pilot_source_prior"] == 512)
    )
    production_ok = bool(
        integer_counts
        and np.all(counts["production_reservoir_draws"] == 131_072)
        and np.all(counts["production_selected_draws"] == 16_384)
        and np.all(counts["production_selected_cycles"] == 2_048)
        and np.all(counts["production_cells"] == 256)
        and np.all(counts["production_source_proxy_t1"] == 8_192)
        and np.all(counts["production_source_proxy_t2"] == 6_144)
        and np.all(counts["production_source_prior"] == 2_048)
    )
    proxy_fallback_ok = bool(
        integer_counts
        and np.all(counts["proxy_t1_prior_fallback_cells"] <= 256)
        and np.all(counts["proxy_t2_prior_fallback_cells"] <= 256)
    )
    exclusion_ok = bool(
        integer_counts
        and np.all(counts["pilot_draws_in_stratum"] == 0)
        and np.all(counts["production_draws_in_stratum"] == 0)
    )
    allocation_ok = bool(
        integer_counts
        and np.all(counts["allocation_min_cycles"] >= 2)
        and np.all(counts["allocation_max_cycles"] <= 64)
        and np.all(counts["production_selected_cycles"] == 2_048)
        and boolean_column_all("allocation_whole_cycle_prefixes", True)
        and boolean_column_all("allocation_zero_variance_fallback", False)
    )
    hash_columns_ok = bool(
        len(frame) == expected_objects
        and all(name in frame for name in _SPATIAL_MANIFEST_HASH_COLUMNS)
        and all(
            _is_sha256(value)
            for name in _SPATIAL_MANIFEST_HASH_COLUMNS
            for value in frame[name]
        )
    )
    support_exclusion_hashes_ok = bool(
        hash_columns_ok
        and frame["stratum_indices_sha256"].astype(str).tolist()
        == frame["pilot_exclusion_indices_sha256"].astype(str).tolist()
        and frame["stratum_indices_sha256"].astype(str).tolist()
        == frame["production_exclusion_indices_sha256"].astype(str).tolist()
    )
    try:
        recomputed_manifest = spatial_neyman_run_manifest_sha256(frame)
    except (KeyError, TypeError, ValueError):
        recomputed_manifest = None
    manifest_ok = bool(
        _is_sha256(spatial.get("run_manifest_sha256"))
        and spatial.get("run_manifest_sha256") == recomputed_manifest
    )
    partition_ok = bool(
        _is_sha256(expected_partition)
        and _is_sha256(expected_partition_cache)
        and _is_sha256(expected_partition_manifest)
        and spatial.get("partition_sha256") == expected_partition
        and spatial.get("partition_cache_sha256") == expected_partition_cache
        and spatial.get("partition_manifest_sha256") == expected_partition_manifest
        and spatial.get("partition_verified") is True
        and int(spatial.get("minimum_cell_active_count", -1)) > 16_384
        and int(spatial.get("maximum_cell_active_count", 262_145)) <= 262_144
    )
    rng_ok = bool(
        spatial.get("pilot_seed") == 1_008_704
        and spatial.get("production_seed") == 7_008_704
        and spatial.get("streams_independent") is True
        and hash_columns_ok
        and np.all(
            frame["pilot_rng_sha256"].astype(str).to_numpy()
            != frame["production_rng_sha256"].astype(str).to_numpy()
        )
        and np.all(
            frame["pilot_rng_sha256"].astype(str).to_numpy()
            != frame["production_reservoir_rng_sha256"].astype(str).to_numpy()
        )
    )
    source_cell_hashes_ok = bool(
        hash_columns_ok
        and pilot_ok
        and production_ok
        and allocation_ok
        and frame["pilot_normalizer_sha256"].astype(str).tolist()
        == frame["production_normalizer_sha256"].astype(str).tolist()
    )
    flow_counts_ok = bool(
        spatial.get("stratum_flow_evaluations") == 8_388_608
        and spatial.get("pilot_flow_evaluations") == 2_097_152
        and spatial.get("production_flow_evaluations") == 8_388_608
        and metadata.get("flow_evaluations") == 835_577_728
        and metadata.get("exact_oracle", {}).get("flow_evaluations") == 816_703_360
    )
    # The row-level flow arrays are intentionally not retained, but their
    # runtime hashes must be bound into the independently recomputed run
    # manifest and agree with the exact evaluation counters.  Do not accept
    # syntactically valid, otherwise-unbound digest strings as verification.
    flow_hashes_ok = bool(hash_columns_ok and manifest_ok and flow_counts_ok)
    calls = spatial.get("execution_call_provenance")
    core_provenance_ok = bool(isinstance(calls, list) and len(calls) == 64)
    if core_provenance_ok:
        for call_index, call in enumerate(calls):
            expected_start = 25_000 + 8 * call_index
            hashes = (
                call.get("partition_sha256"),
                call.get("pilot_record_sha256"),
                call.get("production_record_sha256"),
                call.get("allocation_sha256"),
                call.get("packed_draw_sha256"),
            )
            if not (
                call.get("object_start") == expected_start
                and call.get("object_stop") == expected_start + 8
                and call.get("objects") == 8
                and call.get("pilot_seed") == 1_008_704
                and call.get("production_seed") == 7_008_704
                and call.get("atom_chunk") == 262_144
                and call.get("object_chunk") == 8
                and call.get("partition_sha256") == expected_partition
                and all(_is_sha256(value) for value in hashes)
                and call.get("pilot_record_sha256")
                != call.get("production_record_sha256")
            ):
                core_provenance_ok = False
                break
    finite_positive = bool(
        len(frame) == expected_objects
        and np.isfinite(values("mass_stratum")).all()
        and np.isfinite(values("mass_complement_spatial_neyman")).all()
        and np.isfinite(values("mass_complement_spatial_neyman_standard_error")).all()
        and (values("mass_stratum") > 0.0).all()
        and (values("mass_complement_spatial_neyman") > 0.0).all()
        and (values("mass_complement_spatial_neyman_standard_error") > 0.0).all()
    )
    integrity_gates = {
        "protocol_identity_matched": _gate(protocol_matched, "is", True, protocol_matched),
        "absolute_object_ids_exact": _gate(ids_exact, "is", True, ids_exact),
        "exact_selection_identity_exact": _gate(
            exact_selection_identity_ok, "is", True, exact_selection_identity_ok
        ),
        "partition_hash_predeclared_and_verified": _gate(
            partition_ok, "is", True, partition_ok
        ),
        "integer_typed_count_columns": _gate(integer_counts, "is", True, integer_counts),
        "exact_stratum_unique_active_width_all_objects": _gate(
            support_ok, "is", True, support_ok
        ),
        "pilot_cell_and_fixed_source_counts": _gate(pilot_ok, "is", True, pilot_ok),
        "production_cell_and_fixed_source_counts": _gate(
            production_ok, "is", True, production_ok
        ),
        "proxy_prior_fallback_counts_recorded": _gate(
            proxy_fallback_ok, "is", True, proxy_fallback_ok
        ),
        "conditioned_streams_exclude_exact_stratum": _gate(
            exclusion_ok, "is", True, exclusion_ok
        ),
        "allocation_bounded_whole_cycle_prefixes": _gate(
            allocation_ok, "is", True, allocation_ok
        ),
        "independent_rng_streams": _gate(rng_ok, "is", True, rng_ok),
        "support_and_exclusion_hashes_match": _gate(
            support_exclusion_hashes_ok, "is", True, support_exclusion_hashes_ok
        ),
        "cell_and_source_hashes_verified": _gate(
            source_cell_hashes_ok, "is", True, source_cell_hashes_ok
        ),
        "flow_hashes_verified": _gate(flow_hashes_ok, "is", True, flow_hashes_ok),
        "run_manifest_recomputes": _gate(manifest_ok, "is", True, manifest_ok),
        "flow_evaluation_counts_exact": _gate(flow_counts_ok, "is", True, flow_counts_ok),
        "core_call_provenance_exact": _gate(
            core_provenance_ok, "is", True, core_provenance_ok
        ),
        "finite_positive_masses_and_standard_errors": _gate(
            finite_positive, "is", True, finite_positive
        ),
    }

    k_values = values("pareto_k_spatial_neyman_complement")
    finite_k = np.isfinite(k_values)
    below_0_5 = int(np.sum(finite_k & (k_values < 0.5)))
    below_0_7 = int(np.sum(finite_k & (k_values < 0.7)))
    at_least_1 = int(np.sum(finite_k & (k_values >= 1.0)))
    complement_mass = values("mass_complement_spatial_neyman")
    complement_se = values("mass_complement_spatial_neyman_standard_error")
    relative_error = np.divide(
        complement_se,
        np.abs(complement_mass),
        out=np.full(len(frame), np.nan),
        where=np.abs(complement_mass) > 0.0,
    )
    rel_finite = bool(len(relative_error) == expected_objects and np.isfinite(relative_error).all())
    rel_median = float(np.median(relative_error)) if rel_finite else float("nan")
    rel_p90 = _higher_percentile(relative_error, 90) if rel_finite else float("nan")
    reliability_gates = {
        "pareto_k_defined_all_objects": _gate(
            int((~finite_k).sum()), "undefined_count ==", 0, len(k_values) == 512 and finite_k.all()
        ),
        "pareto_k_below_0_5_count": _gate(
            below_0_5,
            ">=",
            thresholds["pareto_k_below_0_5_count"],
            below_0_5 >= thresholds["pareto_k_below_0_5_count"],
        ),
        "pareto_k_below_0_7_count": _gate(
            below_0_7,
            ">=",
            thresholds["pareto_k_below_0_7_count"],
            below_0_7 >= thresholds["pareto_k_below_0_7_count"],
        ),
        "pareto_k_at_least_1_count": _gate(
            at_least_1,
            "==",
            thresholds["pareto_k_at_least_1_count"],
            at_least_1 == thresholds["pareto_k_at_least_1_count"],
        ),
        "complement_relative_error_median": _gate(
            rel_median,
            "<=",
            thresholds["complement_relative_error_median"],
            np.isfinite(rel_median) and rel_median <= thresholds["complement_relative_error_median"],
        ),
        "complement_relative_error_p90_higher": _gate(
            rel_p90,
            "<=",
            thresholds["complement_relative_error_p90_higher"],
            np.isfinite(rel_p90) and rel_p90 <= thresholds["complement_relative_error_p90_higher"],
        ),
    }

    selected = frame.loc[frame["exact_oracle_selected"].astype(bool)]
    delta_log = np.abs(
        selected["delta_log_total_evidence_spatial_minus_exact"].to_numpy(dtype=np.float64)
    )
    exact_z = selected["mass_complement_spatial_minus_exact_z"].to_numpy(dtype=np.float64)
    crosscheck = selected["mass_stratum_exact_oracle_relative_error"].to_numpy(dtype=np.float64)
    difference = (
        selected["mass_complement_spatial_neyman"].to_numpy(dtype=np.float64)
        - selected["mass_complement_exact_oracle"].to_numpy(dtype=np.float64)
    )
    selected_se = selected["mass_complement_spatial_neyman_standard_error"].to_numpy(
        dtype=np.float64
    )
    exact_finite = bool(
        len(selected) == 64
        and np.isfinite(delta_log).all()
        and np.isfinite(exact_z).all()
        and np.isfinite(crosscheck).all()
        and np.isfinite(difference).all()
        and np.isfinite(selected_se).all()
        and (selected_se > 0.0).all()
    )
    median_delta = float(np.median(delta_log)) if exact_finite else float("nan")
    p90_delta = _higher_percentile(delta_log, 90) if exact_finite else float("nan")
    denominator = float(np.sqrt(np.sum(np.square(selected_se))))
    aggregate_z = (
        float(np.sum(difference) / denominator)
        if exact_finite and denominator > 0.0
        else float("nan")
    )
    z_rms = float(np.sqrt(np.mean(np.square(exact_z)))) if exact_finite else float("nan")
    within_1_96 = int(np.sum(np.isfinite(exact_z) & (np.abs(exact_z) <= 1.96)))
    crosscheck_max = float(np.max(crosscheck)) if exact_finite else float("nan")
    exact_gates = {
        "selected_exact_object_count": _gate(len(selected), "==", 64, len(selected) == 64),
        "stratum_relative_crosscheck_max": _gate(
            crosscheck_max,
            "<=",
            thresholds["stratum_relative_crosscheck_max"],
            np.isfinite(crosscheck_max)
            and crosscheck_max <= thresholds["stratum_relative_crosscheck_max"],
        ),
        "absolute_delta_log_total_evidence_median": _gate(
            median_delta,
            "<=",
            thresholds["absolute_delta_log_total_evidence_median"],
            np.isfinite(median_delta)
            and median_delta <= thresholds["absolute_delta_log_total_evidence_median"],
        ),
        "absolute_delta_log_total_evidence_p90_higher": _gate(
            p90_delta,
            "<=",
            thresholds["absolute_delta_log_total_evidence_p90_higher"],
            np.isfinite(p90_delta)
            and p90_delta <= thresholds["absolute_delta_log_total_evidence_p90_higher"],
        ),
        "absolute_signed_aggregate_z": _gate(
            abs(aggregate_z) if np.isfinite(aggregate_z) else float("nan"),
            "<=",
            thresholds["absolute_signed_aggregate_z"],
            np.isfinite(aggregate_z) and abs(aggregate_z) <= thresholds["absolute_signed_aggregate_z"],
        ),
        "exact_z_rms": _gate(
            z_rms,
            "in_closed_interval",
            thresholds["exact_z_rms"],
            np.isfinite(z_rms)
            and thresholds["exact_z_rms"][0] <= z_rms <= thresholds["exact_z_rms"][1],
        ),
        "exact_z_within_1_96_count": _gate(
            within_1_96,
            ">=",
            thresholds["exact_z_within_1_96_count"],
            within_1_96 >= thresholds["exact_z_within_1_96_count"],
        ),
    }

    total_mass = values("mass_stratum") + complement_mass
    total_relative_error_squared = np.square(
        np.divide(
            complement_se,
            np.abs(total_mass),
            out=np.full(len(frame), np.nan),
            where=np.abs(total_mass) > 0.0,
        )
    )
    finite_efficiency = bool(
        len(total_relative_error_squared) == 512 and np.isfinite(total_relative_error_squared).all()
    )
    phase_seconds = metadata.get("phase_seconds", {})
    method_seconds = float(spatial.get("proposal_initialization_seconds", np.nan))
    try:
        method_seconds += sum(
            float(phase_seconds[name]) for name in SPATIAL_NEYMAN_METHOD_TIME_COMPONENTS
        )
    except (KeyError, TypeError, ValueError):
        method_seconds = float("nan")
    efficiency_mean = (
        float(np.mean(total_relative_error_squared))
        * method_seconds
        / (
            baseline["mean_total_relative_error_squared"] * baseline["method_seconds"]
        )
        if finite_efficiency and np.isfinite(method_seconds) and method_seconds > 0.0
        else float("nan")
    )
    efficiency_p90 = (
        _higher_percentile(total_relative_error_squared, 90)
        * method_seconds
        / (
            baseline["total_relative_error_squared_p90_higher"] * baseline["method_seconds"]
        )
        if finite_efficiency and np.isfinite(method_seconds) and method_seconds > 0.0
        else float("nan")
    )
    efficiency_gates = {
        "mean_relative_variance_time_efficiency": _gate(
            efficiency_mean,
            "<=",
            thresholds["efficiency_mean"],
            np.isfinite(efficiency_mean) and efficiency_mean <= thresholds["efficiency_mean"],
        ),
        "p90_higher_relative_variance_time_efficiency": _gate(
            efficiency_p90,
            "<=",
            thresholds["efficiency_p90_higher"],
            np.isfinite(efficiency_p90) and efficiency_p90 <= thresholds["efficiency_p90_higher"],
        ),
    }
    families = {
        "integrity": integrity_gates,
        "tail_reliability": reliability_gates,
        "exact_calibration": exact_gates,
        "efficiency": efficiency_gates,
    }
    passed = all(item["passed"] for family in families.values() for item in family.values())
    return {
        "protocol": {
            "expected": expected_protocol,
            "observed": observed_protocol,
            "matched": protocol_matched,
        },
        "method_seconds": method_seconds,
        "proposal_initialization_seconds": spatial.get("proposal_initialization_seconds"),
        "complement_relative_error_median": rel_median,
        "complement_relative_error_p90_higher": rel_p90,
        "absolute_delta_log_total_evidence_median": median_delta,
        "absolute_delta_log_total_evidence_p90_higher": p90_delta,
        "stratum_relative_crosscheck_max": crosscheck_max,
        "signed_aggregate_z": aggregate_z,
        "exact_z_rms": z_rms,
        "exact_z_within_1_96_count": within_1_96,
        "efficiency_mean": efficiency_mean,
        "efficiency_p90_higher": efficiency_p90,
        "gates": {
            "predeclared": protocol_matched,
            "thresholds_predeclared": True,
            **families,
            "passed": bool(passed),
            "policy": "all integrity, tail, exact-calibration, and fair-efficiency gates must pass",
        },
    }


def _partition_sha256(partition: Any) -> str:
    for name in ("sha256", "partition_sha256"):
        value = getattr(partition, name, None)
        if callable(value):
            value = value()
        if _is_sha256(value):
            return value
    raise RuntimeError("spatial partition does not expose a lowercase SHA-256")


def _record_array_hashes(prefix: str, indices, probability, cell_id, source_id) -> dict[str, str]:
    return {
        f"{prefix}_rng_sha256": _canonical_array_sha256(
            "SBSI/spatial-rng-indices/v1", indices, ">i8"
        ),
        f"{prefix}_probability_sha256": _canonical_array_sha256(
            f"SBSI/{prefix}-probability/v1", probability, ">f8"
        ),
        f"{prefix}_cell_ids_sha256": _canonical_array_sha256(
            f"SBSI/{prefix}-cell-id/v1", cell_id, ">i2"
        ),
        f"{prefix}_source_ids_sha256": _canonical_array_sha256(
            f"SBSI/{prefix}-source-id/v1", source_id, ">i2"
        ),
    }


def _row_arrays(record: Any, row: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract the four canonical arrays from a core spatial record."""

    indices = np.asarray(record.indices[row])
    probability = np.asarray(record.probability[row])
    cell_id = np.asarray(record.cell_id)
    component_id = np.asarray(record.component_id)
    if cell_id.ndim == 2:
        cell_id = cell_id[row]
    if component_id.ndim == 2:
        component_id = component_id[row]
    if not (
        indices.ndim == probability.ndim == cell_id.ndim == component_id.ndim == 1
        and indices.shape == probability.shape == cell_id.shape == component_id.shape
    ):
        raise RuntimeError("core spatial record arrays are not aligned")
    return indices, probability, cell_id, component_id


def _partition_cell_lookup(partition: Any, proposal: Any, atom_ids: np.ndarray) -> np.ndarray:
    """Independently map atom ids to cells using the immutable partition."""

    atom_ids = np.asarray(atom_ids, dtype=np.int64)
    for name in ("atom_cell_id", "cell_id_by_atom", "atom_cells"):
        mapping = getattr(partition, name, None)
        if mapping is not None:
            mapping = np.asarray(mapping, dtype=np.int64)
            if len(mapping) == len(proposal.prior_weights):
                return mapping[atom_ids]
    active_cells = getattr(partition, "active_cell_id", None)
    if active_cells is None:
        active_cells = getattr(partition, "cell_id", None)
    if active_cells is not None:
        active = np.asarray(proposal.active_indices, dtype=np.int64)
        positions = np.searchsorted(active, atom_ids)
        if (positions >= len(active)).any() or not np.array_equal(active[positions], atom_ids):
            raise RuntimeError("spatial reservoir contains an inactive atom")
        return np.asarray(active_cells, dtype=np.int64)[positions]
    raise RuntimeError("spatial partition does not expose an atom-to-cell map")


def diagnose_spatial_neyman_complement(
    likelihood,
    mock,
    proposal,
    *,
    partition,
    partition_cache_sha256: str,
    partition_manifest_sha256: str,
    partition_load_seconds: float,
    proposal_initialization_seconds: float,
    center: tuple[float, float] = (0.013610377120038044, 0.002740148300964851),
    observation_start: int = 0,
    observation_stop: Optional[int] = None,
    object_id_offset: int = 25_000,
    object_chunk: int = 8,
    atom_chunk: int = 4_096,
    spatial_atom_chunk: int = 262_144,
    spatial_object_chunk: int = 8,
    candidate_backend: str = "torch",
    n_candidates: int = 16_384,
    prefilter_candidates: int = 131_072,
    proposal_seed: int = 8_701,
    pilot_seed: int = 1_008_704,
    production_seed: int = 7_008_704,
    confirmation_mock_sha256: Optional[dict[str, str]] = None,
) -> SpatialNeymanDiagnostic:
    """Run the strict centre-only spatial pilot/Neyman development gate."""

    # Import only when the diagnostic is invoked so ordinary inference remains
    # independent of this development-only sampler extension.
    from .catalogue_sampling import (
        allocate_spatial_neyman_cycles,
        pack_spatial_production_prefix,
        reduce_spatial_mis,
        spatial_pilot_variance,
    )

    frozen_partition_identities = {
        name: SPATIAL_NEYMAN_PROTOCOL[name]
        for name in (
            "partition_sha256",
            "partition_cache_sha256",
            "partition_manifest_sha256",
        )
    }
    if not all(_is_sha256(value) for value in frozen_partition_identities.values()):
        raise ValueError(
            "science diagnostic is disabled until all outcome-blind partition identities are frozen"
        )
    observed_partition_identities = {
        "partition_sha256": _partition_sha256(partition),
        "partition_cache_sha256": partition_cache_sha256,
        "partition_manifest_sha256": partition_manifest_sha256,
    }
    if observed_partition_identities != frozen_partition_identities:
        raise ValueError("science diagnostic partition identities do not match the frozen protocol")
    if confirmation_mock_sha256 != LEARNED_RETRIEVAL_CENTER_MOCK_SHA256:
        raise ValueError("spatial-Neyman diagnostic requires the frozen mock hashes")
    if getattr(likelihood, "selection", None) is not None:
        raise ValueError("spatial-Neyman diagnostic requires measured selection to be disabled")
    if candidate_backend != "torch":
        raise ValueError("spatial-Neyman frozen protocol requires Torch candidate search")
    if (
        center != tuple(SPATIAL_NEYMAN_PROTOCOL["center"])
        or observation_start != 0
        or (observation_stop is not None and observation_stop != 512)
        or object_id_offset != 25_000
        or object_chunk != 8
        or atom_chunk != 4_096
        or spatial_atom_chunk != 262_144
        or spatial_object_chunk != 8
        or n_candidates != 16_384
        or prefilter_candidates != 131_072
        or proposal_seed != 8_701
        or pilot_seed != 1_008_704
        or production_seed != 7_008_704
    ):
        raise ValueError("spatial-Neyman science runner only accepts its frozen protocol")
    if not np.isfinite(proposal_initialization_seconds) or proposal_initialization_seconds < 0.0:
        raise ValueError("proposal initialization time must be finite and nonnegative")
    if not np.isfinite(partition_load_seconds) or partition_load_seconds < 0.0:
        raise ValueError("partition load time must be finite and nonnegative")
    measurements = mock.measurements
    stop = len(measurements) if observation_stop is None else int(observation_stop)
    if len(measurements) != 512 or stop != 512:
        raise ValueError("mock must be the already-sliced frozen 512-object window")

    started = time.perf_counter()
    phase_seconds = {name: 0.0 for name in SPATIAL_NEYMAN_METHOD_TIME_COMPONENTS}
    phase_seconds["exact_oracle_flow"] = 0.0
    tick = time.perf_counter()
    partition.verify(proposal)
    phase_seconds["partition_load_or_build"] = float(partition_load_seconds)
    phase_seconds["partition_reverification"] = time.perf_counter() - tick
    if (
        partition.recipe_sha256 != SPATIAL_NEYMAN_PROTOCOL["partition_recipe_sha256"]
        or len(partition.cell_id) != SPATIAL_NEYMAN_PROTOCOL["active_atoms"]
        or np.asarray(partition.counts).shape != (SPATIAL_NEYMAN_PROTOCOL["n_cells"],)
    ):
        raise RuntimeError("spatial partition shape or recipe differs from the frozen protocol")
    minimum_cell_active_count = int(np.min(partition.counts))
    maximum_cell_active_count = int(np.max(partition.counts))
    if minimum_cell_active_count <= n_candidates:
        raise RuntimeError("every spatial cell must contain more atoms than the exact support width")
    if maximum_cell_active_count > spatial_atom_chunk:
        raise RuntimeError("a spatial cell exceeds the frozen atom-mapping chunk")

    absolute_start = object_id_offset + observation_start
    absolute_stop = object_id_offset + stop
    selected_ids, selection_manifest = exact_oracle_object_ids(absolute_start, absolute_stop)
    if selection_manifest != SPATIAL_NEYMAN_PROTOCOL["exact_selection_manifest_sha256"]:
        raise RuntimeError("frozen exact-oracle selection manifest changed")
    selected_id_set = set(map(int, selected_ids))
    rows: list[dict[str, Any]] = []
    exact_records: dict[int, tuple[float, float]] = {}
    exact_flow_evaluations = 0
    exact_identity = None
    stratum_flow_evaluations = 0
    pilot_flow_evaluations = 0
    production_flow_evaluations = 0
    execution_call_provenance: list[dict[str, Any]] = []

    for batch_start in range(observation_start, stop, object_chunk):
        batch_stop = min(batch_start + object_chunk, stop)
        observed = measurements.iloc[batch_start:batch_stop].reset_index(drop=True)
        absolute_ids = object_id_offset + np.arange(batch_start, batch_stop, dtype=np.int64)
        observed_targets = likelihood.observed_target_tensor(observed)

        tick = time.perf_counter()
        candidates = proposal.candidates(
            observed,
            n_candidates=n_candidates,
            prefilter_candidates=prefilter_candidates,
            torch_device=likelihood.flow_model.device,
        )
        if candidates.indices.shape != (len(observed), n_candidates):
            raise RuntimeError("candidate query did not return the exact support width")

        active = np.asarray(proposal.active_indices, dtype=np.int64)
        active_positions = np.searchsorted(active, candidates.indices)
        active_valid = active_positions < len(active)
        if active_valid.any():
            active_valid[active_valid] &= active[active_positions[active_valid]] == candidates.indices[
                active_valid
            ]
        if not active_valid.all() or any(
            len(np.unique(row)) != n_candidates for row in candidates.indices
        ):
            raise RuntimeError("exact stratum is not unique active support")
        phase_seconds["candidate_query"] += time.perf_counter() - tick

        tick = time.perf_counter()
        stratum_log_c = (
            likelihood.log_importance_weights_tensor(
                observed,
                center[0],
                center[1],
                atom_indices=candidates.indices,
                proposal_probability=np.ones_like(candidates.indices, dtype=np.float64),
                observed_targets=observed_targets,
                object_chunk=len(observed),
                atom_chunk=atom_chunk,
            )
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        phase_seconds["stratum_exact_flow"] += time.perf_counter() - tick
        stratum_flow_evaluations += int(candidates.indices.size)

        tick = time.perf_counter()
        reservoirs = proposal.draw_spatial_mis_reservoirs(
            candidates,
            observed,
            pilot_seed=pilot_seed,
            production_seed=production_seed,
            partition=partition,
            object_ids=absolute_ids,
            device=likelihood.flow_model.device,
            atom_chunk=spatial_atom_chunk,
            object_chunk=spatial_object_chunk,
        )
        pilot = getattr(reservoirs, "pilot", None)
        production = getattr(reservoirs, "production", None)
        if pilot is None or production is None:
            try:
                pilot, production = reservoirs
            except (TypeError, ValueError) as error:
                raise RuntimeError("spatial sampler did not return pilot and production reservoirs") from error
        for reservoir, domain, seed, cycles in (
            (pilot, "pilot", pilot_seed, 2),
            (production, "production", production_seed, 64),
        ):
            if (
                reservoir.domain != domain
                or reservoir.seed != seed
                or reservoir.cycles_per_cell != cycles
                or reservoir.atom_chunk != spatial_atom_chunk
                or reservoir.object_chunk != spatial_object_chunk
                or reservoir.partition_sha256 != partition.partition_sha256
                or reservoir.partition_assignment_sha256 != partition.assignment_sha256
                or reservoir.partition_counts_sha256 != partition.counts_sha256
                or reservoir.partition_detected_mass_sha256
                != partition.detected_mass_sha256
                or reservoir.partition_prior_mass_sha256 != partition.prior_mass_sha256
                or reservoir.partition_recipe_sha256 != partition.recipe_sha256
                or not np.array_equal(reservoir.object_ids, absolute_ids)
            ):
                raise RuntimeError(f"{domain} reservoir provenance does not match the run")
        if (
            pilot.record_sha256 == production.record_sha256
            or not np.array_equal(
                pilot.component_log_normalizer,
                production.component_log_normalizer,
            )
            or not np.array_equal(pilot.proxy_fallback, production.proxy_fallback)
        ):
            raise RuntimeError("spatial reservoirs are not independent maps of one cell law")
        # The core constructs both output banks while mapping the two random
        # domains through one normalized cell law.  It does not expose a
        # separable timer, so charge the complete call plus provenance checks
        # to construction and retain an explicit zero mapping subphase.
        phase_seconds["reservoir_construction"] += time.perf_counter() - tick

        tick = time.perf_counter()
        pilot_log_ratio = (
            likelihood.log_importance_weights_tensor(
                observed,
                center[0],
                center[1],
                atom_indices=pilot.indices,
                proposal_probability=pilot.probability,
                observed_targets=observed_targets,
                object_chunk=len(observed),
                atom_chunk=atom_chunk,
            )
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        phase_seconds["pilot_flow"] += time.perf_counter() - tick
        pilot_flow_evaluations += int(pilot.indices.size)
        tick = time.perf_counter()
        pilot_variance = _stable_spatial_pilot_variance(
            pilot_log_ratio, pilot, spatial_pilot_variance
        )
        allocation = allocate_spatial_neyman_cycles(pilot_variance)
        selected_production = pack_spatial_production_prefix(production, allocation)
        phase_seconds["allocation"] += time.perf_counter() - tick

        tick = time.perf_counter()
        production_log_ratio = (
            likelihood.log_importance_weights_tensor(
                observed,
                center[0],
                center[1],
                atom_indices=selected_production.indices,
                proposal_probability=selected_production.probability,
                observed_targets=observed_targets,
                object_chunk=len(observed),
                atom_chunk=atom_chunk,
            )
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        phase_seconds["selected_production_flow"] += time.perf_counter() - tick
        production_flow_evaluations += int(selected_production.indices.size)
        tick = time.perf_counter()
        production_scaled = np.empty_like(production_log_ratio)
        production_shift = np.empty(len(observed), dtype=np.float64)
        for row in range(len(observed)):
            production_scaled[row], production_shift[row] = _rescale_log_values(
                production_log_ratio[row], groups=selected_production.cell_id[row]
            )
        core_reduction = reduce_spatial_mis(production_scaled, selected_production)
        if (
            selected_production.partition_sha256 != partition.partition_sha256
            or selected_production.reservoir_record_sha256
            != production.record_sha256
            or selected_production.allocation_sha256 != allocation.allocation_sha256
            or not np.array_equal(selected_production.object_ids, absolute_ids)
            or not np.array_equal(selected_production.cycles, allocation.cycles)
        ):
            raise RuntimeError("packed spatial draw provenance does not match its inputs")
        execution_call_provenance.append(
            {
                "object_start": int(absolute_ids[0]),
                "object_stop": int(absolute_ids[-1]) + 1,
                "objects": int(len(absolute_ids)),
                "pilot_seed": int(pilot.seed),
                "production_seed": int(production.seed),
                "atom_chunk": int(production.atom_chunk),
                "object_chunk": int(production.object_chunk),
                "partition_sha256": partition.partition_sha256,
                "pilot_record_sha256": pilot.record_sha256,
                "production_record_sha256": production.record_sha256,
                "allocation_sha256": allocation.allocation_sha256,
                "packed_draw_sha256": selected_production.record_sha256,
            }
        )
        phase_seconds["reduction"] += time.perf_counter() - tick

        selected_positions = np.fromiter(
            (
                row
                for row, object_id in enumerate(absolute_ids)
                if int(object_id) in selected_id_set
            ),
            dtype=np.int64,
        )
        if len(selected_positions):
            tick = time.perf_counter()
            exact = _stream_exact_oracle_mass(
                likelihood,
                observed.iloc[selected_positions].reset_index(drop=True),
                proposal,
                candidates.indices[selected_positions],
                center=center,
                observed_targets=observed_targets[selected_positions.tolist()],
            )
            phase_seconds["exact_oracle_flow"] += time.perf_counter() - tick
            exact_flow_evaluations += exact.flow_evaluations
            identity = (
                exact.active_atoms,
                exact.atoms_visited,
                exact.atom_blocks,
                exact.active_indices_sha256,
            )
            if exact_identity is None:
                exact_identity = identity
            elif exact_identity != identity:
                raise RuntimeError("exact-oracle active support identity changed")
            for selected_row, position in enumerate(selected_positions):
                exact_records[int(absolute_ids[position])] = (
                    float(exact.log_stratum[selected_row]),
                    float(exact.log_complement[selected_row]),
                )

        tick = time.perf_counter()
        allocation_cycles = np.asarray(
            getattr(allocation, "cycles", getattr(allocation, "cell_cycles", None)),
            dtype=np.int64,
        )
        if allocation_cycles.shape != (len(observed), 256):
            raise RuntimeError("spatial allocation must contain 256 cell-cycle counts per object")
        support_hashes = _candidate_exclusion_manifest_sha256(candidates.indices)
        for row, object_id in enumerate(absolute_ids):
            pilot_arrays = _row_arrays(pilot, row)
            reservoir_arrays = _row_arrays(production, row)
            selected_arrays = _row_arrays(selected_production, row)
            pilot_indices, pilot_probability, pilot_cells, pilot_sources = pilot_arrays
            reservoir_indices, reservoir_probability, reservoir_cells, reservoir_sources = (
                reservoir_arrays
            )
            selected_indices, selected_probability, selected_cells, selected_sources = (
                selected_arrays
            )
            if not (
                np.array_equal(
                    _partition_cell_lookup(partition, proposal, pilot_indices), pilot_cells
                )
                and np.array_equal(
                    _partition_cell_lookup(partition, proposal, reservoir_indices), reservoir_cells
                )
                and np.array_equal(
                    _partition_cell_lookup(partition, proposal, selected_indices), selected_cells
                )
            ):
                raise RuntimeError("spatial reservoir cell ids do not match the partition")
            pilot_member = np.isin(pilot_indices, candidates.indices[row])
            production_member = np.isin(selected_indices, candidates.indices[row])
            if pilot_member.any() or production_member.any():
                raise RuntimeError("conditioned spatial stream leaked into exact support")
            if (
                pilot.object_excluded_candidate_indices_sha256[row] != support_hashes[row]
                or production.object_excluded_candidate_indices_sha256[row]
                != support_hashes[row]
            ):
                raise RuntimeError("spatial sampler exclusion hash does not match exact support")
            if not np.array_equal(
                pilot.component_log_normalizer[row],
                production.component_log_normalizer[row],
            ):
                raise RuntimeError("pilot and production did not reuse one normalized cell law")

            scaled = production_scaled[row]
            shift = float(production_shift[row])
            reduction = _reduce_spatial_mis_arrays(
                scaled, selected_cells, selected_sources, cells=256, sources=3
            )
            if not (
                np.isclose(
                    reduction.mass,
                    core_reduction.estimate[row],
                    rtol=1.0e-13,
                    atol=0.0,
                )
                and np.isclose(
                    reduction.standard_error,
                    core_reduction.standard_error[row],
                    rtol=1.0e-13,
                    atol=0.0,
                )
            ):
                raise RuntimeError("independent spatial reduction disagrees with sampler core")
            factor = float(np.exp(shift))
            if not np.isfinite(factor) or factor <= 0.0:
                raise RuntimeError("spatial importance scale is outside finite float64 range")
            mass_complement = float(reduction.mass * factor)
            complement_se = float(reduction.standard_error * factor)
            if not np.isfinite(mass_complement) or not np.isfinite(complement_se):
                raise RuntimeError("spatial mass reduction is outside finite float64 range")
            mass_stratum, log_mass_stratum = _scaled_sum(stratum_log_c[row])
            cycles = allocation_cycles[row]
            whole_cycles = bool(
                np.all(reduction.cell_draw_counts == 8 * cycles)
                and np.all(
                    reduction.cell_source_draw_counts
                    == cycles[:, None] * np.asarray([4, 3, 1], dtype=np.int64)
                )
            )
            flow_hashes = {
                "stratum_log_target_sha256": _canonical_array_sha256(
                    "SBSI/spatial-stratum-log-target/v1", stratum_log_c[row], ">f8"
                ),
                "pilot_log_ratio_sha256": _canonical_array_sha256(
                    "SBSI/spatial-pilot-log-ratio/v1", pilot_log_ratio[row], ">f8"
                ),
                "selected_production_log_ratio_sha256": _canonical_array_sha256(
                    "SBSI/spatial-production-log-ratio/v1", production_log_ratio[row], ">f8"
                ),
            }
            record = {
                "object_index": int(object_id),
                "stratum_count": n_candidates,
                "stratum_unique_count": int(len(np.unique(candidates.indices[row]))),
                "stratum_active_count": n_candidates,
                "stratum_indices_sha256": support_hashes[row],
                **flow_hashes,
                **_record_array_hashes("pilot", *pilot_arrays),
                "pilot_normalizer_sha256": _canonical_array_sha256(
                    "SBSI/spatial-cell-normalizer/v1",
                    pilot.component_log_normalizer[row],
                    ">f8",
                ),
                **{
                    key.replace("production_rng", "production_reservoir_rng")
                    .replace("production_probability", "production_reservoir_probability")
                    .replace("production_cell", "production_reservoir_cell")
                    .replace("production_source", "production_reservoir_source"): value
                    for key, value in _record_array_hashes(
                        "production", *reservoir_arrays
                    ).items()
                },
                "production_normalizer_sha256": _canonical_array_sha256(
                    "SBSI/spatial-cell-normalizer/v1",
                    production.component_log_normalizer[row],
                    ">f8",
                ),
                **{
                    key.replace("selected_rng", "production_rng")
                    .replace("selected_probability", "selected_production_probability")
                    .replace("selected_cell", "selected_production_cell")
                    .replace("selected_source", "selected_production_source"): value
                    for key, value in _record_array_hashes("selected", *selected_arrays).items()
                },
                "pilot_exclusion_indices_sha256": support_hashes[row],
                "production_exclusion_indices_sha256": support_hashes[row],
                "allocation_sha256": _canonical_array_sha256(
                    "SBSI/spatial-neyman-allocation/v1", cycles, ">i2"
                ),
                "allocation_whole_cycle_prefixes": whole_cycles,
                "allocation_zero_variance_fallback": bool(
                    allocation.all_zero_fallback[row]
                ),
                "pilot_draws": int(len(pilot_indices)),
                "pilot_cells": int(len(np.unique(pilot_cells))),
                "pilot_source_proxy_t1": int(np.sum(pilot_sources == 0)),
                "pilot_source_proxy_t2": int(np.sum(pilot_sources == 1)),
                "pilot_source_prior": int(np.sum(pilot_sources == 2)),
                "proxy_t1_prior_fallback_cells": int(
                    np.sum(pilot.proxy_fallback[row, :, 0])
                ),
                "proxy_t2_prior_fallback_cells": int(
                    np.sum(pilot.proxy_fallback[row, :, 1])
                ),
                "pilot_draws_in_stratum": int(pilot_member.sum()),
                "production_reservoir_draws": int(len(reservoir_indices)),
                "production_selected_draws": int(len(selected_indices)),
                "production_selected_cycles": int(cycles.sum()),
                "production_cells": int(len(np.unique(selected_cells))),
                "production_source_proxy_t1": int(np.sum(selected_sources == 0)),
                "production_source_proxy_t2": int(np.sum(selected_sources == 1)),
                "production_source_prior": int(np.sum(selected_sources == 2)),
                "production_draws_in_stratum": int(production_member.sum()),
                "allocation_min_cycles": int(cycles.min()),
                "allocation_max_cycles": int(cycles.max()),
                "allocation_floor_cells": int(np.sum(cycles == 2)),
                "allocation_cap_cells": int(np.sum(cycles == 64)),
                "log_mass_stratum": log_mass_stratum,
                "mass_stratum": mass_stratum,
                "mass_complement_spatial_neyman": mass_complement,
                "mass_complement_spatial_neyman_standard_error": complement_se,
                "pareto_k_spatial_neyman_complement": reduction.pareto_k,
                "maximum_effective_weight_spatial_neyman": (
                    reduction.maximum_effective_weight * factor
                ),
                "exact_oracle_selected": int(object_id) in selected_id_set,
            }
            if int(object_id) in exact_records:
                exact_log_stratum, exact_log_complement = exact_records[int(object_id)]
                exact_log_total = float(np.logaddexp(exact_log_stratum, exact_log_complement))
                exact_stratum = float(np.exp(exact_log_stratum))
                exact_complement = float(np.exp(exact_log_complement))
                exact_total = float(np.exp(exact_log_total))
                total = mass_stratum + mass_complement
                difference = mass_complement - exact_complement
                record.update(
                    {
                        "mass_stratum_exact_oracle": exact_stratum,
                        "mass_stratum_exact_oracle_relative_error": (
                            abs(mass_stratum - exact_stratum) / abs(exact_stratum)
                            if exact_stratum != 0.0
                            else float("nan")
                        ),
                        "mass_complement_exact_oracle": exact_complement,
                        "mass_total_exact_oracle": exact_total,
                        "mass_complement_spatial_minus_exact_z": (
                            difference / complement_se if complement_se > 0.0 else float("nan")
                        ),
                        "delta_log_total_evidence_spatial_minus_exact": (
                            float(np.log(total) - exact_log_total)
                            if total > 0.0
                            else float("nan")
                        ),
                    }
                )
            rows.append(record)
        phase_seconds["reduction"] += time.perf_counter() - tick

    tick = time.perf_counter()
    frame = pd.DataFrame(rows).sort_values("object_index").reset_index(drop=True)
    if len(exact_records) != 64 or exact_identity is None:
        raise RuntimeError("exact-oracle subset was not completely evaluated")
    active_atoms, atoms_visited, atom_blocks, active_hash = exact_identity
    flow_evaluations = (
        stratum_flow_evaluations
        + pilot_flow_evaluations
        + production_flow_evaluations
        + exact_flow_evaluations
    )
    spatial_metadata = {
        "label": SPATIAL_NEYMAN_CENTER_LABEL,
        "initial_strategy": "mean_observed_shape",
        "expected_partition_sha256": frozen_partition_identities["partition_sha256"],
        "expected_partition_cache_sha256": frozen_partition_identities[
            "partition_cache_sha256"
        ],
        "expected_partition_manifest_sha256": frozen_partition_identities[
            "partition_manifest_sha256"
        ],
        "partition_sha256": observed_partition_identities["partition_sha256"],
        "partition_cache_sha256": partition_cache_sha256,
        "partition_manifest_sha256": partition_manifest_sha256,
        "partition_verified": True,
        "partition_recipe_sha256": partition.recipe_sha256,
        "minimum_cell_active_count": minimum_cell_active_count,
        "maximum_cell_active_count": maximum_cell_active_count,
        "n_cells": 256,
        "spatial_atom_chunk": spatial_atom_chunk,
        "spatial_object_chunk": spatial_object_chunk,
        "pilot_cycles_per_cell": 2,
        "pilot_draws": 4_096,
        "n_global_draws": 16_384,
        "proposal_seed": proposal_seed,
        "pilot_seed": pilot_seed,
        "production_reservoir_cycles_per_cell": 64,
        "production_seed": production_seed,
        "allocation_min_cycles": 2,
        "allocation_max_cycles": 64,
        "allocation_total_cycles": 2_048,
        "tail_proposal_recipe": asdict(TAIL_PROPOSAL_RECIPE),
        "streams_independent": True,
        "pilot_draws_enter_estimate": False,
        "production_uses_whole_cycle_prefixes": True,
        "proposal_initialization_seconds": float(proposal_initialization_seconds),
        "mock_input_sha256": dict(confirmation_mock_sha256),
        "stratum_flow_evaluations": int(stratum_flow_evaluations),
        "pilot_flow_evaluations": int(pilot_flow_evaluations),
        "production_flow_evaluations": int(production_flow_evaluations),
        "proxy_prior_fallback_cells_by_component": {
            "proxy_t1": int(frame["proxy_t1_prior_fallback_cells"].sum()),
            "proxy_t2": int(frame["proxy_t2_prior_fallback_cells"].sum()),
        },
        "objects_with_proxy_prior_fallback": int(
            (
                (frame["proxy_t1_prior_fallback_cells"] > 0)
                | (frame["proxy_t2_prior_fallback_cells"] > 0)
            ).sum()
        ),
        "run_manifest_sha256": spatial_neyman_run_manifest_sha256(frame),
        "execution_call_provenance": execution_call_provenance,
    }
    exact_metadata = {
        "selection_count": 64,
        "selected_object_ids": selected_ids.tolist(),
        "selection_manifest_sha256": selection_manifest,
        "atom_chunk": EXACT_ORACLE_ATOM_CHUNK,
        "object_chunk": EXACT_ORACLE_OBJECT_CHUNK,
        "active_atoms": int(active_atoms),
        "active_atoms_visited_per_selected_object": int(atoms_visited),
        "atom_blocks_per_selected_object": int(atom_blocks),
        "active_indices_sha256": active_hash,
        "flow_evaluations": int(exact_flow_evaluations),
    }
    phase_seconds["reduction"] += time.perf_counter() - tick
    metadata = {
        "center": [float(center[0]), float(center[1])],
        "n_candidates": n_candidates,
        "prefilter_candidates": prefilter_candidates,
        "candidate_backend": candidate_backend,
        "object_chunk": object_chunk,
        "flow_atom_chunk": atom_chunk,
        "observation_start": absolute_start,
        "observation_stop": absolute_stop,
        "local_observation_start": observation_start,
        "local_observation_stop": stop,
        "object_id_offset": object_id_offset,
        "flow_evaluations": int(flow_evaluations),
        "elapsed_seconds": float(time.perf_counter() - started),
        "phase_seconds": {name: float(value) for name, value in phase_seconds.items()},
        "spatial_neyman_center": spatial_metadata,
        "exact_oracle": exact_metadata,
    }
    return SpatialNeymanDiagnostic(frame, metadata)


# Stable public names used by the strict CLI branch.  Keep the shorter names as
# compatibility aliases for early development scripts.
diagnose_spatial_neyman = diagnose_spatial_neyman_complement
spatial_neyman_summary = spatial_neyman_center_summary


__all__ = [
    "SPATIAL_NEYMAN_BASE_PROTOCOL",
    "SPATIAL_NEYMAN_PROTOCOL",
    "SPATIAL_NEYMAN_CENTER_GATE_THRESHOLDS",
    "SPATIAL_NEYMAN_CENTER_LABEL",
    "SPATIAL_NEYMAN_METHOD_TIME_COMPONENTS",
    "SpatialNeymanDiagnostic",
    "diagnose_spatial_neyman",
    "diagnose_spatial_neyman_complement",
    "spatial_neyman_center_protocol",
    "spatial_neyman_center_summary",
    "spatial_neyman_summary",
    "spatial_neyman_run_manifest_sha256",
]
