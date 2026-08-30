"""Frozen centre gate for H256-U8-PS2 spatial complement sampling.

This development-only diagnostic reuses the reviewed 256-cell partition and
the frozen conditioned-DM3 proposal, but it has no pilot, reservoir, or
adaptive allocation.  Each cell receives eight whole DM3 cycles.  Within
source ``l``, its ``B_l = (16, 12, 4)`` draws-per-cycle slots are treated as
equal-mass integer-CDF microstrata and sampled twice (PS2).

The sampler core owns construction of the integer CDFs.  This module owns the
frozen science protocol, the numerically stable pair estimator, provenance,
and gates.  Nothing here changes an inference default.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from hashlib import sha256
import struct
import time
from typing import Any
from typing import Optional

import numpy as np
import pandas as pd

from .catalogue_sampling import (
    SPATIAL_CELL_COUNT,
    SPATIAL_PS_CELLS_PER_SLAB,
    SPATIAL_PS_DRAWS_PER_CELL,
    SPATIAL_PS_HASH_CONTRACT,
    SPATIAL_PS_MICROSTRATA,
    SPATIAL_PS_ORDER_CONTRACT,
    SPATIAL_PS_QUANTIZER_CONTRACT,
    SPATIAL_PS_REPLICATES,
    SPATIAL_PS_RNG_CONTRACT,
    SPATIAL_PS_RNG_DOMAIN_WORD,
    SPATIAL_PS_RNG_VERSION_WORD,
    SPATIAL_PS_SEED,
    SPATIAL_PS_SOURCE_COUNTS,
    SPATIAL_PS_TICKS_PER_MICROSTRATUM,
    SPATIAL_PS_VERSION,
    SpatialProbabilityStratifiedMISDraw,
    TAIL_PROPOSAL_RECIPE,
    draw_spatial_probability_stratified_mis,
    pareto_tail_index,
    reduce_spatial_probability_stratified_mis,
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
from .spatial_neyman_diagnostic import (
    SPATIAL_NEYMAN_CENTER_GATE_THRESHOLDS,
    SPATIAL_NEYMAN_PROTOCOL,
    _partition_cell_lookup,
    _partition_sha256,
    _scaled_sum,
)


SPATIAL_H256_U8_PS2_CENTER_LABEL = (
    "SBSI/complement-spatial-h256-u8-ps2-center-development/v1"
)
SPATIAL_H256_U8_PS2_CENTER_GATE_THRESHOLDS = deepcopy(
    SPATIAL_NEYMAN_CENTER_GATE_THRESHOLDS
)

SPATIAL_H256_U8_PS2_CDF_TICK_BITS = 48
SPATIAL_H256_U8_PS2_CDF_TICKS = 1 << SPATIAL_H256_U8_PS2_CDF_TICK_BITS
SPATIAL_H256_U8_PS2_SOURCE_MICROSTRATA = (16, 12, 4)
SPATIAL_H256_U8_PS2_SOURCE_DRAWS = tuple(
    2 * count for count in SPATIAL_H256_U8_PS2_SOURCE_MICROSTRATA
)
SPATIAL_H256_U8_PS2_BALANCE_TICKS = (
    sum(SPATIAL_H256_U8_PS2_SOURCE_MICROSTRATA)
    * SPATIAL_H256_U8_PS2_CDF_TICKS
)

if SPATIAL_H256_U8_PS2_BALANCE_TICKS != 1 << 53:
    raise RuntimeError("the frozen H256-U8-PS2 balance denominator changed")
if (
    SPATIAL_CELL_COUNT != 256
    or SPATIAL_PS_TICKS_PER_MICROSTRATUM != SPATIAL_H256_U8_PS2_CDF_TICKS
    or tuple(SPATIAL_PS_MICROSTRATA)
    != SPATIAL_H256_U8_PS2_SOURCE_MICROSTRATA
    or SPATIAL_PS_REPLICATES != 2
    or tuple(SPATIAL_PS_SOURCE_COUNTS) != SPATIAL_H256_U8_PS2_SOURCE_DRAWS
    or SPATIAL_PS_DRAWS_PER_CELL != 64
):
    raise RuntimeError("the diagnostic and sampler H256-U8-PS2 laws disagree")


SPATIAL_H256_U8_PS2_METHOD_TIME_COMPONENTS = (
    "partition_load_or_build",
    "candidate_query",
    "stratum_exact_flow",
    "integer_cdf_construction_sampling_and_audit",
    "selected_production_flow",
    "pair_reduction_and_hashing",
)


# Spell out the new protocol instead of inheriting the rejected pilot/Neyman
# protocol wholesale.  The selected artifacts and gate thresholds are reused
# deliberately; pilot/allocation/reservoir semantics are not.
SPATIAL_H256_U8_PS2_PROTOCOL = {
    "label": SPATIAL_H256_U8_PS2_CENTER_LABEL,
    "universe_start": 25_000,
    "universe_stop": 25_512,
    "center": [0.013610377120038044, 0.002740148300964851],
    "initial_strategy": "mean_observed_shape",
    "n_candidates": 16_384,
    "prefilter_candidates": 131_072,
    "n_global_draws": 16_384,
    "proposal_seed": 8_701,
    "production_seed": SPATIAL_PS_SEED,
    "candidate_backend": "torch",
    "diagnostic_object_chunk": 32,
    "flow_atom_chunk": 4_096,
    "spatial_atom_chunk": 262_144,
    "spatial_object_chunk": 32,
    "n_cells": 256,
    "sampler_version": SPATIAL_PS_VERSION,
    "sampler_hash_contract": SPATIAL_PS_HASH_CONTRACT,
    "rng_contract": SPATIAL_PS_RNG_CONTRACT,
    "rng_domain_word": SPATIAL_PS_RNG_DOMAIN_WORD,
    "rng_version_word": SPATIAL_PS_RNG_VERSION_WORD,
    "quantizer_contract": SPATIAL_PS_QUANTIZER_CONTRACT,
    "order_contract": SPATIAL_PS_ORDER_CONTRACT,
    "cells_per_slab": SPATIAL_PS_CELLS_PER_SLAB,
    "cycles_per_cell": 8,
    "source_microstrata": list(SPATIAL_H256_U8_PS2_SOURCE_MICROSTRATA),
    "source_draws_per_cell": list(SPATIAL_H256_U8_PS2_SOURCE_DRAWS),
    "microstrata_per_cell": sum(SPATIAL_H256_U8_PS2_SOURCE_MICROSTRATA),
    "draws_per_microstratum": 2,
    "draws_per_cell": sum(SPATIAL_H256_U8_PS2_SOURCE_DRAWS),
    "production_draws": 16_384,
    "cdf_tick_bits": SPATIAL_H256_U8_PS2_CDF_TICK_BITS,
    "cdf_ticks_per_microstratum": SPATIAL_H256_U8_PS2_CDF_TICKS,
    "source_total_ticks": [
        count * SPATIAL_H256_U8_PS2_CDF_TICKS
        for count in SPATIAL_H256_U8_PS2_SOURCE_MICROSTRATA
    ],
    "balance_total_ticks": SPATIAL_H256_U8_PS2_BALANCE_TICKS,
    "balance_probability_law": "q(a)=K(a)/2^53; K(a)=k0(a)+k1(a)+k2(a)",
    "prior_minimum_atom_ticks": 1,
    "proxy_fallback": "copy_quantized_prior",
    "proxy_fallback_tick_scales": [4, 3],
    "integer_target_law": "target=b*2^48+u48, 0<=u48<2^48",
    "pilot": {
        "used": False,
        "draws": 0,
        "target_flow_evaluations": 0,
        "rng_domains": [],
        "allocation": "fixed_uniform_8",
    },
    "tail_proposal_recipe": deepcopy(
        SPATIAL_NEYMAN_PROTOCOL["tail_proposal_recipe"]
    ),
    "partition_recipe_sha256": SPATIAL_NEYMAN_PROTOCOL[
        "partition_recipe_sha256"
    ],
    "partition_sha256": SPATIAL_NEYMAN_PROTOCOL["partition_sha256"],
    "partition_cache_sha256": SPATIAL_NEYMAN_PROTOCOL[
        "partition_cache_sha256"
    ],
    "partition_manifest_sha256": SPATIAL_NEYMAN_PROTOCOL[
        "partition_manifest_sha256"
    ],
    "exact_selection_manifest_sha256": SPATIAL_NEYMAN_PROTOCOL[
        "exact_selection_manifest_sha256"
    ],
    "exact_selection_count": 64,
    "exact_atom_chunk": SPATIAL_NEYMAN_PROTOCOL["exact_atom_chunk"],
    "exact_object_chunk": SPATIAL_NEYMAN_PROTOCOL["exact_object_chunk"],
    "active_atoms": 12_760_990,
    "stratum_flow_evaluations": 8_388_608,
    "production_flow_evaluations": 8_388_608,
    "exact_flow_evaluations": 816_703_360,
    "run_flow_evaluations": 833_480_576,
    "mock_input_sha256": deepcopy(SPATIAL_NEYMAN_PROTOCOL["mock_input_sha256"]),
}


def _is_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _gate(observed: Any, operator: str, threshold: Any, passed: bool) -> dict:
    return {
        "observed": observed,
        "operator": operator,
        "threshold": threshold,
        "passed": bool(passed),
    }


def _higher_percentile(values: np.ndarray, percentile: float) -> float:
    return float(np.percentile(values, percentile, method="higher"))


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
    array = np.ascontiguousarray(values, dtype=np.dtype(dtype))
    digest = sha256()
    digest.update(label.encode("utf-8") + b"\0")
    digest.update(struct.pack(">Q", array.ndim))
    for width in array.shape:
        digest.update(struct.pack(">Q", int(width)))
    digest.update(array.tobytes())
    return digest.hexdigest()


_SPATIAL_H256_U8_PS2_MANIFEST_HASH_COLUMNS = (
    "stratum_indices_sha256",
    "stratum_log_target_sha256",
    "production_indices_sha256",
    "production_probability_sha256",
    "production_probability_numerator_sha256",
    "production_selected_source_ticks_sha256",
    "production_selected_source_cdf_lower_sha256",
    "production_u48_sha256",
    "production_log_ratio_sha256",
    "production_normalizer_sha256",
    "production_quantization_audit_sha256",
    "production_exclusion_indices_sha256",
    "production_schedule_sha256",
    "production_cell_order_sha256",
    "production_draw_record_sha256",
)


def spatial_h256_u8_ps2_run_manifest_sha256(frame: pd.DataFrame) -> str:
    """Hash all object-local science arrays and enclosing core draw records."""

    required = ("object_index", *_SPATIAL_H256_U8_PS2_MANIFEST_HASH_COLUMNS)
    if any(name not in frame for name in required):
        raise ValueError("H256-U8-PS2 manifest columns are incomplete")
    rows = []
    for record in frame.loc[:, required].to_dict(orient="records"):
        object_id = record["object_index"]
        if (
            isinstance(object_id, (bool, np.bool_))
            or not isinstance(object_id, (int, np.integer))
            or object_id < 0
            or object_id > np.iinfo(np.int64).max
        ):
            raise ValueError("H256-U8-PS2 object ids must be non-negative int64")
        hashes = tuple(
            record[name] for name in _SPATIAL_H256_U8_PS2_MANIFEST_HASH_COLUMNS
        )
        if not all(_is_sha256(value) for value in hashes):
            raise ValueError("H256-U8-PS2 manifest contains an invalid SHA-256")
        rows.append((int(object_id), hashes))
    rows.sort(key=lambda item: item[0])
    if len({object_id for object_id, _ in rows}) != len(rows):
        raise ValueError("H256-U8-PS2 manifest object ids must be unique")
    digest = sha256()
    digest.update(b"SBSI/spatial-h256-u8-ps2-center-run/v1\0")
    digest.update(struct.pack(">Q", len(rows)))
    for object_id, hashes in rows:
        digest.update(struct.pack(">Q", object_id))
        for value in hashes:
            digest.update(bytes.fromhex(value))
    return digest.hexdigest()


@dataclass(frozen=True)
class SpatialH256U8PS2Reduction:
    """Stable fixed-source PS2 reduction for one observation."""

    mass: float
    standard_error: float
    pareto_k: float
    maximum_effective_weight: float
    variance: float
    log_mass: float
    log_variance: float
    log_maximum_effective_weight: float
    cell_mass: np.ndarray
    cell_variance: np.ndarray
    cell_log_mass: np.ndarray
    cell_log_variance: np.ndarray
    cell_source_draw_counts: np.ndarray
    scaled_values: np.ndarray
    cell_source_shift: np.ndarray
    cell_source_scaled_mass: np.ndarray
    cell_source_scaled_pair_squared_difference_sum: np.ndarray

    def __post_init__(self):
        cell_mass = np.ascontiguousarray(self.cell_mass, dtype=np.float64).copy()
        cell_variance = np.ascontiguousarray(
            self.cell_variance, dtype=np.float64
        ).copy()
        cell_log_mass = np.ascontiguousarray(
            self.cell_log_mass, dtype=np.float64
        ).copy()
        cell_log_variance = np.ascontiguousarray(
            self.cell_log_variance, dtype=np.float64
        ).copy()
        source_counts = np.ascontiguousarray(
            self.cell_source_draw_counts, dtype=np.int64
        ).copy()
        scaled_values = np.ascontiguousarray(
            self.scaled_values, dtype=np.float64
        ).copy()
        source_shift = np.ascontiguousarray(
            self.cell_source_shift, dtype=np.float64
        ).copy()
        source_scaled_mass = np.ascontiguousarray(
            self.cell_source_scaled_mass, dtype=np.float64
        ).copy()
        source_scaled_pair_sum = np.ascontiguousarray(
            self.cell_source_scaled_pair_squared_difference_sum,
            dtype=np.float64,
        ).copy()
        cells = len(cell_mass)
        expected_source_counts = np.broadcast_to(
            np.asarray(SPATIAL_H256_U8_PS2_SOURCE_DRAWS, dtype=np.int64),
            (cells, 3),
        )
        if (
            cell_mass.ndim != 1
            or not cells
            or cell_variance.shape != (cells,)
            or cell_log_mass.shape != (cells,)
            or cell_log_variance.shape != (cells,)
            or source_counts.shape != (cells, 3)
            or scaled_values.shape != (cells * 64,)
            or source_shift.shape != (cells, 3)
            or source_scaled_mass.shape != (cells, 3)
            or source_scaled_pair_sum.shape != (cells, 3)
            or not np.isfinite(cell_mass).all()
            or (cell_mass < 0.0).any()
            or not np.isfinite(cell_variance).all()
            or (cell_variance < 0.0).any()
            or not np.isfinite(cell_log_mass).all()
            or np.isnan(cell_log_variance).any()
            or np.isposinf(cell_log_variance).any()
            or not np.array_equal(source_counts, expected_source_counts)
            or not np.isfinite(scaled_values).all()
            or (scaled_values < 0.0).any()
            or np.isnan(source_shift).any()
            or np.isposinf(source_shift).any()
            or not np.isfinite(source_scaled_mass).all()
            or (source_scaled_mass < 0.0).any()
            or not np.isfinite(source_scaled_pair_sum).all()
            or (source_scaled_pair_sum < 0.0).any()
        ):
            raise ValueError("H256-U8-PS2 reduction arrays are invalid")
        scalar = np.asarray(
            (
                self.mass,
                self.standard_error,
                self.maximum_effective_weight,
                self.variance,
                self.log_mass,
                self.log_maximum_effective_weight,
            ),
            dtype=np.float64,
        )
        if (
            not np.isfinite(scalar).all()
            or self.mass <= 0.0
            or self.standard_error < 0.0
            or self.maximum_effective_weight <= 0.0
            or self.variance < 0.0
            or np.isnan(self.log_variance)
            or np.isposinf(self.log_variance)
            or np.isinf(self.pareto_k)
            or not np.allclose(
                self.mass,
                cell_mass.sum(dtype=np.float64),
                # A 256-term exp(logsumexp) round trip can accumulate a few
                # hundred eps relative to the corresponding ordinary sum.
                rtol=512.0 * np.finfo(np.float64).eps,
                atol=0.0,
            )
            or not np.allclose(
                self.variance,
                cell_variance.sum(dtype=np.float64),
                rtol=512.0 * np.finfo(np.float64).eps,
                atol=0.0,
            )
            or not np.allclose(
                self.standard_error,
                (
                    0.0
                    if np.isneginf(self.log_variance)
                    else np.exp(0.5 * self.log_variance)
                ),
                rtol=64.0 * np.finfo(np.float64).eps,
                atol=0.0,
            )
        ):
            raise ValueError("H256-U8-PS2 reduction scalars are invalid")
        for array in (
            cell_mass,
            cell_variance,
            cell_log_mass,
            cell_log_variance,
            source_counts,
            scaled_values,
            source_shift,
            source_scaled_mass,
            source_scaled_pair_sum,
        ):
            array.setflags(write=False)
        object.__setattr__(self, "cell_mass", cell_mass)
        object.__setattr__(self, "cell_variance", cell_variance)
        object.__setattr__(self, "cell_log_mass", cell_log_mass)
        object.__setattr__(self, "cell_log_variance", cell_log_variance)
        object.__setattr__(self, "cell_source_draw_counts", source_counts)
        object.__setattr__(self, "scaled_values", scaled_values)
        object.__setattr__(self, "cell_source_shift", source_shift)
        object.__setattr__(
            self, "cell_source_scaled_mass", source_scaled_mass
        )
        object.__setattr__(
            self,
            "cell_source_scaled_pair_squared_difference_sum",
            source_scaled_pair_sum,
        )


@dataclass
class SpatialH256U8PS2Diagnostic:
    """Per-object H256-U8-PS2 estimates and immutable protocol identity."""

    table: pd.DataFrame
    metadata: dict

    def summary(self) -> dict:
        return spatial_h256_u8_ps2_center_summary(self)


def _finite_logsumexp(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    if not finite.any():
        return -np.inf
    maximum = float(np.max(values[finite]))
    return maximum + float(
        np.log(np.exp(values[finite] - maximum, dtype=np.float64).sum())
    )


def _finite_row_logsumexp(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("row log-sum-exp requires a matrix")
    finite = np.isfinite(values)
    maximum = np.max(
        np.where(finite, values, -np.inf), axis=1, keepdims=True
    )
    usable = np.isfinite(maximum[:, 0])
    result = np.full(len(values), -np.inf, dtype=np.float64)
    result[usable] = maximum[usable, 0] + np.log(
        np.exp(
            values[usable] - maximum[usable], dtype=np.float64
        ).sum(axis=1, dtype=np.float64)
    )
    return result


def _log_absolute_exp_difference(left: float, right: float) -> float:
    """Return ``log(abs(exp(left) - exp(right)))`` without exponentiating."""

    if np.isneginf(left):
        return float(right)
    if np.isneginf(right):
        return float(left)
    if left == right:
        return -np.inf
    high = max(left, right)
    separation = abs(left - right)
    return float(high + np.log(-np.expm1(-separation)))


def _validated_pair_schedule(
    size: int,
    cell_id: np.ndarray,
    source_id: np.ndarray,
    microstratum_id: np.ndarray,
    pair_draw_id: np.ndarray,
    *,
    cells: int,
) -> np.ndarray:
    """Validate the exact canonical H256-U8-PS2 slot schedule."""

    arrays = tuple(
        np.asarray(values)
        for values in (cell_id, source_id, microstratum_id, pair_draw_id)
    )
    if any(values.ndim != 1 or len(values) != size for values in arrays):
        raise ValueError("PS2 schedule arrays must align with the value vector")
    if any(values.dtype.kind not in "iu" for values in arrays):
        raise ValueError("PS2 schedule arrays must have integer dtype")
    cell, source, microstratum, draw = (
        values.astype(np.int64, copy=False) for values in arrays
    )
    if size != cells * sum(SPATIAL_H256_U8_PS2_SOURCE_DRAWS):
        raise ValueError("PS2 schedule does not contain 64 draws per cell")
    expected_cell = np.repeat(np.arange(cells, dtype=np.int64), 64)
    expected_source_block = np.repeat(
        np.arange(3, dtype=np.int64),
        np.asarray(SPATIAL_H256_U8_PS2_SOURCE_DRAWS, dtype=np.int64),
    )
    expected_source = np.tile(expected_source_block, cells)
    expected_microstratum_block = np.concatenate(
        tuple(
            np.repeat(np.arange(count, dtype=np.int64), 2)
            for count in SPATIAL_H256_U8_PS2_SOURCE_MICROSTRATA
        )
    )
    expected_microstratum = np.tile(expected_microstratum_block, cells)
    expected_draw = np.tile(np.arange(2, dtype=np.int64), size // 2)
    if not (
        np.array_equal(cell, expected_cell)
        and np.array_equal(source, expected_source)
        and np.array_equal(microstratum, expected_microstratum)
        and np.array_equal(draw, expected_draw)
    ):
        if len(np.unique(np.stack(arrays, axis=1), axis=0)) != size:
            raise ValueError("a PS2 microstratum/draw slot is duplicated")
        raise ValueError("the PS2 slot schedule is not in canonical order")
    return np.broadcast_to(
        np.asarray(SPATIAL_H256_U8_PS2_SOURCE_DRAWS, dtype=np.int64),
        (cells, 3),
    )


def _reduce_h256_u8_ps2_log_values(
    log_values: np.ndarray,
    cell_id: np.ndarray,
    source_id: np.ndarray,
    microstratum_id: np.ndarray,
    pair_draw_id: np.ndarray,
    *,
    cells: int = 256,
) -> SpatialH256U8PS2Reduction:
    """Reduce one H256-U8-PS2 object with exact fixed-source pair algebra.

    The fixed source proportions equal ``B_l / 32`` and each source has
    ``2 B_l`` draws.  Consequently the cell estimate is exactly the all-64
    sample mean.  With two draws in every equal-probability microstratum, the
    unbiased cell variance estimator simplifies to

    ``sum_{all 32 pairs} (Y_0 - Y_1)^2 / 4096``.

    Masses and pair differences are reconstructed from logs, so a low-scale
    varying source is not erased by a high-scale constant source.
    """

    logs = np.asarray(log_values, dtype=np.float64)
    if logs.ndim != 1 or not len(logs):
        raise ValueError("PS2 log values must be a nonempty vector")
    if np.isnan(logs).any() or np.isposinf(logs).any():
        raise ValueError("PS2 log values must be free of NaN and +inf")
    source_counts = _validated_pair_schedule(
        len(logs),
        cell_id,
        source_id,
        microstratum_id,
        pair_draw_id,
        cells=cells,
    )
    block = logs.reshape(cells, 64)
    cell_log_mass = _finite_row_logsumexp(block) - np.log(64.0)
    if not np.isfinite(cell_log_mass).all():
        raise RuntimeError("every PS2 cell must have positive estimated mass")
    left = block[:, 0::2]
    right = block[:, 1::2]
    high = np.maximum(left, right)
    with np.errstate(invalid="ignore"):
        separation = np.abs(left - right)
    log_absolute_difference = np.full_like(high, -np.inf)
    one_finite = np.isfinite(high) & (
        np.isneginf(left) | np.isneginf(right)
    )
    log_absolute_difference[one_finite] = high[one_finite]
    both_finite_distinct = (
        np.isfinite(left) & np.isfinite(right) & (separation > 0.0)
    )
    log_absolute_difference[both_finite_distinct] = (
        high[both_finite_distinct]
        + np.log(-np.expm1(-separation[both_finite_distinct]))
    )
    cell_log_variance = (
        _finite_row_logsumexp(2.0 * log_absolute_difference) - np.log(4096.0)
    )

    scaled_block = np.zeros_like(block)
    source_shift = np.full((cells, 3), -np.inf, dtype=np.float64)
    source_scaled_mass = np.zeros((cells, 3), dtype=np.float64)
    source_scaled_pair_sum = np.zeros((cells, 3), dtype=np.float64)
    source_start = 0
    for source, width in enumerate(SPATIAL_H256_U8_PS2_SOURCE_DRAWS):
        source_stop = source_start + width
        source_logs = block[:, source_start:source_stop]
        source_finite = np.isfinite(source_logs)
        usable = source_finite.any(axis=1)
        source_shift[usable, source] = np.max(
            np.where(source_finite[usable], source_logs[usable], -np.inf),
            axis=1,
        )
        scaled_source = scaled_block[:, source_start:source_stop]
        scaled_source[usable] = np.exp(
            source_logs[usable] - source_shift[usable, source, None],
            dtype=np.float64,
        )
        source_scaled_mass[:, source] = (
            scaled_source.sum(axis=1, dtype=np.float64) / 64.0
        )
        source_scaled_pair_sum[:, source] = np.square(
            scaled_source[:, 0::2] - scaled_source[:, 1::2]
        ).sum(axis=1, dtype=np.float64)
        source_start = source_stop

    log_mass = _finite_logsumexp(cell_log_mass)
    log_variance = _finite_logsumexp(cell_log_variance)
    mass = float(np.exp(log_mass))
    variance = 0.0 if np.isneginf(log_variance) else float(np.exp(log_variance))
    standard_error = (
        0.0 if np.isneginf(log_variance) else float(np.exp(0.5 * log_variance))
    )
    if not np.isfinite(mass) or mass <= 0.0:
        raise RuntimeError("PS2 mass is outside positive finite float64 range")
    if not np.isfinite(standard_error):
        raise RuntimeError("PS2 standard error is outside finite float64 range")

    finite = np.isfinite(logs)
    if not finite.any():
        raise RuntimeError("all PS2 importance values are zero")
    effective_logs = logs + np.log(float(cells))
    log_maximum_effective_weight = float(np.max(effective_logs[finite]))
    scaled_effective_weights = np.exp(
        effective_logs - log_maximum_effective_weight, dtype=np.float64
    )
    pareto_k = float(pareto_tail_index(scaled_effective_weights[None, :])[0])
    maximum_effective_weight = float(np.exp(log_maximum_effective_weight))
    if not np.isfinite(maximum_effective_weight):
        raise RuntimeError("PS2 maximum effective weight is outside float64 range")

    cell_mass = np.exp(cell_log_mass, dtype=np.float64)
    cell_variance = np.zeros(cells, dtype=np.float64)
    positive_cell_variance = np.isfinite(cell_log_variance)
    cell_variance[positive_cell_variance] = np.exp(
        cell_log_variance[positive_cell_variance], dtype=np.float64
    )
    return SpatialH256U8PS2Reduction(
        mass=mass,
        standard_error=standard_error,
        pareto_k=pareto_k,
        maximum_effective_weight=maximum_effective_weight,
        variance=variance,
        log_mass=log_mass,
        log_variance=log_variance,
        log_maximum_effective_weight=log_maximum_effective_weight,
        cell_mass=cell_mass,
        cell_variance=cell_variance,
        cell_log_mass=cell_log_mass,
        cell_log_variance=cell_log_variance,
        cell_source_draw_counts=source_counts,
        scaled_values=scaled_block.reshape(-1),
        cell_source_shift=source_shift,
        cell_source_scaled_mass=source_scaled_mass,
        cell_source_scaled_pair_squared_difference_sum=(
            source_scaled_pair_sum
        ),
    )


def _integer_contract_consistent(protocol: dict[str, Any]) -> bool:
    """Return whether a protocol obeys the frozen exact integer-tick law."""

    try:
        ticks = int(protocol["cdf_ticks_per_microstratum"])
        microstrata = tuple(int(value) for value in protocol["source_microstrata"])
        totals = tuple(int(value) for value in protocol["source_total_ticks"])
        return bool(
            protocol["sampler_version"] == SPATIAL_PS_VERSION
            and protocol["sampler_hash_contract"] == SPATIAL_PS_HASH_CONTRACT
            and protocol["rng_contract"] == SPATIAL_PS_RNG_CONTRACT
            and int(protocol["rng_domain_word"]) == SPATIAL_PS_RNG_DOMAIN_WORD
            and int(protocol["rng_version_word"]) == SPATIAL_PS_RNG_VERSION_WORD
            and protocol["quantizer_contract"] == SPATIAL_PS_QUANTIZER_CONTRACT
            and protocol["order_contract"] == SPATIAL_PS_ORDER_CONTRACT
            and int(protocol["production_seed"]) == SPATIAL_PS_SEED
            and int(protocol["n_cells"]) == SPATIAL_CELL_COUNT
            and int(protocol["cells_per_slab"]) == SPATIAL_PS_CELLS_PER_SLAB
            and int(protocol["cdf_tick_bits"]) == 48
            and ticks == 1 << 48
            and microstrata == (16, 12, 4)
            and microstrata == tuple(SPATIAL_PS_MICROSTRATA)
            and int(protocol["draws_per_microstratum"])
            == SPATIAL_PS_REPLICATES
            and tuple(int(value) for value in protocol["source_draws_per_cell"])
            == tuple(SPATIAL_PS_SOURCE_COUNTS)
            and int(protocol["draws_per_cell"]) == SPATIAL_PS_DRAWS_PER_CELL
            and totals == tuple(count * ticks for count in microstrata)
            and int(protocol["balance_total_ticks"]) == sum(totals) == 1 << 53
            and int(protocol["prior_minimum_atom_ticks"]) == 1
            and protocol["proxy_fallback"] == "copy_quantized_prior"
            and tuple(int(value) for value in protocol["proxy_fallback_tick_scales"])
            == (4, 3)
            and protocol["pilot"]
            == {
                "used": False,
                "draws": 0,
                "target_flow_evaluations": 0,
                "rng_domains": [],
                "allocation": "fixed_uniform_8",
            }
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def _observed_protocol(diagnostic) -> dict | None:
    metadata = diagnostic.metadata
    spatial = metadata.get("spatial_h256_u8_ps2_center")
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
            "proposal_seed": int(spatial["proposal_seed"]),
            "production_seed": int(spatial["production_seed"]),
            "candidate_backend": metadata["candidate_backend"],
            "diagnostic_object_chunk": int(metadata["object_chunk"]),
            "flow_atom_chunk": int(metadata["flow_atom_chunk"]),
            "spatial_atom_chunk": int(spatial["spatial_atom_chunk"]),
            "spatial_object_chunk": int(spatial["spatial_object_chunk"]),
            "n_cells": int(spatial["n_cells"]),
            "sampler_version": spatial["sampler_version"],
            "sampler_hash_contract": spatial["sampler_hash_contract"],
            "rng_contract": spatial["rng_contract"],
            "rng_domain_word": int(spatial["rng_domain_word"]),
            "rng_version_word": int(spatial["rng_version_word"]),
            "quantizer_contract": spatial["quantizer_contract"],
            "order_contract": spatial["order_contract"],
            "cells_per_slab": int(spatial["cells_per_slab"]),
            "cycles_per_cell": int(spatial["cycles_per_cell"]),
            "source_microstrata": [
                int(value) for value in spatial["source_microstrata"]
            ],
            "source_draws_per_cell": [
                int(value) for value in spatial["source_draws_per_cell"]
            ],
            "microstrata_per_cell": int(spatial["microstrata_per_cell"]),
            "draws_per_microstratum": int(
                spatial["draws_per_microstratum"]
            ),
            "draws_per_cell": int(spatial["draws_per_cell"]),
            "production_draws": int(spatial["production_draws"]),
            "cdf_tick_bits": int(spatial["cdf_tick_bits"]),
            "cdf_ticks_per_microstratum": int(
                spatial["cdf_ticks_per_microstratum"]
            ),
            "source_total_ticks": [
                int(value) for value in spatial["source_total_ticks"]
            ],
            "balance_total_ticks": int(spatial["balance_total_ticks"]),
            "balance_probability_law": spatial["balance_probability_law"],
            "prior_minimum_atom_ticks": int(
                spatial["prior_minimum_atom_ticks"]
            ),
            "proxy_fallback": spatial["proxy_fallback"],
            "proxy_fallback_tick_scales": [
                int(value) for value in spatial["proxy_fallback_tick_scales"]
            ],
            "integer_target_law": spatial["integer_target_law"],
            "pilot": deepcopy(spatial["pilot"]),
            "tail_proposal_recipe": deepcopy(spatial["tail_proposal_recipe"]),
            "partition_recipe_sha256": spatial["partition_recipe_sha256"],
            "partition_sha256": spatial["partition_sha256"],
            "partition_cache_sha256": spatial["partition_cache_sha256"],
            "partition_manifest_sha256": spatial[
                "partition_manifest_sha256"
            ],
            "exact_selection_manifest_sha256": exact[
                "selection_manifest_sha256"
            ],
            "exact_selection_count": int(exact["selection_count"]),
            "exact_atom_chunk": int(exact["atom_chunk"]),
            "exact_object_chunk": int(exact["object_chunk"]),
            "active_atoms": int(exact["active_atoms"]),
            "stratum_flow_evaluations": int(
                spatial["stratum_flow_evaluations"]
            ),
            "production_flow_evaluations": int(
                spatial["production_flow_evaluations"]
            ),
            "exact_flow_evaluations": int(exact["flow_evaluations"]),
            "run_flow_evaluations": int(metadata["flow_evaluations"]),
            "mock_input_sha256": deepcopy(spatial["mock_input_sha256"]),
        }
    except (KeyError, TypeError, ValueError, OverflowError):
        return None


def spatial_h256_u8_ps2_center_summary(diagnostic) -> dict:
    """Evaluate the frozen integrity, tail, exact, and efficiency gates."""

    frame = diagnostic.table
    metadata = diagnostic.metadata
    spatial = metadata.get("spatial_h256_u8_ps2_center")
    if not isinstance(spatial, dict):
        raise ValueError("diagnostic lacks H256-U8-PS2 centre metadata")
    expected_protocol = deepcopy(SPATIAL_H256_U8_PS2_PROTOCOL)
    observed_protocol = _observed_protocol(diagnostic)
    protocol_matched = bool(
        _integer_contract_consistent(expected_protocol)
        and observed_protocol == expected_protocol
    )
    thresholds = SPATIAL_H256_U8_PS2_CENTER_GATE_THRESHOLDS
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
    expected_exact_ids, expected_exact_manifest = exact_oracle_object_ids(
        25_000, 25_512
    )
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
    exact_selection_ok = bool(
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
    counts = {name: _exact_int64_vector(frame.get(name)) for name in integer_names}
    integer_counts = all(value is not None for value in counts.values())
    support_ok = bool(
        integer_counts
        and np.all(counts["stratum_count"] == 16_384)
        and np.all(counts["stratum_unique_count"] == 16_384)
        and np.all(counts["stratum_active_count"] == 16_384)
    )
    fixed_layout_ok = bool(
        integer_counts
        and np.all(counts["production_draws"] == 16_384)
        and np.all(counts["production_cells"] == 256)
        and np.all(counts["production_microstrata"] == 8_192)
        and np.all(counts["production_source_proxy_t1"] == 8_192)
        and np.all(counts["production_source_proxy_t2"] == 6_144)
        and np.all(counts["production_source_prior"] == 2_048)
        and boolean_all("canonical_schedule_exact")
    )
    integer_law_ok = bool(
        integer_counts
        and np.all(counts["source_audit_failed_count"] == 0)
        and np.all(counts["fallback_scaled_law_mismatch_count"] == 0)
        and np.all(counts["source_excluded_tick_sum"] == 0)
        and np.all(counts["selected_source_tick_overflow_count"] == 0)
        and np.all(counts["numerator_sum_failure_count"] == 0)
        and np.all(counts["probability_identity_failure_count"] == 0)
        and np.all(counts["cdf_bracket_failure_count"] == 0)
        and np.all(counts["rng_recomputation_failure_count"] == 0)
        and np.all(counts["probability_numerator_minimum"] >= 1)
        and np.all(
            counts["probability_numerator_maximum"]
            <= SPATIAL_H256_U8_PS2_BALANCE_TICKS
        )
        and np.all(counts["u48_minimum"] >= 0)
        and np.all(counts["u48_maximum"] < SPATIAL_H256_U8_PS2_CDF_TICKS)
        and np.all(counts["prior_minimum_eligible_tick"] >= 1)
        and np.all(counts["terminal_correction_maximum"] >= 0)
        and np.all(counts["terminal_correction_positive_count"] >= 0)
        and boolean_all("integer_source_totals_exact")
        and boolean_all("integer_prior_floor_satisfied")
        and boolean_all("integer_proxy_fallback_scaled_exact")
        and boolean_all("integer_balance_probability_exact")
        and boolean_all("integer_cdf_brackets_exact")
    )
    exclusion_ok = bool(
        integer_counts
        and np.all(counts["production_draws_in_stratum"] == 0)
        and all(
            name in frame
            for name in (
                "stratum_indices_sha256",
                "production_exclusion_indices_sha256",
            )
        )
        and frame["stratum_indices_sha256"].astype(str).tolist()
        == frame["production_exclusion_indices_sha256"].astype(str).tolist()
    )
    cell_membership_ok = bool(
        integer_counts
        and np.all(counts["partition_cell_membership_failure_count"] == 0)
        and boolean_all("partition_cell_membership_exact")
    )

    hash_columns_ok = bool(
        len(frame) == expected_objects
        and all(name in frame for name in _SPATIAL_H256_U8_PS2_MANIFEST_HASH_COLUMNS)
        and all(
            _is_sha256(value)
            for name in _SPATIAL_H256_U8_PS2_MANIFEST_HASH_COLUMNS
            for value in frame[name]
        )
    )
    try:
        recomputed_manifest = spatial_h256_u8_ps2_run_manifest_sha256(frame)
    except (KeyError, TypeError, ValueError):
        recomputed_manifest = None
    manifest_ok = bool(
        _is_sha256(spatial.get("run_manifest_sha256"))
        and spatial.get("run_manifest_sha256") == recomputed_manifest
    )
    partition_ok = bool(
        spatial.get("partition_sha256")
        == SPATIAL_H256_U8_PS2_PROTOCOL["partition_sha256"]
        and spatial.get("partition_cache_sha256")
        == SPATIAL_H256_U8_PS2_PROTOCOL["partition_cache_sha256"]
        and spatial.get("partition_manifest_sha256")
        == SPATIAL_H256_U8_PS2_PROTOCOL["partition_manifest_sha256"]
        and spatial.get("partition_verified") is True
        and int(spatial.get("minimum_cell_active_count", -1)) > 16_384
        and int(spatial.get("maximum_four_cell_slab_active_count", 262_145))
        <= 262_144
    )
    no_pilot_ok = bool(
        spatial.get("pilot") == SPATIAL_H256_U8_PS2_PROTOCOL["pilot"]
        and not any(
            name.startswith(("pilot", "allocation", "reservoir"))
            for name in frame.columns
        )
    )
    flow_counts_ok = bool(
        spatial.get("stratum_flow_evaluations") == 8_388_608
        and spatial.get("production_flow_evaluations") == 8_388_608
        and spatial.get("pilot", {}).get("target_flow_evaluations") == 0
        and metadata.get("exact_oracle", {}).get("flow_evaluations")
        == 816_703_360
        and metadata.get("flow_evaluations") == 833_480_576
    )
    calls = spatial.get("execution_call_provenance")
    calls_ok = bool(ids_exact and isinstance(calls, list) and len(calls) == 16)
    if calls_ok:
        for call_index, call in enumerate(calls):
            expected_start = 25_000 + 32 * call_index
            block = frame.iloc[32 * call_index : 32 * (call_index + 1)]
            row_hashes_match = bool(
                isinstance(call, dict)
                and len(block) == 32
                and all(
                    name in block
                    and block[name].astype(str).eq(call.get(call_name)).all()
                    for name, call_name in (
                        ("production_draw_record_sha256", "record_sha256"),
                        ("production_schedule_sha256", "schedule_sha256"),
                        ("production_cell_order_sha256", "cell_order_sha256"),
                    )
                )
            )
            if not (
                isinstance(call, dict)
                and call.get("object_start") == expected_start
                and call.get("object_stop") == expected_start + 32
                and call.get("objects") == 32
                and call.get("seed") == SPATIAL_PS_SEED
                and call.get("atom_chunk") == 262_144
                and call.get("object_chunk") == 32
                and call.get("cells_per_slab") == SPATIAL_PS_CELLS_PER_SLAB
                and call.get("partition_sha256")
                == SPATIAL_H256_U8_PS2_PROTOCOL["partition_sha256"]
                and call.get("sampler_version") == SPATIAL_PS_VERSION
                and call.get("rng_contract") == SPATIAL_PS_RNG_CONTRACT
                and call.get("quantizer_contract")
                == SPATIAL_PS_QUANTIZER_CONTRACT
                and call.get("hash_contract") == SPATIAL_PS_HASH_CONTRACT
                and call.get("order_contract") == SPATIAL_PS_ORDER_CONTRACT
                and row_hashes_match
                and all(
                    _is_sha256(call.get(name))
                    for name in (
                        "record_sha256",
                        "schedule_sha256",
                        "quantization_sha256",
                        "cell_order_sha256",
                    )
                )
            ):
                calls_ok = False
                break
    mass_crosscheck = values("core_cell_mass_crosscheck_relative_max")
    pair_crosscheck = values("core_cell_pair_sum_crosscheck_relative_max")
    core_crosscheck_ok = bool(
        len(frame) == expected_objects
        and np.isfinite(mass_crosscheck).all()
        and np.isfinite(pair_crosscheck).all()
        and np.all(mass_crosscheck <= 1.0e-13)
        and np.all(pair_crosscheck <= 1.0e-13)
    )
    phase_seconds = metadata.get("phase_seconds", {})
    try:
        proposal_initialization_seconds = float(
            spatial.get("proposal_initialization_seconds", np.nan)
        )
    except (TypeError, ValueError, OverflowError):
        proposal_initialization_seconds = float("nan")
    phase_ok = bool(
        isinstance(phase_seconds, dict)
        and np.isfinite(proposal_initialization_seconds)
        and proposal_initialization_seconds >= 0.0
        and all(
            name in phase_seconds
            and np.isfinite(float(phase_seconds[name]))
            and float(phase_seconds[name]) >= 0.0
            for name in SPATIAL_H256_U8_PS2_METHOD_TIME_COMPONENTS
        )
    )
    stratum_mass = values("mass_stratum")
    stratum_log_mass = values("log_mass_stratum")
    complement_log_mass = values(
        "log_mass_complement_spatial_h256_u8_ps2"
    )
    complement_log_variance = values(
        "log_variance_complement_spatial_h256_u8_ps2"
    )
    complement_mass = values("mass_complement_spatial_h256_u8_ps2")
    complement_se = values(
        "mass_complement_spatial_h256_u8_ps2_standard_error"
    )
    finite_positive = bool(
        len(frame) == expected_objects
        and np.isfinite(stratum_mass).all()
        and np.isfinite(stratum_log_mass).all()
        and np.isfinite(complement_log_mass).all()
        and np.isfinite(complement_log_variance).all()
        and np.isfinite(complement_mass).all()
        and np.isfinite(complement_se).all()
        and (stratum_mass > 0.0).all()
        and (complement_mass > 0.0).all()
        and (complement_se > 0.0).all()
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        mass_and_standard_error_log_identity = bool(
            finite_positive
            and np.allclose(
                np.log(stratum_mass),
                stratum_log_mass,
                rtol=64.0 * np.finfo(np.float64).eps,
                atol=64.0 * np.finfo(np.float64).eps,
            )
            and np.allclose(
                np.log(complement_mass),
                complement_log_mass,
                rtol=64.0 * np.finfo(np.float64).eps,
                atol=64.0 * np.finfo(np.float64).eps,
            )
            and np.allclose(
                2.0 * np.log(complement_se),
                complement_log_variance,
                rtol=64.0 * np.finfo(np.float64).eps,
                atol=64.0 * np.finfo(np.float64).eps,
            )
        )
    integrity_gates = {
        "protocol_identity_matched": _gate(protocol_matched, "is", True, protocol_matched),
        "absolute_object_ids_exact": _gate(ids_exact, "is", True, ids_exact),
        "exact_selection_identity_exact": _gate(exact_selection_ok, "is", True, exact_selection_ok),
        "exact_support_unique_active": _gate(support_ok, "is", True, support_ok),
        "fixed_h256_u8_ps2_layout_exact": _gate(fixed_layout_ok, "is", True, fixed_layout_ok),
        "integer_cdf_law_exact": _gate(integer_law_ok, "is", True, integer_law_ok),
        "conditioned_exclusion_exact": _gate(exclusion_ok, "is", True, exclusion_ok),
        "partition_cell_membership_exact": _gate(cell_membership_ok, "is", True, cell_membership_ok),
        "run_manifest_recomputes": _gate(manifest_ok, "is", True, manifest_ok),
        "partition_identity_exact": _gate(partition_ok, "is", True, partition_ok),
        "no_pilot_or_allocation": _gate(no_pilot_ok, "is", True, no_pilot_ok),
        "flow_evaluation_counts_exact": _gate(flow_counts_ok, "is", True, flow_counts_ok),
        "core_call_provenance_exact": _gate(calls_ok, "is", True, calls_ok),
        "source_isolated_core_reduction_crosscheck": _gate(core_crosscheck_ok, "is", True, core_crosscheck_ok),
        "method_phase_timing_complete": _gate(phase_ok, "is", True, phase_ok),
        "finite_positive_estimates": _gate(finite_positive, "is", True, finite_positive),
        "mass_and_standard_error_log_identity": _gate(
            mass_and_standard_error_log_identity,
            "is",
            True,
            mass_and_standard_error_log_identity,
        ),
        "manifest_hash_columns_complete": _gate(hash_columns_ok, "is", True, hash_columns_ok),
    }

    pareto = values("pareto_k_spatial_h256_u8_ps2_complement")
    finite_k = np.isfinite(pareto)
    rel = np.divide(
        complement_se,
        np.abs(complement_mass),
        out=np.full(len(frame), np.nan),
        where=np.abs(complement_mass) > 0.0,
    )
    k_below_05 = int(np.sum(finite_k & (pareto < 0.5)))
    k_below_07 = int(np.sum(finite_k & (pareto < 0.7)))
    k_at_least_1 = int(np.sum(finite_k & (pareto >= 1.0)))
    rel_median = float(np.median(rel)) if np.isfinite(rel).all() else float("nan")
    rel_p90 = _higher_percentile(rel, 90) if np.isfinite(rel).all() else float("nan")
    reliability_gates = {
        "pareto_k_defined_all_objects": _gate(
            int((~finite_k).sum()),
            "undefined_count ==",
            0,
            len(pareto) == expected_objects and finite_k.all(),
        ),
        "pareto_k_below_0_5_count": _gate(k_below_05, ">=", thresholds["pareto_k_below_0_5_count"], k_below_05 >= thresholds["pareto_k_below_0_5_count"]),
        "pareto_k_below_0_7_count": _gate(k_below_07, ">=", thresholds["pareto_k_below_0_7_count"], k_below_07 >= thresholds["pareto_k_below_0_7_count"]),
        "pareto_k_at_least_1_count": _gate(k_at_least_1, "==", thresholds["pareto_k_at_least_1_count"], k_at_least_1 == thresholds["pareto_k_at_least_1_count"]),
        "complement_relative_error_median": _gate(rel_median, "<=", thresholds["complement_relative_error_median"], np.isfinite(rel_median) and rel_median <= thresholds["complement_relative_error_median"]),
        "complement_relative_error_p90_higher": _gate(rel_p90, "<=", thresholds["complement_relative_error_p90_higher"], np.isfinite(rel_p90) and rel_p90 <= thresholds["complement_relative_error_p90_higher"]),
    }

    selected = (
        frame["exact_oracle_selected"].to_numpy(dtype=bool)
        if exact_mask_valid
        else np.zeros(len(frame), dtype=bool)
    )
    delta = np.abs(
        values("delta_log_total_evidence_spatial_h256_u8_ps2_minus_exact")[selected]
    )
    z = values("mass_complement_spatial_h256_u8_ps2_minus_exact_z")[selected]
    stratum_crosscheck = np.abs(
        values("mass_stratum_exact_oracle_relative_error")[selected]
    )
    valid_exact = bool(
        selected.sum() == 64
        and len(delta) == len(z) == len(stratum_crosscheck) == 64
        and np.isfinite(delta).all()
        and np.isfinite(z).all()
        and np.isfinite(stratum_crosscheck).all()
    )
    median_delta = float(np.median(delta)) if valid_exact else float("nan")
    p90_delta = _higher_percentile(delta, 90) if valid_exact else float("nan")
    crosscheck_max = float(np.max(stratum_crosscheck)) if valid_exact else float("nan")
    selected_se = complement_se[selected]
    selected_difference = (
        complement_mass[selected] - values("mass_complement_exact_oracle")[selected]
    )
    aggregate_denominator = float(np.sqrt(np.square(selected_se).sum()))
    aggregate_z = (
        float(selected_difference.sum() / aggregate_denominator)
        if valid_exact and aggregate_denominator > 0.0
        else float("nan")
    )
    z_rms = float(np.sqrt(np.mean(np.square(z)))) if valid_exact else float("nan")
    within_196 = int(np.sum(np.abs(z) <= 1.96)) if valid_exact else 0
    exact_gates = {
        "selected_exact_object_count": _gate(
            int(selected.sum()), "==", 64, int(selected.sum()) == 64
        ),
        "absolute_delta_log_total_evidence_median": _gate(median_delta, "<=", thresholds["absolute_delta_log_total_evidence_median"], np.isfinite(median_delta) and median_delta <= thresholds["absolute_delta_log_total_evidence_median"]),
        "absolute_delta_log_total_evidence_p90_higher": _gate(p90_delta, "<=", thresholds["absolute_delta_log_total_evidence_p90_higher"], np.isfinite(p90_delta) and p90_delta <= thresholds["absolute_delta_log_total_evidence_p90_higher"]),
        "stratum_relative_crosscheck_max": _gate(crosscheck_max, "<=", thresholds["stratum_relative_crosscheck_max"], np.isfinite(crosscheck_max) and crosscheck_max <= thresholds["stratum_relative_crosscheck_max"]),
        "absolute_signed_aggregate_z": _gate(abs(aggregate_z) if np.isfinite(aggregate_z) else float("nan"), "<=", thresholds["absolute_signed_aggregate_z"], np.isfinite(aggregate_z) and abs(aggregate_z) <= thresholds["absolute_signed_aggregate_z"]),
        "exact_z_rms": _gate(z_rms, "in_closed_interval", thresholds["exact_z_rms"], np.isfinite(z_rms) and thresholds["exact_z_rms"][0] <= z_rms <= thresholds["exact_z_rms"][1]),
        "exact_z_within_1_96_count": _gate(within_196, ">=", thresholds["exact_z_within_1_96_count"], within_196 >= thresholds["exact_z_within_1_96_count"]),
    }

    # The stable reducer's logs are authoritative here as well as in the
    # exact-oracle comparison.  Adding two individually finite physical
    # masses can overflow and would otherwise turn the efficiency score into
    # a spurious zero.  Fail closed if the final binary64 relative variance
    # is not finite and strictly positive.
    total_log_mass = np.logaddexp(stratum_log_mass, complement_log_mass)
    total_log_relative_error_squared = (
        complement_log_variance - 2.0 * total_log_mass
    )
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        total_relative_error_squared = np.exp(
            total_log_relative_error_squared
        )
    method_seconds = proposal_initialization_seconds
    try:
        method_seconds += sum(
            float(phase_seconds[name])
            for name in SPATIAL_H256_U8_PS2_METHOD_TIME_COMPONENTS
        )
    except (KeyError, TypeError, ValueError):
        method_seconds = float("nan")
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
        / (baseline["total_relative_error_squared_p90_higher"] * baseline["method_seconds"])
        if finite_efficiency
        else float("nan")
    )
    efficiency_gates = {
        "mean_relative_variance_time_efficiency": _gate(efficiency_mean, "<=", thresholds["efficiency_mean"], np.isfinite(efficiency_mean) and efficiency_mean <= thresholds["efficiency_mean"]),
        "p90_higher_relative_variance_time_efficiency": _gate(efficiency_p90, "<=", thresholds["efficiency_p90_higher"], np.isfinite(efficiency_p90) and efficiency_p90 <= thresholds["efficiency_p90_higher"]),
    }
    families = {
        "integrity": integrity_gates,
        "tail_reliability": reliability_gates,
        "exact_calibration": exact_gates,
        "efficiency": efficiency_gates,
    }
    passed = all(
        item["passed"] for family in families.values() for item in family.values()
    )
    return {
        "protocol": {
            "expected": expected_protocol,
            "observed": observed_protocol,
            "matched": protocol_matched,
        },
        "method_seconds": method_seconds,
        "proposal_initialization_seconds": spatial.get(
            "proposal_initialization_seconds"
        ),
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
        "gates": {
            "predeclared": protocol_matched,
            "thresholds_predeclared": True,
            **families,
            "passed": bool(passed),
            "policy": "all integrity, tail, exact-calibration, and fair-efficiency gates must pass",
        },
    }


def _core_array_sha256(
    label: str,
    values: np.ndarray,
    *,
    dtype: str,
    sort_rows: bool = False,
) -> str:
    """Independently reproduce the core H256-U8-PS2 array hash contract."""

    array = np.asarray(values)
    if sort_rows:
        if array.ndim != 2:
            raise ValueError("only matrices can be sorted by row")
        array = np.sort(array, axis=1)
    canonical = np.ascontiguousarray(array, dtype=np.dtype(dtype))
    shape = np.ascontiguousarray(canonical.shape, dtype=np.dtype(">u8"))
    digest = sha256()
    digest.update(b"SBSI/spatial-probability-stratified-array/v1\0")
    digest.update(label.encode("utf-8") + b"\0")
    digest.update(shape.tobytes())
    digest.update(canonical.tobytes())
    return digest.hexdigest()


def _exclusion_hashes(indices: np.ndarray) -> tuple[str, ...]:
    array = np.asarray(indices, dtype=np.int64)
    if array.ndim != 2:
        raise ValueError("H256-U8-PS2 exclusions require a matrix")
    return tuple(
        _core_array_sha256(
            "object_conditioned_exclusion_indices",
            array[row : row + 1],
            dtype=">i8",
            sort_rows=True,
        )
        for row in range(len(array))
    )


def _recompute_u48(
    *, seed: int, object_id: int, cells: int = 256
) -> np.ndarray:
    """Independently reconstruct the canonical counter-domain RNG record."""

    result = np.empty(cells * 64, dtype=np.uint64)
    offset = 0
    for cell in range(cells):
        for source, count in enumerate(SPATIAL_PS_SOURCE_COUNTS):
            words = [
                int(seed) & 0xFFFFFFFF,
                int(seed) >> 32,
                int(object_id) & 0xFFFFFFFF,
                int(object_id) >> 32,
                cell,
                source,
                SPATIAL_PS_RNG_DOMAIN_WORD,
                SPATIAL_PS_RNG_VERSION_WORD,
            ]
            generator = np.random.PCG64(np.random.SeedSequence(words))
            raw = generator.random_raw(count)
            result[offset : offset + count] = np.right_shift(
                raw, np.uint64(16)
            )
            offset += count
    if offset != len(result):
        raise RuntimeError("H256-U8-PS2 RNG reconstruction width changed")
    return result


def _relative_max(observed: np.ndarray, expected: np.ndarray) -> np.ndarray:
    observed = np.asarray(observed, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    if observed.shape != expected.shape or observed.ndim != 2:
        raise ValueError("core cross-check matrices must align")
    difference = np.abs(observed - expected)
    relative = np.divide(
        difference,
        np.abs(expected),
        out=np.where(difference == 0.0, 0.0, np.inf),
        where=np.abs(expected) > 0.0,
    )
    return np.max(relative, axis=1)


def _source_isolated_core_crosscheck(
    reductions: list[SpatialH256U8PS2Reduction],
    draw: SpatialProbabilityStratifiedMISDraw,
) -> tuple[np.ndarray, np.ndarray]:
    """Cross-check local means/pair sums through three isolated core calls."""

    scaled = np.stack([item.scaled_values for item in reductions])
    expected_mass = np.stack(
        [item.cell_source_scaled_mass for item in reductions]
    )
    expected_pair_sum = np.stack(
        [
            item.cell_source_scaled_pair_squared_difference_sum
            for item in reductions
        ]
    )
    mass_relative = np.zeros(len(reductions), dtype=np.float64)
    pair_relative = np.zeros(len(reductions), dtype=np.float64)
    for source in range(3):
        source_only = np.where(
            draw.source_id[None, :] == source, scaled, 0.0
        )
        core = reduce_spatial_probability_stratified_mis(source_only, draw)
        mass_relative = np.maximum(
            mass_relative,
            _relative_max(
                core.cell_estimate,
                expected_mass[:, :, source],
            ),
        )
        pair_relative = np.maximum(
            pair_relative,
            _relative_max(
                core.cell_pair_squared_difference_sum,
                expected_pair_sum[:, :, source],
            ),
        )
    return mass_relative, pair_relative


def _row_quantization_audit_sha256(
    draw: SpatialProbabilityStratifiedMISDraw, row: int
) -> str:
    arrays = (
        ("source_tick_sum", draw.source_tick_sum[row], ">u8"),
        ("source_eligible_count", draw.source_eligible_count[row], ">u4"),
        (
            "source_positive_tick_count",
            draw.source_positive_tick_count[row],
            ">u4",
        ),
        (
            "source_minimum_eligible_tick",
            draw.source_minimum_eligible_tick[row],
            ">u8",
        ),
        (
            "source_excluded_tick_sum",
            draw.source_excluded_tick_sum[row],
            ">u8",
        ),
        (
            "source_terminal_correction",
            draw.source_terminal_correction[row],
            ">i8",
        ),
        ("source_audit_passed", draw.source_audit_passed[row], ">u1"),
        (
            "fallback_scaled_law_mismatch_count",
            draw.fallback_scaled_law_mismatch_count[row],
            ">u4",
        ),
    )
    digest = sha256()
    digest.update(b"SBSI/spatial-h256-u8-ps2-object-quantization-audit/v1\0")
    for label, values, dtype in arrays:
        digest.update(bytes.fromhex(_canonical_array_sha256(label, values, dtype)))
    return digest.hexdigest()


def diagnose_spatial_h256_u8_ps2_complement(
    likelihood,
    mock,
    proposal,
    *,
    partition,
    partition_cache_sha256: str,
    partition_manifest_sha256: str,
    partition_load_seconds: float,
    proposal_initialization_seconds: float,
    center: tuple[float, float] = (
        0.013610377120038044,
        0.002740148300964851,
    ),
    observation_start: int = 0,
    observation_stop: Optional[int] = None,
    object_id_offset: int = 25_000,
    object_chunk: int = 32,
    atom_chunk: int = 4_096,
    spatial_atom_chunk: int = 262_144,
    spatial_object_chunk: int = 32,
    candidate_backend: str = "torch",
    n_candidates: int = 16_384,
    prefilter_candidates: int = 131_072,
    proposal_seed: int = 8_701,
    production_seed: int = SPATIAL_PS_SEED,
    confirmation_mock_sha256: Optional[dict[str, str]] = None,
) -> SpatialH256U8PS2Diagnostic:
    """Run the strict centre-only H256-U8-PS2 development gate."""

    protocol = SPATIAL_H256_U8_PS2_PROTOCOL
    if not _integer_contract_consistent(protocol):
        raise ValueError("the H256-U8-PS2 integer protocol is inconsistent")
    frozen_partition_identities = {
        name: protocol[name]
        for name in (
            "partition_sha256",
            "partition_cache_sha256",
            "partition_manifest_sha256",
        )
    }
    if not all(
        _is_sha256(value) for value in frozen_partition_identities.values()
    ):
        raise ValueError("H256-U8-PS2 requires frozen partition identities")
    observed_partition_identities = {
        "partition_sha256": _partition_sha256(partition),
        "partition_cache_sha256": partition_cache_sha256,
        "partition_manifest_sha256": partition_manifest_sha256,
    }
    if observed_partition_identities != frozen_partition_identities:
        raise ValueError("H256-U8-PS2 partition identities do not match")
    if confirmation_mock_sha256 != LEARNED_RETRIEVAL_CENTER_MOCK_SHA256:
        raise ValueError("H256-U8-PS2 requires the frozen mock hashes")
    if getattr(likelihood, "selection", None) is not None:
        raise ValueError("H256-U8-PS2 requires measured selection to be disabled")
    if candidate_backend != "torch":
        raise ValueError("H256-U8-PS2 requires Torch candidate search")
    if (
        center != tuple(protocol["center"])
        or observation_start != 0
        or (observation_stop is not None and observation_stop != 512)
        or object_id_offset != 25_000
        or object_chunk != 32
        or atom_chunk != 4_096
        or spatial_atom_chunk != 262_144
        or spatial_object_chunk != 32
        or n_candidates != 16_384
        or prefilter_candidates != 131_072
        or proposal_seed != 8_701
        or production_seed != SPATIAL_PS_SEED
    ):
        raise ValueError("H256-U8-PS2 runner only accepts its frozen protocol")
    if (
        not np.isfinite(proposal_initialization_seconds)
        or proposal_initialization_seconds < 0.0
        or not np.isfinite(partition_load_seconds)
        or partition_load_seconds < 0.0
    ):
        raise ValueError("H256-U8-PS2 setup times must be finite and nonnegative")
    measurements = mock.measurements
    stop = len(measurements) if observation_stop is None else int(observation_stop)
    if len(measurements) != 512 or stop != 512:
        raise ValueError("mock must be the frozen 512-object window")

    started = time.perf_counter()
    phase_seconds = {
        name: 0.0 for name in SPATIAL_H256_U8_PS2_METHOD_TIME_COMPONENTS
    }
    phase_seconds["partition_reverification"] = 0.0
    phase_seconds["exact_oracle_flow"] = 0.0
    tick = time.perf_counter()
    partition.verify(proposal)
    phase_seconds["partition_load_or_build"] = float(partition_load_seconds)
    phase_seconds["partition_reverification"] = time.perf_counter() - tick
    counts = np.asarray(partition.counts, dtype=np.int64)
    if (
        partition.recipe_sha256 != protocol["partition_recipe_sha256"]
        or len(partition.cell_id) != protocol["active_atoms"]
        or counts.shape != (SPATIAL_CELL_COUNT,)
    ):
        raise RuntimeError("H256-U8-PS2 partition shape or recipe changed")
    minimum_cell_active_count = int(counts.min())
    maximum_cell_active_count = int(counts.max())
    four_cell_slab_counts = counts.reshape(
        SPATIAL_CELL_COUNT // SPATIAL_PS_CELLS_PER_SLAB,
        SPATIAL_PS_CELLS_PER_SLAB,
    ).sum(axis=1, dtype=np.int64)
    maximum_four_cell_slab_active_count = int(four_cell_slab_counts.max())
    if minimum_cell_active_count <= n_candidates:
        raise RuntimeError("each spatial cell must exceed the exact support width")
    if maximum_four_cell_slab_active_count > spatial_atom_chunk:
        raise RuntimeError("a four-cell H256-U8-PS2 slab exceeds its atom chunk")
    active = np.asarray(proposal.active_indices, dtype=np.int64)
    cell_order = np.argsort(partition.cell_id, kind="stable")
    ordered_atom_ids = active[cell_order]
    expected_cell_order_sha256 = _core_array_sha256(
        "ascending_cell_atom_order", ordered_atom_ids, dtype=">i8"
    )

    absolute_start = object_id_offset + observation_start
    absolute_stop = object_id_offset + stop
    selected_ids, selection_manifest = exact_oracle_object_ids(
        absolute_start, absolute_stop
    )
    if selection_manifest != protocol["exact_selection_manifest_sha256"]:
        raise RuntimeError("H256-U8-PS2 exact-oracle selection changed")
    selected_id_set = set(map(int, selected_ids))
    rows: list[dict[str, Any]] = []
    exact_records: dict[int, tuple[float, float]] = {}
    exact_flow_evaluations = 0
    exact_identity = None
    stratum_flow_evaluations = 0
    production_flow_evaluations = 0
    execution_call_provenance: list[dict[str, Any]] = []

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
        if candidates.indices.shape != (len(observed), n_candidates):
            raise RuntimeError("candidate query returned the wrong support width")
        active_positions = np.searchsorted(active, candidates.indices)
        active_valid = active_positions < len(active)
        if active_valid.any():
            active_valid[active_valid] &= (
                active[active_positions[active_valid]]
                == candidates.indices[active_valid]
            )
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
        phase_seconds["stratum_exact_flow"] += time.perf_counter() - tick
        stratum_flow_evaluations += int(candidates.indices.size)

        tick = time.perf_counter()
        draw = draw_spatial_probability_stratified_mis(
            proposal,
            candidates,
            observed,
            seed=production_seed,
            partition=partition,
            object_ids=absolute_ids,
            device=likelihood.flow_model.device,
            atom_chunk=spatial_atom_chunk,
            object_chunk=spatial_object_chunk,
            recipe=TAIL_PROPOSAL_RECIPE,
        )
        if not isinstance(draw, SpatialProbabilityStratifiedMISDraw):
            raise RuntimeError("core did not return an H256-U8-PS2 draw")
        if (
            draw.seed != production_seed
            or draw.atom_chunk != spatial_atom_chunk
            or draw.object_chunk != spatial_object_chunk
            or draw.cells_per_slab != SPATIAL_PS_CELLS_PER_SLAB
            or draw.partition_sha256 != partition.partition_sha256
            or draw.partition_assignment_sha256 != partition.assignment_sha256
            or draw.partition_counts_sha256 != partition.counts_sha256
            or draw.partition_detected_mass_sha256
            != partition.detected_mass_sha256
            or draw.partition_prior_mass_sha256 != partition.prior_mass_sha256
            or draw.partition_recipe_sha256 != partition.recipe_sha256
            or draw.cell_order_sha256 != expected_cell_order_sha256
            or not np.array_equal(draw.object_ids, absolute_ids)
            or draw.version != SPATIAL_PS_VERSION
            or draw.rng_contract != SPATIAL_PS_RNG_CONTRACT
            or draw.quantizer_contract != SPATIAL_PS_QUANTIZER_CONTRACT
            or draw.hash_contract != SPATIAL_PS_HASH_CONTRACT
            or draw.order_contract != SPATIAL_PS_ORDER_CONTRACT
        ):
            raise RuntimeError("H256-U8-PS2 draw provenance does not match")
        phase_seconds[
            "integer_cdf_construction_sampling_and_audit"
        ] += time.perf_counter() - tick

        tick = time.perf_counter()
        production_log_ratio = (
            likelihood.log_importance_weights_tensor(
                observed,
                center[0],
                center[1],
                atom_indices=draw.indices,
                proposal_probability=draw.probability,
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
        production_flow_evaluations += int(draw.indices.size)

        tick = time.perf_counter()
        reductions = [
            _reduce_h256_u8_ps2_log_values(
                production_log_ratio[row],
                draw.cell_id,
                draw.source_id,
                draw.microstratum_id,
                draw.replicate_id,
            )
            for row in range(len(observed))
        ]
        core_mass_relative, core_pair_relative = (
            _source_isolated_core_crosscheck(reductions, draw)
        )
        if (
            not np.isfinite(core_mass_relative).all()
            or not np.isfinite(core_pair_relative).all()
            or (core_mass_relative > 1.0e-13).any()
            or (core_pair_relative > 1.0e-13).any()
        ):
            raise RuntimeError(
                "source-isolated H256-U8-PS2 reduction disagrees with core"
            )
        phase_seconds["pair_reduction_and_hashing"] += (
            time.perf_counter() - tick
        )

        execution_call_provenance.append(
            {
                "object_start": int(absolute_ids[0]),
                "object_stop": int(absolute_ids[-1]) + 1,
                "objects": int(len(absolute_ids)),
                "seed": int(draw.seed),
                "atom_chunk": int(draw.atom_chunk),
                "object_chunk": int(draw.object_chunk),
                "cells_per_slab": int(draw.cells_per_slab),
                "partition_sha256": draw.partition_sha256,
                "record_sha256": draw.record_sha256,
                "schedule_sha256": draw.schedule_sha256,
                "quantization_sha256": draw.quantization_sha256,
                "cell_order_sha256": draw.cell_order_sha256,
                "sampler_version": draw.version,
                "rng_contract": draw.rng_contract,
                "quantizer_contract": draw.quantizer_contract,
                "hash_contract": draw.hash_contract,
                "order_contract": draw.order_contract,
            }
        )

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
        support_hashes = _exclusion_hashes(candidates.indices)
        source_totals = np.asarray(
            protocol["source_total_ticks"], dtype=np.uint64
        )
        scheduled_source = draw.source_id.astype(np.int64)
        target = (
            draw.microstratum_id.astype(np.uint64)
            * np.uint64(SPATIAL_H256_U8_PS2_CDF_TICKS)
        )[None, :] + draw.u48
        for row, object_id in enumerate(absolute_ids):
            indices = draw.indices[row]
            probability = draw.probability[row]
            numerator = draw.probability_numerator[row]
            selected_ticks = draw.selected_source_ticks[row]
            selected_lower = draw.selected_source_cdf_lower[row]
            scheduled_ticks = selected_ticks[
                np.arange(len(indices)), scheduled_source
            ]
            expected_probability = np.ldexp(
                numerator.astype(np.float64), -53
            )
            numerator_sum = selected_ticks.sum(axis=1, dtype=np.uint64)
            bracket_valid = (
                (selected_lower <= target[row])
                & (target[row] < selected_lower + scheduled_ticks)
                & (scheduled_ticks > 0)
            )
            tick_valid = selected_ticks <= source_totals[None, :]
            expected_u48 = _recompute_u48(
                seed=production_seed, object_id=int(object_id)
            )
            cell_lookup = _partition_cell_lookup(
                partition, proposal, indices
            )
            member = np.isin(indices, candidates.indices[row])
            if draw.object_excluded_candidate_indices_sha256[row] != support_hashes[row]:
                raise RuntimeError("core exclusion hash differs from exact support")
            if draw.object_indices_sha256[row] != _core_array_sha256(
                "object_indices", draw.indices[row : row + 1], dtype=">i8"
            ):
                raise RuntimeError("object index hash did not independently recompute")
            if draw.object_probability_sha256[row] != _core_array_sha256(
                "object_probability",
                draw.probability[row : row + 1],
                dtype=">f8",
            ):
                raise RuntimeError("object probability hash did not recompute")
            if draw.object_numerator_sha256[row] != _core_array_sha256(
                "object_probability_numerator",
                draw.probability_numerator[row : row + 1],
                dtype=">u8",
            ):
                raise RuntimeError("object numerator hash did not recompute")
            if draw.object_selected_source_ticks_sha256[row] != _core_array_sha256(
                "object_selected_source_ticks",
                draw.selected_source_ticks[row : row + 1],
                dtype=">u8",
            ):
                raise RuntimeError("object selected-tick hash did not recompute")
            if draw.object_selected_source_cdf_lower_sha256[row] != _core_array_sha256(
                "object_selected_source_cdf_lower",
                draw.selected_source_cdf_lower[row : row + 1],
                dtype=">u8",
            ):
                raise RuntimeError("object CDF-lower hash did not recompute")
            if draw.object_u48_sha256[row] != _core_array_sha256(
                "object_u48", draw.u48[row : row + 1], dtype=">u8"
            ):
                raise RuntimeError("object u48 hash did not recompute")

            reduction = reductions[row]
            mass_complement = reduction.mass
            complement_se = reduction.standard_error
            mass_stratum, log_mass_stratum = _scaled_sum(stratum_log_c[row])
            fallback = draw.proxy_fallback[row]
            fallback_exact = True
            for component, scale in enumerate((4, 3)):
                selected = fallback[:, component]
                if selected.any():
                    fallback_exact = bool(
                        fallback_exact
                        and np.all(
                            draw.source_positive_tick_count[row, :, component][
                                selected
                            ]
                            == draw.source_positive_tick_count[row, :, 2][selected]
                        )
                        and np.all(
                            draw.source_minimum_eligible_tick[
                                row, :, component
                            ][selected]
                            == scale
                            * draw.source_minimum_eligible_tick[row, :, 2][
                                selected
                            ]
                        )
                        and np.all(
                            draw.source_terminal_correction[
                                row, :, component
                            ][selected]
                            == scale
                            * draw.source_terminal_correction[row, :, 2][
                                selected
                            ]
                        )
                    )
            flow_hashes = {
                "stratum_log_target_sha256": _canonical_array_sha256(
                    "SBSI/spatial-h256-u8-ps2-stratum-log-target/v1",
                    stratum_log_c[row],
                    ">f8",
                ),
                "production_log_ratio_sha256": _canonical_array_sha256(
                    "SBSI/spatial-h256-u8-ps2-production-log-ratio/v1",
                    production_log_ratio[row],
                    ">f8",
                ),
            }
            record = {
                "object_index": int(object_id),
                "stratum_count": n_candidates,
                "stratum_unique_count": int(
                    len(np.unique(candidates.indices[row]))
                ),
                "stratum_active_count": n_candidates,
                "stratum_indices_sha256": support_hashes[row],
                **flow_hashes,
                "production_indices_sha256": draw.object_indices_sha256[row],
                "production_probability_sha256": draw.object_probability_sha256[
                    row
                ],
                "production_probability_numerator_sha256": (
                    draw.object_numerator_sha256[row]
                ),
                "production_selected_source_ticks_sha256": (
                    draw.object_selected_source_ticks_sha256[row]
                ),
                "production_selected_source_cdf_lower_sha256": (
                    draw.object_selected_source_cdf_lower_sha256[row]
                ),
                "production_u48_sha256": draw.object_u48_sha256[row],
                "production_normalizer_sha256": _canonical_array_sha256(
                    "SBSI/spatial-h256-u8-ps2-cell-normalizer/v1",
                    draw.component_log_normalizer[row],
                    ">f8",
                ),
                "production_quantization_audit_sha256": (
                    _row_quantization_audit_sha256(draw, row)
                ),
                "production_exclusion_indices_sha256": (
                    draw.object_excluded_candidate_indices_sha256[row]
                ),
                "production_schedule_sha256": draw.schedule_sha256,
                "production_cell_order_sha256": draw.cell_order_sha256,
                "production_draw_record_sha256": draw.record_sha256,
                "production_draws": int(len(indices)),
                "production_cells": int(len(np.unique(draw.cell_id))),
                "production_microstrata": int(len(indices) // 2),
                "production_source_proxy_t1": int(
                    np.sum(draw.source_id == 0)
                ),
                "production_source_proxy_t2": int(
                    np.sum(draw.source_id == 1)
                ),
                "production_source_prior": int(np.sum(draw.source_id == 2)),
                "production_draws_in_stratum": int(member.sum()),
                "canonical_schedule_exact": True,
                "integer_source_totals_exact": bool(
                    np.all(draw.source_tick_sum[row] == source_totals)
                ),
                "integer_prior_floor_satisfied": bool(
                    np.all(draw.source_minimum_eligible_tick[row, :, 2] >= 1)
                ),
                "integer_proxy_fallback_scaled_exact": fallback_exact,
                "integer_balance_probability_exact": bool(
                    np.array_equal(numerator, numerator_sum)
                    and np.array_equal(probability, expected_probability)
                ),
                "integer_cdf_brackets_exact": bool(bracket_valid.all()),
                "source_audit_failed_count": int(
                    np.count_nonzero(draw.source_audit_passed[row] != 1)
                ),
                "fallback_scaled_law_mismatch_count": int(
                    draw.fallback_scaled_law_mismatch_count[row].sum(
                        dtype=np.uint64
                    )
                ),
                "source_excluded_tick_sum": int(
                    draw.source_excluded_tick_sum[row].sum(dtype=np.uint64)
                ),
                "selected_source_tick_overflow_count": int(
                    np.count_nonzero(~tick_valid)
                ),
                "numerator_sum_failure_count": int(
                    np.count_nonzero(numerator != numerator_sum)
                ),
                "probability_identity_failure_count": int(
                    np.count_nonzero(probability != expected_probability)
                ),
                "cdf_bracket_failure_count": int(
                    np.count_nonzero(~bracket_valid)
                ),
                "rng_recomputation_failure_count": int(
                    np.count_nonzero(draw.u48[row] != expected_u48)
                ),
                "partition_cell_membership_failure_count": int(
                    np.count_nonzero(cell_lookup != draw.cell_id)
                ),
                "partition_cell_membership_exact": bool(
                    np.array_equal(cell_lookup, draw.cell_id)
                ),
                "probability_numerator_minimum": int(numerator.min()),
                "probability_numerator_maximum": int(numerator.max()),
                "u48_minimum": int(draw.u48[row].min()),
                "u48_maximum": int(draw.u48[row].max()),
                "prior_minimum_eligible_tick": int(
                    draw.source_minimum_eligible_tick[row, :, 2].min()
                ),
                "terminal_correction_maximum": int(
                    draw.source_terminal_correction[row].max()
                ),
                "terminal_correction_positive_count": int(
                    np.count_nonzero(draw.source_terminal_correction[row] > 0)
                ),
                "proxy_t1_prior_fallback_cells": int(fallback[:, 0].sum()),
                "proxy_t2_prior_fallback_cells": int(fallback[:, 1].sum()),
                "core_cell_mass_crosscheck_relative_max": float(
                    core_mass_relative[row]
                ),
                "core_cell_pair_sum_crosscheck_relative_max": float(
                    core_pair_relative[row]
                ),
                "log_mass_stratum": log_mass_stratum,
                "mass_stratum": mass_stratum,
                "log_mass_complement_spatial_h256_u8_ps2": (
                    reduction.log_mass
                ),
                "mass_complement_spatial_h256_u8_ps2": mass_complement,
                "mass_complement_spatial_h256_u8_ps2_standard_error": (
                    complement_se
                ),
                "log_variance_complement_spatial_h256_u8_ps2": (
                    reduction.log_variance
                ),
                "pareto_k_spatial_h256_u8_ps2_complement": reduction.pareto_k,
                "maximum_effective_weight_spatial_h256_u8_ps2": (
                    reduction.maximum_effective_weight
                ),
                "exact_oracle_selected": int(object_id) in selected_id_set,
            }
            if int(object_id) in exact_records:
                exact_log_stratum, exact_log_complement = exact_records[
                    int(object_id)
                ]
                exact_log_total = float(
                    np.logaddexp(exact_log_stratum, exact_log_complement)
                )
                exact_stratum = float(np.exp(exact_log_stratum))
                exact_complement = float(np.exp(exact_log_complement))
                exact_total = float(np.exp(exact_log_total))
                estimated_log_total = float(
                    np.logaddexp(log_mass_stratum, reduction.log_mass)
                )
                difference = mass_complement - exact_complement
                record.update(
                    {
                        "mass_stratum_exact_oracle": exact_stratum,
                        "mass_stratum_exact_oracle_relative_error": (
                            abs(mass_stratum - exact_stratum)
                            / abs(exact_stratum)
                            if exact_stratum != 0.0
                            else float("nan")
                        ),
                        "mass_complement_exact_oracle": exact_complement,
                        "mass_total_exact_oracle": exact_total,
                        "mass_complement_spatial_h256_u8_ps2_minus_exact_z": (
                            difference / complement_se
                            if complement_se > 0.0
                            else float("nan")
                        ),
                        "delta_log_total_evidence_spatial_h256_u8_ps2_minus_exact": (
                            estimated_log_total - exact_log_total
                        ),
                    }
                )
            rows.append(record)
        phase_seconds["pair_reduction_and_hashing"] += (
            time.perf_counter() - tick
        )

    tick = time.perf_counter()
    frame = pd.DataFrame(rows).sort_values("object_index").reset_index(drop=True)
    if len(exact_records) != 64 or exact_identity is None:
        raise RuntimeError("exact-oracle subset was not completely evaluated")
    active_atoms, atoms_visited, atom_blocks, active_hash = exact_identity
    flow_evaluations = (
        stratum_flow_evaluations
        + production_flow_evaluations
        + exact_flow_evaluations
    )
    spatial_metadata = {
        "label": SPATIAL_H256_U8_PS2_CENTER_LABEL,
        "initial_strategy": protocol["initial_strategy"],
        "partition_sha256": observed_partition_identities["partition_sha256"],
        "partition_cache_sha256": partition_cache_sha256,
        "partition_manifest_sha256": partition_manifest_sha256,
        "partition_verified": True,
        "partition_recipe_sha256": partition.recipe_sha256,
        "minimum_cell_active_count": minimum_cell_active_count,
        "maximum_cell_active_count": maximum_cell_active_count,
        "maximum_four_cell_slab_active_count": (
            maximum_four_cell_slab_active_count
        ),
        "n_cells": SPATIAL_CELL_COUNT,
        "spatial_atom_chunk": spatial_atom_chunk,
        "spatial_object_chunk": spatial_object_chunk,
        "n_global_draws": protocol["n_global_draws"],
        "proposal_seed": proposal_seed,
        "production_seed": production_seed,
        "sampler_version": SPATIAL_PS_VERSION,
        "sampler_hash_contract": SPATIAL_PS_HASH_CONTRACT,
        "rng_contract": SPATIAL_PS_RNG_CONTRACT,
        "rng_domain_word": SPATIAL_PS_RNG_DOMAIN_WORD,
        "rng_version_word": SPATIAL_PS_RNG_VERSION_WORD,
        "quantizer_contract": SPATIAL_PS_QUANTIZER_CONTRACT,
        "order_contract": SPATIAL_PS_ORDER_CONTRACT,
        "cells_per_slab": SPATIAL_PS_CELLS_PER_SLAB,
        "cycles_per_cell": protocol["cycles_per_cell"],
        "source_microstrata": list(SPATIAL_PS_MICROSTRATA),
        "source_draws_per_cell": list(SPATIAL_PS_SOURCE_COUNTS),
        "microstrata_per_cell": protocol["microstrata_per_cell"],
        "draws_per_microstratum": SPATIAL_PS_REPLICATES,
        "draws_per_cell": SPATIAL_PS_DRAWS_PER_CELL,
        "production_draws": protocol["production_draws"],
        "cdf_tick_bits": protocol["cdf_tick_bits"],
        "cdf_ticks_per_microstratum": SPATIAL_PS_TICKS_PER_MICROSTRATUM,
        "source_total_ticks": list(protocol["source_total_ticks"]),
        "balance_total_ticks": protocol["balance_total_ticks"],
        "balance_probability_law": protocol["balance_probability_law"],
        "prior_minimum_atom_ticks": protocol["prior_minimum_atom_ticks"],
        "proxy_fallback": protocol["proxy_fallback"],
        "proxy_fallback_tick_scales": list(
            protocol["proxy_fallback_tick_scales"]
        ),
        "integer_target_law": protocol["integer_target_law"],
        "pilot": deepcopy(protocol["pilot"]),
        "tail_proposal_recipe": asdict(TAIL_PROPOSAL_RECIPE),
        "proposal_initialization_seconds": float(
            proposal_initialization_seconds
        ),
        "mock_input_sha256": dict(confirmation_mock_sha256),
        "stratum_flow_evaluations": int(stratum_flow_evaluations),
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
        "run_manifest_sha256": spatial_h256_u8_ps2_run_manifest_sha256(frame),
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
    phase_seconds["pair_reduction_and_hashing"] += time.perf_counter() - tick
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
        "phase_seconds": {
            name: float(value) for name, value in phase_seconds.items()
        },
        "spatial_h256_u8_ps2_center": spatial_metadata,
        "exact_oracle": exact_metadata,
    }
    return SpatialH256U8PS2Diagnostic(frame, metadata)


__all__ = [
    "SPATIAL_H256_U8_PS2_BALANCE_TICKS",
    "SPATIAL_H256_U8_PS2_CDF_TICK_BITS",
    "SPATIAL_H256_U8_PS2_CDF_TICKS",
    "SPATIAL_H256_U8_PS2_CENTER_GATE_THRESHOLDS",
    "SPATIAL_H256_U8_PS2_CENTER_LABEL",
    "SPATIAL_H256_U8_PS2_METHOD_TIME_COMPONENTS",
    "SPATIAL_H256_U8_PS2_PROTOCOL",
    "SPATIAL_H256_U8_PS2_SOURCE_DRAWS",
    "SPATIAL_H256_U8_PS2_SOURCE_MICROSTRATA",
    "SpatialH256U8PS2Diagnostic",
    "SpatialH256U8PS2Reduction",
    "diagnose_spatial_h256_u8_ps2_complement",
    "spatial_h256_u8_ps2_center_summary",
    "spatial_h256_u8_ps2_run_manifest_sha256",
]
