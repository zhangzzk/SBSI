#!/usr/bin/env python3
"""Outcome-free, production-scale A40 preflight for H256-U8-PS2.

This command loads only the frozen proposal inputs and spatial partition.  It
constructs deterministic synthetic observations and exclusions, invokes the
proposal constructor once, validates structural invariants, and prints one
JSON object containing runtime and memory observability.  It deliberately has
no mock-catalogue, target-flow, reduction, exact-oracle, or gate path.
"""

from __future__ import annotations

import argparse
from dataclasses import fields
from hashlib import sha256
import json
from pathlib import Path
import resource
import subprocess
import time

import numpy as np
import torch

from sbsi.catalogue_likelihood import CatalogueModelCache
from sbsi.catalogue_sampling import (
    SPATIAL_CELL_COUNT,
    SPATIAL_PS_CELLS_PER_SLAB,
    SPATIAL_PS_DRAWS_PER_CELL,
    SPATIAL_PS_MICROSTRATA,
    SPATIAL_PS_REPLICATES,
    SPATIAL_PS_SEED,
    SPATIAL_PS_SOURCE_COUNTS,
    SPATIAL_PS_TICKS_PER_MICROSTRATUM,
    SPATIAL_PS_VERSION,
    DefensiveLocalProposal,
    ProposalCandidates,
    ProposalCoordinateTable,
    SpatialCellPartition,
    SpatialProbabilityStratifiedMISDraw,
)
from sbsi.scene_prior import ScenePrior


PREFLIGHT_VERSION = "spatial_h256_u8_ps2_a40_structural_v1"
PREFLIGHT_OBJECT_COUNT = 32
PREFLIGHT_EXCLUSIONS_PER_OBJECT = 16_384
PREFLIGHT_OBJECT_ID_BASE = 1 << 40
PREFLIGHT_OBSERVATION_SEED = 20_260_830
PREFLIGHT_EXCLUSION_SEED = 20_260_831
PREFLIGHT_OBJECT_CHUNK = 32
PREFLIGHT_ATOM_CHUNK = 262_144
PREFLIGHT_RNG_CONTRACT = "pcg64_seedsequence_seed_object_lo_hi_v1"
PREFLIGHT_PARTITION_SHA256 = "0da871a79abfd124360a2d93f02707bb88de42e5281edfda741a819a58b557ab"
PREFLIGHT_PARTITION_CACHE_SHA256 = "1a63e4a7e435f79fa319f1a909698ee117bcb6394a257e0258ace9daf0248e3c"
PREFLIGHT_PARTITION_MANIFEST_SHA256 = "7cd8ae16a65c8488b5ff6c8abfeef5edbdce75b0035dcb37ad83fa51ff29cdef"

ZERO_SCIENCE_COUNTERS = {
    "candidate_queries": 0,
    "exact_oracle_evaluations": 0,
    "gate_evaluations": 0,
    "mock_rows_loaded": 0,
    "reducer_evaluations": 0,
    "target_flow_evaluations": 0,
}

PREFLIGHT_CORE_CONTRACT = {
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-store", type=Path, required=True)
    parser.add_argument("--model-cache", type=Path, required=True)
    parser.add_argument("--proposal-cache", type=Path, required=True)
    parser.add_argument("--partition-cache", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser


def _object_rng(seed: int, object_id: int) -> np.random.Generator:
    object_id = int(object_id)
    return np.random.Generator(
        np.random.PCG64(
            np.random.SeedSequence(
                [
                    int(seed),
                    object_id & 0xFFFF_FFFF,
                    (object_id >> 32) & 0xFFFF_FFFF,
                ]
            )
        )
    )


def _unique_random_positions(rng: np.random.Generator, population: int, count: int) -> np.ndarray:
    """Draw a fixed-order unique sample without an O(population) workspace."""

    if not 0 < int(count) < int(population):
        raise ValueError("unique sample requires 0 < count < population")
    selected: list[int] = []
    seen: set[int] = set()
    while len(selected) < count:
        remaining = count - len(selected)
        candidates = rng.integers(
            0,
            population,
            size=max(1_024, 2 * remaining),
            dtype=np.int64,
        )
        for candidate in candidates:
            value = int(candidate)
            if value in seen:
                continue
            seen.add(value)
            selected.append(value)
            if len(selected) == count:
                break
    return np.asarray(selected, dtype=np.int64)


def _synthetic_inputs(
    coordinates: ProposalCoordinateTable,
    active_indices: np.ndarray,
) -> tuple[np.ndarray, ProposalCandidates, np.ndarray]:
    active = np.asarray(active_indices, dtype=np.int64)
    if active.ndim != 1 or len(active) <= PREFLIGHT_EXCLUSIONS_PER_OBJECT or (np.diff(active) <= 0).any():
        raise ValueError("preflight requires ordered production-size active support")

    object_ids = PREFLIGHT_OBJECT_ID_BASE + np.arange(PREFLIGHT_OBJECT_COUNT, dtype=np.int64)
    observations = np.empty(
        (PREFLIGHT_OBJECT_COUNT, len(coordinates.target_names)),
        dtype=np.float64,
    )
    exclusions = np.empty(
        (PREFLIGHT_OBJECT_COUNT, PREFLIGHT_EXCLUSIONS_PER_OBJECT),
        dtype=np.int64,
    )
    for row, object_id in enumerate(object_ids):
        observation_rng = _object_rng(PREFLIGHT_OBSERVATION_SEED, object_id)
        standardized = np.clip(
            observation_rng.standard_normal(len(coordinates.target_names)),
            -2.0,
            2.0,
        )
        observations[row] = coordinates.center + standardized * coordinates.scale

        exclusion_rng = _object_rng(PREFLIGHT_EXCLUSION_SEED, object_id)
        positions = _unique_random_positions(
            exclusion_rng,
            len(active),
            PREFLIGHT_EXCLUSIONS_PER_OBJECT,
        )
        exclusions[row] = active[positions]

    candidates = ProposalCandidates(
        indices=exclusions,
        distances=np.zeros(exclusions.shape, dtype=np.float64),
        radius=np.zeros(PREFLIGHT_OBJECT_COUNT, dtype=np.float64),
    )
    return observations, candidates, object_ids


def _draw_array_bytes(draw: SpatialProbabilityStratifiedMISDraw) -> int:
    return int(
        sum(
            value.nbytes
            for item in fields(draw)
            if isinstance((value := getattr(draw, item.name)), np.ndarray)
        )
    )


def _validate_structural_draw(
    draw: SpatialProbabilityStratifiedMISDraw,
    candidates: ProposalCandidates,
    object_ids: np.ndarray,
) -> None:
    expected_shape = (
        PREFLIGHT_OBJECT_COUNT,
        SPATIAL_CELL_COUNT * SPATIAL_PS_DRAWS_PER_CELL,
    )
    if not isinstance(draw, SpatialProbabilityStratifiedMISDraw):
        raise TypeError("preflight constructor returned the wrong draw type")
    if draw.indices.shape != expected_shape:
        raise RuntimeError("preflight draw shape changed")
    if (
        draw.version != SPATIAL_PS_VERSION
        or draw.seed != SPATIAL_PS_SEED
        or draw.cells_per_slab != SPATIAL_PS_CELLS_PER_SLAB
        or draw.atom_chunk != PREFLIGHT_ATOM_CHUNK
        or draw.object_chunk != PREFLIGHT_OBJECT_CHUNK
        or not np.array_equal(draw.object_ids, object_ids)
    ):
        raise RuntimeError("preflight draw contract changed")

    expected_totals = np.asarray(SPATIAL_PS_MICROSTRATA, dtype=np.uint64) * np.uint64(
        SPATIAL_PS_TICKS_PER_MICROSTRATUM
    )
    if (
        not np.all(draw.source_tick_sum == expected_totals)
        or not draw.source_audit_passed.all()
        or draw.source_excluded_tick_sum.any()
        or draw.source_terminal_correction.any()
        or draw.fallback_scaled_law_mismatch_count.any()
        or not np.array_equal(
            draw.probability_numerator,
            draw.selected_source_ticks.sum(axis=2, dtype=np.uint64),
        )
        or not np.array_equal(
            draw.probability,
            np.ldexp(draw.probability_numerator.astype(np.float64), -53),
        )
    ):
        raise RuntimeError("preflight integer-law invariant failed")

    for row in range(PREFLIGHT_OBJECT_COUNT):
        if np.isin(draw.indices[row], candidates.indices[row]).any():
            raise RuntimeError("preflight draw leaked into its exclusion set")


def _repository_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _verify_partition_artifact(partition_root: Path) -> Path:
    partition_path = partition_root / "spatial_partition.npz"
    manifest_path = partition_root / "manifest.json"
    if sha256(partition_path.read_bytes()).hexdigest() != (PREFLIGHT_PARTITION_CACHE_SHA256):
        raise RuntimeError("preflight partition cache bytes changed")
    manifest_bytes = manifest_path.read_bytes()
    if sha256(manifest_bytes).hexdigest() != PREFLIGHT_PARTITION_MANIFEST_SHA256:
        raise RuntimeError("preflight partition manifest bytes changed")
    manifest = json.loads(manifest_bytes)
    expected = {
        "partition_cache_sha256": PREFLIGHT_PARTITION_CACHE_SHA256,
        "partition_sha256": PREFLIGHT_PARTITION_SHA256,
        "n_cells": SPATIAL_CELL_COUNT,
        "science_rows_read": 0,
        "candidate_queries": 0,
        "target_flow_evaluations": 0,
    }
    if any(manifest.get(name) != value for name, value in expected.items()):
        raise RuntimeError("preflight partition manifest contract changed")
    return partition_path


def _verify_core_contract() -> None:
    observed = {
        "sampler_version": SPATIAL_PS_VERSION,
        "cells": SPATIAL_CELL_COUNT,
        "cells_per_slab": SPATIAL_PS_CELLS_PER_SLAB,
        "source_microstrata": SPATIAL_PS_MICROSTRATA,
        "source_draws_per_cell": SPATIAL_PS_SOURCE_COUNTS,
        "replicates": SPATIAL_PS_REPLICATES,
        "draws_per_cell": SPATIAL_PS_DRAWS_PER_CELL,
        "ticks_per_microstratum": SPATIAL_PS_TICKS_PER_MICROSTRATUM,
        "sampler_seed": SPATIAL_PS_SEED,
    }
    if observed != PREFLIGHT_CORE_CONTRACT:
        raise RuntimeError("frozen H256-U8-PS2 preflight contract changed")


def run_preflight(args: argparse.Namespace) -> dict:
    _verify_core_contract()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the production-scale preflight requires CUDA")
    device_index = torch.cuda.current_device() if device.index is None else device.index
    device = torch.device("cuda", device_index)
    properties = torch.cuda.get_device_properties(device)
    if "A40" not in properties.name.upper():
        raise RuntimeError(f"the frozen preflight requires an A40, got {properties.name}")

    phase_seconds: dict[str, float] = {}
    total_started = time.perf_counter()

    started = time.perf_counter()
    prior = ScenePrior.load(args.scene_store)
    phase_seconds["scene_store_load"] = time.perf_counter() - started

    started = time.perf_counter()
    cache = CatalogueModelCache.load(args.model_cache, prior=prior)
    coordinates = ProposalCoordinateTable.load(args.proposal_cache)
    phase_seconds["proposal_inputs_load"] = time.perf_counter() - started

    started = time.perf_counter()
    proposal = DefensiveLocalProposal(
        coordinates,
        prior.weights,
        local_base_weights=cache.get(0.0, 0.0).detection_probability,
        build_coordinate_tree=False,
    )
    phase_seconds["proposal_initialize"] = time.perf_counter() - started
    if proposal.tree is not None:
        raise RuntimeError("preflight unexpectedly built a coordinate tree")

    started = time.perf_counter()
    partition_path = _verify_partition_artifact(args.partition_cache)
    partition = SpatialCellPartition.load(partition_path, proposal=proposal)
    phase_seconds["partition_load_verify"] = time.perf_counter() - started
    proposal._spatial_cell_partition = partition

    counts = np.asarray(partition.counts, dtype=np.uint64)
    slab_counts = counts.reshape(-1, SPATIAL_PS_CELLS_PER_SLAB).sum(axis=1)
    if len(proposal.active_indices) != int(counts.sum()) or int(slab_counts.max()) > PREFLIGHT_ATOM_CHUNK:
        raise RuntimeError("partition geometry is incompatible with the preflight")

    started = time.perf_counter()
    observations, candidates, object_ids = _synthetic_inputs(coordinates, proposal.active_indices)
    phase_seconds["synthetic_inputs"] = time.perf_counter() - started

    started = time.perf_counter()
    proposal._tilted_proxy(device)
    # Materialize the CUDA context and float64 GEMM machinery before resetting
    # peak accounting.  The timed constructor still includes its cold spatial
    # order/inverse cache and exclusion bucketing.
    warmup = torch.ones((1, 1), dtype=torch.float64, device=device)
    warmup @ warmup
    torch.cuda.synchronize(device)
    phase_seconds["proxy_device_initialize"] = time.perf_counter() - started
    if hasattr(proposal, "_spatial_ps_cell_order"):
        raise RuntimeError("preflight spatial-order cache was not cold")

    cuda_baseline_allocated = int(torch.cuda.memory_allocated(device))
    cuda_baseline_reserved = int(torch.cuda.memory_reserved(device))
    torch.cuda.reset_peak_memory_stats(device)

    started = time.perf_counter()
    draw = proposal.draw_spatial_probability_stratified_mis(
        candidates,
        observations,
        seed=SPATIAL_PS_SEED,
        partition=partition,
        object_ids=object_ids,
        device=device,
        atom_chunk=PREFLIGHT_ATOM_CHUNK,
        object_chunk=PREFLIGHT_OBJECT_CHUNK,
    )
    torch.cuda.synchronize(device)
    phase_seconds["constructor"] = time.perf_counter() - started

    cuda_peak_allocated = int(torch.cuda.max_memory_allocated(device))
    cuda_peak_reserved = int(torch.cuda.max_memory_reserved(device))
    cuda_end_allocated = int(torch.cuda.memory_allocated(device))
    cuda_end_reserved = int(torch.cuda.memory_reserved(device))

    started = time.perf_counter()
    _validate_structural_draw(draw, candidates, object_ids)
    phase_seconds["structural_validation"] = time.perf_counter() - started
    phase_seconds["total"] = time.perf_counter() - total_started

    cuda_free, cuda_total = torch.cuda.mem_get_info(device)
    return {
        "preflight_version": PREFLIGHT_VERSION,
        "sampler_version": SPATIAL_PS_VERSION,
        "partition_sha256": partition.partition_sha256,
        "repository_commit": _repository_commit(),
        "outcome_free": True,
        "device": {
            "name": properties.name,
            "capability": [properties.major, properties.minor],
            "total_bytes": int(cuda_total),
            "free_bytes_after": int(cuda_free),
        },
        "workload": {
            "objects": PREFLIGHT_OBJECT_COUNT,
            "exclusions_per_object": PREFLIGHT_EXCLUSIONS_PER_OBJECT,
            "draws_per_cell": SPATIAL_PS_DRAWS_PER_CELL,
            "draws_per_object": SPATIAL_CELL_COUNT * SPATIAL_PS_DRAWS_PER_CELL,
            "cells": SPATIAL_CELL_COUNT,
            "cells_per_slab": SPATIAL_PS_CELLS_PER_SLAB,
            "source_draws_per_cell": list(SPATIAL_PS_SOURCE_COUNTS),
            "replicates": SPATIAL_PS_REPLICATES,
            "object_chunk": PREFLIGHT_OBJECT_CHUNK,
            "atom_chunk": PREFLIGHT_ATOM_CHUNK,
            "sampler_seed": SPATIAL_PS_SEED,
            "observation_seed": PREFLIGHT_OBSERVATION_SEED,
            "exclusion_seed": PREFLIGHT_EXCLUSION_SEED,
            "object_id_base": PREFLIGHT_OBJECT_ID_BASE,
            "rng_contract": PREFLIGHT_RNG_CONTRACT,
        },
        "geometry": {
            "active_atoms": len(proposal.active_indices),
            "cell_count_minimum": int(counts.min()),
            "cell_count_median": float(np.median(counts)),
            "cell_count_maximum": int(counts.max()),
            "four_cell_slab_count_minimum": int(slab_counts.min()),
            "four_cell_slab_count_median": float(np.median(slab_counts)),
            "four_cell_slab_count_maximum": int(slab_counts.max()),
        },
        "phase_seconds": phase_seconds,
        "memory": {
            "cuda_baseline_allocated_bytes": cuda_baseline_allocated,
            "cuda_baseline_reserved_bytes": cuda_baseline_reserved,
            "cuda_peak_allocated_bytes": cuda_peak_allocated,
            "cuda_peak_reserved_bytes": cuda_peak_reserved,
            "cuda_incremental_peak_allocated_bytes": max(0, cuda_peak_allocated - cuda_baseline_allocated),
            "cuda_incremental_peak_reserved_bytes": max(0, cuda_peak_reserved - cuda_baseline_reserved),
            "cuda_end_allocated_bytes": cuda_end_allocated,
            "cuda_end_reserved_bytes": cuda_end_reserved,
            "retained_draw_array_bytes": _draw_array_bytes(draw),
            "process_max_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1_024,
        },
        "structural_invariant_failures": 0,
        "science_counters": dict(ZERO_SCIENCE_COUNTERS),
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = run_preflight(args)
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
