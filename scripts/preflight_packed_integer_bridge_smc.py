#!/usr/bin/env python3
"""Outcome-free production-shape preflight for the packed integer bridge.

The command binds the real full-catalogue prior and zero-shear detection arrays
to the exact global integer base, draws the frozen production slot geometry,
and exercises the packed bridge with deterministic synthetic targets.  It
loads no mock, performs no candidate query or target-flow evaluation, invokes
no likelihood reducer or exact oracle, and evaluates no science gate.
"""

from __future__ import annotations

import argparse
from dataclasses import fields
import gc
from hashlib import sha256
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch

from sbsi.catalogue_likelihood import CatalogueModelCache
from sbsi.catalogue_sampling import DefensiveLocalProposal, ProposalCoordinateTable
from sbsi.integer_mixture_base import IntegerMixtureBase, IntegerMixtureBaseDraw
from sbsi.packed_integer_bridge_smc import (
    PACKED_BRIDGE_BETAS,
    PACKED_BRIDGE_MOVE_STAGES,
    PACKED_BRIDGE_ROUNDS,
    PACKED_BRIDGE_STAGES,
    PACKED_BRIDGE_TICKS,
    PACKED_BRIDGE_VERSION,
    PackedIntegerBridgeSMCResult,
    run_packed_integer_bridge_smc,
)
from sbsi.scene_prior import ScenePrior


PREFLIGHT_VERSION = "packed_integer_bridge_smc_a40_structural_v1"
PREFLIGHT_OBJECT_COUNT = 32
PREFLIGHT_PRODUCTION_ATOMS = 12_760_990
PREFLIGHT_EXCLUSIONS_PER_OBJECT = 16_384
PREFLIGHT_ISLANDS = 16
PREFLIGHT_PARTICLES = 256
PREFLIGHT_SLOTS_PER_OBJECT = (
    PREFLIGHT_ISLANDS * PACKED_BRIDGE_ROUNDS * PREFLIGHT_PARTICLES
)
PREFLIGHT_OBJECT_ID_BASE = 1 << 40
PREFLIGHT_BASE_SEED = 7_008_705
PREFLIGHT_BASE_DOMAIN = 0x425249444745
PREFLIGHT_BRIDGE_SEED = 7_008_706
PREFLIGHT_EXCLUSION_SEED = 20_260_832
PREFLIGHT_OBJECT_REPLAY_CHUNK = 7
PREFLIGHT_ISLAND_REPLAY_CHUNK = 5
PREFLIGHT_CONSTANT_LOG_RATIO = float(np.log(3.0))
PREFLIGHT_SYNTHETIC_EVEN_MULTIPLIER = 0.5
PREFLIGHT_SYNTHETIC_ODD_MULTIPLIER = 1.5
PREFLIGHT_SYNTHETIC_MAX_ABS_Z = 8.0
PREFLIGHT_SYNTHETIC_RMS_Z = 3.0
PREFLIGHT_MERGE_REPORT_SHA256 = (
    "797747617121b980c0f221a858021b509acef1b95017a3264d08967e905e3af6"
)
PREFLIGHT_SOURCE_SHA256 = {
    "scene_store": {
        "manifest.json": "dea4295c836b0970f3b0d26adbd992ef3d1a05be6cfa7e0b6c2f73521a809002",
        "galaxies.parquet": "fae16fcdb6ec8756f315188896ad241bbd450fe85e759412767ae8a3a4f2f968",
        "neighbours.npz": "4403f2bc57be671e4509c7151458eb3f0981f907d116b344e70d6fd9691c88ac",
    },
    "model_cache": {
        "manifest.json": "40129ac029c350ecd53f11a1462001926fcc9a5293ded231eb94aa9a572525a7",
        "flow_zero.parquet": "50b8842ba0f7c2f3c61c6f09ff4c68d83d6ebc69295bd895bc2e4e05d6cdd070",
        "detection_zero.parquet": "b98bdde6ea6685ee5be942d7d3495d48862f8f938693b3b0cfab26463cc70f98",
    },
    "proposal_cache": {
        "manifest.json": "a3a037752b1e8dba3195134d8f4a57e634ca5f52f21b51ed0715077bfef15bc6",
        "coordinates.npz": "634ff5b5c87cd2a415bc37ed7cb758e8968d1612d3f69b83f4daa8c287a704cf",
    },
}
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

ZERO_SCIENCE_COUNTERS = {
    "candidate_queries": 0,
    "catalogue_target_evaluations": 0,
    "exact_oracle_evaluations": 0,
    "gate_evaluations": 0,
    "mock_rows_loaded": 0,
    "reducer_evaluations": 0,
    "target_flow_evaluations": 0,
}

PREFLIGHT_CORE_CONTRACT = {
    "base_version": "integer_mixture_base_v1",
    "bridge_version": "packed_integer_bridge_smc_v1",
    "bridge_stages": 8,
    "move_stages": 7,
    "proposal_rounds": 8,
    "resampling_ticks": 1 << 48,
    "objects": 32,
    "production_atoms": 12_760_990,
    "exclusions_per_object": 16_384,
    "islands": 16,
    "particles": 256,
    "slots_per_object": 32_768,
    "base_seed": 7_008_705,
    "base_domain": 0x425249444745,
    "bridge_seed": 7_008_706,
    "exclusion_seed": 20_260_832,
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-store", type=Path, required=True)
    parser.add_argument("--model-cache", type=Path, required=True)
    parser.add_argument("--proposal-cache", type=Path, required=True)
    parser.add_argument("--integer-base-cache", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser


def _git(*arguments: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(REPOSITORY_ROOT), *arguments],
        check=check,
        capture_output=True,
        text=True,
    )


def _repository_identity() -> dict[str, object]:
    commit = _git("rev-parse", "--verify", "HEAD").stdout.strip()
    branch = _git("symbolic-ref", "-q", "--short", "HEAD", check=False)
    status = _git("status", "--porcelain", "--untracked-files=normal").stdout
    detached = branch.returncode != 0
    clean = status == ""
    if not detached or not clean:
        raise RuntimeError("preflight requires a clean detached repository checkout")
    return {
        "commit": commit,
        "detached": detached,
        "clean": clean,
        "path": str(REPOSITORY_ROOT),
    }


def _canonical_report_bytes(report: dict) -> bytes:
    return (
        json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _write_report(path: Path, report: dict) -> bytes:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError("preflight report path already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_report_bytes(report)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.tmp"
    )
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return payload


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_source_artifacts(args: argparse.Namespace) -> dict[str, object]:
    roots = {
        "scene_store": Path(args.scene_store),
        "model_cache": Path(args.model_cache),
        "proposal_cache": Path(args.proposal_cache),
    }
    parents = {root.resolve().parent for root in roots.values()}
    if len(parents) != 1:
        raise RuntimeError("source artifacts do not share one compact-global root")
    compact_root = parents.pop()
    merge_report_path = compact_root / "merge_report.json"
    if _file_sha256(merge_report_path) != PREFLIGHT_MERGE_REPORT_SHA256:
        raise RuntimeError("compact-global merge report identity changed")
    try:
        merge_report = json.loads(merge_report_path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("compact-global merge report is invalid") from error
    if (
        merge_report.get("status") != "complete"
        or merge_report.get("n_positive_prior_atoms") != PREFLIGHT_PRODUCTION_ATOMS
        or merge_report.get("output_sha256") != PREFLIGHT_SOURCE_SHA256
    ):
        raise RuntimeError("compact-global merge report contract changed")

    result: dict[str, object] = {
        "compact_root": str(compact_root),
        "merge_report_sha256": PREFLIGHT_MERGE_REPORT_SHA256,
    }
    for name, root in roots.items():
        observed_files: dict[str, str] = {}
        for relative, expected in PREFLIGHT_SOURCE_SHA256[name].items():
            path = root / relative
            if not path.is_file():
                raise RuntimeError(f"missing {name} payload {relative}")
            observed = _file_sha256(path)
            if observed != expected:
                raise RuntimeError(f"{name} payload identity changed: {relative}")
            observed_files[relative] = observed
        result[name] = {
            "path": str(root.resolve()),
            "files": observed_files,
        }
    return result


def _array_sha256(label: str, values, *, dtype: str) -> str:
    array = np.ascontiguousarray(values, dtype=np.dtype(dtype))
    digest = sha256()
    digest.update(b"SBSI/packed-bridge-preflight-array/v1\0")
    digest.update(label.encode("utf-8"))
    digest.update(b"\0")
    digest.update(np.asarray(array.shape, dtype=np.dtype(">u8")).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _record_sha256(label: str, values) -> str:
    digest = sha256()
    digest.update(b"SBSI/packed-bridge-preflight-record/v1\0")
    digest.update(label.encode("utf-8"))
    digest.update(b"\0")
    for value in values:
        encoded = str(value).encode("ascii")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _progress(phase: str, **values) -> None:
    print(
        json.dumps(
            {"preflight_progress": str(phase), **values},
            sort_keys=True,
            allow_nan=False,
        ),
        file=sys.stderr,
        flush=True,
    )


def _object_rng(seed: int, object_id: int) -> np.random.Generator:
    value = int(object_id)
    return np.random.Generator(
        np.random.PCG64(
            np.random.SeedSequence(
                [
                    int(seed),
                    value & 0xFFFF_FFFF,
                    (value >> 32) & 0xFFFF_FFFF,
                ]
            )
        )
    )


def _unique_random_positions(
    rng: np.random.Generator, population: int, count: int
) -> np.ndarray:
    """Return an ordered unique sample without an O(population) workspace."""

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


def _synthetic_exclusions(
    active_indices: np.ndarray,
    object_ids: np.ndarray,
    *,
    count: int = PREFLIGHT_EXCLUSIONS_PER_OBJECT,
    seed: int = PREFLIGHT_EXCLUSION_SEED,
) -> np.ndarray:
    active = np.asarray(active_indices, dtype=np.int64)
    ids = np.asarray(object_ids, dtype=np.uint64)
    if (
        active.ndim != 1
        or len(active) <= count
        or (np.diff(active) <= 0).any()
        or ids.ndim != 1
        or not len(ids)
    ):
        raise ValueError("synthetic exclusions require ordered active support")
    result = np.empty((len(ids), count), dtype=np.int64)
    for row, object_id in enumerate(ids):
        positions = _unique_random_positions(
            _object_rng(seed, int(object_id)), len(active), count
        )
        result[row] = np.sort(active[positions])
    return result


def _synthetic_log_ratio(
    atom_ids: np.ndarray,
    exclusions: np.ndarray,
    *,
    base: IntegerMixtureBase,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, int]]:
    """Form ``log(c)-log(q0)`` once per unique atom and scatter by slot."""

    atoms = np.asarray(atom_ids, dtype=np.int64)
    excluded = np.asarray(exclusions, dtype=np.int64)
    if atoms.ndim != 3 or (atoms < 0).any():
        raise ValueError("synthetic target requires nonnegative packed atom ids")
    if excluded.ndim != 1 or (np.diff(excluded) <= 0).any():
        raise ValueError("synthetic target exclusions must be sorted and unique")
    unique, inverse = np.unique(atoms.reshape(-1), return_inverse=True)
    if unique[-1] >= base.n_atoms or not base.active[unique].all():
        raise ValueError("synthetic target atom ids leave the integer-base support")
    numerator = (
        7 * base.detected_prior_ticks[unique] + base.raw_prior_ticks[unique]
    )
    probability = np.ldexp(numerator.astype(np.float64), -51)
    if (numerator == 0).any() or (probability <= 0.0).any():
        raise RuntimeError("synthetic target encountered zero marginal q0")
    unique_device = torch.as_tensor(unique, dtype=torch.int64, device=device)
    probability_device = torch.as_tensor(
        probability, dtype=torch.float64, device=device
    )
    log_q0 = torch.log(probability_device)
    multiplier = torch.where(
        torch.remainder(unique_device, 2) == 0,
        torch.as_tensor(
            PREFLIGHT_SYNTHETIC_EVEN_MULTIPLIER,
            dtype=torch.float64,
            device=device,
        ),
        torch.as_tensor(
            PREFLIGHT_SYNTHETIC_ODD_MULTIPLIER,
            dtype=torch.float64,
            device=device,
        ),
    )
    log_c = log_q0 + torch.log(multiplier)
    unique_ratio = log_c - log_q0
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    unique_ratio_cpu = unique_ratio.cpu().numpy()
    unique_zero = np.isin(unique, excluded, assume_unique=True)
    unique_ratio_cpu[unique_zero] = -np.inf
    ratio = unique_ratio_cpu[inverse].reshape(atoms.shape)
    ratio = np.ascontiguousarray(ratio, dtype=np.float64)
    zero_slots = int(np.count_nonzero(np.isneginf(ratio)))
    return ratio, {
        "requested_slots": int(atoms.size),
        "unique_evaluations": int(len(unique)),
        "marginal_probability_lookups": int(len(unique)),
        "cache_hits": int(atoms.size - len(unique)),
        "zero_unique_atoms": int(np.count_nonzero(unique_zero)),
        "zero_slots": zero_slots,
    }


def _parity_probability_numerators(
    base: IntegerMixtureBase,
) -> tuple[int, int]:
    detected_even = int(
        base.detected_prior_ticks[::2].sum(dtype=np.uint64)
    )
    prior_even = int(base.raw_prior_ticks[::2].sum(dtype=np.uint64))
    even = 7 * detected_even + prior_even
    total = 1 << 51
    if not 0 < even < total:
        raise RuntimeError("synthetic parity control requires both parities")
    return even, total - even


def _known_synthetic_normalizer(
    base: IntegerMixtureBase,
    exclusions: np.ndarray,
    *,
    parity_numerator: tuple[int, int],
) -> float:
    excluded = np.asarray(exclusions, dtype=np.int64)
    if excluded.ndim != 1 or (np.diff(excluded) <= 0).any():
        raise ValueError("known synthetic target requires sorted exclusions")
    numerator = (
        7 * base.detected_prior_ticks[excluded]
        + base.raw_prior_ticks[excluded]
    )
    even_excluded = int(numerator[excluded % 2 == 0].sum(dtype=np.uint64))
    odd_excluded = int(numerator[excluded % 2 == 1].sum(dtype=np.uint64))
    even = int(parity_numerator[0]) - even_excluded
    odd = int(parity_numerator[1]) - odd_excluded
    if even < 0 or odd < 0:
        raise RuntimeError("synthetic exclusion mass exceeded its parity source")
    # Multipliers 1/2 and 3/2 make this exactly (even + 3*odd)/2**52.
    return float(np.ldexp(float(even + 3 * odd), -52))


def _draw_array_bytes(draw: IntegerMixtureBaseDraw) -> int:
    return int(
        sum(
            value.nbytes
            for item in fields(draw)
            if isinstance((value := getattr(draw, item.name)), np.ndarray)
        )
    )


def _result_array_bytes(result: PackedIntegerBridgeSMCResult) -> int:
    return int(
        sum(
            value.nbytes
            for item in fields(result)
            if isinstance((value := getattr(result, item.name)), np.ndarray)
        )
    )


def _verify_core_contract() -> None:
    observed = {
        "base_version": "integer_mixture_base_v1",
        "bridge_version": PACKED_BRIDGE_VERSION,
        "bridge_stages": PACKED_BRIDGE_STAGES,
        "move_stages": PACKED_BRIDGE_MOVE_STAGES,
        "proposal_rounds": PACKED_BRIDGE_ROUNDS,
        "resampling_ticks": PACKED_BRIDGE_TICKS,
        "objects": PREFLIGHT_OBJECT_COUNT,
        "production_atoms": PREFLIGHT_PRODUCTION_ATOMS,
        "exclusions_per_object": PREFLIGHT_EXCLUSIONS_PER_OBJECT,
        "islands": PREFLIGHT_ISLANDS,
        "particles": PREFLIGHT_PARTICLES,
        "slots_per_object": PREFLIGHT_SLOTS_PER_OBJECT,
        "base_seed": PREFLIGHT_BASE_SEED,
        "base_domain": PREFLIGHT_BASE_DOMAIN,
        "bridge_seed": PREFLIGHT_BRIDGE_SEED,
        "exclusion_seed": PREFLIGHT_EXCLUSION_SEED,
    }
    if observed != PREFLIGHT_CORE_CONTRACT:
        raise RuntimeError("frozen packed-bridge preflight contract changed")
    if not np.array_equal(PACKED_BRIDGE_BETAS, np.arange(9) / 8.0):
        raise RuntimeError("frozen packed-bridge beta grid changed")


def _load_or_build_base(
    cache_path: Path,
    raw_prior: np.ndarray,
    detected_prior: np.ndarray,
) -> tuple[IntegerMixtureBase, str, dict[str, float]]:
    phases: dict[str, float] = {}
    cache_path = Path(cache_path)
    if cache_path.is_dir() and any(cache_path.iterdir()):
        _progress("integer_base_load_started", cache=str(cache_path))
        started = time.perf_counter()
        base = IntegerMixtureBase.load(cache_path)
        phases["integer_base_load"] = time.perf_counter() - started
        mode = "loaded"
    elif cache_path.exists():
        raise RuntimeError("integer-base cache path exists but is not complete")
    else:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = cache_path.parent / f".{cache_path.name}.build.lock"
        try:
            lock_path.mkdir()
        except FileExistsError as error:
            raise RuntimeError("integer-base cache build lock already exists") from error
        staging_path: Path | None = None
        try:
            if cache_path.exists():
                raise RuntimeError("integer-base cache appeared during lock acquisition")
            staging_path = Path(
                tempfile.mkdtemp(
                    prefix=f".{cache_path.name}.staging-",
                    dir=cache_path.parent,
                )
            )
            _progress("integer_base_construct_started", cache=str(cache_path))
            started = time.perf_counter()
            built = IntegerMixtureBase.from_weights(raw_prior, detected_prior)
            phases["integer_base_construct"] = time.perf_counter() - started
            law_sha256 = built.law_sha256
            started = time.perf_counter()
            built.save(staging_path)
            phases["integer_base_staged_save"] = time.perf_counter() - started
            del built
            gc.collect()
            started = time.perf_counter()
            staged = IntegerMixtureBase.load(staging_path)
            phases["integer_base_staged_verify"] = time.perf_counter() - started
            if staged.law_sha256 != law_sha256:
                raise RuntimeError("staged integer base changed its law identity")
            del staged
            gc.collect()
            os.rename(staging_path, cache_path)
            staging_path = None
            _progress("integer_base_published", cache=str(cache_path))
            started = time.perf_counter()
            base = IntegerMixtureBase.load(cache_path)
            phases["integer_base_published_reload"] = time.perf_counter() - started
            if base.law_sha256 != law_sha256:
                raise RuntimeError("published integer base changed its law identity")
            mode = "constructed"
        finally:
            if staging_path is not None and staging_path.is_dir():
                expected_prefix = f".{cache_path.name}.staging-"
                if (
                    staging_path.parent != cache_path.parent
                    or not staging_path.name.startswith(expected_prefix)
                ):
                    raise RuntimeError("refusing to clean an unexpected staging path")
                shutil.rmtree(staging_path)
            lock_path.rmdir()
    started = time.perf_counter()
    base.verify_weights(raw_prior, detected_prior)
    phases["integer_base_live_weight_verify"] = time.perf_counter() - started
    return base, mode, phases


def _validate_draw(
    draw: IntegerMixtureBaseDraw,
    base: IntegerMixtureBase,
    object_ids: np.ndarray,
) -> None:
    expected_shape = (PREFLIGHT_OBJECT_COUNT, PREFLIGHT_SLOTS_PER_OBJECT)
    if draw.indices.shape != expected_shape:
        raise RuntimeError("integer-base preflight draw shape changed")
    draw.verify(base)
    if (
        not np.array_equal(draw.object_ids, object_ids)
        or draw.seed != PREFLIGHT_BASE_SEED
        or draw.domain != PREFLIGHT_BASE_DOMAIN
        or draw.draw_offset != 0
        or draw.n_atoms != base.n_atoms
        or draw.version != PREFLIGHT_CORE_CONTRACT["base_version"]
        or draw.base_sha256 != base.law_sha256
        or not np.array_equal(
            draw.source_tick_sum,
            np.full(2, PACKED_BRIDGE_TICKS, dtype=np.uint64),
        )
        or not draw.source_audit_passed.all()
        or draw.source_off_support_tick_sum.any()
        or (draw.indices < 0).any()
        or (draw.indices >= base.n_atoms).any()
        or not base.active[draw.indices].all()
        or not np.array_equal(
            draw.probability,
            np.ldexp(draw.probability_numerator.astype(np.float64), -51),
        )
    ):
        raise RuntimeError("integer-base preflight law invariant failed")
    packed = draw.indices.reshape(
        PREFLIGHT_OBJECT_COUNT,
        PREFLIGHT_ISLANDS,
        PACKED_BRIDGE_ROUNDS,
        PREFLIGHT_PARTICLES,
    )
    if not np.array_equal(packed.reshape(draw.indices.shape), draw.indices):
        raise RuntimeError("integer-base canonical slot reshape changed")


def _validate_draw_replay(
    base: IntegerMixtureBase,
    draw: IntegerMixtureBaseDraw,
    object_ids: np.ndarray,
) -> dict[str, int]:
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
    replayed = 0
    for start in range(0, len(object_ids), PREFLIGHT_OBJECT_REPLAY_CHUNK):
        stop = min(start + PREFLIGHT_OBJECT_REPLAY_CHUNK, len(object_ids))
        chunk = base.sample(
            object_ids[start:stop],
            PREFLIGHT_SLOTS_PER_OBJECT,
            seed=PREFLIGHT_BASE_SEED,
            domain=PREFLIGHT_BASE_DOMAIN,
        )
        chunk.verify(base)
        if any(
            not np.array_equal(getattr(chunk, name), getattr(draw, name)[start:stop])
            for name in names
        ):
            raise RuntimeError("integer-base object chunk replay changed a draw")
        replayed += stop - start

    permutation = np.asarray([2, 0, 1], dtype=np.int64)
    reordered = base.sample(
        object_ids[permutation],
        PREFLIGHT_SLOTS_PER_OBJECT,
        seed=PREFLIGHT_BASE_SEED,
        domain=PREFLIGHT_BASE_DOMAIN,
    )
    reordered.verify(base)
    if any(
        not np.array_equal(
            getattr(reordered, name), getattr(draw, name)[permutation]
        )
        for name in names
    ):
        raise RuntimeError("integer-base object order replay changed a draw")
    return {"chunk_replayed_objects": replayed, "order_replayed_objects": 3}


def _validate_constant_control(result: PackedIntegerBridgeSMCResult) -> None:
    if (
        not np.isclose(result.normalizer, 3.0, rtol=5e-13, atol=0.0)
        or not np.isclose(
            result.log_normalizer,
            PREFLIGHT_CONSTANT_LOG_RATIO,
            rtol=0.0,
            atol=5e-13,
        )
        or result.variance != 0.0
        or result.standard_error != 0.0
        or result.collapsed_stage.any()
        or not np.allclose(
            result.stage_effective_sample_size,
            PREFLIGHT_PARTICLES,
            rtol=0.0,
            atol=1e-10,
        )
        or not np.all(result.stage_acceptance_rate == 1.0)
        or not np.array_equal(
            result.stage_atom_ids[:, -1], result.stage_atom_ids[:, -2]
        )
        or result.proposal_slot_count != PREFLIGHT_SLOTS_PER_OBJECT
        or result.target_finite_slot_count != PREFLIGHT_SLOTS_PER_OBJECT
        or result.target_zero_slot_count != 0
        or not result.mh_moved.any()
    ):
        raise RuntimeError("packed-bridge constant-target control failed")


def _mixed_collapse_control(*, object_id: int) -> PackedIntegerBridgeSMCResult:
    shape = (
        PREFLIGHT_ISLANDS,
        PACKED_BRIDGE_ROUNDS,
        PREFLIGHT_PARTICLES,
    )
    # This pathwise structural fixture uses unique IDs so its forced
    # island-specific zero/live split cannot assign inconsistent targets to a
    # naturally duplicated q0 atom. It is not a proposal-law validation run.
    atoms = np.arange(PREFLIGHT_SLOTS_PER_OBJECT, dtype=np.int64).reshape(shape)
    ratio = np.zeros(shape, dtype=np.float64)
    ratio[: PREFLIGHT_ISLANDS // 2] = -np.inf
    result = run_packed_integer_bridge_smc(
        atoms,
        ratio,
        seed=PREFLIGHT_BRIDGE_SEED,
        object_id=object_id,
        island_ids=np.arange(PREFLIGHT_ISLANDS, dtype=np.int64),
        island_chunk=PREFLIGHT_ISLANDS,
    )
    expected_collapse = np.concatenate(
        (
            np.ones(PREFLIGHT_ISLANDS // 2, dtype=np.int64),
            np.zeros(PREFLIGHT_ISLANDS // 2, dtype=np.int64),
        )
    )
    if (
        not np.array_equal(result.collapsed_stage, expected_collapse)
        or not np.isclose(result.normalizer, 0.5, rtol=0.0, atol=5e-13)
        or not np.isclose(
            result.standard_error,
            np.sqrt(1.0 / 60.0),
            rtol=0.0,
            atol=5e-13,
        )
    ):
        raise RuntimeError("packed-bridge mixed-collapse control failed")
    return result


def _summary(values) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise RuntimeError("preflight summary received invalid values")
    return {
        "minimum": float(array.min()),
        "median": float(np.median(array)),
        "maximum": float(array.max()),
    }


def run_preflight(args: argparse.Namespace) -> dict:
    phases: dict[str, float] = {}
    total_started = time.perf_counter()
    started = time.perf_counter()
    _verify_core_contract()
    repository_identity = _repository_identity()
    phases["repository_identity_verify"] = time.perf_counter() - started
    started = time.perf_counter()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the production-shape preflight requires CUDA")
    device_index = torch.cuda.current_device() if device.index is None else device.index
    device = torch.device("cuda", device_index)
    properties = torch.cuda.get_device_properties(device)
    if "A40" not in properties.name.upper():
        raise RuntimeError(f"the frozen preflight requires an A40, got {properties.name}")
    phases["cuda_identity_verify"] = time.perf_counter() - started
    started = time.perf_counter()
    source_artifacts = _verify_source_artifacts(args)
    phases["source_payload_verify"] = time.perf_counter() - started

    _progress("proposal_inputs_load_started")
    started = time.perf_counter()
    prior = ScenePrior.load(args.scene_store)
    phases["scene_store_load"] = time.perf_counter() - started
    started = time.perf_counter()
    cache = CatalogueModelCache.load(args.model_cache, prior=prior)
    coordinates = ProposalCoordinateTable.load(args.proposal_cache)
    proposal = DefensiveLocalProposal(
        coordinates,
        prior.weights,
        local_base_weights=cache.get(0.0, 0.0).detection_probability,
        build_coordinate_tree=False,
    )
    phases["proposal_inputs_load_initialize"] = time.perf_counter() - started
    if proposal.tree is not None:
        raise RuntimeError("preflight unexpectedly built a coordinate tree")
    raw_prior = proposal.prior_weights
    detected_prior = proposal.local_base_weights
    active_indices = proposal.active_indices.copy()

    del cache, coordinates, proposal, prior
    gc.collect()

    base, base_mode, base_phases = _load_or_build_base(
        args.integer_base_cache, raw_prior, detected_prior
    )
    phases.update(base_phases)
    if not np.array_equal(np.flatnonzero(base.active), active_indices):
        raise RuntimeError("integer base active order changed from proposal inputs")
    active_count = int(np.count_nonzero(base.active))
    if (
        base.n_atoms != PREFLIGHT_PRODUCTION_ATOMS
        or active_count != PREFLIGHT_PRODUCTION_ATOMS
    ):
        raise RuntimeError("preflight did not bind the frozen production atom count")
    del raw_prior, detected_prior
    gc.collect()
    _progress("integer_base_ready", mode=base_mode, atoms=base.n_atoms)

    object_ids = (
        PREFLIGHT_OBJECT_ID_BASE
        + np.arange(PREFLIGHT_OBJECT_COUNT, dtype=np.uint64)
    )
    started = time.perf_counter()
    exclusions = _synthetic_exclusions(active_indices, object_ids)
    phases["synthetic_exclusions"] = time.perf_counter() - started
    exclusion_q_mass = np.empty(PREFLIGHT_OBJECT_COUNT, dtype=np.float64)
    parity_numerator = _parity_probability_numerators(base)
    known_synthetic_normalizer = np.empty(
        PREFLIGHT_OBJECT_COUNT, dtype=np.float64
    )
    for row in range(PREFLIGHT_OBJECT_COUNT):
        atoms = exclusions[row]
        numerator = (
            7 * base.detected_prior_ticks[atoms] + base.raw_prior_ticks[atoms]
        )
        exclusion_q_mass[row] = np.ldexp(
            float(numerator.sum(dtype=np.uint64)), -51
        )
        known_synthetic_normalizer[row] = _known_synthetic_normalizer(
            base,
            exclusions[row],
            parity_numerator=parity_numerator,
        )

    started = time.perf_counter()
    draw = base.sample(
        object_ids,
        PREFLIGHT_SLOTS_PER_OBJECT,
        seed=PREFLIGHT_BASE_SEED,
        domain=PREFLIGHT_BASE_DOMAIN,
    )
    phases["integer_base_draw"] = time.perf_counter() - started
    started = time.perf_counter()
    _validate_draw(draw, base, object_ids)
    phases["integer_base_draw_validate"] = time.perf_counter() - started
    started = time.perf_counter()
    replay_report = _validate_draw_replay(base, draw, object_ids)
    phases["integer_base_draw_replay"] = time.perf_counter() - started
    _progress("integer_base_draw_replay_complete", objects=PREFLIGHT_OBJECT_COUNT)

    warmup = torch.ones((1, 1), dtype=torch.float64, device=device)
    warmup @ warmup
    torch.cuda.synchronize(device)
    cuda_baseline_allocated = int(torch.cuda.memory_allocated(device))
    cuda_baseline_reserved = int(torch.cuda.memory_reserved(device))
    torch.cuda.reset_peak_memory_stats(device)

    first_atoms = draw.indices[0].reshape(
        PREFLIGHT_ISLANDS, PACKED_BRIDGE_ROUNDS, PREFLIGHT_PARTICLES
    )
    constant_ratio = np.full(
        first_atoms.shape, PREFLIGHT_CONSTANT_LOG_RATIO, dtype=np.float64
    )
    started = time.perf_counter()
    constant_result = run_packed_integer_bridge_smc(
        first_atoms,
        constant_ratio,
        seed=PREFLIGHT_BRIDGE_SEED,
        object_id=int(object_ids[0]),
        island_ids=np.arange(PREFLIGHT_ISLANDS, dtype=np.int64),
        island_chunk=PREFLIGHT_ISLANDS,
    )
    phases["constant_control_kernel"] = time.perf_counter() - started
    _validate_constant_control(constant_result)

    started = time.perf_counter()
    mixed_collapse_result = _mixed_collapse_control(
        object_id=int(object_ids[0])
    )
    phases["mixed_collapse_control_kernel"] = time.perf_counter() - started

    started = time.perf_counter()
    ratios = np.empty(draw.indices.shape, dtype=np.float64)
    synthetic_accounting: list[dict[str, int]] = []
    for row, object_id in enumerate(object_ids):
        packed_atoms = draw.indices[row].reshape(
            PREFLIGHT_ISLANDS,
            PACKED_BRIDGE_ROUNDS,
            PREFLIGHT_PARTICLES,
        )
        packed_ratio, accounting = _synthetic_log_ratio(
            packed_atoms,
            exclusions[row],
            base=base,
            device=device,
        )
        ratios[row] = packed_ratio.reshape(-1)
        synthetic_accounting.append(accounting)
    torch.cuda.synchronize(device)
    phases["synthetic_unique_target_pack_scatter"] = time.perf_counter() - started

    island_ids = np.arange(PREFLIGHT_ISLANDS, dtype=np.int64)
    object_kernel_seconds: list[float] = []
    object_normalizer: list[float] = []
    object_standard_error: list[float] = []
    object_known_z_score: list[float] = []
    object_record_sha256: list[str] = []
    object_proposal_sha256: list[str] = []
    stage_acceptance_sum = np.zeros(PACKED_BRIDGE_MOVE_STAGES, dtype=np.float64)
    stage_movement_sum = np.zeros(PACKED_BRIDGE_MOVE_STAGES, dtype=np.float64)
    stage_ess_sum = np.zeros(PACKED_BRIDGE_STAGES, dtype=np.float64)
    final_lineage: list[float] = []
    total_collapsed_islands = 0
    partially_collapsed_objects = 0
    all_collapsed_objects = 0
    first_result: PackedIntegerBridgeSMCResult | None = None
    started_all_kernels = time.perf_counter()
    for row, object_id in enumerate(object_ids):
        packed_atoms = draw.indices[row].reshape(
            PREFLIGHT_ISLANDS,
            PACKED_BRIDGE_ROUNDS,
            PREFLIGHT_PARTICLES,
        )
        packed_ratio = ratios[row].reshape(packed_atoms.shape)
        started = time.perf_counter()
        result = run_packed_integer_bridge_smc(
            packed_atoms,
            packed_ratio,
            seed=PREFLIGHT_BRIDGE_SEED,
            object_id=int(object_id),
            island_ids=island_ids,
            island_chunk=PREFLIGHT_ISLANDS,
        )
        object_kernel_seconds.append(time.perf_counter() - started)
        accounting = synthetic_accounting[row]
        if (
            result.proposal_slot_count != PREFLIGHT_SLOTS_PER_OBJECT
            or result.target_finite_slot_count
            != PREFLIGHT_SLOTS_PER_OBJECT - accounting["zero_slots"]
            or result.target_zero_slot_count != accounting["zero_slots"]
            or result.target_unique_atom_count != accounting["unique_evaluations"]
            or result.target_cache_hit_count != accounting["cache_hits"]
            or not np.isfinite(result.log_normalizer)
            or result.normalizer <= 0.0
            or result.standard_error < 0.0
        ):
            raise RuntimeError("synthetic packed-bridge accounting failed")
        delta = result.normalizer - known_synthetic_normalizer[row]
        if abs(delta) > max(
            PREFLIGHT_SYNTHETIC_MAX_ABS_Z * result.standard_error,
            5e-12,
        ):
            raise RuntimeError("synthetic packed-bridge known-Z check failed")
        z_scale = max(result.standard_error, 5e-13)
        object_known_z_score.append(delta / z_scale)
        object_normalizer.append(result.normalizer)
        object_standard_error.append(result.standard_error)
        object_record_sha256.append(result.record_sha256)
        object_proposal_sha256.append(result.proposal_atom_ids_sha256)
        stage_acceptance_sum += result.stage_acceptance_rate.mean(axis=0)
        stage_movement_sum += result.stage_movement_rate.mean(axis=0)
        stage_ess_sum += result.stage_effective_sample_size.mean(axis=0)
        final_lineage.extend(result.stage_unique_lineage_count[:, -1].tolist())
        collapsed = int(np.count_nonzero(result.collapsed_stage))
        total_collapsed_islands += collapsed
        partially_collapsed_objects += int(0 < collapsed < PREFLIGHT_ISLANDS)
        all_collapsed_objects += int(collapsed == PREFLIGHT_ISLANDS)
        if row == 0:
            first_result = result
        if (row + 1) % 8 == 0:
            _progress("synthetic_bridge_progress", completed_objects=row + 1)
    phases["synthetic_bridge_kernels"] = time.perf_counter() - started_all_kernels
    if first_result is None or not np.any(stage_movement_sum > 0.0):
        raise RuntimeError("synthetic bridge recorded no actual movement")
    known_z_score = np.asarray(object_known_z_score, dtype=np.float64)
    if (
        np.max(np.abs(known_z_score)) > PREFLIGHT_SYNTHETIC_MAX_ABS_Z
        or float(np.sqrt(np.mean(np.square(known_z_score))))
        > PREFLIGHT_SYNTHETIC_RMS_Z
    ):
        raise RuntimeError("synthetic packed-bridge known-Z ensemble check failed")

    started = time.perf_counter()
    first_ratio = ratios[0].reshape(first_atoms.shape)
    split_result = run_packed_integer_bridge_smc(
        first_atoms,
        first_ratio,
        seed=PREFLIGHT_BRIDGE_SEED,
        object_id=int(object_ids[0]),
        island_ids=island_ids,
        island_chunk=PREFLIGHT_ISLAND_REPLAY_CHUNK,
    )
    reverse = np.arange(PREFLIGHT_ISLANDS - 1, -1, -1)
    reordered_result = run_packed_integer_bridge_smc(
        first_atoms[reverse],
        first_ratio[reverse],
        seed=PREFLIGHT_BRIDGE_SEED,
        object_id=int(object_ids[0]),
        island_ids=island_ids[reverse],
        island_chunk=PREFLIGHT_ISLAND_REPLAY_CHUNK,
    )
    if (
        split_result.record_sha256 != first_result.record_sha256
        or split_result.trace_sha256 != first_result.trace_sha256
        or split_result.execution_sha256 == first_result.execution_sha256
        or reordered_result.record_sha256 != split_result.record_sha256
        or reordered_result.execution_sha256 != split_result.execution_sha256
        or not np.array_equal(
            reordered_result.stage_atom_ids, split_result.stage_atom_ids
        )
        or not np.array_equal(
            reordered_result.stage_log_unnormalized_weight,
            split_result.stage_log_unnormalized_weight,
        )
    ):
        raise RuntimeError("packed-bridge island order/chunk replay failed")
    phases["bridge_island_order_chunk_replay"] = time.perf_counter() - started

    cuda_peak_allocated = int(torch.cuda.max_memory_allocated(device))
    cuda_peak_reserved = int(torch.cuda.max_memory_reserved(device))
    cuda_end_allocated = int(torch.cuda.memory_allocated(device))
    cuda_end_reserved = int(torch.cuda.memory_reserved(device))
    cuda_free, cuda_total = torch.cuda.mem_get_info(device)

    started = time.perf_counter()
    manifest_path = args.integer_base_cache / "manifest.json"
    arrays_path = args.integer_base_cache / "arrays.npz"
    integer_base_arrays_file_sha256 = _file_sha256(arrays_path)
    integer_base_manifest_file_sha256 = _file_sha256(manifest_path)
    requested_slots = sum(item["requested_slots"] for item in synthetic_accounting)
    unique_evaluations = sum(
        item["unique_evaluations"] for item in synthetic_accounting
    )
    cache_hits = sum(item["cache_hits"] for item in synthetic_accounting)
    zero_slots = sum(item["zero_slots"] for item in synthetic_accounting)
    if requested_slots != PREFLIGHT_OBJECT_COUNT * PREFLIGHT_SLOTS_PER_OBJECT:
        raise RuntimeError("synthetic target slot total changed")

    exclusion_sha256 = _array_sha256(
        "synthetic_exact_stratum", exclusions, dtype=">i8"
    )
    ratio_sha256 = _array_sha256(
        "synthetic_target_log_ratio", ratios, dtype=">f8"
    )
    draw_object_record_sha256 = [
        _record_sha256(
            "integer_draw_object",
            (
                int(object_ids[row]),
                _array_sha256(
                    "indices", draw.indices[row], dtype=">i8"
                ),
                _array_sha256(
                    "source", draw.source[row], dtype=">u1"
                ),
                _array_sha256("u48", draw.u48[row], dtype=">u8"),
                _array_sha256(
                    "probability_numerator",
                    draw.probability_numerator[row],
                    dtype=">u8",
                ),
            ),
        )
        for row in range(PREFLIGHT_OBJECT_COUNT)
    ]
    source_identity_sha256 = [
        str(source_artifacts["merge_report_sha256"])
    ]
    for source_name in sorted(PREFLIGHT_SOURCE_SHA256):
        source_record = source_artifacts[source_name]
        if not isinstance(source_record, dict):
            raise RuntimeError("source-artifact report schema changed")
        files = source_record.get("files")
        if not isinstance(files, dict):
            raise RuntimeError("source-artifact file report schema changed")
        source_identity_sha256.extend(
            str(files[name]) for name in sorted(files)
        )
    run_manifest_sha256 = _record_sha256(
        "preflight_run",
        (
            repository_identity["commit"],
            *source_identity_sha256,
            base.law_sha256,
            integer_base_arrays_file_sha256,
            integer_base_manifest_file_sha256,
            draw.record_sha256,
            exclusion_sha256,
            ratio_sha256,
            constant_result.record_sha256,
            mixed_collapse_result.record_sha256,
            *draw_object_record_sha256,
            *object_record_sha256,
        ),
    )
    if _repository_identity() != repository_identity:
        raise RuntimeError("repository identity changed during preflight")
    phases["final_provenance_hashing"] = time.perf_counter() - started

    report = {
        "preflight_version": PREFLIGHT_VERSION,
        "repository": repository_identity,
        "run_manifest_sha256": run_manifest_sha256,
        "outcome_free": True,
        "bridge_version": PACKED_BRIDGE_VERSION,
        "source_artifacts": source_artifacts,
        "integer_base": {
            "mode": base_mode,
            "n_atoms": base.n_atoms,
            "active_atoms": active_count,
            "law_sha256": base.law_sha256,
            "audit_sha256": base.audit_sha256,
            "arrays_file_sha256": integer_base_arrays_file_sha256,
            "manifest_file_sha256": integer_base_manifest_file_sha256,
            "raw_prior_terminal_correction": base.raw_prior_terminal_correction,
            "detected_prior_terminal_correction": (
                base.detected_prior_terminal_correction
            ),
        },
        "device": {
            "name": properties.name,
            "capability": [properties.major, properties.minor],
            "total_bytes": int(cuda_total),
            "free_bytes_after": int(cuda_free),
        },
        "runtime": {
            "python_executable": sys.executable,
            "python_version": sys.version,
            "numpy_version": np.__version__,
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
        },
        "scope": {
            "production_target_evaluation_exercised": False,
            "production_likelihood_integration_exercised": False,
            "gpu_scope": "synthetic unique marginal-q0 log-ratio formation only",
            "cpu_scope": "exact base draw/replay and packed bridge kernel",
            "dense_object_by_catalogue_target_allocated": False,
        },
        "workload": {
            "objects": PREFLIGHT_OBJECT_COUNT,
            "exclusions_per_object": PREFLIGHT_EXCLUSIONS_PER_OBJECT,
            "islands": PREFLIGHT_ISLANDS,
            "particles_per_island": PREFLIGHT_PARTICLES,
            "proposal_rounds": PACKED_BRIDGE_ROUNDS,
            "proposal_slots_per_object": PREFLIGHT_SLOTS_PER_OBJECT,
            "proposal_slots_total": requested_slots,
            "base_seed": PREFLIGHT_BASE_SEED,
            "base_domain": PREFLIGHT_BASE_DOMAIN,
            "bridge_seed": PREFLIGHT_BRIDGE_SEED,
            "exclusion_seed": PREFLIGHT_EXCLUSION_SEED,
            "object_id_base": PREFLIGHT_OBJECT_ID_BASE,
            "object_replay_chunk": PREFLIGHT_OBJECT_REPLAY_CHUNK,
            "island_replay_chunk": PREFLIGHT_ISLAND_REPLAY_CHUNK,
        },
        "integer_draw": {
            "record_sha256": draw.record_sha256,
            "indices_sha256": draw.indices_sha256,
            "probability_sha256": draw.probability_sha256,
            "source_counts": [
                int(np.count_nonzero(draw.source == source)) for source in (0, 1)
            ],
            "retained_array_bytes": _draw_array_bytes(draw),
            "object_record_sha256": draw_object_record_sha256,
            **replay_report,
        },
        "synthetic_target": {
            "contract": (
                "unique_atom_parity_multiplier_half_threehalves_"
                "logc_minus_exact_marginal_logq_v1"
            ),
            "requested_slots": requested_slots,
            "unique_evaluations": unique_evaluations,
            "marginal_probability_lookups": sum(
                item["marginal_probability_lookups"]
                for item in synthetic_accounting
            ),
            "cache_hits": cache_hits,
            "zero_slots": zero_slots,
            "exact_stratum_sha256": exclusion_sha256,
            "exclusion_q_mass": _summary(exclusion_q_mass),
            "known_normalizer": _summary(known_synthetic_normalizer),
            "known_z_score": {
                **_summary(object_known_z_score),
                "rms": float(
                    np.sqrt(np.mean(np.square(object_known_z_score)))
                ),
                "maximum_absolute_gate": PREFLIGHT_SYNTHETIC_MAX_ABS_Z,
                "rms_gate": PREFLIGHT_SYNTHETIC_RMS_Z,
            },
            "ratio_sha256": ratio_sha256,
        },
        "constant_control": {
            "normalizer": constant_result.normalizer,
            "log_normalizer": constant_result.log_normalizer,
            "variance": constant_result.variance,
            "standard_error": constant_result.standard_error,
            "record_sha256": constant_result.record_sha256,
            "actual_movement_rate": float(
                constant_result.stage_movement_rate.mean()
            ),
        },
        "mixed_collapse_control": {
            "contract": "unique_id_pathwise_reduction_fixture_v1",
            "normalizer": mixed_collapse_result.normalizer,
            "standard_error": mixed_collapse_result.standard_error,
            "collapsed_islands": int(
                np.count_nonzero(mixed_collapse_result.collapsed_stage)
            ),
            "record_sha256": mixed_collapse_result.record_sha256,
        },
        "bridge_summary": {
            "normalizer": _summary(object_normalizer),
            "standard_error": _summary(object_standard_error),
            "kernel_seconds_per_object": _summary(object_kernel_seconds),
            "collapsed_islands": total_collapsed_islands,
            "partially_collapsed_objects": partially_collapsed_objects,
            "all_collapsed_objects": all_collapsed_objects,
            "independent_island_se_valid_objects": (
                PREFLIGHT_OBJECT_COUNT - all_collapsed_objects
            ),
            "stage_mean_effective_sample_size": (
                stage_ess_sum / PREFLIGHT_OBJECT_COUNT
            ).tolist(),
            "stage_mean_acceptance_rate": (
                stage_acceptance_sum / PREFLIGHT_OBJECT_COUNT
            ).tolist(),
            "stage_mean_actual_movement_rate": (
                stage_movement_sum / PREFLIGHT_OBJECT_COUNT
            ).tolist(),
            "final_unique_lineage_count": _summary(final_lineage),
            "object_record_sha256": object_record_sha256,
            "object_proposal_sha256": object_proposal_sha256,
            "replay_record_sha256": split_result.record_sha256,
            "retained_first_result_array_bytes": _result_array_bytes(first_result),
        },
        "phase_seconds": phases,
        "memory": {
            "cuda_baseline_allocated_bytes": cuda_baseline_allocated,
            "cuda_baseline_reserved_bytes": cuda_baseline_reserved,
            "cuda_peak_allocated_bytes": cuda_peak_allocated,
            "cuda_peak_reserved_bytes": cuda_peak_reserved,
            "cuda_incremental_peak_allocated_bytes": max(
                0, cuda_peak_allocated - cuda_baseline_allocated
            ),
            "cuda_incremental_peak_reserved_bytes": max(
                0, cuda_peak_reserved - cuda_baseline_reserved
            ),
            "cuda_end_allocated_bytes": cuda_end_allocated,
            "cuda_end_reserved_bytes": cuda_end_reserved,
            "process_max_rss_bytes": int(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            )
            * 1_024,
        },
        "structural_invariant_failures": 0,
        "science_counters": dict(ZERO_SCIENCE_COUNTERS),
        "science_counter_semantics": (
            "scope declarations: this preflight imports no catalogue target, "
            "mock, candidate-query, reducer, oracle, or science-gate path"
        ),
    }
    phases["total_before_report_serialization"] = (
        time.perf_counter() - total_started
    )
    return report


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.report.exists():
        raise FileExistsError("preflight report path already exists")
    report = run_preflight(args)
    payload = _write_report(args.report, report)
    _progress(
        "preflight_complete",
        report=str(args.report),
        run_manifest_sha256=report["run_manifest_sha256"],
    )
    sys.stdout.buffer.write(payload)
    sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
