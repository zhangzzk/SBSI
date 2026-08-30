"""Packed exact-law bridge SMC for pre-evaluated independent proposals.

This module is deliberately catalogue-agnostic.  A caller supplies eight IID
``q0`` proposal rounds and their target log ratios ``log(c / q0)`` in canonical
``(island, round, particle)`` arrays.  Round zero initializes the particles.
After bridge stages one through seven, particles are resampled and moved with
the corresponding proposal round.  Stage eight applies the exact terminal
weight and performs no resampling or mutation.

The seven provisional intermediate bridges are integer laws

``gamma_t(x) = q0(x) H_t(log(c(x) / q0(x))),  t = 1, ..., 7``.

For every finite target log ratio, ``H_t`` is a strictly positive signed-int64
score.  A true zero target (``-inf`` log ratio) has score zero.  The frozen
prototype mapping is

``H_t(l) = max(1, floor(2**24 * exp(clip(t*l/8, -24, 24))))``.

The scale and clipping affect only intermediate mixing.  The incremental
potentials ``H_1``, ``H_t/H_{t-1}``, and the exact final ``exp(l)/H_7``
telescope to ``exp(l)`` for every finite target state.

Resampling uses an exact ``2**48``-tick realized categorical law.  Every
positive carried weight receives at least one tick.  If a parent's realized
resampling probability is ``r_a``, its child carries

``U_child = U_parent / (P * r_a)``.

This auxiliary correction makes the conditional expected unnormalized
empirical measure identical to the pre-resampling measure even though the
integer resampling probabilities approximate the normalized float weights.
Independence-MH acceptance is the exact integer ratio
``min(1, H_t(y) / H_t(x))`` using unbiased PCG64 ``randbelow``.  Thus the move
is invariant for the realized integer bridge, conditional on the caller's IID
``q0`` proposal contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Optional

import numpy as np


PACKED_BRIDGE_VERSION = "packed_integer_bridge_smc_v1"
PACKED_BRIDGE_RNG_CONTRACT = (
    "pcg64_seedsequence_seed_object_island_stage_particle_domain_version_v1"
)
PACKED_BRIDGE_SCORE_CONTRACT = "exp_clip24_scale2p24_floor_positive_int64_v1"
PACKED_BRIDGE_RESAMPLING_CONTRACT = (
    "cdf_2p48_positive_tick_auxiliary_inverse_probability_v1"
)
PACKED_BRIDGE_HASH_CONTRACT = "packed_integer_bridge_sha256_v1"
PACKED_BRIDGE_INPUT_CONTRACT = "iid_q0_eight_preproposed_rounds_v1"

PACKED_BRIDGE_ROUNDS = 8
PACKED_BRIDGE_STAGES = 8
PACKED_BRIDGE_MOVE_STAGES = 7
PACKED_BRIDGE_TICKS = 1 << 48
PACKED_BRIDGE_SCORE_SCALE = 1 << 24
PACKED_BRIDGE_LOG_CLIP = 24.0
PACKED_BRIDGE_BETAS = np.arange(9, dtype=np.float64) / 8.0
PACKED_BRIDGE_BETAS.setflags(write=False)

_RNG_VERSION_WORD = 1
_DOMAIN_RESAMPLE = 0x52534D50
_DOMAIN_MH_ACCEPT = 0x4D484143
_UINT64_MODULUS = 1 << 64


def _exact_nonnegative_integer(value, name: str, maximum: int) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or int(value) < 0
        or int(value) > maximum
    ):
        raise ValueError(f"{name} must be a non-negative integer <= {maximum}")
    return int(value)


def _positive_integer(value, name: str) -> int:
    result = _exact_nonnegative_integer(value, name, np.iinfo(np.int64).max)
    if result == 0:
        raise ValueError(f"{name} must be positive")
    return result


def _u64_words(value: int) -> tuple[int, int]:
    return value & 0xFFFF_FFFF, (value >> 32) & 0xFFFF_FFFF


def _keyed_rng(
    *,
    seed: int,
    object_id: int,
    island_id: int,
    stage: int,
    domain: int,
    particle: int = 0,
) -> np.random.Generator:
    seed_low, seed_high = _u64_words(seed)
    object_low, object_high = _u64_words(object_id)
    island_low, island_high = _u64_words(island_id)
    words = [
        seed_low,
        seed_high,
        object_low,
        object_high,
        island_low,
        island_high,
        int(stage),
        int(particle),
        int(domain),
        _RNG_VERSION_WORD,
    ]
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(words)))


def _array_sha256(label: str, values: np.ndarray, *, dtype: str) -> str:
    canonical = np.ascontiguousarray(values, dtype=np.dtype(dtype))
    shape = np.ascontiguousarray(canonical.shape, dtype=np.dtype(">u8"))
    digest = sha256()
    digest.update(b"SBSI/packed-integer-bridge-array/v1\0")
    digest.update(label.encode("utf-8"))
    digest.update(b"\0")
    digest.update(shape.tobytes())
    digest.update(canonical.tobytes())
    return digest.hexdigest()


def _text_sha256(label: str, value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    digest = sha256()
    digest.update(b"SBSI/packed-integer-bridge-text/v1\0")
    digest.update(label.encode("utf-8"))
    digest.update(b"\0")
    digest.update(payload)
    return digest.hexdigest()


def _record_sha256(label: str, values: tuple[str, ...]) -> str:
    digest = sha256()
    digest.update(b"SBSI/packed-integer-bridge-record/v1\0")
    digest.update(label.encode("utf-8"))
    digest.update(b"\0")
    for value in values:
        encoded = value.encode("ascii")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _validate_sha256(value: str, name: str) -> str:
    result = str(value)
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return result


def integer_bridge_score(log_ratio, stage: int) -> np.ndarray:
    """Map target log ratios to the frozen provisional integer bridge score.

    Stage zero is the base law and returns score one for every state, including
    states whose final target mass is zero.  Stages one through seven use the
    frozen mapping documented in the module docstring.  Stage eight is the
    exact target and therefore has no integer-score representation.
    """

    stage = _exact_nonnegative_integer(stage, "stage", PACKED_BRIDGE_MOVE_STAGES)
    values = np.asarray(log_ratio, dtype=np.float64)
    if np.isnan(values).any() or np.isposinf(values).any():
        raise ValueError("log ratios may be finite or -inf, but not NaN/+inf")
    if stage == 0:
        return np.ones(values.shape, dtype=np.int64)

    result = np.zeros(values.shape, dtype=np.int64)
    finite = np.isfinite(values)
    if finite.any():
        beta = float(stage) / 8.0
        exponent = np.clip(
            beta * values[finite],
            -PACKED_BRIDGE_LOG_CLIP,
            PACKED_BRIDGE_LOG_CLIP,
        )
        score_float = np.floor(
            float(PACKED_BRIDGE_SCORE_SCALE) * np.exp(exponent, dtype=np.float64)
        )
        maximum = np.iinfo(np.int64).max
        if (
            not np.isfinite(score_float).all()
            or (score_float > maximum).any()
        ):
            raise RuntimeError("integer bridge score exceeded signed int64")
        result[finite] = np.maximum(score_float, 1.0).astype(np.int64)
    return result


def _logsumexp(values: np.ndarray) -> float:
    maximum = float(np.max(values))
    if np.isneginf(maximum):
        return -np.inf
    return maximum + float(
        np.log(np.exp(values - maximum, dtype=np.float64).sum(dtype=np.float64))
    )


def _logmeanexp(values: np.ndarray) -> float:
    total = _logsumexp(values)
    return total - float(np.log(len(values))) if np.isfinite(total) else -np.inf


def _effective_sample_size(log_weight: np.ndarray) -> float:
    log_total = _logsumexp(log_weight)
    if np.isneginf(log_total):
        return 0.0
    log_square_total = _logsumexp(2.0 * log_weight)
    return float(np.exp(2.0 * log_total - log_square_total))


@dataclass(frozen=True)
class _ResamplingLaw:
    ticks: np.ndarray
    terminal_correction: int
    positive_count: int
    minimum_positive_tick: int


def _quantize_resampling_log_weights(log_weight: np.ndarray) -> _ResamplingLaw:
    """Return the exact 2**48 realized resampling law.

    One tick is reserved for every positive unnormalized particle weight.  The
    remaining ticks use cumulative-floor apportionment of max-shifted float64
    weights.  Float arithmetic chooses the integer law; all sampling and the
    auxiliary correction subsequently use those realized integers exactly.
    """

    values = np.asarray(log_weight, dtype=np.float64)
    if values.ndim != 1 or not len(values):
        raise ValueError("resampling log weights must be a nonempty vector")
    if np.isnan(values).any() or np.isposinf(values).any():
        raise ValueError("resampling log weights may be finite or -inf only")
    positive = np.isfinite(values)
    positive_count = int(np.count_nonzero(positive))
    if positive_count == 0:
        raise ValueError("cannot quantize an all-zero resampling measure")
    if positive_count >= PACKED_BRIDGE_TICKS:
        raise ValueError("positive particle count exceeds the resampling tick budget")

    shifted = np.zeros(values.shape, dtype=np.float64)
    maximum = float(np.max(values[positive]))
    shifted[positive] = np.exp(values[positive] - maximum, dtype=np.float64)
    cumulative = np.cumsum(shifted, dtype=np.float64)
    total = float(cumulative[-1])
    if not np.isfinite(total) or total <= 0.0:
        raise RuntimeError("resampling weights have an invalid shifted total")

    remaining = PACKED_BRIDGE_TICKS - positive_count
    positive_cumulative = np.cumsum(positive, dtype=np.int64)
    boundary = positive_cumulative + np.floor(
        (cumulative / total) * float(remaining)
    ).astype(np.int64)
    last_positive = int(np.flatnonzero(positive)[-1])
    terminal_correction = int(PACKED_BRIDGE_TICKS - int(boundary[last_positive]))
    boundary[last_positive:] = PACKED_BRIDGE_TICKS
    ticks = np.diff(
        np.concatenate((np.zeros(1, dtype=np.int64), boundary))
    )
    if (
        (ticks < 0).any()
        or int(ticks.sum(dtype=np.int64)) != PACKED_BRIDGE_TICKS
        or (ticks[~positive] != 0).any()
        or (ticks[positive] < 1).any()
    ):
        raise RuntimeError("resampling integer quantizer failed its exact audit")
    ticks = np.ascontiguousarray(ticks, dtype=np.uint64)
    ticks.setflags(write=False)
    return _ResamplingLaw(
        ticks=ticks,
        terminal_correction=terminal_correction,
        positive_count=positive_count,
        minimum_positive_tick=int(ticks[positive].min()),
    )


def _resampling_thresholds(
    *, seed: int, object_id: int, island_id: int, stage: int, particles: int
) -> np.ndarray:
    rng = _keyed_rng(
        seed=seed,
        object_id=object_id,
        island_id=island_id,
        stage=stage,
        domain=_DOMAIN_RESAMPLE,
    )
    raw = rng.bit_generator.random_raw(particles)
    return np.right_shift(raw, np.uint64(16)).astype(np.uint64, copy=False)


def _inverse_tick_cdf(ticks: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    mass = np.asarray(ticks, dtype=np.uint64)
    targets = np.asarray(thresholds, dtype=np.uint64)
    if (
        mass.ndim != 1
        or targets.ndim != 1
        or not len(mass)
        or int(mass.sum(dtype=np.uint64)) != PACKED_BRIDGE_TICKS
        or (targets >= PACKED_BRIDGE_TICKS).any()
    ):
        raise ValueError("resampling ticks or thresholds are invalid")
    cumulative = np.cumsum(mass, dtype=np.uint64)
    selected = np.searchsorted(cumulative, targets, side="right")
    if (selected >= len(mass)).any() or (mass[selected] == 0).any():
        raise RuntimeError("integer resampling selected a zero-tick ancestor")
    return selected.astype(np.int64, copy=False)


def _unbiased_randbelow(
    bound: int,
    *,
    seed: int,
    object_id: int,
    island_id: int,
    stage: int,
    particle: int,
) -> tuple[int, int]:
    """Draw uniformly from ``range(bound)`` using rejection on raw PCG64."""

    bound = _positive_integer(bound, "randbelow bound")
    if bound > np.iinfo(np.int64).max:
        raise ValueError("randbelow bound must fit signed int64")
    rng = _keyed_rng(
        seed=seed,
        object_id=object_id,
        island_id=island_id,
        stage=stage,
        particle=particle,
        domain=_DOMAIN_MH_ACCEPT,
    )
    limit = _UINT64_MODULUS - (_UINT64_MODULUS % bound)
    attempts = 0
    while True:
        attempts += 1
        raw = int(rng.bit_generator.random_raw())
        if raw < limit:
            return raw % bound, attempts


def _increment_log_weight(log_ratio: np.ndarray, stage: int) -> np.ndarray:
    values = np.asarray(log_ratio, dtype=np.float64)
    if stage < 1 or stage > PACKED_BRIDGE_STAGES:
        raise ValueError("bridge stage must lie in 1..8")
    finite = np.isfinite(values)
    result = np.full(values.shape, -np.inf, dtype=np.float64)
    if stage == PACKED_BRIDGE_STAGES:
        score = integer_bridge_score(values, PACKED_BRIDGE_MOVE_STAGES)
        result[finite] = values[finite] - np.log(score[finite].astype(np.float64))
        return result
    score = integer_bridge_score(values, stage)
    if stage == 1:
        result[finite] = np.log(score[finite].astype(np.float64))
    else:
        previous = integer_bridge_score(values, stage - 1)
        result[finite] = (
            np.log(score[finite].astype(np.float64))
            - np.log(previous[finite].astype(np.float64))
        )
    return result


def _validate_duplicate_targets(atom_ids: np.ndarray, log_ratio: np.ndarray) -> int:
    flat_atom = atom_ids.reshape(-1)
    flat_ratio = log_ratio.reshape(-1)
    order = np.argsort(flat_atom, kind="stable")
    ordered_atom = flat_atom[order]
    ordered_ratio = flat_ratio[order]
    boundary = np.concatenate(
        (
            np.ones(1, dtype=bool),
            ordered_atom[1:] != ordered_atom[:-1],
        )
    )
    starts = np.flatnonzero(boundary)
    stops = np.concatenate((starts[1:], np.asarray([len(order)])))
    for start, stop in zip(starts, stops):
        values = ordered_ratio[start:stop]
        if len(values) > 1 and not np.all(values == values[0]):
            raise ValueError("duplicate atom ids must carry identical target log ratios")
    return int(len(starts))


def _summarize_islands(
    island_log_normalizer: np.ndarray,
) -> tuple[float, float, float, float, float, np.ndarray]:
    logs = np.asarray(island_log_normalizer, dtype=np.float64)
    islands = len(logs)
    finite = np.isfinite(logs)
    if not finite.any():
        return 0.0, -np.inf, 0.0, -np.inf, 0.0, np.zeros(islands)
    shift = float(logs[finite].max())
    scaled = np.zeros(islands, dtype=np.float64)
    scaled[finite] = np.exp(logs[finite] - shift, dtype=np.float64)
    mean_scaled = float(scaled.mean(dtype=np.float64))
    log_normalizer = shift + float(np.log(mean_scaled))
    centered = scaled - mean_scaled
    squared_sum = float(np.square(centered).sum(dtype=np.float64))
    if squared_sum == 0.0:
        variance = 0.0
        log_variance = -np.inf
        standard_error = 0.0
    else:
        log_variance = (
            2.0 * shift
            + float(np.log(squared_sum))
            - float(np.log(islands * (islands - 1)))
        )
        with np.errstate(over="ignore"):
            variance = float(np.exp(log_variance))
            standard_error = float(np.exp(0.5 * log_variance))
    with np.errstate(over="ignore"):
        normalizer = float(np.exp(log_normalizer))
        island_normalizer = np.exp(logs, dtype=np.float64)
    return (
        normalizer,
        log_normalizer,
        variance,
        log_variance,
        standard_error,
        island_normalizer,
    )


@dataclass(frozen=True)
class PackedIntegerBridgeSMCResult:
    """Immutable trace and independent-island normalizer summary."""

    normalizer: float
    log_normalizer: float
    variance: float
    log_variance: float
    standard_error: float
    island_ids: np.ndarray
    island_normalizer: np.ndarray
    island_log_normalizer: np.ndarray
    stage_log_normalizer: np.ndarray
    stage_effective_sample_size: np.ndarray
    stage_acceptance_rate: np.ndarray
    stage_movement_rate: np.ndarray
    stage_unique_lineage_count: np.ndarray
    stage_unique_state_count: np.ndarray
    collapsed_stage: np.ndarray
    stage_atom_ids: np.ndarray
    stage_log_unnormalized_weight: np.ndarray
    ancestor_index: np.ndarray
    resampling_ticks: np.ndarray
    resampling_u48: np.ndarray
    resampling_tick_sum: np.ndarray
    resampling_positive_count: np.ndarray
    resampling_minimum_positive_tick: np.ndarray
    resampling_terminal_correction: np.ndarray
    mh_current_score: np.ndarray
    mh_proposal_score: np.ndarray
    mh_randbelow_value: np.ndarray
    mh_randbelow_attempts: np.ndarray
    mh_accepted: np.ndarray
    mh_moved: np.ndarray
    terminal_atom_ids: np.ndarray
    terminal_log_ratio: np.ndarray
    terminal_log_unnormalized_weight: np.ndarray
    betas: np.ndarray
    seed: int
    object_id: int
    particles_per_island: int
    island_chunk: int
    proposal_slot_count: int
    target_finite_slot_count: int
    target_zero_slot_count: int
    target_unique_atom_count: int
    target_cache_hit_count: int
    proposal_atom_ids_sha256: str
    target_log_ratio_sha256: str
    trace_sha256: str
    record_sha256: str
    execution_sha256: str
    version: str = PACKED_BRIDGE_VERSION
    rng_contract: str = PACKED_BRIDGE_RNG_CONTRACT
    score_contract: str = PACKED_BRIDGE_SCORE_CONTRACT
    resampling_contract: str = PACKED_BRIDGE_RESAMPLING_CONTRACT
    hash_contract: str = PACKED_BRIDGE_HASH_CONTRACT
    input_contract: str = PACKED_BRIDGE_INPUT_CONTRACT

    def __post_init__(self):
        if (
            self.version != PACKED_BRIDGE_VERSION
            or self.rng_contract != PACKED_BRIDGE_RNG_CONTRACT
            or self.score_contract != PACKED_BRIDGE_SCORE_CONTRACT
            or self.resampling_contract != PACKED_BRIDGE_RESAMPLING_CONTRACT
            or self.hash_contract != PACKED_BRIDGE_HASH_CONTRACT
            or self.input_contract != PACKED_BRIDGE_INPUT_CONTRACT
        ):
            raise ValueError("packed bridge result uses an unsupported contract")
        ids = np.ascontiguousarray(self.island_ids, dtype=np.int64).copy()
        islands = len(ids)
        particles = int(self.particles_per_island)
        if (
            islands < 2
            or ids.ndim != 1
            or (ids < 0).any()
            or len(np.unique(ids)) != islands
            or not np.array_equal(ids, np.sort(ids))
            or particles <= 0
        ):
            raise ValueError("packed bridge island identity is invalid")

        specs = {
            "island_normalizer": (np.float64, (islands,)),
            "island_log_normalizer": (np.float64, (islands,)),
            "stage_log_normalizer": (np.float64, (islands, PACKED_BRIDGE_STAGES)),
            "stage_effective_sample_size": (
                np.float64,
                (islands, PACKED_BRIDGE_STAGES),
            ),
            "stage_acceptance_rate": (
                np.float64,
                (islands, PACKED_BRIDGE_MOVE_STAGES),
            ),
            "stage_movement_rate": (
                np.float64,
                (islands, PACKED_BRIDGE_MOVE_STAGES),
            ),
            "stage_unique_lineage_count": (
                np.int64,
                (islands, PACKED_BRIDGE_STAGES),
            ),
            "stage_unique_state_count": (
                np.int64,
                (islands, PACKED_BRIDGE_STAGES),
            ),
            "collapsed_stage": (np.int64, (islands,)),
            "stage_atom_ids": (
                np.int64,
                (islands, PACKED_BRIDGE_STAGES, particles),
            ),
            "stage_log_unnormalized_weight": (
                np.float64,
                (islands, PACKED_BRIDGE_STAGES, particles),
            ),
            "ancestor_index": (
                np.int64,
                (islands, PACKED_BRIDGE_MOVE_STAGES, particles),
            ),
            "resampling_ticks": (
                np.uint64,
                (islands, PACKED_BRIDGE_MOVE_STAGES, particles),
            ),
            "resampling_u48": (
                np.uint64,
                (islands, PACKED_BRIDGE_MOVE_STAGES, particles),
            ),
            "resampling_tick_sum": (
                np.uint64,
                (islands, PACKED_BRIDGE_MOVE_STAGES),
            ),
            "resampling_positive_count": (
                np.int64,
                (islands, PACKED_BRIDGE_MOVE_STAGES),
            ),
            "resampling_minimum_positive_tick": (
                np.uint64,
                (islands, PACKED_BRIDGE_MOVE_STAGES),
            ),
            "resampling_terminal_correction": (
                np.int64,
                (islands, PACKED_BRIDGE_MOVE_STAGES),
            ),
            "mh_current_score": (
                np.int64,
                (islands, PACKED_BRIDGE_MOVE_STAGES, particles),
            ),
            "mh_proposal_score": (
                np.int64,
                (islands, PACKED_BRIDGE_MOVE_STAGES, particles),
            ),
            "mh_randbelow_value": (
                np.uint64,
                (islands, PACKED_BRIDGE_MOVE_STAGES, particles),
            ),
            "mh_randbelow_attempts": (
                np.uint16,
                (islands, PACKED_BRIDGE_MOVE_STAGES, particles),
            ),
            "mh_accepted": (
                np.bool_,
                (islands, PACKED_BRIDGE_MOVE_STAGES, particles),
            ),
            "mh_moved": (
                np.bool_,
                (islands, PACKED_BRIDGE_MOVE_STAGES, particles),
            ),
            "terminal_atom_ids": (np.int64, (islands, particles)),
            "terminal_log_ratio": (np.float64, (islands, particles)),
            "terminal_log_unnormalized_weight": (
                np.float64,
                (islands, particles),
            ),
            "betas": (np.float64, (PACKED_BRIDGE_STAGES + 1,)),
        }
        arrays: dict[str, np.ndarray] = {"island_ids": ids}
        for name, (dtype, shape) in specs.items():
            array = np.ascontiguousarray(getattr(self, name), dtype=dtype).copy()
            if array.shape != shape:
                raise ValueError(f"packed bridge {name} has the wrong shape")
            arrays[name] = array

        if not np.array_equal(arrays["betas"], PACKED_BRIDGE_BETAS):
            raise ValueError("packed bridge beta grid changed")
        if (
            np.isnan(arrays["island_log_normalizer"]).any()
            or np.isposinf(arrays["island_log_normalizer"]).any()
            or np.isnan(arrays["stage_log_normalizer"]).any()
            or np.isposinf(arrays["stage_log_normalizer"]).any()
            or np.isnan(arrays["stage_log_unnormalized_weight"]).any()
            or np.isposinf(arrays["stage_log_unnormalized_weight"]).any()
            or np.isnan(arrays["terminal_log_ratio"]).any()
            or np.isposinf(arrays["terminal_log_ratio"]).any()
            or np.isnan(arrays["terminal_log_unnormalized_weight"]).any()
            or np.isposinf(arrays["terminal_log_unnormalized_weight"]).any()
        ):
            raise ValueError("packed bridge log arrays may be finite or -inf only")
        if (
            not np.isfinite(arrays["stage_effective_sample_size"]).all()
            or (arrays["stage_effective_sample_size"] < 0.0).any()
            or (arrays["stage_effective_sample_size"] > particles * (1.0 + 1e-12)).any()
            or not np.isfinite(arrays["stage_acceptance_rate"]).all()
            or not np.isfinite(arrays["stage_movement_rate"]).all()
            or (arrays["stage_acceptance_rate"] < 0.0).any()
            or (arrays["stage_acceptance_rate"] > 1.0).any()
            or (arrays["stage_movement_rate"] < 0.0).any()
            or (arrays["stage_movement_rate"] > arrays["stage_acceptance_rate"]).any()
        ):
            raise ValueError("packed bridge stage diagnostics are invalid")
        if (
            (arrays["stage_atom_ids"] < 0).any()
            or (arrays["terminal_atom_ids"] < 0).any()
            or (arrays["collapsed_stage"] < 0).any()
            or (arrays["collapsed_stage"] > PACKED_BRIDGE_STAGES).any()
            or (arrays["stage_unique_lineage_count"] < 0).any()
            or (arrays["stage_unique_lineage_count"] > particles).any()
            or (arrays["stage_unique_state_count"] < 0).any()
            or (arrays["stage_unique_state_count"] > particles).any()
            or (arrays["mh_current_score"] < 0).any()
            or (arrays["mh_proposal_score"] < 0).any()
            or np.any(arrays["mh_moved"] & ~arrays["mh_accepted"])
        ):
            raise ValueError("packed bridge integer or movement diagnostics are invalid")
        if not np.array_equal(arrays["terminal_atom_ids"], arrays["stage_atom_ids"][:, -1]):
            raise ValueError("packed bridge terminal atom identity is inconsistent")
        if not np.array_equal(
            arrays["stage_atom_ids"][:, -1], arrays["stage_atom_ids"][:, -2]
        ):
            raise ValueError("packed bridge performed a forbidden terminal move")
        if not np.array_equal(
            arrays["terminal_log_unnormalized_weight"],
            arrays["stage_log_unnormalized_weight"][:, -1],
        ):
            raise ValueError("packed bridge terminal weights are inconsistent")

        expected = _summarize_islands(arrays["island_log_normalizer"])
        observed_scalars = np.asarray(
            (
                self.normalizer,
                self.log_normalizer,
                self.variance,
                self.log_variance,
                self.standard_error,
            ),
            dtype=np.float64,
        )
        expected_scalars = np.asarray(expected[:5], dtype=np.float64)
        if (
            np.isnan(observed_scalars).any()
            or not np.allclose(
                observed_scalars,
                expected_scalars,
                rtol=64.0 * np.finfo(np.float64).eps,
                atol=0.0,
                equal_nan=False,
            )
            or not np.array_equal(arrays["island_normalizer"], expected[5])
        ):
            raise ValueError("packed bridge island summary is inconsistent")

        slots = islands * PACKED_BRIDGE_ROUNDS * particles
        if (
            int(self.proposal_slot_count) != slots
            or int(self.target_finite_slot_count) < 0
            or int(self.target_zero_slot_count) < 0
            or int(self.target_finite_slot_count) + int(self.target_zero_slot_count) != slots
            or int(self.target_unique_atom_count) <= 0
            or int(self.target_unique_atom_count) > slots
            or int(self.target_cache_hit_count) != slots - int(self.target_unique_atom_count)
        ):
            raise ValueError("packed bridge target accounting is inconsistent")

        seed = _exact_nonnegative_integer(
            self.seed, "seed", np.iinfo(np.uint64).max
        )
        object_id = _exact_nonnegative_integer(
            self.object_id, "object_id", np.iinfo(np.uint64).max
        )
        chunk = _positive_integer(self.island_chunk, "island_chunk")
        if not np.array_equal(
            arrays["stage_log_normalizer"][:, -1],
            arrays["island_log_normalizer"],
        ):
            raise ValueError("final stage and island normalizers disagree")

        for island in range(islands):
            collapse = int(arrays["collapsed_stage"][island])
            lineage = np.arange(particles, dtype=np.int64)
            for stage_index in range(PACKED_BRIDGE_MOVE_STAGES):
                stage = stage_index + 1
                active = collapse == 0 or collapse > stage
                ticks = arrays["resampling_ticks"][island, stage_index]
                thresholds = arrays["resampling_u48"][island, stage_index]
                parents = arrays["ancestor_index"][island, stage_index]
                if active:
                    positive_ticks = ticks[ticks > 0]
                    if (
                        int(ticks.sum(dtype=np.uint64)) != PACKED_BRIDGE_TICKS
                        or int(arrays["resampling_tick_sum"][island, stage_index])
                        != PACKED_BRIDGE_TICKS
                        or not len(positive_ticks)
                        or int(
                            arrays["resampling_positive_count"][island, stage_index]
                        )
                        != len(positive_ticks)
                        or int(
                            arrays["resampling_minimum_positive_tick"][
                                island, stage_index
                            ]
                        )
                        != int(positive_ticks.min())
                        or (thresholds >= PACKED_BRIDGE_TICKS).any()
                        or (parents < 0).any()
                        or (parents >= particles).any()
                        or (ticks[parents] == 0).any()
                        or not np.array_equal(
                            parents, _inverse_tick_cdf(ticks, thresholds)
                        )
                    ):
                        raise ValueError("packed bridge resampling audit failed")
                    lineage = lineage[parents]
                    current = arrays["mh_current_score"][island, stage_index]
                    proposed = arrays["mh_proposal_score"][island, stage_index]
                    draws = arrays["mh_randbelow_value"][island, stage_index]
                    attempts = arrays["mh_randbelow_attempts"][island, stage_index]
                    accepted = arrays["mh_accepted"][island, stage_index]
                    deterministic_accept = proposed >= current
                    deterministic_reject = proposed == 0
                    random_case = (proposed > 0) & (proposed < current)
                    if (
                        (current <= 0).any()
                        or not np.array_equal(accepted[deterministic_accept], np.ones(
                            np.count_nonzero(deterministic_accept), dtype=bool
                        ))
                        or accepted[deterministic_reject & ~deterministic_accept].any()
                        or (attempts[random_case] == 0).any()
                        or (attempts[~random_case] != 0).any()
                        or np.any(draws[random_case] >= current[random_case].astype(np.uint64))
                        or not np.array_equal(
                            accepted[random_case],
                            draws[random_case] < proposed[random_case].astype(np.uint64),
                        )
                    ):
                        raise ValueError("packed bridge integer MH audit failed")
                    expected_acceptance = float(accepted.mean(dtype=np.float64))
                    expected_movement = float(
                        arrays["mh_moved"][island, stage_index].mean(
                            dtype=np.float64
                        )
                    )
                    if (
                        arrays["stage_acceptance_rate"][island, stage_index]
                        != expected_acceptance
                        or arrays["stage_movement_rate"][island, stage_index]
                        != expected_movement
                        or arrays["stage_unique_state_count"][island, stage_index]
                        != len(
                            np.unique(
                                arrays["stage_atom_ids"][island, stage_index]
                            )
                        )
                        or arrays["stage_unique_lineage_count"][
                            island, stage_index
                        ]
                        != len(np.unique(lineage))
                    ):
                        raise ValueError("packed bridge move summary audit failed")
                elif (
                    ticks.any()
                    or thresholds.any()
                    or np.any(parents != -1)
                    or arrays["mh_current_score"][island, stage_index].any()
                    or arrays["mh_proposal_score"][island, stage_index].any()
                    or arrays["mh_randbelow_value"][island, stage_index].any()
                    or arrays["mh_randbelow_attempts"][island, stage_index].any()
                    or arrays["mh_accepted"][island, stage_index].any()
                    or arrays["mh_moved"][island, stage_index].any()
                ):
                    raise ValueError("collapsed bridge stage retained live move state")
            if collapse:
                if not np.all(
                    np.isneginf(
                        arrays["stage_log_normalizer"][island, collapse - 1 :]
                    )
                ):
                    raise ValueError("collapsed bridge recovered positive mass")
            elif arrays["stage_unique_state_count"][island, -1] != len(
                np.unique(arrays["stage_atom_ids"][island, -1])
            ):
                raise ValueError("final unique-state audit failed")

        for name in (
            "proposal_atom_ids_sha256",
            "target_log_ratio_sha256",
            "trace_sha256",
            "record_sha256",
            "execution_sha256",
        ):
            _validate_sha256(getattr(self, name), name)
        trace_arrays = {
            name: arrays[name]
            for name in (
                "island_log_normalizer",
                "stage_log_normalizer",
                "stage_effective_sample_size",
                "stage_acceptance_rate",
                "stage_movement_rate",
                "stage_unique_lineage_count",
                "stage_unique_state_count",
                "stage_atom_ids",
                "stage_log_unnormalized_weight",
                "ancestor_index",
                "resampling_ticks",
                "resampling_u48",
                "resampling_tick_sum",
                "resampling_positive_count",
                "resampling_minimum_positive_tick",
                "resampling_terminal_correction",
                "mh_current_score",
                "mh_proposal_score",
                "mh_randbelow_value",
                "mh_randbelow_attempts",
                "mh_accepted",
                "mh_moved",
                "collapsed_stage",
                "terminal_atom_ids",
                "terminal_log_ratio",
                "terminal_log_unnormalized_weight",
            )
        }
        expected_trace_hash = _trace_sha256(trace_arrays)
        expected_record_hash = _run_record_sha256(
            proposal_hash=self.proposal_atom_ids_sha256,
            ratio_hash=self.target_log_ratio_sha256,
            trace_hash=expected_trace_hash,
            island_ids=ids,
            seed=seed,
            object_id=object_id,
            particles=particles,
            slots=slots,
            finite_slots=int(self.target_finite_slot_count),
            unique_atoms=int(self.target_unique_atom_count),
        )
        expected_execution_hash = _record_sha256(
            "execution",
            (expected_record_hash, _text_sha256("island_chunk", chunk)),
        )
        if (
            self.trace_sha256 != expected_trace_hash
            or self.record_sha256 != expected_record_hash
            or self.execution_sha256 != expected_execution_hash
        ):
            raise ValueError("packed bridge record hash does not match its trace")
        for name, array in arrays.items():
            array.setflags(write=False)
            object.__setattr__(self, name, array)


def _trace_sha256(arrays: dict[str, np.ndarray]) -> str:
    specifications = (
        ("island_log_normalizer", ">f8"),
        ("stage_log_normalizer", ">f8"),
        ("stage_effective_sample_size", ">f8"),
        ("stage_acceptance_rate", ">f8"),
        ("stage_movement_rate", ">f8"),
        ("stage_unique_lineage_count", ">i8"),
        ("stage_unique_state_count", ">i8"),
        ("stage_atom_ids", ">i8"),
        ("stage_log_unnormalized_weight", ">f8"),
        ("ancestor_index", ">i8"),
        ("resampling_ticks", ">u8"),
        ("resampling_u48", ">u8"),
        ("resampling_tick_sum", ">u8"),
        ("resampling_positive_count", ">i8"),
        ("resampling_minimum_positive_tick", ">u8"),
        ("resampling_terminal_correction", ">i8"),
        ("mh_current_score", ">i8"),
        ("mh_proposal_score", ">i8"),
        ("mh_randbelow_value", ">u8"),
        ("mh_randbelow_attempts", ">u2"),
        ("mh_accepted", "u1"),
        ("mh_moved", "u1"),
        ("collapsed_stage", ">i8"),
        ("terminal_atom_ids", ">i8"),
        ("terminal_log_ratio", ">f8"),
        ("terminal_log_unnormalized_weight", ">f8"),
    )
    return _record_sha256(
        "trace",
        tuple(
            _array_sha256(name, arrays[name], dtype=dtype)
            for name, dtype in specifications
        ),
    )


def _run_record_sha256(
    *,
    proposal_hash: str,
    ratio_hash: str,
    trace_hash: str,
    island_ids: np.ndarray,
    seed: int,
    object_id: int,
    particles: int,
    slots: int,
    finite_slots: int,
    unique_atoms: int,
) -> str:
    return _record_sha256(
        "run",
        (
            proposal_hash,
            ratio_hash,
            trace_hash,
            _array_sha256("island_ids", island_ids, dtype=">i8"),
            _array_sha256("betas", PACKED_BRIDGE_BETAS, dtype=">f8"),
            _text_sha256("version", PACKED_BRIDGE_VERSION),
            _text_sha256("rng_contract", PACKED_BRIDGE_RNG_CONTRACT),
            _text_sha256("score_contract", PACKED_BRIDGE_SCORE_CONTRACT),
            _text_sha256(
                "resampling_contract", PACKED_BRIDGE_RESAMPLING_CONTRACT
            ),
            _text_sha256("input_contract", PACKED_BRIDGE_INPUT_CONTRACT),
            _text_sha256("seed", seed),
            _text_sha256("object_id", object_id),
            _text_sha256("particles", particles),
            _text_sha256("proposal_slots", slots),
            _text_sha256("target_finite_slots", finite_slots),
            _text_sha256("target_unique_atoms", unique_atoms),
        ),
    )


def run_packed_integer_bridge_smc(
    proposal_atom_ids,
    target_log_ratio,
    *,
    seed: int,
    object_id: int = 0,
    island_ids=None,
    island_chunk: Optional[int] = None,
) -> PackedIntegerBridgeSMCResult:
    """Run the fixed eight-round packed integer-bridge SMC kernel.

    ``proposal_atom_ids`` and ``target_log_ratio`` must have shape
    ``(islands, 8, particles)``.  Every slot is a caller-prepared IID draw from
    the same full-support ``q0`` law.  The kernel validates target consistency
    for duplicate atom IDs but cannot statistically verify IID generation.
    """

    raw_atoms = np.asarray(proposal_atom_ids)
    ratios = np.asarray(target_log_ratio, dtype=np.float64)
    if raw_atoms.ndim != 3 or raw_atoms.dtype.kind not in "iu":
        raise ValueError("proposal atom ids must be a three-dimensional integer array")
    if raw_atoms.shape[1] != PACKED_BRIDGE_ROUNDS or ratios.shape != raw_atoms.shape:
        raise ValueError("packed proposals must have shape (islands, 8, particles)")
    if raw_atoms.shape[0] < 2 or raw_atoms.shape[2] <= 0:
        raise ValueError("packed bridge needs at least two islands and one particle")
    if raw_atoms.dtype.kind == "u" and np.any(raw_atoms > np.iinfo(np.int64).max):
        raise ValueError("proposal atom ids must fit non-negative int64")
    atoms = np.ascontiguousarray(raw_atoms, dtype=np.int64)
    if (atoms < 0).any():
        raise ValueError("proposal atom ids must be non-negative")
    if np.isnan(ratios).any() or np.isposinf(ratios).any():
        raise ValueError("target log ratios may be finite or -inf, but not NaN/+inf")
    ratios = np.ascontiguousarray(ratios, dtype=np.float64)

    seed = _exact_nonnegative_integer(seed, "seed", np.iinfo(np.uint64).max)
    object_id = _exact_nonnegative_integer(
        object_id, "object_id", np.iinfo(np.uint64).max
    )
    islands, _, particles = atoms.shape
    if island_ids is None:
        ids = np.arange(islands, dtype=np.int64)
    else:
        raw_ids = np.asarray(island_ids)
        if raw_ids.ndim != 1 or len(raw_ids) != islands or raw_ids.dtype.kind not in "iu":
            raise ValueError("island_ids must be an aligned integer vector")
        if raw_ids.dtype.kind == "u" and np.any(raw_ids > np.iinfo(np.int64).max):
            raise ValueError("island_ids must fit non-negative int64")
        ids = np.ascontiguousarray(raw_ids, dtype=np.int64)
        if (ids < 0).any() or len(np.unique(ids)) != islands:
            raise ValueError("island_ids must be unique non-negative integers")
    order = np.argsort(ids, kind="stable")
    ids = np.ascontiguousarray(ids[order])
    atoms = np.ascontiguousarray(atoms[order])
    ratios = np.ascontiguousarray(ratios[order])
    chunk = islands if island_chunk is None else _positive_integer(
        island_chunk, "island_chunk"
    )

    unique_atoms = _validate_duplicate_targets(atoms, ratios)
    slots = int(atoms.size)
    finite_slots = int(np.isfinite(ratios).sum())

    stage_shape = (islands, PACKED_BRIDGE_STAGES)
    move_shape = (islands, PACKED_BRIDGE_MOVE_STAGES)
    stage_particle_shape = (*stage_shape, particles)
    move_particle_shape = (*move_shape, particles)
    stage_log_z = np.full(stage_shape, -np.inf, dtype=np.float64)
    stage_ess = np.zeros(stage_shape, dtype=np.float64)
    stage_acceptance = np.zeros(move_shape, dtype=np.float64)
    stage_movement = np.zeros(move_shape, dtype=np.float64)
    stage_unique_lineage = np.zeros(stage_shape, dtype=np.int64)
    stage_unique_state = np.zeros(stage_shape, dtype=np.int64)
    collapsed_stage = np.zeros(islands, dtype=np.int64)
    stage_atoms = np.empty(stage_particle_shape, dtype=np.int64)
    stage_log_u = np.full(stage_particle_shape, -np.inf, dtype=np.float64)
    ancestor = np.full(move_particle_shape, -1, dtype=np.int64)
    resampling_ticks = np.zeros(move_particle_shape, dtype=np.uint64)
    resampling_u48 = np.zeros(move_particle_shape, dtype=np.uint64)
    resampling_tick_sum = np.zeros(move_shape, dtype=np.uint64)
    resampling_positive_count = np.zeros(move_shape, dtype=np.int64)
    resampling_minimum_tick = np.zeros(move_shape, dtype=np.uint64)
    resampling_correction = np.zeros(move_shape, dtype=np.int64)
    mh_current_score = np.zeros(move_particle_shape, dtype=np.int64)
    mh_proposal_score = np.zeros(move_particle_shape, dtype=np.int64)
    mh_randbelow_value = np.zeros(move_particle_shape, dtype=np.uint64)
    mh_randbelow_attempts = np.zeros(move_particle_shape, dtype=np.uint16)
    mh_accepted = np.zeros(move_particle_shape, dtype=bool)
    mh_moved = np.zeros(move_particle_shape, dtype=bool)
    terminal_atoms = np.empty((islands, particles), dtype=np.int64)
    terminal_ratio = np.empty((islands, particles), dtype=np.float64)
    terminal_log_u = np.full((islands, particles), -np.inf, dtype=np.float64)
    island_log_z = np.full(islands, -np.inf, dtype=np.float64)

    log_particle_count = float(np.log(particles))
    log_tick_total = float(np.log(PACKED_BRIDGE_TICKS))

    for chunk_start in range(0, islands, chunk):
        chunk_stop = min(chunk_start + chunk, islands)
        for island_position in range(chunk_start, chunk_stop):
            island_id = int(ids[island_position])
            current_atom = atoms[island_position, 0].copy()
            current_ratio = ratios[island_position, 0].copy()
            current_log_u = np.zeros(particles, dtype=np.float64)
            lineage = np.arange(particles, dtype=np.int64)
            collapsed = False

            for stage in range(1, PACKED_BRIDGE_STAGES + 1):
                stage_index = stage - 1
                if collapsed:
                    stage_atoms[island_position, stage_index] = current_atom
                    continue
                increment = _increment_log_weight(current_ratio, stage)
                weighted_log_u = current_log_u + increment
                if np.isnan(weighted_log_u).any() or np.isposinf(weighted_log_u).any():
                    raise RuntimeError("packed bridge produced an invalid carried weight")
                stage_log_z[island_position, stage_index] = _logmeanexp(weighted_log_u)
                stage_ess[island_position, stage_index] = _effective_sample_size(
                    weighted_log_u
                )

                if not np.isfinite(weighted_log_u).any():
                    collapsed_stage[island_position] = stage
                    current_log_u.fill(-np.inf)
                    stage_log_u[island_position, stage_index] = current_log_u
                    stage_atoms[island_position, stage_index] = current_atom
                    collapsed = True
                    continue

                if stage == PACKED_BRIDGE_STAGES:
                    current_log_u = weighted_log_u
                    stage_log_u[island_position, stage_index] = current_log_u
                    stage_atoms[island_position, stage_index] = current_atom
                    stage_unique_lineage[island_position, stage_index] = len(
                        np.unique(lineage)
                    )
                    stage_unique_state[island_position, stage_index] = len(
                        np.unique(current_atom)
                    )
                    continue

                law = _quantize_resampling_log_weights(weighted_log_u)
                thresholds = _resampling_thresholds(
                    seed=seed,
                    object_id=object_id,
                    island_id=island_id,
                    stage=stage,
                    particles=particles,
                )
                parent = _inverse_tick_cdf(law.ticks, thresholds)
                parent_tick = law.ticks[parent]
                if (parent_tick == 0).any():
                    raise RuntimeError("resampling chose an unsupported parent")
                current_atom = current_atom[parent]
                current_ratio = current_ratio[parent]
                lineage = lineage[parent]
                current_log_u = (
                    weighted_log_u[parent]
                    + log_tick_total
                    - log_particle_count
                    - np.log(parent_tick.astype(np.float64))
                )
                if not np.isfinite(current_log_u).all():
                    raise RuntimeError("auxiliary resampling correction is invalid")

                ancestor[island_position, stage_index] = parent
                resampling_ticks[island_position, stage_index] = law.ticks
                resampling_u48[island_position, stage_index] = thresholds
                resampling_tick_sum[island_position, stage_index] = np.uint64(
                    law.ticks.sum(dtype=np.uint64)
                )
                resampling_positive_count[island_position, stage_index] = (
                    law.positive_count
                )
                resampling_minimum_tick[island_position, stage_index] = np.uint64(
                    law.minimum_positive_tick
                )
                resampling_correction[island_position, stage_index] = (
                    law.terminal_correction
                )

                proposal_atom = atoms[island_position, stage]
                proposal_ratio = ratios[island_position, stage]
                current_score = integer_bridge_score(current_ratio, stage)
                proposal_score = integer_bridge_score(proposal_ratio, stage)
                if (current_score <= 0).any():
                    raise RuntimeError("resampling retained a zero-score bridge state")
                accepted = np.zeros(particles, dtype=bool)
                for particle in range(particles):
                    old_score = int(current_score[particle])
                    new_score = int(proposal_score[particle])
                    if new_score >= old_score:
                        accepted[particle] = True
                    elif new_score > 0:
                        draw, attempts = _unbiased_randbelow(
                            old_score,
                            seed=seed,
                            object_id=object_id,
                            island_id=island_id,
                            stage=stage,
                            particle=particle,
                        )
                        if attempts > np.iinfo(np.uint16).max:
                            raise RuntimeError("MH randbelow attempts exceed uint16 audit")
                        mh_randbelow_value[
                            island_position, stage_index, particle
                        ] = np.uint64(draw)
                        mh_randbelow_attempts[
                            island_position, stage_index, particle
                        ] = np.uint16(attempts)
                        accepted[particle] = draw < new_score
                moved = accepted & (proposal_atom != current_atom)
                current_atom = np.where(accepted, proposal_atom, current_atom)
                current_ratio = np.where(accepted, proposal_ratio, current_ratio)

                mh_current_score[island_position, stage_index] = current_score
                mh_proposal_score[island_position, stage_index] = proposal_score
                mh_accepted[island_position, stage_index] = accepted
                mh_moved[island_position, stage_index] = moved
                stage_acceptance[island_position, stage_index] = float(
                    accepted.mean(dtype=np.float64)
                )
                stage_movement[island_position, stage_index] = float(
                    moved.mean(dtype=np.float64)
                )
                stage_log_u[island_position, stage_index] = current_log_u
                stage_atoms[island_position, stage_index] = current_atom
                stage_unique_lineage[island_position, stage_index] = len(
                    np.unique(lineage)
                )
                stage_unique_state[island_position, stage_index] = len(
                    np.unique(current_atom)
                )

            terminal_atoms[island_position] = current_atom
            terminal_ratio[island_position] = current_ratio
            terminal_log_u[island_position] = current_log_u
            island_log_z[island_position] = _logmeanexp(current_log_u)

    summary = _summarize_islands(island_log_z)
    proposal_hash = _array_sha256("proposal_atom_ids", atoms, dtype=">i8")
    ratio_hash = _array_sha256("target_log_ratio", ratios, dtype=">f8")
    trace_arrays = {
        "island_log_normalizer": island_log_z,
        "stage_log_normalizer": stage_log_z,
        "stage_effective_sample_size": stage_ess,
        "stage_acceptance_rate": stage_acceptance,
        "stage_movement_rate": stage_movement,
        "stage_unique_lineage_count": stage_unique_lineage,
        "stage_unique_state_count": stage_unique_state,
        "stage_atom_ids": stage_atoms,
        "stage_log_unnormalized_weight": stage_log_u,
        "ancestor_index": ancestor,
        "resampling_ticks": resampling_ticks,
        "resampling_u48": resampling_u48,
        "resampling_tick_sum": resampling_tick_sum,
        "resampling_positive_count": resampling_positive_count,
        "resampling_minimum_positive_tick": resampling_minimum_tick,
        "resampling_terminal_correction": resampling_correction,
        "mh_current_score": mh_current_score,
        "mh_proposal_score": mh_proposal_score,
        "mh_randbelow_value": mh_randbelow_value,
        "mh_randbelow_attempts": mh_randbelow_attempts,
        "mh_accepted": mh_accepted,
        "mh_moved": mh_moved,
        "collapsed_stage": collapsed_stage,
        "terminal_atom_ids": terminal_atoms,
        "terminal_log_ratio": terminal_ratio,
        "terminal_log_unnormalized_weight": terminal_log_u,
    }
    trace_hash = _trace_sha256(trace_arrays)
    record_hash = _run_record_sha256(
        proposal_hash=proposal_hash,
        ratio_hash=ratio_hash,
        trace_hash=trace_hash,
        island_ids=ids,
        seed=seed,
        object_id=object_id,
        particles=particles,
        slots=slots,
        finite_slots=finite_slots,
        unique_atoms=unique_atoms,
    )
    execution_hash = _record_sha256(
        "execution",
        (record_hash, _text_sha256("island_chunk", chunk)),
    )

    return PackedIntegerBridgeSMCResult(
        normalizer=summary[0],
        log_normalizer=summary[1],
        variance=summary[2],
        log_variance=summary[3],
        standard_error=summary[4],
        island_ids=ids,
        island_normalizer=summary[5],
        island_log_normalizer=island_log_z,
        stage_log_normalizer=stage_log_z,
        stage_effective_sample_size=stage_ess,
        stage_acceptance_rate=stage_acceptance,
        stage_movement_rate=stage_movement,
        stage_unique_lineage_count=stage_unique_lineage,
        stage_unique_state_count=stage_unique_state,
        collapsed_stage=collapsed_stage,
        stage_atom_ids=stage_atoms,
        stage_log_unnormalized_weight=stage_log_u,
        ancestor_index=ancestor,
        resampling_ticks=resampling_ticks,
        resampling_u48=resampling_u48,
        resampling_tick_sum=resampling_tick_sum,
        resampling_positive_count=resampling_positive_count,
        resampling_minimum_positive_tick=resampling_minimum_tick,
        resampling_terminal_correction=resampling_correction,
        mh_current_score=mh_current_score,
        mh_proposal_score=mh_proposal_score,
        mh_randbelow_value=mh_randbelow_value,
        mh_randbelow_attempts=mh_randbelow_attempts,
        mh_accepted=mh_accepted,
        mh_moved=mh_moved,
        terminal_atom_ids=terminal_atoms,
        terminal_log_ratio=terminal_ratio,
        terminal_log_unnormalized_weight=terminal_log_u,
        betas=PACKED_BRIDGE_BETAS,
        seed=seed,
        object_id=object_id,
        particles_per_island=particles,
        island_chunk=chunk,
        proposal_slot_count=slots,
        target_finite_slot_count=finite_slots,
        target_zero_slot_count=slots - finite_slots,
        target_unique_atom_count=unique_atoms,
        target_cache_hit_count=slots - unique_atoms,
        proposal_atom_ids_sha256=proposal_hash,
        target_log_ratio_sha256=ratio_hash,
        trace_sha256=trace_hash,
        record_sha256=record_hash,
        execution_sha256=execution_hash,
    )


__all__ = [
    "PACKED_BRIDGE_BETAS",
    "PACKED_BRIDGE_HASH_CONTRACT",
    "PACKED_BRIDGE_INPUT_CONTRACT",
    "PACKED_BRIDGE_LOG_CLIP",
    "PACKED_BRIDGE_RESAMPLING_CONTRACT",
    "PACKED_BRIDGE_RNG_CONTRACT",
    "PACKED_BRIDGE_SCORE_CONTRACT",
    "PACKED_BRIDGE_SCORE_SCALE",
    "PACKED_BRIDGE_TICKS",
    "PACKED_BRIDGE_VERSION",
    "PackedIntegerBridgeSMCResult",
    "integer_bridge_score",
    "run_packed_integer_bridge_smc",
]
