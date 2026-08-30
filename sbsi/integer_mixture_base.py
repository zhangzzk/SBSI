"""Exact global two-source integer mixture for finite-catalogue sampling.

The public construction fixes two independently quantized source laws with
``2**48`` ticks each.  An IID three-bit source selector chooses the detected
source with probability ``7/8`` and the defensive raw-prior source with
probability ``1/8``.  Consequently the realized mixture probability of atom
``j`` is exactly

``(7 * detected_ticks[j] + raw_prior_ticks[j]) / 2**51``.

This module deliberately implements only the outcome-independent, unconditioned
global base.  Object-specific complement conditioning belongs to a separate
layer and must not silently alter this cached law.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
import os
from pathlib import Path
import re
from typing import Mapping
import zipfile

import numpy as np


_VERSION = "integer_mixture_base_v1"
_CACHE_VERSION = 1
_TOTAL_TICKS = 1 << 48
_MIXTURE_DENOMINATOR = 8
_DETECTED_SOURCE_NUMERATOR = 7
_PRIOR_SOURCE_NUMERATOR = 1
_PROBABILITY_DENOMINATOR_POWER = 51
_RNG_VERSION_WORD = 1
_RNG_SOURCE_PURPOSE_WORD = 0x534F5552
_RNG_THRESHOLD_PURPOSE_WORD = 0x55343854
_RNG_CONTRACT = (
    "pcg64_seedsequence_seed_object_domain_lo_hi_version_purpose_"
    "source_low3_threshold_independent_high48_v1"
)
_QUANTIZER_CONTRACT = "global_cumulative_floor_uint48_prior_floor_v1"
_HASH_CONTRACT = "sha256_domain_shape_big_endian_integer_mixture_base_v1"
_ORDER_CONTRACT = "canonical_full_catalogue_atom_order_v1"
_CACHE_FORMAT = "deterministic_npz_manifest_integer_mixture_base_v1"
_ARRAY_FILE = "arrays.npz"
_MANIFEST_FILE = "manifest.json"
_CACHE_ARRAY_NAMES = (
    "active",
    "detected_prior_cdf",
    "detected_prior_ticks",
    "raw_prior_cdf",
    "raw_prior_ticks",
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _exact_integer(value, name: str, maximum: int) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or int(value) < 0
        or int(value) > maximum
    ):
        raise ValueError(f"{name} must be an integer in [0, {maximum}]")
    return int(value)


def _validate_sha256(value: str, name: str) -> str:
    parsed = str(value)
    if _SHA256_RE.fullmatch(parsed) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return parsed


def _array_sha256(label: str, values, *, dtype: str) -> str:
    canonical = np.ascontiguousarray(values, dtype=np.dtype(dtype))
    shape = np.ascontiguousarray(canonical.shape, dtype=np.dtype(">u8"))
    digest = sha256()
    digest.update(b"SBSI/integer-mixture-base-array/v1\0")
    digest.update(label.encode("utf-8"))
    digest.update(b"\0")
    digest.update(shape.tobytes())
    digest.update(canonical.tobytes())
    return digest.hexdigest()


def _text_sha256(label: str, value) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    digest = sha256()
    digest.update(b"SBSI/integer-mixture-base-text/v1\0")
    digest.update(label.encode("utf-8"))
    digest.update(b"\0")
    digest.update(payload)
    return digest.hexdigest()


def _record_sha256(label: str, values) -> str:
    digest = sha256()
    digest.update(b"SBSI/integer-mixture-base-record/v1\0")
    digest.update(label.encode("utf-8"))
    digest.update(b"\0")
    for value in values:
        encoded = str(value).encode("ascii")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_bytes(value: Mapping) -> bytes:
    return (
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        + "\n"
    ).encode("utf-8")


def _weights(values, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if (
        array.ndim != 1
        or not len(array)
        or not np.isfinite(array).all()
        or (array < 0.0).any()
    ):
        raise ValueError(f"{name} must be a nonempty finite nonnegative vector")
    return np.ascontiguousarray(array)


def _readonly_copy(values, dtype) -> np.ndarray:
    array = np.ascontiguousarray(values, dtype=dtype).copy()
    array.setflags(write=False)
    return array


def _quantize_detected(
    weights: np.ndarray, active: np.ndarray
) -> tuple[np.ndarray, int]:
    if (weights[~active] != 0.0).any():
        raise ValueError("detected-prior weight must be zero off raw-prior support")
    values = np.where(active, weights, 0.0)
    positive = values > 0.0
    if not positive.any():
        raise ValueError("detected-prior source must have positive total weight")
    values = values / float(values.max())
    cumulative = np.cumsum(values, dtype=np.float64)
    total = float(cumulative[-1])
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("detected-prior source has invalid total weight")
    boundary = np.floor(
        (cumulative / total) * float(_TOTAL_TICKS)
    ).astype(np.int64)
    last_positive = int(np.flatnonzero(positive)[-1])
    correction = int(_TOTAL_TICKS - int(boundary[last_positive]))
    boundary[last_positive:] = _TOTAL_TICKS
    ticks = np.diff(
        np.concatenate((np.zeros(1, dtype=np.int64), boundary))
    )
    if (
        (ticks < 0).any()
        or (np.diff(boundary) < 0).any()
        or int(ticks.sum(dtype=np.int64)) != _TOTAL_TICKS
        or (ticks[~active] != 0).any()
    ):
        raise RuntimeError("detected-prior integer quantization audit failed")
    return np.ascontiguousarray(ticks, dtype=np.uint64), correction


def _quantize_prior(
    weights: np.ndarray, active: np.ndarray
) -> tuple[np.ndarray, int]:
    active_count = int(np.count_nonzero(active))
    if active_count > _TOTAL_TICKS:
        raise ValueError("raw-prior support exceeds the uint48 tick total")
    if active_count == _TOTAL_TICKS:
        ticks = active.astype(np.uint64)
        return np.ascontiguousarray(ticks), 0
    scaled = weights / float(weights[active].max())
    cumulative = np.cumsum(np.where(active, scaled, 0.0), dtype=np.float64)
    total = float(cumulative[-1])
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("raw-prior source has invalid total weight")
    remainder = _TOTAL_TICKS - active_count
    active_cumulative = np.cumsum(active, dtype=np.int64)
    boundary = active_cumulative + np.floor(
        (cumulative / total) * float(remainder)
    ).astype(np.int64)
    last_active = int(np.flatnonzero(active)[-1])
    correction = int(_TOTAL_TICKS - int(boundary[last_active]))
    boundary[last_active:] = _TOTAL_TICKS
    ticks = np.diff(
        np.concatenate((np.zeros(1, dtype=np.int64), boundary))
    )
    if (
        (ticks < 0).any()
        or (np.diff(boundary) < 0).any()
        or int(ticks.sum(dtype=np.int64)) != _TOTAL_TICKS
        or (ticks[~active] != 0).any()
        or (ticks[active] < 1).any()
    ):
        raise RuntimeError("raw-prior integer quantization audit failed")
    return np.ascontiguousarray(ticks, dtype=np.uint64), correction


def _u64_words(value: int) -> tuple[int, int]:
    return value & 0xFFFF_FFFF, (value >> 32) & 0xFFFF_FFFF


def _keyed_raw(
    *,
    seed: int,
    object_id: int,
    domain: int,
    purpose: int,
    size: int,
    offset: int = 0,
) -> np.ndarray:
    """Return canonical PCG64 raw words for one object and purpose."""

    maximum = int(np.iinfo(np.uint64).max)
    seed = _exact_integer(seed, "seed", maximum)
    object_id = _exact_integer(object_id, "object_id", maximum)
    domain = _exact_integer(domain, "domain", maximum)
    size = _exact_integer(size, "size", maximum)
    offset = _exact_integer(offset, "offset", maximum)
    if offset + size > maximum:
        raise ValueError("raw stream offset and size exceed uint64")
    seed_low, seed_high = _u64_words(seed)
    object_low, object_high = _u64_words(object_id)
    domain_low, domain_high = _u64_words(domain)
    words = [
        seed_low,
        seed_high,
        object_low,
        object_high,
        domain_low,
        domain_high,
        _RNG_VERSION_WORD,
        int(purpose),
    ]
    bit_generator = np.random.PCG64(np.random.SeedSequence(words))
    if offset:
        bit_generator.advance(offset)
    result = np.ascontiguousarray(
        bit_generator.random_raw(size), dtype=np.uint64
    )
    result.setflags(write=False)
    return result


def _strict_u64_vector(values, name: str) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1 or not len(raw) or raw.dtype.kind not in "iu":
        raise ValueError(f"{name} must be a nonempty one-dimensional integer vector")
    if raw.dtype.kind == "i" and (raw < 0).any():
        raise ValueError(f"{name} must contain nonnegative integers")
    if raw.dtype.kind == "u" and (
        raw > np.iinfo(np.uint64).max
    ).any():
        raise ValueError(f"{name} must fit uint64")
    result = np.ascontiguousarray(raw, dtype=np.uint64)
    if len(np.unique(result)) != len(result):
        raise ValueError(f"{name} must be unique")
    return result


def _write_deterministic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    if tuple(sorted(arrays)) != _CACHE_ARRAY_NAMES:
        raise RuntimeError("internal integer-base cache array schema changed")
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with zipfile.ZipFile(
            temporary,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
            allowZip64=True,
        ) as archive:
            for name in _CACHE_ARRAY_NAMES:
                info = zipfile.ZipInfo(f"{name}.npy", (1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.create_system = 3
                info.external_attr = 0o600 << 16
                with archive.open(info, mode="w", force_zip64=True) as stream:
                    np.lib.format.write_array(
                        stream,
                        np.ascontiguousarray(arrays[name]),
                        allow_pickle=False,
                    )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


@dataclass(frozen=True)
class IntegerMixtureBaseDraw:
    """Immutable audit record for exact draws from an integer mixture base."""

    indices: np.ndarray
    source: np.ndarray
    source_bits: np.ndarray
    u48: np.ndarray
    selected_source_ticks: np.ndarray
    selected_source_cdf_lower: np.ndarray
    probability_numerator: np.ndarray
    probability: np.ndarray
    object_ids: np.ndarray
    source_tick_sum: np.ndarray
    source_positive_tick_count: np.ndarray
    source_minimum_active_tick: np.ndarray
    source_off_support_tick_sum: np.ndarray
    source_terminal_correction: np.ndarray
    source_audit_passed: np.ndarray
    seed: int
    domain: int
    draw_offset: int
    n_atoms: int
    version: str
    base_sha256: str
    indices_sha256: str
    source_sha256: str
    source_bits_sha256: str
    u48_sha256: str
    selected_source_ticks_sha256: str
    selected_source_cdf_lower_sha256: str
    probability_numerator_sha256: str
    probability_sha256: str
    object_ids_sha256: str
    audit_sha256: str
    record_sha256: str

    def __post_init__(self):
        arrays = {
            "indices": _readonly_copy(self.indices, np.int64),
            "source": _readonly_copy(self.source, np.uint8),
            "source_bits": _readonly_copy(self.source_bits, np.uint8),
            "u48": _readonly_copy(self.u48, np.uint64),
            "selected_source_ticks": _readonly_copy(
                self.selected_source_ticks, np.uint64
            ),
            "selected_source_cdf_lower": _readonly_copy(
                self.selected_source_cdf_lower, np.uint64
            ),
            "probability_numerator": _readonly_copy(
                self.probability_numerator, np.uint64
            ),
            "probability": _readonly_copy(self.probability, np.float64),
            "object_ids": _readonly_copy(self.object_ids, np.uint64),
            "source_tick_sum": _readonly_copy(self.source_tick_sum, np.uint64),
            "source_positive_tick_count": _readonly_copy(
                self.source_positive_tick_count, np.uint64
            ),
            "source_minimum_active_tick": _readonly_copy(
                self.source_minimum_active_tick, np.uint64
            ),
            "source_off_support_tick_sum": _readonly_copy(
                self.source_off_support_tick_sum, np.uint64
            ),
            "source_terminal_correction": _readonly_copy(
                self.source_terminal_correction, np.uint64
            ),
            "source_audit_passed": _readonly_copy(
                self.source_audit_passed, np.bool_
            ),
        }
        shape = arrays["indices"].shape
        audit_shape = (2,)
        if (
            len(shape) != 2
            or not shape[0]
            or not shape[1]
            or arrays["source"].shape != shape
            or arrays["source_bits"].shape != shape
            or arrays["u48"].shape != shape
            or arrays["selected_source_ticks"].shape != shape + (2,)
            or arrays["selected_source_cdf_lower"].shape != shape
            or arrays["probability_numerator"].shape != shape
            or arrays["probability"].shape != shape
            or arrays["object_ids"].shape != (shape[0],)
            or any(
                arrays[name].shape != audit_shape
                for name in (
                    "source_tick_sum",
                    "source_positive_tick_count",
                    "source_minimum_active_tick",
                    "source_off_support_tick_sum",
                    "source_terminal_correction",
                    "source_audit_passed",
                )
            )
            or (arrays["indices"] < 0).any()
            or (arrays["source"] > 1).any()
            or (arrays["source_bits"] > 7).any()
            or not np.array_equal(
                arrays["source"], (arrays["source_bits"] == 7).astype(np.uint8)
            )
            or (arrays["u48"] >= _TOTAL_TICKS).any()
            or len(np.unique(arrays["object_ids"])) != shape[0]
        ):
            raise ValueError("integer-mixture draw shapes or discrete fields are invalid")

        scheduled_ticks = np.take_along_axis(
            arrays["selected_source_ticks"],
            arrays["source"][..., None].astype(np.int64),
            axis=2,
        )[..., 0]
        lower = arrays["selected_source_cdf_lower"]
        upper = lower + scheduled_ticks
        expected_numerator = (
            _DETECTED_SOURCE_NUMERATOR
            * arrays["selected_source_ticks"][..., 0]
            + arrays["selected_source_ticks"][..., 1]
        )
        expected_probability = np.ldexp(
            expected_numerator.astype(np.float64),
            -_PROBABILITY_DENOMINATOR_POWER,
        )
        if (
            (scheduled_ticks == 0).any()
            or (lower > arrays["u48"]).any()
            or (arrays["u48"] >= upper).any()
            or (upper > _TOTAL_TICKS).any()
            or not np.array_equal(
                arrays["probability_numerator"], expected_numerator
            )
            or (expected_numerator == 0).any()
            or not np.array_equal(arrays["probability"], expected_probability)
            or not np.array_equal(
                arrays["source_tick_sum"],
                np.full(2, _TOTAL_TICKS, dtype=np.uint64),
            )
            or (arrays["source_positive_tick_count"] == 0).any()
            or arrays["source_minimum_active_tick"][1] < 1
            or arrays["source_off_support_tick_sum"].any()
            or not arrays["source_audit_passed"].all()
        ):
            raise ValueError("integer-mixture draw law audit failed")

        maximum = int(np.iinfo(np.uint64).max)
        seed = _exact_integer(self.seed, "seed", maximum)
        domain = _exact_integer(self.domain, "domain", maximum)
        draw_offset = _exact_integer(self.draw_offset, "draw_offset", maximum)
        n_atoms = _exact_integer(self.n_atoms, "n_atoms", np.iinfo(np.int64).max)
        if n_atoms == 0 or (arrays["indices"] >= n_atoms).any():
            raise ValueError("integer-mixture draw atom indices are out of range")
        if draw_offset + shape[1] > maximum:
            raise ValueError("draw offset and width exceed uint64")
        if self.version != _VERSION:
            raise ValueError("unsupported integer-mixture draw version")
        base_hash = _validate_sha256(self.base_sha256, "base_sha256")

        expected_hashes = {
            "indices_sha256": _array_sha256(
                "draw_indices", arrays["indices"], dtype=">i8"
            ),
            "source_sha256": _array_sha256(
                "draw_source", arrays["source"], dtype=">u1"
            ),
            "source_bits_sha256": _array_sha256(
                "draw_source_bits", arrays["source_bits"], dtype=">u1"
            ),
            "u48_sha256": _array_sha256(
                "draw_u48", arrays["u48"], dtype=">u8"
            ),
            "selected_source_ticks_sha256": _array_sha256(
                "draw_selected_source_ticks",
                arrays["selected_source_ticks"],
                dtype=">u8",
            ),
            "selected_source_cdf_lower_sha256": _array_sha256(
                "draw_selected_source_cdf_lower",
                arrays["selected_source_cdf_lower"],
                dtype=">u8",
            ),
            "probability_numerator_sha256": _array_sha256(
                "draw_probability_numerator",
                arrays["probability_numerator"],
                dtype=">u8",
            ),
            "probability_sha256": _array_sha256(
                "draw_probability", arrays["probability"], dtype=">f8"
            ),
            "object_ids_sha256": _array_sha256(
                "draw_object_ids", arrays["object_ids"], dtype=">u8"
            ),
        }
        audit_hash = _record_sha256(
            "draw_audit",
            (
                _array_sha256(
                    "source_tick_sum", arrays["source_tick_sum"], dtype=">u8"
                ),
                _array_sha256(
                    "source_positive_tick_count",
                    arrays["source_positive_tick_count"],
                    dtype=">u8",
                ),
                _array_sha256(
                    "source_minimum_active_tick",
                    arrays["source_minimum_active_tick"],
                    dtype=">u8",
                ),
                _array_sha256(
                    "source_off_support_tick_sum",
                    arrays["source_off_support_tick_sum"],
                    dtype=">u8",
                ),
                _array_sha256(
                    "source_terminal_correction",
                    arrays["source_terminal_correction"],
                    dtype=">u8",
                ),
                _array_sha256(
                    "source_audit_passed",
                    arrays["source_audit_passed"],
                    dtype=">u1",
                ),
            ),
        )
        expected_record_hash = _record_sha256(
            "draw_record",
            (
                base_hash,
                *expected_hashes.values(),
                audit_hash,
                _text_sha256("seed", seed),
                _text_sha256("domain", domain),
                _text_sha256("draw_offset", draw_offset),
                _text_sha256("n_atoms", n_atoms),
                _text_sha256("version", _VERSION),
                _text_sha256("rng_contract", _RNG_CONTRACT),
                _text_sha256("hash_contract", _HASH_CONTRACT),
            ),
        )
        for name, expected in expected_hashes.items():
            if getattr(self, name) != expected:
                raise ValueError(f"integer-mixture draw {name} is invalid")
        if self.audit_sha256 != audit_hash:
            raise ValueError("integer-mixture draw audit hash is invalid")
        if self.record_sha256 != expected_record_hash:
            raise ValueError("integer-mixture draw record hash is invalid")

        for name, array in arrays.items():
            object.__setattr__(self, name, array)
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "domain", domain)
        object.__setattr__(self, "draw_offset", draw_offset)
        object.__setattr__(self, "n_atoms", n_atoms)
        object.__setattr__(self, "base_sha256", base_hash)

    def verify(self, base: "IntegerMixtureBase") -> None:
        """Recompute the keyed RNG record and every base-dependent law field."""

        if not isinstance(base, IntegerMixtureBase):
            raise TypeError("base must be an IntegerMixtureBase")
        if self.base_sha256 != base.law_sha256 or self.n_atoms != base.n_atoms:
            raise ValueError("integer-mixture draw base identity mismatch")
        expected_source_bits = np.empty(self.source_bits.shape, dtype=np.uint8)
        expected_u48 = np.empty(self.u48.shape, dtype=np.uint64)
        for row, object_id in enumerate(self.object_ids):
            source_raw = _keyed_raw(
                seed=self.seed,
                object_id=int(object_id),
                domain=self.domain,
                purpose=_RNG_SOURCE_PURPOSE_WORD,
                size=self.source_bits.shape[1],
                offset=self.draw_offset,
            )
            threshold_raw = _keyed_raw(
                seed=self.seed,
                object_id=int(object_id),
                domain=self.domain,
                purpose=_RNG_THRESHOLD_PURPOSE_WORD,
                size=self.u48.shape[1],
                offset=self.draw_offset,
            )
            expected_source_bits[row] = np.bitwise_and(
                source_raw, np.uint64(7)
            ).astype(np.uint8)
            expected_u48[row] = np.right_shift(
                threshold_raw, np.uint64(16)
            ).astype(np.uint64)
        if (
            not np.array_equal(self.source_bits, expected_source_bits)
            or not np.array_equal(self.u48, expected_u48)
        ):
            raise ValueError("integer-mixture draw does not match its keyed RNG record")
        expected_ticks = np.stack(
            (
                base.detected_prior_ticks[self.indices],
                base.raw_prior_ticks[self.indices],
            ),
            axis=2,
        )
        selected_cdf = np.where(
            self.source == 0,
            base.detected_prior_cdf[self.indices],
            base.raw_prior_cdf[self.indices],
        )
        scheduled_ticks = np.take_along_axis(
            expected_ticks, self.source[..., None].astype(np.int64), axis=2
        )[..., 0]
        expected_audits = base._audit_arrays()
        if (
            not np.array_equal(self.selected_source_ticks, expected_ticks)
            or not np.array_equal(
                self.selected_source_cdf_lower,
                selected_cdf - scheduled_ticks,
            )
            or any(
                not np.array_equal(getattr(self, name), expected)
                for name, expected in expected_audits.items()
            )
        ):
            raise ValueError("integer-mixture draw does not match its base law")


@dataclass(frozen=True)
class IntegerMixtureBase:
    """Immutable exact-tick full-catalogue base law."""

    active: np.ndarray
    raw_prior_ticks: np.ndarray
    detected_prior_ticks: np.ndarray
    raw_prior_input_sha256: str
    detected_prior_input_sha256: str
    raw_prior_terminal_correction: int = 0
    detected_prior_terminal_correction: int = 0
    raw_prior_cdf: np.ndarray = field(init=False, repr=False)
    detected_prior_cdf: np.ndarray = field(init=False, repr=False)
    active_sha256: str = field(init=False)
    raw_prior_ticks_sha256: str = field(init=False)
    detected_prior_ticks_sha256: str = field(init=False)
    raw_prior_cdf_sha256: str = field(init=False)
    detected_prior_cdf_sha256: str = field(init=False)
    audit_sha256: str = field(init=False)
    law_sha256: str = field(init=False)

    def __post_init__(self):
        active_raw = np.asarray(self.active)
        prior_raw = np.asarray(self.raw_prior_ticks)
        detected_raw = np.asarray(self.detected_prior_ticks)
        if (
            active_raw.ndim != 1
            or not len(active_raw)
            or active_raw.dtype.kind not in "bu"
            or (active_raw.dtype.kind == "u" and (active_raw > 1).any())
            or prior_raw.shape != active_raw.shape
            or detected_raw.shape != active_raw.shape
            or prior_raw.dtype.kind not in "iu"
            or detected_raw.dtype.kind not in "iu"
            or (prior_raw.dtype.kind == "i" and (prior_raw < 0).any())
            or (detected_raw.dtype.kind == "i" and (detected_raw < 0).any())
        ):
            raise ValueError("integer-mixture base arrays are invalid")
        active = _readonly_copy(active_raw, np.bool_)
        prior = _readonly_copy(prior_raw, np.uint64)
        detected = _readonly_copy(detected_raw, np.uint64)
        if (
            not active.any()
            or int(prior.sum(dtype=np.uint64)) != _TOTAL_TICKS
            or int(detected.sum(dtype=np.uint64)) != _TOTAL_TICKS
            or (prior[~active] != 0).any()
            or (detected[~active] != 0).any()
            or (prior[active] < 1).any()
        ):
            raise ValueError("integer-mixture base support or tick totals are invalid")
        prior_cdf = _readonly_copy(np.cumsum(prior, dtype=np.uint64), np.uint64)
        detected_cdf = _readonly_copy(
            np.cumsum(detected, dtype=np.uint64), np.uint64
        )
        if (
            int(prior_cdf[-1]) != _TOTAL_TICKS
            or int(detected_cdf[-1]) != _TOTAL_TICKS
            or (prior_cdf[1:] < prior_cdf[:-1]).any()
            or (detected_cdf[1:] < detected_cdf[:-1]).any()
        ):
            raise ValueError("integer-mixture base CDF audit failed")

        raw_input_hash = _validate_sha256(
            self.raw_prior_input_sha256, "raw_prior_input_sha256"
        )
        detected_input_hash = _validate_sha256(
            self.detected_prior_input_sha256, "detected_prior_input_sha256"
        )
        raw_correction = _exact_integer(
            self.raw_prior_terminal_correction,
            "raw_prior_terminal_correction",
            _TOTAL_TICKS,
        )
        detected_correction = _exact_integer(
            self.detected_prior_terminal_correction,
            "detected_prior_terminal_correction",
            _TOTAL_TICKS,
        )
        hashes = {
            "active_sha256": _array_sha256("active", active, dtype=">u1"),
            "raw_prior_ticks_sha256": _array_sha256(
                "raw_prior_ticks", prior, dtype=">u8"
            ),
            "detected_prior_ticks_sha256": _array_sha256(
                "detected_prior_ticks", detected, dtype=">u8"
            ),
            "raw_prior_cdf_sha256": _array_sha256(
                "raw_prior_cdf", prior_cdf, dtype=">u8"
            ),
            "detected_prior_cdf_sha256": _array_sha256(
                "detected_prior_cdf", detected_cdf, dtype=">u8"
            ),
        }
        audit_hash = _record_sha256(
            "base_audit",
            (
                _text_sha256("active_count", int(np.count_nonzero(active))),
                _text_sha256(
                    "source_tick_sum", [_TOTAL_TICKS, _TOTAL_TICKS]
                ),
                _text_sha256(
                    "source_positive_tick_count",
                    [int(np.count_nonzero(detected)), int(np.count_nonzero(prior))],
                ),
                _text_sha256(
                    "source_minimum_active_tick",
                    [int(detected[active].min()), int(prior[active].min())],
                ),
                _text_sha256("source_off_support_tick_sum", [0, 0]),
                _text_sha256(
                    "source_terminal_correction",
                    [detected_correction, raw_correction],
                ),
            ),
        )
        law_hash = _record_sha256(
            "base_law",
            (
                raw_input_hash,
                detected_input_hash,
                *hashes.values(),
                audit_hash,
                _text_sha256("version", _VERSION),
                _text_sha256("total_ticks", _TOTAL_TICKS),
                _text_sha256("mixture_numerators", [7, 1]),
                _text_sha256("mixture_denominator", 8),
                _text_sha256("quantizer_contract", _QUANTIZER_CONTRACT),
                _text_sha256("hash_contract", _HASH_CONTRACT),
                _text_sha256("order_contract", _ORDER_CONTRACT),
            ),
        )
        object.__setattr__(self, "active", active)
        object.__setattr__(self, "raw_prior_ticks", prior)
        object.__setattr__(self, "detected_prior_ticks", detected)
        object.__setattr__(self, "raw_prior_cdf", prior_cdf)
        object.__setattr__(self, "detected_prior_cdf", detected_cdf)
        object.__setattr__(self, "raw_prior_input_sha256", raw_input_hash)
        object.__setattr__(self, "detected_prior_input_sha256", detected_input_hash)
        object.__setattr__(self, "raw_prior_terminal_correction", raw_correction)
        object.__setattr__(
            self, "detected_prior_terminal_correction", detected_correction
        )
        for name, value in hashes.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "audit_sha256", audit_hash)
        object.__setattr__(self, "law_sha256", law_hash)

    @classmethod
    def from_weights(cls, raw_prior, detected_prior) -> "IntegerMixtureBase":
        raw = _weights(raw_prior, "raw_prior")
        detected = _weights(detected_prior, "detected_prior")
        if detected.shape != raw.shape:
            raise ValueError("raw-prior and detected-prior weights must align")
        active = raw > 0.0
        if not active.any():
            raise ValueError("raw-prior source must have positive support")
        detected_ticks, detected_correction = _quantize_detected(
            detected, active
        )
        prior_ticks, prior_correction = _quantize_prior(raw, active)
        return cls(
            active=active,
            raw_prior_ticks=prior_ticks,
            detected_prior_ticks=detected_ticks,
            raw_prior_input_sha256=_array_sha256(
                "raw_prior_input", raw, dtype=">f8"
            ),
            detected_prior_input_sha256=_array_sha256(
                "detected_prior_input", detected, dtype=">f8"
            ),
            raw_prior_terminal_correction=prior_correction,
            detected_prior_terminal_correction=detected_correction,
        )

    @property
    def n_atoms(self) -> int:
        return len(self.active)

    @property
    def probability_numerator(self) -> np.ndarray:
        numerator = (
            _DETECTED_SOURCE_NUMERATOR * self.detected_prior_ticks
            + self.raw_prior_ticks
        )
        numerator = _readonly_copy(numerator, np.uint64)
        return numerator

    @property
    def probability(self) -> np.ndarray:
        result = np.ldexp(
            self.probability_numerator.astype(np.float64),
            -_PROBABILITY_DENOMINATOR_POWER,
        )
        result.setflags(write=False)
        return result

    def _audit_arrays(self) -> dict[str, np.ndarray]:
        active = self.active
        detected = self.detected_prior_ticks
        prior = self.raw_prior_ticks
        return {
            "source_tick_sum": np.asarray(
                [detected.sum(dtype=np.uint64), prior.sum(dtype=np.uint64)],
                dtype=np.uint64,
            ),
            "source_positive_tick_count": np.asarray(
                [np.count_nonzero(detected), np.count_nonzero(prior)],
                dtype=np.uint64,
            ),
            "source_minimum_active_tick": np.asarray(
                [detected[active].min(), prior[active].min()], dtype=np.uint64
            ),
            "source_off_support_tick_sum": np.asarray(
                [
                    detected[~active].sum(dtype=np.uint64),
                    prior[~active].sum(dtype=np.uint64),
                ],
                dtype=np.uint64,
            ),
            "source_terminal_correction": np.asarray(
                [
                    self.detected_prior_terminal_correction,
                    self.raw_prior_terminal_correction,
                ],
                dtype=np.uint64,
            ),
            "source_audit_passed": np.ones(2, dtype=np.bool_),
        }

    def sample(
        self,
        object_ids,
        n_draws: int,
        *,
        seed: int,
        domain: int,
        draw_offset: int = 0,
    ) -> IntegerMixtureBaseDraw:
        """Draw exact IID mixture atoms for each absolute object ID."""

        ids = _strict_u64_vector(object_ids, "object_ids")
        maximum = int(np.iinfo(np.uint64).max)
        n_draws = _exact_integer(n_draws, "n_draws", maximum)
        if n_draws == 0:
            raise ValueError("n_draws must be positive")
        seed = _exact_integer(seed, "seed", maximum)
        domain = _exact_integer(domain, "domain", maximum)
        draw_offset = _exact_integer(draw_offset, "draw_offset", maximum)
        if draw_offset + n_draws > maximum:
            raise ValueError("draw offset and width exceed uint64")

        shape = (len(ids), n_draws)
        source_bits = np.empty(shape, dtype=np.uint8)
        u48 = np.empty(shape, dtype=np.uint64)
        for row, object_id in enumerate(ids):
            source_raw = _keyed_raw(
                seed=seed,
                object_id=int(object_id),
                domain=domain,
                purpose=_RNG_SOURCE_PURPOSE_WORD,
                size=n_draws,
                offset=draw_offset,
            )
            threshold_raw = _keyed_raw(
                seed=seed,
                object_id=int(object_id),
                domain=domain,
                purpose=_RNG_THRESHOLD_PURPOSE_WORD,
                size=n_draws,
                offset=draw_offset,
            )
            source_bits[row] = np.bitwise_and(
                source_raw, np.uint64(7)
            ).astype(np.uint8)
            u48[row] = np.right_shift(
                threshold_raw, np.uint64(16)
            ).astype(np.uint64)
        source = (source_bits == 7).astype(np.uint8)
        indices = np.empty(shape, dtype=np.int64)
        detected_slots = source == 0
        prior_slots = ~detected_slots
        if detected_slots.any():
            indices[detected_slots] = np.searchsorted(
                self.detected_prior_cdf,
                u48[detected_slots],
                side="right",
            )
        if prior_slots.any():
            indices[prior_slots] = np.searchsorted(
                self.raw_prior_cdf,
                u48[prior_slots],
                side="right",
            )
        selected_ticks = np.stack(
            (
                self.detected_prior_ticks[indices],
                self.raw_prior_ticks[indices],
            ),
            axis=2,
        )
        source_cdf = np.where(
            source == 0,
            self.detected_prior_cdf[indices],
            self.raw_prior_cdf[indices],
        )
        scheduled_ticks = np.take_along_axis(
            selected_ticks, source[..., None].astype(np.int64), axis=2
        )[..., 0]
        lower = source_cdf - scheduled_ticks
        numerator = (
            _DETECTED_SOURCE_NUMERATOR * selected_ticks[..., 0]
            + selected_ticks[..., 1]
        )
        probability = np.ldexp(
            numerator.astype(np.float64), -_PROBABILITY_DENOMINATOR_POWER
        )
        arrays = {
            "indices": indices,
            "source": source,
            "source_bits": source_bits,
            "u48": u48,
            "selected_source_ticks": selected_ticks,
            "selected_source_cdf_lower": lower,
            "probability_numerator": numerator,
            "probability": probability,
            "object_ids": ids,
        }
        hashes = {
            "indices_sha256": _array_sha256(
                "draw_indices", indices, dtype=">i8"
            ),
            "source_sha256": _array_sha256(
                "draw_source", source, dtype=">u1"
            ),
            "source_bits_sha256": _array_sha256(
                "draw_source_bits", source_bits, dtype=">u1"
            ),
            "u48_sha256": _array_sha256("draw_u48", u48, dtype=">u8"),
            "selected_source_ticks_sha256": _array_sha256(
                "draw_selected_source_ticks", selected_ticks, dtype=">u8"
            ),
            "selected_source_cdf_lower_sha256": _array_sha256(
                "draw_selected_source_cdf_lower", lower, dtype=">u8"
            ),
            "probability_numerator_sha256": _array_sha256(
                "draw_probability_numerator", numerator, dtype=">u8"
            ),
            "probability_sha256": _array_sha256(
                "draw_probability", probability, dtype=">f8"
            ),
            "object_ids_sha256": _array_sha256(
                "draw_object_ids", ids, dtype=">u8"
            ),
        }
        audits = self._audit_arrays()
        audit_hash = _record_sha256(
            "draw_audit",
            (
                _array_sha256(
                    "source_tick_sum", audits["source_tick_sum"], dtype=">u8"
                ),
                _array_sha256(
                    "source_positive_tick_count",
                    audits["source_positive_tick_count"],
                    dtype=">u8",
                ),
                _array_sha256(
                    "source_minimum_active_tick",
                    audits["source_minimum_active_tick"],
                    dtype=">u8",
                ),
                _array_sha256(
                    "source_off_support_tick_sum",
                    audits["source_off_support_tick_sum"],
                    dtype=">u8",
                ),
                _array_sha256(
                    "source_terminal_correction",
                    audits["source_terminal_correction"],
                    dtype=">u8",
                ),
                _array_sha256(
                    "source_audit_passed",
                    audits["source_audit_passed"],
                    dtype=">u1",
                ),
            ),
        )
        record_hash = _record_sha256(
            "draw_record",
            (
                self.law_sha256,
                *hashes.values(),
                audit_hash,
                _text_sha256("seed", seed),
                _text_sha256("domain", domain),
                _text_sha256("draw_offset", draw_offset),
                _text_sha256("n_atoms", self.n_atoms),
                _text_sha256("version", _VERSION),
                _text_sha256("rng_contract", _RNG_CONTRACT),
                _text_sha256("hash_contract", _HASH_CONTRACT),
            ),
        )
        return IntegerMixtureBaseDraw(
            **arrays,
            **audits,
            seed=seed,
            domain=domain,
            draw_offset=draw_offset,
            n_atoms=self.n_atoms,
            version=_VERSION,
            base_sha256=self.law_sha256,
            **hashes,
            audit_sha256=audit_hash,
            record_sha256=record_hash,
        )

    def verify_weights(self, raw_prior, detected_prior) -> None:
        """Require external source weights to match this artifact's provenance."""

        raw = _weights(raw_prior, "raw_prior")
        detected = _weights(detected_prior, "detected_prior")
        if raw.shape != self.active.shape or detected.shape != self.active.shape:
            raise ValueError("verification weights do not align with the cached base")
        if (
            _array_sha256("raw_prior_input", raw, dtype=">f8")
            != self.raw_prior_input_sha256
            or _array_sha256("detected_prior_input", detected, dtype=">f8")
            != self.detected_prior_input_sha256
        ):
            raise ValueError("verification weights do not match the cached base")

    def _manifest(self, arrays_file_sha256: str) -> dict:
        arrays_file_hash = _validate_sha256(
            arrays_file_sha256, "arrays_file_sha256"
        )
        return {
            "array_sha256": {
                "active": self.active_sha256,
                "detected_prior_cdf": self.detected_prior_cdf_sha256,
                "detected_prior_ticks": self.detected_prior_ticks_sha256,
                "raw_prior_cdf": self.raw_prior_cdf_sha256,
                "raw_prior_ticks": self.raw_prior_ticks_sha256,
            },
            "arrays_file_sha256": arrays_file_hash,
            "audit": {
                "active_count": int(np.count_nonzero(self.active)),
                "detected_prior_positive_tick_count": int(
                    np.count_nonzero(self.detected_prior_ticks)
                ),
                "detected_prior_terminal_correction": (
                    self.detected_prior_terminal_correction
                ),
                "off_support_tick_sum": [0, 0],
                "raw_prior_minimum_active_tick": int(
                    self.raw_prior_ticks[self.active].min()
                ),
                "raw_prior_positive_tick_count": int(
                    np.count_nonzero(self.raw_prior_ticks)
                ),
                "raw_prior_terminal_correction": (
                    self.raw_prior_terminal_correction
                ),
                "source_tick_sum": [_TOTAL_TICKS, _TOTAL_TICKS],
            },
            "audit_sha256": self.audit_sha256,
            "cache_format": _CACHE_FORMAT,
            "cache_version": _CACHE_VERSION,
            "contracts": {
                "hash": _HASH_CONTRACT,
                "order": _ORDER_CONTRACT,
                "quantizer": _QUANTIZER_CONTRACT,
                "rng": _RNG_CONTRACT,
            },
            "input_sha256": {
                "detected_prior": self.detected_prior_input_sha256,
                "raw_prior": self.raw_prior_input_sha256,
            },
            "law_sha256": self.law_sha256,
            "mixture": {
                "denominator": _MIXTURE_DENOMINATOR,
                "detected_numerator": _DETECTED_SOURCE_NUMERATOR,
                "prior_numerator": _PRIOR_SOURCE_NUMERATOR,
                "probability_denominator_power": _PROBABILITY_DENOMINATOR_POWER,
            },
            "n_atoms": self.n_atoms,
            "total_ticks_per_source": _TOTAL_TICKS,
            "version": _VERSION,
        }

    def save(self, path: str | Path) -> None:
        """Write a deterministic two-file cache into a new or empty directory."""

        root = Path(path)
        if root.exists() and (not root.is_dir() or any(root.iterdir())):
            raise FileExistsError("integer-mixture cache path must be new or empty")
        root.mkdir(parents=True, exist_ok=True)
        arrays_path = root / _ARRAY_FILE
        _write_deterministic_npz(
            arrays_path,
            {
                "active": self.active.astype(np.uint8),
                "detected_prior_cdf": self.detected_prior_cdf,
                "detected_prior_ticks": self.detected_prior_ticks,
                "raw_prior_cdf": self.raw_prior_cdf,
                "raw_prior_ticks": self.raw_prior_ticks,
            },
        )
        manifest = self._manifest(_file_sha256(arrays_path))
        manifest_path = root / _MANIFEST_FILE
        temporary = manifest_path.with_name(f".{manifest_path.name}.tmp")
        try:
            temporary.write_bytes(_canonical_json_bytes(manifest))
            os.replace(temporary, manifest_path)
        finally:
            if temporary.exists():
                temporary.unlink()

    @classmethod
    def load(cls, path: str | Path) -> "IntegerMixtureBase":
        """Load and fully validate a deterministic integer-base cache."""

        root = Path(path)
        if not root.is_dir() or {item.name for item in root.iterdir()} != {
            _ARRAY_FILE,
            _MANIFEST_FILE,
        }:
            raise ValueError("integer-mixture cache file schema is invalid")
        try:
            manifest_bytes = (root / _MANIFEST_FILE).read_bytes()
            manifest = json.loads(manifest_bytes)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("integer-mixture cache manifest is invalid") from error
        if not isinstance(manifest, dict):
            raise ValueError("integer-mixture cache manifest must be an object")
        if manifest_bytes != _canonical_json_bytes(manifest):
            raise ValueError("integer-mixture cache manifest is not canonical")
        arrays_path = root / _ARRAY_FILE
        observed_file_hash = _file_sha256(arrays_path)
        if manifest.get("arrays_file_sha256") != observed_file_hash:
            raise ValueError("integer-mixture cache array-file hash mismatch")
        try:
            with zipfile.ZipFile(arrays_path) as archive:
                expected_entries = [f"{name}.npy" for name in _CACHE_ARRAY_NAMES]
                if archive.namelist() != expected_entries:
                    raise ValueError("integer-mixture cache NPZ schema is invalid")
            with np.load(arrays_path, allow_pickle=False) as archive:
                if tuple(sorted(archive.files)) != _CACHE_ARRAY_NAMES:
                    raise ValueError("integer-mixture cache array schema is invalid")
                active = np.asarray(archive["active"])
                detected_cdf = np.asarray(archive["detected_prior_cdf"])
                detected_ticks = np.asarray(archive["detected_prior_ticks"])
                prior_cdf = np.asarray(archive["raw_prior_cdf"])
                prior_ticks = np.asarray(archive["raw_prior_ticks"])
        except (OSError, ValueError, zipfile.BadZipFile) as error:
            if isinstance(error, ValueError) and str(error).startswith(
                "integer-mixture cache"
            ):
                raise
            raise ValueError("integer-mixture cache arrays are invalid") from error
        if (
            active.dtype != np.dtype(np.uint8)
            or detected_cdf.dtype != np.dtype(np.uint64)
            or detected_ticks.dtype != np.dtype(np.uint64)
            or prior_cdf.dtype != np.dtype(np.uint64)
            or prior_ticks.dtype != np.dtype(np.uint64)
            or active.ndim != 1
            or any(
                value.shape != active.shape
                for value in (detected_cdf, detected_ticks, prior_cdf, prior_ticks)
            )
        ):
            raise ValueError("integer-mixture cache array dtypes or shapes are invalid")
        try:
            input_hashes = manifest["input_sha256"]
            audit = manifest["audit"]
            result = cls(
                active=active.astype(bool),
                raw_prior_ticks=prior_ticks,
                detected_prior_ticks=detected_ticks,
                raw_prior_input_sha256=input_hashes["raw_prior"],
                detected_prior_input_sha256=input_hashes["detected_prior"],
                raw_prior_terminal_correction=audit[
                    "raw_prior_terminal_correction"
                ],
                detected_prior_terminal_correction=audit[
                    "detected_prior_terminal_correction"
                ],
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("integer-mixture cache manifest schema is invalid") from error
        expected_manifest = result._manifest(observed_file_hash)
        if (
            not np.array_equal(result.raw_prior_cdf, prior_cdf)
            or not np.array_equal(result.detected_prior_cdf, detected_cdf)
            or manifest_bytes != _canonical_json_bytes(expected_manifest)
        ):
            raise ValueError("integer-mixture cache semantic identity mismatch")
        return result


__all__ = ["IntegerMixtureBase", "IntegerMixtureBaseDraw"]
