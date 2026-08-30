from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

import sbsi.integer_mixture_base as base_module
from sbsi.integer_mixture_base import IntegerMixtureBase


TICKS = 1 << 48


def _base() -> IntegerMixtureBase:
    return IntegerMixtureBase.from_weights(
        np.asarray([1.0, 1.0, 0.0, 2.0], dtype=np.float64),
        np.asarray([1.0, 0.0, 0.0, 3.0], dtype=np.float64),
    )


def _draw_arrays(draw):
    return (
        "indices",
        "source",
        "source_bits",
        "u48",
        "selected_source_ticks",
        "selected_source_cdf_lower",
        "probability_numerator",
        "probability",
    )


def test_tiny_ticks_and_exact_mixture_law_identity():
    base = _base()
    remainder = TICKS - 3
    expected_prior_boundary = np.asarray(
        [
            1 + remainder // 4,
            2 + remainder // 2,
            2 + remainder // 2,
            TICKS,
        ],
        dtype=np.uint64,
    )
    expected_prior_ticks = np.diff(
        np.concatenate((np.zeros(1, dtype=np.uint64), expected_prior_boundary))
    )
    expected_detected_ticks = np.asarray(
        [TICKS // 4, 0, 0, 3 * TICKS // 4], dtype=np.uint64
    )

    np.testing.assert_array_equal(base.active, [True, True, False, True])
    np.testing.assert_array_equal(base.raw_prior_ticks, expected_prior_ticks)
    np.testing.assert_array_equal(
        base.detected_prior_ticks, expected_detected_ticks
    )
    np.testing.assert_array_equal(base.raw_prior_cdf, expected_prior_boundary)
    assert int(base.raw_prior_ticks.sum(dtype=np.uint64)) == TICKS
    assert int(base.detected_prior_ticks.sum(dtype=np.uint64)) == TICKS
    assert (base.raw_prior_ticks[base.active] >= 1).all()
    assert base.raw_prior_ticks[~base.active].sum() == 0
    assert base.detected_prior_ticks[~base.active].sum() == 0

    expected_numerator = (
        7 * base.detected_prior_ticks + base.raw_prior_ticks
    )
    np.testing.assert_array_equal(base.probability_numerator, expected_numerator)
    np.testing.assert_array_equal(
        base.probability,
        np.ldexp(expected_numerator.astype(np.float64), -51),
    )
    exact = [Fraction(int(value), 1 << 51) for value in expected_numerator]
    assert sum(exact) == 1
    assert exact[2] == 0


def test_single_atom_has_unit_probability_for_both_sources():
    base = IntegerMixtureBase.from_weights(
        np.asarray([4.0]), np.asarray([9.0])
    )
    np.testing.assert_array_equal(base.raw_prior_ticks, [TICKS])
    np.testing.assert_array_equal(base.detected_prior_ticks, [TICKS])
    np.testing.assert_array_equal(base.probability_numerator, [1 << 51])
    np.testing.assert_array_equal(base.probability, [1.0])

    draw = base.sample([17], 256, seed=3, domain=9)
    assert not draw.indices.any()
    assert (draw.selected_source_ticks == TICKS).all()
    assert not draw.selected_source_cdf_lower.any()
    assert (draw.probability_numerator == 1 << 51).all()
    assert (draw.probability == 1.0).all()
    assert (draw.source == 0).any()
    assert (draw.source == 1).any()


def test_detected_zero_tick_atoms_keep_defensive_mixture_support():
    base = IntegerMixtureBase.from_weights(
        np.ones(4, dtype=np.float64),
        np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float64),
    )
    np.testing.assert_array_equal(base.detected_prior_ticks, [0, TICKS, 0, 0])
    np.testing.assert_array_equal(
        base.detected_prior_cdf, [0, TICKS, TICKS, TICKS]
    )
    assert (base.raw_prior_ticks >= 1).all()
    assert (base.probability_numerator > 0).all()

    draw = base.sample([4], 4096, seed=21, domain=77)
    detected = draw.source[0] == 0
    assert detected.any()
    assert np.all(draw.indices[0, detected] == 1)
    prior = draw.source[0] == 1
    assert set(draw.indices[0, prior]) == {0, 1, 2, 3}


def test_empirical_source_and_atom_frequencies_match_exact_law():
    base = IntegerMixtureBase.from_weights(
        np.asarray([1.0, 1.0]), np.asarray([3.0, 1.0])
    )
    draw = base.sample([808], 200_000, seed=61, domain=29)
    source_prior = float(np.mean(draw.source))
    atom_zero = float(np.mean(draw.indices == 0))
    exact_atom_zero = float(base.probability[0])

    assert source_prior == pytest.approx(1.0 / 8.0, abs=0.003)
    assert atom_zero == pytest.approx(exact_atom_zero, abs=0.004)


def test_draw_cdf_brackets_ticks_and_probability_are_recomputable():
    base = _base()
    draw = base.sample(
        np.asarray([2**40 + 3, 2**40 + 9], dtype=np.uint64),
        97,
        seed=2**41 + 5,
        domain=2**42 + 7,
    )
    scheduled = np.take_along_axis(
        draw.selected_source_ticks,
        draw.source[..., None].astype(np.int64),
        axis=2,
    )[..., 0]
    assert (scheduled > 0).all()
    assert (draw.selected_source_cdf_lower <= draw.u48).all()
    assert (draw.u48 < draw.selected_source_cdf_lower + scheduled).all()
    np.testing.assert_array_equal(
        draw.selected_source_ticks[..., 0],
        base.detected_prior_ticks[draw.indices],
    )
    np.testing.assert_array_equal(
        draw.selected_source_ticks[..., 1],
        base.raw_prior_ticks[draw.indices],
    )
    expected_numerator = (
        7 * draw.selected_source_ticks[..., 0]
        + draw.selected_source_ticks[..., 1]
    )
    np.testing.assert_array_equal(draw.probability_numerator, expected_numerator)
    np.testing.assert_array_equal(
        draw.probability,
        np.ldexp(expected_numerator.astype(np.float64), -51),
    )
    assert draw.source_tick_sum.tolist() == [TICKS, TICKS]
    assert draw.source_off_support_tick_sum.tolist() == [0, 0]
    assert draw.source_audit_passed.all()
    draw.verify(base)


def test_draw_verify_recomputes_the_keyed_rng_record(monkeypatch):
    base = IntegerMixtureBase.from_weights(
        np.asarray([0.2, 0.3, 0.5]),
        np.asarray([0.1, 0.7, 0.2]),
    )
    draw = base.sample(
        np.asarray([11, 29], dtype=np.uint64),
        17,
        seed=37,
        domain=41,
    )
    original = base_module._keyed_raw

    def shifted_keyed_raw(**kwargs):
        values = original(**kwargs).copy()
        values[0] ^= np.uint64(1 << 20)
        return values

    monkeypatch.setattr(base_module, "_keyed_raw", shifted_keyed_raw)
    with pytest.raises(ValueError, match="keyed RNG record"):
        draw.verify(base)


def test_object_order_and_draw_chunk_do_not_change_stream_mapping():
    base = _base()
    ids = np.asarray([4, 2**38 + 7, 2**39 + 11], dtype=np.uint64)
    arguments = dict(seed=2**40 + 13, domain=2**41 + 17)
    whole = base.sample(ids, 17, **arguments)
    reverse = base.sample(ids[::-1], 17, **arguments)
    first = base.sample(ids, 5, draw_offset=0, **arguments)
    second = base.sample(ids, 12, draw_offset=5, **arguments)

    for name in _draw_arrays(whole):
        np.testing.assert_array_equal(
            getattr(whole, name), getattr(reverse, name)[::-1]
        )
        np.testing.assert_array_equal(
            getattr(whole, name),
            np.concatenate((getattr(first, name), getattr(second, name)), axis=1),
        )


def test_keyed_raw_stream_and_decoding_are_frozen():
    source_raw = base_module._keyed_raw(
        seed=2**40 + 7,
        object_id=2**39 + 11,
        domain=2**38 + 13,
        purpose=base_module._RNG_SOURCE_PURPOSE_WORD,
        size=6,
    )
    threshold_raw = base_module._keyed_raw(
        seed=2**40 + 7,
        object_id=2**39 + 11,
        domain=2**38 + 13,
        purpose=base_module._RNG_THRESHOLD_PURPOSE_WORD,
        size=6,
    )
    np.testing.assert_array_equal(
        source_raw,
        np.asarray(
            [
                15074973826001120041,
                2383694968654272960,
                14491788088031997043,
                5427771255171588163,
                1693204416289836495,
                9519392633123596370,
            ],
            dtype=np.uint64,
        ),
    )
    np.testing.assert_array_equal(
        threshold_raw,
        np.asarray(
            [
                1039605262246674904,
                10853635705495221897,
                17123201528359627879,
                6732900925395891637,
                11282178789404663140,
                10253535862032652641,
            ],
            dtype=np.uint64,
        ),
    )
    np.testing.assert_array_equal(source_raw & np.uint64(7), [1, 0, 3, 3, 7, 2])
    np.testing.assert_array_equal(
        threshold_raw >> np.uint64(16),
        [
            15863117404887,
            165613337791369,
            261279320195917,
            102735914999326,
            172152386312937,
            156456540863535,
        ],
    )


def test_high_words_of_every_key_field_change_stream():
    kwargs = dict(
        seed=19,
        object_id=23,
        domain=29,
        purpose=base_module._RNG_SOURCE_PURPOSE_WORD,
        size=8,
    )
    low = base_module._keyed_raw(**kwargs)
    for name in ("seed", "object_id", "domain"):
        changed = dict(kwargs)
        changed[name] += 1 << 32
        assert not np.array_equal(low, base_module._keyed_raw(**changed))
    threshold = dict(kwargs)
    threshold["purpose"] = base_module._RNG_THRESHOLD_PURPOSE_WORD
    assert not np.array_equal(low, base_module._keyed_raw(**threshold))


def test_inputs_base_and_draw_arrays_are_defensively_immutable():
    raw = np.asarray([1.0, 2.0, 3.0])
    detected = np.asarray([3.0, 2.0, 1.0])
    base = IntegerMixtureBase.from_weights(raw, detected)
    raw[:] = 0.0
    detected[:] = 0.0
    assert int(base.raw_prior_ticks.sum(dtype=np.uint64)) == TICKS
    assert int(base.detected_prior_ticks.sum(dtype=np.uint64)) == TICKS

    draw = base.sample([1, 2], 8, seed=3, domain=4)
    for array in (
        base.active,
        base.raw_prior_ticks,
        base.detected_prior_ticks,
        base.raw_prior_cdf,
        base.detected_prior_cdf,
        draw.indices,
        draw.source,
        draw.u48,
        draw.selected_source_ticks,
        draw.probability,
        draw.source_audit_passed,
    ):
        assert not array.flags.writeable
        with pytest.raises(ValueError, match="read-only"):
            array.flat[0] = 0

    tampered = draw.probability_numerator.copy()
    tampered[0, 0] += 1
    with pytest.raises(ValueError, match="law audit"):
        replace(draw, probability_numerator=tampered)


def test_extreme_weights_are_stably_scaled_and_rescaling_keeps_ticks():
    raw = np.asarray([1.0e-300, 8.0, 1.0e300, 1.0e-200])
    detected = np.asarray([0.0, 1.0e-300, 1.0e300, 0.0])
    base = IntegerMixtureBase.from_weights(raw, detected)
    scaled = IntegerMixtureBase.from_weights(raw * 2.0, detected * 2.0)

    assert (base.raw_prior_ticks >= 1).all()
    assert int(base.raw_prior_ticks.sum(dtype=np.uint64)) == TICKS
    assert int(base.detected_prior_ticks.sum(dtype=np.uint64)) == TICKS
    np.testing.assert_array_equal(base.raw_prior_ticks, scaled.raw_prior_ticks)
    np.testing.assert_array_equal(
        base.detected_prior_ticks, scaled.detected_prior_ticks
    )
    assert base.raw_prior_input_sha256 != scaled.raw_prior_input_sha256
    assert base.detected_prior_input_sha256 != scaled.detected_prior_input_sha256


def test_cache_roundtrip_is_byte_deterministic_and_verifies_inputs(tmp_path):
    raw = np.asarray([1.0, 1.0, 0.0, 2.0])
    detected = np.asarray([1.0, 0.0, 0.0, 3.0])
    base = IntegerMixtureBase.from_weights(raw, detected)
    first = tmp_path / "first"
    second = tmp_path / "second"
    base.save(first)
    base.save(second)

    assert (first / "arrays.npz").read_bytes() == (
        second / "arrays.npz"
    ).read_bytes()
    assert (first / "manifest.json").read_bytes() == (
        second / "manifest.json"
    ).read_bytes()
    loaded = IntegerMixtureBase.load(first)
    assert loaded.law_sha256 == base.law_sha256
    for name in (
        "active",
        "raw_prior_ticks",
        "detected_prior_ticks",
        "raw_prior_cdf",
        "detected_prior_cdf",
        "probability_numerator",
        "probability",
    ):
        np.testing.assert_array_equal(getattr(loaded, name), getattr(base, name))
    loaded.verify_weights(raw, detected)
    with pytest.raises(ValueError, match="do not match"):
        loaded.verify_weights(raw * 2.0, detected)
    original = base.sample([7], 31, seed=8, domain=9)
    restored = loaded.sample([7], 31, seed=8, domain=9)
    for name in _draw_arrays(original):
        np.testing.assert_array_equal(getattr(original, name), getattr(restored, name))
    assert original.record_sha256 == restored.record_sha256


def _copy_cache(source: Path, target: Path) -> None:
    shutil.copytree(source, target)


def test_cache_tampering_and_noncanonical_manifest_fail_closed(tmp_path):
    valid = tmp_path / "valid"
    _base().save(valid)

    array_tamper = tmp_path / "array_tamper"
    _copy_cache(valid, array_tamper)
    path = array_tamper / "arrays.npz"
    payload = bytearray(path.read_bytes())
    payload[len(payload) // 2] ^= 1
    path.write_bytes(payload)
    with pytest.raises(ValueError, match="array-file hash mismatch"):
        IntegerMixtureBase.load(array_tamper)

    manifest_tamper = tmp_path / "manifest_tamper"
    _copy_cache(valid, manifest_tamper)
    path = manifest_tamper / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["mixture"]["detected_numerator"] = 6
    path.write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n")
    with pytest.raises(ValueError, match="semantic identity mismatch"):
        IntegerMixtureBase.load(manifest_tamper)

    whitespace_tamper = tmp_path / "whitespace_tamper"
    _copy_cache(valid, whitespace_tamper)
    path = whitespace_tamper / "manifest.json"
    path.write_bytes(b" " + path.read_bytes())
    with pytest.raises(ValueError, match="not canonical"):
        IntegerMixtureBase.load(whitespace_tamper)

    extra_file = tmp_path / "extra_file"
    _copy_cache(valid, extra_file)
    (extra_file / "unexpected").write_text("x")
    with pytest.raises(ValueError, match="file schema"):
        IntegerMixtureBase.load(extra_file)


@pytest.mark.parametrize(
    ("raw", "detected", "match"),
    (
        ([], [], "nonempty"),
        ([1.0], [1.0, 2.0], "align"),
        ([0.0, 0.0], [1.0, 0.0], "positive support"),
        ([1.0, np.nan], [1.0, 0.0], "finite nonnegative"),
        ([1.0, np.inf], [1.0, 0.0], "finite nonnegative"),
        ([1.0, -1.0], [1.0, 0.0], "finite nonnegative"),
        ([1.0, 1.0], [0.0, 0.0], "positive total"),
        ([1.0, 0.0], [1.0, 1.0], "zero off raw-prior support"),
        ([1.0, 1.0], [1.0, np.nan], "finite nonnegative"),
        ([1.0, 1.0], [1.0, -1.0], "finite nonnegative"),
    ),
)
def test_invalid_weight_inputs_fail_closed(raw, detected, match):
    with pytest.raises(ValueError, match=match):
        IntegerMixtureBase.from_weights(raw, detected)


@pytest.mark.parametrize(
    ("object_ids", "n_draws", "kwargs", "match"),
    (
        ([], 1, {}, "nonempty"),
        ([1.0], 1, {}, "integer vector"),
        ([-1], 1, {}, "nonnegative"),
        ([1, 1], 1, {}, "unique"),
        ([1], 0, {}, "positive"),
        ([1], True, {}, "must be an integer"),
        ([1], 1, {"seed": True}, "must be an integer"),
        ([1], 1, {"domain": -1}, "must be an integer"),
        ([1], 1, {"draw_offset": -1}, "must be an integer"),
        (
            [1],
            2,
            {"draw_offset": np.iinfo(np.uint64).max - 1},
            "exceed uint64",
        ),
    ),
)
def test_invalid_sampling_inputs_fail_closed(object_ids, n_draws, kwargs, match):
    arguments = {"seed": 3, "domain": 4, **kwargs}
    with pytest.raises(ValueError, match=match):
        _base().sample(object_ids, n_draws, **arguments)


def test_cache_path_must_be_new_or_empty(tmp_path):
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "user-data").write_text("preserve")
    with pytest.raises(FileExistsError, match="new or empty"):
        _base().save(occupied)
    assert (occupied / "user-data").read_text() == "preserve"
