"""A bright stratum may be over-sampled, so long as its weight pays it back.

The uniform subset gives every atom weight 1/n.  A stratified subset retains
every source row brighter than a truth cut and pays for the over-sampling with
inverse-probability weights, so both describe the same population.
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sbsi.crowding import FLOW_FEATURES
from sbsi.disk_inference_store import (FrozenDiskCache, load_subset_manifest,
                                       subset_weights)

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "subsample_disk_prior.py"
SPEC = importlib.util.spec_from_file_location("subsample_disk_prior", SCRIPT)
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)

SHARDS, ROWS_PER_SHARD, NEIGHBOURS = 3, 400, 2
TOTAL = SHARDS * ROWS_PER_SHARD
PAIR_FEATURES = ["dx_arcsec", "dy_arcsec", "flux_ratio"]


def write_source(root, seed=7):
    """A miniature `uncut_disk_inference_truth_features_v1` tree."""
    rng = np.random.default_rng(seed)
    root.mkdir(parents=True)
    shared = dict(source_prior_manifest_sha256="0"*64, conditions={"seeing": 0.8},
        pairing=dict(k=NEIGHBOURS, r_max_arcsec=7.0, cuts=None),
        flow_features=list(FLOW_FEATURES), pair_features=PAIR_FEATURES)
    for index in range(SHARDS):
        shard = root / f"shard_{index:02d}"
        shard.mkdir()
        # Magnitudes concentrate at the faint end, as the real prior's do.
        galaxies = pd.DataFrame({"r": 28.0 - rng.exponential(2.0, ROWS_PER_SHARD),
                                 "source_row": np.arange(ROWS_PER_SHARD)})
        galaxies.to_parquet(shard / "galaxies.parquet", index=False)
        zero = pd.DataFrame(np.abs(rng.normal(1.0, 0.1, (ROWS_PER_SHARD, len(FLOW_FEATURES)))),
                            columns=list(FLOW_FEATURES))
        zero.index = pd.RangeIndex(ROWS_PER_SHARD, name="primary_row")
        zero.to_parquet(shard / "flow_zero.parquet")
        indptr = np.arange(ROWS_PER_SHARD+1, dtype=np.int64) * NEIGHBOURS
        np.save(shard / "pair_indptr.npy", indptr)
        np.save(shard / "pair_features.npy",
                rng.normal(0.0, 1.0, (int(indptr[-1]), len(PAIR_FEATURES))))
        secondary = (np.arange(int(indptr[-1]), dtype=np.int64) // NEIGHBOURS + 1) % ROWS_PER_SHARD
        np.save(shard / "pair_secondary.npy", secondary)
        names = ("galaxies.parquet", "flow_zero.parquet", "pair_indptr.npy",
                 "pair_features.npy", "pair_secondary.npy")
        manifest = dict(status="complete", shard_index=index,
            format="uncut_disk_inference_truth_features_v1",
            cases=[f"case_{index}"], n_rows=ROWS_PER_SHARD, n_pairs=int(indptr[-1]),
            output_sha256={n: builder.file_hash(shard / n) for n in names}, **shared)
        (shard / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    return root


def truth_magnitudes(root):
    return np.concatenate([
        pd.read_parquet(root / f"shard_{i:02d}" / "galaxies.parquet")["r"].to_numpy(float)
        for i in range(SHARDS)])


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    return write_source(tmp_path_factory.mktemp("source") / "prior")


def test_the_uniform_draw_is_unchanged_when_no_cut_is_asked_for():
    rows, ranks, probability = builder.stratified_rows(TOTAL, 100, 3, None)
    expected_rows, expected_ranks = builder.sample_rows(TOTAL, 100, 3)
    assert np.array_equal(rows, expected_rows) and np.array_equal(ranks, expected_ranks)
    assert np.allclose(probability, 100/TOTAL)


def test_every_bright_row_survives_the_draw():
    bright = np.array([5, 17, 300, 1100], dtype=np.int64)
    rows, _, probability = builder.stratified_rows(TOTAL, 100, 3, bright)
    assert np.all(np.isin(bright, rows))
    assert np.array_equal(rows, np.unique(rows))
    assert np.allclose(probability[np.isin(rows, bright)], 1.0)
    assert np.allclose(probability[~np.isin(rows, bright)], 100/TOTAL)


def test_a_bright_row_the_uniform_draw_already_found_is_not_duplicated():
    drawn, _ = builder.sample_rows(TOTAL, 300, 11)
    bright = drawn[:5]
    rows, _, _ = builder.stratified_rows(TOTAL, 300, 11, bright)
    assert len(rows) == 300 and np.array_equal(rows, drawn)


def test_a_bright_index_outside_the_population_is_refused():
    with pytest.raises(ValueError, match="outside the source population"):
        builder.stratified_rows(TOTAL, 100, 3, np.array([TOTAL], dtype=np.int64))


def test_the_uniform_build_still_writes_a_v1_manifest(source, tmp_path):
    out = tmp_path / "uniform"
    result = builder.prepare(source, out, size=200, seed=5, expected_shards=SHARDS)
    assert result["format"] == "uncut_disk_prior_subset_v1"
    assert result["sampling"] == "uniform_without_replacement"
    assert result["prior_weight"] == 1/200 and result["n_rows"] == 200
    assert result["bright_stratum"] is None
    manifest = load_subset_manifest(out / "manifest.json")
    assert np.allclose(subset_weights(manifest), 1/200)


def test_the_stratified_build_keeps_every_bright_source_row(source, tmp_path):
    out = tmp_path / "stratified"
    cut = 22.0
    result = builder.prepare(source, out, size=200, seed=5, expected_shards=SHARDS,
                             bright_cut=cut)
    magnitudes = truth_magnitudes(source)
    bright = np.flatnonzero(magnitudes < cut)
    assert len(bright), "the fixture must contain bright rows for this to mean anything"
    assert result["format"] == "uncut_disk_prior_subset_v2"
    assert result["sampling"] == "uniform_plus_certain_stratum"
    assert result["prior_weight"] is None
    assert result["bright_stratum"]["source_rows"] == len(bright)
    assert result["n_rows"] == 200 + result["bright_stratum"]["extra_rows"]
    kept = np.concatenate([np.load(Path(s["root"]) / "source_atom_ids.npy")
                           for s in result["shards"] if s["n_rows"]])
    assert np.all(np.isin(bright, kept))
    assert np.all(truth_magnitudes(source)[kept][np.isin(kept, bright)] < cut)


def test_the_stratified_weights_restore_the_uniform_population(source, tmp_path):
    out = tmp_path / "weighted"
    cut = 22.0
    result = builder.prepare(source, out, size=200, seed=5, expected_shards=SHARDS,
                             bright_cut=cut)
    manifest = load_subset_manifest(out / "manifest.json")
    weights = subset_weights(manifest)
    assert weights.shape == (result["n_rows"],)
    assert weights.sum() == pytest.approx(1.0)
    kept = np.concatenate([np.load(Path(s["root"]) / "source_atom_ids.npy")
                           for s in result["shards"] if s["n_rows"]])
    bright = truth_magnitudes(source)[kept] < cut
    # A bright atom stood in for itself alone; a faint one for TOTAL/size others.
    assert np.allclose(weights[~bright] / weights[bright][0], TOTAL/200)
    assert len(np.unique(np.round(weights, 15))) == 2
    # The bright end is over-represented by count and restored by weight.
    assert bright.mean() > (truth_magnitudes(source) < cut).mean()
    assert weights[bright].sum() == pytest.approx(
        (truth_magnitudes(source) < cut).mean(), rel=0.35)


def test_the_per_shard_weights_line_up_with_the_atoms(source, tmp_path):
    out = tmp_path / "aligned"
    result = builder.prepare(source, out, size=200, seed=5, expected_shards=SHARDS,
                             bright_cut=22.0)
    manifest = load_subset_manifest(out / "manifest.json")
    for index, receipt in enumerate(manifest["shards"]):
        if not receipt["n_rows"]:
            continue
        root = Path(receipt["root"])
        weights = subset_weights(manifest, index)
        assert weights.shape == (receipt["n_rows"],)
        assert np.allclose(weights,
            pd.read_parquet(root / "galaxies.parquet")["prior_weight"].to_numpy(float))
        assert "prior_weight.npy" in receipt["output_sha256"]
        assert builder.file_hash(root / "prior_weight.npy") == receipt["output_sha256"]["prior_weight.npy"]


def test_a_stratified_manifest_claiming_a_scalar_weight_is_refused(source, tmp_path):
    out = tmp_path / "tampered"
    builder.prepare(source, out, size=200, seed=5, expected_shards=SHARDS, bright_cut=22.0)
    path = out / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["prior_weight"] = 1/manifest["n_rows"]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="invalid complete subset manifest"):
        load_subset_manifest(path)


def test_an_unknown_subset_format_is_refused(source, tmp_path):
    out = tmp_path / "unknown"
    builder.prepare(source, out, size=200, seed=5, expected_shards=SHARDS)
    path = out / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["format"] = "uncut_disk_prior_subset_v9"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="invalid complete subset manifest"):
        load_subset_manifest(path)


def test_weights_that_do_not_sum_to_one_are_refused(source, tmp_path):
    out = tmp_path / "unbalanced"
    result = builder.prepare(source, out, size=200, seed=5, expected_shards=SHARDS,
                             bright_cut=22.0)
    receipt = next(s for s in result["shards"] if s["n_rows"])
    path = Path(receipt["root"]) / "prior_weight.npy"
    np.save(path, np.load(path) * 2.0)
    with pytest.raises(ValueError, match="must sum to one"):
        subset_weights(load_subset_manifest(out / "manifest.json"))


def test_the_frozen_cache_carries_the_weights_it_is_given():
    zero = pd.DataFrame(np.abs(np.random.default_rng(1).normal(1.0, 0.1, (6, len(FLOW_FEATURES)))),
                        columns=list(FLOW_FEATURES))
    probabilities = {(0., 0.): np.full(6, 0.5)}
    weights = np.array([0.5, 0.1, 0.1, 0.1, 0.1, 0.1])
    cache = FrozenDiskCache(zero, probabilities, {"seeing": 0.8}, weights=weights)
    assert np.allclose(cache.prior.weights, weights)
    assert np.allclose(FrozenDiskCache(zero, probabilities, {"seeing": 0.8}).prior.weights, 1/6)
    for bad in (np.full(5, 0.2), np.r_[np.full(5, 0.2), 0.0], np.r_[np.full(5, 0.2), np.nan]):
        with pytest.raises(ValueError, match="positive finite prior weight"):
            FrozenDiskCache(zero, probabilities, {"seeing": 0.8}, weights=bad)


# --- a truth cut narrows the population, and must say so --------------------

FAINT = 26.0


def kept_atoms(result):
    return np.concatenate([np.load(Path(r["root"]) / "source_atom_ids.npy")
                           for r in result["shards"] if r["n_rows"]])


def test_the_truth_cut_build_keeps_only_eligible_atoms(source, tmp_path):
    out = tmp_path / "framed"
    result = builder.prepare(source, out, size=200, seed=5, expected_shards=SHARDS,
                             bright_cut=22.0, faint_cut=FAINT)
    magnitudes = truth_magnitudes(source)
    eligible = np.flatnonzero(magnitudes < FAINT)
    assert 200 < len(eligible) < TOTAL, "the fixture must exclude rows for this to mean anything"
    assert result["format"] == "truth_cut_disk_prior_subset_v3"
    assert result["sampling"] == "truth_frame_uniform_plus_certain_stratum"
    assert result["truth_cuts"]["keep_below"] == FAINT
    assert result["truth_cuts"]["frame_rows"] == len(eligible)
    assert result["truth_cuts"]["discarded_rows"] == TOTAL - len(eligible)
    kept = kept_atoms(result)
    # Nothing above the cut survives, and every bright row still does.
    assert np.all(magnitudes[kept] < FAINT)
    assert np.all(np.isin(np.flatnonzero(magnitudes < 22.0), kept))


def test_the_truth_cut_weights_represent_the_frame_not_the_source(source, tmp_path):
    out = tmp_path / "frameweights"
    result = builder.prepare(source, out, size=200, seed=5, expected_shards=SHARDS,
                             bright_cut=22.0, faint_cut=FAINT)
    magnitudes = truth_magnitudes(source)
    frame = int((magnitudes < FAINT).sum())
    manifest = load_subset_manifest(out / "manifest.json")
    weights = subset_weights(manifest)
    assert weights.sum() == pytest.approx(1.0)
    kept = kept_atoms(result)
    bright = magnitudes[kept] < 22.0
    # The inverse probability is taken against the frame, not the whole source.
    assert np.allclose(weights[~bright] / weights[bright][0], frame / 200)
    assert result["bright_stratum"]["uniform_probability"] == pytest.approx(200 / frame)
    # Bright weight recovers the bright share OF THE FRAME, which exceeds its
    # share of the source; that difference is exactly what the cut changed.
    assert weights[bright].sum() == pytest.approx(
        int((magnitudes < 22.0).sum()) / frame, rel=0.35)
    assert weights[bright].sum() > (magnitudes < 22.0).mean()


def test_a_bright_stratum_outside_the_frame_is_refused(source, tmp_path):
    with pytest.raises(ValueError, match="strictly inside the eligible frame"):
        builder.prepare(source, tmp_path / "inverted", size=200, seed=5,
                        expected_shards=SHARDS, bright_cut=26.0, faint_cut=22.0)


def test_a_truth_cut_manifest_must_declare_its_cut(source, tmp_path):
    out = tmp_path / "undeclared"
    builder.prepare(source, out, size=200, seed=5, expected_shards=SHARDS,
                    bright_cut=22.0, faint_cut=FAINT)
    path = out / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["truth_cuts"] = None
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="invalid complete subset manifest"):
        load_subset_manifest(path)


def test_an_uncut_manifest_claiming_a_truth_cut_is_refused(source, tmp_path):
    out = tmp_path / "falsecut"
    builder.prepare(source, out, size=200, seed=5, expected_shards=SHARDS, bright_cut=22.0)
    path = out / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["truth_cuts"] = {"column": "r", "keep_below": 26.0}
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="invalid complete subset manifest"):
        load_subset_manifest(path)


def test_the_frame_restricts_the_draw_even_without_a_bright_stratum():
    eligible = np.arange(0, TOTAL, 3, dtype=np.int64)
    rows, _, probability = builder.stratified_rows(TOTAL, 100, 11, None, eligible)
    assert np.all(np.isin(rows, eligible))
    assert np.allclose(probability, 100 / len(eligible))


def test_an_eligible_index_outside_the_population_is_refused():
    with pytest.raises(ValueError, match="eligible row index outside"):
        builder.stratified_rows(TOTAL, 10, 3, None, np.array([TOTAL], dtype=np.int64))
