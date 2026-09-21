import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location("disk_prior_subsample", Path(__file__).resolve().parents[1] / "scripts/subsample_disk_prior.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_uniform_sample_and_retained_nested_order():
    rows, ranks = module.sample_rows(100, 24, 42)
    expected = np.random.default_rng(42).choice(100, size=24, replace=False)
    np.testing.assert_array_equal(rows[np.argsort(ranks)], expected)
    assert np.all(np.diff(rows) > 0)
    np.testing.assert_array_equal(rows, module.sample_rows(100, 24, 42)[0])
    assert len(rows[ranks < 12]) == 12
    with pytest.raises(ValueError):
        module.sample_rows(10, 11, 42)


def make_source(tmp_path):
    root = tmp_path / "source"
    for i in range(2):
        shard = root / f"shard_{i:02d}"
        shard.mkdir(parents=True)
        pd.DataFrame(dict(case=[i]*4, index=range(4), r=[20., 25., 27., 30.],
                          prior_weight=[1., 0., 0., 0.])).to_parquet(shard / "galaxies.parquet", index=False)
        zero = pd.DataFrame(dict(feature=np.arange(4)+10*i))
        zero.index.name = "primary_row"
        zero.to_parquet(shard / "flow_zero.parquet")
        np.save(shard / "pair_indptr.npy", [0, 1, 1, 2, 3])
        np.save(shard / "pair_secondary.npy", [3, 0, 1])
        np.save(shard / "pair_features.npy", np.arange(6).reshape(3, 2)+10*i)
        names = ("galaxies.parquet", "flow_zero.parquet", "pair_indptr.npy", "pair_secondary.npy", "pair_features.npy")
        manifest = dict(status="complete", format="uncut_disk_inference_truth_features_v1",
            shard_index=i, cases=[i], n_rows=4, n_pairs=3, source_prior_manifest_sha256="source",
            conditions={}, pairing={}, flow_features=["feature"], pair_features=["a", "b"],
            output_sha256={name: module.file_hash(shard / name) for name in names})
        (shard / "manifest.json").write_text(json.dumps(manifest))
    return root


def test_subsample_retains_all_neighbours_and_uniform_uncut_weights(tmp_path):
    source = make_source(tmp_path)
    output = tmp_path / "output"
    result = module.prepare(source, output, size=5, seed=42, row_chunk=1)
    ids, ranks = [], []
    for receipt in result["shards"]:
        target = Path(receipt["root"])
        shard_ids = np.load(target / "source_atom_ids.npy")
        ids.extend(shard_ids)
        ranks.extend(np.load(target / "sample_rank.npy"))
        original = source / f'shard_{receipt["source_shard"]:02d}'
        old_ptr = np.load(original / "pair_indptr.npy")
        edges, counts = module.selected_edges(old_ptr, shard_ids % 4)
        np.testing.assert_array_equal(np.diff(np.load(target / "pair_indptr.npy")), counts)
        np.testing.assert_array_equal(np.load(target / "pair_features.npy"), np.load(original / "pair_features.npy")[edges])
        np.testing.assert_array_equal(np.load(target / "pair_secondary.npy"),
            np.load(original / "pair_secondary.npy")[edges]+receipt["source_row_offset"])
        galaxies = pd.read_parquet(target / "galaxies.parquet")
        np.testing.assert_array_equal(galaxies.prior_weight, np.full(len(shard_ids), .2))
        np.testing.assert_array_equal(galaxies["index"], shard_ids % 4)
    np.testing.assert_array_equal(ids, module.sample_rows(8, 5, 42)[0])
    np.testing.assert_array_equal(ranks, module.sample_rows(8, 5, 42)[1])
    assert result["truth_cuts"] is None
    with pytest.raises(ValueError, match="new output"):
        module.prepare(source, output, size=5, seed=42)


def test_reject_modified_source(tmp_path):
    source = make_source(tmp_path)
    np.save(source / "shard_00/pair_secondary.npy", [2, 0, 1])
    with pytest.raises(ValueError, match="hash mismatch"):
        module.prepare(source, tmp_path / "output", size=5, seed=42)
