#!/usr/bin/env python
"""Uniformly subsample uncut primary atoms, retaining their full pair context."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd


def file_hash(path):
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sample_rows(total, size, seed):
    if not 0 < size <= total:
        raise ValueError("sample size must be positive and no larger than the population")
    drawn = np.random.default_rng(seed).choice(total, size=size, replace=False)
    order = np.argsort(drawn)
    # The ranks retain random draw order, permitting nested smaller subsets.
    return drawn[order], order


def selected_edges(indptr, rows):
    counts = indptr[rows+1] - indptr[rows]
    offsets = np.r_[0, np.cumsum(counts)]
    edges = (np.repeat(indptr[rows], counts) + np.arange(offsets[-1])
             - np.repeat(offsets[:-1], counts))
    return edges, counts


def prepare(source, output, *, size, seed, row_chunk=50000, expected_shards=None):
    source, output = Path(source), Path(output)
    if output.exists() or row_chunk < 1:
        raise ValueError("new output directory and positive row chunk required")
    started = time.monotonic()
    paths = sorted(source.glob("shard_*/manifest.json"))
    if not paths:
        raise ValueError("no source manifests")
    if expected_shards is not None and len(paths) != expected_shards:
        raise ValueError("source shard count differs from requested population")
    manifests = [json.loads(path.read_text()) for path in paths]
    reference = manifests[0]
    shared = ("source_prior_manifest_sha256", "conditions", "pairing", "flow_features", "pair_features")
    cases = []
    for i, manifest in enumerate(manifests):
        if (manifest["status"] != "complete" or manifest["shard_index"] != i
                or manifest["format"] != "uncut_disk_inference_truth_features_v1"
                or any(manifest[key] != reference[key] for key in shared)):
            raise ValueError("incomplete, noncontiguous or incompatible source shards")
        cases.extend(manifest["cases"])
    if len(set(cases)) != len(cases):
        raise ValueError("overlapping source cases")
    offsets = np.r_[0, np.cumsum([m["n_rows"] for m in manifests])]
    rows, ranks = sample_rows(int(offsets[-1]), size, seed)
    output.mkdir(parents=True)
    receipts = []
    for i, (path, manifest) in enumerate(zip(paths, manifests)):
        root = path.parent
        print(f"VERIFY shard={i}", flush=True)
        for name, expected in manifest["output_sha256"].items():
            if file_hash(root / name) != expected:
                raise ValueError(f"source hash mismatch: {root / name}")
        lo, hi = np.searchsorted(rows, offsets[i:i+2])
        local = rows[lo:hi] - offsets[i]
        if not len(local):
            receipts.append(dict(source_shard=i, n_rows=0, source_manifest_sha256=file_hash(path)))
            continue
        target = output / f"shard_{i:02d}"
        target.mkdir()
        np.save(target / "source_atom_ids.npy", rows[lo:hi])
        np.save(target / "sample_rank.npy", ranks[lo:hi])
        galaxies = pd.read_parquet(root / "galaxies.parquet")
        if len(galaxies) != manifest["n_rows"]:
            raise ValueError("source galaxy count mismatch")
        galaxies = galaxies.iloc[local].copy().reset_index(drop=True)
        galaxies["prior_weight"] = 1.0 / size
        galaxies.to_parquet(target / "galaxies.parquet", index=False)
        del galaxies
        zero = pd.read_parquet(root / "flow_zero.parquet")
        if (len(zero) != manifest["n_rows"] or list(zero.columns) != manifest["flow_features"]
                or not zero.index.equals(pd.RangeIndex(len(zero), name="primary_row"))):
            raise ValueError("source flow alignment mismatch")
        zero = zero.iloc[local].reset_index(drop=True)
        zero.index.name = "primary_row"
        zero.to_parquet(target / "flow_zero.parquet")
        del zero
        ptr = np.load(root / "pair_indptr.npy", mmap_mode="r")
        features = np.load(root / "pair_features.npy", mmap_mode="r")
        secondary = np.load(root / "pair_secondary.npy", mmap_mode="r")
        if (ptr.shape != (manifest["n_rows"]+1,) or ptr[0] != 0
                or np.any(np.diff(ptr) < 0) or ptr[-1] != manifest["n_pairs"]
                or features.shape != (ptr[-1], len(manifest["pair_features"]))
                or secondary.shape != (ptr[-1],)):
            raise ValueError("source CSR alignment mismatch")
        selected_ptr = np.r_[0, np.cumsum(ptr[local+1]-ptr[local])]
        np.save(target / "pair_indptr.npy", selected_ptr)
        dest = np.lib.format.open_memmap(target / "pair_features.npy", mode="w+",
            dtype=features.dtype, shape=(int(selected_ptr[-1]), features.shape[1]))
        dest_secondary = np.lib.format.open_memmap(target / "pair_secondary.npy", mode="w+",
            dtype=np.int64, shape=(int(selected_ptr[-1]),))
        for start in range(0, len(local), row_chunk):
            stop = min(start+row_chunk, len(local))
            edges, counts = selected_edges(ptr, local[start:stop])
            neighbours = secondary[edges]
            if (np.any(neighbours < 0) or np.any(neighbours >= manifest["n_rows"])
                    or np.any(neighbours == np.repeat(local[start:stop], counts))):
                raise ValueError("invalid source neighbour identity")
            dest[selected_ptr[start]:selected_ptr[stop]] = features[edges]
            dest_secondary[selected_ptr[start]:selected_ptr[stop]] = neighbours + offsets[i]
        dest.flush()
        dest_secondary.flush()
        del dest, dest_secondary, features, secondary, ptr
        names = ("source_atom_ids.npy", "sample_rank.npy", "galaxies.parquet", "flow_zero.parquet",
                 "pair_indptr.npy", "pair_features.npy", "pair_secondary.npy")
        receipt = dict(source_shard=i, root=str(target.resolve()), n_rows=len(local),
            n_pairs=int(selected_ptr[-1]), source_row_offset=int(offsets[i]),
            source_manifest=str(path.resolve()), source_manifest_sha256=file_hash(path),
            output_sha256={name: file_hash(target / name) for name in names})
        (target / "manifest.json").write_text(json.dumps(receipt, indent=2)+"\n")
        receipts.append(receipt)
        print(f"SUBSET shard={i} atoms={len(local)} pairs={selected_ptr[-1]} elapsed={time.monotonic()-started:.1f}s", flush=True)
    result = dict(status="complete", format="uncut_disk_prior_subset_v1", n_rows=size,
        source_n_rows=int(offsets[-1]), sampling="uniform_without_replacement", seed=seed,
        numpy_version=np.__version__, prior_weight=1.0/size, truth_cuts=None,
        secondary_identity="global source row; neighbours need not belong to sampled atoms",
        sample_rank="zero-based original random order, enabling nested subsamples",
        source_manifest_sha256=[file_hash(p) for p in paths],
        **{key: reference[key] for key in shared}, shards=receipts,
        implementation_sha256=file_hash(__file__), elapsed_seconds=time.monotonic()-started)
    (output / "manifest.json").write_text(json.dumps(result, indent=2)+"\n")
    print(f"SUBSET_COMPLETE atoms={size} output={output}", flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=24000000)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--expected-shards", type=int, default=20)
    args = parser.parse_args()
    prepare(args.source, args.output, size=args.size, seed=args.seed, expected_shards=args.expected_shards)
