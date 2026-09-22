#!/usr/bin/env python
"""Subsample uncut primary atoms, retaining their full pair context.

The draw is uniform by default.  Given ``--bright-cut`` it is uniform plus a
certain stratum: every source row brighter than the cut on a truth column is
retained whether or not the uniform draw found it.  That over-samples the
bright end without changing the population the atoms represent, because each
atom then carries the inverse of its own inclusion probability as its weight.

``--faint-cut`` is different in kind.  It restricts the uniform draw to rows
below a truth magnitude, so rows above it have inclusion probability zero and
no weight can bring them back.  The prior then represents that truth-selected
frame rather than the source, and every result read from it is conditional on
the cut.  Such a subset is written in a distinct format that declares the cut,
so a reader cannot mistake it for the uncut population.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from sbsi.disk_inference_store import TRUTH_CUT_SUBSET


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


def source_rows_below(paths, manifests, offsets, *, column, cuts):
    """Global row indices below each cut, from one pass over the truth column.

    Read before the per-shard hash verification below, which still runs and
    still aborts the build, so a corrupted source cannot reach the output.
    """
    found = {cut: [] for cut in cuts}
    for i, (path, manifest) in enumerate(zip(paths, manifests)):
        values = (pq.read_table(path.parent / "galaxies.parquet", columns=[column])
                    .column(column).to_numpy(zero_copy_only=False).astype(np.float64))
        if len(values) != manifest["n_rows"] or not np.isfinite(values).all():
            raise ValueError(f"unusable truth column {column!r} in source shard {i}")
        for cut in cuts:
            found[cut].append(np.flatnonzero(values < cut).astype(np.int64) + offsets[i])
        print(f"TRUTH shard={i} "
              + " ".join(f"{column}<{cut}={len(found[cut][-1])}" for cut in cuts), flush=True)
    return {cut: np.concatenate(parts) for cut, parts in found.items()}


def stratified_rows(total, size, seed, bright, eligible=None):
    """Uniform draw of `size` rows from `eligible`, plus every bright row missed.

    `eligible` defaults to the whole source.  When it is a truth-selected frame
    the uniform probability is taken against that frame, so the weights below
    represent the frame and not the source: that is a genuine narrowing of the
    population, unlike the bright stratum, which leaves it unchanged.

    Returns globally sorted row indices, their draw ranks, and the probability
    with which each was included: one for a bright row, `size/frame` otherwise.
    """
    if eligible is None:
        drawn, ranks = sample_rows(total, size, seed)
        frame = total
    else:
        eligible = np.unique(np.asarray(eligible, dtype=np.int64))
        if len(eligible) and (eligible[0] < 0 or eligible[-1] >= total):
            raise ValueError("eligible row index outside the source population")
        local, ranks = sample_rows(len(eligible), size, seed)
        drawn, frame = eligible[local], len(eligible)
    uniform = size / frame
    if bright is None:
        return drawn, ranks, np.full(len(drawn), uniform)
    bright = np.unique(np.asarray(bright, dtype=np.int64))
    if len(bright) and (bright[0] < 0 or bright[-1] >= total):
        raise ValueError("bright row index outside the source population")
    if eligible is not None and not np.isin(bright, eligible, assume_unique=True).all():
        raise ValueError("bright stratum must lie inside the eligible frame")
    extra = bright[~np.isin(bright, drawn, assume_unique=True)]
    rows = np.concatenate([drawn, extra])
    ranks = np.concatenate([ranks, size+np.arange(len(extra))])
    probability = np.where(np.isin(rows, bright, assume_unique=True), 1., uniform)
    order = np.argsort(rows, kind="stable")
    return rows[order], ranks[order], probability[order]


def selected_edges(indptr, rows):
    counts = indptr[rows+1] - indptr[rows]
    offsets = np.r_[0, np.cumsum(counts)]
    edges = (np.repeat(indptr[rows], counts) + np.arange(offsets[-1])
             - np.repeat(offsets[:-1], counts))
    return edges, counts


def prepare(source, output, *, size, seed, row_chunk=50000, expected_shards=None,
            bright_cut=None, bright_column="r", faint_cut=None):
    source, output = Path(source), Path(output)
    if output.exists() or row_chunk < 1:
        raise ValueError("new output directory and positive row chunk required")
    if faint_cut is not None and bright_cut is not None and not bright_cut < faint_cut:
        raise ValueError("the bright stratum must be strictly inside the eligible frame")
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
    total = int(offsets[-1])
    wanted = [cut for cut in (bright_cut, faint_cut) if cut is not None]
    below = (source_rows_below(paths, manifests, offsets, column=bright_column, cuts=wanted)
             if wanted else {})
    bright = below.get(bright_cut)
    eligible = below.get(faint_cut)
    rows, ranks, probability = stratified_rows(total, size, seed, bright, eligible)
    # Hajek normalization: the inverse-probability weights are scaled to sum to
    # one over the subset, exactly as the uniform 1/n weights do by construction.
    weights = (1./probability) / np.sum(1./probability)
    n_atoms = len(rows)
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
        np.save(target / "prior_weight.npy", weights[lo:hi])
        galaxies = pd.read_parquet(root / "galaxies.parquet")
        if len(galaxies) != manifest["n_rows"]:
            raise ValueError("source galaxy count mismatch")
        galaxies = galaxies.iloc[local].copy().reset_index(drop=True)
        galaxies["prior_weight"] = weights[lo:hi]
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
        names = ("source_atom_ids.npy", "sample_rank.npy", "prior_weight.npy",
                 "galaxies.parquet", "flow_zero.parquet",
                 "pair_indptr.npy", "pair_features.npy", "pair_secondary.npy")
        receipt = dict(source_shard=i, root=str(target.resolve()), n_rows=len(local),
            n_pairs=int(selected_ptr[-1]), source_row_offset=int(offsets[i]),
            source_manifest=str(path.resolve()), source_manifest_sha256=file_hash(path),
            output_sha256={name: file_hash(target / name) for name in names})
        (target / "manifest.json").write_text(json.dumps(receipt, indent=2)+"\n")
        receipts.append(receipt)
        print(f"SUBSET shard={i} atoms={len(local)} pairs={selected_ptr[-1]} elapsed={time.monotonic()-started:.1f}s", flush=True)
    frame = total if eligible is None else int(len(eligible))
    stratum = None if bright is None else dict(column=bright_column, cut=bright_cut,
        source_rows=int(len(bright)), uniform_rows=int(size), extra_rows=int(n_atoms-size),
        uniform_probability=size/frame, weight_bright=float(weights.min()),
        weight_uniform=float(weights.max()))
    # Only a truth cut narrows the represented population, so only it declares
    # cuts; the bright stratum is paid back in weight and leaves it unchanged.
    truth_cuts = None if faint_cut is None else dict(column=bright_column,
        keep_below=faint_cut, frame_rows=frame, source_rows=total,
        frame_fraction=frame/total, discarded_rows=total-frame)
    if faint_cut is not None:
        fmt, sampling = TRUTH_CUT_SUBSET, "truth_frame_uniform_plus_certain_stratum"
    elif bright is None:
        fmt, sampling = "uncut_disk_prior_subset_v1", "uniform_without_replacement"
    else:
        fmt, sampling = "uncut_disk_prior_subset_v2", "uniform_plus_certain_stratum"
    result = dict(status="complete", n_rows=n_atoms, format=fmt,
        source_n_rows=total, seed=seed, sampling=sampling,
        numpy_version=np.__version__,
        prior_weight=1.0/size if fmt == "uncut_disk_prior_subset_v1" else None,
        truth_cuts=truth_cuts, bright_stratum=stratum,
        secondary_identity="global source row; neighbours need not belong to sampled atoms",
        sample_rank="zero-based original random order; nesting holds below the uniform size",
        source_manifest_sha256=[file_hash(p) for p in paths],
        **{key: reference[key] for key in shared}, shards=receipts,
        implementation_sha256=file_hash(__file__), elapsed_seconds=time.monotonic()-started)
    (output / "manifest.json").write_text(json.dumps(result, indent=2)+"\n")
    print(f"SUBSET_COMPLETE atoms={n_atoms} output={output}", flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=24000000)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--expected-shards", type=int, default=20)
    parser.add_argument("--bright-cut", type=float, default=None,
        help="also retain every source row below this truth magnitude, with "
             "inverse-probability weights; leaving it unset draws uniformly")
    parser.add_argument("--bright-column", default="r",
        help="truth magnitude column both cuts read (default: r)")
    parser.add_argument("--faint-cut", type=float, default=None,
        help="draw the uniform part only from source rows below this truth "
             "magnitude.  Unlike the bright stratum this NARROWS the population "
             "the prior represents to that frame, so the result is conditional "
             "on it; leaving it unset keeps the full source")
    args = parser.parse_args()
    prepare(args.source, args.output, size=args.size, seed=args.seed,
        expected_shards=args.expected_shards, bright_cut=args.bright_cut,
        bright_column=args.bright_column, faint_cut=args.faint_cut)
