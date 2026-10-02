#!/usr/bin/env python
"""Prepare uncut, neighbour-complete truth features for disk inference.

No likelihood probabilities or old response predictions are reused. This is
a full-scene feature store, not a normalized production prior or a fit.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from sbsi.coordinates import ellipticity_from_axis_ratio_angle
from sbsi.crowding import FLOW_FEATURES
from sbsi.flow_size_condition import circularized_radius
from sbsi.forward_catalogue import rescale_emulator_pairs
from sbsi.output_conditioned_response import TRUTH_FEATURES
from sbsi.scene_prior import ScenePrior


def file_hash(path):
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def select_edges(indptr, secondary, distance, start, stop, *, radius, neighbours):
    """Nearest-k strict-aperture edges in the original CSR row convention."""
    edges = np.arange(indptr[start], indptr[stop], dtype=np.int64)
    primary = np.repeat(np.arange(start, stop), np.diff(indptr[start:stop+1]))
    keep = distance[edges] < radius
    edges, primary = edges[keep], primary[keep]
    order = np.lexsort((distance[edges], primary))
    edges, primary = edges[order], primary[order]
    _, starts, counts = np.unique(primary, return_index=True, return_counts=True)
    rank = np.arange(len(edges)) - np.repeat(starts, counts)
    keep = rank < neighbours
    edges, primary = edges[keep], primary[keep]
    return primary, secondary[edges], distance[edges]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-manifest", type=Path, required=True)
    parser.add_argument("--shard-cache-root", type=Path, required=True)
    parser.add_argument("--pairing-metadata", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--row-chunk", type=int, default=50000)
    args = parser.parse_args()
    started = time.monotonic()
    if args.output.exists() or args.row_chunk < 1:
        raise ValueError("new output and positive chunk size required")
    manifest = json.loads(args.prior_manifest.read_text())
    if not 0 <= args.shard_index < len(manifest["shards"]):
        raise ValueError("invalid shard index")
    shard = manifest["shards"][args.shard_index]
    source = Path(shard["root"]) / "scene_store"
    feature_root = args.shard_cache_root / f"shard_{args.shard_index:02d}"
    report = json.loads((feature_root / "report.json").read_text())
    feature_manifest = json.loads((feature_root / "model_cache/manifest.json").read_text())
    if (report["default_prior_manifest_sha256"] != file_hash(args.prior_manifest)
            or report["shard_index"] != args.shard_index or report["cases"] != shard["cases"]):
        raise ValueError("source shard identity differs")
    hashes = {name: file_hash(source / name) for name in ("manifest.json", "galaxies.parquet", "neighbours.npz")}
    if hashes != report["scene_sha256"] or any(hashes[name] != shard["sha256"][f"scene_store/{name}"] for name in hashes):
        raise ValueError("source scene hash mismatch")
    feature_path = feature_root / "model_cache/flow_zero.parquet"
    for name in ("manifest.json", "flow_zero.parquet"):
        if file_hash(feature_root / "model_cache" / name) != report["output_sha256"][f"model_cache/{name}"]:
            raise ValueError("source feature cache hash mismatch")
    expected_pairing_hash = "b3f251d59685e964d81dc744fd301fc9a65015e764d9da6e86de910619d7e261"
    if file_hash(args.pairing_metadata) != expected_pairing_hash:
        raise ValueError("V3.6 development pair-construction metadata differs")
    pairing = json.loads(args.pairing_metadata.read_text())["tasks"]["regression"]
    if pairing["cuts"] is not None or pairing["truth_analysis_cuts_applied"] is not False:
        raise ValueError("pair construction must not impose truth-property cuts")
    radius, neighbours = float(pairing["r_max"]), int(pairing["k"])
    conditions = feature_manifest["conditions"]
    expected_conditions = dict(pixel_size=.2, zero_point=30., psf_fwhm=.73, moffat_beta=2.224, pixel_rms=.312)
    if (conditions != expected_conditions or feature_manifest["flow_neighbour_radius_arcsec"] != 7.
            or feature_manifest["crowding_radii_arcsec"] != [3., 7.]):
        raise ValueError("source observing/flow geometry differs")
    print(f"VERIFIED shard={args.shard_index}; loading full uncut scene", flush=True)
    prior = ScenePrior.load(source)
    if radius > prior.guard_radius_arcsec:
        raise ValueError("source neighbour graph does not cover response aperture")
    galaxies = prior.galaxies
    zero = pd.read_parquet(feature_path).set_index("primary_row")
    if not zero.index.equals(pd.Index(np.arange(len(galaxies)), name="primary_row")):
        raise ValueError("zero context does not cover every scene row")
    for feature, truth in (("Re_input_p", "Re"), ("r_input_p", "r"), ("sersic_n_input_p", "sersic_n")):
        np.testing.assert_allclose(zero[feature], galaxies[truth], rtol=0, atol=1e-12)
    e1, e2 = ellipticity_from_axis_ratio_angle(galaxies.axis_ratio, galaxies.position_angle)
    np.testing.assert_allclose(zero.e1_input_p, e1, rtol=0, atol=1e-12)
    np.testing.assert_allclose(zero.e2_input_p, e2, rtol=0, atol=1e-12)
    zero["circularized_Re_input_p"] = circularized_radius(galaxies.Re, galaxies.axis_ratio)
    zero = zero.loc[:, FLOW_FEATURES]
    args.output.mkdir(parents=True)
    zero.to_parquet(args.output / "flow_zero.parquet")
    galaxies.to_parquet(args.output / "galaxies.parquet", index=False)
    # Count first so the persistent CSR contains no padding or unused rows.
    distance = np.hypot(prior.dx_arcsec, prior.dy_arcsec)
    counts = np.zeros(len(galaxies), dtype=np.int64)
    for start in range(0, len(galaxies), args.row_chunk):
        stop = min(start + args.row_chunk, len(galaxies))
        p, _, _ = select_edges(prior.indptr, prior.secondary_row, distance, start, stop,
                               radius=radius, neighbours=neighbours)
        counts[start:stop] = np.bincount(p-start, minlength=stop-start)
    indptr = np.r_[0, np.cumsum(counts)]
    np.save(args.output / "pair_indptr.npy", indptr)
    features = np.lib.format.open_memmap(args.output / "pair_features.npy", mode="w+",
                                         dtype=np.float64, shape=(int(indptr[-1]), len(TRUTH_FEATURES)))
    secondaries = np.lib.format.open_memmap(args.output / "pair_secondary.npy", mode="w+",
                                            dtype=np.int64, shape=(int(indptr[-1]),))
    truth_values = galaxies[["Re", "r", "sersic_n"]].to_numpy(float)
    for start in range(0, len(galaxies), args.row_chunk):
        stop = min(start + args.row_chunk, len(galaxies))
        p, s, d = select_edges(prior.indptr, prior.secondary_row, distance, start, stop,
                               radius=radius, neighbours=neighbours)
        frame = pd.DataFrame({f"{name}_input_{side}": truth_values[rows, j]
                              for side, rows in (("p", p), ("s", s))
                              for j, name in enumerate(("Re", "r", "sersic_n"))})
        frame["distance"] = d
        scaled = rescale_emulator_pairs(frame, conditions)
        values = scaled.loc[:, TRUTH_FEATURES].to_numpy(float)
        if not np.isfinite(values).all():
            raise ValueError("nonfinite uncut response-pair features")
        features[indptr[start]:indptr[stop]] = values
        secondaries[indptr[start]:indptr[stop]] = s
        print(f"PAIRS shard={args.shard_index} rows={stop}/{len(galaxies)} elapsed={time.monotonic()-started:.1f}s", flush=True)
    features.flush()
    secondaries.flush()
    names = ("galaxies.parquet", "flow_zero.parquet", "pair_indptr.npy", "pair_features.npy", "pair_secondary.npy")
    result = {
        "status": "complete", "format": "uncut_disk_inference_truth_features_v1",
        "shard_index": args.shard_index, "cases": shard["cases"], "n_rows": len(galaxies),
        "n_pairs": int(indptr[-1]), "population": "all source rows; no truth cuts; not yet a normalized prior",
        "source_prior_manifest_sha256": file_hash(args.prior_manifest), "source_scene_sha256": hashes,
        "source_feature_sha256": file_hash(feature_path), "conditions": conditions,
        "pairing": {"r_max_arcsec": radius, "k": neighbours, "cuts": None, "metadata_sha256": expected_pairing_hash},
        "flow_features": list(FLOW_FEATURES), "pair_features": list(TRUTH_FEATURES),
        "true_r_ge26_rows": int((galaxies.r >= 26).sum()),
        "elapsed_seconds": time.monotonic()-started,
        "output_sha256": {name: file_hash(args.output / name) for name in names},
        "implementation_sha256": file_hash(__file__),
    }
    (args.output / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"SHARD_COMPLETE {args.output}", flush=True)


if __name__ == "__main__":
    main()
