#!/usr/bin/env python
"""Rebuild active prior support from complete scene shards and verified truth views.

Reuse only deterministic neighbour-complete features. Recompute R_blend on the
requested active atoms. A later model-cache build evaluates the new classifier
and flow coordinates. No old probabilities or flow samples are carried over.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sbsi.catalogue_blend import CatalogueBlendResponse
from sbsi.coordinates import ellipticity_from_axis_ratio_angle
from sbsi.forward_catalogue import EmulatorPairingConfig
from sbsi.models import ModelPaths, load_emulator
from sbsi.scene_prior import ScenePrior


def file_hash(path):
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def select_active(galaxies, flow, mag, radius):
    if not (mag[0] < mag[1] and 0 < radius[0] < radius[1]):
        raise ValueError("require ordered magnitude and positive radius bounds")
    expected = pd.Index(np.arange(len(galaxies)), name="primary_row")
    if not flow.index.equals(expected):
        raise ValueError("source truth features do not cover every aligned scene row")
    for feature, truth in (("Re_input_p", "Re"), ("r_input_p", "r"), ("sersic_n_input_p", "sersic_n")):
        np.testing.assert_allclose(flow[feature], galaxies[truth], rtol=0, atol=1e-12)
    e1, e2 = ellipticity_from_axis_ratio_angle(galaxies.axis_ratio, galaxies.position_angle)
    np.testing.assert_allclose(flow.e1_input_p, e1, rtol=0, atol=1e-12)
    np.testing.assert_allclose(flow.e2_input_p, e2, rtol=0, atol=1e-12)
    mask = galaxies.r.between(*mag, inclusive="neither") & galaxies.Re.between(*radius, inclusive="neither")
    active = np.flatnonzero(mask.to_numpy())
    if not len(active):
        raise ValueError("requested prior domain has no atoms")
    return active


def source_identity(args, manifest):
    return {
        "source_prior_manifest": str(args.prior_manifest.resolve()),
        "source_prior_manifest_sha256": file_hash(args.prior_manifest),
        "source_shard_cache_root": str(args.shard_cache_root.resolve()),
        "primary_domain": {"mag": args.primary_mag, "Re": args.primary_re, "radius_kind": "semi_major"},
        "likelihood_config_sha256": file_hash(args.likelihood_config),
        "n_source_shards": len(manifest["shards"]),
    }


def prepare_shard(args, manifest, config):
    index = args.shard_index
    output = args.output / f"shard_{index:02d}"
    if output.exists():
        raise ValueError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    shard = manifest["shards"][index]
    scene_root = Path(shard["root"]) / "scene_store"
    cache_root = args.shard_cache_root / f"shard_{index:02d}"
    report = json.loads((cache_root / "report.json").read_text())
    source = json.loads((cache_root / "model_cache/manifest.json").read_text())
    identity = source_identity(args, manifest)
    if report["default_prior_manifest_sha256"] != identity["source_prior_manifest_sha256"]:
        raise ValueError("source feature cache used a different prior")
    if report["shard_index"] != index or report["cases"] != shard["cases"]:
        raise ValueError("source feature cache shard/cases differ")
    scene_hashes = {name: file_hash(scene_root / name) for name in ("manifest.json", "galaxies.parquet", "neighbours.npz")}
    if scene_hashes != report["scene_sha256"] or any(
        scene_hashes[name] != shard["sha256"][f"scene_store/{name}"] for name in scene_hashes
    ) or source["metadata"]["scene_sha256"] != scene_hashes:
        raise ValueError("source scene hashes differ")
    for name in ("manifest.json", "flow_zero.parquet"):
        if file_hash(cache_root / "model_cache" / name) != report["output_sha256"][f"model_cache/{name}"]:
            raise ValueError("source zero-feature cache hash differs")
    geometry = config["geometry"]
    for name, expected in (("conditions", config["observing_conditions"]),
                           ("flow_neighbour_radius_arcsec", geometry["flow_neighbour_radius_arcsec"]),
                           ("crowding_radii_arcsec", geometry["crowding_radii_arcsec"])):
        if source[name] != expected:
            raise ValueError(f"source feature geometry mismatch: {name}")
    print(f"SHARD_VERIFIED {index}; loading full scene and zero features", flush=True)
    prior = ScenePrior.load(scene_root)
    frame = pd.read_parquet(cache_root / "model_cache/flow_zero.parquet").set_index("primary_row")
    active = select_active(prior.galaxies, frame, args.primary_mag, args.primary_re)
    weights = np.zeros(len(prior.galaxies))
    weights[active] = 1.0
    prior = replace(prior, weights=weights)
    print(f"SHARD_ACTIVE {index} atoms={len(active)}; evaluating response", flush=True)
    emulator_config = config["emulator"]
    for key in ("model", "metadata"):
        if file_hash(emulator_config[key]["path"]) != emulator_config[key]["sha256"]:
            raise ValueError("response model identity mismatch")
    emulator = load_emulator(ModelPaths(
        flow_checkpoints=(args.measurement_model,),
        emulator_model=Path(emulator_config["model"]["path"]),
        emulator_metadata=Path(emulator_config["metadata"]["path"]),
    ), conditions=config["observing_conditions"], device=args.device)
    pairing = EmulatorPairingConfig.from_emulator(emulator, task="regression")
    if pairing.cuts[3][0] > args.primary_re[0] or pairing.cuts[3][1] < args.primary_re[1]:
        raise ValueError("requested primary radius exceeds response training support")
    response = CatalogueBlendResponse.from_emulator(prior, emulator, config=pairing, active_indices=active)
    galaxies = prior.galaxies.iloc[active].reset_index(drop=True).copy()
    galaxies["prior_weight"] = 1.0
    galaxies.to_parquet(output / "galaxies.parquet", index=False)
    frame.iloc[active].reset_index(drop=True).to_parquet(output / "flow_zero.parquet", index=False)
    np.save(output / "active_rows.npy", active)
    np.save(output / "r_blend.npy", response.values[active])
    names = ("galaxies.parquet", "flow_zero.parquet", "active_rows.npy", "r_blend.npy")
    write(output / "report.json", {
        "status": "complete", "identity": identity, "shard_index": index, "cases": shard["cases"],
        "n_source_rows": len(prior.galaxies), "n_active_atoms": len(active),
        "source_scene_sha256": scene_hashes, "source_feature_report_sha256": file_hash(cache_root / "report.json"),
        "source_feature_sha256": file_hash(cache_root / "model_cache/flow_zero.parquet"),
        "pairing_config": asdict(pairing), "response_report": response.report,
        "output_sha256": {name: file_hash(output / name) for name in names},
    })
    print(f"SHARD_COMPLETE {index} atoms={len(active)}", flush=True)


def merge(args, manifest, config):
    output = args.output / "compact"
    if output.exists():
        raise ValueError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    identity = source_identity(args, manifest)
    galaxies_parts, flow_parts, response_parts, reports = [], [], [], []
    pairing = None
    for index, shard in enumerate(manifest["shards"]):
        root = args.output / f"shard_{index:02d}"
        report = json.loads((root / "report.json").read_text())
        if report["identity"] != identity or report["shard_index"] != index or report["cases"] != shard["cases"]:
            raise ValueError("mixed source/domain/shard identities")
        for name, digest in report["output_sha256"].items():
            if file_hash(root / name) != digest:
                raise ValueError(f"shard {index} changed: {name}")
        galaxies = pd.read_parquet(root / "galaxies.parquet")
        frame = pd.read_parquet(root / "flow_zero.parquet")
        response = np.load(root / "r_blend.npy")
        if len(galaxies) != report["n_active_atoms"] or len(frame) != len(galaxies) or response.shape != (len(galaxies),):
            raise ValueError("shard row coverage mismatch")
        if pairing is not None and report["pairing_config"] != pairing:
            raise ValueError("mixed response pairing settings")
        pairing = report["pairing_config"]
        galaxies_parts.append(galaxies)
        flow_parts.append(frame)
        response_parts.append(response)
        reports.append({"shard_index": index, "cases": shard["cases"], "n_active_atoms": len(galaxies),
                        "report_sha256": file_hash(root / "report.json")})
        print(f"MERGE_LOADED {index} atoms={len(galaxies)}", flush=True)
    galaxies = pd.concat(galaxies_parts, ignore_index=True)
    if galaxies.duplicated(["case", "index"]).any():
        raise ValueError("duplicate source atom identities")
    frame = pd.concat(flow_parts, ignore_index=True)
    frame.index.name = "primary_row"
    n = len(galaxies)
    np.testing.assert_array_equal(select_active(galaxies, frame, args.primary_mag, args.primary_re), np.arange(n))
    prior = ScenePrior(galaxies, np.ones(n), np.zeros(n+1, dtype=np.int64), np.empty(0, dtype=np.int64),
                       np.empty(0), np.empty(0), 11.0, group_column="case", metadata={
                           **identity, "n_positive_prior_atoms": n,
                           "purpose": "active_only_inference_cache_with_precomputed_neighbour_views",
                           "neighbour_graph_usage": "precomputed_in_shard_model_views_only"})
    scene_root = output / "scene_store"
    prior.save(scene_root)
    scene_hashes = {name: file_hash(scene_root / name) for name in ("manifest.json", "galaxies.parquet", "neighbours.npz")}
    zero = output / "zero_conditions"
    zero.mkdir()
    frame.reset_index().to_parquet(zero / "flow_zero.parquet", index=False)
    geometry = config["geometry"]
    write(zero / "manifest.json", {
        "kind": "neighbour_complete_zero_conditions", "conditions": config["observing_conditions"],
        "flow_neighbour_radius_arcsec": geometry["flow_neighbour_radius_arcsec"],
        "crowding_radii_arcsec": geometry["crowding_radii_arcsec"],
        "views": [{"g1": 0.0, "g2": 0.0, "flow": "flow_zero.parquet"}],
        "metadata": {"scene_sha256": scene_hashes, "compaction": "positive_atoms_only_after_neighbour_complete_shard_views",
                     "identity": identity, "contains_probabilities_or_flow_samples": False},
    })
    emulator = config["emulator"]
    response = CatalogueBlendResponse(np.concatenate(response_parts), {
        "scene_sha256": scene_hashes, "emulator_sha256": {key: emulator[key]["sha256"] for key in ("model", "metadata")},
        "conditions": config["observing_conditions"], "pairing_config": pairing,
        "source_prior_identity": identity,
    }, {"n_scene_rows": n, "n_active_atoms": n, "n_shards": len(reports)})
    response.save(output / "blend_response")
    write(output / "merge_report.json", {
        "status": "complete", "identity": identity, "n_active_atoms": n, "shards": reports,
        "output_sha256": {
            "scene_store": scene_hashes,
            "zero_conditions": {name: file_hash(zero / name) for name in ("manifest.json", "flow_zero.parquet")},
            "blend_response": {name: file_hash(output / "blend_response" / name) for name in ("manifest.json", "r_blend.npy")},
        },
    })
    print(f"PRIOR_DOMAIN_COMPLETE atoms={n}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-manifest", type=Path, required=True)
    parser.add_argument("--shard-cache-root", type=Path, required=True)
    parser.add_argument("--likelihood-config", type=Path, required=True)
    parser.add_argument("--measurement-model", type=Path, required=True)
    parser.add_argument("--primary-mag", type=float, nargs=2, required=True)
    parser.add_argument("--primary-re", type=float, nargs=2, required=True)
    parser.add_argument("--output", type=Path, required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--shard-index", type=int)
    action.add_argument("--merge", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    manifest = json.loads(args.prior_manifest.read_text())
    config = json.loads(args.likelihood_config.read_text())
    if manifest["kind"] != "sharded_scene_prior":
        raise ValueError("require a complete sharded source prior")
    if file_hash(args.measurement_model) != config["measurement_model"]["sha256"]:
        raise ValueError("flow checkpoint differs from likelihood")
    if args.merge:
        merge(args, manifest, config)
    else:
        if not 0 <= args.shard_index < len(manifest["shards"]):
            raise ValueError("invalid shard index")
        prepare_shard(args, manifest, config)


if __name__ == "__main__":
    main()
