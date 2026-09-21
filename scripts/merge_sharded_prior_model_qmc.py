#!/usr/bin/env python
"""Merge neighbour-complete shard views into an exact active-atom prior cache."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sbsi.catalogue_likelihood import CatalogueModelCache, CatalogueModelView
from sbsi.catalogue_sampling import (
    ProposalCoordinateTable,
    floored_dispersion,
    fractional_mask,
)
from sbsi.scene_prior import ScenePrior


def _sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_hashes(root: str | Path, names) -> dict[str, str]:
    root = Path(root)
    return {name: _sha256(root / name) for name in names}


def _robust_location_scale(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.median(values, axis=0)
    q25, q75 = np.percentile(values, [25, 75], axis=0)
    scale = (q75 - q25) / 1.3489795003921634
    fallback = np.std(values, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 0), scale, fallback)
    scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)
    return center, scale




def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--default-prior-manifest", required=True)
    parser.add_argument("--shard-cache-root", required=True)
    parser.add_argument("--measurement-model", required=True)
    parser.add_argument("--emulator-metadata", required=True)
    parser.add_argument("--emulator-model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--dispersion-floor-percentile",
        type=float,
        default=50.0,
        help=(
            "floor the concatenated per-atom scatter at this percentile of "
            "itself (v1.3-infer: the median)"
        ),
    )
    parser.add_argument(
        "--fractional-floor",
        nargs="*",
        default=("measured_flux_from_mag_auto",),
        metavar="TARGET",
        help="coordinates floored on sigma/|x| rather than on sigma",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    manifest_path = Path(args.default_prior_manifest).resolve()
    default_manifest = json.loads(manifest_path.read_text())
    shards = default_manifest.get("shards", ())
    masses = np.asarray(default_manifest.get("global_shard_mass", ()), dtype=float)
    if (
        default_manifest.get("kind") != "sharded_scene_prior"
        or default_manifest.get("status") != "default_fs2_prior_catalogue"
        or len(shards) != 20
        or masses.shape != (20,)
        or not np.isclose(masses.sum(), 1.0, rtol=0, atol=1e-12)
    ):
        raise RuntimeError("default prior manifest is not a normalized 20-shard prior")
    output = Path(args.output).resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    shard_cache_root = Path(args.shard_cache_root).resolve()
    manifest_sha = _sha256(manifest_path)
    model_hashes = {
        "measurement": _sha256(args.measurement_model),
        "emulator": _sha256(args.emulator_model),
        "emulator_metadata": _sha256(args.emulator_metadata),
    }

    galaxies_parts = []
    weight_parts = []
    flow_parts = []
    detection_parts = []
    probability_parts = []
    coordinate_parts = []
    dispersion_parts = []
    shard_reports = []
    cache_config = None
    coordinate_config = None
    target_names = None
    total_positive = 0
    for index, (shard, mass) in enumerate(zip(shards, masses)):
        shard_output = shard_cache_root / f"shard_{index:02d}"
        report_path = shard_output / "report.json"
        report = json.loads(report_path.read_text())
        if (
            report.get("status") != "complete"
            or report.get("shard_index") != index
            or report.get("default_prior_manifest_sha256") != manifest_sha
            or report.get("model_sha256") != model_hashes
        ):
            raise RuntimeError(f"shard {index} cache provenance mismatch")
        for relative, expected in report["output_sha256"].items():
            if _sha256(shard_output / relative) != expected:
                raise RuntimeError(f"shard {index} cache hash mismatch: {relative}")

        scene_root = Path(shard["root"]) / "scene_store"
        prior = ScenePrior.load(scene_root)
        active = np.flatnonzero(prior.weights > 0)
        expected_positive = int(shard["n_positive_prior_atoms"])
        if len(active) != expected_positive:
            raise RuntimeError(f"shard {index} active count changed")
        total_positive += len(active)
        galaxies_parts.append(prior.galaxies.iloc[active].reset_index(drop=True))
        weight_parts.append(prior.weights[active] * float(mass))

        cache = CatalogueModelCache.load(shard_output / "model_cache", prior=prior)
        view = cache.get(0.0, 0.0)
        flow_parts.append(view.flow.iloc[active].reset_index(drop=True))
        detection_parts.append(view.detection.iloc[active].reset_index(drop=True))
        probability_parts.append(view.detection_probability[active])
        this_cache_config = {
            "conditions": cache.conditions,
            "detection_radius_arcsec": cache.detection_radius_arcsec,
            "flow_neighbour_radius_arcsec": cache.flow_neighbour_radius_arcsec,
            "crowding_radii_arcsec": list(cache.crowding_radii_arcsec),
            "flow_features": list(cache.flow_features or ()),
            "detection_features": list(cache.detection_features),
        }
        if cache_config is None:
            cache_config = this_cache_config
        elif this_cache_config != cache_config:
            raise RuntimeError(f"shard {index} model-cache configuration mismatch")

        coordinates = ProposalCoordinateTable.load(shard_output / "proposal_cache")
        if target_names is None:
            target_names = coordinates.target_names
            coordinate_config = {
                "n_flow_samples": coordinates.n_flow_samples,
                "statistic": coordinates.statistic,
                "dispersion_statistic": coordinates.dispersion_statistic,
                "seed": int(coordinates.metadata["coordinate_config"]["seed"]),
            }
        elif (
            coordinates.target_names != target_names
            or coordinates.n_flow_samples != coordinate_config["n_flow_samples"]
            or coordinates.statistic != coordinate_config["statistic"]
            or coordinates.dispersion_statistic
            != coordinate_config["dispersion_statistic"]
        ):
            raise RuntimeError(f"shard {index} proposal-coordinate mismatch")
        coordinate_parts.append(coordinates.values[active])
        dispersion_parts.append(coordinates.dispersion[active])
        shard_reports.append({
            "shard_index": index,
            "cases": shard["cases"],
            "mass": float(mass),
            "n_positive_prior_atoms": len(active),
            "report_sha256": _sha256(report_path),
        })

    expected_total = int(default_manifest["n_positive_prior_atoms"])
    if total_positive != expected_total:
        raise RuntimeError(
            f"global active count mismatch: expected {expected_total}, found {total_positive}"
        )
    galaxies = pd.concat(galaxies_parts, ignore_index=True)
    weights = np.concatenate(weight_parts)
    if not np.isclose(weights.sum(), 1.0, rtol=0, atol=1e-12):
        raise RuntimeError("merged prior masses do not sum to one")
    compact_prior = ScenePrior(
        galaxies=galaxies,
        weights=weights,
        indptr=np.zeros(len(galaxies) + 1, dtype=np.int64),
        secondary_row=np.empty(0, dtype=np.int64),
        dx_arcsec=np.empty(0, dtype=np.float64),
        dy_arcsec=np.empty(0, dtype=np.float64),
        guard_radius_arcsec=11.0,
        metadata={
            "purpose": "active_only_inference_cache_with_precomputed_neighbour_views",
            "default_prior_manifest": str(manifest_path),
            "default_prior_manifest_sha256": manifest_sha,
            "n_source_rows": int(default_manifest["n_rows"]),
            "n_positive_prior_atoms": expected_total,
            "neighbour_graph_usage": "precomputed_in_shard_model_views_only",
        },
    )
    scene_store = output / "scene_store"
    compact_prior.save(scene_store)
    scene_hashes = _file_hashes(
        scene_store, ("manifest.json", "galaxies.parquet", "neighbours.npz")
    )

    flow = pd.concat(flow_parts, ignore_index=True)
    detection = pd.concat(detection_parts, ignore_index=True)
    probability = np.concatenate(probability_parts)
    aligned_index = pd.Index(
        np.arange(expected_total, dtype=np.int64), name="primary_row"
    )
    flow.index = aligned_index
    detection.index = aligned_index
    cache = CatalogueModelCache(
        compact_prior,
        detector=None,
        conditions=cache_config["conditions"],
        detection_radius_arcsec=cache_config["detection_radius_arcsec"],
        flow_neighbour_radius_arcsec=cache_config["flow_neighbour_radius_arcsec"],
        crowding_radii_arcsec=cache_config["crowding_radii_arcsec"],
        flow_features=cache_config["flow_features"],
        detection_features=cache_config["detection_features"],
    )
    cache._views[(0.0, 0.0)] = CatalogueModelView(
        0.0,
        0.0,
        flow,
        detection,
        probability,
        np.zeros((expected_total, 2), dtype=np.float64),
    )
    model_cache = output / "model_cache"
    cache.save(
        model_cache,
        metadata={
            "purpose": "numerical_catalogue_likelihood_recenter_base",
            "model_sha256": model_hashes,
            "scene_sha256": scene_hashes,
            "flow_features": cache_config["flow_features"],
            "selection": None,
            "r_blend": {"enabled": False, "cache_sha256": None},
            "default_prior_manifest": str(manifest_path),
            "default_prior_manifest_sha256": manifest_sha,
            "compaction": "positive_atoms_only_after_neighbour_complete_shard_views",
        },
    )

    values = np.concatenate(coordinate_parts)
    dispersion = np.concatenate(dispersion_parts)
    center, scale = _robust_location_scale(values)
    dispersion = floored_dispersion(
        values,
        dispersion,
        percentile=args.dispersion_floor_percentile,
        fractional=fractional_mask(target_names, args.fractional_floor),
        fallback=np.maximum(1.0e-3 * scale, np.finfo(np.float64).eps),
    )
    proposal = ProposalCoordinateTable(
        values,
        target_names,
        center,
        scale,
        dispersion=dispersion,
        statistic=coordinate_config["statistic"],
        dispersion_statistic=coordinate_config["dispersion_statistic"],
        n_flow_samples=coordinate_config["n_flow_samples"],
        metadata={
            "model_sha256": model_hashes,
            "scene_sha256": scene_hashes,
            "conditions": cache_config["conditions"],
            "flow_neighbour_radius_arcsec": cache_config[
                "flow_neighbour_radius_arcsec"
            ],
            "crowding_radii_arcsec": cache_config["crowding_radii_arcsec"],
            "coordinate_config": coordinate_config,
            "default_prior_manifest": str(manifest_path),
            "default_prior_manifest_sha256": manifest_sha,
        },
    )
    proposal_cache = output / "proposal_cache"
    proposal.save(proposal_cache)

    report = {
        "status": "complete",
        "default_prior_manifest": str(manifest_path),
        "default_prior_manifest_sha256": manifest_sha,
        "n_source_rows": int(default_manifest["n_rows"]),
        "n_positive_prior_atoms": expected_total,
        "global_shard_mass": masses.tolist(),
        "model_sha256": model_hashes,
        "scene_sha256": scene_hashes,
        "coordinate_config": coordinate_config,
        "shards": shard_reports,
        "output_sha256": {
            "scene_store": _file_hashes(
                scene_store, ("manifest.json", "galaxies.parquet", "neighbours.npz")
            ),
            "model_cache": _file_hashes(
                model_cache,
                ("manifest.json", "flow_zero.parquet", "detection_zero.parquet"),
            ),
            "proposal_cache": _file_hashes(
                proposal_cache, ("manifest.json", "coordinates.npz")
            ),
        },
    }
    (output / "merge_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in (
        "status", "n_source_rows", "n_positive_prior_atoms", "coordinate_config"
    )}, indent=2))


if __name__ == "__main__":
    main()
