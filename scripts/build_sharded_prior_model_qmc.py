#!/usr/bin/env python
"""Build one neighbour-complete model/QMC cache for a sharded ScenePrior."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from sbsi.catalogue_likelihood import CatalogueLikelihood, CatalogueModelCache
from sbsi.catalogue_sampling import ProposalCoordinateTable
from sbsi.forward_catalogue import EmulatorPairingConfig
from sbsi.measurement_model import load_measurement_model
from sbsi.models import ModelPaths, load_emulator
from sbsi.scene_prior import ScenePrior


def _sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hashes(root: str | Path, names) -> dict[str, str]:
    root = Path(root)
    return {name: _sha256(root / name) for name in names}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--default-prior-manifest", required=True)
    parser.add_argument("--shard-index", required=True, type=int)
    parser.add_argument("--measurement-model", required=True)
    parser.add_argument("--emulator-metadata", required=True)
    parser.add_argument("--emulator-model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--qmc-samples", type=int, default=128)
    parser.add_argument("--qmc-row-chunk", type=int, default=8192)
    parser.add_argument("--qmc-seed", type=int, default=8201)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pixel-size", type=float, default=0.2)
    parser.add_argument("--zero-point", type=float, default=30.0)
    parser.add_argument("--psf-fwhm", type=float, default=0.73)
    parser.add_argument("--moffat-beta", type=float, default=2.224)
    parser.add_argument("--pixel-rms", type=float, default=0.312)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    manifest_path = Path(args.default_prior_manifest).resolve()
    manifest = json.loads(manifest_path.read_text())
    shards = manifest.get("shards", ())
    if (
        manifest.get("kind") != "sharded_scene_prior"
        or manifest.get("status") != "default_fs2_prior_catalogue"
        or len(shards) != 20
    ):
        raise RuntimeError("default prior manifest is not the complete 20-shard prior")
    if not 0 <= args.shard_index < len(shards):
        raise SystemExit("--shard-index is outside the manifest")
    output = Path(args.output).resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}")
    output.mkdir(parents=True)

    shard = shards[args.shard_index]
    scene_root = Path(shard["root"]) / "scene_store"
    prior = ScenePrior.load(scene_root)
    expected = int(shard["n_positive_prior_atoms"])
    actual = int(np.count_nonzero(prior.weights > 0))
    if actual != expected:
        raise RuntimeError(f"positive atom mismatch: expected {expected}, found {actual}")

    conditions = {
        "pixel_size": args.pixel_size,
        "zero_point": args.zero_point,
        "psf_fwhm": args.psf_fwhm,
        "moffat_beta": args.moffat_beta,
        "pixel_rms": args.pixel_rms,
    }
    model_hashes = {
        "measurement": _sha256(args.measurement_model),
        "emulator": _sha256(args.emulator_model),
        "emulator_metadata": _sha256(args.emulator_metadata),
    }
    scene_hashes = _hashes(
        scene_root, ("manifest.json", "galaxies.parquet", "neighbours.npz")
    )
    # The shard manifest also hashes its assembly report; compare only the
    # three ScenePrior entries declared above.
    expected_scene = {
        name: shard["sha256"][f"scene_store/{name}"] for name in scene_hashes
    }
    if scene_hashes != expected_scene:
        raise RuntimeError("shard ScenePrior hashes do not match the default manifest")

    flow = load_measurement_model(args.measurement_model, device=args.device)
    detector = load_emulator(
        ModelPaths(
            flow_checkpoints=(Path(args.measurement_model),),
            emulator_model=Path(args.emulator_model),
            emulator_metadata=Path(args.emulator_metadata),
        ),
        conditions=conditions,
        device=args.device,
    )
    cache = CatalogueModelCache(
        prior,
        detector=detector,
        conditions=conditions,
        detection_radius_arcsec=3.0,
        flow_neighbour_radius_arcsec=7.0,
        crowding_radii_arcsec=(3.0, 7.0),
        flow_features=flow.condition_preprocessor.feature_names,
    )
    cache.get(0.0, 0.0)
    cache.validate_model_features(flow)
    detection_features = cache.validate_detection_shear_invariance()
    model_cache = output / "model_cache"
    cache.save(
        model_cache,
        metadata={
            "purpose": "fs2_default_shard_zero_view_for_compaction",
            "default_prior_manifest_sha256": _sha256(manifest_path),
            "shard_index": int(args.shard_index),
            "global_shard_mass": float(manifest["global_shard_mass"][args.shard_index]),
            "model_sha256": model_hashes,
            "scene_sha256": scene_hashes,
            "pairing_config": json.loads(json.dumps(asdict(
                EmulatorPairingConfig.from_emulator(detector, task="regression")
            ))),
            "r_blend": {"enabled": False, "cache_sha256": None},
        },
    )
    likelihood = CatalogueLikelihood(flow, cache)
    coordinates = ProposalCoordinateTable.from_flow(
        likelihood,
        target_names=likelihood.target_names,
        n_flow_samples=args.qmc_samples,
        statistic="mean",
        dispersion_statistic="std",
        row_chunk=args.qmc_row_chunk,
        seed=args.qmc_seed,
        # Preserve raw positive standard deviations.  The global one-percent
        # floor is applied only after all twenty shards are concatenated.
        dispersion_floor_percentile=0.0,
        metadata={
            "purpose": "fs2_default_shard_qmc_moments_for_compaction",
            "default_prior_manifest_sha256": _sha256(manifest_path),
            "shard_index": int(args.shard_index),
            "coordinate_config": {
                "n_flow_samples": int(args.qmc_samples),
                "statistic": "mean",
                "dispersion_statistic": "std",
                "seed": int(args.qmc_seed),
            },
        },
    )
    proposal_cache = output / "proposal_cache"
    coordinates.save(proposal_cache)
    report = {
        "status": "complete",
        "shard_index": int(args.shard_index),
        "cases": shard["cases"],
        "n_scene_rows": int(len(prior.galaxies)),
        "n_positive_prior_atoms": actual,
        "global_shard_mass": float(manifest["global_shard_mass"][args.shard_index]),
        "default_prior_manifest": str(manifest_path),
        "default_prior_manifest_sha256": _sha256(manifest_path),
        "model_sha256": model_hashes,
        "scene_sha256": scene_hashes,
        "detection_features": list(detection_features),
        "qmc_samples": int(args.qmc_samples),
        "output_sha256": {
            "model_cache/manifest.json": _sha256(model_cache / "manifest.json"),
            "model_cache/flow_zero.parquet": _sha256(model_cache / "flow_zero.parquet"),
            "model_cache/detection_zero.parquet": _sha256(model_cache / "detection_zero.parquet"),
            "proposal_cache/manifest.json": _sha256(proposal_cache / "manifest.json"),
            "proposal_cache/coordinates.npz": _sha256(proposal_cache / "coordinates.npz"),
        },
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in (
        "status", "shard_index", "n_positive_prior_atoms", "qmc_samples"
    )}, indent=2))


if __name__ == "__main__":
    main()
