#!/usr/bin/env python
"""Build one fixed, atom-aligned BlendEMU response cache for a scene prior."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path

import torch

from sbsi.catalogue_blend import CatalogueBlendResponse
from sbsi.forward_catalogue import EmulatorPairingConfig
from sbsi.models import ModelPaths, load_emulator
from sbsi.scene_prior import ScenePrior


def _sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-store", required=True)
    parser.add_argument(
        "--measurement-model",
        required=True,
        help="companion flow checkpoint required by the shared ModelPaths identity",
    )
    parser.add_argument("--emulator-model", required=True)
    parser.add_argument("--emulator-metadata", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pixel-size", type=float, required=True)
    parser.add_argument("--zero-point", type=float, required=True)
    parser.add_argument("--psf-fwhm", type=float, required=True)
    parser.add_argument("--moffat-beta", type=float, required=True)
    parser.add_argument("--pixel-rms", type=float, required=True)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    output = Path(args.output)
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing path {output}")

    conditions = {
        "pixel_size": args.pixel_size,
        "zero_point": args.zero_point,
        "psf_fwhm": args.psf_fwhm,
        "moffat_beta": args.moffat_beta,
        "pixel_rms": args.pixel_rms,
    }
    scene_root = Path(args.scene_store)
    emulator_model = Path(args.emulator_model)
    emulator_metadata = Path(args.emulator_metadata)
    prior = ScenePrior.load(scene_root)
    paths = ModelPaths(
        flow_checkpoints=(Path(args.measurement_model),),
        emulator_model=emulator_model,
        emulator_metadata=emulator_metadata,
    )
    emulator = load_emulator(paths, conditions=conditions, device=args.device)
    pairing = EmulatorPairingConfig.from_emulator(emulator, task="regression")
    metadata = {
        "scene_sha256": {
            name: _sha256(scene_root / name)
            for name in ("manifest.json", "galaxies.parquet", "neighbours.npz")
        },
        "emulator_sha256": {
            "model": _sha256(emulator_model),
            "metadata": _sha256(emulator_metadata),
        },
        "conditions": conditions,
        "pairing_config": asdict(pairing),
    }
    response = CatalogueBlendResponse.from_emulator(
        prior,
        emulator,
        config=pairing,
        metadata=metadata,
    )
    response.save(output)
    print(json.dumps({"output": str(output), **response.report}, indent=2))


if __name__ == "__main__":
    main()
