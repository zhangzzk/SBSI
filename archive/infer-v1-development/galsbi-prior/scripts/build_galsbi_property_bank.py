#!/usr/bin/env python
# Archived GalSBI-prior experiment.
"""Merge independently generated GalSBI catalogues into one property bank.

The quality processing is delegated to the exact BlendSim catalogue loader so
the bank and image-simulation catalogue step cannot drift apart.  Sky density
is deliberately absent here; randomized one-degree cases choose their own
fixed row count later.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_identity(repository: Path) -> dict:
    revision = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    commit = revision.stdout.strip() if revision.returncode == 0 else None
    dirty = None
    if commit is not None:
        dirty = subprocess.run(
            ["git", "-C", str(repository), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    catalogue_source = repository / "blendsim" / "catalog.py"
    if not catalogue_source.is_file():
        raise FileNotFoundError(catalogue_source)
    return {
        "repository": str(repository),
        "git_commit": commit,
        "dirty_paths": dirty,
        "catalogue_source": str(catalogue_source),
        "catalogue_source_sha256": _sha256(catalogue_source),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalogue-manifest", action="append", required=True,
        help="manifest from generate_galsbi_prior_base.py; repeat for each seed",
    )
    parser.add_argument("--blendsim-repository", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model", default="Fischbacher+24")
    parser.add_argument("--model-index", type=int, default=0)
    parser.add_argument("--min-usable", type=int, default=1_000_000)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.min_usable <= 0:
        raise SystemExit("--min-usable must be positive")
    output = Path(args.output).resolve()
    manifest_output = Path(args.manifest).resolve()
    if manifest_output.exists():
        raise SystemExit("refusing to overwrite property-bank manifest")
    resume_output = output.exists()

    blendsim_repository = Path(args.blendsim_repository).resolve()
    sys.path.insert(0, str(blendsim_repository))
    from blendsim.catalog import load_and_process_base_catalog

    frames = []
    inputs = []
    seen_seeds = set()
    for manifest_name in args.catalogue_manifest:
        manifest_path = Path(manifest_name).resolve()
        payload = json.loads(manifest_path.read_text())
        identity = (payload.get("model"), payload.get("mode"), payload.get("model_index"))
        expected = (args.model, "intrinsic", args.model_index)
        if identity != expected:
            raise ValueError(
                f"{manifest_path} declares {identity}, expected {expected}"
            )
        seed = int(payload["seed"])
        if seed in seen_seeds:
            raise ValueError(f"duplicate GalSBI seed {seed}")
        seen_seeds.add(seed)
        catalogue = Path(payload["r_band_catalogue"]).resolve()
        if not catalogue.is_file():
            raise FileNotFoundError(catalogue)
        frame, reported_density = load_and_process_base_catalog(path=str(catalogue))
        frame = frame.copy()
        frame["property_seed"] = seed
        frame["source_usable_row"] = range(len(frame))
        frames.append(frame)
        inputs.append(
            {
                "seed": seed,
                "manifest": str(manifest_path),
                "manifest_sha256": _sha256(manifest_path),
                "catalogue": str(catalogue),
                "catalogue_sha256": _sha256(catalogue),
                "n_usable": int(len(frame)),
                "legacy_reported_density_per_degree2": float(reported_density),
            }
        )

    bank = pd.concat(frames, ignore_index=True)
    if len(bank) < args.min_usable:
        raise RuntimeError(
            f"only {len(bank):,} usable galaxies; generate another independent seed "
            f"and rerun to reach {args.min_usable:,}"
        )
    bank.insert(0, "property_id", range(len(bank)))
    if bank["property_id"].duplicated().any():
        raise RuntimeError("property identifiers are not unique")

    output.parent.mkdir(parents=True, exist_ok=True)
    if resume_output:
        existing = pd.read_parquet(output)
        if not existing.equals(bank):
            raise RuntimeError(
                "existing property bank without a manifest does not equal the rebuilt bank"
            )
    else:
        bank.to_parquet(output, index=False)
    report = {
        "format_version": 1,
        "purpose": "property bank; size is independent of scene sky density",
        "model": args.model,
        "mode": "intrinsic",
        "model_index": args.model_index,
        "seeds": sorted(seen_seeds),
        "min_usable": args.min_usable,
        "n_usable": int(len(bank)),
        "quality_pipeline": "blendsim.catalog.load_and_process_base_catalog",
        "blendsim": _git_identity(blendsim_repository),
        "inputs": inputs,
        "output": str(output),
        "output_sha256": _sha256(output),
        "columns": list(bank.columns),
    }
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
