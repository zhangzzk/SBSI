#!/usr/bin/env python
# Archived GalSBI-prior experiment.
"""Generate a fresh GalSBI intrinsic base catalogue with explicit provenance."""

from __future__ import annotations

import argparse
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import subprocess

from galsbi import GalSBI


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-index", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--galsbi-repository", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    prefix = Path(args.output_prefix).resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    existing = sorted(prefix.parent.glob(prefix.name + "_*"))
    manifest_path = prefix.parent / f"{prefix.name}_manifest.json"
    if existing or manifest_path.exists():
        rendered = ", ".join(str(path) for path in [*existing, manifest_path])
        raise SystemExit(f"refusing to overwrite existing GalSBI outputs: {rendered}")

    repository = Path(args.galsbi_repository).resolve()
    commit = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(repository), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()

    model = GalSBI(name=args.model)
    model(
        mode="intrinsic",
        model_index=args.model_index,
        file_name=str(prefix),
        seed=args.seed,
    )

    outputs = sorted(
        path for path in prefix.parent.glob(prefix.name + "_*") if path.is_file()
    )
    if not outputs:
        raise RuntimeError("GalSBI completed without producing any catalogue files")
    r_band = prefix.parent / f"{prefix.name}_{args.model_index}_r_ucat.gal.cat"
    if not r_band.is_file():
        raise RuntimeError(f"GalSBI did not produce the required r-band catalogue {r_band}")
    manifest = {
        "generator": "galsbi.GalSBI",
        "model": args.model,
        "mode": "intrinsic",
        "model_index": args.model_index,
        "seed": args.seed,
        "output_prefix": str(prefix),
        "r_band_catalogue": str(r_band),
        "galsbi_version": version("galsbi"),
        "galsbi_repository": str(repository),
        "galsbi_git_commit": commit,
        "galsbi_dirty_paths": dirty,
        "files": {
            path.name: {"size_bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in outputs
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
