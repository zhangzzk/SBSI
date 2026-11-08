#!/usr/bin/env python
"""Build the reusable, model-independent catalogue scene store."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from sbsi.scene_prior import ScenePrior


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalogue", required=True, help="input truth catalogue")
    parser.add_argument("--output", required=True, help="output scene-store directory")
    parser.add_argument(
        "--guard-radius-arcsec",
        required=True,
        type=float,
        help="neighbour radius stored before shear; must exceed every model aperture",
    )
    parser.add_argument("--weight-column", default=None)
    parser.add_argument("--group-column", default=None)
    parser.add_argument(
        "--provenance-manifest",
        action="append",
        default=[],
        help="upstream JSON manifest to hash into the scene-store identity; repeatable",
    )
    parser.add_argument(
        "--duplicate-policy", choices=("error", "drop"), default="error"
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    prior = ScenePrior.from_catalogue(
        args.catalogue,
        guard_radius_arcsec=args.guard_radius_arcsec,
        weight_column=args.weight_column,
        group_column=args.group_column,
        duplicate_policy=args.duplicate_policy,
    )
    catalogue = Path(args.catalogue).resolve()
    manifests = [Path(path).resolve() for path in args.provenance_manifest]
    missing = [str(path) for path in manifests if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing provenance manifests: {missing}")
    prior.save(
        args.output,
        metadata={
            "source_catalogue": str(catalogue),
            "source_catalogue_sha256": _sha256(catalogue),
            "weight_column": args.weight_column,
            "group_column": args.group_column,
            "upstream_manifests": {
                str(path): _sha256(path) for path in manifests
            },
        },
    )
    print(json.dumps(prior.report.__dict__, indent=2))
    print(f"scene store -> {args.output}")


if __name__ == "__main__":
    main()
