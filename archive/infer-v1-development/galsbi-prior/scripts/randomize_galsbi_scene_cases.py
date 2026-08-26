#!/usr/bin/env python
# Archived GalSBI-prior experiment.
"""Draw fixed-density randomized one-degree scenes from a GalSBI property bank."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--property-bank", required=True)
    parser.add_argument("--property-manifest", required=True)
    parser.add_argument("--blendsim-repository", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--n-cases", type=int, default=100)
    parser.add_argument("--case-start", type=int, default=0)
    parser.add_argument("--rows-per-case", type=int, default=71_104)
    parser.add_argument("--primary-mag-min", type=float, default=18.0)
    parser.add_argument("--primary-mag-max", type=float, default=25.8)
    parser.add_argument("--primary-re-min", type=float, default=0.5)
    parser.add_argument("--primary-re-max", type=float, default=1.5)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.n_cases <= 0 or args.rows_per_case <= 0:
        raise SystemExit("case and row counts must be positive")
    if args.rows_per_case % 2:
        raise SystemExit("--rows-per-case must be even to match the BlendSim helper")
    output = Path(args.output).resolve()
    report_path = Path(args.report).resolve()
    if output.exists() or report_path.exists():
        raise SystemExit("refusing to overwrite randomized scenes or report")

    bank_path = Path(args.property_bank).resolve()
    bank_manifest_path = Path(args.property_manifest).resolve()
    bank_manifest = json.loads(bank_manifest_path.read_text())
    if bank_manifest.get("output_sha256") != _sha256(bank_path):
        raise RuntimeError("property-bank hash does not match its manifest")
    bank = pd.read_parquet(bank_path)
    required = {
        "property_id", "property_seed", "shape/sersic_n", "sdss_r",
        "Re_arcsec", "BA", "redshift", "angle",
    }
    missing = sorted(required - set(bank))
    if missing:
        raise KeyError(f"property bank lacks columns: {missing}")

    blendsim_repository = Path(args.blendsim_repository).resolve()
    sys.path.insert(0, str(blendsim_repository))
    from blendsim.catalog import generate_catalog_realization

    frames = []
    case_reports = []
    for case in range(args.case_start, args.case_start + args.n_cases):
        seed = case + 123
        frame = generate_catalog_realization(
            bank, args.rows_per_case, seed=seed, g=0.0
        )
        bank_rows = frame["cata_idx"].to_numpy(dtype=np.int64)
        frame["property_id"] = bank["property_id"].to_numpy()[bank_rows]
        frame["property_seed"] = bank["property_seed"].to_numpy()[bank_rows]
        frame["case"] = int(case)
        frame["case_seed"] = int(seed)
        eligible = (
            frame["r"].between(
                args.primary_mag_min, args.primary_mag_max, inclusive="neither"
            )
            & frame["Re"].between(
                args.primary_re_min, args.primary_re_max, inclusive="neither"
            )
        )
        frame["prior_weight"] = eligible.astype(np.float64)
        frames.append(frame)
        case_reports.append(
            {
                "case": int(case),
                "seed": int(seed),
                "n_rows": int(len(frame)),
                "n_positive_prior_atoms": int(eligible.sum()),
                "n_unique_properties": int(frame["property_id"].nunique()),
            }
        )

    scene = pd.concat(frames, ignore_index=True)
    expected_rows = args.n_cases * args.rows_per_case
    if len(scene) != expected_rows:
        raise RuntimeError(f"assembled {len(scene)} rows, expected {expected_rows}")
    if scene.duplicated(["case", "index"]).any():
        raise RuntimeError("randomized scenes have duplicate (case, index) keys")
    output.parent.mkdir(parents=True, exist_ok=True)
    scene.to_parquet(output, index=False)

    revision = subprocess.run(
        ["git", "-C", str(blendsim_repository), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    commit = revision.stdout.strip() if revision.returncode == 0 else None
    catalogue_source = blendsim_repository / "blendsim" / "catalog.py"
    if not catalogue_source.is_file():
        raise FileNotFoundError(catalogue_source)
    report = {
        "format_version": 1,
        "generator": "blendsim.catalog.generate_catalog_realization",
        "blendsim_repository": str(blendsim_repository),
        "blendsim_git_commit": commit,
        "blendsim_catalogue_source": str(catalogue_source),
        "blendsim_catalogue_source_sha256": _sha256(catalogue_source),
        "shear": 0.0,
        "n_cases": args.n_cases,
        "rows_per_case": args.rows_per_case,
        "n_rows": int(len(scene)),
        "position_distribution": {"RA": [180.0, 181.0], "DEC": [0.0, 1.0]},
        "draw_with_replacement": True,
        "primary_domain": {
            "mag": [args.primary_mag_min, args.primary_mag_max],
            "Re": [args.primary_re_min, args.primary_re_max],
        },
        "n_positive_prior_atoms": int((scene["prior_weight"] > 0).sum()),
        "n_zero_weight_neighbours": int((scene["prior_weight"] == 0).sum()),
        "property_bank": str(bank_path),
        "property_bank_sha256": _sha256(bank_path),
        "property_manifest": str(bank_manifest_path),
        "property_manifest_sha256": _sha256(bank_manifest_path),
        "cases": case_reports,
        "output": str(output),
        "output_sha256": _sha256(output),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
