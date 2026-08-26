#!/usr/bin/env python
# Archived GalSBI-prior experiment.
"""Assemble randomized-position blendsim cases into a finite scene prior."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd


TRUTH_COLUMNS = (
    "RA",
    "DEC",
    "redshift",
    "r",
    "Re",
    "sersic_n",
    "axis_ratio",
    "position_angle",
)


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases-directory", required=True)
    parser.add_argument("--case-start", type=int, required=True)
    parser.add_argument("--n-cases", type=int, required=True)
    parser.add_argument("--shear-label", required=True)
    parser.add_argument("--base-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--primary-mag-min", type=float, required=True)
    parser.add_argument("--primary-mag-max", type=float, required=True)
    parser.add_argument("--primary-re-min", type=float, required=True)
    parser.add_argument("--primary-re-max", type=float, required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.n_cases <= 0:
        raise SystemExit("--n-cases must be positive")
    if not (
        args.primary_mag_min < args.primary_mag_max
        and args.primary_re_min < args.primary_re_max
    ):
        raise SystemExit("primary-domain bounds must be strictly increasing")
    output = Path(args.output)
    report_path = Path(args.report)
    if output.exists() or report_path.exists():
        raise SystemExit("refusing to overwrite scene catalogue or report")

    case_root = Path(args.cases_directory)
    case_paths = [
        case_root / f"gals{case}_{args.shear_label}.feather"
        for case in range(args.case_start, args.case_start + args.n_cases)
    ]
    missing = [str(path) for path in case_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing randomized-position cases: {missing}")

    frames = []
    case_reports = []
    for case, path in zip(
        range(args.case_start, args.case_start + args.n_cases), case_paths
    ):
        frame = pd.read_feather(path)
        missing_columns = sorted(set(TRUTH_COLUMNS) - set(frame))
        if missing_columns:
            raise KeyError(f"{path} lacks truth columns: {missing_columns}")
        if "index" not in frame or frame["index"].duplicated().any():
            raise ValueError(f"{path} lacks a unique within-case index")
        finite = np.isfinite(frame.loc[:, TRUTH_COLUMNS].to_numpy(float)).all(axis=1)
        if not finite.all():
            raise ValueError(f"{path} contains {int((~finite).sum())} non-finite rows")
        frame = frame.copy()
        frame["case"] = int(case)
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
                "seed": int(case + 123),
                "n_rows": int(len(frame)),
                "n_positive_prior_atoms": int(eligible.sum()),
                "source": str(path),
                "source_sha256": _sha256(path),
            }
        )

    scene = pd.concat(frames, ignore_index=True)
    if scene.duplicated(["case", "index"]).any():
        raise RuntimeError("assembled scene has duplicate (case, index) keys")
    output.parent.mkdir(parents=True, exist_ok=True)
    scene.to_parquet(output, index=False)
    base_manifest = Path(args.base_manifest)
    report = {
        "n_cases": args.n_cases,
        "n_rows": int(len(scene)),
        "n_positive_prior_atoms": int((scene["prior_weight"] > 0).sum()),
        "n_zero_weight_neighbours": int((scene["prior_weight"] == 0).sum()),
        "primary_domain": {
            "mag": [args.primary_mag_min, args.primary_mag_max],
            "Re": [args.primary_re_min, args.primary_re_max],
        },
        "base_manifest": str(base_manifest),
        "base_manifest_sha256": _sha256(base_manifest),
        "cases": case_reports,
        "output": str(output),
        "output_sha256": _sha256(output),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
