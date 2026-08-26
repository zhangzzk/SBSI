#!/usr/bin/env python
"""Assemble complete existing FS2 truth cases into one empirical-prior catalogue."""

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


def _sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_cases(spec: str) -> tuple[int, ...]:
    cases = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lower, upper = part.split("-", 1)
            cases.extend(range(int(lower), int(upper) + 1))
        else:
            cases.append(int(part))
    result = tuple(dict.fromkeys(cases))
    if not result:
        raise argparse.ArgumentTypeError("case specification is empty")
    return result


def assemble_fs2_scene_catalogue(
    paths,
    *,
    cases,
    expected_g1: float,
    expected_g2: float,
    primary_mag_bounds: tuple[float, float],
    primary_re_bounds: tuple[float, float],
) -> tuple[pd.DataFrame, list[dict]]:
    """Read full truth cases, verify their null shear, and assign prior mass."""

    paths = tuple(Path(path) for path in paths)
    cases = tuple(int(case) for case in cases)
    if len(paths) != len(cases) or not paths:
        raise ValueError("paths and cases must be non-empty and aligned")
    mag_lo, mag_hi = map(float, primary_mag_bounds)
    re_lo, re_hi = map(float, primary_re_bounds)
    if not (mag_lo < mag_hi and re_lo < re_hi):
        raise ValueError("primary-domain bounds must be ordered")

    frames = []
    reports = []
    required = {
        "index",
        "g1",
        "g2",
        "shear_component_convention",
        *TRUTH_COLUMNS,
    }
    for case, path in zip(cases, paths):
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_feather(path)
        missing = sorted(required - set(frame))
        if missing:
            raise KeyError(f"{path} lacks FS2 truth columns: {missing}")
        if frame["index"].isna().any() or frame["index"].duplicated().any():
            raise ValueError(f"{path} does not have a unique within-case index")
        values = frame.loc[:, TRUTH_COLUMNS].to_numpy(float)
        if not np.isfinite(values).all():
            raise ValueError(f"{path} contains non-finite truth properties")
        shear = frame[["g1", "g2"]].to_numpy(float)
        expected = np.asarray([expected_g1, expected_g2], dtype=float)
        if not np.isclose(shear, expected[None, :], rtol=0, atol=1e-14).all():
            examples = shear[
                ~np.isclose(shear, expected[None, :], rtol=0, atol=1e-14).all(axis=1)
            ][:5]
            raise ValueError(
                f"{path} does not have the declared shear {expected.tolist()}; "
                f"examples: {examples.tolist()}"
            )
        conventions = tuple(frame["shear_component_convention"].dropna().unique())
        if conventions != ("sky_cos_sin",):
            raise ValueError(f"{path} has unsupported shear conventions {conventions}")

        frame = frame.copy()
        frame["case"] = int(case)
        eligible = (
            frame["r"].between(mag_lo, mag_hi, inclusive="neither")
            & frame["Re"].between(re_lo, re_hi, inclusive="neither")
        )
        frame["prior_weight"] = eligible.astype(np.float64)
        frames.append(frame)
        reports.append(
            {
                "case": int(case),
                "source": str(path.resolve()),
                "source_sha256": _sha256(path),
                "n_rows": int(len(frame)),
                "n_positive_prior_atoms": int(eligible.sum()),
            }
        )

    scene = pd.concat(frames, ignore_index=True)
    if scene.duplicated(["case", "index"]).any():
        raise RuntimeError("assembled scene has duplicate (case, index) keys")
    return scene, reports


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases-directory", required=True)
    parser.add_argument("--cases", required=True, type=parse_cases)
    parser.add_argument("--shear-label", default="0.0")
    parser.add_argument("--expected-g1", type=float, default=0.0)
    parser.add_argument("--expected-g2", type=float, default=0.0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--primary-mag-min", type=float, required=True)
    parser.add_argument("--primary-mag-max", type=float, required=True)
    parser.add_argument("--primary-re-min", type=float, required=True)
    parser.add_argument("--primary-re-max", type=float, required=True)
    parser.add_argument("--expected-rows", type=int, default=None)
    parser.add_argument("--expected-positive-atoms", type=int, default=None)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    output = Path(args.output)
    report_path = Path(args.report)
    if output.exists() or report_path.exists():
        raise SystemExit("refusing to overwrite scene catalogue or report")
    root = Path(args.cases_directory)
    paths = [root / f"gals{case}_{args.shear_label}.feather" for case in args.cases]
    scene, case_reports = assemble_fs2_scene_catalogue(
        paths,
        cases=args.cases,
        expected_g1=args.expected_g1,
        expected_g2=args.expected_g2,
        primary_mag_bounds=(args.primary_mag_min, args.primary_mag_max),
        primary_re_bounds=(args.primary_re_min, args.primary_re_max),
    )
    n_positive = int((scene["prior_weight"] > 0).sum())
    if args.expected_rows is not None and len(scene) != args.expected_rows:
        raise RuntimeError(
            f"assembled {len(scene)} rows, expected {args.expected_rows}"
        )
    if (
        args.expected_positive_atoms is not None
        and n_positive != args.expected_positive_atoms
    ):
        raise RuntimeError(
            f"assembled {n_positive} positive atoms, expected "
            f"{args.expected_positive_atoms}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    scene.to_parquet(output, index=False)
    report = {
        "source": "existing FS2 truth catalogues; no image simulation performed",
        "cases_directory": str(root.resolve()),
        "cases": list(args.cases),
        "shear_label": args.shear_label,
        "expected_shear": [args.expected_g1, args.expected_g2],
        "n_rows": int(len(scene)),
        "n_positive_prior_atoms": n_positive,
        "n_zero_weight_neighbours": int((scene["prior_weight"] == 0).sum()),
        "primary_domain": {
            "mag": [args.primary_mag_min, args.primary_mag_max],
            "Re": [args.primary_re_min, args.primary_re_max],
        },
        "case_reports": case_reports,
        "output": str(output.resolve()),
        "output_sha256": _sha256(output),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
