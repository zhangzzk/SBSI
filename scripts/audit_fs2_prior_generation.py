#!/usr/bin/env python
"""Validate fresh FS2 prior cases and freeze their generation provenance."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd


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


AUDIT_VERSION = 1
AUDIT_COLUMNS = ("index", "g1", "g2", "shear_component_convention")


def _sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_cases(
    cases_directory: str | Path,
    *,
    cases,
    expected_rows_per_case: int,
    seed_offset: int,
    forbidden_cases=(),
) -> list[dict]:
    """Return file-level provenance after strict null-shear validation."""

    root = Path(cases_directory)
    cases = tuple(int(case) for case in cases)
    forbidden = set(int(case) for case in forbidden_cases)
    overlap = sorted(set(cases) & forbidden)
    if overlap:
        raise ValueError(f"prior cases overlap forbidden simulation cases: {overlap[:10]}")
    if len(set(cases)) != len(cases):
        raise ValueError("prior cases must be unique")
    seeds = [case + int(seed_offset) for case in cases]
    if len(set(seeds)) != len(seeds):
        raise ValueError("derived generator seeds must be unique")

    expected_names = {f"gals{case}_0.0.feather" for case in cases}
    actual_names = {path.name for path in root.glob("gals*_*.feather")}
    missing = sorted(expected_names - actual_names)
    unexpected = sorted(actual_names - expected_names)
    if missing or unexpected:
        raise ValueError(
            f"raw prior case set mismatch; missing={missing[:5]}, "
            f"unexpected={unexpected[:5]}"
        )

    reports = []
    for case, seed in zip(cases, seeds):
        path = root / f"gals{case}_0.0.feather"
        frame = pd.read_feather(path, columns=list(AUDIT_COLUMNS))
        if len(frame) != int(expected_rows_per_case):
            raise ValueError(
                f"{path} has {len(frame)} rows, expected {expected_rows_per_case}"
            )
        index = frame["index"].to_numpy(dtype=np.int64)
        if not np.array_equal(index, np.arange(len(frame), dtype=np.int64)):
            raise ValueError(f"{path} index is not the complete ordered row range")
        shear = frame[["g1", "g2"]].to_numpy(dtype=np.float64)
        if not np.allclose(shear, 0.0, rtol=0, atol=0):
            raise ValueError(f"{path} is not an exactly unsheared prior case")
        conventions = tuple(frame["shear_component_convention"].dropna().unique())
        if conventions != ("sky_cos_sin",):
            raise ValueError(f"{path} has unsupported shear conventions {conventions}")
        reports.append(
            {
                "case": case,
                "generator_seed": seed,
                "path": str(path.resolve()),
                "sha256": _sha256(path),
                "n_rows": int(len(frame)),
            }
        )
    return reports


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases-directory", required=True)
    parser.add_argument("--cases", required=True, type=parse_cases)
    parser.add_argument("--forbidden-cases", required=True, type=parse_cases)
    parser.add_argument("--expected-rows-per-case", required=True, type=int)
    parser.add_argument("--seed-offset", type=int, default=123)
    parser.add_argument("--base-catalogue", required=True)
    parser.add_argument("--pipeline-config", required=True)
    parser.add_argument("--generator", required=True)
    parser.add_argument("--generator-catalog-module", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    output = Path(args.output)
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}")
    reports = audit_cases(
        args.cases_directory,
        cases=args.cases,
        expected_rows_per_case=args.expected_rows_per_case,
        seed_offset=args.seed_offset,
        forbidden_cases=args.forbidden_cases,
    )
    inputs = {
        "base_catalogue": args.base_catalogue,
        "pipeline_config": args.pipeline_config,
        "generator": args.generator,
        "generator_catalog_module": args.generator_catalog_module,
    }
    payload = {
        "version": AUDIT_VERSION,
        "generator": "blendemu.scripts.run_pipeline step 1",
        "seed_rule": f"seed = case + {args.seed_offset}",
        "cases": list(args.cases),
        "generator_seeds": [report["generator_seed"] for report in reports],
        "forbidden_cases": list(args.forbidden_cases),
        "n_cases": len(reports),
        "n_rows": int(sum(report["n_rows"] for report in reports)),
        "expected_shear": [0.0, 0.0],
        "source_identity": {
            name: {"path": str(Path(path).resolve()), "sha256": _sha256(path)}
            for name, path in inputs.items()
        },
        "case_reports": reports,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
