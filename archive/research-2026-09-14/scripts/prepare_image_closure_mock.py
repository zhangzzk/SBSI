#!/usr/bin/env python
"""Prepare a provenance-checked image-generated mock for catalogue inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sbsi.image_closure import (
    blendemu_case_paths,
    build_image_mock,
    file_sha256,
)
from sbsi.measurement_model import load_measurement_model


def parse_cases(spec: str) -> tuple[int, ...]:
    cases: list[int] = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            cases.extend(range(int(lo), int(hi) + 1))
        else:
            cases.append(int(part))
    result = tuple(dict.fromkeys(cases))
    if not result:
        raise argparse.ArgumentTypeError("case specification is empty")
    return result


def _tree_sha256(root: Path) -> dict[str, str]:
    return {
        name: file_sha256(root / name)
        for name in ("measurements.parquet", "truth.parquet")
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulation-root", required=True)
    parser.add_argument("--cases", required=True, type=parse_cases)
    parser.add_argument("--shear-label", required=True)
    parser.add_argument("--injected-g1", required=True, type=float)
    parser.add_argument("--injected-g2", required=True, type=float)
    parser.add_argument("--measurement-model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-objects", type=int, default=None)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--primary-mag-min", type=float, required=True)
    parser.add_argument("--primary-mag-max", type=float, required=True)
    parser.add_argument("--primary-re-min", type=float, required=True)
    parser.add_argument("--primary-re-max", type=float, required=True)
    parser.add_argument("--real", default="real0")
    parser.add_argument("--tile-name", default="tile180.0_-0.5")
    parser.add_argument(
        "--shape-name",
        default=None,
        help=(
            "filename below catalogues/Shapes; {tile_name} is expanded. "
            "If omitted, exactly one recognized primary/all product must exist"
        ),
    )
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    output = Path(args.output)
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}")
    model = load_measurement_model(args.measurement_model, device=args.device)
    target_names = tuple(model.target_transform.target_names)
    paths = [
        blendemu_case_paths(
            args.simulation_root,
            case=case,
            shear_label=args.shear_label,
            real=args.real,
            tile_name=args.tile_name,
            shape_name=args.shape_name,
        )
        for case in args.cases
    ]
    mock, report = build_image_mock(
        paths,
        target_names=target_names,
        injected_g1=args.injected_g1,
        injected_g2=args.injected_g2,
        primary_mag_bounds=(args.primary_mag_min, args.primary_mag_max),
        primary_re_bounds=(args.primary_re_min, args.primary_re_max),
        n_objects=args.n_objects,
        sample_seed=args.sample_seed,
    )
    mock.save(output)
    for case_paths, case_report in zip(paths, report["cases"]):
        hashes = {
            "input": file_sha256(case_paths.input_catalogue),
            "match": file_sha256(case_paths.match_catalogue),
            "shape": file_sha256(case_paths.shape_catalogue),
        }
        case_report["sha256"] = hashes
    model_digest = file_sha256(args.measurement_model)
    manifest = {
        **report,
        "simulation_root": str(Path(args.simulation_root).resolve()),
        "measurement_model": str(Path(args.measurement_model).resolve()),
        "measurement_model_sha256": model_digest,
        "output_sha256": _tree_sha256(output),
        "config": {
            "cases": list(args.cases),
            "shear_label": args.shear_label,
            "real": args.real,
            "tile_name": args.tile_name,
            "shape_name": args.shape_name,
            "n_objects": args.n_objects,
        },
    }
    (output / "image_mock_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))
    print(f"image closure mock -> {output}")


if __name__ == "__main__":
    main()
