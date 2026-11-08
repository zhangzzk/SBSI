#!/usr/bin/env python
"""Audit FS2 scene shards and write the top-level default-prior manifest."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

MANIFEST_VERSION = 1


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


def _sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finalize(root: str | Path, *, cases, cases_per_shard: int) -> dict:
    root = Path(root)
    cases = tuple(int(case) for case in cases)
    if cases_per_shard <= 0 or len(cases) % cases_per_shard:
        raise ValueError("cases must divide exactly into positive-size shards")
    generation_path = root / "generation_manifest.json"
    generation = json.loads(generation_path.read_text())
    if generation.get("cases") != list(cases):
        raise ValueError("generation manifest cases do not match requested prior cases")

    shards = []
    all_reported_cases = []
    for start in range(0, len(cases), cases_per_shard):
        selected = cases[start : start + cases_per_shard]
        shard_root = root / "shards" / f"cases{selected[0]}_{selected[-1]}"
        report_path = shard_root / "scene_catalogue_report.json"
        store_root = shard_root / "scene_store"
        store_manifest_path = store_root / "manifest.json"
        report = json.loads(report_path.read_text())
        store_manifest = json.loads(store_manifest_path.read_text())
        reported_cases = report.get("cases")
        if reported_cases != list(selected):
            raise ValueError(f"{shard_root} reports cases {reported_cases}, expected {selected}")
        store_report = store_manifest.get("report") or {}
        if store_report.get("n_galaxies") != report.get("n_rows"):
            raise ValueError(f"{shard_root} scene/store row counts disagree")
        all_reported_cases.extend(reported_cases)
        shards.append(
            {
                "cases": list(selected),
                "root": str(shard_root.resolve()),
                "n_rows": int(report["n_rows"]),
                "n_positive_prior_atoms": int(report["n_positive_prior_atoms"]),
                "n_directed_edges": int(store_report["n_directed_edges"]),
                "sha256": {
                    "scene_catalogue_report.json": _sha256(report_path),
                    "scene_store/manifest.json": _sha256(store_manifest_path),
                    "scene_store/galaxies.parquet": _sha256(
                        store_root / "galaxies.parquet"
                    ),
                    "scene_store/neighbours.npz": _sha256(
                        store_root / "neighbours.npz"
                    ),
                },
            }
        )
    if all_reported_cases != list(cases):
        raise ValueError("scene shards are not an exact ordered partition of cases")
    n_active = sum(shard["n_positive_prior_atoms"] for shard in shards)
    return {
        "version": MANIFEST_VERSION,
        "kind": "sharded_scene_prior",
        "status": "default_fs2_prior_catalogue",
        "cases": list(cases),
        "n_cases": len(cases),
        "n_rows": sum(shard["n_rows"] for shard in shards),
        "n_positive_prior_atoms": n_active,
        "n_directed_edges": sum(shard["n_directed_edges"] for shard in shards),
        "global_shard_mass": [
            shard["n_positive_prior_atoms"] / n_active for shard in shards
        ],
        "generation_manifest": {
            "path": str(generation_path.resolve()),
            "sha256": _sha256(generation_path),
        },
        "shards": shards,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--cases", required=True, type=parse_cases)
    parser.add_argument("--cases-per-shard", type=int, default=10)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    output = Path(args.output)
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}")
    payload = finalize(
        args.root, cases=args.cases, cases_per_shard=args.cases_per_shard
    )
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
