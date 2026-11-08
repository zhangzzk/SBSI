#!/usr/bin/env python
"""Audit sharded ConstGold R_blend lookups and optionally compare a reference."""

from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.feather as pf


SHARD_RE = re.compile(r"_c(?P<first>\d+)-(?P<last>\d+)\.feather$")
KEYS = ["case", "input_index"]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lookup-glob", required=True)
    parser.add_argument("--reference-glob")
    parser.add_argument("--expected-first", type=int, required=True)
    parser.add_argument("--expected-last", type=int, required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def shard_key(path: str | Path) -> tuple[int, int]:
    match = SHARD_RE.search(str(path))
    if match is None:
        raise ValueError(f"cannot parse case range from {path}")
    return int(match["first"]), int(match["last"])


def read_shard(path: str | Path) -> pd.DataFrame:
    frame = pf.read_table(path, columns=KEYS + ["R_blend"]).to_pandas()
    if frame.duplicated(KEYS).any():
        raise RuntimeError(f"duplicate lookup keys in {path}")
    if not np.isfinite(frame["R_blend"]).all():
        raise RuntimeError(f"non-finite R_blend in {path}")
    return frame


def distribution(values: np.ndarray) -> dict[str, float | int]:
    return {
        "rows": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "p01": float(np.quantile(values, 0.01)),
        "p50": float(np.quantile(values, 0.50)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(values.max()),
    }


def main(argv=None):
    args = parse_args(argv)
    output = Path(args.output)
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}")

    paths = sorted(glob.glob(args.lookup_glob), key=shard_key)
    if not paths:
        raise FileNotFoundError(args.lookup_glob)
    references = {}
    if args.reference_glob:
        references = {shard_key(path): path for path in glob.glob(args.reference_glob)}
        if set(references) != {shard_key(path) for path in paths}:
            raise RuntimeError("lookup and reference shard ranges differ")

    values = []
    seen_cases: set[int] = set()
    shared_new = []
    shared_reference = []
    lookup_only = 0
    reference_only = 0
    reference_rows = 0
    for path in paths:
        key = shard_key(path)
        frame = read_shard(path)
        values.append(frame["R_blend"].to_numpy(dtype=np.float64))
        seen_cases.update(frame["case"].unique().tolist())
        if references:
            reference = read_shard(references[key])
            reference_rows += len(reference)
            joined = frame.merge(
                reference, on=KEYS, how="outer", suffixes=("_lookup", "_reference"),
                indicator=True, validate="one_to_one",
            )
            lookup_only += int((joined["_merge"] == "left_only").sum())
            reference_only += int((joined["_merge"] == "right_only").sum())
            both = joined["_merge"] == "both"
            shared_new.append(joined.loc[both, "R_blend_lookup"].to_numpy(np.float64))
            shared_reference.append(
                joined.loc[both, "R_blend_reference"].to_numpy(np.float64)
            )

    expected_cases = set(range(args.expected_first, args.expected_last + 1))
    if seen_cases != expected_cases:
        raise RuntimeError(
            f"case coverage mismatch: missing={sorted(expected_cases - seen_cases)}, "
            f"unexpected={sorted(seen_cases - expected_cases)}"
        )
    all_values = np.concatenate(values)
    report: dict[str, object] = {
        "lookup_glob": args.lookup_glob,
        "shards": len(paths),
        "cases": len(seen_cases),
        "case_range": [args.expected_first, args.expected_last],
        "lookup": distribution(all_values),
    }
    if references:
        current = np.concatenate(shared_new)
        prior = np.concatenate(shared_reference)
        delta = current - prior
        report["reference"] = {
            "glob": args.reference_glob,
            "rows": reference_rows,
            "shared_rows": int(current.size),
            "lookup_only_rows": lookup_only,
            "reference_only_rows": reference_only,
            "lookup_only_fraction": lookup_only / all_values.size,
            "shared_lookup_mean": float(current.mean()),
            "shared_reference_mean": float(prior.mean()),
            "delta_mean": float(delta.mean()),
            "delta_std": float(delta.std()),
            "delta_rmse": float(np.sqrt(np.mean(delta * delta))),
            "correlation": float(np.corrcoef(current, prior)[0, 1]),
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
