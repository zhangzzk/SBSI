"""Merge disjoint constgold-bin decomposition truth chunks safely."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


def sidecar(path: str) -> str:
    return os.path.splitext(path)[0] + ".json"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", action="append", required=True)
    ap.add_argument("--cases", type=int, nargs="+", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    if os.path.exists(args.output) or os.path.exists(sidecar(args.output)):
        raise FileExistsError(f"refusing existing output/sidecar for {args.output}")

    parts: list[pd.DataFrame] = []
    metadata = []
    reference_columns = None
    for path in args.input:
        frame = pd.read_feather(path)
        if reference_columns is None:
            reference_columns = frame.columns.tolist()
        elif frame.columns.tolist() != reference_columns:
            raise RuntimeError(f"schema/order differs in {path}")
        with open(sidecar(path), encoding="utf-8") as handle:
            meta = json.load(handle)
        file_cases = sorted(frame["case"].unique().astype(int).tolist())
        if file_cases != sorted(int(case) for case in meta["cases"]):
            raise RuntimeError(f"frame/sidecar cases differ for {path}")
        if len(frame) != int(meta["n_rows"]):
            raise RuntimeError(f"frame/sidecar row count differs for {path}")
        parts.append(frame)
        metadata.append(meta)

    out = pd.concat(parts, ignore_index=True)
    keys = ["case", "input_index"]
    if out.duplicated(keys).any():
        dup = out.loc[out.duplicated(keys, keep=False), keys].head().to_dict("records")
        raise RuntimeError(f"duplicate merged keys: {dup}")
    got_cases = sorted(out["case"].unique().astype(int).tolist())
    expected_cases = sorted(args.cases)
    if got_cases != expected_cases:
        raise RuntimeError(f"merged cases {got_cases} != expected {expected_cases}")
    numeric = out.select_dtypes(include=[np.number])
    if not np.isfinite(numeric.to_numpy(float)).all():
        raise RuntimeError("merged truth contains non-finite numeric values")
    out = out.sort_values(keys, kind="mergesort").reset_index(drop=True)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_feather(args.output)
    with open(sidecar(args.output), "x", encoding="utf-8") as handle:
        json.dump({
            "cases": expected_cases, "n_cases": len(expected_cases),
            "n_rows": len(out), "inputs": args.input,
            "rows_per_case": {
                str(int(case)): int(count)
                for case, count in out.groupby("case").size().items()
            },
            "component_identity": metadata[0].get("component_identity"),
            "common_detection_policy": metadata[0].get("common_detection_policy"),
            "population_policy": metadata[0].get("population_policy"),
            "source_sidecars": metadata,
        }, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(
        f"merged {len(args.input)} truth parts -> {args.output}: "
        f"{len(out):,} anchors, {len(expected_cases)} cases",
        flush=True,
    )
    print("V22_CONSTGOLD_Q3_DECOMP_MERGE_DONE", flush=True)


if __name__ == "__main__":
    main()
