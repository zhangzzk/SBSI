"""Extract one strict per-object feature from a pair-annotated IPC catalogue."""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
import pyarrow.feather as pf


KEY_MULTIPLIER = 1_000_003


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--catalogue", nargs="+", required=True)
    ap.add_argument("--input-column", required=True)
    ap.add_argument("--output-column", required=True)
    ap.add_argument("--min-case", type=int, default=None)
    ap.add_argument("--max-case", type=int, default=None)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    if os.path.exists(args.output):
        raise FileExistsError(f"refusing to overwrite {args.output}")

    parts = [pf.read_table(
        catalogue, columns=["case", "input_index", args.input_column],
    ).to_pandas() for catalogue in args.catalogue]
    frame = pd.concat(parts, ignore_index=True)
    if args.min_case is not None:
        frame = frame.loc[frame["case"].to_numpy(np.int64) >= args.min_case]
    if args.max_case is not None:
        frame = frame.loc[frame["case"].to_numpy(np.int64) <= args.max_case]
    frame = frame.reset_index(drop=True)
    values = frame[args.input_column].to_numpy(float)
    if not np.isfinite(values).all():
        raise RuntimeError(f"non-finite values in {args.input_column}")

    input_index = frame["input_index"].to_numpy(np.int64)
    if len(input_index) and (input_index.min() < 0 or input_index.max() >= KEY_MULTIPLIER):
        raise RuntimeError(
            f"input_index range [{input_index.min()},{input_index.max()}] is incompatible "
            f"with key multiplier {KEY_MULTIPLIER}"
        )
    key = frame["case"].to_numpy(np.int64) * KEY_MULTIPLIER + input_index
    order = np.argsort(key, kind="stable")
    key = key[order]
    values = values[order]
    repeated = key[1:] == key[:-1]
    if repeated.any() and not np.array_equal(values[1:][repeated], values[:-1][repeated]):
        difference = np.abs(values[1:][repeated] - values[:-1][repeated])
        raise RuntimeError(
            f"{int(repeated.sum()):,} duplicate rows are not value-consistent; "
            f"max difference={difference.max():.6g}"
        )
    keep = np.r_[True, ~repeated]
    source_index = order[keep]
    output = pd.DataFrame({
        "case": frame["case"].to_numpy(np.int64)[source_index],
        "input_index": frame["input_index"].to_numpy(np.int64)[source_index],
        args.output_column: values[keep],
    })
    if output.duplicated(["case", "input_index"]).any():
        raise RuntimeError("compacted keys are not unique")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    output.to_feather(args.output)
    metadata = {
        "catalogues": [os.path.abspath(path) for path in args.catalogue],
        "input_column": args.input_column,
        "output_column": args.output_column,
        "input_rows_after_case_cut": int(len(frame)),
        "output_rows": int(len(output)),
        "duplicate_rows_removed": int(len(frame) - len(output)),
        "min_case": args.min_case,
        "max_case": args.max_case,
    }
    with open(os.path.splitext(args.output)[0] + ".json", "x", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(
        f"wrote {args.output}: {len(frame):,} rows -> {len(output):,} unique keys; "
        f"column {args.input_column}->{args.output_column}",
        flush=True,
    )


if __name__ == "__main__":
    main()
