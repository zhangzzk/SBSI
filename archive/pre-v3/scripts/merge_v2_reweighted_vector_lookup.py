#!/usr/bin/env python3
"""Merge guarded 10-case V2 ConstGold lookup shards."""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd
import pyarrow.feather as pf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-glob", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-case", type=int, default=40)
    parser.add_argument("--max-case", type=int, default=139)
    parser.add_argument("--expected-shards", type=int, default=10)
    args = parser.parse_args()
    if os.path.exists(args.output):
        raise FileExistsError(f"refusing to overwrite {args.output}")
    paths = sorted(glob.glob(args.input_glob))
    if len(paths) != args.expected_shards:
        raise RuntimeError(f"expected {args.expected_shards} shards, found {len(paths)}")
    parts = [pf.read_table(path).to_pandas() for path in paths]
    output = pd.concat(parts, ignore_index=True)
    if list(output.columns) != ["case", "input_index", "R_blend"]:
        raise RuntimeError(f"unexpected columns: {list(output.columns)}")
    expected_cases = np.arange(args.min_case, args.max_case + 1)
    cases = np.sort(output["case"].unique())
    if not np.array_equal(cases, expected_cases):
        raise RuntimeError(f"case coverage differs: {cases.tolist()}")
    if output.duplicated(["case", "input_index"]).any():
        raise RuntimeError("duplicate (case,input_index) keys across lookup shards")
    if not np.isfinite(output["R_blend"].to_numpy(float)).all():
        raise RuntimeError("non-finite R_blend in lookup shards")
    output = output.sort_values(["case", "input_index"], kind="stable").reset_index(drop=True)
    output.to_feather(args.output)
    print(
        f"wrote {args.output}: {len(output):,} rows over {len(cases)} cases from {len(paths)} shards",
        flush=True,
    )
    print("V2_REWEIGHTED_VECTOR_LOOKUP_MERGE_DONE", flush=True)


if __name__ == "__main__":
    main()
