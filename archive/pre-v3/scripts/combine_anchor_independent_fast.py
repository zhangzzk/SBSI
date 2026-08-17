"""Combine fast per-case independent-neighbour anchor measurements."""
from __future__ import annotations

import argparse
import os

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--case-min", type=int, default=200)
    parser.add_argument("--case-max", type=int, default=299)
    args = parser.parse_args()
    if os.path.exists(args.output):
        raise FileExistsError(f"refusing to overwrite {args.output}")
    paths = [os.path.join(args.input_dir, f"case{case}.feather")
             for case in range(args.case_min, args.case_max + 1)]
    missing = [path for path in paths if not os.path.isfile(path) or os.path.getsize(path) == 0]
    if missing:
        raise RuntimeError(f"missing {len(missing)} case parts; first={missing[:3]}")
    parts = []
    for path in paths:
        part = pd.read_feather(path)
        if "input_index" not in part and "index" in part:
            part = part.rename(columns={"index": "input_index"})
        parts.append(part)
    frame = pd.concat(parts, ignore_index=True)
    if frame.duplicated(["case", "input_index"]).any():
        raise RuntimeError("duplicate combined anchor keys")
    frame.to_feather(args.output)
    print(f"wrote {args.output}: rows={len(frame):,}")
    print("ANCHOR_INDEPENDENT_FAST_COMBINE_DONE", flush=True)


if __name__ == "__main__":
    main()
