#!/usr/bin/env python3
"""Merge disjoint fixed-g0 ConstGold R_blend lookup blocks."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8_388_608), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", type=int, action="append", required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    expected = set(args.case)
    if len(expected) != len(args.case):
        raise ValueError("duplicate expected cases")
    frames = []
    inputs = []
    for path in args.input:
        frame = pd.read_feather(path, columns=["case", "input_index", "R_blend"])
        frames.append(frame)
        inputs.append({"path": str(path.resolve()), "sha256": file_sha256(path)})
    merged = pd.concat(frames, ignore_index=True)
    if merged.duplicated(["case", "input_index"]).any():
        raise ValueError("lookup blocks overlap or contain duplicate keys")
    actual = set(merged["case"].unique())
    if actual != expected:
        raise ValueError(f"case coverage mismatch: expected={expected}, actual={actual}")
    if not np.isfinite(merged["R_blend"].to_numpy(float)).all():
        raise ValueError("merged lookup contains non-finite responses")
    merged = merged.sort_values(["case", "input_index"], kind="stable").reset_index(
        drop=True
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_feather(args.output)
    args.output.with_suffix(".json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "rows": int(len(merged)),
                "cases": sorted(int(case) for case in actual),
                "inputs": inputs,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        f"CONSTGOLD_FIXED_G0_RBLEND_MERGED output={args.output} "
        f"rows={len(merged):,} cases={len(actual)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
