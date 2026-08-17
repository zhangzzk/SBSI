"""Concatenate disjoint per-object BlendEMU lookups with strict key guards."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


KEYS = ["case", "input_index"]
COLUMNS = KEYS + ["R_blend"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--cases", type=int, nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    expected_cases = sorted(set(args.cases))
    if expected_cases != args.cases:
        raise ValueError("--cases must be unique and sorted")

    parts = []
    for input_name in args.inputs:
        frame = pd.read_feather(input_name, columns=COLUMNS)
        if frame[KEYS].isna().any().any():
            raise ValueError(f"null lookup key in {input_name}")
        if not np.isfinite(frame["R_blend"].to_numpy(float)).all():
            raise ValueError(f"non-finite R_blend in {input_name}")
        print(
            f"{input_name}: {len(frame):,} rows, cases "
            f"{int(frame['case'].min())}--{int(frame['case'].max())}",
            flush=True,
        )
        parts.append(frame)

    combined = pd.concat(parts, ignore_index=True)
    duplicated = combined.duplicated(KEYS, keep=False)
    if duplicated.any():
        raise ValueError(f"combined lookup has {int(duplicated.sum())} duplicate-key rows")
    actual_cases = sorted(combined["case"].unique().astype(int).tolist())
    if actual_cases != expected_cases:
        missing = sorted(set(expected_cases).difference(actual_cases))
        extra = sorted(set(actual_cases).difference(expected_cases))
        raise ValueError(f"case mismatch: missing={missing}, extra={extra}")

    combined = combined.sort_values(KEYS, kind="stable").reset_index(drop=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_feather(output)
    print(
        f"wrote {output}: {len(combined):,} unique rows over {len(actual_cases)} cases",
        flush=True,
    )


if __name__ == "__main__":
    main()
