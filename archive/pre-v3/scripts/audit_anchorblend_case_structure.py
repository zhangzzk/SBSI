"""Audit coherent-anchor case means for block drift and serial structure."""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd


def stat(values) -> dict:
    x = np.asarray(values, float)
    return {
        "mean": float(x.mean()),
        "case_sem": float(x.std(ddof=1) / np.sqrt(len(x))),
        "case_sd": float(x.std(ddof=1)),
        "n_cases": int(len(x)),
    }


def correlation(x, y) -> float:
    return float(np.corrcoef(np.asarray(x, float), np.asarray(y, float))[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--response", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--block-size", type=int, default=25)
    args = parser.parse_args()
    if os.path.exists(args.output):
        raise FileExistsError(f"refusing to overwrite {args.output}")
    pieces = [pd.read_feather(path) for path in args.response]
    frame = pd.concat(pieces, ignore_index=True)
    key = ["case", "input_index"]
    if frame.duplicated(key).any():
        raise RuntimeError("duplicate anchor key across response shards")
    required = ["R_blend_truth", "R_blend_lsst_r_extnbr_v22"]
    if not np.isfinite(frame[required].to_numpy(float)).all():
        raise RuntimeError("non-finite response")
    case = frame.groupby("case", sort=True)[required].agg(["mean", "size"])
    case.columns = ["truth", "n_rows", "prediction", "n_rows_replay"]
    if not np.array_equal(case.n_rows, case.n_rows_replay):
        raise RuntimeError("case aggregation row counts differ")
    case = case.drop(columns="n_rows_replay")
    expected = np.arange(int(case.index.min()), int(case.index.max()) + 1)
    if not np.array_equal(case.index.to_numpy(int), expected):
        raise RuntimeError("case IDs are not consecutive")
    case["gap"] = case.prediction - case.truth
    case["block"] = (case.index.to_numpy(int) // args.block_size).astype(int)
    block = case.groupby("block", sort=True)[["truth", "prediction", "gap", "n_rows"]].mean()
    block_payload = {
        str(int(index)): {
            "cases": [
                int(index * args.block_size),
                int((index + 1) * args.block_size - 1),
            ],
            **{name: float(value) for name, value in row.items()},
        }
        for index, row in block.iterrows()
    }
    index = case.index.to_numpy(float)
    design = np.column_stack([np.ones(len(index)), index - index.mean()])
    slope = np.linalg.lstsq(design, case.gap.to_numpy(float), rcond=None)[0][1]
    residual = case.gap.to_numpy(float) - design @ np.linalg.lstsq(
        design, case.gap.to_numpy(float), rcond=None,
    )[0]
    autocorrelation = {}
    for lag in range(1, min(26, len(residual))):
        autocorrelation[str(lag)] = correlation(residual[:-lag], residual[lag:])
    payload = {
        "design": "case-balanced audit of consecutive coherent anchor simulations; descriptive only",
        "cases": [int(case.index.min()), int(case.index.max())],
        "block_size": int(args.block_size),
        "overall": {name: stat(case[name]) for name in ("truth", "prediction", "gap", "n_rows")},
        "block_means": block_payload,
        "linear_gap_slope_per_case": float(slope),
        "gap_correlations": {
            "truth": correlation(case.gap, case.truth),
            "prediction": correlation(case.gap, case.prediction),
            "n_rows": correlation(case.gap, case.n_rows),
        },
        "detrended_gap_autocorrelation": autocorrelation,
        "block_gap_sd": float(block.gap.std(ddof=1)),
        "expected_block_mean_sd_from_case_sd": float(
            case.gap.std(ddof=1) / np.sqrt(args.block_size)
        ),
        "constgold_opened": False,
    }
    with open(args.output, "x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    print("ANCHORBLEND_CASE_STRUCTURE_AUDIT_DONE", flush=True)


if __name__ == "__main__":
    main()
