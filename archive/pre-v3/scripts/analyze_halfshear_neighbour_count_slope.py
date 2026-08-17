"""Stratify aggregate-vector closure by supported neighbours per primary."""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd


def stat(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(values)),
        "case_sem": float(np.std(values, ddof=1) / np.sqrt(len(values))),
        "n_cases": int(len(values)),
    }


def summarize(frame: pd.DataFrame, edges: list[int]) -> dict:
    total_power = float(frame.power.sum())
    total_dot = float(frame["dot"].sum())
    output = {}
    for low, high in zip(edges[:-1], edges[1:]):
        selected = frame.loc[(frame.n_pairs >= low) & (frame.n_pairs < high)]
        case = selected.groupby("case", sort=True)[["dot", "power"]].sum()
        case = case.loc[case.power > 0]
        slopes = case["dot"].to_numpy(float) / case.power.to_numpy(float)
        output[f"[{low},{high})"] = {
            "n_primaries": int(len(selected)),
            "pooled_slope": float(selected["dot"].sum() / selected.power.sum()),
            "case_slope": stat(slopes),
            "fraction_total_model_power": float(selected.power.sum() / total_power),
            "fraction_total_dot": float(selected["dot"].sum() / total_dot),
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", required=True)
    parser.add_argument("--split-case", type=int, default=120)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if os.path.exists(args.output):
        raise FileExistsError(f"refusing to overwrite {args.output}")
    frame = pd.read_feather(args.table, columns=["case", "n_pairs", "dot", "power"])
    edges = [1, 2, 3, 4, 6, 11, 20, 21]
    payload = {
        "design": "case-level aggregate-vector slope stratified only by exact supported pair count",
        "table": args.table,
        "edges_left_closed_right_open": edges,
        "all": summarize(frame, edges),
        f"first_cases_below_{args.split_case}": summarize(
            frame.loc[frame.case < args.split_case], edges,
        ),
        f"second_cases_at_least_{args.split_case}": summarize(
            frame.loc[frame.case >= args.split_case], edges,
        ),
        "constgold_opened": False,
        "anchor_truth_opened": False,
    }
    with open(args.output, "x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
