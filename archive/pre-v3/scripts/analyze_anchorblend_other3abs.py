"""Evaluate the frozen pair-other-flux emulator on anchor responses.

Coherent and three disjoint local-random arms use cases 200--249 as a labeled
development summary and cases 250--299 as the predeclared validation summary.
The candidate was fixed before either arm was scored.  This is mechanism
evidence rather than a new blinded certification because baseline anchor truth
has been examined previously.  No correction is fitted and constgold is not
read.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd


KEY = ["case", "input_index"]
BASE = "R_blend_lsst_r_extnbr_v22"
NEW = "R_blend_lsst_r_extnbr_v22_other3abs"


def sem(values):
    values = np.asarray(values, float)
    return float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else None


def stat(values):
    values = np.asarray(values, float)
    if not len(values):
        return {"mean": None, "case_sem": None, "n_cases": 0}
    return {"mean": float(values.mean()), "case_sem": sem(values), "n_cases": int(len(values))}


def max_abs_conditional(rows, label):
    values = [abs(row[label]["mean"]) for row in rows if row[label]["mean"] is not None]
    if not values:
        raise RuntimeError(f"no populated conditional bins for {label}")
    return float(max(values))


def read_disjoint(paths):
    frame = pd.concat([pd.read_feather(path) for path in paths], ignore_index=True)
    if frame.duplicated(KEY).any():
        raise RuntimeError("duplicate keys across disjoint anchor layers")
    required = {"R_blend_truth", BASE, NEW, *KEY}
    missing = required - set(frame)
    if missing:
        raise KeyError(f"anchor response lacks {sorted(missing)}")
    return frame


def conditional_summary(frame, edges):
    work = frame.copy()
    work["bin"] = np.clip(
        np.searchsorted(edges, work[BASE].to_numpy(float), side="right") - 1,
        0, len(edges) - 2,
    )
    rows = []
    for index in range(len(edges) - 1):
        subset = work[work.bin == index]
        by_case = subset.groupby("case", sort=True)[["gap_v22", "gap_new", "delta"]].mean()
        rows.append({
            "bin": index,
            "edges": [
                None if not np.isfinite(value) else float(value)
                for value in (edges[index], edges[index + 1])
            ],
            "n_rows": int(len(subset)),
            "v22": stat(by_case.gap_v22), "other3abs": stat(by_case.gap_new),
            "other3abs_minus_v22": stat(by_case.delta),
        })
    return rows


def summarize(frame, development_max):
    work = frame.copy()
    work["gap_v22"] = work[BASE] - work.R_blend_truth
    work["gap_new"] = work[NEW] - work.R_blend_truth
    work["delta"] = work[NEW] - work[BASE]
    development = work.case <= development_max
    edges = np.quantile(
        work.loc[development, BASE].to_numpy(float), np.linspace(0, 1, 6),
    )
    edges[0] = -np.inf; edges[-1] = np.inf
    result = {}
    for label, mask in (
        ("development", development), ("validation", ~development),
        ("all", np.ones(len(work), bool)),
    ):
        subset = work.loc[mask]
        case = subset.groupby("case", sort=True)[
            ["R_blend_truth", BASE, NEW, "gap_v22", "gap_new", "delta"]
        ].mean()
        conditional = conditional_summary(subset, edges)
        result[label] = {
            "n_rows": int(len(subset)),
            "n_cases": int(subset.case.nunique()),
            "truth": stat(case.R_blend_truth),
            "v22_prediction": stat(case[BASE]),
            "other3abs_prediction": stat(case[NEW]),
            "v22_minus_truth": stat(case.gap_v22),
            "other3abs_minus_truth": stat(case.gap_new),
            "other3abs_minus_v22": stat(case.delta),
            "conditional_v22_prediction_quintiles": conditional,
            "max_abs_conditional_v22": max_abs_conditional(conditional, "v22"),
            "max_abs_conditional_other3abs": max_abs_conditional(
                conditional, "other3abs",
            ),
        }
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--coherent", required=True)
    ap.add_argument("--random", action="append", required=True)
    ap.add_argument("--development-max", type=int, default=249)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    if os.path.exists(args.output):
        raise FileExistsError(f"refusing to overwrite {args.output}")

    coherent = summarize(read_disjoint([args.coherent]), args.development_max)
    random = summarize(read_disjoint(args.random), args.development_max)
    gates = {
        "coherent_validation_global_abs_residual_smaller": (
            abs(coherent["validation"]["other3abs_minus_truth"]["mean"])
            < abs(coherent["validation"]["v22_minus_truth"]["mean"])
        ),
        "random_validation_global_abs_residual_smaller": (
            abs(random["validation"]["other3abs_minus_truth"]["mean"])
            < abs(random["validation"]["v22_minus_truth"]["mean"])
        ),
        "coherent_validation_max_conditional_smaller": (
            coherent["validation"]["max_abs_conditional_other3abs"]
            < coherent["validation"]["max_abs_conditional_v22"]
        ),
        "random_validation_max_conditional_smaller": (
            random["validation"]["max_abs_conditional_other3abs"]
            < random["validation"]["max_abs_conditional_v22"]
        ),
    }
    result = {
        "design": (
            "frozen other3abs emulator; cases 200--249 development and 250--299 "
            "validation; case is uncertainty unit"
        ),
        "independence_caveat": (
            "candidate predictions are new and cases >=200 are outside training, but the "
            "baseline anchor truths were previously examined; mechanism diagnostic only"
        ),
        "coherent": coherent, "random_local10_layers012": random,
        "gates": gates, "gate_passed": bool(all(gates.values())),
        "constgold_opened": False,
    }
    with open(args.output, "x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    print("ANCHORBLEND_OTHER3ABS_ANALYSIS_DONE", flush=True)


if __name__ == "__main__":
    main()
