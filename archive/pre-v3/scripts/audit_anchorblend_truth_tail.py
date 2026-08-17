"""Decompose coherent-anchor gaps by measured truth-response tails."""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd


KEY = ["case", "input_index"]


def stat(values) -> dict:
    x = np.asarray(values, float)
    return {
        "mean": float(x.mean()),
        "case_sem": float(x.std(ddof=1) / np.sqrt(len(x))),
        "n_cases": int(len(x)),
    }


def summarize_bin(frame: pd.DataFrame, mask: np.ndarray, gap: str) -> dict:
    work = frame[["case", gap]].copy()
    work["selected"] = np.asarray(mask, bool)
    work["selected_gap"] = np.where(work.selected, work[gap], 0.0)
    grouped = work.groupby("case", sort=True)
    case = grouped.agg(
        n_total=(gap, "size"), n_selected=("selected", "sum"),
        contribution=("selected_gap", "mean"),
    )
    case["row_fraction"] = case.n_selected / case.n_total
    selected = work.loc[work.selected].groupby("case", sort=True)[gap].mean()
    return {
        "n_rows": int(mask.sum()),
        "row_fraction": stat(case.row_fraction),
        "contribution_to_global_gap": stat(case.contribution),
        "conditional_gap": stat(selected) if len(selected) else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--response", action="append", required=True)
    parser.add_argument("--parts-root", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--case-min", type=int, default=200)
    parser.add_argument("--case-max", type=int, default=399)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if os.path.exists(args.output):
        raise FileExistsError(f"refusing to overwrite {args.output}")
    truth = pd.concat([
        pd.read_feather(path, columns=KEY + [
            "R_blend_truth", "R_blend_lsst_r_extnbr_v22",
        ])
        for path in args.response
    ], ignore_index=True)
    truth = truth.loc[truth.case.between(args.case_min, args.case_max)].copy()
    paths = [
        os.path.join(args.parts_root, args.tag, f"case{case}.feather")
        for case in range(args.case_min, args.case_max + 1)
    ]
    missing = [path for path in paths if not os.path.isfile(path)]
    if missing:
        raise RuntimeError(f"missing {len(missing)} candidate case files")
    candidate = pd.concat([
        pd.read_feather(path, columns=KEY + ["prediction_model", "prediction_baseline"])
        for path in paths
    ], ignore_index=True)
    if truth.duplicated(KEY).any() or candidate.duplicated(KEY).any():
        raise RuntimeError("duplicate anchor key")
    frame = truth.merge(candidate, on=KEY, validate="one_to_one")
    replay = float(np.max(np.abs(
        frame.R_blend_lsst_r_extnbr_v22 - frame.prediction_baseline
    )))
    if replay > 5e-7:
        raise RuntimeError(f"baseline replay differs by {replay:.3e}")
    frame["gap_v22"] = frame.prediction_baseline - frame.R_blend_truth
    frame["gap_candidate"] = frame.prediction_model - frame.R_blend_truth
    truth_value = frame.R_blend_truth.to_numpy(float)
    edges = np.asarray([-np.inf, -1.0, -0.5, -0.2, 0.0, 0.2, 0.5, 1.0, np.inf])
    bins = np.searchsorted(edges, truth_value, side="right") - 1
    payload = {
        "design": "descriptive post-response decomposition; measured truth is not a deployable feature",
        "cases": [args.case_min, args.case_max],
        "candidate_tag": args.tag,
        "n_rows": int(len(frame)),
        "baseline_replay_max_abs": replay,
        "global": {}, "truth_response_bins": {}, "absolute_truth_tails": {},
        "constgold_opened": False,
    }
    for gap in ("gap_v22", "gap_candidate"):
        case_gap = frame.groupby("case", sort=True)[gap].mean()
        payload["global"][gap] = stat(case_gap)
    for index in range(len(edges) - 1):
        entry = {
            "lo": None if not np.isfinite(edges[index]) else float(edges[index]),
            "hi": None if not np.isfinite(edges[index + 1]) else float(edges[index + 1]),
        }
        mask = bins == index
        for gap in ("gap_v22", "gap_candidate"):
            entry[gap] = summarize_bin(frame, mask, gap)
        payload["truth_response_bins"][str(index)] = entry
    absolute = np.abs(truth_value)
    for threshold in (0.2, 0.5, 1.0, 2.0, 5.0):
        mask = absolute > threshold
        payload["absolute_truth_tails"][str(threshold)] = {
            gap: summarize_bin(frame, mask, gap)
            for gap in ("gap_v22", "gap_candidate")
        }
    with open(args.output, "x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    print("ANCHORBLEND_TRUTH_TAIL_AUDIT_DONE", flush=True)


if __name__ == "__main__":
    main()
