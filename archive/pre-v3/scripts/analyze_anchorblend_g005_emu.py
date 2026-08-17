"""Evaluate the frozen g=0.05 pair emulator on coherent and random anchors."""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd


KEY = ["case", "input_index"]
BASE_TAG = "lsst_r_extnbr_v22"
CAND_TAG = os.environ.get("G005_CAND_TAG", "lsst_r_extnbr_v22_g005")
TRAINING_CASE_DESCRIPTION = os.environ.get(
    "G005_TRAINING_CASE_DESCRIPTION", "g=0.05 response cases 40--99",
)
BASE = f"R_blend_{BASE_TAG}"
CAND = f"R_blend_{CAND_TAG}"


def sem(values):
    values = np.asarray(values, float)
    return float(values.std(ddof=1) / np.sqrt(len(values)))


def stat(values):
    values = np.asarray(values, float)
    return {"mean": float(values.mean()), "case_sem": sem(values), "n_cases": len(values)}


def read(paths):
    frame = pd.concat([pd.read_feather(path) for path in paths], ignore_index=True)
    # The coherent replay source also contains older cases 100--199.  They are
    # intentionally outside this frozen gate; random arms already start at 200.
    frame = frame[(frame["case"] >= 200) & (frame["case"] <= 299)].copy()
    if frame.duplicated(KEY).any():
        raise RuntimeError("duplicate anchor keys")
    missing = {"R_blend_truth", BASE, CAND, *KEY} - set(frame)
    if missing:
        raise KeyError(f"anchor table lacks {sorted(missing)}")
    if frame["case"].nunique() != 100:
        raise RuntimeError("anchor gate must contain every case 200--299")
    return frame


def conditional(frame, edges):
    values = frame[BASE].to_numpy(float)
    rows = []
    for index, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (values >= lo) & (values < hi)
        subset = frame.loc[mask]
        if len(subset) < 100:
            continue
        case = subset.groupby("case", sort=True)[["base_gap", "cand_gap", "shift"]].mean()
        rows.append({
            "bin": index,
            "edges": [
                None if not np.isfinite(value) else float(value) for value in (lo, hi)
            ],
            "n_rows": int(len(subset)),
            "baseline_minus_truth": stat(case["base_gap"]),
            "candidate_minus_truth": stat(case["cand_gap"]),
            "candidate_minus_baseline": stat(case["shift"]),
        })
    return rows


def summarize(frame, development_max=249):
    work = frame.copy()
    work["base_gap"] = work[BASE] - work["R_blend_truth"]
    work["cand_gap"] = work[CAND] - work["R_blend_truth"]
    work["shift"] = work[CAND] - work[BASE]
    development = work.case <= development_max
    edges = np.quantile(work.loc[development, BASE], np.linspace(0, 1, 6))
    edges[0] = -np.inf
    edges[-1] = np.inf
    output = {}
    for name, mask in (
        ("development", development),
        ("validation", ~development),
        ("all", np.ones(len(work), bool)),
    ):
        subset = work.loc[mask]
        case = subset.groupby("case", sort=True)[
            ["R_blend_truth", BASE, CAND, "base_gap", "cand_gap", "shift"]
        ].mean()
        output[name] = {
            "n_rows": int(len(subset)), "n_cases": int(len(case)),
            "truth": stat(case["R_blend_truth"]),
            "baseline_prediction": stat(case[BASE]),
            "candidate_prediction": stat(case[CAND]),
            "baseline_minus_truth": stat(case["base_gap"]),
            "candidate_minus_truth": stat(case["cand_gap"]),
            "candidate_minus_baseline": stat(case["shift"]),
            "conditional_baseline_prediction_quintiles": conditional(subset, edges),
        }
        rows = output[name]["conditional_baseline_prediction_quintiles"]
        output[name]["max_abs_conditional_baseline_gap"] = float(max(
            abs(row["baseline_minus_truth"]["mean"]) for row in rows
        ))
        output[name]["max_abs_conditional_candidate_gap"] = float(max(
            abs(row["candidate_minus_truth"]["mean"]) for row in rows
        ))
    return output


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--coherent", required=True)
    ap.add_argument("--random", action="append", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    if os.path.exists(args.output):
        raise FileExistsError(f"refusing to overwrite {args.output}")
    coherent = summarize(read([args.coherent]))
    random = summarize(read(args.random))
    cv = coherent["validation"]
    rv = random["validation"]
    cshift = cv["candidate_minus_baseline"]
    result = {
        "design": (
            f"candidate trained on {TRAINING_CASE_DESCRIPTION}; anchors 200--249 "
            "development and 250--299 validation; case is uncertainty unit"
        ),
        "baseline_tag": BASE_TAG, "candidate_tag": CAND_TAG,
        "coherent": coherent, "random_local10_layers012": random,
        "gates": {
            "coherent_validation_absolute_gap_smaller": bool(
                abs(cv["candidate_minus_truth"]["mean"])
                < abs(cv["baseline_minus_truth"]["mean"])
            ),
            "coherent_validation_shift_positive": bool(cshift["mean"] > 0),
            "coherent_validation_shift_resolved": bool(
                cshift["mean"] > 2 * cshift["case_sem"]
            ),
            "random_validation_absolute_gap_smaller": bool(
                abs(rv["candidate_minus_truth"]["mean"])
                < abs(rv["baseline_minus_truth"]["mean"])
            ),
            "coherent_validation_max_conditional_gap_smaller": bool(
                cv["max_abs_conditional_candidate_gap"]
                < cv["max_abs_conditional_baseline_gap"]
            ),
        },
        "constgold_opened": False,
    }
    result["gate_passed"] = bool(all(result["gates"].values()))
    with open(args.output, "x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    print("ANCHORBLEND_G005_EMULATOR_ANALYSIS_DONE", flush=True)


if __name__ == "__main__":
    main()
