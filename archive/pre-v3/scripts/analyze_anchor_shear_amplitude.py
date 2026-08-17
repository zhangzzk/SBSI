"""Paired coherent-anchor gap comparison at g=.02 and g=.05."""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd


KEY = ["case", "input_index"]


def stat(values: np.ndarray) -> dict:
    values = np.asarray(values, float)
    return {
        "mean": float(values.mean()),
        "case_sem": float(values.std(ddof=1) / np.sqrt(len(values))),
        "n_cases": int(len(values)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--g002", required=True)
    ap.add_argument("--g005", required=True)
    ap.add_argument("--case-min", type=int, default=400)
    ap.add_argument("--case-max", type=int, default=499)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    if os.path.exists(args.output):
        raise FileExistsError(f"refusing to overwrite {args.output}")
    columns = KEY + ["R_blend_truth", "R_blend_lsst_r_extnbr_v22"]
    low = pd.read_feather(args.g002, columns=columns)
    high = pd.read_feather(args.g005, columns=columns)
    low = low[low.case.between(args.case_min, args.case_max)]
    high = high[high.case.between(args.case_min, args.case_max)]
    if low.duplicated(KEY).any() or high.duplicated(KEY).any():
        raise RuntimeError("duplicate response keys")
    frame = low.merge(high, on=KEY, suffixes=("_g002", "_g005"), validate="one_to_one")
    coverage = min(len(frame) / len(low), len(frame) / len(high))
    if coverage < 0.95:
        raise RuntimeError(f"common response coverage below 95%: {coverage:.3%}")
    prediction_replay = np.max(np.abs(
        frame.R_blend_lsst_r_extnbr_v22_g002 - frame.R_blend_lsst_r_extnbr_v22_g005
    ))
    if prediction_replay > 2e-7:
        raise RuntimeError(f"paired V2.2 predictions differ by {prediction_replay:.3e}")
    frame["gap_g002"] = (
        frame.R_blend_lsst_r_extnbr_v22_g002 - frame.R_blend_truth_g002
    )
    frame["gap_g005"] = (
        frame.R_blend_lsst_r_extnbr_v22_g005 - frame.R_blend_truth_g005
    )
    frame["gap_g002_minus_g005"] = frame.gap_g002 - frame.gap_g005
    frame["truth_g002_minus_g005"] = frame.R_blend_truth_g002 - frame.R_blend_truth_g005
    case = frame.groupby("case", sort=True)[
        ["gap_g002", "gap_g005", "gap_g002_minus_g005", "truth_g002_minus_g005"]
    ].mean()
    result = {column: stat(case[column].to_numpy(float)) for column in case}
    low_abs = abs(result["gap_g002"]["mean"])
    high_abs = abs(result["gap_g005"]["mean"])
    payload = {
        "design": "same latent anchor cases and V2.2 predictions; coherent neighbour-only central response at g=.02 versus g=.05",
        "case_window": [args.case_min, args.case_max],
        "n_g002": int(len(low)), "n_g005": int(len(high)),
        "n_common": int(len(frame)), "common_coverage": float(coverage),
        "prediction_replay_max_abs": float(prediction_replay),
        "statistics": result,
        "predeclared_gates": {
            "finite_amplitude_carrier_abs_gap_g002_below_half_g005": bool(
                low_abs < 0.5 * high_abs
            ),
            "persistent_gap_g002_consistent_with_g005_at_2sem": bool(
                abs(result["gap_g002_minus_g005"]["mean"])
                <= 2.0 * result["gap_g002_minus_g005"]["case_sem"]
            ),
        },
        "constgold_opened": False,
    }
    with open(args.output, "x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    print("ANCHOR_SHEAR_AMPLITUDE_ANALYSIS_DONE", flush=True)


if __name__ == "__main__":
    main()
