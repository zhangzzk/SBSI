"""Compare detected-centroid and fixed-truth-position anchor responses."""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd


KEY = ["case", "input_index"]


def stat(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(values.mean()),
        "case_sem": float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else None,
        "n_cases": int(len(values)),
    }


def summarize(frame: pd.DataFrame) -> dict:
    columns = [
        "R_stored", "R_detected", "R_truthpos", "null_detected", "null_truthpos",
        "gap_stored", "gap_detected", "gap_truthpos", "truthpos_minus_detected",
        "detected_minus_stored",
    ]
    case = frame.groupby("case", sort=True)[columns].mean()
    return {
        "n_rows": int(len(frame)),
        "n_cases": int(len(case)),
        **{column: stat(case[column].to_numpy(float)) for column in columns},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--response", required=True)
    parser.add_argument("--g", type=float, default=0.05)
    parser.add_argument("--case-min", type=int, default=200)
    parser.add_argument("--case-max", type=int, default=299)
    parser.add_argument("--development-max", type=int, default=249)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if os.path.exists(args.output):
        raise FileExistsError(f"refusing to overwrite {args.output}")

    paths = sorted(glob.glob(os.path.join(args.input_dir, "case*.feather")))
    if len(paths) != args.case_max - args.case_min + 1:
        raise RuntimeError(f"expected 100 case files, found {len(paths)}")
    measured = pd.concat([pd.read_feather(path) for path in paths], ignore_index=True)
    measured = measured[measured.case.between(args.case_min, args.case_max)]
    if measured.duplicated(KEY).any():
        raise RuntimeError("duplicate measured anchor keys")
    shape_columns = [column for column in measured if column.startswith(("g1_", "g2_"))]
    finite = np.isfinite(measured[shape_columns].to_numpy(float)).all(axis=1)
    measured = measured.loc[finite].copy()

    response = pd.read_feather(args.response)
    response = response[response.case.between(args.case_min, args.case_max)][
        KEY + ["R_blend_truth", "R_blend_lsst_r_extnbr_v22"]
    ]
    frame = response.merge(measured, on=KEY, validate="one_to_one")
    if len(frame) < 0.98 * len(response):
        raise RuntimeError(f"common finite coverage below 98%: {len(frame)}/{len(response)}")
    scale = 2.0 * args.g
    frame["R_stored"] = frame.R_blend_truth
    frame["R_detected"] = (
        frame.g1_detected_plus.to_numpy(float) - frame.g1_detected_minus.to_numpy(float)
    ) / scale
    frame["R_truthpos"] = (
        frame.g1_truthpos_plus.to_numpy(float) - frame.g1_truthpos_minus.to_numpy(float)
    ) / scale
    frame["null_detected"] = (
        frame.g2_detected_plus.to_numpy(float) - frame.g2_detected_minus.to_numpy(float)
    ) / scale
    frame["null_truthpos"] = (
        frame.g2_truthpos_plus.to_numpy(float) - frame.g2_truthpos_minus.to_numpy(float)
    ) / scale
    prediction = frame.R_blend_lsst_r_extnbr_v22.to_numpy(float)
    frame["gap_stored"] = prediction - frame.R_stored.to_numpy(float)
    frame["gap_detected"] = prediction - frame.R_detected.to_numpy(float)
    frame["gap_truthpos"] = prediction - frame.R_truthpos.to_numpy(float)
    frame["truthpos_minus_detected"] = (
        frame.R_truthpos.to_numpy(float) - frame.R_detected.to_numpy(float)
    )
    frame["detected_minus_stored"] = (
        frame.R_detected.to_numpy(float) - frame.R_stored.to_numpy(float)
    )

    development = frame[frame.case <= args.development_max]
    validation = frame[frame.case > args.development_max]
    edges = np.quantile(
        development.R_blend_lsst_r_extnbr_v22.to_numpy(float), np.linspace(0, 1, 6),
    )
    edges[0] = -np.inf
    edges[-1] = np.inf
    bins = np.searchsorted(
        edges, validation.R_blend_lsst_r_extnbr_v22.to_numpy(float), side="right",
    ) - 1
    bins = np.clip(bins, 0, 4)
    conditional = {}
    for index in range(5):
        conditional[str(index)] = {
            "edges": [
                None if not np.isfinite(value) else float(value)
                for value in edges[index:index + 2]
            ],
            **summarize(validation.loc[bins == index]),
        }
    development_summary = summarize(development)
    validation_summary = summarize(validation)
    all_summary = summarize(frame)
    payload = {
        "design": (
            "same pixels and anchors; deterministic common ngmix initialization; "
            "per-leg detected centroid versus fixed input truth position"
        ),
        "case_window": [args.case_min, args.case_max],
        "development": development_summary,
        "validation": validation_summary,
        "all": all_summary,
        "validation_conditional_v22_prediction_quintiles": conditional,
        "n_response_rows": int(len(response)),
        "n_common_finite": int(len(frame)),
        "common_finite_fraction": float(len(frame) / len(response)),
        "carrier_scale_required": [-0.008, -0.006],
        "interpretation_gate": {
            "detected_position_reproduces_stored_within_2_case_sem": bool(
                abs(validation_summary["detected_minus_stored"]["mean"])
                <= 2.0 * validation_summary["detected_minus_stored"]["case_sem"]
            ),
            "fixed_position_shift_has_required_negative_sign": bool(
                validation_summary["truthpos_minus_detected"]["mean"] < 0.0
            ),
            "fixed_position_reduces_absolute_gap": bool(
                abs(validation_summary["gap_truthpos"]["mean"])
                < abs(validation_summary["gap_detected"]["mean"])
            ),
        },
        "constgold_opened": False,
    }
    with open(args.output, "x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    print("ANCHOR_FIXED_POSITION_ANALYSIS_DONE", flush=True)


if __name__ == "__main__":
    main()
