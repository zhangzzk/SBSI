"""Refit the validated six-bin pair calibration on all c0--39 cases."""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from fit_halfshear_pair_vector_calibration import vector_fit


def fit_coefficients(frame: pd.DataFrame, n_bins: int,
                     ridge_fraction: float) -> tuple[np.ndarray, float, float]:
    cosine = [f"b{i}_cos" for i in range(n_bins)]
    sine = [f"b{i}_sin" for i in range(n_bins)]
    x = np.vstack([frame[cosine].to_numpy(float), frame[sine].to_numpy(float)])
    y = np.concatenate([frame.y1.to_numpy(float), frame.y2.to_numpy(float)])
    count_by_case = frame.groupby("case").size()
    row_weight = 1.0 / frame.case.map(count_by_case).to_numpy(float)
    weight = np.concatenate([row_weight, row_weight])
    weight *= len(weight) / weight.sum()
    xw, yw = x * np.sqrt(weight[:, None]), y * np.sqrt(weight)
    gram, rhs = xw.T @ xw, xw.T @ yw
    ridge = float(ridge_fraction * np.trace(gram) / n_bins)
    regularized = gram + ridge * np.eye(n_bins)
    coefficient = np.linalg.solve(regularized, rhs + ridge * np.ones(n_bins))
    return coefficient, ridge, float(np.linalg.cond(regularized))


def apply(frame: pd.DataFrame, coefficient: np.ndarray) -> pd.DataFrame:
    frame = frame.copy()
    n_bins = len(coefficient)
    frame["corrected_cos"] = frame[[f"b{i}_cos" for i in range(n_bins)]].to_numpy(float) @ coefficient
    frame["corrected_sin"] = frame[[f"b{i}_sin" for i in range(n_bins)]].to_numpy(float) @ coefficient
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", required=True)
    parser.add_argument("--source-calibration", required=True)
    parser.add_argument("--ridge-fraction", type=float, default=1e-6)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if os.path.exists(args.output):
        raise FileExistsError(f"refusing to overwrite {args.output}")
    source = json.load(open(args.source_calibration, encoding="utf-8"))
    n_bins = len(source["coefficients"])
    columns = ["case", "input_index", "y1", "y2", "raw_cos", "raw_sin"]
    for index in range(n_bins):
        columns.extend([f"b{index}_cos", f"b{index}_sin"])
    frame = pd.read_feather(args.table, columns=columns)
    first = frame.case <= 19
    coefficient_first, _, _ = fit_coefficients(frame.loc[first], n_bins, args.ridge_fraction)
    coefficient_second, _, _ = fit_coefficients(frame.loc[~first], n_bins, args.ridge_fraction)
    coefficient_all, ridge, condition = fit_coefficients(frame, n_bins, args.ridge_fraction)
    corrected = apply(frame, coefficient_all)
    payload = {
        "candidate": "validated six-bin rotation-equivariant pair calibration refit on all half-shear cases 0--39",
        "response_bin_edges": source["response_bin_edges"],
        "coefficients": [float(x) for x in coefficient_all],
        "coefficients_c0_19": [float(x) for x in coefficient_first],
        "coefficients_c20_39": [float(x) for x in coefficient_second],
        "coefficient_half_max_abs_difference": float(np.max(np.abs(
            coefficient_first - coefficient_second
        ))),
        "ridge_fraction": float(args.ridge_fraction), "ridge_absolute": ridge,
        "gram_condition_number": condition,
        "all40_raw": vector_fit(frame, "raw"),
        "all40_refit_in_sample": vector_fit(corrected, "corrected"),
        "fresh_validation_required": True,
        "deployable_model_written": False,
        "constgold_opened": False, "anchor_truth_opened": False,
    }
    with open(args.output, "x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    print("HALFSHEAR_PAIR_VECTOR_CALIBRATION_REFIT_DONE", flush=True)


if __name__ == "__main__":
    main()
