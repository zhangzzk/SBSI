"""Analyze matched random-direction anchor response at 10 and 15 arcsec.

The two arms use identical sparse anchors, stable spin-2 directions, latent
catalogues and noise.  Their difference directly estimates the 10--15 arcsec
response shell.  Cases 200--249 fix a single additive shell response; cases
250--299 validate it unchanged.  Constgold is never read.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd


KEY = ["case", "input_index"]


def sem(values) -> float:
    values = np.asarray(values, float)
    return float(values.std(ddof=1) / np.sqrt(len(values)))


def stat(values) -> dict:
    values = np.asarray(values, float)
    return {"mean": float(values.mean()), "case_sem": sem(values)}


def assert_matching_manifests(root10: str, root15: str, cases: list[int]) -> None:
    columns = ["index", "u1", "u2"]
    for case in cases:
        a = pd.read_feather(os.path.join(root10, f"anchors_case{case}.feather"), columns=columns)
        b = pd.read_feather(os.path.join(root15, f"anchors_case{case}.feather"), columns=columns)
        if not a.equals(b):
            raise RuntimeError(f"case {case}: radius arms have different anchors/directions")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--response10", required=True)
    ap.add_argument("--response15", required=True)
    ap.add_argument("--manifest-root10", required=True)
    ap.add_argument("--manifest-root15", required=True)
    ap.add_argument("--case-min", type=int, default=200)
    ap.add_argument("--case-max", type=int, default=299)
    ap.add_argument("--development-max", type=int, default=249)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    if os.path.exists(args.output):
        raise FileExistsError(f"refusing to overwrite {args.output}")
    cases = list(range(args.case_min, args.case_max + 1))
    assert_matching_manifests(args.manifest_root10, args.manifest_root15, cases)

    columns = KEY + ["R_blend_truth", "R_blend_null", "R_blend_lsst_r_extnbr_v22",
                     "r_input_p_plus", "Re_input_p_plus"]
    r10 = pd.read_feather(args.response10, columns=columns)
    r15 = pd.read_feather(args.response15, columns=columns)
    r10 = r10[(r10.case >= args.case_min) & (r10.case <= args.case_max)]
    r15 = r15[(r15.case >= args.case_min) & (r15.case <= args.case_max)]
    joined = r10.merge(r15, on=KEY, how="inner", suffixes=("_10", "_15"),
                       validate="one_to_one")
    coverage = len(joined) / max(len(r10), 1)
    if coverage < 0.99 or len(joined) / max(len(r15), 1) < 0.99:
        raise RuntimeError(f"common radius-arm key coverage below 99%: {coverage:.3%}")
    for column in ("R_blend_lsst_r_extnbr_v22", "r_input_p_plus", "Re_input_p_plus"):
        if not np.array_equal(joined[f"{column}_10"], joined[f"{column}_15"]):
            raise RuntimeError(f"radius arms differ in frozen column {column}")
    joined["shell_10_15"] = joined["R_blend_truth_15"] - joined["R_blend_truth_10"]
    joined["shell_null_10_15"] = joined["R_blend_null_15"] - joined["R_blend_null_10"]
    joined["gap10"] = (
        joined["R_blend_lsst_r_extnbr_v22_10"] - joined["R_blend_truth_10"]
    )
    joined["gap15"] = (
        joined["R_blend_lsst_r_extnbr_v22_10"] - joined["R_blend_truth_15"]
    )
    case = joined.groupby("case", sort=True)[[
        "R_blend_truth_10", "R_blend_truth_15", "R_blend_null_10", "R_blend_null_15",
        "R_blend_lsst_r_extnbr_v22_10", "shell_10_15", "shell_null_10_15", "gap10", "gap15",
    ]].mean()
    development = case.index.to_numpy(int) <= args.development_max
    validation = ~development
    if development.sum() < 20 or validation.sum() < 20:
        raise RuntimeError("need at least 20 independent cases per split")
    dev_shell = float(case.loc[development, "shell_10_15"].mean())
    corrected_validation = case.loc[validation, "gap15"].to_numpy(float) + dev_shell
    raw_validation = case.loc[validation, "gap15"].to_numpy(float)
    transfer_uncertainty = float(np.hypot(
        sem(case.loc[development, "shell_10_15"]), sem(raw_validation),
    ))
    shell_all = case["shell_10_15"].to_numpy(float)
    shell_dev = case.loc[development, "shell_10_15"].to_numpy(float)
    shell_val = case.loc[validation, "shell_10_15"].to_numpy(float)
    split_uncertainty = float(np.hypot(sem(shell_dev), sem(shell_val)))
    gates = {
        "local10_gap_consistent_with_zero_at_2_case_sem": bool(
            abs(case["gap10"].mean()) <= 2.0 * sem(case["gap10"])
        ),
        "shell_10_15_detected_at_3_case_sem": bool(
            abs(shell_all.mean()) >= 3.0 * sem(shell_all)
        ),
        "shell_split_consistent_at_2_combined_sem": bool(
            abs(shell_dev.mean() - shell_val.mean()) <= 2.0 * split_uncertainty
        ),
        "corrected_validation_gap_smaller": bool(
            abs(corrected_validation.mean()) < abs(raw_validation.mean())
        ),
        "corrected_validation_gap_consistent_with_zero_at_2_combined_sem": bool(
            abs(corrected_validation.mean()) <= 2.0 * transfer_uncertainty
        ),
        "truth_and_shell_nulls_within_3_case_sem": bool(all(
            abs(case[column].mean()) <= 3.0 * sem(case[column])
            for column in ("R_blend_null_10", "R_blend_null_15", "shell_null_10_15")
        )),
    }
    payload = {
        "design": "independent stable spin-2 direction per non-overlapping anchor neighbourhood",
        "case_window": [args.case_min, args.case_max],
        "development_cases": [int(case.index[development].min()), int(case.index[development].max())],
        "validation_cases": [int(case.index[validation].min()), int(case.index[validation].max())],
        "n_common_rows": len(joined), "n_cases": len(case),
        "common_key_coverage_of_radius10": float(coverage),
        "truth_local10": stat(case["R_blend_truth_10"]),
        "truth_local15": stat(case["R_blend_truth_15"]),
        "v22_prediction": stat(case["R_blend_lsst_r_extnbr_v22_10"]),
        "v22_minus_local10": stat(case["gap10"]),
        "v22_minus_local15": stat(case["gap15"]),
        "shell_10_15": stat(shell_all),
        "shell_10_15_development": stat(shell_dev),
        "shell_10_15_validation": stat(shell_val),
        "null_local10": stat(case["R_blend_null_10"]),
        "null_local15": stat(case["R_blend_null_15"]),
        "shell_null_10_15": stat(case["shell_null_10_15"]),
        "fixed_development_shell_offset": dev_shell,
        "validation_raw_v22_minus_local15": stat(raw_validation),
        "validation_corrected_v22_minus_local15": stat(corrected_validation),
        "validation_corrected_mean_uncertainty_including_development": transfer_uncertainty,
        "gates": gates, "gate_passed": bool(all(gates.values())),
        "constgold_opened": False,
    }
    with open(args.output, "x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    print("ANCHORBLEND_RANDOM_RADII_ANALYSIS_DONE", flush=True)


if __name__ == "__main__":
    main()
