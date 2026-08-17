"""Compare all-field-sheared and aperture-matched local10 anchor truths."""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd


def sem(x) -> float:
    x = np.asarray(x, float)
    return float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan")


def stat(values: np.ndarray) -> dict:
    values = np.asarray(values, float)
    return {"mean": float(values.mean()), "case_sem": sem(values)}


def split_additive_aperture_test(case: pd.DataFrame, development_max: int) -> dict:
    """Fit an additive outside-aperture term on cases before the fixed split.

    The correction is deliberately a single response offset, not a rescaling of
    the in-aperture emulator.  It is fixed on the development cases and then
    applied unchanged to later cases.
    """
    development = case.index.to_numpy(int) <= int(development_max)
    validation = ~development
    if development.sum() < 2 or validation.sum() < 2:
        raise RuntimeError(
            f"development split at {development_max} leaves "
            f"{development.sum()}/{validation.sum()} cases"
        )
    all_truth = case["R_blend_truth_all"].to_numpy(float)
    local_truth = case["R_blend_truth_local"].to_numpy(float)
    prediction = case["R_blend_lsst_r_extnbr_v22_all"].to_numpy(float)
    outside = all_truth - local_truth
    correction = float(outside[development].mean())
    raw_gap = prediction - all_truth
    corrected_gap = prediction + correction - all_truth
    consistency_sem = float(np.hypot(sem(outside[development]), sem(outside[validation])))
    validation_residual_se = float(
        np.hypot(sem(outside[development]), sem(raw_gap[validation]))
    )
    return {
        "development_cases": [int(case.index[development].min()), int(case.index[development].max())],
        "validation_cases": [int(case.index[validation].min()), int(case.index[validation].max())],
        "n_development_cases": int(development.sum()),
        "n_validation_cases": int(validation.sum()),
        "development_outside10_response": stat(outside[development]),
        "validation_outside10_response": stat(outside[validation]),
        "fixed_additive_response": correction,
        "validation_raw_v22_minus_all_truth": stat(raw_gap[validation]),
        "validation_corrected_v22_minus_all_truth": stat(corrected_gap[validation]),
        "validation_corrected_mean_uncertainty_including_development": validation_residual_se,
        "gates": {
            "outside_response_split_consistent_at_2_combined_sem": bool(
                abs(outside[development].mean() - outside[validation].mean())
                <= 2.0 * consistency_sem
            ),
            "corrected_validation_gap_smaller": bool(
                abs(corrected_gap[validation].mean()) < abs(raw_gap[validation].mean())
            ),
            "corrected_validation_gap_consistent_with_zero_at_2_case_sem": bool(
                abs(corrected_gap[validation].mean()) <= 2.0 * validation_residual_se
            ),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--all-response", required=True)
    ap.add_argument("--local-response", required=True)
    ap.add_argument("--case-min", type=int, default=200)
    ap.add_argument("--case-max", type=int, default=299)
    ap.add_argument("--development-max", type=int, default=249)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    if os.path.exists(args.output):
        raise SystemExit(f"REFUSING to overwrite {args.output}")
    columns = ["case", "input_index", "R_blend_truth", "R_blend_lsst_r_extnbr_v22",
               "r_input_p_plus", "Re_input_p_plus"]
    all_frame = pd.read_feather(args.all_response, columns=columns)
    all_frame = all_frame[(all_frame.case >= args.case_min) & (all_frame.case <= args.case_max)]
    local = pd.read_feather(args.local_response, columns=columns)
    joined = all_frame.merge(local, on=["case", "input_index"], how="inner",
                             suffixes=("_all", "_local"), validate="one_to_one")
    for column in ("r_input_p_plus", "Re_input_p_plus"):
        if not np.array_equal(joined[f"{column}_all"], joined[f"{column}_local"]):
            raise RuntimeError(f"primary truth mismatch in {column}")
    pred_all = joined.R_blend_lsst_r_extnbr_v22_all.to_numpy(float)
    pred_local = joined.R_blend_lsst_r_extnbr_v22_local.to_numpy(float)
    if not np.array_equal(pred_all, pred_local):
        raise RuntimeError(
            f"intrinsic V2.2 prediction changed; max abs={np.max(np.abs(pred_all-pred_local)):.3e}"
        )
    n_expected = len(all_frame)
    coverage = len(joined) / max(n_expected, 1)
    if coverage < 0.95:
        raise RuntimeError(f"common-key coverage {coverage:.2%} below 95%")

    case = joined.groupby("case", sort=True)[[
        "R_blend_truth_all", "R_blend_truth_local", "R_blend_lsst_r_extnbr_v22_all",
    ]].mean()
    truth_all = case.R_blend_truth_all.to_numpy(float)
    truth_local = case.R_blend_truth_local.to_numpy(float)
    pred = case.R_blend_lsst_r_extnbr_v22_all.to_numpy(float)
    outside = truth_all - truth_local
    gap_all = pred - truth_all
    gap_local = pred - truth_local
    additive_test = split_additive_aperture_test(case, args.development_max)
    deployment_offset = stat(outside)
    physical_gates = {
        "local_gap_smaller_than_all_gap": bool(abs(gap_local.mean()) < abs(gap_all.mean())),
        "local_gap_consistent_with_zero_at_2_case_sem": bool(
            abs(gap_local.mean()) <= 2 * sem(gap_local)
        ),
        "outside10_detected_at_3_case_sem": bool(abs(outside.mean()) >= 3 * sem(outside)),
        "note": (
            "All-minus-local is a matched-render aperture response on common detected anchors. "
            "It is not an emulator correction."
        ),
    }
    payload = {
        "case_window": [args.case_min, args.case_max],
        "all_rows": int(len(all_frame)), "local_rows": int(len(local)),
        "common_rows": int(len(joined)), "common_key_coverage_of_all": float(coverage),
        "n_cases": int(len(case)),
        "all_field_truth": stat(truth_all),
        "local_within10_truth": stat(truth_local),
        "all_minus_local_response": stat(outside),
        "v22_prediction": stat(pred),
        "v22_minus_all_truth": stat(gap_all),
        "v22_minus_local_truth": stat(gap_local),
        "fraction_all_deficit_explained_by_outside10": float(
            outside.mean() / max(-gap_all.mean(), np.finfo(float).tiny)
        ),
        "predeclared_additive_aperture_test": additive_test,
        "all_case_additive_response_after_gate": deployment_offset,
        "additive_calibration": {
            "tag": "lsst_r_extnbr_v22",
            "method": "matched_outside10_additive_response",
            "dev_offset": additive_test["fixed_additive_response"],
            "test_residual": additive_test[
                "validation_corrected_v22_minus_all_truth"
            ]["mean"],
            "test_residual_se": additive_test[
                "validation_corrected_mean_uncertainty_including_development"
            ],
            "deployment_offset": deployment_offset["mean"],
            "deployment_offset_se": deployment_offset["case_sem"],
            "gate_passed": bool(
                all(additive_test["gates"].values())
                and all(value for key, value in physical_gates.items() if key != "note")
            ),
            "fit_population": "unseen coherent-anchor cases 200--299",
        },
    }
    payload["gates"] = physical_gates
    with open(args.output, "w") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    print("ANCHORBLEND_LOCAL10_ANALYSIS_DONE", flush=True)


if __name__ == "__main__":
    main()
