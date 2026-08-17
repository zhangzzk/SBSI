"""Held-out test of a fixed outer-scene response correction.

Cases 200--249 train one predeclared histogram-gradient regressor for the
matched response contributed by sources outside 10 arcsec.  Cases 250--299 are
opened once for validation.  The model uses only intrinsic primary properties,
the frozen V2.2 prediction, and deterministic outer catalogue summaries; it
does not read constgold.  A deployment model is saved only when the unchanged
candidate improves both the raw and constant-offset validation residuals and
is consistent with zero.
"""
from __future__ import annotations

import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


KEY = ["case", "input_index"]
FEATURES = [
    "r_input_p_plus", "Re_input_p_plus", "R_blend_lsst_r_extnbr_v22_all",
    "logflux_outer_10_15", "logflux_outer_15_20", "logflux_outer_20_30",
    "logflux_re2_over_d2_outer_10_30", "max_re_over_d_outer_10_30",
    "n_outer_10_30",
]


def sem(values: np.ndarray) -> float:
    values = np.asarray(values, float)
    return float(values.std(ddof=1) / np.sqrt(len(values)))


def stat(values: np.ndarray) -> dict:
    values = np.asarray(values, float)
    return {"mean": float(values.mean()), "case_sem": sem(values)}


def make_model(seed: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="squared_error", learning_rate=0.05, max_iter=100,
        max_leaf_nodes=15, min_samples_leaf=1000, l2_regularization=10.0,
        random_state=seed,
    )


def case_mean(frame: pd.DataFrame, column: str) -> np.ndarray:
    return frame.groupby("case", sort=True)[column].mean().to_numpy(float)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--all-response", required=True)
    ap.add_argument("--local-response", required=True)
    ap.add_argument("--outer-features", required=True)
    ap.add_argument("--development-max", type=int, default=249)
    ap.add_argument("--seed", type=int, default=25876)
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--output-model", required=True)
    args = ap.parse_args()
    for path in (args.output_json, args.output_model):
        if os.path.exists(path):
            raise FileExistsError(f"refusing to overwrite {path}")

    columns = KEY + ["R_blend_truth", "R_blend_lsst_r_extnbr_v22",
                     "r_input_p_plus", "Re_input_p_plus"]
    all_frame = pd.read_feather(args.all_response, columns=columns)
    local = pd.read_feather(args.local_response, columns=columns)
    if local.empty:
        raise RuntimeError("local response table is empty")
    all_frame = all_frame[
        (all_frame.case >= int(local.case.min()))
        & (all_frame.case <= int(local.case.max()))
    ].copy()
    joined = all_frame.merge(local, on=KEY, how="inner", suffixes=("_all", "_local"),
                             validate="one_to_one")
    for column in ("r_input_p_plus", "Re_input_p_plus"):
        if not np.array_equal(joined[f"{column}_all"], joined[f"{column}_local"]):
            raise RuntimeError(f"primary truth changed in {column}")
        joined[column] = joined[f"{column}_all"]
    features = pd.read_feather(args.outer_features)
    joined = joined.merge(features, on=KEY, how="inner", validate="one_to_one")
    if len(joined) < 0.99 * len(all_frame):
        raise RuntimeError("joined outer-correction coverage below 99%")
    if not np.array_equal(
        joined["R_blend_lsst_r_extnbr_v22_all"],
        joined["R_blend_lsst_r_extnbr_v22_local"],
    ):
        raise RuntimeError("V2.2 prediction changed between controlled arms")
    joined["outside_truth"] = (
        joined["R_blend_truth_all"] - joined["R_blend_truth_local"]
    )
    joined["raw_gap"] = (
        joined["R_blend_lsst_r_extnbr_v22_all"] - joined["R_blend_truth_all"]
    )
    finite = np.isfinite(joined[FEATURES + ["outside_truth", "raw_gap"]].to_numpy(float)).all(1)
    if not finite.all():
        raise RuntimeError(f"{(~finite).sum()} non-finite training rows")
    development = joined.case <= args.development_max
    validation = ~development
    if joined.loc[development, "case"].nunique() < 20 or joined.loc[validation, "case"].nunique() < 20:
        raise RuntimeError("need at least 20 cases on both sides of the split")

    model = make_model(args.seed)
    model.fit(
        joined.loc[development, FEATURES].to_numpy(float),
        joined.loc[development, "outside_truth"].to_numpy(float),
    )
    joined.loc[validation, "outer_prediction"] = model.predict(
        joined.loc[validation, FEATURES].to_numpy(float),
    )
    dev_offset = float(joined.loc[development, "outside_truth"].mean())
    joined.loc[validation, "constant_corrected_gap"] = (
        joined.loc[validation, "raw_gap"] + dev_offset
    )
    joined.loc[validation, "feature_corrected_gap"] = (
        joined.loc[validation, "raw_gap"] + joined.loc[validation, "outer_prediction"]
    )
    validation_frame = joined.loc[validation].copy()
    raw_case = case_mean(validation_frame, "raw_gap")
    constant_case = case_mean(validation_frame, "constant_corrected_gap")
    feature_case = case_mean(validation_frame, "feature_corrected_gap")
    truth_case = case_mean(validation_frame, "outside_truth")
    prediction_case = case_mean(validation_frame, "outer_prediction")
    raw = stat(raw_case)
    constant = stat(constant_case)
    feature = stat(feature_case)
    gates = {
        "feature_gap_smaller_than_raw": bool(abs(feature["mean"]) < abs(raw["mean"])),
        "feature_gap_smaller_than_constant": bool(abs(feature["mean"]) < abs(constant["mean"])),
        "feature_gap_consistent_with_zero_at_2_case_sem": bool(
            abs(feature["mean"]) <= 2.0 * feature["case_sem"]
        ),
        "feature_case_mse_better_than_constant": bool(
            np.mean((prediction_case - truth_case) ** 2)
            < np.mean((dev_offset - truth_case) ** 2)
        ),
    }
    passed = bool(all(gates.values()))
    payload = {
        "tag": "lsst_r_extnbr_v22", "method": "outer_scene_hgb_additive_response",
        "features": FEATURES, "model_parameters": model.get_params(),
        "development_cases": [int(joined.loc[development, "case"].min()),
                              int(joined.loc[development, "case"].max())],
        "validation_cases": [int(validation_frame.case.min()), int(validation_frame.case.max())],
        "n_development_rows": int(development.sum()),
        "n_validation_rows": int(validation.sum()),
        "development_constant_offset": dev_offset,
        "validation_outside_truth": stat(truth_case),
        "validation_feature_prediction": stat(prediction_case),
        "validation_raw_gap": raw,
        "validation_constant_corrected_gap": constant,
        "validation_feature_corrected_gap": feature,
        "validation_outer_case_mse_constant": float(np.mean((dev_offset - truth_case) ** 2)),
        "validation_outer_case_mse_feature": float(np.mean((prediction_case - truth_case) ** 2)),
        "validation_outer_row_mse_constant": float(np.mean(
            (dev_offset - validation_frame["outside_truth"].to_numpy(float)) ** 2
        )),
        "validation_outer_row_mse_feature": float(np.mean(
            (validation_frame["outer_prediction"].to_numpy(float)
             - validation_frame["outside_truth"].to_numpy(float)) ** 2
        )),
        "gates": gates, "gate_passed": passed,
        "constgold_opened": False,
    }
    if passed:
        deployment = make_model(args.seed)
        deployment.fit(joined[FEATURES].to_numpy(float), joined["outside_truth"].to_numpy(float))
        artifact = {
            "model": deployment, "features": FEATURES, "tag": payload["tag"],
            "method": payload["method"], "validation": payload,
        }
        joblib.dump(artifact, args.output_model)
        payload["deployment_model"] = os.path.abspath(args.output_model)
    else:
        payload["deployment_model"] = None
    with open(args.output_json, "x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    print("ANCHORBLEND_OUTER_CORRECTION_DONE", flush=True)


if __name__ == "__main__":
    main()
