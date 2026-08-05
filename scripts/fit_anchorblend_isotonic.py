"""Fit a monotonic per-primary R_blend calibration and validate it by case.

The development half fixes both the isotonic step function and raw-prediction
quintile edges.  The test half is opened only for the reported gates.  Once
those gates pass, the unchanged method is refit on all anchor cases for the
deployment artifact.  No constgold input is read.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


def case_residual(frame, truth, prediction):
    means = frame.groupby("case")[[truth, prediction]].mean()
    values = means[truth] - means[prediction]
    return float(values.mean()), float(values.std(ddof=1) / np.sqrt(len(values)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input")
    ap.add_argument("--tag", default="lsst_r_extnbr_v21")
    ap.add_argument("--split-case", type=int, default=50)
    ap.add_argument("--output-npz", required=True)
    ap.add_argument("--output-json", required=True)
    args = ap.parse_args()

    truth = "R_blend_truth"
    raw = f"R_blend_{args.tag}"
    data = pd.read_feather(args.input)
    finite = np.isfinite(data[[truth, raw]].to_numpy(float)).all(axis=1)
    data = data.loc[finite, ["case", truth, raw]].copy()
    dev = data[data["case"] < args.split_case].copy()
    test = data[data["case"] >= args.split_case].copy()
    if dev["case"].nunique() < 10 or test["case"].nunique() < 10:
        raise RuntimeError("need at least 10 independent cases in each split")

    model = IsotonicRegression(out_of_bounds="clip").fit(dev[raw], dev[truth])
    dev["calibrated"] = model.predict(dev[raw])
    test["calibrated"] = model.predict(test[raw])
    edges = np.unique(np.quantile(dev[raw], np.linspace(0.0, 1.0, 6)))
    if len(edges) != 6:
        raise RuntimeError("raw response quintile edges are not unique")
    test["bin"] = np.clip(np.searchsorted(edges, test[raw], side="right") - 1, 0, 4)

    global_residual, global_se = case_residual(test, truth, "calibrated")
    bins = []
    for index in range(5):
        rows = test[test["bin"] == index]
        raw_residual, raw_se = case_residual(rows, truth, raw)
        cal_residual, cal_se = case_residual(rows, truth, "calibrated")
        bins.append({
            "bin": index,
            "n": len(rows),
            "raw_residual": raw_residual,
            "raw_residual_se": raw_se,
            "calibrated_residual": cal_residual,
            "calibrated_residual_se": cal_se,
        })

    summary = {
        "tag": args.tag,
        "method": "isotonic",
        "split_case": args.split_case,
        "n_dev_cases": int(dev["case"].nunique()),
        "n_test_cases": int(test["case"].nunique()),
        "n_thresholds": int(len(model.X_thresholds_)),
        "test_residual": global_residual,
        "test_residual_se": global_se,
        "test_bins": bins,
    }
    # Gate the method before writing a deployable artifact.  The global mean
    # must close, and both response-dominant upper quintiles must improve in
    # absolute residual without becoming >2 sigma discrepancies.
    if abs(global_residual) > 0.006 or abs(global_residual) > 2.0 * global_se:
        raise RuntimeError(f"global held-out isotonic gate failed: {summary}")
    for result in bins[-2:]:
        if abs(result["calibrated_residual"]) >= abs(result["raw_residual"]):
            raise RuntimeError(f"upper-quintile improvement gate failed: {result}")
        if abs(result["calibrated_residual"]) > 2.0 * result["calibrated_residual_se"]:
            raise RuntimeError(f"upper-quintile consistency gate failed: {result}")

    # The method and every gate are frozen before this point.  Refit the same
    # parameter-free estimator on all available anchor scenes for deployment;
    # constgold remains an external acceptance set and is never read here.
    deployment_model = IsotonicRegression(out_of_bounds="clip").fit(data[raw], data[truth])
    summary["deployment_fit_cases"] = int(data["case"].nunique())
    summary["deployment_n_thresholds"] = int(len(deployment_model.X_thresholds_))
    np.savez(
        args.output_npz,
        x_thresholds=np.asarray(deployment_model.X_thresholds_, dtype=float),
        y_thresholds=np.asarray(deployment_model.y_thresholds_, dtype=float),
        validation_x_thresholds=np.asarray(model.X_thresholds_, dtype=float),
        validation_y_thresholds=np.asarray(model.y_thresholds_, dtype=float),
        quintile_edges=edges,
        tag=np.array(args.tag),
        split_case=np.array(args.split_case),
    )
    with open(args.output_json, "x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("ANCHORBLEND_ISOTONIC_DONE", flush=True)


if __name__ == "__main__":
    main()
