"""Test centroid motion in independent-direction anchor scenes.

The same image stamps and deterministic ngmix initializations are measured at
the production detected centroid and at the fixed input truth position.  The
resulting response vectors are compared with both the baseline V2.2 emulator
and the positive-tail weighted candidate.  No images are rendered and
constgold is not read.
"""
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
        "case_sem": float(values.std(ddof=1) / np.sqrt(len(values))),
        "n_cases": int(len(values)),
    }


def vector_fit(frame: pd.DataFrame, mode: str, g: float) -> dict:
    y1 = (frame[f"g1_{mode}_plus"] - frame[f"g1_{mode}_minus"]) / (2.0 * g)
    y2 = (frame[f"g2_{mode}_plus"] - frame[f"g2_{mode}_minus"]) / (2.0 * g)
    work = pd.DataFrame({
        "case": frame.case.to_numpy(int),
        "power": frame.prediction_u1.to_numpy(float) ** 2
                 + frame.prediction_u2.to_numpy(float) ** 2,
        "dot_product": frame.prediction_u1.to_numpy(float) * y1.to_numpy(float)
                       + frame.prediction_u2.to_numpy(float) * y2.to_numpy(float),
        "cross": -frame.prediction_u2.to_numpy(float) * y1.to_numpy(float)
                 + frame.prediction_u1.to_numpy(float) * y2.to_numpy(float),
    })
    case = work.groupby("case", sort=True).sum()
    if (case.power <= 0).any():
        raise RuntimeError("non-positive case model-vector power")
    slope = case.dot_product / case.power
    null = case.cross / case.power
    return {
        "slope_measured_on_predicted": stat(slope.to_numpy(float)),
        "slope_minus_one": stat((slope - 1.0).to_numpy(float)),
        "orthogonal_slope_null": stat(null.to_numpy(float)),
        "pooled_slope": float(case.dot_product.sum() / case.power.sum()),
        "pooled_orthogonal_slope": float(case.cross.sum() / case.power.sum()),
        "n_rows": int(len(frame)),
        "n_cases": int(len(case)),
    }


def position_shift(frame: pd.DataFrame, g: float) -> dict:
    detected_y1 = (frame.g1_detected_plus - frame.g1_detected_minus) / (2.0 * g)
    detected_y2 = (frame.g2_detected_plus - frame.g2_detected_minus) / (2.0 * g)
    truth_y1 = (frame.g1_truthpos_plus - frame.g1_truthpos_minus) / (2.0 * g)
    truth_y2 = (frame.g2_truthpos_plus - frame.g2_truthpos_minus) / (2.0 * g)
    power = frame.prediction_u1**2 + frame.prediction_u2**2
    delta_dot = (
        frame.prediction_u1 * (truth_y1 - detected_y1)
        + frame.prediction_u2 * (truth_y2 - detected_y2)
    )
    work = pd.DataFrame({"case": frame.case, "power": power, "delta_dot": delta_dot})
    case = work.groupby("case", sort=True).sum()
    return {
        "truthpos_minus_detected_slope": stat((case.delta_dot / case.power).to_numpy(float)),
    }


def summarize(frame: pd.DataFrame, g: float, development_max: int) -> dict:
    development = frame.case <= development_max
    output = {}
    for name, mask in (
        ("development", development),
        ("validation", ~development),
        ("all", np.ones(len(frame), dtype=bool)),
    ):
        subset = frame.loc[mask]
        output[name] = {
            "detected": vector_fit(subset, "detected", g),
            "truth_position": vector_fit(subset, "truthpos", g),
            **position_shift(subset, g),
        }
    return output


def load_prediction(path: str) -> pd.DataFrame:
    frame = pd.read_feather(path)
    if "index" in frame:
        if "input_index" not in frame:
            frame["input_index"] = frame["index"]
        else:
            missing = frame.input_index.isna()
            if np.any(missing & frame["index"].isna()):
                raise RuntimeError(f"missing input_index has no fallback in {path}")
            frame.loc[missing, "input_index"] = frame.loc[missing, "index"]
    required = KEY + ["prediction_u1", "prediction_u2"]
    if frame.duplicated(KEY).any():
        raise RuntimeError(f"duplicate prediction keys in {path}")
    return frame[required]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurement-dir", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--case-min", type=int, default=200)
    parser.add_argument("--case-max", type=int, default=299)
    parser.add_argument("--development-max", type=int, default=249)
    parser.add_argument("--g", type=float, default=0.05)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if os.path.exists(args.output):
        raise FileExistsError(f"refusing to overwrite {args.output}")

    paths = sorted(glob.glob(os.path.join(args.measurement_dir, "case*.feather")))
    expected = args.case_max - args.case_min + 1
    if len(paths) != expected:
        raise RuntimeError(f"expected {expected} case files, found {len(paths)}")
    measured = pd.concat([pd.read_feather(path) for path in paths], ignore_index=True)
    measured = measured[measured.case.between(args.case_min, args.case_max)]
    if measured.duplicated(KEY).any():
        raise RuntimeError("duplicate measurement keys")
    shape_columns = [column for column in measured if column.startswith(("g1_", "g2_"))]
    finite = np.isfinite(measured[shape_columns].to_numpy(float)).all(axis=1)
    measured = measured.loc[finite].copy()

    baseline = measured.merge(load_prediction(args.baseline), on=KEY, validate="one_to_one")
    candidate = measured.merge(load_prediction(args.candidate), on=KEY, validate="one_to_one")
    common_keys = baseline[KEY].merge(candidate[KEY], on=KEY, validate="one_to_one")
    baseline = common_keys.merge(baseline, on=KEY, validate="one_to_one")
    candidate = common_keys.merge(candidate, on=KEY, validate="one_to_one")
    if len(common_keys) < 0.995 * len(measured):
        raise RuntimeError(
            f"prediction coverage below 99.5%: {len(common_keys)}/{len(measured)}"
        )

    baseline_summary = summarize(baseline, args.g, args.development_max)
    candidate_summary = summarize(candidate, args.g, args.development_max)
    baseline_validation = baseline_summary["validation"]
    position_shift_value = baseline_validation[
        "truthpos_minus_detected_slope"
    ]["mean"]
    baseline_gap_value = baseline_validation["detected"]["slope_minus_one"]["mean"]
    payload = {
        "design": (
            "same independent-direction pixels and anchors; deterministic common ngmix "
            "initialization; detected centroid versus fixed input truth position"
        ),
        "case_window": [args.case_min, args.case_max],
        "development_max": args.development_max,
        "n_measured_finite": int(len(measured)),
        "n_common_predictions": int(len(common_keys)),
        "common_prediction_fraction": float(len(common_keys) / len(measured)),
        "baseline": baseline_summary,
        "candidate": candidate_summary,
        "interpretation_gate": {
            "centroid_motion_explains_baseline_vector_gap": bool(
                abs(position_shift_value) >= 0.5 * abs(baseline_gap_value)
            ),
            "baseline_validation_position_shift": float(position_shift_value),
            "baseline_validation_detected_slope_minus_one": float(baseline_gap_value),
        },
        "constgold_opened": False,
    }
    with open(args.output, "x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    print("ANCHOR_INDEPENDENT_FIXED_POSITION_ANALYSIS_DONE", flush=True)


if __name__ == "__main__":
    main()
