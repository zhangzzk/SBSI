"""Decompose the fixed weighted-emulator correction into mean and structural parts.

The correction is defined against the deployed V2.2 source model on its own fitting
population (half-shear cases 40--199):

    delta(x) = f_weighted(x) - f_v22(x)
             = mean(delta) + [delta(x) - mean(delta)].

No response truth from an endpoint is used to choose the centering constant.  Existing
per-primary scored tables are replayed analytically, so no model is retrained or rescored.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd


KEY = ["case", "input_index"]
ARMS = ("baseline", "constant_only", "zero_mean_structural", "full_candidate")


def stat(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(values.mean()),
        "case_sem": float(values.std(ddof=1) / np.sqrt(len(values))),
        "n_cases": int(len(values)),
    }


def normalize_key(frame: pd.DataFrame) -> pd.DataFrame:
    if "index" not in frame:
        return frame
    result = frame.copy()
    if "input_index" not in result:
        result["input_index"] = result["index"]
    else:
        missing = result.input_index.isna()
        result.loc[missing, "input_index"] = result.loc[missing, "index"]
    return result.drop(columns="index")


def strict_merge(left: pd.DataFrame, right: pd.DataFrame, suffixes: tuple[str, str]) -> pd.DataFrame:
    for name, frame in (("left", left), ("right", right)):
        if frame.duplicated(KEY).any():
            raise RuntimeError(f"{name} endpoint table has duplicate keys")
    joined = left.merge(right, on=KEY, how="inner", suffixes=suffixes, validate="one_to_one")
    if len(joined) != len(left) or len(joined) != len(right):
        raise RuntimeError(
            f"endpoint key mismatch: left={len(left)} right={len(right)} common={len(joined)}"
        )
    return joined


def check_equal(frame: pd.DataFrame, left: str, right: str, tolerance: float = 1e-10) -> float:
    delta = np.nanmax(np.abs(frame[left].to_numpy(float) - frame[right].to_numpy(float)))
    if delta > tolerance:
        raise RuntimeError(f"identity drift {left} vs {right}: max={delta:.3e}")
    return float(delta)


def vector_fit(case: np.ndarray, y1: np.ndarray, y2: np.ndarray,
               p1: np.ndarray, p2: np.ndarray) -> dict:
    work = pd.DataFrame({
        "case": np.asarray(case, int),
        "power": p1 * p1 + p2 * p2,
        "dot": p1 * y1 + p2 * y2,
        "cross": -p2 * y1 + p1 * y2,
    }).groupby("case", sort=True).sum()
    if (work.power <= 0).any():
        raise RuntimeError("non-positive model-vector power")
    slope = (work["dot"] / work["power"]).to_numpy(float)
    null = (work["cross"] / work["power"]).to_numpy(float)
    return {
        "slope_measured_on_predicted": stat(slope),
        "slope_minus_one": stat(slope - 1.0),
        "orthogonal_slope_null": stat(null),
        "pooled_slope": float(work["dot"].sum() / work["power"].sum()),
        "pooled_orthogonal_slope": float(work["cross"].sum() / work["power"].sum()),
    }


def split_vector(case: np.ndarray, y1: np.ndarray, y2: np.ndarray,
                 vectors: dict[str, tuple[np.ndarray, np.ndarray]], split: int) -> dict:
    groups = {
        "all": np.ones(len(case), dtype=bool),
        "development": case <= split,
        "validation": case > split,
    }
    return {
        group: {
            arm: vector_fit(case[mask], y1[mask], y2[mask], p1[mask], p2[mask])
            for arm, (p1, p2) in vectors.items()
        }
        for group, mask in groups.items()
    }


def halfshear(baseline_path: str, candidate_path: str, mean_delta: float) -> dict:
    columns = [
        "case", "input_index", "label", "null", "cos_shear", "sin_shear",
        "prediction_cos", "prediction_sin", "n_pairs",
    ]
    baseline = pd.read_feather(baseline_path, columns=columns)
    candidate = pd.read_feather(candidate_path, columns=columns)
    frame = strict_merge(baseline, candidate, ("_base", "_candidate"))
    checks = {
        name: check_equal(frame, f"{name}_base", f"{name}_candidate")
        for name in ("label", "null", "cos_shear", "sin_shear", "n_pairs")
    }
    direction_power = frame.cos_shear_base.to_numpy(float) ** 2 + frame.sin_shear_base.to_numpy(float) ** 2
    if (direction_power <= 1e-8).any():
        raise RuntimeError("halfshear direction sum vanished")
    label = frame.label_base.to_numpy(float)
    null = frame.null_base.to_numpy(float)
    c1 = frame.cos_shear_base.to_numpy(float)
    c2 = frame.sin_shear_base.to_numpy(float)
    y1 = (label * c1 - null * c2) / direction_power
    y2 = (label * c2 + null * c1) / direction_power
    b1 = frame.prediction_cos_base.to_numpy(float)
    b2 = frame.prediction_sin_base.to_numpy(float)
    f1 = frame.prediction_cos_candidate.to_numpy(float)
    f2 = frame.prediction_sin_candidate.to_numpy(float)
    vectors = {
        "baseline": (b1, b2),
        "constant_only": (b1 + mean_delta * c1, b2 + mean_delta * c2),
        "zero_mean_structural": (f1 - mean_delta * c1, f2 - mean_delta * c2),
        "full_candidate": (f1, f2),
    }
    return {
        "rows": int(len(frame)), "cases": int(frame.case.nunique()), "identity_checks": checks,
        "fits": split_vector(frame.case.to_numpy(int), y1, y2, vectors, split=19),
    }


def independent(baseline_path: str, candidate_path: str, mean_delta: float, g: float = 0.05) -> dict:
    columns = [
        "case", "input_index", "measured_e1_plus", "measured_e2_plus",
        "measured_e1_minus", "measured_e2_minus", "n_pairs", "sum_u1", "sum_u2",
        "prediction_u1", "prediction_u2",
    ]
    baseline = normalize_key(pd.read_feather(baseline_path, columns=[*columns, "index"]))
    candidate = normalize_key(pd.read_feather(candidate_path, columns=[*columns, "index"]))
    frame = strict_merge(baseline, candidate, ("_base", "_candidate"))
    checks = {
        name: check_equal(frame, f"{name}_base", f"{name}_candidate")
        for name in (
            "measured_e1_plus", "measured_e2_plus", "measured_e1_minus", "measured_e2_minus",
            "n_pairs", "sum_u1", "sum_u2",
        )
    }
    y1 = (frame.measured_e1_plus_base.to_numpy(float) - frame.measured_e1_minus_base.to_numpy(float)) / (2 * g)
    y2 = (frame.measured_e2_plus_base.to_numpy(float) - frame.measured_e2_minus_base.to_numpy(float)) / (2 * g)
    u1 = frame.sum_u1_base.to_numpy(float)
    u2 = frame.sum_u2_base.to_numpy(float)
    b1 = frame.prediction_u1_base.to_numpy(float)
    b2 = frame.prediction_u2_base.to_numpy(float)
    f1 = frame.prediction_u1_candidate.to_numpy(float)
    f2 = frame.prediction_u2_candidate.to_numpy(float)
    vectors = {
        "baseline": (b1, b2),
        "constant_only": (b1 + mean_delta * u1, b2 + mean_delta * u2),
        "zero_mean_structural": (f1 - mean_delta * u1, f2 - mean_delta * u2),
        "full_candidate": (f1, f2),
    }
    return {
        "rows": int(len(frame)), "cases": int(frame.case.nunique()), "identity_checks": checks,
        "fits": split_vector(frame.case.to_numpy(int), y1, y2, vectors, split=249),
    }


def coherent(parts_root: str, tag: str, case_min: int, case_max: int, mean_delta: float) -> dict:
    paths = [os.path.join(parts_root, tag, f"case{case}.feather") for case in range(case_min, case_max + 1)]
    missing = [path for path in paths if not os.path.isfile(path)]
    if missing:
        raise RuntimeError(f"coherent block missing {len(missing)} parts")
    frame = pd.concat([pd.read_feather(path) for path in paths], ignore_index=True)
    if frame.duplicated(KEY).any() or frame.case.nunique() != case_max - case_min + 1:
        raise RuntimeError("invalid coherent key/case coverage")
    if set(frame.model_tag.unique()) != {tag}:
        raise RuntimeError(f"coherent model tag drift: {frame.model_tag.unique()}")
    truth = frame.R_blend_truth.to_numpy(float)
    baseline = frame.prediction_baseline.to_numpy(float)
    full = frame.prediction_model.to_numpy(float)
    count = frame.n_pairs.to_numpy(float)
    predictions = {
        "baseline": baseline,
        "constant_only": baseline + mean_delta * count,
        "zero_mean_structural": full - mean_delta * count,
        "full_candidate": full,
    }
    case = frame.case.to_numpy(int)
    payload = {}
    for arm, prediction in predictions.items():
        work = pd.DataFrame({"case": case, "gap": prediction - truth, "correction": prediction - baseline})
        by_case = work.groupby("case", sort=True).mean()
        payload[arm] = {"gap": stat(by_case.gap), "correction_from_baseline": stat(by_case.correction)}
    replay = np.max(np.abs((predictions["constant_only"] - baseline) +
                           (predictions["zero_mean_structural"] - baseline) - (full - baseline)))
    if replay > 1e-12:
        raise RuntimeError(f"coherent correction decomposition fails to replay: {replay:.3e}")
    return {
        "rows": int(len(frame)), "cases": int(frame.case.nunique()),
        "case_window": [case_min, case_max], "replay_max_abs": float(replay), "arms": payload,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mean-json", required=True)
    ap.add_argument("--halfshear-baseline", required=True)
    ap.add_argument("--halfshear-candidate", required=True)
    ap.add_argument("--independent-baseline", required=True)
    ap.add_argument("--independent-candidate", required=True)
    ap.add_argument("--coherent-old-root", required=True)
    ap.add_argument("--coherent-fresh-root", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    if os.path.exists(args.output):
        raise FileExistsError(f"refusing to overwrite {args.output}")
    with open(args.mean_json, encoding="utf-8") as handle:
        mean_info = json.load(handle)
    if mean_info["source_tag"] != "lsst_r_extnbr_v22" or mean_info["candidate_tag"] != args.tag:
        raise RuntimeError("mean-model provenance does not match requested decomposition")
    if mean_info["minimum_training_case"] != 40:
        raise RuntimeError("centering mean must come from the exact cases 40--199 fitting population")
    mean_delta = float(mean_info["global"]["candidate_minus_source"])
    payload = {
        "design": "fixed candidate correction = fitting-row mean shift + zero-mean structural remainder; endpoints are replayed without retraining",
        "source_tag": mean_info["source_tag"], "candidate_tag": args.tag,
        "centering_population": "V2.2 half-shear selected pair rows, cases 40--199",
        "mean_pair_correction": mean_delta,
        "candidate_minus_label_global": float(mean_info["global"]["candidate_minus_label"]),
        "halfshear_c0_39": halfshear(args.halfshear_baseline, args.halfshear_candidate, mean_delta),
        "independent_anchor_c200_299": independent(args.independent_baseline, args.independent_candidate, mean_delta),
        "coherent_anchor_c200_399": coherent(args.coherent_old_root, args.tag, 200, 399, mean_delta),
        "coherent_anchor_c400_599": coherent(args.coherent_fresh_root, args.tag, 400, 599, mean_delta),
        "constgold_opened_for_decomposition": False,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    print("WEIGHTED_CORRECTION_DECOMPOSITION_DONE")


if __name__ == "__main__":
    main()
