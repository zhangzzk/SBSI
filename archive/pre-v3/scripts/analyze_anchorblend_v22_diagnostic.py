"""Compare current V2.2 BlendEMU with direct coherent-neighbour anchor truth.

The anchor simulations are independent of constgold.  This script reports raw
prediction-minus-truth response; it does not fit or apply a correction.  V2.1
is rescored in the same files and compared with the original stored V2.1 result
as a reproducibility control.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd


KEYS = ["case", "input_index"]


def read_disjoint(paths: list[str]) -> pd.DataFrame:
    parts = [pd.read_feather(path) for path in paths]
    out = pd.concat(parts, ignore_index=True)
    if out.duplicated(KEYS).any():
        raise RuntimeError("duplicate keys across response inputs")
    return out


def summary(frame: pd.DataFrame, prediction: str) -> dict:
    cols = ["R_blend_truth", prediction]
    clean = frame.loc[np.isfinite(frame[cols].to_numpy(float)).all(axis=1)]
    by_case = clean.groupby("case", sort=True)[cols].mean()
    truth = by_case["R_blend_truth"].to_numpy(float)
    pred = by_case[prediction].to_numpy(float)
    residual = pred - truth
    return {
        "n_objects": int(len(clean)),
        "n_cases": int(len(by_case)),
        "truth_case_mean": float(truth.mean()),
        "prediction_case_mean": float(pred.mean()),
        "prediction_minus_truth": float(residual.mean()),
        "prediction_minus_truth_case_sem": (
            float(residual.std(ddof=1) / np.sqrt(len(residual))) if len(residual) > 1 else None
        ),
        "prediction_over_truth_minus_one": float(pred.mean() / truth.mean() - 1.0),
        "truth_over_prediction_scale": float(truth.mean() / pred.mean()),
    }


def split_summaries(frame: pd.DataFrame, prediction: str, split_case: int) -> dict:
    return {
        "development_0_to_split": summary(frame.loc[frame["case"] < split_case], prediction),
        "fresh_split_and_above": summary(frame.loc[frame["case"] >= split_case], prediction),
        "all": summary(frame, prediction),
    }


def population_summaries(frame: pd.DataFrame, prediction: str, split_case: int) -> dict:
    native = split_summaries(frame, prediction, split_case)
    if not {"r_input_p_plus", "Re_input_p_plus"}.issubset(frame.columns):
        raise KeyError("anchor response lacks primary truth columns for the V2.2-box intersection")
    in_box = (
        (frame["r_input_p_plus"].to_numpy(float) < 25.8)
        & (frame["Re_input_p_plus"].to_numpy(float) > 0.5)
    )
    return {
        "native_v21_anchor_population": native,
        "intersection_with_v22_primary_box": {
            "fraction_of_native_objects": float(in_box.mean()),
            **split_summaries(frame.loc[in_box], prediction, split_case),
        },
    }


def control_against_original(new: pd.DataFrame, old: pd.DataFrame, column: str) -> dict:
    joined = new[KEYS + ["R_blend_truth", column]].merge(
        old[KEYS + ["R_blend_truth", column]], on=KEYS, how="inner",
        validate="one_to_one", suffixes=("_new", "_old"),
    )
    if len(joined) != len(new) or len(joined) != len(old):
        raise RuntimeError(
            f"new/original key sets differ: new={len(new)}, old={len(old)}, shared={len(joined)}"
        )
    result = {"n": int(len(joined))}
    for name in ("R_blend_truth", column):
        diff = (
            joined[f"{name}_new"].to_numpy(float)
            - joined[f"{name}_old"].to_numpy(float)
        )
        result[name] = {
            "max_abs_difference": float(np.max(np.abs(diff))),
            "mean_difference": float(diff.mean()),
            "exact_equal_fraction": float(np.mean(diff == 0.0)),
        }
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--new", nargs="+", required=True)
    ap.add_argument("--original", nargs="+", required=True)
    ap.add_argument("--v21-tag", default="lsst_r_extnbr_v21")
    ap.add_argument("--v22-tag", default="lsst_r_extnbr_v22")
    ap.add_argument("--split-case", type=int, default=100)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    if os.path.exists(args.output):
        raise FileExistsError(f"refusing to overwrite {args.output}")

    new = read_disjoint(args.new)
    old = read_disjoint(args.original)
    v21 = f"R_blend_{args.v21_tag}"
    v22 = f"R_blend_{args.v22_tag}"
    missing = [name for name in ["R_blend_truth", v21, v22] if name not in new]
    if missing:
        raise KeyError(f"new response files missing {missing}")
    if v21 not in old:
        raise KeyError(f"original response files missing {v21}")

    result = {
        "status": "diagnostic only; no correction fitted or applied",
        "independence_caveat": (
            "The anchor truth is independent of constgold, but this instrument was previously "
            "used during V2.1 development. Treat V2.2 scoring as mechanism evidence, not a new "
            "blinded certification."
        ),
        "split_case": args.split_case,
        "v21_reproduction_control": control_against_original(new, old, v21),
        "models": {
            args.v21_tag: population_summaries(new, v21, args.split_case),
            args.v22_tag: population_summaries(new, v22, args.split_case),
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    for tag, rows in result["models"].items():
        all_row = rows["native_v21_anchor_population"]["all"]
        print(
            f"{tag}: truth={all_row['truth_case_mean']:+.6f}, "
            f"pred={all_row['prediction_case_mean']:+.6f}, "
            f"pred-truth={all_row['prediction_minus_truth']:+.6f} +/- "
            f"{all_row['prediction_minus_truth_case_sem']:.6f}",
            flush=True,
        )
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
