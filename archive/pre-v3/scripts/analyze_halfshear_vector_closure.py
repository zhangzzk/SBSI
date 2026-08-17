"""Low-variance V2.2 vector closure on existing random half-shear primaries.

For a primary with measured response vector y and neighbour unit directions
u_i, the stored aggregate coordinates obey

    label = y dot sum(u_i)
    null  = cross(y, sum(u_i)).

They therefore reconstruct y exactly away from a vanishing direction sum.
We regress this measured vector on the frozen model vector sum_i R_i u_i,
using rendered cases as the independent uncertainty units.  This is the same
endpoint used for the independent-neighbour anchor scenes.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd


def stat(values) -> dict:
    values = np.asarray(values, float)
    return {
        "mean": float(values.mean()),
        "case_sem": float(values.std(ddof=1) / np.sqrt(len(values))),
        "n_cases": int(len(values)),
    }


def fit(frame: pd.DataFrame) -> dict:
    case = frame.groupby("case", sort=True)[["power", "dot", "cross"]].sum()
    slope = case["dot"] / case["power"]
    null = case["cross"] / case["power"]
    return {
        "slope_measured_on_predicted": stat(slope),
        "slope_minus_one": stat(slope - 1.0),
        "orthogonal_slope_null": stat(null),
        "pooled_slope": float(case["dot"].sum() / case["power"].sum()),
        "pooled_orthogonal_slope": float(case["cross"].sum() / case["power"].sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", required=True)
    parser.add_argument("--development-max", type=int, default=19)
    parser.add_argument("--min-direction-power", type=float, default=1e-8)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if os.path.exists(args.output):
        raise FileExistsError(f"refusing to overwrite {args.output}")
    columns = [
        "case", "input_index", "label", "null", "cos_shear", "sin_shear",
        "prediction_cos", "prediction_sin", "n_pairs",
    ]
    frame = pd.read_feather(args.table, columns=columns)
    direction_power = frame.cos_shear**2 + frame.sin_shear**2
    finite = np.isfinite(frame[columns[2:]].to_numpy(float)).all(axis=1)
    keep = finite & (direction_power > args.min_direction_power)
    work = frame.loc[keep].copy()
    denominator = direction_power.loc[keep].to_numpy(float)
    # label = y1*S1 + y2*S2; null = -y1*S2 + y2*S1.
    work["y1"] = (
        work.label * work.cos_shear - work.null * work.sin_shear
    ) / denominator
    work["y2"] = (
        work.label * work.sin_shear + work.null * work.cos_shear
    ) / denominator
    work["power"] = work.prediction_cos**2 + work.prediction_sin**2
    work["dot"] = work.prediction_cos * work.y1 + work.prediction_sin * work.y2
    work["cross"] = -work.prediction_sin * work.y1 + work.prediction_cos * work.y2
    work = work.loc[work.power > 0].copy()
    development = work.case <= args.development_max
    payload = {
        "design": "existing c0--39 random half-shear scenes; reconstruct measured 2-vector from stored projected label/null and fit against sum_i(R_i u_i)",
        "all": fit(work),
        "development_c0_19": fit(work.loc[development]),
        "validation_c20_39": fit(work.loc[~development]),
        "n_input_rows": int(len(frame)),
        "n_used_rows": int(len(work)),
        "excluded_small_direction_sum": int((finite & ~keep).sum()),
        "minimum_direction_power": float(args.min_direction_power),
        "constgold_opened": False,
        "anchor_truth_opened": False,
    }
    with open(args.output, "x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    print("HALFSHEAR_VECTOR_CLOSURE_DONE", flush=True)


if __name__ == "__main__":
    main()
