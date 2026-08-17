"""Paired g=0.05 versus g=0.2 vector response on exact common V2.2 scenes."""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from analyze_halfshear_vector_closure import stat


KEY = ["case", "input_index"]


def prepare(path: str, suffix: str) -> pd.DataFrame:
    columns = [
        *KEY, "label", "null", "n_pairs", "cos_shear", "sin_shear",
        "prediction_cos", "prediction_sin",
    ]
    frame = pd.read_feather(path, columns=columns)
    direction_power = frame.cos_shear**2 + frame.sin_shear**2
    frame[f"y1_{suffix}"] = (
        frame.label * frame.cos_shear - frame.null * frame.sin_shear
    ) / direction_power
    frame[f"y2_{suffix}"] = (
        frame.label * frame.sin_shear + frame.null * frame.cos_shear
    ) / direction_power
    return frame[[
        *KEY, "n_pairs", "prediction_cos", "prediction_sin",
        f"y1_{suffix}", f"y2_{suffix}",
    ]]


def paired_fit(frame: pd.DataFrame) -> dict:
    frame = frame.copy()
    frame["power"] = frame.prediction_cos_20**2 + frame.prediction_sin_20**2
    for suffix in ("05", "20"):
        frame[f"dot_{suffix}"] = (
            frame.prediction_cos_20 * frame[f"y1_{suffix}"]
            + frame.prediction_sin_20 * frame[f"y2_{suffix}"]
        )
        frame[f"cross_{suffix}"] = (
            -frame.prediction_sin_20 * frame[f"y1_{suffix}"]
            + frame.prediction_cos_20 * frame[f"y2_{suffix}"]
        )
    case = frame.groupby("case", sort=True)[
        ["power", "dot_05", "dot_20", "cross_05", "cross_20"]
    ].sum()
    slope05 = case.dot_05 / case.power
    slope20 = case.dot_20 / case.power
    null05 = case.cross_05 / case.power
    null20 = case.cross_20 / case.power
    return {
        "slope_g005": stat(slope05), "slope_g020": stat(slope20),
        "slope_g005_minus_g020": stat(slope05 - slope20),
        "orthogonal_g005": stat(null05), "orthogonal_g020": stat(null20),
        "orthogonal_g005_minus_g020": stat(null05 - null20),
        "n_rows": int(len(frame)), "n_cases": int(len(case)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g005", required=True)
    parser.add_argument("--g020", required=True)
    parser.add_argument("--development-max", type=int, default=19)
    parser.add_argument("--model-tolerance", type=float, default=2e-7)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if os.path.exists(args.output):
        raise FileExistsError(f"refusing to overwrite {args.output}")
    g05 = prepare(args.g005, "05")
    g20 = prepare(args.g020, "20")
    common = g05.merge(g20, on=KEY, how="inner", validate="one_to_one", suffixes=("_05", "_20"))
    model_difference = np.hypot(
        common.prediction_cos_05 - common.prediction_cos_20,
        common.prediction_sin_05 - common.prediction_sin_20,
    )
    exact = (common.n_pairs_05 == common.n_pairs_20) & (model_difference <= args.model_tolerance)
    matched = common.loc[exact].copy()
    development = matched.case <= args.development_max
    payload = {
        "design": "exact common primary keys with identical supported-pair count and aggregate frozen V2.2 response vector; paired case slopes use the g=0.2 model vector",
        "all": paired_fit(matched),
        "development_c0_19": paired_fit(matched.loc[development]),
        "validation_c20_39": paired_fit(matched.loc[~development]),
        "n_g005_rows": int(len(g05)), "n_g020_rows": int(len(g20)),
        "n_common_primary_keys": int(len(common)),
        "n_exact_model_vector_keys": int(len(matched)),
        "exact_fraction_of_common": float(len(matched) / len(common)),
        "model_tolerance": float(args.model_tolerance),
        "constgold_opened": False, "anchor_truth_opened": False,
    }
    with open(args.output, "x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    print("HALFSHEAR_VECTOR_AMPLITUDE_COMPARISON_DONE", flush=True)


if __name__ == "__main__":
    main()
