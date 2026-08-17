"""Compare all-active and training-density independent-neighbour anchor arms."""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from analyze_anchorblend_independent_response import stat, summary, vector_fit


KEY = ["case", "input_index"]
TRACE_COLUMNS = [
    "prediction", "label_sum", "projected_prediction", "desired_gap",
    "projected_gap", "null_sum",
]


def case_vector_slopes(frame: pd.DataFrame, g: float) -> pd.DataFrame:
    work = frame.copy()
    work["y1"] = (work.measured_e1_plus - work.measured_e1_minus) / (2 * g)
    work["y2"] = (work.measured_e2_plus - work.measured_e2_minus) / (2 * g)
    work["power"] = work.prediction_u1**2 + work.prediction_u2**2
    work["dot"] = work.prediction_u1 * work.y1 + work.prediction_u2 * work.y2
    work["cross"] = -work.prediction_u2 * work.y1 + work.prediction_u1 * work.y2
    case = work.groupby("case", sort=True)[["power", "dot", "cross"]].sum()
    case["slope"] = case["dot"] / case["power"]
    case["orthogonal"] = case["cross"] / case["power"]
    return case


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-active", required=True)
    parser.add_argument("--half-active", required=True)
    parser.add_argument("--g", type=float, default=0.05)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if os.path.exists(args.output):
        raise FileExistsError(f"refusing to overwrite {args.output}")
    all_active = pd.read_feather(args.all_active)
    half_active = pd.read_feather(args.half_active)
    if all_active.duplicated(KEY).any() or half_active.duplicated(KEY).any():
        raise RuntimeError("duplicate anchor keys")
    common = all_active.merge(
        half_active, on=KEY, how="inner", validate="one_to_one",
        suffixes=("_all", "_half"),
    )
    all_common = pd.DataFrame({
        column: common[f"{column}_all"] for column in all_active.columns if column not in KEY
    })
    all_common[KEY] = common[KEY]
    half_common = pd.DataFrame({
        column: common[f"{column}_half"] for column in half_active.columns if column not in KEY
    })
    half_common[KEY] = common[KEY]
    all_case = case_vector_slopes(all_common, args.g)
    half_case = case_vector_slopes(half_common, args.g)
    paired = all_case.join(half_case, lsuffix="_all", rsuffix="_half", how="inner")
    payload = {
        "design": "same latent anchor scenes/noise and independent directions; all local sources active versus upper input-index half active",
        "all_active": {
            "trace": summary(all_active, TRACE_COLUMNS),
            "vector_fit": vector_fit(all_active, args.g),
        },
        "half_active": {
            "trace": summary(half_active, TRACE_COLUMNS),
            "vector_fit": vector_fit(half_active, args.g),
        },
        "exact_common": {
            "n_rows": int(len(common)),
            "fraction_all": float(len(common) / len(all_active)),
            "fraction_half": float(len(common) / len(half_active)),
            "all_vector_fit": vector_fit(all_common, args.g),
            "half_vector_fit": vector_fit(half_common, args.g),
            "paired_all_minus_half_slope": stat(
                (paired.slope_all - paired.slope_half).to_numpy(float)
            ),
            "paired_all_minus_half_orthogonal": stat(
                (paired.orthogonal_all - paired.orthogonal_half).to_numpy(float)
            ),
        },
        "interpretation": {
            "primary_endpoint": "case-level measured-on-predicted vector slope; one is closure",
            "causal_test": "whether reducing simultaneously active neighbour density moves slope toward one",
            "trace_warning": "desired/projected trace gaps are unbiased but have large direction cross-term case variance",
        },
        "constgold_opened": False,
    }
    with open(args.output, "x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    print("ANCHOR_INDEPENDENT_ACTIVE_DENSITY_DONE", flush=True)


if __name__ == "__main__":
    main()
