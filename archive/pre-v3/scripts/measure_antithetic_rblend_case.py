"""Compare forward and central random-direction R_blend labels for one case.

The existing half-shear construction measures primaries while all secondary
galaxies receive independent random spin-2 shear directions.  This script
uses exact common pairs from 0 -> +0.02 and -0.02 -> +0.02, so finite-
difference convention and common-detection population are the only intended
differences.  It never reads constgold or coherent-anchor truth.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd


BE = "/home/z/Zekang.Zhang/blendemu"
if BE not in sys.path:
    sys.path.insert(0, BE)

from blendemu import data_utils  # noqa: E402
from blendemu.inference import BlendingPredictor  # noqa: E402
from blendemu.response import retrieve_response  # noqa: E402


COND = dict(pixel_size=0.2, zero_point=30.0, psf_fwhm=0.73,
            moffat_beta=2.224, pixel_rms=0.312)
CUTS = [[13, 29], [18, 25.8], [0.0, 10.0], [0.5, 1.5], [0, 10]]
KEY = ["input_index", "_ra_s", "_dec_s"]
FEATURES = [
    "Re_input_p", "Re_input_s", "r_input_p", "r_input_s",
    "sersic_n_input_p", "sersic_n_input_s", "distance",
]
SEPARATION_EDGES = np.asarray([0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0])


def select(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    frame = data_utils.source_select_reg(frame, cuts=CUTS).copy()
    frame = frame.loc[np.isfinite(frame[["delta_et1", "delta_et2"]]).all(axis=1)].copy()
    frame["_ra_s"] = np.round(frame.RA_input_s.to_numpy(float), 9)
    frame["_dec_s"] = np.round(frame.DEC_input_s.to_numpy(float), 9)
    duplicate = frame.duplicated(KEY, keep=False)
    if duplicate.any():
        check = frame.loc[duplicate].groupby(KEY, sort=False)[
            FEATURES + ["delta_et1", "delta_et2"]
        ].nunique(dropna=False)
        if (check.to_numpy() > 1).any():
            raise RuntimeError(f"{label}: non-identical duplicate pair keys")
        frame = frame.drop_duplicates(KEY)
    return frame


def mean(values) -> float:
    return float(np.asarray(values, float).mean())


def summarize_pairs(frame: pd.DataFrame) -> dict:
    columns = [
        "R_forward", "R_central", "R_model", "R_model_central",
        "N_forward", "N_central", "distance_central_minus_forward",
    ]
    result = {column: mean(frame[column]) for column in columns}
    result.update({
        "central_minus_forward": mean(frame.R_central - frame.R_forward),
        "model_minus_central": mean(frame.R_model_central - frame.R_central),
        "model_minus_forward": mean(frame.R_model - frame.R_forward),
        "model_central_minus_forward": mean(frame.R_model_central - frame.R_model),
        "n_pairs": int(len(frame)),
    })
    return result


def summarize_primary(frame: pd.DataFrame) -> dict:
    primary = frame.groupby("input_index", sort=False)[
        ["R_forward", "R_central", "R_model", "R_model_central", "N_forward", "N_central"]
    ].sum()
    result = {f"sum_{column}": mean(primary[column]) for column in primary}
    result.update({
        "sum_central_minus_forward": mean(primary.R_central - primary.R_forward),
        "sum_model_minus_central": mean(primary.R_model_central - primary.R_central),
        "sum_model_minus_forward": mean(primary.R_model - primary.R_forward),
        "sum_model_central_minus_forward": mean(primary.R_model_central - primary.R_model),
        "mean_pairs_per_primary": float(len(frame) / len(primary)),
        "n_primaries": int(len(primary)),
    })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=int, required=True)
    parser.add_argument("--base", default="/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876")
    parser.add_argument("--g", type=float, default=0.02)
    parser.add_argument("--tag", default="lsst_r_extnbr_v22")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    output = os.path.join(args.output_dir, f"case{args.case:03d}.json")
    if os.path.exists(output):
        raise FileExistsError(f"refusing to overwrite {output}")

    forward_raw = retrieve_response(
        args.case, r_max=10.0, r_min=0.0, k=20, real="real0",
        data_path=args.base, shear_cases=["0.0", str(args.g)],
    )
    central_raw = retrieve_response(
        args.case, r_max=10.0, r_min=0.0, k=20, real="real0",
        data_path=args.base, shear_cases=[str(-args.g), str(args.g)],
    )
    forward = select(forward_raw, "forward")
    central = select(central_raw, "central")
    keep = KEY + FEATURES + ["delta_et1", "delta_et2"]
    common = forward[keep].merge(
        central[KEY + ["distance", "delta_et1", "delta_et2"]],
        on=KEY, how="inner", validate="one_to_one", suffixes=("_forward", "_central"),
    )
    if len(common) < 1000:
        raise RuntimeError(f"case {args.case}: only {len(common)} exact common pairs")
    distance_change = (
        common.distance_central.to_numpy(float) - common.distance_forward.to_numpy(float)
    )
    distance_error = np.max(np.abs(distance_change))
    common = common.rename(columns={"distance_forward": "distance"})
    common["distance_central_minus_forward"] = distance_change
    common["R_forward"] = common.delta_et1_forward / args.g
    common["R_central"] = common.delta_et1_central / (2.0 * args.g)
    common["N_forward"] = common.delta_et2_forward / args.g
    common["N_central"] = common.delta_et2_central / (2.0 * args.g)
    predictor = BlendingPredictor.load(
        os.path.join(BE, "models"), tag=args.tag, conditions=COND, device="cpu",
    )
    common["R_model"] = predictor.predict_on_pairs(
        common[FEATURES], task="response", warn_extrapolation=False,
    ).response.to_numpy(float)
    central_features = common[FEATURES].copy()
    central_features["distance"] = common.distance_central.to_numpy(float)
    common["R_model_central"] = predictor.predict_on_pairs(
        central_features, task="response", warn_extrapolation=False,
    ).response.to_numpy(float)

    bins = []
    for index, (lo, hi) in enumerate(zip(SEPARATION_EDGES[:-1], SEPARATION_EDGES[1:])):
        mask = (common.distance >= lo) & (common.distance < hi)
        if mask.any():
            bins.append({"index": index, "lo": float(lo), "hi": float(hi),
                         **summarize_pairs(common.loc[mask])})
    payload = {
        "case": int(args.case),
        "g": float(args.g),
        "design": "exact common V2.2-domain pairs: 0->+g versus -g->+g random secondary directions",
        "n_selected_forward": int(len(forward)),
        "n_selected_central": int(len(central)),
        "common_fraction_forward": float(len(common) / len(forward)),
        "common_fraction_central": float(len(common) / len(central)),
        "distance_change_max_abs": float(distance_error),
        "distance_change_mean": mean(distance_change),
        "pair": summarize_pairs(common),
        "primary_sum": summarize_primary(common),
        "separation_bins": bins,
        "constgold_opened": False,
    }
    with open(output, "x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    print("ANTITHETIC_RBLEND_CASE_DONE", flush=True)


if __name__ == "__main__":
    main()
