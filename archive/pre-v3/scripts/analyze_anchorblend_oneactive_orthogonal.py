"""Measure the isotropic response of one active neighbour per anchor."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BLENDEMU_ROOT = "/home/z/Zekang.Zhang/blendemu"
if BLENDEMU_ROOT not in sys.path:
    sys.path.insert(0, BLENDEMU_ROOT)

from blendemu.response import retrieve_constant_shear  # noqa: E402


KEY = ["case", "input_index"]


def label(g: float) -> str:
    return str(float(g))


def stat(values: np.ndarray) -> dict:
    values = np.asarray(values, float)
    return {
        "mean": float(values.mean()),
        "case_sem": float(values.std(ddof=1) / np.sqrt(len(values))),
        "n_cases": int(len(values)),
    }


def read_leg(base: str, case: int, sign: float, anchors: np.ndarray) -> pd.DataFrame:
    frame = retrieve_constant_shear(
        case, label(sign), r_max=10.0, r_min=0.0, k=20,
        data_path=base, shape_suffix="_all", include_isolated=True,
    )
    frame = frame[frame["input_index"].isin(anchors)][
        ["input_index", "measured_e1", "measured_e2"]
    ].copy()
    if frame["input_index"].duplicated().any():
        raise RuntimeError(f"case {case} sign {sign:+g}: duplicate anchor detection")
    finite = np.isfinite(frame[["measured_e1", "measured_e2"]].to_numpy(float)).all(axis=1)
    return frame.loc[finite]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-u", required=True)
    ap.add_argument("--base-v", required=True)
    ap.add_argument("--manifest-dir", required=True)
    ap.add_argument("--coherent", required=True)
    ap.add_argument("--cases", type=int, nargs="+", required=True)
    ap.add_argument("--g", type=float, default=0.05)
    ap.add_argument("--output-feather", required=True)
    ap.add_argument("--output-json", required=True)
    args = ap.parse_args()
    for path in (args.output_feather, args.output_json):
        if os.path.exists(path):
            raise FileExistsError(f"refusing existing output {path}")

    coherent = pd.read_feather(args.coherent, columns=[
        "case", "input_index", "R_blend_truth", "R_blend_lsst_r_extnbr_v22",
    ])
    coherent = coherent[coherent["case"].isin(args.cases)].copy()
    if coherent.duplicated(KEY).any():
        raise RuntimeError("duplicate coherent response key")

    parts = []
    audit = []
    manifest_dir = Path(args.manifest_dir)
    for case in args.cases:
        anchors = pd.read_feather(manifest_dir / f"anchors_case{case}.feather")
        pairs = pd.read_feather(manifest_dir / f"pairs_case{case}.feather")
        anchor_ids = anchors["index"].to_numpy(np.int64)
        selected = pairs.loc[pairs.selected].copy()
        if selected["anchor_index"].duplicated().any():
            raise RuntimeError(f"case {case}: multiple selected pairs per anchor")

        joined = pd.DataFrame({"input_index": anchor_ids})
        for mode, base in (("u", args.base_u), ("v", args.base_v)):
            for side, sign in (("plus", +args.g), ("minus", -args.g)):
                leg = read_leg(base, case, sign, anchor_ids)
                rename = {
                    c: f"{c}_{mode}_{side}" for c in leg if c != "input_index"
                }
                joined = joined.merge(
                    leg.rename(columns=rename), on="input_index", how="inner",
                    validate="one_to_one",
                )
        four_leg_coverage = len(joined) / len(anchor_ids)
        if four_leg_coverage < 0.80:
            raise RuntimeError(
                f"case {case}: four-leg anchor coverage {four_leg_coverage:.2%} < 80%"
            )
        joined.insert(0, "case", int(case))
        joined = joined.merge(
            coherent[coherent.case == case], on=KEY, how="inner", validate="one_to_one",
        )
        common_coverage = len(joined) / len(anchor_ids)
        if common_coverage < 0.75:
            raise RuntimeError(
                f"case {case}: coherent/four-leg coverage {common_coverage:.2%} < 75%"
            )
        joined = joined.merge(
            selected[[
                "anchor_index", "secondary_index", "response", "n_pairs",
                "selection_probability", "u1", "u2", "v1", "v2",
            ]].rename(columns={"anchor_index": "input_index", "response": "selected_prediction"}),
            on="input_index", how="left", validate="one_to_one",
        )
        has_pair = joined["secondary_index"].notna().to_numpy()
        for mode in ("u", "v"):
            de1 = (
                joined[f"measured_e1_{mode}_plus"].to_numpy(float)
                - joined[f"measured_e1_{mode}_minus"].to_numpy(float)
            ) / (2.0 * args.g)
            de2 = (
                joined[f"measured_e2_{mode}_plus"].to_numpy(float)
                - joined[f"measured_e2_{mode}_minus"].to_numpy(float)
            ) / (2.0 * args.g)
            d1 = joined[f"{mode}1"].to_numpy(float)
            d2 = joined[f"{mode}2"].to_numpy(float)
            response = np.zeros(len(joined), dtype=float)
            null = np.zeros(len(joined), dtype=float)
            response[has_pair] = de1[has_pair] * d1[has_pair] + de2[has_pair] * d2[has_pair]
            null[has_pair] = -de1[has_pair] * d2[has_pair] + de2[has_pair] * d1[has_pair]
            joined[f"R_{mode}"] = response
            joined[f"N_{mode}"] = null
        joined["R_one_pair"] = 0.5 * (joined["R_u"] + joined["R_v"])
        joined["R_direction_difference"] = 0.5 * (joined["R_u"] - joined["R_v"])
        joined["N_one_pair"] = 0.5 * (joined["N_u"] + joined["N_v"])
        weight = joined["n_pairs"].fillna(0.0).to_numpy(float)
        joined["HT_truth_sum"] = weight * joined["R_one_pair"]
        joined["HT_null_sum"] = weight * joined["N_one_pair"]
        joined["HT_model_sum"] = weight * joined["selected_prediction"].fillna(0.0)
        exact = pairs.groupby("anchor_index", sort=False)["response"].sum()
        joined["manifest_model_sum"] = joined["input_index"].map(exact).fillna(0.0)
        # The stored coherent response is the endpoint actually being compared.
        # Retain an explicit replay audit of the freshly reconstructed manifest,
        # but do not replace that endpoint with a second inference pass.  Tiny
        # CPU/order differences near a pair cut can affect an individual sum.
        joined["exact_model_sum"] = joined["R_blend_lsst_r_extnbr_v22"]
        replay_delta = joined["manifest_model_sum"] - joined["exact_model_sum"]
        replay = float(np.max(np.abs(replay_delta)))
        replay_mean = float(replay_delta.mean())
        if replay > 1e-2 or abs(replay_mean) > 1e-5:
            raise RuntimeError(
                f"case {case}: manifest/coherent prediction replay is material: "
                f"max={replay:.3e} mean={replay_mean:+.3e}"
            )
        joined["oneactive_gap_ht"] = joined["HT_model_sum"] - joined["HT_truth_sum"]
        joined["oneactive_gap_exact_model"] = joined["exact_model_sum"] - joined["HT_truth_sum"]
        joined["coherent_gap"] = (
            joined["R_blend_lsst_r_extnbr_v22"] - joined["R_blend_truth"]
        )
        joined["ht_model_sampling_balance"] = joined["HT_model_sum"] - joined["exact_model_sum"]
        parts.append(joined)
        audit.append({
            "case": int(case), "n_manifest_anchors": int(len(anchor_ids)),
            "n_four_leg": int(round(four_leg_coverage * len(anchor_ids))),
            "n_common_coherent": int(len(joined)),
            "four_leg_coverage": float(four_leg_coverage),
            "common_coherent_coverage": float(common_coverage),
            "n_common_with_selected_pair": int(has_pair.sum()),
            "prediction_replay_max_abs": float(replay),
            "prediction_replay_mean": float(replay_mean),
        })
        print(
            f"case {case}: common={len(joined):,}/{len(anchor_ids):,} "
            f"with-pair={has_pair.sum():,} replay={replay:.2e}", flush=True,
        )

    frame = pd.concat(parts, ignore_index=True)
    finite_columns = [
        "R_one_pair", "N_one_pair", "HT_truth_sum", "HT_null_sum",
        "HT_model_sum", "exact_model_sum", "oneactive_gap_ht",
        "oneactive_gap_exact_model", "coherent_gap", "ht_model_sampling_balance",
    ]
    if not np.isfinite(frame[finite_columns].to_numpy(float)).all():
        raise RuntimeError("non-finite final one-active response")
    frame.to_feather(args.output_feather)

    case = frame.groupby("case", sort=True)[finite_columns].mean()
    statistics = {column: stat(case[column].to_numpy(float)) for column in finite_columns}
    difference = case["oneactive_gap_exact_model"] - case["coherent_gap"]
    statistics["oneactive_minus_coherent_gap"] = stat(difference.to_numpy(float))
    coherent_abs = abs(statistics["coherent_gap"]["mean"])
    one_abs = abs(statistics["oneactive_gap_exact_model"]["mean"])
    combined_sem = float(np.hypot(
        statistics["oneactive_gap_exact_model"]["case_sem"],
        statistics["coherent_gap"]["case_sem"],
    ))
    payload = {
        "design": "one uniformly selected deployed neighbour per anchor, antithetic in two orthogonal spin-2 directions; HT reconstruction of full neighbour sum",
        "case_window": [int(min(args.cases)), int(max(args.cases))],
        "n_cases": int(frame["case"].nunique()),
        "n_common_rows": int(len(frame)),
        "n_common_with_selected_pair": int(frame["secondary_index"].notna().sum()),
        "statistics": statistics,
        "audit": audit,
        "predeclared_interpretation": {
            "individual_pair_carrier": bool(
                statistics["oneactive_gap_exact_model"]["mean"] < 0.0
                and one_abs >= 0.5 * coherent_abs
                and abs(statistics["oneactive_minus_coherent_gap"]["mean"])
                <= 2.0 * combined_sem
            ),
            "multi_neighbour_carrier": bool(
                one_abs < 0.5 * coherent_abs
                and abs(statistics["oneactive_minus_coherent_gap"]["mean"])
                > 2.0 * combined_sem
            ),
            "otherwise": "inconclusive or mixed mechanism",
        },
        "constgold_opened": False,
    }
    with open(args.output_json, "x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    print(f"wrote {args.output_feather}: {len(frame):,} rows")
    print("ANCHORBLEND_ONEACTIVE_ANALYSIS_DONE", flush=True)


if __name__ == "__main__":
    main()
