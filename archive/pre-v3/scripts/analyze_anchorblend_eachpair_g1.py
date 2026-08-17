"""Compare exact individual-pair sums, V2.2 predictions, and coherent truth."""
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
from scripts.prepare_anchorblend_eachpair_g1 import arm_path  # noqa: E402


KEY = ["case", "input_index"]


def stat(values: np.ndarray) -> dict:
    values = np.asarray(values, float)
    return {
        "mean": float(values.mean()),
        "case_sem": float(values.std(ddof=1) / np.sqrt(len(values))),
        "n_cases": int(len(values)),
        "case_values": values.tolist(),
    }


def read_leg(base: str, case: int, sign: float, anchors: np.ndarray) -> pd.DataFrame:
    frame = retrieve_constant_shear(
        case, str(float(sign)), r_max=10.0, r_min=0.0, k=20,
        data_path=base, shape_suffix="_all", include_isolated=True,
    )
    frame = frame.loc[frame["input_index"].isin(anchors), [
        "input_index", "measured_e1", "measured_e2",
    ]].copy()
    if frame["input_index"].duplicated().any():
        raise RuntimeError(f"case {case} sign {sign:+g}: duplicate anchor detection")
    finite = np.isfinite(frame[["measured_e1", "measured_e2"]].to_numpy(float)).all(axis=1)
    return frame.loc[finite]


def add_decomposition(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["emulator_gap"] = out["R_model_sum"] - out["R_pair_sum"]
    out["additivity_gap"] = out["R_coherent_truth"] - out["R_pair_sum"]
    out["coherent_gap"] = out["R_model_sum"] - out["R_coherent_truth"]
    out["decomposition_replay"] = (
        out["emulator_gap"] - out["additivity_gap"] - out["coherent_gap"]
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-prefix", required=True)
    ap.add_argument("--manifest-dir", required=True)
    ap.add_argument("--coherent", required=True)
    ap.add_argument("--cases", type=int, nargs="+", required=True)
    ap.add_argument("--g", type=float, default=0.05)
    ap.add_argument("--max-rank", type=int, default=18)
    ap.add_argument("--output-feather", required=True)
    ap.add_argument("--output-pairs", required=True)
    ap.add_argument("--output-json", required=True)
    args = ap.parse_args()
    for output in (args.output_feather, args.output_pairs, args.output_json):
        if os.path.exists(output):
            raise FileExistsError(f"refusing existing output {output}")

    coherent = pd.read_feather(args.coherent, columns=[
        "case", "input_index", "R_blend_truth", "R_blend_lsst_r_extnbr_v22",
    ])
    coherent = coherent.loc[coherent["case"].isin(args.cases)].copy()
    if coherent.duplicated(KEY).any():
        raise RuntimeError("duplicate coherent response key")
    coherent = coherent.rename(columns={
        "R_blend_truth": "R_coherent_truth",
        "R_blend_lsst_r_extnbr_v22": "R_coherent_stored_model",
    })

    manifest_dir = Path(args.manifest_dir)
    anchor_parts = []
    pair_parts = []
    audit = []
    rank_local_case = []
    for case in args.cases:
        anchors = pd.read_feather(manifest_dir / f"anchors_case{case}.feather")
        pairs = pd.read_feather(manifest_dir / f"pairs_case{case}.feather")
        anchor_ids = anchors["index"].to_numpy(np.int64)
        joined = pd.DataFrame({"input_index": anchor_ids})
        local_rank_rows = []
        for rank in range(args.max_rank + 1):
            base = arm_path(args.output_prefix, rank)
            plus = read_leg(base, case, +args.g, anchor_ids).rename(columns={
                "measured_e1": "e1_plus", "measured_e2": "e2_plus",
            })
            minus = read_leg(base, case, -args.g, anchor_ids).rename(columns={
                "measured_e1": "e1_minus", "measured_e2": "e2_minus",
            })
            leg = plus.merge(minus, on="input_index", how="inner", validate="one_to_one")
            leg[f"R_rank{rank:02d}"] = (leg["e1_plus"] - leg["e1_minus"]) / (2.0 * args.g)
            leg[f"N_rank{rank:02d}"] = (leg["e2_plus"] - leg["e2_minus"]) / (2.0 * args.g)
            selected = pairs.loc[pairs["rank"] == rank, [
                "anchor_index", "secondary_index", "response", "distance",
            ]].rename(columns={"anchor_index": "input_index", "response": "R_model_pair"})
            local = leg[["input_index", f"R_rank{rank:02d}", f"N_rank{rank:02d}"]].merge(
                selected, on="input_index", how="inner", validate="one_to_one",
            )
            local["case"] = int(case)
            local["rank"] = int(rank)
            local["R_pair_truth"] = local[f"R_rank{rank:02d}"]
            local["pair_gap"] = local["R_model_pair"] - local["R_pair_truth"]
            local_rank_rows.append(local)
            local_case = {
                "case": int(case), "rank": int(rank),
                "coverage": float(len(leg) / len(anchor_ids)),
                "n_pairs": int(len(local)),
                "model_mean_over_anchors": float(local["R_model_pair"].sum() / len(anchor_ids)),
                "truth_mean_over_anchors": float(local["R_pair_truth"].sum() / len(anchor_ids)),
            }
            rank_local_case.append(local_case)
            joined = joined.merge(
                leg[["input_index", f"R_rank{rank:02d}", f"N_rank{rank:02d}"]],
                on="input_index", how="inner", validate="one_to_one",
            )
        all_leg_coverage = len(joined) / len(anchor_ids)
        if all_leg_coverage < 0.65:
            raise RuntimeError(f"case {case}: all-rank coverage {all_leg_coverage:.2%} < 65%")
        joined.insert(0, "case", int(case))
        joined = joined.merge(
            coherent.loc[coherent.case == case], on=KEY, how="inner", validate="one_to_one",
        )
        coherent_common_coverage = len(joined) / len(anchor_ids)
        if coherent_common_coverage < 0.60:
            raise RuntimeError(
                f"case {case}: all-rank/coherent coverage {coherent_common_coverage:.2%} < 60%"
            )
        rcols = [f"R_rank{rank:02d}" for rank in range(args.max_rank + 1)]
        ncols = [f"N_rank{rank:02d}" for rank in range(args.max_rank + 1)]
        joined["R_pair_sum"] = joined[rcols].sum(axis=1)
        joined["N_pair_sum"] = joined[ncols].sum(axis=1)
        model_sum = pairs.groupby("anchor_index", sort=False)["response"].sum()
        joined["R_model_sum"] = joined["input_index"].map(model_sum).fillna(0.0)
        replay = joined["R_model_sum"] - joined["R_coherent_stored_model"]
        if float(np.max(np.abs(replay))) > 1e-2 or abs(float(replay.mean())) > 1e-5:
            raise RuntimeError(
                f"case {case}: fresh/stored model replay material: "
                f"max={np.max(np.abs(replay)):.3e} mean={replay.mean():+.3e}"
            )
        joined = add_decomposition(joined)
        if float(np.max(np.abs(joined["decomposition_replay"]))) > 1e-12:
            raise RuntimeError(f"case {case}: decomposition algebra does not close")
        common_ids = joined["input_index"]
        case_pairs = pd.concat(local_rank_rows, ignore_index=True)
        case_pairs = case_pairs.loc[case_pairs["input_index"].isin(common_ids)].copy()
        pair_parts.append(case_pairs)
        anchor_parts.append(joined)
        audit.append({
            "case": int(case), "n_manifest_anchors": int(len(anchor_ids)),
            "n_all_rank_detected": int(round(all_leg_coverage * len(anchor_ids))),
            "n_common_coherent": int(len(joined)),
            "all_rank_coverage": float(all_leg_coverage),
            "common_coherent_coverage": float(coherent_common_coverage),
            "n_common_pairs": int(len(case_pairs)),
            "prediction_replay_max_abs": float(np.max(np.abs(replay))),
            "prediction_replay_mean": float(replay.mean()),
        })
        print(
            f"case {case}: anchors={len(anchor_ids):,} common={len(joined):,} "
            f"pairs={len(case_pairs):,} model={joined.R_model_sum.mean():+.5f} "
            f"individual={joined.R_pair_sum.mean():+.5f} "
            f"coherent={joined.R_coherent_truth.mean():+.5f}", flush=True,
        )

    anchors = pd.concat(anchor_parts, ignore_index=True)
    pair_rows = pd.concat(pair_parts, ignore_index=True)
    anchors.to_feather(args.output_feather)
    pair_rows.to_feather(args.output_pairs)

    columns = [
        "R_model_sum", "R_pair_sum", "R_coherent_truth", "N_pair_sum",
        "emulator_gap", "additivity_gap", "coherent_gap",
    ]
    by_case = anchors.groupby("case", sort=True)[columns].mean()
    statistics = {column: stat(by_case[column].to_numpy(float)) for column in columns}
    full_case = coherent.groupby("case", sort=True).agg(
        full_model=("R_coherent_stored_model", "mean"),
        full_truth=("R_coherent_truth", "mean"),
    )
    full_case["full_coherent_gap"] = full_case["full_model"] - full_case["full_truth"]
    common_vs_full = by_case[["coherent_gap"]].join(full_case, how="inner")
    common_vs_full["conditioning_shift"] = (
        common_vs_full["coherent_gap"] - common_vs_full["full_coherent_gap"]
    )

    pair_case = pair_rows.groupby("case", sort=True).agg(
        R_model_pair=("R_model_pair", "mean"),
        R_pair_truth=("R_pair_truth", "mean"),
        pair_gap=("pair_gap", "mean"),
    )
    local = pd.DataFrame(rank_local_case)
    local["gap_mean_over_anchors"] = (
        local["model_mean_over_anchors"] - local["truth_mean_over_anchors"]
    )
    local_case = local.groupby("case", sort=True)[[
        "model_mean_over_anchors", "truth_mean_over_anchors", "gap_mean_over_anchors",
    ]].sum()
    payload = {
        "design": "every deployed neighbour sheared alone in coherent g1; stable rank layers; all other sources present unsheared",
        "estimands": {
            "emulator_gap": "sum emulator predictions minus sum individually measured g1 responses",
            "additivity_gap": "coherent all-neighbour g1 response minus sum individually measured g1 responses",
            "coherent_gap": "sum emulator predictions minus coherent all-neighbour g1 response",
            "identity": "coherent_gap = emulator_gap - additivity_gap",
            "conditioning_shift": "all-rank-common coherent gap minus ordinary coherent-population gap",
        },
        "case_window": [int(min(args.cases)), int(max(args.cases))],
        "n_cases": int(anchors["case"].nunique()),
        "n_common_anchors": int(len(anchors)),
        "n_common_pairs": int(len(pair_rows)),
        "statistics": statistics,
        "pair_level_common_population": {
            column: stat(pair_case[column].to_numpy(float))
            for column in ["R_model_pair", "R_pair_truth", "pair_gap"]
        },
        "rank_local_population_sum": {
            column: stat(local_case[column].to_numpy(float))
            for column in local_case
        },
        "ordinary_coherent_population": {
            column: stat(full_case[column].to_numpy(float))
            for column in full_case
        },
        "conditioning_shift": stat(common_vs_full["conditioning_shift"].to_numpy(float)),
        "rank_local_coverage": {
            "mean": float(local["coverage"].mean()),
            "min": float(local["coverage"].min()),
            "max": float(local["coverage"].max()),
        },
        "audit": audit,
        "constgold_opened": False,
    }
    with open(args.output_json, "x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    print("ANCHORBLEND_EACHPAIR_G1_ANALYSIS_DONE", flush=True)


if __name__ == "__main__":
    main()
