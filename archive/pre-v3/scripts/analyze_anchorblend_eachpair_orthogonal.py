"""Combine matched g1/g2 pair and coherent arms into response traces."""
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
from scripts.prepare_anchorblend_eachpair_g2 import pair_arm_path  # noqa: E402


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


def add_trace_decomposition(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["R_pair_trace"] = 0.5 * (out["R11_pair"] + out["R22_pair"])
    out["R_coherent_trace"] = 0.5 * (out["R11_coherent"] + out["R22_coherent"])
    out["emulator_gap_trace"] = out["R_model_sum"] - out["R_pair_trace"]
    out["additivity_gap_trace"] = out["R_coherent_trace"] - out["R_pair_trace"]
    out["coherent_gap_trace"] = out["R_model_sum"] - out["R_coherent_trace"]
    out["trace_replay"] = (
        out["emulator_gap_trace"] - out["additivity_gap_trace"]
        - out["coherent_gap_trace"]
    )
    out["pair_cross_symmetric"] = 0.5 * (out["R12_pair"] + out["R21_pair"])
    out["pair_cross_rotation"] = 0.5 * (out["R21_pair"] - out["R12_pair"])
    out["coherent_cross_symmetric"] = 0.5 * (
        out["R12_coherent"] + out["R21_coherent"]
    )
    out["coherent_cross_rotation"] = 0.5 * (
        out["R21_coherent"] - out["R12_coherent"]
    )
    return out


def subset_case_stats(frame: pd.DataFrame, mask: np.ndarray, columns: list[str]) -> dict:
    selected = frame.loc[mask]
    case = selected.groupby("case", sort=True)[columns].mean()
    return {
        "n_rows": int(len(selected)),
        "row_fraction": float(len(selected) / len(frame)),
        "statistics": {column: stat(case[column].to_numpy(float)) for column in columns},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--g1-anchor", required=True)
    ap.add_argument("--pair-prefix-g2", required=True)
    ap.add_argument("--coherent-g2-base", required=True)
    ap.add_argument("--coherent-g1", required=True)
    ap.add_argument("--manifest-dir", required=True)
    ap.add_argument("--cases", type=int, nargs="+", required=True)
    ap.add_argument("--g", type=float, default=0.05)
    ap.add_argument("--max-rank", type=int, default=18)
    ap.add_argument("--dominance-threshold", type=float, default=0.7752772106835227)
    ap.add_argument("--output-feather", required=True)
    ap.add_argument("--output-json", required=True)
    args = ap.parse_args()
    for output in (args.output_feather, args.output_json):
        if os.path.exists(output):
            raise FileExistsError(f"refusing existing output {output}")

    g1 = pd.read_feather(args.g1_anchor)
    g1 = g1.loc[g1["case"].isin(args.cases)].copy()
    if g1.duplicated(KEY).any():
        raise RuntimeError("duplicate g1 anchor key")
    coherent_g1 = pd.read_feather(args.coherent_g1, columns=[
        "case", "input_index", "measured_e2_plus", "measured_e2_minus",
    ])
    coherent_g1 = coherent_g1.loc[coherent_g1["case"].isin(args.cases)].copy()
    coherent_g1["R21_coherent"] = (
        coherent_g1["measured_e2_plus"] - coherent_g1["measured_e2_minus"]
    ) / (2.0 * args.g)
    coherent_g1 = coherent_g1[KEY + ["R21_coherent"]]

    manifest_dir = Path(args.manifest_dir)
    parts = []
    audit = []
    for case in args.cases:
        anchors = pd.read_feather(manifest_dir / f"anchors_case{case}.feather")
        pairs = pd.read_feather(manifest_dir / f"pairs_case{case}.feather")
        anchor_ids = anchors["index"].to_numpy(np.int64)
        joined = g1.loc[g1.case == case].copy()
        n_g1_common = len(joined)

        r22_columns = []
        r12_columns = []
        for rank in range(args.max_rank + 1):
            base = pair_arm_path(args.pair_prefix_g2, rank)
            plus = read_leg(base, case, +args.g, anchor_ids).rename(columns={
                "measured_e1": "e1_plus", "measured_e2": "e2_plus",
            })
            minus = read_leg(base, case, -args.g, anchor_ids).rename(columns={
                "measured_e1": "e1_minus", "measured_e2": "e2_minus",
            })
            leg = plus.merge(minus, on="input_index", how="inner", validate="one_to_one")
            r12 = f"R12_rank{rank:02d}"
            r22 = f"R22_rank{rank:02d}"
            leg[r12] = (leg["e1_plus"] - leg["e1_minus"]) / (2.0 * args.g)
            leg[r22] = (leg["e2_plus"] - leg["e2_minus"]) / (2.0 * args.g)
            joined = joined.merge(
                leg[["input_index", r12, r22]], on="input_index", how="inner",
                validate="one_to_one",
            )
            r12_columns.append(r12)
            r22_columns.append(r22)

        coherent_plus = read_leg(
            args.coherent_g2_base, case, +args.g, anchor_ids,
        ).rename(columns={"measured_e1": "e1_plus", "measured_e2": "e2_plus"})
        coherent_minus = read_leg(
            args.coherent_g2_base, case, -args.g, anchor_ids,
        ).rename(columns={"measured_e1": "e1_minus", "measured_e2": "e2_minus"})
        coherent = coherent_plus.merge(
            coherent_minus, on="input_index", how="inner", validate="one_to_one",
        )
        coherent["R12_coherent"] = (
            coherent["e1_plus"] - coherent["e1_minus"]
        ) / (2.0 * args.g)
        coherent["R22_coherent"] = (
            coherent["e2_plus"] - coherent["e2_minus"]
        ) / (2.0 * args.g)
        joined = joined.merge(
            coherent[["input_index", "R12_coherent", "R22_coherent"]],
            on="input_index", how="inner", validate="one_to_one",
        )
        joined = joined.merge(
            coherent_g1.loc[coherent_g1.case == case], on=KEY, how="inner",
            validate="one_to_one",
        )
        coverage = len(joined) / len(anchor_ids)
        if coverage < 0.55:
            raise RuntimeError(f"case {case}: full orthogonal coverage {coverage:.2%} < 55%")

        joined["R11_pair"] = joined["R_pair_sum"]
        joined["R21_pair"] = joined["N_pair_sum"]
        joined["R11_coherent"] = joined["R_coherent_truth"]
        joined["R12_pair"] = joined[r12_columns].sum(axis=1)
        joined["R22_pair"] = joined[r22_columns].sum(axis=1)
        grouped = pairs.groupby("anchor_index", sort=False)["response"]
        abs_sum = grouped.apply(lambda x: float(np.abs(x).sum()))
        max_abs = grouped.apply(lambda x: float(np.abs(x).max()))
        dominance = (max_abs / abs_sum.replace(0.0, np.nan)).fillna(0.0)
        joined["top_abs_fraction"] = joined["input_index"].map(dominance).fillna(0.0)
        joined = add_trace_decomposition(joined)
        if float(np.max(np.abs(joined["trace_replay"]))) > 1e-12:
            raise RuntimeError(f"case {case}: trace decomposition does not close")
        parts.append(joined)
        audit.append({
            "case": int(case),
            "n_manifest_anchors": int(len(anchor_ids)),
            "n_g1_common": int(n_g1_common),
            "n_full_orthogonal_common": int(len(joined)),
            "full_orthogonal_coverage": float(coverage),
            "retention_from_g1_common": float(len(joined) / n_g1_common),
        })
        print(
            f"case {case}: common={len(joined):,}/{len(anchor_ids):,} "
            f"model={joined.R_model_sum.mean():+.5f} "
            f"pair-trace={joined.R_pair_trace.mean():+.5f} "
            f"coherent-trace={joined.R_coherent_trace.mean():+.5f}", flush=True,
        )

    frame = pd.concat(parts, ignore_index=True)
    frame.to_feather(args.output_feather)
    columns = [
        "R_model_sum", "R11_pair", "R22_pair", "R_pair_trace",
        "R11_coherent", "R22_coherent", "R_coherent_trace",
        "emulator_gap_trace", "additivity_gap_trace", "coherent_gap_trace",
        "R12_pair", "R21_pair", "pair_cross_symmetric", "pair_cross_rotation",
        "R12_coherent", "R21_coherent", "coherent_cross_symmetric",
        "coherent_cross_rotation",
    ]
    all_stats = subset_case_stats(frame, np.ones(len(frame), dtype=bool), columns)
    dominant = frame["top_abs_fraction"].to_numpy(float) > args.dominance_threshold
    payload = {
        "design": (
            "matched fixed g1/g2 response trace: every frozen neighbour rank sheared "
            "alone, plus coherent all-non-anchor g1/g2 arms"
        ),
        "estimands": {
            "R_pair_trace": "0.5 * (sum R11_pair + sum R22_pair)",
            "R_coherent_trace": "0.5 * (R11_coherent + R22_coherent)",
            "emulator_gap_trace": "model scalar minus individually measured trace",
            "additivity_gap_trace": "coherent trace minus individually measured trace",
            "coherent_gap_trace": "model scalar minus coherent trace",
            "identity": "coherent_gap_trace = emulator_gap_trace - additivity_gap_trace",
        },
        "case_window": [int(min(args.cases)), int(max(args.cases))],
        "n_cases": int(frame["case"].nunique()),
        "n_common_anchors": int(len(frame)),
        "all_common": all_stats,
        "dominant_pair_tail": {
            "definition": f"top_abs_fraction > {args.dominance_threshold:.15g}",
            **subset_case_stats(frame, dominant, columns),
        },
        "audit": audit,
        "constgold_opened": False,
    }
    with open(args.output_json, "x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    print("ANCHORBLEND_EACHPAIR_ORTHOGONAL_ANALYSIS_DONE", flush=True)


if __name__ == "__main__":
    main()
