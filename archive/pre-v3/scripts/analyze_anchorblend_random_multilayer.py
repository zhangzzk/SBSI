"""Analyze the matched random-direction radius test across anchor layers.

Layers are disjoint deterministic partitions of the original coherent anchor
manifest.  The primary result combines layers 0--2 before case averaging;
per-layer values are retained as consistency diagnostics.  Cases 200--249 fix
the only allowed additive shell offset and 250--299 test it unchanged.
Constgold is never read.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

try:
    from scripts.analyze_anchorblend_random_radii import assert_matching_manifests, sem, stat
except ModuleNotFoundError:  # direct ``python scripts/...`` execution
    from analyze_anchorblend_random_radii import assert_matching_manifests, sem, stat


KEY = ["case", "input_index"]
VALUE_COLUMNS = [
    "R_blend_truth", "R_blend_null", "R_blend_lsst_r_extnbr_v22",
    "r_input_p_plus", "Re_input_p_plus",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--response10", action="append", required=True)
    ap.add_argument("--response15", action="append", required=True)
    ap.add_argument("--manifest-root10", action="append", required=True)
    ap.add_argument("--manifest-root15", action="append", required=True)
    ap.add_argument("--case-min", type=int, default=200)
    ap.add_argument("--case-max", type=int, default=299)
    ap.add_argument("--development-max", type=int, default=249)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    if os.path.exists(args.output):
        raise FileExistsError(f"refusing to overwrite {args.output}")
    lengths = {
        len(args.response10), len(args.response15),
        len(args.manifest_root10), len(args.manifest_root15),
    }
    if len(lengths) != 1:
        raise RuntimeError("radius response/manifest argument counts differ")
    cases = list(range(args.case_min, args.case_max + 1))
    common_parts = []
    layer_payload = {}
    total10 = total15 = 0
    for layer, paths in enumerate(zip(
        args.response10, args.response15, args.manifest_root10, args.manifest_root15,
    )):
        path10, path15, root10, root15 = paths
        assert_matching_manifests(root10, root15, cases)
        r10 = pd.read_feather(path10, columns=KEY + VALUE_COLUMNS)
        r15 = pd.read_feather(path15, columns=KEY + VALUE_COLUMNS)
        r10 = r10[r10.case.between(args.case_min, args.case_max)]
        r15 = r15[r15.case.between(args.case_min, args.case_max)]
        total10 += len(r10); total15 += len(r15)
        joined = r10.merge(r15, on=KEY, suffixes=("_10", "_15"),
                           validate="one_to_one")
        coverage10 = len(joined) / len(r10)
        coverage15 = len(joined) / len(r15)
        if min(coverage10, coverage15) < 0.99:
            raise RuntimeError(f"layer {layer}: common-key coverage below 99%")
        for column in ("R_blend_lsst_r_extnbr_v22", "r_input_p_plus", "Re_input_p_plus"):
            if not np.array_equal(joined[f"{column}_10"], joined[f"{column}_15"]):
                raise RuntimeError(f"layer {layer}: frozen column differs: {column}")
        joined["layer"] = layer
        joined["shell"] = joined.R_blend_truth_15 - joined.R_blend_truth_10
        joined["shell_null"] = joined.R_blend_null_15 - joined.R_blend_null_10
        joined["gap10"] = (
            joined.R_blend_lsst_r_extnbr_v22_10 - joined.R_blend_truth_10
        )
        joined["gap15"] = (
            joined.R_blend_lsst_r_extnbr_v22_10 - joined.R_blend_truth_15
        )
        common_parts.append(joined)
        layer_case = joined.groupby("case", sort=True)[[
            "R_blend_truth_10", "R_blend_truth_15", "shell", "gap10", "gap15",
            "R_blend_null_10", "R_blend_null_15", "shell_null",
        ]].mean()
        layer_payload[str(layer)] = {
            "n_common_rows": int(len(joined)),
            "coverage10": float(coverage10), "coverage15": float(coverage15),
            "truth10": stat(layer_case.R_blend_truth_10),
            "truth15": stat(layer_case.R_blend_truth_15),
            "shell10_15": stat(layer_case.shell),
            "v22_minus_truth10": stat(layer_case.gap10),
            "v22_minus_truth15": stat(layer_case.gap15),
        }

    joined = pd.concat(common_parts, ignore_index=True)
    if joined.duplicated(KEY).any():
        raise RuntimeError("anchor keys overlap across layers")
    case = joined.groupby("case", sort=True)[[
        "R_blend_truth_10", "R_blend_truth_15", "R_blend_null_10",
        "R_blend_null_15", "R_blend_lsst_r_extnbr_v22_10", "shell",
        "shell_null", "gap10", "gap15",
    ]].mean()
    development = case.index.to_numpy(int) <= args.development_max
    validation = ~development
    if development.sum() < 20 or validation.sum() < 20:
        raise RuntimeError("need at least 20 independent cases per split")
    shell_dev = case.loc[development, "shell"].to_numpy(float)
    shell_val = case.loc[validation, "shell"].to_numpy(float)
    raw_val = case.loc[validation, "gap15"].to_numpy(float)
    offset = float(shell_dev.mean())
    corrected_val = raw_val + offset
    transfer_sem = float(np.hypot(sem(shell_dev), sem(raw_val)))
    split_sem = float(np.hypot(sem(shell_dev), sem(shell_val)))
    shell_all = case.shell.to_numpy(float)
    gates = {
        "local10_gap_consistent_with_zero_at_2_case_sem": bool(
            abs(case.gap10.mean()) <= 2.0 * sem(case.gap10)
        ),
        "shell_10_15_detected_at_3_case_sem": bool(
            abs(shell_all.mean()) >= 3.0 * sem(shell_all)
        ),
        "shell_split_consistent_at_2_combined_sem": bool(
            abs(shell_dev.mean() - shell_val.mean()) <= 2.0 * split_sem
        ),
        "corrected_validation_gap_smaller": bool(
            abs(corrected_val.mean()) < abs(raw_val.mean())
        ),
        "corrected_validation_gap_consistent_with_zero_at_2_combined_sem": bool(
            abs(corrected_val.mean()) <= 2.0 * transfer_sem
        ),
        "truth_and_shell_nulls_within_3_case_sem": bool(all(
            abs(case[column].mean()) <= 3.0 * sem(case[column])
            for column in ("R_blend_null_10", "R_blend_null_15", "shell_null")
        )),
    }
    payload = {
        "design": "disjoint random-direction anchor layers combined before case averaging",
        "layers": layer_payload, "n_layers": len(common_parts),
        "case_window": [args.case_min, args.case_max],
        "development_cases": [int(case.index[development].min()), int(case.index[development].max())],
        "validation_cases": [int(case.index[validation].min()), int(case.index[validation].max())],
        "n_response10_rows": int(total10), "n_response15_rows": int(total15),
        "n_common_rows": int(len(joined)),
        "combined_coverage10": float(len(joined) / total10),
        "combined_coverage15": float(len(joined) / total15),
        "truth_local10": stat(case.R_blend_truth_10),
        "truth_local15": stat(case.R_blend_truth_15),
        "v22_prediction": stat(case.R_blend_lsst_r_extnbr_v22_10),
        "v22_minus_local10": stat(case.gap10),
        "v22_minus_local15": stat(case.gap15),
        "shell_10_15": stat(shell_all),
        "shell_10_15_development": stat(shell_dev),
        "shell_10_15_validation": stat(shell_val),
        "null_local10": stat(case.R_blend_null_10),
        "null_local15": stat(case.R_blend_null_15),
        "shell_null_10_15": stat(case.shell_null),
        "fixed_development_shell_offset": offset,
        "validation_raw_v22_minus_local15": stat(raw_val),
        "validation_corrected_v22_minus_local15": stat(corrected_val),
        "validation_corrected_mean_uncertainty_including_development": transfer_sem,
        "gates": gates, "gate_passed": bool(all(gates.values())),
        "constgold_opened": False,
    }
    with open(args.output, "x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    print("ANCHORBLEND_RANDOM_MULTILAYER_ANALYSIS_DONE", flush=True)


if __name__ == "__main__":
    main()
