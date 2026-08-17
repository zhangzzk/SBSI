"""Compare disjoint random-direction anchor layers with coherent shear.

The random-radius layers are deterministic, disjoint subsets of the original
coherent-anchor population.  This analysis concatenates exact-key matches from
all supplied layers before case averaging.  It reports direction and aperture
contrasts without reading constgold or fitting a correction.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

try:
    from scripts.analyze_anchorblend_random_radii import sem, stat
except ModuleNotFoundError:  # direct ``python scripts/...`` execution
    from analyze_anchorblend_random_radii import sem, stat


KEY = ["case", "input_index"]


def case_stat(frame: pd.DataFrame, column: str) -> dict:
    return stat(frame.groupby("case", sort=True)[column].mean().to_numpy(float))


def renamed(frame: pd.DataFrame, suffix: str) -> pd.DataFrame:
    return frame.rename(columns={
        column: f"{column}_{suffix}" for column in frame.columns if column not in KEY
    })


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--coherent-response", required=True)
    ap.add_argument("--random10-response", action="append", required=True)
    ap.add_argument("--random15-response", action="append", required=True)
    ap.add_argument("--manifest-root", action="append", required=True)
    ap.add_argument("--case-min", type=int, default=200)
    ap.add_argument("--case-max", type=int, default=299)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    if os.path.exists(args.output):
        raise FileExistsError(f"refusing to overwrite {args.output}")
    lengths = {
        len(args.random10_response), len(args.random15_response),
        len(args.manifest_root),
    }
    if len(lengths) != 1:
        raise RuntimeError("random response/manifest argument counts differ")

    columns = KEY + ["R_blend_truth", "R_blend_lsst_r_extnbr_v22"]
    coherent = pd.read_feather(args.coherent_response, columns=columns)
    coherent = coherent[coherent.case.between(args.case_min, args.case_max)].copy()
    if coherent.duplicated(KEY).any():
        raise RuntimeError("duplicate coherent response key")
    coherent["gap"] = (
        coherent.R_blend_lsst_r_extnbr_v22 - coherent.R_blend_truth
    )

    manifest_parts = []
    joined_parts = []
    layer_case_tables = {}
    layer_payload = {}
    total_random10 = 0
    for layer, (path10, path15, root) in enumerate(zip(
        args.random10_response, args.random15_response, args.manifest_root,
    )):
        layer_manifest = []
        for case in range(args.case_min, args.case_max + 1):
            part = pd.read_feather(
                os.path.join(root, f"anchors_case{case}.feather"), columns=["index"],
            ).rename(columns={"index": "input_index"})
            part["case"] = case
            layer_manifest.append(part[KEY])
        layer_manifest = pd.concat(layer_manifest, ignore_index=True)
        if layer_manifest.duplicated(KEY).any():
            raise RuntimeError(f"layer {layer}: duplicate manifest anchor key")
        layer_manifest["layer"] = layer
        manifest_parts.append(layer_manifest)

        random10 = pd.read_feather(path10, columns=columns + ["R_blend_null"])
        random15 = pd.read_feather(path15, columns=columns + ["R_blend_null"])
        random10 = random10[random10.case.between(args.case_min, args.case_max)]
        random15 = random15[random15.case.between(args.case_min, args.case_max)]
        total_random10 += len(random10)
        joined = (
            renamed(coherent[columns], "coherent")
            .merge(renamed(random10, "random10"), on=KEY, validate="one_to_one")
            .merge(renamed(random15, "random15"), on=KEY, validate="one_to_one")
        )
        if len(joined) / len(random10) < 0.99 or len(joined) / len(random15) < 0.99:
            raise RuntimeError(f"layer {layer}: exact common-key coverage below 99%")
        predictions = [
            joined.R_blend_lsst_r_extnbr_v22_coherent.to_numpy(float),
            joined.R_blend_lsst_r_extnbr_v22_random10.to_numpy(float),
            joined.R_blend_lsst_r_extnbr_v22_random15.to_numpy(float),
        ]
        if not all(np.array_equal(predictions[0], value) for value in predictions[1:]):
            raise RuntimeError(f"layer {layer}: V2.2 prediction differs across arms")
        prediction = joined.R_blend_lsst_r_extnbr_v22_coherent
        joined["coherent_minus_random10"] = (
            joined.R_blend_truth_coherent - joined.R_blend_truth_random10
        )
        joined["coherent_minus_random15"] = (
            joined.R_blend_truth_coherent - joined.R_blend_truth_random15
        )
        joined["random_shell_10_15"] = (
            joined.R_blend_truth_random15 - joined.R_blend_truth_random10
        )
        joined["gap_coherent"] = prediction - joined.R_blend_truth_coherent
        joined["gap_random10"] = prediction - joined.R_blend_truth_random10
        joined["gap_random15"] = prediction - joined.R_blend_truth_random15
        layer_case = joined.groupby("case", sort=True)[[
            "R_blend_truth_coherent", "R_blend_truth_random10",
            "R_blend_truth_random15", "R_blend_lsst_r_extnbr_v22_coherent",
            "coherent_minus_random10", "coherent_minus_random15",
            "random_shell_10_15", "gap_coherent", "gap_random10", "gap_random15",
            "R_blend_null_random10", "R_blend_null_random15",
        ]].mean()
        layer_case_tables[layer] = layer_case
        joined["layer"] = layer
        joined_parts.append(joined)
        layer_payload[str(layer)] = {
            "n_manifest_anchors": int(len(layer_manifest)),
            "n_exact_common_rows": int(len(joined)),
            "coverage_of_random10": float(len(joined) / len(random10)),
            "coverage_of_random15": float(len(joined) / len(random15)),
            "coherent_truth": stat(layer_case.R_blend_truth_coherent),
            "random10_truth": stat(layer_case.R_blend_truth_random10),
            "random15_truth": stat(layer_case.R_blend_truth_random15),
            "v22_prediction": stat(layer_case.R_blend_lsst_r_extnbr_v22_coherent),
            "v22_minus_coherent": stat(layer_case.gap_coherent),
            "v22_minus_random10": stat(layer_case.gap_random10),
            "v22_minus_random15": stat(layer_case.gap_random15),
            "coherent_minus_random10": stat(layer_case.coherent_minus_random10),
            "coherent_minus_random15": stat(layer_case.coherent_minus_random15),
            "random_shell_10_15": stat(layer_case.random_shell_10_15),
        }

    manifest = pd.concat(manifest_parts, ignore_index=True)
    joined = pd.concat(joined_parts, ignore_index=True)
    if manifest.duplicated(KEY).any():
        raise RuntimeError("anchor keys overlap across manifests")
    if joined.duplicated(KEY).any():
        raise RuntimeError("exact matched keys overlap across layers")

    coherent_population = coherent.merge(
        manifest[KEY].assign(retained=True), on=KEY, how="left", validate="one_to_one",
    )
    coherent_population["retained"] = coherent_population.retained.eq(True)
    retained = coherent_population[coherent_population.retained]
    excluded = coherent_population[~coherent_population.retained]

    value_columns = [
        "R_blend_truth_coherent", "R_blend_truth_random10",
        "R_blend_truth_random15", "R_blend_lsst_r_extnbr_v22_coherent",
        "coherent_minus_random10", "coherent_minus_random15",
        "random_shell_10_15", "gap_coherent", "gap_random10", "gap_random15",
        "R_blend_null_random10", "R_blend_null_random15",
    ]
    case = joined.groupby("case", sort=True)[value_columns].mean()
    expected_cases = args.case_max - args.case_min + 1
    if len(case) != expected_cases:
        raise RuntimeError("combined exact common set does not cover every case")

    contrast10 = case.coherent_minus_random10.to_numpy(float)
    contrast15 = case.coherent_minus_random15.to_numpy(float)
    shell = case.random_shell_10_15.to_numpy(float)
    null10 = case.R_blend_null_random10.to_numpy(float)
    null15 = case.R_blend_null_random15.to_numpy(float)
    paired_layer_contrasts = {}
    for left in range(len(layer_case_tables)):
        for right in range(left + 1, len(layer_case_tables)):
            paired = layer_case_tables[right].join(
                layer_case_tables[left], how="inner", lsuffix=f"_l{right}",
                rsuffix=f"_l{left}", validate="one_to_one",
            )
            if len(paired) != expected_cases:
                raise RuntimeError(f"layers {left}/{right}: incomplete paired case set")
            entry = {}
            for column in ("gap_coherent", "gap_random10", "gap_random15"):
                difference = (
                    paired[f"{column}_l{right}"] - paired[f"{column}_l{left}"]
                ).to_numpy(float)
                entry[f"layer{right}_minus_layer{left}_{column}"] = stat(difference)
            paired_layer_contrasts[f"{right}_minus_{left}"] = entry
    payload = {
        "design": "disjoint random-direction anchor layers versus exact-key coherent response",
        "case_window": [args.case_min, args.case_max],
        "n_layers": len(joined_parts),
        "layers": layer_payload,
        "paired_layer_contrasts": paired_layer_contrasts,
        "n_coherent_rows": int(len(coherent)),
        "n_manifest_anchors": int(len(manifest)),
        "n_coherent_retained_rows": int(len(retained)),
        "n_coherent_excluded_rows": int(len(excluded)),
        "n_exact_common_rows": int(len(joined)),
        "coverage_of_random10": float(len(joined) / total_random10),
        "coherent_full_gap": case_stat(coherent, "gap"),
        "coherent_retained_gap": case_stat(retained, "gap"),
        "coherent_excluded_gap": case_stat(excluded, "gap"),
        "exact_common_coherent_truth": stat(case.R_blend_truth_coherent),
        "exact_common_random10_truth": stat(case.R_blend_truth_random10),
        "exact_common_random15_truth": stat(case.R_blend_truth_random15),
        "exact_common_v22_prediction": stat(case.R_blend_lsst_r_extnbr_v22_coherent),
        "exact_common_v22_minus_coherent": stat(case.gap_coherent),
        "exact_common_v22_minus_random10": stat(case.gap_random10),
        "exact_common_v22_minus_random15": stat(case.gap_random15),
        "exact_common_coherent_minus_random10": stat(contrast10),
        "exact_common_coherent_minus_random15": stat(contrast15),
        "exact_common_random_shell_10_15": stat(shell),
        "null_random10": stat(null10),
        "null_random15": stat(null15),
        "gates": {
            "random_shell_10_15_detected_at_3_case_sem": bool(
                abs(shell.mean()) >= 3.0 * sem(shell)
            ),
            "coherent_minus_random10_detected_at_3_case_sem": bool(
                abs(contrast10.mean()) >= 3.0 * sem(contrast10)
            ),
            "coherent_minus_random15_detected_at_3_case_sem": bool(
                abs(contrast15.mean()) >= 3.0 * sem(contrast15)
            ),
            "random_nulls_within_3_case_sem": bool(
                abs(null10.mean()) <= 3.0 * sem(null10)
                and abs(null15.mean()) <= 3.0 * sem(null15)
            ),
        },
        "constgold_opened": False,
    }
    with open(args.output, "x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    print("ANCHORBLEND_MULTILAYER_VS_COHERENT_ANALYSIS_DONE", flush=True)


if __name__ == "__main__":
    main()
