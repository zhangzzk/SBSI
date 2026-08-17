"""Separate aperture/direction effects from anchor re-sparsification.

The random-direction radius experiment re-sparsified the original coherent
anchor manifests from 20 arcsec to 30.01 arcsec so 15-arcsec neighbourhoods do
not overlap.  This script uses exact case/input keys to report the original
coherent response on the full, retained-manifest, and excluded populations,
then compares coherent, random-local10, and random-local15 truth only on their
common measured anchors.  Constgold is never read.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd


KEY = ["case", "input_index"]


def sem(values) -> float:
    values = np.asarray(values, float)
    return float(values.std(ddof=1) / np.sqrt(len(values)))


def stat(values) -> dict:
    values = np.asarray(values, float)
    return {"mean": float(values.mean()), "case_sem": sem(values)}


def case_stat(frame: pd.DataFrame, column: str) -> dict:
    return stat(frame.groupby("case", sort=True)[column].mean().to_numpy(float))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--coherent-response", required=True)
    ap.add_argument("--random10-response", required=True)
    ap.add_argument("--random15-response", required=True)
    ap.add_argument("--manifest-root", required=True)
    ap.add_argument("--case-min", type=int, default=200)
    ap.add_argument("--case-max", type=int, default=299)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    if os.path.exists(args.output):
        raise FileExistsError(f"refusing to overwrite {args.output}")

    columns = KEY + ["R_blend_truth", "R_blend_lsst_r_extnbr_v22"]
    coherent = pd.read_feather(args.coherent_response, columns=columns)
    coherent = coherent[
        coherent.case.between(args.case_min, args.case_max)
    ].copy()
    random10 = pd.read_feather(
        args.random10_response,
        columns=columns + ["R_blend_null"],
    )
    random15 = pd.read_feather(
        args.random15_response,
        columns=columns + ["R_blend_null"],
    )
    for frame in (random10, random15):
        frame.drop(
            frame.index[~frame.case.between(args.case_min, args.case_max)],
            inplace=True,
        )

    manifest_parts = []
    for case in range(args.case_min, args.case_max + 1):
        part = pd.read_feather(
            os.path.join(args.manifest_root, f"anchors_case{case}.feather"),
            columns=["index"],
        ).rename(columns={"index": "input_index"})
        part["case"] = case
        manifest_parts.append(part[KEY])
    manifest = pd.concat(manifest_parts, ignore_index=True)
    if manifest.duplicated(KEY).any():
        raise RuntimeError("duplicate retained anchor key in manifests")

    coherent = coherent.merge(
        manifest.assign(retained=True), on=KEY, how="left", validate="one_to_one",
    )
    coherent["retained"] = coherent["retained"].eq(True)
    coherent["gap"] = (
        coherent.R_blend_lsst_r_extnbr_v22 - coherent.R_blend_truth
    )
    retained = coherent[coherent.retained]
    excluded = coherent[~coherent.retained]
    if retained.case.nunique() != args.case_max - args.case_min + 1:
        raise RuntimeError("retained coherent subset does not cover every case")
    if excluded.case.nunique() != args.case_max - args.case_min + 1:
        raise RuntimeError("excluded coherent subset does not cover every case")

    def named(frame: pd.DataFrame, suffix: str) -> pd.DataFrame:
        return frame.rename(columns={
            column: f"{column}_{suffix}" for column in frame.columns if column not in KEY
        })

    joined = (
        named(coherent[KEY + ["R_blend_truth", "R_blend_lsst_r_extnbr_v22"]],
              "coherent")
        .merge(named(random10, "random10"), on=KEY, validate="one_to_one")
        .merge(named(random15, "random15"), on=KEY, validate="one_to_one")
    )

    predictions = [
        joined.R_blend_lsst_r_extnbr_v22_coherent.to_numpy(float),
        joined.R_blend_lsst_r_extnbr_v22_random10.to_numpy(float),
        joined.R_blend_lsst_r_extnbr_v22_random15.to_numpy(float),
    ]
    if not all(np.array_equal(predictions[0], values) for values in predictions[1:]):
        raise RuntimeError("V2.2 prediction differs across exact matched scene arms")
    joined["coherent_minus_random10"] = (
        joined.R_blend_truth_coherent - joined.R_blend_truth_random10
    )
    joined["coherent_minus_random15"] = (
        joined.R_blend_truth_coherent - joined.R_blend_truth_random15
    )
    joined["random_shell_10_15"] = (
        joined.R_blend_truth_random15 - joined.R_blend_truth_random10
    )
    joined["gap_coherent"] = predictions[0] - joined.R_blend_truth_coherent
    joined["gap_random10"] = predictions[0] - joined.R_blend_truth_random10
    joined["gap_random15"] = predictions[0] - joined.R_blend_truth_random15

    case = joined.groupby("case", sort=True)[[
        "R_blend_truth_coherent", "R_blend_truth_random10", "R_blend_truth_random15",
        "R_blend_lsst_r_extnbr_v22_coherent", "coherent_minus_random10",
        "coherent_minus_random15", "random_shell_10_15", "gap_coherent",
        "gap_random10", "gap_random15", "R_blend_null_random10",
        "R_blend_null_random15",
    ]].mean()
    if len(case) != args.case_max - args.case_min + 1:
        raise RuntimeError("triple common set does not cover every case")

    contrast = case.coherent_minus_random10.to_numpy(float)
    shell = case.random_shell_10_15.to_numpy(float)
    population_shift = (
        retained.groupby("case").gap.mean() - excluded.groupby("case").gap.mean()
    ).to_numpy(float)
    gates = {
        "random_shell_10_15_within_3_case_sem": bool(
            abs(shell.mean()) < 3.0 * sem(shell)
        ),
        "coherent_minus_random10_detected_at_3_case_sem": bool(
            abs(contrast.mean()) >= 3.0 * sem(contrast)
        ),
        "retained_minus_excluded_gap_detected_at_3_case_sem": bool(
            abs(population_shift.mean()) >= 3.0 * sem(population_shift)
        ),
        "random_nulls_within_3_case_sem": bool(all(
            abs(case[column].mean()) <= 3.0 * sem(case[column])
            for column in ("R_blend_null_random10", "R_blend_null_random15")
        )),
    }
    payload = {
        "case_window": [args.case_min, args.case_max],
        "n_coherent_rows": int(len(coherent)),
        "n_manifest_anchors": int(len(manifest)),
        "n_coherent_retained_rows": int(len(retained)),
        "n_coherent_excluded_rows": int(len(excluded)),
        "n_triple_common_rows": int(len(joined)),
        "triple_coverage_of_random10": float(len(joined) / len(random10)),
        "coherent_full_truth": case_stat(coherent, "R_blend_truth"),
        "coherent_full_gap": case_stat(coherent, "gap"),
        "coherent_retained_truth": case_stat(retained, "R_blend_truth"),
        "coherent_retained_gap": case_stat(retained, "gap"),
        "coherent_excluded_truth": case_stat(excluded, "R_blend_truth"),
        "coherent_excluded_gap": case_stat(excluded, "gap"),
        "retained_minus_excluded_coherent_gap": stat(population_shift),
        "triple_common_coherent_truth": stat(case.R_blend_truth_coherent),
        "triple_common_random10_truth": stat(case.R_blend_truth_random10),
        "triple_common_random15_truth": stat(case.R_blend_truth_random15),
        "triple_common_v22_prediction": stat(case.R_blend_lsst_r_extnbr_v22_coherent),
        "triple_common_v22_minus_coherent": stat(case.gap_coherent),
        "triple_common_v22_minus_random10": stat(case.gap_random10),
        "triple_common_v22_minus_random15": stat(case.gap_random15),
        "triple_common_coherent_minus_random10": stat(contrast),
        "triple_common_coherent_minus_random15": stat(case.coherent_minus_random15),
        "triple_common_random_shell_10_15": stat(shell),
        "null_random10": stat(case.R_blend_null_random10),
        "null_random15": stat(case.R_blend_null_random15),
        "gates": gates,
        "constgold_opened": False,
    }
    with open(args.output, "x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    print("ANCHORBLEND_RANDOM_VS_COHERENT_ANALYSIS_DONE", flush=True)


if __name__ == "__main__":
    main()
