"""Diagnose the scene population removed by 30-arcsec anchor sparsification.

Uses only existing coherent-anchor responses, retained manifests, and intrinsic
10--30 arcsec scene summaries.  No rendering, fitting, or constgold access.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd


KEY = ["case", "input_index"]
FEATURES = [
    "logflux_outer_10_15", "logflux_outer_15_20", "logflux_outer_20_30",
    "logflux_re2_over_d2_outer_10_30", "max_re_over_d_outer_10_30",
    "n_outer_10_30",
]


def sem(values) -> float:
    values = np.asarray(values, float)
    return float(values.std(ddof=1) / np.sqrt(len(values)))


def stat(values) -> dict:
    values = np.asarray(values, float)
    return {"mean": float(values.mean()), "case_sem": sem(values)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--coherent-response", required=True)
    ap.add_argument("--outer-features", required=True)
    ap.add_argument("--manifest-root", required=True)
    ap.add_argument("--case-min", type=int, default=200)
    ap.add_argument("--case-max", type=int, default=299)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    if os.path.exists(args.output):
        raise FileExistsError(f"refusing to overwrite {args.output}")

    response = pd.read_feather(
        args.coherent_response,
        columns=KEY + ["R_blend_truth", "R_blend_lsst_r_extnbr_v22"],
    )
    response = response[response.case.between(args.case_min, args.case_max)]
    outer = pd.read_feather(args.outer_features, columns=KEY + FEATURES)
    frame = response.merge(outer, on=KEY, how="inner", validate="one_to_one")
    coverage = len(frame) / len(response)
    if coverage < 0.99:
        raise RuntimeError(f"outer-feature coverage below 99%: {coverage:.3%}")

    manifests = []
    for case in range(args.case_min, args.case_max + 1):
        part = pd.read_feather(
            os.path.join(args.manifest_root, f"anchors_case{case}.feather"),
            columns=["index"],
        ).rename(columns={"index": "input_index"})
        part["case"] = case
        manifests.append(part[KEY])
    manifest = pd.concat(manifests, ignore_index=True)
    frame = frame.merge(
        manifest.assign(retained=True), on=KEY, how="left", validate="one_to_one",
    )
    frame["retained"] = frame["retained"].eq(True)
    frame["gap"] = frame.R_blend_lsst_r_extnbr_v22 - frame.R_blend_truth

    retained = frame[frame.retained]
    excluded = frame[~frame.retained]
    payload = {
        "case_window": [args.case_min, args.case_max],
        "n_rows": int(len(frame)), "outer_feature_coverage": float(coverage),
        "n_retained": int(len(retained)), "n_excluded": int(len(excluded)),
        "features": {}, "gap_quartiles": {}, "constgold_opened": False,
    }
    for feature in FEATURES:
        retained_case = retained.groupby("case")[feature].mean()
        excluded_case = excluded.groupby("case")[feature].mean()
        difference = (retained_case - excluded_case).to_numpy(float)
        payload["features"][feature] = {
            "retained": stat(retained_case), "excluded": stat(excluded_case),
            "retained_minus_excluded": stat(difference),
            "difference_significance_case_sem": float(
                difference.mean() / sem(difference)
            ),
        }

        edges = np.quantile(frame[feature].to_numpy(float), np.linspace(0.0, 1.0, 5))
        edges = np.unique(edges)
        if len(edges) < 5:
            continue
        bin_id = np.clip(
            np.searchsorted(edges, frame[feature].to_numpy(float), side="right") - 1,
            0, 3,
        )
        quartiles = []
        for index in range(4):
            subset = frame[bin_id == index]
            case_gap = subset.groupby("case").gap.mean().to_numpy(float)
            quartiles.append({
                "bin": index, "lo": float(edges[index]), "hi": float(edges[index + 1]),
                "n_rows": int(len(subset)), "gap": stat(case_gap),
            })
        payload["gap_quartiles"][feature] = quartiles

    with open(args.output, "x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    print("ANCHORBLEND_RESPARSIFICATION_DIAGNOSTIC_DONE", flush=True)


if __name__ == "__main__":
    main()
