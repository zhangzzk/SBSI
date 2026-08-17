"""Recompute the coherent V2.2 gap after true bright-secondary cuts."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


KEY = ["case", "input_index"]


def stat(values: np.ndarray) -> dict:
    values = np.asarray(values, float)
    return {
        "mean": float(values.mean()),
        "case_sem": float(values.std(ddof=1) / np.sqrt(len(values))),
        "case_sd": float(values.std(ddof=1)),
        "n_cases": int(len(values)),
    }


def summarize_cut(frame: pd.DataFrame, remove: np.ndarray) -> dict:
    work = frame[["case", "gap"]].copy()
    work["remove"] = np.asarray(remove, bool)
    by_case = work.groupby("case", sort=True).agg(
        baseline_gap=("gap", "mean"),
        removed_fraction=("remove", "mean"),
    )
    kept = work.loc[~work.remove].groupby("case", sort=True).gap.mean()
    removed = work.loc[work.remove].groupby("case", sort=True).gap.mean()
    contribution = work.assign(
        kept_gap=np.where(~work.remove, work.gap, 0.0),
        removed_gap=np.where(work.remove, work.gap, 0.0),
    ).groupby("case", sort=True).agg(
        kept_contribution=("kept_gap", "mean"),
        removed_contribution=("removed_gap", "mean"),
    )
    by_case = by_case.join(kept.rename("kept_gap"), how="inner")
    by_case = by_case.join(removed.rename("removed_gap"), how="inner")
    by_case = by_case.join(contribution, how="inner")
    by_case["kept_minus_baseline"] = by_case.kept_gap - by_case.baseline_gap
    by_case["removed_minus_kept"] = by_case.removed_gap - by_case.kept_gap
    return {
        "n_rows": int(len(work)),
        "n_removed": int(work.remove.sum()),
        "removed_fraction": stat(by_case.removed_fraction.to_numpy(float)),
        "baseline_gap": stat(by_case.baseline_gap.to_numpy(float)),
        "kept_gap": stat(by_case.kept_gap.to_numpy(float)),
        "removed_gap": stat(by_case.removed_gap.to_numpy(float)),
        "kept_contribution_to_baseline": stat(
            by_case.kept_contribution.to_numpy(float)
        ),
        "removed_contribution_to_baseline": stat(
            by_case.removed_contribution.to_numpy(float)
        ),
        "kept_minus_baseline": stat(by_case.kept_minus_baseline.to_numpy(float)),
        "removed_minus_kept": stat(by_case.removed_minus_kept.to_numpy(float)),
        "kept_gap_p_raw": float(stats.ttest_1samp(by_case.kept_gap, 0.0).pvalue),
        "removed_minus_kept_p_raw": float(
            stats.ttest_1samp(by_case.removed_minus_kept, 0.0).pvalue
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--response", required=True)
    ap.add_argument("--dominant-table", required=True)
    ap.add_argument("--manifest-dir", required=True)
    ap.add_argument("--reference-base", required=True)
    ap.add_argument("--ratio-threshold", type=float, default=5.0)
    ap.add_argument("--case-min", type=int, default=400)
    ap.add_argument("--case-max", type=int, default=499)
    ap.add_argument("--development-max", type=int, default=449)
    ap.add_argument("--output-json", required=True)
    args = ap.parse_args()
    if os.path.exists(args.output_json):
        raise FileExistsError(f"refusing existing output {args.output_json}")

    response = pd.read_feather(args.response, columns=[
        "case", "input_index", "R_blend_truth", "R_blend_lsst_r_extnbr_v22",
    ]).rename(columns={
        "R_blend_truth": "truth", "R_blend_lsst_r_extnbr_v22": "model",
    })
    response = response.loc[response.case.between(args.case_min, args.case_max)].copy()
    response["gap"] = response.model - response.truth
    dominant = pd.read_feather(args.dominant_table, columns=[
        "case", "input_index", "flux_ratio_secondary_primary",
    ]).rename(columns={
        "flux_ratio_secondary_primary": "dominant_pair_flux_ratio",
    })

    root = Path(args.manifest_dir)
    parts = []
    for case in range(args.case_min, args.case_max + 1):
        anchors = pd.read_feather(
            root / f"anchors_case{case}.feather", columns=["index", "r"],
        ).rename(columns={"index": "input_index", "r": "primary_mag"})
        pairs = pd.read_feather(
            root / f"pairs_case{case}.feather",
            columns=["anchor_index", "secondary_index"],
        )
        latent = pd.read_feather(
            Path(args.reference_base) / f"gals{case}_0.05.feather",
            columns=["index", "r"],
        ).rename(columns={"index": "secondary_index", "r": "secondary_mag"})
        pair_features = pairs.merge(
            anchors.rename(columns={"input_index": "anchor_index"}),
            on="anchor_index", how="left", validate="many_to_one",
        ).merge(latent, on="secondary_index", how="left", validate="many_to_one")
        pair_features["flux_ratio"] = np.power(
            10.0, -0.4 * (pair_features.secondary_mag - pair_features.primary_mag),
        )
        maximum = pair_features.groupby("anchor_index", sort=False).flux_ratio.max()
        local = response.loc[response.case == case].merge(
            maximum.rename("max_deployed_flux_ratio"),
            left_on="input_index", right_index=True, how="inner", validate="one_to_one",
        ).merge(
            dominant.loc[dominant.case == case], on=KEY,
            how="inner", validate="one_to_one",
        )
        if not np.isfinite(local[[
            "gap", "max_deployed_flux_ratio", "dominant_pair_flux_ratio",
        ]].to_numpy(float)).all():
            raise RuntimeError(f"case {case}: non-finite cut coordinate")
        parts.append(local)
        print(
            f"case {case}: common={len(local):,} "
            f"any>{args.ratio_threshold:g}="
            f"{(local.max_deployed_flux_ratio > args.ratio_threshold).mean():.2%}",
            flush=True,
        )
    frame = pd.concat(parts, ignore_index=True)
    payload = {
        "design": (
            "true input flux-ratio diagnostic on coherent-common anchors; "
            "gap is emulator minus coherent truth; rendered case is uncertainty unit"
        ),
        "ratio_threshold": float(args.ratio_threshold),
        "case_window": [args.case_min, args.case_max],
        "development_window": [args.case_min, args.development_max],
        "validation_window": [args.development_max + 1, args.case_max],
        "n_rows": int(len(frame)),
        "blocks": {},
        "constgold_opened": False,
    }
    for block_name, block in (
        ("development", frame.loc[frame.case <= args.development_max]),
        ("validation", frame.loc[frame.case > args.development_max]),
    ):
        payload["blocks"][block_name] = {
            "remove_if_dominant_response_pair_exceeds_threshold": summarize_cut(
                block, block.dominant_pair_flux_ratio.to_numpy(float) > args.ratio_threshold,
            ),
            "remove_if_any_deployed_pair_exceeds_threshold": summarize_cut(
                block, block.max_deployed_flux_ratio.to_numpy(float) > args.ratio_threshold,
            ),
        }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    print("ANCHORBLEND_BRIGHT_SECONDARY_CUT_DONE", flush=True)


if __name__ == "__main__":
    main()
