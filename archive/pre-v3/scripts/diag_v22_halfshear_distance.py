"""Test the constgold nearest-distance localization on half-shear self response.

This uses the fresh c40--139 half-shear rows and the frozen 16-seed V2.2 flow.
There is no BlendEMU term: the diagnostic is exactly ``R_self - R_flow`` in
fixed nearest-neighbour distance groups.  Cases 40--89 and 90--139 are reported
separately to match the constgold localization split.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
import pyarrow.feather as pf
from scipy import stats

from scripts.localize_anchorblend_coherent_gap import stat


SEEDS = [501, 502, 503, *range(505, 518)]


def summarize(frame: pd.DataFrame, mask: np.ndarray) -> dict:
    local = frame.loc[mask]
    by_case = local.groupby("case", sort=True).agg(
        R_self=("R_self", "mean"), R_flow=("R_flow", "mean"),
        gap=("gap", "mean"), n=("gap", "size"),
    )
    m = 100.0 * (by_case.R_self.to_numpy(float) / by_case.R_flow.to_numpy(float) - 1.0)
    return {
        "n_rows": int(len(local)),
        "fraction": float(np.mean(mask)),
        "R_self": stat(by_case.R_self.to_numpy(float)),
        "R_flow": stat(by_case.R_flow.to_numpy(float)),
        "gap_R_self_minus_R_flow": stat(by_case.gap.to_numpy(float)),
        "m_self_percent": stat(m),
    }


def summarize_block(frame: pd.DataFrame) -> dict:
    neighbored = frame.neighbored.to_numpy(bool)
    distance = frame.distance.to_numpy(float)
    masks = {
        "not_flagged": ~neighbored,
        "flagged_0_1_arcsec": neighbored & (distance >= 0.0) & (distance < 1.0),
        "flagged_1_2_arcsec": neighbored & (distance >= 1.0) & (distance < 2.0),
        "flagged_2_3p01_arcsec": neighbored & (distance >= 2.0) & (distance < 3.01),
    }
    output = {name: summarize(frame, mask) for name, mask in masks.items()}
    # Directly test the sign-changing pair that localized constgold.
    rows = []
    for name in ("flagged_1_2_arcsec", "flagged_2_3p01_arcsec"):
        mask = masks[name]
        values = frame.loc[mask].groupby("case", sort=True).gap.mean()
        rows.append(values.rename(name))
    paired = pd.concat(rows, axis=1, join="inner")
    contrast = (
        paired["flagged_2_3p01_arcsec"].to_numpy(float)
        - paired["flagged_1_2_arcsec"].to_numpy(float)
    )
    test = stats.ttest_1samp(contrast, popmean=0.0)
    output["contrast_2_3_minus_1_2"] = {
        "gap": stat(contrast), "paired_t": float(test.statistic),
        "paired_p": float(test.pvalue),
    }
    return output


def compare_constgold(frame: pd.DataFrame, constgold: pd.DataFrame) -> dict:
    """Case-paired constgold total gap minus half-shear self gap by fixed group."""
    def masks(local: pd.DataFrame) -> dict[str, np.ndarray]:
        neighbored = local.neighbored.to_numpy(bool)
        distance = local.distance.to_numpy(float)
        return {
            "not_flagged": ~neighbored,
            "flagged_0_1_arcsec": neighbored & (distance >= 0.0) & (distance < 1.0),
            "flagged_1_2_arcsec": neighbored & (distance >= 1.0) & (distance < 2.0),
            "flagged_2_3p01_arcsec": neighbored & (distance >= 2.0) & (distance < 3.01),
        }

    output = {}
    half_masks = masks(frame)
    const_masks = masks(constgold)
    for name in half_masks:
        half = frame.loc[half_masks[name]].groupby("case", sort=True).gap.mean()
        constant = constgold.loc[const_masks[name]].groupby("case", sort=True).gap.mean()
        paired = pd.concat([constant.rename("constgold"), half.rename("halfshear")], axis=1).dropna()
        difference = paired.constgold.to_numpy(float) - paired.halfshear.to_numpy(float)
        test = stats.ttest_1samp(difference, popmean=0.0)
        output[name] = {
            "constgold_gap": stat(paired.constgold.to_numpy(float)),
            "halfshear_gap": stat(paired.halfshear.to_numpy(float)),
            "constgold_minus_halfshear_gap": stat(difference),
            "paired_t": float(test.statistic), "paired_p": float(test.pvalue),
        }
    constant = constgold.groupby("case", sort=True).gap.mean()
    half = frame.groupby("case", sort=True).gap.mean()
    paired = pd.concat([constant.rename("constgold"), half.rename("halfshear")], axis=1).dropna()
    difference = paired.constgold.to_numpy(float) - paired.halfshear.to_numpy(float)
    test = stats.ttest_1samp(difference, popmean=0.0)
    output["global"] = {
        "constgold_gap": stat(paired.constgold.to_numpy(float)),
        "halfshear_gap": stat(paired.halfshear.to_numpy(float)),
        "constgold_minus_halfshear_gap": stat(difference),
        "paired_t": float(test.statistic), "paired_p": float(test.pvalue),
        "warning": "case-paired but not same-object; detection-conditioned populations differ",
    }
    return output


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True)
    ap.add_argument("--selfresp", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--constgold-feather", default=None,
                    help="optional compact constgold gap table for case-paired branch comparison")
    ap.add_argument("--case-min", type=int, default=40)
    ap.add_argument("--case-max", type=int, default=139)
    ap.add_argument("--development-max", type=int, default=89)
    ap.add_argument("--mag-max", type=float, default=25.8)
    ap.add_argument("--re-min", type=float, default=0.5)
    args = ap.parse_args()
    if os.path.exists(args.output):
        raise FileExistsError(args.output)

    base = pf.read_table(
        args.base, columns=["case", "input_index", "neighbored", "distance",
                            "r_input_p", "Re_input_p"],
    ).to_pandas()
    seed_columns = [f"R_flow_s{seed}" for seed in SEEDS]
    response = pf.read_table(
        args.selfresp, columns=["case", "input_index", "r_sim_self", *seed_columns],
    ).to_pandas()
    if not base[["case", "input_index"]].equals(response[["case", "input_index"]]):
        raise RuntimeError("half-shear base/self-response keys differ")
    use = (
        base.case.between(args.case_min, args.case_max)
        & (base.r_input_p.to_numpy(float) < args.mag_max)
        & (base.Re_input_p.to_numpy(float) > args.re_min)
    )
    finite = np.isfinite(response[["r_sim_self", *seed_columns]].to_numpy(float)).all(axis=1)
    use &= finite
    frame = base.loc[
        use, ["case", "input_index", "neighbored", "distance"]
    ].reset_index(drop=True)
    frame["R_self"] = response.loc[use, "r_sim_self"].to_numpy(float)
    frame["R_flow"] = response.loc[use, seed_columns].to_numpy(float).mean(axis=1)
    frame["gap"] = frame.R_self - frame.R_flow
    development = frame.loc[frame.case <= args.development_max].copy()
    validation = frame.loc[frame.case > args.development_max].copy()
    if development.case.nunique() != 50 or validation.case.nunique() != 50:
        raise RuntimeError("expected two 50-case blocks")
    payload = {
        "design": (
            "fresh half-shear self response c40--139; fixed nearest-distance bins; "
            "16-seed V2.2 flow; no emulator term; case is uncertainty unit"
        ),
        "domain": {"primary_mag_max": args.mag_max, "primary_size_min": args.re_min},
        "n_rows": int(len(frame)),
        "n_seeds": len(SEEDS),
        "development_window": [args.case_min, args.development_max],
        "validation_window": [args.development_max + 1, args.case_max],
        "development": summarize_block(development),
        "validation": summarize_block(validation),
    }
    if args.constgold_feather:
        constgold = pf.read_table(
            args.constgold_feather,
            columns=["case", "input_index", "neighbored", "distance", "gap"],
        ).to_pandas()
        constgold = constgold.loc[
            constgold.case.between(args.case_min, args.case_max)
        ].copy()
        const_development = constgold.loc[constgold.case <= args.development_max].copy()
        const_validation = constgold.loc[constgold.case > args.development_max].copy()
        payload["constgold_minus_halfshear_case_paired"] = {
            "development": compare_constgold(development, const_development),
            "validation": compare_constgold(validation, const_validation),
        }
    with open(args.output, "x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    print("V22_HALFSHEAR_DISTANCE_DONE", flush=True)


if __name__ == "__main__":
    main()
