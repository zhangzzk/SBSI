"""Characterize the pair dominating each coherent-anchor emulator prediction.

Cases 400--449 retain the development role used to freeze the dominance cut;
cases 450--499 are the held-out validation block.  The rendered case is the
uncertainty unit.  Pair properties are true input properties from the matched
latent catalogue; constgold and post-shear detection properties are not read.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


KEY = ["case", "input_index"]
FEATURES = [
    "primary_mag", "primary_size", "secondary_mag", "secondary_size",
    "delta_mag_secondary_minus_primary", "log10_flux_ratio_secondary_primary",
    "flux_ratio_secondary_primary", "distance", "dominant_response",
    "dominant_abs_response", "top_abs_fraction", "n_pairs",
]


def stat(values: np.ndarray) -> dict:
    values = np.asarray(values, float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("stat requires at least two finite values")
    return {
        "mean": float(values.mean()),
        "case_sem": float(values.std(ddof=1) / np.sqrt(len(values))),
        "case_sd": float(values.std(ddof=1)),
        "n_cases": int(len(values)),
    }


def select_dominant_pairs(pairs: pd.DataFrame) -> pd.DataFrame:
    """Select one stable maximum-|response| pair and retain anchor totals."""
    required = {"anchor_index", "secondary_index", "response", "distance"}
    if missing := required - set(pairs):
        raise KeyError(f"pair table lacks {sorted(missing)}")
    if pairs.duplicated(["anchor_index", "secondary_index"]).any():
        raise RuntimeError("duplicate deployed pair")
    work = pairs[["anchor_index", "secondary_index", "response", "distance"]].copy()
    work["abs_response"] = work["response"].abs()
    grouped = work.groupby("anchor_index", sort=False)
    totals = grouped.agg(
        R_model_sum=("response", "sum"),
        R_abs_sum=("abs_response", "sum"),
        n_pairs=("response", "size"),
    )
    ordered = work.sort_values(
        ["anchor_index", "abs_response", "secondary_index"],
        ascending=[True, False, True], kind="mergesort",
    )
    dominant = ordered.groupby("anchor_index", sort=False).head(1).rename(columns={
        "response": "dominant_response",
        "abs_response": "dominant_abs_response",
    })
    dominant = dominant.merge(
        totals, left_on="anchor_index", right_index=True,
        how="left", validate="one_to_one",
    )
    dominant["top_abs_fraction"] = (
        dominant["dominant_abs_response"] / dominant["R_abs_sum"]
    )
    if not np.isfinite(dominant[[
        "dominant_response", "dominant_abs_response", "R_model_sum",
        "R_abs_sum", "top_abs_fraction",
    ]].to_numpy(float)).all():
        raise RuntimeError("non-finite dominant-pair summary")
    return dominant


def conditional_gap_summary(frame: pd.DataFrame, selected: np.ndarray) -> dict:
    work = frame[["case", "gap"]].copy()
    work["selected"] = np.asarray(selected, bool)
    all_cases = np.sort(work.case.unique())
    local = work.loc[work.selected].groupby("case", sort=True).gap.mean()
    complement = work.loc[~work.selected].groupby("case", sort=True).gap.mean()
    paired = local.to_frame("selected").join(
        complement.to_frame("outside"), how="inner",
    )
    contribution = work.assign(
        selected_gap=np.where(work.selected, work.gap, 0.0),
    ).groupby("case", sort=True).agg(
        contribution=("selected_gap", "mean"),
        fraction=("selected", "mean"),
    ).reindex(all_cases, fill_value=0.0)
    outside_contribution = work.assign(
        outside_gap=np.where(~work.selected, work.gap, 0.0),
    ).groupby("case", sort=True).outside_gap.mean().reindex(all_cases, fill_value=0.0)
    contrast = paired.selected - paired.outside
    return {
        "n_selected": int(np.asarray(selected, bool).sum()),
        "n_outside": int((~np.asarray(selected, bool)).sum()),
        "selected_fraction": stat(contribution.fraction.to_numpy(float)),
        "selected_conditional_gap": stat(local.to_numpy(float)),
        "outside_conditional_gap": stat(complement.to_numpy(float)),
        "selected_contribution_to_global_gap": stat(
            contribution.contribution.to_numpy(float)
        ),
        "outside_contribution_to_global_gap": stat(
            outside_contribution.to_numpy(float)
        ),
        "selected_minus_outside_gap": stat(contrast.to_numpy(float)),
        "selected_gap_p_raw": float(stats.ttest_1samp(local, 0.0).pvalue),
        "outside_gap_p_raw": float(stats.ttest_1samp(complement, 0.0).pvalue),
        "selected_minus_outside_p_raw": float(stats.ttest_1samp(contrast, 0.0).pvalue),
    }


def pooled_quantiles(values: pd.Series) -> dict:
    q = values.quantile([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
    return {f"q{int(100 * key):02d}": float(value) for key, value in q.items()}


def feature_comparison(frame: pd.DataFrame, selected: np.ndarray, feature: str) -> dict:
    work = frame[["case", feature]].copy()
    work["selected"] = np.asarray(selected, bool)
    selected_rows = work.loc[work.selected]
    outside_rows = work.loc[~work.selected]
    by_case = work.groupby(["case", "selected"], sort=True)[feature].mean().unstack()
    difference = by_case[True] - by_case[False]
    return {
        "selected_quantiles": pooled_quantiles(selected_rows[feature]),
        "outside_quantiles": pooled_quantiles(outside_rows[feature]),
        "selected_case_mean": stat(
            selected_rows.groupby("case", sort=True)[feature].mean().to_numpy(float)
        ),
        "outside_case_mean": stat(
            outside_rows.groupby("case", sort=True)[feature].mean().to_numpy(float)
        ),
        "selected_minus_outside_case_mean": stat(difference.to_numpy(float)),
        "selected_minus_outside_p_raw": float(
            stats.ttest_1samp(difference.to_numpy(float), 0.0).pvalue
        ),
    }


def fraction_comparison(frame: pd.DataFrame, selected: np.ndarray,
                        condition: np.ndarray) -> dict:
    work = frame[["case"]].copy()
    work["selected"] = np.asarray(selected, bool)
    work["condition"] = np.asarray(condition, bool)
    by_case = work.groupby(["case", "selected"], sort=True).condition.mean().unstack()
    difference = by_case[True] - by_case[False]
    return {
        "selected": stat(by_case[True].to_numpy(float)),
        "outside": stat(by_case[False].to_numpy(float)),
        "selected_minus_outside": stat(difference.to_numpy(float)),
    }


def report(payload: dict) -> str:
    val = payload["blocks"]["validation"]
    gap = val["gap_summary"]
    lines = [
        "# V2.2 dominant-pair domain audit",
        "",
        "Cases 400–449 retain the development role; cases 450–499 are held-out "
        "validation. The case is the uncertainty unit. Pair features are true latent "
        "input properties from the emulator's deployed pair list.",
        "",
        f"Dominant tail: `top_abs_fraction > {payload['dominance_threshold']:.6f}`.",
        "",
        "## Held-out gap split",
        "",
        f"- Tail conditional gap: `{gap['selected_conditional_gap']['mean']:+.6f} +- "
        f"{gap['selected_conditional_gap']['case_sem']:.6f}`.",
        f"- Outside conditional gap: `{gap['outside_conditional_gap']['mean']:+.6f} +- "
        f"{gap['outside_conditional_gap']['case_sem']:.6f}`.",
        f"- Tail minus outside: `{gap['selected_minus_outside_gap']['mean']:+.6f} +- "
        f"{gap['selected_minus_outside_gap']['case_sem']:.6f}`.",
        "",
        "## Held-out pair-property medians",
        "",
        "| Feature | Tail | Outside |",
        "|---|---:|---:|",
    ]
    for feature in FEATURES:
        item = val["feature_comparisons"][feature]
        lines.append(
            f"| `{feature}` | {item['selected_quantiles']['q50']:.6g} | "
            f"{item['outside_quantiles']['q50']:.6g} |"
        )
    lines.extend([
        "", "## Interpretation limits", "",
        "- The tail was defined from emulator predictions, so pair properties are correlated.",
        "- This is descriptive localization, not a fitted correction or causal attribution.",
        "- Every reported dominant pair already passed the deployed V2.2 regression cuts.",
        "- Constgold and post-shear detection properties were not opened.", "",
    ])
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--response", required=True)
    ap.add_argument("--manifest-dir", required=True)
    ap.add_argument("--reference-base", required=True)
    ap.add_argument("--design-json", required=True)
    ap.add_argument("--case-min", type=int, default=400)
    ap.add_argument("--case-max", type=int, default=499)
    ap.add_argument("--development-max", type=int, default=449)
    ap.add_argument("--dominance-threshold", type=float, required=True)
    ap.add_argument("--output-feather", required=True)
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--output-md", required=True)
    args = ap.parse_args()
    for output in (args.output_feather, args.output_json, args.output_md):
        if os.path.exists(output):
            raise FileExistsError(f"refusing existing output {output}")

    response = pd.read_feather(args.response, columns=[
        "case", "input_index", "R_blend_truth", "R_blend_lsst_r_extnbr_v22",
    ]).rename(columns={
        "R_blend_truth": "R_coherent_truth",
        "R_blend_lsst_r_extnbr_v22": "R_stored_model_sum",
    })
    response = response.loc[response.case.between(args.case_min, args.case_max)].copy()
    if response.duplicated(KEY).any():
        raise RuntimeError("duplicate coherent response key")
    with open(args.design_json, encoding="utf-8") as handle:
        design = json.load(handle)
    cuts = design["regression_cuts"]
    if len(cuts) != 5:
        raise RuntimeError(f"expected five regression cuts, got {cuts}")

    parts = []
    root = Path(args.manifest_dir)
    for case in range(args.case_min, args.case_max + 1):
        anchors = pd.read_feather(
            root / f"anchors_case{case}.feather", columns=["index", "r", "Re"],
        ).rename(columns={
            "index": "input_index", "r": "primary_mag", "Re": "primary_size",
        })
        pairs = pd.read_feather(root / f"pairs_case{case}.feather")
        dominant = select_dominant_pairs(pairs).rename(columns={
            "anchor_index": "input_index",
        })
        latent = pd.read_feather(
            Path(args.reference_base) / f"gals{case}_0.05.feather",
            columns=["index", "r", "Re", "sersic_n", "axis_ratio", "redshift"],
        ).rename(columns={
            "index": "secondary_index", "r": "secondary_mag",
            "Re": "secondary_size", "sersic_n": "secondary_sersic_n",
            "axis_ratio": "secondary_axis_ratio", "redshift": "secondary_redshift",
        })
        local = anchors.merge(dominant, on="input_index", how="inner", validate="one_to_one")
        local = local.merge(latent, on="secondary_index", how="left", validate="many_to_one")
        local.insert(0, "case", int(case))
        local = local.merge(
            response.loc[response.case == case], on=KEY, how="inner", validate="one_to_one",
        )
        replay = local.R_model_sum - local.R_stored_model_sum
        if float(np.max(np.abs(replay))) > 1e-2 or abs(float(replay.mean())) > 1e-5:
            raise RuntimeError(f"case {case}: model replay is material")
        local["gap"] = local.R_stored_model_sum - local.R_coherent_truth
        local["delta_mag_secondary_minus_primary"] = (
            local.secondary_mag - local.primary_mag
        )
        local["log10_flux_ratio_secondary_primary"] = (
            -0.4 * local.delta_mag_secondary_minus_primary
        )
        local["flux_ratio_secondary_primary"] = np.power(
            10.0, local.log10_flux_ratio_secondary_primary,
        )
        local["is_dominant_tail"] = (
            local.top_abs_fraction > args.dominance_threshold
        )
        if not np.isfinite(local[[*FEATURES, "gap"]].to_numpy(float)).all():
            raise RuntimeError(f"case {case}: non-finite final feature")
        parts.append(local)
        print(
            f"case {case}: coherent-common={len(local):,} "
            f"tail={local.is_dominant_tail.mean():.2%}", flush=True,
        )
    frame = pd.concat(parts, ignore_index=True)
    frame.to_feather(args.output_feather)

    # The deployed-pair domain is [secondary mag, primary mag, secondary size,
    # primary size, distance]. Strict cuts mirror blendemu.data_utils.
    domain = (
        frame.secondary_mag.between(*cuts[0], inclusive="neither")
        & frame.primary_mag.between(*cuts[1], inclusive="neither")
        & frame.secondary_size.between(*cuts[2], inclusive="neither")
        & frame.primary_size.between(*cuts[3], inclusive="neither")
        & frame.distance.between(*cuts[4], inclusive="neither")
    )
    if not domain.all():
        raise RuntimeError(f"{int((~domain).sum())} dominant pairs outside deployed cuts")

    threshold_definitions = {
        "secondary_brighter_than_primary": frame.flux_ratio_secondary_primary > 1.0,
        "secondary_at_least_half_primary_flux": frame.flux_ratio_secondary_primary >= 0.5,
        "secondary_at_least_twice_primary_flux": frame.flux_ratio_secondary_primary >= 2.0,
        "secondary_at_least_ten_times_primary_flux": frame.flux_ratio_secondary_primary >= 10.0,
        "secondary_mag_lt18": frame.secondary_mag < 18.0,
        "secondary_mag_gt28": frame.secondary_mag > 28.0,
        "secondary_size_lt0p1": frame.secondary_size < 0.1,
        "secondary_size_gt1p5": frame.secondary_size > 1.5,
        "distance_lt1": frame.distance < 1.0,
        "distance_lt2": frame.distance < 2.0,
    }
    payload = {
        "design": (
            "dominant pair is stable argmax absolute emulator response; c400--449 "
            "development and c450--499 held-out validation; case is uncertainty unit"
        ),
        "case_window": [args.case_min, args.case_max],
        "development_window": [args.case_min, args.development_max],
        "validation_window": [args.development_max + 1, args.case_max],
        "dominance_threshold": float(args.dominance_threshold),
        "regression_cuts": cuts,
        "n_rows": int(len(frame)),
        "n_cases": int(frame.case.nunique()),
        "domain_audit": {
            "n_outside_deployed_regression_domain": int((~domain).sum()),
            "cut_order": [
                "secondary_mag", "primary_mag", "secondary_size",
                "primary_size", "distance",
            ],
        },
        "blocks": {},
        "constgold_opened": False,
    }
    for name, block in (
        ("development", frame.loc[frame.case <= args.development_max]),
        ("validation", frame.loc[frame.case > args.development_max]),
    ):
        selected = block.is_dominant_tail.to_numpy(bool)
        payload["blocks"][name] = {
            "n_rows": int(len(block)),
            "gap_summary": conditional_gap_summary(block, selected),
            "feature_comparisons": {
                feature: feature_comparison(block, selected, feature)
                for feature in FEATURES
            },
            "threshold_fractions": {
                key: fraction_comparison(
                    block, selected, condition.loc[block.index].to_numpy(bool),
                )
                for key, condition in threshold_definitions.items()
            },
        }
    with open(args.output_json, "x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    with open(args.output_md, "x", encoding="utf-8") as handle:
        handle.write(report(payload))
    print(json.dumps({
        "n_rows": payload["n_rows"],
        "development_gap": payload["blocks"]["development"]["gap_summary"],
        "validation_gap": payload["blocks"]["validation"]["gap_summary"],
        "validation_flux_ratio": payload["blocks"]["validation"][
            "feature_comparisons"
        ]["flux_ratio_secondary_primary"],
        "validation_threshold_fractions": payload["blocks"]["validation"][
            "threshold_fractions"
        ],
    }, indent=2, sort_keys=True, allow_nan=False))
    print("ANCHORBLEND_DOMINANT_PAIR_DOMAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
