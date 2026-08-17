"""Explain what the V2.2 maximum-pair response-dominance tail represents."""
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

from blendemu import data_utils  # noqa: E402
from blendemu.inference import BlendingPredictor  # noqa: E402


COND = dict(pixel_size=0.2, zero_point=30.0, psf_fwhm=0.73,
            moffat_beta=2.224, pixel_rms=0.312)
MODEL_DIR = "/home/z/Zekang.Zhang/blendemu/models"
SCALED_FEATURES = [
    "Re_input_p_scaled", "Re_input_s_scaled", "r_input_p_scaled",
    "r_input_s_scaled", "sersic_n_input_p", "sersic_n_input_s",
    "distance_scaled",
]
SUMMARY_FEATURES = [
    "top_abs_fraction", "dominant_abs_response", "runner_up_abs_response",
    "other_abs_response_sum", "dominant_to_runner_up_abs_response",
    "n_response_pairs_ge_10pct_max", "dominant_flux_ratio_primary",
    "dominant_flux_share_neighbours", "dominant_secondary_mag",
    "dominant_secondary_size", "dominant_size_ratio_primary",
    "dominant_distance", "dominant_flux_rank", "dominant_distance_rank",
    *SCALED_FEATURES,
]


def stat(values: np.ndarray) -> dict:
    values = np.asarray(values, float)
    return {
        "mean": float(values.mean()),
        "case_sem": float(values.std(ddof=1) / np.sqrt(len(values))),
        "n_cases": int(len(values)),
    }


def quantiles(values: pd.Series) -> dict:
    result = values.quantile([0.05, 0.25, 0.50, 0.75, 0.95])
    return {f"q{int(100 * q):02d}": float(v) for q, v in result.items()}


def build_case(case: int, manifest_dir: Path, reference_base: Path,
               threshold: float, boundaries: np.ndarray,
               reference_shear: str = "0.05",
               pair_prefix: str = "pairs") -> pd.DataFrame:
    anchors = pd.read_feather(
        manifest_dir / f"anchors_case{case}.feather",
        columns=["index", "r", "Re"],
    ).rename(columns={
        "index": "anchor_index", "r": "r_input_p", "Re": "Re_input_p",
    })
    pairs = pd.read_feather(
        manifest_dir / f"{pair_prefix}_case{case}.feather",
        columns=["anchor_index", "secondary_index", "response", "distance"],
    )
    latent = pd.read_feather(
        reference_base / f"gals{case}_{reference_shear}.feather",
        columns=["index", "r", "Re", "sersic_n"],
    ).rename(columns={
        "index": "secondary_index", "r": "r_input_s", "Re": "Re_input_s",
        "sersic_n": "sersic_n_input_s",
    })
    # Primary Sérsic is read from the same latent catalogue to avoid relying on
    # the intentionally compact anchor manifest.
    primary_latent = latent.rename(columns={
        "secondary_index": "anchor_index", "r_input_s": "primary_r_replay",
        "Re_input_s": "primary_Re_replay",
        "sersic_n_input_s": "sersic_n_input_p",
    })[["anchor_index", "primary_r_replay", "primary_Re_replay", "sersic_n_input_p"]]
    anchors = anchors.merge(primary_latent, on="anchor_index", validate="one_to_one")
    if not np.allclose(anchors.r_input_p, anchors.primary_r_replay, rtol=0, atol=0):
        raise RuntimeError(f"case {case}: primary magnitude replay mismatch")
    if not np.allclose(anchors.Re_input_p, anchors.primary_Re_replay, rtol=0, atol=0):
        raise RuntimeError(f"case {case}: primary size replay mismatch")
    anchors = anchors.drop(columns=["primary_r_replay", "primary_Re_replay"])

    work = pairs.merge(anchors, on="anchor_index", how="left", validate="many_to_one")
    work = work.merge(latent, on="secondary_index", how="left", validate="many_to_one")
    if work[["r_input_p", "Re_input_p", "r_input_s", "Re_input_s"]].isna().any().any():
        raise RuntimeError(f"case {case}: missing pair latent properties")
    work["abs_response"] = work.response.abs()
    work["secondary_flux"] = np.power(10.0, -0.4 * work.r_input_s)
    work["primary_flux"] = np.power(10.0, -0.4 * work.r_input_p)
    work["flux_ratio_primary"] = work.secondary_flux / work.primary_flux

    # Stable response, flux, and distance ranks within every anchor.
    response_order = work.sort_values(
        ["anchor_index", "abs_response", "secondary_index"],
        ascending=[True, False, True], kind="mergesort",
    ).copy()
    response_order["response_rank"] = response_order.groupby(
        "anchor_index", sort=False,
    ).cumcount() + 1
    flux_order = work.sort_values(
        ["anchor_index", "secondary_flux", "secondary_index"],
        ascending=[True, False, True], kind="mergesort",
    ).copy()
    flux_order["flux_rank"] = flux_order.groupby("anchor_index", sort=False).cumcount() + 1
    distance_order = work.sort_values(
        ["anchor_index", "distance", "secondary_index"],
        ascending=[True, True, True], kind="mergesort",
    ).copy()
    distance_order["distance_rank"] = distance_order.groupby(
        "anchor_index", sort=False,
    ).cumcount() + 1
    rank_lookup = flux_order[["anchor_index", "secondary_index", "flux_rank"]].merge(
        distance_order[["anchor_index", "secondary_index", "distance_rank"]],
        on=["anchor_index", "secondary_index"], validate="one_to_one",
    )
    response_order = response_order.merge(
        rank_lookup, on=["anchor_index", "secondary_index"], validate="one_to_one",
    )

    grouped = work.groupby("anchor_index", sort=False)
    totals = grouped.agg(
        R_abs_sum=("abs_response", "sum"),
        neighbour_flux_sum=("secondary_flux", "sum"),
        n_pairs=("response", "size"),
    )
    maximum = grouped.abs_response.max().rename("maximum")
    strong = work.assign(
        strong=work.abs_response >= 0.1 * work.anchor_index.map(maximum),
    ).groupby("anchor_index", sort=False).strong.sum().rename(
        "n_response_pairs_ge_10pct_max"
    )
    dominant = response_order.loc[response_order.response_rank == 1].copy()
    runner = response_order.loc[response_order.response_rank == 2, [
        "anchor_index", "abs_response",
    ]].rename(columns={"abs_response": "runner_up_abs_response"})
    dominant = dominant.merge(totals, on="anchor_index", validate="one_to_one")
    dominant = dominant.merge(
        runner, on="anchor_index", how="left", validate="one_to_one",
    )
    dominant["runner_up_abs_response"] = dominant.runner_up_abs_response.fillna(0.0)
    dominant = dominant.merge(strong, on="anchor_index", validate="one_to_one")
    dominant["top_abs_fraction"] = dominant.abs_response / dominant.R_abs_sum
    dominant["other_abs_response_sum"] = dominant.R_abs_sum - dominant.abs_response
    dominant["dominant_to_runner_up_abs_response"] = (
        dominant.abs_response / np.maximum(dominant.runner_up_abs_response, 1.0e-12)
    )
    dominant["dominant_flux_share_neighbours"] = (
        dominant.secondary_flux / dominant.neighbour_flux_sum
    )
    dominant["dominant_size_ratio_primary"] = (
        dominant.Re_input_s / dominant.Re_input_p
    )
    dominant["is_response_dominant_tail"] = dominant.top_abs_fraction > threshold
    dominant = dominant.rename(columns={
        "abs_response": "dominant_abs_response",
        "flux_ratio_primary": "dominant_flux_ratio_primary",
        "r_input_s": "dominant_secondary_mag",
        "Re_input_s": "dominant_secondary_size",
        "distance": "dominant_distance",
        "flux_rank": "dominant_flux_rank",
        "distance_rank": "dominant_distance_rank",
    })

    raw = dominant.rename(columns={
        "dominant_secondary_mag": "r_input_s",
        "dominant_secondary_size": "Re_input_s",
        "dominant_distance": "distance",
    })
    scaled = data_utils.rescale(
        raw[[
            "Re_input_p", "Re_input_s", "r_input_p", "r_input_s",
            "sersic_n_input_p", "sersic_n_input_s", "distance",
        ]].copy(), **{
            "pixel_rms": COND["pixel_rms"], "pixel_size": COND["pixel_size"],
            "zero_mag": COND["zero_point"], "psf_fwhm": COND["psf_fwhm"],
            "moffat_beta": COND["moffat_beta"],
        }
    )
    for index, feature in enumerate(SCALED_FEATURES):
        dominant[feature] = scaled[feature].to_numpy(float)
        dominant[f"oob_{feature}"] = (
            (dominant[feature] < boundaries[index, 0])
            | (dominant[feature] > boundaries[index, 1])
        )
    dominant["any_scaled_feature_oob"] = dominant[
        [f"oob_{feature}" for feature in SCALED_FEATURES]
    ].any(axis=1)
    dominant.insert(0, "case", int(case))
    if not np.isfinite(dominant[SUMMARY_FEATURES].to_numpy(float)).all():
        raise RuntimeError(f"case {case}: non-finite dominance feature")
    return dominant


def summarize_block(frame: pd.DataFrame) -> dict:
    tail = frame.is_response_dominant_tail.to_numpy(bool)
    result = {
        "n_rows": int(len(frame)),
        "n_tail": int(tail.sum()),
        "tail_fraction": stat(
            frame.assign(x=tail.astype(float)).groupby("case").x.mean().to_numpy(float)
        ),
        "features": {},
        "fractions": {},
    }
    for feature in SUMMARY_FEATURES:
        selected = frame.loc[tail]
        outside = frame.loc[~tail]
        selected_case = selected.groupby("case", sort=True)[feature].mean()
        outside_case = outside.groupby("case", sort=True)[feature].mean()
        paired = selected_case.to_frame("tail").join(
            outside_case.to_frame("outside"), how="inner",
        )
        result["features"][feature] = {
            "tail_quantiles": quantiles(selected[feature]),
            "outside_quantiles": quantiles(outside[feature]),
            "tail_case_mean": stat(selected_case.to_numpy(float)),
            "outside_case_mean": stat(outside_case.to_numpy(float)),
            "tail_minus_outside_case_mean": stat(
                (paired["tail"] - paired["outside"]).to_numpy(float)
            ),
        }
    predicates = {
        "dominant_pair_is_brightest_neighbour": frame.dominant_flux_rank == 1,
        "dominant_pair_is_closest_neighbour": frame.dominant_distance_rank == 1,
        "dominant_pair_carries_majority_neighbour_flux": (
            frame.dominant_flux_share_neighbours > 0.5
        ),
        "dominant_pair_brighter_than_primary": frame.dominant_flux_ratio_primary > 1.0,
        "dominant_pair_flux_ratio_gt5": frame.dominant_flux_ratio_primary > 5.0,
        "dominant_pair_abs_response_gt0p1": frame.dominant_abs_response > 0.1,
        "dominant_pair_abs_response_gt0p2": frame.dominant_abs_response > 0.2,
        "dominant_pair_at_least_10x_runner_up": (
            frame.dominant_to_runner_up_abs_response >= 10.0
        ),
        "only_one_pair_above_10pct_max": frame.n_response_pairs_ge_10pct_max == 1,
        "any_scaled_feature_oob": frame.any_scaled_feature_oob,
    }
    predicates.update({
        f"oob_{feature}": frame[f"oob_{feature}"] for feature in SCALED_FEATURES
    })
    for name, values in predicates.items():
        work = frame[["case"]].copy()
        work["tail"] = tail
        work["value"] = np.asarray(values, bool).astype(float)
        by_case = work.groupby(["case", "tail"], sort=True).value.mean().unstack()
        result["fractions"][name] = {
            "tail": stat(by_case[True].to_numpy(float)),
            "outside": stat(by_case[False].to_numpy(float)),
            "tail_minus_outside": stat(
                (by_case[True] - by_case[False]).to_numpy(float)
            ),
        }
    if "gap" in frame:
        gap_masks = {
            "response_dominant_tail": tail,
            "outside_response_dominant_tail": ~tail,
            "tail_and_brightest_neighbour": tail & (frame.dominant_flux_rank == 1),
            "tail_and_not_brightest_neighbour": tail & (frame.dominant_flux_rank > 1),
            "tail_and_majority_neighbour_flux": tail & (
                frame.dominant_flux_share_neighbours > 0.5
            ),
            "tail_and_not_majority_neighbour_flux": tail & (
                frame.dominant_flux_share_neighbours <= 0.5
            ),
        }
        result["gap_subgroups"] = {}
        for name, mask in gap_masks.items():
            conditional = frame.loc[mask].groupby("case", sort=True).gap.mean()
            contribution = frame.assign(
                selected_gap=np.where(mask, frame.gap, 0.0),
                selected=np.asarray(mask, bool).astype(float),
            ).groupby("case", sort=True).agg(
                contribution=("selected_gap", "mean"),
                fraction=("selected", "mean"),
            )
            result["gap_subgroups"][name] = {
                "n_rows": int(np.asarray(mask, bool).sum()),
                "fraction": stat(contribution.fraction.to_numpy(float)),
                "conditional_gap": stat(conditional.to_numpy(float)),
                "contribution_to_global_gap": stat(
                    contribution.contribution.to_numpy(float)
                ),
            }
    return result


def markdown(payload: dict) -> str:
    val = payload["blocks"]["validation"]
    lines = [
        "# What V2.2 response dominance means",
        "",
        "The tail is defined only from emulator predictions: maximum absolute pair "
        "response divided by the sum of absolute pair responses exceeds the frozen "
        f"threshold `{payload['dominance_threshold']:.6f}`.",
        "",
        "## Held-out medians", "",
        "| Coordinate | Tail | Outside |", "|---|---:|---:|",
    ]
    for feature in SUMMARY_FEATURES:
        item = val["features"][feature]
        lines.append(
            f"| `{feature}` | {item['tail_quantiles']['q50']:.6g} | "
            f"{item['outside_quantiles']['q50']:.6g} |"
        )
    lines.extend(["", "## Held-out fractions", "", "| Statement | Tail | Outside |",
                  "|---|---:|---:|"])
    for name, item in val["fractions"].items():
        lines.append(
            f"| `{name}` | {item['tail']['mean']:.2%} | {item['outside']['mean']:.2%} |"
        )
    lines.extend([
        "", "## Limits", "",
        "- This is descriptive localization of a model-output-defined tail.",
        "- Model inputs are correlated; no coordinate is identified as causal.",
        "- Constgold and post-shear detection properties were not opened.", "",
    ])
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest-dir", required=True)
    ap.add_argument("--reference-base", required=True)
    ap.add_argument("--reference-shear", default="0.05",
                    help="shear label of the gals{case}_{g}.feather latent "
                         "catalogue inside --reference-base; only the input "
                         "properties (r, Re, sersic_n) are read, so any rendered "
                         "leg of the same case gives identical values")
    ap.add_argument(
        "--pair-prefix", default="pairs",
        help="pair-manifest prefix before _case{case}.feather",
    )
    ap.add_argument("--tag", default="lsst_r_extnbr_v22")
    ap.add_argument("--case-min", type=int, default=400)
    ap.add_argument("--case-max", type=int, default=499)
    ap.add_argument("--development-max", type=int, default=449)
    ap.add_argument("--dominance-threshold", type=float, required=True)
    ap.add_argument("--output-feather", required=True)
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--output-md", required=True)
    ap.add_argument(
        "--common-table",
        help="optional coherent-common table with case,input_index,gap for filtering",
    )
    ap.add_argument(
        "--reuse-feather", action="store_true",
        help="reuse an already completed output feather after a summary-only failure",
    )
    args = ap.parse_args()
    protected_outputs = (args.output_json, args.output_md) if args.reuse_feather else (
        args.output_feather, args.output_json, args.output_md,
    )
    for output in protected_outputs:
        if os.path.exists(output):
            raise FileExistsError(f"refusing existing output {output}")
    predictor = BlendingPredictor.load(
        MODEL_DIR, tag=args.tag, conditions=COND, device="cpu",
    )
    if predictor.bst_reg.feature_names != SCALED_FEATURES:
        raise RuntimeError(
            f"unexpected regression features {predictor.bst_reg.feature_names}"
        )
    boundaries = np.asarray(predictor.boundaries_reg, float)
    if boundaries.shape != (len(SCALED_FEATURES), 2):
        raise RuntimeError(f"unexpected regression boundary shape {boundaries.shape}")

    if args.reuse_feather:
        if not os.path.isfile(args.output_feather):
            raise FileNotFoundError(f"missing reusable feather {args.output_feather}")
        frame = pd.read_feather(args.output_feather)
        expected_cases = set(range(args.case_min, args.case_max + 1))
        observed_cases = set(frame.case.unique().tolist())
        if observed_cases != expected_cases:
            raise RuntimeError(
                f"reusable feather case mismatch: expected {len(expected_cases)}, "
                f"observed {len(observed_cases)}"
            )
        print(f"reused completed per-anchor table: {len(frame):,} rows", flush=True)
    else:
        parts = []
        for case in range(args.case_min, args.case_max + 1):
            local = build_case(
                case, Path(args.manifest_dir), Path(args.reference_base),
                args.dominance_threshold, boundaries,
                reference_shear=args.reference_shear,
                pair_prefix=args.pair_prefix,
            )
            parts.append(local)
            print(
                f"case {case}: anchors={len(local):,} "
                f"tail={local.is_response_dominant_tail.mean():.2%}", flush=True,
            )
        frame = pd.concat(parts, ignore_index=True)
        frame.to_feather(args.output_feather)
    if args.common_table:
        common = pd.read_feather(
            args.common_table, columns=["case", "input_index", "gap"],
        )
        frame = frame.merge(
            common, left_on=["case", "anchor_index"],
            right_on=["case", "input_index"], how="inner", validate="one_to_one",
        )
        print(f"restricted to coherent-common population: {len(frame):,} rows", flush=True)
    payload = {
        "design": (
            "stable maximum-absolute-response pair versus runner-up and remaining "
            "deployed neighbour list; c400--449 development, c450--499 validation"
        ),
        "model_tag": args.tag,
        "pair_prefix": args.pair_prefix,
        "model_features": SCALED_FEATURES,
        "training_boundaries": boundaries.tolist(),
        "dominance_threshold": float(args.dominance_threshold),
        "n_rows": int(len(frame)),
        "summary_population": (
            "coherent-common anchors" if args.common_table else "all frozen anchors"
        ),
        "blocks": {
            "development": summarize_block(frame.loc[frame.case <= args.development_max]),
            "validation": summarize_block(frame.loc[frame.case > args.development_max]),
        },
        "constgold_opened": False,
    }
    with open(args.output_json, "x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    with open(args.output_md, "x", encoding="utf-8") as handle:
        handle.write(markdown(payload))
    print(json.dumps({
        "n_rows": payload["n_rows"],
        "validation": payload["blocks"]["validation"],
    }, indent=2, sort_keys=True, allow_nan=False))
    print("ANCHORBLEND_RESPONSE_DOMINANCE_LOCALIZATION_DONE", flush=True)


if __name__ == "__main__":
    main()
