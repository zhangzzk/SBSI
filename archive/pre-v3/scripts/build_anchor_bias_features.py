"""Build a leak-free per-anchor feature table for V2.2 bias emulation.

The target is the direct coherent-anchor residual

    bias = R_blend_truth - R_blend_V2.2,

so positive values mean V2.2 underpredicts.  Features contain only latent input
properties and frozen V2.2 pair predictions.  Neither the measured truth nor
constgold enters a feature.  Exact renderer-catalogue pair manifests are
summarized in fixed distance shells to expose multi-neighbour scene structure
without rereplaying the model.
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
TARGET = "bias_truth_minus_model"
SHELLS = [
    (0.0, 1.0, "d0_1"),
    (1.0, 2.0, "d1_2"),
    (2.0, 3.0, "d2_3"),
    (3.0, 5.0, "d3_5"),
    (5.0, 7.0, "d5_7"),
    (7.0, 10.000001, "d7_10"),
]

PRIMARY_FEATURES = [
    "primary_mag", "log10_primary_size", "log10_primary_sersic_n",
]
DOMINANT_PHYSICS_FEATURES = [
    "dominant_secondary_mag", "log10_dominant_secondary_size",
    "log10_dominant_secondary_sersic_n", "dominant_distance",
    "log10_flux_ratio", "log10_size_ratio", "log10_overlap_scale",
    "log10_surface_brightness_ratio", "dominant_flux_share_neighbours",
    "log1p_dominant_flux_rank", "log1p_dominant_distance_rank",
]
SCENE_PHYSICS_FEATURES = [
    "has_deployed_pair", "log10_one_plus_total_neighbour_flux_ratio",
    "log1p_n_pairs",
    "min_pair_distance", "mean_pair_distance", "std_pair_distance",
    *[f"log1p_n_pair_{name}" for _, _, name in SHELLS],
]
RESPONSE_FEATURES = [
    "scene_prediction", "abs_scene_prediction", "dominant_response",
    "dominant_abs_response", "runner_up_abs_response",
    "other_abs_response_sum", "R_abs_sum", "top_abs_fraction",
    "log1p_dominant_to_runner_up_abs_response", "signed_to_abs_response",
    "cancelled_abs_response_fraction", "n_response_pairs_ge_10pct_max",
    *[f"R_pair_{name}" for _, _, name in SHELLS],
    *[f"R_abs_pair_{name}" for _, _, name in SHELLS],
]
PHYSICAL_FEATURES = [
    *PRIMARY_FEATURES, *DOMINANT_PHYSICS_FEATURES, *SCENE_PHYSICS_FEATURES,
]
FULL_FEATURES = [*PHYSICAL_FEATURES, *RESPONSE_FEATURES]
FEATURE_SETS = {
    "primary_only": PRIMARY_FEATURES,
    "physical_scene": PHYSICAL_FEATURES,
    "response_structure": RESPONSE_FEATURES,
    "full": FULL_FEATURES,
}
FEATURE_LABELS = {
    "primary_mag": "Primary magnitude",
    "log10_primary_size": "log10 primary Re (arcsec)",
    "log10_primary_sersic_n": "log10 primary Sersic n",
    "dominant_secondary_mag": "Dominant secondary magnitude",
    "log10_dominant_secondary_size": "log10 dominant secondary Re (arcsec)",
    "log10_dominant_secondary_sersic_n": "log10 dominant secondary Sersic n",
    "dominant_distance": "Dominant-pair distance (arcsec)",
    "log10_flux_ratio": "log10(Fs/Fp)",
    "log10_size_ratio": "log10(Res/Rep)",
    "log10_overlap_scale": "log10[(Rep+Res)/distance]",
    "log10_surface_brightness_ratio": "log10(SBs/SBp)",
    "dominant_flux_share_neighbours": "Dominant fraction of neighbour flux",
    "log1p_dominant_flux_rank": "log(1 + dominant flux rank)",
    "log1p_dominant_distance_rank": "log(1 + dominant distance rank)",
    "has_deployed_pair": "Has at least one deployed pair",
    "log10_one_plus_total_neighbour_flux_ratio": (
        "log10(1 + total neighbour flux/Fp)"
    ),
    "log1p_n_pairs": "log(1 + deployed pair count)",
    "min_pair_distance": "Closest deployed-pair distance (arcsec)",
    "mean_pair_distance": "Mean deployed-pair distance (arcsec)",
    "std_pair_distance": "SD of deployed-pair distance (arcsec)",
    "scene_prediction": "V2.2 scene response",
    "abs_scene_prediction": "abs(V2.2 scene response)",
    "dominant_response": "Dominant-pair signed response",
    "dominant_abs_response": "Dominant-pair abs response",
    "runner_up_abs_response": "Runner-up abs response",
    "other_abs_response_sum": "Other-pair abs-response sum",
    "R_abs_sum": "All-pair abs-response sum",
    "top_abs_fraction": "Dominant fraction of abs response",
    "log1p_dominant_to_runner_up_abs_response": (
        "log(1 + dominant/runner-up abs response)"
    ),
    "signed_to_abs_response": "Signed/absolute scene response",
    "cancelled_abs_response_fraction": "Cancelled fraction of abs response",
    "n_response_pairs_ge_10pct_max": "Pairs above 10% of maximum response",
}
for _, _, name in SHELLS:
    label = name.replace("d", "").replace("_", "-")
    FEATURE_LABELS[f"log1p_n_pair_{name}"] = f"log(1 + pair count), {label} arcsec"
    FEATURE_LABELS[f"R_pair_{name}"] = f"Signed response sum, {label} arcsec"
    FEATURE_LABELS[f"R_abs_pair_{name}"] = f"Abs-response sum, {label} arcsec"


def basic_stat(values: np.ndarray) -> dict:
    values = np.asarray(values, float)
    if values.size < 2 or not np.isfinite(values).all():
        raise ValueError("expected at least two finite values")
    sd = float(values.std(ddof=1))
    return {
        "mean": float(values.mean()),
        "sd": sd,
        "sem": sd / np.sqrt(values.size),
        "n": int(values.size),
    }


def pair_scene_summary(pairs: pd.DataFrame,
                       anchors: pd.Index) -> pd.DataFrame:
    required = {"anchor_index", "secondary_index", "distance", "response"}
    if missing := required - set(pairs):
        raise KeyError(f"pair manifest lacks {sorted(missing)}")
    if pairs.duplicated(["anchor_index", "secondary_index"]).any():
        raise RuntimeError("duplicate deployed pair")
    work = pairs[["anchor_index", "distance", "response"]].copy()
    work["abs_response"] = work.response.abs()
    grouped = work.groupby("anchor_index", sort=False)
    out = grouped.agg(
        pair_count_replay=("response", "size"),
        min_pair_distance=("distance", "min"),
        mean_pair_distance=("distance", "mean"),
        std_pair_distance=("distance", "std"),
        R_pair_replay=("response", "sum"),
        R_abs_pair_replay=("abs_response", "sum"),
    ).reindex(anchors)
    for column in ("pair_count_replay", "R_pair_replay", "R_abs_pair_replay"):
        out[column] = out[column].fillna(0.0)
    out.loc[
        (out.pair_count_replay == 1) & out.std_pair_distance.isna(),
        "std_pair_distance",
    ] = 0.0
    distance = work.distance.to_numpy(float)
    for lo, hi, name in SHELLS:
        local = work.loc[(distance >= lo) & (distance < hi)]
        local_group = local.groupby("anchor_index", sort=False)
        out[f"n_pair_{name}"] = local_group.size().reindex(
            anchors, fill_value=0,
        ).to_numpy(np.int64)
        out[f"R_pair_{name}"] = local_group.response.sum().reindex(
            anchors, fill_value=0.0,
        ).to_numpy(float)
        out[f"R_abs_pair_{name}"] = local_group.abs_response.sum().reindex(
            anchors, fill_value=0.0,
        ).to_numpy(float)
    out.index.name = "input_index"
    return out.reset_index()


def derive_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    primary_positive = ["primary_size", "primary_sersic_n", "primary_flux"]
    if (~np.isfinite(frame[primary_positive].to_numpy(float))).any():
        raise RuntimeError("non-finite primary input")
    if (frame[primary_positive].to_numpy(float) <= 0).any():
        raise RuntimeError("non-positive primary input")
    paired = frame.has_deployed_pair.to_numpy(bool)
    paired_positive = [
        "dominant_secondary_size",
        "dominant_secondary_sersic_n", "dominant_distance",
        "dominant_flux_ratio_primary", "dominant_size_ratio_primary",
        "dominant_abs_response", "dominant_to_runner_up_abs_response",
    ]
    if (~np.isfinite(frame.loc[paired, paired_positive].to_numpy(float))).any():
        raise RuntimeError("non-finite paired positive-domain input")
    if (frame.loc[paired, paired_positive].to_numpy(float) <= 0).any():
        raise RuntimeError("non-positive paired positive-domain input")
    frame["log10_primary_size"] = np.log10(frame.primary_size)
    frame["log10_primary_sersic_n"] = np.log10(frame.primary_sersic_n)
    frame["log10_dominant_secondary_size"] = np.log10(
        frame.dominant_secondary_size
    )
    frame["log10_dominant_secondary_sersic_n"] = np.log10(
        frame.dominant_secondary_sersic_n
    )
    frame["log10_flux_ratio"] = np.log10(frame.dominant_flux_ratio_primary)
    frame["log10_size_ratio"] = np.log10(frame.dominant_size_ratio_primary)
    frame["log10_overlap_scale"] = np.log10(
        (frame.primary_size + frame.dominant_secondary_size)
        / frame.dominant_distance
    )
    frame["log10_surface_brightness_ratio"] = (
        frame.log10_flux_ratio - 2.0 * frame.log10_size_ratio
    )
    frame["log10_one_plus_total_neighbour_flux_ratio"] = np.log10(
        1.0 + frame.neighbour_flux_sum / frame.primary_flux
    )
    frame["log1p_dominant_flux_rank"] = np.log1p(frame.dominant_flux_rank)
    frame["log1p_dominant_distance_rank"] = np.log1p(
        frame.dominant_distance_rank
    )
    frame["log1p_n_pairs"] = np.log1p(frame.n_pairs)
    for _, _, name in SHELLS:
        frame[f"log1p_n_pair_{name}"] = np.log1p(frame[f"n_pair_{name}"])
    frame["scene_prediction"] = frame.R_blend_lsst_r_extnbr_v22
    frame["abs_scene_prediction"] = frame.scene_prediction.abs()
    frame["dominant_response"] = frame.response
    frame["log1p_dominant_to_runner_up_abs_response"] = np.log1p(
        frame.dominant_to_runner_up_abs_response
    )
    abs_sum = frame.R_abs_sum.to_numpy(float)
    frame["signed_to_abs_response"] = np.divide(
        frame.scene_prediction.to_numpy(float), abs_sum,
        out=np.zeros(len(frame), dtype=float), where=abs_sum > 0,
    )
    frame["cancelled_abs_response_fraction"] = np.divide(
        abs_sum - frame.abs_scene_prediction.to_numpy(float), abs_sum,
        out=np.zeros(len(frame), dtype=float), where=abs_sum > 0,
    )
    if frame.cancelled_abs_response_fraction.min() < -2.0e-5:
        raise RuntimeError("signed pair sum exceeds absolute pair sum")
    frame[TARGET] = frame.R_blend_truth - frame.scene_prediction
    frame["carrier_ratio_gt5_positive"] = (
        (frame.dominant_to_runner_up_abs_response > 5.0)
        & (frame.dominant_response >= 0.0)
    )
    frame["carrier_top_fraction_gt0p7_positive"] = (
        (frame.top_abs_fraction > 0.7) & (frame.dominant_response >= 0.0)
    )
    required = [*FULL_FEATURES, TARGET]
    values = frame[required].to_numpy(float)
    if np.isinf(values).any():
        raise RuntimeError("infinite derived feature")
    if frame.loc[paired, required].isna().any().any():
        bad = frame.loc[paired, required].isna().sum()
        raise RuntimeError(f"missing paired feature: {bad.loc[bad > 0].to_dict()}")
    return frame


def load_block(response_path: str, dominance_path: str,
               design_path: str) -> tuple[pd.DataFrame, dict]:
    response_columns = [
        *KEY, "R_blend_truth", "R_blend_lsst_r_extnbr_v22",
        "r_input_p_plus", "Re_input_p_plus",
    ]
    dominance_columns = [
        "case", "anchor_index", "response", "dominant_distance",
        "r_input_p", "Re_input_p", "sersic_n_input_p",
        "dominant_secondary_mag", "dominant_secondary_size",
        "sersic_n_input_s", "dominant_abs_response", "primary_flux",
        "dominant_flux_ratio_primary", "dominant_flux_rank",
        "dominant_distance_rank", "R_abs_sum", "neighbour_flux_sum", "n_pairs",
        "runner_up_abs_response", "n_response_pairs_ge_10pct_max",
        "top_abs_fraction", "other_abs_response_sum",
        "dominant_to_runner_up_abs_response", "dominant_flux_share_neighbours",
        "dominant_size_ratio_primary",
    ]
    response = pd.read_feather(response_path, columns=response_columns)
    dominance = pd.read_feather(
        dominance_path, columns=dominance_columns
    ).rename(columns={
        "anchor_index": "input_index",
        "r_input_p": "primary_mag",
        "Re_input_p": "primary_size",
        "sersic_n_input_p": "primary_sersic_n",
        "sersic_n_input_s": "dominant_secondary_sersic_n",
    })
    if response.duplicated(KEY).any() or dominance.duplicated(KEY).any():
        raise RuntimeError("duplicate response/dominance key")
    with open(design_path, encoding="utf-8") as handle:
        design = json.load(handle)
    manifest = Path(design["manifest_dir"])
    pair_prefix = design.get("pair_prefix", "pairs")
    if design.get("catalogue_source", "generated") != "rendered":
        raise RuntimeError(
            "bias-emulator features require exact renderer-catalogue pair manifests"
        )
    frame = response.merge(dominance, on=KEY, how="left", validate="one_to_one")
    frame["has_deployed_pair"] = frame.response.notna().astype(np.int8)
    missing_pair = ~frame.has_deployed_pair.astype(bool)
    if missing_pair.any():
        for case in frame.loc[missing_pair, "case"].unique():
            local_mask = missing_pair & (frame.case == int(case))
            latent = pd.read_feather(
                manifest / f"gals{int(case)}_{float(design['g'])}.feather",
                columns=["index", "sersic_n"],
            ).set_index("index", verify_integrity=True)
            frame.loc[local_mask, "primary_sersic_n"] = latent.loc[
                frame.loc[local_mask, "input_index"], "sersic_n"
            ].to_numpy(float)
        frame.loc[missing_pair, "primary_mag"] = frame.loc[
            missing_pair, "r_input_p_plus"
        ]
        frame.loc[missing_pair, "primary_size"] = frame.loc[
            missing_pair, "Re_input_p_plus"
        ]
        frame.loc[missing_pair, "primary_flux"] = np.power(
            10.0, -0.4 * frame.loc[missing_pair, "primary_mag"]
        )
        zero_columns = [
            "response", "dominant_abs_response", "R_abs_sum",
            "neighbour_flux_sum", "n_pairs", "runner_up_abs_response",
            "n_response_pairs_ge_10pct_max", "top_abs_fraction",
            "other_abs_response_sum", "dominant_to_runner_up_abs_response",
            "dominant_flux_share_neighbours",
        ]
        frame.loc[missing_pair, zero_columns] = 0.0
    if not np.allclose(
        frame.r_input_p_plus, frame.primary_mag, rtol=0, atol=0,
    ):
        raise RuntimeError("primary magnitude mismatch across input tables")
    if not np.allclose(
        frame.Re_input_p_plus, frame.primary_size, rtol=0, atol=0,
    ):
        raise RuntimeError("primary size mismatch across input tables")
    expected_cases = {int(item["case"]) for item in design["per_case"]}
    observed_cases = set(frame.case.unique().tolist())
    if observed_cases != expected_cases:
        raise RuntimeError("pair-design cases do not match response block")

    pieces = []
    replay_max = 0.0
    replay_case_mean_max = 0.0
    for case, local in frame.groupby("case", sort=True):
        case = int(case)
        pairs = pd.read_feather(
            manifest / f"{pair_prefix}_case{case}.feather",
            columns=["anchor_index", "secondary_index", "distance", "response"],
        )
        anchors = pd.Index(local.input_index.to_numpy(np.int64), name="input_index")
        scene = pair_scene_summary(pairs, anchors)
        joined = local.merge(scene, on="input_index", how="left", validate="one_to_one")
        if not np.array_equal(
            joined.n_pairs.to_numpy(np.int64),
            joined.pair_count_replay.to_numpy(np.int64),
        ):
            raise RuntimeError(f"case {case}: pair count replay mismatch")
        response_delta = (
            joined.R_pair_replay - joined.R_blend_lsst_r_extnbr_v22
        ).to_numpy(float)
        replay_max = max(replay_max, float(np.max(np.abs(response_delta))))
        replay_case_mean_max = max(
            replay_case_mean_max, abs(float(response_delta.mean()))
        )
        if replay_max > 1.0e-6 or replay_case_mean_max > 1.0e-8:
            raise RuntimeError(
                f"case {case}: material pair-response replay mismatch"
            )
        shell_count = sum(
            joined[f"n_pair_{name}"] for _, _, name in SHELLS
        ).to_numpy(np.int64)
        if not np.array_equal(shell_count, joined.n_pairs.to_numpy(np.int64)):
            raise RuntimeError(f"case {case}: distance shells do not cover pairs")
        pieces.append(joined)
        if case % 10 == 0:
            print(
                f"case {case}: anchors={len(joined):,} pairs={len(pairs):,} "
                f"replay_max={np.max(np.abs(response_delta)):.2e}", flush=True,
            )
    result = pd.concat(pieces, ignore_index=True)
    audit = {
        "response_path": os.path.abspath(response_path),
        "dominance_path": os.path.abspath(dominance_path),
        "pair_design_path": os.path.abspath(design_path),
        "manifest_dir": str(manifest),
        "pair_prefix": pair_prefix,
        "catalogue_source": design["catalogue_source"],
        "case_window": [int(result.case.min()), int(result.case.max())],
        "n_rows": int(len(result)),
        "pair_response_replay_max_abs": replay_max,
        "pair_response_replay_max_abs_case_mean": replay_case_mean_max,
    }
    return result, audit


def quantile_dict(values: pd.Series) -> dict:
    q = values.quantile([0, 0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99,
                         0.999, 1.0])
    return {f"q{1000 * key:04.0f}": float(value) for key, value in q.items()}


def describe_split(frame: pd.DataFrame) -> dict:
    case_mean = frame.groupby("case", sort=True)[TARGET].mean().to_numpy(float)
    return {
        "case_window": [int(frame.case.min()), int(frame.case.max())],
        "n_cases": int(frame.case.nunique()),
        "n_rows": int(len(frame)),
        "rows_per_case": {
            "minimum": int(frame.groupby("case").size().min()),
            "median": float(frame.groupby("case").size().median()),
            "maximum": int(frame.groupby("case").size().max()),
        },
        "target_row_quantiles": quantile_dict(frame[TARGET]),
        "target_row_mean": float(frame[TARGET].mean()),
        "target_row_sd": float(frame[TARGET].std(ddof=1)),
        "target_case_mean": basic_stat(case_mean),
        "carrier_ratio_gt5_positive_fraction": basic_stat(
            frame.groupby("case").carrier_ratio_gt5_positive.mean().to_numpy(float)
        ),
        "carrier_top_fraction_gt0p7_positive_fraction": basic_stat(
            frame.groupby("case").carrier_top_fraction_gt0p7_positive.mean().to_numpy(float)
        ),
    }


def markdown(payload: dict) -> str:
    lines = [
        "# V2.2 coherent-anchor bias-emulator feature EDA",
        "",
        "Feather tables contain direct coherent-anchor truth joined one-to-one "
        "to latent/model-only predictors. The target is `truth - V2.2`; positive "
        "means V2.2 underpredicts. Constgold is not opened.",
        "",
        "## Structure and quality",
        "",
        f"- Rows: `{payload['n_rows']:,}` across `{payload['n_cases']}` cases.",
        f"- Features: `{len(payload['feature_sets']['full'])}` in the full model.",
        f"- Duplicate keys: `{payload['quality']['duplicate_keys']}`.",
        f"- Missing model cells: `{payload['quality']['nan_feature_cells']}` "
        "(explicit no-deployed-pair state).",
        f"- Infinite model cells: `{payload['quality']['infinite_feature_cells']}`.",
        f"- Maximum pair-response replay difference: "
        f"`{payload['quality']['pair_response_replay_max_abs']:.3e}`.",
        "",
        "## Predeclared case split",
        "",
        "| Split | Cases | Rows | Mean target | Case SEM | Row SD |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ("train", "tune", "test"):
        item = payload["splits"][name]
        lines.append(
            f"| {name} | {item['case_window'][0]}-{item['case_window'][1]} | "
            f"{item['n_rows']:,} | {item['target_case_mean']['mean']:+.6f} | "
            f"{item['target_case_mean']['sem']:.6f} | {item['target_row_sd']:.4f} |"
        )
    lines.extend([
        "",
        "## Largest sampled rank correlations with the target",
        "",
        "These are descriptive only; model selection and final claims use the "
        "case-separated tuning/test design.",
        "",
        "| Feature | Spearman rho |",
        "|---|---:|",
    ])
    for item in payload["sample_spearman_with_target"][:15]:
        lines.append(f"| `{item['feature']}` | {item['rho']:+.4f} |")
    lines.extend([
        "", "## Recommendations", "",
        "- Use case-balanced training weights; row counts differ slightly by case.",
        "- Judge usefulness by held-out conditional-mean calibration, not raw "
        "per-row R2: the direct per-anchor response target is heavy-tailed.",
        "- Keep the final c700-899 block out of hyperparameter and feature-pair "
        "selection.",
        "- Treat any fitted residual model as a diagnostic, not a deployed "
        "correction or constgold calibration.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--response", nargs="+", required=True)
    ap.add_argument("--dominance", nargs="+", required=True)
    ap.add_argument("--pair-design", nargs="+", required=True)
    ap.add_argument("--train-max", type=int, default=599)
    ap.add_argument("--tune-max", type=int, default=699)
    ap.add_argument("--output-feather", required=True)
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--output-md", required=True)
    args = ap.parse_args()
    if not (len(args.response) == len(args.dominance) == len(args.pair_design)):
        raise ValueError("response/dominance/pair-design block counts differ")
    for output in (args.output_feather, args.output_json, args.output_md):
        if os.path.exists(output):
            raise FileExistsError(f"refusing existing output {output}")

    parts = []
    audits = []
    for response, dominance, design in zip(
        args.response, args.dominance, args.pair_design
    ):
        block, audit = load_block(response, dominance, design)
        parts.append(block)
        audits.append(audit)
    frame = derive_features(pd.concat(parts, ignore_index=True))
    if frame.duplicated(KEY).any():
        raise RuntimeError("duplicate final anchor key")
    if not frame.case.is_monotonic_increasing:
        frame = frame.sort_values(KEY, kind="mergesort").reset_index(drop=True)

    keep = [
        *KEY, "R_blend_truth", "scene_prediction", TARGET,
        "carrier_ratio_gt5_positive", "carrier_top_fraction_gt0p7_positive",
        *[feature for feature in FULL_FEATURES if feature != "scene_prediction"],
    ]
    frame = frame[keep]
    Path(args.output_feather).parent.mkdir(parents=True, exist_ok=True)
    frame.to_feather(args.output_feather)

    train = frame.loc[frame.case <= args.train_max]
    tune = frame.loc[frame.case.between(args.train_max + 1, args.tune_max)]
    test = frame.loc[frame.case > args.tune_max]
    if min(local.case.nunique() for local in (train, tune, test)) < 50:
        raise RuntimeError("case split unexpectedly small")
    rng = np.random.default_rng(7301)
    sample = frame.iloc[
        rng.choice(len(frame), size=min(200_000, len(frame)), replace=False)
    ]
    correlations = sample[[*FULL_FEATURES, TARGET]].corr(method="spearman")[TARGET]
    correlations = correlations.drop(TARGET)
    correlations = correlations.loc[np.isfinite(correlations)].sort_values(
        key=lambda values: values.abs(), ascending=False
    )
    payload = {
        "title": "V2.2 coherent-anchor bias-emulator feature EDA",
        "format": "Apache Feather tabular scientific data",
        "output_feather": os.path.abspath(args.output_feather),
        "target": TARGET,
        "target_definition": "R_blend_truth - R_blend_lsst_r_extnbr_v22",
        "target_sign": "positive means V2.2 underpredicts",
        "n_rows": int(len(frame)),
        "n_cases": int(frame.case.nunique()),
        "feature_sets": FEATURE_SETS,
        "feature_labels": FEATURE_LABELS,
        "physical_features": PHYSICAL_FEATURES,
        "response_features": RESPONSE_FEATURES,
        "shells_arcsec": [[lo, hi, name] for lo, hi, name in SHELLS],
        "splits": {
            "train": describe_split(train),
            "tune": describe_split(tune),
            "test": describe_split(test),
        },
        "quality": {
            "duplicate_keys": int(frame.duplicated(KEY).sum()),
            "nan_feature_cells": int(frame[FULL_FEATURES].isna().sum().sum()),
            "infinite_feature_cells": int(np.isinf(
                frame[FULL_FEATURES].to_numpy(float)
            ).sum()),
            "pair_response_replay_max_abs": float(max(
                item["pair_response_replay_max_abs"] for item in audits
            )),
            "pair_response_replay_max_abs_case_mean": float(max(
                item["pair_response_replay_max_abs_case_mean"] for item in audits
            )),
        },
        "feature_quantiles": {
            feature: quantile_dict(frame[feature]) for feature in FULL_FEATURES
        },
        "sample_spearman_n_rows": int(len(sample)),
        "sample_spearman_with_target": [
            {"feature": feature, "rho": float(value)}
            for feature, value in correlations.items()
        ],
        "source_blocks": audits,
        "constgold_opened": False,
    }
    with open(args.output_json, "x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    with open(args.output_md, "x", encoding="utf-8") as handle:
        handle.write(markdown(payload))
    print(markdown(payload), flush=True)
    print("ANCHOR_BIAS_FEATURES_DONE", flush=True)


if __name__ == "__main__":
    main()
