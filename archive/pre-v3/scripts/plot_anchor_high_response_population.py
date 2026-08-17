#!/usr/bin/env python3
"""Compare high-scene-response coherent anchors with the remaining population."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from string import ascii_uppercase
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


KEY = ["case", "input_index"]
GROUP_ORDER = ["high", "outside"]
GROUP_LABEL = {
    "high": r"$R_{\rm scene}>0.1$",
    "outside": r"$R_{\rm scene}\leq0.1$",
}
GROUP_COLOR = {"high": "#D55E00", "outside": "#0072B2"}
GROUP_STYLE = {"high": "-", "outside": "--"}


@dataclass(frozen=True)
class PanelSpec:
    name: str
    label: str
    source: str
    bins: np.ndarray
    log_x: bool = False
    discrete: bool = False
    digits: int = 2


PANELS = [
    PanelSpec(
        "n_pairs", "Number of deployed neighbours", "anchor",
        np.arange(-0.5, 20.5, 1.0), discrete=True, digits=2,
    ),
    PanelSpec(
        "primary_mag", r"Primary magnitude $r_p$", "anchor",
        np.linspace(18.0, 25.8, 32), digits=2,
    ),
    PanelSpec(
        "primary_size", r"Primary size $R_{e,p}$ (arcsec)", "anchor",
        np.linspace(0.5, 1.5, 31), digits=3,
    ),
    PanelSpec(
        "primary_sersic_n", r"Primary Sérsic index $n_p$", "anchor",
        np.geomspace(0.1, 10.0, 31), log_x=True, digits=2,
    ),
    PanelSpec(
        "secondary_mag", r"Neighbour magnitude $r_s$", "neighbour",
        np.linspace(13.0, 29.0, 33), digits=2,
    ),
    PanelSpec(
        "secondary_size", r"Neighbour size $R_{e,s}$ (arcsec)", "neighbour",
        np.geomspace(0.005, 10.0, 34), log_x=True, digits=3,
    ),
    PanelSpec(
        "secondary_sersic_n", r"Neighbour Sérsic index $n_s$", "neighbour",
        np.geomspace(0.1, 10.0, 31), log_x=True, digits=2,
    ),
    PanelSpec(
        "distance", "Primary–neighbour separation (arcsec)", "neighbour",
        np.linspace(0.0, 10.0, 31), digits=2,
    ),
    PanelSpec(
        "log10_flux_ratio", r"$\log_{10}(F_s/F_p)$", "neighbour",
        np.linspace(-4.5, 5.2, 34), digits=2,
    ),
]


def finite_stat(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        raise ValueError("at least two finite case values are required")
    sd = float(np.std(values, ddof=1))
    return {
        "mean": float(np.mean(values)),
        "case_sd": sd,
        "case_sem": float(sd / np.sqrt(values.size)),
        "n_cases": int(values.size),
    }


def normalized_histogram(
    values: np.ndarray,
    bins: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    discrete: bool = False,
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    bins = np.asarray(bins, dtype=float)
    if weights is None:
        weights = np.ones(values.size, dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights >= 0.0)
    values = values[valid]
    weights = weights[valid]
    if values.size == 0 or float(weights.sum()) <= 0.0:
        raise ValueError("empty weighted histogram")
    tolerance = 1.0e-10
    if (
        np.any(values < bins[0] - tolerance)
        or np.any(values > bins[-1] + tolerance)
    ):
        raise ValueError(
            f"values [{values.min()}, {values.max()}] exceed "
            f"histogram support [{bins[0]}, {bins[-1]}]"
        )
    counts, _ = np.histogram(values, bins=bins, weights=weights)
    mass = counts.astype(float) / float(weights.sum())
    if discrete:
        return mass
    return mass / np.diff(bins)


def bin_centers(spec: PanelSpec) -> np.ndarray:
    if spec.log_x:
        return np.sqrt(spec.bins[:-1] * spec.bins[1:])
    return 0.5 * (spec.bins[:-1] + spec.bins[1:])


def find_case_file(
    directories: list[Path], pattern: str, case: int,
) -> Path:
    matches = [directory / pattern.format(case=case) for directory in directories]
    matches = [path for path in matches if path.is_file()]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one file for case {case}, found {len(matches)}: {matches}"
        )
    return matches[0]


def prepare_anchors(
    features_path: Path,
    predictions_path: Path,
    case_min: int,
    case_max: int,
    threshold: float,
) -> pd.DataFrame:
    feature_columns = [
        *KEY,
        "scene_prediction",
        "bias_truth_minus_model",
        "primary_mag",
        "log10_primary_size",
        "log10_primary_sersic_n",
        "log1p_n_pairs",
    ]
    anchors = pd.read_feather(features_path, columns=feature_columns)
    anchors = anchors.loc[anchors.case.between(case_min, case_max)].copy()
    predictions = pd.read_feather(
        predictions_path,
        columns=[*KEY, "predicted_bias_full"],
    )
    predictions = predictions.loc[
        predictions.case.between(case_min, case_max)
    ].copy()
    if anchors.duplicated(KEY).any() or predictions.duplicated(KEY).any():
        raise RuntimeError("duplicate anchor key")
    anchors = anchors.merge(
        predictions, on=KEY, how="outer", validate="one_to_one",
        indicator=True,
    )
    if not anchors["_merge"].eq("both").all():
        raise RuntimeError("anchor/prediction join failed")
    anchors = anchors.drop(columns="_merge")
    expected_cases = np.arange(case_min, case_max + 1)
    if not np.array_equal(np.sort(anchors.case.unique()), expected_cases):
        raise RuntimeError("anchor table does not cover the requested cases exactly")
    numeric = anchors.drop(columns=KEY).to_numpy(float)
    if not np.isfinite(numeric).all():
        raise RuntimeError("non-finite anchor value")
    anchors["n_pairs"] = np.rint(
        np.expm1(anchors["log1p_n_pairs"].to_numpy(float))
    ).astype(int)
    replay = np.log1p(anchors["n_pairs"].to_numpy(float))
    if np.max(np.abs(replay - anchors["log1p_n_pairs"].to_numpy(float))) > 1e-8:
        raise RuntimeError("neighbour count does not replay log1p_n_pairs")
    if anchors.n_pairs.min() < 0 or anchors.n_pairs.max() > 19:
        raise RuntimeError("deployed neighbour count lies outside 0--19")
    anchors["primary_size"] = np.power(
        10.0, anchors["log10_primary_size"].to_numpy(float)
    )
    anchors["primary_sersic_n"] = np.power(
        10.0, anchors["log10_primary_sersic_n"].to_numpy(float)
    )
    anchors["group"] = np.where(
        anchors.scene_prediction.to_numpy(float) > threshold,
        "high",
        "outside",
    )
    anchors["remaining_gap"] = (
        anchors.bias_truth_minus_model - anchors.predicted_bias_full
    )
    return anchors


def initialize_store() -> tuple[
    dict[str, dict[str, list[np.ndarray]]],
    dict[str, dict[str, list[float]]],
]:
    histograms = {
        spec.name: {group: [] for group in GROUP_ORDER} for spec in PANELS
    }
    means = {
        spec.name: {group: [] for group in GROUP_ORDER} for spec in PANELS
    }
    return histograms, means


def record_distribution(
    histograms: dict[str, dict[str, list[np.ndarray]]],
    means: dict[str, dict[str, list[float]]],
    spec: PanelSpec,
    group: str,
    values: np.ndarray,
    weights: np.ndarray | None = None,
) -> None:
    values = np.asarray(values, dtype=float)
    if weights is None:
        weights = np.ones(values.size, dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)
    histograms[spec.name][group].append(normalized_histogram(
        values, spec.bins, weights=weights, discrete=spec.discrete,
    ))
    finite = np.isfinite(values) & np.isfinite(weights) & (weights >= 0.0)
    if not finite.any() or float(weights[finite].sum()) <= 0.0:
        raise ValueError(f"empty mean for {spec.name}/{group}")
    means[spec.name][group].append(float(np.average(
        values[finite], weights=weights[finite],
    )))


def process_cases(
    anchors: pd.DataFrame,
    directories: list[Path],
    pair_prefix: str,
    galaxy_shear: str,
) -> tuple[
    dict[str, dict[str, list[np.ndarray]]],
    dict[str, dict[str, list[float]]],
    pd.DataFrame,
]:
    histograms, means = initialize_store()
    cases = []
    anchor_specs = [spec for spec in PANELS if spec.source == "anchor"]
    neighbour_specs = [spec for spec in PANELS if spec.source == "neighbour"]

    for case_index, (case, local) in enumerate(
        anchors.groupby("case", sort=True), start=1,
    ):
        local = local.copy()
        row = {
            "case": int(case),
            "n_anchors": int(len(local)),
            "n_high": int(local.group.eq("high").sum()),
        }
        row["fraction_high"] = row["n_high"] / row["n_anchors"]
        for group in GROUP_ORDER:
            selected = local.loc[local.group.eq(group)]
            if len(selected) < 2:
                raise RuntimeError(f"case {case} has too few {group} anchors")
            row[f"n_{group}"] = int(len(selected))
            row[f"raw_gap_{group}"] = float(
                selected.bias_truth_minus_model.mean()
            )
            row[f"predicted_bias_{group}"] = float(
                selected.predicted_bias_full.mean()
            )
            row[f"remaining_gap_{group}"] = float(
                selected.remaining_gap.mean()
            )
            for spec in anchor_specs:
                record_distribution(
                    histograms,
                    means,
                    spec,
                    group,
                    selected[spec.name].to_numpy(float),
                )
        row["predicted_bias_global"] = float(local.predicted_bias_full.mean())
        row["predicted_bias_high_contribution"] = float(
            np.where(
                local.group.eq("high"), local.predicted_bias_full, 0.0,
            ).mean()
        )
        row["predicted_bias_outside_contribution"] = float(
            np.where(
                local.group.eq("outside"), local.predicted_bias_full, 0.0,
            ).mean()
        )
        if not np.isclose(
            row["predicted_bias_global"],
            row["predicted_bias_high_contribution"]
            + row["predicted_bias_outside_contribution"],
            rtol=0.0,
            atol=2.0e-15,
        ):
            raise RuntimeError("bias-contribution decomposition does not close")

        pair_path = find_case_file(
            directories, f"{pair_prefix}_case{{case}}.feather", int(case),
        )
        galaxy_path = find_case_file(
            directories, f"gals{{case}}_{galaxy_shear}.feather", int(case),
        )
        pairs = pd.read_feather(
            pair_path,
            columns=[
                "anchor_index", "secondary_index", "distance", "n_pairs",
            ],
        )
        anchor_join = local[[
            "input_index", "group", "primary_mag", "n_pairs",
        ]].rename(columns={"n_pairs": "anchor_n_pairs"})
        pairs = pairs.merge(
            anchor_join,
            left_on="anchor_index",
            right_on="input_index",
            how="inner",
            validate="many_to_one",
        )
        if not pairs.n_pairs.eq(pairs.anchor_n_pairs).all():
            raise RuntimeError(f"pair neighbour count mismatch in case {case}")
        observed = pairs.groupby("anchor_index", sort=False).size()
        expected = local.set_index("input_index").n_pairs
        expected = expected.loc[expected > 0].sort_index()
        observed = observed.sort_index()
        if not observed.index.equals(expected.index):
            raise RuntimeError(f"pair anchor coverage mismatch in case {case}")
        if not np.array_equal(observed.to_numpy(int), expected.to_numpy(int)):
            raise RuntimeError(f"pair row-count mismatch in case {case}")

        galaxies = pd.read_feather(
            galaxy_path, columns=["index", "r", "Re", "sersic_n"],
        ).rename(columns={
            "index": "secondary_index",
            "r": "secondary_mag",
            "Re": "secondary_size",
            "sersic_n": "secondary_sersic_n",
        })
        if galaxies.secondary_index.duplicated().any():
            raise RuntimeError(f"duplicate galaxy index in case {case}")
        pairs = pairs.merge(
            galaxies,
            on="secondary_index",
            how="left",
            validate="many_to_one",
        )
        pair_numeric = pairs[[
            "distance", "primary_mag", "secondary_mag", "secondary_size",
            "secondary_sersic_n",
        ]].to_numpy(float)
        if not np.isfinite(pair_numeric).all():
            raise RuntimeError(f"missing/non-finite pair properties in case {case}")
        pairs["log10_flux_ratio"] = -0.4 * (
            pairs.secondary_mag - pairs.primary_mag
        )
        pairs["anchor_weight"] = 1.0 / pairs.anchor_n_pairs
        for group in GROUP_ORDER:
            selected = pairs.loc[pairs.group.eq(group)]
            if selected.empty:
                raise RuntimeError(f"case {case} has no {group} pair rows")
            row[f"n_{group}_with_neighbours"] = int(
                selected.anchor_index.nunique()
            )
            for spec in neighbour_specs:
                record_distribution(
                    histograms,
                    means,
                    spec,
                    group,
                    selected[spec.name].to_numpy(float),
                    selected.anchor_weight.to_numpy(float),
                )
        cases.append(row)
        if case_index % 20 == 0:
            print(f"processed {case_index}/{anchors.case.nunique()} cases", flush=True)

    return histograms, means, pd.DataFrame(cases)


def summarize(
    anchors: pd.DataFrame,
    histograms: dict[str, dict[str, list[np.ndarray]]],
    means: dict[str, dict[str, list[float]]],
    cases: pd.DataFrame,
    threshold: float,
    features_path: Path,
    predictions_path: Path,
    directories: list[Path],
) -> tuple[dict[str, Any], pd.DataFrame]:
    variables: dict[str, Any] = {}
    distribution_rows = []
    for spec in PANELS:
        variables[spec.name] = {
            "label": spec.label,
            "source": spec.source,
            "groups": {},
        }
        centers = bin_centers(spec)
        for group in GROUP_ORDER:
            matrix = np.stack(histograms[spec.name][group])
            mean_curve = matrix.mean(axis=0)
            sem_curve = matrix.std(axis=0, ddof=1) / np.sqrt(matrix.shape[0])
            mean_summary = finite_stat(np.asarray(means[spec.name][group]))
            variables[spec.name]["groups"][group] = {
                "case_balanced_mean": mean_summary,
            }
            for index, (center, value, sem) in enumerate(zip(
                centers, mean_curve, sem_curve,
            )):
                distribution_rows.append({
                    "variable": spec.name,
                    "source": spec.source,
                    "group": group,
                    "bin": int(index),
                    "lower": float(spec.bins[index]),
                    "upper": float(spec.bins[index + 1]),
                    "center": float(center),
                    "density_or_mass": float(value),
                    "case_sem": float(sem),
                    "n_cases": int(matrix.shape[0]),
                })
        paired_difference = (
            np.asarray(means[spec.name]["high"])
            - np.asarray(means[spec.name]["outside"])
        )
        variables[spec.name]["high_minus_outside_case_paired_mean"] = (
            finite_stat(paired_difference)
        )

    group_summary = {}
    for group in GROUP_ORDER:
        selected = anchors.loc[anchors.group.eq(group)]
        group_summary[group] = {
            "n_anchors": int(len(selected)),
            "fraction_of_anchors": float(len(selected) / len(anchors)),
            "raw_gap": finite_stat(cases[f"raw_gap_{group}"].to_numpy(float)),
            "predicted_bias": finite_stat(
                cases[f"predicted_bias_{group}"].to_numpy(float)
            ),
            "remaining_gap": finite_stat(
                cases[f"remaining_gap_{group}"].to_numpy(float)
            ),
            "n_anchors_with_neighbours": int(
                cases[f"n_{group}_with_neighbours"].sum()
            ),
        }
    global_prediction = finite_stat(cases.predicted_bias_global.to_numpy(float))
    high_contribution = finite_stat(
        cases.predicted_bias_high_contribution.to_numpy(float)
    )
    outside_contribution = finite_stat(
        cases.predicted_bias_outside_contribution.to_numpy(float)
    )
    payload = {
        "title": "Held-out coherent-anchor properties split by V2.2 scene response",
        "case_window": [int(anchors.case.min()), int(anchors.case.max())],
        "n_cases": int(anchors.case.nunique()),
        "n_anchors": int(len(anchors)),
        "threshold": {
            "feature": "scene_prediction",
            "operator": ">",
            "value": float(threshold),
        },
        "groups": group_summary,
        "case_balanced_high_fraction": finite_stat(
            cases.fraction_high.to_numpy(float)
        ),
        "bias_emulator_shift_decomposition": {
            "global": global_prediction,
            "high_additive_contribution": high_contribution,
            "outside_additive_contribution": outside_contribution,
            "high_share_of_mean_shift": float(
                high_contribution["mean"] / global_prediction["mean"]
            ),
            "closure_absolute": float(abs(
                global_prediction["mean"]
                - high_contribution["mean"]
                - outside_contribution["mean"]
            )),
        },
        "variables": variables,
        "weighting": {
            "primary_panels": "one unit per anchor",
            "neighbour_panels": (
                "each pair receives 1/n_pairs, so every anchor with at least "
                "one deployed neighbour has total weight one"
            ),
            "curves": (
                "within-case normalized histograms averaged equally over cases; "
                "bands are one SEM across rendered cases"
            ),
        },
        "sources": {
            "features": str(features_path.resolve()),
            "predictions": str(predictions_path.resolve()),
            "manifest_directories": [str(path.resolve()) for path in directories],
            "pair_prefix": "pairs_renderer_v22",
            "galaxy_shear_leg": "+0.02 catalogue; latent properties are shear-invariant",
        },
        "constgold_opened": False,
    }
    return payload, pd.DataFrame(distribution_rows)


def configure_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 8.0,
        "axes.labelsize": 8.5,
        "axes.titlesize": 8.5,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 8.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def plot_distributions(
    distributions: pd.DataFrame,
    payload: dict[str, Any],
    output_prefix: Path,
) -> None:
    configure_style()
    figure, axes = plt.subplots(3, 3, figsize=(10.8, 8.0))
    for panel_index, (axis, spec) in enumerate(zip(axes.flat, PANELS)):
        local_all = distributions.loc[distributions.variable.eq(spec.name)]
        for group in GROUP_ORDER:
            local = local_all.loc[local_all.group.eq(group)].sort_values("bin")
            x = local.center.to_numpy(float)
            y = local.density_or_mass.to_numpy(float)
            sem = local.case_sem.to_numpy(float)
            axis.plot(
                x,
                y,
                color=GROUP_COLOR[group],
                linestyle=GROUP_STYLE[group],
                linewidth=1.35,
                label=GROUP_LABEL[group],
            )
            axis.fill_between(
                x,
                np.maximum(0.0, y - sem),
                y + sem,
                color=GROUP_COLOR[group],
                alpha=0.14,
                linewidth=0.0,
            )
        if spec.log_x:
            axis.set_xscale("log")
        axis.set_ylim(bottom=0.0)
        axis.set_xlabel(spec.label)
        axis.set_ylabel("Probability" if spec.discrete else "Probability density")
        axis.spines[["top", "right"]].set_visible(False)
        axis.text(
            -0.14,
            1.05,
            ascii_uppercase[panel_index],
            transform=axis.transAxes,
            fontsize=9.5,
            fontweight="bold",
            va="top",
        )
        high = payload["variables"][spec.name]["groups"]["high"][
            "case_balanced_mean"
        ]["mean"]
        outside = payload["variables"][spec.name]["groups"]["outside"][
            "case_balanced_mean"
        ]["mean"]
        axis.text(
            0.98,
            0.96,
            f"means (> / ≤): {high:.{spec.digits}f} / {outside:.{spec.digits}f}",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=6.8,
            color="0.25",
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.78,
                "pad": 1.2,
            },
        )

    high = payload["groups"]["high"]
    outside = payload["groups"]["outside"]
    handles, _ = axes.flat[0].get_legend_handles_labels()
    labels = [
        f"{GROUP_LABEL['high']}  (N={high['n_anchors']:,}, "
        f"{100.0 * high['fraction_of_anchors']:.1f}%)",
        f"{GROUP_LABEL['outside']}  (N={outside['n_anchors']:,})",
    ]
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.935),
        ncol=2,
        frameon=False,
    )
    figure.suptitle(
        "What distinguishes the high-V2.2-response coherent anchors?",
        fontsize=11.0,
        y=0.992,
    )
    figure.text(
        0.5,
        0.955,
        "Held-out c700–899; neighbour curves give each anchor total weight one; "
        "bands are one SEM across cases",
        ha="center",
        va="top",
        fontsize=8.0,
    )
    figure.subplots_adjust(
        left=0.075, right=0.985, bottom=0.065, top=0.885,
        hspace=0.42, wspace=0.30,
    )
    figure.savefig(f"{output_prefix}.pdf", bbox_inches="tight")
    figure.savefig(f"{output_prefix}.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def write_markdown(payload: dict[str, Any], output: Path) -> None:
    high = payload["groups"]["high"]
    outside = payload["groups"]["outside"]
    shift = payload["bias_emulator_shift_decomposition"]
    lines = [
        "# Held-out coherent anchors split at raw V2.2 scene response 0.1",
        "",
        f"The high-response subset contains `{high['n_anchors']:,}` of "
        f"`{payload['n_anchors']:,}` anchors "
        f"(`{100.0 * high['fraction_of_anchors']:.3f}%`).",
        "",
        "## Bias-emulator shift",
        "",
        f"- Global predicted correction: `{shift['global']['mean']:+.6f} "
        f"+- {shift['global']['case_sem']:.6f}`.",
        f"- High-response additive contribution: "
        f"`{shift['high_additive_contribution']['mean']:+.6f} "
        f"+- {shift['high_additive_contribution']['case_sem']:.6f}` "
        f"(`{100.0 * shift['high_share_of_mean_shift']:.2f}%` of the mean shift).",
        f"- Outside additive contribution: "
        f"`{shift['outside_additive_contribution']['mean']:+.6f} "
        f"+- {shift['outside_additive_contribution']['case_sem']:.6f}`.",
        "",
        "## Case-balanced property means",
        "",
        "| property | response > 0.1 | outside | paired difference |",
        "|---|---:|---:|---:|",
    ]
    for spec in PANELS:
        item = payload["variables"][spec.name]
        high_mean = item["groups"]["high"]["case_balanced_mean"]
        outside_mean = item["groups"]["outside"]["case_balanced_mean"]
        difference = item["high_minus_outside_case_paired_mean"]
        lines.append(
            f"| {spec.label} | {high_mean['mean']:.6f} +- "
            f"{high_mean['case_sem']:.6f} | {outside_mean['mean']:.6f} +- "
            f"{outside_mean['case_sem']:.6f} | {difference['mean']:+.6f} +- "
            f"{difference['case_sem']:.6f} |"
        )
    lines.extend([
        "",
        "Primary panels weight anchors equally. Neighbour panels weight each "
        "pair by `1/n_pairs`, so every anchor with at least one deployed "
        "neighbour contributes total weight one. Distribution bands are one "
        "SEM across the 200 rendered cases.",
        "",
    ])
    output.write_text("\n".join(lines), encoding="utf-8")


def json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_clean(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--manifest-dir", nargs="+", required=True)
    parser.add_argument("--pair-prefix", default="pairs_renderer_v22")
    parser.add_argument("--galaxy-shear", default="0.02")
    parser.add_argument("--case-min", type=int, default=700)
    parser.add_argument("--case-max", type=int, default=899)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()

    if args.case_min > args.case_max or not np.isfinite(args.threshold):
        raise ValueError("invalid case window or threshold")
    features_path = Path(args.features)
    predictions_path = Path(args.predictions)
    directories = [Path(path) for path in args.manifest_dir]
    for path in [features_path, predictions_path, *directories]:
        if not path.exists():
            raise FileNotFoundError(path)
    output_prefix = Path(args.output_prefix)
    outputs = [Path(f"{output_prefix}.{suffix}") for suffix in (
        "csv", "json", "md", "pdf", "png",
    )]
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(f"refusing existing outputs: {existing}")
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    anchors = prepare_anchors(
        features_path,
        predictions_path,
        args.case_min,
        args.case_max,
        args.threshold,
    )
    histograms, means, cases = process_cases(
        anchors, directories, args.pair_prefix, args.galaxy_shear,
    )
    payload, distributions = summarize(
        anchors,
        histograms,
        means,
        cases,
        args.threshold,
        features_path,
        predictions_path,
        directories,
    )
    distributions.to_csv(f"{output_prefix}.csv", index=False)
    with open(f"{output_prefix}.json", "x", encoding="utf-8") as handle:
        json.dump(json_clean(payload), handle, indent=2, allow_nan=False)
        handle.write("\n")
    write_markdown(payload, Path(f"{output_prefix}.md"))
    plot_distributions(distributions, payload, output_prefix)
    print(
        f"saved high-response population comparison to {output_prefix}.*",
        flush=True,
    )


if __name__ == "__main__":
    main()
