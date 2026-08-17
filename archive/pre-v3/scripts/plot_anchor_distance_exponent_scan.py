"""Compare inverse-distance exponents for a physical anchor-scene proxy.

For each coherent anchor, construct

    Q_alpha = sum_j (F_s,j / F_p) / d_j**alpha

on the exact deployed V2.2 renderer-neighbour support.  The figure compares
alpha = 0, 0.5, 1, 1.5, and 2 using equal-count physical-proxy bins.  Top-row
curves show measured coherent response; bottom-row curves show the V2.2
truth-minus-prediction residual with anchor histograms behind them.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from scripts.plot_anchor_physical_proxy_truth_residual import (
    case_balanced_curve,
    log_with_zero_floor,
    plot_density,
    proxy_edges,
    response_transform,
)
from scripts.plot_anchor_scene_flux_distance_proxy import (
    base_for_case,
    input_path,
    refuse,
)


ALPHAS = (0.0, 0.5, 1.0, 1.5, 2.0)


def alpha_tag(alpha: float) -> str:
    return f"a{alpha:g}".replace(".", "p")


def raw_name(alpha: float) -> str:
    return f"q_{alpha_tag(alpha)}"


def log_name(alpha: float) -> str:
    return f"log10_{raw_name(alpha)}"


def case_proxies(
    case: int,
    base_root: Path,
    pair_prefix: str,
    shear: float,
    expected: pd.DataFrame,
) -> pd.DataFrame:
    base = base_for_case(base_root, case)
    pairs = pd.read_feather(
        base / f"{pair_prefix}_case{case}.feather",
        columns=["anchor_index", "secondary_index", "distance"],
    )
    catalogue = pd.read_feather(
        input_path(base, case, shear), columns=["index_input", "r_input"],
    ).rename(columns={"index_input": "index"})
    if catalogue["index"].duplicated().any():
        raise RuntimeError(f"case {case}: duplicate rendered input ID")
    latent = catalogue.set_index("index")
    secondary_mag = latent.reindex(pairs.secondary_index).r_input.to_numpy(float)
    primary_mag = latent.reindex(pairs.anchor_index).r_input.to_numpy(float)
    distance = pairs.distance.to_numpy(float)
    if not all(np.isfinite(value).all() for value in (
        secondary_mag, primary_mag, distance
    )):
        raise RuntimeError(f"case {case}: non-finite physical pair input")
    if np.any(distance <= 0.0):
        raise RuntimeError(f"case {case}: non-positive separation")

    ratio = np.power(10.0, -0.4 * (secondary_mag - primary_mag))
    terms = pd.DataFrame({
        "input_index": pairs.anchor_index.to_numpy(np.int64),
        **{
            raw_name(alpha): ratio / np.power(distance, alpha)
            for alpha in ALPHAS
        },
    })
    scene = terms.groupby("input_index", sort=False).sum().reset_index()
    scene["case"] = case
    merged = expected.merge(
        scene, on=["case", "input_index"], how="left", validate="one_to_one"
    )
    proxy_columns = [raw_name(alpha) for alpha in ALPHAS]
    merged[proxy_columns] = merged[proxy_columns].fillna(0.0)
    print(
        f"case {case}: anchors={len(merged):,} pairs={len(pairs):,}",
        flush=True,
    )
    return merged


def metric_summary(
    frame: pd.DataFrame,
    proxy: str,
    truth_curve: pd.DataFrame,
    residual_curve: pd.DataFrame,
) -> dict:
    x = frame[proxy].to_numpy(float)
    truth = frame.R_blend_truth.to_numpy(float)
    residual = frame.bias_truth_minus_model.to_numpy(float)
    curve_residual = residual_curve["mean"].to_numpy(float)
    n_top = max(1, int(np.ceil(0.30 * len(curve_residual))))
    denominator = float(np.sum(curve_residual))
    top_share = (
        float(np.sum(curve_residual[-n_top:]) / denominator)
        if denominator != 0.0 else float("nan")
    )
    return {
        "individual_truth_spearman_rho": float(stats.spearmanr(x, truth).statistic),
        "individual_residual_spearman_rho": float(
            stats.spearmanr(x, residual).statistic
        ),
        "binned_truth_spearman_rho": float(stats.spearmanr(
            truth_curve.x.to_numpy(float), truth_curve["mean"].to_numpy(float)
        ).statistic),
        "residual_curve_rms_about_global": float(np.sqrt(np.mean(np.square(
            curve_residual - np.mean(curve_residual)
        )))),
        "upper_30pct_gap_share": top_share,
        "highest_bin_residual": float(curve_residual[-1]),
        "peak_bin_residual": float(np.max(curve_residual)),
        "minimum_bin_residual": float(np.min(curve_residual)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True)
    parser.add_argument("--base-root", required=True)
    parser.add_argument("--pair-prefix", default="pairs_renderer_v22")
    parser.add_argument("--case-min", type=int, default=400)
    parser.add_argument("--case-max", type=int, default=899)
    parser.add_argument("--shear", type=float, default=0.02)
    parser.add_argument("--n-bins", type=int, default=20)
    parser.add_argument("--x-quantile-min", type=float, default=0.20)
    parser.add_argument("--x-quantile-max", type=float, default=0.98)
    parser.add_argument("--hist-bins", type=int, default=32)
    parser.add_argument("--table-output", required=True)
    parser.add_argument("--curve-output", required=True)
    parser.add_argument("--figure-output", required=True)
    parser.add_argument("--pdf-output", required=True)
    parser.add_argument("--summary-output", required=True)
    args = parser.parse_args()

    if not 0.0 <= args.x_quantile_min < args.x_quantile_max <= 1.0:
        raise ValueError("invalid x-axis quantile limits")
    outputs = [
        Path(args.table_output), Path(args.curve_output),
        Path(args.figure_output), Path(args.pdf_output),
        Path(args.summary_output),
    ]
    refuse(outputs)
    for output in outputs:
        output.parent.mkdir(parents=True, exist_ok=True)

    features = pd.read_feather(
        args.features,
        columns=[
            "case", "input_index", "R_blend_truth", "scene_prediction",
            "bias_truth_minus_model",
        ],
    )
    features = features[
        features.case.between(args.case_min, args.case_max)
    ].copy()
    if features.duplicated(["case", "input_index"]).any():
        raise RuntimeError("anchor feature keys are not unique")
    cases = np.arange(args.case_min, args.case_max + 1)
    if not np.array_equal(np.sort(features.case.unique()), cases):
        raise RuntimeError("feature table does not cover requested cases")

    parts = []
    base_root = Path(args.base_root)
    for case in cases:
        expected = features[features.case == case].copy()
        parts.append(case_proxies(
            int(case), base_root, args.pair_prefix, args.shear, expected
        ))
    frame = pd.concat(parts, ignore_index=True)

    floors = {}
    for alpha in ALPHAS:
        raw = raw_name(alpha)
        proxy = log_name(alpha)
        frame[proxy], floors[raw] = log_with_zero_floor(
            frame[raw].to_numpy(float)
        )
    frame.to_feather(args.table_output)

    curves = {}
    curve_parts = []
    summary = {
        "dataset": f"coherent_anchor_c{args.case_min}-{args.case_max}",
        "n_anchors": int(len(frame)),
        "n_cases": int(len(cases)),
        "n_bins": int(args.n_bins),
        "proxy_definition": "Q_alpha = sum_j (F_s,j/F_p) / d_j**alpha",
        "gap_definition": "R_blend_truth - frozen V2.2 scene prediction",
        "bin_definition": "global equal-count bins using each physical proxy only",
        "uncertainty": "one SEM across per-case bin means",
        "display_quantiles": [args.x_quantile_min, args.x_quantile_max],
        "zero_proxy_plot_floors": floors,
        "alphas": {},
    }
    for alpha in ALPHAS:
        proxy = log_name(alpha)
        edges = proxy_edges(frame[proxy].to_numpy(float), args.n_bins)
        truth_curve = case_balanced_curve(frame, proxy, "R_blend_truth", edges)
        residual_curve = case_balanced_curve(
            frame, proxy, "bias_truth_minus_model", edges
        )
        truth_curve["target"] = "coherent_truth"
        residual_curve["target"] = "truth_minus_v22"
        for curve in (truth_curve, residual_curve):
            curve["alpha"] = alpha
            curve["proxy"] = proxy
        curve_parts.extend([truth_curve, residual_curve])
        curves[alpha] = (truth_curve, residual_curve)
        summary["alphas"][f"{alpha:g}"] = {
            "finite_quantile_edges": [float(value) for value in edges[1:-1]],
            **metric_summary(frame, proxy, truth_curve, residual_curve),
        }
    pd.concat(curve_parts, ignore_index=True).to_csv(args.curve_output, index=False)

    plt.rcParams.update({
        "font.size": 8.0, "axes.labelsize": 8.5, "axes.titlesize": 10.0,
        "xtick.labelsize": 7.0, "ytick.labelsize": 7.0,
        "font.family": "sans-serif",
    })
    fig, axes = plt.subplots(
        2, len(ALPHAS), figsize=(17.0, 6.6), sharey="row"
    )
    truth_ticks = np.asarray([
        -30, -10, -3, -1, -0.3, -0.1, 0.0, 0.1, 0.3, 1, 3, 10, 30,
    ], float)
    density = None
    for column, alpha in enumerate(ALPHAS):
        proxy = log_name(alpha)
        top, bottom = axes[:, column]
        density = plot_density(top, frame, proxy)
        truth_curve, residual_curve = curves[alpha]
        metrics = summary["alphas"][f"{alpha:g}"]

        top.errorbar(
            truth_curve.x, response_transform(truth_curve["mean"]),
            yerr=np.vstack((
                response_transform(truth_curve["mean"])
                - response_transform(truth_curve["mean"] - truth_curve["sem"]),
                response_transform(truth_curve["mean"] + truth_curve["sem"])
                - response_transform(truth_curve["mean"]),
            )),
            color="#D55E00", marker="o", markersize=3.0, linewidth=1.4,
            capsize=1.8, label="Case-balanced mean ± SEM",
        )
        top.set_yticks(response_transform(truth_ticks))
        top.set_yticklabels([f"{value:g}" for value in truth_ticks])
        top.set_title(rf"$\alpha={alpha:g}$")
        top.text(
            0.03, 0.96,
            rf"truth $\rho={metrics['individual_truth_spearman_rho']:+.3f}$",
            transform=top.transAxes, ha="left", va="top", fontsize=7.5,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.80, pad=2),
        )

        xlim = np.quantile(
            frame[proxy].to_numpy(float),
            [args.x_quantile_min, args.x_quantile_max],
        )
        hist_ax = bottom.twinx()
        counts, _, _ = hist_ax.hist(
            frame[proxy].to_numpy(float), bins=args.hist_bins, range=xlim,
            color="0.72", alpha=0.55, edgecolor="none", zorder=0,
        )
        hist_ax.set_ylim(0.0, max(float(counts.max()) / 0.32, 1.0))
        hist_ax.set_yticks([])
        hist_ax.spines[["top", "right", "left"]].set_visible(False)
        hist_ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        bottom.set_zorder(hist_ax.get_zorder() + 1)
        bottom.patch.set_visible(False)
        top.set_xlim(*xlim)
        bottom.errorbar(
            residual_curve.x, residual_curve["mean"], yerr=residual_curve["sem"],
            color="#0072B2", marker="o", markersize=3.0, linewidth=1.4,
            capsize=1.8, zorder=3,
        )
        bottom.axhline(0.0, color="0.45", linestyle="--", linewidth=1.0)
        bottom.set_xlim(*xlim)
        bottom.set_xlabel(rf"$\log_{{10}} Q_{{{alpha:g}}}$")
        bottom.text(
            0.03, 0.96,
            f"curve RMS={metrics['residual_curve_rms_about_global']:.3f}\n"
            f"upper 30% share={100*metrics['upper_30pct_gap_share']:.0f}%",
            transform=bottom.transAxes, ha="left", va="top", fontsize=7.2,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.80, pad=2),
            zorder=4,
        )
        for row, ax in enumerate((top, bottom)):
            ax.spines[["top", "right"]].set_visible(False)
            panel = row * len(ALPHAS) + column
            ax.text(-0.10, 1.04, chr(ord("A") + panel), transform=ax.transAxes,
                    fontsize=10.5, fontweight="bold", va="top")

    axes[0, 0].set_ylabel("Measured coherent response")
    axes[1, 0].set_ylabel(r"V2.2 residual $R_{\rm coherent}-P_s$")
    axes[0, 0].legend(frameon=False, fontsize=7.3, loc="lower right")
    colorbar = fig.colorbar(density, ax=axes[0, :], pad=0.009, fraction=0.012)
    colorbar.set_label("logarithmic anchor count per hexagon")
    fig.suptitle(
        r"Distance-exponent scan: $Q_\alpha=\sum_j(F_{s,j}/F_p)/d_j^\alpha$"
        "\nSame anchors and neighbour support; x axes display proxy quantiles "
        f"{100*args.x_quantile_min:.0f}--{100*args.x_quantile_max:.0f}%",
        fontsize=12,
    )
    fig.subplots_adjust(
        left=0.055, right=0.925, bottom=0.12, top=0.85,
        hspace=0.13, wspace=0.18,
    )
    fig.savefig(args.figure_output, dpi=300, bbox_inches="tight")
    fig.savefig(args.pdf_output, bbox_inches="tight")
    plt.close(fig)

    finite_metrics = {
        alpha: summary["alphas"][f"{alpha:g}"] for alpha in ALPHAS
    }
    summary["largest_truth_spearman_alpha"] = float(max(
        finite_metrics,
        key=lambda alpha: finite_metrics[alpha]["individual_truth_spearman_rho"],
    ))
    summary["largest_residual_curve_rms_alpha"] = float(max(
        finite_metrics,
        key=lambda alpha: finite_metrics[alpha]["residual_curve_rms_about_global"],
    ))
    with Path(args.summary_output).open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    print("ANCHOR_DISTANCE_EXPONENT_SCAN_DONE", flush=True)


if __name__ == "__main__":
    main()
