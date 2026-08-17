"""Measured coherent response and V2.2 residual versus physical scene proxies.

Three candidate axes are evaluated on the exact same c400--899 coherent-anchor
population and exact deployed renderer-input neighbour support:

1. inverse-square flux ratio: sum (F_s/F_p) / d^2;
2. inverse-square surface-brightness ratio:
   sum (F_s/F_p) (Re_p/Re_s)^2 / d^2;
3. Gaussian overlap: sum (F_s/F_p) exp[-d^2/(2 W^2)], with
   W^2 = Re_p^2 + Re_s^2 + Re_PSF^2.

The top row is the individual-anchor density against the measured coherent
response, with equal-count case-balanced means.  The bottom row is the V2.2
truth-minus-prediction residual in the identical proxy bins.  Quantile edges
depend only on each physical proxy; no response or residual is used to define
them.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from scripts.plot_anchor_scene_flux_distance_proxy import (
    base_for_case,
    input_path,
    refuse,
)


PSF_RE = 0.5268
RESPONSE_SCALE = 0.1
PROXIES = {
    "log10_fluxratio_d2": {
        "title": "Flux ratio / separation²",
        "xlabel": r"$\log_{10}\sum_j (F_{s,j}/F_p)/d_j^2$ (arcsec$^{-2}$)",
        "raw": "fluxratio_d2",
    },
    "log10_sb_d2": {
        "title": "Surface-brightness ratio / separation²",
        "xlabel": (
            r"$\log_{10}\sum_j (F_{s,j}/F_p)(R_{e,p}/R_{e,s})^2/d_j^2$ "
            r"(arcsec$^{-2}$)"
        ),
        "raw": "sb_d2",
    },
    "log10_overlap": {
        "title": "Size + PSF overlap",
        "xlabel": (
            r"$\log_{10}\sum_j (F_{s,j}/F_p)"
            r"\exp[-d_j^2/(2W_j^2)]$"
        ),
        "raw": "overlap",
    },
}


def case_proxy(case: int, base_root: Path, pair_prefix: str,
               shear: float, expected: pd.DataFrame) -> pd.DataFrame:
    base = base_for_case(base_root, case)
    pairs = pd.read_feather(
        base / f"{pair_prefix}_case{case}.feather",
        columns=["anchor_index", "secondary_index", "distance", "response"],
    )
    catalogue = pd.read_feather(
        input_path(base, case, shear),
        columns=["index_input", "r_input", "Re_input"],
    ).rename(columns={"index_input": "index"})
    if catalogue["index"].duplicated().any():
        raise RuntimeError(f"case {case}: duplicate rendered input ID")
    latent = catalogue.set_index("index")
    secondary = latent.reindex(pairs.secondary_index)
    primary = latent.reindex(pairs.anchor_index)
    secondary_mag = secondary.r_input.to_numpy(float)
    primary_mag = primary.r_input.to_numpy(float)
    secondary_re = secondary.Re_input.to_numpy(float)
    primary_re = primary.Re_input.to_numpy(float)
    distance = pairs.distance.to_numpy(float)
    arrays = [secondary_mag, primary_mag, secondary_re, primary_re, distance]
    if any(not np.isfinite(value).all() for value in arrays):
        raise RuntimeError(f"case {case}: non-finite physical pair input")
    if np.any(secondary_re <= 0.0) or np.any(primary_re <= 0.0) or np.any(distance <= 0.0):
        raise RuntimeError(f"case {case}: non-positive size or distance")

    ratio = np.power(10.0, -0.4 * (secondary_mag - primary_mag))
    inv_d2 = np.reciprocal(np.square(distance))
    width2 = np.square(primary_re) + np.square(secondary_re) + PSF_RE**2
    terms = pd.DataFrame({
        "input_index": pairs.anchor_index.to_numpy(np.int64),
        "fluxratio_d2": ratio * inv_d2,
        "sb_d2": ratio * np.square(primary_re / secondary_re) * inv_d2,
        "overlap": ratio * np.exp(-0.5 * np.square(distance) / width2),
        "pair_response": pairs.response.to_numpy(float),
    })
    scene = terms.groupby("input_index", sort=False).agg(
        fluxratio_d2=("fluxratio_d2", "sum"),
        sb_d2=("sb_d2", "sum"),
        overlap=("overlap", "sum"),
        pair_response_sum=("pair_response", "sum"),
        n_pairs=("pair_response", "size"),
    ).reset_index()
    scene["case"] = case
    merged = expected.merge(
        scene, on=["case", "input_index"], how="left", validate="one_to_one"
    )
    no_pair = merged.fluxratio_d2.isna()
    fill = ["fluxratio_d2", "sb_d2", "overlap", "pair_response_sum", "n_pairs"]
    merged.loc[no_pair, fill] = 0.0
    merged["n_pairs"] = merged.n_pairs.astype(np.int64)
    replay = float(np.max(np.abs(
        merged.scene_prediction.to_numpy(float)
        - merged.pair_response_sum.to_numpy(float)
    )))
    if replay > 5e-6:
        raise RuntimeError(f"case {case}: V2.2 replay mismatch {replay:.3e}")
    print(
        f"case {case}: anchors={len(merged):,} pairs={len(pairs):,} "
        f"zero_pair={int(no_pair.sum())} replay={replay:.2e}", flush=True,
    )
    return merged


def log_with_zero_floor(value: np.ndarray) -> tuple[np.ndarray, float]:
    value = np.asarray(value, float)
    positive = value[value > 0.0]
    if not len(positive):
        raise RuntimeError("proxy has no positive values")
    floor = float(0.5 * positive.min())
    return np.log10(np.maximum(value, floor)), floor


def response_transform(value: np.ndarray) -> np.ndarray:
    return np.arcsinh(np.asarray(value, float) / RESPONSE_SCALE)


def proxy_edges(value: np.ndarray, n_bins: int) -> np.ndarray:
    edges = np.unique(np.quantile(value, np.linspace(0.0, 1.0, n_bins + 1)))
    if len(edges) != n_bins + 1:
        raise RuntimeError("physical-proxy quantile edges collapsed")
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def case_balanced_curve(frame: pd.DataFrame, proxy: str, target: str,
                        edges: np.ndarray) -> pd.DataFrame:
    value = frame[proxy].to_numpy(float)
    index = np.clip(
        np.searchsorted(edges, value, side="right") - 1, 0, len(edges) - 2
    )
    work = pd.DataFrame({
        "case": frame.case.to_numpy(np.int64),
        "bin": index,
        "proxy": value,
        "target": frame[target].to_numpy(float),
    })
    x = work.groupby("bin", sort=True).proxy.median()
    count = work.groupby("bin", sort=True).size()
    by_case = work.groupby(["case", "bin"], sort=True).target.mean().unstack()
    mean = by_case.mean(axis=0)
    sem = by_case.std(axis=0, ddof=1) / np.sqrt(by_case.count(axis=0))
    return pd.DataFrame({
        "bin": x.index.to_numpy(int),
        "x": x.to_numpy(float),
        "mean": mean.reindex(x.index).to_numpy(float),
        "sem": sem.reindex(x.index).to_numpy(float),
        "n_anchors": count.reindex(x.index).to_numpy(int),
        "n_cases": by_case.count(axis=0).reindex(x.index).to_numpy(int),
    })


def correlation_summary(frame: pd.DataFrame, proxy: str,
                        response_curve: pd.DataFrame) -> dict:
    x = frame[proxy].to_numpy(float)
    truth = frame.R_blend_truth.to_numpy(float)
    residual = frame.bias_truth_minus_model.to_numpy(float)
    truth_spearman = stats.spearmanr(x, truth)
    residual_spearman = stats.spearmanr(x, residual)
    curve_spearman = stats.spearmanr(
        response_curve.x.to_numpy(float), response_curve["mean"].to_numpy(float)
    )
    return {
        "individual_truth_spearman_rho": float(truth_spearman.statistic),
        "individual_truth_spearman_p": float(truth_spearman.pvalue),
        "individual_residual_spearman_rho": float(residual_spearman.statistic),
        "individual_residual_spearman_p": float(residual_spearman.pvalue),
        "binned_case_mean_truth_spearman_rho": float(curve_spearman.statistic),
        "binned_case_mean_truth_spearman_p": float(curve_spearman.pvalue),
    }


def plot_density(ax, frame: pd.DataFrame, proxy: str):
    x = frame[proxy].to_numpy(float)
    y = response_transform(frame.R_blend_truth.to_numpy(float))
    return ax.hexbin(
        x, y, gridsize=(90, 80), bins="log", mincnt=1,
        cmap="viridis", linewidths=0.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True)
    parser.add_argument("--base-root", required=True)
    parser.add_argument("--pair-prefix", default="pairs_renderer_v22")
    parser.add_argument("--case-min", type=int, default=400)
    parser.add_argument("--case-max", type=int, default=899)
    parser.add_argument("--shear", type=float, default=0.02)
    parser.add_argument("--n-bins", type=int, default=20)
    parser.add_argument("--table-output", required=True)
    parser.add_argument("--curve-output", required=True)
    parser.add_argument("--figure-output", required=True)
    parser.add_argument("--pdf-output", required=True)
    parser.add_argument("--summary-output", required=True)
    args = parser.parse_args()

    outputs = [Path(args.table_output), Path(args.curve_output),
               Path(args.figure_output), Path(args.pdf_output),
               Path(args.summary_output)]
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

    base_root = Path(args.base_root)
    parts = []
    for case in cases:
        expected = features[features.case == case].copy()
        parts.append(case_proxy(
            int(case), base_root, args.pair_prefix, args.shear, expected
        ))
    frame = pd.concat(parts, ignore_index=True)
    floors = {}
    for proxy, spec in PROXIES.items():
        frame[proxy], floors[spec["raw"]] = log_with_zero_floor(
            frame[spec["raw"]].to_numpy(float)
        )

    table_columns = [
        "case", "input_index", "R_blend_truth", "scene_prediction",
        "bias_truth_minus_model", "n_pairs",
        *[spec["raw"] for spec in PROXIES.values()], *PROXIES,
    ]
    frame[table_columns].to_feather(args.table_output)

    curve_parts = []
    summary = {
        "dataset": f"coherent_anchor_c{args.case_min}-{args.case_max}",
        "n_anchors": int(len(frame)), "n_cases": int(len(cases)),
        "n_bins": int(args.n_bins),
        "gap_definition": "R_blend_truth - frozen V2.2 scene prediction",
        "bin_definition": "global equal-count bins using physical proxy only",
        "uncertainty": "one SEM across per-case bin means",
        "psf_re_arcsec": PSF_RE,
        "zero_proxy_plot_floors": floors,
        "proxies": {},
    }
    curves = {}
    for proxy, spec in PROXIES.items():
        edges = proxy_edges(frame[proxy].to_numpy(float), args.n_bins)
        truth_curve = case_balanced_curve(frame, proxy, "R_blend_truth", edges)
        residual_curve = case_balanced_curve(
            frame, proxy, "bias_truth_minus_model", edges
        )
        truth_curve["target"] = "coherent_truth"
        residual_curve["target"] = "truth_minus_v22"
        for curve in (truth_curve, residual_curve):
            curve["proxy"] = proxy
        curve_parts.extend([truth_curve, residual_curve])
        curves[proxy] = (truth_curve, residual_curve)
        summary["proxies"][proxy] = {
            "definition": spec["xlabel"],
            "finite_quantile_edges": [
                float(value) for value in edges[1:-1]
            ],
            "correlations": correlation_summary(frame, proxy, truth_curve),
            "lowest_bin_residual": {
                "mean": float(residual_curve.iloc[0]["mean"]),
                "case_sem": float(residual_curve.iloc[0]["sem"]),
            },
            "highest_bin_residual": {
                "mean": float(residual_curve.iloc[-1]["mean"]),
                "case_sem": float(residual_curve.iloc[-1]["sem"]),
            },
        }
    pd.concat(curve_parts, ignore_index=True).to_csv(args.curve_output, index=False)

    plt.rcParams.update({
        "font.size": 8.5, "axes.labelsize": 9.5, "axes.titlesize": 10.5,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "font.family": "sans-serif",
    })
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 7.0), sharex="col")
    truth_ticks = np.asarray([
        -30, -10, -3, -1, -0.3, -0.1, 0.0, 0.1, 0.3, 1, 3, 10, 30,
    ], float)
    density = None
    for column, (proxy, spec) in enumerate(PROXIES.items()):
        top, bottom = axes[:, column]
        density = plot_density(top, frame, proxy)
        truth_curve, residual_curve = curves[proxy]
        top.errorbar(
            truth_curve.x, response_transform(truth_curve["mean"]),
            yerr=np.vstack((
                response_transform(truth_curve["mean"])
                - response_transform(truth_curve["mean"] - truth_curve["sem"]),
                response_transform(truth_curve["mean"] + truth_curve["sem"])
                - response_transform(truth_curve["mean"]),
            )),
            color="#D55E00", marker="o", markersize=3.5, linewidth=1.5,
            capsize=2, label="Case-balanced mean ± SEM",
        )
        top.set_yticks(response_transform(truth_ticks))
        top.set_yticklabels([f"{value:g}" for value in truth_ticks])
        top.set_title(spec["title"])
        rho = summary["proxies"][proxy]["correlations"]
        top.text(
            0.03, 0.97,
            f"Individual $\\rho$={rho['individual_truth_spearman_rho']:+.3f}\n"
            f"20-bin mean $\\rho$={rho['binned_case_mean_truth_spearman_rho']:+.3f}",
            transform=top.transAxes, ha="left", va="top", fontsize=8,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=2.5),
        )
        bottom.errorbar(
            residual_curve.x, residual_curve["mean"], yerr=residual_curve["sem"],
            color="#0072B2", marker="o", markersize=3.5, linewidth=1.5,
            capsize=2,
        )
        bottom.axhline(0.0, color="0.45", linestyle="--", linewidth=1.0)
        bottom.set_xlabel(spec["xlabel"])
        for row, ax in enumerate((top, bottom)):
            ax.spines[["top", "right"]].set_visible(False)
            panel = row * len(PROXIES) + column
            ax.text(-0.10, 1.04, chr(ord("A") + panel), transform=ax.transAxes,
                    fontsize=11, fontweight="bold", va="top")
    axes[0, 0].set_ylabel("Measured coherent response")
    axes[1, 0].set_ylabel(r"V2.2 residual $R_{\rm coherent}-P_s$")
    axes[0, 0].legend(frameon=False, fontsize=8, loc="lower right")
    colorbar = fig.colorbar(density, ax=axes[0, :], pad=0.012, fraction=0.025)
    colorbar.set_label("logarithmic anchor count per hexagon")
    fig.suptitle(
        "Coherent-anchor measurement and V2.2 residual in fixed physical-scene bins\n"
        f"Same {len(frame):,} anchors, cases {args.case_min}--{args.case_max}; "
        "top response axes use an asinh stretch",
        fontsize=12,
    )
    fig.subplots_adjust(
        left=0.075, right=0.92, bottom=0.12, top=0.87, hspace=0.13, wspace=0.16
    )
    fig.savefig(args.figure_output, dpi=300, bbox_inches="tight")
    fig.savefig(args.pdf_output, bbox_inches="tight")
    plt.close(fig)

    with Path(args.summary_output).open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    print("ANCHOR_PHYSICAL_PROXY_TRUTH_RESIDUAL_DONE", flush=True)


if __name__ == "__main__":
    main()
