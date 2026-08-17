"""Replot the physical-proxy anchor diagnostic with robust x-axis limits."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.plot_anchor_physical_proxy_truth_residual import (
    PROXIES,
    plot_density,
    response_transform,
)


def refuse(paths: list[Path]) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError("refusing to overwrite: " + ", ".join(existing))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", required=True)
    parser.add_argument("--curves", required=True)
    parser.add_argument("--figure-output", required=True)
    parser.add_argument("--pdf-output", required=True)
    parser.add_argument("--x-quantile-min", type=float, default=0.20)
    parser.add_argument("--x-quantile-max", type=float, default=0.98)
    parser.add_argument("--hist-bins", type=int, default=32)
    args = parser.parse_args()

    outputs = [Path(args.figure_output), Path(args.pdf_output)]
    refuse(outputs)
    for output in outputs:
        output.parent.mkdir(parents=True, exist_ok=True)

    frame = pd.read_feather(args.table)
    curves = pd.read_csv(args.curves)
    if not 0.0 <= args.x_quantile_min < args.x_quantile_max <= 1.0:
        raise ValueError("invalid x-axis quantile limits")

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
        selected = curves[curves.proxy.eq(proxy)]
        truth_curve = selected[selected.target.eq("coherent_truth")]
        residual_curve = selected[selected.target.eq("truth_minus_v22")]

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

        bottom.errorbar(
            residual_curve.x, residual_curve["mean"], yerr=residual_curve["sem"],
            color="#0072B2", marker="o", markersize=3.5, linewidth=1.5,
            capsize=2, zorder=3,
        )
        bottom.axhline(0.0, color="0.45", linestyle="--", linewidth=1.0)
        bottom.set_xlabel(spec["xlabel"])
        bottom.set_xlim(*xlim)
        bottom.text(
            0.98, 0.04, "anchor histogram", transform=bottom.transAxes,
            ha="right", va="bottom", color="0.38", fontsize=7,
        )
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
        f"X axes display proxy quantiles {100*args.x_quantile_min:.0f}--"
        f"{100*args.x_quantile_max:.0f}%; gray histograms show anchor distributions",
        fontsize=12,
    )
    fig.subplots_adjust(
        left=0.075, right=0.92, bottom=0.12, top=0.87, hspace=0.13, wspace=0.16
    )
    fig.savefig(args.figure_output, dpi=300, bbox_inches="tight")
    fig.savefig(args.pdf_output, bbox_inches="tight")
    plt.close(fig)

    print("ANCHOR_PHYSICAL_PROXY_TRUTH_RESIDUAL_ZOOM_DONE", flush=True)


if __name__ == "__main__":
    main()
