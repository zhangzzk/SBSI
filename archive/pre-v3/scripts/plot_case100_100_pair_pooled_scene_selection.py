#!/usr/bin/env python3
"""Plot half-shear-only selection diagnostics for pooled scene moments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-stem", required=True)
    args = parser.parse_args()
    run = Path(args.run_dir).resolve()
    stem = Path(args.output_stem).resolve()
    stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        if stem.with_suffix(suffix).exists():
            raise FileExistsError(stem.with_suffix(suffix))
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    metrics = pd.read_csv(run / "selection_metrics.csv").sort_values("lambda")
    curves = pd.read_csv(run / "selection_curves.csv")
    selected = float(summary["selection"]["selected_lambda"])

    plt.rcParams.update({
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.5), constrained_layout=True)
    axis = axes[0, 0]
    axis.plot(
        metrics["lambda"], 100.0 * (metrics.pair_mse_over_raw - 1.0),
        color="#0072B2", marker="o", linewidth=1.2,
    )
    axis.axhline(0.0, color="0.55", linestyle=":", linewidth=0.8)
    axis.axvline(selected, color="#D55E00", linestyle="--", linewidth=1.0)
    axis.set_xscale("symlog", linthresh=0.001)
    axis.set_xlabel(r"Scene-moment strength $\lambda$")
    axis.set_ylabel("Pair MSE change from raw base (%)")
    axis.set_title("Pair-level validation loss")

    axis = axes[0, 1]
    axis.plot(
        metrics["lambda"], metrics.normalized_scene_score,
        color="#009E73", marker="s", linewidth=1.2,
    )
    axis.axhline(1.0, color="0.55", linestyle=":", linewidth=0.8)
    axis.axvline(selected, color="#D55E00", linestyle="--", linewidth=1.0)
    axis.set_xscale("symlog", linthresh=0.001)
    axis.set_xlabel(r"Scene-moment strength $\lambda$")
    axis.set_ylabel("Combined scene-curve RMS / raw")
    axis.set_title("Half-shear selection score")

    styles = {
        "Raw base": ("#0072B2", "o", "-"),
        "lambda=0": ("#009E73", "^", "-."),
        f"lambda={selected:g}": ("#D55E00", "s", "--"),
    }
    for panel, (axis_name, title, xlabel) in enumerate((
        (
            "full_base_scene_prediction",
            "Residual by full-scene base prediction",
            "Full-scene prediction quantile bin",
        ),
        (
            "log10_q_d2",
            r"Residual by physical $Q_{d2}$",
            r"$\log_{10}Q_{d2}$ quantile bin",
        ),
    )):
        axis = axes[1, panel]
        for model, (color, marker, linestyle) in styles.items():
            if model == "Raw base":
                local = curves.loc[(curves.axis == axis_name) & curves["lambda"].isna()]
            else:
                value = 0.0 if model == "lambda=0" else selected
                local = curves.loc[
                    (curves.axis == axis_name)
                    & np.isclose(curves["lambda"], value, rtol=0.0, atol=1.0e-12)
                ]
            local = local.sort_values("bin")
            axis.errorbar(
                local.bin,
                local.residual_mean,
                yerr=local.residual_case_sem,
                color=color,
                marker=marker,
                linestyle=linestyle,
                linewidth=1.1,
                markersize=3.5,
                capsize=2.0,
                label=model,
            )
        axis.axhline(0.0, color="0.55", linestyle=":", linewidth=0.8)
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Mean summed residual")
        axis.set_title(title)
        axis.legend(frameon=False)

    for label, axis in zip("ABCD", axes.ravel()):
        axis.text(
            -0.14, 1.08, label, transform=axis.transAxes,
            fontsize=11, fontweight="bold", va="top",
        )
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Pair + pooled-scene correction selected only on half-shear cases 160–199\n"
        "error bars in conditional panels: one SEM across cases",
        fontsize=10,
    )
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"selected_lambda={selected:g}")
    print("PLOT_PAIR_POOLED_SCENE_SELECTION_DONE", flush=True)


if __name__ == "__main__":
    main()
