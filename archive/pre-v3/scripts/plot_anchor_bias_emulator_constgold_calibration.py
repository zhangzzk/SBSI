"""Plot frozen anchor-bias predictions against noisy ConstGold gap labels.

This is a presentation-only consumer of the already-frozen transfer result.
No model is fit and no bin boundary is chosen from ConstGold: the ten bins and
their boundaries were frozen on coherent-anchor development cases by the
transfer analysis.  Vertical errors use rendered ConstGold case as the unit.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd


REQUIRED = {
    "bin",
    "n_rows",
    "n_cases",
    "raw_gap_mean",
    "raw_gap_case_sem",
    "predicted_bias_mean",
    "predicted_bias_case_sem",
    "corrected_gap_mean",
    "corrected_gap_case_sem",
}


def read_inputs(deciles_csv: str, summary_json: str) -> tuple[pd.DataFrame, dict]:
    frame = pd.read_csv(deciles_csv)
    missing = REQUIRED - set(frame.columns)
    if missing:
        raise KeyError(f"decile table lacks {sorted(missing)}")
    if frame.empty or frame.bin.duplicated().any():
        raise RuntimeError("decile table is empty or has duplicate bins")
    frame = frame.sort_values("predicted_bias_mean").reset_index(drop=True)
    numeric = frame[list(REQUIRED - {"bin"})].to_numpy(float)
    if not np.isfinite(numeric).all():
        raise RuntimeError("decile table contains non-finite plot values")
    with open(summary_json, encoding="utf-8") as handle:
        summary = json.load(handle)
    expected = summary["population"]
    if int(frame.n_rows.sum()) != int(expected["n_rows"]):
        raise RuntimeError("decile row count does not close to summary population")
    if not np.all(frame.n_cases.to_numpy(int) == int(expected["n_cases"])):
        raise RuntimeError("decile case count disagrees with summary population")
    return frame, summary


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 8.0,
        "axes.labelsize": 8.5,
        "axes.titlesize": 8.5,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 7.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    })


def plot_curve(frame: pd.DataFrame, summary: dict, output_prefix: str) -> None:
    configure_style()
    prediction = frame.predicted_bias_mean.to_numpy(float)
    prediction_sem = frame.predicted_bias_case_sem.to_numpy(float)
    label = frame.raw_gap_mean.to_numpy(float)
    label_sem = frame.raw_gap_case_sem.to_numpy(float)
    remaining = frame.corrected_gap_mean.to_numpy(float)
    remaining_sem = frame.corrected_gap_case_sem.to_numpy(float)

    fig, axes = plt.subplots(
        1, 2, figsize=(7.1, 3.15), constrained_layout=True
    )
    axes[0].errorbar(
        prediction, label, xerr=prediction_sem, yerr=label_sem,
        color="#0072B2", marker="o", markersize=4.2, linewidth=1.15,
        capsize=2.0, label="ConstGold label",
    )
    lower = float(np.min(np.r_[prediction - prediction_sem, label - label_sem]))
    upper = float(np.max(np.r_[prediction + prediction_sem, label + label_sem]))
    pad = 0.07 * max(upper - lower, 1.0e-3)
    limits = (lower - pad, upper + pad)
    axes[0].plot(
        limits, limits, color="0.35", linestyle="--", linewidth=0.9,
        label="Label = prediction",
    )
    axes[0].set_xlim(limits)
    axes[0].set_ylim(limits)
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set_xlabel("Frozen anchor-bias prediction")
    axes[0].set_ylabel(r"Mean ConstGold $R_{\rm sim}-R_{\rm model}$")
    axes[0].set_title("Conditional transfer curve")
    axes[0].legend(frameon=False)

    axes[1].errorbar(
        prediction, remaining, xerr=prediction_sem, yerr=remaining_sem,
        color="#D55E00", marker="o", markersize=4.2, linewidth=1.15,
        capsize=2.0,
    )
    axes[1].axhline(0.0, color="0.35", linestyle="--", linewidth=0.9)
    axes[1].set_xlim(limits)
    axes[1].set_xlabel("Frozen anchor-bias prediction")
    axes[1].set_ylabel("Mean label - prediction")
    axes[1].set_title("Gap remaining after correction")

    for panel, axis in enumerate(axes):
        axis.spines[["top", "right"]].set_visible(False)
        axis.text(
            -0.14, 1.06, chr(ord("A") + panel), transform=axis.transAxes,
            fontweight="bold", fontsize=10, va="top",
        )
    conditional = summary["conditional_transfer"]
    fig.suptitle(
        "Anchor-bias emulator on V2.2 ConstGold (cases 40-139)\n"
        f"Frozen bins; error bars: SEM across "
        f"{summary['population']['n_cases']} cases; "
        f"slope={conditional['bin_mean_slope_total_gap_on_predicted_blend_bias']:.3f}",
        fontsize=9.2,
    )
    fig.savefig(f"{output_prefix}.pdf", bbox_inches="tight")
    fig.savefig(f"{output_prefix}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--deciles-csv", required=True)
    ap.add_argument("--summary-json", required=True)
    ap.add_argument("--output-prefix", required=True)
    args = ap.parse_args()
    outputs = [f"{args.output_prefix}.{suffix}" for suffix in ("png", "pdf")]
    existing = [path for path in outputs if os.path.exists(path)]
    if existing:
        raise FileExistsError(f"refusing existing outputs: {existing}")
    Path(args.output_prefix).parent.mkdir(parents=True, exist_ok=True)
    frame, summary = read_inputs(args.deciles_csv, args.summary_json)
    plot_curve(frame, summary, args.output_prefix)
    print("ANCHOR_BIAS_EMULATOR_CONSTGOLD_CALIBRATION_PLOT_DONE")


if __name__ == "__main__":
    main()
