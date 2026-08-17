#!/usr/bin/env python3
"""Plot held-out coherent-anchor gap before and after bias-emulator correction."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "population",
    "feature",
    "bin",
    "feature_mean",
    "n_rows",
    "n_cases",
    "target_mean",
    "target_case_sem",
    "prediction_mean",
    "target_minus_prediction_mean",
    "target_minus_prediction_case_sem",
}


def configure_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 8.5,
        "axes.labelsize": 9.0,
        "axes.titlesize": 9.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 8.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def select_curve(table: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS.difference(table.columns)
    if missing:
        raise ValueError(f"curve table is missing columns: {sorted(missing)}")
    curve = table.loc[
        (table["population"] == "all")
        & (table["feature"] == "scene_prediction")
    ].sort_values("bin").copy()
    if curve.empty:
        raise ValueError("no all-anchor scene_prediction curve found")
    expected_bins = np.arange(len(curve), dtype=int)
    if not np.array_equal(curve["bin"].to_numpy(int), expected_bins):
        raise ValueError("scene-prediction curve bins are not contiguous")
    numeric = [
        "feature_mean",
        "target_mean",
        "target_case_sem",
        "prediction_mean",
        "target_minus_prediction_mean",
        "target_minus_prediction_case_sem",
    ]
    if not np.isfinite(curve[numeric].to_numpy(float)).all():
        raise ValueError("scene-prediction curve contains non-finite plot values")
    if (curve[["n_rows", "n_cases"]].to_numpy(int) <= 0).any():
        raise ValueError("scene-prediction curve contains an empty bin")
    return curve


def write_plot_table(curve: pd.DataFrame, output: Path) -> None:
    plotted = curve[[
        "bin",
        "lower",
        "upper",
        "feature_mean",
        "n_rows",
        "n_cases",
        "target_mean",
        "target_case_sem",
        "prediction_mean",
        "target_minus_prediction_mean",
        "target_minus_prediction_case_sem",
    ]].rename(columns={
        "feature_mean": "raw_scene_prediction_mean",
        "target_mean": "raw_gap_truth_minus_v22",
        "target_case_sem": "raw_gap_case_sem",
        "prediction_mean": "bias_emulator_prediction",
        "target_minus_prediction_mean": (
            "corrected_gap_truth_minus_v22_plus_bias_emulator"
        ),
        "target_minus_prediction_case_sem": "corrected_gap_case_sem",
    })
    plotted.to_csv(output, index=False)


def plot(curve: pd.DataFrame, output_prefix: Path) -> None:
    configure_style()
    x = curve["feature_mean"].to_numpy(float)
    raw = curve["target_mean"].to_numpy(float)
    raw_sem = curve["target_case_sem"].to_numpy(float)
    corrected = curve["target_minus_prediction_mean"].to_numpy(float)
    corrected_sem = curve[
        "target_minus_prediction_case_sem"
    ].to_numpy(float)

    fig, axis = plt.subplots(figsize=(5.15, 3.45), constrained_layout=True)
    axis.errorbar(
        x,
        raw,
        yerr=raw_sem,
        color="#0072B2",
        marker="o",
        markersize=4.2,
        linewidth=1.25,
        capsize=2.2,
        label="Raw V2.2",
    )
    axis.errorbar(
        x,
        corrected,
        yerr=corrected_sem,
        color="#D55E00",
        marker="s",
        markersize=4.0,
        linewidth=1.25,
        capsize=2.2,
        label="V2.2 + bias emulator",
    )
    axis.axhline(0.0, color="0.40", linestyle="--", linewidth=0.85)
    axis.set_xscale("symlog", linthresh=1.0e-3, linscale=1.0)
    axis.set_xlabel("Raw V2.2 scene prediction")
    axis.set_ylabel(r"Coherent truth $-$ prediction")
    axis.set_title(
        "Conditional coherent gap after bias-emulator correction\n"
        "Held-out c700–899; bins frozen on c400–699; errors are one case SEM"
    )
    axis.legend(frameon=False, loc="upper left")
    axis.spines[["top", "right"]].set_visible(False)

    fig.savefig(f"{output_prefix}.pdf", bbox_inches="tight")
    fig.savefig(f"{output_prefix}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curves", required=True)
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()

    curves_path = Path(args.curves)
    output_prefix = Path(args.output_prefix)
    outputs = [
        Path(f"{output_prefix}.pdf"),
        Path(f"{output_prefix}.png"),
        Path(f"{output_prefix}.csv"),
    ]
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(f"refusing existing outputs: {existing}")
    if not curves_path.is_file():
        raise FileNotFoundError(curves_path)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    curve = select_curve(pd.read_csv(curves_path))
    write_plot_table(curve, Path(f"{output_prefix}.csv"))
    plot(curve, output_prefix)
    print(
        f"saved {len(curve)} held-out bins to {output_prefix}.{{csv,pdf,png}}",
        flush=True,
    )


if __name__ == "__main__":
    main()
