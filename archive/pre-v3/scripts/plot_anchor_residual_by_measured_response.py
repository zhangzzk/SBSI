#!/usr/bin/env python3
"""Plot coherent-anchor residuals conditioned on measured anchor response."""

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


MODELS = {
    "V2.2": "R_blend_v22_replay",
    "Tuned pair": "R_blend_pair",
    "Tuned pair+scene": "R_blend_pair_scene",
}
COLORS = {
    "V2.2": "#0072B2",
    "Tuned pair": "#D55E00",
    "Tuned pair+scene": "#009E73",
}
MARKERS = {"V2.2": "o", "Tuned pair": "s", "Tuned pair+scene": "v"}
LINESTYLES = {"V2.2": "--", "Tuned pair": "-", "Tuned pair+scene": "-."}


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def quantile_edges(values: np.ndarray, n_bins: int) -> np.ndarray:
    edges = np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1))
    edges[0] = -np.inf
    edges[-1] = np.inf
    if np.any(np.diff(edges) <= 0.0):
        raise RuntimeError("measured-response quantile edges are not unique")
    return edges


def summarize(
    case: np.ndarray,
    truth: np.ndarray,
    prediction: np.ndarray,
    edges: np.ndarray,
) -> list[dict]:
    n_bins = len(edges) - 1
    cases = np.unique(case)
    bin_index = np.searchsorted(edges[1:-1], truth, side="right")
    residual = truth - prediction
    rows = []
    for index in range(n_bins):
        mask = bin_index == index
        frame = pd.DataFrame({
            "case": case[mask],
            "truth": truth[mask],
            "residual": residual[mask],
        })
        grouped = frame.groupby("case", sort=True).agg(
            n=("residual", "size"),
            truth=("truth", "mean"),
            residual=("residual", "mean"),
        )
        if not np.array_equal(grouped.index.to_numpy(), cases):
            raise RuntimeError(f"bin {index} does not contain every rendered case")
        rows.append({
            "bin": int(index),
            "lower": None if not np.isfinite(edges[index]) else float(edges[index]),
            "upper": None if not np.isfinite(edges[index + 1]) else float(edges[index + 1]),
            "n_anchors": int(mask.sum()),
            "measured_response": {
                "case_balanced_mean": float(grouped.truth.mean()),
                "case_sem": float(grouped.truth.std(ddof=1) / np.sqrt(len(grouped))),
            },
            "residual": {
                "case_balanced_mean": float(grouped.residual.mean()),
                "case_sem": float(grouped.residual.std(ddof=1) / np.sqrt(len(grouped))),
                "row_weighted_mean": float(residual[mask].mean()),
            },
        })
    return rows


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 9.0,
        "axes.labelsize": 10.0,
        "axes.titlesize": 10.5,
        "xtick.labelsize": 8.2,
        "ytick.labelsize": 8.2,
        "legend.fontsize": 8.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    })


def make_plot(payload: dict, output_prefix: Path) -> None:
    configure_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.3))
    handles = []
    for panel, axis in enumerate(axes):
        for model in MODELS:
            rows = payload["curves"][model]
            if panel == 1:
                rows = rows[1:-1]
            x = np.asarray([
                row["measured_response"]["case_balanced_mean"] for row in rows
            ])
            y = np.asarray([
                row["residual"]["case_balanced_mean"] for row in rows
            ])
            sem = np.asarray([row["residual"]["case_sem"] for row in rows])
            handle = axis.errorbar(
                x,
                y,
                yerr=sem,
                color=COLORS[model],
                marker=MARKERS[model],
                linestyle=LINESTYLES[model],
                markersize=4.2,
                linewidth=1.35,
                capsize=2.0,
                alpha=0.97,
                label=model,
            )
            if panel == 0:
                handles.append(handle)
        axis.axhline(0.0, color="0.42", linestyle=":", linewidth=0.9)
        axis.set_xscale("symlog", linthresh=1.0e-2, linscale=1.0)
        axis.set_xlabel(
            r"Measured coherent-anchor response $R_{\rm blend}^{\rm anchor}$"
        )
        axis.set_ylabel(
            r"Mean residual $R_{\rm blend}^{\rm anchor}-\widehat R_{\rm blend}$"
        )
        axis.spines[["top", "right"]].set_visible(False)
        axis.text(
            -0.12,
            1.03,
            "AB"[panel],
            transform=axis.transAxes,
            fontsize=11,
            fontweight="bold",
            va="top",
        )
    axes[0].set_yscale("symlog", linthresh=2.0e-2, linscale=1.0)
    axes[0].set_title("Full measured-response range")
    axes[1].set_title("Central 90% of measured responses")
    axes[1].set_ylim(-0.24, 0.24)
    fig.legend(
        handles,
        list(MODELS),
        title="Pair-response emulator",
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=3,
    )
    fig.suptitle(
        "Coherent anchors: common conditioning on measured response",
        fontsize=11.2,
        y=0.98,
    )
    fig.text(
        0.5,
        0.055,
        "20 equal-population bins; error bars are one SEM across 500 rendered cases",
        ha="center",
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.13, 1, 0.94), w_pad=2.2)
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-dir", required=True)
    parser.add_argument("--anchor-features", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--n-bins", type=int, default=20)
    args = parser.parse_args()

    score_dir = Path(args.score_dir).resolve()
    anchor_path = Path(args.anchor_features).resolve()
    output_prefix = Path(args.output_prefix).resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    outputs = [
        output_prefix.with_suffix(suffix)
        for suffix in (".json", ".csv", ".pdf", ".png")
    ]
    if existing := [str(path) for path in outputs if path.exists()]:
        raise FileExistsError(f"refusing existing outputs: {existing}")

    score = pd.concat(
        [
            pd.read_feather(score_dir / f"case{case}.feather")
            for case in range(400, 900)
        ],
        ignore_index=True,
    )
    truth_frame = pd.read_feather(
        anchor_path, columns=["case", "input_index", "R_blend_truth"]
    )
    joined = truth_frame.merge(
        score,
        on=["case", "input_index"],
        how="left",
        validate="one_to_one",
    )
    if len(joined) != 1_703_884 or joined.isna().any().any():
        raise RuntimeError("anchor truth/score coverage differs")
    case = joined.case.to_numpy(np.int16)
    if not np.array_equal(np.unique(case), np.arange(400, 900)):
        raise RuntimeError("expected coherent-anchor cases 400--899")
    truth = joined.R_blend_truth.to_numpy(np.float64)
    edges = quantile_edges(truth, args.n_bins)

    curves = {
        model: summarize(
            case,
            truth,
            joined[column].to_numpy(np.float64),
            edges,
        )
        for model, column in MODELS.items()
    }
    records = []
    for model, rows in curves.items():
        for row in rows:
            records.append({
                "model": model,
                "bin": row["bin"],
                "lower": row["lower"],
                "upper": row["upper"],
                "n_anchors": row["n_anchors"],
                "measured_response": row["measured_response"]["case_balanced_mean"],
                "residual": row["residual"]["case_balanced_mean"],
                "residual_case_sem": row["residual"]["case_sem"],
            })
    pd.DataFrame(records).to_csv(output_prefix.with_suffix(".csv"), index=False)
    payload = {
        "schema_version": 1,
        "dataset": "coherent_anchor_c400_899",
        "n_anchors": int(len(joined)),
        "n_cases": int(len(np.unique(case))),
        "n_bins": int(args.n_bins),
        "conditioning_coordinate": "measured coherent-anchor R_blend_truth",
        "binning": "global equal-population quantiles; common membership for all models",
        "residual_definition": "R_blend_truth - R_blend_prediction",
        "uncertainty": "one SEM across per-case conditional means",
        "curves": curves,
    }
    atomic_json(output_prefix.with_suffix(".json"), payload)
    make_plot(payload, output_prefix)
    print(json.dumps({
        "outputs": [str(path) for path in outputs],
        "rightmost_bin": {
            model: rows[-1]["residual"] for model, rows in curves.items()
        },
    }, indent=2, sort_keys=True, allow_nan=False))
    print("ANCHOR_RESIDUAL_BY_MEASURED_RESPONSE_PLOT_DONE", flush=True)


if __name__ == "__main__":
    main()
