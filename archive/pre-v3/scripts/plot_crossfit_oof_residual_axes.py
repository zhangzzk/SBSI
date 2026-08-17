#!/usr/bin/env python3
"""Plot three half-shear emulators' OOF residuals over five coordinates."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from string import ascii_uppercase

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd

from scripts.plot_v22_emulator_label_calibration import quantile_edges
from scripts.v22_grouped_rscene_common import (
    load_source_metadata,
    mmap_array,
    sha256,
    strict_json,
)


MODEL_ORDER = ("V2.2", "Tuned pair", "Tuned pair+scene")
COLORS = {
    "V2.2": "#0072B2",
    "Tuned pair": "#D55E00",
    "Tuned pair+scene": "#009E73",
}
MARKERS = {"V2.2": "o", "Tuned pair": "s", "Tuned pair+scene": "v"}
LINESTYLES = {"V2.2": "--", "Tuned pair": "-", "Tuned pair+scene": "-."}
RAW_COLUMNS = {
    "primary_mag": 2,
    "secondary_mag": 3,
    "separation": 6,
}


def finite_stat(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        raise RuntimeError("statistic needs at least two finite cases")
    return {
        "mean": float(values.mean()),
        "case_sem": float(values.std(ddof=1) / np.sqrt(len(values))),
        "n_cases": int(len(values)),
    }


def summarize_bins(
    case: np.ndarray,
    coordinate: np.ndarray,
    residual: np.ndarray,
    edges: np.ndarray,
    *,
    item_name: str,
) -> list[dict]:
    case = np.asarray(case, dtype=np.int64)
    coordinate = np.asarray(coordinate, dtype=np.float64)
    residual = np.asarray(residual, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.float64)
    n_cases = 200
    n_bins = len(edges) - 1
    bin_index = np.searchsorted(edges[1:-1], coordinate, side="right")
    flat = case * n_bins + bin_index
    length = n_cases * n_bins
    count = np.bincount(flat, minlength=length).reshape(n_cases, n_bins)
    coordinate_sum = np.bincount(
        flat, weights=coordinate, minlength=length
    ).reshape(n_cases, n_bins)
    residual_sum = np.bincount(
        flat, weights=residual, minlength=length
    ).reshape(n_cases, n_bins)
    rows = []
    for index in range(n_bins):
        present = count[:, index] > 0
        local_count = count[present, index]
        case_coordinate = coordinate_sum[present, index] / local_count
        case_residual = residual_sum[present, index] / local_count
        rows.append({
            "bin": int(index),
            "lower": float(edges[index]),
            "upper": float(edges[index + 1]),
            f"n_{item_name}": int(local_count.sum()),
            "coordinate": finite_stat(case_coordinate),
            "residual": finite_stat(case_residual),
        })
    return rows


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 8.5,
        "axes.labelsize": 9.0,
        "axes.titlesize": 9.2,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 8.2,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    })


def make_plot(payload: dict, output_prefix: Path) -> None:
    configure_style()
    fig, axes = plt.subplots(2, 3, figsize=(11.0, 6.4))
    panels = [
        ("primary_mag", "Primary magnitude $r_p$", "Mean pair residual"),
        ("secondary_mag", "Secondary magnitude $r_s$", "Mean pair residual"),
        ("separation", "Pair separation (arcsec)", "Mean pair residual"),
        ("pair_prediction", r"Predicted pair $R_{\rm blend}$", "Mean pair residual"),
        ("scene_prediction", r"Predicted scene $R_{\rm blend}$", "Mean cumulative scene residual"),
    ]
    handles = []
    for panel_index, (name, xlabel, ylabel) in enumerate(panels):
        axis = axes.ravel()[panel_index]
        for model in MODEL_ORDER:
            rows = payload["curves"][name][model]
            x = np.asarray([row["coordinate"]["mean"] for row in rows])
            y = np.asarray([row["residual"]["mean"] for row in rows])
            sem = np.asarray([row["residual"]["case_sem"] for row in rows])
            handle = axis.errorbar(
                x,
                y,
                yerr=sem,
                color=COLORS[model],
                marker=MARKERS[model],
                linestyle=LINESTYLES[model],
                markersize=3.4,
                linewidth=1.15,
                capsize=1.5,
                alpha=0.96,
                label=model,
            )
            if panel_index == 0:
                handles.append(handle)
        axis.axhline(0.0, color="0.45", linestyle=":", linewidth=0.85)
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.spines[["top", "right"]].set_visible(False)
        axis.text(
            -0.14,
            1.04,
            ascii_uppercase[panel_index],
            transform=axis.transAxes,
            fontsize=11,
            fontweight="bold",
            va="top",
        )
        if name == "pair_prediction":
            axis.set_xscale("symlog", linthresh=1.0e-3, linscale=1.0)
        elif name == "scene_prediction":
            axis.set_xscale("symlog", linthresh=1.0e-2, linscale=1.0)

    legend_axis = axes.ravel()[5]
    legend_axis.axis("off")
    legend_axis.legend(
        handles,
        MODEL_ORDER,
        title="Pair-response emulator",
        frameon=False,
        loc="center left",
        handlelength=2.8,
    )
    legend_axis.text(
        0.0,
        0.30,
        "A–D: pair label − pair prediction\n"
        "E: sum(pair labels) − sum(pair predictions)\n"
        "Error bars: one SEM across 200 rendered cases",
        transform=legend_axis.transAxes,
        fontsize=8.2,
        linespacing=1.5,
        va="top",
    )
    fig.suptitle(
        "Half-shear validation: reproduced case-level out-of-fold residuals",
        fontsize=11.0,
        y=0.985,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96), w_pad=2.0, h_pad=2.0)
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-cache", required=True)
    parser.add_argument("--pair-oof", required=True)
    parser.add_argument("--pair-scene-oof", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--n-bins", type=int, default=20)
    args = parser.parse_args()

    output_prefix = Path(args.output_prefix).resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    outputs = [
        output_prefix.with_suffix(suffix)
        for suffix in (".json", ".csv", ".pdf", ".png")
    ]
    if existing := [str(path) for path in outputs if path.exists()]:
        raise FileExistsError(f"refusing existing outputs: {existing}")
    source_cache = Path(args.source_cache).resolve()
    metadata = load_source_metadata(source_cache)
    case = mmap_array(source_cache, metadata, "case")
    primary = mmap_array(source_cache, metadata, "input_index")
    label = mmap_array(source_cache, metadata, "label")
    raw = mmap_array(source_cache, metadata, "x_raw")
    predictions = {
        "V2.2": mmap_array(source_cache, metadata, "v22_prediction"),
        "Tuned pair": np.load(args.pair_oof, mmap_mode="r"),
        "Tuned pair+scene": np.load(args.pair_scene_oof, mmap_mode="r"),
    }
    n_rows = len(case)
    if n_rows != 47_310_214 or any(
        prediction.shape != (n_rows,) for prediction in predictions.values()
    ):
        raise RuntimeError("OOF/source row coverage differs")
    if not np.array_equal(np.unique(case), np.arange(200)):
        raise RuntimeError("expected complete cases 0--199")

    curves: dict[str, dict[str, list[dict]]] = {}
    # Physical coordinates use shared, label-free bins and pair residuals.
    for name in ("primary_mag", "secondary_mag", "separation"):
        coordinate = np.asarray(raw[:, RAW_COLUMNS[name]], dtype=np.float32)
        if name == "separation":
            edges = np.linspace(0.0, 10.0, args.n_bins + 1)
        else:
            edges = quantile_edges(coordinate, args.n_bins)
        curves[name] = {}
        for model, prediction in predictions.items():
            residual = np.asarray(label, dtype=np.float32) - np.asarray(
                prediction, dtype=np.float32
            )
            curves[name][model] = summarize_bins(
                case, coordinate, residual, edges, item_name="pairs"
            )
            del residual
        print(f"summarized {name}", flush=True)

    # Each emulator is conditioned on its own pair prediction, in shared
    # label-free V2.2-prediction quantile edges.
    pair_edges = quantile_edges(
        np.asarray(predictions["V2.2"], dtype=np.float32), args.n_bins
    )
    curves["pair_prediction"] = {}
    for model, prediction in predictions.items():
        residual = np.asarray(label, dtype=np.float32) - np.asarray(
            prediction, dtype=np.float32
        )
        curves["pair_prediction"][model] = summarize_bins(
            case,
            prediction,
            residual,
            pair_edges,
            item_name="pairs",
        )
        del residual
    print("summarized pair prediction", flush=True)

    # Scene panel uses cumulative scene labels and cumulative prediction.
    case_array = np.asarray(case)
    primary_array = np.asarray(primary)
    change = np.empty(n_rows, dtype=bool)
    change[0] = True
    change[1:] = (
        (case_array[1:] != case_array[:-1])
        | (primary_array[1:] != primary_array[:-1])
    )
    starts = np.flatnonzero(change)
    scene_case = case_array[starts]
    scene_truth = np.add.reduceat(
        np.asarray(label, dtype=np.float64), starts
    )
    scene_predictions = {
        model: np.add.reduceat(np.asarray(prediction, dtype=np.float64), starts)
        for model, prediction in predictions.items()
    }
    scene_edges = quantile_edges(
        scene_predictions["V2.2"], args.n_bins
    )
    curves["scene_prediction"] = {}
    for model, prediction in scene_predictions.items():
        curves["scene_prediction"][model] = summarize_bins(
            scene_case,
            prediction,
            scene_truth - prediction,
            scene_edges,
            item_name="scenes",
        )
    print("summarized scene prediction", flush=True)

    payload = {
        "schema_version": 1,
        "dataset": "half_shear_cases_0_199",
        "prediction_protocol": "reproduced case-level OOF for tuned models",
        "n_pairs": int(n_rows),
        "n_scenes": int(len(starts)),
        "n_cases": 200,
        "n_bins": args.n_bins,
        "uncertainty": "one SEM across rendered cases",
        "pair_response_panels": [
            "primary_mag", "secondary_mag", "separation", "pair_prediction"
        ],
        "cumulative_scene_response_panel": "scene_prediction",
        "pair_prediction_edges_source": "frozen V2.2 prediction quantiles",
        "scene_prediction_edges_source": "frozen V2.2 scene-sum quantiles",
        "curves": curves,
        "provenance": {
            "source_metadata": str(source_cache / "metadata.json"),
            "source_metadata_sha256": sha256(source_cache / "metadata.json"),
            "pair_oof": str(Path(args.pair_oof).resolve()),
            "pair_scene_oof": str(Path(args.pair_scene_oof).resolve()),
            "coherent_anchor_truth_opened": False,
            "constgold_opened": False,
        },
    }
    rows = []
    for axis, models in curves.items():
        for model, bins in models.items():
            for item in bins:
                rows.append({
                    "axis": axis,
                    "model": model,
                    "bin": item["bin"],
                    "lower": item["lower"],
                    "upper": item["upper"],
                    "coordinate": item["coordinate"]["mean"],
                    "coordinate_case_sem": item["coordinate"]["case_sem"],
                    "residual": item["residual"]["mean"],
                    "residual_case_sem": item["residual"]["case_sem"],
                    "n_items": item.get("n_pairs", item.get("n_scenes")),
                })
    pd.DataFrame(rows).to_csv(output_prefix.with_suffix(".csv"), index=False)
    strict_json(output_prefix.with_suffix(".json"), payload)
    make_plot(payload, output_prefix)
    print("CROSSFIT_OOF_RESIDUAL_AXES_PLOT_DONE", flush=True)


if __name__ == "__main__":
    main()
