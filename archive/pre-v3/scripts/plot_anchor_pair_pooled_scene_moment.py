#!/usr/bin/env python3
"""Plot coherent-anchor transfer for pair plus pooled-scene corrections."""

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


KEY = ["case", "input_index"]
MODELS = (
    ("V2.2", "R_blend_v22_replay"),
    ("Raw base", "R_blend_raw_base"),
    ("Ordinary correction", "R_blend_ordinary"),
    ("Pair-only + Q", "R_blend_pair_only_q"),
    ("Pair + pooled scene", "R_blend_pooled_scene"),
)


def stat(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=float)
    sd = float(values.std(ddof=1))
    return {
        "mean": float(values.mean()),
        "case_sd": sd,
        "case_sem": float(sd / np.sqrt(len(values))),
        "n_cases": int(len(values)),
    }


def edges(values: np.ndarray, n_bins: int = 12) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    inner = np.unique(np.quantile(values, np.arange(1, n_bins) / n_bins))
    if len(inner) != n_bins - 1:
        raise RuntimeError("anchor coordinate has tied quantile edges")
    return np.r_[-np.inf, inner, np.inf]


def curve(
    frame: pd.DataFrame, coordinate: str, bin_edges: np.ndarray
) -> list[dict]:
    columns = list(dict.fromkeys([
        "case", coordinate, *[column for _, column in MODELS],
        "R_blend_truth",
    ]))
    local = frame.loc[np.isfinite(frame[coordinate]), columns].copy()
    local["bin"] = np.searchsorted(
        bin_edges[1:-1], local[coordinate].to_numpy(float), side="right"
    )
    rows = []
    for index in range(len(bin_edges) - 1):
        cell = local.loc[local.bin == index]
        for model, column in MODELS:
            residual = cell.R_blend_truth.to_numpy(float) - cell[column].to_numpy(float)
            by_case = pd.DataFrame({"case": cell.case, "residual": residual}).groupby("case").residual.mean().to_numpy(float)
            result = stat(by_case)
            rows.append({
                "coordinate": coordinate,
                "model": model,
                "bin": index,
                "lower": float(bin_edges[index]),
                "upper": float(bin_edges[index + 1]),
                "coordinate_median": float(cell[coordinate].median()),
                "n_anchors": int(len(cell)),
                "residual_mean": result["mean"],
                "residual_case_sem": result["case_sem"],
            })
    return rows


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor-features", required=True)
    parser.add_argument("--score-dir", required=True)
    parser.add_argument("--pooled-summary", required=True)
    parser.add_argument("--output-stem", required=True)
    args = parser.parse_args()
    summary = json.loads(Path(args.pooled_summary).read_text(encoding="utf-8"))
    shards = []
    for case in range(400, 900):
        sidecar = Path(args.score_dir) / f"case{case}.json"
        audit = json.loads(sidecar.read_text(encoding="utf-8"))
        if audit["coherent_response_truth_opened"] or audit["constgold_opened"]:
            raise RuntimeError(f"case {case}: truth firewall failed")
        shards.append(pd.read_feather(Path(args.score_dir) / f"case{case}.feather"))
    scores = pd.concat(shards, ignore_index=True)
    truth = pd.read_feather(
        args.anchor_features,
        columns=["case", "input_index", "R_blend_truth", "scene_prediction"],
    )
    frame = truth.merge(scores, on=KEY, how="left", validate="one_to_one")
    replay = float(np.max(np.abs(frame.scene_prediction - frame.R_blend_v22_replay)))
    if replay > 2.0e-6:
        raise RuntimeError(f"V2.2 replay mismatch {replay}")
    development = frame.case.between(400, 699)
    test = frame.case.between(700, 899)
    test_frame = frame.loc[test].copy()
    coordinates = ("R_blend_raw_base", "log10_scene_proxy_q_d2")
    edge_map = {
        coordinate: edges(frame.loc[development, coordinate].to_numpy(float))
        for coordinate in coordinates
    }
    rows = []
    for coordinate in coordinates:
        rows.extend(curve(test_frame, coordinate, edge_map[coordinate]))
    curves = pd.DataFrame(rows)

    styles = {
        "V2.2": ("#0072B2", "o", "-"),
        "Raw base": ("#009E73", "^", "-."),
        "Ordinary correction": ("#D55E00", "s", "--"),
        "Pair-only + Q": ("#CC79A7", "D", ":"),
        "Pair + pooled scene": ("#000000", "v", "-"),
    }
    plt.rcParams.update({
        "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, axes_plot = plt.subplots(1, 2, figsize=(10.2, 3.9), constrained_layout=True)
    labels = {
        "R_blend_raw_base": "Raw-base full-neighbour scene prediction",
        "log10_scene_proxy_q_d2": r"Physical scene proxy $\log_{10}Q_{d2}$",
    }
    titles = {
        "R_blend_raw_base": "Conditioned on learned scene response",
        "log10_scene_proxy_q_d2": "Conditioned on physical crowding",
    }
    for axis_plot, coordinate in zip(axes_plot, coordinates):
        local = curves.loc[curves.coordinate == coordinate]
        for model, _ in MODELS:
            line = local.loc[local.model == model].sort_values("bin")
            color, marker, linestyle = styles[model]
            axis_plot.errorbar(
                line.coordinate_median,
                line.residual_mean,
                yerr=line.residual_case_sem,
                color=color,
                marker=marker,
                linestyle=linestyle,
                linewidth=1.1,
                markersize=3.6,
                capsize=2.0,
                label=model,
            )
        axis_plot.axhline(0.0, color="0.55", linestyle=":", linewidth=0.8)
        axis_plot.set_xlabel(labels[coordinate])
        axis_plot.set_ylabel("Coherent truth - prediction")
        axis_plot.set_title(titles[coordinate])
        axis_plot.spines[["top", "right"]].set_visible(False)
    axes_plot[0].legend(frameon=False, loc="best")
    for label, axis_plot in zip("AB", axes_plot):
        axis_plot.text(
            -0.14, 1.08, label, transform=axis_plot.transAxes,
            fontsize=11, fontweight="bold", va="top",
        )
    fig.suptitle(
        "Half-shear-selected pooled-scene correction transferred to coherent anchors\n"
        "held-out anchor cases 700–899; errors: one SEM across cases",
        fontsize=10,
    )
    stem = Path(args.output_stem).resolve()
    stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".csv", ".json", ".png", ".pdf"):
        if stem.with_suffix(suffix).exists():
            raise FileExistsError(stem.with_suffix(suffix))
    curves.to_csv(stem.with_suffix(".csv"), index=False)
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    global_stats = {}
    for model, column in MODELS:
        residual = test_frame.R_blend_truth.to_numpy(float) - test_frame[column].to_numpy(float)
        by_case = pd.DataFrame({"case": test_frame.case, "residual": residual}).groupby("case").residual.mean().to_numpy(float)
        global_stats[model] = stat(by_case)
    rightmost = {}
    for coordinate in coordinates:
        local = curves.loc[
            (curves.coordinate == coordinate)
            & (curves.bin == curves.loc[curves.coordinate == coordinate, "bin"].max())
        ]
        rightmost[coordinate] = {
            row.model: {
                "residual_mean": float(row.residual_mean),
                "residual_case_sem": float(row.residual_case_sem),
            }
            for row in local.itertuples()
        }
    payload = {
        "schema_version": 1,
        "dataset": "coherent_anchor_c700_899",
        "n_anchors": int(len(test_frame)),
        "n_cases": int(test_frame.case.nunique()),
        "selected_lambda": float(summary["selection"]["selected_lambda"]),
        "gap_definition": "R_blend_truth - prediction",
        "global_case_balanced_residual": global_stats,
        "rightmost_bins": rightmost,
        "binning": {
            "development_cases": [400, 699],
            "test_cases": [700, 899],
            "labels_used_to_define_bins": False,
            "n_bins": 12,
        },
        "audit": {
            "v22_replay_max_abs": replay,
            "score_stage_opened_anchor_truth": False,
            "coherent_truth_used_for_training_or_selection": False,
            "constgold_opened": False,
        },
    }
    atomic_json(stem.with_suffix(".json"), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    print("PLOT_ANCHOR_PAIR_POOLED_SCENE_DONE", flush=True)


if __name__ == "__main__":
    main()
