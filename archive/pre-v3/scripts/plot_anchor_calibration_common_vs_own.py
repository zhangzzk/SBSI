#!/usr/bin/env python3
"""Compare four coherent-anchor models in common and own-prediction bins."""

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

from scripts.v22_grouped_rscene_common import (
    assign_scene_bins,
    load_scene_metadata,
    scene_edges_from_metadata,
)


MODELS = {
    "V2.2": "scene_prediction",
    "Old residual": "R_blend_grouped",
    "Tuned pair": "R_blend_pair",
    "Tuned pair+scene": "R_blend_pair_scene",
}
COLORS = {
    "V2.2": "#0072B2",
    "Old residual": "#CC79A7",
    "Tuned pair": "#D55E00",
    "Tuned pair+scene": "#009E73",
}
MARKERS = {
    "V2.2": "o",
    "Old residual": "^",
    "Tuned pair": "s",
    "Tuned pair+scene": "v",
}
LINESTYLES = {
    "V2.2": "--",
    "Old residual": ":",
    "Tuned pair": "-",
    "Tuned pair+scene": "-.",
}
EXTRA_MODEL = "Corrected-own bins"
SELECTIONS = {
    "All": lambda value: np.ones(len(value), dtype=bool),
    r"$P_s\leq0.1$": lambda value: value <= 0.1,
    r"$P_s>0.1$": lambda value: value > 0.1,
    r"$P_s>0.2$": lambda value: value > 0.2,
}


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def finite_stat(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        raise RuntimeError("statistic requires at least two rendered cases")
    return {
        "mean": float(values.mean()),
        "case_sem": float(values.std(ddof=1) / np.sqrt(len(values))),
        "n_cases": int(len(values)),
    }


def quantile_edges(values: np.ndarray, n_bins: int) -> np.ndarray:
    edges = np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1))
    if np.any(np.diff(edges) <= 0.0):
        raise RuntimeError("prediction quantile edges are not unique")
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def summarize_bins(
    case_index: np.ndarray,
    coordinate: np.ndarray,
    residual: np.ndarray,
    bin_index: np.ndarray,
    n_bins: int,
) -> list[dict]:
    n_cases = 500
    flat = case_index * n_bins + bin_index
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
        local_coordinate = coordinate_sum[present, index] / local_count
        local_residual = residual_sum[present, index] / local_count
        rows.append({
            "bin": int(index),
            "n_anchors": int(local_count.sum()),
            "coordinate": finite_stat(local_coordinate),
            "residual": finite_stat(local_residual),
            "row_weighted_residual": float(
                residual_sum[present, index].sum() / local_count.sum()
            ),
        })
    return rows


def selection_summary(
    case_index: np.ndarray,
    frozen: np.ndarray,
    truth: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, dict]:
    residual = truth - prediction
    output = {}
    for name, select in SELECTIONS.items():
        mask = select(frozen)
        counts = np.bincount(case_index[mask], minlength=500)
        sums = np.bincount(
            case_index[mask], weights=residual[mask], minlength=500
        )
        if np.any(counts == 0):
            raise RuntimeError(f"selection {name} lacks a rendered case")
        item = finite_stat(sums / counts)
        item["n_anchors"] = int(mask.sum())
        item["row_weighted_mean"] = float(residual[mask].mean())
        output[name] = item
    return output


def load_scores(directory: Path, columns: list[str]) -> pd.DataFrame:
    return pd.concat(
        [
            pd.read_feather(directory / f"case{case}.feather", columns=columns)
            for case in range(400, 900)
        ],
        ignore_index=True,
    )


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 8.8,
        "axes.labelsize": 9.5,
        "axes.titlesize": 10.0,
        "xtick.labelsize": 7.8,
        "ytick.labelsize": 7.8,
        "legend.fontsize": 8.2,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    })


def curve(axis: plt.Axes, rows: list[dict], model: str):
    x = np.asarray([row["coordinate"]["mean"] for row in rows])
    y = np.asarray([row["residual"]["mean"] for row in rows])
    sem = np.asarray([row["residual"]["case_sem"] for row in rows])
    return axis.errorbar(
        x,
        y,
        yerr=sem,
        color=COLORS[model],
        marker=MARKERS[model],
        linestyle=LINESTYLES[model],
        markersize=3.4,
        linewidth=1.15,
        capsize=1.5,
        alpha=0.97,
        label=model,
    )


def make_plot(payload: dict, output_prefix: Path) -> None:
    configure_style()
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.25))
    handles = []
    for model in MODELS:
        handles.append(curve(axes[0], payload["common_bins"][model], model))
        curve(axes[1], payload["own_prediction_bins"][model], model)
    for axis in axes[:2]:
        axis.axhline(0.0, color="0.45", linestyle="--", linewidth=0.85)
        axis.axvline(0.1, color="0.68", linestyle=":", linewidth=0.85)
        axis.set_xscale("symlog", linthresh=1.0e-3, linscale=1.0)
        axis.set_ylabel(r"Mean residual $R_{\rm anchor}-\widehat R_{\rm blend}$")
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_title("Same anchors in every bin")
    axes[0].set_xlabel(r"Frozen V2.2 scene response $P_s$")
    axes[1].set_title("Each model's self-calibration")
    axes[1].set_xlabel(r"Model's own predicted scene response")

    categories = list(SELECTIONS)
    x = np.arange(len(categories), dtype=float)
    offsets = np.linspace(-0.24, 0.24, len(MODELS))
    for offset, model in zip(offsets, MODELS):
        values = payload["fixed_population_gaps"][model]
        mean = [values[name]["mean"] for name in categories]
        sem = [values[name]["case_sem"] for name in categories]
        axes[2].errorbar(
            x + offset,
            mean,
            yerr=sem,
            color=COLORS[model],
            marker=MARKERS[model],
            linestyle="none",
            markersize=4.2,
            capsize=2.2,
            label=model,
        )
    axes[2].axhline(0.0, color="0.45", linestyle="--", linewidth=0.85)
    axes[2].set_xticks(x, categories)
    axes[2].set_ylabel(r"Mean residual $R_{\rm anchor}-\widehat R_{\rm blend}$")
    axes[2].set_title("Fixed V2.2-defined populations")
    axes[2].spines[["top", "right"]].set_visible(False)

    for label, axis in zip("ABC", axes):
        axis.text(
            -0.14,
            1.04,
            label,
            transform=axis.transAxes,
            fontsize=11,
            fontweight="bold",
            va="top",
        )
    fig.legend(
        handles,
        list(MODELS),
        title="Pair-response emulator",
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=4,
        handlelength=2.5,
    )
    fig.suptitle(
        "Coherent-anchor calibration: common populations versus self-calibration",
        fontsize=11.3,
        y=0.98,
    )
    fig.text(
        0.5,
        0.055,
        "Residual is anchor truth minus prediction; error bars are one SEM across 500 rendered cases",
        ha="center",
        fontsize=8.4,
    )
    fig.tight_layout(rect=(0, 0.14, 1, 0.94), w_pad=2.0)
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-score-dir", required=True)
    parser.add_argument("--old-score-dir", required=True)
    parser.add_argument("--anchor-features", required=True)
    parser.add_argument("--scene-cache", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--n-own-bins", type=int, default=20)
    parser.add_argument(
        "--extra-score-dir",
        help=(
            "optional score directory containing R_blend_corrected_own; adds "
            "the corrected-own-bin model to all panels"
        ),
    )
    args = parser.parse_args()

    output_prefix = Path(args.output_prefix).resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    outputs = [
        output_prefix.with_suffix(suffix)
        for suffix in (".json", ".csv", ".pdf", ".png")
    ]
    if existing := [str(path) for path in outputs if path.exists()]:
        raise FileExistsError(f"refusing existing outputs: {existing}")

    truth_frame = pd.read_feather(
        args.anchor_features,
        columns=["case", "input_index", "R_blend_truth", "scene_prediction"],
    )
    new = load_scores(
        Path(args.new_score_dir),
        [
            "case", "input_index", "R_blend_v22_replay",
            "R_blend_pair", "R_blend_pair_scene",
        ],
    )
    old = load_scores(
        Path(args.old_score_dir),
        ["case", "input_index", "R_blend_v22_replay", "R_blend_grouped"],
    ).rename(columns={"R_blend_v22_replay": "R_blend_v22_replay_old"})
    joined = truth_frame.merge(
        new, on=["case", "input_index"], how="left", validate="one_to_one"
    ).merge(
        old, on=["case", "input_index"], how="left", validate="one_to_one"
    )
    replay_columns = ["R_blend_v22_replay", "R_blend_v22_replay_old"]
    if args.extra_score_dir:
        extra = load_scores(
            Path(args.extra_score_dir),
            [
                "case", "input_index", "R_blend_v22_replay",
                "R_blend_corrected_own",
            ],
        ).rename(columns={
            "R_blend_v22_replay": "R_blend_v22_replay_extra"
        })
        joined = joined.merge(
            extra,
            on=["case", "input_index"],
            how="left",
            validate="one_to_one",
        )
        MODELS[EXTRA_MODEL] = "R_blend_corrected_own"
        COLORS[EXTRA_MODEL] = "#E69F00"
        MARKERS[EXTRA_MODEL] = "D"
        LINESTYLES[EXTRA_MODEL] = (0, (3, 1, 1, 1))
        replay_columns.append("R_blend_v22_replay_extra")
    if len(joined) != 1_703_884 or joined.isna().any().any():
        raise RuntimeError("anchor truth/score coverage differs")
    frozen = joined.scene_prediction.to_numpy(np.float64)
    replay = [
        joined[column].to_numpy(np.float64) for column in replay_columns
    ]
    replay_max = float(max(
        *[np.max(np.abs(frozen - value)) for value in replay],
        *[
            np.max(np.abs(replay[0] - value)) for value in replay[1:]
        ],
    ))
    if replay_max > 2.0e-6:
        raise RuntimeError(f"V2.2 replay mismatch {replay_max:.3e}")
    case = joined.case.to_numpy(np.int64)
    if not np.array_equal(np.unique(case), np.arange(400, 900)):
        raise RuntimeError("expected coherent-anchor cases 400--899")
    case_index = case - 400
    truth = joined.R_blend_truth.to_numpy(np.float64)
    prediction = {
        model: joined[column].to_numpy(np.float64)
        for model, column in MODELS.items()
    }

    scene_meta = load_scene_metadata(Path(args.scene_cache))
    common_edges = scene_edges_from_metadata(scene_meta)
    common_index = assign_scene_bins(frozen, common_edges).astype(np.int64)
    n_common_bins = len(common_edges) - 1
    common_bins = {}
    own_bins = {}
    own_edges = {}
    fixed_gaps = {}
    own_gaps = {}
    for model, local_prediction in prediction.items():
        residual = truth - local_prediction
        common_bins[model] = summarize_bins(
            case_index, frozen, residual, common_index, n_common_bins
        )
        edges = quantile_edges(local_prediction, args.n_own_bins)
        index = np.searchsorted(edges[1:-1], local_prediction, side="right")
        own_bins[model] = summarize_bins(
            case_index,
            local_prediction,
            residual,
            index,
            args.n_own_bins,
        )
        own_edges[model] = [
            None if not np.isfinite(value) else float(value) for value in edges
        ]
        fixed_gaps[model] = selection_summary(
            case_index, frozen, truth, local_prediction
        )
        own_gaps[model] = selection_summary(
            case_index, local_prediction, truth, local_prediction
        )

    payload = {
        "schema_version": 1,
        "dataset": "coherent_anchor_c400_899",
        "n_anchors": int(len(joined)),
        "n_cases": 500,
        "models": list(MODELS),
        "residual_definition": "R_blend_truth - R_blend_prediction",
        "common_bins": common_bins,
        "common_bin_coordinate": "frozen V2.2 scene prediction",
        "common_bin_edges": [
            None if not np.isfinite(value) else float(value)
            for value in common_edges
        ],
        "own_prediction_bins": own_bins,
        "own_prediction_edges": own_edges,
        "fixed_population_gaps": fixed_gaps,
        "fixed_population_coordinate": "frozen V2.2 scene prediction",
        "own_population_gaps": own_gaps,
        "own_population_coordinate": "each model's own scene prediction",
        "uncertainty": "one SEM across per-case conditional means",
        "audit": {"v22_replay_max_abs": replay_max},
    }
    records = []
    for family in ("common_bins", "own_prediction_bins"):
        for model, rows in payload[family].items():
            for row in rows:
                records.append({
                    "family": family,
                    "model": model,
                    "bin": row["bin"],
                    "n_anchors": row["n_anchors"],
                    "coordinate": row["coordinate"]["mean"],
                    "residual": row["residual"]["mean"],
                    "residual_case_sem": row["residual"]["case_sem"],
                })
    pd.DataFrame(records).to_csv(output_prefix.with_suffix(".csv"), index=False)
    atomic_json(output_prefix.with_suffix(".json"), payload)
    make_plot(payload, output_prefix)
    print(json.dumps({
        "outputs": [str(path) for path in outputs],
        "fixed_population_gaps": fixed_gaps,
        "own_population_gaps": own_gaps,
        "v22_replay_max_abs": replay_max,
    }, indent=2, sort_keys=True, allow_nan=False))
    print("ANCHOR_COMMON_VS_OWN_CALIBRATION_PLOT_DONE", flush=True)


if __name__ == "__main__":
    main()
