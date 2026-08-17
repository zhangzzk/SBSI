#!/usr/bin/env python3
"""Remake coherent-anchor Panel E for the case-disjoint 100/100 stack."""

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
STACK = "case100_100"
RAW_COLUMN = f"R_blend_{STACK}_base"
CORRECTED_COLUMN = f"R_blend_{STACK}"


def finite_stat(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size < 2:
        raise RuntimeError("need at least two finite case values")
    sd = float(values.std(ddof=1))
    return {
        "mean": float(values.mean()),
        "case_sd": sd,
        "case_sem": float(sd / np.sqrt(values.size)),
        "n_cases": int(values.size),
    }


def case_stat(frame: pd.DataFrame, column: str) -> dict[str, float | int]:
    values = frame.groupby("case", sort=True)[column].mean().to_numpy(float)
    return finite_stat(values)


def quantile_edges(values: np.ndarray, n_bins: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    inner = np.unique(np.quantile(values, np.arange(1, n_bins) / n_bins))
    if len(inner) != n_bins - 1:
        raise RuntimeError("raw-scene prediction has tied quantile edges")
    return np.r_[-np.inf, inner, np.inf]


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def load_scores(directory: Path, case_min: int, case_max: int) -> pd.DataFrame:
    columns = [
        "case",
        "input_index",
        "R_blend_v22_replay",
        RAW_COLUMN,
        CORRECTED_COLUMN,
    ]
    shards = []
    for case in range(case_min, case_max + 1):
        feather = directory / f"case{case}.feather"
        sidecar = directory / f"case{case}.json"
        if not feather.is_file() or not sidecar.is_file():
            raise FileNotFoundError(f"missing score shard for case {case}")
        audit = json.loads(sidecar.read_text(encoding="utf-8"))
        if audit["coherent_response_truth_opened"] or audit["constgold_opened"]:
            raise RuntimeError(f"truth firewall failed in case {case}")
        if STACK not in audit["summaries"]:
            raise RuntimeError(f"wrong stack in case {case}")
        shards.append(pd.read_feather(feather, columns=columns))
    return pd.concat(shards, ignore_index=True)


def conditional_curves(
    frame: pd.DataFrame,
    development_prediction: np.ndarray,
    n_bins: int = 12,
) -> tuple[pd.DataFrame, np.ndarray]:
    edges = quantile_edges(development_prediction, n_bins)
    residual_columns = [
        "residual_v22",
        "residual_raw",
        "residual_corrected",
    ]
    finite = np.isfinite(
        frame[[RAW_COLUMN, *residual_columns]].to_numpy(float)
    ).all(axis=1)
    local = frame.loc[
        finite, ["case", RAW_COLUMN, *residual_columns]
    ].copy()
    local["bin"] = np.digitize(
        local[RAW_COLUMN].to_numpy(float), edges[1:-1]
    )
    model_columns = (
        ("V2.2", "residual_v22"),
        ("Raw 100/100 base", "residual_raw"),
        ("100/100 base + correction", "residual_corrected"),
    )
    rows = []
    for index in range(n_bins):
        cell = local.loc[local.bin == index]
        if cell.empty or cell.case.nunique() < 2:
            raise RuntimeError(f"empty or one-case bin {index}")
        by_case = cell.groupby("case", sort=True)[residual_columns].mean()
        common = {
            "bin": int(index),
            "lower": float(edges[index]),
            "upper": float(edges[index + 1]),
            "raw_prediction_median": float(cell[RAW_COLUMN].median()),
            "raw_prediction_mean": float(cell[RAW_COLUMN].mean()),
            "n_anchors": int(len(cell)),
            "n_cases": int(len(by_case)),
        }
        for model, column in model_columns:
            stats = finite_stat(by_case[column].to_numpy(float))
            rows.append({
                **common,
                "model": model,
                "residual_mean": stats["mean"],
                "residual_case_sem": stats["case_sem"],
                "residual_case_sd": stats["case_sd"],
            })
    return pd.DataFrame(rows), edges


def configure_style() -> None:
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


def plot_panel(
    curves: pd.DataFrame,
    stem: Path,
    n_anchors: int,
    n_cases: int,
) -> None:
    configure_style()
    styles = {
        "V2.2": ("#0072B2", "o", "-", "V2.2"),
        "Raw 100/100 base": (
            "#009E73", "^", "-.", "Raw 100/100 base",
        ),
        "100/100 base + correction": (
            "#D55E00", "s", "--", "100/100 base + correction",
        ),
    }
    fig, axis = plt.subplots(figsize=(5.2, 3.8), constrained_layout=True)
    bounds = []
    for row in curves.itertuples():
        bounds.extend([
            abs(float(row.residual_mean) - float(row.residual_case_sem)),
            abs(float(row.residual_mean) + float(row.residual_case_sem)),
        ])
    bound = max(0.01, 1.08 * max(bounds))
    for model in styles:
        line = curves.loc[curves.model == model].sort_values("bin")
        color, marker, linestyle, label = styles[model]
        axis.errorbar(
            line.raw_prediction_median,
            line.residual_mean,
            yerr=line.residual_case_sem,
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=1.15,
            markersize=4.0,
            capsize=2.2,
            label=label,
        )
    axis.axhline(0.0, color="0.60", linestyle=":", linewidth=0.7)
    axis.set_ylim(-bound, bound)
    axis.set_xlabel("Raw 100/100-base scene response")
    axis.set_ylabel("truth - prediction")
    axis.set_title(
        "Panel E: case-disjoint 100/100 model on held-out coherent anchors\n"
        f"cases 700–899; n={n_anchors:,}; error bars: one case SEM"
    )
    axis.legend(frameon=False, loc="best")
    axis.spines[["top", "right"]].set_visible(False)
    for suffix, options in ((".png", {"dpi": 300}), (".pdf", {})):
        output = stem.with_suffix(suffix)
        if output.exists():
            raise FileExistsError(output)
        fig.savefig(output, bbox_inches="tight", **options)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor-features", required=True)
    parser.add_argument("--score-dir", required=True)
    parser.add_argument("--original-json", required=True)
    parser.add_argument("--output-stem", required=True)
    args = parser.parse_args()

    metadata_path = Path(args.original_json).resolve()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    development_min = int(metadata["case_windows"]["train"][0])
    development_max = int(metadata["case_windows"]["tune"][1])
    test_min, test_max = map(int, metadata["case_windows"]["test"])
    expected_development = int(
        metadata["rows"]["train"] + metadata["rows"]["tune"]
    )
    expected_test = int(metadata["rows"]["test"])

    anchor_columns = [
        "case",
        "input_index",
        "R_blend_truth",
        "scene_prediction",
    ]
    all_anchors = pd.read_feather(
        args.anchor_features, columns=anchor_columns
    )
    development = all_anchors.loc[
        all_anchors.case.between(development_min, development_max),
        ["case", "input_index", "scene_prediction"],
    ].copy()
    test = all_anchors.loc[
        all_anchors.case.between(test_min, test_max),
        anchor_columns,
    ].copy()
    del all_anchors
    scores = load_scores(
        Path(args.score_dir).resolve(), development_min, test_max
    )
    development = development.merge(
        scores.loc[
            scores.case.between(development_min, development_max)
        ],
        on=KEY,
        how="left",
        validate="one_to_one",
    )
    frame = test.merge(
        scores.loc[scores.case.between(test_min, test_max)],
        on=KEY,
        how="left",
        validate="one_to_one",
    )
    if len(development) != expected_development or len(frame) != expected_test:
        raise RuntimeError("anchor population size differs from original panel")
    if development.duplicated(KEY).any() or frame.duplicated(KEY).any():
        raise RuntimeError("duplicate anchor key")
    required_development = [
        "scene_prediction",
        "R_blend_v22_replay",
        RAW_COLUMN,
        CORRECTED_COLUMN,
    ]
    required_test = [
        "R_blend_truth",
        *required_development,
    ]
    if not np.isfinite(
        development[required_development].to_numpy(float)
    ).all() or not np.isfinite(frame[required_test].to_numpy(float)).all():
        raise RuntimeError("non-finite or missing anchor score")

    replay_max = max(
        float(np.max(np.abs(
            development.scene_prediction.to_numpy(float)
            - development.R_blend_v22_replay.to_numpy(float)
        ))),
        float(np.max(np.abs(
            frame.scene_prediction.to_numpy(float)
            - frame.R_blend_v22_replay.to_numpy(float)
        ))),
    )
    if replay_max > 2.0e-6:
        raise RuntimeError(f"V2.2 replay mismatch: {replay_max:.3e}")

    truth = frame.R_blend_truth.to_numpy(float)
    frame["residual_v22"] = truth - frame.scene_prediction.to_numpy(float)
    frame["residual_raw"] = truth - frame[RAW_COLUMN].to_numpy(float)
    frame["residual_corrected"] = (
        truth - frame[CORRECTED_COLUMN].to_numpy(float)
    )
    curves, edges = conditional_curves(
        frame,
        development[RAW_COLUMN].to_numpy(float),
    )

    stem = Path(args.output_stem).resolve()
    stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".csv", ".json", ".png", ".pdf"):
        if stem.with_suffix(suffix).exists():
            raise FileExistsError(stem.with_suffix(suffix))
    atomic_csv(stem.with_suffix(".csv"), curves)
    plot_panel(curves, stem, len(frame), frame.case.nunique())

    rightmost = curves.loc[curves.bin == curves.bin.max()].copy()
    payload = {
        "schema_version": 1,
        "dataset": f"coherent_anchor_c{test_min}_{test_max}",
        "n_anchors": int(len(frame)),
        "n_cases": int(frame.case.nunique()),
        "gap_definition": "R_blend_truth - R_blend_prediction",
        "uncertainty": "one SEM across per-case conditional means",
        "coordinate": "raw case-disjoint 100/100 base full-neighbour scene sum",
        "binning": {
            "n_bins": int(len(edges) - 1),
            "development_cases": [development_min, development_max],
            "internal_edges": [float(value) for value in edges[1:-1]],
            "outer_bounds": "negative and positive infinity",
            "labels_used_to_define_bins": False,
        },
        "global_case_balanced_residual": {
            "V2.2": case_stat(frame, "residual_v22"),
            "Raw 100/100 base": case_stat(frame, "residual_raw"),
            "100/100 base + correction": case_stat(
                frame, "residual_corrected"
            ),
        },
        "rightmost_bin": {
            "raw_prediction_median": float(
                rightmost.raw_prediction_median.iloc[0]
            ),
            "n_anchors": int(rightmost.n_anchors.iloc[0]),
            "models": {
                row.model: {
                    "residual_mean": float(row.residual_mean),
                    "residual_case_sem": float(row.residual_case_sem),
                }
                for row in rightmost.itertuples()
            },
        },
        "audit": {
            "v22_replay_max_abs": replay_max,
            "score_stage_opened_anchor_truth": False,
            "coherent_anchor_truth_used_for_training_or_selection": False,
        },
        "provenance": {
            "anchor_features": str(Path(args.anchor_features).resolve()),
            "score_dir": str(Path(args.score_dir).resolve()),
            "original_metadata": str(metadata_path),
        },
        "outputs": {
            "curves_csv": str(stem.with_suffix(".csv")),
            "figure_png": str(stem.with_suffix(".png")),
            "figure_pdf": str(stem.with_suffix(".pdf")),
        },
    }
    atomic_json(stem.with_suffix(".json"), payload)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    print("PLOT_ANCHOR_CASE100_100_PANEL_E_DONE", flush=True)


if __name__ == "__main__":
    main()
