#!/usr/bin/env python3
"""Compare V2.2, the raw new base, and its correction on coherent anchors.

The panel coordinates, their order, and all bin edges come from the original
anchor bias-emulator diagnostic.  Only coherent-anchor cases 700--899 are
evaluated.  No model is fit or selected here, and the bias-emulator prediction
is deliberately not plotted.  A second figure reproduces Panel E using the
raw new-base scene prediction, rather than V2.2, as the shared coordinate.
"""

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


KEY = ["case", "input_index"]


def finite_stat(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        raise ValueError("need at least two finite case values")
    sd = float(values.std(ddof=1))
    return {
        "mean": float(values.mean()),
        "case_sd": sd,
        "case_sem": float(sd / np.sqrt(values.size)),
        "n_cases": int(values.size),
    }


def case_stat(frame: pd.DataFrame, column: str) -> dict:
    means = frame.groupby("case", sort=True)[column].mean().to_numpy(float)
    return finite_stat(means)


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
        "case", "input_index", "R_blend_v22_replay",
        "R_blend_orig_full_base", "R_blend_orig_full",
    ]
    shards = []
    for case in range(case_min, case_max + 1):
        path = directory / f"case{case}.feather"
        if not path.is_file():
            raise FileNotFoundError(path)
        shards.append(pd.read_feather(path, columns=columns))
    return pd.concat(shards, ignore_index=True)


def quantile_edges(values: np.ndarray, n_bins: int) -> np.ndarray:
    """Replay the original diagnostic's tie-aware quantile binning exactly."""
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        raise RuntimeError("feature has fewer than two finite values")
    inner = np.unique(np.quantile(values, np.arange(1, n_bins) / n_bins))
    if inner.size < min(3, n_bins - 1):
        unique = np.unique(values)
        if unique.size < 2:
            raise RuntimeError("feature is constant")
        if unique.size <= n_bins:
            inner = (unique[:-1] + unique[1:]) / 2.0
        else:
            inner = np.unique(np.quantile(
                unique, np.arange(1, n_bins) / n_bins
            ))
    return np.r_[-np.inf, inner, np.inf]


def fixed_bin_curve(
    frame: pd.DataFrame, feature: str, saved: pd.DataFrame,
    development_values: np.ndarray,
) -> list[dict]:
    saved = saved.sort_values("bin").reset_index(drop=True)
    expected_bins = np.arange(len(saved), dtype=int)
    if not np.array_equal(saved.bin.to_numpy(int), expected_bins):
        raise RuntimeError(f"non-contiguous saved bins for {feature}")
    edges = quantile_edges(development_values, 12)
    if len(edges) - 1 != len(saved):
        raise RuntimeError(f"recomputed bin count differs for {feature}")
    if not np.isneginf(edges[0]) or not np.isposinf(edges[-1]):
        raise RuntimeError(f"saved edges do not span the real line for {feature}")

    finite = np.isfinite(frame[feature].to_numpy(float))
    residual_columns = [
        "residual_v22", "residual_newbase_raw", "residual_newbase_corrected",
    ]
    local = frame.loc[finite, ["case", feature, *residual_columns]].copy()
    local["bin"] = np.digitize(
        local[feature].to_numpy(float), edges[1:-1]
    )
    rows = []
    for index in expected_bins:
        cell = local.loc[local.bin == index]
        if cell.empty or cell.case.nunique() < 2:
            raise RuntimeError(f"unexpected empty/one-case bin {feature}:{index}")
        by_case = cell.groupby("case", sort=True)[residual_columns].mean()
        old = saved.loc[saved.bin == index].iloc[0]
        x_median = float(cell[feature].median())
        x_mean = float(cell[feature].mean())
        v22 = finite_stat(by_case.residual_v22.to_numpy(float))
        newbase_raw = finite_stat(by_case.residual_newbase_raw.to_numpy(float))
        newbase_corrected = finite_stat(
            by_case.residual_newbase_corrected.to_numpy(float)
        )

        # This is an exact replay audit of the old held-out black curve.  It
        # protects the comparison against an accidental population/bin change.
        if abs(x_median - float(old.feature_median)) > 1.0e-10:
            raise RuntimeError(f"feature median replay mismatch {feature}:{index}")
        if abs(v22["mean"] - float(old.target_mean)) > 2.0e-10:
            raise RuntimeError(f"V2.2 mean replay mismatch {feature}:{index}")
        if abs(v22["case_sem"] - float(old.target_case_sem)) > 2.0e-10:
            raise RuntimeError(f"V2.2 SEM replay mismatch {feature}:{index}")

        common = {
            "feature": feature,
            "bin": int(index),
            "lower": float(edges[index]),
            "upper": float(edges[index + 1]),
            "feature_median": x_median,
            "feature_mean": x_mean,
            "n_rows": int(len(cell)),
            "n_cases": int(len(by_case)),
        }
        for name, stats in (
            ("V2.2", v22),
            ("new_base_raw", newbase_raw),
            ("new_base_corrected", newbase_corrected),
        ):
            rows.append({
                **common,
                "model": name,
                "residual_mean": stats["mean"],
                "residual_case_sem": stats["case_sem"],
                "residual_case_sd": stats["case_sd"],
            })
    return rows


def newbase_scene_curve(
    frame: pd.DataFrame, development_prediction: np.ndarray, n_bins: int = 12,
) -> pd.DataFrame:
    """Condition all residuals on the raw new-base scene prediction."""
    feature = "R_blend_orig_full_base"
    residual_columns = [
        "residual_v22", "residual_newbase_raw", "residual_newbase_corrected",
    ]
    edges = quantile_edges(development_prediction, n_bins)
    finite = np.isfinite(
        frame[[feature, *residual_columns]].to_numpy(float)
    ).all(axis=1)
    local = frame.loc[finite, ["case", feature, *residual_columns]].copy()
    local["bin"] = np.digitize(local[feature].to_numpy(float), edges[1:-1])
    rows = []
    models = (
        ("V2.2", "residual_v22"),
        ("new_base_raw", "residual_newbase_raw"),
        ("new_base_corrected", "residual_newbase_corrected"),
    )
    for index in range(len(edges) - 1):
        cell = local.loc[local.bin == index]
        if cell.empty or cell.case.nunique() < 2:
            raise RuntimeError(f"unexpected empty/one-case new-base bin {index}")
        by_case = cell.groupby("case", sort=True)[residual_columns].mean()
        common = {
            "feature": feature,
            "bin": int(index),
            "lower": float(edges[index]),
            "upper": float(edges[index + 1]),
            "feature_median": float(cell[feature].median()),
            "feature_mean": float(cell[feature].mean()),
            "n_rows": int(len(cell)),
            "n_cases": int(len(by_case)),
        }
        for model, column in models:
            stats = finite_stat(by_case[column].to_numpy(float))
            rows.append({
                **common,
                "model": model,
                "residual_mean": stats["mean"],
                "residual_case_sem": stats["case_sem"],
                "residual_case_sd": stats["case_sd"],
            })
    return pd.DataFrame(rows)


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


def save_figure(fig: plt.Figure, stem: Path) -> None:
    for suffix, options in (
        (".png", {"dpi": 300}),
        (".pdf", {}),
    ):
        output = stem.with_suffix(suffix)
        if output.exists():
            raise FileExistsError(f"refusing existing output {output}")
        fig.savefig(output, bbox_inches="tight", **options)
    plt.close(fig)


def plot_curves(
    curves: pd.DataFrame, features: list[str], labels: dict[str, str], stem: Path,
) -> None:
    configure_style()
    fig, axes = plt.subplots(
        4, 4, figsize=(11.0, 9.8), constrained_layout=True, squeeze=False,
    )
    bound_values = []
    for row in curves.itertuples():
        bound_values.extend([
            abs(float(row.residual_mean)),
            abs(float(row.residual_mean) + float(row.residual_case_sem)),
            abs(float(row.residual_mean) - float(row.residual_case_sem)),
        ])
    bound = max(float(np.nanpercentile(bound_values, 98) * 1.15), 0.01)

    styles = {
        "V2.2": {
            "color": "#0072B2", "marker": "o", "linestyle": "-",
            "label": "V2.2",
        },
        "new_base_raw": {
            "color": "#009E73", "marker": "^", "linestyle": "-.",
            "label": "Raw new base",
        },
        "new_base_corrected": {
            "color": "#D55E00", "marker": "s", "linestyle": "--",
            "label": "New base + full-neighbour correction",
        },
    }
    for panel, (axis, feature) in enumerate(zip(axes.flat, features)):
        local = curves.loc[curves.feature == feature]
        for model in ("V2.2", "new_base_raw", "new_base_corrected"):
            line = local.loc[local.model == model].sort_values("bin")
            style = styles[model]
            axis.errorbar(
                line.feature_median,
                line.residual_mean,
                yerr=line.residual_case_sem,
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=1.0,
                markersize=3.0,
                capsize=1.8,
                label=style["label"],
            )
        axis.axhline(0, color="0.60", linestyle=":", linewidth=0.6)
        axis.set_ylim(-bound, bound)
        axis.set_xlabel(labels.get(feature, feature))
        axis.set_ylabel("truth - prediction")
        axis.spines[["top", "right"]].set_visible(False)
        axis.text(
            -0.14, 1.05, ascii_uppercase[panel], transform=axis.transAxes,
            fontweight="bold", fontsize=9, va="top",
        )
        if panel == 0:
            axis.legend(frameon=False, loc="best")
    fig.suptitle(
        "Held-out coherent-anchor conditional residuals (cases 700-899)\n"
        "Original development-fixed coordinates and bins",
        fontsize=10,
    )
    save_figure(fig, stem)


def plot_newbase_scene_panel(curves: pd.DataFrame, stem: Path) -> None:
    configure_style()
    fig, axis = plt.subplots(figsize=(5.2, 3.8), constrained_layout=True)
    styles = {
        "V2.2": ("#0072B2", "o", "-", "V2.2"),
        "new_base_raw": ("#009E73", "^", "-.", "Raw new base"),
        "new_base_corrected": (
            "#D55E00", "s", "--", "New base + full-neighbour correction",
        ),
    }
    bound_values = []
    for row in curves.itertuples():
        bound_values.extend([
            abs(float(row.residual_mean) + float(row.residual_case_sem)),
            abs(float(row.residual_mean) - float(row.residual_case_sem)),
        ])
    bound = max(0.01, 1.08 * max(bound_values))
    for model in ("V2.2", "new_base_raw", "new_base_corrected"):
        line = curves.loc[curves.model == model].sort_values("bin")
        color, marker, linestyle, label = styles[model]
        axis.errorbar(
            line.feature_median, line.residual_mean,
            yerr=line.residual_case_sem, color=color, marker=marker,
            linestyle=linestyle, linewidth=1.15, markersize=4.0,
            capsize=2.2, label=label,
        )
    axis.axhline(0, color="0.60", linestyle=":", linewidth=0.7)
    axis.set_ylim(-bound, bound)
    axis.set_xlabel("Raw new-base scene response")
    axis.set_ylabel("truth - prediction")
    axis.set_title(
        "Panel E re-binned on raw new-base scene response\n"
        "Held-out coherent anchors, cases 700-899"
    )
    axis.legend(frameon=False, loc="best")
    axis.spines[["top", "right"]].set_visible(False)
    save_figure(fig, stem)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor-features", required=True)
    parser.add_argument("--score-dir", required=True)
    parser.add_argument("--original-json", required=True)
    parser.add_argument("--original-curves", required=True)
    parser.add_argument("--output-stem", required=True)
    parser.add_argument("--panel-e-output-stem", required=True)
    args = parser.parse_args()

    metadata_path = Path(args.original_json).resolve()
    original_curves_path = Path(args.original_curves).resolve()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    features = list(metadata["selected_curve_features"])
    labels = dict(metadata["feature_labels"])
    case_min, case_max = map(int, metadata["case_windows"]["test"])
    expected_rows = int(metadata["rows"]["test"])

    columns = list(dict.fromkeys([
        "case", "input_index", "R_blend_truth", "scene_prediction", *features,
    ]))
    anchors_all = pd.read_feather(args.anchor_features, columns=columns)
    development = anchors_all.loc[
        anchors_all.case.between(
            int(metadata["case_windows"]["train"][0]),
            int(metadata["case_windows"]["tune"][1]),
        ),
        ["case", "input_index", *features],
    ].copy()
    anchors = anchors_all.loc[
        anchors_all.case.between(case_min, case_max), columns
    ].copy()
    del anchors_all
    development_min = int(metadata["case_windows"]["train"][0])
    development_max = int(metadata["case_windows"]["tune"][1])
    scores = load_scores(Path(args.score_dir), development_min, case_max)
    development_scores = scores.loc[
        scores.case.between(development_min, development_max)
    ].copy()
    test_scores = scores.loc[scores.case.between(case_min, case_max)].copy()
    del scores
    development = development.merge(
        development_scores, on=KEY, how="left", validate="one_to_one"
    )
    frame = anchors.merge(
        test_scores, on=KEY, how="left", validate="one_to_one"
    )
    required_finite = [
        "R_blend_truth", "scene_prediction", "R_blend_v22_replay",
        "R_blend_orig_full_base", "R_blend_orig_full",
    ]
    if (
        len(frame) != expected_rows
        or not np.isfinite(frame[required_finite].to_numpy(float)).all()
    ):
        raise RuntimeError(
            f"anchor/score coverage mismatch: rows={len(frame)} expected={expected_rows}"
        )
    if frame.duplicated(KEY).any():
        raise RuntimeError("duplicate coherent-anchor key")

    development_expected = int(metadata["rows"]["train"] + metadata["rows"]["tune"])
    development_required = [
        "scene_prediction", "R_blend_v22_replay",
        "R_blend_orig_full_base", "R_blend_orig_full",
    ]
    if (
        len(development) != development_expected
        or not np.isfinite(development[development_required].to_numpy(float)).all()
    ):
        raise RuntimeError("development feature/score coverage mismatch")
    if development.duplicated(KEY).any():
        raise RuntimeError("duplicate development anchor key")

    replay_max = max(
        float(np.max(np.abs(
            frame.scene_prediction.to_numpy(float)
            - frame.R_blend_v22_replay.to_numpy(float)
        ))),
        float(np.max(np.abs(
            development.scene_prediction.to_numpy(float)
            - development.R_blend_v22_replay.to_numpy(float)
        ))),
    )
    if replay_max > 2.0e-6:
        raise RuntimeError(f"V2.2 score replay mismatch: {replay_max:.3e}")
    frame["residual_v22"] = (
        frame.R_blend_truth.to_numpy(float)
        - frame.scene_prediction.to_numpy(float)
    )
    frame["residual_newbase_raw"] = (
        frame.R_blend_truth.to_numpy(float)
        - frame.R_blend_orig_full_base.to_numpy(float)
    )
    frame["residual_newbase_corrected"] = (
        frame.R_blend_truth.to_numpy(float)
        - frame.R_blend_orig_full.to_numpy(float)
    )

    old = pd.read_csv(original_curves_path)
    old = old.loc[old.population == "all"]
    curve_rows = []
    for feature in features:
        saved = old.loc[old.feature == feature]
        if saved.empty:
            raise RuntimeError(f"missing original curve bins for {feature}")
        curve_rows.extend(fixed_bin_curve(
            frame, feature, saved, development[feature].to_numpy(float)
        ))
    curves = pd.DataFrame(curve_rows)

    panel_e_curves = newbase_scene_curve(
        frame, development.R_blend_orig_full_base.to_numpy(float)
    )
    stem = Path(args.output_stem).resolve()
    panel_e_stem = Path(args.panel_e_output_stem).resolve()
    stem.parent.mkdir(parents=True, exist_ok=True)
    panel_e_stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".csv", ".json", ".png", ".pdf"):
        if stem.with_suffix(suffix).exists():
            raise FileExistsError(f"refusing existing output {stem.with_suffix(suffix)}")
    for suffix in (".csv", ".png", ".pdf"):
        if panel_e_stem.with_suffix(suffix).exists():
            raise FileExistsError(
                f"refusing existing output {panel_e_stem.with_suffix(suffix)}"
            )
    atomic_csv(stem.with_suffix(".csv"), curves)
    atomic_csv(panel_e_stem.with_suffix(".csv"), panel_e_curves)
    plot_curves(curves, features, labels, stem)
    plot_newbase_scene_panel(panel_e_curves, panel_e_stem)

    payload = {
        "schema_version": 1,
        "dataset": f"coherent_anchor_c{case_min}_{case_max}",
        "n_anchors": int(len(frame)),
        "n_cases": int(frame.case.nunique()),
        "gap_definition": "R_blend_truth - R_blend_prediction",
        "uncertainty": "one SEM across per-case conditional means",
        "panel_features": features,
        "panel_labels": {feature: labels.get(feature, feature) for feature in features},
        "binning": {
            "definition": "unchanged development-case quantile bins from original plot",
            "source_curves": str(original_curves_path),
            "development_cases": metadata["case_windows"]["train"][:1]
            + metadata["case_windows"]["tune"][1:],
            "panel_e_definition": (
                "all residual curves in raw-new-base scene-response bins"
            ),
        },
        "global_case_balanced_residual": {
            "V2.2": case_stat(frame, "residual_v22"),
            "new_base_raw": case_stat(frame, "residual_newbase_raw"),
            "new_base_corrected": case_stat(
                frame, "residual_newbase_corrected"
            ),
        },
        "provenance": {
            "anchor_features": str(Path(args.anchor_features).resolve()),
            "score_dir": str(Path(args.score_dir).resolve()),
            "original_metadata": str(metadata_path),
            "coherent_anchor_truth_used_for_training_or_selection": False,
        },
        "audit": {
            "v22_replay_max_abs": replay_max,
            "old_v22_curve_replayed_exactly": True,
        },
        "outputs": {
            "curves_csv": str(stem.with_suffix(".csv")),
            "figure_png": str(stem.with_suffix(".png")),
            "figure_pdf": str(stem.with_suffix(".pdf")),
            "panel_e_curves_csv": str(panel_e_stem.with_suffix(".csv")),
            "panel_e_figure_png": str(panel_e_stem.with_suffix(".png")),
            "panel_e_figure_pdf": str(panel_e_stem.with_suffix(".pdf")),
        },
    }
    atomic_json(stem.with_suffix(".json"), payload)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    print("PLOT_ANCHOR_NEWBASE_ORIGINAL_CASES_4X4_DONE", flush=True)


if __name__ == "__main__":
    main()
