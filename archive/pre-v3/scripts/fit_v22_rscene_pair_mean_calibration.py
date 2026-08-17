#!/usr/bin/env python3
"""Fit a frozen-V2.2 correction on the predicted scene-response axis.

The base emulator is never retrained.  For each primary scene ``s`` we first
freeze

    P_s = sum_j p_v22(s, j).

Scenes are binned by ``P_s`` without using a response label.  Within each bin
the additive per-pair correction is the exact minimizer of squared bin-mean
residual,

    c_b = sum(label - p_v22) / number_of_pairs.

The calibration fit uses only the official 20 per cent V2.2 validation rows in
cases 40--199.  It is then frozen and tested on external half-shear cases
0--39 and on coherent anchors.  ConstGold is never read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd


DEFAULT_FORCED_EDGES = (0.05, 0.1, 0.2)
DEFAULT_STRENGTHS = (0.0, 0.25, 0.5, 0.75, 1.0)


def json_clean(value: Any) -> Any:
    """Return a strict-JSON-compatible copy."""
    if isinstance(value, dict):
        return {str(key): json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_clean(value.tolist())
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def stat(values: Iterable[float]) -> dict[str, float | int]:
    values = np.asarray(list(values) if not isinstance(values, np.ndarray) else values,
                        dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"mean": np.nan, "case_sd": np.nan, "case_sem": np.nan, "n_cases": 0}
    sd = float(values.std(ddof=1)) if len(values) > 1 else np.nan
    return {
        "mean": float(values.mean()),
        "case_sd": sd,
        "case_sem": float(sd / np.sqrt(len(values))) if len(values) > 1 else np.nan,
        "n_cases": int(len(values)),
    }


def case_mean_stat(
    case: np.ndarray,
    values: np.ndarray,
    mask: np.ndarray | None = None,
) -> dict[str, float | int]:
    """Case-balanced mean and SEM for row- or scene-level values."""
    case = np.asarray(case, dtype=np.int64)
    values = np.asarray(values, dtype=np.float64)
    if case.shape != values.shape:
        raise ValueError("case/value shape mismatch")
    keep = np.isfinite(values)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != keep.shape:
            raise ValueError("mask shape mismatch")
        keep &= mask
    if not keep.any():
        return stat(np.empty(0))
    unique, inverse = np.unique(case[keep], return_inverse=True)
    count = np.bincount(inverse, minlength=len(unique))
    total = np.bincount(inverse, weights=values[keep], minlength=len(unique))
    return stat(total / count)


def make_scene_edges(
    scene_prediction: np.ndarray,
    n_quantile_bins: int,
    forced_edges: Iterable[float] = DEFAULT_FORCED_EDGES,
) -> np.ndarray:
    """Build label-independent scene-quantile edges plus fixed physical cuts."""
    prediction = np.asarray(scene_prediction, dtype=np.float64)
    prediction = prediction[np.isfinite(prediction)]
    if len(prediction) < n_quantile_bins or n_quantile_bins < 2:
        raise ValueError("insufficient finite scenes for requested quantile bins")
    quantiles = np.quantile(prediction, np.linspace(0.0, 1.0, n_quantile_bins + 1))
    internal = list(np.unique(quantiles[1:-1]))
    if len(internal) != n_quantile_bins - 1:
        raise RuntimeError("scene prediction has tied quantile edges")
    forced = np.asarray(tuple(forced_edges), dtype=np.float64)
    forced = forced[np.isfinite(forced)]
    # Replace the nearest quantile edge rather than simply inserting each
    # physical edge.  Inserting 0.05 beside the original 0.050226 quantile, for
    # example, creates a tiny and scientifically useless sliver bin.
    for value in np.unique(forced):
        nearest = int(np.argmin(np.abs(np.asarray(internal) - value)))
        internal[nearest] = float(value)
        internal = sorted(set(internal))
        if len(internal) != n_quantile_bins - 1:
            raise RuntimeError("forced boundary collided with another scene edge")
    edges = np.asarray([-np.inf, *internal, np.inf], dtype=np.float64)
    if len(edges) < 3 or not np.all(np.diff(edges) > 0):
        raise RuntimeError("scene edges are not strictly increasing")
    return edges


def assign_bins(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("bin coordinate contains non-finite values")
    index = np.searchsorted(edges, values, side="right") - 1
    if np.any(index < 0) or np.any(index >= len(edges) - 1):
        raise RuntimeError("bin assignment escaped finite edge support")
    return index.astype(np.int16)


def fit_bin_correction(
    scene_prediction: np.ndarray,
    validation_pair_count: np.ndarray,
    validation_residual_sum: np.ndarray,
    edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the exact constant correction and bin index for each scene.

    The returned correction minimizes the squared mean pair residual separately
    in every frozen scene-response bin.
    """
    prediction = np.asarray(scene_prediction, dtype=np.float64)
    count = np.asarray(validation_pair_count, dtype=np.int64)
    residual = np.asarray(validation_residual_sum, dtype=np.float64)
    if prediction.shape != count.shape or prediction.shape != residual.shape:
        raise ValueError("fit arrays have different shapes")
    if np.any(count < 0) or not np.isfinite(residual).all():
        raise ValueError("invalid validation counts or residuals")
    bins = assign_bins(prediction, edges)
    n_bins = len(edges) - 1
    pair_count = np.bincount(bins, weights=count, minlength=n_bins).astype(np.float64)
    residual_sum = np.bincount(
        bins, weights=residual, minlength=n_bins
    ).astype(np.float64)
    if np.any(pair_count <= 0):
        raise RuntimeError("at least one R_scene bin contains no validation pair")
    correction = residual_sum / pair_count
    closure = residual_sum - correction * pair_count
    if np.max(np.abs(closure)) > 2.0e-12:
        raise RuntimeError("fitted bin correction does not close its training mean")
    return correction, bins


def crossfit_bin_correction(
    scenes: pd.DataFrame,
    edges: np.ndarray,
    fold_by_case: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit the lookup on other rendered cases and predict each omitted fold."""
    case = scenes.case.to_numpy(np.int64)
    fold_by_case = np.asarray(fold_by_case, dtype=np.int16)
    if np.any(case < 0) or int(case.max()) >= len(fold_by_case):
        raise ValueError("case lies outside fold lookup")
    fold = fold_by_case[case]
    unique_fold = np.unique(fold)
    if np.any(unique_fold < 0) or not np.array_equal(
        unique_fold, np.arange(len(unique_fold))
    ):
        raise RuntimeError("case-fold labels are not contiguous from zero")
    prediction = scenes.scene_prediction.to_numpy(np.float64)
    count = scenes.validation_pair_count.to_numpy(np.int64)
    residual = scenes.validation_residual_sum.to_numpy(np.float64)
    bins = assign_bins(prediction, edges)
    oof = np.full(len(scenes), np.nan, dtype=np.float64)
    models = np.empty((len(unique_fold), len(edges) - 1), dtype=np.float64)
    for heldout in unique_fold:
        train = fold != heldout
        held = ~train
        model, _ = fit_bin_correction(
            prediction[train], count[train], residual[train], edges
        )
        models[int(heldout)] = model
        oof[held] = model[bins[held]]
    if not np.isfinite(oof).all():
        raise RuntimeError("case-cross-fitted correction has incomplete coverage")
    return oof, models


def aggregate_contiguous_scenes(
    case: np.ndarray,
    input_index: np.ndarray,
    label: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, np.ndarray]:
    """Collapse lexicographically ordered pair rows to primary scenes."""
    case = np.asarray(case, dtype=np.int64)
    input_index = np.asarray(input_index, dtype=np.int64)
    label = np.asarray(label, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if not (case.shape == input_index.shape == label.shape == prediction.shape):
        raise ValueError("pair array shape mismatch")
    if len(case) == 0 or not np.isfinite(label).all() or not np.isfinite(prediction).all():
        raise ValueError("empty or non-finite pair arrays")
    backwards = (case[1:] < case[:-1]) | (
        (case[1:] == case[:-1]) & (input_index[1:] < input_index[:-1])
    )
    if np.any(backwards):
        raise RuntimeError("pair rows are not ordered by (case,input_index)")
    new_scene = (case[1:] != case[:-1]) | (input_index[1:] != input_index[:-1])
    starts = np.concatenate(([0], np.flatnonzero(new_scene) + 1)).astype(np.int64)
    stops = np.concatenate((starts[1:], [len(case)]))
    counts = (stops - starts).astype(np.int32)
    residual = label - prediction
    output = {
        "case": case[starts],
        "input_index": input_index[starts],
        "n_pairs": counts,
        "label_sum": np.add.reduceat(label, starts),
        "prediction_sum": np.add.reduceat(prediction, starts),
        "residual_sum": np.add.reduceat(residual, starts),
        "residual_square_sum": np.add.reduceat(np.square(residual), starts),
        "starts": starts,
    }
    if int(counts.sum()) != len(case):
        raise RuntimeError("scene pair counts do not close to the input rows")
    return output


def selection_masks(scene_prediction: np.ndarray) -> dict[str, np.ndarray]:
    prediction = np.asarray(scene_prediction, dtype=np.float64)
    return {
        "all": np.ones(len(prediction), dtype=bool),
        "outside_le_0p1": prediction <= 0.1,
        "tail_gt_0p1": prediction > 0.1,
        "tail_gt_0p2": prediction > 0.2,
    }


def summarize_scenes(
    case: np.ndarray,
    truth: np.ndarray,
    baseline: np.ndarray,
    n_pairs: np.ndarray,
    correction_per_pair: np.ndarray,
    strength: float,
    mask: np.ndarray,
) -> dict[str, Any]:
    case = np.asarray(case, dtype=np.int64)
    truth = np.asarray(truth, dtype=np.float64)
    baseline = np.asarray(baseline, dtype=np.float64)
    n_pairs = np.asarray(n_pairs, dtype=np.float64)
    correction_per_pair = np.asarray(correction_per_pair, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    corrected = baseline + float(strength) * n_pairs * correction_per_pair
    gap = truth - corrected
    baseline_gap = truth - baseline
    return {
        "n_scenes": int(mask.sum()),
        "n_pairs": int(np.rint(n_pairs[mask].sum())),
        "mean_pairs_per_scene": float(n_pairs[mask].mean()),
        "mean_frozen_scene_prediction": float(baseline[mask].mean()),
        "truth": case_mean_stat(case, truth, mask),
        "prediction": case_mean_stat(case, corrected, mask),
        "truth_minus_prediction": case_mean_stat(case, gap, mask),
        "baseline_truth_minus_prediction": case_mean_stat(case, baseline_gap, mask),
        "applied_scene_correction": case_mean_stat(
            case, corrected - baseline, mask
        ),
    }


def summarize_pairs(
    case: np.ndarray,
    label: np.ndarray,
    prediction: np.ndarray,
    correction: np.ndarray,
    strength: float,
    mask: np.ndarray,
) -> dict[str, Any]:
    case = np.asarray(case, dtype=np.int64)
    label = np.asarray(label, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    correction = np.asarray(correction, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    residual0 = label - prediction
    residual = residual0 - float(strength) * correction
    mse0 = np.square(residual0)
    mse = np.square(residual)
    unique, inverse = np.unique(case[mask], return_inverse=True)
    count = np.bincount(inverse, minlength=len(unique))
    mse0_case = np.bincount(
        inverse, weights=mse0[mask], minlength=len(unique)
    ) / count
    mse_case = np.bincount(
        inverse, weights=mse[mask], minlength=len(unique)
    ) / count
    return {
        "n_pairs": int(mask.sum()),
        "label_mean": case_mean_stat(case, label, mask),
        "prediction_mean": case_mean_stat(
            case, prediction + float(strength) * correction, mask
        ),
        "label_minus_prediction": case_mean_stat(case, residual, mask),
        "mse": stat(mse_case),
        "baseline_mse": stat(mse0_case),
        "mse_minus_baseline": stat(mse_case - mse0_case),
        "mse_percent_change_from_baseline": stat(
            100.0 * (mse_case - mse0_case) / mse0_case
        ),
    }


def summarize_sampled_pairs(
    case: np.ndarray,
    pair_count: np.ndarray,
    residual_sum: np.ndarray,
    residual_square_sum: np.ndarray,
    correction: np.ndarray,
    strength: float,
) -> dict[str, Any]:
    """Summarize pair loss from scene-aggregated validation sufficient statistics."""
    case = np.asarray(case, dtype=np.int64)
    count = np.asarray(pair_count, dtype=np.float64)
    residual_sum = np.asarray(residual_sum, dtype=np.float64)
    square_sum = np.asarray(residual_square_sum, dtype=np.float64)
    correction = np.asarray(correction, dtype=np.float64)
    if not (
        case.shape == count.shape == residual_sum.shape
        == square_sum.shape == correction.shape
    ):
        raise ValueError("sampled-pair aggregate shape mismatch")
    corrected_sum = residual_sum - strength * count * correction
    corrected_square = (
        square_sum - 2.0 * strength * correction * residual_sum
        + np.square(strength * correction) * count
    )
    unique, inverse = np.unique(case, return_inverse=True)
    case_count = np.bincount(inverse, weights=count, minlength=len(unique))
    if np.any(case_count <= 0):
        raise RuntimeError("case-cross-fit contains a case without validation pairs")
    residual_case = np.bincount(
        inverse, weights=corrected_sum, minlength=len(unique)
    ) / case_count
    mse_case = np.bincount(
        inverse, weights=corrected_square, minlength=len(unique)
    ) / case_count
    baseline_mse_case = np.bincount(
        inverse, weights=square_sum, minlength=len(unique)
    ) / case_count
    return {
        "n_pairs": int(np.rint(count.sum())),
        "label_minus_prediction": stat(residual_case),
        "mse": stat(mse_case),
        "baseline_mse": stat(baseline_mse_case),
        "mse_minus_baseline": stat(mse_case - baseline_mse_case),
        "mse_percent_change_from_baseline": stat(
            100.0 * (mse_case - baseline_mse_case) / baseline_mse_case
        ),
    }


def summarize_sampled_scene_gap(
    case: np.ndarray,
    validation_residual_sum: np.ndarray,
    n_pairs: np.ndarray,
    correction_per_pair: np.ndarray,
    strength: float,
    mask: np.ndarray,
    validation_scale: float,
) -> dict[str, Any]:
    """Evaluate a full-scene gap using inverse-sampled validation labels."""
    case = np.asarray(case, dtype=np.int64)
    residual = validation_scale * np.asarray(
        validation_residual_sum, dtype=np.float64
    )
    n_pairs = np.asarray(n_pairs, dtype=np.float64)
    correction = np.asarray(correction_per_pair, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    corrected = residual - strength * n_pairs * correction
    return {
        "n_scenes": int(mask.sum()),
        "n_pairs": int(np.rint(n_pairs[mask].sum())),
        "mean_pairs_per_scene": float(n_pairs[mask].mean()),
        "baseline_truth_minus_prediction": case_mean_stat(case, residual, mask),
        "truth_minus_prediction": case_mean_stat(case, corrected, mask),
        "applied_scene_correction": case_mean_stat(
            case, strength * n_pairs * correction, mask
        ),
    }


def fit_table(
    scenes: pd.DataFrame,
    edges: np.ndarray,
    correction: np.ndarray,
    bins: np.ndarray,
) -> pd.DataFrame:
    n_bins = len(edges) - 1
    case = scenes.case.to_numpy(np.int64)
    count = scenes.validation_pair_count.to_numpy(np.int64)
    residual = scenes.validation_residual_sum.to_numpy(np.float64)
    prediction = scenes.scene_prediction.to_numpy(np.float64)
    unique_case = np.unique(case)
    case_index = np.searchsorted(unique_case, case)
    flat = case_index * n_bins + bins
    count_case = np.bincount(
        flat, weights=count, minlength=len(unique_case) * n_bins
    ).reshape(len(unique_case), n_bins)
    residual_case = np.bincount(
        flat, weights=residual, minlength=len(unique_case) * n_bins
    ).reshape(len(unique_case), n_bins)
    rows = []
    for index in range(n_bins):
        mask = bins == index
        valid = count_case[:, index] > 0
        case_residual = residual_case[valid, index] / count_case[valid, index]
        pair_count = int(count[mask].sum())
        residual_sum = float(residual[mask].sum())
        x_pair_weighted = float(
            np.average(prediction[mask], weights=count[mask])
        )
        info = stat(case_residual)
        rows.append({
            "bin": index,
            "lower": edges[index],
            "upper": edges[index + 1],
            "x_pair_weighted_mean": x_pair_weighted,
            "n_scenes": int(mask.sum()),
            "n_validation_pairs": pair_count,
            "residual_sum": residual_sum,
            "correction": float(correction[index]),
            "case_mean_pair_residual": info["mean"],
            "case_sem_pair_residual": info["case_sem"],
            "n_cases": info["n_cases"],
            "postfit_pooled_pair_residual": (
                residual_sum - correction[index] * pair_count
            ) / pair_count,
        })
    return pd.DataFrame(rows)


def conditional_scene_table(
    name: str,
    case: np.ndarray,
    truth: np.ndarray,
    baseline: np.ndarray,
    n_pairs: np.ndarray,
    correction_per_pair: np.ndarray,
    edges: np.ndarray,
    strength: float,
) -> pd.DataFrame:
    bins = assign_bins(baseline, edges)
    rows = []
    corrected = baseline + strength * n_pairs * correction_per_pair
    for index in range(len(edges) - 1):
        mask = bins == index
        if not mask.any():
            continue
        raw = case_mean_stat(case, truth - baseline, mask)
        calibrated = case_mean_stat(case, truth - corrected, mask)
        rows.append({
            "dataset": name,
            "bin": index,
            "lower": edges[index],
            "upper": edges[index + 1],
            "x_mean": float(np.mean(baseline[mask])),
            "n_scenes": int(mask.sum()),
            "n_pairs": int(np.rint(np.sum(n_pairs[mask]))),
            "raw_gap_mean": raw["mean"],
            "raw_gap_sem": raw["case_sem"],
            "corrected_gap_mean": calibrated["mean"],
            "corrected_gap_sem": calibrated["case_sem"],
            "n_cases": calibrated["n_cases"],
        })
    return pd.DataFrame(rows)


def configure_plot() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 8.5,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def set_scene_xscale(axis: plt.Axes) -> None:
    axis.set_xscale("symlog", linthresh=0.01, linscale=0.8)
    axis.axvline(0.1, color="0.55", linestyle=":", linewidth=0.8)
    axis.axvline(0.2, color="0.70", linestyle=":", linewidth=0.7)


def make_figure(
    fit_bins: pd.DataFrame,
    external_bins: pd.DataFrame,
    anchor_bins: pd.DataFrame,
    strength_table: pd.DataFrame,
    output_prefix: Path,
) -> None:
    configure_plot()
    blue, orange, green = "#0072B2", "#D55E00", "#009E73"
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), constrained_layout=True)

    axis = axes[0, 0]
    axis.errorbar(
        fit_bins.x_pair_weighted_mean,
        fit_bins.correction,
        yerr=fit_bins.case_sem_pair_residual,
        color=blue,
        marker="o",
        markersize=3.5,
        linewidth=1.0,
        capsize=2,
    )
    axis.axhline(0.0, color="0.45", linestyle="--", linewidth=0.8)
    set_scene_xscale(axis)
    axis.set_title("Learned per-pair correction")
    axis.set_xlabel(r"Frozen V2.2 scene sum $P_s$")
    axis.set_ylabel(r"$c(P_s)$")

    for axis, table, title in (
        (axes[0, 1], external_bins, "External half-shear cases 20–39"),
        (axes[1, 0], anchor_bins, "Coherent anchors, cases 400–899"),
    ):
        axis.errorbar(
            table.x_mean, table.raw_gap_mean, yerr=table.raw_gap_sem,
            color=blue, marker="o", markersize=3.2, linewidth=1.0,
            capsize=2, label="V2.2",
        )
        axis.errorbar(
            table.x_mean, table.corrected_gap_mean,
            yerr=table.corrected_gap_sem,
            color=orange, marker="s", markersize=3.2, linewidth=1.0,
            capsize=2, label=r"V2.2 + $c(P_s)$",
        )
        axis.axhline(0.0, color="0.45", linestyle="--", linewidth=0.8)
        set_scene_xscale(axis)
        axis.set_title(title)
        axis.set_xlabel(r"Frozen V2.2 scene sum $P_s$")
        axis.set_ylabel("Truth − prediction")
        axis.legend(frameon=False)

    axis = axes[1, 1]
    x = strength_table.strength.to_numpy(float)
    mse = strength_table.external_validation_pair_mse_percent.to_numpy(float)
    mse_sem = strength_table.external_validation_pair_mse_percent_sem.to_numpy(float)
    axis.errorbar(
        x, mse, yerr=mse_sem, color=orange, marker="s", capsize=2,
        linewidth=1.1, label="External pair MSE change",
    )
    axis.axhline(0.0, color="0.55", linestyle="--", linewidth=0.8)
    axis.set_xlabel("Correction strength")
    axis.set_ylabel("Pair-MSE change (%)", color=orange)
    axis.tick_params(axis="y", colors=orange)
    twin = axis.twinx()
    twin.errorbar(
        x,
        strength_table.external_validation_tail_gap,
        yerr=strength_table.external_validation_tail_gap_sem,
        color=blue,
        marker="o",
        capsize=2,
        linewidth=1.1,
        label="External half-shear tail",
    )
    twin.errorbar(
        x,
        strength_table.anchor_tail_gap,
        yerr=strength_table.anchor_tail_gap_sem,
        color=green,
        marker="^",
        capsize=2,
        linewidth=1.1,
        label="Coherent-anchor tail",
    )
    twin.axhline(0.0, color="0.55", linestyle="--", linewidth=0.8)
    twin.set_ylabel(r"$P_s>0.1$ truth − prediction")
    handles1, labels1 = axis.get_legend_handles_labels()
    handles2, labels2 = twin.get_legend_handles_labels()
    axis.legend(handles1 + handles2, labels1 + labels2, frameon=False, loc="best")
    axis.set_title("Shrinkage trade-off")

    for label, axis in zip("ABCD", axes.flat):
        axis.text(-0.13, 1.05, label, transform=axis.transAxes,
                  fontweight="bold", fontsize=11, va="top")
        axis.spines[["top", "right"]].set_visible(False)
    twin.spines["top"].set_visible(False)
    fig.suptitle(
        r"Scene-axis calibration of frozen V2.2 ($c$ fitted on internal validation pairs)",
        fontsize=12,
    )
    fig.savefig(output_prefix.with_suffix(".png"), dpi=260, bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def fmt_stat(item: dict[str, Any], digits: int = 6) -> str:
    return f"{item['mean']:+.{digits}f} ± {item['case_sem']:.{digits}f}"


def write_markdown(
    path: Path,
    payload: dict[str, Any],
    strength_table: pd.DataFrame,
) -> None:
    literal = payload["strength_results"]["1.0"]
    lines = [
        "# Frozen-V2.2 scene-axis pair-mean calibration",
        "",
        "The lookup is fitted only from official V2.2 internal-validation pair labels in "
        "cases 40–199. Bins use the full frozen V2.2 scene sum; labels never enter the "
        "bin definition. ConstGold was not opened.",
        "",
        "## Literal unshrunk correction",
        "",
        "| dataset / selection | V2.2 truth − prediction | corrected truth − prediction |",
        "|---|---:|---:|",
    ]
    for dataset, selection in (
        ("fit_case_oof", "tail_gt_0p1"),
        ("external_all", "tail_gt_0p1"),
        ("external_validation", "all"),
        ("external_validation", "tail_gt_0p1"),
        ("coherent_anchor_all", "all"),
        ("coherent_anchor_all", "tail_gt_0p1"),
        ("coherent_anchor_c700_899", "tail_gt_0p1"),
    ):
        item = literal[dataset][selection]
        lines.append(
            f"| {dataset}: {selection} | "
            f"{fmt_stat(item['baseline_truth_minus_prediction'])} | "
            f"{fmt_stat(item['truth_minus_prediction'])} |"
        )
    pair = literal["external_validation_pairs"]
    lines.extend([
        "",
        "## Ordinary pair loss on external cases 20–39",
        "",
        f"- Pair residual: {fmt_stat(pair['label_minus_prediction'])}",
        f"- MSE change from V2.2: "
        f"{fmt_stat(pair['mse_percent_change_from_baseline'])}%",
        "",
        "## Strength scan",
        "",
        "| strength | external pair MSE change (%) | external tail gap | anchor tail gap |",
        "|---:|---:|---:|---:|",
    ])
    for row in strength_table.itertuples(index=False):
        lines.append(
            f"| {row.strength:.2f} | "
            f"{row.external_validation_pair_mse_percent:+.6f} ± "
            f"{row.external_validation_pair_mse_percent_sem:.6f} | "
            f"{row.external_validation_tail_gap:+.6f} ± "
            f"{row.external_validation_tail_gap_sem:.6f} | "
            f"{row.anchor_tail_gap:+.6f} ± {row.anchor_tail_gap_sem:.6f} |"
        )
    lines.extend([
        "",
        "Errors are one SEM across rendered cases. The coherent-anchor result is an "
        "external transfer check and was not used to fit the lookup or choose a strength.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-scenes", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--anchor-features", required=True)
    parser.add_argument("--n-quantile-bins", type=int, default=20)
    parser.add_argument("--forced-edge", type=float, action="append")
    parser.add_argument("--strength", type=float, action="append")
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()

    fit_path = Path(args.fit_scenes)
    cache = Path(args.cache)
    anchor_path = Path(args.anchor_features)
    output_prefix = Path(args.output_prefix)
    strengths = tuple(args.strength or DEFAULT_STRENGTHS)
    forced_edges = tuple(args.forced_edge or DEFAULT_FORCED_EDGES)
    if 0.0 not in strengths or 1.0 not in strengths:
        raise ValueError("strength scan must contain the baseline 0 and literal fit 1")
    if len(set(strengths)) != len(strengths) or min(strengths) < 0:
        raise ValueError("strengths must be unique and non-negative")

    outputs = [
        output_prefix.with_suffix(suffix)
        for suffix in (
            ".json", ".md", ".png", ".pdf", ".fit_bins.csv",
            ".external_bins.csv", ".anchor_bins.csv", ".strengths.csv",
            ".model.json",
        )
    ]
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(f"refusing existing outputs: {existing}")

    fit_columns = [
        "case", "input_index", "n_pairs", "scene_prediction",
        "validation_pair_count", "validation_residual_sum",
        "validation_residual_square_sum",
    ]
    fit_scenes = pd.read_feather(fit_path, columns=fit_columns)
    if len(fit_scenes) != 4_506_407:
        raise RuntimeError(f"unexpected fitting-scene count: {len(fit_scenes):,}")
    if set(fit_scenes.case.unique()) != set(range(40, 200)):
        raise RuntimeError("fitting scenes do not cover exactly cases 40--199")
    if int(fit_scenes.validation_pair_count.sum()) != 7_570_479:
        raise RuntimeError("official validation-pair count drifted")
    edges = make_scene_edges(
        fit_scenes.scene_prediction.to_numpy(np.float64),
        args.n_quantile_bins,
        forced_edges,
    )
    correction, fit_bins_index = fit_bin_correction(
        fit_scenes.scene_prediction.to_numpy(np.float64),
        fit_scenes.validation_pair_count.to_numpy(np.int64),
        fit_scenes.validation_residual_sum.to_numpy(np.float64),
        edges,
    )
    fit_bins = fit_table(fit_scenes, edges, correction, fit_bins_index)
    print(
        f"fit: scenes={len(fit_scenes):,} validation_pairs="
        f"{int(fit_scenes.validation_pair_count.sum()):,} bins={len(correction)}",
        flush=True,
    )

    metadata_path = cache / "metadata.json"
    with metadata_path.open(encoding="utf-8") as handle:
        cache_metadata = json.load(handle)
    if cache_metadata["source_tag"] != "lsst_r_extnbr_v22":
        raise RuntimeError("external cache is not frozen V2.2")
    if cache_metadata["evaluation_case_window"] != [0, 39]:
        raise RuntimeError("external cache case window drifted")
    fold_by_case = np.load(
        cache / cache_metadata["arrays"]["fold_by_case"], mmap_mode="r"
    )
    fit_oof_correction, fit_fold_models = crossfit_bin_correction(
        fit_scenes, edges, fold_by_case
    )
    arrays = {
        name: np.load(cache / filename, mmap_mode="r")
        for name, filename in cache_metadata["arrays"].items()
        if name in {"case", "input_index", "label", "v22_prediction"}
    }
    case_all = arrays["case"]
    stop = int(np.searchsorted(case_all, 40, side="left"))
    if stop != cache_metadata["n_evaluation_rows"]:
        raise RuntimeError("external-row boundary does not match cache metadata")
    case = np.asarray(case_all[:stop], dtype=np.int64)
    input_index = np.asarray(arrays["input_index"][:stop], dtype=np.int64)
    label = np.asarray(arrays["label"][:stop], dtype=np.float64)
    prediction = np.asarray(arrays["v22_prediction"][:stop], dtype=np.float64)
    external = aggregate_contiguous_scenes(case, input_index, label, prediction)
    external_bins_index = assign_bins(external["prediction_sum"], edges)
    external_correction_scene = correction[external_bins_index]
    external_correction_row = np.repeat(
        external_correction_scene, external["n_pairs"]
    )
    if len(external_correction_row) != len(case):
        raise RuntimeError("row-level correction expansion failed")
    print(
        f"external: pairs={len(case):,} scenes={len(external['case']):,}",
        flush=True,
    )

    anchor_columns = [
        "case", "input_index", "R_blend_truth", "scene_prediction",
        "bias_truth_minus_model", "log1p_n_pairs",
    ]
    anchor = pd.read_feather(anchor_path, columns=anchor_columns)
    if len(anchor) != 1_703_884 or set(anchor.case.unique()) != set(range(400, 900)):
        raise RuntimeError("coherent-anchor population drifted")
    identity = (
        anchor.R_blend_truth.to_numpy(np.float64)
        - anchor.scene_prediction.to_numpy(np.float64)
        - anchor.bias_truth_minus_model.to_numpy(np.float64)
    )
    if np.max(np.abs(identity)) > 1.0e-10:
        raise RuntimeError("anchor truth-model residual identity failed")
    anchor_n = np.rint(np.expm1(anchor.log1p_n_pairs.to_numpy(np.float64))).astype(np.int32)
    if np.any(anchor_n < 0) or np.max(
        np.abs(np.log1p(anchor_n) - anchor.log1p_n_pairs.to_numpy(np.float64))
    ) > 1.0e-10:
        raise RuntimeError("anchor pair multiplicity is not integer-valued")
    anchor_prediction = anchor.scene_prediction.to_numpy(np.float64)
    anchor_bin_index = assign_bins(anchor_prediction, edges)
    anchor_correction = correction[anchor_bin_index]
    print(
        f"anchor: scenes={len(anchor):,} mean_supported_pairs={anchor_n.mean():.3f}",
        flush=True,
    )

    external_masks = selection_masks(external["prediction_sum"])
    anchor_masks = selection_masks(anchor_prediction)
    external_development_rows = case <= 19
    external_validation_rows = (case >= 20) & (case <= 39)
    external_all_rows = np.ones(len(case), dtype=bool)
    external_development_scenes = external["case"] <= 19
    external_validation_scenes = (external["case"] >= 20) & (external["case"] <= 39)
    external_all_scenes = np.ones(len(external["case"]), dtype=bool)
    anchor_all = np.ones(len(anchor), dtype=bool)
    anchor_c700 = anchor.case.to_numpy(np.int64) >= 700

    strength_results: dict[str, Any] = {}
    strength_rows = []
    for strength in strengths:
        key = str(float(strength))
        result: dict[str, Any] = {
            "fit_case_oof_pairs": summarize_sampled_pairs(
                fit_scenes.case.to_numpy(np.int64),
                fit_scenes.validation_pair_count.to_numpy(np.int64),
                fit_scenes.validation_residual_sum.to_numpy(np.float64),
                fit_scenes.validation_residual_square_sum.to_numpy(np.float64),
                fit_oof_correction, strength,
            ),
            "fit_case_oof": {},
            "external_all_pairs": summarize_pairs(
                case, label, prediction, external_correction_row, strength,
                external_all_rows,
            ),
            "external_development_pairs": summarize_pairs(
                case, label, prediction, external_correction_row, strength,
                external_development_rows,
            ),
            "external_validation_pairs": summarize_pairs(
                case, label, prediction, external_correction_row, strength,
                external_validation_rows,
            ),
            "external_development": {},
            "external_validation": {},
            "external_all": {},
            "coherent_anchor_all": {},
            "coherent_anchor_c700_899": {},
        }
        for name, mask in selection_masks(
            fit_scenes.scene_prediction.to_numpy(np.float64)
        ).items():
            result["fit_case_oof"][name] = summarize_sampled_scene_gap(
                fit_scenes.case.to_numpy(np.int64),
                fit_scenes.validation_residual_sum.to_numpy(np.float64),
                fit_scenes.n_pairs.to_numpy(np.int64),
                fit_oof_correction, strength, mask, validation_scale=5.0,
            )
        for name, mask in external_masks.items():
            result["external_all"][name] = summarize_scenes(
                external["case"], external["label_sum"],
                external["prediction_sum"], external["n_pairs"],
                external_correction_scene, strength,
                mask & external_all_scenes,
            )
            result["external_development"][name] = summarize_scenes(
                external["case"], external["label_sum"],
                external["prediction_sum"], external["n_pairs"],
                external_correction_scene, strength,
                mask & external_development_scenes,
            )
            result["external_validation"][name] = summarize_scenes(
                external["case"], external["label_sum"],
                external["prediction_sum"], external["n_pairs"],
                external_correction_scene, strength,
                mask & external_validation_scenes,
            )
        for name, mask in anchor_masks.items():
            result["coherent_anchor_all"][name] = summarize_scenes(
                anchor.case.to_numpy(np.int64),
                anchor.R_blend_truth.to_numpy(np.float64), anchor_prediction,
                anchor_n, anchor_correction, strength, mask & anchor_all,
            )
            result["coherent_anchor_c700_899"][name] = summarize_scenes(
                anchor.case.to_numpy(np.int64),
                anchor.R_blend_truth.to_numpy(np.float64), anchor_prediction,
                anchor_n, anchor_correction, strength, mask & anchor_c700,
            )
        strength_results[key] = result
        pair_mse = result["external_validation_pairs"][
            "mse_percent_change_from_baseline"
        ]
        fit_oof_mse = result["fit_case_oof_pairs"][
            "mse_percent_change_from_baseline"
        ]
        fit_oof_tail = result["fit_case_oof"]["tail_gt_0p1"][
            "truth_minus_prediction"
        ]
        external_all_tail = result["external_all"]["tail_gt_0p1"][
            "truth_minus_prediction"
        ]
        external_tail = result["external_validation"]["tail_gt_0p1"][
            "truth_minus_prediction"
        ]
        anchor_tail = result["coherent_anchor_all"]["tail_gt_0p1"][
            "truth_minus_prediction"
        ]
        strength_rows.append({
            "strength": float(strength),
            "fit_oof_pair_mse_percent": fit_oof_mse["mean"],
            "fit_oof_pair_mse_percent_sem": fit_oof_mse["case_sem"],
            "fit_oof_tail_gap": fit_oof_tail["mean"],
            "fit_oof_tail_gap_sem": fit_oof_tail["case_sem"],
            "external_all_tail_gap": external_all_tail["mean"],
            "external_all_tail_gap_sem": external_all_tail["case_sem"],
            "external_validation_pair_mse_percent": pair_mse["mean"],
            "external_validation_pair_mse_percent_sem": pair_mse["case_sem"],
            "external_validation_tail_gap": external_tail["mean"],
            "external_validation_tail_gap_sem": external_tail["case_sem"],
            "anchor_tail_gap": anchor_tail["mean"],
            "anchor_tail_gap_sem": anchor_tail["case_sem"],
        })
        print(
            f"strength={strength:.2f}: external tail="
            f"{external_tail['mean']:+.6f}±{external_tail['case_sem']:.6f}, "
            f"fit-OOF tail={fit_oof_tail['mean']:+.6f}±{fit_oof_tail['case_sem']:.6f}, "
            f"anchor tail={anchor_tail['mean']:+.6f}±{anchor_tail['case_sem']:.6f}, "
            f"pair MSE={pair_mse['mean']:+.6f}%",
            flush=True,
        )

    strength_table = pd.DataFrame(strength_rows).sort_values("strength")
    literal_strength = 1.0
    external_bins = conditional_scene_table(
        "external_half_shear_c20_39",
        external["case"][external_validation_scenes],
        external["label_sum"][external_validation_scenes],
        external["prediction_sum"][external_validation_scenes],
        external["n_pairs"][external_validation_scenes],
        external_correction_scene[external_validation_scenes],
        edges,
        literal_strength,
    )
    anchor_bins = conditional_scene_table(
        "coherent_anchor_c400_899",
        anchor.case.to_numpy(np.int64),
        anchor.R_blend_truth.to_numpy(np.float64),
        anchor_prediction, anchor_n, anchor_correction, edges, literal_strength,
    )

    fit_residual = fit_scenes.validation_residual_sum.to_numpy(np.float64)
    fit_square = fit_scenes.validation_residual_square_sum.to_numpy(np.float64)
    fit_count = fit_scenes.validation_pair_count.to_numpy(np.float64)
    fit_c = correction[fit_bins_index]
    training_mse_before = float(fit_square.sum() / fit_count.sum())
    training_mse_after = float(
        np.sum(fit_square - 2.0 * fit_c * fit_residual + fit_count * np.square(fit_c))
        / fit_count.sum()
    )
    model_payload = {
        "schema_version": 1,
        "base_model": "lsst_r_extnbr_v22",
        "kind": "additive_per_pair_lookup_conditioned_on_frozen_scene_sum",
        "scene_coordinate": "sum of frozen V2.2 predictions over its supported pair list",
        "bin_edge_convention": "lower-inclusive, upper-exclusive; final bin includes +inf",
        "edges": edges,
        "per_pair_correction": correction,
        "literal_strength": literal_strength,
        "fit_population": {
            "cases": [40, 199],
            "split": "official sklearn random-row validation (20%, random_state=321)",
            "n_scenes": len(fit_scenes),
            "n_validation_pairs": int(fit_count.sum()),
        },
    }
    payload = {
        "schema_version": 1,
        "objective": (
            "within each frozen R_scene bin, minimize squared mean pair residual "
            "with an additive constant per pair"
        ),
        "fit": {
            "scene_file": str(fit_path),
            "n_scenes": len(fit_scenes),
            "n_validation_pairs": int(fit_count.sum()),
            "n_bins": len(correction),
            "n_quantile_bins_requested": args.n_quantile_bins,
            "forced_edges": forced_edges,
            "training_pair_mse_before": training_mse_before,
            "training_pair_mse_after": training_mse_after,
            "training_pair_mse_percent_change": 100.0 * (
                training_mse_after / training_mse_before - 1.0
            ),
            "max_abs_postfit_pooled_bin_residual": float(
                fit_bins.postfit_pooled_pair_residual.abs().max()
            ),
            "case_crossfit_folds": int(fit_fold_models.shape[0]),
            "case_crossfit_lookup_max_sd_across_folds": float(
                np.max(np.std(fit_fold_models, axis=0, ddof=1))
            ),
        },
        "model": model_payload,
        "strength_results": strength_results,
        "provenance": {
            "cache_metadata": str(metadata_path),
            "cache_metadata_sha256": sha256(metadata_path),
            "anchor_features": str(anchor_path),
            "fit_scenes": str(fit_path),
            "external_cases": [0, 39],
            "external_development_cases": [0, 19],
            "external_validation_cases": [20, 39],
            "coherent_anchor_cases": [400, 899],
            "constgold_opened": False,
            "coherent_truth_used_for_fitting_or_strength_selection": False,
        },
    }

    fit_bins.to_csv(output_prefix.with_suffix(".fit_bins.csv"), index=False)
    external_bins.to_csv(output_prefix.with_suffix(".external_bins.csv"), index=False)
    anchor_bins.to_csv(output_prefix.with_suffix(".anchor_bins.csv"), index=False)
    strength_table.to_csv(output_prefix.with_suffix(".strengths.csv"), index=False)
    with output_prefix.with_suffix(".model.json").open("x", encoding="utf-8") as handle:
        json.dump(json_clean(model_payload), handle, indent=2, sort_keys=True,
                  allow_nan=False)
        handle.write("\n")
    with output_prefix.with_suffix(".json").open("x", encoding="utf-8") as handle:
        json.dump(json_clean(payload), handle, indent=2, sort_keys=True,
                  allow_nan=False)
        handle.write("\n")
    write_markdown(output_prefix.with_suffix(".md"), payload, strength_table)
    make_figure(fit_bins, external_bins, anchor_bins, strength_table, output_prefix)
    print(f"wrote {output_prefix}.*", flush=True)
    print("V22_RSCENE_PAIR_MEAN_CALIBRATION_DONE", flush=True)


if __name__ == "__main__":
    main()
