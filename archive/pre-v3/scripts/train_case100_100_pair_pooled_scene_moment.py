#!/usr/bin/env python3
"""Tune and fit a pair-MSE plus pooled scene-moment correction.

The frozen c0--99 base is reused. Candidate corrections fit cases 100--159
and are selected only on cases 160--199. The selected strength and a lambda=0
control are then refit on cases 100--199. Scene moments use separate 20-bin
coordinates in the full-neighbour base sum and physical log10(Q_d2).
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import time
from typing import Any, Callable

import numpy as np
import pandas as pd
import xgboost as xgb

from scripts.train_case100_100_binned_sheared_scene_mse import load_frozen_base
from scripts.train_newbase_oldway_fullneighbour_160_40 import (
    CORRECTION_PARAMS,
    CORRECTION_TREES,
    N_SCENE_BINS,
    build_full_base_context,
    case_balanced_pair_metrics,
    load_full_metadata,
    physical_base_prediction,
    scene_index,
)
from scripts.tune_sequential_rblend_optuna import (
    CONDITIONAL_FEATURES,
    atomic_json,
    atomic_model,
    json_clean,
)
from scripts.v22_grouped_rscene_common import (
    assign_scene_bins,
    case_offsets,
    conditional_matrix,
    load_source_metadata,
    make_scene_edges,
    mmap_array,
    sha256,
)
from scripts.v22_proxy_correction_common import (
    PROXY_DEFINITION,
    PROXY_FEATURE,
    aligned_full_scene_context,
)


FIT_CASE_MIN = 100
FIT_CASE_MAX = 159
SELECT_CASE_MIN = 160
SELECT_CASE_MAX = 199
FINAL_CASE_MIN = 100
FINAL_CASE_MAX = 199
FEATURES = [*CONDITIONAL_FEATURES, PROXY_FEATURE]
DEFAULT_LAMBDAS = (0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3)


def parse_lambdas(text: str) -> list[float]:
    values = sorted(set(float(value) for value in text.split(",")))
    if not values or values[0] < 0.0 or 0.0 not in values:
        raise ValueError("lambda scan must be non-negative and include zero")
    return values


def case_window_mask(values: np.ndarray, low: int, high: int) -> np.ndarray:
    values = np.asarray(values)
    return (values >= low) & (values <= high)


def add_proxy_feature(base_features: np.ndarray, log_q: np.ndarray) -> np.ndarray:
    base_features = np.asarray(base_features, dtype=np.float32)
    log_q = np.asarray(log_q, dtype=np.float32)
    if log_q.shape != (len(base_features),):
        raise ValueError("physical-proxy feature does not align")
    output = np.empty((len(base_features), len(FEATURES)), dtype=np.float32)
    output[:, :-1] = base_features
    output[:, -1] = log_q
    if not np.isfinite(output).all():
        raise RuntimeError("non-finite pooled-scene correction feature")
    return output


def load_q_context(
    source_cache: Path,
    full_cache: Path,
    source_meta: dict[str, Any],
    full_meta: dict[str, Any],
    primary: np.ndarray,
    case_min: int,
    case_max: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return row log-Q/count and one log-Q per source scene in case order."""
    offsets = case_offsets(source_meta)
    row_logq: list[np.ndarray] = []
    row_count: list[np.ndarray] = []
    scene_logq: list[np.ndarray] = []
    for case in range(case_min, case_max + 1):
        start, stop = int(offsets[case]), int(offsets[case + 1])
        local_primary = np.asarray(primary[start:stop], dtype=np.int64)
        logq, _, count = aligned_full_scene_context(
            local_primary, full_cache, full_meta, case
        )
        change = np.empty(len(local_primary), dtype=bool)
        change[0] = True
        change[1:] = local_primary[1:] != local_primary[:-1]
        local_starts = np.flatnonzero(change)
        local_counts = np.diff(np.r_[local_starts, len(local_primary)])
        # local_counts is the labelled/sheared multiplicity, whereas count is
        # deliberately the all-neighbour deployment multiplicity.  They must
        # not be equated; only constancy of the full count within each source
        # scene is required here.  Its exact values are checked later against
        # build_full_base_context's independently constructed full counts.
        if not np.array_equal(
            count.astype(np.int64),
            np.repeat(count[local_starts].astype(np.int64), local_counts),
        ):
            raise RuntimeError(f"case {case}: full count varies within scene")
        if not np.allclose(
            logq, np.repeat(logq[local_starts], local_counts), rtol=0.0, atol=0.0
        ):
            raise RuntimeError(f"case {case}: log-Q is not constant within scene")
        row_logq.append(logq.astype(np.float32))
        row_count.append(count.astype(np.int16))
        scene_logq.append(logq[local_starts].astype(np.float64))
    return (
        np.concatenate(row_logq),
        np.concatenate(row_count),
        np.concatenate(scene_logq),
    )


def make_axis(
    name: str,
    scene_case: np.ndarray,
    scene_count: np.ndarray,
    coordinate: np.ndarray,
    edges: np.ndarray | None,
    forced_edges: tuple[float, ...],
) -> dict[str, Any]:
    scene_case = np.asarray(scene_case, dtype=np.int64)
    scene_count = np.asarray(scene_count, dtype=np.int64)
    coordinate = np.asarray(coordinate, dtype=np.float64)
    if not (len(scene_case) == len(scene_count) == len(coordinate)):
        raise ValueError(f"{name}: scene arrays do not align")
    if edges is None:
        edges = make_scene_edges(
            coordinate,
            n_quantile_bins=N_SCENE_BINS,
            forced_edges=forced_edges,
        )
    bins = assign_scene_bins(coordinate, edges).astype(np.int64)
    unique_case, case_inverse = np.unique(scene_case, return_inverse=True)
    flat = case_inverse * N_SCENE_BINS + bins
    cell_scene_count = np.bincount(
        flat, minlength=len(unique_case) * N_SCENE_BINS
    ).reshape(len(unique_case), N_SCENE_BINS).astype(np.float64)
    if np.any(cell_scene_count <= 0.0):
        raise RuntimeError(f"{name}: an occupied case/bin cell is missing")
    bin_case_count = np.sum(cell_scene_count > 0.0, axis=0).astype(np.float64)
    scene_coeff = 1.0 / (
        bin_case_count[bins] * cell_scene_count[case_inverse, bins]
    )
    row_bin = np.repeat(bins, scene_count).astype(np.uint8)
    row_coeff = np.repeat(scene_coeff, scene_count).astype(np.float64)
    if len(row_bin) != int(scene_count.sum()):
        raise RuntimeError(f"{name}: row expansion does not close")
    return {
        "name": name,
        "edges": np.asarray(edges, dtype=np.float64),
        "scene_bin": bins.astype(np.uint8),
        "scene_flat_cell": flat.astype(np.int32),
        "cell_scene_count": cell_scene_count,
        "bin_case_count": bin_case_count,
        "row_bin": row_bin,
        "row_coeff": row_coeff,
        "n_cases": int(len(unique_case)),
    }


def axis_curve(
    error: np.ndarray,
    scene_starts: np.ndarray,
    axis: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return scene errors, case-bin means, and across-case bin means."""
    scene_error = np.add.reduceat(np.asarray(error, dtype=np.float64), scene_starts)
    flat = np.asarray(axis["scene_flat_cell"], dtype=np.int64)
    cell_count = np.asarray(axis["cell_scene_count"], dtype=np.float64)
    cell_sum = np.bincount(
        flat, weights=scene_error, minlength=cell_count.size
    ).reshape(cell_count.shape)
    cell_mean = cell_sum / cell_count
    curve = np.mean(cell_mean, axis=0)
    return scene_error, cell_mean, curve


def prepare_objective(
    target: np.ndarray,
    scene_starts: np.ndarray,
    axes: list[dict[str, Any]],
) -> dict[str, Any]:
    target = np.asarray(target, dtype=np.float64)
    zero_error = -target
    pair_mse0 = float(np.mean(np.square(zero_error)))
    if pair_mse0 <= 0.0 or not np.isfinite(pair_mse0):
        raise RuntimeError("invalid raw pair MSE normalization")
    for axis in axes:
        _, _, curve = axis_curve(zero_error, scene_starts, axis)
        mse0 = float(np.mean(np.square(curve)))
        if mse0 <= 0.0 or not np.isfinite(mse0):
            raise RuntimeError(f"{axis['name']}: invalid scene normalization")
        axis["curve_mse0"] = mse0
        axis["gradient_scale"] = (
            len(target) * pair_mse0 / (N_SCENE_BINS * mse0)
        )
    return {"pair_mse0": pair_mse0, "axes": axes}


def objective_value(
    prediction: np.ndarray,
    target: np.ndarray,
    scene_starts: np.ndarray,
    prepared: dict[str, Any],
    strength: float,
) -> float:
    error = np.asarray(prediction, dtype=np.float64) - np.asarray(
        target, dtype=np.float64
    )
    value = 0.5 * float(np.sum(np.square(error)))
    scene_value = 0.0
    for axis in prepared["axes"]:
        _, _, curve = axis_curve(error, scene_starts, axis)
        scene_value += 0.5 * float(np.mean(np.square(curve))) / float(
            axis["curve_mse0"]
        )
    value += (
        float(strength)
        * len(error)
        * float(prepared["pair_mse0"])
        * scene_value
        / len(prepared["axes"])
    )
    return float(value)


def make_objective(
    scene_starts: np.ndarray,
    prepared: dict[str, Any],
    strength: float,
) -> Callable[[np.ndarray, xgb.DMatrix], tuple[np.ndarray, np.ndarray]]:
    axes = prepared["axes"]
    axis_weight = float(strength) / len(axes)
    hessian = np.ones(len(axes[0]["row_bin"]), dtype=np.float64)
    for axis in axes:
        row_bin = np.asarray(axis["row_bin"], dtype=np.int64)
        row_coeff = np.asarray(axis["row_coeff"], dtype=np.float64)
        bin_total_coeff = np.bincount(
            row_bin, weights=row_coeff, minlength=N_SCENE_BINS
        )
        # Block approximation: if a tree leaf contains a complete pooled bin,
        # the summed Hessian equals its exact common-leaf curvature, including
        # the cross-row terms.  The literal diagonal (coeff**2) is far too
        # small for this coupled moment and permits explosive leaf steps.
        hessian += (
            axis_weight
            * float(axis["gradient_scale"])
            * row_coeff
            * bin_total_coeff[row_bin]
        )
    hessian32 = hessian.astype(np.float32)

    def objective(
        prediction: np.ndarray, matrix: xgb.DMatrix
    ) -> tuple[np.ndarray, np.ndarray]:
        target = matrix.get_label()
        error = np.asarray(prediction, dtype=np.float64) - np.asarray(
            target, dtype=np.float64
        )
        gradient = error.copy()
        for axis in axes:
            _, _, curve = axis_curve(error, scene_starts, axis)
            gradient += (
                axis_weight
                * float(axis["gradient_scale"])
                * curve[np.asarray(axis["row_bin"], dtype=np.int64)]
                * np.asarray(axis["row_coeff"], dtype=np.float64)
            )
        return gradient.astype(np.float32), hessian32

    return objective


def finite_difference_self_test() -> dict[str, float]:
    scene_count = np.asarray([2, 1, 2, 1, 1, 2], dtype=np.int64)
    scene_case = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64)
    scene_starts = np.r_[0, np.cumsum(scene_count[:-1])]
    coordinate1 = np.asarray([-2, 2, -1, 1, -3, 3], dtype=float)
    coordinate2 = np.asarray([-4, 4, -2, 2, -1, 1], dtype=float)
    # The production bin count is temporarily replaced in the fake axes.
    def fake_axis(name: str, coordinate: np.ndarray) -> dict[str, Any]:
        bins = (coordinate > 0).astype(np.int64)
        flat = scene_case * 2 + bins
        cell_count = np.bincount(flat, minlength=6).reshape(3, 2).astype(float)
        coeff = 1.0 / (3.0 * cell_count[scene_case, bins])
        return {
            "name": name,
            "scene_flat_cell": flat,
            "cell_scene_count": cell_count,
            "row_bin": np.repeat(bins, scene_count),
            "row_coeff": np.repeat(coeff, scene_count),
        }
    axes = [fake_axis("a", coordinate1), fake_axis("b", coordinate2)]
    target = np.linspace(-0.8, 0.9, int(scene_count.sum()))
    prediction = np.linspace(0.2, -0.3, len(target))
    error0 = -target
    pair_mse0 = float(np.mean(error0**2))
    for axis in axes:
        _, _, curve = axis_curve(error0, scene_starts, axis)
        axis["curve_mse0"] = float(np.mean(curve**2))
        axis["gradient_scale"] = (
            len(target) * pair_mse0 / (2 * axis["curve_mse0"])
        )
    prepared = {"pair_mse0": pair_mse0, "axes": axes}
    strength = 0.037
    objective = make_objective(scene_starts, prepared, strength)
    class FakeMatrix:
        def get_label(self) -> np.ndarray:
            return target.astype(np.float32)
    analytic, hessian = objective(prediction.astype(np.float32), FakeMatrix())
    eps = 2.0e-5
    numeric = np.empty(len(target))
    for index in range(len(target)):
        plus = prediction.copy()
        minus = prediction.copy()
        plus[index] += eps
        minus[index] -= eps
        numeric[index] = (
            objective_value(plus, target, scene_starts, prepared, strength)
            - objective_value(minus, target, scene_starts, prepared, strength)
        ) / (2.0 * eps)
    max_abs = float(np.max(np.abs(analytic.astype(float) - numeric)))
    if max_abs > 2.0e-4 or np.any(hessian <= 0.0):
        raise RuntimeError(f"pooled-scene objective self-test failed: {max_abs}")
    return {
        "gradient_max_abs_finite_difference_error": max_abs,
        "hessian_min": float(hessian.min()),
        "hessian_max": float(hessian.max()),
    }


def model_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    case: np.ndarray,
    scene_starts: np.ndarray,
    axes: list[dict[str, Any]],
    target_scale: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    physical_truth = target * target_scale
    physical_prediction = prediction * target_scale
    pair = case_balanced_pair_metrics(
        physical_truth, physical_prediction, np.asarray(case)
    )
    error = prediction - target
    rows: list[dict[str, Any]] = []
    scene: dict[str, Any] = {}
    for axis in axes:
        _, cell_mean, curve = axis_curve(error, scene_starts, axis)
        physical_curve = -curve * target_scale
        cell_residual = -cell_mean * target_scale
        sem = cell_residual.std(axis=0, ddof=1) / np.sqrt(cell_residual.shape[0])
        scene[axis["name"]] = {
            "curve_rms": float(np.sqrt(np.mean(physical_curve**2))),
            "curve_mean_abs": float(np.mean(np.abs(physical_curve))),
            "curve_max_abs": float(np.max(np.abs(physical_curve))),
            "curve": physical_curve,
            "curve_case_sem": sem,
        }
        edges = np.asarray(axis["edges"], dtype=float)
        for index in range(N_SCENE_BINS):
            rows.append({
                "axis": axis["name"],
                "bin": int(index),
                "lower": float(edges[index]),
                "upper": float(edges[index + 1]),
                "residual_mean": float(physical_curve[index]),
                "residual_case_sem": float(sem[index]),
            })
    return {"pair": pair, "scene": scene}, rows


def global_scene_stat(
    prediction: np.ndarray,
    target: np.ndarray,
    scene_starts: np.ndarray,
    scene_case: np.ndarray,
    target_scale: float,
) -> dict[str, float | int]:
    error = (np.asarray(target) - np.asarray(prediction)) * target_scale
    scene_residual = np.add.reduceat(error.astype(np.float64), scene_starts)
    unique, inverse = np.unique(scene_case, return_inverse=True)
    counts = np.bincount(inverse)
    means = np.bincount(inverse, weights=scene_residual) / counts
    return {
        "mean": float(means.mean()),
        "case_sem": float(means.std(ddof=1) / np.sqrt(len(means))),
        "case_sd": float(means.std(ddof=1)),
        "n_cases": int(len(unique)),
    }


def fit_booster(
    features: np.ndarray,
    target: np.ndarray,
    scene_starts: np.ndarray,
    prepared: dict[str, Any],
    strength: float,
    n_jobs: int,
) -> tuple[xgb.Booster, float]:
    matrix = xgb.DMatrix(features, label=target, feature_names=FEATURES)
    params = CORRECTION_PARAMS | {
        "n_jobs": n_jobs,
        "disable_default_eval_metric": 1,
    }
    started = time.time()
    booster = xgb.train(
        params,
        matrix,
        obj=make_objective(scene_starts, prepared, strength),
        num_boost_round=CORRECTION_TREES,
        verbose_eval=False,
    )
    return booster, time.time() - started


def save_csv(path: Path, frame: pd.DataFrame) -> None:
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-cache", required=True)
    parser.add_argument("--full-neighbour-cache", required=True)
    parser.add_argument("--source-stack-summary", required=True)
    parser.add_argument(
        "--lambdas",
        default=",".join(str(value) for value in DEFAULT_LAMBDAS),
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "summary.json"
    if summary_path.exists():
        print(f"pooled scene-moment training already complete: {summary_path}")
        return
    lambdas = parse_lambdas(args.lambdas)
    self_test = finite_difference_self_test()
    n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", "16"))
    source_cache = Path(args.source_cache).resolve()
    full_cache = Path(args.full_neighbour_cache).resolve()
    source_summary_path = Path(args.source_stack_summary).resolve()
    source_summary, base, base_path = load_frozen_base(source_summary_path, n_jobs)
    source_meta = load_source_metadata(source_cache)
    full_meta = load_full_metadata(full_cache, source_cache)
    if sha256(source_cache / "metadata.json") != source_summary["source"][
        "metadata_sha256"
    ]:
        raise RuntimeError("source cache differs from frozen base")

    case = mmap_array(source_cache, source_meta, "case")
    primary = mmap_array(source_cache, source_meta, "input_index")
    label = mmap_array(source_cache, source_meta, "label")
    x_scaled = mmap_array(source_cache, source_meta, "x_scaled")
    x_raw = mmap_array(source_cache, source_meta, "x_raw")
    starts, counts, scene_case_all, scene_offsets = scene_index(case, primary)
    final_row_mask = case_window_mask(case, FINAL_CASE_MIN, FINAL_CASE_MAX)
    final_index = np.flatnonzero(final_row_mask).astype(np.int32)
    final_scene_mask = case_window_mask(
        scene_case_all, FINAL_CASE_MIN, FINAL_CASE_MAX
    )
    final_scene_case = np.asarray(scene_case_all[final_scene_mask], dtype=np.int16)
    final_scene_counts = np.asarray(counts[final_scene_mask], dtype=np.int32)
    final_scene_starts = np.r_[
        0, np.cumsum(final_scene_counts[:-1], dtype=np.int64)
    ]
    if int(final_scene_counts.sum()) != len(final_index):
        raise RuntimeError("final c100--199 scene indexing does not close")

    target_mean = float(source_summary["source"]["target_mean"])
    target_scale = float(source_summary["source"]["target_scale"])
    all_base = physical_base_prediction(
        base, x_scaled, target_mean, target_scale
    )
    full_scene_base, full_scene_count = build_full_base_context(
        base=base,
        full_cache=full_cache,
        full_meta=full_meta,
        source_primary=primary,
        source_scene_starts=starts,
        source_scene_offsets=scene_offsets,
        target_mean=target_mean,
        target_scale=target_scale,
    )
    row_logq, q_row_count, scene_logq = load_q_context(
        source_cache,
        full_cache,
        source_meta,
        full_meta,
        primary,
        FINAL_CASE_MIN,
        FINAL_CASE_MAX,
    )
    if len(row_logq) != len(final_index) or len(scene_logq) != len(final_scene_case):
        raise RuntimeError("physical-proxy context coverage mismatch")
    if not np.array_equal(
        q_row_count.astype(np.int64),
        np.repeat(full_scene_count[final_scene_mask], final_scene_counts).astype(np.int64),
    ):
        raise RuntimeError("proxy and base full-neighbour multiplicities differ")

    row_scene_base = np.repeat(full_scene_base, counts)
    row_scene_count = np.repeat(full_scene_count, counts).astype(np.int32)
    original_features = conditional_matrix(
        x_scaled,
        x_raw,
        all_base,
        row_scene_base,
        row_scene_count,
        final_index,
    )
    features = add_proxy_feature(original_features, row_logq)
    del original_features, row_scene_base, row_scene_count, q_row_count
    gc.collect()
    physical_target = (
        np.asarray(label[final_index], dtype=np.float64)
        - np.asarray(all_base[final_index], dtype=np.float64)
    )
    target = (physical_target / target_scale).astype(np.float32)
    final_case = np.asarray(case[final_index], dtype=np.int16)
    final_scene_base = np.asarray(full_scene_base[final_scene_mask], dtype=np.float64)

    fit_row = case_window_mask(final_case, FIT_CASE_MIN, FIT_CASE_MAX)
    select_row = case_window_mask(final_case, SELECT_CASE_MIN, SELECT_CASE_MAX)
    fit_scene = case_window_mask(final_scene_case, FIT_CASE_MIN, FIT_CASE_MAX)
    select_scene = case_window_mask(
        final_scene_case, SELECT_CASE_MIN, SELECT_CASE_MAX
    )
    if not np.array_equal(fit_row, np.arange(len(final_case)) < int(fit_row.sum())):
        raise RuntimeError("fit rows are not a contiguous prefix")
    if not np.array_equal(fit_scene, np.arange(len(final_scene_case)) < int(fit_scene.sum())):
        raise RuntimeError("fit scenes are not a contiguous prefix")
    n_fit_row = int(fit_row.sum())
    n_fit_scene = int(fit_scene.sum())
    fit_scene_counts = final_scene_counts[:n_fit_scene]
    fit_scene_starts = np.r_[0, np.cumsum(fit_scene_counts[:-1], dtype=np.int64)]
    select_scene_counts = final_scene_counts[n_fit_scene:]
    select_scene_starts = np.r_[
        0, np.cumsum(select_scene_counts[:-1], dtype=np.int64)
    ]

    p_edges = make_scene_edges(final_scene_base[fit_scene], N_SCENE_BINS)
    q_edges = make_scene_edges(
        scene_logq[fit_scene], N_SCENE_BINS, forced_edges=()
    )
    fit_axes = [
        make_axis(
            "full_base_scene_prediction",
            final_scene_case[fit_scene],
            fit_scene_counts,
            final_scene_base[fit_scene],
            p_edges,
            (),
        ),
        make_axis(
            "log10_q_d2",
            final_scene_case[fit_scene],
            fit_scene_counts,
            scene_logq[fit_scene],
            q_edges,
            (),
        ),
    ]
    select_axes = [
        make_axis(
            "full_base_scene_prediction",
            final_scene_case[select_scene],
            select_scene_counts,
            final_scene_base[select_scene],
            p_edges,
            (),
        ),
        make_axis(
            "log10_q_d2",
            final_scene_case[select_scene],
            select_scene_counts,
            scene_logq[select_scene],
            q_edges,
            (),
        ),
    ]
    fit_prepared = prepare_objective(
        target[:n_fit_row], fit_scene_starts, fit_axes
    )

    candidate_rows: list[dict[str, Any]] = []
    candidate_curves: list[dict[str, Any]] = []
    raw_select_metrics, raw_select_curves = model_metrics(
        np.zeros(int(select_row.sum())),
        target[n_fit_row:],
        final_case[select_row],
        select_scene_starts,
        select_axes,
        target_scale,
    )
    raw_scene_global = global_scene_stat(
        np.zeros(int(select_row.sum())),
        target[n_fit_row:],
        select_scene_starts,
        final_scene_case[select_scene],
        target_scale,
    )
    for row in raw_select_curves:
        candidate_curves.append({"lambda": np.nan, "model": "Raw base", **row})

    candidate_dir = output / "selection_models"
    candidate_dir.mkdir(exist_ok=True)
    for strength in lambdas:
        print(f"fitting selection candidate lambda={strength:g}", flush=True)
        booster, fit_seconds = fit_booster(
            features[:n_fit_row],
            target[:n_fit_row],
            fit_scene_starts,
            fit_prepared,
            strength,
            n_jobs,
        )
        candidate_path = candidate_dir / f"lambda_{strength:g}.json"
        atomic_model(booster, candidate_path)
        select_prediction = booster.inplace_predict(
            features[n_fit_row:]
        ).astype(np.float64)
        metrics, curves = model_metrics(
            select_prediction,
            target[n_fit_row:],
            final_case[select_row],
            select_scene_starts,
            select_axes,
            target_scale,
        )
        scene_global = global_scene_stat(
            select_prediction,
            target[n_fit_row:],
            select_scene_starts,
            final_scene_case[select_scene],
            target_scale,
        )
        p_rms = metrics["scene"]["full_base_scene_prediction"]["curve_rms"]
        q_rms = metrics["scene"]["log10_q_d2"]["curve_rms"]
        p_raw = raw_select_metrics["scene"]["full_base_scene_prediction"]["curve_rms"]
        q_raw = raw_select_metrics["scene"]["log10_q_d2"]["curve_rms"]
        score = float(np.sqrt(0.5 * ((p_rms / p_raw) ** 2 + (q_rms / q_raw) ** 2)))
        pair_mse = float(metrics["pair"]["mse"])
        candidate_rows.append({
            "lambda": strength,
            "fit_seconds": fit_seconds,
            "model": str(candidate_path),
            "model_sha256": sha256(candidate_path),
            "pair_mse": pair_mse,
            "pair_mse_over_raw": pair_mse / float(raw_select_metrics["pair"]["mse"]),
            "pair_case_residual": metrics["pair"]["case_balanced_residual_mean"],
            "pair_case_sem": metrics["pair"]["case_residual_sem"],
            "p_curve_rms": p_rms,
            "q_curve_rms": q_rms,
            "normalized_scene_score": score,
            "global_scene_residual": scene_global["mean"],
            "global_scene_sem": scene_global["case_sem"],
        })
        for row in curves:
            candidate_curves.append({
                "lambda": strength,
                "model": f"lambda={strength:g}",
                **row,
            })
        del booster, select_prediction
        gc.collect()

    candidate_frame = pd.DataFrame(candidate_rows).sort_values("lambda")
    eligible = candidate_frame.loc[candidate_frame.pair_mse_over_raw <= 1.0]
    if eligible.empty:
        raise RuntimeError("no lambda preserves or improves validation pair MSE")
    selected_row = eligible.sort_values(
        ["normalized_scene_score", "lambda"], kind="stable"
    ).iloc[0]
    selected_lambda = float(selected_row["lambda"])
    print(f"selected lambda={selected_lambda:g}", flush=True)

    final_p_edges = make_scene_edges(final_scene_base, N_SCENE_BINS)
    final_q_edges = make_scene_edges(scene_logq, N_SCENE_BINS, forced_edges=())
    final_axes = [
        make_axis(
            "full_base_scene_prediction",
            final_scene_case,
            final_scene_counts,
            final_scene_base,
            final_p_edges,
            (),
        ),
        make_axis(
            "log10_q_d2",
            final_scene_case,
            final_scene_counts,
            scene_logq,
            final_q_edges,
            (),
        ),
    ]
    final_prepared = prepare_objective(target, final_scene_starts, final_axes)
    final_models: dict[str, Any] = {}
    final_curve_rows: list[dict[str, Any]] = []
    for name, strength in (("pair_only_q", 0.0), ("selected", selected_lambda)):
        booster, fit_seconds = fit_booster(
            features,
            target,
            final_scene_starts,
            final_prepared,
            strength,
            n_jobs,
        )
        path = output / f"correction_{name}.json"
        atomic_model(booster, path)
        prediction = booster.inplace_predict(features).astype(np.float64)
        metrics, curves = model_metrics(
            prediction,
            target,
            final_case,
            final_scene_starts,
            final_axes,
            target_scale,
        )
        scene_global = global_scene_stat(
            prediction,
            target,
            final_scene_starts,
            final_scene_case,
            target_scale,
        )
        final_models[name] = {
            "lambda": strength,
            "model": str(path),
            "model_sha256": sha256(path),
            "fit_seconds": fit_seconds,
            "diagnostic_in_sample_metrics": metrics,
            "global_scene_residual": scene_global,
        }
        for row in curves:
            final_curve_rows.append({"model": name, **row})
        del booster, prediction
        gc.collect()

    save_csv(output / "selection_metrics.csv", candidate_frame)
    save_csv(output / "selection_curves.csv", pd.DataFrame(candidate_curves))
    save_csv(output / "final_fit_curves.csv", pd.DataFrame(final_curve_rows))
    payload = {
        "schema_version": 1,
        "kind": "frozen-base pair-MSE plus pooled-across-cases scene-moment correction",
        "features": {
            "base": source_summary["features"]["base"],
            "correction": FEATURES,
        },
        "source": {
            "cache": str(source_cache),
            "metadata_sha256": sha256(source_cache / "metadata.json"),
            "full_neighbour_cache": str(full_cache),
            "full_neighbour_metadata_sha256": sha256(full_cache / "metadata.json"),
            "source_stack_summary": str(source_summary_path),
            "source_stack_summary_sha256": sha256(source_summary_path),
            "target_mean": target_mean,
            "target_scale": target_scale,
        },
        "objective": {
            "formula": (
                "ordinary pair MSE plus lambda times the mean of two normalized "
                "scene-curve MSEs; each scene curve averages case-bin mean summed "
                "sheared-pair residuals across cases before squaring"
            ),
            "scene_axes": [
                "sum frozen-base predictions over all deployed neighbours",
                f"log10({PROXY_DEFINITION}) over all deployed neighbours",
            ],
            "scene_bins_per_axis": N_SCENE_BINS,
            "pair_target": "labelled/sheared pair label minus frozen base prediction",
            "self_test": self_test,
            "tree_params": CORRECTION_PARAMS | {"n_jobs": n_jobs},
            "rounds": CORRECTION_TREES,
        },
        "selection": {
            "fit_cases": [FIT_CASE_MIN, FIT_CASE_MAX],
            "fit_rows": n_fit_row,
            "fit_scenes": n_fit_scene,
            "score_cases": [SELECT_CASE_MIN, SELECT_CASE_MAX],
            "score_rows": int(select_row.sum()),
            "score_scenes": int(select_scene.sum()),
            "lambda_candidates": lambdas,
            "eligibility": "validation pair MSE no worse than raw frozen base",
            "score": (
                "quadrature mean of P_s-curve RMS/raw RMS and Q_d2-curve RMS/raw RMS"
            ),
            "raw_pair_metrics": raw_select_metrics["pair"],
            "raw_scene": raw_select_metrics["scene"],
            "raw_global_scene_residual": raw_scene_global,
            "selected_lambda": selected_lambda,
            "selected_candidate": selected_row.to_dict(),
        },
        "final_refit": {
            "case_window": [FINAL_CASE_MIN, FINAL_CASE_MAX],
            "n_rows": int(len(final_index)),
            "n_scenes": int(len(final_scene_case)),
            "models": final_models,
            "note": (
                "lambda was frozen using c160--199 before both deployment controls "
                "were refit on all c100--199"
            ),
        },
        "deployment_stack": {
            "base_model": str(base_path),
            "base_sha256": sha256(base_path),
            "correction_model": final_models["selected"]["model"],
            "correction_sha256": final_models["selected"]["model_sha256"],
            "base_is_the_exact_frozen_model_used_for_correction_targets": True,
        },
        "audit": {
            "coherent_anchor_truth_opened": False,
            "constgold_opened": False,
            "selection_uses_only_half_shear_cases_160_199": True,
        },
    }
    atomic_json(summary_path, json_clean(payload))
    print(json.dumps(json_clean(payload), indent=2, sort_keys=True), flush=True)
    print("PAIR_POOLED_SCENE_MOMENT_TRAINING_DONE", flush=True)


if __name__ == "__main__":
    main()
