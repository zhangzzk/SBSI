#!/usr/bin/env python3
"""Shared utilities for the frozen-V2.2 grouped scene-residual experiment.

The learned quantity is an additive physical pair correction to the deployed
V2.2 response.  The feature coordinate is deliberately label-free: the
original pair features, the frozen pair prediction, two physical pair ratios,
the frozen sum of pair predictions for the primary scene, and scene
multiplicity.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import xgboost as xgb


SOURCE_TAG = "lsst_r_extnbr_v22"
FIT_CASE_MIN = 40
CASE_MAX = 199
TRAIN_CASE_MAX = 159
INTERNAL_VALIDATION_CASE_MIN = 160
EXTERNAL_DEVELOPMENT_CASE_MAX = 19
EXTERNAL_FINAL_CASE_MIN = 20
EXTERNAL_FINAL_CASE_MAX = 39
VALIDATION_SCALE = 5.0
DEFAULT_FORCED_EDGES = (0.05, 0.1, 0.2)
DEFAULT_N_SCENE_BINS = 20

CONDITIONAL_FEATURES = [
    "Re_input_p_scaled",
    "Re_input_s_scaled",
    "r_input_p_scaled",
    "r_input_s_scaled",
    "sersic_n_input_p",
    "sersic_n_input_s",
    "distance_scaled",
    "v22_pair_prediction",
    "log10_flux_s_over_flux_p",
    "log10_Re_s_over_Re_p",
    "v22_scene_prediction",
    "log1p_n_pairs",
]


def strict_json(path: Path, payload: dict[str, Any]) -> None:
    """Write strict JSON once, refusing accidental replacement."""
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    with path.open("x", encoding="utf-8") as handle:
        json.dump(json_clean(payload), handle, indent=2, sort_keys=True,
                  allow_nan=False)
        handle.write("\n")


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


def load_source_metadata(cache: Path) -> dict[str, Any]:
    path = cache / "metadata.json"
    with path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata["source_tag"] != SOURCE_TAG:
        raise RuntimeError("source cache is not frozen V2.2")
    if metadata["case_window"] != [0, CASE_MAX]:
        raise RuntimeError("source-cache case window drifted")
    return metadata


def load_scene_metadata(scene_cache: Path) -> dict[str, Any]:
    path = scene_cache / "metadata.json"
    with path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata["source_tag"] != SOURCE_TAG:
        raise RuntimeError("scene cache is not frozen V2.2")
    if metadata["source_case_window"] != [0, CASE_MAX]:
        raise RuntimeError("scene-cache case window drifted")
    if metadata["feature_names"] != CONDITIONAL_FEATURES:
        raise RuntimeError("scene-cache feature definition drifted")
    return metadata


def scene_edges_from_metadata(metadata: dict[str, Any]) -> np.ndarray:
    """Restore the infinite endpoint sentinels used in strict JSON."""
    raw = list(metadata["scene_bin_definition"]["edges"])
    if len(raw) < 3 or raw[0] is not None or raw[-1] is not None:
        raise RuntimeError("scene metadata lacks strict-JSON infinite sentinels")
    if any(value is None for value in raw[1:-1]):
        raise RuntimeError("scene metadata has an internal null edge")
    edges = np.asarray([-np.inf, *raw[1:-1], np.inf], dtype=np.float64)
    if not np.all(np.diff(edges) > 0):
        raise RuntimeError("restored scene edges are not strictly increasing")
    return edges


def mmap_array(cache: Path, metadata: dict[str, Any], name: str) -> np.ndarray:
    return np.load(cache / metadata["arrays"][name], mmap_mode="r")


def make_scene_edges(
    scene_prediction: np.ndarray,
    n_quantile_bins: int = DEFAULT_N_SCENE_BINS,
    forced_edges: Iterable[float] = DEFAULT_FORCED_EDGES,
) -> np.ndarray:
    """Make label-independent scene quantile bins with physical landmarks."""
    prediction = np.asarray(scene_prediction, dtype=np.float64)
    prediction = prediction[np.isfinite(prediction)]
    if n_quantile_bins < 2 or len(prediction) < n_quantile_bins:
        raise ValueError("insufficient finite scenes for requested bins")
    quantiles = np.quantile(
        prediction, np.linspace(0.0, 1.0, n_quantile_bins + 1)
    )
    internal = list(np.unique(quantiles[1:-1]))
    if len(internal) != n_quantile_bins - 1:
        raise RuntimeError("scene prediction has tied quantile edges")
    for value in np.unique(np.asarray(tuple(forced_edges), dtype=np.float64)):
        if not np.isfinite(value):
            continue
        nearest = int(np.argmin(np.abs(np.asarray(internal) - value)))
        internal[nearest] = float(value)
        internal = sorted(set(internal))
        if len(internal) != n_quantile_bins - 1:
            raise RuntimeError("forced scene edge collided with a quantile edge")
    edges = np.asarray([-np.inf, *internal, np.inf], dtype=np.float64)
    if not np.all(np.diff(edges) > 0):
        raise RuntimeError("scene edges are not strictly increasing")
    return edges


def assign_scene_bins(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    edges = np.asarray(edges, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("scene coordinate contains non-finite values")
    bins = np.searchsorted(edges, values, side="right") - 1
    if np.any(bins < 0) or np.any(bins >= len(edges) - 1):
        raise RuntimeError("scene-bin assignment escaped edge support")
    return bins.astype(np.uint8)


def case_offsets(metadata: dict[str, Any]) -> np.ndarray:
    counts = np.asarray(
        [metadata["case_counts"][str(case)] for case in range(CASE_MAX + 1)],
        dtype=np.int64,
    )
    offsets = np.concatenate(([0], np.cumsum(counts)))
    if int(offsets[-1]) != int(metadata["n_rows"]):
        raise RuntimeError("case counts do not close to source-cache row count")
    return offsets


def conditional_matrix(
    x_scaled: np.ndarray,
    x_raw: np.ndarray,
    pair_prediction: np.ndarray,
    scene_prediction: np.ndarray,
    n_pairs: np.ndarray,
    index: np.ndarray | slice,
) -> np.ndarray:
    """Build the fixed pair-plus-scene feature coordinate in float32."""
    raw = np.asarray(x_raw[index], dtype=np.float32)
    scaled = np.asarray(x_scaled[index], dtype=np.float32)
    pair = np.asarray(pair_prediction[index], dtype=np.float32)
    scene = np.asarray(scene_prediction[index], dtype=np.float32)
    multiplicity = np.asarray(n_pairs[index], dtype=np.float32)
    if scaled.ndim != 2 or scaled.shape[1] != 7 or raw.shape != scaled.shape:
        raise ValueError("unexpected cached pair-feature shape")
    n_row = len(scaled)
    if pair.shape != (n_row,) or scene.shape != (n_row,) or multiplicity.shape != (n_row,):
        raise ValueError("pair/scene feature shape mismatch")
    if np.any(raw[:, :2] <= 0.0) or np.any(multiplicity <= 0.0):
        raise RuntimeError("supported pair has non-positive size or multiplicity")
    output = np.empty((n_row, len(CONDITIONAL_FEATURES)), dtype=np.float32)
    output[:, :7] = scaled
    output[:, 7] = pair
    output[:, 8] = -0.4 * (raw[:, 3] - raw[:, 2])
    output[:, 9] = np.log10(raw[:, 1] / raw[:, 0])
    output[:, 10] = scene
    output[:, 11] = np.log1p(multiplicity)
    if not np.isfinite(output).all():
        raise RuntimeError("non-finite conditional feature")
    return output


def grouped_loss_value(
    prediction: np.ndarray,
    target: np.ndarray,
    bin_index: np.ndarray,
    strength: float,
) -> float:
    """Return mean row MSE/2 plus equal-bin mean-residual MSE/2."""
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    bins = np.asarray(bin_index, dtype=np.int64)
    if prediction.shape != target.shape or bins.shape != prediction.shape:
        raise ValueError("grouped-loss array shape mismatch")
    n_bins = int(bins.max()) + 1
    count = np.bincount(bins, minlength=n_bins).astype(np.float64)
    if np.any(count <= 0):
        raise RuntimeError("grouped loss contains an empty scene bin")
    error = prediction - target
    mean = np.bincount(bins, weights=error, minlength=n_bins) / count
    return float(
        0.5 * np.mean(np.square(error))
        + 0.5 * float(strength) * np.mean(np.square(mean))
    )


def grouped_gradient_hessian(
    prediction: np.ndarray,
    target: np.ndarray,
    bin_index: np.ndarray,
    strength: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Gradient and block-curvature approximation for the grouped loss.

    The gradient is exact up to a harmless common factor of ``n_rows``.  The
    Hessian distributes each bin's off-diagonal block curvature over its rows.
    Consequently a leaf containing a whole bin has the exact Newton curvature,
    while XGBoost still receives a positive diagonal Hessian.
    """
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    bins = np.asarray(bin_index, dtype=np.int64)
    if prediction.shape != target.shape or bins.shape != prediction.shape:
        raise ValueError("grouped-objective array shape mismatch")
    if len(prediction) == 0 or strength < 0 or not np.isfinite(strength):
        raise ValueError("empty arrays or invalid grouped-loss strength")
    n_bins = int(bins.max()) + 1
    count = np.bincount(bins, minlength=n_bins).astype(np.float64)
    if np.any(count <= 0):
        raise RuntimeError("grouped objective contains an empty scene bin")
    error = prediction - target
    mean = np.bincount(bins, weights=error, minlength=n_bins) / count
    mean_bin_count = len(error) / n_bins
    weight = float(strength) * mean_bin_count / count
    gradient = error + weight[bins] * mean[bins]
    # This includes the bin block curvature relevant to a shared tree-leaf
    # update, rather than pretending the true off-diagonal Hessian is absent.
    hessian = 1.0 + weight[bins]
    return gradient.astype(np.float32), hessian.astype(np.float32)


def grouped_objective(bin_index: np.ndarray, strength: float):
    """Create an XGBoost custom objective for one immutable training matrix."""
    bins = np.asarray(bin_index, dtype=np.uint8)

    def objective(prediction: np.ndarray, matrix: xgb.DMatrix):
        target = matrix.get_label()
        if len(prediction) != len(bins):
            raise RuntimeError("custom-objective row order changed")
        return grouped_gradient_hessian(prediction, target, bins, strength)

    return objective


def model_params() -> dict[str, Any]:
    """Fixed conservative residual-tree recipe for every strength candidate."""
    return {
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "device": os.environ.get("XGB_DEVICE", "cpu"),
        "n_jobs": int(os.environ.get("SLURM_CPUS_PER_TASK", "8")),
        "max_depth": 5,
        "min_child_weight": 2000,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.9,
        "gamma": 0.0,
        "reg_alpha": 0.0,
        "reg_lambda": 10.0,
        "max_bin": 256,
        "base_score": 0.0,
        "seed": 20260815,
        "disable_default_eval_metric": 1,
    }


def physical_correction(
    booster: xgb.Booster,
    features: np.ndarray,
    target_scale: float,
) -> np.ndarray:
    correction = booster.inplace_predict(features).astype(np.float64)
    correction *= float(target_scale)
    if not np.isfinite(correction).all():
        raise RuntimeError("grouped residual model returned non-finite correction")
    return correction


def finite_stat(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"mean": np.nan, "case_sd": np.nan, "case_sem": np.nan,
                "n_cases": 0}
    sd = float(values.std(ddof=1)) if len(values) > 1 else np.nan
    return {
        "mean": float(values.mean()),
        "case_sd": sd,
        "case_sem": float(sd / np.sqrt(len(values))) if len(values) > 1 else np.nan,
        "n_cases": int(len(values)),
    }
