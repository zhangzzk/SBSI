#!/usr/bin/env python3
"""Shared definitions for half-shear-only tuning of the V2.2 correction."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import xgboost as xgb


TUNE_TRAIN_CASES = (40, 139)
EARLY_STOP_CASES = (140, 159)
SELECTION_CASES = (160, 199)
DEVELOPMENT_CASES = (0, 19)
FINAL_HALF_SHEAR_CASES = (20, 39)
N_CURVE_BINS = 20
EARLY_STOPPING_ROUNDS = 50


def _candidate(
    name: str,
    *,
    max_depth: int = 5,
    min_child_weight: float = 2000.0,
    reg_lambda: float = 10.0,
    learning_rate: float = 0.05,
    subsample: float = 0.8,
    colsample_bytree: float = 0.9,
    max_rounds: int = 600,
) -> dict[str, Any]:
    return {
        "name": name,
        "max_depth": max_depth,
        "min_child_weight": min_child_weight,
        "reg_lambda": reg_lambda,
        "learning_rate": learning_rate,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "max_rounds": max_rounds,
    }


# A compact deterministic search around the conservative V1 recipe.  The first
# entry is the exact old hyperparameter recipe, now with early stopping.  The
# remaining entries probe capacity, leaf support, shrinkage, and randomness;
# no entry was chosen using coherent-anchor or ConstGold outcomes.
CANDIDATES = (
    _candidate("baseline_d5_m2000_l10"),
    _candidate("shallow_d3_m2000_l10", max_depth=3),
    _candidate("shallow_d4_m2000_l10", max_depth=4),
    _candidate("deep_d6_m2000_l10", max_depth=6),
    _candidate("deep_d7_m2000_l30", max_depth=7, reg_lambda=30.0),
    _candidate("leaf_m500_l10", min_child_weight=500.0),
    _candidate("leaf_m1000_l10", min_child_weight=1000.0),
    _candidate("leaf_m5000_l10", min_child_weight=5000.0),
    _candidate("leaf_m10000_l30", min_child_weight=10000.0, reg_lambda=30.0),
    _candidate("ridge_l1", reg_lambda=1.0),
    _candidate("ridge_l3", reg_lambda=3.0),
    _candidate("ridge_l30", reg_lambda=30.0),
    _candidate("ridge_l100", reg_lambda=100.0),
    _candidate(
        "conservative_d4_m8000_l30",
        max_depth=4,
        min_child_weight=8000.0,
        reg_lambda=30.0,
    ),
    _candidate(
        "flexible_d6_m500_l3",
        max_depth=6,
        min_child_weight=500.0,
        reg_lambda=3.0,
    ),
    _candidate("full_sample", subsample=1.0, colsample_bytree=1.0),
    _candidate(
        "more_random",
        subsample=0.65,
        colsample_bytree=0.75,
    ),
    _candidate(
        "slow_eta003",
        learning_rate=0.03,
        max_rounds=900,
    ),
)


def candidate_by_id(candidate_id: int) -> dict[str, Any]:
    """Return a defensive copy of one predefined candidate."""
    if candidate_id < 0 or candidate_id >= len(CANDIDATES):
        raise ValueError(
            f"candidate id {candidate_id} outside 0--{len(CANDIDATES) - 1}"
        )
    return dict(CANDIDATES[candidate_id])


def xgb_params(config: dict[str, Any]) -> dict[str, Any]:
    """Translate a candidate into the fixed XGBoost training parameters."""
    return {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "device": os.environ.get("XGB_DEVICE", "cpu"),
        "n_jobs": int(os.environ.get("SLURM_CPUS_PER_TASK", "8")),
        "max_depth": int(config["max_depth"]),
        "min_child_weight": float(config["min_child_weight"]),
        "learning_rate": float(config["learning_rate"]),
        "subsample": float(config["subsample"]),
        "colsample_bytree": float(config["colsample_bytree"]),
        "gamma": 0.0,
        "reg_alpha": 0.0,
        "reg_lambda": float(config["reg_lambda"]),
        "max_bin": 256,
        "base_score": 0.0,
        "seed": 20260815,
    }


def select_case_rows(
    case: np.ndarray,
    official_train: np.ndarray,
    case_window: tuple[int, int],
    *,
    official_validation_only: bool,
) -> np.ndarray:
    """Select one complete rendered-case block without row leakage."""
    case_min, case_max = case_window
    mask = (np.asarray(case) >= case_min) & (np.asarray(case) <= case_max)
    if official_validation_only:
        mask &= ~np.asarray(official_train, dtype=bool)
    index = np.flatnonzero(mask).astype(np.int32)
    if len(index) == 0:
        raise RuntimeError(f"empty case selection {case_window}")
    observed = np.unique(np.asarray(case)[index]).astype(int)
    expected = np.arange(case_min, case_max + 1)
    if not np.array_equal(observed, expected):
        raise RuntimeError(
            f"case selection {case_window} does not cover every requested case"
        )
    return index


def prediction_edges(prediction: np.ndarray, n_bins: int = N_CURVE_BINS) -> np.ndarray:
    """Return strictly increasing label-free quantile edges."""
    values = np.asarray(prediction, dtype=np.float64)
    if values.ndim != 1 or len(values) < n_bins or not np.isfinite(values).all():
        raise ValueError("prediction values are not valid for quantile binning")
    edges = np.unique(np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1)))
    if len(edges) != n_bins + 1:
        raise RuntimeError("prediction quantiles contain tied edges")
    return edges


def physical_prediction(
    booster: xgb.Booster,
    features: np.ndarray,
    target_scale: float,
) -> np.ndarray:
    """Predict the additive pair correction in physical response units."""
    prediction = booster.inplace_predict(features).astype(np.float64)
    prediction *= float(target_scale)
    if not np.isfinite(prediction).all():
        raise RuntimeError("hyperparameter candidate returned a non-finite correction")
    return prediction


def rms(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("RMS requires a non-empty finite vector")
    return float(np.sqrt(np.mean(np.square(values))))
