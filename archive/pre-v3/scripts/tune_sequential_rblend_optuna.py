#!/usr/bin/env python3
"""Tune a case-separated base BlendEMU plus scene-informed residual model.

Each Optuna trial trains two boosters in sequence:

1. a seven-feature pair model for the physical half-shear R_blend label;
2. a twelve-feature additive model for the remaining pair residual, informed
   by the first-stage pair prediction and its sum over the primary scene.

The two supported studies use the same data, search space, and seed.  The
``pair`` study maximizes the historical BlendEMU validation-R2 objective.  The
``pair_scene`` study adds a case-balanced cumulative-scene calibration term.
No coherent-anchor or ConstGold labels are read.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import optuna
import xgboost as xgb

from scripts.v22_grouped_rscene_common import (
    CONDITIONAL_FEATURES,
    case_offsets,
    load_scene_metadata,
    load_source_metadata,
    mmap_array,
    sha256,
)


BASE_FEATURES = [
    "Re_input_p_scaled",
    "Re_input_s_scaled",
    "r_input_p_scaled",
    "r_input_s_scaled",
    "sersic_n_input_p",
    "sersic_n_input_s",
    "distance_scaled",
]
BASE_TRAIN_CASES = (40, 99)
BASE_EARLY_CASES = (100, 119)
CORRECTION_TRAIN_CASES = (120, 159)
OPTUNA_VALIDATION_CASES = (160, 199)
EXTERNAL_CHECK_CASES = (0, 39)
N_SCENE_BINS = 20
MAX_BOOST_ROUNDS = 1000
EARLY_STOPPING_ROUNDS = 50
R2_GAP_WEIGHT = 0.6
SCENE_WEIGHT = 1.0
SEED = 20260815


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(json_clean(payload), handle, indent=2, sort_keys=True,
                  allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def atomic_model(booster: xgb.Booster, path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.json")
    booster.save_model(temporary)
    os.replace(temporary, path)


def json_clean(value: Any) -> Any:
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


def r2_score(target: np.ndarray, prediction: np.ndarray) -> float:
    target64 = np.asarray(target, dtype=np.float64)
    prediction64 = np.asarray(prediction, dtype=np.float64)
    residual_ss = float(np.sum(np.square(target64 - prediction64)))
    total_ss = float(np.sum(np.square(target64 - target64.mean())))
    if not total_ss > 0.0:
        raise RuntimeError("R2 target has zero variance")
    return 1.0 - residual_ss / total_ss


class CaseBlock:
    """One contiguous, complete range of rendered half-shear cases."""

    def __init__(
        self,
        *,
        name: str,
        case_window: tuple[int, int],
        source_meta: dict[str, Any],
        source_arrays: dict[str, np.ndarray],
        scene_arrays: dict[str, np.ndarray],
        need_raw: bool,
    ) -> None:
        self.name = name
        self.case_window = case_window
        offsets = case_offsets(source_meta)
        case_min, case_max = case_window
        start = int(offsets[case_min])
        stop = int(offsets[case_max + 1])
        self.row_slice = slice(start, stop)
        self.case = np.asarray(source_arrays["case"][self.row_slice])
        expected_cases = np.arange(case_min, case_max + 1)
        if not np.array_equal(np.unique(self.case), expected_cases):
            raise RuntimeError(f"{name}: incomplete case coverage")
        self.primary = np.asarray(
            source_arrays["input_index"][self.row_slice], dtype=np.int64
        )
        self.label = np.asarray(
            source_arrays["label"][self.row_slice], dtype=np.float32
        )
        self.x_scaled = np.asarray(
            source_arrays["x_scaled"][self.row_slice], dtype=np.float32
        )
        if self.x_scaled.shape != (len(self.label), len(BASE_FEATURES)):
            raise RuntimeError(f"{name}: unexpected base feature shape")
        self.x_raw = None
        if need_raw:
            self.x_raw = np.asarray(
                source_arrays["x_raw"][self.row_slice], dtype=np.float32
            )

        change = np.empty(len(self.primary), dtype=bool)
        change[0] = True
        change[1:] = (
            (self.case[1:] != self.case[:-1])
            | (self.primary[1:] != self.primary[:-1])
        )
        self.scene_starts = np.flatnonzero(change).astype(np.int64)
        self.scene_counts = np.diff(
            np.append(self.scene_starts, len(self.primary))
        ).astype(np.int32)
        if int(self.scene_counts.sum()) != len(self.primary):
            raise RuntimeError(f"{name}: scene counts do not close")
        self.scene_case = self.case[self.scene_starts].astype(np.int16)
        self.truth_scene = np.add.reduceat(
            self.label.astype(np.float64), self.scene_starts
        )
        self.fixed_scene_bin = np.asarray(
            scene_arrays["scene_bin"][self.row_slice][self.scene_starts],
            dtype=np.int16,
        )
        self.fixed_scene_coordinate = np.asarray(
            scene_arrays["scene_prediction"][self.row_slice][self.scene_starts],
            dtype=np.float64,
        )
        if (
            self.fixed_scene_bin.min() < 0
            or self.fixed_scene_bin.max() >= N_SCENE_BINS
        ):
            raise RuntimeError(f"{name}: invalid fixed scene bins")
        self._conditional = None

    @property
    def n_rows(self) -> int:
        return len(self.label)

    @property
    def n_scenes(self) -> int:
        return len(self.scene_starts)

    def base_matrix(self, target_mean: float, target_scale: float) -> xgb.DMatrix:
        target = ((self.label.astype(np.float64) - target_mean) / target_scale)
        return xgb.DMatrix(
            self.x_scaled,
            label=target.astype(np.float32),
            feature_names=BASE_FEATURES,
        )

    def prepare_conditional_static(self) -> None:
        if self.x_raw is None:
            raise RuntimeError(f"{self.name}: raw features were not loaded")
        raw = self.x_raw
        if np.any(raw[:, :2] <= 0.0):
            raise RuntimeError(f"{self.name}: non-positive galaxy size")
        output = np.empty(
            (self.n_rows, len(CONDITIONAL_FEATURES)), dtype=np.float32
        )
        output[:, :7] = self.x_scaled
        output[:, 7] = 0.0
        output[:, 8] = -0.4 * (raw[:, 3] - raw[:, 2])
        output[:, 9] = np.log10(raw[:, 1] / raw[:, 0])
        output[:, 10] = 0.0
        row_counts = np.repeat(self.scene_counts, self.scene_counts)
        if len(row_counts) != self.n_rows:
            raise RuntimeError(f"{self.name}: multiplicity expansion failed")
        output[:, 11] = np.log1p(row_counts.astype(np.float32))
        if not np.isfinite(output).all():
            raise RuntimeError(f"{self.name}: non-finite conditional feature")
        self._conditional = output

    def conditional_matrix(self, pair_prediction: np.ndarray) -> np.ndarray:
        if self._conditional is None:
            self.prepare_conditional_static()
        prediction = np.asarray(pair_prediction, dtype=np.float32)
        if prediction.shape != (self.n_rows,) or not np.isfinite(prediction).all():
            raise RuntimeError(f"{self.name}: invalid base prediction")
        scene_prediction = np.add.reduceat(
            prediction.astype(np.float64), self.scene_starts
        ).astype(np.float32)
        row_scene_prediction = np.repeat(scene_prediction, self.scene_counts)
        if len(row_scene_prediction) != self.n_rows:
            raise RuntimeError(f"{self.name}: scene prediction expansion failed")
        self._conditional[:, 7] = prediction
        self._conditional[:, 10] = row_scene_prediction
        return self._conditional

    def scene_curve(self, pair_prediction: np.ndarray) -> dict[str, Any]:
        prediction_scene = np.add.reduceat(
            np.asarray(pair_prediction, dtype=np.float64), self.scene_starts
        )
        residual = self.truth_scene - prediction_scene
        case_min, case_max = self.case_window
        n_cases = case_max - case_min + 1
        case_index = self.scene_case.astype(np.int64) - case_min
        flat_index = case_index * N_SCENE_BINS + self.fixed_scene_bin
        count = np.bincount(
            flat_index, minlength=n_cases * N_SCENE_BINS
        ).reshape(n_cases, N_SCENE_BINS)
        total = np.bincount(
            flat_index,
            weights=residual,
            minlength=n_cases * N_SCENE_BINS,
        ).reshape(n_cases, N_SCENE_BINS)
        if np.any(count == 0):
            where = np.argwhere(count == 0)[0]
            raise RuntimeError(
                f"{self.name}: empty case/scene bin {tuple(where)}"
            )
        case_bin_mean = total / count
        curve = case_bin_mean.mean(axis=0)
        case_mean = np.bincount(
            case_index, weights=residual, minlength=n_cases
        ) / np.bincount(case_index, minlength=n_cases)
        return {
            "curve": curve,
            "curve_rms": float(np.sqrt(np.mean(np.square(curve)))),
            "case_mean": case_mean,
            "case_balanced_mean": float(case_mean.mean()),
            "row_weighted_mean": float(residual.mean()),
        }


def xgb_common() -> dict[str, Any]:
    return {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "device": os.environ.get("XGB_DEVICE", "cuda"),
        "n_jobs": int(os.environ.get("SLURM_CPUS_PER_TASK", "8")),
        "booster": "gbtree",
        "verbosity": 0,
        "disable_default_eval_metric": 0,
        "max_bin": 256,
        "base_score": 0.0,
        "seed": SEED,
    }


def suggest_params(trial: optuna.Trial) -> dict[str, dict[str, Any]]:
    """Use the historical base space and a conservative residual-tree space."""
    base = {
        "subsample": trial.suggest_float("base_subsample", 0.3, 1.0),
        "colsample_bytree": trial.suggest_float(
            "base_colsample_bytree", 0.3, 1.0
        ),
        "learning_rate": trial.suggest_float(
            "base_learning_rate", 1.0e-3, 0.2, log=True
        ),
        "max_depth": trial.suggest_int("base_max_depth", 3, 15),
        "min_child_weight": trial.suggest_int(
            "base_min_child_weight", 1, 200
        ),
        "reg_lambda": trial.suggest_float(
            "base_reg_lambda", 1.0e-3, 10.0, log=True
        ),
        "reg_alpha": trial.suggest_float(
            "base_reg_alpha", 1.0e-3, 10.0, log=True
        ),
        "gamma": trial.suggest_float("base_gamma", 0.0, 5.0),
    }
    correction = {
        "subsample": trial.suggest_float("correction_subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float(
            "correction_colsample_bytree", 0.5, 1.0
        ),
        "learning_rate": trial.suggest_float(
            "correction_learning_rate", 3.0e-3, 0.12, log=True
        ),
        "max_depth": trial.suggest_int("correction_max_depth", 2, 8),
        "min_child_weight": trial.suggest_float(
            "correction_min_child_weight", 100.0, 10000.0, log=True
        ),
        "reg_lambda": trial.suggest_float(
            "correction_reg_lambda", 1.0e-2, 100.0, log=True
        ),
        "reg_alpha": trial.suggest_float(
            "correction_reg_alpha", 1.0e-3, 10.0, log=True
        ),
        "gamma": trial.suggest_float("correction_gamma", 0.0, 5.0),
    }
    return {"base": base, "correction": correction}


def params_from_flat(flat: dict[str, Any]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {"base": {}, "correction": {}}
    for name, value in flat.items():
        for stage in output:
            prefix = f"{stage}_"
            if name.startswith(prefix):
                output[stage][name[len(prefix):]] = value
                break
    expected = 8
    if any(len(stage_params) != expected for stage_params in output.values()):
        raise RuntimeError("stored Optuna parameters do not cover both stages")
    return output


def baseline_trial() -> dict[str, Any]:
    return {
        "base_subsample": 0.9474434509427369,
        "base_colsample_bytree": 0.9801141338727593,
        "base_learning_rate": 0.025150914825693486,
        "base_max_depth": 8,
        "base_min_child_weight": 181,
        "base_reg_lambda": 0.7037806884244237,
        "base_reg_alpha": 0.002305343603936417,
        "base_gamma": 3.998163258286566,
        "correction_subsample": 0.8,
        "correction_colsample_bytree": 0.9,
        "correction_learning_rate": 0.05,
        "correction_max_depth": 3,
        "correction_min_child_weight": 2000.0,
        "correction_reg_lambda": 10.0,
        "correction_reg_alpha": 0.001,
        "correction_gamma": 0.0,
    }


class SequentialExperiment:
    def __init__(self, source_cache: Path, scene_cache: Path) -> None:
        self.source_cache = source_cache
        self.scene_cache = scene_cache
        self.source_meta = load_source_metadata(source_cache)
        self.scene_meta = load_scene_metadata(scene_cache)
        if self.scene_meta["source_metadata_sha256"] != sha256(
            source_cache / "metadata.json"
        ):
            raise RuntimeError("scene cache does not match the source cache")
        source_arrays = {
            name: mmap_array(source_cache, self.source_meta, name)
            for name in ("case", "input_index", "label", "x_raw", "x_scaled")
        }
        scene_arrays = {
            name: mmap_array(scene_cache, self.scene_meta, name)
            for name in ("scene_bin", "scene_prediction")
        }
        self.source_arrays = source_arrays
        self.scene_arrays = scene_arrays
        standardization = self.source_meta["source_standardization"]
        self.target_mean = float(standardization["mean"])
        self.target_scale = float(standardization["std"])
        self.base_train = CaseBlock(
            name="base_train",
            case_window=BASE_TRAIN_CASES,
            source_meta=self.source_meta,
            source_arrays=source_arrays,
            scene_arrays=scene_arrays,
            need_raw=False,
        )
        self.base_early = CaseBlock(
            name="base_early",
            case_window=BASE_EARLY_CASES,
            source_meta=self.source_meta,
            source_arrays=source_arrays,
            scene_arrays=scene_arrays,
            need_raw=False,
        )
        self.correction_train = CaseBlock(
            name="correction_train",
            case_window=CORRECTION_TRAIN_CASES,
            source_meta=self.source_meta,
            source_arrays=source_arrays,
            scene_arrays=scene_arrays,
            need_raw=True,
        )
        self.validation = CaseBlock(
            name="optuna_validation",
            case_window=OPTUNA_VALIDATION_CASES,
            source_meta=self.source_meta,
            source_arrays=source_arrays,
            scene_arrays=scene_arrays,
            need_raw=True,
        )
        self.correction_train.prepare_conditional_static()
        self.validation.prepare_conditional_static()
        self.base_matrices = {
            "train": self.base_train.base_matrix(
                self.target_mean, self.target_scale
            ),
            "early": self.base_early.base_matrix(
                self.target_mean, self.target_scale
            ),
            "correction": self.correction_train.base_matrix(
                self.target_mean, self.target_scale
            ),
            "validation": self.validation.base_matrix(
                self.target_mean, self.target_scale
            ),
        }
        print(
            "Loaded case-separated blocks: "
            f"base={self.base_train.n_rows:,}, "
            f"base_early={self.base_early.n_rows:,}, "
            f"correction={self.correction_train.n_rows:,}, "
            f"validation={self.validation.n_rows:,}",
            flush=True,
        )

    def physical_base_prediction(
        self, booster: xgb.Booster, matrix: xgb.DMatrix
    ) -> np.ndarray:
        standardized = booster.predict(matrix).astype(np.float64)
        prediction = standardized * self.target_scale + self.target_mean
        if not np.isfinite(prediction).all():
            raise RuntimeError("base booster returned non-finite predictions")
        return prediction

    def fit_pipeline(
        self,
        stage_params: dict[str, dict[str, Any]],
        *,
        keep_models: bool,
    ) -> tuple[dict[str, Any], tuple[xgb.Booster, xgb.Booster] | None]:
        started = time.time()
        base_params = xgb_common() | stage_params["base"]
        base_history: dict[str, dict[str, list[float]]] = {}
        base_full = xgb.train(
            base_params,
            self.base_matrices["train"],
            num_boost_round=MAX_BOOST_ROUNDS,
            evals=[
                (self.base_matrices["train"], "train"),
                (self.base_matrices["early"], "early"),
            ],
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            evals_result=base_history,
            verbose_eval=False,
        )
        base_rounds = int(base_full.best_iteration) + 1
        base_best_score = float(base_full.best_score)
        base = base_full[:base_rounds]
        del base_full

        base_prediction_train = self.physical_base_prediction(
            base, self.base_matrices["correction"]
        )
        base_prediction_validation = self.physical_base_prediction(
            base, self.base_matrices["validation"]
        )
        correction_target_train = (
            self.correction_train.label.astype(np.float64)
            - base_prediction_train
        ) / self.target_scale
        correction_target_validation = (
            self.validation.label.astype(np.float64)
            - base_prediction_validation
        ) / self.target_scale
        correction_features_train = self.correction_train.conditional_matrix(
            base_prediction_train
        )
        correction_features_validation = self.validation.conditional_matrix(
            base_prediction_validation
        )
        dcorrection_train = xgb.DMatrix(
            correction_features_train,
            label=correction_target_train.astype(np.float32),
            feature_names=CONDITIONAL_FEATURES,
        )
        dcorrection_validation = xgb.DMatrix(
            correction_features_validation,
            label=correction_target_validation.astype(np.float32),
            feature_names=CONDITIONAL_FEATURES,
        )
        correction_params = xgb_common() | stage_params["correction"]
        correction_history: dict[str, dict[str, list[float]]] = {}
        correction_full = xgb.train(
            correction_params,
            dcorrection_train,
            num_boost_round=MAX_BOOST_ROUNDS,
            evals=[
                (dcorrection_train, "train"),
                (dcorrection_validation, "validation"),
            ],
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            evals_result=correction_history,
            verbose_eval=False,
        )
        correction_rounds = int(correction_full.best_iteration) + 1
        correction_best_score = float(correction_full.best_score)
        correction = correction_full[:correction_rounds]
        del correction_full
        correction_train = (
            correction.predict(dcorrection_train).astype(np.float64)
            * self.target_scale
        )
        correction_validation = (
            correction.predict(dcorrection_validation).astype(np.float64)
            * self.target_scale
        )
        combined_train = base_prediction_train + correction_train
        combined_validation = base_prediction_validation + correction_validation
        train_r2 = r2_score(self.correction_train.label, combined_train)
        validation_r2 = r2_score(self.validation.label, combined_validation)
        pair_score = validation_r2 - R2_GAP_WEIGHT * abs(
            train_r2 - validation_r2
        )
        scene = self.validation.scene_curve(combined_validation)
        scene_curve_rms_standardized = (
            float(scene["curve_rms"]) / self.target_scale
        )
        pair_scene_score = pair_score - SCENE_WEIGHT * scene_curve_rms_standardized
        validation_residual = (
            self.validation.label.astype(np.float64) - combined_validation
        )
        train_residual = (
            self.correction_train.label.astype(np.float64) - combined_train
        )
        metrics = {
            "base_rounds": base_rounds,
            "correction_rounds": correction_rounds,
            "base_best_early_rmse_standardized": base_best_score,
            "correction_best_validation_rmse_standardized": correction_best_score,
            "train_r2": train_r2,
            "validation_r2": validation_r2,
            "r2_train_validation_gap_abs": abs(train_r2 - validation_r2),
            "historical_pair_score": pair_score,
            "validation_scene_curve": scene["curve"],
            "validation_scene_curve_rms": scene["curve_rms"],
            "validation_scene_curve_rms_standardized": scene_curve_rms_standardized,
            "pair_scene_score": pair_scene_score,
            "train_pair_residual_mean": float(train_residual.mean()),
            "validation_pair_residual_mean": float(validation_residual.mean()),
            "validation_pair_mse": float(
                np.mean(np.square(validation_residual))
            ),
            "validation_scene_case_balanced_mean": scene["case_balanced_mean"],
            "fit_seconds": time.time() - started,
        }
        del (
            dcorrection_train,
            dcorrection_validation,
            base_prediction_train,
            base_prediction_validation,
            correction_target_train,
            correction_target_validation,
            correction_train,
            correction_validation,
            combined_train,
            combined_validation,
        )
        gc.collect()
        models = (base, correction) if keep_models else None
        if not keep_models:
            del base, correction
            gc.collect()
        return metrics, models

    def external_evaluation(
        self, base: xgb.Booster, correction: xgb.Booster
    ) -> dict[str, Any]:
        external = CaseBlock(
            name="external_check",
            case_window=EXTERNAL_CHECK_CASES,
            source_meta=self.source_meta,
            source_arrays=self.source_arrays,
            scene_arrays=self.scene_arrays,
            need_raw=True,
        )
        base_matrix = external.base_matrix(self.target_mean, self.target_scale)
        base_prediction = self.physical_base_prediction(base, base_matrix)
        conditional = external.conditional_matrix(base_prediction)
        correction_prediction = (
            correction.inplace_predict(conditional).astype(np.float64)
            * self.target_scale
        )
        combined = base_prediction + correction_prediction
        output = {
            "base_only": evaluation_metrics(external, base_prediction, self.target_scale),
            "combined": evaluation_metrics(external, combined, self.target_scale),
        }
        del external, base_matrix, base_prediction, conditional, correction_prediction
        gc.collect()
        return output


def evaluation_metrics(
    block: CaseBlock, prediction: np.ndarray, target_scale: float
) -> dict[str, Any]:
    residual = block.label.astype(np.float64) - np.asarray(
        prediction, dtype=np.float64
    )
    scene = block.scene_curve(prediction)
    case_min, case_max = block.case_window
    pair_case_index = block.case.astype(np.int64) - case_min
    n_cases = case_max - case_min + 1
    pair_case_mean = np.bincount(
        pair_case_index, weights=residual, minlength=n_cases
    ) / np.bincount(pair_case_index, minlength=n_cases)
    return {
        "n_rows": block.n_rows,
        "n_scenes": block.n_scenes,
        "r2": r2_score(block.label, prediction),
        "pair_residual_row_weighted_mean": float(residual.mean()),
        "pair_residual_case_balanced_mean": float(pair_case_mean.mean()),
        "pair_residual_case_sem": float(
            pair_case_mean.std(ddof=1) / np.sqrt(n_cases)
        ),
        "pair_mse": float(np.mean(np.square(residual))),
        "scene_residual_case_balanced_mean": scene["case_balanced_mean"],
        "scene_residual_case_sem": float(
            np.asarray(scene["case_mean"]).std(ddof=1) / np.sqrt(n_cases)
        ),
        "scene_curve": scene["curve"],
        "scene_curve_rms": scene["curve_rms"],
        "scene_curve_rms_standardized": scene["curve_rms"] / target_scale,
    }


def study_completed_trials(study: optuna.Study) -> int:
    return sum(
        trial.state == optuna.trial.TrialState.COMPLETE
        for trial in study.trials
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-cache", required=True)
    parser.add_argument("--scene-cache", required=True)
    parser.add_argument(
        "--objective", choices=("pair", "pair_scene"), required=True
    )
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if args.n_trials <= 0:
        raise ValueError("n-trials must be positive")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        print(f"Study already complete: {summary_path}", flush=True)
        return

    experiment = SequentialExperiment(
        Path(args.source_cache).resolve(), Path(args.scene_cache).resolve()
    )
    database = output_dir / "study.sqlite3"
    sampler = optuna.samplers.TPESampler(
        seed=SEED,
        n_startup_trials=10,
        multivariate=True,
    )
    study_name = f"sequential_rblend_{args.objective}_optuna50_v1"
    study = optuna.create_study(
        study_name=study_name,
        storage=f"sqlite:///{database}",
        direction="maximize",
        sampler=sampler,
        load_if_exists=True,
    )
    if len(study.trials) == 0:
        study.enqueue_trial(baseline_trial())

    def objective(trial: optuna.Trial) -> float:
        stage_params = suggest_params(trial)
        metrics, _ = experiment.fit_pipeline(stage_params, keep_models=False)
        for key, value in metrics.items():
            if key != "validation_scene_curve":
                trial.set_user_attr(key, json_clean(value))
        score_name = (
            "historical_pair_score"
            if args.objective == "pair"
            else "pair_scene_score"
        )
        score = float(metrics[score_name])
        print(
            f"TRIAL_DONE objective={args.objective} trial={trial.number} "
            f"score={score:+.8f} pair={metrics['historical_pair_score']:+.8f} "
            f"scene_std={metrics['validation_scene_curve_rms_standardized']:.8f} "
            f"trees={metrics['base_rounds']}+{metrics['correction_rounds']} "
            f"seconds={metrics['fit_seconds']:.1f}",
            flush=True,
        )
        return score

    remaining = args.n_trials - study_completed_trials(study)
    if remaining > 0:
        study.optimize(
            objective,
            n_trials=remaining,
            gc_after_trial=True,
            show_progress_bar=False,
        )
    if study_completed_trials(study) < args.n_trials:
        raise RuntimeError("study ended before the requested completed-trial count")

    trial_table = study.trials_dataframe(
        attrs=("number", "value", "params", "user_attrs", "state")
    )
    trial_table.to_csv(output_dir / "trials.csv", index=False)
    best = study.best_trial
    best_stage_params = params_from_flat(best.params)
    best_metrics, models = experiment.fit_pipeline(
        best_stage_params, keep_models=True
    )
    if models is None:
        raise RuntimeError("best pipeline was not retained")
    base, correction = models
    external = experiment.external_evaluation(base, correction)
    base_model_path = output_dir / "best_base.json"
    correction_model_path = output_dir / "best_correction.json"
    atomic_model(base, base_model_path)
    atomic_model(correction, correction_model_path)

    payload = {
        "schema_version": 1,
        "kind": "case-separated sequential base plus scene-informed residual Optuna pilot",
        "objective_variant": args.objective,
        "objective": {
            "historical_pair_score": (
                "validation_R2 - 0.6*abs(correction_train_R2-validation_R2)"
            ),
            "scene_term": (
                None
                if args.objective == "pair"
                else (
                    "case-balanced RMS over 20 fixed bins of exact cumulative "
                    "validation-scene (truth-prediction), divided by the frozen "
                    "V2.2 target standard deviation"
                )
            ),
            "optimized_score": (
                "historical_pair_score"
                if args.objective == "pair"
                else "historical_pair_score - 1.0*scene_curve_RMS_standardized"
            ),
        },
        "study": {
            "name": study.study_name,
            "database": str(database),
            "requested_completed_trials": args.n_trials,
            "completed_trials": study_completed_trials(study),
            "best_trial": best.number,
            "best_value": best.value,
            "sampler": "Optuna TPESampler",
            "sampler_seed": SEED,
            "startup_trials": 10,
        },
        "splits": {
            "base_train_cases": BASE_TRAIN_CASES,
            "base_early_stopping_cases": BASE_EARLY_CASES,
            "correction_train_cases": CORRECTION_TRAIN_CASES,
            "optuna_validation_cases": OPTUNA_VALIDATION_CASES,
            "external_post_selection_check_cases": EXTERNAL_CHECK_CASES,
            "row_counts": {
                "base_train": experiment.base_train.n_rows,
                "base_early_stopping": experiment.base_early.n_rows,
                "correction_train": experiment.correction_train.n_rows,
                "optuna_validation": experiment.validation.n_rows,
            },
        },
        "features": {
            "base": BASE_FEATURES,
            "correction": CONDITIONAL_FEATURES,
            "correction_scene_coordinate": (
                "sum of this trial's first-stage pair predictions per primary"
            ),
            "scene_selection_bins": (
                "fixed label-free V2.2 scene-prediction bins from scene cache"
            ),
        },
        "best_params": best_stage_params,
        "best_internal_metrics": best_metrics,
        "external_post_selection_metrics": external,
        "models": {
            "base": str(base_model_path),
            "base_sha256": sha256(base_model_path),
            "correction": str(correction_model_path),
            "correction_sha256": sha256(correction_model_path),
        },
        "source": {
            "cache": str(experiment.source_cache),
            "metadata_sha256": sha256(experiment.source_cache / "metadata.json"),
            "scene_cache": str(experiment.scene_cache),
            "scene_metadata_sha256": sha256(
                experiment.scene_cache / "metadata.json"
            ),
            "target_mean": experiment.target_mean,
            "target_scale": experiment.target_scale,
        },
        "protocol": {
            "all_stage_splits_are_case_disjoint": True,
            "correction_training_residuals_are_out_of_case_for_base": True,
            "external_cases_used_by_optuna": False,
            "external_cases_opened_only_after_best_trial_frozen": True,
            "coherent_anchor_truth_opened": False,
            "constgold_opened": False,
        },
    }
    atomic_json(summary_path, payload)
    print(json.dumps(json_clean(payload), indent=2, sort_keys=True))
    print(
        f"SEQUENTIAL_RBLEND_OPTUNA_DONE objective={args.objective} "
        f"trials={study_completed_trials(study)} best={best.number}",
        flush=True,
    )


if __name__ == "__main__":
    main()
