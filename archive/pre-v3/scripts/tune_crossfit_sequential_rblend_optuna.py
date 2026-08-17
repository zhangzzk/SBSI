#!/usr/bin/env python3
"""Five-fold all-case tuning of a sequential R_blend base and correction.

All 200 rendered cases rotate through three disjoint roles.  In each rotation,
80 cases fit the base, 80 different cases fit the correction on base-out-of-
fold residuals, and the remaining 40 cases receive a prediction from both
stages without either stage seeing their labels.  Concatenating the five
scored folds gives one clean prediction for every pair and complete scene.
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

from scripts.tune_sequential_rblend_optuna import (
    BASE_FEATURES,
    CONDITIONAL_FEATURES,
    R2_GAP_WEIGHT,
    SCENE_WEIGHT,
    SEED,
    atomic_json,
    atomic_model,
    baseline_trial,
    json_clean,
    r2_score,
    xgb_common,
)
from scripts.v22_grouped_rscene_common import (
    case_offsets,
    load_scene_metadata,
    load_source_metadata,
    mmap_array,
    sha256,
)


N_CASES = 200
N_FOLDS = 5
N_SCENE_BINS = 20
ALL_CASES = (0, N_CASES - 1)


def balanced_case_folds() -> np.ndarray:
    """Return a reproducible balanced assignment of 40 cases to each fold."""
    generator = np.random.default_rng(SEED)
    shuffled = generator.permutation(N_CASES)
    output = np.empty(N_CASES, dtype=np.int8)
    output[shuffled] = np.repeat(np.arange(N_FOLDS), N_CASES // N_FOLDS)
    if not np.array_equal(
        np.bincount(output, minlength=N_FOLDS), np.repeat(40, N_FOLDS)
    ):
        raise RuntimeError("case-fold assignment is not balanced")
    return output


def suggest_crossfit_params(
    trial: optuna.Trial,
) -> dict[str, Any]:
    """Search around the established base and residual recipes.

    The historical BlendEMU objective is retained exactly, while the pilot
    range avoids depth/learning-rate extremes that are impractical when every
    trial contains ten fits over all 200 cases.
    """
    base = {
        "subsample": trial.suggest_float("base_subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float(
            "base_colsample_bytree", 0.5, 1.0
        ),
        "learning_rate": trial.suggest_float(
            "base_learning_rate", 5.0e-3, 0.1, log=True
        ),
        "max_depth": trial.suggest_int("base_max_depth", 5, 11),
        "min_child_weight": trial.suggest_int(
            "base_min_child_weight", 20, 500, log=True
        ),
        "reg_lambda": trial.suggest_float(
            "base_reg_lambda", 1.0e-2, 10.0, log=True
        ),
        "reg_alpha": trial.suggest_float(
            "base_reg_alpha", 1.0e-3, 3.0, log=True
        ),
        "gamma": trial.suggest_float("base_gamma", 0.0, 5.0),
    }
    correction = {
        "subsample": trial.suggest_float("correction_subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float(
            "correction_colsample_bytree", 0.6, 1.0
        ),
        "learning_rate": trial.suggest_float(
            "correction_learning_rate", 5.0e-3, 0.1, log=True
        ),
        "max_depth": trial.suggest_int("correction_max_depth", 2, 6),
        "min_child_weight": trial.suggest_float(
            "correction_min_child_weight", 500.0, 10000.0, log=True
        ),
        "reg_lambda": trial.suggest_float(
            "correction_reg_lambda", 0.1, 100.0, log=True
        ),
        "reg_alpha": trial.suggest_float(
            "correction_reg_alpha", 1.0e-3, 3.0, log=True
        ),
        "gamma": trial.suggest_float("correction_gamma", 0.0, 3.0),
    }
    return {
        "base": base,
        "correction": correction,
        "base_rounds": trial.suggest_int(
            "base_num_boost_round", 100, 600
        ),
        "correction_rounds": trial.suggest_int(
            "correction_num_boost_round", 100, 600
        ),
    }


def baseline_crossfit_trial() -> dict[str, Any]:
    output = baseline_trial()
    output["base_num_boost_round"] = 271
    output["correction_num_boost_round"] = 283
    return output


def crossfit_params_from_flat(flat: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {
        "base": {},
        "correction": {},
        "base_rounds": int(flat["base_num_boost_round"]),
        "correction_rounds": int(flat["correction_num_boost_round"]),
    }
    for name, value in flat.items():
        if name.endswith("_num_boost_round"):
            continue
        for stage in ("base", "correction"):
            prefix = f"{stage}_"
            if name.startswith(prefix):
                output[stage][name[len(prefix):]] = value
                break
    if len(output["base"]) != 8 or len(output["correction"]) != 8:
        raise RuntimeError("stored Optuna parameters do not cover both stages")
    return output


class CrossfitExperiment:
    def __init__(self, source_cache: Path, scene_cache: Path) -> None:
        self.source_cache = source_cache
        self.scene_cache = scene_cache
        self.source_meta = load_source_metadata(source_cache)
        self.scene_meta = load_scene_metadata(scene_cache)
        if self.scene_meta["source_metadata_sha256"] != sha256(
            source_cache / "metadata.json"
        ):
            raise RuntimeError("scene cache does not match source cache")
        self.case = mmap_array(source_cache, self.source_meta, "case")
        self.primary = mmap_array(source_cache, self.source_meta, "input_index")
        self.label = mmap_array(source_cache, self.source_meta, "label")
        self.x_scaled = mmap_array(source_cache, self.source_meta, "x_scaled")
        self.x_raw = mmap_array(source_cache, self.source_meta, "x_raw")
        self.frozen_pair_prediction = mmap_array(
            source_cache, self.source_meta, "v22_prediction"
        )
        self.scene_bin_rows = mmap_array(scene_cache, self.scene_meta, "scene_bin")
        self.scene_coordinate_rows = mmap_array(
            scene_cache, self.scene_meta, "scene_prediction"
        )
        self.n_rows = int(self.source_meta["n_rows"])
        if len(self.case) != self.n_rows:
            raise RuntimeError("source-cache row count drifted")
        if not np.array_equal(np.unique(self.case), np.arange(N_CASES)):
            raise RuntimeError("source cache does not contain cases 0--199")
        standardization = self.source_meta["source_standardization"]
        self.target_mean = float(standardization["mean"])
        self.target_scale = float(standardization["std"])
        self.case_fold = balanced_case_folds()
        self.row_fold = self.case_fold[np.asarray(self.case, dtype=np.int64)]
        self.fold_index = tuple(
            np.flatnonzero(self.row_fold == fold).astype(np.int32)
            for fold in range(N_FOLDS)
        )
        if sum(len(index) for index in self.fold_index) != self.n_rows:
            raise RuntimeError("row folds do not close")
        self._build_scene_index()
        print(
            "Loaded all 200 cases: "
            f"rows={self.n_rows:,}, scenes={self.n_scenes:,}; "
            f"fold rows={[len(index) for index in self.fold_index]}",
            flush=True,
        )

    def _build_scene_index(self) -> None:
        case = np.asarray(self.case)
        primary = np.asarray(self.primary)
        change = np.empty(self.n_rows, dtype=bool)
        change[0] = True
        change[1:] = (
            (case[1:] != case[:-1]) | (primary[1:] != primary[:-1])
        )
        self.scene_starts = np.flatnonzero(change).astype(np.int64)
        del change
        self.scene_counts = np.diff(
            np.append(self.scene_starts, self.n_rows)
        ).astype(np.int32)
        self.n_scenes = len(self.scene_starts)
        self.scene_case = case[self.scene_starts].astype(np.int16)
        self.scene_bin = np.asarray(
            self.scene_bin_rows[self.scene_starts], dtype=np.int16
        )
        self.scene_coordinate = np.asarray(
            self.scene_coordinate_rows[self.scene_starts], dtype=np.float64
        )
        self.truth_scene = np.add.reduceat(
            np.asarray(self.label, dtype=np.float64), self.scene_starts
        )
        if (
            self.scene_bin.min() < 0
            or self.scene_bin.max() >= N_SCENE_BINS
        ):
            raise RuntimeError("invalid fixed scene bins")

    def split_indices(
        self, outer_fold: int
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        list[int],
        list[int],
    ]:
        correction_folds = [
            (outer_fold + 1) % N_FOLDS,
            (outer_fold + 2) % N_FOLDS,
        ]
        base_folds = [
            (outer_fold + 3) % N_FOLDS,
            (outer_fold + 4) % N_FOLDS,
        ]
        if sorted([outer_fold, *correction_folds, *base_folds]) != list(
            range(N_FOLDS)
        ):
            raise RuntimeError("rotation roles are not disjoint and exhaustive")
        base_index = np.sort(np.concatenate([
            self.fold_index[fold] for fold in base_folds
        ]))
        correction_index = np.sort(np.concatenate([
            self.fold_index[fold] for fold in correction_folds
        ]))
        outer_index = self.fold_index[outer_fold]
        return (
            base_index,
            correction_index,
            outer_index,
            base_folds,
            correction_folds,
        )

    def base_matrix(self, index: np.ndarray) -> xgb.DMatrix:
        features = np.asarray(self.x_scaled[index], dtype=np.float32)
        target = (
            np.asarray(self.label[index], dtype=np.float64) - self.target_mean
        ) / self.target_scale
        matrix = xgb.DMatrix(
            features,
            label=target.astype(np.float32),
            feature_names=BASE_FEATURES,
        )
        del features, target
        return matrix

    def physical_base_prediction(
        self, booster: xgb.Booster, matrix: xgb.DMatrix
    ) -> np.ndarray:
        prediction = booster.predict(matrix).astype(np.float64)
        prediction = prediction * self.target_scale + self.target_mean
        if not np.isfinite(prediction).all():
            raise RuntimeError("base model returned non-finite prediction")
        return prediction

    def conditional_features(
        self, index: np.ndarray, pair_prediction: np.ndarray
    ) -> np.ndarray:
        """Build pair-plus-scene features for a union of complete cases."""
        prediction = np.asarray(pair_prediction, dtype=np.float32)
        if prediction.shape != (len(index),):
            raise RuntimeError("conditional pair-prediction length mismatch")
        case = np.asarray(self.case[index])
        primary = np.asarray(self.primary[index], dtype=np.int64)
        change = np.empty(len(index), dtype=bool)
        change[0] = True
        change[1:] = (
            (case[1:] != case[:-1]) | (primary[1:] != primary[:-1])
        )
        starts = np.flatnonzero(change)
        counts = np.diff(np.append(starts, len(index))).astype(np.int32)
        scene_prediction = np.add.reduceat(
            prediction.astype(np.float64), starts
        ).astype(np.float32)
        row_scene_prediction = np.repeat(scene_prediction, counts)
        row_counts = np.repeat(counts, counts)
        if len(row_scene_prediction) != len(index) or len(row_counts) != len(index):
            raise RuntimeError("conditional scene expansion failed")
        scaled = np.asarray(self.x_scaled[index], dtype=np.float32)
        raw = np.asarray(self.x_raw[index], dtype=np.float32)
        if np.any(raw[:, :2] <= 0.0):
            raise RuntimeError("non-positive galaxy size in supported rows")
        output = np.empty(
            (len(index), len(CONDITIONAL_FEATURES)), dtype=np.float32
        )
        output[:, :7] = scaled
        output[:, 7] = prediction
        output[:, 8] = -0.4 * (raw[:, 3] - raw[:, 2])
        output[:, 9] = np.log10(raw[:, 1] / raw[:, 0])
        output[:, 10] = row_scene_prediction
        output[:, 11] = np.log1p(row_counts.astype(np.float32))
        del (
            case,
            primary,
            change,
            starts,
            counts,
            scene_prediction,
            row_scene_prediction,
            row_counts,
            scaled,
            raw,
        )
        if not np.isfinite(output).all():
            raise RuntimeError("conditional feature matrix is non-finite")
        return output

    def correction_matrix(
        self,
        index: np.ndarray,
        base_prediction: np.ndarray,
    ) -> xgb.DMatrix:
        features = self.conditional_features(index, base_prediction)
        target = (
            np.asarray(self.label[index], dtype=np.float64) - base_prediction
        ) / self.target_scale
        matrix = xgb.DMatrix(
            features,
            label=target.astype(np.float32),
            feature_names=CONDITIONAL_FEATURES,
        )
        del features, target
        return matrix

    def scene_metrics(self, pair_prediction: np.ndarray) -> dict[str, Any]:
        prediction_scene = np.add.reduceat(
            np.asarray(pair_prediction, dtype=np.float64), self.scene_starts
        )
        residual = self.truth_scene - prediction_scene
        flat_index = (
            self.scene_case.astype(np.int64) * N_SCENE_BINS + self.scene_bin
        )
        count = np.bincount(
            flat_index, minlength=N_CASES * N_SCENE_BINS
        ).reshape(N_CASES, N_SCENE_BINS)
        total = np.bincount(
            flat_index,
            weights=residual,
            minlength=N_CASES * N_SCENE_BINS,
        ).reshape(N_CASES, N_SCENE_BINS)
        if np.any(count == 0):
            raise RuntimeError("at least one case lacks a fixed scene bin")
        case_bin_mean = total / count
        curve = case_bin_mean.mean(axis=0)
        scene_case_count = np.bincount(self.scene_case, minlength=N_CASES)
        scene_case_mean = np.bincount(
            self.scene_case, weights=residual, minlength=N_CASES
        ) / scene_case_count
        output: dict[str, Any] = {
            "curve": curve,
            "curve_rms": float(np.sqrt(np.mean(np.square(curve)))),
            "case_balanced_mean": float(scene_case_mean.mean()),
            "case_sem": float(
                scene_case_mean.std(ddof=1) / np.sqrt(N_CASES)
            ),
            "row_weighted_mean": float(residual.mean()),
        }
        for name, mask in {
            "tail_gt_0p1": self.scene_coordinate > 0.1,
            "tail_gt_0p2": self.scene_coordinate > 0.2,
        }.items():
            local_case = self.scene_case[mask]
            local_residual = residual[mask]
            count_by_case = np.bincount(local_case, minlength=N_CASES)
            total_by_case = np.bincount(
                local_case, weights=local_residual, minlength=N_CASES
            )
            present = count_by_case > 0
            case_mean = total_by_case[present] / count_by_case[present]
            output[name] = {
                "n_scenes": int(mask.sum()),
                "case_balanced_mean": float(case_mean.mean()),
                "case_sem": float(
                    case_mean.std(ddof=1) / np.sqrt(len(case_mean))
                ),
            }
        return output

    def pair_metrics(self, prediction: np.ndarray) -> dict[str, Any]:
        residual = np.asarray(self.label, dtype=np.float64) - np.asarray(
            prediction, dtype=np.float64
        )
        case = np.asarray(self.case, dtype=np.int64)
        count = np.bincount(case, minlength=N_CASES)
        case_mean = np.bincount(
            case, weights=residual, minlength=N_CASES
        ) / count
        return {
            "r2": r2_score(self.label, prediction),
            "mse": float(np.mean(np.square(residual))),
            "row_weighted_residual_mean": float(residual.mean()),
            "case_balanced_residual_mean": float(case_mean.mean()),
            "case_residual_sem": float(
                case_mean.std(ddof=1) / np.sqrt(N_CASES)
            ),
        }

    def fit_crossfit(
        self,
        stage_params: dict[str, Any],
        *,
        keep_oof: bool,
    ) -> tuple[dict[str, Any], np.ndarray | None]:
        started = time.time()
        combined_oof = np.full(self.n_rows, np.nan, dtype=np.float32)
        train_r2_by_fold: list[float] = []
        fold_records: list[dict[str, Any]] = []
        base_params = xgb_common() | stage_params["base"]
        correction_params = xgb_common() | stage_params["correction"]
        base_rounds = int(stage_params["base_rounds"])
        correction_rounds = int(stage_params["correction_rounds"])

        for outer_fold in range(N_FOLDS):
            fold_started = time.time()
            (
                base_index,
                correction_index,
                outer_index,
                base_folds,
                correction_folds,
            ) = self.split_indices(outer_fold)
            dbase_train = self.base_matrix(base_index)
            dbase_correction = self.base_matrix(correction_index)
            dbase_outer = self.base_matrix(outer_index)
            base = xgb.train(
                base_params,
                dbase_train,
                num_boost_round=base_rounds,
                verbose_eval=False,
            )
            base_correction = self.physical_base_prediction(
                base, dbase_correction
            )
            base_outer = self.physical_base_prediction(base, dbase_outer)
            del dbase_train, dbase_correction, dbase_outer, base
            gc.collect()

            dcorrection_train = self.correction_matrix(
                correction_index, base_correction
            )
            dcorrection_outer = self.correction_matrix(outer_index, base_outer)
            correction = xgb.train(
                correction_params,
                dcorrection_train,
                num_boost_round=correction_rounds,
                verbose_eval=False,
            )
            correction_train = (
                correction.predict(dcorrection_train).astype(np.float64)
                * self.target_scale
            )
            correction_outer = (
                correction.predict(dcorrection_outer).astype(np.float64)
                * self.target_scale
            )
            combined_train = base_correction + correction_train
            combined_outer = base_outer + correction_outer
            fold_train_r2 = r2_score(
                self.label[correction_index], combined_train
            )
            fold_outer_r2 = r2_score(self.label[outer_index], combined_outer)
            train_r2_by_fold.append(fold_train_r2)
            combined_oof[outer_index] = combined_outer.astype(np.float32)
            fold_records.append({
                "outer_fold": outer_fold,
                "outer_cases": np.flatnonzero(self.case_fold == outer_fold),
                "base_folds": base_folds,
                "base_cases": np.flatnonzero(
                    np.isin(self.case_fold, base_folds)
                ),
                "correction_folds": correction_folds,
                "correction_cases": np.flatnonzero(
                    np.isin(self.case_fold, correction_folds)
                ),
                "n_base_rows": len(base_index),
                "n_correction_rows": len(correction_index),
                "n_outer_rows": len(outer_index),
                "base_rounds": base_rounds,
                "correction_rounds": correction_rounds,
                "correction_fit_r2": fold_train_r2,
                "outer_r2": fold_outer_r2,
                "fit_seconds": time.time() - fold_started,
            })
            print(
                f"FOLD_DONE outer={outer_fold} "
                f"r2={fold_outer_r2:+.7f} trees={base_rounds}+{correction_rounds} "
                f"seconds={fold_records[-1]['fit_seconds']:.1f}",
                flush=True,
            )
            del (
                dcorrection_train,
                dcorrection_outer,
                correction,
                base_correction,
                base_outer,
                correction_train,
                correction_outer,
                combined_train,
                combined_outer,
                base_index,
                correction_index,
                outer_index,
            )
            gc.collect()

        if not np.isfinite(combined_oof).all():
            raise RuntimeError("OOF combined prediction is incomplete")
        pair = self.pair_metrics(combined_oof)
        scene = self.scene_metrics(combined_oof)
        train_r2 = float(np.mean(train_r2_by_fold))
        validation_r2 = float(pair["r2"])
        pair_score = validation_r2 - R2_GAP_WEIGHT * abs(
            train_r2 - validation_r2
        )
        scene_standardized = float(scene["curve_rms"]) / self.target_scale
        pair_scene_score = pair_score - SCENE_WEIGHT * scene_standardized
        metrics = {
            "train_r2_mean_over_folds": train_r2,
            "oof_r2": validation_r2,
            "r2_train_oof_gap_abs": abs(train_r2 - validation_r2),
            "historical_pair_score": pair_score,
            "oof_scene_curve_rms": scene["curve_rms"],
            "oof_scene_curve_rms_standardized": scene_standardized,
            "pair_scene_score": pair_scene_score,
            "oof_pair_metrics": pair,
            "oof_scene_metrics": scene,
            "folds": fold_records,
            "fit_seconds": time.time() - started,
        }
        return metrics, combined_oof if keep_oof else None

    def fit_deployment(
        self,
        stage_params: dict[str, Any],
        output_dir: Path,
    ) -> dict[str, Any]:
        """Fit the frozen stack on all cases using base-OOF residual targets."""
        base_rounds = int(stage_params["base_rounds"])
        correction_rounds = int(stage_params["correction_rounds"])
        base_params = xgb_common() | stage_params["base"]
        correction_params = xgb_common() | stage_params["correction"]
        all_index = np.arange(self.n_rows, dtype=np.int32)

        base_oof = np.full(self.n_rows, np.nan, dtype=np.float32)
        for outer_fold in range(N_FOLDS):
            outer_index = self.fold_index[outer_fold]
            train_index = np.sort(np.concatenate([
                self.fold_index[fold]
                for fold in range(N_FOLDS)
                if fold != outer_fold
            ]))
            dtrain = self.base_matrix(train_index)
            douter = self.base_matrix(outer_index)
            fold_base = xgb.train(
                base_params,
                dtrain,
                num_boost_round=base_rounds,
                verbose_eval=False,
            )
            base_oof[outer_index] = self.physical_base_prediction(
                fold_base, douter
            ).astype(np.float32)
            del dtrain, douter, fold_base, train_index, outer_index
            gc.collect()
        if not np.isfinite(base_oof).all():
            raise RuntimeError("deployment base-OOF prediction is incomplete")
        dcorrection_oof = self.correction_matrix(all_index, base_oof)
        correction = xgb.train(
            correction_params,
            dcorrection_oof,
            num_boost_round=correction_rounds,
            verbose_eval=False,
        )
        del dcorrection_oof, base_oof
        gc.collect()

        dbase = self.base_matrix(all_index)
        base = xgb.train(
            base_params,
            dbase,
            num_boost_round=base_rounds,
            verbose_eval=False,
        )
        base_prediction = self.physical_base_prediction(base, dbase)
        dcorrection = self.correction_matrix(all_index, base_prediction)
        correction_prediction = (
            correction.predict(dcorrection).astype(np.float64)
            * self.target_scale
        )
        combined = base_prediction + correction_prediction
        in_sample = {
            "pair": self.pair_metrics(combined),
            "scene": self.scene_metrics(combined),
        }
        base_path = output_dir / "best_base_all200.json"
        correction_path = output_dir / "best_correction_all200.json"
        atomic_model(base, base_path)
        atomic_model(correction, correction_path)
        return {
            "base_rounds": base_rounds,
            "correction_rounds": correction_rounds,
            "base_model": str(base_path),
            "base_sha256": sha256(base_path),
            "correction_model": str(correction_path),
            "correction_sha256": sha256(correction_path),
            "correction_target": (
                "all-200-case residual to five-fold base-OOF prediction"
            ),
            "in_sample_deployment_diagnostic_not_used_for_selection": in_sample,
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

    experiment = CrossfitExperiment(
        Path(args.source_cache).resolve(), Path(args.scene_cache).resolve()
    )
    database = output_dir / "study.sqlite3"
    sampler = optuna.samplers.TPESampler(
        seed=SEED,
        n_startup_trials=10,
        multivariate=True,
    )
    study = optuna.create_study(
        study_name=f"crossfit_all200_sequential_{args.objective}_optuna50_v3",
        storage=f"sqlite:///{database}",
        direction="maximize",
        sampler=sampler,
        load_if_exists=True,
    )
    if len(study.trials) == 0:
        study.enqueue_trial(baseline_crossfit_trial())

    def objective(trial: optuna.Trial) -> float:
        stage_params = suggest_crossfit_params(trial)
        metrics, _ = experiment.fit_crossfit(stage_params, keep_oof=False)
        compact = {
            key: value
            for key, value in metrics.items()
            if key not in ("oof_scene_metrics", "folds")
        }
        compact["base_rounds_by_fold"] = [
            item["base_rounds"] for item in metrics["folds"]
        ]
        compact["correction_rounds_by_fold"] = [
            item["correction_rounds"] for item in metrics["folds"]
        ]
        for key, value in compact.items():
            trial.set_user_attr(key, json_clean(value))
        score_key = (
            "historical_pair_score"
            if args.objective == "pair"
            else "pair_scene_score"
        )
        score = float(metrics[score_key])
        print(
            f"TRIAL_DONE objective={args.objective} trial={trial.number} "
            f"score={score:+.8f} pair={metrics['historical_pair_score']:+.8f} "
            f"scene_std={metrics['oof_scene_curve_rms_standardized']:.8f} "
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
        raise RuntimeError("study ended before 50 completed trials")
    study.trials_dataframe(
        attrs=("number", "value", "params", "user_attrs", "state")
    ).to_csv(output_dir / "trials.csv", index=False)

    best = study.best_trial
    best_stage_params = crossfit_params_from_flat(best.params)
    best_metrics, best_oof = experiment.fit_crossfit(
        best_stage_params, keep_oof=True
    )
    if best_oof is None:
        raise RuntimeError("best OOF prediction was not retained")
    del best_oof
    gc.collect()
    deployment = experiment.fit_deployment(best_stage_params, output_dir)
    frozen_v22 = {
        "pair": experiment.pair_metrics(experiment.frozen_pair_prediction),
        "scene": experiment.scene_metrics(experiment.frozen_pair_prediction),
        "warning": (
            "descriptive only: frozen V2.2 is not case-OOF over these 200 cases"
        ),
    }
    payload = {
        "schema_version": 3,
        "kind": (
            "all-200-case rotating disjoint-stage sequential R_blend tuning"
        ),
        "objective_variant": args.objective,
        "objective": {
            "historical_pair_score": (
                "global OOF R2 - 0.6*abs(mean fold-fit R2 - global OOF R2)"
            ),
            "scene_term": (
                None
                if args.objective == "pair"
                else (
                    "case-balanced RMS over 20 fixed bins of exact cumulative "
                    "OOF scene residual, divided by target standard deviation"
                )
            ),
            "optimized_score": (
                "historical_pair_score"
                if args.objective == "pair"
                else "historical_pair_score - scene_curve_RMS_standardized"
            ),
        },
        "crossfit": {
            "n_cases": N_CASES,
            "n_folds": N_FOLDS,
            "case_fold": experiment.case_fold,
            "per_rotation": {
                "base_fit_cases": 80,
                "correction_fit_cases": 80,
                "outer_scored_cases": 40,
                "roles_are_case_disjoint": True,
                "tree_counts_are_optuna_hyperparameters": True,
            },
            "role_rotation": (
                "each case base-fits in two rotations, correction-fits in two, "
                "and is outer-scored exactly once"
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
        "features": {
            "base": BASE_FEATURES,
            "correction": CONDITIONAL_FEATURES,
            "correction_scene_coordinate": (
                "sum of base predictions from a model fit on disjoint cases"
            ),
            "selection_scene_bins": (
                "fixed label-free frozen-V2.2 scene-prediction bins"
            ),
        },
        "best_params": best_stage_params,
        "best_oof_metrics": best_metrics,
        "deployment_fit_all_200_cases_after_selection": deployment,
        "frozen_v22_descriptive_reference": frozen_v22,
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
            "every_case_used_in_base_fit_rotations": True,
            "every_case_used_in_correction_fit_rotations": True,
            "every_case_has_one_outer_oof_combined_prediction": True,
            "correction_fit_residual_is_out_of-case_for_base": True,
            "outer_case_labels_used_by_corresponding_fold_models": False,
            "coherent_anchor_truth_opened": False,
            "constgold_opened": False,
        },
    }
    atomic_json(summary_path, payload)
    print(json.dumps(json_clean(payload), indent=2, sort_keys=True))
    print(
        f"CROSSFIT_SEQUENTIAL_RBLEND_DONE objective={args.objective} "
        f"trials={study_completed_trials(study)} best={best.number}",
        flush=True,
    )


if __name__ == "__main__":
    main()
