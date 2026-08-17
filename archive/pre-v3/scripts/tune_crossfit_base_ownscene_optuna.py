#!/usr/bin/env python3
"""Tune one base R_blend model with pair and self-calibration criteria.

Every Optuna trial runs five-fold cross-validation over all 200 half-shear
cases.  Each fold trains the seven-feature base model on 160 complete cases
and predicts the remaining 40.  The concatenated predictions are therefore
out-of-fold for every pair and complete scene.

The scene term is recomputed for every trial from that trial's own OOF scene
prediction.  Twenty approximately equal-population bins are formed from the
sum of pair predictions per scene, with 0.05, 0.1, and 0.2 retained as exact
physical boundaries.  No coherent-anchor or ConstGold labels are read.
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

from scripts.tune_crossfit_sequential_rblend_optuna import (
    CrossfitExperiment,
    N_CASES,
    N_FOLDS,
    N_SCENE_BINS,
    study_completed_trials,
)
from scripts.tune_sequential_rblend_optuna import (
    BASE_FEATURES,
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
    assign_scene_bins,
    make_scene_edges,
    sha256,
)


def suggest_base_params(trial: optuna.Trial) -> dict[str, Any]:
    """Use exactly the base-stage search space from the sequential pilot."""
    model = {
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
    return {
        "model": model,
        "rounds": trial.suggest_int("base_num_boost_round", 100, 600),
    }


def baseline_base_trial() -> dict[str, Any]:
    """Enqueue the established V2.2-like base recipe as trial zero."""
    historical = baseline_trial()
    output = {
        name: value
        for name, value in historical.items()
        if name.startswith("base_")
    }
    output["base_num_boost_round"] = 271
    return output


def base_params_from_flat(flat: dict[str, Any]) -> dict[str, Any]:
    output = {
        name.removeprefix("base_"): value
        for name, value in flat.items()
        if name.startswith("base_") and name != "base_num_boost_round"
    }
    if len(output) != 8:
        raise RuntimeError("stored Optuna parameters do not cover base stage")
    return {
        "model": output,
        "rounds": int(flat["base_num_boost_round"]),
    }


def atomic_npy(path: Path, array: np.ndarray) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
    os.replace(temporary, path)


class BaseOwnSceneExperiment(CrossfitExperiment):
    """Base-only cross-fit with trial-dependent self-calibration bins."""

    def base_split_indices(
        self, outer_fold: int
    ) -> tuple[np.ndarray, np.ndarray, list[int]]:
        train_folds = [
            fold for fold in range(N_FOLDS) if fold != outer_fold
        ]
        train_index = np.sort(np.concatenate([
            self.fold_index[fold] for fold in train_folds
        ]))
        outer_index = self.fold_index[outer_fold]
        if len(train_index) + len(outer_index) != self.n_rows:
            raise RuntimeError("base fold does not partition all rows")
        return train_index, outer_index, train_folds

    def own_scene_metrics(
        self, pair_prediction: np.ndarray
    ) -> dict[str, Any]:
        """Evaluate calibration in bins made from this prediction itself."""
        prediction_scene = np.add.reduceat(
            np.asarray(pair_prediction, dtype=np.float64), self.scene_starts
        )
        if not np.isfinite(prediction_scene).all():
            raise RuntimeError("own scene prediction is non-finite")
        residual = self.truth_scene - prediction_scene
        edges = make_scene_edges(
            prediction_scene,
            n_quantile_bins=N_SCENE_BINS,
        )
        scene_bin = assign_scene_bins(prediction_scene, edges).astype(np.int64)
        flat_index = self.scene_case.astype(np.int64) * N_SCENE_BINS + scene_bin
        count = np.bincount(
            flat_index, minlength=N_CASES * N_SCENE_BINS
        ).reshape(N_CASES, N_SCENE_BINS)
        total = np.bincount(
            flat_index,
            weights=residual,
            minlength=N_CASES * N_SCENE_BINS,
        ).reshape(N_CASES, N_SCENE_BINS)
        if np.any(count == 0):
            missing = np.argwhere(count == 0)
            raise RuntimeError(
                f"trial-dependent bins leave {len(missing)} empty case bins"
            )
        case_bin_mean = total / count
        curve = case_bin_mean.mean(axis=0)
        scene_count = np.bincount(scene_bin, minlength=N_SCENE_BINS)
        prediction_mean = np.bincount(
            scene_bin,
            weights=prediction_scene,
            minlength=N_SCENE_BINS,
        ) / scene_count
        truth_mean = np.bincount(
            scene_bin,
            weights=self.truth_scene,
            minlength=N_SCENE_BINS,
        ) / scene_count

        scene_case_count = np.bincount(self.scene_case, minlength=N_CASES)
        scene_case_mean = np.bincount(
            self.scene_case,
            weights=residual,
            minlength=N_CASES,
        ) / scene_case_count
        output: dict[str, Any] = {
            "coordinate": "trial-specific OOF base scene prediction",
            "edges": edges,
            "edge_convention": "lower-inclusive, upper-exclusive",
            "curve": curve,
            "curve_rms": float(np.sqrt(np.mean(np.square(curve)))),
            "prediction_mean": prediction_mean,
            "truth_mean": truth_mean,
            "scene_count": scene_count,
            "case_balanced_mean": float(scene_case_mean.mean()),
            "case_sem": float(
                scene_case_mean.std(ddof=1) / np.sqrt(N_CASES)
            ),
            "row_weighted_mean": float(residual.mean()),
        }
        for name, mask in {
            "tail_gt_0p1": prediction_scene > 0.1,
            "tail_gt_0p2": prediction_scene > 0.2,
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
                "n_cases": int(present.sum()),
                "case_balanced_mean": float(case_mean.mean()),
                "case_sem": float(
                    case_mean.std(ddof=1) / np.sqrt(len(case_mean))
                ),
            }
        return output

    def fit_crossfit(
        self,
        stage_params: dict[str, Any],
        *,
        keep_oof: bool,
    ) -> tuple[dict[str, Any], np.ndarray | None]:
        started = time.time()
        base_oof = np.full(self.n_rows, np.nan, dtype=np.float32)
        train_r2_by_fold: list[float] = []
        fold_records: list[dict[str, Any]] = []
        base_params = xgb_common() | stage_params["model"]
        base_rounds = int(stage_params["rounds"])

        for outer_fold in range(N_FOLDS):
            fold_started = time.time()
            train_index, outer_index, train_folds = self.base_split_indices(
                outer_fold
            )
            dtrain = self.base_matrix(train_index)
            douter = self.base_matrix(outer_index)
            base = xgb.train(
                base_params,
                dtrain,
                num_boost_round=base_rounds,
                verbose_eval=False,
            )
            train_prediction = self.physical_base_prediction(base, dtrain)
            outer_prediction = self.physical_base_prediction(base, douter)
            fold_train_r2 = r2_score(self.label[train_index], train_prediction)
            fold_outer_r2 = r2_score(self.label[outer_index], outer_prediction)
            train_r2_by_fold.append(fold_train_r2)
            base_oof[outer_index] = outer_prediction.astype(np.float32)
            fold_records.append({
                "outer_fold": outer_fold,
                "outer_cases": np.flatnonzero(self.case_fold == outer_fold),
                "train_folds": train_folds,
                "train_cases": np.flatnonzero(
                    np.isin(self.case_fold, train_folds)
                ),
                "n_train_rows": len(train_index),
                "n_outer_rows": len(outer_index),
                "base_rounds": base_rounds,
                "train_r2": fold_train_r2,
                "outer_r2": fold_outer_r2,
                "fit_seconds": time.time() - fold_started,
            })
            print(
                f"FOLD_DONE outer={outer_fold} "
                f"train_r2={fold_train_r2:+.7f} "
                f"outer_r2={fold_outer_r2:+.7f} trees={base_rounds} "
                f"seconds={fold_records[-1]['fit_seconds']:.1f}",
                flush=True,
            )
            del (
                dtrain,
                douter,
                base,
                train_prediction,
                outer_prediction,
                train_index,
                outer_index,
            )
            gc.collect()

        if not np.isfinite(base_oof).all():
            raise RuntimeError("OOF base prediction is incomplete")
        pair = self.pair_metrics(base_oof)
        own_scene = self.own_scene_metrics(base_oof)
        fixed_scene = self.scene_metrics(base_oof)
        train_r2 = float(np.mean(train_r2_by_fold))
        validation_r2 = float(pair["r2"])
        pair_score = validation_r2 - R2_GAP_WEIGHT * abs(
            train_r2 - validation_r2
        )
        own_scene_standardized = (
            float(own_scene["curve_rms"]) / self.target_scale
        )
        fixed_scene_standardized = (
            float(fixed_scene["curve_rms"]) / self.target_scale
        )
        score = pair_score - SCENE_WEIGHT * own_scene_standardized
        metrics = {
            "train_r2_mean_over_folds": train_r2,
            "oof_r2": validation_r2,
            "r2_train_oof_gap_abs": abs(train_r2 - validation_r2),
            "historical_pair_score": pair_score,
            "oof_own_scene_curve_rms": own_scene["curve_rms"],
            "oof_own_scene_curve_rms_standardized": own_scene_standardized,
            "oof_fixed_v22_scene_curve_rms": fixed_scene["curve_rms"],
            "oof_fixed_v22_scene_curve_rms_standardized": (
                fixed_scene_standardized
            ),
            "pair_own_scene_score": score,
            "oof_pair_metrics": pair,
            "oof_own_scene_metrics": own_scene,
            "oof_fixed_v22_scene_metrics": fixed_scene,
            "folds": fold_records,
            "fit_seconds": time.time() - started,
        }
        return metrics, base_oof if keep_oof else None

    def fit_deployment(
        self,
        stage_params: dict[str, Any],
        output_dir: Path,
    ) -> dict[str, Any]:
        """Fit and save one base booster using all 200 cases."""
        base_params = xgb_common() | stage_params["model"]
        base_rounds = int(stage_params["rounds"])
        all_index = np.arange(self.n_rows, dtype=np.int32)
        dtrain = self.base_matrix(all_index)
        base = xgb.train(
            base_params,
            dtrain,
            num_boost_round=base_rounds,
            verbose_eval=False,
        )
        prediction = self.physical_base_prediction(base, dtrain)
        in_sample = {
            "pair": self.pair_metrics(prediction),
            "own_scene": self.own_scene_metrics(prediction),
            "fixed_v22_scene": self.scene_metrics(prediction),
        }
        model_path = output_dir / "best_base_all200.json"
        atomic_model(base, model_path)
        del dtrain, base, prediction, all_index
        gc.collect()
        return {
            "base_rounds": base_rounds,
            "base_model": str(model_path),
            "base_sha256": sha256(model_path),
            "in_sample_deployment_diagnostic_not_used_for_selection": in_sample,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-cache", required=True)
    parser.add_argument("--scene-cache", required=True)
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

    experiment = BaseOwnSceneExperiment(
        Path(args.source_cache).resolve(), Path(args.scene_cache).resolve()
    )
    database = output_dir / "study.sqlite3"
    sampler = optuna.samplers.TPESampler(
        seed=SEED,
        n_startup_trials=10,
        multivariate=True,
    )
    study = optuna.create_study(
        study_name="crossfit_all200_base_ownscene_optuna50_v1",
        storage=f"sqlite:///{database}",
        direction="maximize",
        sampler=sampler,
        load_if_exists=True,
    )
    if study_completed_trials(study) == 0:
        study.enqueue_trial(baseline_base_trial())

    def objective(trial: optuna.Trial) -> float:
        stage_params = suggest_base_params(trial)
        metrics, _ = experiment.fit_crossfit(stage_params, keep_oof=False)
        own_scene = metrics["oof_own_scene_metrics"]
        fixed_scene = metrics["oof_fixed_v22_scene_metrics"]
        compact = {
            key: value
            for key, value in metrics.items()
            if key not in (
                "oof_own_scene_metrics",
                "oof_fixed_v22_scene_metrics",
                "folds",
            )
        }
        compact.update({
            "own_scene_edges": own_scene["edges"],
            "own_tail_gt_0p1_gap": own_scene["tail_gt_0p1"][
                "case_balanced_mean"
            ],
            "own_tail_gt_0p2_gap": own_scene["tail_gt_0p2"][
                "case_balanced_mean"
            ],
            "fixed_tail_gt_0p1_gap": fixed_scene["tail_gt_0p1"][
                "case_balanced_mean"
            ],
            "fixed_tail_gt_0p2_gap": fixed_scene["tail_gt_0p2"][
                "case_balanced_mean"
            ],
            "base_rounds_by_fold": [
                item["base_rounds"] for item in metrics["folds"]
            ],
        })
        for key, value in compact.items():
            trial.set_user_attr(key, json_clean(value))
        score = float(metrics["pair_own_scene_score"])
        print(
            f"TRIAL_DONE trial={trial.number} score={score:+.8f} "
            f"pair={metrics['historical_pair_score']:+.8f} "
            "own_scene_std="
            f"{metrics['oof_own_scene_curve_rms_standardized']:.8f} "
            "fixed_scene_std="
            f"{metrics['oof_fixed_v22_scene_curve_rms_standardized']:.8f} "
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
        raise RuntimeError(
            f"study ended with {study_completed_trials(study)} completed trials"
        )
    study.trials_dataframe(
        attrs=("number", "value", "params", "user_attrs", "state")
    ).to_csv(output_dir / "trials.csv", index=False)

    best = study.best_trial
    best_stage_params = base_params_from_flat(best.params)
    best_metrics, best_oof = experiment.fit_crossfit(
        best_stage_params, keep_oof=True
    )
    if best_oof is None:
        raise RuntimeError("best OOF prediction was not retained")
    oof_path = output_dir / "best_oof_pair_prediction.npy"
    atomic_npy(oof_path, best_oof)
    del best_oof
    gc.collect()
    deployment = experiment.fit_deployment(best_stage_params, output_dir)
    frozen_v22 = {
        "pair": experiment.pair_metrics(experiment.frozen_pair_prediction),
        "own_scene": experiment.own_scene_metrics(
            experiment.frozen_pair_prediction
        ),
        "fixed_v22_scene": experiment.scene_metrics(
            experiment.frozen_pair_prediction
        ),
        "warning": (
            "descriptive only: frozen V2.2 is not case-OOF over these 200 cases"
        ),
    }
    payload = {
        "schema_version": 1,
        "kind": "all-200-case base-only R_blend self-calibration tuning",
        "objective_variant": "base_pair_own_scene",
        "objective": {
            "historical_pair_score": (
                "global OOF R2 - 0.6*abs(mean fold-fit R2 - global OOF R2)"
            ),
            "scene_term": (
                "case-balanced RMS over 20 trial-specific bins of exact "
                "cumulative OOF scene residual, divided by target standard "
                "deviation"
            ),
            "scene_coordinate": (
                "sum of the same trial's OOF base pair predictions per scene"
            ),
            "scene_bin_construction": (
                "20 approximate quantile bins with nearest boundaries replaced "
                "by exact 0.05, 0.1, and 0.2 landmarks"
            ),
            "optimized_score": (
                "historical_pair_score - 1.0*own_scene_curve_RMS_standardized"
            ),
        },
        "crossfit": {
            "n_cases": N_CASES,
            "n_folds": N_FOLDS,
            "case_fold": experiment.case_fold,
            "per_fold": {
                "base_fit_cases": 160,
                "outer_scored_cases": 40,
                "roles_are_case_disjoint": True,
                "tree_count_is_an_optuna_hyperparameter": True,
            },
            "role_rotation": (
                "each case base-fits in four folds and is outer-scored once"
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
            "selection_scene_bins": (
                "label-free bins recomputed from each trial's own OOF base "
                "scene predictions"
            ),
            "fixed_v22_bins_role": "diagnostic only; not optimized",
        },
        "best_params": best_stage_params,
        "best_oof_metrics": best_metrics,
        "best_oof_prediction": {
            "path": str(oof_path),
            "sha256": sha256(oof_path),
            "dtype": "float32",
            "row_order": "source cache row order",
        },
        "deployment_fit_all_200_cases_after_selection": deployment,
        "frozen_v22_descriptive_reference": frozen_v22,
        "source": {
            "cache": str(experiment.source_cache),
            "metadata_sha256": sha256(
                experiment.source_cache / "metadata.json"
            ),
            "scene_cache": str(experiment.scene_cache),
            "scene_metadata_sha256": sha256(
                experiment.scene_cache / "metadata.json"
            ),
            "target_mean": experiment.target_mean,
            "target_scale": experiment.target_scale,
        },
        "protocol": {
            "every_case_used_for_base_fit": True,
            "every_case_has_one_outer_oof_prediction": True,
            "outer_case_labels_used_by_corresponding_fold_model": False,
            "scene_bins_use_labels": False,
            "coherent_anchor_truth_opened": False,
            "constgold_opened": False,
        },
    }
    atomic_json(summary_path, payload)
    print(json.dumps(json_clean(payload), indent=2, sort_keys=True))
    print(
        "CROSSFIT_BASE_OWNSCENE_DONE "
        f"trials={study_completed_trials(study)} best={best.number}",
        flush=True,
    )


if __name__ == "__main__":
    main()
