#!/usr/bin/env python3
"""Tune one blendness-aware base model on pair and scene calibration.

This is the all-200-case, five-fold base-only tuning experiment with three
additional pair-excluded shell-flux inputs.  Each trial is scored entirely on
its concatenated out-of-fold predictions.  Its own predictions define both
the pair-R_blend calibration bins and the cumulative-scene calibration bins.
No correction model is trained and no coherent-anchor or ConstGold truth is
read.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import xgboost as xgb

from scripts.tune_crossfit_base_ownscene_optuna import (
    BaseOwnSceneExperiment,
    atomic_npy,
    base_params_from_flat,
    baseline_base_trial,
    suggest_base_params,
)
from scripts.tune_crossfit_sequential_rblend_optuna import (
    N_CASES,
    N_SCENE_BINS,
    study_completed_trials,
)
from scripts.tune_sequential_rblend_optuna import (
    BASE_FEATURES,
    SEED,
    atomic_json,
    atomic_model,
    json_clean,
    xgb_common,
)
from scripts.v22_grouped_rscene_common import sha256


BLENDNESS_FEATURES = [
    "scene_logfluxabs_other_near_0_1",
    "scene_logfluxabs_other_mid_1_3",
    "scene_logfluxabs_other_far_3_10",
]
EXTENDED_BASE_FEATURES = [*BASE_FEATURES, *BLENDNESS_FEATURES]
PAIR_AXIS_WEIGHT = 0.5
SCENE_AXIS_WEIGHT = 0.5


class Blend3PairSceneExperiment(BaseOwnSceneExperiment):
    """Base-only cross-fit with three pair-excluded scene-flux features."""

    def __init__(
        self,
        source_cache: Path,
        scene_cache: Path,
        blendness_cache: Path,
    ) -> None:
        super().__init__(source_cache, scene_cache)
        self.blendness_cache = blendness_cache
        metadata_path = blendness_cache / "metadata.json"
        with metadata_path.open(encoding="utf-8") as handle:
            self.blendness_meta = json.load(handle)
        if self.blendness_meta["source_metadata_sha256"] != sha256(
            source_cache / "metadata.json"
        ):
            raise RuntimeError("blendness cache does not match source cache")
        if self.blendness_meta["features"] != BLENDNESS_FEATURES:
            raise RuntimeError("blendness feature names or order have drifted")
        self.x_blendness = np.load(
            blendness_cache / self.blendness_meta["array"], mmap_mode="r"
        )
        if self.x_blendness.shape != (self.n_rows, len(BLENDNESS_FEATURES)):
            raise RuntimeError("blendness feature matrix has the wrong shape")
        if not np.isfinite(self.x_blendness).all():
            raise RuntimeError("blendness feature matrix is non-finite")
        print(
            "Loaded pair-excluded blendness features: "
            f"shape={self.x_blendness.shape}",
            flush=True,
        )

    def base_matrix(self, index: np.ndarray) -> xgb.DMatrix:
        pair = np.asarray(self.x_scaled[index], dtype=np.float32)
        blendness = np.asarray(self.x_blendness[index], dtype=np.float32)
        features = np.column_stack((pair, blendness))
        target = (
            np.asarray(self.label[index], dtype=np.float64) - self.target_mean
        ) / self.target_scale
        matrix = xgb.DMatrix(
            features,
            label=target.astype(np.float32),
            feature_names=EXTENDED_BASE_FEATURES,
        )
        del pair, blendness, features, target
        return matrix

    def own_pair_metrics(self, prediction: np.ndarray) -> dict[str, Any]:
        """Case-balanced residual curve in trial-specific prediction bins."""
        prediction = np.asarray(prediction, dtype=np.float64)
        if prediction.shape != (self.n_rows,) or not np.isfinite(prediction).all():
            raise RuntimeError("pair prediction is incomplete or non-finite")
        edges = np.quantile(
            prediction, np.linspace(0.0, 1.0, N_SCENE_BINS + 1)
        )
        if np.any(np.diff(edges) <= 0.0):
            raise RuntimeError("pair prediction has repeated quantile edges")
        pair_bin = np.searchsorted(edges[1:-1], prediction, side="right")
        case = np.asarray(self.case, dtype=np.int64)
        residual = np.asarray(self.label, dtype=np.float64) - prediction
        flat = case * N_SCENE_BINS + pair_bin
        length = N_CASES * N_SCENE_BINS
        count = np.bincount(flat, minlength=length).reshape(
            N_CASES, N_SCENE_BINS
        )
        residual_total = np.bincount(
            flat, weights=residual, minlength=length
        ).reshape(N_CASES, N_SCENE_BINS)
        prediction_total = np.bincount(
            flat, weights=prediction, minlength=length
        ).reshape(N_CASES, N_SCENE_BINS)
        label_total = np.bincount(
            flat,
            weights=np.asarray(self.label, dtype=np.float64),
            minlength=length,
        ).reshape(N_CASES, N_SCENE_BINS)
        if np.any(count == 0):
            raise RuntimeError("trial-specific pair bins leave an empty case bin")
        case_bin_residual = residual_total / count
        curve = case_bin_residual.mean(axis=0)
        return {
            "coordinate": "trial-specific OOF base pair prediction",
            "edges": edges,
            "edge_convention": "lower-inclusive, upper-exclusive",
            "curve": curve,
            "curve_rms": float(np.sqrt(np.mean(np.square(curve)))),
            "prediction_mean": (prediction_total / count).mean(axis=0),
            "label_mean": (label_total / count).mean(axis=0),
            "pair_count": count.sum(axis=0),
            "case_balanced_global_mean": float(
                (
                    residual_total.sum(axis=1) / count.sum(axis=1)
                ).mean()
            ),
            "row_weighted_global_mean": float(residual.mean()),
        }

    def fit_crossfit(
        self,
        stage_params: dict[str, Any],
        *,
        keep_oof: bool,
    ) -> tuple[dict[str, Any], np.ndarray | None]:
        metrics, prediction = super().fit_crossfit(
            stage_params, keep_oof=True
        )
        if prediction is None:
            raise RuntimeError("parent cross-fit did not return OOF prediction")
        pair_axis = self.own_pair_metrics(prediction)
        pair_axis_standardized = float(pair_axis["curve_rms"]) / self.target_scale
        scene_axis_standardized = float(
            metrics["oof_own_scene_curve_rms_standardized"]
        )
        score = (
            float(metrics["historical_pair_score"])
            - PAIR_AXIS_WEIGHT * pair_axis_standardized
            - SCENE_AXIS_WEIGHT * scene_axis_standardized
        )
        metrics.update({
            "oof_own_pair_curve_rms": pair_axis["curve_rms"],
            "oof_own_pair_curve_rms_standardized": pair_axis_standardized,
            "oof_own_pair_metrics": pair_axis,
            "pair_axis_weight": PAIR_AXIS_WEIGHT,
            "scene_axis_weight": SCENE_AXIS_WEIGHT,
            "pair_own_pair_scene_score": score,
        })
        if keep_oof:
            return metrics, prediction
        del prediction
        gc.collect()
        return metrics, None

    def fit_deployment(
        self,
        stage_params: dict[str, Any],
        output_dir: Path,
    ) -> dict[str, Any]:
        """Fit and save one ten-feature booster using all 200 cases."""
        params = xgb_common() | stage_params["model"]
        rounds = int(stage_params["rounds"])
        all_index = np.arange(self.n_rows, dtype=np.int32)
        dtrain = self.base_matrix(all_index)
        booster = xgb.train(
            params, dtrain, num_boost_round=rounds, verbose_eval=False
        )
        prediction = self.physical_base_prediction(booster, dtrain)
        diagnostics = {
            "pair": self.pair_metrics(prediction),
            "own_pair": self.own_pair_metrics(prediction),
            "own_scene": self.own_scene_metrics(prediction),
            "fixed_v22_scene": self.scene_metrics(prediction),
        }
        model_path = output_dir / "best_base_all200.json"
        atomic_model(booster, model_path)
        del dtrain, booster, prediction, all_index
        gc.collect()
        return {
            "base_rounds": rounds,
            "base_model": str(model_path),
            "base_sha256": sha256(model_path),
            "in_sample_deployment_diagnostic_not_used_for_selection": diagnostics,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-cache", required=True)
    parser.add_argument("--scene-cache", required=True)
    parser.add_argument("--blendness-cache", required=True)
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

    experiment = Blend3PairSceneExperiment(
        Path(args.source_cache).resolve(),
        Path(args.scene_cache).resolve(),
        Path(args.blendness_cache).resolve(),
    )
    database = output_dir / "study.sqlite3"
    sampler = optuna.samplers.TPESampler(
        seed=SEED, n_startup_trials=10, multivariate=True
    )
    study = optuna.create_study(
        study_name="crossfit_all200_base_blend3_pairscene_optuna50_v1",
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
        pair_axis = metrics["oof_own_pair_metrics"]
        own_scene = metrics["oof_own_scene_metrics"]
        fixed_scene = metrics["oof_fixed_v22_scene_metrics"]
        excluded = {
            "oof_own_pair_metrics",
            "oof_own_scene_metrics",
            "oof_fixed_v22_scene_metrics",
            "folds",
        }
        compact = {
            key: value for key, value in metrics.items() if key not in excluded
        }
        compact.update({
            "own_pair_edges": pair_axis["edges"],
            "own_pair_curve": pair_axis["curve"],
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
        score = float(metrics["pair_own_pair_scene_score"])
        print(
            f"TRIAL_DONE trial={trial.number} score={score:+.8f} "
            f"pair={metrics['historical_pair_score']:+.8f} "
            "pair_axis_std="
            f"{metrics['oof_own_pair_curve_rms_standardized']:.8f} "
            "scene_axis_std="
            f"{metrics['oof_own_scene_curve_rms_standardized']:.8f} "
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
        raise RuntimeError("study ended before requested completed-trial count")
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
        "own_pair": experiment.own_pair_metrics(
            experiment.frozen_pair_prediction
        ),
        "own_scene": experiment.own_scene_metrics(
            experiment.frozen_pair_prediction
        ),
        "fixed_v22_scene": experiment.scene_metrics(
            experiment.frozen_pair_prediction
        ),
        "warning": "descriptive only; frozen V2.2 is not case-OOF here",
    }
    payload = {
        "schema_version": 1,
        "kind": "all-200-case blendness-aware base-only R_blend tuning",
        "objective_variant": "base_pair_own_pair_own_scene",
        "objective": {
            "historical_pair_score": (
                "global OOF R2 - 0.6*abs(mean fold-fit R2 - global OOF R2)"
            ),
            "pair_axis_term": (
                "case-balanced RMS over 20 trial-specific quantile bins of "
                "pair label minus pair prediction, divided by target std"
            ),
            "scene_axis_term": (
                "case-balanced RMS over 20 trial-specific bins of cumulative "
                "scene residual, divided by target std"
            ),
            "weights": {
                "pair_axis": PAIR_AXIS_WEIGHT,
                "scene_axis": SCENE_AXIS_WEIGHT,
                "combined_calibration_weight": (
                    PAIR_AXIS_WEIGHT + SCENE_AXIS_WEIGHT
                ),
            },
            "optimized_score": (
                "historical_pair_score - 0.5*pair_axis_RMS_standardized "
                "- 0.5*scene_axis_RMS_standardized"
            ),
        },
        "crossfit": {
            "n_cases": N_CASES,
            "n_folds": 5,
            "case_fold": experiment.case_fold,
            "per_fold": {
                "base_fit_cases": 160,
                "outer_scored_cases": 40,
                "roles_are_case_disjoint": True,
            },
        },
        "features": {
            "base": EXTENDED_BASE_FEATURES,
            "added_blendness": BLENDNESS_FEATURES,
            "blendness_definition": experiment.blendness_meta[
                "feature_definition"
            ],
            "pair_axis_bins": (
                "20 equal-population bins from each trial's own OOF pair "
                "predictions"
            ),
            "scene_axis_bins": (
                "20 approximate quantile bins from each trial's own OOF "
                "scene sums, retaining exact 0.05/0.1/0.2 landmarks"
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
            "blendness_cache": str(experiment.blendness_cache),
            "blendness_metadata_sha256": sha256(
                experiment.blendness_cache / "metadata.json"
            ),
            "target_mean": experiment.target_mean,
            "target_scale": experiment.target_scale,
        },
        "protocol": {
            "every_case_used_for_base_fit": True,
            "every_case_has_one_outer_oof_prediction": True,
            "outer_case_labels_used_by_corresponding_fold_model": False,
            "pair_and_scene_bins_use_labels": False,
            "coherent_anchor_truth_opened": False,
            "constgold_opened": False,
        },
    }
    atomic_json(summary_path, payload)
    print(json.dumps(json_clean(payload), indent=2, sort_keys=True))
    print(
        "CROSSFIT_BASE_BLEND3_PAIRSCENE_DONE "
        f"trials={study_completed_trials(study)} best={best.number}",
        flush=True,
    )


if __name__ == "__main__":
    main()
