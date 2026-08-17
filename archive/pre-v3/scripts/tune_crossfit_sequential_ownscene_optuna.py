#!/usr/bin/env python3
"""Tune a base-plus-correction R_blend stack with endogenous scene bins.

Both stages are retrained in every Optuna trial.  The five-fold rotating
protocol keeps base-fit, correction-fit, and outer-scored cases disjoint:
80 cases fit the base, 80 different cases fit the correction to base-OOF
residuals, and 40 cases score the final stack.  Each case is outer-scored
once.

The scene term is computed after concatenating the final corrected OOF pair
predictions.  Twenty approximately equal-population bins are rebuilt from
that trial's own corrected scene predictions, with 0.05, 0.1, and 0.2 kept as
exact physical boundaries.  Frozen V2.2 bins are diagnostics only.  No
coherent-anchor or ConstGold labels are read.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import optuna

from scripts.tune_crossfit_base_ownscene_optuna import (
    BaseOwnSceneExperiment,
    atomic_npy,
)
from scripts.tune_crossfit_sequential_rblend_optuna import (
    CrossfitExperiment,
    N_CASES,
    N_FOLDS,
    baseline_crossfit_trial,
    crossfit_params_from_flat,
    study_completed_trials,
    suggest_crossfit_params,
)
from scripts.tune_sequential_rblend_optuna import (
    BASE_FEATURES,
    CONDITIONAL_FEATURES,
    SCENE_WEIGHT,
    SEED,
    atomic_json,
    json_clean,
)
from scripts.v22_grouped_rscene_common import sha256


def flatten_stage_params(stage_params: dict[str, Any]) -> dict[str, Any]:
    """Convert saved nested parameters back to the Optuna naming scheme."""
    output: dict[str, Any] = {
        "base_num_boost_round": int(stage_params["base_rounds"]),
        "correction_num_boost_round": int(stage_params["correction_rounds"]),
    }
    for stage in ("base", "correction"):
        output.update({
            f"{stage}_{name}": value
            for name, value in stage_params[stage].items()
        })
    return output


def load_seed_trial(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return flatten_stage_params(payload["best_params"])


class SequentialOwnSceneExperiment(CrossfitExperiment):
    """Two-stage cross-fit scored in bins of its final own prediction."""

    # This method only needs scene-index attributes supplied by
    # CrossfitExperiment, so reuse the exactly matching base-only definition.
    own_scene_metrics = BaseOwnSceneExperiment.own_scene_metrics

    def fit_crossfit(
        self,
        stage_params: dict[str, Any],
        *,
        keep_oof: bool,
    ) -> tuple[dict[str, Any], Any]:
        metrics, combined_oof = super().fit_crossfit(
            stage_params, keep_oof=True
        )
        if combined_oof is None:
            raise RuntimeError("combined OOF prediction was not retained")

        own_scene = self.own_scene_metrics(combined_oof)
        fixed_scene = metrics.pop("oof_scene_metrics")
        fixed_rms = metrics.pop("oof_scene_curve_rms")
        fixed_standardized = metrics.pop(
            "oof_scene_curve_rms_standardized"
        )
        fixed_score = metrics.pop("pair_scene_score")
        own_standardized = float(own_scene["curve_rms"]) / self.target_scale
        own_score = (
            float(metrics["historical_pair_score"])
            - SCENE_WEIGHT * own_standardized
        )
        metrics.update({
            "oof_own_scene_curve_rms": own_scene["curve_rms"],
            "oof_own_scene_curve_rms_standardized": own_standardized,
            "oof_fixed_v22_scene_curve_rms": fixed_rms,
            "oof_fixed_v22_scene_curve_rms_standardized": fixed_standardized,
            "pair_own_scene_score": own_score,
            "fixed_v22_pair_scene_score_diagnostic": fixed_score,
            "oof_own_scene_metrics": own_scene,
            "oof_fixed_v22_scene_metrics": fixed_scene,
        })
        if keep_oof:
            return metrics, combined_oof
        del combined_oof
        gc.collect()
        return metrics, None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-cache", required=True)
    parser.add_argument("--scene-cache", required=True)
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--seed-summary",
        action="append",
        default=[],
        help="optional prior sequential summary whose best recipe is enqueued",
    )
    args = parser.parse_args()
    if args.n_trials <= 0:
        raise ValueError("n-trials must be positive")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        print(f"Study already complete: {summary_path}", flush=True)
        return

    experiment = SequentialOwnSceneExperiment(
        Path(args.source_cache).resolve(), Path(args.scene_cache).resolve()
    )
    database = output_dir / "study.sqlite3"
    sampler = optuna.samplers.TPESampler(
        seed=SEED,
        n_startup_trials=10,
        multivariate=True,
    )
    study = optuna.create_study(
        study_name="crossfit_all200_sequential_ownscene_optuna50_v1",
        storage=f"sqlite:///{database}",
        direction="maximize",
        sampler=sampler,
        load_if_exists=True,
    )
    seed_paths = [Path(item).resolve() for item in args.seed_summary]
    if len(study.trials) == 0:
        study.enqueue_trial(baseline_crossfit_trial())
        for path in seed_paths:
            study.enqueue_trial(load_seed_trial(path))

    def objective(trial: optuna.Trial) -> float:
        stage_params = suggest_crossfit_params(trial)
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
            "correction_rounds_by_fold": [
                item["correction_rounds"] for item in metrics["folds"]
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
    best_stage_params = crossfit_params_from_flat(best.params)
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
        "kind": (
            "all-200-case sequential R_blend tuning with corrected-own-scene "
            "calibration"
        ),
        "objective_variant": "sequential_pair_corrected_own_scene",
        "objective": {
            "historical_pair_score": (
                "global final OOF R2 - 0.6*abs(mean correction-fit R2 - "
                "global final OOF R2)"
            ),
            "scene_term": (
                "case-balanced RMS over 20 trial-specific bins of exact "
                "cumulative final corrected OOF scene residual, divided by "
                "target standard deviation"
            ),
            "scene_coordinate": (
                "sum of the same trial's final base-plus-correction OOF pair "
                "predictions per scene"
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
            "per_rotation": {
                "base_fit_cases": 80,
                "correction_fit_cases": 80,
                "outer_scored_cases": 40,
                "roles_are_case_disjoint": True,
                "both_stages_retrained_every_trial": True,
                "both_tree_counts_are_optuna_hyperparameters": True,
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
            "enqueued_seed_summaries": [str(path) for path in seed_paths],
        },
        "features": {
            "base": BASE_FEATURES,
            "correction": CONDITIONAL_FEATURES,
            "correction_scene_coordinate": (
                "sum of base predictions from a model fit on disjoint cases"
            ),
            "selection_scene_bins": (
                "label-free bins recomputed from each trial's own final "
                "corrected OOF scene predictions"
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
            "every_case_used_in_base_fit_rotations": True,
            "every_case_used_in_correction_fit_rotations": True,
            "every_case_has_one_outer_oof_combined_prediction": True,
            "correction_fit_residual_is_out_of-case_for_base": True,
            "outer_case_labels_used_by_corresponding_fold_models": False,
            "scene_bins_use_labels": False,
            "coherent_anchor_truth_opened": False,
            "constgold_opened": False,
        },
    }
    atomic_json(summary_path, payload)
    print(json.dumps(json_clean(payload), indent=2, sort_keys=True))
    print(
        "CROSSFIT_SEQUENTIAL_OWNSCENE_DONE "
        f"trials={study_completed_trials(study)} best={best.number}",
        flush=True,
    )


if __name__ == "__main__":
    main()
