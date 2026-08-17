#!/usr/bin/env python3
"""Tune a freshly trained V2.2 base followed by its response-weighted refit.

Every Optuna trial performs two complete fits on the official V2.2 cases
40--199 and official random-row train/validation split:

1. fit an ordinary squared-error seven-feature pair model with early stopping;
2. derive positive-response weights from that trial's base predictions and fit
   a new model from scratch with the same parameters and selected tree count.

The optimized value is ``abs(beta - 1)`` on all 40 case-disjoint half-shear
cases 0--39, where beta is the case-balanced projection slope of the measured
scene response vector on the summed predicted pair-response vector.  Coherent
anchors and ConstGold are never read; they remain possible transfer tests.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import optuna
import xgboost as xgb


ROOT = Path(__file__).resolve().parents[1]
BLENDEMU = Path("/home/z/Zekang.Zhang/blendemu")
FEATURES = [
    "Re_input_p_scaled",
    "Re_input_s_scaled",
    "r_input_p_scaled",
    "r_input_s_scaled",
    "sersic_n_input_p",
    "sersic_n_input_s",
    "distance_scaled",
]
FIT_CASES = (40, 199)
OPTUNA_CASES = (0, 39)
WEIGHT_CAP = 50.0
MAX_BOOST_ROUNDS = 1000
EARLY_STOPPING_ROUNDS = 50
SEED = 20260816


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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def stat(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("stat requires at least two finite values")
    return {
        "mean": float(values.mean()),
        "case_sd": float(values.std(ddof=1)),
        "case_sem": float(values.std(ddof=1) / np.sqrt(len(values))),
        "n_cases": int(len(values)),
    }


def r2(target: np.ndarray, prediction: np.ndarray) -> float:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    residual_ss = float(np.square(target - prediction).sum())
    total_ss = float(np.square(target - target.mean()).sum())
    if total_ss <= 0.0:
        raise RuntimeError("target variance is zero")
    return 1.0 - residual_ss / total_ss


def positive_response_weights(
    train_prediction: np.ndarray,
    validation_prediction: np.ndarray,
    alpha: float,
    cap: float = WEIGHT_CAP,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    train_signal = np.maximum(
        np.asarray(train_prediction, dtype=np.float64), 0.0
    ) ** 2
    validation_signal = np.maximum(
        np.asarray(validation_prediction, dtype=np.float64), 0.0
    ) ** 2
    mean_power = float(train_signal.mean())
    if not mean_power > 0.0:
        raise RuntimeError("base prediction has no positive response power")
    train_ratio = np.minimum(train_signal / mean_power, cap)
    validation_ratio = np.minimum(validation_signal / mean_power, cap)
    normalization = float((1.0 + alpha * train_ratio).mean())
    train_weight = ((1.0 + alpha * train_ratio) / normalization).astype(
        np.float32
    )
    validation_weight = (
        (1.0 + alpha * validation_ratio) / normalization
    ).astype(np.float32)
    metadata = {
        "alpha": float(alpha),
        "cap": float(cap),
        "train_mean_prediction_power": mean_power,
        "raw_weight_normalization": normalization,
        "train_weight_p50": float(np.quantile(train_weight, 0.50)),
        "train_weight_p99": float(np.quantile(train_weight, 0.99)),
        "train_weight_max": float(train_weight.max()),
    }
    return train_weight, validation_weight, metadata


class VectorEvaluation:
    """Fast reconstruction and scoring of half-shear scene-response vectors."""

    def __init__(
        self,
        case: np.ndarray,
        primary: np.ndarray,
        shear_angle_degree: np.ndarray,
        label: np.ndarray,
        null: np.ndarray,
    ) -> None:
        self.case = np.asarray(case, dtype=np.int16)
        self.primary = np.asarray(primary, dtype=np.int64)
        phase = np.deg2rad(
            2.0 * np.asarray(shear_angle_degree, dtype=np.float64)
        )
        self.cosine = np.cos(phase).astype(np.float32)
        self.sine = np.sin(phase).astype(np.float32)
        key = (self.case.astype(np.int64) << np.int64(48)) + self.primary
        if np.any(key[1:] < key[:-1]):
            raise RuntimeError("evaluation rows are not ordered by case/primary")
        change = np.empty(len(key), dtype=bool)
        change[0] = True
        change[1:] = key[1:] != key[:-1]
        self.starts = np.flatnonzero(change).astype(np.int64)
        self.scene_case = self.case[self.starts]
        csum = np.add.reduceat(self.cosine.astype(np.float64), self.starts)
        ssum = np.add.reduceat(self.sine.astype(np.float64), self.starts)
        lsum = np.add.reduceat(
            np.asarray(label, dtype=np.float64), self.starts
        )
        nsum = np.add.reduceat(
            np.asarray(null, dtype=np.float64), self.starts
        )
        direction_power = np.square(csum) + np.square(ssum)
        if np.any(direction_power <= 1.0e-8):
            raise RuntimeError("encountered vanishing summed shear direction")
        self.measured1 = (lsum * csum - nsum * ssum) / direction_power
        self.measured2 = (lsum * ssum + nsum * csum) / direction_power

    def score(
        self,
        pair_prediction: np.ndarray,
        case_window: tuple[int, int],
    ) -> dict[str, Any]:
        prediction = np.asarray(pair_prediction, dtype=np.float64)
        if prediction.shape != self.cosine.shape:
            raise ValueError("pair prediction length differs from evaluation rows")
        pred1 = np.add.reduceat(prediction * self.cosine, self.starts)
        pred2 = np.add.reduceat(prediction * self.sine, self.starts)
        power = np.square(pred1) + np.square(pred2)
        dot = pred1 * self.measured1 + pred2 * self.measured2
        cross = -pred2 * self.measured1 + pred1 * self.measured2
        case_min, case_max = case_window
        use = (self.scene_case >= case_min) & (self.scene_case <= case_max)
        local_case = self.scene_case[use].astype(np.int64) - case_min
        n_cases = case_max - case_min + 1
        case_power = np.bincount(
            local_case, weights=power[use], minlength=n_cases
        )
        case_dot = np.bincount(
            local_case, weights=dot[use], minlength=n_cases
        )
        case_cross = np.bincount(
            local_case, weights=cross[use], minlength=n_cases
        )
        if np.any(case_power <= 0.0):
            raise RuntimeError("evaluation case has zero predicted vector power")
        slopes = case_dot / case_power
        nulls = case_cross / case_power
        return {
            "case_window": [case_min, case_max],
            "n_pairs": int(
                ((self.case >= case_min) & (self.case <= case_max)).sum()
            ),
            "n_primaries": int(use.sum()),
            "slope_measured_on_predicted": stat(slopes),
            "slope_minus_one": stat(slopes - 1.0),
            "orthogonal_slope_null": stat(nulls),
            "pooled_slope": float(case_dot.sum() / case_power.sum()),
        }


def common_params() -> dict[str, Any]:
    return {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "device": os.environ.get("XGB_DEVICE", "cuda"),
        "n_jobs": int(os.environ.get("SLURM_CPUS_PER_TASK", "12")),
        "booster": "gbtree",
        "verbosity": 0,
        "max_bin": 256,
        "seed": SEED,
    }


def suggest_params(trial: optuna.Trial) -> tuple[dict[str, Any], float]:
    params = {
        "subsample": trial.suggest_float("subsample", 0.3, 1.0),
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree", 0.3, 1.0
        ),
        "learning_rate": trial.suggest_float(
            "learning_rate", 1.0e-3, 0.2, log=True
        ),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 200),
        "reg_lambda": trial.suggest_float(
            "reg_lambda", 1.0e-3, 10.0, log=True
        ),
        "reg_alpha": trial.suggest_float(
            "reg_alpha", 1.0e-3, 10.0, log=True
        ),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
    }
    alpha = trial.suggest_float("response_weight_alpha", 0.01, 0.20, log=True)
    return params, float(alpha)


def completed_trials(study: optuna.Study) -> int:
    return sum(
        trial.state == optuna.trial.TrialState.COMPLETE
        for trial in study.get_trials(deepcopy=False)
    )


class Experiment:
    def __init__(self, cache: Path) -> None:
        self.cache = cache
        with (cache / "metadata.json").open(encoding="utf-8") as handle:
            self.metadata = json.load(handle)
        arrays = {
            name: np.load(cache / filename, mmap_mode="r")
            for name, filename in self.metadata["arrays"].items()
        }
        if self.metadata["case_window"] != [0, 199]:
            raise RuntimeError("unexpected source-cache case window")
        if self.metadata["model_features"] != FEATURES:
            raise RuntimeError("source-cache feature order changed")
        case = np.asarray(arrays["case"])
        official_train = np.asarray(arrays["official_train"], dtype=bool)
        fit = (case >= FIT_CASES[0]) & (case <= FIT_CASES[1])
        train = fit & official_train
        validation = fit & ~official_train
        external = (case >= OPTUNA_CASES[0]) & (case <= OPTUNA_CASES[1])
        expected = self.metadata["official_random_row_split"]
        if int(train.sum()) != int(expected["train_rows"]):
            raise RuntimeError("official training-row count changed")
        if int(validation.sum()) != int(expected["validation_rows"]):
            raise RuntimeError("official validation-row count changed")
        self.x_train = np.asarray(arrays["x_scaled"][train], dtype=np.float32)
        self.x_validation = np.asarray(
            arrays["x_scaled"][validation], dtype=np.float32
        )
        self.x_external = np.asarray(
            arrays["x_scaled"][external], dtype=np.float32
        )
        self.y_train_physical = np.asarray(
            arrays["label"][train], dtype=np.float32
        )
        self.y_validation_physical = np.asarray(
            arrays["label"][validation], dtype=np.float32
        )
        standardization = self.metadata["source_standardization"]
        self.target_mean = float(standardization["mean"])
        self.target_scale = float(standardization["std"])
        self.y_train = (
            (self.y_train_physical - self.target_mean) / self.target_scale
        ).astype(np.float32)
        self.y_validation = (
            (self.y_validation_physical - self.target_mean) / self.target_scale
        ).astype(np.float32)
        self.dm_train = xgb.DMatrix(
            self.x_train, label=self.y_train, feature_names=FEATURES
        )
        self.dm_validation = xgb.DMatrix(
            self.x_validation,
            label=self.y_validation,
            feature_names=FEATURES,
        )
        self.dm_external = xgb.DMatrix(
            self.x_external, feature_names=FEATURES
        )
        self.vector = VectorEvaluation(
            case=case[external],
            primary=arrays["input_index"][external],
            shear_angle_degree=arrays["shear_angle"][external],
            label=arrays["label"][external],
            null=arrays["null"][external],
        )
        self.n_train = int(train.sum())
        self.n_validation = int(validation.sum())
        self.n_external = int(external.sum())

    def physical_predict(
        self,
        booster: xgb.Booster,
        matrix: xgb.DMatrix,
        n_trees: int,
    ) -> np.ndarray:
        standardized = booster.predict(matrix, iteration_range=(0, n_trees))
        return (
            standardized.astype(np.float64) * self.target_scale
            + self.target_mean
        ).astype(np.float32)

    def fit_candidate(
        self,
        params: dict[str, Any],
        alpha: float,
        *,
        keep_models: bool,
    ) -> tuple[dict[str, Any], xgb.Booster | None, xgb.Booster | None]:
        fit_params = common_params()
        fit_params.update(params)
        started = time.time()
        base_evals: dict[str, dict[str, list[float]]] = {}
        base = xgb.train(
            fit_params,
            self.dm_train,
            evals=[
                (self.dm_train, "train"),
                (self.dm_validation, "validation"),
            ],
            evals_result=base_evals,
            num_boost_round=MAX_BOOST_ROUNDS,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            verbose_eval=False,
        )
        n_trees = int(base.best_iteration) + 1
        base_train = self.physical_predict(base, self.dm_train, n_trees)
        base_validation = self.physical_predict(
            base, self.dm_validation, n_trees
        )
        base_external = self.physical_predict(base, self.dm_external, n_trees)
        train_weight, validation_weight, weight_meta = (
            positive_response_weights(
                base_train, base_validation, alpha, WEIGHT_CAP
            )
        )
        weighted_train = xgb.DMatrix(
            self.x_train,
            label=self.y_train,
            weight=train_weight,
            feature_names=FEATURES,
        )
        weighted_validation = xgb.DMatrix(
            self.x_validation,
            label=self.y_validation,
            weight=validation_weight,
            feature_names=FEATURES,
        )
        weighted_evals: dict[str, dict[str, list[float]]] = {}
        weighted = xgb.train(
            fit_params,
            weighted_train,
            evals=[
                (weighted_train, "train"),
                (weighted_validation, "validation"),
            ],
            evals_result=weighted_evals,
            num_boost_round=n_trees,
            verbose_eval=False,
        )
        weighted_validation_physical = self.physical_predict(
            weighted, self.dm_validation, n_trees
        )
        weighted_external = self.physical_predict(
            weighted, self.dm_external, n_trees
        )
        base_vector = {
            "development": self.vector.score(base_external, OPTUNA_CASES),
        }
        weighted_vector = {
            "development": self.vector.score(
                weighted_external, OPTUNA_CASES
            ),
        }
        metrics = {
            "alpha": alpha,
            "n_trees": n_trees,
            "fit_seconds": float(time.time() - started),
            "base": {
                "ordinary_validation_r2": r2(
                    self.y_validation_physical, base_validation
                ),
                "vector": base_vector,
                "best_validation_rmse_standardized": float(base.best_score),
            },
            "weighted": {
                "ordinary_validation_r2": r2(
                    self.y_validation_physical,
                    weighted_validation_physical,
                ),
                "vector": weighted_vector,
                "final_weighted_validation_rmse_standardized": float(
                    weighted_evals["validation"]["rmse"][-1]
                ),
            },
            "weight": weight_meta,
        }
        del (
            base_train,
            base_validation,
            base_external,
            weighted_validation_physical,
            weighted_external,
            train_weight,
            validation_weight,
            weighted_train,
            weighted_validation,
        )
        gc.collect()
        if keep_models:
            return metrics, base, weighted
        del base, weighted
        gc.collect()
        return metrics, None, None


def source_v22_params() -> dict[str, Any]:
    path = BLENDEMU / "models/emulator_metadata_lsst_r_extnbr_v22.json"
    with path.open(encoding="utf-8") as handle:
        task = json.load(handle)["tasks"]["regression"]
    keep = (
        "subsample",
        "colsample_bytree",
        "learning_rate",
        "max_depth",
        "min_child_weight",
        "reg_lambda",
        "reg_alpha",
        "gamma",
    )
    return {name: task["params"][name] for name in keep}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-trials", type=int, default=30)
    args = parser.parse_args()
    if args.n_trials <= 0:
        raise ValueError("n-trials must be positive")
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "summary.json"
    if summary_path.exists():
        print(f"Study already finalized: {summary_path}", flush=True)
        return

    experiment = Experiment(Path(args.cache).resolve())
    sampler = optuna.samplers.TPESampler(
        seed=SEED,
        n_startup_trials=10,
        multivariate=True,
    )
    database = output / "study.sqlite3"
    study = optuna.create_study(
        study_name="v22_fresh_base_positive_reweight_vector_optuna30_v1",
        storage=f"sqlite:///{database}",
        direction="minimize",
        sampler=sampler,
        load_if_exists=True,
    )
    if len(study.trials) == 0:
        baseline = source_v22_params()
        for alpha in (0.035, 0.05, 0.065, 0.10):
            study.enqueue_trial({
                **baseline,
                "response_weight_alpha": alpha,
            })

    def objective(trial: optuna.Trial) -> float:
        params, alpha = suggest_params(trial)
        metrics, _, _ = experiment.fit_candidate(
            params, alpha, keep_models=False
        )
        for key, value in metrics.items():
            trial.set_user_attr(key, json_clean(value))
        slope = metrics["weighted"]["vector"]["development"][
            "slope_measured_on_predicted"
        ]["mean"]
        score = abs(float(slope) - 1.0)
        base_slope = metrics["base"]["vector"]["development"][
            "slope_measured_on_predicted"
        ]["mean"]
        print(
            f"TRIAL_DONE trial={trial.number} score={score:.8f} "
            f"alpha={alpha:.6f} trees={metrics['n_trees']} "
            f"base_dev_slope={base_slope:.6f} "
            f"weighted_dev_slope={slope:.6f} "
            f"pair_r2={metrics['weighted']['ordinary_validation_r2']:.8f} "
            f"seconds={metrics['fit_seconds']:.1f}",
            flush=True,
        )
        return score

    remaining = args.n_trials - completed_trials(study)
    if remaining > 0:
        study.optimize(
            objective,
            n_trials=remaining,
            gc_after_trial=True,
            show_progress_bar=False,
        )
    if completed_trials(study) < args.n_trials:
        raise RuntimeError(
            f"only {completed_trials(study)} completed of {args.n_trials} trials"
        )
    study.trials_dataframe(
        attrs=("number", "value", "params", "user_attrs", "state")
    ).to_csv(output / "trials.csv", index=False)

    best = study.best_trial
    best_params = {
        key: value
        for key, value in best.params.items()
        if key != "response_weight_alpha"
    }
    best_alpha = float(best.params["response_weight_alpha"])
    best_metrics, base_model, weighted_model = experiment.fit_candidate(
        best_params, best_alpha, keep_models=True
    )
    if base_model is None or weighted_model is None:
        raise RuntimeError("best model refit did not return boosters")
    base_path = output / "best_base_model.json"
    weighted_path = output / "best_weighted_model.json"
    atomic_model(base_model, base_path)
    atomic_model(weighted_model, weighted_path)
    payload = {
        "schema_version": 1,
        "kind": "fresh ordinary V2.2 base followed by positive-response weighted refit",
        "study": {
            "name": study.study_name,
            "database": str(database),
            "requested_completed_trials": args.n_trials,
            "completed_trials": completed_trials(study),
            "best_trial": best.number,
            "best_value_abs_development_slope_minus_one": best.value,
            "sampler": "Optuna TPESampler",
            "seed": SEED,
        },
        "objective": {
            "metric": "abs(case-balanced half-shear vector slope - 1)",
            "selection_cases": list(OPTUNA_CASES),
            "internal_half_shear_validation_cases": None,
            "external_validation_note": (
                "all available case-disjoint half-shear cases enter tuning; "
                "use coherent anchors or newly simulated half-shear cases for "
                "an external transfer test"
            ),
            "ordinary_pair_loss_in_objective": False,
            "coherent_anchor_truth_used": False,
            "constgold_used": False,
        },
        "training": {
            "fit_cases": list(FIT_CASES),
            "official_random_row_split": experiment.metadata[
                "official_random_row_split"
            ],
            "n_train_rows": experiment.n_train,
            "n_validation_rows": experiment.n_validation,
            "base_loss": "ordinary squared error with RMSE early stopping",
            "weighted_loss": "squared error under positive-response sample weights",
            "weighted_refit_starts_from_scratch": True,
            "weighted_refit_uses_base_selected_tree_count": True,
            "weight_formula": (
                "(1 + alpha * min(max(base_prediction,0)^2 / "
                "train_mean_power, 50)) / train_mean_raw_weight"
            ),
            "weight_uses_label": False,
            "both_stages_retrained_for_every_trial": True,
        },
        "features": FEATURES,
        "best_params": best_params,
        "best_alpha": best_alpha,
        "best_metrics": best_metrics,
        "artifacts": {
            "base_model": str(base_path),
            "base_model_sha256": sha256(base_path),
            "weighted_model": str(weighted_path),
            "weighted_model_sha256": sha256(weighted_path),
        },
        "source": {
            "cache": str(experiment.cache),
            "cache_metadata_sha256": sha256(experiment.cache / "metadata.json"),
            "target_mean": experiment.target_mean,
            "target_scale": experiment.target_scale,
        },
    }
    atomic_json(summary_path, payload)
    print(json.dumps(json_clean(payload), indent=2, sort_keys=True))
    print(
        "V22_REWEIGHTED_VECTOR_OPTUNA_DONE "
        f"trials={completed_trials(study)} best={best.number}",
        flush=True,
    )


if __name__ == "__main__":
    main()
