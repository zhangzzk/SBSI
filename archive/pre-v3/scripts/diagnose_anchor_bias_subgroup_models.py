"""Exploratory carrier-specific ablations for the anchor bias emulator.

The global bias emulator showed strong conditional ordering but failed its
predeclared case-paired MSE gate.  This follow-up asks the narrower mechanism
question suggested by its one-dimensional curves: after freezing the existing
``ratio > 5 and positive dominant response`` carrier, can latent physical
variables predict residual structure inside that carrier or its complement?

Cases 400--599 train, 600--699 select capacity separately for each population
and feature set, and 700--899 score the fitted development model.  Those final
cases were already opened by the global emulator, so this is explicitly an
exploratory subgroup audit rather than a second untouched validation.  Constgold
is never read and no correction is written.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from scripts.build_anchor_bias_features import (
    DOMINANT_PHYSICS_FEATURES,
    FULL_FEATURES,
    PHYSICAL_FEATURES,
    PRIMARY_FEATURES,
    RESPONSE_FEATURES,
)
from scripts.train_anchor_bias_emulator import (
    CANDIDATES,
    TARGET,
    case_balanced_mean,
    fit_model,
    json_clean,
    metric_summary,
    prediction_calibration,
    quantile_edges,
)


FEATURE_SETS = {
    "dominant_pair_physics": [*PRIMARY_FEATURES, *DOMINANT_PHYSICS_FEATURES],
    "physical_scene": PHYSICAL_FEATURES,
    "response_structure": RESPONSE_FEATURES,
    "full": FULL_FEATURES,
}
POPULATIONS = {
    "carrier_ratio_gt5_positive": True,
    "outside_carrier": False,
}


def gates(metrics: dict, calibration: dict) -> dict:
    reduction = metrics["mse_reduction_model_minus_constant"]
    error = metrics["global_target_minus_prediction"]
    span = calibration["observed_high_minus_low"]
    return {
        "case_paired_mse_reduction_gt2sem": bool(
            reduction["mean"] > 2.0 * reduction["case_sem"]
        ),
        "positive_test_mse_skill": bool(metrics["mse_skill"] > 0.0),
        "observed_prediction_decile_span_gt3sem": bool(
            span["mean"] > 3.0 * span["case_sem"]
        ),
        "prediction_decile_rank_correlation_gt0p8": bool(
            calibration["bin_mean_spearman"] > 0.8
        ),
        "global_calibration_within_2_case_sem": bool(
            abs(error["mean"]) <= 2.0 * error["case_sem"]
        ),
    }


def markdown(payload: dict) -> str:
    lines = [
        "# Carrier-specific anchor bias-emulator ablation",
        "",
        "This is an exploratory follow-up: c700-899 had already been opened by "
        "the global emulator. The carrier is frozen as dominant/runner-up "
        "absolute response >5 with non-negative dominant response. Constgold is "
        "not opened.",
        "",
        "| Population | Feature set | Chosen capacity | Test MSE skill | "
        "MSE reduction +- case SEM | Decile span +- case SEM | rho | Gates |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for population, result in payload["populations"].items():
        for feature_set, item in result["feature_sets"].items():
            metrics = item["test_metrics"]
            reduction = metrics["mse_reduction_model_minus_constant"]
            span = item["test_calibration"]["observed_high_minus_low"]
            lines.append(
                f"| `{population}` | `{feature_set}` | "
                f"`{item['chosen_candidate']}` | {metrics['mse_skill']:.3%} | "
                f"{reduction['mean']:+.5f} +- {reduction['case_sem']:.5f} | "
                f"{span['mean']:+.4f} +- {span['case_sem']:.4f} | "
                f"{item['test_calibration']['bin_mean_spearman']:.3f} | "
                f"{sum(item['gates'].values())}/{len(item['gates'])} |"
            )
    lines.extend([
        "",
        "A physical-root candidate would require the physical-scene model to "
        "pass all gates inside the frozen carrier, not merely show a structured "
        "one-dimensional curve. Response-only success remains localization in "
        "model coordinates, not a physical explanation.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", required=True)
    ap.add_argument("--train-max", type=int, default=599)
    ap.add_argument("--tune-max", type=int, default=699)
    ap.add_argument("--seed", type=int, default=9182)
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--output-md", required=True)
    args = ap.parse_args()
    for output in (args.output_json, args.output_md):
        if os.path.exists(output):
            raise FileExistsError(f"refusing existing output {output}")

    columns = [
        "case", TARGET, "carrier_ratio_gt5_positive", *FULL_FEATURES,
    ]
    frame = pd.read_feather(args.features, columns=columns)
    train_all = frame.loc[frame.case <= args.train_max]
    tune_all = frame.loc[
        frame.case.between(args.train_max + 1, args.tune_max)
    ]
    test_all = frame.loc[frame.case > args.tune_max]
    payload = {
        "design": (
            "exploratory carrier/complement models; capacity selected on c600-699 "
            "separately by population and feature set; c700-899 scored after it "
            "had already been opened by the global emulator"
        ),
        "test_status": "previously opened; exploratory follow-up",
        "carrier_definition": (
            "dominant/runner-up absolute response > 5 and dominant response >= 0"
        ),
        "case_windows": {
            "train": [400, args.train_max],
            "tune": [args.train_max + 1, args.tune_max],
            "test": [args.tune_max + 1, 899],
        },
        "feature_sets": FEATURE_SETS,
        "populations": {},
        "constgold_opened": False,
        "correction_written": False,
    }

    for pop_index, (population, carrier_value) in enumerate(POPULATIONS.items()):
        train = train_all.loc[
            train_all.carrier_ratio_gt5_positive == carrier_value
        ].copy()
        tune = tune_all.loc[
            tune_all.carrier_ratio_gt5_positive == carrier_value
        ].copy()
        test = test_all.loc[
            test_all.carrier_ratio_gt5_positive == carrier_value
        ].copy()
        development = pd.concat([train, tune], ignore_index=True)
        train_constant = case_balanced_mean(
            train[TARGET].to_numpy(float), train.case.to_numpy(np.int64)
        )
        development_constant = case_balanced_mean(
            development[TARGET].to_numpy(float),
            development.case.to_numpy(np.int64),
        )
        pop_result = {
            "rows": {
                "train": int(len(train)), "tune": int(len(tune)),
                "test": int(len(test)),
            },
            "development_constant": development_constant,
            "feature_sets": {},
        }
        print(
            f"{population}: rows={len(train):,}/{len(tune):,}/{len(test):,} "
            f"development_mean={development_constant:+.5f}", flush=True,
        )
        for set_index, (set_name, features) in enumerate(FEATURE_SETS.items()):
            tuning = {}
            tune_predictions = {}
            for candidate_index, (candidate, parameters) in enumerate(
                CANDIDATES.items()
            ):
                model = fit_model(
                    train, features, parameters,
                    args.seed + 10_000 * pop_index + 1_000 * set_index
                    + 100 * candidate_index,
                )
                prediction = model.predict(
                    tune[features].to_numpy(np.float32)
                )
                tuning[candidate] = metric_summary(
                    tune, prediction, train_constant
                )
                tune_predictions[candidate] = prediction
            chosen = min(
                tuning,
                key=lambda candidate: tuning[candidate]["case_balanced_mse"],
            )
            final_model = fit_model(
                development, features, CANDIDATES[chosen],
                args.seed + 100_000 + 10_000 * pop_index + 1_000 * set_index,
            )
            test_prediction = final_model.predict(
                test[features].to_numpy(np.float32)
            )
            metrics = metric_summary(test, test_prediction, development_constant)
            edges = quantile_edges(tune_predictions[chosen], 10)
            calibration = prediction_calibration(test, test_prediction, edges)
            model_gates = gates(metrics, calibration)
            pop_result["feature_sets"][set_name] = {
                "chosen_candidate": chosen,
                "chosen_parameters": CANDIDATES[chosen],
                "tuning_metrics": tuning,
                "test_metrics": metrics,
                "test_calibration": calibration,
                "gates": model_gates,
                "all_gates_pass": bool(all(model_gates.values())),
            }
            reduction = metrics["mse_reduction_model_minus_constant"]
            print(
                f"  {set_name}: {chosen} skill={metrics['mse_skill']:.4%} "
                f"reduction={reduction['mean']:+.5g}+-"
                f"{reduction['case_sem']:.5g} gates="
                f"{sum(model_gates.values())}/{len(model_gates)}",
                flush=True,
            )
        payload["populations"][population] = pop_result

    payload = json_clean(payload)
    with open(args.output_json, "x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    with open(args.output_md, "x", encoding="utf-8") as handle:
        handle.write(markdown(payload))
    print(markdown(payload), flush=True)
    print("ANCHOR_BIAS_SUBGROUP_MODELS_DONE", flush=True)


if __name__ == "__main__":
    main()
