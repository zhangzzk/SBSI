#!/usr/bin/env python3
"""Test whether the small-radius response is predictable from flow context.

This is a diagnostic control, not a replacement calibration model.  A flexible
tree regressor is trained on the same declared truth/blending contexts as the
flow to predict the individual measured projected response.  A second oracle
regressor additionally receives the realised g=0 FLUX_RADIUS.  Comparing their
radius-binned predictions on unseen objects and held-out cases distinguishes a
truth-context capacity/optimization failure from response information carried
primarily by the realised measurement.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import roc_auc_score

from sbsi.fixed_g0_domain import FLOW_FEATURES, matched_key_indices


DEFAULT_EDGES_PIXELS = (3.0, 3.077, 3.149, 3.218, 3.286, math.inf)


def load_leg(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as stored:
        return {name: stored[name] for name in stored.files}


def case_arrays(domain_root: Path, case: int) -> dict[str, np.ndarray]:
    """Return aligned truth-only features, response, and realised g=0 radius."""

    zero = load_leg(domain_root / "flow" / "g0" / f"case{case:03d}.npz")
    sheared = load_leg(domain_root / "flow" / "g05" / f"case{case:03d}.npz")
    left, right, _ = matched_key_indices(
        zero["case"], zero["input_index"], sheared["case"], sheared["input_index"]
    )
    gamma = np.asarray(sheared["gamma"][right], dtype=np.float64)
    delta = np.asarray(sheared["target"][right, :2], dtype=np.float64)
    delta -= np.asarray(zero["target"][left, :2], dtype=np.float64)
    squared = np.einsum("ij,ij->i", gamma, gamma)
    keep = np.isfinite(gamma).all(axis=1) & (squared > 0.0)
    keep &= np.isfinite(delta).all(axis=1)
    left, right = left[keep], right[keep]
    gamma, delta, squared = gamma[keep], delta[keep], squared[keep]
    features = np.column_stack(
        (
            np.asarray(zero["context"][left], dtype=np.float32),
            np.asarray(sheared["context"][right], dtype=np.float32),
            gamma.astype(np.float32),
        )
    )
    response = np.einsum("ij,ij->i", delta, gamma) / squared
    radius = np.asarray(zero["target"][left, 2], dtype=np.float64)
    if not np.isfinite(features).all() or not np.isfinite(response).all():
        raise RuntimeError(f"case {case}: non-finite diagnostic row")
    if np.any(radius <= 3.0):
        raise RuntimeError(f"case {case}: strict selected-domain radius violated")
    return {
        "features": features,
        "response": response.astype(np.float32),
        "radius": radius.astype(np.float32),
    }


def sampled_case(
    arrays: dict[str, np.ndarray],
    *,
    fit_rows: int,
    evaluation_rows: int,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Draw disjoint deterministic fit/evaluation subsets from one case."""

    count = len(arrays["response"])
    fit_count = min(int(fit_rows), count)
    evaluation_count = min(int(evaluation_rows), count - fit_count)
    order = np.random.default_rng(seed).permutation(count)

    def take(indices):
        return {name: values[indices] for name, values in arrays.items()}

    return take(order[:fit_count]), take(order[fit_count : fit_count + evaluation_count])


def concatenate(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not parts:
        raise ValueError("at least one case part is required")
    return {name: np.concatenate([part[name] for part in parts]) for name in parts[0]}


def bin_labels(edges: tuple[float, ...]) -> list[str]:
    return [
        (f"[{left:.3f},{right:.3f})" if math.isfinite(right) else f">={left:.3f}")
        for left, right in zip(edges[:-1], edges[1:])
    ]


def case_bin_summary(
    case: np.ndarray,
    radius: np.ndarray,
    measured: np.ndarray,
    predictions: dict[str, np.ndarray],
    edges: tuple[float, ...],
) -> list[dict]:
    """Return equal-case response means and paired residual SEMs by radius."""

    labels = bin_labels(edges)
    assignment = np.searchsorted(np.asarray(edges), radius, side="right") - 1
    frame = pd.DataFrame(
        {
            "case": case,
            "bin": assignment,
            "measured": measured,
            **predictions,
        }
    )
    grouped = frame.groupby(["case", "bin"], sort=True)
    means = grouped[["measured", *predictions]].mean()
    counts = grouped.size().rename("objects")
    rows = []
    for index, label in enumerate(labels):
        if index not in means.index.get_level_values("bin"):
            continue
        local = means.xs(index, level="bin", drop_level=False)
        local_counts = counts.xs(index, level="bin", drop_level=False)
        measured_case = local["measured"].to_numpy(np.float64)
        for name in predictions:
            predicted_case = local[name].to_numpy(np.float64)
            residual = measured_case - predicted_case
            rows.append(
                {
                    "radius_bin_pixels": label,
                    "model": name,
                    "cases": int(len(local)),
                    "objects": int(local_counts.sum()),
                    "measured_response": float(measured_case.mean()),
                    "predicted_response": float(predicted_case.mean()),
                    "measured_minus_predicted": float(residual.mean()),
                    "residual_case_sem": (
                        float(residual.std(ddof=1) / np.sqrt(len(residual))) if len(residual) > 1 else None
                    ),
                }
            )
    return rows


def model_parameters(seed: int, threads: int) -> dict:
    return {
        "learning_rate": 0.05,
        "num_leaves": 127,
        "min_child_samples": 500,
        "n_estimators": 400,
        "reg_lambda": 1.0,
        "max_bin": 255,
        "random_state": seed,
        "n_jobs": threads,
        "deterministic": True,
        "force_col_wise": True,
        "verbosity": -1,
    }


def regression_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    residual = np.asarray(target, dtype=np.float64) - np.asarray(prediction, dtype=np.float64)
    centered = np.asarray(target, dtype=np.float64) - float(np.mean(target))
    return {
        "rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "r2": float(1.0 - np.sum(np.square(residual)) / np.sum(np.square(centered))),
        "mean_residual": float(residual.mean()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--fit-rows-per-case", type=int, default=15000)
    parser.add_argument("--train-evaluation-rows-per-case", type=int, default=5000)
    parser.add_argument("--validation-rows-per-case", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--threads", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite {output_root}")
    output_root.mkdir(parents=True)
    manifest_path = args.domain_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    train_cases = tuple(map(int, manifest["split"]["train_cases"]))
    validation_cases = tuple(map(int, manifest["split"]["validation_cases"]))

    fit_parts, train_evaluation_parts = [], []
    for offset, case in enumerate(train_cases):
        arrays = case_arrays(args.domain_root, case)
        fit, evaluation = sampled_case(
            arrays,
            fit_rows=args.fit_rows_per_case,
            evaluation_rows=args.train_evaluation_rows_per_case,
            seed=args.seed + 1_000_003 * offset,
        )
        fit_parts.append(fit)
        evaluation["case"] = np.full(len(evaluation["response"]), case, np.int16)
        train_evaluation_parts.append(evaluation)
        if (offset + 1) % 20 == 0:
            print(f"loaded training cases {offset + 1}/{len(train_cases)}", flush=True)
    fit = concatenate(fit_parts)
    train_evaluation = concatenate(train_evaluation_parts)

    validation_parts = []
    for offset, case in enumerate(validation_cases):
        arrays = case_arrays(args.domain_root, case)
        _, evaluation = sampled_case(
            arrays,
            fit_rows=0,
            evaluation_rows=args.validation_rows_per_case,
            seed=args.seed + 500_000_003 + 1_000_003 * offset,
        )
        evaluation["case"] = np.full(len(evaluation["response"]), case, np.int16)
        validation_parts.append(evaluation)
        if (offset + 1) % 10 == 0:
            print(
                f"loaded validation cases {offset + 1}/{len(validation_cases)}",
                flush=True,
            )
    validation = concatenate(validation_parts)

    parameters = model_parameters(args.seed, args.threads)
    truth_only = lgb.LGBMRegressor(objective="regression_l2", **parameters)
    truth_only.fit(fit["features"], fit["response"])
    truth_plus_radius = lgb.LGBMRegressor(objective="regression_l2", **parameters)
    truth_plus_radius.fit(np.column_stack((fit["features"], fit["radius"])), fit["response"])
    membership = lgb.LGBMClassifier(objective="binary", **parameters)
    membership.fit(fit["features"], fit["radius"] < DEFAULT_EDGES_PIXELS[1])

    truth_only.booster_.save_model(output_root / "truth_only_response.txt")
    truth_plus_radius.booster_.save_model(output_root / "truth_plus_radius_response.txt")
    membership.booster_.save_model(output_root / "truth_only_first_bin_classifier.txt")

    evaluations = {}
    for split, values in (
        ("unseen_training_objects", train_evaluation),
        ("held_out_cases", validation),
    ):
        truth_prediction = truth_only.predict(values["features"])
        radius_prediction = truth_plus_radius.predict(np.column_stack((values["features"], values["radius"])))
        probability = membership.predict_proba(values["features"])[:, 1]
        evaluations[split] = {
            "rows": int(len(values["response"])),
            "truth_only_metrics": regression_metrics(values["response"], truth_prediction),
            "truth_plus_radius_metrics": regression_metrics(values["response"], radius_prediction),
            "truth_only_first_bin_auc": float(
                roc_auc_score(values["radius"] < DEFAULT_EDGES_PIXELS[1], probability)
            ),
            "radius_bins": case_bin_summary(
                values["case"],
                values["radius"],
                values["response"],
                {
                    "truth_only": truth_prediction,
                    "truth_plus_radius": radius_prediction,
                },
                DEFAULT_EDGES_PIXELS,
            ),
        }

    result = {
        "format_version": 1,
        "purpose": "truth-only predictability control for the selected small-radius response",
        "domain_root": str(args.domain_root.resolve()),
        "domain_manifest": str(manifest_path.resolve()),
        "train_cases": list(train_cases),
        "validation_cases": list(validation_cases),
        "feature_names": [
            *[f"g0_{name}" for name in FLOW_FEATURES],
            *[f"g05_{name}" for name in FLOW_FEATURES],
            "gamma1",
            "gamma2",
        ],
        "fit_rows": int(len(fit["response"])),
        "fit_rows_per_case": int(args.fit_rows_per_case),
        "train_evaluation_rows_per_case": int(args.train_evaluation_rows_per_case),
        "validation_rows_per_case": int(args.validation_rows_per_case),
        "radius_edges_pixels": [
            None if not math.isfinite(value) else value for value in DEFAULT_EDGES_PIXELS
        ],
        "lightgbm_version": lgb.__version__,
        "sklearn_version": sklearn.__version__,
        "model_parameters": parameters,
        "evaluations": evaluations,
        "interpretation_contract": [
            "The truth-only regressor is more direct than the flow but uses no measured condition.",
            "The truth-plus-radius regressor is a diagnostic oracle and is not a proposed model.",
            "Agreement by the oracle but not the truth-only model localizes information in the realised radius after conditioning on declared truth features.",
            "Failure of both models would not distinguish inadequate features from inadequate tree capacity.",
        ],
    }
    (output_root / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for split, values in evaluations.items():
        print(f"\n### {split}", flush=True)
        print(
            json.dumps({key: value for key, value in values.items() if key != "radius_bins"}, indent=2),
            flush=True,
        )
        print(pd.DataFrame(values["radius_bins"]).to_string(index=False), flush=True)
    print(f"TRUTH_ONLY_BOUNDARY_COMPLETE output={output_root}", flush=True)


if __name__ == "__main__":
    main()
