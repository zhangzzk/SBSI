"""Train the frozen scalar scene recipe on coherent anchors, validate unseen cases.

This is an exploratory simulation-only improvement attempt.  Cases 0--199
train five case-blocked HGB models; cases 200--299 are scored once.  The
feature list and HGB hyperparameters are imported unchanged from the earlier
random-local scene correction.  No constgold data are read and no deployable
model is written.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd


HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
BE = "/home/z/Zekang.Zhang/blendemu"
if BE not in sys.path:
    sys.path.insert(0, BE)

from blendemu.inference import BlendingPredictor  # noqa: E402
from fit_anchorblend_scene_correction import (  # noqa: E402
    COND, KEY, MODEL_FEATURES, aggregate_scene_features,
    fit_case_blocked_ensemble, stat,
)


FEATURE_SETS = {
    "full": MODEL_FEATURES,
    "primary_plus": ["r_primary", "log_Re_primary", "log_n_primary", "R_v22"],
}


def summarize(frame: pd.DataFrame, correction: np.ndarray,
              constant: float, edges: np.ndarray) -> dict:
    gap = (frame.R_blend_lsst_r_extnbr_v22 - frame.R_blend_truth).to_numpy(float)
    work = pd.DataFrame({
        "case": frame.case.to_numpy(int),
        "raw": gap,
        "corrected": gap + correction,
        "constant": gap + constant,
        "prediction": frame.R_blend_lsst_r_extnbr_v22.to_numpy(float),
    })
    case = work.groupby("case", sort=True)[["raw", "corrected", "constant"]].mean()
    bins = np.clip(np.searchsorted(edges, work.prediction, side="right") - 1, 0, len(edges) - 2)
    work["bin"] = bins
    conditional = {}
    for index in range(len(edges) - 1):
        subset = work[work.bin == index]
        by_case = subset.groupby("case", sort=True)[["raw", "corrected", "constant"]].mean()
        conditional[str(index)] = {
            "lo": None if not np.isfinite(edges[index]) else float(edges[index]),
            "hi": None if not np.isfinite(edges[index + 1]) else float(edges[index + 1]),
            "n_rows": int(len(subset)),
            "raw": stat(by_case.raw),
            "corrected": stat(by_case.corrected),
            "constant": stat(by_case.constant),
        }
    return {
        "n_rows": int(len(work)), "n_cases": int(len(case)),
        "raw": stat(case.raw), "corrected": stat(case.corrected),
        "constant": stat(case.constant),
        "paired_scene_correction": stat(case.corrected - case.raw),
        "paired_constant_correction": stat(case.constant - case.raw),
        "conditional_prediction_quintiles": conditional,
        "max_abs_conditional_raw": float(max(abs(x["raw"]["mean"]) for x in conditional.values())),
        "max_abs_conditional_corrected": float(max(
            abs(x["corrected"]["mean"]) for x in conditional.values()
        )),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", action="append", required=True)
    parser.add_argument("--response", action="append", required=True)
    parser.add_argument("--development-max", type=int, default=199)
    parser.add_argument("--validation-min", type=int, default=200)
    parser.add_argument("--validation-max", type=int, default=299)
    parser.add_argument("--fresh-validation", action="store_true")
    parser.add_argument(
        "--candidate-parts-root",
        help=(
            "optional root containing <candidate-tag>/caseN.feather files; "
            "fit the unchanged scene recipe to truth minus prediction_model"
        ),
    )
    parser.add_argument("--candidate-tag")
    parser.add_argument("--feature-set", choices=sorted(FEATURE_SETS), default="full")
    parser.add_argument("--sign", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=311)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if os.path.exists(args.output):
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if len(args.base) != len(args.response):
        raise RuntimeError("each --response requires its corresponding --base")
    if bool(args.candidate_parts_root) != bool(args.candidate_tag):
        raise RuntimeError("--candidate-parts-root and --candidate-tag are required together")
    observed_parts = [pd.read_feather(path) for path in args.response]
    keep_parts = []
    for part in observed_parts:
        keep_parts.append(part[
            (part.case <= args.development_max)
            | part.case.between(args.validation_min, args.validation_max)
        ].copy())
    observed = pd.concat(keep_parts, ignore_index=True)
    expected_cases = (
        list(range(0, args.development_max + 1))
        + list(range(args.validation_min, args.validation_max + 1))
    )
    if observed.duplicated(KEY).any() or sorted(observed.case.unique()) != expected_cases:
        raise RuntimeError(f"response inputs do not provide expected cases {expected_cases}")
    predictor = BlendingPredictor.load(
        os.path.join(BE, "models"), tag="lsst_r_extnbr_v22",
        conditions=COND, device="cpu",
    )
    frame = pd.concat([
        aggregate_scene_features(base, part, predictor, args.sign)
        for base, part in zip(args.base, keep_parts) if len(part)
    ], ignore_index=True)
    candidate_source = None
    if args.candidate_parts_root:
        paths = [
            os.path.join(
                args.candidate_parts_root, args.candidate_tag, f"case{case}.feather",
            )
            for case in expected_cases
        ]
        missing = [path for path in paths if not os.path.isfile(path)]
        if missing:
            raise RuntimeError(f"candidate score is missing {len(missing)} case parts")
        scored = pd.concat([
            pd.read_feather(path, columns=KEY + [
                "prediction_model", "prediction_baseline",
            ])
            for path in paths
        ], ignore_index=True)
        if scored.duplicated(KEY).any():
            raise RuntimeError("candidate score has duplicate anchor keys")
        frame = frame.merge(scored, on=KEY, how="left", validate="one_to_one")
        if frame.prediction_model.isna().any():
            raise RuntimeError("candidate score does not cover every response anchor")
        baseline_error = float(np.max(np.abs(
            frame.prediction_baseline.to_numpy(float)
            - frame.R_blend_lsst_r_extnbr_v22.to_numpy(float)
        )))
        if baseline_error > 5e-7:
            raise RuntimeError(
                f"candidate baseline replay failed, max abs={baseline_error:.3e}"
            )
        frame["R_blend_lsst_r_extnbr_v22"] = frame.prediction_model.to_numpy(float)
        candidate_source = {
            "tag": args.candidate_tag,
            "parts_root": args.candidate_parts_root,
            "baseline_scene_features_retained": True,
            "baseline_replay_max_abs": baseline_error,
        }
    correction, target, stability = fit_case_blocked_ensemble(
        frame, args.development_max, args.seed,
        model_features=FEATURE_SETS[args.feature_set],
    )
    development = frame.case.to_numpy(int) <= args.development_max
    validation = ~development
    dev_case_target = pd.DataFrame({
        "case": frame.loc[development, "case"].to_numpy(int),
        "target": target[development],
    }).groupby("case", sort=True).target.mean()
    constant = float(dev_case_target.mean())
    edges = np.quantile(
        frame.loc[development, "R_blend_lsst_r_extnbr_v22"].to_numpy(float),
        np.linspace(0.0, 1.0, 6),
    )
    edges[0], edges[-1] = -np.inf, np.inf
    development_summary = summarize(
        frame.loc[development], correction[development], constant, edges,
    )
    validation_summary = summarize(
        frame.loc[validation], correction[validation], constant, edges,
    )
    validation_midpoint = (args.validation_min + args.validation_max) // 2
    validation_first = validation & (frame.case.to_numpy(int) <= validation_midpoint)
    validation_second = validation & (frame.case.to_numpy(int) > validation_midpoint)
    validation_first_summary = summarize(
        frame.loc[validation_first], correction[validation_first], constant, edges,
    )
    validation_second_summary = summarize(
        frame.loc[validation_second], correction[validation_second], constant, edges,
    )
    gates = {
        "validation_global_abs_smaller": bool(
            abs(validation_summary["corrected"]["mean"])
            < abs(validation_summary["raw"]["mean"])
        ),
        "validation_max_conditional_abs_smaller": bool(
            validation_summary["max_abs_conditional_corrected"]
            < validation_summary["max_abs_conditional_raw"]
        ),
        "validation_global_within_2_case_sem": bool(
            abs(validation_summary["corrected"]["mean"])
            <= 2.0 * np.hypot(validation_summary["corrected"]["case_sem"], stability)
        ),
        "training_stability_below_0p003": bool(stability < 0.003),
    }
    payload = {
        "candidate": (
            "frozen HGB scalar scene correction stacked on fixed pair candidate"
            if candidate_source else
            "frozen HGB scalar scene correction trained directly on coherent neighbour response"
        ),
        "candidate_prediction_source": candidate_source,
        "feature_set": args.feature_set,
        "features": FEATURE_SETS[args.feature_set],
        "development_cases": [0, args.development_max],
        "validation_cases": [args.validation_min, args.validation_max],
        "validation_previously_inspected": bool(not args.fresh_validation),
        "constant_case_weighted_development_correction": constant,
        "ensemble_validation_mean_stability_sd": stability,
        "development_oof": development_summary,
        "validation": validation_summary,
        "validation_first_half": validation_first_summary,
        "validation_second_half": validation_second_summary,
        "gates": gates, "gate_passed": bool(all(gates.values())),
        "deployable_model_written": False,
        "constgold_opened": False,
    }
    with open(args.output, "x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    print("ANCHORBLEND_COHERENT_SCENE_CORRECTION_DONE", flush=True)


if __name__ == "__main__":
    main()
