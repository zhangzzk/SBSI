#!/usr/bin/env python3
"""Evaluate the two-by-two split/context ablation on coherent anchors."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


SELECTIONS = {
    "all": lambda value: np.ones(len(value), dtype=bool),
    "le_0p1": lambda value: value <= 0.1,
    "gt_0p1": lambda value: value > 0.1,
    "gt_0p2": lambda value: value > 0.2,
}


def load_scores(directory: Path, columns: list[str]) -> pd.DataFrame:
    return pd.concat(
        [
            pd.read_feather(directory / f"case{case}.feather", columns=columns)
            for case in range(400, 900)
        ],
        ignore_index=True,
    )


def stat(case: np.ndarray, value: np.ndarray, mask: np.ndarray) -> dict:
    frame = pd.DataFrame({"case": case[mask], "value": value[mask]})
    case_mean = frame.groupby("case", sort=True).value.mean().to_numpy(float)
    if len(case_mean) != 500:
        raise RuntimeError("selection does not cover every coherent-anchor case")
    return {
        "mean": float(case_mean.mean()),
        "case_sem": float(case_mean.std(ddof=1) / np.sqrt(len(case_mean))),
        "n_cases": int(len(case_mean)),
        "n_anchors": int(mask.sum()),
        "row_weighted_mean": float(np.asarray(value)[mask].mean()),
    }


def selected(case: np.ndarray, value: np.ndarray, coordinate: np.ndarray) -> dict:
    return {
        name: stat(case, value, select(coordinate))
        for name, select in SELECTIONS.items()
    }


def atomic_json(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(f"refusing existing output {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def correction_training_record(path: Path) -> dict:
    summary = json.loads(path.read_text(encoding="utf-8"))
    correction = summary["training"]["correction"]
    context = summary["source"].get("scene_context_population")
    if context is None:
        context = (
            "full"
            if "all deployed neighbours" in correction["scene_context"]
            else "half"
        )
    return {
        "summary": str(path),
        "fit_case_window": summary["training"]["case_window"],
        "scene_context_population": context,
        "n_rows": correction["n_rows"],
        "residual_mean_before": correction["before_metrics"]["residual_mean"],
        "residual_mean_after": correction["after_metrics"]["residual_mean"],
        "mse_before": correction["before_metrics"]["mse"],
        "mse_after": correction["after_metrics"]["mse"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--missing-score-dir", required=True)
    parser.add_argument("--orig-half-score-dir", required=True)
    parser.add_argument("--newsplit-full-score-dir", required=True)
    parser.add_argument("--anchor-features", required=True)
    parser.add_argument("--orig-half-summary", required=True)
    parser.add_argument("--orig-full-summary", required=True)
    parser.add_argument("--newsplit-half-summary", required=True)
    parser.add_argument("--newsplit-full-summary", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    truth = pd.read_feather(
        args.anchor_features,
        columns=["case", "input_index", "R_blend_truth", "scene_prediction"],
    )
    missing = load_scores(
        Path(args.missing_score_dir),
        [
            "case", "input_index", "R_blend_v22_replay",
            "R_blend_orig_full_base", "R_blend_orig_full_correction",
            "R_blend_orig_full", "R_blend_newsplit_half_base",
            "R_blend_newsplit_half_correction", "R_blend_newsplit_half",
        ],
    )
    orig_half = load_scores(
        Path(args.orig_half_score_dir),
        [
            "case", "input_index", "R_blend_v22_replay",
            "R_blend_newbase_oldway_base",
            "R_blend_newbase_oldway_correction", "R_blend_newbase_oldway",
        ],
    ).rename(columns={
        "R_blend_v22_replay": "R_blend_v22_orig_half",
        "R_blend_newbase_oldway_base": "R_blend_orig_half_base",
        "R_blend_newbase_oldway_correction": "R_blend_orig_half_correction",
        "R_blend_newbase_oldway": "R_blend_orig_half",
    })
    newsplit_full = load_scores(
        Path(args.newsplit_full_score_dir),
        [
            "case", "input_index", "R_blend_v22_replay",
            "R_blend_newbase_oldway_base",
            "R_blend_newbase_oldway_correction", "R_blend_newbase_oldway",
        ],
    ).rename(columns={
        "R_blend_v22_replay": "R_blend_v22_newsplit_full",
        "R_blend_newbase_oldway_base": "R_blend_newsplit_full_base",
        "R_blend_newbase_oldway_correction": "R_blend_newsplit_full_correction",
        "R_blend_newbase_oldway": "R_blend_newsplit_full",
    })
    joined = truth.merge(
        missing, on=["case", "input_index"], how="left", validate="one_to_one"
    ).merge(
        orig_half, on=["case", "input_index"], how="left", validate="one_to_one"
    ).merge(
        newsplit_full,
        on=["case", "input_index"], how="left", validate="one_to_one",
    )
    if len(joined) != 1_703_884 or joined.isna().any().any():
        raise RuntimeError("anchor truth/score coverage differs")

    replay_columns = [
        "R_blend_v22_replay", "R_blend_v22_orig_half",
        "R_blend_v22_newsplit_full",
    ]
    frozen = joined.scene_prediction.to_numpy(np.float64)
    replay_max = max(
        float(np.max(np.abs(joined[column].to_numpy(np.float64) - frozen)))
        for column in replay_columns
    )
    if replay_max > 2.0e-6:
        raise RuntimeError(f"V2.2 replay mismatch {replay_max:.3e}")

    case = joined.case.to_numpy(np.int16)
    truth_value = joined.R_blend_truth.to_numpy(np.float64)
    predictions = {
        "V2.2": frozen,
        "original_split_half_context": joined.R_blend_orig_half.to_numpy(float),
        "original_split_full_context": joined.R_blend_orig_full.to_numpy(float),
        "new_split_half_context": joined.R_blend_newsplit_half.to_numpy(float),
        "new_split_full_context": joined.R_blend_newsplit_full.to_numpy(float),
    }
    gaps = {
        name: truth_value - prediction for name, prediction in predictions.items()
    }
    fixed = {
        name: selected(case, gap, frozen) for name, gap in gaps.items()
    }
    effects = {
        "full_minus_half_original_split": selected(
            case,
            gaps["original_split_full_context"]
            - gaps["original_split_half_context"],
            frozen,
        ),
        "full_minus_half_new_split": selected(
            case,
            gaps["new_split_full_context"] - gaps["new_split_half_context"],
            frozen,
        ),
        "new_minus_original_half_context": selected(
            case,
            gaps["new_split_half_context"]
            - gaps["original_split_half_context"],
            frozen,
        ),
        "new_minus_original_full_context": selected(
            case,
            gaps["new_split_full_context"]
            - gaps["original_split_full_context"],
            frozen,
        ),
    }
    corrections = {
        name: selected(
            case, joined[f"R_blend_{name}_correction"].to_numpy(float), frozen
        )
        for name in (
            "orig_half", "orig_full", "newsplit_half", "newsplit_full"
        )
    }
    payload = {
        "schema_version": 1,
        "dataset": "coherent_anchor_c400_899",
        "n_anchors": int(len(joined)),
        "n_cases": 500,
        "gap_definition": "R_blend_truth - R_blend_prediction",
        "uncertainty": "one SEM across per-case conditional means",
        "fixed_v22_population_gaps": fixed,
        "own_final_prediction_population_gaps": {
            name: selected(case, gaps[name], prediction)
            for name, prediction in predictions.items()
        },
        "paired_effects_on_fixed_v22_populations": effects,
        "applied_corrections_on_fixed_v22_populations": corrections,
        "base_reproduction": {
            "original_split_max_abs_half_vs_full": float(np.max(np.abs(
                joined.R_blend_orig_half_base.to_numpy(float)
                - joined.R_blend_orig_full_base.to_numpy(float)
            ))),
            "new_split_max_abs_half_vs_full": float(np.max(np.abs(
                joined.R_blend_newsplit_half_base.to_numpy(float)
                - joined.R_blend_newsplit_full_base.to_numpy(float)
            ))),
        },
        "correction_training": {
            "original_split_half_context": correction_training_record(
                Path(args.orig_half_summary)
            ),
            "original_split_full_context": correction_training_record(
                Path(args.orig_full_summary)
            ),
            "new_split_half_context": correction_training_record(
                Path(args.newsplit_half_summary)
            ),
            "new_split_full_context": correction_training_record(
                Path(args.newsplit_full_summary)
            ),
        },
        "audit": {
            "v22_replay_max_abs": replay_max,
            "anchor_truth_used_for_training_or_selection": False,
        },
    }
    output = Path(args.output_json).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    print("ANCHOR_NEWBASE_OLDWAY_2X2_EVALUATION_DONE", flush=True)


if __name__ == "__main__":
    main()
