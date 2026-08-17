#!/usr/bin/env python3
"""Compare the new-base old-way residual stack with V2.2 and the early stack."""

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
        raise RuntimeError("selection does not cover all anchor cases")
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-score-dir", required=True)
    parser.add_argument("--early-score-dir", required=True)
    parser.add_argument("--anchor-features", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    truth = pd.read_feather(
        args.anchor_features,
        columns=["case", "input_index", "R_blend_truth", "scene_prediction"],
    )
    new = load_scores(
        Path(args.new_score_dir),
        [
            "case", "input_index", "R_blend_v22_replay",
            "R_blend_newbase_oldway_base",
            "R_blend_newbase_oldway_correction",
            "R_blend_newbase_oldway",
        ],
    )
    early = load_scores(
        Path(args.early_score_dir),
        [
            "case", "input_index", "R_blend_v22_replay",
            "grouped_pair_correction_sum", "R_blend_grouped",
        ],
    ).rename(columns={"R_blend_v22_replay": "R_blend_v22_early"})
    joined = truth.merge(
        new, on=["case", "input_index"], how="left", validate="one_to_one"
    ).merge(
        early, on=["case", "input_index"], how="left", validate="one_to_one"
    )
    if len(joined) != 1_703_884 or joined.isna().any().any():
        raise RuntimeError("anchor truth/score coverage differs")
    case = joined.case.to_numpy(np.int16)
    frozen = joined.scene_prediction.to_numpy(np.float64)
    replay = joined.R_blend_v22_replay.to_numpy(np.float64)
    replay_early = joined.R_blend_v22_early.to_numpy(np.float64)
    replay_max = float(max(
        np.max(np.abs(frozen - replay)),
        np.max(np.abs(frozen - replay_early)),
        np.max(np.abs(replay - replay_early)),
    ))
    if replay_max > 2.0e-6:
        raise RuntimeError(f"V2.2 replay mismatch {replay_max:.3e}")
    truth_value = joined.R_blend_truth.to_numpy(np.float64)
    predictions = {
        "V2.2": replay,
        "Early V2.2 residual": joined.R_blend_grouped.to_numpy(np.float64),
        "New-base old-way residual": joined.R_blend_newbase_oldway.to_numpy(
            np.float64
        ),
    }
    gaps = {
        model: truth_value - prediction
        for model, prediction in predictions.items()
    }
    payload = {
        "schema_version": 1,
        "dataset": "coherent_anchor_c400_899",
        "n_anchors": int(len(joined)),
        "n_cases": 500,
        "gap_definition": "R_blend_truth - R_blend_prediction",
        "uncertainty": "one SEM across per-case conditional means",
        "fixed_v22_population_gaps": {
            model: selected(case, gap, frozen) for model, gap in gaps.items()
        },
        "own_final_prediction_population_gaps": {
            model: selected(case, gaps[model], prediction)
            for model, prediction in predictions.items()
        },
        "paired_fixed_v22_population_gap_changes": {
            "New-base old-way minus V2.2": selected(
                case, gaps["New-base old-way residual"] - gaps["V2.2"], frozen
            ),
            "New-base old-way minus Early V2.2 residual": selected(
                case,
                gaps["New-base old-way residual"]
                - gaps["Early V2.2 residual"],
                frozen,
            ),
        },
        "applied_corrections": {
            "Early V2.2 residual": selected(
                case,
                joined.grouped_pair_correction_sum.to_numpy(np.float64),
                frozen,
            ),
            "New-base old-way residual": selected(
                case,
                joined.R_blend_newbase_oldway_correction.to_numpy(np.float64),
                frozen,
            ),
        },
        "new_base_minus_v22_scene_prediction": selected(
            case,
            joined.R_blend_newbase_oldway_base.to_numpy(np.float64) - replay,
            frozen,
        ),
        "audit": {
            "v22_replay_max_abs": replay_max,
            "anchor_truth_used_for_training_or_selection": False,
        },
    }
    output = Path(args.output_json).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    print("ANCHOR_NEWBASE_OLDWAY_EVALUATION_DONE", flush=True)


if __name__ == "__main__":
    main()
