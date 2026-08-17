#!/usr/bin/env python3
"""Evaluate the fixed full-neighbour stack on coherent anchors 400--899.

The new stack is compared with frozen V2.2 and the previous corrected-own
stack.  Gaps are reported both on common populations selected by frozen V2.2
and on each model's own final prediction.  Uncertainties are one SEM across
the 500 independently rendered cases.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


MODELS = {
    "V2.2": "R_blend_v22_replay",
    "Previous corrected-own": "R_blend_corrected_own",
    "Full-neighbour context": "R_blend_fullneighbour_fixed",
}
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


def case_balanced_stat(
    case: np.ndarray, residual: np.ndarray, mask: np.ndarray
) -> dict[str, float | int]:
    selected_case = case[mask]
    selected_residual = residual[mask]
    frame = pd.DataFrame({"case": selected_case, "residual": selected_residual})
    case_mean = frame.groupby("case", sort=True).residual.mean().to_numpy(float)
    if len(case_mean) != 500:
        raise RuntimeError("selection does not cover all 500 rendered cases")
    return {
        "mean": float(case_mean.mean()),
        "case_sem": float(case_mean.std(ddof=1) / np.sqrt(len(case_mean))),
        "n_cases": int(len(case_mean)),
        "n_anchors": int(mask.sum()),
        "row_weighted_mean": float(selected_residual.mean()),
    }


def selection_metrics(
    case: np.ndarray,
    truth: np.ndarray,
    prediction: np.ndarray,
    coordinate: np.ndarray,
) -> dict[str, dict[str, float | int]]:
    residual = truth - prediction
    return {
        name: case_balanced_stat(case, residual, select(coordinate))
        for name, select in SELECTIONS.items()
    }


def paired_gap_change_metrics(
    case: np.ndarray,
    reference_prediction: np.ndarray,
    candidate_prediction: np.ndarray,
    coordinate: np.ndarray,
) -> dict[str, dict[str, float | int]]:
    """Candidate gap minus reference gap on identical selected anchors."""
    paired_change = reference_prediction - candidate_prediction
    return {
        name: case_balanced_stat(case, paired_change, select(coordinate))
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
    parser.add_argument("--previous-score-dir", required=True)
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
            "R_blend_fullneighbour_fixed",
        ],
    )
    previous = load_scores(
        Path(args.previous_score_dir),
        [
            "case", "input_index", "R_blend_v22_replay",
            "R_blend_corrected_own",
        ],
    ).rename(columns={"R_blend_v22_replay": "R_blend_v22_previous"})
    joined = truth.merge(
        new, on=["case", "input_index"], how="left", validate="one_to_one"
    ).merge(
        previous,
        on=["case", "input_index"],
        how="left",
        validate="one_to_one",
    )
    if len(joined) != 1_703_884 or joined.isna().any().any():
        raise RuntimeError("anchor truth/score coverage differs")
    case = joined.case.to_numpy(np.int16)
    if not np.array_equal(np.unique(case), np.arange(400, 900)):
        raise RuntimeError("expected coherent-anchor cases 400--899")
    frozen = joined.scene_prediction.to_numpy(np.float64)
    replay = joined.R_blend_v22_replay.to_numpy(np.float64)
    replay_previous = joined.R_blend_v22_previous.to_numpy(np.float64)
    replay_max = float(max(
        np.max(np.abs(frozen - replay)),
        np.max(np.abs(frozen - replay_previous)),
        np.max(np.abs(replay - replay_previous)),
    ))
    if replay_max > 2.0e-6:
        raise RuntimeError(f"V2.2 replay mismatch {replay_max:.3e}")
    truth_value = joined.R_blend_truth.to_numpy(np.float64)
    prediction = {
        model: joined[column].to_numpy(np.float64)
        for model, column in MODELS.items()
    }
    payload = {
        "schema_version": 1,
        "dataset": "coherent_anchor_c400_899",
        "n_anchors": int(len(joined)),
        "n_cases": 500,
        "gap_definition": "R_blend_truth - R_blend_prediction",
        "uncertainty": "one SEM across per-case conditional means",
        "fixed_v22_population_gaps": {
            model: selection_metrics(
                case, truth_value, local_prediction, frozen
            )
            for model, local_prediction in prediction.items()
        },
        "own_final_prediction_population_gaps": {
            model: selection_metrics(
                case, truth_value, local_prediction, local_prediction
            )
            for model, local_prediction in prediction.items()
        },
        "paired_fixed_v22_population_gap_changes": {
            "Full-neighbour context minus V2.2": paired_gap_change_metrics(
                case,
                prediction["V2.2"],
                prediction["Full-neighbour context"],
                frozen,
            ),
            "Full-neighbour context minus Previous corrected-own": (
                paired_gap_change_metrics(
                    case,
                    prediction["Previous corrected-own"],
                    prediction["Full-neighbour context"],
                    frozen,
                )
            ),
        },
        "audit": {
            "v22_replay_max_abs": replay_max,
            "new_model_tuned_without_coherent_anchor_truth": True,
            "new_model_tuning_performed": False,
        },
    }
    output = Path(args.output_json).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    print("ANCHOR_FULLNEIGHBOUR_TRANSFER_EVALUATION_DONE", flush=True)


if __name__ == "__main__":
    main()
