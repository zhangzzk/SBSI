#!/usr/bin/env python3
"""Evaluate frozen sequential R_blend models on coherent anchors 400--899."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


MODELS = {
    "V2.2": "R_blend_v22_replay",
    "Tuned pair objective": "R_blend_pair",
    "Tuned pair+scene objective": "R_blend_pair_scene",
}


def finite_stat(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return {
        "mean": float(values.mean()),
        "case_sem": float(values.std(ddof=1) / np.sqrt(len(values))),
        "n_cases": int(len(values)),
    }


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-dir", required=True)
    parser.add_argument("--anchor-features", required=True)
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()

    score_dir = Path(args.score_dir).resolve()
    anchor_path = Path(args.anchor_features).resolve()
    output_prefix = Path(args.output_prefix).resolve()
    output_json = output_prefix.with_suffix(".json")
    output_csv = output_prefix.with_suffix(".case_metrics.csv")
    if output_json.exists() or output_csv.exists():
        raise FileExistsError("refusing existing evaluation output")

    score = pd.concat(
        [
            pd.read_feather(score_dir / f"case{case}.feather")
            for case in range(400, 900)
        ],
        ignore_index=True,
    )
    truth = pd.read_feather(
        anchor_path,
        columns=[
            "case", "input_index", "R_blend_truth", "scene_prediction",
            "log1p_n_pairs",
        ],
    )
    joined = truth.merge(
        score, on=["case", "input_index"], how="left", validate="one_to_one"
    )
    if len(joined) != 1_703_884 or joined.isna().any().any():
        raise RuntimeError("anchor score coverage differs from truth table")
    replay_error = float(np.max(np.abs(
        joined.R_blend_v22_replay.to_numpy(float)
        - joined.scene_prediction.to_numpy(float)
    )))
    if replay_error > 2.0e-6:
        raise RuntimeError(f"V2.2 replay differs by {replay_error:.3e}")
    expected_n = np.rint(np.expm1(joined.log1p_n_pairs)).astype(np.int16)
    if not np.array_equal(expected_n, joined.n_pairs.to_numpy(np.int16)):
        raise RuntimeError("pair multiplicity differs from truth table")

    coordinate = joined.scene_prediction.to_numpy(float)
    selections = {
        "all": np.ones(len(joined), dtype=bool),
        "outside_le_0p1": coordinate <= 0.1,
        "tail_gt_0p1": coordinate > 0.1,
        "tail_gt_0p2": coordinate > 0.2,
    }
    truth_value = joined.R_blend_truth.to_numpy(float)
    records: list[dict] = []
    summary: dict[str, dict] = {}
    for model_name, prediction_column in MODELS.items():
        prediction = joined[prediction_column].to_numpy(float)
        residual = truth_value - prediction
        summary[model_name] = {}
        for selection, mask in selections.items():
            case_frame = pd.DataFrame({
                "case": joined.loc[mask, "case"].to_numpy(np.int16),
                "residual": residual[mask],
            })
            case_mean = case_frame.groupby("case", sort=True).residual.mean()
            stats = finite_stat(case_mean.to_numpy(float))
            stats["n_scenes"] = int(mask.sum())
            stats["row_weighted_mean"] = float(residual[mask].mean())
            summary[model_name][selection] = stats
            for case, value in case_mean.items():
                records.append({
                    "model": model_name,
                    "selection": selection,
                    "case": int(case),
                    "truth_minus_prediction": float(value),
                    "n_scenes": int(
                        np.sum(mask & (joined.case.to_numpy() == case))
                    ),
                })

    payload = {
        "schema_version": 1,
        "dataset": "coherent_anchor_c400_899",
        "gap_definition": "R_blend_truth - R_blend_prediction",
        "selection_coordinate": "frozen V2.2 scene prediction",
        "uncertainty": "one SEM across rendered cases",
        "metrics": summary,
        "audit": {
            "n_rows": int(len(joined)),
            "n_cases": int(joined.case.nunique()),
            "v22_replay_max_abs": replay_error,
            "models_tuned_without_coherent_anchor_truth": True,
            "correction_physical_scaling": "one target_scale factor",
        },
    }
    pd.DataFrame(records).to_csv(output_csv, index=False)
    atomic_json(output_json, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    print("ANCHOR_CROSSFIT_SEQUENTIAL_EVALUATION_DONE", flush=True)


if __name__ == "__main__":
    main()
