"""Freeze the localized coherent-anchor population for matched noiseless toys.

The selected rows are the final-test anchors in panel E's already-frozen
rightmost ``scene_prediction`` bin.  Selection uses only the model-side scene
prediction, never the per-anchor measured residual.  The resulting manifest is
the immutable population used by both coherent and one-neighbour-at-a-time toy
measurements.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.plot_anchor_bias_scene_tail_curves import frozen_tail_definition


KEY = ["case", "input_index"]


def select_population(frame: pd.DataFrame, case_min: int, case_max: int,
                      threshold: float, compact_size: float) -> pd.DataFrame:
    """Return the outcome-free high-scene-response toy population."""
    required = {
        *KEY, "scene_prediction", "bias_truth_minus_model",
        "R_blend_truth", "has_deployed_pair",
        "log10_dominant_secondary_size", "log1p_n_pairs",
    }
    if missing := required - set(frame):
        raise KeyError(f"feature table lacks {sorted(missing)}")
    selected = frame.loc[
        frame.case.between(case_min, case_max)
        & (frame.scene_prediction >= threshold)
        & frame.has_deployed_pair.astype(bool),
        sorted(required),
    ].copy()
    if selected.empty:
        raise RuntimeError("localized toy population is empty")
    if selected.duplicated(KEY).any():
        raise RuntimeError("duplicate selected anchor key")
    selected["dominant_secondary_size"] = np.power(
        10.0, selected.log10_dominant_secondary_size.to_numpy(float)
    )
    selected["compact_dominant_secondary"] = (
        selected.dominant_secondary_size < float(compact_size)
    )
    selected["n_deployed_pairs"] = np.rint(
        np.expm1(selected.log1p_n_pairs.to_numpy(float))
    ).astype(np.int16)
    if (selected.n_deployed_pairs <= 0).any():
        raise RuntimeError("selected anchor lacks a deployed pair")
    return selected.sort_values(KEY, kind="mergesort").reset_index(drop=True)


def case_stat(frame: pd.DataFrame, column: str) -> dict:
    values = frame.groupby("case", sort=True)[column].mean().to_numpy(float)
    return {
        "mean": float(values.mean()),
        "case_sem": float(values.std(ddof=1) / np.sqrt(len(values))),
        "n_cases": int(len(values)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", required=True)
    ap.add_argument("--parent-curves", required=True)
    ap.add_argument("--case-min", type=int, default=700)
    ap.add_argument("--case-max", type=int, default=899)
    ap.add_argument("--compact-size", type=float, default=0.4)
    ap.add_argument("--expected-rows", type=int, default=0)
    ap.add_argument("--output-feather", required=True)
    ap.add_argument("--output-json", required=True)
    args = ap.parse_args()
    for path in (args.output_feather, args.output_json):
        if os.path.exists(path):
            raise FileExistsError(f"refusing existing output {path}")

    curves = pd.read_csv(args.parent_curves)
    definition = frozen_tail_definition(curves)
    threshold = float(definition["threshold"])
    frame = pd.read_feather(args.features)
    selected = select_population(
        frame, args.case_min, args.case_max, threshold, args.compact_size,
    )
    if args.expected_rows and len(selected) != args.expected_rows:
        raise RuntimeError(
            f"localized row count {len(selected):,} != expected "
            f"{args.expected_rows:,}"
        )
    expected_cases = set(range(args.case_min, args.case_max + 1))
    observed_cases = set(selected.case.unique().tolist())
    if observed_cases != expected_cases:
        raise RuntimeError(
            f"localized cases differ: missing={sorted(expected_cases-observed_cases)} "
            f"extra={sorted(observed_cases-expected_cases)}"
        )

    keep = [
        *KEY, "scene_prediction", "R_blend_truth",
        "bias_truth_minus_model", "n_deployed_pairs",
        "dominant_secondary_size", "compact_dominant_secondary",
    ]
    output = selected[keep].copy()
    Path(args.output_feather).parent.mkdir(parents=True, exist_ok=True)
    output.to_feather(args.output_feather)

    groups = {
        "all_tail": np.ones(len(output), dtype=bool),
        "compact_dominant_secondary": output.compact_dominant_secondary.to_numpy(bool),
        "noncompact_dominant_secondary": ~output.compact_dominant_secondary.to_numpy(bool),
    }
    payload = {
        "design": (
            "outcome-free frozen panel-E rightmost scene-prediction bin; "
            "same exact anchor keys feed coherent and individual-neighbour toys"
        ),
        "source_features": os.path.abspath(args.features),
        "source_parent_curves": os.path.abspath(args.parent_curves),
        "output_feather": os.path.abspath(args.output_feather),
        "case_window": [int(args.case_min), int(args.case_max)],
        "tail_definition": definition,
        "compact_secondary_threshold_arcsec": float(args.compact_size),
        "selection_uses_measured_residual": False,
        "n_rows": int(len(output)),
        "n_cases": int(output.case.nunique()),
        "per_case_rows": {
            str(int(case)): int(count)
            for case, count in output.groupby("case", sort=True).size().items()
        },
        "groups": {},
    }
    for name, mask in groups.items():
        local = output.loc[mask]
        payload["groups"][name] = {
            "n_rows": int(len(local)),
            "row_fraction": float(len(local) / len(output)),
            "n_cases": int(local.case.nunique()),
            "original_truth_minus_model": case_stat(
                local, "bias_truth_minus_model"
            ),
            "mean_deployed_pairs": float(local.n_deployed_pairs.mean()),
        }
    with open(args.output_json, "x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(payload["groups"], indent=2, sort_keys=True))
    print(
        f"wrote {args.output_feather}: {len(output):,} anchors; "
        "LOCALIZED_ANCHOR_TOY_MANIFEST_DONE",
        flush=True,
    )


if __name__ == "__main__":
    main()
