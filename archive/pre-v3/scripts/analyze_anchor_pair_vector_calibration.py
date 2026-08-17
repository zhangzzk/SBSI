"""Case-balanced diagnostic of the half-shear pair calibration on anchors."""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

from fit_anchorblend_scene_correction import stat


def summary(frame: pd.DataFrame, edges: np.ndarray) -> dict:
    columns = ["gap_raw", "gap_corrected", "correction"]
    case = frame.groupby("case", sort=True)[columns].mean()
    bins = np.clip(
        np.searchsorted(edges, frame.prediction_raw.to_numpy(float), side="right") - 1,
        0, len(edges) - 2,
    )
    conditional = {}
    for index in range(len(edges) - 1):
        subset = frame.loc[bins == index]
        by_case = subset.groupby("case", sort=True)[columns].mean()
        conditional[str(index)] = {
            "lo": None if not np.isfinite(edges[index]) else float(edges[index]),
            "hi": None if not np.isfinite(edges[index + 1]) else float(edges[index + 1]),
            "n_rows": int(len(subset)),
            **{column: stat(by_case[column]) for column in columns},
        }
    return {
        "n_rows": int(len(frame)), "n_cases": int(len(case)),
        **{column: stat(case[column]) for column in columns},
        "conditional_prediction_quintiles": conditional,
        "max_abs_conditional_raw": float(max(abs(x["gap_raw"]["mean"]) for x in conditional.values())),
        "max_abs_conditional_corrected": float(max(
            abs(x["gap_corrected"]["mean"]) for x in conditional.values()
        )),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--case-min", type=int, required=True)
    parser.add_argument("--case-max", type=int, required=True)
    parser.add_argument("--development-max", type=int, default=249)
    parser.add_argument("--fresh-validation", action="store_true")
    parser.add_argument("--table-output", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    for path in (args.table_output, args.output):
        if os.path.exists(path):
            raise FileExistsError(f"refusing to overwrite {path}")
    paths = [
        os.path.join(args.input_dir, f"case{case}.feather")
        for case in range(args.case_min, args.case_max + 1)
    ]
    missing = [path for path in paths if not os.path.isfile(path)]
    if missing:
        raise RuntimeError(f"missing {len(missing)} case parts")
    frame = pd.concat([pd.read_feather(path) for path in paths], ignore_index=True)
    if frame.duplicated(["case", "input_index"]).any():
        raise RuntimeError("duplicate anchor keys")
    frame.to_feather(args.table_output)
    development = frame.case <= args.development_max
    edge_source = frame.loc[development] if np.any(development) else frame
    edges = np.quantile(edge_source.prediction_raw, np.linspace(0, 1, 6))
    edges[0], edges[-1] = -np.inf, np.inf
    all_summary = summary(frame, edges)
    development_summary = summary(frame.loc[development], edges) if np.any(development) else None
    validation_summary = summary(frame.loc[~development], edges) if np.any(~development) else all_summary
    payload = {
        "candidate": (
            "six-bin pair-response calibration recipe fixed on half-shear c0--19; "
            "coefficients refit on c0--39 before fresh truth"
            if args.fresh_validation else
            "six-bin pair-response calibration trained only on half-shear cases 0--19"
        ),
        "cases": [args.case_min, args.case_max],
        "validation_previously_inspected": bool(not args.fresh_validation),
        "all": all_summary, "development": development_summary,
        "validation": validation_summary,
        "gates": {
            "validation_global_abs_smaller": bool(
                abs(validation_summary["gap_corrected"]["mean"])
                < abs(validation_summary["gap_raw"]["mean"])
            ),
            "validation_worst_conditional_abs_smaller": bool(
                validation_summary["max_abs_conditional_corrected"]
                < validation_summary["max_abs_conditional_raw"]
            ),
        },
        "deployable_model_written": False, "constgold_opened": False,
    }
    with open(args.output, "x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    print("ANCHOR_PAIR_VECTOR_CALIBRATION_ANALYSIS_DONE", flush=True)


if __name__ == "__main__":
    main()
