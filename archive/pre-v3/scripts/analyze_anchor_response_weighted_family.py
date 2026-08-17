"""Case-balanced fresh comparison of the fixed response-weighted V2.2 family."""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd


TAGS = [
    "lsst_r_extnbr_v22_rpowa03",
    "lsst_r_extnbr_v22_rpowa10",
    "lsst_r_extnbr_v22_rpowa30",
]


def stat(values) -> dict:
    array = np.asarray(values, float)
    return {
        "mean": float(array.mean()),
        "case_sem": float(array.std(ddof=1) / np.sqrt(len(array))),
        "n_cases": int(len(array)),
    }


def summarize(frame: pd.DataFrame, edges: np.ndarray) -> dict:
    columns = ["gap_baseline", "gap_model", "model_minus_baseline"]
    case = frame.groupby("case", sort=True)[columns].mean()
    bins = np.clip(
        np.searchsorted(edges, frame.prediction_baseline.to_numpy(float), side="right") - 1,
        0, len(edges) - 2,
    )
    conditional = {}
    for index in range(len(edges) - 1):
        sub = frame.loc[bins == index]
        by_case = sub.groupby("case", sort=True)[columns].mean()
        conditional[str(index)] = {
            "lo": None if not np.isfinite(edges[index]) else float(edges[index]),
            "hi": None if not np.isfinite(edges[index + 1]) else float(edges[index + 1]),
            "n_rows": int(len(sub)),
            **{column: stat(by_case[column]) for column in columns},
        }
    gap_model_mean = float(case.gap_model.mean())
    gap_model_sem = float(case.gap_model.std(ddof=1) / np.sqrt(len(case)))
    worst_baseline = float(max(
        abs(value["gap_baseline"]["mean"]) for value in conditional.values()
    ))
    worst_model = float(max(
        abs(value["gap_model"]["mean"]) for value in conditional.values()
    ))
    return {
        "n_rows": int(len(frame)),
        **{column: stat(case[column]) for column in columns},
        "conditional_baseline_prediction_quintiles": conditional,
        "global_abs_gap_improved": bool(abs(case.gap_model.mean()) < abs(case.gap_baseline.mean())),
        "global_gap_within_2_case_sem": bool(abs(gap_model_mean) <= 2 * gap_model_sem),
        "worst_conditional_abs_gap_baseline": worst_baseline,
        "worst_conditional_abs_gap_model": worst_model,
        "worst_conditional_abs_gap_improved": bool(worst_model < worst_baseline),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parts-root", required=True)
    parser.add_argument("--case-min", type=int, default=300)
    parser.add_argument("--case-max", type=int, default=399)
    parser.add_argument("--tags", nargs="+", default=TAGS)
    parser.add_argument("--previously-inspected", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if os.path.exists(args.output):
        raise FileExistsError(f"refusing to overwrite {args.output}")
    frames = {}
    for tag in args.tags:
        paths = [
            os.path.join(args.parts_root, tag, f"case{case}.feather")
            for case in range(args.case_min, args.case_max + 1)
        ]
        missing = [path for path in paths if not os.path.isfile(path)]
        if missing:
            raise RuntimeError(f"{tag}: missing {len(missing)} case parts")
        frame = pd.concat([pd.read_feather(path) for path in paths], ignore_index=True)
        if frame.duplicated(["case", "input_index"]).any():
            raise RuntimeError(f"{tag}: duplicate anchor keys")
        frames[tag] = frame
    first = frames[args.tags[0]]
    edges = np.quantile(first.prediction_baseline, np.linspace(0, 1, 6))
    edges[0], edges[-1] = -np.inf, np.inf
    payload = {
        "design": "fixed response-model family; feature-only loss allocation where applicable; rendered case is uncertainty unit",
        "cases": [args.case_min, args.case_max],
        "models": {tag: summarize(frame, edges) for tag, frame in frames.items()},
        "family_size": len(args.tags),
        "fresh_validation": bool(not args.previously_inspected),
        "constgold_opened": False,
        "deployable_model_selected": False,
    }
    with open(args.output, "x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    print("ANCHOR_RESPONSE_WEIGHTED_FAMILY_ANALYSIS_DONE", flush=True)


if __name__ == "__main__":
    main()
