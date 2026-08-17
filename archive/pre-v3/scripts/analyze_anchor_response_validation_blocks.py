"""Block stability of the preselected positive-tail emulator on c200--599."""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd


def stat(values) -> dict:
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(values)),
        "case_sem": float(np.std(values, ddof=1) / np.sqrt(len(values))),
        "n_cases": int(len(values)),
    }


def summarize(frame: pd.DataFrame) -> dict:
    case = frame.groupby("case", sort=True)[
        ["gap_baseline", "gap_model", "model_minus_baseline"]
    ].mean()
    return {
        "gap_baseline": stat(case.gap_baseline),
        "gap_model": stat(case.gap_model),
        "model_minus_baseline": stat(case.model_minus_baseline),
        "global_abs_gap_improved": bool(
            abs(case.gap_model.mean()) < abs(case.gap_baseline.mean())
        ),
        "model_gap_within_2_case_sem": bool(
            abs(case.gap_model.mean())
            <= 2.0 * case.gap_model.std(ddof=1) / np.sqrt(len(case))
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-root", required=True)
    parser.add_argument("--fresh-root", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if os.path.exists(args.output):
        raise FileExistsError(f"refusing to overwrite {args.output}")
    parts = []
    for case in range(200, 600):
        root = args.old_root if case < 400 else args.fresh_root
        path = os.path.join(root, args.tag, f"case{case}.feather")
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        parts.append(pd.read_feather(path))
    frame = pd.concat(parts, ignore_index=True)
    if frame.duplicated(["case", "input_index"]).any():
        raise RuntimeError("duplicate anchor keys across validation blocks")
    blocks = {}
    for start in range(200, 600, 100):
        blocks[f"c{start}_{start + 99}"] = summarize(
            frame.loc[(frame.case >= start) & (frame.case < start + 100)]
        )
    old = frame.loc[frame.case < 400]
    fresh = frame.loc[frame.case >= 400]
    old_case = old.groupby("case").model_minus_baseline.mean()
    fresh_case = fresh.groupby("case").model_minus_baseline.mean()
    difference = float(fresh_case.mean() - old_case.mean())
    difference_sem = float(np.hypot(
        old_case.std(ddof=1) / np.sqrt(len(old_case)),
        fresh_case.std(ddof=1) / np.sqrt(len(fresh_case)),
    ))
    payload = {
        "design": "fixed positive-tail alpha=0.065 candidate; four non-overlapping 100-case coherent blocks; no refit",
        "tag": args.tag,
        "blocks": blocks,
        "previously_inspected_c200_399": summarize(old),
        "fresh_c400_599": summarize(fresh),
        "all_out_of_training_c200_599": summarize(frame),
        "fresh_minus_previous_correction": {
            "mean": difference,
            "combined_case_sem": difference_sem,
            "z": float(difference / difference_sem),
        },
        "constgold_opened": False,
        "candidate_changed_after_fresh_truth": False,
    }
    with open(args.output, "x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    print("ANCHOR_RESPONSE_VALIDATION_BLOCKS_DONE", flush=True)


if __name__ == "__main__":
    main()
