"""Compare two per-object BlendEMU lookups on identical constgold keys."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


KEYS = ["case", "input_index"]
VALUE = "R_blend"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--cases", type=int, nargs="+", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load_lookup(path: str, label: str) -> pd.DataFrame:
    frame = pd.read_feather(path, columns=KEYS + [VALUE])
    missing = set(KEYS + [VALUE]).difference(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing columns: {sorted(missing)}")
    if frame[KEYS].isna().any().any():
        raise ValueError(f"{label} contains null keys")
    if not np.isfinite(frame[VALUE].to_numpy(float)).all():
        raise ValueError(f"{label} contains non-finite {VALUE}")
    duplicated = frame.duplicated(KEYS, keep=False)
    if duplicated.any():
        raise ValueError(f"{label} contains {int(duplicated.sum())} duplicate-key rows")
    return frame


def mean_summary(values: pd.Series) -> dict[str, float]:
    array = values.to_numpy(float)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "std": float(array.std(ddof=1)),
        "p01": float(np.quantile(array, 0.01)),
        "p05": float(np.quantile(array, 0.05)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
    }


def main() -> None:
    args = parse_args()
    requested_cases = sorted(set(args.cases))
    if requested_cases != args.cases:
        raise ValueError("--cases must be unique and sorted")

    baseline_all = load_lookup(args.baseline, "baseline")
    baseline = baseline_all.loc[baseline_all["case"].isin(requested_cases)].copy()
    candidate = load_lookup(args.candidate, "candidate")

    candidate_cases = sorted(candidate["case"].unique().astype(int).tolist())
    baseline_cases = sorted(baseline["case"].unique().astype(int).tolist())
    if candidate_cases != requested_cases:
        raise ValueError(
            f"candidate cases {candidate_cases} do not equal requested cases {requested_cases}"
        )
    if baseline_cases != requested_cases:
        raise ValueError(
            f"baseline cases {baseline_cases} do not equal requested cases {requested_cases}"
        )

    joined = baseline.merge(
        candidate,
        on=KEYS,
        how="outer",
        suffixes=("_baseline", "_candidate"),
        indicator=True,
        validate="one_to_one",
    )
    coverage_counts = joined["_merge"].value_counts().to_dict()
    if coverage_counts.get("left_only", 0) or coverage_counts.get("right_only", 0):
        raise ValueError(f"lookup key mismatch: {coverage_counts}")
    joined = joined.drop(columns="_merge")
    joined["delta"] = joined[f"{VALUE}_candidate"] - joined[f"{VALUE}_baseline"]

    rows = []
    for case_id, group in joined.groupby("case", sort=True):
        baseline_mean = float(group[f"{VALUE}_baseline"].mean())
        candidate_mean = float(group[f"{VALUE}_candidate"].mean())
        delta_mean = candidate_mean - baseline_mean
        rows.append(
            {
                "case": int(case_id),
                "rows": int(len(group)),
                "baseline_mean": baseline_mean,
                "candidate_mean": candidate_mean,
                "delta_mean": delta_mean,
                "relative_change_percent": float(100.0 * delta_mean / baseline_mean),
            }
        )
    per_case = pd.DataFrame(rows)

    baseline_case_balanced = float(per_case["baseline_mean"].mean())
    candidate_case_balanced = float(per_case["candidate_mean"].mean())
    delta_case_balanced = candidate_case_balanced - baseline_case_balanced
    baseline_row_mean = float(joined[f"{VALUE}_baseline"].mean())
    candidate_row_mean = float(joined[f"{VALUE}_candidate"].mean())
    delta_row_mean = candidate_row_mean - baseline_row_mean

    result = {
        "design": {
            "purpose": "constgold prediction pilot; evaluation only, not model selection or tuning",
            "cases": requested_cases,
            "baseline_path": str(Path(args.baseline).resolve()),
            "candidate_path": str(Path(args.candidate).resolve()),
            "key": KEYS,
        },
        "coverage": {
            "baseline_rows_selected": int(len(baseline)),
            "candidate_rows": int(len(candidate)),
            "matched_rows": int(len(joined)),
            "baseline_only_rows": int(coverage_counts.get("left_only", 0)),
            "candidate_only_rows": int(coverage_counts.get("right_only", 0)),
            "exact_key_match": True,
        },
        "global": {
            "case_balanced": {
                "baseline_mean": baseline_case_balanced,
                "candidate_mean": candidate_case_balanced,
                "delta_mean": delta_case_balanced,
                "relative_change_percent": float(
                    100.0 * delta_case_balanced / baseline_case_balanced
                ),
            },
            "row_weighted": {
                "baseline_mean": baseline_row_mean,
                "candidate_mean": candidate_row_mean,
                "delta_mean": delta_row_mean,
                "relative_change_percent": float(100.0 * delta_row_mean / baseline_row_mean),
            },
            "candidate": mean_summary(joined[f"{VALUE}_candidate"]),
            "baseline": mean_summary(joined[f"{VALUE}_baseline"]),
            "delta": mean_summary(joined["delta"]),
            "mae_between_models": float(np.abs(joined["delta"]).mean()),
            "rmse_between_models": float(np.sqrt(np.mean(joined["delta"] ** 2))),
            "pearson_correlation": float(
                joined[[f"{VALUE}_baseline", f"{VALUE}_candidate"]].corr().iloc[0, 1]
            ),
            "fraction_candidate_greater": float((joined["delta"] > 0).mean()),
        },
        "per_case": rows,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
