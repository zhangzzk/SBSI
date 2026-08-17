"""Exploratory cross-fitted variance reduction for the one-active anchor test.

The primary predeclared result is the raw estimator in the input artifact.  This
follow-up uses only two quantities with zero expectation under random spin-2
directions: the orthogonal null and the difference between the two orthogonal
response projections.  Slopes are learned on the opposite case half and then
applied out of sample.  A within-half bootstrap repeats fitting and evaluation
to include coefficient uncertainty.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd


def stat(values: np.ndarray) -> dict:
    values = np.asarray(values, float)
    return {
        "mean": float(values.mean()),
        "case_sem": float(values.std(ddof=1) / np.sqrt(len(values))),
        "n_cases": int(len(values)),
    }


def slopes(train: pd.DataFrame, controls: list[str]) -> np.ndarray:
    x = train[controls].to_numpy(float)
    y = train["target"].to_numpy(float)
    design = np.column_stack([np.ones(len(x)), x])
    return np.linalg.lstsq(design, y, rcond=None)[0][1:]


def crossfit(case: pd.DataFrame, controls: list[str]) -> tuple[np.ndarray, dict]:
    split = case["case"].to_numpy(int) < 450
    corrected = np.empty(len(case), dtype=float)
    betas = {}
    for name, test in (("development", split), ("validation", ~split)):
        train = case.loc[~test]
        beta = slopes(train, controls)
        corrected[test] = (
            case.loc[test, "target"].to_numpy(float)
            - case.loc[test, controls].to_numpy(float) @ beta
        )
        betas[name] = beta.tolist()
    return corrected, betas


def bootstrap_sem(case: pd.DataFrame, controls: list[str], n_boot: int,
                  seed: int) -> float:
    rng = np.random.default_rng(seed)
    left = case[case.case < 450].reset_index(drop=True)
    right = case[case.case >= 450].reset_index(drop=True)
    means = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        li = rng.integers(0, len(left), len(left))
        ri = rng.integers(0, len(right), len(right))
        l = left.iloc[li]
        r = right.iloc[ri]
        beta_l = slopes(r, controls)
        beta_r = slopes(l, controls)
        yc_l = l["target"].to_numpy(float) - l[controls].to_numpy(float) @ beta_l
        yc_r = r["target"].to_numpy(float) - r[controls].to_numpy(float) @ beta_r
        means[b] = np.concatenate([yc_l, yc_r]).mean()
    return float(means.std(ddof=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--n-bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260812)
    args = ap.parse_args()
    if os.path.exists(args.output):
        raise FileExistsError(f"refusing existing output {args.output}")
    frame = pd.read_feather(args.input)
    frame["HT_direction_difference"] = (
        frame["n_pairs"].to_numpy(float)
        * frame["R_direction_difference"].to_numpy(float)
    )
    columns = [
        "oneactive_gap_exact_model", "HT_null_sum", "HT_direction_difference",
        "selected_prediction", "R_one_pair", "n_pairs",
    ]
    case = frame.groupby("case", sort=True)[columns].mean().reset_index()
    case = case.rename(columns={"oneactive_gap_exact_model": "target"})
    case["pair_gap"] = case["selected_prediction"] - case["R_one_pair"]
    raw = stat(case["target"].to_numpy(float))
    payload = {
        "design": "exploratory two-fold cross-fitted zero-expectation control variates; raw predeclared endpoint retained",
        "case_split": {"development": [400, 449], "validation": [450, 499]},
        "n_bootstrap": args.n_bootstrap,
        "raw_target": raw,
        "direct_selected_pair": {
            "prediction": stat(case["selected_prediction"].to_numpy(float)),
            "truth": stat(case["R_one_pair"].to_numpy(float)),
            "prediction_minus_truth": stat(case["pair_gap"].to_numpy(float)),
        },
        "controls": {
            "HT_null_sum": stat(case["HT_null_sum"].to_numpy(float)),
            "HT_direction_difference": stat(
                case["HT_direction_difference"].to_numpy(float)
            ),
        },
        "cross_fitted": {},
        "constgold_opened": False,
    }
    designs = {
        "null_only": ["HT_null_sum"],
        "direction_difference_only": ["HT_direction_difference"],
        "both": ["HT_null_sum", "HT_direction_difference"],
    }
    for offset, (name, controls) in enumerate(designs.items()):
        corrected, betas = crossfit(case, controls)
        result = stat(corrected)
        result["bootstrap_sem"] = bootstrap_sem(
            case, controls, args.n_bootstrap, args.seed + offset,
        )
        result["opposite_half_slopes"] = betas
        result["sem_reduction_factor_vs_raw"] = float(
            result["bootstrap_sem"] / raw["case_sem"]
        )
        payload["cross_fitted"][name] = result
    with open(args.output, "x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    print("ONEACTIVE_CONTROL_VARIATES_DONE", flush=True)


if __name__ == "__main__":
    main()
