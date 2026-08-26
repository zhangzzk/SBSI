#!/usr/bin/env python3
"""Measure case-block uncertainty of the mixed-shear simulation response."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sbsi.flow_response_target import _load_leg, estimate_response_matrix


KEYS = ["case", "input_index"]
BASE_COLUMNS = [
    "case", "input_index", "r_input_p", "Re_input_p", "r_blend",
    "neighbored", "distance", "measured_ngmix_g1", "measured_ngmix_g2",
]


def _summarize(frame, cases, *, n_boot, seed):
    cases = sorted(int(case) for case in cases)
    case_values = []
    normals = []
    right_sides = []
    counts = []
    case_array = frame["case"].to_numpy(dtype=np.int64)
    for case in cases:
        selected = case_array == case
        if not selected.any():
            raise RuntimeError(f"case {case} has no matched finite response rows")
        gamma = frame.loc[selected, ["gamma1_input_p", "gamma2_input_p"]].to_numpy(float)
        delta_e = frame.loc[selected, ["delta_e1", "delta_e2"]].to_numpy(float)
        normal = gamma.T @ gamma
        right_side = delta_e.T @ gamma
        matrix = estimate_response_matrix(delta_e, gamma)
        normals.append(normal)
        right_sides.append(right_side)
        counts.append(int(selected.sum()))
        case_values.append(float(0.5 * np.trace(matrix)))

    normal = np.sum(normals, axis=0)
    right_side = np.sum(right_sides, axis=0)
    matrix = right_side @ np.linalg.inv(normal)
    case_values = np.asarray(case_values)

    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=np.float64)
    normals = np.asarray(normals)
    right_sides = np.asarray(right_sides)
    for index in range(n_boot):
        selected = rng.integers(0, len(cases), size=len(cases))
        draw_matrix = right_sides[selected].sum(axis=0) @ np.linalg.inv(
            normals[selected].sum(axis=0)
        )
        draws[index] = 0.5 * np.trace(draw_matrix)

    response = float(0.5 * np.trace(matrix))
    return {
        "n_cases": len(cases),
        "n_rows": int(sum(counts)),
        "cases": cases,
        "response_matrix": matrix.tolist(),
        "mean_measured_response": response,
        "case_response_standard_deviation": float(case_values.std(ddof=1)),
        "case_mean_standard_error": float(case_values.std(ddof=1) / np.sqrt(len(cases))),
        "case_bootstrap_standard_error": float(draws.std(ddof=1)),
        "case_bootstrap_95_interval": np.quantile(draws, [0.025, 0.975]).tolist(),
        "relative_case_bootstrap_standard_error": float(draws.std(ddof=1) / response),
        "per_case_response": case_values.tolist(),
        "per_case_counts": counts,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--g0-catalogue", type=Path, required=True)
    parser.add_argument("--sheared-catalogue", type=Path, required=True)
    parser.add_argument("--response-target", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-boot", type=int, default=100_000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    target = np.load(args.response_target)
    train_cases = [int(value) for value in target["train_cases"]]
    validation_cases = [int(value) for value in target["validation_cases"]]
    all_cases = sorted(train_cases + validation_cases)

    zero = _load_leg(args.g0_catalogue, BASE_COLUMNS, all_cases).rename(columns={
        "measured_ngmix_g1": "e1_zero", "measured_ngmix_g2": "e2_zero",
    })
    sheared = _load_leg(
        args.sheared_catalogue,
        [*BASE_COLUMNS, "gamma1_input_p", "gamma2_input_p"],
        all_cases,
    ).rename(columns={
        "measured_ngmix_g1": "e1_sheared", "measured_ngmix_g2": "e2_sheared",
    })
    before_zero = len(zero)
    before_sheared = len(sheared)
    frame = zero.merge(
        sheared[[
            "case", "input_index", "e1_sheared", "e2_sheared",
            "gamma1_input_p", "gamma2_input_p",
        ]],
        on=KEYS,
        how="inner",
        validate="one_to_one",
    )
    matched_before_cuts = len(frame)
    keep = (
        (frame["r_input_p"] > 18.0)
        & (frame["r_input_p"] < float(target["primary_mag_max"]))
        & (frame["Re_input_p"] > float(target["primary_re_min"]))
        & (frame["Re_input_p"] < 1.5)
        & (
            ((frame["distance"] > 0.0) & (frame["distance"] < 5.0))
            | (~frame["neighbored"].astype(bool))
        )
    )
    frame = frame.loc[keep].reset_index(drop=True)
    frame["delta_e1"] = frame["e1_sheared"] - frame["e1_zero"]
    frame["delta_e2"] = frame["e2_sheared"] - frame["e2_zero"]
    finite_columns = [
        "delta_e1", "delta_e2", "gamma1_input_p", "gamma2_input_p",
        "r_input_p", "Re_input_p", "r_blend",
    ]
    finite = np.isfinite(frame[finite_columns].to_numpy(dtype=np.float64)).all(axis=1)
    frame = frame.loc[finite].reset_index(drop=True)

    result = {
        "method": "case-block bootstrap of pooled forward-response least squares",
        "difference_convention": "forward g=0 to |g|=0.05",
        "matrix_convention": "rows=measured e1/e2; columns=applied primary g1/g2",
        "g0_catalogue": str(args.g0_catalogue),
        "sheared_catalogue": str(args.sheared_catalogue),
        "response_target": str(args.response_target),
        "loaded_g0_rows": int(before_zero),
        "loaded_sheared_rows": int(before_sheared),
        "matched_rows_before_cuts": int(matched_before_cuts),
        "matched_finite_rows_after_target_cuts": int(len(frame)),
        "dropped_unmatched_g0": int(before_zero - matched_before_cuts),
        "dropped_unmatched_sheared": int(before_sheared - matched_before_cuts),
        "bootstrap_replicates": args.n_boot,
        "bootstrap_seed": args.bootstrap_seed,
        "all_200_cases": _summarize(
            frame, all_cases, n_boot=args.n_boot, seed=args.bootstrap_seed
        ),
        "target_160_training_cases": _summarize(
            frame, train_cases, n_boot=args.n_boot, seed=args.bootstrap_seed
        ),
        "heldout_40_validation_cases": _summarize(
            frame, validation_cases, n_boot=args.n_boot, seed=args.bootstrap_seed
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    brief = {
        name: {
            key: value[key]
            for key in (
                "n_cases", "n_rows", "mean_measured_response",
                "case_bootstrap_standard_error",
                "relative_case_bootstrap_standard_error",
            )
        }
        for name, value in result.items()
        if name in (
            "all_200_cases", "target_160_training_cases", "heldout_40_validation_cases"
        )
    }
    print(json.dumps(brief, indent=2))
    print(args.output)


if __name__ == "__main__":
    main()
