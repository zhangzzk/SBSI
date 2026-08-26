#!/usr/bin/env python3
"""Measure the full component response matrix in a random-direction simulation.

The historical flow target stores only the response projected onto each random
shear direction.  This audit goes back to the same measured catalogue and fits
the full 2x2 matrix, case by case.  Repeated all-pairs rows are collapsed to one
row per measured primary before fitting.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.feather as feather


def _fit_case(frame, case: int) -> np.ndarray:
    select = frame["case"] == case
    g1 = frame["gamma1_input_p"][select]
    g2 = frame["gamma2_input_p"][select]
    e1 = frame["e1_input_rot0_p"][select]
    e2 = frame["e2_input_rot0_p"][select]
    design = np.column_stack((np.ones(select.sum()), g1, g2, e1, e2))
    measured = np.column_stack(
        (frame["measured_ngmix_g1"][select], frame["measured_ngmix_g2"][select])
    )
    coefficient, _, rank, _ = np.linalg.lstsq(design, measured, rcond=None)
    if rank != design.shape[1]:
        raise RuntimeError(f"case {case} response design has rank {rank}")
    return coefficient[1:3].T


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalogue", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-case", type=int, default=99)
    parser.add_argument("--primary-mag-min", type=float, default=18.0)
    parser.add_argument("--primary-mag-max", type=float, default=28.0)
    parser.add_argument("--primary-re-min", type=float, default=0.1)
    parser.add_argument("--primary-re-max", type=float, default=1.5)
    args = parser.parse_args()

    columns = [
        "case",
        "input_index",
        "detected",
        "r_input_p",
        "Re_input_p",
        "gamma1_input_p",
        "gamma2_input_p",
        "e1_input_rot0_p",
        "e2_input_rot0_p",
        "measured_ngmix_g1",
        "measured_ngmix_g2",
    ]
    table = feather.read_table(args.catalogue, columns=columns, memory_map=True)
    frame = {name: table[name].to_numpy(zero_copy_only=False) for name in columns}
    finite = np.ones(len(table), dtype=bool)
    for name in columns:
        if name == "detected":
            continue
        finite &= np.isfinite(frame[name])
    keep = (
        finite
        & frame["detected"].astype(bool)
        & (frame["case"] <= args.max_case)
        & (frame["r_input_p"] >= args.primary_mag_min)
        & (frame["r_input_p"] <= args.primary_mag_max)
        & (frame["Re_input_p"] >= args.primary_re_min)
        & (frame["Re_input_p"] <= args.primary_re_max)
    )
    frame = {name: values[keep] for name, values in frame.items()}

    # The all-pairs catalogue repeats one measured primary for each neighbour.
    # Keep the first occurrence of each (case, input_index), exactly matching an
    # equal 1/n_pairs weight without needing to retain the group counts.
    key = frame["case"].astype(np.int64) * 1_000_003 + frame["input_index"].astype(np.int64)
    _, first = np.unique(key, return_index=True)
    first.sort()
    frame = {name: values[first] for name, values in frame.items()}

    cases = np.unique(frame["case"]).astype(int)
    matrices = np.stack([_fit_case(frame, int(case)) for case in cases])
    mean = matrices.mean(axis=0)
    standard_error = matrices.std(axis=0, ddof=1) / np.sqrt(len(matrices))
    result = {
        "method": "case-block full component response regression",
        "catalogue": str(args.catalogue),
        "n_raw_rows": int(len(table)),
        "n_unique_detected_primaries": int(len(first)),
        "cases": cases.tolist(),
        "n_cases": int(len(cases)),
        "design": ["intercept", "g1", "g2", "intrinsic_e1", "intrinsic_e2"],
        "matrix_convention": "rows=measured_e1,e2; columns=injected_g1,g2",
        "response_matrix": mean.tolist(),
        "case_standard_error": standard_error.tolist(),
        "cross_response_pull": [
            float(mean[0, 1] / standard_error[0, 1]),
            float(mean[1, 0] / standard_error[1, 0]),
        ],
        "per_case_response_matrix": matrices.tolist(),
        "cuts": {
            "max_case": args.max_case,
            "primary_mag": [args.primary_mag_min, args.primary_mag_max],
            "primary_re": [args.primary_re_min, args.primary_re_max],
            "detected": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps({key: result[key] for key in (
        "n_unique_detected_primaries",
        "n_cases",
        "response_matrix",
        "case_standard_error",
        "cross_response_pull",
    )}, indent=2))
    print(args.output)


if __name__ == "__main__":
    main()
