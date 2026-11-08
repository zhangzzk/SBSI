#!/usr/bin/env python3
"""Verify that antithetic ConstGold legs share one invariant truth scene."""

from __future__ import annotations

import argparse
from hashlib import sha256
import os
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.evaluate_constgold_fixed_g0_response import load_anchor, raw_root, write_json


INVARIANT_COLUMNS = (
    "index_input",
    "RA_input",
    "DEC_input",
    "redshift_input",
    "Re_input",
    "axis_ratio_input",
    "position_angle_input",
    "sersic_n_input",
    "r_input",
    "e1_input_rot0",
    "e2_input_rot0",
)


def truth_path(root: Path, case: int, shear: float) -> Path:
    return raw_root(root, case, shear) / "input" / "gals_info_tile180.0_-0.5.feather"


def canonical_truth(path: Path) -> pd.DataFrame:
    frame = pd.read_feather(
        path, columns=[*INVARIANT_COLUMNS, "gamma1_input", "gamma2_input"]
    )
    if frame["index_input"].duplicated().any():
        raise ValueError(f"duplicate truth identity in {path}")
    frame = frame.sort_values("index_input", kind="stable").reset_index(drop=True)
    if not np.isfinite(frame.to_numpy(np.float64)).all():
        raise ValueError(f"non-finite truth value in {path}")
    return frame


def frame_sha256(frame: pd.DataFrame) -> str:
    digest = sha256()
    for column in frame:
        values = frame[column].to_numpy()
        digest.update(column.encode())
        digest.update(values.dtype.str.encode())
        digest.update(np.ascontiguousarray(values).tobytes())
    return digest.hexdigest()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--constgold-root", type=Path, required=True)
    parser.add_argument("--anchor-pattern", required=True)
    parser.add_argument("--case", type=int, action="append", required=True)
    parser.add_argument("--h", type=float, default=0.02)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("ConstGold truth pairing audit must run under Slurm")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    cases = tuple(dict.fromkeys(int(case) for case in args.case))
    if len(cases) != len(args.case) or args.h <= 0:
        raise ValueError("cases must be unique and h must be positive")

    reports = {}
    for case in cases:
        plus = canonical_truth(truth_path(args.constgold_root, case, +args.h))
        minus = canonical_truth(truth_path(args.constgold_root, case, -args.h))
        plus_invariant = plus.loc[:, INVARIANT_COLUMNS]
        minus_invariant = minus.loc[:, INVARIANT_COLUMNS]
        if not plus_invariant.equals(minus_invariant):
            raise RuntimeError(f"case {case} antithetic legs have different truth scenes")
        expected_shear = {
            "plus": (args.h, 0.0),
            "minus": (-args.h, 0.0),
        }
        for direction, frame in (("plus", plus), ("minus", minus)):
            g1, g2 = expected_shear[direction]
            if not np.array_equal(frame["gamma1_input"].to_numpy(), np.full(len(frame), g1)):
                raise RuntimeError(f"case {case} {direction} has unexpected gamma1")
            if not np.array_equal(frame["gamma2_input"].to_numpy(), np.full(len(frame), g2)):
                raise RuntimeError(f"case {case} {direction} has unexpected gamma2")
        anchor_ids, _ = load_anchor(Path(args.anchor_pattern.format(case=case)))
        truth_ids = plus["index_input"].to_numpy(np.int64)
        missing = np.setdiff1d(anchor_ids, truth_ids, assume_unique=True)
        if len(missing):
            raise RuntimeError(f"case {case} truth misses {len(missing)} anchor identities")
        reports[str(case)] = {
            "truth_rows": int(len(plus)),
            "anchor_rows": int(len(anchor_ids)),
            "missing_anchor_rows": 0,
            "invariant_truth_sha256": frame_sha256(plus_invariant),
        }
        print(
            f"CONSTGOLD_TRUTH_PAIRING case={case} truth={len(plus)} "
            f"anchor={len(anchor_ids)}",
            flush=True,
        )

    result = {
        "format_version": 1,
        "cases": list(cases),
        "h": float(args.h),
        "invariant_columns": list(INVARIANT_COLUMNS),
        "expected_applied_shear": {
            "plus": [float(args.h), 0.0],
            "minus": [-float(args.h), 0.0],
        },
        "truth_analysis_cut": None,
        "case_reports": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print(f"CONSTGOLD_TRUTH_PAIRING_DONE output={args.output}", flush=True)


if __name__ == "__main__":
    main()
