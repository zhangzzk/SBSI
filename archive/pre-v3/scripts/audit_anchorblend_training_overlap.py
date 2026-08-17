"""Audit whether anchorblend cases reuse the response-emulator input fields.

An anchor response render can be independent as a measurement while still
sharing the exact latent galaxy field used to train the response emulator.
This script compares every non-shear input column exactly and writes a strict
JSON record so held-out language is based on the actual provenance.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

NON_SHEAR = [
    "index", "cata_idx", "RA", "DEC", "position_angle", "redshift",
    "Re", "axis_ratio", "sersic_n", "r",
]
INTRINSIC = [
    "cata_idx", "position_angle", "redshift", "Re", "axis_ratio", "sersic_n", "r",
]


def compare_case(response_root: str, anchor_root: str, case: int) -> dict:
    response_path = os.path.join(response_root, f"gals{case}_0.2.feather")
    anchor_path = os.path.join(anchor_root, f"gals{case}_0.05.feather")
    response = pd.read_feather(response_path, columns=NON_SHEAR)
    anchor = pd.read_feather(anchor_path, columns=NON_SHEAR)
    result = {
        "case": int(case),
        "response_path": response_path,
        "anchor_path": anchor_path,
        "response_rows": int(len(response)),
        "anchor_rows": int(len(anchor)),
        "columns": {},
    }
    all_equal = len(response) == len(anchor)
    for column in NON_SHEAR:
        a = response[column].to_numpy()
        b = anchor[column].to_numpy()
        equal = a.shape == b.shape and np.array_equal(a, b, equal_nan=True)
        result["columns"][column] = bool(equal)
        all_equal &= equal
    result["all_non_shear_columns_exact"] = bool(all_equal)
    prefix_intrinsic = len(response) >= len(anchor)
    for column in INTRINSIC:
        prefix_intrinsic &= np.array_equal(
            response[column].to_numpy()[:len(anchor)],
            anchor[column].to_numpy(),
            equal_nan=True,
        )
    result["anchor_is_exact_intrinsic_prefix"] = bool(prefix_intrinsic)
    result["anchor_cata_idx_in_response_fraction"] = float(np.mean(np.isin(
        anchor["cata_idx"].to_numpy(), response["cata_idx"].to_numpy(),
    )))
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--response-root", required=True)
    ap.add_argument("--anchor-root", required=True)
    ap.add_argument("--cases", type=int, nargs="+", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    if os.path.exists(args.output):
        raise SystemExit(f"REFUSING to overwrite {args.output}")
    rows = [compare_case(args.response_root, args.anchor_root, case) for case in args.cases]
    payload = {
        "response_root": args.response_root,
        "anchor_root": args.anchor_root,
        "cases": rows,
        "all_cases_exact": bool(all(row["all_non_shear_columns_exact"] for row in rows)),
        "all_cases_exact_intrinsic_prefix": bool(all(
            row["anchor_is_exact_intrinsic_prefix"] for row in rows
        )),
    }
    with open(args.output, "w") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    for row in rows:
        print(
            f"case {row['case']}: rows={row['anchor_rows']:,} "
            f"whole-exact={row['all_non_shear_columns_exact']} "
            f"intrinsic-prefix={row['anchor_is_exact_intrinsic_prefix']} "
            f"cata-in-response={row['anchor_cata_idx_in_response_fraction']:.3f}", flush=True,
        )
    print(f"wrote {args.output}")
    print("ANCHORBLEND_TRAINING_OVERLAP_AUDIT_DONE", flush=True)


if __name__ == "__main__":
    main()
