"""Scan cuts on maximum/runner-up absolute V2.2 pair response."""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from scripts.diagnose_anchorblend_bright_secondary_cut import summarize_cut


KEY = ["case", "input_index"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dominance-table", required=True)
    ap.add_argument("--common-table", required=True)
    ap.add_argument("--thresholds", type=float, nargs="+", required=True)
    ap.add_argument("--primary-threshold", type=float, default=20.0)
    ap.add_argument("--development-max", type=int, default=449)
    ap.add_argument("--output-json", required=True)
    args = ap.parse_args()
    if os.path.exists(args.output_json):
        raise FileExistsError(f"refusing existing output {args.output_json}")
    thresholds = sorted(set(float(x) for x in args.thresholds))
    if args.primary_threshold not in thresholds:
        raise ValueError("primary threshold must appear in threshold scan")
    if any(x <= 0 for x in thresholds):
        raise ValueError("thresholds must be positive")

    dominance = pd.read_feather(args.dominance_table, columns=[
        "case", "anchor_index", "dominant_to_runner_up_abs_response",
    ])
    common = pd.read_feather(args.common_table, columns=["case", "input_index", "gap"])
    frame = dominance.merge(
        common, left_on=["case", "anchor_index"],
        right_on=KEY, how="inner", validate="one_to_one",
    )
    coordinate = frame.dominant_to_runner_up_abs_response.to_numpy(float)
    if not np.isfinite(coordinate).all():
        raise RuntimeError("non-finite maximum/runner-up response ratio")
    payload = {
        "design": (
            "user-requested primary cut max|R_pair|/runner-up|R_pair| > 20; "
            "additional thresholds are explicitly exploratory; coherent-common "
            "anchors; rendered case is uncertainty unit"
        ),
        "primary_threshold": float(args.primary_threshold),
        "thresholds": thresholds,
        "development_window": [int(frame.case.min()), args.development_max],
        "validation_window": [args.development_max + 1, int(frame.case.max())],
        "n_rows": int(len(frame)),
        "blocks": {},
        "constgold_opened": False,
    }
    for block_name, block in (
        ("development", frame.loc[frame.case <= args.development_max]),
        ("validation", frame.loc[frame.case > args.development_max]),
    ):
        payload["blocks"][block_name] = {}
        local_coordinate = block.dominant_to_runner_up_abs_response.to_numpy(float)
        for threshold in thresholds:
            payload["blocks"][block_name][str(threshold)] = summarize_cut(
                block, local_coordinate > threshold,
            )
    with open(args.output_json, "x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    print("ANCHORBLEND_RESPONSE_RATIO_CUT_SCAN_DONE", flush=True)


if __name__ == "__main__":
    main()
