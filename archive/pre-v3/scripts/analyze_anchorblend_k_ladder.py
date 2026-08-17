"""Summarize the no-retrain nearest-neighbour k ladder on anchorblend."""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd


def sem(x) -> float:
    x = np.asarray(x, float)
    return float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--ks", type=int, nargs="+", default=[20, 32, 48, 64])
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    if os.path.exists(args.output):
        raise SystemExit(f"REFUSING to overwrite {args.output}")
    frame = pd.concat([pd.read_feather(path) for path in args.inputs], ignore_index=True)
    splits = [("0_99", 0, 99), ("100_199", 100, 199),
              ("200_299_unseen", 200, 299), ("all", 0, 299)]
    payload = {"ks": args.ks, "splits": {}}
    for name, lo, hi in splits:
        sub = frame[(frame.case >= lo) & (frame.case <= hi)]
        cols = ["R_blend_truth", "raw_neighbours_within_10",
                *[f"R_blend_k{k}" for k in args.ks]]
        case = sub.groupby("case", sort=True)[cols].mean()
        truth = case.R_blend_truth.to_numpy(float)
        row = {
            "n_rows": int(len(sub)), "n_cases": int(len(case)),
            "truth": float(truth.mean()), "truth_case_sem": sem(truth),
            "fraction_anchors_raw_neighbours_ge20": float(
                np.mean(sub.raw_neighbours_within_10.to_numpy(int) >= 20)
            ),
            "arms": {},
        }
        base = case.R_blend_k20.to_numpy(float)
        for k in args.ks:
            pred = case[f"R_blend_k{k}"].to_numpy(float)
            gap = pred - truth
            shift = pred - base
            row["arms"][f"k{k}"] = {
                "prediction": float(pred.mean()),
                "prediction_minus_truth": float(gap.mean()),
                "gap_case_sem": sem(gap),
                "shift_from_k20": float(shift.mean()),
                "shift_case_sem": sem(shift),
            }
        payload["splits"][name] = row
    final = payload["splits"]["200_299_unseen"]
    payload["conclusion_gates"] = {
        "k64_reduces_absolute_gap": bool(
            abs(final["arms"]["k64"]["prediction_minus_truth"])
            < abs(final["arms"]["k20"]["prediction_minus_truth"])
        ),
        "k64_closes_at_least_half_of_deficit": bool(
            final["arms"]["k64"]["shift_from_k20"]
            >= 0.5 * -final["arms"]["k20"]["prediction_minus_truth"]
        ),
        "note": "Larger k is an inference diagnostic, not an authorized extrapolation or correction.",
    }
    with open(args.output, "w") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    for name, row in payload["splits"].items():
        print(f"\n[{name}] truth={row['truth']:+.6f} frac_n>=20={row['fraction_anchors_raw_neighbours_ge20']:.3%}")
        for arm, values in row["arms"].items():
            print(f"  {arm}: pred={values['prediction']:+.6f} gap={values['prediction_minus_truth']:+.6f} "
                  f"+-{values['gap_case_sem']:.6f} shift={values['shift_from_k20']:+.6f}")
    print(f"gates={payload['conclusion_gates']}")
    print(f"wrote {args.output}\nANCHORBLEND_K_LADDER_ANALYSIS_DONE", flush=True)


if __name__ == "__main__":
    main()
