"""Compare two summed held-out g=0.05 closure results with case uncertainty."""
from __future__ import annotations

import argparse
import json
import os

import numpy as np


def sem(values):
    values = np.asarray(values, float)
    return float(values.std(ddof=1) / np.sqrt(len(values)))


def jackknife_ratio(numerator, denominator):
    numerator = np.asarray(numerator, float)
    denominator = np.asarray(denominator, float)
    n = len(numerator)
    estimates = np.asarray([
        np.delete(numerator, index).mean() / np.delete(denominator, index).mean() - 1.0
        for index in range(n)
    ])
    estimate = float(numerator.mean() / denominator.mean() - 1.0)
    error = float(np.sqrt((n - 1) / n * np.square(estimates - estimates.mean()).sum()))
    return {"value": estimate, "delete_one_case_jackknife_sem": error}


def load(path):
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = {int(row["case"]): row for row in payload["case_table"]}
    return payload, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    if os.path.exists(args.output):
        raise FileExistsError(f"refusing to overwrite {args.output}")

    baseline_payload, baseline = load(args.baseline)
    candidate_payload, candidate = load(args.candidate)
    if sorted(baseline) != sorted(candidate):
        raise RuntimeError("held-out case sets differ")
    cases = sorted(baseline)
    for case in cases:
        for key in ("n_primaries", "n_pairs", "label_sum_mean", "null_sum_mean"):
            if baseline[case][key] != candidate[case][key]:
                raise RuntimeError(f"case {case}: shared quantity {key} differs")

    label = np.asarray([baseline[case]["label_sum_mean"] for case in cases])
    null = np.asarray([baseline[case]["null_sum_mean"] for case in cases])
    base = np.asarray([baseline[case]["prediction_sum_mean"] for case in cases])
    cand = np.asarray([candidate[case]["prediction_sum_mean"] for case in cases])
    base_gap = base - label
    cand_gap = cand - label
    shift = cand - base
    label_sem = sem(label)
    null_sem = sem(null)
    result = {
        "design": (
            "g=0.05 response cases 40--99 train; cases 0--39 held out; "
            "per-pair responses summed before case aggregation"
        ),
        "uncertainty_unit": "simulation case",
        "n_cases": len(cases),
        "baseline_tag": baseline_payload["tag"],
        "candidate_tag": candidate_payload["tag"],
        "label": {
            "mean": float(label.mean()), "case_sem": label_sem,
            "fractional_sem": float(label_sem / abs(label.mean())),
        },
        "null": {
            "mean": float(null.mean()), "case_sem": null_sem,
            "z_from_zero": float(abs(null.mean()) / null_sem),
        },
        "baseline": {
            "prediction_mean": float(base.mean()),
            "prediction_minus_label": float(base_gap.mean()),
            "gap_case_sem": sem(base_gap),
            "prediction_over_label_minus_one": jackknife_ratio(base, label),
        },
        "candidate": {
            "prediction_mean": float(cand.mean()),
            "prediction_minus_label": float(cand_gap.mean()),
            "gap_case_sem": sem(cand_gap),
            "prediction_over_label_minus_one": jackknife_ratio(cand, label),
        },
        "candidate_minus_baseline": {
            "mean": float(shift.mean()), "case_sem": sem(shift),
            "z_from_zero": float(abs(shift.mean()) / sem(shift)),
        },
    }
    result["gates"] = {
        "null_within_two_case_sem": bool(abs(null.mean()) <= 2 * null_sem),
        "candidate_absolute_gap_smaller": bool(abs(cand_gap.mean()) < abs(base_gap.mean())),
        "candidate_gap_within_two_case_sem": bool(abs(cand_gap.mean()) <= 2 * sem(cand_gap)),
        "candidate_shift_resolved": bool(abs(shift.mean()) > 2 * sem(shift)),
    }
    result["gate_passed"] = bool(all(result["gates"].values()))
    result["constgold_opened"] = False
    with open(args.output, "x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    print("V22_G005_CLOSURE_COMPARE_DONE", flush=True)


if __name__ == "__main__":
    main()
