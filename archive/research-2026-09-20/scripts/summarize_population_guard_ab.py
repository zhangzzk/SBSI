#!/usr/bin/env python3
"""Compare frozen population-guard A/B diagnostics using paired case resamples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from plots.constgold_property_bias import (
    BRANCHES, PROPERTY_ORDER, bootstrap_response, collapse_bins, make_case_weights,
)
from scripts.train_fixed_g0_flow import write_json


def collapsed_arrays(result, branch):
    raw = result["per_case_sufficient"][PROPERTY_ORDER[0]][branch]
    arrays = {
        source: {
            direction: {field: np.asarray(value) for field, value in stats.items()}
            for direction, stats in directions.items()
        } for source, directions in raw.items()
    }
    return collapse_bins(arrays)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    output = args.root / "comparison.json"
    if output.exists():
        raise FileExistsError(output)
    result = {"cohorts": {}, "training": {}}
    for arm in ("shared", "independent"):
        result["training"][arm] = json.loads((args.root / arm / "completed.json").read_text())
    for cohort in ("uncut", "fixed_cut"):
        data = {arm: json.loads((args.root / "constgold" / cohort / f"{arm}.json").read_text())
                for arm in ("baseline", "shared", "independent")}
        baseline = data["baseline"]
        for arm, candidate in data.items():
            for key in ("cases", "h", "draws", "sampling_seed", "bootstrap",
                        "population", "anchor_rows_by_case", "binning"):
                if candidate[key] != baseline[key]:
                    raise ValueError(f"non-comparable {cohort} {arm}: {key}")
            for key in ("anchor_pattern", "rblend", "classifiers"):
                if candidate["inputs"][key] != baseline["inputs"][key]:
                    raise ValueError(f"non-comparable inputs: {cohort} {arm} {key}")
        weights = make_case_weights(10000, len(baseline["cases"]), 20260914)
        branches = {}
        for branch in BRANCHES:
            draws = {}
            measured_reference = None
            for arm, candidate in data.items():
                arrays = collapsed_arrays(candidate, branch)
                measured = bootstrap_response(
                    arrays["measured"]["plus"], arrays["measured"]["minus"], weights, baseline["h"],
                )[:, 0, 0]
                if measured_reference is not None:
                    np.testing.assert_array_equal(measured, measured_reference)
                measured_reference = measured
                predicted = bootstrap_response(
                    arrays["model"]["plus"], arrays["model"]["minus"], weights, baseline["h"],
                )[:, 0, 0]
                draws[arm] = 100 * (measured / predicted - 1)
            entries = {arm: candidate["overall"][branch] for arm, candidate in data.items()}
            comparisons = {}
            for left, right in (("independent", "shared"), ("independent", "baseline"),
                                ("shared", "baseline")):
                delta = draws[left] - draws[right]
                comparisons[f"{left}_minus_{right}"] = {
                    "delta_m_percentage_points": entries[left]["m_percent"] - entries[right]["m_percent"],
                    "paired_case_bootstrap_se_pp": float(delta.std(ddof=1)),
                    "paired_case_bootstrap_ci95_pp": np.quantile(delta, [0.025, 0.975]),
                }
            branches[branch] = {"models": entries, "paired_comparisons": comparisons}
            print(f"JOINT_AB {cohort} {branch} " + " ".join(
                f"{arm}={entries[arm]['m_percent']:+.4f}%" for arm in entries
            ), flush=True)
        result["cohorts"][cohort] = branches
    result["limitations"] = [
        "Reused ConstGold development cases; no checkpoint is selected using these results.",
        "Both cohorts require an observed usable zero-shear anchor.",
        "Fixed-cut cohort is not a per-leg reselected likelihood sample.",
        "One training seed and one common latent integration seed; no fresh acceptance claim.",
        "The fixed-g0-trained blending emulator is extrapolated on the uncut cohort.",
    ]
    write_json(output, result)
    print(f"POPULATION_GUARD_COMPARISON_COMPLETE output={output}", flush=True)


if __name__ == "__main__":
    main()
