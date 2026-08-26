#!/usr/bin/env python
# Archived Infer V1 optimization experiment.
"""Recombine one retained fixed ladder under independent-pilot allocation rules."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-ess", nargs="+", type=float, default=(8, 16, 32, 64, 128))
    parser.add_argument(
        "--max-weight-fraction", nargs="+", type=float, default=(0.25, 0.5, 0.75)
    )
    parser.add_argument(
        "--safety-factor", nargs="+", type=float, default=(1.0, 1.5, 2.0, 3.0, 4.0)
    )
    parser.add_argument(
        "--bias-correction",
        choices=("none", "richardson_1_over_m"),
        default="none",
    )
    return parser.parse_args(argv)


def _estimate(score, information):
    return np.linalg.solve(np.sum(information, axis=0), np.sum(score, axis=0))


def _influence(score, information):
    estimate = _estimate(score, information)
    residual = score - np.einsum("nij,j->ni", information, estimate)
    return residual @ np.linalg.inv(np.mean(information, axis=0)).T


def _allocate(ess_fraction, maximum, ladder, pilot_draws, min_ess, max_weight, safety):
    counts = np.full(len(ess_fraction), ladder[-1], dtype=np.int64)
    unresolved = np.ones(len(ess_fraction), dtype=bool)
    for rung in ladder:
        passed = (
            unresolved
            & (rung * ess_fraction / safety >= min_ess)
            & (maximum * pilot_draws * safety / rung <= max_weight)
        )
        counts[passed] = rung
        unresolved[passed] = False
    return counts


def main(argv=None):
    args = parse_args(argv)
    benchmark = Path(args.benchmark).resolve()
    payload = json.loads((benchmark / "result.json").read_text())
    arrays = np.load(benchmark / payload["moments"])
    ladder = tuple(sorted(int(value) for value in payload["config"]["draw_ladder"]))
    allocation_ladder = (
        ladder[1:] if args.bias_correction == "richardson_1_over_m" else ladder
    )
    if not allocation_ladder:
        raise ValueError("Richardson recombination requires at least two nested rungs")
    fixed_score = arrays[f"fixed_score_m{ladder[-1]}"]
    fixed_information = arrays[f"fixed_information_m{ladder[-1]}"]
    fixed_estimate = _estimate(fixed_score, fixed_information)
    fixed_influence = _influence(fixed_score, fixed_information)
    ess_fraction = arrays["adaptive_pilot_ess_fraction"]
    maximum = arrays["adaptive_pilot_max_weight_fraction"]
    pilot_draws = int(payload["config"]["pilot_draws"])
    n_candidates = int(payload["config"]["proposal_candidates"])
    epsilon = float(payload["config"]["proposal_epsilon"])
    n_views = 9 if payload["config"].get("full_information") else 5
    fixed_proxy = n_candidates + (n_views - 1 + epsilon) * ladder[-1]

    reports = []
    for min_ess in args.min_ess:
        for max_weight in args.max_weight_fraction:
            for safety in args.safety_factor:
                counts = _allocate(
                    ess_fraction,
                    maximum,
                    allocation_ladder,
                    pilot_draws,
                    min_ess,
                    max_weight,
                    safety,
                )
                score = np.empty_like(fixed_score)
                information = np.empty_like(fixed_information)
                for rung in allocation_ladder:
                    selected = counts == rung
                    rung_score = arrays[f"fixed_score_m{rung}"]
                    rung_information = arrays[f"fixed_information_m{rung}"]
                    if args.bias_correction == "richardson_1_over_m":
                        previous = ladder[ladder.index(rung) - 1]
                        rung_score = (
                            2.0 * rung_score - arrays[f"fixed_score_m{previous}"]
                        )
                        rung_information = (
                            2.0 * rung_information
                            - arrays[f"fixed_information_m{previous}"]
                        )
                    score[selected] = rung_score[selected]
                    information[selected] = rung_information[selected]
                estimate = _estimate(score, information)
                influence = _influence(score, information)
                difference = estimate - fixed_estimate
                paired = influence - fixed_influence
                paired_se = np.std(paired, axis=0, ddof=1) / np.sqrt(len(paired))
                mean_draws = float(np.mean(counts))
                adaptive_proxy = (
                    n_candidates
                    + epsilon * pilot_draws
                    + (n_views - 1 + epsilon) * mean_draws
                )
                reports.append(
                    {
                        "min_ess": float(min_ess),
                        "max_weight_fraction": float(max_weight),
                        "safety_factor": float(safety),
                        "estimate": estimate.tolist(),
                        "difference": difference.tolist(),
                        "paired_standard_error": paired_se.tolist(),
                        "paired_pull": np.divide(
                            difference,
                            paired_se,
                            out=np.full(2, np.nan),
                            where=paired_se > 0,
                        ).tolist(),
                        "passed_025_paired_se": bool(
                            np.all(np.abs(difference) <= 0.25 * paired_se)
                        ),
                        "mean_draws": mean_draws,
                        "draw_count_percentiles": np.percentile(
                            counts, [0, 25, 50, 75, 90, 100]
                        ).tolist(),
                        "idealized_flow_speedup": float(fixed_proxy / adaptive_proxy),
                    }
                )
    reports.sort(
        key=lambda item: (
            not item["passed_025_paired_se"],
            -item["idealized_flow_speedup"],
        )
    )
    output = Path(args.output).resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "method": "independent_pilot_grid_from_common_draws_v1",
        "benchmark": str(benchmark),
        "fixed_estimate": fixed_estimate.tolist(),
        "ladder": list(ladder),
        "allocation_ladder": list(allocation_ladder),
        "bias_correction": args.bias_correction,
        "reports": reports,
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
