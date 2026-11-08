#!/usr/bin/env python
"""Print a compact summary from one or more catalogue-closure result files."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+")
    args = parser.parse_args(argv)
    print(
        f"{'result':32s} {'method':18s} {'N':>8s} {'g_true':>10s} "
        f"{'g_hat':>10s} {'delta':>10s} {'SE':>9s} {'pull':>8s} {'ESS':>9s}"
    )
    for filename in args.results:
        payload = json.loads(Path(filename).read_text())
        result = payload["result"]
        n_detected = int(payload["config"]["n_detected"])
        if result is not None:
            delta = result["estimated_shear"] - result["injected_shear"]
            print(
                f"{filename:32.32s} {'exact':18s} {n_detected:8d} "
                f"{result['injected_shear']:+10.5f} {result['estimated_shear']:+10.5f} "
                f"{delta:+10.5f} {result.get('robust_standard_error', float('nan')):9.5f} "
                f"{result.get('closure_pull', float('nan')):+8.2f} {'-':>9s}"
            )
        for arm in payload.get("importance", []):
            for rung in arm["rungs"]:
                delta = rung["estimated_shear"] - arm["injected_shear"]
                method = (
                    f"IS K{arm['n_candidates']} s{arm['proposal_seed']} "
                    f"M{rung['n_draws']}"
                )
                print(
                    f"{filename:32.32s} {method:18.18s} {n_detected:8d} "
                    f"{arm['injected_shear']:+10.5f} {rung['estimated_shear']:+10.5f} "
                    f"{delta:+10.5f} {'-':>9s} {'-':>8s} "
                    f"{rung['diagnostics']['mean_ess']:9.1f}"
                )
        for arm in payload.get("profile", []):
            for rung in arm["rungs"]:
                estimate = rung["estimated_shear"]
                delta = estimate - arm["injected_shear"]
                information = rung.get("quadratic_information")
                standard_error = (
                    float("nan")
                    if information is None or information <= 0
                    else 1.0 / math.sqrt(information)
                )
                pull = (
                    float("nan")
                    if not math.isfinite(standard_error)
                    else delta / standard_error
                )
                method = f"P s{arm['proposal_seed']} M{rung['n_draws']}"
                print(
                    f"{filename:32.32s} {method:18.18s} {n_detected:8d} "
                    f"{arm['injected_shear']:+10.5f} {estimate:+10.5f} "
                    f"{delta:+10.5f} {standard_error:9.5f} {pull:+8.2f} {'-':>9s}"
                )


if __name__ == "__main__":
    main()
