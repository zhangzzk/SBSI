#!/usr/bin/env python3
"""Turn the quiet-blend flow read into a cohort-wide flow error.

`diagnose_constgold_response_split.py --bin-column R_blend_abs` reads the flow
against the simulation with no model subtracted, but only where blending is
negligible -- and those primaries are not a random slice of the cohort: the
quietest twelfth is 30% brighter than r=23 against 1.7% cohort-wide.  Quoting
its `m` as the flow's error is therefore quoting a bright-primary number as a
global one.

Crossing the blending axis with primary magnitude fixes that.  Within a
magnitude band the quiet cells give the flow's fractional error at that
magnitude, and the cohort-wide error is the response-weighted average of those:
a band enters weighted by the response it contributes, `share x R_flow`, not by
its object count, because the quantity being corrected is a summed response.

The blending term follows without a second measurement, because ConstGold
measures the sum: `R_blend_true = R_measured - R_flow_true`.

What this does NOT do is verify that the flow's error at fixed magnitude is the
same for quiet and crowded primaries.  That is the assumption the whole
extrapolation rests on, and the reported coverage per band is how far it is
being stretched.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

TERMS = ("measured", "flow")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True,
                        help="result.json from the crossed split")
    parser.add_argument("--arm", default=None,
                        help="blending arm to report; default the first")
    parser.add_argument("--quiet-bins", type=int, default=4,
                        help="how many of the lowest blending bins count as quiet")
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260915)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def gather(result: dict, arm: str) -> tuple:
    """Per-case sufficient statistics as (case, cell, term, direction) arrays."""
    per_case = result["per_case_sufficient"]
    bands = tuple(result["binning"]["magnitude_bands"])
    if not bands:
        raise ValueError("this result has no magnitude axis; rerun with --magnitude-edge")
    cases = [str(case) for case in result["cases"]]
    cells = [cell for cell in per_case[cases[0]] if cell != "all"]
    terms = ("measured", "flow", f"blend::{arm}")
    numerator = np.zeros((len(cases), len(cells), len(terms), 2), dtype=np.float64)
    denominator = np.zeros((len(cases), len(cells), len(terms), 2), dtype=np.float64)
    for case_index, case in enumerate(cases):
        for cell_index, cell in enumerate(cells):
            entry = per_case[case][cell]
            for term_index, term in enumerate(terms):
                for side, direction in enumerate(("plus", "minus")):
                    record = entry[term][direction]
                    #  only the first response component is used
                    numerator[case_index, cell_index, term_index, side] = record[
                        "numerator"][0]
                    denominator[case_index, cell_index, term_index, side] = record[
                        "denominator"]
    return cells, bands, numerator, denominator


def responses(numerator: np.ndarray, denominator: np.ndarray,
              members: np.ndarray, h: float) -> tuple[np.ndarray, np.ndarray]:
    """Pooled (group, term) response and object count for 0/1 membership rows."""
    summed_n = np.tensordot(members, numerator.sum(axis=0), axes=([1], [0]))
    summed_d = np.tensordot(members, denominator.sum(axis=0), axes=([1], [0]))
    with np.errstate(divide="ignore", invalid="ignore"):
        means = summed_n / np.where(summed_d > 0, summed_d, np.nan)
    return (means[..., 0] - means[..., 1]) / (2.0 * h), summed_d[:, 0, 0]


def memberships(cells, bands, quiet_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """Rows: each band over all blending bins, then each band over quiet ones."""
    prefix = sorted({cell.split("|")[0] for cell in cells})
    quiet = set(prefix[:quiet_bins])
    whole = np.zeros((len(bands), len(cells)))
    narrow = np.zeros((len(bands), len(cells)))
    for column, cell in enumerate(cells):
        blend_bin, band = cell.split("|")
        row = bands.index(band)
        whole[row, column] = 1.0
        if blend_bin in quiet:
            narrow[row, column] = 1.0
    return whole, narrow


def estimate(numerator, denominator, whole, narrow, h) -> dict:
    """Band errors from the quiet cells, re-weighted to the cohort."""
    band_all, objects = responses(numerator, denominator, whole, h)
    band_quiet, quiet_objects = responses(numerator, denominator, narrow, h)
    #  m in the quiet cells is the flow's fractional under-prediction there
    epsilon = band_quiet[:, 0] / (band_quiet[:, 1] + band_quiet[:, 2]) - 1.0
    share = objects / objects.sum()
    contribution = share * band_all[:, 1]
    flow_model = float(contribution.sum())
    flow_true = float((contribution * (1.0 + epsilon)).sum())
    cohort = np.ones((1, whole.shape[1]))
    pooled, _ = responses(numerator, denominator, cohort, h)
    measured, blend_model = float(pooled[0, 0]), float(pooled[0, 2])
    blend_true = measured - flow_true
    return {
        "epsilon": epsilon,
        "share": share,
        "band_R_flow": band_all[:, 1],
        "band_R_measured": band_all[:, 0],
        "band_R_blend": band_all[:, 2],
        "quiet_R_blend_fraction": np.abs(band_quiet[:, 2]) / band_quiet[:, 0],
        "coverage": quiet_objects / objects,
        "objects": objects,
        "quiet_objects": quiet_objects,
        "flow_model": flow_model,
        "flow_true": flow_true,
        "flow_error": 100.0 * (flow_true / flow_model - 1.0),
        "blend_model": blend_model,
        "blend_true": blend_true,
        "blend_error": 100.0 * (blend_model / blend_true - 1.0) if blend_true else np.nan,
        "R_measured": measured,
        "m_percent": 100.0 * (measured / (flow_model + blend_model) - 1.0),
    }


def described(point: float, samples: np.ndarray) -> dict:
    usable = samples[np.isfinite(samples)]
    return {
        "value": float(point),
        "standard_error": float(usable.std(ddof=1)) if usable.size > 1 else None,
        "ci95": [float(edge) for edge in np.quantile(usable, (0.025, 0.975))]
        if usable.size > 1 else None,
        "resampled": int(usable.size),
    }


def main(argv=None):
    args = parse_args(argv)
    result = json.loads(args.result.read_text())
    arm = args.arm or next(iter(result["inputs"]["rblend"]))
    h = float(result["h"])
    cells, bands, numerator, denominator = gather(result, arm)
    whole, narrow = memberships(cells, bands, args.quiet_bins)
    point = estimate(numerator, denominator, whole, narrow, h)

    rng = np.random.default_rng(args.bootstrap_seed)
    draws = rng.integers(0, numerator.shape[0], size=(args.n_boot, numerator.shape[0]))
    scalars = ("flow_error", "blend_error", "flow_true", "blend_true")
    samples = {name: np.empty(args.n_boot) for name in scalars}
    epsilon_samples = np.empty((args.n_boot, len(bands)))
    for index, choice in enumerate(draws):
        drawn = estimate(numerator[choice], denominator[choice], whole, narrow, h)
        for name in scalars:
            samples[name][index] = drawn[name]
        epsilon_samples[index] = drawn["epsilon"]

    print(f"arm={arm}  quiet bins = the lowest {args.quiet_bins} of "
          f"{len(cells) // len(bands)}  cases={numerator.shape[0]}")
    print()
    print(f"{'band':>12} {'objects':>10} {'share':>7} {'cohort R_flow':>13} "
          f"{'flow error %':>20} {'quiet |Rb|/R':>13} {'quiet covers':>13}")
    for row, band in enumerate(bands):
        entry = described(100.0 * point["epsilon"][row], 100.0 * epsilon_samples[:, row])
        error = f"{entry['value']:8.2f} +- {entry['standard_error']:6.2f}"
        print(f"{band:>12} {point['objects'][row]:10,.0f} {point['share'][row]:7.4f} "
              f"{point['band_R_flow'][row]:13.5f} {error:>20} "
              f"{point['quiet_R_blend_fraction'][row]:13.4f} "
              f"{point['coverage'][row]:12.1%}")
    print()
    for name, label in (("flow_error", "GLOBAL FLOW ERROR  (model low by)"),
                        ("blend_error", "GLOBAL BLEND ERROR (model high by)")):
        entry = described(point[name], samples[name])
        print(f"{label:>36}: {entry['value']:+7.3f} +- {entry['standard_error']:.3f} %"
              f"   95% CI [{entry['ci95'][0]:+.3f}, {entry['ci95'][1]:+.3f}]")
    print(f"{'model R_flow / implied true':>36}: {point['flow_model']:.5f} / "
          f"{described(point['flow_true'], samples['flow_true'])['value']:.5f}")
    print(f"{'model R_blend / implied true':>36}: {point['blend_model']:.5f} / "
          f"{described(point['blend_true'], samples['blend_true'])['value']:.5f}")
    print(f"{'R_measured, cohort':>36}: {point['R_measured']:.5f}")
    print(f"{'pooled m (unchanged by any of this)':>36}: {point['m_percent']:+.3f} %")

    if args.output:
        payload = {
            "format_version": 1,
            "source": str(args.result),
            "arm": arm,
            "quiet_bins": args.quiet_bins,
            "bands": list(bands),
            "per_band": {
                band: {
                    "objects": float(point["objects"][row]),
                    "cohort_share": float(point["share"][row]),
                    "cohort_R_flow": float(point["band_R_flow"][row]),
                    "quiet_coverage": float(point["coverage"][row]),
                    "quiet_blend_fraction": float(point["quiet_R_blend_fraction"][row]),
                    "flow_error_percent": described(100.0 * point["epsilon"][row],
                                                    100.0 * epsilon_samples[:, row]),
                }
                for row, band in enumerate(bands)
            },
            "global": {name: described(point[name], samples[name]) for name in scalars},
            "R_measured": point["R_measured"],
            "model_R_flow": point["flow_model"],
            "model_R_blend": point["blend_model"],
            "m_percent": point["m_percent"],
            "limitations": [
                "The flow's error at fixed magnitude is assumed the same for quiet "
                "and crowded primaries; nothing here tests that, and the per-band "
                "coverage says how far it is stretched -- the faint bands least.",
                "Bands are truth primary magnitude only; a flow error tracking size "
                "or ellipticity at fixed magnitude is not resolved.",
                "One flow-training seed and one antithetic integration seed.",
            ],
        }
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
