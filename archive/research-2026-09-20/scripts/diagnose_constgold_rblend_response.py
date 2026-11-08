#!/usr/bin/env python3
"""Measure the exact R_blend contribution to fixed-g0 matched response."""

from __future__ import annotations

import argparse
from hashlib import sha256
import os
from pathlib import Path

import numpy as np
import pandas as pd

from sbsi.shear_map import apply_shear_to_ellipticity
from scripts.diagnose_constgold_fixed_g0_truth_support import (
    GROUPS,
    bootstrap_response,
    bootstrap_weights,
    load_rblend_allow_missing,
    load_truth_groups,
    pooled,
    uncertainty,
)
from scripts.evaluate_constgold_fixed_g0_response import (
    load_anchor,
    load_measured_leg,
    parse_mapping,
    response,
    sufficient,
    write_json,
)


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8_388_608), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--constgold-root", type=Path, required=True)
    parser.add_argument("--anchor-pattern", required=True)
    parser.add_argument("--rblend", action="append", required=True)
    parser.add_argument("--case", type=int, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--h", type=float, default=0.02)
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260915)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("ConstGold component diagnostics must run under Slurm")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    cases = tuple(dict.fromkeys(args.case))
    if len(cases) != len(args.case) or args.h <= 0:
        raise ValueError("unique cases and positive h required")
    paths = parse_mapping(args.rblend, "--rblend")
    tables = {
        label: pd.read_feather(path, columns=["case", "input_index", "R_blend"])
        for label, path in paths.items()
    }

    per_case = {label: {} for label in paths}
    unavailable = {label: {group: [] for group in GROUPS} for label in paths}
    reports = {}
    for case in cases:
        anchor_ids, context = load_anchor(Path(args.anchor_pattern.format(case=case)))
        groups = load_truth_groups(args.constgold_root, case, args.h, anchor_ids)
        plus = load_measured_leg(args.constgold_root, case, args.h, anchor_ids)
        minus = load_measured_leg(args.constgold_root, case, -args.h, anchor_ids)
        common = plus["usable"] & minus["usable"]
        base_e1, base_e2 = context[:, 0], context[:, 1]
        plus_e1, plus_e2 = apply_shear_to_ellipticity(
            base_e1, base_e2, args.h, 0.0
        )
        minus_e1, minus_e2 = apply_shear_to_ellipticity(
            base_e1, base_e2, -args.h, 0.0
        )
        delta = {
            "plus": np.column_stack((plus_e1 - base_e1, plus_e2 - base_e2)),
            "minus": np.column_stack((minus_e1 - base_e1, minus_e2 - base_e2)),
        }
        reports[str(case)] = {"matched_rows": int(common.sum()), "coverage": {}}
        for label, table in tables.items():
            rblend = load_rblend_allow_missing(table, case, anchor_ids)
            available = np.isfinite(rblend)
            reports[str(case)]["coverage"][label] = int(available.sum())
            values = {
                direction: rblend[:, None] * delta[direction]
                for direction in ("plus", "minus")
            }
            per_case[label][str(case)] = {}
            for group in GROUPS:
                mask = common & groups[group]
                if np.any(mask & ~available):
                    unavailable[label][group].append(int(case))
                    continue
                per_case[label][str(case)][group] = {
                    direction: sufficient(
                        values[direction][mask], np.ones(int(mask.sum()))
                    )
                    for direction in ("plus", "minus")
                }
        print(
            f"CONSTGOLD_RBLEND_COMPONENT case={case} matched={int(common.sum())}",
            flush=True,
        )

    weights = bootstrap_weights(cases, args.n_boot, args.bootstrap_seed)
    summaries = {}
    for label, values in per_case.items():
        summaries[label] = {}
        for group in GROUPS:
            missing = unavailable[label][group]
            if missing:
                summaries[label][group] = {
                    "status": "not_evaluable_on_complete_group",
                    "cases_with_missing_rblend_rows": missing,
                }
                continue
            point = response(
                pooled(values, cases, group, "plus"),
                pooled(values, cases, group, "minus"),
                args.h,
            )
            draws = bootstrap_response(values, cases, group, weights, args.h)
            summaries[label][group] = {
                "status": "complete",
                "rblend_response": point,
                "bootstrap": uncertainty(draws),
            }

    result = {
        "format_version": 1,
        "purpose": (
            "exact additive R_blend response component on the fixed measured-g0, "
            "both-leg-usable cohort"
        ),
        "cases": list(cases),
        "h": float(args.h),
        "domain": {
            "anchor": (
                "measured g=0 detected, MAG_AUTO<25.8, strict "
                "FLUX_RADIUS>3.0 pixels (0.6 arcsec)"
            ),
            "truth_analysis_cut": None,
            "sheared_leg_magnitude_radius_recut": False,
            "former_truth_support_is_diagnostic_partition_only": True,
        },
        "inputs": {
            label: {
                "path": str(path.resolve()),
                "sha256": file_sha256(path),
            }
            for label, path in paths.items()
        },
        "case_reports": reports,
        "per_case_sufficient": per_case,
        "summaries": summaries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print(f"CONSTGOLD_RBLEND_COMPONENT_COMPLETE output={args.output}", flush=True)


if __name__ == "__main__":
    main()
