#!/usr/bin/env python3
"""Audit the measured side of a fixed-g0 ConstGold response comparison."""

from __future__ import annotations

import argparse
from hashlib import sha256
import os
from pathlib import Path

import numpy as np

from scripts.evaluate_constgold_fixed_g0_response import (
    load_anchor,
    load_measured_leg,
    parse_subsets,
    pooled_sufficient,
    response,
    sufficient,
    write_json,
)


BRANCHES = ("matched_usable", "actual_usable_flags")


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8_388_608), b""):
            digest.update(block)
    return digest.hexdigest()


def bootstrap_response(
    per_case: dict,
    cases: tuple[int, ...],
    branch: str,
    *,
    h: float,
    replicates: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(cases), size=(replicates, len(cases)))
    offsets = len(cases) * np.arange(replicates, dtype=np.int64)[:, None]
    weights = np.bincount(
        (sampled + offsets).ravel(), minlength=replicates * len(cases)
    ).reshape(replicates, len(cases))

    means = []
    for direction in ("plus", "minus"):
        denominators = np.asarray(
            [
                per_case[str(case)][branch][direction]["denominator"]
                for case in cases
            ],
            dtype=np.float64,
        )
        numerators = np.asarray(
            [per_case[str(case)][branch][direction]["numerator"] for case in cases],
            dtype=np.float64,
        )
        means.append(
            np.einsum("bc,cj->bj", weights, numerators)
            / np.einsum("bc,c->b", weights, denominators)[:, None]
        )
    draws = (means[0] - means[1]) / (2.0 * h)
    return {
        "case_bootstrap_standard_error": draws.std(axis=0, ddof=1).tolist(),
        "case_bootstrap_ci95": np.quantile(draws, [0.025, 0.975], axis=0).tolist(),
        "replicates": int(replicates),
        "seed": int(seed),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--constgold-root", type=Path, required=True)
    parser.add_argument("--anchor-pattern", required=True)
    parser.add_argument("--case", type=int, action="append", required=True)
    parser.add_argument("--subset", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--h", type=float, default=0.02)
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260914)
    args = parser.parse_args(argv)
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("ConstGold measured audit must run under Slurm")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    cases = tuple(dict.fromkeys(int(case) for case in args.case))
    subsets = parse_subsets(args.subset, cases)
    per_case = {}
    reports = {}
    anchor_hashes = {}
    for case in cases:
        anchor_path = Path(args.anchor_pattern.format(case=case))
        anchor_ids, _ = load_anchor(anchor_path)
        anchor_hashes[str(case)] = file_sha256(anchor_path)
        legs = {
            "plus": load_measured_leg(
                args.constgold_root, case, +args.h, anchor_ids
            ),
            "minus": load_measured_leg(
                args.constgold_root, case, -args.h, anchor_ids
            ),
        }
        common = legs["plus"]["usable"] & legs["minus"]["usable"]
        per_case[str(case)] = {}
        for branch in BRANCHES:
            per_case[str(case)][branch] = {}
            for direction in ("plus", "minus"):
                mask = common if branch == "matched_usable" else legs[direction]["usable"]
                per_case[str(case)][branch][direction] = sufficient(
                    legs[direction]["values"][mask], np.ones(int(mask.sum()))
                )
        reports[str(case)] = {
            "anchor_rows": int(len(anchor_ids)),
            "matched_usable_rows": int(common.sum()),
            "plus_raw_detected": int(legs["plus"]["anchor_raw_detected"]),
            "minus_raw_detected": int(legs["minus"]["anchor_raw_detected"]),
            "plus_usable": int(legs["plus"]["anchor_usable"]),
            "minus_usable": int(legs["minus"]["anchor_usable"]),
            "plus_only_usable": int(
                np.sum(legs["plus"]["usable"] & ~legs["minus"]["usable"])
            ),
            "minus_only_usable": int(
                np.sum(legs["minus"]["usable"] & ~legs["plus"]["usable"])
            ),
            "plus_cross_without_shape": int(legs["plus"]["cross_without_shape"]),
            "minus_cross_without_shape": int(legs["minus"]["cross_without_shape"]),
        }
        print(
            f"CONSTGOLD_MEASURED_AUDIT case={case} anchor={len(anchor_ids)} "
            f"matched={int(common.sum())}",
            flush=True,
        )

    summaries = {}
    for subset_index, (label, selected) in enumerate(subsets.items()):
        summaries[label] = {"cases": list(selected), "branches": {}}
        for branch_index, branch in enumerate(BRANCHES):
            value = response(
                pooled_sufficient(per_case, selected, branch, "plus"),
                pooled_sufficient(per_case, selected, branch, "minus"),
                args.h,
            )
            summaries[label]["branches"][branch] = {
                "measured_response": value.tolist(),
                "bootstrap": bootstrap_response(
                    per_case,
                    selected,
                    branch,
                    h=args.h,
                    replicates=args.n_boot,
                    seed=args.bootstrap_seed
                    + 100_003 * subset_index
                    + 97 * branch_index,
                ),
            }
    result = {
        "format_version": 1,
        "domain": {
            "anchor": (
                "measured g=0 detected and MAG_AUTO<25.8 and "
                "FLUX_RADIUS>3.0 pixels"
            ),
            "truth_analysis_cut": None,
            "sheared_leg_magnitude_radius_recut": False,
            "usable_definition": "finite ngmix e1/e2 and e1^2+e2^2<1",
        },
        "h": float(args.h),
        "case_reports": reports,
        "per_case_sufficient": per_case,
        "summaries": summaries,
        "anchor_sha256_by_case": anchor_hashes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print(f"CONSTGOLD_MEASURED_AUDIT_DONE output={args.output}", flush=True)


if __name__ == "__main__":
    main()
