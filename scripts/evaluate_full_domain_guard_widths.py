#!/usr/bin/env python3
"""Evaluate a trained full-domain flow across progressively sharper guards."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sbsi.fixed_g0_domain import FLOW_FEATURES
from sbsi.flow_guard_response import (
    GuardResponsePopulation,
    evaluate_guard_response,
    make_guard_cuts,
)
from sbsi.measurement_model import load_measurement_model
from scripts import train_fixed_g0_flow as base
from scripts.train_full_domain_guard_flow import (
    MAGNITUDE_THRESHOLDS,
    RADIUS_THRESHOLDS,
)


WIDTHS = (
    (0.05, 0.05),
    (0.02, 0.05),
    (0.01, 0.05),
    (0.01, 0.02),
    (0.005, 0.01),
)
SUBSET_SEED = 900000613
EVALUATION_SEED = 614_000_000_001


def scalar_rows(cuts, result):
    rows = []
    for cut, target, model, residual in zip(
        cuts,
        result["target"],
        result["mean_response"],
        result["mean_residual"],
    ):
        target = np.asarray(target, dtype=np.float64).reshape(2, 2)
        model = np.asarray(model, dtype=np.float64).reshape(2, 2)
        residual = np.asarray(residual, dtype=np.float64).reshape(2, 2)
        target_self = float(np.trace(target) / 2.0)
        model_self = float(np.trace(model) / 2.0)
        residual_self = float(np.trace(residual) / 2.0)
        rows.append(
            {
                "cut": cut,
                "measured_R_self": target_self,
                "model_R_self": model_self,
                "model_minus_measured": residual_self,
                "relative_residual": residual_self / target_self,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain-root", type=Path, required=True)
    parser.add_argument("--flow", type=Path, required=True)
    parser.add_argument("--reference-indices", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=400_000)
    parser.add_argument("--draws", type=int, default=16)
    parser.add_argument("--groups", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2048)
    args = parser.parse_args()

    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("guard-width evaluation must run under Slurm")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    manifest = json.loads((args.domain_root / "manifest.json").read_text())
    validation = base.load_split(
        args.domain_root, manifest["split"]["validation_cases"]
    )
    bundle = load_measurement_model(args.flow, device=torch.device("cuda"))
    context_np = bundle.condition_preprocessor.transform_frame(
        pd.DataFrame(validation["context_raw"], columns=FLOW_FEATURES)
    )
    context = torch.as_tensor(context_np, dtype=torch.float32, device="cuda")

    reference = np.load(args.reference_indices)
    if reference.ndim != 1 or np.any(reference < 0) or np.any(
        reference >= len(validation["pairs"])
    ):
        raise ValueError("reference indices are outside the validation pair table")
    available = np.ones(len(validation["pairs"]), dtype=bool)
    available[reference] = False
    candidates = np.flatnonzero(available)
    if args.pairs < 1 or args.pairs > len(candidates):
        raise ValueError("requested disjoint validation subset is unavailable")
    indices = np.sort(
        np.random.default_rng(SUBSET_SEED).choice(
            candidates, size=args.pairs, replace=False
        )
    )
    cuts = make_guard_cuts(RADIUS_THRESHOLDS, MAGNITUDE_THRESHOLDS)
    results = []
    for radius_softness, magnitude_softness in WIDTHS:
        population = GuardResponsePopulation(
            context,
            validation["pairs"],
            validation["pair_zero_target"],
            validation["pair_shear_target"],
            validation["gamma"],
            cuts,
            radius_softness=radius_softness,
            magnitude_softness=magnitude_softness,
            response_scale=np.ones(len(cuts), dtype=np.float64),
        )
        evaluated = evaluate_guard_response(
            bundle.model,
            population,
            indices,
            draws=args.draws,
            groups=args.groups,
            batch_size=args.batch_size,
            seed=EVALUATION_SEED,
        )
        results.append(
            {
                "radius_softness_pixels": radius_softness,
                "magnitude_softness": magnitude_softness,
                "rows": scalar_rows(cuts, evaluated),
            }
        )
        radius_row = next(row for row in results[-1]["rows"] if row["cut"]["name"] == "radius_gt_3")
        joint_row = next(
            row
            for row in results[-1]["rows"]
            if row["cut"]["name"] == "radius_gt_3_magnitude_lt_25.8"
        )
        print(
            f"GUARD_WIDTH radius={radius_softness:g} magnitude={magnitude_softness:g} "
            f"radius_rel={radius_row['relative_residual']:+.6%} "
            f"joint_rel={joint_row['relative_residual']:+.6%}",
            flush=True,
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "format_version": 1,
                "purpose": "soft-guard width sweep on validation pairs excluded from checkpoint selection",
                "domain_root": str(args.domain_root.resolve()),
                "flow": str(args.flow.resolve()),
                "reference_indices": str(args.reference_indices.resolve()),
                "reference_pairs_excluded": int(len(reference)),
                "evaluation_pairs": int(len(indices)),
                "subset_seed": SUBSET_SEED,
                "draws": args.draws,
                "groups": args.groups,
                "evaluation_seed": EVALUATION_SEED,
                "common_random_numbers_across_widths": True,
                "results": results,
            },
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    print(f"GUARD_WIDTH_SWEEP_COMPLETE output={output}", flush=True)


if __name__ == "__main__":
    main()
