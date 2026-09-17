#!/usr/bin/env python3
"""Evaluate the per-leg hard-cut shear response of a full-domain flow.

Context
-------
Three different response estimands appear in this project and they are not
interchangeable:

``per-leg selection`` (this script)
    Each leg is selected by its *own* measurement, the guarded mean is
    linearised by the influence function of ``sbsi.flow_guard_response``, and
    the two legs are differenced.  Because each leg's contribution is an
    expectation under that leg's marginal law, the estimand is a functional of
    the per-leg marginals alone.  It is invariant to the cross-leg latent
    pairing, it contains the selection response, and it is the quantity a real
    survey measures when it applies catalogue cuts to observed shapes.  It is
    also exactly the quantity the guard training loss minimises.

``zero-leg anchor applied to both legs``
    ``scripts/diagnose_flow_generated_radius_coupling.py`` takes the anchor
    from the g=0 draw and applies it to the same-index sheared draw.  That is a
    cross-leg conditional expectation and therefore depends on the assumed
    copula, which a marginal flow does not identify.  Its reported numbers must
    not be read as the per-leg bias.

``unconditional model mean binned on a measured quantity``
    Selects on information the model is not conditioned on, and mixes two
    different selections between the model and measured sides.

The training loss is copula-free, so the headline number must be too.  This
script supplies the missing measurement: the per-leg response under the exact
catalogue indicator, on held-out cases, with a case-level uncertainty.

Nothing is fitted and no empirical correction is applied.  The measured and
model sides use the identical cut definition, the identical influence-function
linearisation and, when both cut conventions are requested, identical latent
draws.
"""

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


# The widths the refined checkpoint was trained at.  They are only used for the
# optional soft companion curve; the hard convention ignores them.
TRAINED_RADIUS_SOFTNESS = 0.005
TRAINED_MAGNITUDE_SOFTNESS = 0.01
EVALUATION_SEED = 733_000_000_001
BOOTSTRAP_SEED = 733_000_017


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def self_response(values) -> np.ndarray:
    """Collapse stacked ``(R11, R12, R21, R22)`` rows to the scalar trace/2."""

    matrices = np.asarray(values, dtype=np.float64).reshape(-1, 2, 2)
    return 0.5 * (matrices[:, 0, 0] + matrices[:, 1, 1])


def case_populations(
    context: torch.Tensor,
    split: dict,
    case: int,
    cuts,
    *,
    hard: bool,
    radius_softness: float,
    magnitude_softness: float,
) -> GuardResponsePopulation:
    mask = np.asarray(split["pair_case"]) == case
    if not mask.any():
        raise ValueError(f"case {case} contributes no validation response pair")
    return GuardResponsePopulation(
        context,
        split["pairs"][mask],
        split["pair_zero_target"][mask],
        split["pair_shear_target"][mask],
        split["gamma"][mask],
        cuts,
        radius_softness=radius_softness,
        magnitude_softness=magnitude_softness,
        response_scale=np.ones(len(cuts), dtype=np.float64),
        hard=hard,
    )


def bootstrap_relative_bias(
    model_by_case: np.ndarray,
    measured_by_case: np.ndarray,
    resamples: int,
) -> dict:
    """Case-resampled percentage bias of the equal-weight case means."""

    if model_by_case.shape != measured_by_case.shape or model_by_case.ndim != 2:
        raise ValueError("aligned per-case response matrices required")
    cases = model_by_case.shape[0]
    if cases < 2 or resamples < 2:
        raise ValueError("at least two cases and two resamples required")
    point = 100.0 * (
        model_by_case.mean(axis=0) / measured_by_case.mean(axis=0) - 1.0
    )
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = np.empty((resamples, model_by_case.shape[1]), dtype=np.float64)
    for index in range(resamples):
        pick = rng.integers(cases, size=cases)
        draws[index] = 100.0 * (
            model_by_case[pick].mean(axis=0) / measured_by_case[pick].mean(axis=0)
            - 1.0
        )
    if not np.isfinite(draws).all():
        raise ValueError("nonfinite case-bootstrap resample")
    return {
        "m_percent": point,
        "m_standard_error_percentage_points": draws.std(axis=0, ddof=1),
        "m_ci95_percent": np.quantile(draws, [0.025, 0.975], axis=0).T,
    }


def evaluate_convention(
    bundle,
    context: torch.Tensor,
    split: dict,
    cases: tuple[int, ...],
    cuts,
    *,
    hard: bool,
    radius_softness: float,
    magnitude_softness: float,
    draws: int,
    groups: int,
    batch_size: int,
    resamples: int,
) -> dict:
    measured = np.empty((len(cases), len(cuts)), dtype=np.float64)
    model = np.empty((len(cases), len(cuts)), dtype=np.float64)
    pair_counts = []
    pass_fraction = np.empty((len(cases), len(cuts)), dtype=np.float64)
    for index, case in enumerate(cases):
        population = case_populations(
            context,
            split,
            case,
            cuts,
            hard=hard,
            radius_softness=radius_softness,
            magnitude_softness=magnitude_softness,
        )
        evaluated = evaluate_guard_response(
            bundle.model,
            population,
            np.arange(len(population.pairs)),
            draws=draws,
            groups=groups,
            batch_size=batch_size,
            # A shared seed across conventions makes the hard/soft comparison
            # a common-random-number difference rather than a noisy one.
            seed=EVALUATION_SEED + 1_000_003 * case,
        )
        measured[index] = self_response(evaluated["target"])
        model[index] = self_response(evaluated["mean_response"])
        pair_counts.append(int(len(population.pairs)))
        pass_fraction[index] = population.statistics["baseline_mass"]
        print(
            f"case{case:03d} pairs={pair_counts[-1]} "
            f"measured_joint={measured[index][-5]:+.6f} "
            f"model_joint={model[index][-5]:+.6f}",
            flush=True,
        )
    if not np.isfinite(measured).all() or not np.isfinite(model).all():
        raise ValueError("nonfinite per-case response")
    if np.any(np.abs(measured.mean(axis=0)) <= 0.0):
        raise ValueError("a guard has zero measured response and no defined ratio")
    summary = bootstrap_relative_bias(model, measured, resamples)
    rows = []
    for position, cut in enumerate(cuts):
        rows.append(
            {
                "cut": cut["name"],
                "radius_min_pixels": cut["radius_min"],
                "magnitude_max": cut["magnitude_max"],
                "measured_R_self": float(measured[:, position].mean()),
                "model_R_self": float(model[:, position].mean()),
                "model_minus_measured": float(
                    model[:, position].mean() - measured[:, position].mean()
                ),
                "measured_R_self_case_sem": float(
                    measured[:, position].std(ddof=1) / np.sqrt(len(cases))
                ),
                "m_percent": float(summary["m_percent"][position]),
                "m_standard_error_percentage_points": float(
                    summary["m_standard_error_percentage_points"][position]
                ),
                "m_ci95_percent": [
                    float(summary["m_ci95_percent"][position][0]),
                    float(summary["m_ci95_percent"][position][1]),
                ],
                "mean_pass_fraction": float(pass_fraction[:, position].mean()),
            }
        )
    return {
        "hard": bool(hard),
        "radius_softness": None if hard else float(radius_softness),
        "magnitude_softness": None if hard else float(magnitude_softness),
        "rows": rows,
        "per_case_measured_R_self": measured,
        "per_case_model_R_self": model,
        "per_case_pairs": pair_counts,
        "cases": [int(case) for case in cases],
        "draws": int(draws),
        "groups": int(groups),
        "bootstrap_resamples": int(resamples),
    }


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain-root", type=Path, required=True)
    parser.add_argument("--flow", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--case",
        type=int,
        action="append",
        help="evaluate only these cases; defaults to the manifest validation split",
    )
    parser.add_argument("--draws", type=int, default=16)
    parser.add_argument("--groups", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument(
        "--skip-soft",
        action="store_true",
        help="report only the hard catalogue indicator convention",
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("per-leg hard-cut evaluation must run under Slurm")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((args.domain_root / "manifest.json").read_text())
    cases = tuple(
        int(case)
        for case in (args.case or manifest["split"]["validation_cases"])
    )
    if len(set(cases)) != len(cases):
        raise ValueError("duplicate case requested")
    split = base.load_split(args.domain_root, cases)
    device = torch.device(args.device)
    bundle = load_measurement_model(args.flow, device=device)
    context_np = bundle.condition_preprocessor.transform_frame(
        pd.DataFrame(split["context_raw"], columns=FLOW_FEATURES)
    )
    context = torch.as_tensor(context_np, dtype=torch.float32, device=device)
    cuts = make_guard_cuts(RADIUS_THRESHOLDS, MAGNITUDE_THRESHOLDS)

    conventions = [("hard_catalogue_indicator", True)]
    if not args.skip_soft:
        conventions.append(("trained_soft_guard", False))
    results = {}
    for name, hard in conventions:
        print(f"=== convention {name} (hard={hard}) ===", flush=True)
        results[name] = evaluate_convention(
            bundle,
            context,
            split,
            cases,
            cuts,
            hard=hard,
            radius_softness=TRAINED_RADIUS_SOFTNESS,
            magnitude_softness=TRAINED_MAGNITUDE_SOFTNESS,
            draws=args.draws,
            groups=args.groups,
            batch_size=args.batch_size,
            resamples=args.bootstrap_resamples,
        )

    payload = {
        "format_version": 1,
        "purpose": (
            "per-leg selected shear response under the exact catalogue "
            "indicator; copula-free and equal to the guard training estimand"
        ),
        "flow": str(args.flow),
        "flow_sha256": base.sha256(args.flow),
        "domain_root": str(args.domain_root),
        "cases": list(cases),
        "radius_thresholds_pixels": list(RADIUS_THRESHOLDS),
        "magnitude_thresholds": list(MAGNITUDE_THRESHOLDS),
        "conventions": results,
        "notes": [
            "Each leg is selected by its own generated or measured row, so the "
            "estimand depends only on the per-leg marginals and is invariant "
            "to the cross-leg latent pairing.",
            "The reported response is the scalar trace/2 of the two-by-two "
            "forward response, fitted by the same influence-function "
            "linearisation on the measured and model sides.",
            "Uncertainties are equal-weight case bootstrap resamples of the "
            "held-out cases; they do not include flow-training variance.",
            "This number is not comparable with the zero-leg-anchor extraction "
            "in diagnose_flow_generated_radius_coupling.py, which is a "
            "cross-leg conditional and depends on an unidentified copula.",
        ],
    }
    output.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True))
    joint = next(
        row
        for row in results["hard_catalogue_indicator"]["rows"]
        if row["cut"] == "radius_gt_3_magnitude_lt_25.8"
    )
    print(
        "PER_LEG_HARD_CUT_COMPLETE "
        f"joint_m={joint['m_percent']:+.4f}%"
        f" +/- {joint['m_standard_error_percentage_points']:.4f} "
        f"output={output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
