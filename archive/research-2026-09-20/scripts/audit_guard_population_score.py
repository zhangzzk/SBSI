#!/usr/bin/env python3
"""Audit row-sampling bias and validation coverage of the response guard loss.

Read every pair in the requested training or validation split. Independent latent replicas on identical rows
estimate the between-row covariance of latent-mean responses. This gives the
extra expected penalty in the historical shared-row training objective.
Compare full-population predictions with the exact subset used for checkpoint
selection. All responses here reselect each leg, unlike fixed-g0 diagnostics.
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
    GuardResponsePopulation, make_guard_cuts, numpy_guard_weights,
)
from sbsi.measurement_model import load_measurement_model
from scripts import train_fixed_g0_flow as base
from scripts.train_full_domain_guard_flow import (
    MAGNITUDE_THRESHOLDS, RADIUS_THRESHOLDS,
)


def measured_contributions(population, zero, shear, design):
    stats = population.statistics
    influence = []
    for values in (zero, shear):
        weights = numpy_guard_weights(
            values, population.cuts,
            radius_softness=population.radius_softness,
            magnitude_softness=population.magnitude_softness,
        )
        influence.append(
            weights[..., None]
            * (values[:, None, :2] - np.asarray(stats["baseline_mean_shape"])[None])
            / np.asarray(stats["baseline_mass"])[None, :, None]
        )
    return np.einsum("bco,bi->bcoi", influence[1] - influence[0], design).reshape(
        len(zero), len(population.cuts), 4
    )


def scalar(values):
    return (values[..., 0] + values[..., 3]) / 2


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain-root", type=Path, required=True)
    parser.add_argument("--flow", type=Path, required=True)
    parser.add_argument("--reference-indices", type=Path,
                        help="Historical comparison subset; omitted means the complete chosen split")
    parser.add_argument("--split", choices=("train", "validation"), default="validation")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--radius-softness", type=float, default=0.05)
    parser.add_argument("--magnitude-softness", type=float, default=0.05)
    parser.add_argument("--response-scale", type=float, default=0.02)
    parser.add_argument("--training-pairs", type=int, default=32768)
    parser.add_argument("--draws", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=731000000001)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("full-population guard audit must run under Slurm")
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = json.loads((args.domain_root / "manifest.json").read_text())
    data = base.load_split(args.domain_root, manifest["split"][f"{args.split}_cases"])
    bundle = load_measurement_model(args.flow, device=torch.device("cuda"))
    bundle.model.eval()
    context = torch.as_tensor(
        bundle.condition_preprocessor.transform_frame(
            pd.DataFrame(data["context_raw"], columns=FLOW_FEATURES)
        ), dtype=torch.float32, device="cuda",
    )
    cuts = make_guard_cuts(RADIUS_THRESHOLDS, MAGNITUDE_THRESHOLDS)
    scales = np.full(len(cuts), args.response_scale)
    scales[0] = 0.01
    population = GuardResponsePopulation(
        context, data["pairs"], data["pair_zero_target"], data["pair_shear_target"],
        data["gamma"], cuts, radius_softness=args.radius_softness,
        magnitude_softness=args.magnitude_softness, response_scale=scales,
    )
    n = len(data["pairs"])
    subset = np.load(args.reference_indices) if args.reference_indices is not None else np.arange(n)
    if (subset.ndim != 1 or not np.issubdtype(subset.dtype, np.integer)
            or not len(subset) or np.any(subset < 0) or np.any(subset >= n)
            or len(np.unique(subset)) != len(subset)):
        raise ValueError("invalid reference subset")
    mask = np.zeros(n, dtype=bool)
    mask[subset] = True
    sums = np.zeros((2, len(cuts), 4))
    products = np.zeros((len(cuts), 4))
    subset_sums = np.zeros_like(sums)
    subset_measured = np.zeros_like(products)
    case_ids, case_inverse = np.unique(data["pair_case"], return_inverse=True)
    case_counts = np.bincount(case_inverse)
    case_model = np.zeros((len(case_ids), len(cuts), 4))
    case_measured = np.zeros_like(case_model)
    for block, start in enumerate(range(0, n, args.batch_size)):
        stop = min(n, start + args.batch_size)
        ids = torch.arange(start, stop, device="cuda")
        values = [population.contributions(
            bundle.model, ids, args.draws, args.seed + stream * 1000000000 + block,
        ).cpu().numpy() for stream in range(2)]
        local_mask = mask[start:stop]
        for stream in range(2):
            sums[stream] += values[stream].sum(0)
            subset_sums[stream] += values[stream][local_mask].sum(0)
        products += (values[0] * values[1]).sum(0)
        measured = measured_contributions(
            population, data["pair_zero_target"][start:stop],
            data["pair_shear_target"][start:stop],
            population.design[ids].cpu().numpy(),
        )
        subset_measured += measured[local_mask].sum(0)
        np.add.at(case_measured, case_inverse[start:stop], measured)
        np.add.at(case_model, case_inverse[start:stop], (values[0] + values[1]) / 2)
        if block % 100 == 0:
            print(f"GUARD_POPULATION_AUDIT pairs={stop}/{n}", flush=True)
    means = sums / n
    target = population.target.cpu().numpy()
    np.testing.assert_allclose(case_measured.sum(0) / n, target, atol=1e-11)
    row_covariance = (products - n * means[0] * means[1]) / (n - 1)
    extra_penalty = 0.5 * row_covariance / args.training_pairs / scales[:, None] ** 2
    full_score = 0.5 * ((means[0] - target) * (means[1] - target)
                        / scales[:, None] ** 2).mean()
    subset_means = subset_sums / len(subset)
    subset_score = 0.5 * ((subset_means[0] - target) * (subset_means[1] - target)
                          / scales[:, None] ** 2).mean()
    bootstrap = np.random.default_rng(20260917).integers(
        len(case_ids), size=(10000, len(case_ids))
    )
    rows = []
    for i, cut in enumerate(cuts):
        model_r = float(scalar(means.mean(0)[i]))
        measured_r = float(scalar(target[i]))
        model_cases = scalar(case_model[:, i])
        measured_cases = scalar(case_measured[:, i])
        boot_bias = (measured_cases[bootstrap].sum(1)
                     / model_cases[bootstrap].sum(1) - 1)
        rows.append({
            "cut": cut, "measured_R_self": measured_r, "model_R_self": model_r,
            "m": measured_r / model_r - 1,
            "m_case_bootstrap_se": float(boot_bias.std(ddof=1)),
            "subset_model_R_self": float(scalar(subset_means.mean(0)[i])),
            "subset_measured_R_self": float(scalar(subset_measured[i] / len(subset))),
            "subset_model_minus_full": float(scalar(subset_means.mean(0)[i]
                                                      - means.mean(0)[i])),
            "shared_row_extra_loss": float(extra_penalty[i].mean()),
            "full_target_matrix": target[i].reshape(2, 2),
            "full_model_matrix": means.mean(0)[i].reshape(2, 2),
        })
    result = {
        "arguments": vars(args) | {key: str(value) for key, value in vars(args).items()
                                   if isinstance(value, Path)},
        "flow_sha256": base.sha256(args.flow),
        "domain_manifest_sha256": base.sha256(args.domain_root / "manifest.json"),
        "script_sha256": base.sha256(Path(__file__)),
        "population": manifest["population"], "per_case_matching": data["per_case_matching"],
        "pairs": n, "subset_pairs": len(subset), "case_ids": case_ids,
        "full_population_score": full_score, "historical_subset_score": subset_score,
        "shared_row_extra_loss": float(extra_penalty.mean()),
        "full_replica_means": means, "target": target,
        "case_counts": case_counts, "case_model_sum": case_model,
        "case_measured_sum": case_measured, "rows": rows,
        "estimand": "matched usable half-shear pairs; per-leg soft reselection; full-population baseline influence response",
        "limitations": ["forward amplitude 0.05", "self response only; excludes emulator",
                        f"{args.split} cases reused from model development", "two finite latent replicas",
                        "case bootstrap holds baseline mass/shape nuisance estimates fixed",
                        "case bootstrap excludes training and latent integration uncertainty"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    base.write_json(args.output, result)
    for row in rows:
        if row["cut"]["name"] in ("global", "radius_gt_3", "radius_gt_3_magnitude_lt_25.8"):
            print(f"AUDIT_RESULT {row['cut']['name']} m={row['m']:+.6%} "
                  f"case_se={row['m_case_bootstrap_se']:.6%} "
                  f"subset_shift={row['subset_model_minus_full']:+.6g}", flush=True)
    print(f"GUARD_POPULATION_AUDIT_COMPLETE full_score={full_score:.8g} "
          f"subset_score={subset_score:.8g} "
          f"shared_row_extra_loss={extra_penalty.mean():.8g}", flush=True)


if __name__ == "__main__":
    main()
