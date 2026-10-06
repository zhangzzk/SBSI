#!/usr/bin/env python3
"""Measure response-gradient variance before changing a density checkpoint.

Integrate four independent reference replicas on every training pair. Compare
ordinary independent-row gradients and two independent reference banks using
identical row/latent streams. No optimizer update or ConstGold target is used.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sbsi.fixed_g0_domain import FLOW_FEATURES
from sbsi.flow_guard_response import GuardResponsePopulation, make_guard_cuts, sampled_guard_response_backward
from sbsi.measurement_model import load_measurement_model
from scripts import train_fixed_g0_flow as base
from scripts.train_full_domain_guard_flow import MAGNITUDE_THRESHOLDS, RADIUS_THRESHOLDS


def gradient_summary(gradients):
    values = np.asarray(gradients, dtype=np.float64)
    mean = values.mean(0)
    noise = np.square(values - mean).sum(1).mean()
    signal = np.square(mean).sum()
    return {"mean_gradient_norm": float(np.sqrt(signal)), "rms_gradient_noise": float(np.sqrt(noise)),
            "relative_rms_noise": float(np.sqrt(noise / signal)) if signal else None,
            "individual_gradient_norms": np.linalg.norm(values, axis=1)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain-root", type=Path, required=True)
    parser.add_argument("--flow", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=8)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("full training-population gradient audit requires Slurm")
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    if args.replicates < 2:
        raise ValueError("at least two gradient replicas required")
    args.output_root.mkdir(parents=True)
    manifest_path = args.domain_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    print("CONTROL_VARIATE_LOADING_TRAINING_PAIRS", flush=True)
    data = base.load_split(args.domain_root, manifest["split"]["train_cases"])
    bundle = load_measurement_model(args.flow, device=torch.device("cuda"))
    model = bundle.model.eval()
    reference = copy.deepcopy(model).eval().requires_grad_(False)
    context = torch.as_tensor(bundle.condition_preprocessor.transform_frame(
        pd.DataFrame(data["context_raw"], columns=FLOW_FEATURES)), dtype=torch.float32, device="cuda")
    cuts = make_guard_cuts(RADIUS_THRESHOLDS, MAGNITUDE_THRESHOLDS)
    population = GuardResponsePopulation(
        context, data["pairs"], data["pair_zero_target"], data["pair_shear_target"], data["gamma"], cuts,
        radius_softness=.05, magnitude_softness=.05, response_scale=.01,
    )
    n = len(population.pairs)
    protocol = {"flow": str(args.flow), "flow_sha256": base.sha256(args.flow),
                "domain_manifest_sha256": base.sha256(manifest_path), "training_cases": manifest["split"]["train_cases"],
                "pairs": n, "reference_groups": 4, "reference_draws": 4, "reference_seed": 741000000001,
                "reference_batch_size": 2048, "gradient_replicates": args.replicates,
                "gradient_seed": 748000000001,
                "gradient_pairs_per_group": 65536, "gradient_draws": 8, "gradient_chunk_size": 1024,
                "guard_module_sha256": base.sha256(Path(__file__).resolve().parents[1] / "sbsi/flow_guard_response.py"),
                "script_sha256": base.sha256(Path(__file__)), "cuts": cuts, "response_scale": .01,
                "radius_softness": .05, "magnitude_softness": .05}
    base.write_json(args.output_root / "protocol.json", protocol)
    means = []
    with torch.no_grad():
        for group in range(4):
            total = torch.zeros_like(population.target)
            for block, start in enumerate(range(0, n, 2048)):
                ids = torch.arange(start, min(start + 2048, n), device="cuda")
                total += population.contributions(reference, ids, 4, 741000000001 + group * 1000000000 + block).sum(0)
                if block % 1000 == 0:
                    print(f"CONTROL_VARIATE_REFERENCE group={group} pairs={min(start+2048,n)}/{n}", flush=True)
            means.append((total / n).cpu().numpy())
            print(f"CONTROL_VARIATE_REFERENCE_GROUP_COMPLETE group={group}", flush=True)
    means = np.array(means)
    base.write_json(args.output_root / "reference.json", {
        "group_means": means, "mean_response": means.mean(0),
        "integration_se_of_four_group_mean": means.std(0, ddof=1) / 2,
        "target": population.target.cpu().numpy(),
        "note": "integration uncertainty only, conditional on training rows and checkpoint; never a response correction",
    })
    all_gradients, results = {}, {}
    for arm in ("independent", "reference_bank0", "reference_bank1"):
        gradients, records = [], []
        options = {} if arm == "independent" else {
            "reference_model": reference,
            "reference_group_means": means[:2] if arm.endswith("0") else means[2:],
        }
        for replicate in range(args.replicates):
            model.zero_grad(set_to_none=True)
            result = sampled_guard_response_backward(
                model, population, 65536, 8, chunk_size=1024,
                seed=748000000001 + replicate * 1000003, independent_pairs=True, **options,
            )
            gradient = torch.cat([p.grad.flatten() if p.grad is not None else torch.zeros_like(p).flatten()
                                  for p in model.parameters()]).detach().cpu().numpy()
            if not np.isfinite(gradient).all():
                raise ValueError("nonfinite gradient")
            gradients.append(gradient)
            records.append(result)
            print(f"CONTROL_VARIATE_GRADIENT arm={arm} replicate={replicate} "
                  f"loss={result['loss']:.8g} norm={np.linalg.norm(gradient):.8g}", flush=True)
        all_gradients[arm] = np.array(gradients)
        results[arm] = {"gradient_summary": gradient_summary(gradients), "records": records}
    # Capture sensitivity to independently integrated references separately
    # from repeated minibatch noise conditional on a single reference bank.
    left = all_gradients["reference_bank0"].mean(0)
    right = all_gradients["reference_bank1"].mean(0)
    agreement = {"reference_bank_mean_gradient_distance": float(np.linalg.norm(left-right)),
                 "reference_bank_mean_gradient_cosine": float(np.dot(left, right) / np.linalg.norm(left) / np.linalg.norm(right))}
    np.savez_compressed(args.output_root / "gradients.npz", **all_gradients)
    base.write_json(args.output_root / "result.json", {
        "arms": results, "reference_bank_agreement": agreement,
        "protocol_sha256": base.sha256(args.output_root / "protocol.json"),
        "limitations": ["variance diagnostic at the frozen reference checkpoint only",
                        "four finite reference integral replicas", "no model update, density or calibration gain claimed"],
    })
    print(f"CONTROL_VARIATE_COMPLETE {agreement}", flush=True)


if __name__ == "__main__":
    main()
