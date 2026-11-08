#!/usr/bin/env python3
"""Small response-gradient steps with explicit training/validation density gates.

Reuse independently integrated reference means, never fitted response offsets.
Backtracking limits NLL degradation to 0.001 nat per row from the original
checkpoint. Validation selects only density-eligible response improvements;
the unchanged original checkpoint competes as step zero. No ConstGold is read.
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
from sbsi.flow_guard_response import GuardResponsePopulation, evaluate_guard_response, make_guard_cuts, sampled_guard_response_backward
from sbsi.measurement_model import load_measurement_model
from scripts import train_fixed_g0_flow as base
from scripts.train_full_domain_guard_flow import MAGNITUDE_THRESHOLDS, RADIUS_THRESHOLDS

DENSITY_TOLERANCE = .001


def acceptable_step(before_loss, after_loss, after_nll, baseline_nll):
    """A numerical improvement must also remain inside the density budget."""
    return (np.isfinite([before_loss, after_loss, after_nll, baseline_nll]).all()
            and after_loss < before_loss - 1e-6 and after_nll <= baseline_nll + DENSITY_TOLERANCE)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain-root", type=Path, required=True)
    parser.add_argument("--flow", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("refinement requires Slurm")
    args.output_root.mkdir(parents=True, exist_ok=False)
    reference_protocol = json.loads((args.reference_root / "protocol.json").read_text())
    reference_result = json.loads((args.reference_root / "reference.json").read_text())
    variance_result = json.loads((args.reference_root / "result.json").read_text())
    if (reference_protocol["flow_sha256"] != base.sha256(args.flow)
            or reference_protocol["domain_manifest_sha256"] != base.sha256(args.domain_root / "manifest.json")
            or reference_protocol["response_scale"] != .01
            or reference_protocol["radius_softness"] != .05 or reference_protocol["magnitude_softness"] != .05):
        raise ValueError("reference integral is incompatible with this model/population")
    if variance_result["reference_bank_agreement"]["reference_bank_mean_gradient_cosine"] < .99:
        raise ValueError("reference banks do not establish a stable update direction")
    manifest = json.loads((args.domain_root / "manifest.json").read_text())
    bundle = load_measurement_model(args.flow, device=torch.device("cuda"))
    model, reference = bundle.model.eval(), copy.deepcopy(bundle.model).eval().requires_grad_(False)
    checkpoint = torch.load(args.flow, map_location="cpu", weights_only=False)
    cuts = make_guard_cuts(RADIUS_THRESHOLDS, MAGNITUDE_THRESHOLDS)
    if list(cuts) != reference_protocol["cuts"]:
        raise ValueError("reference guard bank differs")
    groups = np.asarray(reference_result["group_means"])
    if groups.shape != (4, len(cuts), 4):
        raise ValueError("four aligned reference integrals required")
    arrays = {}
    for split, seed in (("train", 753000001), ("validation", 754000001)):
        print(f"DENSITY_GUARD_LOADING split={split}", flush=True)
        data = base.load_split(args.domain_root, manifest["split"][f"{split}_cases"])
        contexts = torch.as_tensor(bundle.condition_preprocessor.transform_frame(
            pd.DataFrame(data["context_raw"], columns=FLOW_FEATURES)), dtype=torch.float32, device="cuda")
        targets = torch.as_tensor(bundle.target_transform.transform_array(data["target_raw"]),
                                  dtype=torch.float64, device="cuda")
        pop = GuardResponsePopulation(contexts, data["pairs"], data["pair_zero_target"], data["pair_shear_target"],
                                      data["gamma"], cuts, radius_softness=.05, magnitude_softness=.05, response_scale=.01)
        ids = torch.as_tensor(np.random.default_rng(seed).choice(len(targets), min(500000, len(targets)), replace=False),
                              dtype=torch.long, device="cuda")
        arrays[split] = {"population": pop, "context": contexts, "targets": targets, "nll_ids": ids}
        del data
    train, validation = arrays["train"], arrays["validation"]
    np.testing.assert_allclose(train["population"].target.cpu().numpy(), reference_result["target"], rtol=0, atol=1e-12)
    if len(train["population"].pairs) != reference_protocol["pairs"]:
        raise ValueError("reference pair count mismatch")

    def nll(split, full=False):
        block = arrays[split]
        ids = slice(None) if full else block["nll_ids"]
        with torch.no_grad():
            return base.nll_pass(model, block["targets"][ids], block["context"][ids])["nll"]

    def training_score():
        return sampled_guard_response_backward(
            model, train["population"], 262144, 4, chunk_size=2048, seed=757000000001,
            weight=0, independent_pairs=True, reference_model=reference, reference_group_means=groups[:2])

    def validation_score():
        pop = validation["population"]
        return evaluate_guard_response(model, pop, np.arange(len(pop.pairs)), draws=4, groups=2,
                                       batch_size=2048, seed=758000000001)

    protocol = {"initial_flow_sha256": base.sha256(args.flow), "reference_root": str(args.reference_root),
                "reference_protocol_sha256": base.sha256(args.reference_root / "protocol.json"),
                "reference_result_sha256": base.sha256(args.reference_root / "reference.json"),
                "density_tolerance_nats_per_row": DENSITY_TOLERANCE, "steps": 6,
                "learning_rates": [1e-5 / 2**i for i in range(6)],
                "optimizer": "unclipped gradient descent with backtracking; no Adam moments",
                "gradient_pairs": 65536, "gradient_draws": 8, "gradient_seed": 756000000001,
                "line_search_pairs": 262144, "line_search_draws": 4, "line_search_seed": 757000000001,
                "nll_subset_seeds": [753000001, 754000001], "validation_seed": 758000000001,
                "selection": "validation guard loss with a fixed baseline-relative NLL gate; initial checkpoint included",
                "script_sha256": base.sha256(Path(__file__))}
    base.write_json(args.output_root / "protocol.json", protocol)
    baseline_train_nll, baseline_val_nll = nll("train"), nll("validation")
    baseline_full_nll = nll("validation", full=True)
    initial_guard = validation_score()
    best_loss, best_step = initial_guard["loss"], 0
    history = [{"step": 0, "validation_nll": baseline_val_nll, "validation_guard": initial_guard,
                "train_nll": baseline_train_nll, "full_validation_nll": baseline_full_nll}]

    def save_selected(step, val_nll, val_guard):
        base.save_checkpoint(args.output_root / "selected.pt", model, bundle.condition_preprocessor,
                             bundle.target_transform, checkpoint["model_config"],
                             {"training_phase": "density_gated_reference_guard", "step": step,
                              "validation_nll": val_nll, "validation_guard_loss": val_guard["loss"],
                              "protocol_sha256": base.sha256(args.output_root / "protocol.json"),
                              "population": manifest["population"]})
    save_selected(0, baseline_val_nll, initial_guard)
    print(f"DENSITY_GUARD_BASELINE nll={baseline_val_nll:.9f} guard={best_loss:.9f}", flush=True)
    for step in range(1, 7):
        model.zero_grad(set_to_none=True)
        gradient_result = sampled_guard_response_backward(
            model, train["population"], 65536, 8, chunk_size=1024,
            seed=756000000001 + step*1000003, independent_pairs=True,
            reference_model=reference, reference_group_means=groups[:2])
        gradients = {name: p.grad.detach().clone() for name, p in model.named_parameters() if p.grad is not None}
        if not gradients or any(not torch.isfinite(g).all() for g in gradients.values()):
            raise ValueError("invalid guard gradient")
        state = copy.deepcopy(model.state_dict())
        before = training_score()
        attempts = []
        accepted = False
        for rate in protocol["learning_rates"]:
            model.load_state_dict(state)
            with torch.no_grad():
                for name, p in model.named_parameters():
                    if name in gradients:
                        p.add_(gradients[name], alpha=-rate)
            after = training_score()
            after_nll = nll("train")
            accepted = acceptable_step(before["loss"], after["loss"], after_nll, baseline_train_nll)
            attempts.append({"learning_rate": rate, "guard": after, "nll": after_nll, "accepted": bool(accepted)})
            print(f"DENSITY_GUARD_TRIAL step={step} lr={rate:.4g} train_guard={after['loss']:.8f} "
                  f"delta_nll={after_nll-baseline_train_nll:+.8f} accepted={accepted}", flush=True)
            if accepted:
                break
        if not accepted:
            model.load_state_dict(state)
            history.append({"step": step, "accepted": False, "attempts": attempts})
            break
        val_nll, val_guard = nll("validation"), validation_score()
        eligible = val_nll <= baseline_val_nll + DENSITY_TOLERANCE
        if eligible and val_guard["loss"] < best_loss:
            best_loss, best_step = val_guard["loss"], step
            save_selected(step, val_nll, val_guard)
        history.append({"step": step, "accepted": True, "gradient": gradient_result,
                        "gradient_norm": float(torch.sqrt(sum(g.square().sum() for g in gradients.values()))),
                        "attempts": attempts, "validation_nll": val_nll, "validation_guard": val_guard,
                        "density_eligible": bool(eligible), "selected_step": best_step})
        base.write_json(args.output_root / "history.json", {"history": history})
        print(f"DENSITY_GUARD_STEP step={step} val_nll={val_nll:.9f} guard={val_guard['loss']:.9f} best={best_step}", flush=True)
    selected = torch.load(args.output_root / "selected.pt", map_location="cuda", weights_only=False)
    model.load_state_dict(selected["state_dict"])
    final_nll = nll("validation", full=True)
    full_density_pass = final_nll <= baseline_full_nll + DENSITY_TOLERANCE
    base.write_json(args.output_root / "history.json", {"history": history})
    base.write_json(args.output_root / "completed.json", {
        "best_step": best_step, "baseline_full_validation_nll": baseline_full_nll,
        "selected_full_validation_nll": final_nll, "full_density_gate_passed": bool(full_density_pass),
        "selected_sha256": base.sha256(args.output_root / "selected.pt"), "best_validation_guard_loss": best_loss,
        "limitations": ["one frozen-reference integration bank used for refinement",
                        "soft forward0.05 response targets and reused validation cases",
                        "a density gate does not prove joint distribution or joint calibration acceptance"],
    })
    if not full_density_pass:
        raise RuntimeError("selected model failed full validation density gate; do not promote")
    print(f"DENSITY_GUARD_COMPLETE best_step={best_step} delta_full_nll={final_nll-baseline_full_nll:+.9g}", flush=True)


if __name__ == "__main__":
    main()
