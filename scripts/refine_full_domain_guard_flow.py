#!/usr/bin/env python3
"""Refine a full-domain flow with sharp cut-boundary response guards."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sbsi.fixed_g0_domain import FLOW_FEATURES, FLOW_TARGETS
from sbsi.flow_guard_response import (
    GuardResponsePopulation,
    evaluate_guard_response,
    make_guard_cuts,
    sampled_guard_response_backward,
)
from sbsi.measurement_model import load_measurement_model
from scripts import train_fixed_g0_flow as base
from scripts.train_full_domain_guard_flow import (
    MAGNITUDE_THRESHOLDS,
    RADIUS_THRESHOLDS,
    compact_guard_result,
    json_guard_statistics,
)


SEED = 621
EPOCHS = 40
ROWS_PER_EPOCH = 4_000_000
BATCH_SIZE = 8192
LEARNING_RATE = 3.0e-6
MIN_LEARNING_RATE = 1.0e-7
GRADIENT_CLIP = 5.0
VALIDATION_NLL_ROWS = 500_000
VALIDATION_NLL_SEED = 900000621

RADIUS_SOFTNESS = 0.005
MAGNITUDE_SOFTNESS = 0.01
RESPONSE_SCALE = 0.01
GUARD_STEPS_PER_EPOCH = 4
GUARD_PAIRS_PER_STEP = 65_536
GUARD_TRAINING_DRAWS = 8
GUARD_TRAINING_CHUNK = 1024
GUARD_TRAINING_SEED = 622_000_000_001
VALIDATION_GUARD_PAIRS = 400_000
VALIDATION_GUARD_DRAWS = 16
VALIDATION_GUARD_GROUPS = 4
VALIDATION_GUARD_BATCH = 2048
VALIDATION_GUARD_SEED = 623_000_000_001
VALIDATION_GUARD_SUBSET_SEED = 900000622


def sharp_guard_training_epoch(model, population, optimizer, epoch_seed: int) -> dict:
    model.train()
    losses = []
    norms = []
    for step in range(GUARD_STEPS_PER_EPOCH):
        optimizer.zero_grad(set_to_none=True)
        result = sampled_guard_response_backward(
            model,
            population,
            GUARD_PAIRS_PER_STEP,
            GUARD_TRAINING_DRAWS,
            chunk_size=GUARD_TRAINING_CHUNK,
            seed=GUARD_TRAINING_SEED + epoch_seed * 1_000_003 + step * 100_003,
        )
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), GRADIENT_CLIP, error_if_nonfinite=True
        )
        optimizer.step()
        losses.append(result["loss"])
        norms.append(float(norm))
    return {
        "guard_loss_mean": float(np.mean(losses)),
        "guard_loss_steps": losses,
        "steps": GUARD_STEPS_PER_EPOCH,
        "pairs_per_step": GUARD_PAIRS_PER_STEP,
        "draws": GUARD_TRAINING_DRAWS,
        "gradient_norm_mean": float(np.mean(norms)),
        "gradient_norm_max": float(np.max(norms)),
        "clipped_step_fraction": float(np.mean(np.asarray(norms) > GRADIENT_CLIP)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain-root", type=Path, required=True)
    parser.add_argument("--initial-flow", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("sharp guard refinement must run under Slurm")
    domain_root = args.domain_root.resolve()
    initial_flow = args.initial_flow.resolve()
    output_root = args.output_root.resolve()
    manifest_path = domain_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    population = manifest["population"]
    if (
        population.get("truth_analysis_cut")
        != {"column": "r_input_p", "operator": "<", "value": 26.0}
        or population.get("measured_selection_cut") is not None
        or not population.get("independent_per_leg_validity", False)
    ):
        raise ValueError("sharp refinement requires the true-mag-limited full domain")
    if manifest["flow"]["features"] != list(FLOW_FEATURES) or manifest["flow"][
        "targets"
    ] != list(FLOW_TARGETS):
        raise ValueError("full-domain flow contract mismatch")
    if output_root.exists() and not args.resume:
        raise FileExistsError(f"preserving existing refinement root: {output_root}")
    output_root.mkdir(parents=True, exist_ok=args.resume)

    print("LOADING_SHARP_GUARD_DATA", flush=True)
    train_data = base.load_split(domain_root, manifest["split"]["train_cases"])
    validation_data = base.load_split(
        domain_root, manifest["split"]["validation_cases"]
    )
    device = torch.device("cuda")
    bundle = load_measurement_model(initial_flow, device=device)
    if bundle.condition_preprocessor.feature_names != list(FLOW_FEATURES):
        raise ValueError("initial flow condition contract mismatch")
    if bundle.target_transform.target_names != list(FLOW_TARGETS):
        raise ValueError("initial flow target contract mismatch")
    initial_checkpoint = torch.load(
        initial_flow, map_location="cpu", weights_only=False
    )
    model_config = initial_checkpoint["model_config"]
    condition = bundle.condition_preprocessor
    target = bundle.target_transform
    model = bundle.model

    train_context_np = condition.transform_frame(
        pd.DataFrame(train_data["context_raw"], columns=FLOW_FEATURES)
    )
    validation_context_np = condition.transform_frame(
        pd.DataFrame(validation_data["context_raw"], columns=FLOW_FEATURES)
    )
    train_values_np = target.transform_array(train_data["target_raw"])
    validation_values_np = target.transform_array(validation_data["target_raw"])
    train_context = torch.as_tensor(train_context_np, dtype=torch.float32, device=device)
    train_values = torch.as_tensor(train_values_np, dtype=torch.float64, device=device)
    validation_context = torch.as_tensor(
        validation_context_np, dtype=torch.float32, device=device
    )
    validation_values = torch.as_tensor(
        validation_values_np, dtype=torch.float64, device=device
    )
    validation_nll_indices = np.sort(
        np.random.default_rng(VALIDATION_NLL_SEED).choice(
            len(validation_values),
            min(VALIDATION_NLL_ROWS, len(validation_values)),
            replace=False,
        )
    )
    validation_guard_indices = np.sort(
        np.random.default_rng(VALIDATION_GUARD_SUBSET_SEED).choice(
            len(validation_data["pairs"]),
            min(VALIDATION_GUARD_PAIRS, len(validation_data["pairs"])),
            replace=False,
        )
    )
    validation_nll_ids = torch.as_tensor(
        validation_nll_indices, dtype=torch.long, device=device
    )
    cuts = make_guard_cuts(RADIUS_THRESHOLDS, MAGNITUDE_THRESHOLDS)
    scales = np.full(len(cuts), RESPONSE_SCALE, dtype=np.float64)
    train_population = GuardResponsePopulation(
        train_context,
        train_data["pairs"],
        train_data["pair_zero_target"],
        train_data["pair_shear_target"],
        train_data["gamma"],
        cuts,
        radius_softness=RADIUS_SOFTNESS,
        magnitude_softness=MAGNITUDE_SOFTNESS,
        response_scale=scales,
    )
    validation_population = GuardResponsePopulation(
        validation_context,
        validation_data["pairs"],
        validation_data["pair_zero_target"],
        validation_data["pair_shear_target"],
        validation_data["gamma"],
        cuts,
        radius_softness=RADIUS_SOFTNESS,
        magnitude_softness=MAGNITUDE_SOFTNESS,
        response_scale=scales,
    )
    protocol = {
        "format_version": 1,
        "domain_manifest": str(manifest_path),
        "domain_manifest_sha256": base.sha256(manifest_path),
        "initial_flow": str(initial_flow),
        "initial_flow_sha256": base.sha256(initial_flow),
        "population": population,
        "features": list(FLOW_FEATURES),
        "targets": list(FLOW_TARGETS),
        "model_config": model_config,
        "counts": {
            "train_nll_rows": len(train_values),
            "validation_nll_rows": len(validation_values),
            "train_matched_response_pairs": len(train_data["pairs"]),
            "validation_matched_response_pairs": len(validation_data["pairs"]),
        },
        "guard": {
            "cuts": list(cuts),
            "radius_softness_pixels": RADIUS_SOFTNESS,
            "magnitude_softness": MAGNITUDE_SOFTNESS,
            "response_scale": RESPONSE_SCALE,
            "train_statistics": json_guard_statistics(train_population),
            "validation_statistics": json_guard_statistics(validation_population),
        },
        "recipe": {
            "seed": SEED,
            "epochs": EPOCHS,
            "rows_per_epoch": ROWS_PER_EPOCH,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "minimum_learning_rate": MIN_LEARNING_RATE,
            "gradient_clip": GRADIENT_CLIP,
            "validation_nll_rows": len(validation_nll_indices),
            "guard_steps_per_epoch": GUARD_STEPS_PER_EPOCH,
            "guard_pairs_per_step": GUARD_PAIRS_PER_STEP,
            "guard_training_draws": GUARD_TRAINING_DRAWS,
            "validation_guard_pairs": len(validation_guard_indices),
            "validation_guard_draws": VALIDATION_GUARD_DRAWS,
            "validation_guard_groups": VALIDATION_GUARD_GROUPS,
            "measurement_conditions_used": False,
        },
        "script_sha256": base.sha256(Path(__file__).resolve()),
    }
    protocol_path = output_root / "protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != base.json_ready(protocol):
            raise RuntimeError("resume protocol differs from sharp refinement")
    else:
        base.write_json(protocol_path, protocol)
        np.save(output_root / "validation_nll_indices.npy", validation_nll_indices)
        np.save(
            output_root / "validation_guard_pair_indices.npy",
            validation_guard_indices,
        )
    print(
        f"SHARP_GUARD_DATA_READY train_nll={len(train_values):,} "
        f"validation_nll={len(validation_values):,} "
        f"train_pairs={len(train_data['pairs']):,} "
        f"validation_pairs={len(validation_data['pairs']):,}",
        flush=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=0.0
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=4,
        threshold=1.0e-4,
        threshold_mode="abs",
        min_lr=MIN_LEARNING_RATE,
    )
    state_path = output_root / "latest_optimizer.pt"
    history = []
    best_metric = float("inf")
    best_epoch = 0
    stale = 0
    start_epoch = 1
    if args.resume and state_path.exists():
        state = torch.load(state_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        history = state["history"]
        best_metric = state["best_metric"]
        best_epoch = state["best_epoch"]
        stale = state["stale"]
        start_epoch = state["epoch"] + 1
    if (output_root / "completed.json").exists():
        print("SHARP_GUARD_ALREADY_COMPLETE", flush=True)
        return

    for epoch in range(start_epoch, EPOCHS + 1):
        epoch_seed = SEED + epoch * 1_000_003
        selected = np.random.default_rng(epoch_seed).choice(
            len(train_values), min(ROWS_PER_EPOCH, len(train_values)), replace=False
        )
        ids = torch.as_tensor(selected, dtype=torch.long, device=device)
        trained_nll = base.nll_pass(
            model,
            train_values[ids],
            train_context[ids],
            optimizer=optimizer,
            seed=epoch_seed,
        )
        trained_guard = sharp_guard_training_epoch(
            model, train_population, optimizer, epoch_seed
        )
        validation_nll = base.nll_pass(
            model,
            validation_values[validation_nll_ids],
            validation_context[validation_nll_ids],
        )
        validation_guard = evaluate_guard_response(
            model,
            validation_population,
            validation_guard_indices,
            draws=VALIDATION_GUARD_DRAWS,
            groups=VALIDATION_GUARD_GROUPS,
            batch_size=VALIDATION_GUARD_BATCH,
            seed=VALIDATION_GUARD_SEED,
        )
        metric = validation_nll["nll"] + validation_guard["loss"]
        if not math.isfinite(metric):
            raise ValueError("nonfinite sharp-guard validation objective")
        scheduler.step(metric)
        record = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_nll": trained_nll,
            "train_guard": trained_guard,
            "validation_nll": validation_nll,
            "validation_guard": compact_guard_result(validation_guard),
            "selection_metric": metric,
        }
        history.append(record)
        if metric < best_metric:
            best_metric = metric
            best_epoch = epoch
            stale = 0
            base.save_checkpoint(
                output_root / "selected.pt",
                model,
                condition,
                target,
                model_config,
                {
                    "training_phase": "full_domain_sharp_guard_refinement",
                    "epoch": epoch,
                    "validation_nll": validation_nll["nll"],
                    "guard_response_loss": validation_guard["loss"],
                    "selection_metric": metric,
                    "population": population,
                    "protocol_sha256": base.sha256(protocol_path),
                    "measurement_conditions_used": False,
                },
            )
        else:
            stale += 1
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "history": history,
                "best_metric": best_metric,
                "best_epoch": best_epoch,
                "stale": stale,
            },
            output_root / "latest_optimizer.tmp",
        )
        os.replace(output_root / "latest_optimizer.tmp", state_path)
        base.write_json(
            output_root / "history.json",
            {"history": history, "best_epoch": best_epoch, "best_metric": best_metric},
        )
        print(
            f"SHARP_GUARD epoch={epoch} nll={validation_nll['nll']:.8f} "
            f"guard={validation_guard['loss']:.8f} objective={metric:.8f} "
            f"best={best_metric:.8f}@{best_epoch} "
            f"lr={optimizer.param_groups[0]['lr']:.3g}",
            flush=True,
        )
        if stale >= 15 and optimizer.param_groups[0]["lr"] <= MIN_LEARNING_RATE:
            break

    selected_path = output_root / "selected.pt"
    checkpoint = torch.load(selected_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    final_nll = base.nll_pass(model, validation_values, validation_context)
    final_guard = evaluate_guard_response(
        model,
        validation_population,
        validation_guard_indices,
        draws=VALIDATION_GUARD_DRAWS,
        groups=VALIDATION_GUARD_GROUPS,
        batch_size=VALIDATION_GUARD_BATCH,
        seed=VALIDATION_GUARD_SEED,
    )
    base.write_json(
        output_root / "completed.json",
        {
            "checks_passed": True,
            "selected_checkpoint": str(selected_path),
            "selected_sha256": base.sha256(selected_path),
            "best_epoch": best_epoch,
            "stop_epoch": history[-1]["epoch"],
            "best_selection_metric": best_metric,
            "full_validation_nll": final_nll,
            "fixed_subset_guard_response": compact_guard_result(final_guard),
            "protocol_sha256": base.sha256(protocol_path),
            "limitations": [
                "single refinement seed",
                "finite generated-draw integration",
                "validation cases participate in checkpoint selection",
                "not an end-to-end SBSI calibration acceptance result",
            ],
        },
    )
    print(
        f"SHARP_GUARD_COMPLETE selected={selected_path} "
        f"nll={final_nll['nll']:.8f} guard={final_guard['loss']:.8f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
