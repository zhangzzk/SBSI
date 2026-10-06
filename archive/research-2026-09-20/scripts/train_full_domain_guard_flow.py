#!/usr/bin/env python3
"""Train a full-domain physical flow with guard-band response supervision.

Stage one is an ordinary conditional maximum-likelihood fit on every valid
measurement in each leg.  Stage two retains that full-domain NLL and adds a
global plus radius/magnitude guard-bank loss on the induced two-by-two mean
shape response.  Measured properties remain flow outputs and are never model
conditions.
"""

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
from sbsi.measurement_model import build_flow
from scripts import train_fixed_g0_flow as base


SEED = 611
NLL_EPOCHS = 140
GUARD_EPOCHS = 60
ROWS_PER_EPOCH = 4_000_000
BATCH_SIZE = 8192
LEARNING_RATE = 1.0e-4
MIN_LEARNING_RATE = 1.0e-7
GRADIENT_CLIP = 5.0
VALIDATION_NLL_ROWS = 500_000
VALIDATION_NLL_SEED = 900000611

RADIUS_THRESHOLDS = (2.8, 3.0, 3.2)
MAGNITUDE_THRESHOLDS = (25.6, 25.8, 26.0)
RADIUS_SOFTNESS = 0.05
MAGNITUDE_SOFTNESS = 0.05
GLOBAL_RESPONSE_SCALE = 0.01
GUARD_RESPONSE_SCALE = 0.02
GUARD_WEIGHT = 1.0
GUARD_STEPS_PER_EPOCH = 4
GUARD_PAIRS_PER_STEP = 32_768
GUARD_TRAINING_DRAWS = 4
GUARD_TRAINING_CHUNK = 1024
GUARD_TRAINING_SEED = 612_000_000_001
VALIDATION_GUARD_PAIRS = 200_000
VALIDATION_GUARD_DRAWS = 8
VALIDATION_GUARD_GROUPS = 4
VALIDATION_GUARD_BATCH = 2048
VALIDATION_GUARD_SEED = 613_000_000_001
VALIDATION_GUARD_SUBSET_SEED = 900000612


def guard_scales(cuts) -> np.ndarray:
    return np.asarray(
        [
            GLOBAL_RESPONSE_SCALE
            if cut["radius_min"] is None and cut["magnitude_max"] is None
            else GUARD_RESPONSE_SCALE
            for cut in cuts
        ],
        dtype=np.float64,
    )


def json_guard_statistics(population) -> dict:
    return {
        key: value
        for key, value in population.statistics.items()
        if key != "shape_normal"
    } | {"shape_normal": population.statistics["shape_normal"]}


def guard_training_epoch(model, population, optimizer, epoch_seed: int) -> dict:
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
            weight=GUARD_WEIGHT,
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


def compact_guard_result(result: dict) -> dict:
    return {
        key: value
        for key, value in result.items()
        if key != "group_means"
    }


def train(args: argparse.Namespace) -> None:
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("full-domain flow training must run under Slurm")
    domain_root = args.domain_root.resolve()
    output_root = args.output_root.resolve()
    manifest_path = domain_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    population = manifest["population"]
    expected_truth_cut = {"column": "r_input_p", "operator": "<", "value": 26.0}
    if (
        population.get("truth_analysis_cut") != expected_truth_cut
        or population.get("measured_selection_cut") is not None
        or not population.get("independent_per_leg_validity", False)
        or population.get("anchor_leg") is not None
    ):
        raise ValueError("guard-flow training requires an independently valid full domain")
    if manifest["flow"]["features"] != list(FLOW_FEATURES):
        raise ValueError("full-domain feature contract differs from the physical flow")
    if manifest["flow"]["targets"] != list(FLOW_TARGETS):
        raise ValueError("full-domain target contract differs from the physical flow")
    if output_root.exists() and not args.resume:
        raise FileExistsError(f"preserving existing full-domain training root: {output_root}")
    output_root.mkdir(parents=True, exist_ok=args.resume)

    print("LOADING_FULL_DOMAIN_FLOW_DATA", flush=True)
    train_data = base.load_split(domain_root, manifest["split"]["train_cases"])
    validation_data = base.load_split(
        domain_root, manifest["split"]["validation_cases"]
    )
    condition, target, model_config = base.build_transforms_and_config(train_data)
    train_context_np = condition.transform_frame(
        pd.DataFrame(train_data["context_raw"], columns=FLOW_FEATURES)
    )
    validation_context_np = condition.transform_frame(
        pd.DataFrame(validation_data["context_raw"], columns=FLOW_FEATURES)
    )
    train_values_np = target.transform_array(train_data["target_raw"])
    validation_values_np = target.transform_array(validation_data["target_raw"])

    device = torch.device("cuda")
    train_context = torch.as_tensor(train_context_np, dtype=torch.float32, device=device)
    train_values = torch.as_tensor(train_values_np, dtype=torch.float64, device=device)
    validation_context = torch.as_tensor(
        validation_context_np, dtype=torch.float32, device=device
    )
    validation_values = torch.as_tensor(
        validation_values_np, dtype=torch.float64, device=device
    )
    validation_ids_np = np.sort(
        np.random.default_rng(VALIDATION_NLL_SEED).choice(
            len(validation_values),
            min(VALIDATION_NLL_ROWS, len(validation_values)),
            replace=False,
        )
    )
    validation_ids = torch.as_tensor(
        validation_ids_np, dtype=torch.long, device=device
    )
    validation_guard_indices = np.sort(
        np.random.default_rng(VALIDATION_GUARD_SUBSET_SEED).choice(
            len(validation_data["pairs"]),
            min(VALIDATION_GUARD_PAIRS, len(validation_data["pairs"])),
            replace=False,
        )
    )
    cuts = make_guard_cuts(RADIUS_THRESHOLDS, MAGNITUDE_THRESHOLDS)
    scales = guard_scales(cuts)
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
        "population": population,
        "features": list(FLOW_FEATURES),
        "targets": list(FLOW_TARGETS),
        "model_config": model_config,
        "counts": {
            "train_nll_rows": len(train_values),
            "validation_nll_rows": len(validation_values),
            "train_matched_response_pairs": len(train_data["pairs"]),
            "validation_matched_response_pairs": len(validation_data["pairs"]),
            "train_leg_rows": train_data["leg_rows"],
            "validation_leg_rows": validation_data["leg_rows"],
            "train_per_case_matching": train_data["per_case_matching"],
            "validation_per_case_matching": validation_data["per_case_matching"],
        },
        "guard": {
            "cuts": list(cuts),
            "radius_softness_pixels": RADIUS_SOFTNESS,
            "magnitude_softness": MAGNITUDE_SOFTNESS,
            "global_response_scale": GLOBAL_RESPONSE_SCALE,
            "guard_response_scale": GUARD_RESPONSE_SCALE,
            "target_definition": (
                "first-order influence response of soft-cut mean measured shape"
            ),
            "train_statistics": json_guard_statistics(train_population),
            "validation_statistics": json_guard_statistics(validation_population),
        },
        "recipe": {
            "seed": SEED,
            "nll_epochs": NLL_EPOCHS,
            "guard_epochs": GUARD_EPOCHS,
            "rows_per_epoch": ROWS_PER_EPOCH,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "minimum_learning_rate": MIN_LEARNING_RATE,
            "gradient_clip": GRADIENT_CLIP,
            "validation_nll_rows": len(validation_ids_np),
            "guard_weight": GUARD_WEIGHT,
            "guard_steps_per_epoch": GUARD_STEPS_PER_EPOCH,
            "guard_pairs_per_step": GUARD_PAIRS_PER_STEP,
            "guard_training_draws": GUARD_TRAINING_DRAWS,
            "guard_training_chunk": GUARD_TRAINING_CHUNK,
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
            raise RuntimeError("resume protocol differs from existing guard-flow training")
    else:
        base.write_json(protocol_path, protocol)
        np.save(output_root / "validation_nll_indices.npy", validation_ids_np)
        np.save(
            output_root / "validation_guard_pair_indices.npy",
            validation_guard_indices,
        )
    print(
        f"FULL_DOMAIN_FLOW_DATA_READY train_nll={len(train_values):,} "
        f"validation_nll={len(validation_values):,} "
        f"train_pairs={len(train_data['pairs']):,} "
        f"validation_pairs={len(validation_data['pairs']):,} guards={len(cuts)}",
        flush=True,
    )

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    model = build_flow(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=4,
        threshold=1.0e-4,
        threshold_mode="abs",
        min_lr=MIN_LEARNING_RATE,
    )
    nll_dir = output_root / "nll"
    nll_dir.mkdir(exist_ok=args.resume)
    nll_state_path = nll_dir / "latest_optimizer.pt"
    nll_history = []
    best_nll = float("inf")
    best_nll_epoch = 0
    start_epoch = 1
    if (
        args.resume
        and nll_state_path.exists()
        and not (output_root / "nll_completed.json").exists()
    ):
        state = torch.load(nll_state_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        nll_history = state["history"]
        best_nll = state["best_nll"]
        best_nll_epoch = state["best_epoch"]
        start_epoch = state["epoch"] + 1

    if not (output_root / "nll_completed.json").exists():
        for epoch in range(start_epoch, NLL_EPOCHS + 1):
            epoch_seed = SEED + epoch * 1_000_003
            selected = np.random.default_rng(epoch_seed).choice(
                len(train_values), min(ROWS_PER_EPOCH, len(train_values)), replace=False
            )
            ids = torch.as_tensor(selected, dtype=torch.long, device=device)
            trained = base.nll_pass(
                model,
                train_values[ids],
                train_context[ids],
                optimizer=optimizer,
                seed=epoch_seed,
            )
            validation = base.nll_pass(
                model,
                validation_values[validation_ids],
                validation_context[validation_ids],
            )
            scheduler.step(validation["nll"])
            record = {
                "epoch": epoch,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "train": trained,
                "validation": validation,
            }
            nll_history.append(record)
            if validation["nll"] < best_nll:
                best_nll = validation["nll"]
                best_nll_epoch = epoch
                base.save_checkpoint(
                    nll_dir / "selected.pt",
                    model,
                    condition,
                    target,
                    model_config,
                    {
                        "training_phase": "full_domain_fresh_nll",
                        "epoch": epoch,
                        "validation_nll": best_nll,
                        "population": population,
                        "protocol_sha256": base.sha256(protocol_path),
                    },
                )
                torch.save(
                    {
                        "epoch": epoch,
                        "model": {
                            key: value.detach().cpu()
                            for key, value in model.state_dict().items()
                        },
                        "optimizer": base.cpu_optimizer_state(optimizer),
                        "learning_rate": optimizer.param_groups[0]["lr"],
                    },
                    nll_dir / "selected_optimizer.pt",
                )
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "history": nll_history,
                    "best_nll": best_nll,
                    "best_epoch": best_nll_epoch,
                },
                nll_dir / "latest_optimizer.tmp",
            )
            os.replace(nll_dir / "latest_optimizer.tmp", nll_state_path)
            base.write_json(
                nll_dir / "history.json",
                {"history": nll_history, "best_epoch": best_nll_epoch, "best_nll": best_nll},
            )
            print(
                f"FULL_DOMAIN_NLL epoch={epoch} train={trained['nll']:.8f} "
                f"validation={validation['nll']:.8f} best={best_nll:.8f}@{best_nll_epoch} "
                f"lr={optimizer.param_groups[0]['lr']:.3g}",
                flush=True,
            )
        base.write_json(
            output_root / "nll_completed.json",
            {
                "checks_passed": True,
                "best_epoch": best_nll_epoch,
                "best_validation_nll": best_nll,
                "selected_checkpoint": str(nll_dir / "selected.pt"),
                "selected_sha256": base.sha256(nll_dir / "selected.pt"),
            },
        )

    nll_selected = torch.load(
        nll_dir / "selected.pt", map_location=device, weights_only=False
    )
    selected_optimizer = torch.load(
        nll_dir / "selected_optimizer.pt", map_location=device, weights_only=False
    )
    model.load_state_dict(nll_selected["state_dict"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0)
    optimizer.load_state_dict(selected_optimizer["optimizer"])
    for group in optimizer.param_groups:
        group["lr"] = float(selected_optimizer["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=4,
        threshold=1.0e-4,
        threshold_mode="abs",
        min_lr=MIN_LEARNING_RATE,
    )
    guard_dir = output_root / "guard"
    guard_dir.mkdir(exist_ok=args.resume)
    guard_state_path = guard_dir / "latest_optimizer.pt"
    history = []
    best_metric = float("inf")
    best_phase_epoch = 0
    start_phase = 1
    stale = 0
    if args.resume and guard_state_path.exists():
        state = torch.load(guard_state_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        history = state["history"]
        best_metric = state["best_metric"]
        best_phase_epoch = state["best_epoch"]
        stale = state["stale"]
        start_phase = state["epoch"] + 1
    if (output_root / "completed.json").exists():
        print("FULL_DOMAIN_GUARD_FLOW_ALREADY_COMPLETE", flush=True)
        return

    for phase_epoch in range(start_phase, GUARD_EPOCHS + 1):
        epoch = selected_optimizer["epoch"] + phase_epoch
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
        trained_guard = guard_training_epoch(
            model, train_population, optimizer, epoch_seed
        )
        validation = base.nll_pass(
            model,
            validation_values[validation_ids],
            validation_context[validation_ids],
        )
        guard = evaluate_guard_response(
            model,
            validation_population,
            validation_guard_indices,
            draws=VALIDATION_GUARD_DRAWS,
            groups=VALIDATION_GUARD_GROUPS,
            batch_size=VALIDATION_GUARD_BATCH,
            seed=VALIDATION_GUARD_SEED,
        )
        metric = validation["nll"] + GUARD_WEIGHT * guard["loss"]
        if not math.isfinite(metric):
            raise ValueError("nonfinite full-domain guard validation objective")
        scheduler.step(metric)
        record = {
            "phase_epoch": phase_epoch,
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_nll": trained_nll,
            "train_guard": trained_guard,
            "validation_nll": validation,
            "validation_guard": compact_guard_result(guard),
            "selection_metric": metric,
        }
        history.append(record)
        if metric < best_metric:
            best_metric = metric
            best_phase_epoch = phase_epoch
            stale = 0
            base.save_checkpoint(
                guard_dir / "selected.pt",
                model,
                condition,
                target,
                model_config,
                {
                    "training_phase": "full_domain_guard_response",
                    "epoch": epoch,
                    "phase_epoch": phase_epoch,
                    "validation_nll": validation["nll"],
                    "guard_response_loss": guard["loss"],
                    "selection_metric": metric,
                    "population": population,
                    "protocol_sha256": base.sha256(protocol_path),
                    "guard_weight": GUARD_WEIGHT,
                    "guard_cuts": list(cuts),
                    "measurement_conditions_used": False,
                },
            )
        else:
            stale += 1
        torch.save(
            {
                "epoch": phase_epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "history": history,
                "best_metric": best_metric,
                "best_epoch": best_phase_epoch,
                "stale": stale,
            },
            guard_dir / "latest_optimizer.tmp",
        )
        os.replace(guard_dir / "latest_optimizer.tmp", guard_state_path)
        base.write_json(
            guard_dir / "history.json",
            {
                "history": history,
                "best_phase_epoch": best_phase_epoch,
                "best_metric": best_metric,
            },
        )
        print(
            f"FULL_DOMAIN_GUARD epoch={epoch} phase={phase_epoch} "
            f"nll={validation['nll']:.8f} guard={guard['loss']:.8f} "
            f"objective={metric:.8f} best={best_metric:.8f}@{best_phase_epoch} "
            f"lr={optimizer.param_groups[0]['lr']:.3g}",
            flush=True,
        )
        if stale >= 20 and optimizer.param_groups[0]["lr"] <= MIN_LEARNING_RATE:
            break

    selected_path = guard_dir / "selected.pt"
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
            "best_phase_epoch": best_phase_epoch,
            "stop_phase_epoch": history[-1]["phase_epoch"],
            "best_selection_metric": best_metric,
            "full_validation_nll": final_nll,
            "fixed_subset_guard_response": compact_guard_result(final_guard),
            "domain_manifest_sha256": base.sha256(manifest_path),
            "protocol_sha256": base.sha256(protocol_path),
            "limitations": [
                "single flow-training seed",
                "finite generated-draw integration",
                "soft guard thresholds approximate hard deployment cuts",
                "not an end-to-end SBSI calibration acceptance result",
            ],
        },
    )
    print(
        f"FULL_DOMAIN_GUARD_FLOW_COMPLETE selected={selected_path} "
        f"nll={final_nll['nll']:.8f} guard={final_guard['loss']:.8f}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
