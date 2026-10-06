#!/usr/bin/env python3
"""Train a fresh fixed-g0 flow with response supervision from epoch one.

This is the schedule-control counterpart to ``train_fixed_g0_flow``.  It uses
the same data, split, architecture, NLL batches, paired-response estimator,
loss weight, and validation subsets, but it does not perform or restore an
NLL-only warm-up.  Every optimizer epoch starts from the joint objective

    NLL + 10 * paired(shape + physical-radius response score).

Flux remains a joint density output and is not response-supervised.
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
from sbsi.flow_paired_shape import PairedResponsePopulation, evaluate_paired_response
from sbsi.measurement_model import build_flow
from scripts import train_fixed_g0_flow as base


MAXIMUM_EPOCHS = base.NLL_EPOCHS + base.RESPONSE_EPOCHS
EARLY_STOPPING_STALE = 20
MIN_LEARNING_RATE = base.MIN_LEARNING_RATE


def train(args: argparse.Namespace) -> None:
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("direct-response flow training must run under Slurm")
    domain_root = args.domain_root.resolve()
    output_root = args.output_root.resolve()
    manifest_path = domain_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["population"]["truth_analysis_cut"] is not None:
        raise ValueError("flow training refuses a truth-cut fixed-domain manifest")
    if manifest["population"]["sheared_leg_recut"]:
        raise ValueError("flow training refuses a sheared-leg selection recut")
    if output_root.exists() and not args.resume:
        raise FileExistsError(
            f"preserving existing direct-response training root: {output_root}"
        )
    output_root.mkdir(parents=True, exist_ok=args.resume)

    print("LOADING_FIXED_G0_DIRECT_RESPONSE_DATA", flush=True)
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
    train_context = torch.as_tensor(
        train_context_np, dtype=torch.float32, device=device
    )
    train_values = torch.as_tensor(
        train_values_np, dtype=torch.float64, device=device
    )
    validation_context = torch.as_tensor(
        validation_context_np, dtype=torch.float32, device=device
    )
    validation_values = torch.as_tensor(
        validation_values_np, dtype=torch.float64, device=device
    )
    validation_ids_np = np.sort(
        np.random.default_rng(base.VALIDATION_NLL_SEED).choice(
            len(validation_values),
            min(base.VALIDATION_NLL_ROWS, len(validation_values)),
            replace=False,
        )
    )
    validation_ids = torch.as_tensor(
        validation_ids_np, dtype=torch.long, device=device
    )
    response_subset = np.sort(
        np.random.default_rng(base.VALIDATION_RESPONSE_SUBSET_SEED).choice(
            len(validation_data["pairs"]),
            min(base.VALIDATION_RESPONSE_PAIRS, len(validation_data["pairs"])),
            replace=False,
        )
    )
    train_population = PairedResponsePopulation(
        train_context,
        train_data["pairs"],
        train_data["measured_delta"],
        train_data["gamma"],
        radius_scale=base.RADIUS_SCALE_PIXELS,
    )
    validation_population = PairedResponsePopulation(
        validation_context,
        validation_data["pairs"],
        validation_data["measured_delta"],
        validation_data["gamma"],
        radius_scale=base.RADIUS_SCALE_PIXELS,
    )

    protocol = {
        "format_version": 1,
        "schedule": "joint_nll_plus_paired_response_from_epoch_one",
        "from_scratch": True,
        "warm_start_checkpoint": None,
        "domain_manifest": str(manifest_path),
        "domain_manifest_sha256": base.sha256(manifest_path),
        "population": manifest["population"],
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
        "recipe": {
            "seed": base.SEED,
            "maximum_epochs": MAXIMUM_EPOCHS,
            "early_stopping_stale": EARLY_STOPPING_STALE,
            "rows_per_epoch": base.ROWS_PER_EPOCH,
            "batch_size": base.BATCH_SIZE,
            "learning_rate": base.LEARNING_RATE,
            "minimum_learning_rate": MIN_LEARNING_RATE,
            "gradient_clip": base.GRADIENT_CLIP,
            "validation_nll_rows": len(validation_ids_np),
            "response_weight": base.RESPONSE_WEIGHT,
            "response_pairs_per_step": base.RESPONSE_PAIRS_PER_STEP,
            "response_training_draws": base.RESPONSE_TRAINING_DRAWS,
            "response_training_chunk": base.RESPONSE_TRAINING_CHUNK,
            "response_components": [
                "ngmix_g1",
                "ngmix_g2",
                "flux_radius_pixels",
            ],
            "response_divisor": 4.0,
            "response_pair_weighting": "uniform matched fixed-anchor pairs",
            "flux_response_supervision": False,
            "validation_response_pairs": len(response_subset),
            "validation_response_draws": base.VALIDATION_RESPONSE_DRAWS,
            "validation_response_groups": base.VALIDATION_RESPONSE_GROUPS,
        },
        "implementation_sha256": base.sha256(Path(__file__).resolve()),
        "shared_training_implementation_sha256": base.sha256(
            Path(base.__file__).resolve()
        ),
    }
    protocol_path = output_root / "protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != base.json_ready(protocol):
            raise RuntimeError(
                "resume protocol differs from existing direct-response training"
            )
    else:
        base.write_json(protocol_path, protocol)
        np.save(output_root / "validation_nll_indices.npy", validation_ids_np)
        np.save(
            output_root / "validation_response_pair_indices.npy", response_subset
        )
    print(
        f"FIXED_G0_DIRECT_DATA_READY train_nll={len(train_values):,} "
        f"validation_nll={len(validation_values):,} "
        f"train_pairs={len(train_data['pairs']):,} "
        f"validation_pairs={len(validation_data['pairs']):,}",
        flush=True,
    )

    torch.manual_seed(base.SEED)
    torch.cuda.manual_seed_all(base.SEED)
    model = build_flow(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=base.LEARNING_RATE, weight_decay=0.0
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
        print("FIXED_G0_DIRECT_FLOW_ALREADY_COMPLETE", flush=True)
        return
    for epoch in range(start_epoch, MAXIMUM_EPOCHS + 1):
        epoch_seed = base.SEED + epoch * 1_000_003
        selected = np.random.default_rng(epoch_seed).choice(
            len(train_values),
            min(base.ROWS_PER_EPOCH, len(train_values)),
            replace=False,
        )
        ids = torch.as_tensor(selected, dtype=torch.long, device=device)
        trained = base.response_epoch(
            model,
            train_values[ids],
            train_context[ids],
            train_population,
            optimizer,
            seed=epoch_seed,
        )
        validation = base.nll_pass(
            model,
            validation_values[validation_ids],
            validation_context[validation_ids],
        )
        response = evaluate_paired_response(
            model,
            validation_population,
            response_subset,
            draws=base.VALIDATION_RESPONSE_DRAWS,
            groups=base.VALIDATION_RESPONSE_GROUPS,
            batch_size=base.VALIDATION_RESPONSE_BATCH,
            seed=base.VALIDATION_RESPONSE_SEED,
        )
        metric = validation["nll"] + base.RESPONSE_WEIGHT * response["response_loss"]
        if not math.isfinite(metric):
            raise ValueError("non-finite direct-response validation objective")
        scheduler.step(metric)
        record = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": trained,
            "validation": validation,
            "response": {
                key: value
                for key, value in response.items()
                if key not in ("group_residuals", "per_pair")
            },
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
                    "training_phase": "fixed_g0_direct_paired_shape_radius",
                    "from_scratch": True,
                    "warm_start_checkpoint": None,
                    "epoch": epoch,
                    "validation_nll": validation["nll"],
                    "paired_response_loss": response["response_loss"],
                    "selection_metric": metric,
                    "population": manifest["population"],
                    "truth_analysis_cut": None,
                    "sheared_leg_recut": False,
                    "protocol_sha256": base.sha256(protocol_path),
                    "response_weight": base.RESPONSE_WEIGHT,
                    "response_components": ["shape", "radius"],
                    "flux_response_supervision": False,
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
            {
                "history": history,
                "best_epoch": best_epoch,
                "best_metric": best_metric,
            },
        )
        print(
            f"FIXED_G0_DIRECT epoch={epoch} nll={validation['nll']:.8f} "
            f"response={response['response_loss']:.8f} objective={metric:.8f} "
            f"best={best_metric:.8f}@{best_epoch} "
            f"lr={optimizer.param_groups[0]['lr']:.3g}",
            flush=True,
        )
        if stale >= EARLY_STOPPING_STALE and (
            optimizer.param_groups[0]["lr"] <= MIN_LEARNING_RATE
        ):
            break

    selected_path = output_root / "selected.pt"
    checkpoint = torch.load(selected_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    final_nll = base.nll_pass(model, validation_values, validation_context)
    final_response = evaluate_paired_response(
        model,
        validation_population,
        response_subset,
        draws=base.VALIDATION_RESPONSE_DRAWS,
        groups=base.VALIDATION_RESPONSE_GROUPS,
        batch_size=base.VALIDATION_RESPONSE_BATCH,
        seed=base.VALIDATION_RESPONSE_SEED,
    )
    base.write_json(
        output_root / "completed.json",
        {
            "checks_passed": True,
            "schedule": "joint_nll_plus_paired_response_from_epoch_one",
            "from_scratch": True,
            "warm_start_checkpoint": None,
            "selected_checkpoint": str(selected_path),
            "selected_sha256": base.sha256(selected_path),
            "best_epoch": best_epoch,
            "stop_epoch": history[-1]["epoch"],
            "best_selection_metric": best_metric,
            "full_validation_nll": final_nll,
            "fixed_subset_response": {
                key: value
                for key, value in final_response.items()
                if key not in ("group_residuals", "per_pair")
            },
            "domain_manifest_sha256": base.sha256(manifest_path),
            "protocol_sha256": base.sha256(protocol_path),
            "limitations": [
                "single direct-response training seed",
                "finite paired-response Monte Carlo",
                "fixed g=0 cohort changes likelihood conditioning semantics",
                "not an end-to-end shear-calibration acceptance result",
            ],
        },
    )
    print(
        f"FIXED_G0_DIRECT_FLOW_COMPLETE selected={selected_path} "
        f"nll={final_nll['nll']:.8f} "
        f"response={final_response['response_loss']:.8f}",
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
