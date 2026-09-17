#!/usr/bin/env python3
"""Train the joint physical flow on the fixed-g0 measured cohort.

Stage one is a fresh 140-epoch conditional NLL fit.  Stage two restores the
selected NLL checkpoint and AdamW moments, then runs at most 60 epochs of

    validation NLL + 10 * paired(shape + physical-radius response score).

The pair score uses the same fixed g=0 anchor, exact key intersections, common
antithetic random numbers between legs, two independent latent replicas, and
the historical divisor four.  Flux remains a joint NLL output but is not
response-supervised.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from sbsi.fixed_g0_domain import FLOW_FEATURES, FLOW_TARGETS, matched_key_indices
from sbsi.flow_disk import radial_atanh
from sbsi.flow_paired_shape import (
    PairedResponsePopulation,
    evaluate_paired_response,
    paired_response_backward,
    positive_shear_pair_mask,
)
from sbsi.flow_physical_disk import positive_inverse
from sbsi.measurement_model import (
    TargetStandardizer,
    build_flow,
    save_measurement_model,
)
from sbsi.selection_model import TabularPreprocessor


SEED = 501
NLL_EPOCHS = 140
RESPONSE_EPOCHS = 60
ROWS_PER_EPOCH = 4_000_000
BATCH_SIZE = 8192
LEARNING_RATE = 1.0e-4
MIN_LEARNING_RATE = 1.0e-7
GRADIENT_CLIP = 5.0
VALIDATION_NLL_ROWS = 500_000
VALIDATION_NLL_SEED = 900000504
RESPONSE_WEIGHT = 10.0
RESPONSE_PAIRS_PER_STEP = 1536
RESPONSE_TRAINING_DRAWS = 16
RESPONSE_TRAINING_CHUNK = 256
RESPONSE_TRAINING_SEED = 3250000000001
VALIDATION_RESPONSE_PAIRS = 50_000
VALIDATION_RESPONSE_DRAWS = 64
VALIDATION_RESPONSE_GROUPS = 4
VALIDATION_RESPONSE_BATCH = 1024
VALIDATION_RESPONSE_SEED = 3260000000001
VALIDATION_RESPONSE_SUBSET_SEED = 2671000501
RADIUS_SCALE_PIXELS = 1.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8_388_608), b""):
            digest.update(block)
    return digest.hexdigest()


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_ready(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    os.replace(temporary, path)


def load_case(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as stored:
        result = {name: stored[name] for name in stored.files}
    expected = {"case", "input_index", "context", "target", "gamma"}
    if set(result) != expected:
        raise ValueError(f"unexpected prepared arrays in {path}: {sorted(result)}")
    n = len(result["case"])
    if (
        result["input_index"].shape != (n,)
        or result["context"].shape != (n, len(FLOW_FEATURES))
        or result["target"].shape != (n, len(FLOW_TARGETS))
        or result["gamma"].shape != (n, 2)
    ):
        raise ValueError(f"misaligned prepared arrays in {path}")
    if (
        not np.isfinite(result["context"]).all()
        or not np.isfinite(result["target"]).all()
        or not np.isfinite(result["gamma"]).all()
    ):
        raise ValueError(f"non-finite prepared flow row in {path}")
    return result


def load_split(domain_root: Path, cases) -> dict:
    contexts = []
    targets = []
    pairs = []
    deltas = []
    pair_zero_targets = []
    pair_shear_targets = []
    pair_cases = []
    gammas = []
    per_case = []
    offset = 0
    leg_rows = {"g0": 0, "g05": 0}
    for case in map(int, cases):
        zero = load_case(domain_root / "flow" / "g0" / f"case{case:03d}.npz")
        shear = load_case(domain_root / "flow" / "g05" / f"case{case:03d}.npz")
        if not np.all(zero["case"] == case) or not np.all(shear["case"] == case):
            raise ValueError(f"prepared case label mismatch for case{case:03d}")
        left, right, counts = matched_key_indices(
            zero["case"],
            zero["input_index"],
            shear["case"],
            shear["input_index"],
        )
        if len(left) == 0:
            raise RuntimeError(f"case{case:03d} has no usable matched response pairs")
        pair_gamma = shear["gamma"][right]
        response_eligible = positive_shear_pair_mask(pair_gamma)
        excluded_zero_shear = int((~response_eligible).sum())
        left = left[response_eligible]
        right = right[response_eligible]
        pair_gamma = pair_gamma[response_eligible]
        if len(left) == 0:
            raise RuntimeError(
                f"case{case:03d} has no matched pair with nonzero response shear"
            )
        n0 = len(zero["case"])
        ng = len(shear["case"])
        contexts.extend((zero["context"], shear["context"]))
        targets.extend((zero["target"], shear["target"]))
        pairs.append(
            np.column_stack((offset + left, offset + n0 + right)).astype(np.int64)
        )
        pair_zero_targets.append(zero["target"][left])
        pair_shear_targets.append(shear["target"][right])
        pair_cases.append(np.full(len(left), case, dtype=np.int32))
        deltas.append((shear["target"][right, :3] - zero["target"][left, :3]))
        gammas.append(pair_gamma)
        per_case.append(
            {
                "case": case,
                **counts,
                "response_pairs": int(len(left)),
                "excluded_zero_shear_pairs": excluded_zero_shear,
            }
        )
        offset += n0 + ng
        leg_rows["g0"] += n0
        leg_rows["g05"] += ng
    return {
        "context_raw": np.concatenate(contexts).astype(np.float32, copy=False),
        "target_raw": np.concatenate(targets).astype(np.float64, copy=False),
        "pairs": np.concatenate(pairs),
        "pair_zero_target": np.concatenate(pair_zero_targets).astype(
            np.float64, copy=False
        ),
        "pair_shear_target": np.concatenate(pair_shear_targets).astype(
            np.float64, copy=False
        ),
        "pair_case": np.concatenate(pair_cases),
        "measured_delta": np.concatenate(deltas).astype(np.float64, copy=False),
        "gamma": np.concatenate(gammas).astype(np.float64, copy=False),
        "per_case_matching": per_case,
        "leg_rows": leg_rows,
    }


def build_transforms_and_config(train: dict):
    context_frame = pd.DataFrame(train["context_raw"], columns=FLOW_FEATURES)
    condition = TabularPreprocessor.fit(
        context_frame, FLOW_FEATURES, add_missing_indicators=True
    )
    target_frame = pd.DataFrame(train["target_raw"], columns=FLOW_TARGETS)
    target = TargetStandardizer.fit(target_frame, FLOW_TARGETS, dtype="float64")

    physical = torch.as_tensor(train["target_raw"], dtype=torch.float64)
    shape_coordinate, _ = radial_atanh(physical[:, :2])
    units = torch.as_tensor(target.scales[2:], dtype=torch.float64)
    positive_coordinate = positive_inverse(physical[:, 2:] / units)
    disk_means = shape_coordinate.mean(0)
    disk_scales = shape_coordinate.std(0, correction=1)
    physical_coordinate_means = positive_coordinate.mean(0)
    physical_coordinate_scales = positive_coordinate.std(0, correction=1)
    for name, value in (
        ("disk coordinate scales", disk_scales),
        ("physical coordinate scales", physical_coordinate_scales),
    ):
        if not bool(torch.isfinite(value).all() & (value > 0).all()):
            raise ValueError(f"invalid {name}")
    config = {
        "flow_type": "physical_disk_affine",
        "target_dim": len(FLOW_TARGETS),
        "context_dim": condition.output_dim,
        "hidden_dim": 256,
        "n_layers": 3,
        "n_flows": 10,
        "scale_limit": 3.0,
        "activation": "silu",
        "num_bins": 8,
        "tail_bound": 5.0,
        "mean_hidden": 0,
        "disk_map": "radial_tanh",
        "disk_shape_means": target.means[:2].tolist(),
        "disk_shape_scales": target.scales[:2].tolist(),
        "disk_coordinate_means": disk_means.numpy().tolist(),
        "disk_coordinate_scales": disk_scales.numpy().tolist(),
        "physical_map": "softplus",
        "physical_means": target.means[2:].tolist(),
        "physical_scales": target.scales[2:].tolist(),
        "physical_units": units.numpy().tolist(),
        "physical_coordinate_means": physical_coordinate_means.numpy().tolist(),
        "physical_coordinate_scales": physical_coordinate_scales.numpy().tolist(),
    }
    return condition, target, config


def nll_pass(model, values, context, *, optimizer=None, seed=None):
    training = optimizer is not None
    model.train(training)
    if training:
        generator = torch.Generator(device=values.device).manual_seed(int(seed))
        order = torch.randperm(len(values), generator=generator, device=values.device)
    else:
        order = torch.arange(len(values), device=values.device)
    total = 0.0
    norms = []
    started = time.monotonic()
    for rows in order.split(BATCH_SIZE):
        if training:
            optimizer.zero_grad(set_to_none=True)
        log_probability = model.log_prob(values[rows], context[rows])
        loss = -log_probability.mean()
        if not bool(torch.isfinite(loss)):
            raise ValueError("non-finite flow NLL")
        if training:
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), GRADIENT_CLIP, error_if_nonfinite=True
            )
            optimizer.step()
            norms.append(float(norm))
        total += float(loss.detach()) * len(rows)
    return {
        "nll": total / len(values),
        "rows": int(len(values)),
        "optimizer_steps": len(norms),
        "gradient_norm_mean": float(np.mean(norms)) if norms else 0.0,
        "gradient_norm_max": float(np.max(norms)) if norms else 0.0,
        "clipped_step_fraction": float(np.mean(np.asarray(norms) > GRADIENT_CLIP))
        if norms
        else 0.0,
        "seconds": time.monotonic() - started,
    }


def response_epoch(model, values, context, population, optimizer, *, seed):
    model.train()
    generator = torch.Generator(device=values.device).manual_seed(int(seed))
    order = torch.randperm(len(values), generator=generator, device=values.device)
    nll_total = 0.0
    response_total = 0.0
    component_total = np.zeros(3, dtype=float)
    norms = []
    started = time.monotonic()
    for step, rows in enumerate(order.split(BATCH_SIZE)):
        optimizer.zero_grad(set_to_none=True)
        nll = -model.log_prob(values[rows], context[rows]).mean()
        if not bool(torch.isfinite(nll)):
            raise ValueError("non-finite response-stage NLL")
        nll.backward()
        response = paired_response_backward(
            model,
            population,
            RESPONSE_PAIRS_PER_STEP,
            RESPONSE_TRAINING_DRAWS,
            chunk_size=RESPONSE_TRAINING_CHUNK,
            seed=RESPONSE_TRAINING_SEED + seed * 1_000_003 + step * 100_003,
            weight=RESPONSE_WEIGHT,
        )
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), GRADIENT_CLIP, error_if_nonfinite=True
        )
        optimizer.step()
        nll_total += float(nll.detach()) * len(rows)
        response_total += response["loss"] * len(rows)
        component_total += response["component_losses"] * len(rows)
        norms.append(float(norm))
    return {
        "nll": nll_total / len(values),
        "response_loss": response_total / len(values),
        "response_component_losses": (component_total / len(values)).tolist(),
        "rows": int(len(values)),
        "optimizer_steps": len(norms),
        "gradient_norm_mean": float(np.mean(norms)),
        "gradient_norm_max": float(np.max(norms)),
        "clipped_step_fraction": float(np.mean(np.asarray(norms) > GRADIENT_CLIP)),
        "seconds": time.monotonic() - started,
    }


def save_checkpoint(path, model, condition, target, model_config, metadata):
    temporary = path.with_suffix(path.suffix + ".tmp")
    save_measurement_model(
        temporary, model, condition, target, model_config, metadata=metadata
    )
    os.replace(temporary, path)


def cpu_optimizer_state(optimizer):
    state = copy.deepcopy(optimizer.state_dict())
    for values in state["state"].values():
        for key, value in values.items():
            if torch.is_tensor(value):
                values[key] = value.detach().cpu()
    return state


def train(args: argparse.Namespace) -> None:
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("physical-flow training must run under Slurm")
    domain_root = args.domain_root.resolve()
    output_root = args.output_root.resolve()
    manifest_path = domain_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["population"]["truth_analysis_cut"] is not None:
        raise ValueError("flow training refuses a truth-cut fixed-domain manifest")
    if manifest["population"]["sheared_leg_recut"]:
        raise ValueError("flow training refuses a sheared-leg selection recut")
    if output_root.exists() and not args.resume:
        raise FileExistsError(f"preserving existing flow training root: {output_root}")
    output_root.mkdir(parents=True, exist_ok=args.resume)

    print("LOADING_FIXED_G0_FLOW_DATA", flush=True)
    train_data = load_split(domain_root, manifest["split"]["train_cases"])
    validation_data = load_split(domain_root, manifest["split"]["validation_cases"])
    condition, target, model_config = build_transforms_and_config(train_data)
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
    validation_ids = torch.as_tensor(validation_ids_np, dtype=torch.long, device=device)
    response_subset = np.sort(
        np.random.default_rng(VALIDATION_RESPONSE_SUBSET_SEED).choice(
            len(validation_data["pairs"]),
            min(VALIDATION_RESPONSE_PAIRS, len(validation_data["pairs"])),
            replace=False,
        )
    )
    train_population = PairedResponsePopulation(
        train_context,
        train_data["pairs"],
        train_data["measured_delta"],
        train_data["gamma"],
        radius_scale=RADIUS_SCALE_PIXELS,
    )
    validation_population = PairedResponsePopulation(
        validation_context,
        validation_data["pairs"],
        validation_data["measured_delta"],
        validation_data["gamma"],
        radius_scale=RADIUS_SCALE_PIXELS,
    )
    protocol = {
        "format_version": 1,
        "domain_manifest": str(manifest_path),
        "domain_manifest_sha256": sha256(manifest_path),
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
            "seed": SEED,
            "nll_epochs": NLL_EPOCHS,
            "response_epochs": RESPONSE_EPOCHS,
            "rows_per_epoch": ROWS_PER_EPOCH,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "minimum_learning_rate": MIN_LEARNING_RATE,
            "gradient_clip": GRADIENT_CLIP,
            "validation_nll_rows": len(validation_ids_np),
            "response_weight": RESPONSE_WEIGHT,
            "response_pairs_per_step": RESPONSE_PAIRS_PER_STEP,
            "response_training_draws": RESPONSE_TRAINING_DRAWS,
            "response_training_chunk": RESPONSE_TRAINING_CHUNK,
            "response_components": ["ngmix_g1", "ngmix_g2", "flux_radius_pixels"],
            "response_divisor": 4.0,
            "response_pair_weighting": "uniform matched fixed-anchor pairs",
            "flux_response_supervision": False,
            "validation_response_pairs": len(response_subset),
            "validation_response_draws": VALIDATION_RESPONSE_DRAWS,
            "validation_response_groups": VALIDATION_RESPONSE_GROUPS,
        },
        "script_sha256": sha256(Path(__file__).resolve()),
    }
    protocol_path = output_root / "protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != json_ready(protocol):
            raise RuntimeError("resume protocol differs from existing flow training")
    else:
        write_json(protocol_path, protocol)
        np.save(output_root / "validation_nll_indices.npy", validation_ids_np)
        np.save(output_root / "validation_response_pair_indices.npy", response_subset)
    print(
        f"FIXED_G0_FLOW_DATA_READY train_nll={len(train_values):,} "
        f"validation_nll={len(validation_values):,} "
        f"train_pairs={len(train_data['pairs']):,} "
        f"validation_pairs={len(validation_data['pairs']):,}",
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
    if args.resume and nll_state_path.exists() and not (output_root / "nll_completed.json").exists():
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
            trained = nll_pass(
                model, train_values[ids], train_context[ids], optimizer=optimizer, seed=epoch_seed
            )
            validation = nll_pass(
                model, validation_values[validation_ids], validation_context[validation_ids]
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
                save_checkpoint(
                    nll_dir / "selected.pt",
                    model,
                    condition,
                    target,
                    model_config,
                    {
                        "training_phase": "fixed_g0_fresh_nll",
                        "epoch": epoch,
                        "validation_nll": best_nll,
                        "population": manifest["population"],
                        "protocol_sha256": sha256(protocol_path),
                    },
                )
                torch.save(
                    {
                        "epoch": epoch,
                        "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                        "optimizer": cpu_optimizer_state(optimizer),
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
            write_json(
                nll_dir / "history.json",
                {"history": nll_history, "best_epoch": best_nll_epoch, "best_nll": best_nll},
            )
            print(
                f"FIXED_G0_NLL epoch={epoch} train={trained['nll']:.8f} "
                f"validation={validation['nll']:.8f} best={best_nll:.8f}@{best_nll_epoch} "
                f"lr={optimizer.param_groups[0]['lr']:.3g}",
                flush=True,
            )
        write_json(
            output_root / "nll_completed.json",
            {
                "checks_passed": True,
                "best_epoch": best_nll_epoch,
                "best_validation_nll": best_nll,
                "selected_checkpoint": str(nll_dir / "selected.pt"),
                "selected_sha256": sha256(nll_dir / "selected.pt"),
            },
        )

    nll_selected = torch.load(nll_dir / "selected.pt", map_location=device, weights_only=False)
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
    response_dir = output_root / "paired"
    response_dir.mkdir(exist_ok=args.resume)
    response_state_path = response_dir / "latest_optimizer.pt"
    response_history = []
    best_metric = float("inf")
    best_phase_epoch = 0
    response_start = 1
    stale = 0
    if args.resume and response_state_path.exists():
        state = torch.load(response_state_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        response_history = state["history"]
        best_metric = state["best_metric"]
        best_phase_epoch = state["best_epoch"]
        stale = state["stale"]
        response_start = state["epoch"] + 1

    if (output_root / "completed.json").exists():
        print("FIXED_G0_FLOW_ALREADY_COMPLETE", flush=True)
        return
    for phase_epoch in range(response_start, RESPONSE_EPOCHS + 1):
        epoch = selected_optimizer["epoch"] + phase_epoch
        epoch_seed = SEED + epoch * 1_000_003
        selected = np.random.default_rng(epoch_seed).choice(
            len(train_values), min(ROWS_PER_EPOCH, len(train_values)), replace=False
        )
        ids = torch.as_tensor(selected, dtype=torch.long, device=device)
        trained = response_epoch(
            model,
            train_values[ids],
            train_context[ids],
            train_population,
            optimizer,
            seed=epoch_seed,
        )
        validation = nll_pass(
            model, validation_values[validation_ids], validation_context[validation_ids]
        )
        response = evaluate_paired_response(
            model,
            validation_population,
            response_subset,
            draws=VALIDATION_RESPONSE_DRAWS,
            groups=VALIDATION_RESPONSE_GROUPS,
            batch_size=VALIDATION_RESPONSE_BATCH,
            seed=VALIDATION_RESPONSE_SEED,
        )
        metric = validation["nll"] + RESPONSE_WEIGHT * response["response_loss"]
        if not math.isfinite(metric):
            raise ValueError("non-finite paired flow validation objective")
        scheduler.step(metric)
        record = {
            "phase_epoch": phase_epoch,
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": trained,
            "validation": validation,
            "response": {key: value for key, value in response.items() if key != "group_residuals" and key != "per_pair"},
            "selection_metric": metric,
        }
        response_history.append(record)
        if metric < best_metric:
            best_metric = metric
            best_phase_epoch = phase_epoch
            stale = 0
            save_checkpoint(
                response_dir / "selected.pt",
                model,
                condition,
                target,
                model_config,
                {
                    "training_phase": "fixed_g0_paired_shape_radius",
                    "epoch": epoch,
                    "phase_epoch": phase_epoch,
                    "validation_nll": validation["nll"],
                    "paired_response_loss": response["response_loss"],
                    "selection_metric": metric,
                    "population": manifest["population"],
                    "truth_analysis_cut": None,
                    "sheared_leg_recut": False,
                    "protocol_sha256": sha256(protocol_path),
                    "response_weight": RESPONSE_WEIGHT,
                    "response_components": ["shape", "radius"],
                    "flux_response_supervision": False,
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
                "history": response_history,
                "best_metric": best_metric,
                "best_epoch": best_phase_epoch,
                "stale": stale,
            },
            response_dir / "latest_optimizer.tmp",
        )
        os.replace(response_dir / "latest_optimizer.tmp", response_state_path)
        write_json(
            response_dir / "history.json",
            {
                "history": response_history,
                "best_phase_epoch": best_phase_epoch,
                "best_metric": best_metric,
            },
        )
        print(
            f"FIXED_G0_PAIRED epoch={epoch} phase={phase_epoch} "
            f"nll={validation['nll']:.8f} response={response['response_loss']:.8f} "
            f"objective={metric:.8f} best={best_metric:.8f}@{best_phase_epoch} "
            f"lr={optimizer.param_groups[0]['lr']:.3g}",
            flush=True,
        )
        if stale >= 20 and optimizer.param_groups[0]["lr"] <= MIN_LEARNING_RATE:
            break

    selected_path = response_dir / "selected.pt"
    checkpoint = torch.load(selected_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    final_nll = nll_pass(model, validation_values, validation_context)
    final_response = evaluate_paired_response(
        model,
        validation_population,
        response_subset,
        draws=VALIDATION_RESPONSE_DRAWS,
        groups=VALIDATION_RESPONSE_GROUPS,
        batch_size=VALIDATION_RESPONSE_BATCH,
        seed=VALIDATION_RESPONSE_SEED,
    )
    write_json(
        output_root / "completed.json",
        {
            "checks_passed": True,
            "selected_checkpoint": str(selected_path),
            "selected_sha256": sha256(selected_path),
            "best_phase_epoch": best_phase_epoch,
            "stop_phase_epoch": response_history[-1]["phase_epoch"],
            "best_selection_metric": best_metric,
            "full_validation_nll": final_nll,
            "fixed_subset_response": {
                key: value
                for key, value in final_response.items()
                if key not in ("group_residuals", "per_pair")
            },
            "domain_manifest_sha256": sha256(manifest_path),
            "protocol_sha256": sha256(protocol_path),
            "limitations": [
                "single flow-training seed",
                "finite paired-response Monte Carlo",
                "fixed g=0 cohort changes likelihood conditioning semantics",
                "not an end-to-end shear-calibration acceptance result",
            ],
        },
    )
    print(
        f"FIXED_G0_FLOW_COMPLETE selected={selected_path} "
        f"nll={final_nll['nll']:.8f} response={final_response['response_loss']:.8f}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
