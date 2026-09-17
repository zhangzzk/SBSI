#!/usr/bin/env python3
"""Refine a full-domain flow with combined-gradient, band-resolved guards.

This supersedes :mod:`scripts.refine_full_domain_guard_flow` on three points
that the per-leg hard-cut measurement showed to be load bearing.

**The two objectives are now one step.**  The earlier recipe ran a full
likelihood epoch, clipping every one of its 489 minibatch steps, and then ran
four separate guard steps, clipping each of those as well.  Both gradient
norms sit far above the clip (10--20 for the likelihood, 45--105 for the
guard, against a clip of 5), so every step of both objectives was renormalised
to the same length.  The relative weight of the two objectives was therefore
not ``response_scale`` and not any loss scale at all -- it was the step count,
489 to 4.  Rescaling the guard loss, to a relative error or anything else,
cannot change a direction that is renormalised immediately afterwards.  Here
the guard gradient is accumulated into the same buffer as the likelihood
gradient and the sum is clipped once, so ``GUARD_WEIGHT`` sets the balance
directly.  This is the composition
:func:`scripts.train_fixed_g0_flow.response_epoch` already uses for the paired
response.

**The guards now resolve the cut boundary.**  A threshold guard
``radius > 3`` constrains the mean response of every object above three
pixels, which is dominated by the well-resolved majority.  The per-leg
measurement localised the entire residual bias to the two tenths of a pixel
between 2.8 and 3.0, where the response crosses zero just above the PSF
half-light radius, and showed the largest objects biased with the opposite
sign.  A cumulative guard averages those against each other and can read as
satisfied while both are wrong.  The bank therefore adds disjoint radius
bands, whose residuals cannot cancel.

**The near-zero off-diagonals no longer dilute the loss.**  The guard score
averages over ``R11``, ``R12``, ``R21`` and ``R22`` with equal weight, but the
off-diagonals are about 0.4% of ``R11`` and carry no signal, so half of the
averaged gradient was spent on them.  They are kept in the bank, at a tenth
of the diagonal weight, rather than dropped.

Two further points follow from measuring the first version of this recipe on
held-out cases.  The band residuals fell by a factor of 3.5 and the fall
survived a selection-free re-measurement of the final epoch, so the bands work.
The global bias did not fall, and the number reported from ``selected.pt``
turned out to be one lucky draw.

**The cumulative cuts are now weighted to match the bands.**  Band residuals
start about ten times the cumulative ones, so under a uniform
``RESPONSE_SCALE`` the averaged quadratic was roughly fifty times more
sensitive to the nine bands than to the sixteen cumulative cuts.  The cuts
that pin the overall level of the response contributed almost nothing to the
gradient, and the level then drifted freely: over twenty epochs the global
residual wandered between -0.0011 and +0.0017 on ``R_self ~ 0.42``, about
+-1.3% in ``m``, with no trend.  ``CUMULATIVE_SCALE_FACTOR`` divides their
scale by ten, multiplying their share of the loss by a hundred.

**The reportable checkpoint is an epoch average, not a selected epoch.**
``selected.pt`` minimises a validation objective whose guard term is a noisy
Monte Carlo estimate on the same cases the evaluation then measures, so
selecting on it harvests whichever epoch drifted closest to zero on those
cases.  ``averaged.pt`` averages the weights over the tail of the run, which
estimates a level that wanders without a trend far better than any single
epoch and involves no choice made on the measured cases at all.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sbsi.fixed_g0_domain import FLOW_FEATURES, FLOW_TARGETS
from sbsi.flow_guard_response import (
    SHAPE_RESPONSE_COMPONENTS,
    GuardResponsePopulation,
    evaluate_guard_response,
    make_guard_band_cuts,
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


SEED = 821
EPOCHS = 40
ROWS_PER_EPOCH = 4_000_000
BATCH_SIZE = 8192
LEARNING_RATE = 3.0e-6
MIN_LEARNING_RATE = 1.0e-7
GRADIENT_CLIP = 5.0
VALIDATION_NLL_ROWS = 500_000
VALIDATION_NLL_SEED = 900000821

RADIUS_SOFTNESS = 0.005
MAGNITUDE_SOFTNESS = 0.01
RESPONSE_SCALE = 0.01
# The off-diagonal response is consistent with zero and about 0.4% of R11.
# A ten times larger scale divides its share of the averaged quadratic by a
# hundred, which removes the dilution without dropping the component.
OFF_DIAGONAL_SCALE_FACTOR = 10.0
# The band residuals start about ten times the cumulative ones, so under a
# uniform scale the averaged quadratic is a hundred times more sensitive to a
# band than to a cumulative cut.  Nine bands then outweigh sixteen cumulative
# cuts by roughly fifty to one, the cuts that pin the overall level of the
# response contribute almost nothing to the gradient, and that level is free to
# drift.  Dividing the cumulative scale by ten multiplies their share of the
# loss by a hundred, which restores a comparable footing.
CUMULATIVE_SCALE_FACTOR = 0.1

# Epochs whose weights are averaged into ``averaged.pt``.  The level wanders
# from epoch to epoch without a trend, so an average over the tail of the run
# estimates it far better than any single epoch, and unlike ``selected.pt`` it
# involves no choice made on the cases the evaluation then measures.
AVERAGE_FROM_FRACTION = 0.5

# Disjoint radius bands across the resolution cliff, under the deployment
# magnitude limit.  The lowest edge sits below the flux-radius selection so
# the band that the cut actually lands in is supervised from both sides.
BAND_RADIUS_EDGES = (2.2, 2.6, 2.8, 3.0, 3.2, 3.5, 4.0, 5.0, 7.0)
BAND_MAGNITUDE_MAX = 25.8

# One guard gradient per likelihood minibatch, sized like the paired-response
# stage that this composition is borrowed from.
GUARD_WEIGHT = 1.0
GUARD_PAIRS_PER_STEP = 1536
GUARD_TRAINING_DRAWS = 16
GUARD_TRAINING_CHUNK = 256
GUARD_TRAINING_SEED = 822_000_000_001
VALIDATION_GUARD_PAIRS = 400_000
VALIDATION_GUARD_DRAWS = 16
VALIDATION_GUARD_GROUPS = 4
VALIDATION_GUARD_BATCH = 2048
VALIDATION_GUARD_SEED = 823_000_000_001
VALIDATION_GUARD_SUBSET_SEED = 900000822


def build_guard_bank() -> tuple[tuple[dict, ...], int]:
    """Cumulative deployment guards plus disjoint radius bands.

    Returns the bank and the number of leading cumulative cuts, because the
    two kinds carry different residual scales and must be weighted apart.
    """

    cumulative = make_guard_cuts(RADIUS_THRESHOLDS, MAGNITUDE_THRESHOLDS)
    bands = make_guard_band_cuts(BAND_RADIUS_EDGES, magnitude_max=BAND_MAGNITUDE_MAX)
    names = [cut["name"] for cut in cumulative]
    combined = list(cumulative)
    combined.extend(cut for cut in bands if cut["name"] not in names)
    return tuple(combined), len(cumulative)


def build_response_scale(cuts, cumulative_count: int) -> np.ndarray:
    """Per-cut, per-component guard scale.

    Off-diagonal components are de-weighted, and the leading cumulative cuts
    are up-weighted so the overall level of the response is supervised rather
    than left to whatever the band terms happen to imply.
    """

    if not 0 <= cumulative_count <= len(cuts):
        raise ValueError("cumulative_count must index into the guard bank")
    scale = np.full(
        (len(cuts), len(SHAPE_RESPONSE_COMPONENTS)), RESPONSE_SCALE, dtype=np.float64
    )
    scale[:cumulative_count, :] *= CUMULATIVE_SCALE_FACTOR
    for index, component in enumerate(SHAPE_RESPONSE_COMPONENTS):
        if component in {"R12", "R21"}:
            scale[:, index] *= OFF_DIAGONAL_SCALE_FACTOR
    return scale


def combined_epoch(
    model, values, context, population, optimizer, *, seed, guard_weight
) -> dict:
    """One epoch of likelihood and guard gradients summed before clipping."""

    model.train()
    generator = torch.Generator(device=values.device).manual_seed(int(seed))
    order = torch.randperm(len(values), generator=generator, device=values.device)
    nll_total = 0.0
    guard_total = 0.0
    nll_norms = []
    combined_norms = []
    started = time.monotonic()
    steps = 0
    for step, rows in enumerate(order.split(BATCH_SIZE)):
        optimizer.zero_grad(set_to_none=True)
        nll = -model.log_prob(values[rows], context[rows]).mean()
        if not bool(torch.isfinite(nll)):
            raise ValueError("non-finite band-guard stage NLL")
        nll.backward()
        # Unclipped likelihood-only norm, recorded so the balance the clip
        # actually applies is visible rather than assumed.
        nll_norms.append(
            float(
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float("inf"), error_if_nonfinite=True
                )
            )
        )
        guard = sampled_guard_response_backward(
            model,
            population,
            GUARD_PAIRS_PER_STEP,
            GUARD_TRAINING_DRAWS,
            chunk_size=GUARD_TRAINING_CHUNK,
            seed=GUARD_TRAINING_SEED + seed * 1_000_003 + step * 100_003,
            weight=guard_weight,
        )
        combined_norms.append(
            float(
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), GRADIENT_CLIP, error_if_nonfinite=True
                )
            )
        )
        optimizer.step()
        nll_total += float(nll.detach()) * len(rows)
        guard_total += guard["loss"] * len(rows)
        steps += 1
    return {
        "nll": nll_total / len(values),
        "guard_loss": guard_total / len(values),
        "rows": int(len(values)),
        "optimizer_steps": steps,
        "guard_weight": float(guard_weight),
        "guard_pairs_per_step": GUARD_PAIRS_PER_STEP,
        "guard_draws": GUARD_TRAINING_DRAWS,
        "nll_gradient_norm_mean": float(np.mean(nll_norms)),
        "combined_gradient_norm_mean": float(np.mean(combined_norms)),
        "combined_gradient_norm_max": float(np.max(combined_norms)),
        "clipped_step_fraction": float(
            np.mean(np.asarray(combined_norms) > GRADIENT_CLIP)
        ),
        "seconds": time.monotonic() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain-root", type=Path, required=True)
    parser.add_argument("--initial-flow", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--rows-per-epoch", type=int, default=ROWS_PER_EPOCH)
    parser.add_argument("--guard-weight", type=float, default=GUARD_WEIGHT)
    args = parser.parse_args()

    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("band guard refinement must run under Slurm")
    if args.epochs < 1 or args.rows_per_epoch < BATCH_SIZE:
        raise ValueError("at least one epoch and one full batch required")
    if not math.isfinite(args.guard_weight) or args.guard_weight < 0.0:
        raise ValueError("finite nonnegative guard weight required")
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
        raise ValueError("band refinement requires the true-mag-limited full domain")
    if manifest["flow"]["features"] != list(FLOW_FEATURES) or manifest["flow"][
        "targets"
    ] != list(FLOW_TARGETS):
        raise ValueError("full-domain flow contract mismatch")
    if output_root.exists() and not args.resume:
        raise FileExistsError(f"preserving existing refinement root: {output_root}")
    output_root.mkdir(parents=True, exist_ok=args.resume)

    print("LOADING_BAND_GUARD_DATA", flush=True)
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
    train_context = torch.as_tensor(
        train_context_np, dtype=torch.float32, device=device
    )
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
    cuts, cumulative_count = build_guard_bank()
    scales = build_response_scale(cuts, cumulative_count)
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
            "off_diagonal_scale_factor": OFF_DIAGONAL_SCALE_FACTOR,
            "cumulative_scale_factor": CUMULATIVE_SCALE_FACTOR,
            "cumulative_cut_count": cumulative_count,
            "average_from_fraction": AVERAGE_FROM_FRACTION,
            "band_radius_edges": list(BAND_RADIUS_EDGES),
            "band_magnitude_max": BAND_MAGNITUDE_MAX,
            "components": list(SHAPE_RESPONSE_COMPONENTS),
            "train_statistics": json_guard_statistics(train_population),
            "validation_statistics": json_guard_statistics(validation_population),
        },
        "recipe": {
            "seed": SEED,
            "epochs": args.epochs,
            "rows_per_epoch": args.rows_per_epoch,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "minimum_learning_rate": MIN_LEARNING_RATE,
            "gradient_clip": GRADIENT_CLIP,
            "combined_gradient": True,
            "guard_weight": args.guard_weight,
            "guard_pairs_per_step": GUARD_PAIRS_PER_STEP,
            "guard_training_draws": GUARD_TRAINING_DRAWS,
            "validation_nll_rows": len(validation_nll_indices),
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
            raise RuntimeError("resume protocol differs from band refinement")
    else:
        base.write_json(protocol_path, protocol)
        np.save(output_root / "validation_nll_indices.npy", validation_nll_indices)
        np.save(
            output_root / "validation_guard_pair_indices.npy",
            validation_guard_indices,
        )
    print(
        f"BAND_GUARD_DATA_READY guards={len(cuts)} "
        f"train_nll={len(train_values):,} "
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
    # Averaging starts after a burn-in so the early epochs, which are still
    # moving the model rather than wandering around a level, stay out of it.
    average_from = max(1, math.ceil(args.epochs * AVERAGE_FROM_FRACTION))
    average_sum: dict[str, torch.Tensor] = {}
    average_count = 0
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
        average_sum = {k: v.clone() for k, v in state.get("average_sum", {}).items()}
        average_count = state.get("average_count", 0)
    if (output_root / "completed.json").exists():
        print("BAND_GUARD_ALREADY_COMPLETE", flush=True)
        return

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_seed = SEED + epoch * 1_000_003
        selected = np.random.default_rng(epoch_seed).choice(
            len(train_values),
            min(args.rows_per_epoch, len(train_values)),
            replace=False,
        )
        ids = torch.as_tensor(selected, dtype=torch.long, device=device)
        trained = combined_epoch(
            model,
            train_values[ids],
            train_context[ids],
            train_population,
            optimizer,
            seed=epoch_seed,
            guard_weight=args.guard_weight,
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
            raise ValueError("nonfinite band-guard validation objective")
        scheduler.step(metric)
        record = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": trained,
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
                    "training_phase": "full_domain_band_guard_refinement",
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
        if epoch >= average_from:
            for key, value in model.state_dict().items():
                if not value.is_floating_point():
                    # Integer buffers are counters or indices, not parameters;
                    # a mean of them is meaningless, so carry the latest.
                    average_sum[key] = value.detach().cpu().clone()
                    continue
                tensor = value.detach().to("cpu", torch.float64)
                if key in average_sum:
                    average_sum[key] += tensor
                else:
                    average_sum[key] = tensor
            average_count += 1
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "average_sum": average_sum,
                "average_count": average_count,
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
            f"BAND_GUARD epoch={epoch} nll={validation_nll['nll']:.8f} "
            f"guard={validation_guard['loss']:.8f} objective={metric:.8f} "
            f"best={best_metric:.8f}@{best_epoch} "
            f"nll_norm={trained['nll_gradient_norm_mean']:.3f} "
            f"combined_norm={trained['combined_gradient_norm_mean']:.3f} "
            f"lr={optimizer.param_groups[0]['lr']:.3g} "
            f"seconds={trained['seconds']:.1f}",
            flush=True,
        )
        if stale >= 15 and optimizer.param_groups[0]["lr"] <= MIN_LEARNING_RATE:
            break

    averaged_path = output_root / "averaged.pt"
    averaged_report = None
    if average_count:
        reference = model.state_dict()
        averaged_state = {}
        for key, value in average_sum.items():
            target_tensor = reference[key]
            if not target_tensor.is_floating_point():
                averaged_state[key] = value.to(target_tensor.dtype)
            else:
                averaged_state[key] = (value / average_count).to(target_tensor.dtype)
        model.load_state_dict(averaged_state, strict=True)
        model.to(device)
        averaged_nll = base.nll_pass(model, validation_values, validation_context)
        averaged_guard = evaluate_guard_response(
            model,
            validation_population,
            validation_guard_indices,
            draws=VALIDATION_GUARD_DRAWS,
            groups=VALIDATION_GUARD_GROUPS,
            batch_size=VALIDATION_GUARD_BATCH,
            seed=VALIDATION_GUARD_SEED,
        )
        base.save_checkpoint(
            averaged_path,
            model,
            condition,
            target,
            model_config,
            {
                "training_phase": "full_domain_band_guard_epoch_average",
                "averaged_epochs": average_count,
                "average_from_epoch": average_from,
                "stop_epoch": history[-1]["epoch"],
                "selection_free": True,
                "validation_nll": averaged_nll["nll"],
                "guard_response_loss": averaged_guard["loss"],
                "population": population,
                "protocol_sha256": base.sha256(protocol_path),
                "measurement_conditions_used": False,
            },
        )
        averaged_report = {
            "checkpoint": str(averaged_path),
            "sha256": base.sha256(averaged_path),
            "averaged_epochs": average_count,
            "average_from_epoch": average_from,
            "full_validation_nll": averaged_nll,
            "fixed_subset_guard_response": compact_guard_result(averaged_guard),
        }
        print(
            f"BAND_GUARD_AVERAGED epochs={average_count} from={average_from} "
            f"nll={averaged_nll['nll']:.8f} guard={averaged_guard['loss']:.8f}",
            flush=True,
        )

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
            "averaged": averaged_report,
            "best_epoch": best_epoch,
            "stop_epoch": history[-1]["epoch"],
            "best_selection_metric": best_metric,
            "full_validation_nll": final_nll,
            "fixed_subset_guard_response": compact_guard_result(final_guard),
            "protocol_sha256": base.sha256(protocol_path),
            "limitations": [
                "single refinement seed",
                "finite generated-draw integration",
                "validation cases participate in selecting selected.pt; "
                "averaged.pt is selection-free and is the reportable checkpoint",
                "guard weight fixed from one measured gradient-norm ratio",
                "not an end-to-end SBSI calibration acceptance result",
            ],
        },
    )
    print(
        f"BAND_GUARD_COMPLETE best_epoch={best_epoch} metric={best_metric:.8f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
