#!/usr/bin/env python3
"""Prepare and train two 216-cell fixed-g0 physical-response flow variants.

Both variants continue the selected epoch-140 NLL checkpoint on exactly the
fixed zero-shear measured cohort.  Cell membership is frozen on the g=0 leg:

``truth``
    true r magnitude, true circularized radius, cached baseline r_blend.

``measured``
    g=0 MAG_AUTO, g=0 FLUX_RADIUS, cached baseline r_blend.

The sheared leg is never recut and never supplies a grid coordinate.  Each
cell carries the joint four-component shape response matrix and two-component
physical-radius response.  Flux remains NLL-only.
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
import pyarrow as pa
import pyarrow.ipc as ipc
import torch

from sbsi.fixed_g0_domain import (
    FLOW_FEATURES,
    FLOW_TARGETS,
    matched_key_indices,
    packed_keys,
)
from sbsi.flow_grid_physical import (
    PhysicalGridPopulation,
    RESPONSE_COMPONENTS,
    assign_grid_cells,
    baseline_frame_gamma,
    evaluate_grid_response,
    fit_physical_grid_response,
    quantile_grid_edges,
    sampled_grid_response_backward,
)
from sbsi.flow_paired_shape import positive_shear_pair_mask
from sbsi.measurement_model import TargetStandardizer, build_flow
from sbsi.selection_model import TabularPreprocessor
from scripts import train_fixed_g0_flow as base


GRID_BINS = 6
GRID_CELLS = GRID_BINS**3
GRID_WEIGHT = 1.0
CELLS_PER_STEP = 12
PAIRS_PER_CELL = 128
MAXIMUM_PHASE_EPOCHS = base.RESPONSE_EPOCHS
EARLY_STOPPING_STALE = 20
PREPARATION_SUBSET_SEED = base.VALIDATION_RESPONSE_SUBSET_SEED

VARIANTS = {
    "truth": (
        "true_r_magnitude",
        "true_circularized_radius_arcsec",
        "cached_baseline_r_blend",
    ),
    "measured": (
        "g0_MAG_AUTO",
        "g0_FLUX_RADIUS_pixels",
        "cached_baseline_r_blend",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8_388_608), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.view(np.uint8))
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
        json.dumps(json_ready(value), indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    )
    os.replace(temporary, path)


def load_grid_split(domain_root: Path, cases) -> dict:
    """Load the existing domain and retain exact baseline identities."""

    contexts = []
    targets = []
    pairs = []
    deltas = []
    gammas = []
    pair_keys = []
    per_case = []
    offset = 0
    leg_rows = {"g0": 0, "g05": 0}
    for case in map(int, cases):
        zero = base.load_case(domain_root / "flow" / "g0" / f"case{case:03d}.npz")
        shear = base.load_case(
            domain_root / "flow" / "g05" / f"case{case:03d}.npz"
        )
        if not np.all(zero["case"] == case) or not np.all(shear["case"] == case):
            raise ValueError(f"prepared case label mismatch in case{case:03d}")
        left, right, counts = matched_key_indices(
            zero["case"],
            zero["input_index"],
            shear["case"],
            shear["input_index"],
        )
        pair_gamma = shear["gamma"][right]
        response_eligible = positive_shear_pair_mask(pair_gamma)
        excluded_zero_shear = int((~response_eligible).sum())
        left = left[response_eligible]
        right = right[response_eligible]
        pair_gamma = pair_gamma[response_eligible]
        if not len(left):
            raise RuntimeError(f"case{case:03d} has no eligible matched pair")
        n0 = len(zero["case"])
        ng = len(shear["case"])
        contexts.extend((zero["context"], shear["context"]))
        targets.extend((zero["target"], shear["target"]))
        pairs.append(
            np.column_stack((offset + left, offset + n0 + right)).astype(np.int64)
        )
        deltas.append(shear["target"][right, :3] - zero["target"][left, :3])
        gammas.append(pair_gamma)
        pair_keys.append(
            np.column_stack((zero["case"][left], zero["input_index"][left])).astype(
                np.int64
            )
        )
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
        "measured_delta": np.concatenate(deltas).astype(np.float64, copy=False),
        "gamma": np.concatenate(gammas).astype(np.float64, copy=False),
        "pair_keys": np.concatenate(pair_keys),
        "per_case_matching": per_case,
        "leg_rows": leg_rows,
    }


def load_baseline_lookup(path: Path) -> dict:
    """Load grid properties once and index them by exact packed identity."""

    names = (
        "case",
        "input_index",
        "r_input_p",
        "measured_mag_auto",
        "measured_flux_radius",
        "r_blend",
    )
    parts = {name: [] for name in names}
    with ipc.open_file(path) as reader:
        missing = sorted(set(names) - set(reader.schema.names))
        if missing:
            raise KeyError(f"baseline catalogue lacks grid columns: {missing}")
        for batch_index in range(reader.num_record_batches):
            table = pa.Table.from_batches([reader.get_batch(batch_index)]).select(names)
            for name in names:
                parts[name].append(
                    table[name].combine_chunks().to_numpy(zero_copy_only=False)
                )
            if (batch_index + 1) % 100 == 0:
                print(
                    f"BASELINE_LOOKUP_PROGRESS {batch_index + 1}/{reader.num_record_batches}",
                    flush=True,
                )
    case = np.concatenate(parts["case"])
    input_index = np.concatenate(parts["input_index"])
    keys = packed_keys(case, input_index)
    order = np.argsort(keys, kind="stable")
    keys = keys[order]
    if len(keys) > 1 and np.any(keys[1:] == keys[:-1]):
        raise ValueError("baseline grid catalogue contains duplicate identities")
    values = np.column_stack(
        [
            np.concatenate(parts[name]).astype(np.float64, copy=False)[order]
            for name in names[2:]
        ]
    )
    return {"keys": keys, "values": values, "columns": names[2:]}


def matched_baseline_values(lookup: dict, pair_keys) -> np.ndarray:
    packed = packed_keys(pair_keys[:, 0], pair_keys[:, 1])
    locations = np.searchsorted(lookup["keys"], packed)
    matched = locations < len(lookup["keys"])
    matched[matched] &= lookup["keys"][locations[matched]] == packed[matched]
    if not matched.all():
        raise RuntimeError(
            f"baseline lookup is missing {int((~matched).sum()):,} fixed-domain pairs"
        )
    return lookup["values"][locations]


def grid_features(data: dict, baseline_values, variant: str) -> np.ndarray:
    """Build fixed baseline-leg grid coordinates and verify their identity."""

    if variant not in VARIANTS:
        raise ValueError(f"unknown grid variant {variant!r}")
    values = np.asarray(baseline_values, dtype=np.float64)
    if values.shape != (len(data["pairs"]), 4) or not np.isfinite(values).all():
        raise ValueError("finite aligned baseline grid values required")
    baseline_context = data["context_raw"][data["pairs"][:, 0]]
    baseline_target = data["target_raw"][data["pairs"][:, 0]]
    truth_mag, measured_mag, measured_radius, r_blend = values.T
    np.testing.assert_allclose(
        baseline_context[:, FLOW_FEATURES.index("r_input_p")],
        truth_mag,
        rtol=0.0,
        atol=2.0e-6,
    )
    recovered_mag = 30.0 - 2.5 * np.log10(baseline_target[:, 3])
    np.testing.assert_allclose(recovered_mag, measured_mag, rtol=0.0, atol=2.0e-12)
    np.testing.assert_allclose(
        baseline_target[:, 2], measured_radius, rtol=0.0, atol=0.0
    )
    if variant == "truth":
        features = np.column_stack(
            (
                truth_mag,
                baseline_context[
                    :, FLOW_FEATURES.index("circularized_Re_input_p")
                ],
                r_blend,
            )
        )
    else:
        features = np.column_stack((measured_mag, measured_radius, r_blend))
    if not np.isfinite(features).all():
        raise ValueError("nonfinite fixed baseline grid coordinate")
    return features


def scalar_shear(data: dict) -> np.ndarray:
    baseline_shape = data["context_raw"][data["pairs"][:, 0], :2]
    return baseline_frame_gamma(data["gamma"], baseline_shape)


def stratified_subset_indices(cell_ids, total, seed):
    """Allocate a deterministic near-equal validation budget across cells."""

    cells = np.asarray(cell_ids)
    counts = np.bincount(cells, minlength=GRID_CELLS)
    if total > counts.sum() or np.any(counts == 0):
        raise ValueError("validation subset exceeds a completely populated grid")
    allocation = np.minimum(counts, total // GRID_CELLS)
    while int(allocation.sum()) < total:
        eligible = np.flatnonzero(allocation < counts)
        needed = min(len(eligible), total - int(allocation.sum()))
        allocation[eligible[:needed]] += 1
    rng = np.random.default_rng(seed)
    return np.sort(
        np.concatenate(
            [
                rng.choice(np.flatnonzero(cells == cell), int(count), replace=False)
                for cell, count in enumerate(allocation)
            ]
        )
    )


def statistics_report(statistics: dict) -> dict:
    covariance = statistics["covariance"]
    standard_error = np.sqrt(np.diagonal(covariance, axis1=1, axis2=2))
    eigenvalues = statistics["correlation_eigenvalues"]
    return {
        "min_cell_pairs": int(statistics["counts"].min()),
        "max_cell_pairs": int(statistics["counts"].max()),
        "min_cell_case_coverage": int(statistics["case_coverage"].min()),
        "correlation_min_eigenvalue": float(eigenvalues.min()),
        "correlation_max_condition": float(
            np.max(eigenvalues[:, -1] / eigenvalues[:, 0])
        ),
        "component_se_quantiles": np.quantile(
            standard_error, (0.0, 0.5, 0.95, 1.0), axis=0
        ),
    }


def validate_domain(manifest: dict) -> None:
    population = manifest["population"]
    if (
        manifest.get("format_version") != 2
        or population["anchor_leg"] != "g=0"
        or population["mag_auto_max"] != 25.8
        or population["flux_radius_min_pixels"] != 3.0
        or not population["strict_radius_inequality"]
        or population["truth_analysis_cut"] is not None
        or population["sheared_leg_recut"]
    ):
        raise ValueError("grid training requires the corrected fixed-g0 v2 domain")


def prepare(args: argparse.Namespace) -> None:
    if not os.environ.get("SLURM_JOB_ID") and not args.allow_login_smoke:
        raise RuntimeError("full grid preparation must run under Slurm")
    domain_root = args.domain_root.resolve()
    output_root = args.output_root.resolve()
    parent_path = args.parent.resolve()
    optimizer_path = args.parent_optimizer.resolve()
    if output_root.exists():
        raise FileExistsError(f"preserving existing grid output: {output_root}")
    output_root.mkdir(parents=True)
    started = time.monotonic()

    domain_manifest_path = domain_root / "manifest.json"
    domain_manifest = json.loads(domain_manifest_path.read_text())
    validate_domain(domain_manifest)
    parent = torch.load(parent_path, map_location="cpu", weights_only=False)
    if parent["metadata"].get("epoch") != base.NLL_EPOCHS:
        raise ValueError("grid continuation requires the selected epoch-140 NLL parent")
    if parent["metadata"].get("population") != domain_manifest["population"]:
        raise ValueError("parent checkpoint population differs from the domain")
    parent_optimizer = torch.load(
        optimizer_path, map_location="cpu", weights_only=False
    )
    if parent_optimizer.get("epoch") != base.NLL_EPOCHS:
        raise ValueError("parent optimizer does not belong to epoch 140")
    for name, value in parent["state_dict"].items():
        if not torch.equal(value, parent_optimizer["model"][name]):
            raise ValueError("parent checkpoint and optimizer snapshot differ")

    print("LOADING_FIXED_G0_GRID_DOMAIN", flush=True)
    data = {
        "train": load_grid_split(domain_root, domain_manifest["split"]["train_cases"]),
        "validation": load_grid_split(
            domain_root, domain_manifest["split"]["validation_cases"]
        ),
    }
    baseline_path = Path(domain_manifest["paths"]["g0_catalogue"])
    lookup = load_baseline_lookup(baseline_path)
    baseline = {
        name: matched_baseline_values(lookup, subset["pair_keys"])
        for name, subset in data.items()
    }
    del lookup

    protocol = {
        "format_version": 1,
        "experiment": "fixed_g0_grid_shape_radius_6x6x6",
        "domain_manifest": str(domain_manifest_path),
        "domain_manifest_sha256": sha256(domain_manifest_path),
        "population": domain_manifest["population"],
        "split": domain_manifest["split"],
        "baseline_catalogue": str(baseline_path),
        "baseline_catalogue_sha256": sha256(baseline_path),
        "parent_checkpoint": str(parent_path),
        "parent_checkpoint_sha256": sha256(parent_path),
        "parent_optimizer": str(optimizer_path),
        "parent_optimizer_sha256": sha256(optimizer_path),
        "features": list(FLOW_FEATURES),
        "targets": list(FLOW_TARGETS),
        "response_components": list(RESPONSE_COMPONENTS),
        "grid_shape": [GRID_BINS, GRID_BINS, GRID_BINS],
        "grid_cell_weighting": "equal cell; no occupancy weighting",
        "grid_edge_rule": (
            "marginal train-pair quantiles; internal boundaries enter the right "
            "bin; validation tails remain in outer bins"
        ),
        "grid_leg": "g=0 anchor leg only; sheared coordinates are never used",
        "response": (
            "matched forward 0-to-0.05 WLS; four sky-frame shape and two "
            "baseline-frame physical-radius components"
        ),
        "precision": (
            "inverse joint six-component delete-one-case covariance after "
            "algebraically exact correlation equilibration; no floor"
        ),
        "recipe": {
            "seed": base.SEED,
            "maximum_phase_epochs": MAXIMUM_PHASE_EPOCHS,
            "early_stopping_stale": EARLY_STOPPING_STALE,
            "rows_per_epoch": base.ROWS_PER_EPOCH,
            "batch_size": base.BATCH_SIZE,
            "starting_learning_rate": float(parent_optimizer["learning_rate"]),
            "minimum_learning_rate": base.MIN_LEARNING_RATE,
            "gradient_clip": base.GRADIENT_CLIP,
            "grid_response_weight": GRID_WEIGHT,
            "cells_per_step": CELLS_PER_STEP,
            "pairs_per_cell": PAIRS_PER_CELL,
            "pairs_per_group_per_step": CELLS_PER_STEP * PAIRS_PER_CELL,
            "response_training_draws": base.RESPONSE_TRAINING_DRAWS,
            "response_training_chunk": base.RESPONSE_TRAINING_CHUNK,
            "validation_nll_rows": base.VALIDATION_NLL_ROWS,
            "validation_response_pairs": base.VALIDATION_RESPONSE_PAIRS,
            "validation_response_draws": base.VALIDATION_RESPONSE_DRAWS,
            "validation_response_groups": base.VALIDATION_RESPONSE_GROUPS,
            "validation_response_batch": base.VALIDATION_RESPONSE_BATCH,
            "validation_response_seed": base.VALIDATION_RESPONSE_SEED,
            "validation_subset_seed": PREPARATION_SUBSET_SEED,
            "flux_response_supervision": False,
        },
        "counts": {
            name: {
                "nll_rows": int(len(subset["target_raw"])),
                "response_pairs": int(len(subset["pairs"])),
                "pair_keys_sha256": array_sha256(subset["pair_keys"]),
                "leg_rows": subset["leg_rows"],
                "per_case_matching": subset["per_case_matching"],
            }
            for name, subset in data.items()
        },
        "variants": {},
        "implementation_sha256": sha256(Path(__file__).resolve()),
        "grid_module_sha256": sha256(
            Path(__file__).resolve().parents[1] / "sbsi" / "flow_grid_physical.py"
        ),
    }

    scalar = {name: scalar_shear(subset) for name, subset in data.items()}
    for variant, axes in VARIANTS.items():
        variant_root = output_root / variant
        variant_root.mkdir()
        features = {
            name: grid_features(subset, baseline[name], variant)
            for name, subset in data.items()
        }
        edges = quantile_grid_edges(features["train"], GRID_BINS)
        np.save(variant_root / "edges.npy", edges)
        variant_report = {"axes": list(axes), "edges": edges, "splits": {}}
        for split, subset in data.items():
            cell_ids = assign_grid_cells(features[split], edges)
            statistics = fit_physical_grid_response(
                subset["measured_delta"],
                subset["gamma"],
                scalar[split],
                cell_ids,
                subset["pair_keys"][:, 0],
                GRID_CELLS,
            )
            np.save(variant_root / f"{split}_cell_ids.npy", cell_ids.astype(np.uint8))
            np.savez(variant_root / f"{split}_statistics.npz", **statistics)
            split_report = {
                "pairs": int(len(cell_ids)),
                "cases": int(len(np.unique(subset["pair_keys"][:, 0]))),
                **statistics_report(statistics),
                "below_training_minimum": (
                    (features[split] < edges[:, 0]).sum(axis=0)
                ),
                "above_training_maximum": (
                    (features[split] > edges[:, -1]).sum(axis=0)
                ),
            }
            if split == "validation":
                indices = stratified_subset_indices(
                    cell_ids,
                    min(base.VALIDATION_RESPONSE_PAIRS, len(cell_ids)),
                    PREPARATION_SUBSET_SEED,
                )
                subset_statistics = fit_physical_grid_response(
                    subset["measured_delta"][indices],
                    subset["gamma"][indices],
                    scalar[split][indices],
                    cell_ids[indices],
                    subset["pair_keys"][indices, 0],
                    GRID_CELLS,
                )
                np.save(variant_root / "validation_subset_indices.npy", indices)
                np.savez(
                    variant_root / "validation_subset_statistics.npz",
                    **subset_statistics,
                )
                split_report["validation_subset_pairs"] = int(len(indices))
                split_report["validation_subset"] = statistics_report(
                    subset_statistics
                )
            variant_report["splits"][split] = split_report
            print(
                f"GRID_PREPARED variant={variant} split={split} "
                f"pairs={len(cell_ids):,} min_cell={statistics['counts'].min():,} "
                f"min_cases={statistics['case_coverage'].min()}",
                flush=True,
            )
        variant_report["files"] = {
            path.name: sha256(path)
            for path in sorted(variant_root.iterdir())
            if path.is_file()
        }
        protocol["variants"][variant] = variant_report

    validation_ids = np.sort(
        np.random.default_rng(base.VALIDATION_NLL_SEED).choice(
            len(data["validation"]["target_raw"]),
            min(base.VALIDATION_NLL_ROWS, len(data["validation"]["target_raw"])),
            replace=False,
        )
    )
    np.save(output_root / "validation_nll_indices.npy", validation_ids)
    protocol["validation_nll_indices_sha256"] = sha256(
        output_root / "validation_nll_indices.npy"
    )
    protocol["elapsed_seconds"] = time.monotonic() - started
    write_json(output_root / "protocol.json", protocol)
    print(
        f"FIXED_G0_GRID_PREPARATION_COMPLETE root={output_root} "
        f"seconds={protocol['elapsed_seconds']:.1f}",
        flush=True,
    )


def load_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as stored:
        return {name: stored[name] for name in stored.files}


def transform_context(preprocessor, values):
    raw = np.asarray(values, dtype=np.float32)
    raw = preprocessor._safe_log_np(raw, preprocessor._log_cols())
    finite = np.isfinite(raw)
    filled = np.where(finite, raw, preprocessor.fill_values)
    scaled = (filled - preprocessor.means) / preprocessor.scales
    if preprocessor.add_missing_indicators:
        scaled = np.concatenate((scaled, (~finite).astype(np.float32)), axis=1)
    return scaled.astype(np.float32, copy=False)


def cpu_optimizer_state(optimizer):
    state = copy.deepcopy(optimizer.state_dict())
    for values in state["state"].values():
        for key, value in values.items():
            if torch.is_tensor(value):
                values[key] = value.detach().cpu()
    return state


def grid_epoch(
    model,
    values,
    context,
    population,
    optimizer,
    *,
    seed,
):
    model.train()
    generator = torch.Generator(device=values.device).manual_seed(int(seed))
    order = torch.randperm(len(values), generator=generator, device=values.device)
    nll_total = 0.0
    response_total = 0.0
    norms = []
    started = time.monotonic()
    for step, rows in enumerate(order.split(base.BATCH_SIZE)):
        optimizer.zero_grad(set_to_none=True)
        nll = -model.log_prob(values[rows], context[rows]).mean()
        if not bool(torch.isfinite(nll)):
            raise ValueError("nonfinite grid-stage NLL")
        nll.backward()
        response = sampled_grid_response_backward(
            model,
            population,
            CELLS_PER_STEP,
            PAIRS_PER_CELL,
            base.RESPONSE_TRAINING_DRAWS,
            chunk_size=base.RESPONSE_TRAINING_CHUNK,
            seed=base.RESPONSE_TRAINING_SEED + seed * 1_000_003 + step * 100_003,
            weight=GRID_WEIGHT,
        )
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), base.GRADIENT_CLIP, error_if_nonfinite=True
        )
        optimizer.step()
        nll_total += float(nll.detach()) * len(rows)
        response_total += response["loss"] * len(rows)
        norms.append(float(norm))
    return {
        "nll": nll_total / len(values),
        "grid_response_loss": response_total / len(values),
        "rows": int(len(values)),
        "optimizer_steps": len(norms),
        "gradient_norm_mean": float(np.mean(norms)),
        "gradient_norm_max": float(np.max(norms)),
        "clipped_step_fraction": float(
            np.mean(np.asarray(norms) > base.GRADIENT_CLIP)
        ),
        "seconds": time.monotonic() - started,
    }


def train(args: argparse.Namespace) -> None:
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("grid flow training must run under Slurm")
    if args.variant not in VARIANTS:
        raise ValueError(f"unknown grid variant {args.variant!r}")
    domain_root = args.domain_root.resolve()
    prepared_root = args.prepared_root.resolve()
    variant_root = prepared_root / args.variant
    training_root = variant_root / "training"
    protocol_path = prepared_root / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    domain_manifest = json.loads((domain_root / "manifest.json").read_text())
    validate_domain(domain_manifest)
    if sha256(domain_root / "manifest.json") != protocol["domain_manifest_sha256"]:
        raise RuntimeError("fixed-g0 domain manifest changed after grid preparation")
    if protocol["population"] != domain_manifest["population"]:
        raise RuntimeError("prepared grid population differs from the fixed-g0 domain")
    if protocol["variants"][args.variant]["axes"] != list(VARIANTS[args.variant]):
        raise RuntimeError("prepared grid axes differ from the requested definition")
    if sha256(Path(__file__).resolve()) != protocol["implementation_sha256"]:
        raise RuntimeError("grid training script changed after preparation")
    grid_module = Path(__file__).resolve().parents[1] / "sbsi" / "flow_grid_physical.py"
    if sha256(grid_module) != protocol["grid_module_sha256"]:
        raise RuntimeError("physical grid implementation changed after preparation")
    if sha256(
        prepared_root / "validation_nll_indices.npy"
    ) != protocol["validation_nll_indices_sha256"]:
        raise RuntimeError("fixed validation NLL subset changed after preparation")
    for name, expected in protocol["variants"][args.variant]["files"].items():
        if sha256(variant_root / name) != expected:
            raise RuntimeError(f"prepared grid artifact changed: {name}")
    parent_path = Path(protocol["parent_checkpoint"])
    optimizer_path = Path(protocol["parent_optimizer"])
    if sha256(parent_path) != protocol["parent_checkpoint_sha256"] or sha256(
        optimizer_path
    ) != protocol["parent_optimizer_sha256"]:
        raise RuntimeError("NLL parent or optimizer changed after preparation")
    if training_root.exists() and not args.resume:
        raise FileExistsError(f"preserving existing training root: {training_root}")
    training_root.mkdir(exist_ok=args.resume)
    diagnostics_root = training_root / "diagnostics"
    diagnostics_root.mkdir(exist_ok=args.resume)
    if (training_root / "completed.json").exists():
        print(f"FIXED_G0_GRID_ALREADY_COMPLETE variant={args.variant}", flush=True)
        return

    print(f"LOADING_FIXED_G0_GRID_TRAINING variant={args.variant}", flush=True)
    data = {
        "train": load_grid_split(domain_root, domain_manifest["split"]["train_cases"]),
        "validation": load_grid_split(
            domain_root, domain_manifest["split"]["validation_cases"]
        ),
    }
    for split, subset in data.items():
        expected = protocol["counts"][split]
        if len(subset["pairs"]) != expected["response_pairs"] or array_sha256(
            subset["pair_keys"]
        ) != expected["pair_keys_sha256"]:
            raise RuntimeError(f"{split} pair population changed after preparation")

    parent = torch.load(parent_path, map_location="cpu", weights_only=False)
    saved_optimizer = torch.load(
        optimizer_path, map_location="cpu", weights_only=False
    )
    condition = TabularPreprocessor.from_state(parent["condition_preprocessor"])
    target = TargetStandardizer.from_state(parent["target_transform"])
    if condition.feature_names != list(FLOW_FEATURES) or target.target_names != list(
        FLOW_TARGETS
    ):
        raise ValueError("NLL parent feature/target contract differs from fixed-g0")

    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("scheduled grid training requires CUDA")
    tensors = {}
    populations = {}
    for split, subset in data.items():
        context_array = transform_context(condition, subset["context_raw"])
        value_array = target.transform_array(subset["target_raw"])
        context = torch.as_tensor(context_array, dtype=torch.float32, device=device)
        values = torch.as_tensor(value_array, dtype=torch.float64, device=device)
        cell_ids = np.load(variant_root / f"{split}_cell_ids.npy").astype(
            np.int64
        )
        statistics = load_npz(variant_root / f"{split}_statistics.npz")
        scalar = scalar_shear(subset)
        if split == "train":
            populations[split] = PhysicalGridPopulation(
                context,
                subset["pairs"],
                cell_ids,
                subset["gamma"],
                scalar,
                statistics,
            )
        tensors[split] = (values, context)
        del context_array, value_array, cell_ids, statistics, scalar

    validation_subset = np.load(variant_root / "validation_subset_indices.npy")
    validation_cell_ids = np.load(
        variant_root / "validation_cell_ids.npy"
    ).astype(np.int64)
    validation_full_statistics = load_npz(
        variant_root / "validation_statistics.npz"
    )
    validation_subset_statistics = load_npz(
        variant_root / "validation_subset_statistics.npz"
    )
    validation_data = data["validation"]
    validation_scalar = scalar_shear(validation_data)
    validation_population = PhysicalGridPopulation(
        tensors["validation"][1],
        validation_data["pairs"][validation_subset],
        validation_cell_ids[validation_subset],
        validation_data["gamma"][validation_subset],
        validation_scalar[validation_subset],
        validation_subset_statistics,
        precision=validation_full_statistics["precision"],
    )
    del validation_scalar, validation_full_statistics, validation_subset_statistics
    del validation_data, data

    validation_ids_np = np.load(prepared_root / "validation_nll_indices.npy")
    validation_ids = torch.as_tensor(
        validation_ids_np, dtype=torch.long, device=device
    )
    torch.manual_seed(base.SEED)
    torch.cuda.manual_seed_all(base.SEED)
    model = build_flow(parent["model_config"]).to(device)
    model.load_state_dict(parent["state_dict"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(saved_optimizer["learning_rate"]), weight_decay=0.0
    )
    optimizer.load_state_dict(saved_optimizer["optimizer"])
    for group in optimizer.param_groups:
        group["lr"] = float(saved_optimizer["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=4,
        threshold=1.0e-4,
        threshold_mode="abs",
        min_lr=base.MIN_LEARNING_RATE,
    )

    state_path = training_root / "latest_optimizer.pt"
    history = []
    best_metric = float("inf")
    best_phase_epoch = -1
    stale = 0
    response_start = 0
    if args.resume and state_path.exists():
        state = torch.load(state_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        history = state["history"]
        best_metric = state["best_metric"]
        best_phase_epoch = state["best_epoch"]
        stale = state["stale"]
        response_start = state["phase_epoch"] + 1

    def validate(phase_epoch):
        absolute_epoch = saved_optimizer["epoch"] + phase_epoch
        validation = base.nll_pass(
            model,
            tensors["validation"][0][validation_ids],
            tensors["validation"][1][validation_ids],
        )
        response = evaluate_grid_response(
            model,
            validation_population,
            draws=base.VALIDATION_RESPONSE_DRAWS,
            groups=base.VALIDATION_RESPONSE_GROUPS,
            batch_size=base.VALIDATION_RESPONSE_BATCH,
            seed=base.VALIDATION_RESPONSE_SEED,
        )
        metric = validation["nll"] + GRID_WEIGHT * response["loss"]
        if not math.isfinite(metric):
            raise ValueError("nonfinite grid validation objective")
        np.savez(
            diagnostics_root / f"epoch{absolute_epoch:03d}.npz",
            **response,
        )
        return absolute_epoch, validation, response, metric

    if response_start == 0:
        epoch, validation, response, metric = validate(0)
        history.append(
            {
                "phase_epoch": 0,
                "epoch": epoch,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "train": None,
                "validation": validation,
                "response": {
                    key: value
                    for key, value in response.items()
                    if np.asarray(value).ndim == 0
                },
                "selection_metric": metric,
            }
        )
        best_metric = metric
        best_phase_epoch = 0
        base.save_checkpoint(
            training_root / "selected.pt",
            model,
            condition,
            target,
            parent["model_config"],
            {
                "training_phase": "fixed_g0_grid_shape_radius",
                "grid_variant": args.variant,
                "grid_axes": list(VARIANTS[args.variant]),
                "grid_shape": [GRID_BINS] * 3,
                "epoch": epoch,
                "phase_epoch": 0,
                "validation_nll": validation["nll"],
                "grid_response_loss": response["loss"],
                "selection_metric": metric,
                "population": domain_manifest["population"],
                "truth_analysis_cut": None,
                "sheared_leg_recut": False,
                "grid_leg": "g=0",
                "protocol_sha256": sha256(protocol_path),
                "grid_response_weight": GRID_WEIGHT,
                "response_components": list(RESPONSE_COMPONENTS),
                "flux_response_supervision": False,
                "experimental": True,
            },
        )
        response_start = 1
        print(
            f"FIXED_G0_GRID_PARENT variant={args.variant} epoch={epoch} "
            f"nll={validation['nll']:.8f} response={response['loss']:.8f} "
            f"objective={metric:.8f}",
            flush=True,
        )

    for phase_epoch in range(response_start, MAXIMUM_PHASE_EPOCHS + 1):
        absolute_epoch = saved_optimizer["epoch"] + phase_epoch
        epoch_seed = base.SEED + absolute_epoch * 1_000_003
        selected = np.random.default_rng(epoch_seed).choice(
            len(tensors["train"][0]),
            min(base.ROWS_PER_EPOCH, len(tensors["train"][0])),
            replace=False,
        )
        ids = torch.as_tensor(selected, dtype=torch.long, device=device)
        trained = grid_epoch(
            model,
            tensors["train"][0][ids],
            tensors["train"][1][ids],
            populations["train"],
            optimizer,
            seed=epoch_seed,
        )
        epoch, validation, response, metric = validate(phase_epoch)
        scheduler.step(metric)
        record = {
            "phase_epoch": phase_epoch,
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": trained,
            "validation": validation,
            "response": {
                key: value
                for key, value in response.items()
                if np.asarray(value).ndim == 0
            },
            "selection_metric": metric,
        }
        history.append(record)
        if metric < best_metric:
            best_metric = metric
            best_phase_epoch = phase_epoch
            stale = 0
            base.save_checkpoint(
                training_root / "selected.pt",
                model,
                condition,
                target,
                parent["model_config"],
                {
                    "training_phase": "fixed_g0_grid_shape_radius",
                    "grid_variant": args.variant,
                    "grid_axes": list(VARIANTS[args.variant]),
                    "grid_shape": [GRID_BINS] * 3,
                    "epoch": epoch,
                    "phase_epoch": phase_epoch,
                    "validation_nll": validation["nll"],
                    "grid_response_loss": response["loss"],
                    "selection_metric": metric,
                    "population": domain_manifest["population"],
                    "truth_analysis_cut": None,
                    "sheared_leg_recut": False,
                    "grid_leg": "g=0",
                    "protocol_sha256": sha256(protocol_path),
                    "grid_response_weight": GRID_WEIGHT,
                    "response_components": list(RESPONSE_COMPONENTS),
                    "flux_response_supervision": False,
                    "experimental": True,
                },
            )
        else:
            stale += 1
        torch.save(
            {
                "phase_epoch": phase_epoch,
                "model": {
                    key: value.detach().cpu()
                    for key, value in model.state_dict().items()
                },
                "optimizer": cpu_optimizer_state(optimizer),
                "scheduler": scheduler.state_dict(),
                "history": history,
                "best_metric": best_metric,
                "best_epoch": best_phase_epoch,
                "stale": stale,
            },
            training_root / "latest_optimizer.tmp",
        )
        os.replace(training_root / "latest_optimizer.tmp", state_path)
        write_json(
            training_root / "history.json",
            {
                "history": history,
                "best_phase_epoch": best_phase_epoch,
                "best_metric": best_metric,
            },
        )
        print(
            f"FIXED_G0_GRID variant={args.variant} epoch={epoch} "
            f"phase={phase_epoch} nll={validation['nll']:.8f} "
            f"response={response['loss']:.8f} objective={metric:.8f} "
            f"best={best_metric:.8f}@{best_phase_epoch} "
            f"lr={optimizer.param_groups[0]['lr']:.3g}",
            flush=True,
        )
        if stale >= EARLY_STOPPING_STALE and optimizer.param_groups[0][
            "lr"
        ] <= base.MIN_LEARNING_RATE:
            break

    selected_path = training_root / "selected.pt"
    selected = torch.load(selected_path, map_location=device, weights_only=False)
    model.load_state_dict(selected["state_dict"])
    final_nll = base.nll_pass(
        model, tensors["validation"][0], tensors["validation"][1]
    )
    final_response = evaluate_grid_response(
        model,
        validation_population,
        draws=base.VALIDATION_RESPONSE_DRAWS,
        groups=base.VALIDATION_RESPONSE_GROUPS,
        batch_size=base.VALIDATION_RESPONSE_BATCH,
        seed=base.VALIDATION_RESPONSE_SEED,
    )
    write_json(
        training_root / "completed.json",
        {
            "checks_passed": True,
            "grid_variant": args.variant,
            "grid_axes": list(VARIANTS[args.variant]),
            "selected_checkpoint": str(selected_path),
            "selected_sha256": sha256(selected_path),
            "best_phase_epoch": best_phase_epoch,
            "stop_phase_epoch": history[-1]["phase_epoch"],
            "best_selection_metric": best_metric,
            "full_validation_nll": final_nll,
            "fixed_stratified_subset_response": {
                key: value
                for key, value in final_response.items()
                if np.asarray(value).ndim == 0
            },
            "domain_manifest_sha256": protocol["domain_manifest_sha256"],
            "protocol_sha256": sha256(protocol_path),
            "limitations": [
                "experimental single-seed grid continuation",
                "finite forward response at shear amplitude 0.05",
                "grid axes define response cells but are not added as flow conditions",
                "cached baseline r_blend is used only for cell membership",
                "fixed g=0 cohort changes likelihood conditioning semantics",
                "not an end-to-end shear-calibration acceptance result",
            ],
        },
    )
    print(
        f"FIXED_G0_GRID_COMPLETE variant={args.variant} "
        f"selected={selected_path} nll={final_nll['nll']:.8f} "
        f"response={final_response['loss']:.8f}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--domain-root", type=Path, required=True)
    prepare_parser.add_argument("--output-root", type=Path, required=True)
    prepare_parser.add_argument("--parent", type=Path, required=True)
    prepare_parser.add_argument("--parent-optimizer", type=Path, required=True)
    prepare_parser.add_argument("--allow-login-smoke", action="store_true")
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--domain-root", type=Path, required=True)
    train_parser.add_argument("--prepared-root", type=Path, required=True)
    train_parser.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    train_parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    prepare(args) if args.command == "prepare" else train(args)


if __name__ == "__main__":
    main()
