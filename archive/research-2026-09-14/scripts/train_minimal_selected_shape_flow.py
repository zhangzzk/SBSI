#!/usr/bin/env python3
"""Train the minimal two-output flow for the exact selected event.

The frozen selector supplies p(S|x), where S is usable shape, MAG_AUTO<25.8,
and convolved SExtractor FLUX_RADIUS>=0.75 arcsec with no |e| cut.  This script
therefore fits only p(e1,e2|S,x).  It reuses the existing prepared catalogue
arrays and the same nine conditions; it never reads a response statistic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import torch

from sbsi.flow_disk import radial_atanh
from sbsi.measurement_model import TargetStandardizer, build_flow, save_measurement_model
from sbsi.selection_model import TabularPreprocessor


SHAPE_NAMES = ("measured_ngmix_g1", "measured_ngmix_g2")
PHYSICAL_NAMES = (
    "measured_ngmix_g1",
    "measured_ngmix_g2",
    "measured_flux_radius",
    "measured_flux_from_mag_auto",
)
SOURCE_FILES = (
    "train_values.npy",
    "train_context8.npy",
    "train_rblend.npy",
    "train_cases.npy",
    "validation_values.npy",
    "validation_context8.npy",
    "validation_rblend.npy",
    "validation_cases.npy",
    "rblend_normalization.npy",
    "receipt.json",
)
PREPARED_FILES = (
    "train_values.npy",
    "train_context.npy",
    "train_cases.npy",
    "validation_values.npy",
    "validation_context.npy",
    "validation_cases.npy",
    "validation_indices.npy",
)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8_388_608), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: str | Path, value: dict) -> None:
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def load_torch(path: str | Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_protocol(
    root: str | Path, *, verify_source: bool = False, verify_prepared: bool = False
) -> dict:
    root = Path(root).resolve()
    protocol = json.loads((root / "protocol.json").read_text())
    if verify_source:
        for name, value in protocol["source_inputs"].items():
            path = Path(value)
            if sha256(path) != protocol["source_hashes"][name]:
                raise RuntimeError(f"source input changed: {path}")
    if verify_prepared:
        receipt = json.loads((root / "prepared/receipt.json").read_text())
        for name in PREPARED_FILES:
            if sha256(root / "prepared" / name) != receipt["hashes"][name]:
                raise RuntimeError(f"prepared input changed: {name}")
    return protocol


def init_protocol(args: argparse.Namespace) -> None:
    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(exist_ok=True)
    path = root / "protocol.json"
    if path.exists():
        raise FileExistsError(f"preserving existing protocol: {path}")
    source_prepared = Path(args.source_prepared).resolve()
    source_checkpoint = Path(args.source_checkpoint).resolve()
    training_script = Path(__file__).resolve()
    job_script = Path(args.job_script).resolve()
    repository = training_script.parents[1]
    source_paths = {name: source_prepared / name for name in SOURCE_FILES}
    source_paths.update(
        source_checkpoint=source_checkpoint,
        training_script=training_script,
        job_script=job_script,
    )
    source_paths.update(
        {
            f"sbsi/{path.name}": path
            for path in sorted((repository / "sbsi").glob("*.py"))
        }
    )
    missing = [str(value) for value in source_paths.values() if not value.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    parent = load_torch(source_checkpoint)
    target = TargetStandardizer.from_state(parent["target_transform"])
    condition = TabularPreprocessor.from_state(parent["condition_preprocessor"])
    if tuple(target.target_names) != PHYSICAL_NAMES or target.dtype != "float64":
        raise RuntimeError("source checkpoint has the wrong physical target contract")
    if condition.output_dim != 9 or condition.add_missing_indicators:
        raise RuntimeError("source checkpoint is not the frozen compact nine-input flow")
    protocol = {
        "purpose": "response-blind fit of p(e1,e2 | exact selected event, nine frozen inputs)",
        "paths": {
            "root": str(root),
            "source_prepared": str(source_prepared),
            "source_checkpoint": str(source_checkpoint),
            "training_script": str(training_script),
            "job_script": str(job_script),
        },
        "source_inputs": {name: str(value) for name, value in source_paths.items()},
        "source_hashes": {name: sha256(value) for name, value in source_paths.items()},
        "sample": {
            "usable_shape_required": True,
            "mag_auto": "<25.8",
            "flux_radius": ">=0.75 arcsec = >=3.75 pixels at 0.2 arcsec/pixel",
            "flux_radius_semantics": "convolved SExtractor circular half-light-radius measurement",
            "ellipticity_norm_cut": None,
        },
        "factorization": "p(S|x) times p(e1,e2|S,x)",
        "conditions": list(condition.feature_names),
        "outputs": list(SHAPE_NAMES),
        "architecture": {
            "family": "same affine-coupling disk flow as the four-output source",
            "target_dim": 2,
            "context_dim": 9,
            "hidden_dim": int(parent["model_config"]["hidden_dim"]),
            "n_layers": int(parent["model_config"]["n_layers"]),
            "n_flows": int(parent["model_config"]["n_flows"]),
            "initialization": "fresh seed 501; no response-trained weights",
        },
        "recipe": {
            "seed": 501,
            "epochs": 20,
            "rows_per_epoch": 2_000_000,
            "batch_size": 8192,
            "learning_rates": [1.0e-4, 3.0e-4, 1.0e-3],
            "probe_steps": 64,
            "gradient_clip": 5.0,
            "validation_rows": 500_000,
            "validation_seed": 900_000_505,
            "scheduler": "ReduceLROnPlateau(factor=0.5,patience=3,min_lr=1e-6)",
        },
        "splits": {
            "train_cases": list(range(40, 100)),
            "validation_cases": list(range(100, 120)),
            "reused_response_screen_cases": list(range(120, 140)),
        },
        "gate": {
            "checkpoint_selection": "minimum fixed-subset selected-sample validation NLL only",
            "density_readiness": "finite losses, best epoch >0, and best epoch is not the final epoch",
            "next_if_ready": "one reused cases120-139 mechanism screen; not model promotion",
        },
        "scope": {
            "existing_catalogue_arrays_only": True,
            "new_image_simulation": False,
            "new_measurement": False,
            "new_input_features": False,
            "response_used_for_training_or_checkpointing": False,
            "forbidden": [
                "response losses, offsets, rescaling, or response-selected stopping",
                "cases120-139 before the density checkpoint is frozen",
                "photometry outputs already represented by the exact selected event",
                "additional input features",
            ],
        },
    }
    write_json(path, protocol)
    load_protocol(root, verify_source=True)
    print("MINIMAL_SELECTED_SHAPE_PROTOCOL_READY", sha256(path), flush=True)


def _selected_mask(values: np.ndarray, target: TargetStandardizer) -> np.ndarray:
    radius_flux = (
        np.asarray(values[:, 2:], dtype=np.float64) * target.scales[2:]
        + target.means[2:]
    )
    radius = radius_flux[:, 0]
    flux = radius_flux[:, 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        mag = 30.0 - 2.5 * np.log10(flux)
    return np.isfinite(mag) & np.isfinite(radius) & (mag < 25.8) & (radius >= 3.75)


def prepare_data(args: argparse.Namespace) -> None:
    root = Path(args.root).resolve()
    protocol = load_protocol(root, verify_source=True)
    output = root / "prepared"
    output.mkdir(exist_ok=False)
    parent = load_torch(protocol["paths"]["source_checkpoint"])
    target = TargetStandardizer.from_state(parent["target_transform"])
    condition = TabularPreprocessor.from_state(parent["condition_preprocessor"])
    source = Path(protocol["paths"]["source_prepared"])
    rb_mean, rb_scale = np.load(source / "rblend_normalization.npy")
    if not np.isfinite([rb_mean, rb_scale]).all() or rb_scale <= 0:
        raise RuntimeError("invalid frozen R_blend normalization")
    split_report = {}
    coordinate_parts = []
    for split in ("train", "validation"):
        values = np.load(source / f"{split}_values.npy", mmap_mode="r")
        context8 = np.load(source / f"{split}_context8.npy", mmap_mode="r")
        rblend = np.load(source / f"{split}_rblend.npy", mmap_mode="r")
        cases = np.load(source / f"{split}_cases.npy", mmap_mode="r")
        if not (len(values) == len(context8) == len(rblend) == len(cases)):
            raise RuntimeError(f"unaligned source split: {split}")
        keep = _selected_mask(values, target)
        selected_values = np.asarray(values[keep, :2], dtype=np.float64)
        raw_shape = selected_values * target.scales[:2] + target.means[:2]
        norm2 = np.sum(raw_shape**2, axis=1)
        if not np.isfinite(selected_values).all() or np.any(norm2 >= 1.0):
            raise RuntimeError(f"invalid selected shapes in {split}")
        selected_context = np.column_stack(
            (
                np.asarray(context8[keep], dtype=np.float32),
                np.asarray((rblend[keep] - rb_mean) / rb_scale, dtype=np.float32),
            )
        )
        selected_cases = np.asarray(cases[keep], dtype=np.int16)
        if selected_context.shape != (len(selected_values), condition.output_dim):
            raise RuntimeError("selected context has the wrong shape")
        if not np.isfinite(selected_context).all():
            raise RuntimeError("nonfinite selected context")
        np.save(output / f"{split}_values.npy", selected_values)
        np.save(output / f"{split}_context.npy", selected_context)
        np.save(output / f"{split}_cases.npy", selected_cases)
        unique = np.unique(cases)
        split_report[split] = {
            "source_rows": int(len(values)),
            "selected_rows": int(keep.sum()),
            "selected_fraction": float(keep.mean()),
            "cases": [int(value) for value in unique],
            "case_counts": {
                str(int(case)): {
                    "source": int(np.count_nonzero(cases == case)),
                    "selected": int(np.count_nonzero(keep & (cases == case))),
                }
                for case in unique
            },
            "minimum_one_minus_shape_norm_squared": float(np.min(1.0 - norm2)),
        }
        if split == "train":
            coordinate_parts.append(radial_atanh(torch.as_tensor(raw_shape))[0].numpy())
    coordinate = np.concatenate(coordinate_parts)
    coordinate_mean = coordinate.mean(axis=0, dtype=np.float64)
    coordinate_scale = coordinate.std(axis=0, ddof=1, dtype=np.float64)
    if not np.isfinite(coordinate_mean).all() or not np.all(coordinate_scale > 0):
        raise RuntimeError("invalid disk-coordinate normalization")
    rng = np.random.default_rng(protocol["recipe"]["validation_seed"])
    count = min(
        protocol["recipe"]["validation_rows"], split_report["validation"]["selected_rows"]
    )
    indices = np.sort(
        rng.choice(split_report["validation"]["selected_rows"], count, replace=False)
    )
    np.save(output / "validation_indices.npy", indices)
    receipt = {
        "definition": "existing usable-flow rows satisfying MAG_AUTO<25.8 and convolved FLUX_RADIUS>=0.75arcsec; no |e| cut",
        "cuts_reapplied_to_model_draws": False,
        "splits": split_report,
        "disk_coordinate_mean": coordinate_mean.tolist(),
        "disk_coordinate_scale": coordinate_scale.tolist(),
        "validation_subset_rows": int(count),
        "hashes": {name: sha256(output / name) for name in PREPARED_FILES},
    }
    write_json(output / "receipt.json", receipt)
    load_protocol(root, verify_source=True, verify_prepared=True)
    print("MINIMAL_SELECTED_SHAPE_DATA_READY", json.dumps(split_report), flush=True)


def _nll(model, values, context, batch_size: int, indices=None, per_row=False):
    model.eval()
    if indices is None:
        indices = np.arange(len(values))
    outputs = []
    total = 0.0
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            rows = torch.as_tensor(indices[start : start + batch_size], device=values.device)
            loss = -model.log_prob(values[rows], context[rows])
            if not bool(torch.isfinite(loss).all()):
                raise RuntimeError("nonfinite validation NLL")
            if per_row:
                outputs.append(loss.cpu().numpy())
            else:
                total += float(loss.sum())
    return np.concatenate(outputs) if per_row else total / len(indices)


def _train_steps(model, optimizer, values, context, indices, batch_size, clip, max_steps=None):
    model.train()
    total = 0.0
    count = 0
    maximum_norm = 0.0
    steps = 0
    for start in range(0, len(indices), batch_size):
        if max_steps is not None and steps >= max_steps:
            break
        rows = torch.as_tensor(indices[start : start + batch_size], device=values.device)
        optimizer.zero_grad(set_to_none=True)
        loss = -model.log_prob(values[rows], context[rows]).mean()
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("nonfinite training NLL")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip, error_if_nonfinite=True)
        optimizer.step()
        total += float(loss.detach()) * len(rows)
        count += len(rows)
        maximum_norm = max(maximum_norm, float(norm))
        steps += 1
    return {
        "nll": total / count,
        "rows": int(count),
        "steps": int(steps),
        "maximum_preclip_gradient_norm": maximum_norm,
    }


def train(args: argparse.Namespace) -> None:
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("selected-shape flow training requires scheduled CUDA")
    root = Path(args.root).resolve()
    protocol = load_protocol(root, verify_source=True, verify_prepared=True)
    output = root / "training"
    output.mkdir(exist_ok=False)
    parent = load_torch(protocol["paths"]["source_checkpoint"])
    parent_target = TargetStandardizer.from_state(parent["target_transform"])
    target = TargetStandardizer(
        list(SHAPE_NAMES),
        parent_target.means[:2],
        parent_target.scales[:2],
        dtype="float64",
    )
    condition = TabularPreprocessor.from_state(parent["condition_preprocessor"])
    receipt = json.loads((root / "prepared/receipt.json").read_text())
    old = parent["model_config"]
    config = {
        "flow_type": "disk_affine",
        "target_dim": 2,
        "context_dim": 9,
        "hidden_dim": int(old["hidden_dim"]),
        "n_layers": int(old["n_layers"]),
        "n_flows": int(old["n_flows"]),
        "scale_limit": float(old["scale_limit"]),
        "activation": old.get("activation", "silu"),
        "num_bins": int(old.get("num_bins", 8)),
        "tail_bound": float(old.get("tail_bound", 5.0)),
        "mean_hidden": 0,
        "disk_map": "radial_tanh",
        "disk_shape_means": target.means.tolist(),
        "disk_shape_scales": target.scales.tolist(),
        "disk_coordinate_means": receipt["disk_coordinate_mean"],
        "disk_coordinate_scales": receipt["disk_coordinate_scale"],
    }
    prepared = root / "prepared"
    values = {
        split: torch.as_tensor(np.load(prepared / f"{split}_values.npy"), device="cuda")
        for split in ("train", "validation")
    }
    contexts = {
        split: torch.as_tensor(np.load(prepared / f"{split}_context.npy"), device="cuda")
        for split in ("train", "validation")
    }
    validation_indices = np.load(prepared / "validation_indices.npy")
    recipe = protocol["recipe"]
    torch.manual_seed(recipe["seed"])
    model = build_flow(config).cuda()
    initial = {name: value.detach().clone() for name, value in model.state_dict().items()}
    probe_rng = np.random.default_rng(recipe["seed"] + 71_003)
    probe_count = min(recipe["probe_steps"] * recipe["batch_size"], len(values["train"]))
    probe_indices = probe_rng.choice(len(values["train"]), probe_count, replace=False)
    probes = []
    for rate in recipe["learning_rates"]:
        model.load_state_dict(initial, strict=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=rate, weight_decay=0.0)
        trained = _train_steps(
            model,
            optimizer,
            values["train"],
            contexts["train"],
            probe_indices,
            recipe["batch_size"],
            recipe["gradient_clip"],
            max_steps=recipe["probe_steps"],
        )
        validation_nll = _nll(
            model,
            values["validation"],
            contexts["validation"],
            recipe["batch_size"],
            validation_indices,
        )
        probes.append({"learning_rate": rate, "train": trained, "validation_nll": validation_nll})
        print("LR_PROBE", json.dumps(probes[-1]), flush=True)
    chosen_rate = min(probes, key=lambda item: item["validation_nll"])["learning_rate"]
    write_json(output / "probes.json", {"probes": probes, "chosen_learning_rate": chosen_rate})
    model.load_state_dict(initial, strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=chosen_rate, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, threshold=1.0e-4,
        threshold_mode="abs", min_lr=1.0e-6,
    )
    history = []
    best_epoch = 0
    for epoch in range(recipe["epochs"] + 1):
        trained = None
        if epoch:
            epoch_rng = np.random.default_rng(recipe["seed"] + epoch * 1_000_003)
            count = min(recipe["rows_per_epoch"], len(values["train"]))
            indices = epoch_rng.choice(len(values["train"]), count, replace=False)
            trained = _train_steps(
                model,
                optimizer,
                values["train"],
                contexts["train"],
                indices,
                recipe["batch_size"],
                recipe["gradient_clip"],
            )
        validation_nll = _nll(
            model,
            values["validation"],
            contexts["validation"],
            recipe["batch_size"],
            validation_indices,
        )
        record = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": trained,
            "validation_subset_nll": validation_nll,
        }
        history.append(record)
        if validation_nll < history[best_epoch]["validation_subset_nll"]:
            best_epoch = epoch
        metadata = {
            "training_phase": "minimal_selected_shape_NLL_only",
            "epoch": epoch,
            "response_used": False,
            "empirical_response_calibration": False,
            "manifest_sha256": sha256(root / "protocol.json"),
            "source_checkpoint": protocol["paths"]["source_checkpoint"],
            "selected_event": protocol["sample"],
            "factorization": protocol["factorization"],
            "record": record,
        }
        path = output / f"epoch{epoch:03d}.pt"
        save_measurement_model(path, model, condition, target, config, metadata)
        write_json(output / "history.json", {"history": history, "best_epoch": best_epoch})
        print("NLL", epoch, validation_nll, record["learning_rate"], flush=True)
        if epoch:
            scheduler.step(validation_nll)
    selected_path = output / f"epoch{best_epoch:03d}.pt"
    selected = load_torch(selected_path)
    model.load_state_dict(selected["state_dict"])
    row_nll = _nll(
        model,
        values["validation"],
        contexts["validation"],
        recipe["batch_size"],
        per_row=True,
    )
    cases = np.load(prepared / "validation_cases.npy")
    unique = np.unique(cases)
    case_sums = np.array([row_nll[cases == case].sum() for case in unique])
    case_counts = np.array([np.count_nonzero(cases == case) for case in unique])
    np.savez(
        output / "selected_validation.npz",
        cases=unique,
        sums=case_sums,
        counts=case_counts,
        means=case_sums / case_counts,
    )
    result = {
        "best_epoch": int(best_epoch),
        "epochs": int(recipe["epochs"]),
        "chosen_learning_rate": float(chosen_rate),
        "best_validation_subset_nll": float(history[best_epoch]["validation_subset_nll"]),
        "full_validation_nll": float(case_sums.sum() / case_counts.sum()),
        "selected_checkpoint": str(selected_path),
        "selected_sha256": sha256(selected_path),
        "parameter_count": int(sum(value.numel() for value in model.parameters())),
        "response_used": False,
        "selection": "minimum fixed-subset selected-sample validation NLL",
        "final_epoch_is_best": bool(best_epoch == recipe["epochs"]),
    }
    write_json(output / "result.json", result)
    load_protocol(root, verify_source=True, verify_prepared=True)
    print("MINIMAL_SELECTED_SHAPE_TRAINING_COMPLETE", json.dumps(result), flush=True)


def review(args: argparse.Namespace) -> None:
    root = Path(args.root).resolve()
    protocol = load_protocol(root, verify_source=True, verify_prepared=True)
    result = json.loads((root / "training/result.json").read_text())
    checkpoint = load_torch(result["selected_checkpoint"])
    metadata = checkpoint.get("metadata", {})
    reasons = []
    if sha256(result["selected_checkpoint"]) != result["selected_sha256"]:
        reasons.append("selected checkpoint hash mismatch")
    if result.get("response_used") is not False or metadata.get("response_used") is not False:
        reasons.append("response firewall failed")
    if checkpoint["model_config"].get("target_dim") != 2:
        reasons.append("model is not two-output")
    if checkpoint["model_config"].get("context_dim") != 9:
        reasons.append("model is not nine-input")
    if tuple(checkpoint["target_transform"]["target_names"]) != SHAPE_NAMES:
        reasons.append("target contract changed")
    if result["best_epoch"] <= 0:
        reasons.append("training did not improve beyond initialization")
    if result["final_epoch_is_best"]:
        reasons.append("validation NLL is still improving at the frozen boundary")
    if not all(
        math.isfinite(float(result[key]))
        for key in ("best_validation_subset_nll", "full_validation_nll")
    ):
        reasons.append("nonfinite selected-sample NLL")
    ready = not reasons
    decision = {
        "passes_response_blind_density_readiness": ready,
        "decision": (
            "freeze_for_reused_cases120_139_mechanism_screen"
            if ready
            else "do_not_open_response_extend_or_repair_NLL_only_fit"
        ),
        "reasons": reasons,
        "selected_checkpoint": result["selected_checkpoint"],
        "selected_sha256": result["selected_sha256"],
        "response_used": False,
        "new_image_simulation": False,
        "limitations": [
            "cases40-119 are reused development catalogues",
            "a later cases120-139 response is a reused mechanism screen, not acceptance",
            "no fresh multi-axis 0.3% claim is licensed",
        ],
    }
    write_json(root / "decision.json", decision)
    load_protocol(root, verify_source=True, verify_prepared=True)
    print("MINIMAL_SELECTED_SHAPE_REVIEW", json.dumps(decision), flush=True)


def selftest() -> None:
    values = np.array(
        [
            [0.0, 0.0, 3.75, 10 ** ((30.0 - 25.7) / 2.5)],
            [0.0, 0.0, 3.74, 100.0],
            [0.0, 0.0, 4.0, 10 ** ((30.0 - 25.8) / 2.5)],
        ],
        dtype=np.float64,
    )
    target = TargetStandardizer(PHYSICAL_NAMES, np.zeros(4), np.ones(4), dtype="float64")
    np.testing.assert_array_equal(_selected_mask(values, target), [True, False, False])
    config = {
        "flow_type": "disk_affine",
        "target_dim": 2,
        "context_dim": 9,
        "hidden_dim": 8,
        "n_layers": 1,
        "n_flows": 2,
        "scale_limit": 1.0,
        "disk_map": "radial_tanh",
        "disk_shape_means": [0.0, 0.0],
        "disk_shape_scales": [0.3, 0.3],
        "disk_coordinate_means": [0.0, 0.0],
        "disk_coordinate_scales": [0.5, 0.5],
    }
    model = build_flow(config)
    draw = model.sample(torch.zeros(3, 9), n_samples=4, qmc=True)
    assert draw.shape == (3, 4, 2)
    print("MINIMAL_SELECTED_SHAPE_SELFTEST_OK", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    initial = subparsers.add_parser("init")
    initial.add_argument("--root", required=True)
    initial.add_argument("--source-prepared", required=True)
    initial.add_argument("--source-checkpoint", required=True)
    initial.add_argument("--job-script", required=True)
    for command in ("prepare", "train", "review"):
        child = subparsers.add_parser(command)
        child.add_argument("--root", required=True)
    subparsers.add_parser("selftest")
    args = parser.parse_args()
    {
        "init": init_protocol,
        "prepare": prepare_data,
        "train": train,
        "review": review,
        "selftest": lambda _: selftest(),
    }[args.command](args)


if __name__ == "__main__":
    main()
