#!/usr/bin/env python3
"""NLL-only continuation of the minimal selected-shape flow.

The initial fixed 20-epoch schedule ended at its best validation NLL. This
continuation starts from that response-blind checkpoint, uses a fixed lower
learning rate, and stops only after six NLL-only epochs without improvement.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch

from sbsi.measurement_model import (
    TargetStandardizer,
    build_flow,
    save_measurement_model,
)
from sbsi.selection_model import TabularPreprocessor
from scripts.train_minimal_selected_shape_flow import (
    _nll,
    _train_steps,
    load_torch,
    sha256,
    write_json,
)


def load_protocol(root: str | Path, *, verify: bool = False) -> dict:
    root = Path(root).resolve()
    protocol = json.loads((root / "protocol.json").read_text())
    if verify:
        for name, value in protocol["inputs"].items():
            if sha256(value) != protocol["input_hashes"][name]:
                raise RuntimeError(f"continuation input changed: {value}")
    return protocol


def init_protocol(args: argparse.Namespace) -> None:
    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(exist_ok=True)
    if (root / "protocol.json").exists():
        raise FileExistsError("preserving existing continuation protocol")
    base = Path(args.base).resolve()
    result = json.loads((base / "training/result.json").read_text())
    decision = json.loads((base / "decision.json").read_text())
    if result["best_epoch"] != result["epochs"] or decision[
        "passes_response_blind_density_readiness"
    ]:
        raise RuntimeError("base fit does not require an NLL-only continuation")
    script = Path(__file__).resolve()
    repository = script.parents[1]
    inputs = {
        "base_protocol": base / "protocol.json",
        "base_prepared_receipt": base / "prepared/receipt.json",
        "base_result": base / "training/result.json",
        "base_decision": base / "decision.json",
        "initial_checkpoint": Path(result["selected_checkpoint"]),
        "training_script": script,
        "job_script": Path(args.job_script).resolve(),
    }
    for name in (
        "train_values.npy",
        "train_context.npy",
        "train_cases.npy",
        "validation_values.npy",
        "validation_context.npy",
        "validation_cases.npy",
        "validation_indices.npy",
    ):
        inputs[f"prepared/{name}"] = base / "prepared" / name
    inputs.update(
        {f"sbsi/{path.name}": path for path in sorted((repository / "sbsi").glob("*.py"))}
    )
    if any(not path.is_file() for path in inputs.values()):
        raise FileNotFoundError("continuation input is missing")
    checkpoint = load_torch(inputs["initial_checkpoint"])
    if checkpoint.get("metadata", {}).get("response_used") is not False:
        raise RuntimeError("base checkpoint did not pass the response firewall")
    protocol = {
        "purpose": "finish the response-blind selected-shape density fit before any response screen",
        "base_root": str(base),
        "inputs": {name: str(path) for name, path in inputs.items()},
        "input_hashes": {name: sha256(path) for name, path in inputs.items()},
        "recipe": {
            "seed": 501,
            "first_epoch": 21,
            "maximum_epoch": 50,
            "rows_per_epoch": 2_000_000,
            "batch_size": 8192,
            "learning_rate": 3.0e-4,
            "gradient_clip": 5.0,
            "early_stopping_patience": 6,
            "scheduler": "ReduceLROnPlateau(factor=0.5,patience=3,min_lr=1e-6)",
            "optimizer": "fresh AdamW because the initial bounded fit retained no optimizer state",
        },
        "scope": {
            "same_selected_rows": True,
            "same_nine_inputs": True,
            "same_two_outputs": True,
            "response_used": False,
            "new_image_simulation": False,
            "new_measurement": False,
        },
        "gate": "freeze the minimum fixed-subset validation NLL only after six later epochs fail to improve it",
    }
    write_json(root / "protocol.json", protocol)
    load_protocol(root, verify=True)
    print("MINIMAL_SELECTED_SHAPE_CONTINUATION_READY", sha256(root / "protocol.json"), flush=True)


def train(args: argparse.Namespace) -> None:
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("selected-shape continuation requires scheduled CUDA")
    root = Path(args.root).resolve()
    protocol = load_protocol(root, verify=True)
    output = root / "training"
    output.mkdir(exist_ok=False)
    base = Path(protocol["base_root"])
    initial = load_torch(protocol["inputs"]["initial_checkpoint"])
    target = TargetStandardizer.from_state(initial["target_transform"])
    condition = TabularPreprocessor.from_state(initial["condition_preprocessor"])
    config = initial["model_config"]
    if config.get("target_dim") != 2 or config.get("context_dim") != 9:
        raise RuntimeError("initial checkpoint is not the minimal selected-shape flow")
    model = build_flow(config).cuda()
    model.load_state_dict(initial["state_dict"], strict=True)
    values = {
        split: torch.as_tensor(np.load(base / "prepared" / f"{split}_values.npy"), device="cuda")
        for split in ("train", "validation")
    }
    contexts = {
        split: torch.as_tensor(np.load(base / "prepared" / f"{split}_context.npy"), device="cuda")
        for split in ("train", "validation")
    }
    validation_indices = np.load(base / "prepared/validation_indices.npy")
    recipe = protocol["recipe"]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=recipe["learning_rate"], weight_decay=0.0
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, threshold=1.0e-4,
        threshold_mode="abs", min_lr=1.0e-6,
    )
    base_nll = _nll(
        model,
        values["validation"],
        contexts["validation"],
        recipe["batch_size"],
        validation_indices,
    )
    history = [
        {
            "epoch": 20,
            "learning_rate": None,
            "train": None,
            "validation_subset_nll": base_nll,
            "source_checkpoint": protocol["inputs"]["initial_checkpoint"],
        }
    ]
    best_epoch = 20
    selected_path = Path(protocol["inputs"]["initial_checkpoint"])
    stopped_for_patience = False
    for epoch in range(recipe["first_epoch"], recipe["maximum_epoch"] + 1):
        rng = np.random.default_rng(recipe["seed"] + epoch * 1_000_003)
        count = min(recipe["rows_per_epoch"], len(values["train"]))
        indices = rng.choice(len(values["train"]), count, replace=False)
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
        path = output / f"epoch{epoch:03d}.pt"
        metadata = {
            **initial.get("metadata", {}),
            "training_phase": "minimal_selected_shape_NLL_only_continuation",
            "epoch": epoch,
            "response_used": False,
            "empirical_response_calibration": False,
            "continuation_protocol_sha256": sha256(root / "protocol.json"),
            "record": record,
        }
        save_measurement_model(path, model, condition, target, config, metadata)
        if validation_nll < min(row["validation_subset_nll"] for row in history[:-1]):
            best_epoch = epoch
            selected_path = path
        write_json(output / "history.json", {"history": history, "best_epoch": best_epoch})
        print("NLL_CONTINUATION", epoch, validation_nll, record["learning_rate"], flush=True)
        scheduler.step(validation_nll)
        if epoch - best_epoch >= recipe["early_stopping_patience"]:
            stopped_for_patience = True
            break
    selected = load_torch(selected_path)
    model.load_state_dict(selected["state_dict"], strict=True)
    row_nll = _nll(
        model,
        values["validation"],
        contexts["validation"],
        recipe["batch_size"],
        per_row=True,
    )
    cases = np.load(base / "prepared/validation_cases.npy")
    unique = np.unique(cases)
    sums = np.array([row_nll[cases == case].sum() for case in unique])
    counts = np.array([np.count_nonzero(cases == case) for case in unique])
    np.savez(output / "selected_validation.npz", cases=unique, sums=sums, counts=counts, means=sums / counts)
    result = {
        "best_epoch": int(best_epoch),
        "last_epoch": int(epoch),
        "stopped_for_NLL_patience": stopped_for_patience,
        "best_validation_subset_nll": float(
            next(row["validation_subset_nll"] for row in history if row["epoch"] == best_epoch)
        ),
        "full_validation_nll": float(sums.sum() / counts.sum()),
        "selected_checkpoint": str(selected_path),
        "selected_sha256": sha256(selected_path),
        "response_used": False,
        "selection": "minimum fixed-subset selected-sample validation NLL",
    }
    write_json(output / "result.json", result)
    load_protocol(root, verify=True)
    print("MINIMAL_SELECTED_SHAPE_CONTINUATION_COMPLETE", json.dumps(result), flush=True)


def review(args: argparse.Namespace) -> None:
    root = Path(args.root).resolve()
    load_protocol(root, verify=True)
    result = json.loads((root / "training/result.json").read_text())
    checkpoint = load_torch(result["selected_checkpoint"])
    reasons = []
    if result["selected_sha256"] != sha256(result["selected_checkpoint"]):
        reasons.append("selected checkpoint hash mismatch")
    if result.get("response_used") is not False or checkpoint.get("metadata", {}).get("response_used") is not False:
        reasons.append("response firewall failed")
    if result["best_epoch"] <= 20:
        reasons.append("continuation did not improve the initial checkpoint")
    if not result["stopped_for_NLL_patience"]:
        reasons.append("maximum schedule ended before NLL-only early stopping")
    if result["last_epoch"] - result["best_epoch"] < 6:
        reasons.append("fewer than six non-improving epochs follow the selected checkpoint")
    if not all(math.isfinite(float(result[key])) for key in ("best_validation_subset_nll", "full_validation_nll")):
        reasons.append("nonfinite NLL")
    ready = not reasons
    decision = {
        "passes_response_blind_density_readiness": ready,
        "decision": "freeze_for_reused_response_screen" if ready else "do_not_open_response",
        "reasons": reasons,
        "selected_checkpoint": result["selected_checkpoint"],
        "selected_sha256": result["selected_sha256"],
        "response_used": False,
        "new_image_simulation": False,
    }
    write_json(root / "decision.json", decision)
    print("MINIMAL_SELECTED_SHAPE_CONTINUATION_REVIEW", json.dumps(decision), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    initial = subparsers.add_parser("init")
    initial.add_argument("--root", required=True)
    initial.add_argument("--base", required=True)
    initial.add_argument("--job-script", required=True)
    for name in ("train", "review"):
        child = subparsers.add_parser(name)
        child.add_argument("--root", required=True)
    args = parser.parse_args()
    {"init": init_protocol, "train": train, "review": review}[args.command](args)


if __name__ == "__main__":
    main()
