#!/usr/bin/env python3
"""Final low-rate, NLL-only refinement of the selected-shape flow."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch

from sbsi.measurement_model import TargetStandardizer, build_flow, save_measurement_model
from sbsi.selection_model import TabularPreprocessor
from scripts.train_minimal_selected_shape_flow import (
    _nll,
    _train_steps,
    load_torch,
    sha256,
    write_json,
)


def load_protocol(root: str | Path, verify=False):
    root = Path(root).resolve()
    protocol = json.loads((root / "protocol.json").read_text())
    if verify:
        for name, path in protocol["inputs"].items():
            if sha256(path) != protocol["input_hashes"][name]:
                raise RuntimeError(f"refinement input changed: {path}")
    return protocol


def init(args):
    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(exist_ok=True)
    if (root / "protocol.json").exists():
        raise FileExistsError("preserving existing refinement protocol")
    prepared = Path(args.prepared).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    script = Path(__file__).resolve()
    repository = script.parents[1]
    inputs = {
        "initial_checkpoint": checkpoint,
        "training_script": script,
        "job_script": Path(args.job_script).resolve(),
    }
    for name in (
        "train_values.npy", "train_context.npy", "train_cases.npy",
        "validation_values.npy", "validation_context.npy", "validation_cases.npy",
        "validation_indices.npy", "receipt.json",
    ):
        inputs[f"prepared/{name}"] = prepared / name
    inputs.update({f"sbsi/{path.name}": path for path in sorted((repository / "sbsi").glob("*.py"))})
    if any(not path.is_file() for path in inputs.values()):
        raise FileNotFoundError("refinement input missing")
    model = load_torch(checkpoint)
    if model.get("metadata", {}).get("response_used") is not False:
        raise RuntimeError("initial checkpoint did not pass the response firewall")
    if model["model_config"].get("target_dim") != 2 or model["model_config"].get("context_dim") != 9:
        raise RuntimeError("initial checkpoint is not the minimal flow")
    protocol = {
        "purpose": "one bounded low-rate NLL-only refinement after epoch48 lacked six trailing epochs",
        "prepared": str(prepared),
        "inputs": {name: str(path) for name, path in inputs.items()},
        "input_hashes": {name: sha256(path) for name, path in inputs.items()},
        "recipe": {
            "seed": 501,
            "maximum_refinement_epochs": 30,
            "rows_per_epoch": 2_000_000,
            "batch_size": 8192,
            "learning_rate": 5.0e-5,
            "gradient_clip": 5.0,
            "early_stopping_patience": 6,
        },
        "scope": {
            "response_used": False,
            "new_image_simulation": False,
            "new_measurement": False,
            "same_rows_inputs_outputs": True,
        },
        "gate": "freeze only after six fixed-validation-NLL epochs fail to improve the selected checkpoint",
    }
    write_json(root / "protocol.json", protocol)
    load_protocol(root, verify=True)
    print("MINIMAL_SELECTED_SHAPE_REFINEMENT_READY", sha256(root / "protocol.json"), flush=True)


def train(args):
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("selected-shape refinement requires scheduled CUDA")
    root = Path(args.root).resolve()
    protocol = load_protocol(root, verify=True)
    output = root / "training"
    output.mkdir(exist_ok=False)
    prepared = Path(protocol["prepared"])
    checkpoint = load_torch(protocol["inputs"]["initial_checkpoint"])
    target = TargetStandardizer.from_state(checkpoint["target_transform"])
    condition = TabularPreprocessor.from_state(checkpoint["condition_preprocessor"])
    config = checkpoint["model_config"]
    model = build_flow(config).cuda()
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    values = {split: torch.as_tensor(np.load(prepared / f"{split}_values.npy"), device="cuda") for split in ("train", "validation")}
    contexts = {split: torch.as_tensor(np.load(prepared / f"{split}_context.npy"), device="cuda") for split in ("train", "validation")}
    validation_indices = np.load(prepared / "validation_indices.npy")
    recipe = protocol["recipe"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=recipe["learning_rate"], weight_decay=0.0)
    initial_nll = _nll(model, values["validation"], contexts["validation"], recipe["batch_size"], validation_indices)
    history = [{"refinement_epoch": 0, "validation_subset_nll": initial_nll, "train": None}]
    best = 0
    selected_path = Path(protocol["inputs"]["initial_checkpoint"])
    stopped = False
    for epoch in range(1, recipe["maximum_refinement_epochs"] + 1):
        rng = np.random.default_rng(recipe["seed"] + (50 + epoch) * 1_000_003)
        count = min(recipe["rows_per_epoch"], len(values["train"]))
        indices = rng.choice(len(values["train"]), count, replace=False)
        trained = _train_steps(model, optimizer, values["train"], contexts["train"], indices, recipe["batch_size"], recipe["gradient_clip"])
        validation_nll = _nll(model, values["validation"], contexts["validation"], recipe["batch_size"], validation_indices)
        record = {"refinement_epoch": epoch, "validation_subset_nll": validation_nll, "train": trained}
        history.append(record)
        path = output / f"refine{epoch:03d}.pt"
        metadata = {
            **checkpoint.get("metadata", {}),
            "training_phase": "minimal_selected_shape_final_NLL_only_refinement",
            "refinement_epoch": epoch,
            "response_used": False,
            "empirical_response_calibration": False,
            "refinement_protocol_sha256": sha256(root / "protocol.json"),
            "record": record,
        }
        save_measurement_model(path, model, condition, target, config, metadata)
        if validation_nll < history[best]["validation_subset_nll"]:
            best = epoch
            selected_path = path
        write_json(output / "history.json", {"history": history, "best_refinement_epoch": best})
        print("NLL_REFINEMENT", epoch, validation_nll, flush=True)
        if epoch - best >= recipe["early_stopping_patience"]:
            stopped = True
            break
    selected = load_torch(selected_path)
    model.load_state_dict(selected["state_dict"], strict=True)
    row_nll = _nll(model, values["validation"], contexts["validation"], recipe["batch_size"], per_row=True)
    cases = np.load(prepared / "validation_cases.npy")
    unique = np.unique(cases)
    sums = np.array([row_nll[cases == case].sum() for case in unique])
    counts = np.array([np.count_nonzero(cases == case) for case in unique])
    np.savez(output / "selected_validation.npz", cases=unique, sums=sums, counts=counts, means=sums / counts)
    result = {
        "best_refinement_epoch": int(best),
        "last_refinement_epoch": int(epoch),
        "stopped_for_NLL_patience": stopped,
        "best_validation_subset_nll": float(history[best]["validation_subset_nll"]),
        "full_validation_nll": float(sums.sum() / counts.sum()),
        "selected_checkpoint": str(selected_path),
        "selected_sha256": sha256(selected_path),
        "response_used": False,
    }
    write_json(output / "result.json", result)
    load_protocol(root, verify=True)
    print("MINIMAL_SELECTED_SHAPE_REFINEMENT_COMPLETE", json.dumps(result), flush=True)


def review(args):
    root = Path(args.root).resolve()
    load_protocol(root, verify=True)
    result = json.loads((root / "training/result.json").read_text())
    checkpoint = load_torch(result["selected_checkpoint"])
    reasons = []
    if sha256(result["selected_checkpoint"]) != result["selected_sha256"]:
        reasons.append("selected checkpoint hash mismatch")
    if result.get("response_used") is not False or checkpoint.get("metadata", {}).get("response_used") is not False:
        reasons.append("response firewall failed")
    if result["best_refinement_epoch"] <= 0:
        reasons.append("refinement did not improve its initial checkpoint")
    if not result["stopped_for_NLL_patience"]:
        reasons.append("bounded refinement ended before NLL-only early stopping")
    if result["last_refinement_epoch"] - result["best_refinement_epoch"] < 6:
        reasons.append("six trailing NLL epochs are not present")
    if not all(math.isfinite(float(result[key])) for key in ("best_validation_subset_nll", "full_validation_nll")):
        reasons.append("nonfinite NLL")
    ready = not reasons
    decision = {
        "passes_response_blind_density_readiness": ready,
        "decision": "freeze_for_reused_response_screen" if ready else "stop_branch_without_response",
        "reasons": reasons,
        "selected_checkpoint": result["selected_checkpoint"],
        "selected_sha256": result["selected_sha256"],
        "response_used": False,
        "new_image_simulation": False,
    }
    write_json(root / "decision.json", decision)
    print("MINIMAL_SELECTED_SHAPE_REFINEMENT_REVIEW", json.dumps(decision), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    initial = subs.add_parser("init")
    initial.add_argument("--root", required=True)
    initial.add_argument("--prepared", required=True)
    initial.add_argument("--checkpoint", required=True)
    initial.add_argument("--job-script", required=True)
    for name in ("train", "review"):
        child = subs.add_parser(name)
        child.add_argument("--root", required=True)
    args = parser.parse_args()
    {"init": init, "train": train, "review": review}[args.command](args)


if __name__ == "__main__":
    main()
