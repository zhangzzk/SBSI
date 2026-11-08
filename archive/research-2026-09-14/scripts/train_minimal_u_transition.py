#!/usr/bin/env python3
"""Clean nine-input coherent-U transition-loss ablation.

The experiment keeps the V3.5-like classifier data, preprocessing, network,
optimizer, seeds, and train/tune split fixed.  Its sole modeling change is a
lambda=1 conditional direction loss on discordant +/-g usable-label pairs.
No response cases or response moments are read during training or selection.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from sbsi.selection_model import SelectionMLP, load_selection_model


VERSION = "minimal-coherent-u-transition-lambda1-v1"
TRANSITION_WEIGHT = 1.0
SEEDS = (20260913, 20260914, 20260915)
TRAIN_CASES = tuple(range(40, 100))
TUNE_CASES = tuple(range(100, 120))
FEATURES = (
    "e1_input_p",
    "e2_input_p",
    "sersic_n_input_p",
    "r_input_p",
    "circularized_Re_input_p",
    "nbr_flux_near",
    "nbr_flux_far",
    "nbr_flux_max",
    "R_blend",
)
BASE_ROOT = Path(
    "/project/ls-gruen/users/zekang.zhang/sbsi_caches/"
    "minimal_coherent_u_flow_rblend_ablation_v2"
)
CURRENT_MODEL_PATHS = tuple(
    BASE_ROOT / f"models/flow8_rblend/seed{seed}/selected.pt" for seed in SEEDS
)
CURRENT_MODEL_HASHES = (
    "bbbd27a86c18bffb3e8b98e46c455ebcd2a27844ead363913bb5289ae19e6685",
    "031498aa4140be4d2dfc7605f5211a1d2379e14c16f3fbc23b79f2f7f9e028cd",
    "dab1e4cddf3b61da4cc6dee810091209b29df4dbf5dcc86cd20d3e23eebbd702",
)


def digest(path: str | Path) -> str:
    value = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8_388_608), b""):
            value.update(block)
    return value.hexdigest()


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import frozen module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_json(path: str | Path, value: dict) -> None:
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def archived_base_paths() -> tuple[Path, Path]:
    research_root = Path(__file__).resolve().parents[1]
    return (
        research_root / "scripts/train_minimal_coherent_u_rblend_ablation.py",
        research_root / "jobs/job_train_minimal_coherent_u_rblend_ablation.sh",
    )


def load_base_protocol(base) -> dict:
    protocol_path = BASE_ROOT / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    base_script, base_job = archived_base_paths()
    replacements = {"script": base_script, "job_script": base_job}
    for key, expected in protocol["hashes"].items():
        path = replacements.get(key, Path(protocol["paths"][key]))
        if digest(path) != expected:
            raise RuntimeError(f"frozen base input changed: {key} {path}")
    if protocol["version"] != "minimal-coherent-u-flow-rblend-ablation-v2":
        raise RuntimeError("wrong base protocol version")
    return protocol


def prepare(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root).resolve()
    if output_root.exists():
        raise FileExistsError(f"preserving existing experiment: {output_root}")
    script_path = Path(__file__).resolve()
    job_script = Path(args.job_script).resolve()
    base_script, base_job = archived_base_paths()
    base_protocol = BASE_ROOT / "protocol.json"
    resource_report = json.loads(Path(args.resource_report).read_text())
    if tuple(digest(path) for path in CURRENT_MODEL_PATHS) != CURRENT_MODEL_HASHES:
        raise RuntimeError("current V3.5-like model hashes changed")
    output_root.mkdir(parents=False)
    (output_root / "logs").mkdir()
    protocol = {
        "version": VERSION,
        "created_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "purpose": (
            "Causal loss ablation: keep the nine-input coherent-U classifier "
            "fixed and add one historical lambda=1 paired-transition term."
        ),
        "event": "per-leg usable measurement U",
        "features": list(FEATURES),
        "seeds": list(SEEDS),
        "splits": {
            "train": list(TRAIN_CASES),
            "tune": list(TUNE_CASES),
            "response_or_test": [],
        },
        "architecture": {
            "hidden_dim": 64,
            "n_layers": 2,
            "activation": "silu",
            "dropout": 0.0,
        },
        "training": {
            "maximum_epochs": 250,
            "patience": 20,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "batch_pairs": 16384,
            "gradient_clip": 5.0,
            "transition_weight": TRANSITION_WEIGHT,
            "objective": (
                "mean marginal coherent-U BCE plus lambda times "
                "BCEWithLogits(z_plus-z_minus,U_plus) on U-discordant pairs"
            ),
            "checkpoint_rule": "minimum mean per-case tune combined objective",
        },
        "comparison": {
            "baseline": "the same-seed frozen V3.5-like nine-input BCE checkpoint",
            "response_screen_rule": (
                "screen the frozen equal-probability three-seed ensemble once "
                "on cached cases120-139 regardless of tune result; do not select "
                "lambda, seeds, weights, or checkpoints on that response"
            ),
        },
        "scope": {
            "fixed": [
                "catalogues and U labels",
                "nine condition features and preprocessing",
                "network architecture",
                "optimizer, seeds, batches, and train/tune cases",
            ],
            "changed": ["training and tune-selection loss only"],
            "forbidden": [
                "cases120-139 or response moments during training/selection",
                "additional features or larger network",
                "lambda sweep, temperature, threshold, seed weights, offset, or scaling",
            ],
            "interpretation": (
                "Mechanism ablation only; neither a configured likelihood nor "
                "a 0.3%-calibration claim."
            ),
        },
        "paths": {
            "output_root": str(output_root),
            "script": str(script_path),
            "job_script": str(job_script),
            "base_script": str(base_script),
            "base_job_script": str(base_job),
            "base_protocol": str(base_protocol),
            "current_models": [str(path) for path in CURRENT_MODEL_PATHS],
        },
        "hashes": {
            "script": digest(script_path),
            "job_script": digest(job_script),
            "base_script": digest(base_script),
            "base_job_script": digest(base_job),
            "base_protocol": digest(base_protocol),
            "current_models": list(CURRENT_MODEL_HASHES),
        },
        "resource_snapshot": resource_report,
    }
    write_json(output_root / "protocol.json", protocol)
    print(f"MINIMAL_U_TRANSITION_PREPARED root={output_root}", flush=True)


def load_protocol(output_root: str | Path) -> dict:
    output_root = Path(output_root).resolve()
    protocol_path = output_root / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    if protocol.get("version") != VERSION:
        raise RuntimeError("wrong transition protocol version")
    if Path(protocol["paths"]["output_root"]) != output_root:
        raise RuntimeError("protocol/output-root mismatch")
    if protocol["splits"]["response_or_test"]:
        raise RuntimeError("response/test cases entered training protocol")
    for key in (
        "script",
        "job_script",
        "base_script",
        "base_job_script",
        "base_protocol",
    ):
        if digest(protocol["paths"][key]) != protocol["hashes"][key]:
            raise RuntimeError(f"frozen transition input changed: {key}")
    paths = tuple(map(Path, protocol["paths"]["current_models"]))
    if tuple(digest(path) for path in paths) != tuple(
        protocol["hashes"]["current_models"]
    ):
        raise RuntimeError("frozen current checkpoints changed")
    return protocol


def model_parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


@torch.no_grad()
def case_metrics(model: torch.nn.Module, data: dict) -> list[dict]:
    model.eval()
    output = []
    for case, first, last in data["case_slices"]:
        marginal_sum = 0.0
        brier_sum = 0.0
        gap_sum = 0.0
        transition_sum = 0.0
        transition_correct = 0
        transition_count = 0
        for start in range(first, last, 65_536):
            stop = min(start + 65_536, last)
            logits0 = model(data["x0"][start:stop])
            logits1 = model(data["x1"][start:stop])
            u0 = data["u0"][start:stop]
            u1 = data["u1"][start:stop]
            for logits, labels in ((logits0, u0), (logits1, u1)):
                probability = logits.sigmoid()
                marginal_sum += float(
                    F.binary_cross_entropy_with_logits(logits, labels, reduction="sum")
                )
                brier_sum += float((probability - labels).square().sum())
                gap_sum += float((probability - labels).sum())
            discordant = u0 != u1
            if discordant.any():
                direction_logits = logits1[discordant] - logits0[discordant]
                direction_labels = u1[discordant]
                transition_sum += float(
                    F.binary_cross_entropy_with_logits(
                        direction_logits, direction_labels, reduction="sum"
                    )
                )
                transition_correct += int(
                    ((direction_logits > 0) == direction_labels.bool()).sum()
                )
                transition_count += int(discordant.sum())
        n_leg_rows = 2 * (last - first)
        if transition_count == 0:
            raise RuntimeError(f"no U-discordant tune pairs: case={case}")
        bce = marginal_sum / n_leg_rows
        transition_bce = transition_sum / transition_count
        output.append(
            {
                "case": case,
                "n_leg_rows": n_leg_rows,
                "discordant_pairs": transition_count,
                "bce": bce,
                "brier": brier_sum / n_leg_rows,
                "signed_probability_gap": gap_sum / n_leg_rows,
                "transition_bce": transition_bce,
                "transition_accuracy": transition_correct / transition_count,
                "combined_objective": bce + TRANSITION_WEIGHT * transition_bce,
            }
        )
    return output


def mean_metric(entries: list[dict], name: str) -> float:
    return float(np.mean([entry[name] for entry in entries]))


def train(args: argparse.Namespace) -> None:
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("classifier training must run through Slurm")
    protocol = load_protocol(args.output_root)
    base = load_module(Path(protocol["paths"]["base_script"]), "minimal_u_transition_base")
    base_protocol = load_base_protocol(base)
    seed = SEEDS[args.index]
    directory = Path(args.output_root) / "models" / f"seed{seed}"
    if directory.exists():
        raise FileExistsError(f"preserving existing training output: {directory}")
    directory.mkdir(parents=True)
    prep = base.preprocessor(base_protocol, "flow8_rblend")
    if tuple(prep.feature_names) != FEATURES or prep.output_dim != len(FEATURES):
        raise RuntimeError("nine-input preprocessor changed")
    data = {
        "train": base.load_split(base_protocol, TRAIN_CASES, prep, "cuda"),
        "tune": base.load_split(base_protocol, TUNE_CASES, prep, "cuda"),
    }
    settings = protocol["training"]
    architecture = protocol["architecture"]
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model_config = {
        "input_dim": prep.output_dim,
        "hidden_dim": architecture["hidden_dim"],
        "n_layers": architecture["n_layers"],
        "dropout": architecture["dropout"],
        "activation": architecture["activation"],
    }
    model = SelectionMLP(**model_config).cuda()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings["learning_rate"],
        weight_decay=settings["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=6, min_lr=1e-5
    )
    best = float("inf")
    best_epoch = 0
    stale = 0
    history = []
    n = len(data["train"]["u0"])
    for epoch in range(1, settings["maximum_epochs"] + 1):
        started = time.time()
        generator = torch.Generator(device="cuda").manual_seed(seed + epoch)
        order = torch.randperm(n, generator=generator, device="cuda")
        model.train()
        marginal_total = 0.0
        transition_total = 0.0
        objective_total = 0.0
        discordant_total = 0
        clipped = 0
        steps = 0
        for indices in order.split(settings["batch_pairs"]):
            optimizer.zero_grad(set_to_none=True)
            logits0 = model(data["train"]["x0"][indices])
            logits1 = model(data["train"]["x1"][indices])
            u0 = data["train"]["u0"][indices]
            u1 = data["train"]["u1"][indices]
            marginal = 0.5 * (
                F.binary_cross_entropy_with_logits(logits0, u0)
                + F.binary_cross_entropy_with_logits(logits1, u1)
            )
            discordant = u0 != u1
            if not discordant.any():
                raise RuntimeError("training batch contains no U-discordant pairs")
            transition = F.binary_cross_entropy_with_logits(
                logits1[discordant] - logits0[discordant], u1[discordant]
            )
            loss = marginal + settings["transition_weight"] * transition
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite coherent-U transition objective")
            loss.backward()
            norm = float(
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    settings["gradient_clip"],
                    error_if_nonfinite=True,
                )
            )
            optimizer.step()
            marginal_total += float(marginal.detach()) * len(indices)
            transition_total += float(transition.detach()) * len(indices)
            objective_total += float(loss.detach()) * len(indices)
            discordant_total += int(discordant.sum())
            clipped += norm > settings["gradient_clip"]
            steps += 1
        tune = case_metrics(model, data["tune"])
        tune_objective = mean_metric(tune, "combined_objective")
        record = {
            "epoch": epoch,
            "train_marginal_bce_batch_mean": marginal_total / n,
            "train_transition_bce_batch_mean": transition_total / n,
            "train_combined_objective_batch_mean": objective_total / n,
            "train_discordant_pairs_seen": discordant_total,
            "tune_mean_case_bce": mean_metric(tune, "bce"),
            "tune_mean_case_transition_bce": mean_metric(tune, "transition_bce"),
            "tune_mean_case_combined_objective": tune_objective,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "gradient_clipped_steps": clipped,
            "steps": steps,
            "seconds": time.time() - started,
        }
        history.append(record)
        if tune_objective < best - 1e-7:
            best = tune_objective
            best_epoch = epoch
            stale = 0
            temporary = directory / "selected.pt.tmp"
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "preprocessor": prep.to_state(),
                    "model_config": model_config,
                    "temperature": 1.0,
                    "metadata": {
                        "version": VERSION,
                        "seed": seed,
                        "event": "per-leg usable measurement U",
                        "condition_features": list(prep.feature_names),
                        "training_cases": list(TRAIN_CASES),
                        "tune_cases": list(TUNE_CASES),
                        "training_objective": "marginal_bce_plus_discordant_transition_bce",
                        "transition_weight": TRANSITION_WEIGHT,
                        "response_used": False,
                        "selected_epoch": epoch,
                    },
                },
                temporary,
            )
            os.replace(temporary, directory / "selected.pt")
        else:
            stale += 1
        scheduler.step(tune_objective)
        print(
            f"MINIMAL_U_TRANSITION_TRAIN seed={seed} epoch={epoch} "
            f"marginal={marginal_total/n:.8f} transition={transition_total/n:.8f} "
            f"tune={tune_objective:.8f} best={best:.8f}@{best_epoch}",
            flush=True,
        )
        if stale >= settings["patience"]:
            break

    selected = directory / "selected.pt"
    checkpoint = torch.load(selected, map_location="cuda", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    selected_train = case_metrics(model, data["train"])
    selected_tune = case_metrics(model, data["tune"])
    current_path = Path(protocol["paths"]["current_models"][args.index])
    current_bundle = load_selection_model(current_path, device="cuda")
    current_state = current_bundle.preprocessor.to_state()
    trained_state = prep.to_state()
    if (
        current_state["feature_names"] != trained_state["feature_names"]
        or current_state["add_missing_indicators"]
        != trained_state["add_missing_indicators"]
        or current_state["log_features"] != trained_state["log_features"]
    ):
        raise RuntimeError("current and transition preprocessors have different contracts")
    for name in ("fill_values", "means", "scales"):
        np.testing.assert_array_equal(current_state[name], trained_state[name])
    current = current_bundle.model
    current_tune = case_metrics(current, data["tune"])
    write_json(
        directory / "completed.json",
        {
            "checks_passed": True,
            "seed": seed,
            "features": list(prep.feature_names),
            "input_dim": prep.output_dim,
            "parameter_count": model_parameter_count(model),
            "best_epoch": best_epoch,
            "stop_epoch": history[-1]["epoch"],
            "best_tune_mean_case_combined_objective": best,
            "selected_train_case_metrics": selected_train,
            "selected_tune_case_metrics": selected_tune,
            "same_seed_current_tune_case_metrics": current_tune,
            "source_unmatched_or_dropped": {
                "train": data["train"]["source_counts"],
                "tune": data["tune"]["source_counts"],
            },
            "history": history,
            "selected_sha256": digest(selected),
            "protocol_sha256": digest(Path(args.output_root) / "protocol.json"),
            "response_read": False,
        },
    )
    print(
        f"MINIMAL_U_TRANSITION_TRAIN_PASSED seed={seed} "
        f"objective={mean_metric(selected_tune, 'combined_objective'):.8f}",
        flush=True,
    )


def review(args: argparse.Namespace) -> None:
    load_protocol(args.output_root)
    records = []
    per_seed = {}
    for seed in SEEDS:
        directory = Path(args.output_root) / "models" / f"seed{seed}"
        record = json.loads((directory / "completed.json").read_text())
        checkpoint = directory / "selected.pt"
        if (
            not record.get("checks_passed")
            or record.get("response_read")
            or record.get("selected_sha256") != digest(checkpoint)
            or record.get("protocol_sha256")
            != digest(Path(args.output_root) / "protocol.json")
        ):
            raise RuntimeError(f"invalid training record: seed={seed}")
        records.append(record)
        trained = record["selected_tune_case_metrics"]
        current = record["same_seed_current_tune_case_metrics"]
        if [entry["case"] for entry in trained] != list(TUNE_CASES):
            raise RuntimeError(f"trained tune case order changed: seed={seed}")
        if [entry["case"] for entry in current] != list(TUNE_CASES):
            raise RuntimeError(f"current tune case order changed: seed={seed}")
        per_seed[str(seed)] = {
            "selected_epoch": record["best_epoch"],
            "stop_epoch": record["stop_epoch"],
            "trained_mean_case_bce": mean_metric(trained, "bce"),
            "current_mean_case_bce": mean_metric(current, "bce"),
            "delta_bce": mean_metric(trained, "bce") - mean_metric(current, "bce"),
            "trained_mean_case_transition_bce": mean_metric(trained, "transition_bce"),
            "current_mean_case_transition_bce": mean_metric(current, "transition_bce"),
            "delta_transition_bce": mean_metric(trained, "transition_bce")
            - mean_metric(current, "transition_bce"),
            "trained_mean_case_transition_accuracy": mean_metric(
                trained, "transition_accuracy"
            ),
            "current_mean_case_transition_accuracy": mean_metric(
                current, "transition_accuracy"
            ),
        }
    result = {
        "analysis": "frozen tune-only review of clean nine-input U transition-loss ablation",
        "response_read": False,
        "transition_weight": TRANSITION_WEIGHT,
        "per_seed": per_seed,
        "all_seeds_improve_tune_transition_bce": all(
            entry["delta_transition_bce"] < 0.0 for entry in per_seed.values()
        ),
        "all_seeds_preserve_or_improve_tune_marginal_bce": all(
            entry["delta_bce"] <= 0.0 for entry in per_seed.values()
        ),
        "models": [
            {
                "path": str(
                    Path(args.output_root) / "models" / f"seed{seed}/selected.pt"
                ),
                "sha256": record["selected_sha256"],
            }
            for seed, record in zip(SEEDS, records)
        ],
        "protocol_sha256": digest(Path(args.output_root) / "protocol.json"),
        "limitations": [
            "The conditional discordant-pair objective is not a proper marginal-probability score under arbitrarily correlated paired noise.",
            "This fixed lambda=1 ablation is a mechanism test, not a tuned optimum.",
            "No response case, response moment, or m value was read by training, checkpointing, or this review.",
        ],
    }
    write_json(Path(args.output_root) / "review.json", result)
    print("MINIMAL_U_TRANSITION_REVIEW_PASSED", flush=True)


def selftest() -> None:
    logits0 = torch.tensor([-1.0, 1.0, 0.5])
    logits1 = torch.tensor([1.0, -1.0, 0.5])
    u0 = torch.tensor([0.0, 1.0, 1.0])
    u1 = torch.tensor([1.0, 0.0, 1.0])
    discordant = u0 != u1
    good = F.binary_cross_entropy_with_logits(
        logits1[discordant] - logits0[discordant], u1[discordant]
    )
    bad = F.binary_cross_entropy_with_logits(
        logits0[discordant] - logits1[discordant], u1[discordant]
    )
    if not float(good) < float(bad):
        raise RuntimeError("transition direction self-test failed")
    print("MINIMAL_U_TRANSITION_SELFTEST_PASSED", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "train", "review", "selftest"))
    parser.add_argument("--output-root")
    parser.add_argument("--job-script")
    parser.add_argument("--resource-report")
    parser.add_argument("--index", type=int)
    arguments = parser.parse_args()
    if arguments.mode == "selftest":
        selftest()
    elif arguments.mode == "prepare":
        if not all(
            (arguments.output_root, arguments.job_script, arguments.resource_report)
        ):
            parser.error("prepare requires --output-root, --job-script, and --resource-report")
        prepare(arguments)
    elif arguments.mode == "train":
        if arguments.output_root is None or arguments.index not in range(len(SEEDS)):
            parser.error("train requires --output-root and --index in [0,2]")
        train(arguments)
    else:
        if arguments.output_root is None:
            parser.error("review requires --output-root")
        review(arguments)
