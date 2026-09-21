#!/usr/bin/env python3
"""Minimal eight-condition versus eight-condition-plus-R_blend ablation.

This development experiment deliberately evaluates only coherent usable-event
log loss on the training/tune cases.  It never reads response moments or the
reused cases 120--139.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from sbsi.selection_model import SelectionMLP, TabularPreprocessor


FLOW_FEATURES = (
    "e1_input_p",
    "e2_input_p",
    "sersic_n_input_p",
    "r_input_p",
    "circularized_Re_input_p",
    "nbr_flux_near",
    "nbr_flux_far",
    "nbr_flux_max",
)
RBLEND_FEATURE = "R_blend"
ARMS = ("flow8", "flow8_rblend")
SEEDS = (20260913, 20260914, 20260915)
TRAIN_CASES = tuple(range(40, 100))
TUNE_CASES = tuple(range(100, 120))


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8_388_608), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: str | Path, value: dict) -> None:
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False, sort_keys=True) + "\n"
    )
    os.replace(temporary, path)


def model_parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def prepare(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root).resolve()
    if output_root.exists():
        raise FileExistsError(f"preserving existing experiment: {output_root}")
    output_root.mkdir(parents=False)
    (output_root / "logs").mkdir()
    script_path = Path(__file__).resolve()
    job_script = Path(args.job_script).resolve()
    flow_checkpoint = Path(args.flow_checkpoint).resolve()
    selection_source = Path(args.selection_source).resolve()
    protocol = {
        "version": "minimal-coherent-u-flow-rblend-ablation-v2",
        "created_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "purpose": (
            "Paired proper-score ablation of the one conditioning variable "
            "missing from the complete no-|e| moment: R_blend."
        ),
        "arms": {
            "flow8": list(FLOW_FEATURES),
            "flow8_rblend": [*FLOW_FEATURES, RBLEND_FEATURE],
        },
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
            "missing_indicators": False,
        },
        "training": {
            "maximum_epochs": 250,
            "patience": 20,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "batch_pairs": 16384,
            "gradient_clip": 5.0,
            "checkpoint_rule": "minimum mean per-case tune coherent-U BCE",
        },
        "comparison": {
            "unit": "simulation case",
            "contrast": "flow8_rblend minus flow8 tune BCE, averaged over the three seeds within case",
            "replicates": 10000,
            "bootstrap_seed": 20260913,
            "continue_rule": (
                "all three seed contrasts are negative and the paired case-bootstrap "
                "95% interval has an upper endpoint below zero"
            ),
        },
        "scope": {
            "sample": "magnitude and radius cuts retained; measured-|e| cut retired",
            "forbidden": [
                "response moments or m in training, checkpointing, or comparison",
                "cases120-139",
                "cases140-239",
                "probability offsets, temperatures, thresholds, or seed weights",
                "additional scene, representative-neighbour, redshift, or profile features",
            ],
            "interpretation": (
                "A pass supports freezing the nine-input candidate for a newly simulated "
                "acceptance set; it does not demonstrate shear calibration."
            ),
        },
        "paths": {
            "output_root": str(output_root),
            "flow_checkpoint": str(flow_checkpoint),
            "localization_root": str(Path(args.localization_root).resolve()),
            "replication_root": str(Path(args.replication_root).resolve()),
            "selection_source": str(selection_source),
            "script": str(script_path),
            "job_script": str(job_script),
        },
        "hashes": {
            "flow_checkpoint": sha256(flow_checkpoint),
            "selection_source": sha256(selection_source),
            "script": sha256(script_path),
            "job_script": sha256(job_script),
        },
    }
    write_json(output_root / "protocol.json", protocol)
    print(f"MINIMAL_RBLEND_PREPARED root={output_root}", flush=True)


def load_protocol(output_root: str | Path) -> dict:
    output_root = Path(output_root).resolve()
    protocol_path = output_root / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    if Path(protocol["paths"]["output_root"]) != output_root:
        raise RuntimeError("protocol/output-root mismatch")
    if protocol["splits"]["response_or_test"]:
        raise RuntimeError("this worker forbids response/test cases")
    for key in ("flow_checkpoint", "selection_source", "script", "job_script"):
        if sha256(protocol["paths"][key]) != protocol["hashes"][key]:
            raise RuntimeError(f"frozen input changed: {key}")
    return protocol


def case_root(protocol: dict, case: int) -> Path:
    key = "localization_root" if case < 90 else "replication_root"
    return Path(protocol["paths"][key]) / "prepared" / f"case{case:03d}"


def verified_case(protocol: dict, case: int, *, with_frames: bool = True):
    root = case_root(protocol, case)
    receipt_path = root / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    if receipt["case"] != case or not receipt.get("checks_passed", True):
        raise RuntimeError(f"invalid source receipt: case{case:03d}")
    data_path = root / "data.npz"
    rblend_path = root / "rblend.feather"
    required = [data_path]
    if rblend_path.exists():
        required.append(rblend_path)
    if with_frames:
        required.extend((root / "frame0.feather", root / "frame1.feather"))
    for path in required:
        if receipt["hashes"].get(str(path)) != sha256(path):
            raise RuntimeError(f"source hash mismatch: {path}")
    with np.load(data_path) as stored:
        ids = stored["ids"]
        usable = stored["U"].astype(np.float32) if with_frames else None
        if rblend_path.exists():
            shift = es = e0 = None
        else:
            shift = stored["shift"].astype(np.float64)
            es = stored["es"].astype(np.float64)
            e0 = stored["e0"].astype(np.float64)
    if rblend_path.exists():
        rblend = pd.read_feather(rblend_path, columns=["input_index", RBLEND_FEATURE])
        np.testing.assert_array_equal(rblend["input_index"].to_numpy(np.int64), ids)
        rb = rblend[RBLEND_FEATURE].to_numpy(np.float32)
    else:
        delta = es - e0[None]
        denominator = np.square(delta).sum(axis=2)
        if np.any(denominator <= 1e-14):
            raise RuntimeError(f"cannot recover R_blend: case{case:03d}")
        rb_by_leg = (shift * delta).sum(axis=2) / denominator
        np.testing.assert_allclose(rb_by_leg[0], rb_by_leg[1], rtol=2e-6, atol=2e-7)
        rb = rb_by_leg.mean(axis=0).astype(np.float32)
        np.testing.assert_allclose(
            shift, rb[None, :, None] * delta, rtol=2e-6, atol=2e-7
        )
    if len(np.unique(ids)) != len(ids) or not np.isfinite(rb).all():
        raise RuntimeError(f"invalid R_blend source: case{case:03d}")
    if not with_frames:
        return ids, rb, receipt
    frames = []
    for leg in range(2):
        frame = pd.read_feather(
            root / f"frame{leg}.feather", columns=["input_index", *FLOW_FEATURES]
        )
        np.testing.assert_array_equal(frame["input_index"].to_numpy(np.int64), ids)
        if not np.isfinite(frame[list(FLOW_FEATURES)].to_numpy(float)).all():
            raise RuntimeError(f"nonfinite flow condition: case{case:03d} leg{leg}")
        frame[RBLEND_FEATURE] = rb
        frames.append(frame)
    if usable.shape != (2, len(ids)):
        raise RuntimeError(f"usable-label shape mismatch: case{case:03d}")
    return ids, frames, usable, receipt


def preprocessor(protocol: dict, arm: str) -> TabularPreprocessor:
    checkpoint = torch.load(
        protocol["paths"]["flow_checkpoint"], map_location="cpu", weights_only=False
    )
    state = checkpoint["condition_preprocessor"]
    if tuple(state["feature_names"]) != FLOW_FEATURES or state.get("log_features", []):
        raise RuntimeError("flow preprocessor contract changed")
    names = list(FLOW_FEATURES)
    fill = np.asarray(state["fill_values"], dtype=np.float32)
    means = np.asarray(state["means"], dtype=np.float32)
    scales = np.asarray(state["scales"], dtype=np.float32)
    if arm == "flow8_rblend":
        values = []
        for case in TRAIN_CASES:
            _, rb, _ = verified_case(protocol, case, with_frames=False)
            values.append(rb)
        rb = np.concatenate(values).astype(np.float64)
        rb_mean = np.float32(rb.mean())
        rb_scale = np.float32(rb.std(ddof=1))
        if not np.isfinite(rb_scale) or rb_scale < 1e-6:
            raise RuntimeError("degenerate R_blend scale")
        names.append(RBLEND_FEATURE)
        fill = np.append(fill, rb_mean).astype(np.float32)
        means = np.append(means, rb_mean).astype(np.float32)
        scales = np.append(scales, rb_scale).astype(np.float32)
    elif arm != "flow8":
        raise ValueError(f"unknown arm: {arm}")
    return TabularPreprocessor(
        names, fill, means, scales, add_missing_indicators=False, log_features=()
    )


def load_split(protocol: dict, cases: tuple[int, ...], prep: TabularPreprocessor, device: str):
    x = [[], []]
    u = [[], []]
    slices = []
    source_counts = {"cross_without_shape": 0, "shape_without_cross": 0, "outside_support": 0}
    offset = 0
    for case in cases:
        ids, frames, usable, receipt = verified_case(protocol, case)
        for leg in range(2):
            x[leg].append(prep.transform_frame(frames[leg]))
            u[leg].append(usable[leg])
        slices.append((case, offset, offset + len(ids)))
        offset += len(ids)
        for leg_drop in receipt.get("drops", []):
            for name in source_counts:
                source_counts[name] += int(leg_drop.get(name, 0))
    arrays = {
        f"x{leg}": torch.as_tensor(np.concatenate(x[leg]), device=device)
        for leg in range(2)
    }
    arrays.update(
        {
            f"u{leg}": torch.as_tensor(np.concatenate(u[leg]), device=device)
            for leg in range(2)
        }
    )
    arrays["case_slices"] = slices
    arrays["source_counts"] = source_counts
    return arrays


@torch.no_grad()
def case_metrics(model: torch.nn.Module, data: dict) -> list[dict]:
    model.eval()
    output = []
    for case, first, last in data["case_slices"]:
        sums = np.zeros(3, dtype=np.float64)
        count = 0
        for start in range(first, last, 65536):
            stop = min(start + 65536, last)
            for leg in range(2):
                logits = model(data[f"x{leg}"][start:stop])
                labels = data[f"u{leg}"][start:stop]
                probability = logits.sigmoid()
                sums[0] += float(F.binary_cross_entropy_with_logits(logits, labels, reduction="sum"))
                sums[1] += float((probability - labels).square().sum())
                sums[2] += float((probability - labels).sum())
                count += len(logits)
        output.append(
            {
                "case": case,
                "n_leg_rows": count,
                "bce": sums[0] / count,
                "brier": sums[1] / count,
                "signed_probability_gap": sums[2] / count,
            }
        )
    return output


def mean_metric(entries: list[dict], name: str) -> float:
    return float(np.mean([entry[name] for entry in entries]))


def train(args: argparse.Namespace) -> None:
    protocol = load_protocol(args.output_root)
    arm = ARMS[args.index // len(SEEDS)]
    seed = SEEDS[args.index % len(SEEDS)]
    directory = Path(args.output_root) / "models" / arm / f"seed{seed}"
    if directory.exists():
        raise FileExistsError(f"preserving existing training output: {directory}")
    directory.mkdir(parents=True)
    prep = preprocessor(protocol, arm)
    data = {
        "train": load_split(protocol, TRAIN_CASES, prep, "cuda"),
        "tune": load_split(protocol, TUNE_CASES, prep, "cuda"),
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
        model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"]
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
        total = 0.0
        clipped = 0
        steps = 0
        for indices in order.split(settings["batch_pairs"]):
            optimizer.zero_grad(set_to_none=True)
            logits0 = model(data["train"]["x0"][indices])
            logits1 = model(data["train"]["x1"][indices])
            loss = 0.5 * (
                F.binary_cross_entropy_with_logits(logits0, data["train"]["u0"][indices])
                + F.binary_cross_entropy_with_logits(logits1, data["train"]["u1"][indices])
            )
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite coherent-U BCE")
            loss.backward()
            norm = float(
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), settings["gradient_clip"], error_if_nonfinite=True
                )
            )
            optimizer.step()
            total += float(loss.detach()) * len(indices)
            clipped += norm > settings["gradient_clip"]
            steps += 1
        tune = case_metrics(model, data["tune"])
        tune_bce = mean_metric(tune, "bce")
        record = {
            "epoch": epoch,
            "train_object_bce": total / n,
            "tune_mean_case_bce": tune_bce,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "gradient_clipped_steps": clipped,
            "steps": steps,
            "seconds": time.time() - started,
        }
        history.append(record)
        if tune_bce < best - 1e-7:
            best = tune_bce
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
                        "arm": arm,
                        "seed": seed,
                        "event": "per-leg usable measurement U on fully targeted coherent catalogues",
                        "condition_features": list(prep.feature_names),
                        "training_cases": list(TRAIN_CASES),
                        "tune_cases": list(TUNE_CASES),
                        "response_used": False,
                        "selected_epoch": epoch,
                    },
                },
                temporary,
            )
            os.replace(temporary, directory / "selected.pt")
        else:
            stale += 1
        scheduler.step(tune_bce)
        print(
            f"MINIMAL_RBLEND_TRAIN arm={arm} seed={seed} epoch={epoch} "
            f"train={total/n:.8f} tune_case={tune_bce:.8f} best={best:.8f}@{best_epoch}",
            flush=True,
        )
        if stale >= settings["patience"]:
            break
    selected = directory / "selected.pt"
    checkpoint = torch.load(selected, map_location="cuda", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    selected_tune = case_metrics(model, data["tune"])
    selected_train = case_metrics(model, data["train"])
    write_json(
        directory / "completed.json",
        {
            "checks_passed": True,
            "arm": arm,
            "seed": seed,
            "features": list(prep.feature_names),
            "input_dim": prep.output_dim,
            "parameter_count": model_parameter_count(model),
            "best_epoch": best_epoch,
            "stop_epoch": history[-1]["epoch"],
            "best_tune_mean_case_bce": best,
            "selected_train_case_metrics": selected_train,
            "selected_tune_case_metrics": selected_tune,
            "source_unmatched_or_dropped": {
                "train": data["train"]["source_counts"],
                "tune": data["tune"]["source_counts"],
            },
            "history": history,
            "selected_sha256": sha256(selected),
            "protocol_sha256": sha256(Path(args.output_root) / "protocol.json"),
            "selection_used_only_tune_coherent_U_BCE": True,
            "response_read": False,
        },
    )
    print(
        f"MINIMAL_RBLEND_TRAIN_PASSED arm={arm} seed={seed} "
        f"bce={mean_metric(selected_tune, 'bce'):.8f}",
        flush=True,
    )


def review(args: argparse.Namespace) -> None:
    protocol = load_protocol(args.output_root)
    root = Path(args.output_root)
    records = {}
    for arm in ARMS:
        records[arm] = {}
        for seed in SEEDS:
            path = root / "models" / arm / f"seed{seed}" / "completed.json"
            record = json.loads(path.read_text())
            checkpoint = path.with_name("selected.pt")
            if (
                not record["checks_passed"]
                or record["response_read"]
                or record["selected_sha256"] != sha256(checkpoint)
            ):
                raise RuntimeError(f"invalid training record: {arm} seed{seed}")
            records[arm][seed] = record
    cases = list(TUNE_CASES)
    per_seed = {}
    per_case_by_seed = []
    for seed in SEEDS:
        by_arm = {}
        for arm in ARMS:
            entries = records[arm][seed]["selected_tune_case_metrics"]
            if [entry["case"] for entry in entries] != cases:
                raise RuntimeError(f"tune-case order changed: {arm} seed{seed}")
            by_arm[arm] = np.asarray([entry["bce"] for entry in entries])
        difference = by_arm["flow8_rblend"] - by_arm["flow8"]
        per_case_by_seed.append(difference)
        per_seed[str(seed)] = {
            "flow8_mean_case_bce": float(by_arm["flow8"].mean()),
            "flow8_rblend_mean_case_bce": float(by_arm["flow8_rblend"].mean()),
            "paired_delta_bce": float(difference.mean()),
        }
    per_case = np.mean(per_case_by_seed, axis=0)
    rng = np.random.default_rng(protocol["comparison"]["bootstrap_seed"])
    indices = rng.integers(
        0, len(cases), size=(protocol["comparison"]["replicates"], len(cases))
    )
    bootstrap = per_case[indices].mean(axis=1)
    interval = np.quantile(bootstrap, [0.025, 0.975])
    continue_to_fresh_acceptance = bool(
        all(entry["paired_delta_bce"] < 0 for entry in per_seed.values())
        and interval[1] < 0
    )
    result = {
        "checks_passed": True,
        "development_only": True,
        "response_or_test_cases_read": False,
        "per_seed": per_seed,
        "paired_case_contrast": {
            "definition": "flow8_rblend minus flow8 tune BCE",
            "value": float(per_case.mean()),
            "case_se": float(per_case.std(ddof=1) / np.sqrt(len(per_case))),
            "case_bootstrap_ci95": interval.tolist(),
            "n_cases": len(cases),
        },
        "model_size": {
            arm: {
                "input_dim": records[arm][SEEDS[0]]["input_dim"],
                "parameter_count": records[arm][SEEDS[0]]["parameter_count"],
            }
            for arm in ARMS
        },
        "continue_to_newly_simulated_acceptance": continue_to_fresh_acceptance,
        "interpretation": (
            "Proper-score support only; no response or shear-calibration claim."
            if continue_to_fresh_acceptance
            else "The one-feature addition failed its proper-score screen; stop this branch."
        ),
        "protocol_sha256": sha256(root / "protocol.json"),
    }
    output = root / "development_result.json"
    if output.exists():
        raise FileExistsError(f"preserving existing result: {output}")
    write_json(output, result)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    print("MINIMAL_RBLEND_REVIEW_PASSED", flush=True)


def self_test() -> None:
    model8 = SelectionMLP(input_dim=8, hidden_dim=64, n_layers=2, dropout=0.0, activation="silu")
    model9 = SelectionMLP(input_dim=9, hidden_dim=64, n_layers=2, dropout=0.0, activation="silu")
    assert model_parameter_count(model8) == 4801
    assert model_parameter_count(model9) == 4865
    sample = np.asarray([[-1.0, 0.0, 1.0], [1.0, 2.0, 3.0]], dtype=np.float32)
    prep = TabularPreprocessor.fit(
        pd.DataFrame(sample, columns=list("abc")), list("abc"), add_missing_indicators=False
    )
    transformed = prep.transform_frame(pd.DataFrame(sample, columns=list("abc")))
    assert transformed.shape == sample.shape and np.isfinite(transformed).all()
    print("MINIMAL_RBLEND_SELF_TEST_PASSED", flush=True)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subparsers = result.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--output-root", required=True)
    prepare_parser.add_argument("--flow-checkpoint", required=True)
    prepare_parser.add_argument("--localization-root", required=True)
    prepare_parser.add_argument("--replication-root", required=True)
    prepare_parser.add_argument("--selection-source", required=True)
    prepare_parser.add_argument("--job-script", required=True)
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--output-root", required=True)
    train_parser.add_argument("--index", required=True, type=int, choices=range(6))
    review_parser = subparsers.add_parser("review")
    review_parser.add_argument("--output-root", required=True)
    subparsers.add_parser("self-test")
    return result


def main() -> None:
    args = parser().parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "train":
        train(args)
    elif args.command == "review":
        review(args)
    else:
        self_test()


if __name__ == "__main__":
    main()
