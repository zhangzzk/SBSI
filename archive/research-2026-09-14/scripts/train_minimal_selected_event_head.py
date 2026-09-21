#!/usr/bin/env python3
"""Response-blind proper-score test of a direct selected-event head.

The candidate keeps the frozen nine raw inputs and 2x64 MLP, but predicts the
exact deployable no-|e| event rather than usable measurement alone.  It is
compared on cases 100--119 with the frozen composed probability
P(U|x) P(MAG_AUTO<25.8, FLUX_RADIUS>=0.75 arcsec|U,x).  No response catalogue
or case at or above 120 is read.
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

from sbsi.measurement_model import load_measurement_model
from sbsi.selection_model import (
    SelectionMLP,
    TabularPreprocessor,
    load_selection_model,
)


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
TRAIN_CASES = tuple(range(40, 100))
TUNE_CASES = tuple(range(100, 120))
MODEL_SEEDS = (20260913, 20260914, 20260915)
QMC_SEEDS = tuple(range(7301, 7317))
CUT_INDEX = 6  # magnitude + radius, without measured-|e|
MAG_MAX = 25.8
RADIUS_MIN_ARCSEC = 0.75
PIXEL_SIZE_ARCSEC = 0.2
PHYSICAL_TARGETS = (
    "measured_ngmix_g1",
    "measured_ngmix_g2",
    "measured_flux_radius",
    "measured_flux_from_mag_auto",
)


def sha256(path: str | Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8_388_608), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path: str | Path, value: dict) -> None:
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    os.replace(temporary, path)


def case_root(protocol: dict, case: int) -> Path:
    key = "localization_root" if case < 90 else "replication_root"
    return Path(protocol["paths"][key]) / "prepared" / f"case{case:03d}"


def _small_inputs(args: argparse.Namespace) -> list[Path]:
    source_root = Path(args.source_root).resolve()
    source_files = sorted((source_root / "sbsi").glob("*.py"))
    if not source_files:
        raise FileNotFoundError(source_root / "sbsi")
    detector_root = Path(args.detector_root).resolve()
    return [
        Path(__file__).resolve(),
        Path(args.train_job).resolve(),
        Path(args.baseline_job).resolve(),
        Path(args.review_job).resolve(),
        Path(args.flow_checkpoint).resolve(),
        *(detector_root / f"seed{seed}/selected.pt" for seed in MODEL_SEEDS),
        *source_files,
    ]


def prepare(args: argparse.Namespace) -> None:
    root = Path(args.output_root).resolve()
    if root.exists():
        raise FileExistsError(f"preserving existing experiment: {root}")
    root.mkdir(parents=False)
    (root / "logs").mkdir()
    (root / "baseline").mkdir()

    paths = {
        "output_root": str(root),
        "localization_root": str(Path(args.localization_root).resolve()),
        "replication_root": str(Path(args.replication_root).resolve()),
        "flow_checkpoint": str(Path(args.flow_checkpoint).resolve()),
        "detector_root": str(Path(args.detector_root).resolve()),
        "source_root": str(Path(args.source_root).resolve()),
        "script": str(Path(__file__).resolve()),
        "train_job": str(Path(args.train_job).resolve()),
        "baseline_job": str(Path(args.baseline_job).resolve()),
        "review_job": str(Path(args.review_job).resolve()),
    }
    small_inputs = _small_inputs(args)
    for path in small_inputs:
        if not path.is_file():
            raise FileNotFoundError(path)

    receipt_hashes = {}
    for case in (*TRAIN_CASES, *TUNE_CASES):
        receipt = case_root({"paths": paths}, case) / "receipt.json"
        if not receipt.is_file():
            raise FileNotFoundError(receipt)
        receipt_hashes[str(case)] = sha256(receipt)

    detector_states = []
    for seed in MODEL_SEEDS:
        checkpoint = Path(paths["detector_root"]) / f"seed{seed}/selected.pt"
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if (
            tuple(state["preprocessor"]["feature_names"]) != FEATURES
            or state["preprocessor"].get("add_missing_indicators", True)
            or state["model_config"]
            != {
                "input_dim": 9,
                "hidden_dim": 64,
                "n_layers": 2,
                "dropout": 0.0,
                "activation": "silu",
            }
            or state["metadata"].get("response_used") is not False
        ):
            raise RuntimeError(f"unexpected usable-event checkpoint: {checkpoint}")
        detector_states.append(state["preprocessor"])
    for state in detector_states[1:]:
        for key in ("feature_names", "add_missing_indicators", "log_features"):
            if state[key] != detector_states[0][key]:
                raise RuntimeError(f"detector preprocessor mismatch: {key}")
        for key in ("fill_values", "means", "scales"):
            np.testing.assert_array_equal(state[key], detector_states[0][key])

    protocol = {
        "version": "minimal-selected-event-head-v2",
        "created_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "purpose": (
            "Response-blind proper-score comparison of a direct joint selected-event "
            "head with the frozen usable-detector times measurement-flow cut probability."
        ),
        "features": list(FEATURES),
        "event": {
            "label": "S = U and MAG_AUTO<25.8 and FLUX_RADIUS>=0.75arcsec",
            "stored_label": "data.npz C[6] (magnitude_radius, no measured-|e| cut)",
            "flux_radius_semantics": (
                "SExtractor circular half-light radius measured on the PSF-convolved "
                "detection image; not deconvolved ngmix T"
            ),
            "cut_index": CUT_INDEX,
            "mag_max": MAG_MAX,
            "radius_min_arcsec": RADIUS_MIN_ARCSEC,
            "pixel_size_arcsec": PIXEL_SIZE_ARCSEC,
        },
        "splits": {
            "train": list(TRAIN_CASES),
            "tune": list(TUNE_CASES),
            "response_or_test": [],
        },
        "architecture": {
            "input_dim": 9,
            "hidden_dim": 64,
            "n_layers": 2,
            "activation": "silu",
            "dropout": 0.0,
            "parameter_count": 4865,
        },
        "training": {
            "model_seeds": list(MODEL_SEEDS),
            "initialization": (
                "matching frozen same-seed nine-input usable-event checkpoint; "
                "fresh optimizer and exact same architecture"
            ),
            "maximum_epochs": 250,
            "patience": 20,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "batch_pairs": 16384,
            "gradient_clip": 5.0,
            "checkpoint_rule": "minimum mean per-case tune selected-event BCE",
        },
        "baseline": {
            "definition": (
                "equal-three-seed mean P(U|x) times the frozen epoch16 flow probability "
                "of MAG_AUTO<25.8 and FLUX_RADIUS>=0.75arcsec conditional on U,x"
            ),
            "qmc_seeds": list(QMC_SEEDS),
            "draws_per_seed_per_row": 64,
            "common_random_numbers_between_legs": True,
            "probability_integration": "mean of 16 independently shifted Sobol groups",
            "numerical_uncertainty": "delete-one-QMC-group jackknife",
        },
        "gate": {
            "unit": "simulation case",
            "contrast": "direct selected-event BCE minus frozen composed-event BCE",
            "bootstrap_replicates": 10000,
            "bootstrap_seed": 20260913,
            "continue_rule": (
                "all three direct-head seed mean contrasts are negative and the "
                "ensemble paired case-bootstrap 95% upper endpoint plus 1.96 times "
                "the QMC jackknife SE is below zero"
            ),
        },
        "scope": {
            "existing_catalogues_only": True,
            "response_used": False,
            "sealed_cases": "every case >=120, including cases140--239",
            "forbidden": [
                "new images, measurements, or catalogue rows",
                "response moments or m in training, checkpointing, comparison, or stopping",
                "features beyond the frozen nine-input interface",
                "architecture search, calibration offsets, temperatures, thresholds, or seed weights",
                "changing the 0.75arcsec convolved FLUX_RADIUS sample",
            ],
            "interpretation": (
                "A pass qualifies one frozen response test on the still-sealed existing "
                "cases140--239 endpoint; it is not a shear-calibration claim."
            ),
        },
        "paths": paths,
        "small_input_hashes": {
            str(path): sha256(path) for path in small_inputs
        },
        "case_receipt_hashes": receipt_hashes,
    }
    write_json(root / "protocol.json", protocol)
    print(f"MINIMAL_SELECTED_EVENT_PREPARED root={root}", flush=True)


def load_protocol(output_root: str | Path) -> dict:
    root = Path(output_root).resolve()
    protocol = json.loads((root / "protocol.json").read_text())
    if (
        protocol.get("version") != "minimal-selected-event-head-v2"
        or Path(protocol["paths"]["output_root"]) != root
        or protocol["splits"]["train"] != list(TRAIN_CASES)
        or protocol["splits"]["tune"] != list(TUNE_CASES)
        or protocol["splits"]["response_or_test"]
        or protocol["event"]["cut_index"] != CUT_INDEX
        or protocol["event"]["radius_min_arcsec"] != RADIUS_MIN_ARCSEC
    ):
        raise RuntimeError("protocol contract changed")
    for path, expected in protocol["small_input_hashes"].items():
        if sha256(path) != expected:
            raise RuntimeError(f"frozen input changed: {path}")
    return protocol


def _recover_rblend(stored: dict[str, np.ndarray]) -> np.ndarray:
    delta = stored["es"] - stored["e0"][None]
    denominator = np.square(delta).sum(axis=2)
    if np.any(denominator <= 1e-14):
        raise RuntimeError("cannot recover R_blend from zero shear displacement")
    by_leg = (stored["shift"] * delta).sum(axis=2) / denominator
    np.testing.assert_allclose(by_leg[0], by_leg[1], rtol=2e-6, atol=2e-7)
    rblend = by_leg.mean(axis=0)
    np.testing.assert_allclose(
        stored["shift"], rblend[None, :, None] * delta, rtol=2e-6, atol=2e-7
    )
    return rblend.astype(np.float32)


def load_case(protocol: dict, case: int) -> tuple[np.ndarray, list[pd.DataFrame], np.ndarray]:
    if case not in (*TRAIN_CASES, *TUNE_CASES):
        raise RuntimeError(f"response/test case forbidden: {case}")
    root = case_root(protocol, case)
    receipt_path = root / "receipt.json"
    if sha256(receipt_path) != protocol["case_receipt_hashes"][str(case)]:
        raise RuntimeError(f"case receipt changed: {case}")
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("case") != case or not receipt.get("checks_passed", True):
        raise RuntimeError(f"invalid case receipt: {case}")
    required = [root / "data.npz", root / "frame0.feather", root / "frame1.feather"]
    if (root / "rblend.feather").exists():
        required.append(root / "rblend.feather")
    for path in required:
        if receipt["hashes"].get(str(path)) != sha256(path):
            raise RuntimeError(f"case source changed: {path}")

    with np.load(root / "data.npz") as source:
        stored = {name: source[name] for name in source.files}
    ids = stored["ids"].astype(np.int64, copy=False)
    usable = stored["U"].astype(bool, copy=False)
    values = stored["values"]
    selected = stored["C"][CUT_INDEX].astype(bool)
    expected = (
        usable
        & (values[..., 2] < MAG_MAX)
        & (values[..., 3] >= np.log(RADIUS_MIN_ARCSEC / PIXEL_SIZE_ARCSEC))
    )
    np.testing.assert_array_equal(selected, expected)
    if selected.shape != (2, len(ids)) or len(np.unique(ids)) != len(ids):
        raise RuntimeError(f"invalid selected-event arrays: case{case:03d}")

    rblend_path = root / "rblend.feather"
    if rblend_path.exists():
        table = pd.read_feather(rblend_path, columns=["input_index", "R_blend"])
        np.testing.assert_array_equal(table["input_index"].to_numpy(np.int64), ids)
        rblend = table["R_blend"].to_numpy(np.float32)
    else:
        rblend = _recover_rblend(stored)
    if not np.isfinite(rblend).all():
        raise RuntimeError(f"nonfinite R_blend: case{case:03d}")

    frames = []
    for leg in range(2):
        frame = pd.read_feather(
            root / f"frame{leg}.feather", columns=["input_index", *FEATURES[:-1]]
        )
        np.testing.assert_array_equal(frame["input_index"].to_numpy(np.int64), ids)
        frame["R_blend"] = rblend
        if not np.isfinite(frame[list(FEATURES)].to_numpy(float)).all():
            raise RuntimeError(f"nonfinite condition: case{case:03d} leg{leg}")
        frames.append(frame)
    return ids, frames, selected.astype(np.float32)


def probability_metrics(probability: np.ndarray, label: np.ndarray) -> dict:
    probability = np.asarray(probability, dtype=np.float64)
    label = np.asarray(label, dtype=bool)
    if probability.shape != label.shape or not np.isfinite(probability).all():
        raise ValueError("invalid binary-probability arrays")
    if np.any((probability <= 0.0) & label) or np.any((probability >= 1.0) & ~label):
        raise ValueError("finite-draw probability gives an impossible observed event")
    bce = -np.where(
        label,
        np.log(probability, where=probability > 0.0, out=np.zeros_like(probability)),
        np.log1p(-probability),
    )
    return {
        "n_leg_rows": int(label.size),
        "event_fraction": float(label.mean()),
        "bce": float(bce.mean()),
        "brier": float(np.square(probability - label).mean()),
        "signed_probability_gap": float((probability - label).mean()),
        "probability_min": float(probability.min()),
        "probability_max": float(probability.max()),
    }


def load_split(protocol: dict, cases: tuple[int, ...], prep: TabularPreprocessor, device: str) -> dict:
    x = [[], []]
    selected = [[], []]
    slices = []
    offset = 0
    for case in cases:
        ids, frames, label = load_case(protocol, case)
        for leg in range(2):
            x[leg].append(prep.transform_frame(frames[leg]))
            selected[leg].append(label[leg])
        slices.append((case, offset, offset + len(ids)))
        offset += len(ids)
    result = {
        f"x{leg}": torch.as_tensor(np.concatenate(x[leg]), device=device)
        for leg in range(2)
    }
    result.update(
        {
            f"s{leg}": torch.as_tensor(np.concatenate(selected[leg]), device=device)
            for leg in range(2)
        }
    )
    result["case_slices"] = slices
    return result


@torch.no_grad()
def case_metrics_from_model(model: torch.nn.Module, data: dict) -> list[dict]:
    model.eval()
    output = []
    for case, first, last in data["case_slices"]:
        sums = np.zeros(3, dtype=np.float64)
        count = 0
        events = 0
        for start in range(first, last, 65536):
            stop = min(start + 65536, last)
            for leg in range(2):
                logits = model(data[f"x{leg}"][start:stop])
                label = data[f"s{leg}"][start:stop]
                probability = logits.sigmoid()
                sums[0] += float(F.binary_cross_entropy_with_logits(logits, label, reduction="sum"))
                sums[1] += float((probability - label).square().sum())
                sums[2] += float((probability - label).sum())
                count += len(logits)
                events += int(label.sum())
        output.append(
            {
                "case": case,
                "n_leg_rows": count,
                "event_fraction": events / count,
                "bce": sums[0] / count,
                "brier": sums[1] / count,
                "signed_probability_gap": sums[2] / count,
            }
        )
    return output


def _mean(entries: list[dict], key: str) -> float:
    return float(np.mean([entry[key] for entry in entries]))


def train(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("selected-event training requires CUDA")
    protocol = load_protocol(args.output_root)
    seed = MODEL_SEEDS[args.index]
    root = Path(args.output_root)
    directory = root / "models" / f"seed{seed}"
    if directory.exists():
        raise FileExistsError(f"preserving existing output: {directory}")
    directory.mkdir(parents=True)

    parent_path = Path(protocol["paths"]["detector_root"]) / f"seed{seed}/selected.pt"
    parent = torch.load(parent_path, map_location="cpu", weights_only=False)
    prep = TabularPreprocessor.from_state(parent["preprocessor"])
    if tuple(prep.feature_names) != FEATURES or prep.output_dim != 9:
        raise RuntimeError("nine-input preprocessor contract changed")
    model = SelectionMLP(**parent["model_config"]).cuda()
    model.load_state_dict(parent["state_dict"])
    if sum(value.numel() for value in model.parameters()) != 4865:
        raise RuntimeError("model size changed")

    data = {
        "train": load_split(protocol, TRAIN_CASES, prep, "cuda"),
        "tune": load_split(protocol, TUNE_CASES, prep, "cuda"),
    }
    settings = protocol["training"]
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings["learning_rate"],
        weight_decay=settings["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=6, min_lr=1e-5
    )
    history = []
    best = float("inf")
    best_epoch = -1
    stale = 0
    selected_path = directory / "selected.pt"

    def consider(epoch: int, tune: list[dict]) -> float:
        nonlocal best, best_epoch, stale
        score = _mean(tune, "bce")
        if score < best - 1e-7:
            best = score
            best_epoch = epoch
            stale = 0
            temporary = selected_path.with_suffix(".pt.tmp")
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "preprocessor": prep.to_state(),
                    "model_config": parent["model_config"],
                    "temperature": 1.0,
                    "metadata": {
                        "event": (
                            "per-leg U and MAG_AUTO<25.8 and convolved "
                            "FLUX_RADIUS>=0.75arcsec; no measured-|e| cut"
                        ),
                        "condition_features": list(FEATURES),
                        "training_cases": list(TRAIN_CASES),
                        "tune_cases": list(TUNE_CASES),
                        "response_used": False,
                        "selected_epoch": epoch,
                        "warm_start": str(parent_path),
                        "warm_start_sha256": sha256(parent_path),
                    },
                },
                temporary,
            )
            os.replace(temporary, selected_path)
        else:
            stale += 1
        return score

    initial_tune = case_metrics_from_model(model, data["tune"])
    initial_score = consider(0, initial_tune)
    history.append(
        {
            "epoch": 0,
            "train_object_bce": None,
            "tune_mean_case_bce": initial_score,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "gradient_clipped_steps": 0,
            "steps": 0,
            "seconds": 0.0,
        }
    )

    n = len(data["train"]["s0"])
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
                F.binary_cross_entropy_with_logits(logits0, data["train"]["s0"][indices])
                + F.binary_cross_entropy_with_logits(logits1, data["train"]["s1"][indices])
            )
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite selected-event BCE")
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
        tune = case_metrics_from_model(model, data["tune"])
        score = consider(epoch, tune)
        history.append(
            {
                "epoch": epoch,
                "train_object_bce": total / n,
                "tune_mean_case_bce": score,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "gradient_clipped_steps": clipped,
                "steps": steps,
                "seconds": time.time() - started,
            }
        )
        scheduler.step(score)
        print(
            f"SELECTED_EVENT_TRAIN seed={seed} epoch={epoch} train={total/n:.8f} "
            f"tune={score:.8f} best={best:.8f}@{best_epoch}",
            flush=True,
        )
        if stale >= settings["patience"]:
            break

    checkpoint = torch.load(selected_path, map_location="cuda", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    selected_tune = case_metrics_from_model(model, data["tune"])
    selected_train = case_metrics_from_model(model, data["train"])
    write_json(
        directory / "completed.json",
        {
            "checks_passed": True,
            "seed": seed,
            "features": list(FEATURES),
            "parameter_count": 4865,
            "best_epoch": best_epoch,
            "stop_epoch": history[-1]["epoch"],
            "initial_tune_mean_case_bce": initial_score,
            "best_tune_mean_case_bce": best,
            "selected_train_case_metrics": selected_train,
            "selected_tune_case_metrics": selected_tune,
            "history": history,
            "selected_sha256": sha256(selected_path),
            "protocol_sha256": sha256(root / "protocol.json"),
            "selection_used_only_tune_selected_event_bce": True,
            "response_read": False,
        },
    )
    print(f"SELECTED_EVENT_TRAIN_COMPLETE seed={seed} bce={best:.8f}", flush=True)


def _flow_pass_probability(samples: np.ndarray) -> np.ndarray:
    samples = np.asarray(samples)
    if samples.ndim != 3 or samples.shape[-1] != 4 or not np.isfinite(samples).all():
        raise RuntimeError("invalid physical flow samples")
    if np.any(samples[..., 2:] < 0.0):
        raise RuntimeError("negative physical radius or flux")
    radius_pixels = samples[..., 2]
    flux = samples[..., 3]
    flux_min = 10.0 ** ((30.0 - MAG_MAX) / 2.5)
    return (flux > flux_min) & (
        radius_pixels >= RADIUS_MIN_ARCSEC / PIXEL_SIZE_ARCSEC
    )


def baseline(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("flow integration requires CUDA")
    protocol = load_protocol(args.output_root)
    case = TUNE_CASES[args.index]
    destination = Path(args.output_root) / "baseline" / f"case{case:03d}.json"
    if destination.exists():
        raise FileExistsError(f"preserving existing baseline: {destination}")
    ids, frames, selected = load_case(protocol, case)

    detectors = [
        load_selection_model(
            Path(protocol["paths"]["detector_root"]) / f"seed{seed}/selected.pt",
            device="cuda",
        )
        for seed in MODEL_SEEDS
    ]
    p_usable = np.stack(
        [
            np.mean(
                [detector.predict_proba(frames[leg]) for detector in detectors], axis=0
            )
            for leg in range(2)
        ]
    ).astype(np.float64)
    flow = load_measurement_model(protocol["paths"]["flow_checkpoint"], device="cuda")
    if tuple(flow.target_transform.target_names) != PHYSICAL_TARGETS:
        raise RuntimeError("flow target contract changed")
    if tuple(flow.condition_preprocessor.feature_names) != FEATURES:
        raise RuntimeError("flow feature contract changed")

    groups = np.empty((len(QMC_SEEDS), 2, len(ids)), dtype=np.float32)
    for group, seed in enumerate(QMC_SEEDS):
        for leg in range(2):
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            samples = flow.sample(frames[leg], n_samples=64, batch_size=2048, qmc=True)
            groups[group, leg] = _flow_pass_probability(samples).mean(axis=1)
            del samples
        print(f"SELECTED_BASELINE_QMC case={case} seed={seed}", flush=True)

    q_mean = groups.mean(axis=0, dtype=np.float64)
    full = probability_metrics(p_usable * q_mean, selected)
    leave_one_out = []
    q_sum = groups.sum(axis=0, dtype=np.float64)
    for index, seed in enumerate(QMC_SEEDS):
        q = (q_sum - groups[index]) / (len(QMC_SEEDS) - 1)
        record = probability_metrics(p_usable * q, selected)
        leave_one_out.append({"omitted_qmc_seed": seed, **record})
    output = {
        "checks_passed": True,
        "case": case,
        "n_objects": len(ids),
        "label": protocol["event"]["label"],
        "qmc_seeds": list(QMC_SEEDS),
        "draws_per_seed_per_row": 64,
        "common_random_numbers_between_legs": True,
        "composed_case_metrics": full,
        "leave_one_qmc_group_out_metrics": leave_one_out,
        "mean_usable_probability": float(p_usable.mean()),
        "mean_flow_pass_probability": float(q_mean.mean()),
        "mean_composed_probability": float((p_usable * q_mean).mean()),
        "zero_flow_pass_probability_observed_events": int(
            np.count_nonzero(selected.astype(bool) & (q_mean == 0.0))
        ),
        "source_receipt_sha256": protocol["case_receipt_hashes"][str(case)],
        "protocol_sha256": sha256(Path(args.output_root) / "protocol.json"),
        "response_read": False,
    }
    write_json(destination, output)
    print(f"SELECTED_BASELINE_COMPLETE case={case} bce={full['bce']:.8f}", flush=True)


def review(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("ensemble review requires CUDA")
    protocol = load_protocol(args.output_root)
    root = Path(args.output_root)
    completed = {}
    models = []
    for seed in MODEL_SEEDS:
        directory = root / "models" / f"seed{seed}"
        record = json.loads((directory / "completed.json").read_text())
        checkpoint = directory / "selected.pt"
        if (
            not record["checks_passed"]
            or record["response_read"]
            or record["selected_sha256"] != sha256(checkpoint)
        ):
            raise RuntimeError(f"invalid direct-head output: seed{seed}")
        completed[seed] = record
        models.append(load_selection_model(checkpoint, device="cuda"))

    baseline_records = []
    for case in TUNE_CASES:
        record = json.loads((root / "baseline" / f"case{case:03d}.json").read_text())
        if (
            not record["checks_passed"]
            or record["case"] != case
            or record["response_read"]
            or record["qmc_seeds"] != list(QMC_SEEDS)
            or record["source_receipt_sha256"]
            != protocol["case_receipt_hashes"][str(case)]
        ):
            raise RuntimeError(f"invalid composed baseline: case{case:03d}")
        baseline_records.append(record)

    direct_ensemble = []
    for case in TUNE_CASES:
        _, frames, selected = load_case(protocol, case)
        probability = np.stack(
            [
                np.mean([model.predict_proba(frames[leg]) for model in models], axis=0)
                for leg in range(2)
            ]
        )
        direct_ensemble.append({"case": case, **probability_metrics(probability, selected)})

    baseline_bce = np.asarray(
        [record["composed_case_metrics"]["bce"] for record in baseline_records]
    )
    direct_bce = np.asarray([record["bce"] for record in direct_ensemble])
    delta = direct_bce - baseline_bce
    rng = np.random.default_rng(protocol["gate"]["bootstrap_seed"])
    indices = rng.integers(
        0,
        len(TUNE_CASES),
        size=(protocol["gate"]["bootstrap_replicates"], len(TUNE_CASES)),
    )
    bootstrap = delta[indices].mean(axis=1)
    interval = np.quantile(bootstrap, (0.025, 0.975))

    leave_delta = []
    for group in range(len(QMC_SEEDS)):
        baseline_leave = np.mean(
            [
                record["leave_one_qmc_group_out_metrics"][group]["bce"]
                for record in baseline_records
            ]
        )
        leave_delta.append(float(direct_bce.mean() - baseline_leave))
    leave_delta = np.asarray(leave_delta)
    qmc_se = float(
        np.sqrt(
            (len(leave_delta) - 1)
            / len(leave_delta)
            * np.square(leave_delta - leave_delta.mean()).sum()
        )
    )
    expanded_upper = float(interval[1] + 1.96 * qmc_se)

    per_seed = {}
    for seed in MODEL_SEEDS:
        entries = completed[seed]["selected_tune_case_metrics"]
        if [entry["case"] for entry in entries] != list(TUNE_CASES):
            raise RuntimeError(f"case order changed: seed{seed}")
        values = np.asarray([entry["bce"] for entry in entries])
        per_seed[str(seed)] = {
            "direct_mean_case_bce": float(values.mean()),
            "composed_mean_case_bce": float(baseline_bce.mean()),
            "paired_delta_bce": float((values - baseline_bce).mean()),
            "best_epoch": completed[seed]["best_epoch"],
            "selected_sha256": completed[seed]["selected_sha256"],
        }

    passed = bool(
        all(record["paired_delta_bce"] < 0.0 for record in per_seed.values())
        and expanded_upper < 0.0
    )
    result = {
        "checks_passed": True,
        "development_only": True,
        "response_or_test_cases_read": False,
        "event": protocol["event"],
        "features": list(FEATURES),
        "model_parameter_count": 4865,
        "per_seed": per_seed,
        "ensemble": {
            "direct_mean_case_bce": float(direct_bce.mean()),
            "composed_mean_case_bce": float(baseline_bce.mean()),
            "paired_delta_bce": float(delta.mean()),
            "case_standard_error": float(delta.std(ddof=1) / np.sqrt(len(delta))),
            "paired_case_bootstrap_ci95": interval.tolist(),
            "qmc_delete_group_jackknife_se": qmc_se,
            "case_ci95_upper_plus_1p96_qmc_se": expanded_upper,
            "cases_improved": int(np.count_nonzero(delta < 0.0)),
            "n_cases": len(TUNE_CASES),
            "direct_case_metrics": direct_ensemble,
            "composed_case_metrics": [
                {"case": case, **record["composed_case_metrics"]}
                for case, record in zip(TUNE_CASES, baseline_records, strict=True)
            ],
        },
        "gate": protocol["gate"],
        "passed": passed,
        "next_action": (
            "freeze the direct selected-event ensemble before one existing cases140-239 response test"
            if passed
            else "prune the direct selected-event branch; keep cases140-239 sealed"
        ),
        "interpretation": (
            "Proper-score qualification only; no response or 0.3% calibration claim."
        ),
        "protocol_sha256": sha256(root / "protocol.json"),
    }
    output = root / "development_result.json"
    if output.exists():
        raise FileExistsError(f"preserving existing review: {output}")
    write_json(output, result)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), flush=True)
    print("MINIMAL_SELECTED_EVENT_REVIEW_COMPLETE", flush=True)


def self_test() -> None:
    probability = np.asarray([[0.2, 0.8], [0.7, 0.1]])
    label = np.asarray([[False, True], [True, False]])
    record = probability_metrics(probability, label)
    expected = -np.mean([np.log(0.8), np.log(0.8), np.log(0.7), np.log(0.9)])
    np.testing.assert_allclose(record["bce"], expected)
    samples = np.zeros((2, 3, 4))
    samples[..., 2] = [[3.75, 3.74, 4.0], [4.0, 4.0, 4.0]]
    samples[..., 3] = [[100.0, 100.0, 1.0], [100.0, 1.0, 100.0]]
    assert _flow_pass_probability(samples).tolist() == [
        [True, False, False],
        [True, False, True],
    ]
    model = SelectionMLP(input_dim=9, hidden_dim=64, n_layers=2, dropout=0.0)
    assert sum(value.numel() for value in model.parameters()) == 4865
    print("MINIMAL_SELECTED_EVENT_SELF_TEST_PASSED", flush=True)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subparsers = result.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    for name in (
        "output_root",
        "localization_root",
        "replication_root",
        "flow_checkpoint",
        "detector_root",
        "source_root",
        "train_job",
        "baseline_job",
        "review_job",
    ):
        prepare_parser.add_argument(f"--{name.replace('_', '-')}", required=True)
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--output-root", required=True)
    train_parser.add_argument("--index", required=True, type=int, choices=range(3))
    baseline_parser = subparsers.add_parser("baseline")
    baseline_parser.add_argument("--output-root", required=True)
    baseline_parser.add_argument("--index", required=True, type=int, choices=range(20))
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
    elif args.command == "baseline":
        baseline(args)
    elif args.command == "review":
        review(args)
    else:
        self_test()


if __name__ == "__main__":
    main()
