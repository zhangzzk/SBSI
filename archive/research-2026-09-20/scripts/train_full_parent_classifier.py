#!/usr/bin/env python3
"""Train one full-parent coherent-U classifier ensemble member.

No truth or measured selection is applied.  The eight inputs are the physical
flow truth/scene conditions; ``R_blend`` is deliberately excluded because the
response emulator is trained only on the fixed selected cohort and would be an
out-of-domain conditioner over the classifier's full parent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from sbsi.fixed_g0_domain import FLOW_FEATURES
from sbsi.selection_model import SelectionMLP, TabularPreprocessor


SEEDS = (20260913, 20260914, 20260915)
TRAIN_CASES = tuple(range(40, 100))
TUNE_CASES = tuple(range(100, 120))
MAXIMUM_EPOCHS = 100
PATIENCE = 15
BATCH_PAIRS = 16384
PAIRS_PER_EPOCH = 8_000_000
LEARNING_RATE = 1.0e-3
WEIGHT_DECAY = 1.0e-4
GRADIENT_CLIP = 5.0


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


def load_cases(root: Path, cases) -> dict:
    x0 = []
    x1 = []
    u0 = []
    u1 = []
    slices = []
    offset = 0
    for case in cases:
        path = root / "cases" / f"case{case:03d}.npz"
        with np.load(path, allow_pickle=False) as stored:
            ids = stored["ids"]
            a = stored["x0"].astype(np.float32, copy=False)
            b = stored["x1"].astype(np.float32, copy=False)
            ua = stored["u0"].astype(np.float32, copy=False)
            ub = stored["u1"].astype(np.float32, copy=False)
        if (
            a.shape != (len(ids), len(FLOW_FEATURES))
            or b.shape != a.shape
            or ua.shape != (len(ids),)
            or ub.shape != (len(ids),)
            or len(np.unique(ids)) != len(ids)
            or not np.isfinite(a).all()
            or not np.isfinite(b).all()
            or not np.isin(ua, [0.0, 1.0]).all()
            or not np.isin(ub, [0.0, 1.0]).all()
        ):
            raise ValueError(f"invalid prepared classifier case: {path}")
        x0.append(a)
        x1.append(b)
        u0.append(ua)
        u1.append(ub)
        slices.append((int(case), offset, offset + len(ids)))
        offset += len(ids)
    return {
        "x0_raw": np.concatenate(x0),
        "x1_raw": np.concatenate(x1),
        "u0": np.concatenate(u0),
        "u1": np.concatenate(u1),
        "case_slices": slices,
    }


def fit_preprocessor(data: dict) -> TabularPreprocessor:
    n = len(data["x0_raw"]) + len(data["x1_raw"])
    total = data["x0_raw"].sum(axis=0, dtype=np.float64)
    total += data["x1_raw"].sum(axis=0, dtype=np.float64)
    means64 = total / n
    square = np.square(data["x0_raw"].astype(np.float64)).sum(axis=0)
    square += np.square(data["x1_raw"].astype(np.float64)).sum(axis=0)
    variance = (square - n * np.square(means64)) / (n - 1)
    scales = np.sqrt(np.maximum(variance, 0.0)).astype(np.float32)
    scales[~np.isfinite(scales) | (scales < 1.0e-6)] = 1.0
    means = means64.astype(np.float32)
    return TabularPreprocessor(
        feature_names=list(FLOW_FEATURES),
        fill_values=means.copy(),
        means=means,
        scales=scales,
        add_missing_indicators=False,
        log_features=(),
    )


def transform(data: dict, prep: TabularPreprocessor, device: torch.device) -> dict:
    result = {
        "x0": torch.as_tensor(
            (data["x0_raw"] - prep.means) / prep.scales,
            dtype=torch.float32,
            device=device,
        ),
        "x1": torch.as_tensor(
            (data["x1_raw"] - prep.means) / prep.scales,
            dtype=torch.float32,
            device=device,
        ),
        "u0": torch.as_tensor(data["u0"], dtype=torch.float32, device=device),
        "u1": torch.as_tensor(data["u1"], dtype=torch.float32, device=device),
        "case_slices": data["case_slices"],
    }
    return result


@torch.no_grad()
def case_metrics(model, data: dict) -> list[dict]:
    model.eval()
    output = []
    for case, first, last in data["case_slices"]:
        sums = np.zeros(3, dtype=np.float64)
        count = 0
        for start in range(first, last, 65536):
            stop = min(start + 65536, last)
            for leg in (0, 1):
                logits = model(data[f"x{leg}"][start:stop])
                labels = data[f"u{leg}"][start:stop]
                probability = logits.sigmoid()
                sums[0] += float(
                    F.binary_cross_entropy_with_logits(logits, labels, reduction="sum")
                )
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


def train(args: argparse.Namespace) -> None:
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("classifier training must run under Slurm")
    seed = SEEDS[args.index]
    prepared_root = args.prepared_root.resolve()
    manifest_path = prepared_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["population"]["truth_analysis_cut"] is not None:
        raise ValueError("classifier preparation contains a truth analysis cut")
    if manifest["population"]["measured_selection_cut"] is not None:
        raise ValueError("classifier preparation contains a measured selection cut")
    if manifest["features"] != list(FLOW_FEATURES):
        raise ValueError("classifier feature contract differs from the physical flow")
    if manifest["cases"] != list(range(40, 120)):
        raise ValueError("classifier preparation must contain cases 40--119")
    output = args.output_root.resolve() / f"seed{seed}"
    if output.exists():
        raise FileExistsError(f"preserving existing classifier member: {output}")
    output.mkdir(parents=True)

    train_raw = load_cases(prepared_root, TRAIN_CASES)
    tune_raw = load_cases(prepared_root, TUNE_CASES)
    prep = fit_preprocessor(train_raw)
    device = torch.device("cuda")
    data = {
        "train": transform(train_raw, prep, device),
        "tune": transform(tune_raw, prep, device),
    }
    del train_raw, tune_raw
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model_config = {
        "input_dim": prep.output_dim,
        "hidden_dim": 64,
        "n_layers": 2,
        "dropout": 0.0,
        "activation": "silu",
    }
    model = SelectionMLP(**model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=6, min_lr=1.0e-5
    )
    n = len(data["train"]["u0"])
    best = float("inf")
    best_epoch = 0
    stale = 0
    history = []
    for epoch in range(1, MAXIMUM_EPOCHS + 1):
        started = time.monotonic()
        sample_n = min(PAIRS_PER_EPOCH, n)
        sampled = np.random.default_rng(seed + epoch * 1_000_003).choice(
            n, sample_n, replace=False
        )
        order = torch.as_tensor(sampled, dtype=torch.long, device=device)
        model.train()
        total = 0.0
        clipped = 0
        steps = 0
        for indices in order.split(BATCH_PAIRS):
            optimizer.zero_grad(set_to_none=True)
            logits0 = model(data["train"]["x0"][indices])
            logits1 = model(data["train"]["x1"][indices])
            loss = 0.5 * (
                F.binary_cross_entropy_with_logits(
                    logits0, data["train"]["u0"][indices]
                )
                + F.binary_cross_entropy_with_logits(
                    logits1, data["train"]["u1"][indices]
                )
            )
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("non-finite coherent-U BCE")
            loss.backward()
            norm = float(
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), GRADIENT_CLIP, error_if_nonfinite=True
                )
            )
            optimizer.step()
            total += float(loss.detach()) * len(indices)
            clipped += norm > GRADIENT_CLIP
            steps += 1
        tune = case_metrics(model, data["tune"])
        tune_bce = float(np.mean([entry["bce"] for entry in tune]))
        record = {
            "epoch": epoch,
            "sampled_train_pair_bce": total / sample_n,
            "sampled_train_pairs": sample_n,
            "tune_mean_case_bce": tune_bce,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "gradient_clipped_steps": clipped,
            "steps": steps,
            "seconds": time.monotonic() - started,
        }
        history.append(record)
        if tune_bce < best - 1.0e-7:
            best = tune_bce
            best_epoch = epoch
            stale = 0
            temporary = output / "selected.tmp"
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "preprocessor": prep.to_state(),
                    "model_config": model_config,
                    "temperature": 1.0,
                    "metadata": {
                        "seed": seed,
                        "event": "per-leg usable measurement U",
                        "population": "complete half-shear target-role parent",
                        "truth_analysis_cut": None,
                        "measured_selection_cut": None,
                        "parent_boundaries": "inherited simulation generator/target-role only",
                        "condition_features": list(prep.feature_names),
                        "r_blend_conditioned": False,
                        "training_cases": list(TRAIN_CASES),
                        "tune_cases": list(TUNE_CASES),
                        "response_used": False,
                        "selected_epoch": epoch,
                        "prepared_manifest_sha256": sha256(manifest_path),
                    },
                },
                temporary,
            )
            os.replace(temporary, output / "selected.pt")
        else:
            stale += 1
        scheduler.step(tune_bce)
        write_json(
            output / "history.json",
            {"history": history, "best_epoch": best_epoch, "best_tune_mean_case_bce": best},
        )
        print(
            f"FULL_PARENT_CLASSIFIER seed={seed} epoch={epoch} "
            f"train={total/sample_n:.8f} tune_case={tune_bce:.8f} "
            f"best={best:.8f}@{best_epoch} lr={optimizer.param_groups[0]['lr']:.3g}",
            flush=True,
        )
        if stale >= PATIENCE:
            break

    checkpoint = torch.load(output / "selected.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    selected_tune = case_metrics(model, data["tune"])
    selected_train = case_metrics(model, data["train"])
    write_json(
        output / "completed.json",
        {
            "checks_passed": True,
            "seed": seed,
            "features": list(prep.feature_names),
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "best_epoch": best_epoch,
            "stop_epoch": history[-1]["epoch"],
            "best_tune_mean_case_bce": best,
            "selected_train_case_metrics": selected_train,
            "selected_tune_case_metrics": selected_tune,
            "selected_sha256": sha256(output / "selected.pt"),
            "prepared_manifest_sha256": sha256(manifest_path),
            "response_read": False,
            "truth_analysis_cut": None,
            "measured_selection_cut": None,
        },
    )
    print(
        f"FULL_PARENT_CLASSIFIER_COMPLETE seed={seed} bce={best:.8f}", flush=True
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--index", type=int, choices=range(len(SEEDS)), required=True)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
