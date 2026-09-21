#!/usr/bin/env python3
"""Minimal nine-input coherent-U classifier with one response-moment term.

The experiment changes only the loss and checkpoint objective of the frozen
V3.5-like classifier recipe.  Ordinary coherent-U BCE remains the proper
probability score.  A lambda=100 squared constraint matches the pooled truth
response produced by classifier probabilities to the same cached-moment
response produced by the binary U oracle on train/tune cases.  Cases120--139
are read only by the separate, one-shot ``screen`` command.
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


VERSION = "minimal-coherent-u-response-moment-lambda100-v1"
RESPONSE_WEIGHT = 100.0
SHEAR_STEP = 0.04
MOMENT_CUT = 6
SEEDS = (20260913, 20260914, 20260915)
TRAIN_CASES = tuple(range(40, 100))
TUNE_CASES = tuple(range(100, 120))
SCREEN_CASES = tuple(range(120, 140))
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
MOMENT_ROOT = Path(
    "/project/ls-gruen/users/zekang.zhang/sbsi_caches/"
    "coherent_u_crowd_ensemble_v1"
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


def write_json(path: str | Path, value: dict, *, preserve: bool = False) -> None:
    path = Path(path)
    if preserve and path.exists():
        raise FileExistsError(f"preserving existing output: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def archived_base_paths() -> tuple[Path, Path]:
    research_root = Path(__file__).resolve().parents[1]
    return (
        research_root / "scripts/train_minimal_coherent_u_rblend_ablation.py",
        research_root / "jobs/job_train_minimal_coherent_u_rblend_ablation.sh",
    )


def load_base_protocol() -> dict:
    protocol = json.loads((BASE_ROOT / "protocol.json").read_text())
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
    review_job_script = Path(args.review_job_script).resolve()
    moment_worker = MOMENT_ROOT / "worker.py"
    base_script, base_job = archived_base_paths()
    base_protocol = BASE_ROOT / "protocol.json"
    resource_report = json.loads(Path(args.resource_report).read_text())
    if tuple(digest(path) for path in CURRENT_MODEL_PATHS) != CURRENT_MODEL_HASHES:
        raise RuntimeError("current V3.5-like model hashes changed")
    load_base_protocol()
    output_root.mkdir(parents=False)
    (output_root / "logs").mkdir()
    protocol = {
        "version": VERSION,
        "created_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "purpose": (
            "Single-change classifier ablation: retain the nine-input coherent-U "
            "BCE recipe and add one pooled truth-response moment constraint."
        ),
        "event": "per-leg usable measurement U",
        "selection": "MAG_AUTO<25.8; measured FLUX_RADIUS>=0.75 arcsec; no measured-|e| cut",
        "features": list(FEATURES),
        "seeds": list(SEEDS),
        "splits": {
            "train": list(TRAIN_CASES),
            "tune": list(TUNE_CASES),
            "one_shot_reused_screen": list(SCREEN_CASES),
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
            "response_weight": RESPONSE_WEIGHT,
            "response_definition": (
                "(weighted mean es1 in +g leg minus weighted mean es1 in -g leg)/0.04; "
                "weights are predicted_probability*q6, target weights are U*q6"
            ),
            "objective": "coherent-U BCE + 100*(R_truth_probability-R_truth_U)^2",
            "minibatch_gradient": (
                "first-order full-split response-penalty gradient, with pooled "
                "denominators and means refreshed at the start of every epoch"
            ),
            "checkpoint_rule": "minimum tune mean-case BCE plus exact tune response penalty",
            "response_weight_rationale": (
                "A response error of 0.003 contributes 0.0009 BCE units; fixed before training."
            ),
        },
        "comparison": {
            "baseline": "same-seed frozen V3.5-like nine-input BCE checkpoint",
            "screen_rule": (
                "evaluate the equal-probability three-seed ensemble once on cached "
                "cases120--139; do not tune lambda, seeds, weights, or checkpoints"
            ),
        },
        "scope": {
            "fixed": [
                "existing catalogues, U labels, and cached 16-by-64 flow moments",
                "nine features and preprocessing",
                "network architecture, optimizer, seeds, batches, and splits",
            ],
            "changed": ["one training/tune response-moment penalty"],
            "forbidden": [
                "new image simulation or catalogue generation",
                "screen cases during training or checkpoint selection",
                "additional features, network capacity, or lambda sweep",
                "temperature, threshold, seed weights, offset, or empirical scaling",
                "measured y response or m in training/tune objectives",
            ],
            "interpretation": (
                "Development mechanism test only; not the configured likelihood or "
                "a 0.3%-calibration claim."
            ),
        },
        "paths": {
            "output_root": str(output_root),
            "script": str(script_path),
            "job_script": str(job_script),
            "review_job_script": str(review_job_script),
            "base_script": str(base_script),
            "base_job_script": str(base_job),
            "base_protocol": str(base_protocol),
            "moment_worker": str(moment_worker),
            "current_models": [str(path) for path in CURRENT_MODEL_PATHS],
        },
        "hashes": {
            "script": digest(script_path),
            "job_script": digest(job_script),
            "review_job_script": digest(review_job_script),
            "base_script": digest(base_script),
            "base_job_script": digest(base_job),
            "base_protocol": digest(base_protocol),
            "moment_worker": digest(moment_worker),
            "current_models": list(CURRENT_MODEL_HASHES),
        },
        "resource_snapshot": resource_report,
    }
    write_json(output_root / "protocol.json", protocol)
    print(f"MINIMAL_U_RESPONSE_PREPARED root={output_root}", flush=True)


def load_protocol(output_root: str | Path) -> dict:
    output_root = Path(output_root).resolve()
    protocol = json.loads((output_root / "protocol.json").read_text())
    if protocol.get("version") != VERSION:
        raise RuntimeError("wrong response-moment protocol version")
    if Path(protocol["paths"]["output_root"]) != output_root:
        raise RuntimeError("protocol/output-root mismatch")
    for key in (
        "script",
        "job_script",
        "review_job_script",
        "base_script",
        "base_job_script",
        "base_protocol",
        "moment_worker",
    ):
        if digest(protocol["paths"][key]) != protocol["hashes"][key]:
            raise RuntimeError(f"frozen response-moment input changed: {key}")
    models = tuple(map(Path, protocol["paths"]["current_models"]))
    if tuple(digest(path) for path in models) != tuple(
        protocol["hashes"]["current_models"]
    ):
        raise RuntimeError("frozen current checkpoints changed")
    return protocol


def model_parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def load_split(base, base_protocol: dict, coherent, coherent_config: dict,
               cases: tuple[int, ...], prep, device: str) -> dict:
    data = base.load_split(base_protocol, cases, prep, device)
    q = [[], []]
    es1 = [[], []]
    for case in cases:
        source, _, _, moments, _ = coherent.source_case(
            coherent_config, case, with_moments=True
        )
        ids, _, _ = base.verified_case(base_protocol, case, with_frames=False)
        np.testing.assert_array_equal(ids, source["ids"])
        for leg in range(2):
            q[leg].append(moments["q"][MOMENT_CUT, leg].astype(np.float32))
            es1[leg].append(source["es"][leg, :, 0].astype(np.float32))
    for leg in range(2):
        data[f"q{leg}"] = torch.as_tensor(np.concatenate(q[leg]), device=device)
        data[f"es1_{leg}"] = torch.as_tensor(
            np.concatenate(es1[leg]), device=device
        )
    if any(len(data[name]) != len(data["u0"]) for name in ("q0", "q1", "es1_0", "es1_1")):
        raise RuntimeError("response arrays do not match classifier rows")
    data["oracle_response"] = oracle_response(data)
    return data


def weighted_response(parent0: torch.Tensor, parent1: torch.Tensor,
                      data: dict) -> torch.Tensor:
    means = []
    for leg, parent in enumerate((parent0, parent1)):
        weight = parent * data[f"q{leg}"]
        means.append((weight * data[f"es1_{leg}"]).sum() / weight.sum())
    return (means[0] - means[1]) / SHEAR_STEP


@torch.no_grad()
def oracle_response(data: dict) -> float:
    return float(weighted_response(data["u0"], data["u1"], data))


@torch.no_grad()
def probability_response(model: torch.nn.Module, data: dict) -> dict:
    model.eval()
    sums = np.zeros((2, 2), dtype=np.float64)
    n = len(data["u0"])
    for start in range(0, n, 65_536):
        stop = min(start + 65_536, n)
        for leg in range(2):
            probability = model(data[f"x{leg}"][start:stop]).sigmoid()
            weight = probability * data[f"q{leg}"][start:stop]
            sums[leg, 0] += float(weight.sum())
            sums[leg, 1] += float(
                (weight * data[f"es1_{leg}"][start:stop]).sum()
            )
    means = sums[:, 1] / sums[:, 0]
    response = float((means[0] - means[1]) / SHEAR_STEP)
    return {
        "response": response,
        "oracle_response": float(data["oracle_response"]),
        "error": response - float(data["oracle_response"]),
        "denominator_means": (sums[:, 0] / n).tolist(),
        "weighted_shape_means": means.tolist(),
    }


def response_surrogate(probability0: torch.Tensor, probability1: torch.Tensor,
                       indices: torch.Tensor, data: dict, state: dict) -> torch.Tensor:
    terms = []
    for leg, probability in enumerate((probability0, probability1)):
        centered = data[f"es1_{leg}"][indices] - state["weighted_shape_means"][leg]
        term = (
            probability * data[f"q{leg}"][indices] * centered
        ).mean() / state["denominator_means"][leg]
        terms.append(term)
    return (terms[0] - terms[1]) / SHEAR_STEP


def mean_metric(entries: list[dict], name: str) -> float:
    return float(np.mean([entry[name] for entry in entries]))


def evaluate(model: torch.nn.Module, data: dict, base) -> dict:
    cases = base.case_metrics(model, data)
    response = probability_response(model, data)
    bce = mean_metric(cases, "bce")
    return {
        "case_metrics": cases,
        "mean_case_bce": bce,
        "response": response,
        "objective": bce + RESPONSE_WEIGHT * response["error"] ** 2,
    }


def train(args: argparse.Namespace) -> None:
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("classifier training requires a Slurm GPU allocation")
    protocol = load_protocol(args.output_root)
    base = load_module(Path(protocol["paths"]["base_script"]), "response_moment_base")
    coherent = load_module(
        Path(protocol["paths"]["moment_worker"]), "response_moment_worker"
    )
    base_protocol = load_base_protocol()
    coherent_config = coherent.verify()
    seed = SEEDS[args.index]
    directory = Path(args.output_root) / "models" / f"seed{seed}"
    if directory.exists():
        raise FileExistsError(f"preserving existing training output: {directory}")
    directory.mkdir(parents=True)
    prep = base.preprocessor(base_protocol, "flow8_rblend")
    if tuple(prep.feature_names) != FEATURES or prep.output_dim != len(FEATURES):
        raise RuntimeError("nine-input preprocessor changed")
    data = {
        "train": load_split(
            base, base_protocol, coherent, coherent_config, TRAIN_CASES, prep, "cuda"
        ),
        "tune": load_split(
            base, base_protocol, coherent, coherent_config, TUNE_CASES, prep, "cuda"
        ),
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
        response_start = probability_response(model, data["train"])
        state = {
            "denominator_means": tuple(
                torch.tensor(value, device="cuda")
                for value in response_start["denominator_means"]
            ),
            "weighted_shape_means": tuple(
                torch.tensor(value, device="cuda")
                for value in response_start["weighted_shape_means"]
            ),
        }
        penalty_coefficient = 2.0 * RESPONSE_WEIGHT * response_start["error"]
        generator = torch.Generator(device="cuda").manual_seed(seed + epoch)
        order = torch.randperm(n, generator=generator, device="cuda")
        model.train()
        bce_total = 0.0
        clipped = 0
        steps = 0
        for indices in order.split(settings["batch_pairs"]):
            optimizer.zero_grad(set_to_none=True)
            logits0 = model(data["train"]["x0"][indices])
            logits1 = model(data["train"]["x1"][indices])
            bce = 0.5 * (
                F.binary_cross_entropy_with_logits(
                    logits0, data["train"]["u0"][indices]
                )
                + F.binary_cross_entropy_with_logits(
                    logits1, data["train"]["u1"][indices]
                )
            )
            surrogate = response_surrogate(
                logits0.sigmoid(), logits1.sigmoid(), indices, data["train"], state
            )
            loss = bce + penalty_coefficient * surrogate
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite coherent-U response-moment objective")
            loss.backward()
            norm = float(
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    settings["gradient_clip"],
                    error_if_nonfinite=True,
                )
            )
            optimizer.step()
            bce_total += float(bce.detach()) * len(indices)
            clipped += norm > settings["gradient_clip"]
            steps += 1
        tune = evaluate(model, data["tune"], base)
        record = {
            "epoch": epoch,
            "train_object_bce_batch_mean": bce_total / n,
            "train_response_at_epoch_start": response_start,
            "tune_mean_case_bce": tune["mean_case_bce"],
            "tune_response": tune["response"],
            "tune_exact_objective": tune["objective"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "gradient_clipped_steps": clipped,
            "steps": steps,
            "seconds": time.time() - started,
        }
        history.append(record)
        if tune["objective"] < best - 1e-7:
            best = tune["objective"]
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
                        "training_objective": (
                            "coherent_u_bce_plus_squared_truth_response_moment"
                        ),
                        "response_weight": RESPONSE_WEIGHT,
                        "response_used": True,
                        "measured_y_or_m_used": False,
                        "selected_epoch": epoch,
                    },
                },
                temporary,
            )
            os.replace(temporary, directory / "selected.pt")
        else:
            stale += 1
        scheduler.step(tune["objective"])
        print(
            f"MINIMAL_U_RESPONSE_TRAIN seed={seed} epoch={epoch} "
            f"bce={bce_total/n:.8f} train_dR={response_start['error']:+.8f} "
            f"tune_dR={tune['response']['error']:+.8f} "
            f"objective={tune['objective']:.8f} best={best:.8f}@{best_epoch}",
            flush=True,
        )
        if stale >= settings["patience"]:
            break

    selected = directory / "selected.pt"
    checkpoint = torch.load(selected, map_location="cuda", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    selected_train = evaluate(model, data["train"], base)
    selected_tune = evaluate(model, data["tune"], base)
    current_bundle = load_selection_model(
        Path(protocol["paths"]["current_models"][args.index]), device="cuda"
    )
    current_state = current_bundle.preprocessor.to_state()
    trained_state = prep.to_state()
    if current_state["feature_names"] != trained_state["feature_names"]:
        raise RuntimeError("current and trained feature contracts differ")
    for name in ("fill_values", "means", "scales"):
        np.testing.assert_array_equal(current_state[name], trained_state[name])
    current = {
        "train": evaluate(current_bundle.model, data["train"], base),
        "tune": evaluate(current_bundle.model, data["tune"], base),
    }
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
            "best_tune_objective": best,
            "selected": {"train": selected_train, "tune": selected_tune},
            "same_seed_current": current,
            "source_unmatched_or_dropped": {
                split: data[split]["source_counts"] for split in ("train", "tune")
            },
            "history": history,
            "selected_sha256": digest(selected),
            "protocol_sha256": digest(Path(args.output_root) / "protocol.json"),
            "response_used": True,
            "measured_y_or_m_read": False,
            "screen_cases_read": False,
        },
    )
    print(
        f"MINIMAL_U_RESPONSE_TRAIN_PASSED seed={seed} "
        f"tune_dR={selected_tune['response']['error']:+.8f}",
        flush=True,
    )


def preflight(args: argparse.Namespace) -> None:
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("the data preflight must run through Slurm")
    protocol = load_protocol(args.output_root)
    base = load_module(Path(protocol["paths"]["base_script"]), "response_preflight_base")
    coherent = load_module(
        Path(protocol["paths"]["moment_worker"]), "response_preflight_worker"
    )
    base_protocol = load_base_protocol()
    coherent_config = coherent.verify()
    prep = base.preprocessor(base_protocol, "flow8_rblend")
    data = load_split(
        base, base_protocol, coherent, coherent_config, (TRAIN_CASES[0],), prep, "cpu"
    )
    current = load_selection_model(
        Path(protocol["paths"]["current_models"][0]), device="cpu"
    )
    result = {
        "checks_passed": True,
        "case": TRAIN_CASES[0],
        "rows_per_leg": len(data["u0"]),
        "current": evaluate(current.model, data, base),
        "protocol_sha256": digest(Path(args.output_root) / "protocol.json"),
    }
    if not np.isfinite(result["current"]["objective"]):
        raise RuntimeError("nonfinite preflight objective")
    write_json(Path(args.output_root) / "preflight.json", result, preserve=True)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    print("MINIMAL_U_RESPONSE_PREFLIGHT_PASSED", flush=True)


def review(args: argparse.Namespace) -> None:
    load_protocol(args.output_root)
    root = Path(args.output_root)
    per_seed = {}
    models = []
    for seed in SEEDS:
        directory = root / "models" / f"seed{seed}"
        record = json.loads((directory / "completed.json").read_text())
        checkpoint = directory / "selected.pt"
        if (
            not record.get("checks_passed")
            or record.get("measured_y_or_m_read")
            or record.get("screen_cases_read")
            or record.get("selected_sha256") != digest(checkpoint)
            or record.get("protocol_sha256") != digest(root / "protocol.json")
        ):
            raise RuntimeError(f"invalid training record: seed={seed}")
        trained = record["selected"]["tune"]
        current = record["same_seed_current"]["tune"]
        per_seed[str(seed)] = {
            "selected_epoch": record["best_epoch"],
            "stop_epoch": record["stop_epoch"],
            "trained_tune_bce": trained["mean_case_bce"],
            "current_tune_bce": current["mean_case_bce"],
            "delta_bce": trained["mean_case_bce"] - current["mean_case_bce"],
            "trained_tune_response_error": trained["response"]["error"],
            "current_tune_response_error": current["response"]["error"],
            "trained_tune_objective": trained["objective"],
            "current_tune_objective": current["objective"],
        }
        models.append({"path": str(checkpoint), "sha256": record["selected_sha256"]})
    result = {
        "analysis": "frozen tune-only review of the minimal response-moment classifier",
        "response_weight": RESPONSE_WEIGHT,
        "response_definition": "truth es1 response relative to the U oracle",
        "measured_y_or_m_read": False,
        "screen_cases_read": False,
        "per_seed": per_seed,
        "all_seeds_reduce_absolute_tune_response_error": all(
            abs(value["trained_tune_response_error"])
            < abs(value["current_tune_response_error"])
            for value in per_seed.values()
        ),
        "models": models,
        "protocol_sha256": digest(root / "protocol.json"),
        "limitations": [
            "Only the coherent g1 truth-response moment is constrained.",
            "The U oracle is a finite cached-simulation target, not ground-truth calibration.",
            "The response-gradient linearization is refreshed once per epoch.",
        ],
    }
    write_json(root / "review.json", result, preserve=True)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    print("MINIMAL_U_RESPONSE_REVIEW_PASSED", flush=True)


def actual_statistics(coherent, data: dict) -> np.ndarray:
    keep = data["U"].astype(bool) & data["C"][MOMENT_CUT].astype(bool)
    return np.asarray(
        [
            coherent.sufficient(keep, data["y"]),
            coherent.sufficient(keep, data["es"]),
            coherent.sufficient(keep, data["y"] - data["es"]),
        ]
    )


def describe_screen(coherent, actual_r: np.ndarray, model_r: np.ndarray) -> dict:
    denominator = model_r[:, 0, 0]
    m = 100.0 * (actual_r[:, 0, 0] / denominator - 1.0)
    return {
        "actual_R11": coherent.describe(actual_r[:, 0, 0]),
        "model_R11": coherent.describe(denominator),
        "m_percent": coherent.describe(m),
        "component_contribution_to_m_pp": {
            name: coherent.describe(
                -100.0 * (model_r[:, index, 0] - actual_r[:, index, 0]) / denominator
            )
            for index, name in ((1, "truth"), (2, "residual"))
        },
    }


def screen(args: argparse.Namespace) -> None:
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("the screen must run through Slurm")
    protocol = load_protocol(args.output_root)
    root = Path(args.output_root)
    base = load_module(Path(protocol["paths"]["base_script"]), "response_screen_base")
    coherent = load_module(
        Path(protocol["paths"]["moment_worker"]), "response_screen_worker"
    )
    base_protocol = load_base_protocol()
    coherent_config = coherent.verify()
    candidate_paths = []
    for seed in SEEDS:
        directory = root / "models" / f"seed{seed}"
        record = json.loads((directory / "completed.json").read_text())
        path = directory / "selected.pt"
        if record.get("selected_sha256") != digest(path):
            raise RuntimeError(f"candidate hash mismatch: seed={seed}")
        candidate_paths.append(path)
    candidates = [load_selection_model(path, device=args.device) for path in candidate_paths]
    currents = [
        load_selection_model(Path(path), device=args.device)
        for path in protocol["paths"]["current_models"]
    ]
    records = []
    for case in SCREEN_CASES:
        data, _, prepared_receipt, moments, moment_receipt = coherent.source_case(
            coherent_config, case, with_moments=True
        )
        ids, frames, usable, _ = base.verified_case(base_protocol, case)
        np.testing.assert_array_equal(ids, data["ids"])
        np.testing.assert_array_equal(usable, data["U"])
        probabilities = {}
        for name, bundles in (("candidate", candidates), ("current", currents)):
            members = [
                np.stack(
                    [bundle.predict_proba(frame, batch_size=65_536) for frame in frames]
                )
                for bundle in bundles
            ]
            probabilities[name] = np.mean(np.stack(members), axis=0)
        records.append(
            {
                "case": case,
                "actual": actual_statistics(coherent, data),
                "oracle": coherent.model_statistics(
                    data["U"],
                    moments["q"][MOMENT_CUT],
                    moments["u"][MOMENT_CUT],
                    data["es"],
                )[:3],
                "candidate": coherent.model_statistics(
                    probabilities["candidate"],
                    moments["q"][MOMENT_CUT],
                    moments["u"][MOMENT_CUT],
                    data["es"],
                )[:3],
                "current": coherent.model_statistics(
                    probabilities["current"],
                    moments["q"][MOMENT_CUT],
                    moments["u"][MOMENT_CUT],
                    data["es"],
                )[:3],
                "candidate_metrics": coherent.metric_sufficient(
                    data["U"], probabilities["candidate"]
                ),
                "current_metrics": coherent.metric_sufficient(
                    data["U"], probabilities["current"]
                ),
                "prepared_receipt_sha256": digest(prepared_receipt),
                "moment_receipt_sha256": digest(moment_receipt),
            }
        )
        print(f"MINIMAL_U_RESPONSE_SCREEN case={case}", flush=True)
    rng = np.random.default_rng(20260913)
    weights = np.vstack(
        [
            np.ones((1, len(SCREEN_CASES)), dtype=int),
            rng.multinomial(
                len(SCREEN_CASES),
                np.full(len(SCREEN_CASES), 1.0 / len(SCREEN_CASES)),
                10_000,
            ),
        ]
    )
    actual_r = coherent.responses(np.asarray([r["actual"] for r in records]), weights)
    results = {}
    for name in ("oracle", "current", "candidate"):
        model_r = coherent.responses(np.asarray([r[name] for r in records]), weights)
        results[name] = describe_screen(coherent, actual_r, model_r)
        if name != "oracle":
            metric = np.asarray([r[f"{name}_metrics"] for r in records]).sum(axis=(0, 1))
            results[name]["probability_metrics"] = {
                "n_leg_rows": int(metric[0]),
                "observed_usable_fraction": float(metric[1] / metric[0]),
                "predicted_usable_fraction": float(metric[2] / metric[0]),
                "signed_probability_gap": float((metric[2] - metric[1]) / metric[0]),
                "bce": float(metric[3] / metric[0]),
                "brier": float(metric[4] / metric[0]),
            }
    result = {
        "analysis": "one-shot reused-screen evaluation of minimal response-moment classifier",
        "development_only": True,
        "selection": protocol["selection"],
        "cases": list(SCREEN_CASES),
        "results": results,
        "candidate_minus_current_m_pp": (
            results["candidate"]["m_percent"]["value"]
            - results["current"]["m_percent"]["value"]
        ),
        "protocol_sha256": digest(root / "protocol.json"),
        "models": {
            "candidate": [
                {"path": str(path), "sha256": digest(path)} for path in candidate_paths
            ],
            "current": [
                {"path": str(path), "sha256": value}
                for path, value in zip(
                    protocol["paths"]["current_models"], CURRENT_MODEL_HASHES
                )
            ],
        },
        "limitations": [
            "Cases120--139 are repeatedly reused development data.",
            "The screen shares fixed flow and historical simulation lineage with training.",
            "Only coherent g1 +/-0.02 and the declared 0.75-arcsec selection are tested.",
        ],
    }
    write_json(root / "screen.json", result, preserve=True)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    print("MINIMAL_U_RESPONSE_SCREEN_PASSED", flush=True)


def self_test() -> None:
    torch.manual_seed(7)
    logits0 = torch.randn(23, dtype=torch.float64, requires_grad=True)
    logits1 = torch.randn(23, dtype=torch.float64, requires_grad=True)
    q0 = torch.rand(23, dtype=torch.float64)
    q1 = torch.rand(23, dtype=torch.float64)
    es0 = torch.randn(23, dtype=torch.float64)
    es1 = torch.randn(23, dtype=torch.float64)
    target = torch.tensor(0.91, dtype=torch.float64)
    data = {"q0": q0, "q1": q1, "es1_0": es0, "es1_1": es1}
    p0, p1 = logits0.sigmoid(), logits1.sigmoid()
    response = weighted_response(p0, p1, data)
    direct = RESPONSE_WEIGHT * (response - target).square()
    direct_gradient = torch.autograd.grad(direct, (logits0, logits1), retain_graph=True)
    with torch.no_grad():
        weights = (p0 * q0, p1 * q1)
        means = (
            (weights[0] * es0).sum() / weights[0].sum(),
            (weights[1] * es1).sum() / weights[1].sum(),
        )
        state = {
            "denominator_means": (weights[0].mean(), weights[1].mean()),
            "weighted_shape_means": means,
        }
        coefficient = 2.0 * RESPONSE_WEIGHT * float(response - target)
    indices = torch.arange(len(logits0))
    surrogate = coefficient * response_surrogate(p0, p1, indices, data, state)
    surrogate_gradient = torch.autograd.grad(surrogate, (logits0, logits1))
    for exact, approximate in zip(direct_gradient, surrogate_gradient):
        torch.testing.assert_close(exact, approximate, rtol=2e-12, atol=2e-12)
    model = SelectionMLP(
        input_dim=9, hidden_dim=64, n_layers=2, dropout=0.0, activation="silu"
    )
    assert model_parameter_count(model) == 4865
    print("MINIMAL_U_RESPONSE_SELF_TEST_PASSED", flush=True)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--output-root", required=True)
    prepare_parser.add_argument("--job-script", required=True)
    prepare_parser.add_argument("--review-job-script", required=True)
    prepare_parser.add_argument("--resource-report", required=True)
    train_parser = commands.add_parser("train")
    train_parser.add_argument("--output-root", required=True)
    train_parser.add_argument("--index", required=True, type=int, choices=range(3))
    preflight_parser = commands.add_parser("preflight")
    preflight_parser.add_argument("--output-root", required=True)
    review_parser = commands.add_parser("review")
    review_parser.add_argument("--output-root", required=True)
    screen_parser = commands.add_parser("screen")
    screen_parser.add_argument("--output-root", required=True)
    screen_parser.add_argument("--device", default="cpu")
    commands.add_parser("self-test")
    return result


def main() -> None:
    args = parser().parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "train":
        train(args)
    elif args.command == "preflight":
        preflight(args)
    elif args.command == "review":
        review(args)
    elif args.command == "screen":
        screen(args)
    else:
        self_test()


if __name__ == "__main__":
    main()
