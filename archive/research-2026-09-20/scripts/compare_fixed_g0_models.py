#!/usr/bin/env python3
"""Compare flow checkpoints on one fixed-g0 validation cohort.

Every checkpoint is evaluated on the same NLL rows and matched response pairs.
The response calculation uses common antithetic latent draws across models.
Pairwise differences include case-level standard errors, avoiding an
independent-model comparison of highly correlated scores.
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
import pandas as pd
import torch

from sbsi.fixed_g0_domain import FLOW_FEATURES, FLOW_TARGETS
from sbsi.flow_paired_shape import PairedResponsePopulation, evaluate_paired_response
from sbsi.measurement_model import load_measurement_model
from scripts.train_fixed_g0_flow import load_split


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


def parse_models(values: list[str]) -> dict[str, Path]:
    models = {}
    for value in values:
        label, separator, path = value.partition("=")
        if not separator or not label or not path:
            raise ValueError("each --model must have the form LABEL=PATH")
        if label in models:
            raise ValueError(f"duplicate model label: {label}")
        models[label] = Path(path).resolve()
    if len(models) < 2:
        raise ValueError("at least two models are required")
    missing = [str(path) for path in models.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing model checkpoints: {missing}")
    return models


def pair_component_scores(group_residuals: np.ndarray) -> np.ndarray:
    """Return cross-replica response scores for every pair and component."""

    residuals = np.asarray(group_residuals, dtype=np.float64)
    if (
        residuals.ndim != 3
        or residuals.shape[0] < 2
        or residuals.shape[1] < 1
        or residuals.shape[2] != 3
        or not np.isfinite(residuals).all()
    ):
        raise ValueError("finite [group,pair,3] residuals with >=2 groups required")
    groups = len(residuals)
    return (
        residuals.sum(0) ** 2 - (residuals**2).sum(0)
    ) / (4.0 * groups * (groups - 1))


def case_summary(values: np.ndarray, cases: np.ndarray) -> dict:
    """Summarize a paired difference by pooled objects and independent cases."""

    values = np.asarray(values, dtype=np.float64)
    cases = np.asarray(cases)
    if values.ndim != 1 or cases.shape != values.shape or not np.isfinite(values).all():
        raise ValueError("finite aligned one-dimensional values and case labels required")
    unique = np.unique(cases)
    case_means = np.asarray([values[cases == case].mean() for case in unique])
    return {
        "pooled_mean": float(values.mean()),
        "case_balanced_mean": float(case_means.mean()),
        "case_sem": float(case_means.std(ddof=1) / np.sqrt(len(case_means))),
        "n_cases": int(len(unique)),
        "n_rows": int(len(values)),
    }


def response_mc_difference_se(
    left: np.ndarray, right: np.ndarray
) -> dict[str, float]:
    """Paired latent-group jackknife SEs for model response-score differences."""

    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 3 or left.shape[0] < 4:
        raise ValueError("aligned model residual groups required")
    deleted = []
    for group in range(len(left)):
        left_score = pair_component_scores(np.delete(left, group, axis=0)).mean(0)
        right_score = pair_component_scores(np.delete(right, group, axis=0)).mean(0)
        deleted.append(left_score - right_score)
    deleted = np.asarray(deleted)
    def jackknife_se(values: np.ndarray) -> float:
        return float(
            np.sqrt(
                (len(values) - 1)
                / len(values)
                * np.sum((values - values.mean()) ** 2)
            )
        )

    return {
        "total": jackknife_se(deleted.sum(1)),
        "shape": jackknife_se(deleted[:, :2].sum(1)),
        "radius": jackknife_se(deleted[:, 2]),
    }


def target_baseline(population: PairedResponsePopulation, indices: np.ndarray) -> dict:
    rows = torch.as_tensor(indices, dtype=torch.long, device=population.context.device)
    amplitude = population.amplitude[rows].cpu().numpy()
    delta = population.measured_delta[rows].cpu().numpy()
    scaled = delta / amplitude[:, None] / population.output_scale.cpu().numpy()
    contributions = scaled**2 / 4.0
    component = contributions.mean(0)
    tails = {}
    for column, name in enumerate(("shape1", "shape2", "radius")):
        ordered = np.sort(contributions[:, column])
        total = ordered.sum()
        tails[name] = {
            "mean": float(component[column]),
            "median": float(np.median(ordered)),
            "p99": float(np.quantile(ordered, 0.99)),
            "maximum": float(ordered[-1]),
            "top_1pct_fraction": float(
                ordered[-max(1, len(ordered) // 100) :].sum() / total
            ),
            "top_0p1pct_fraction": float(
                ordered[-max(1, len(ordered) // 1000) :].sum() / total
            ),
        }
    return {
        "zero_model_component_losses": component,
        "zero_model_response_loss": float(component.sum()),
        "per_component_target_loss_distribution": tails,
    }


def expanded_case_labels(data: dict) -> tuple[np.ndarray, np.ndarray]:
    row_cases = []
    pair_cases = []
    for counts in data["per_case_matching"]:
        case = int(counts["case"])
        row_cases.append(
            np.full(int(counts["left_rows"] + counts["right_rows"]), case, dtype=np.int16)
        )
        pair_cases.append(np.full(int(counts["response_pairs"]), case, dtype=np.int16))
    rows = np.concatenate(row_cases)
    pairs = np.concatenate(pair_cases)
    if len(rows) != len(data["context_raw"]) or len(pairs) != len(data["pairs"]):
        raise RuntimeError("case-label reconstruction does not match prepared arrays")
    return rows, pairs


def compare(args: argparse.Namespace) -> None:
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("fixed-g0 model comparison must run under Slurm")
    models = parse_models(args.model)
    output = args.output.resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"preserving existing result: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    progress_output = output.with_suffix(".progress.json")
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    device = torch.device(args.device)

    domain_root = args.domain_root.resolve()
    manifest_path = domain_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["population"]["truth_analysis_cut"] is not None:
        raise ValueError("comparison requires the no-truth-cut fixed-g0 domain")
    if manifest["population"]["sheared_leg_recut"]:
        raise ValueError("comparison refuses a sheared-leg selection recut")
    print("LOADING_COMMON_FIXED_G0_VALIDATION", flush=True)
    data = load_split(domain_root, manifest["split"]["validation_cases"])
    row_cases, pair_cases = expanded_case_labels(data)
    nll_indices = np.load(args.nll_indices)
    response_indices = np.load(args.response_indices)
    if (
        nll_indices.ndim != 1
        or response_indices.ndim != 1
        or np.any(nll_indices < 0)
        or np.any(nll_indices >= len(data["target_raw"]))
        or np.any(response_indices < 0)
        or np.any(response_indices >= len(data["pairs"]))
    ):
        raise ValueError("invalid common validation indices")
    context_frame = pd.DataFrame(data["context_raw"], columns=FLOW_FEATURES)
    nll_frame = pd.DataFrame(
        np.column_stack(
            (data["context_raw"][nll_indices], data["target_raw"][nll_indices])
        ),
        columns=FLOW_FEATURES + FLOW_TARGETS,
    )
    selected_nll_cases = row_cases[nll_indices]
    selected_pair_cases = pair_cases[response_indices]

    baseline_population = PairedResponsePopulation(
        torch.zeros((len(data["context_raw"]), 1), dtype=torch.float32),
        data["pairs"],
        data["measured_delta"],
        data["gamma"],
        radius_scale=1.0,
    )
    result = {
        "format_version": 1,
        "comparison_domain": "corrected fixed-g0 validation cases and rows",
        "domain_manifest": str(manifest_path),
        "domain_manifest_sha256": sha256(manifest_path),
        "nll_indices": str(args.nll_indices.resolve()),
        "nll_indices_sha256": sha256(args.nll_indices.resolve()),
        "response_indices": str(args.response_indices.resolve()),
        "response_indices_sha256": sha256(args.response_indices.resolve()),
        "nll_rows": int(len(nll_indices)),
        "response_pairs": int(len(response_indices)),
        "response_draws": int(args.response_draws),
        "response_groups": int(args.response_groups),
        "response_seed": int(args.response_seed),
        "response_crn_across_models": True,
        "target_baseline": target_baseline(baseline_population, response_indices),
        "models": {},
        "pairwise_differences": {},
    }
    del baseline_population

    raw_nll_rows = {}
    response_groups = {}
    response_component = {}
    for label, path in models.items():
        started = time.monotonic()
        print(f"SCORING_MODEL {label}", flush=True)
        bundle = load_measurement_model(str(path), device=device)
        if bundle.condition_preprocessor.feature_names != list(FLOW_FEATURES):
            raise ValueError(f"feature mismatch for {label}")
        if bundle.target_transform.target_names != list(FLOW_TARGETS):
            raise ValueError(f"target mismatch for {label}")
        standardized_log_probability = bundle.log_prob(
            nll_frame, batch_size=args.nll_batch_size
        )
        standardized_nll = float(-standardized_log_probability.mean())
        log_scale = float(np.log(bundle.target_transform.scales).sum())
        raw_nll = -standardized_log_probability + log_scale
        context = bundle.context_tensor(context_frame)
        population = PairedResponsePopulation(
            context,
            data["pairs"],
            data["measured_delta"],
            data["gamma"],
            radius_scale=1.0,
        )
        response = evaluate_paired_response(
            bundle.model,
            population,
            response_indices,
            draws=args.response_draws,
            groups=args.response_groups,
            batch_size=args.response_batch_size,
            seed=args.response_seed,
        )
        raw_nll_rows[label] = raw_nll
        response_groups[label] = response["group_residuals"]
        response_component[label] = pair_component_scores(response["group_residuals"])
        result["models"][label] = {
            "path": str(path),
            "sha256": sha256(path),
            "epoch": bundle.metadata.get("epoch"),
            "phase_epoch": bundle.metadata.get("phase_epoch"),
            "training_phase": bundle.metadata.get("training_phase"),
            "checkpoint_validation_nll": bundle.metadata.get("validation_nll"),
            "checkpoint_response_loss": bundle.metadata.get("paired_response_loss"),
            "common_standardized_nll": standardized_nll,
            "common_raw_coordinate_nll": float(raw_nll.mean()),
            "target_scales": bundle.target_transform.scales,
            "common_response": {
                key: value
                for key, value in response.items()
                if key not in ("group_residuals", "per_pair")
            },
            "elapsed_seconds": time.monotonic() - started,
        }
        write_json(progress_output, result)
        del bundle, context, population, response
        if device.type == "cuda":
            torch.cuda.empty_cache()

    labels = list(models)
    for left_index, left in enumerate(labels):
        for right in labels[:left_index]:
            key = f"{left}_minus_{right}"
            component_difference = response_component[left] - response_component[right]
            result["pairwise_differences"][key] = {
                "interpretation": "negative favors the left model",
                "raw_coordinate_nll": case_summary(
                    raw_nll_rows[left] - raw_nll_rows[right], selected_nll_cases
                ),
                "response_total": case_summary(
                    component_difference.sum(1), selected_pair_cases
                ),
                "response_shape": case_summary(
                    component_difference[:, :2].sum(1), selected_pair_cases
                ),
                "response_radius": case_summary(
                    component_difference[:, 2], selected_pair_cases
                ),
                "response_paired_mc_jackknife_se": response_mc_difference_se(
                    response_groups[left], response_groups[right]
                ),
            }
    write_json(output, result)
    progress_output.unlink(missing_ok=True)
    print(f"COMMON_FIXED_G0_COMPARISON_COMPLETE output={output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain-root", type=Path, required=True)
    parser.add_argument("--nll-indices", type=Path, required=True)
    parser.add_argument("--response-indices", type=Path, required=True)
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--nll-batch-size", type=int, default=8192)
    parser.add_argument("--response-draws", type=int, default=64)
    parser.add_argument("--response-groups", type=int, default=4)
    parser.add_argument("--response-batch-size", type=int, default=1024)
    parser.add_argument("--response-seed", type=int, default=3260000000001)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if (
        args.nll_batch_size < 1
        or args.response_batch_size < 1
        or args.response_draws < 2
        or args.response_draws % 2
        or args.response_groups < 4
    ):
        parser.error("positive batches, even draws >=2, and groups >=4 required")
    compare(args)


if __name__ == "__main__":
    main()
