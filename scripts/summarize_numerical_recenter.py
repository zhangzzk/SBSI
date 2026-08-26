#!/usr/bin/env python
"""Summarize paired numerical-recentering arms with simulation cases as blocks."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def _sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_recenter_summary(path: str | Path) -> dict:
    """Load one converged result and its frozen case provenance."""

    path = Path(path)
    payload = json.loads(path.read_text())
    result = payload["result"]
    if not result.get("converged"):
        raise ValueError(f"{path} did not converge: {result.get('reason')}")
    estimate = np.asarray(result["estimate"], dtype=float)
    injected = np.asarray(payload["injected_shear"], dtype=float)
    if estimate.shape != (2,) or injected.shape != (2,):
        raise ValueError(f"{path} does not contain two-component shears")
    iterations = result.get("iterations") or []
    if not iterations:
        raise ValueError(f"{path} has no final information matrix")
    information = np.asarray(iterations[-1]["information"], dtype=float)
    if information.shape != (2, 2) or np.linalg.eigvalsh(information)[0] <= 0:
        raise ValueError(f"{path} has non-positive final information")
    covariance = np.linalg.inv(information)
    importance = result.get("importance")
    required_importance = {
        "mean_ess",
        "median_ess",
        "p10_ess",
        "mean_ess_fraction",
        "p90_max_weight_fraction",
        "mean_outside_local_contribution",
        "mean_global_draw_contribution",
    }
    if not isinstance(importance, dict) or not required_importance <= set(importance):
        raise ValueError(f"{path} lacks final-shear importance diagnostics")
    elapsed_seconds = float(result["elapsed_seconds"])
    flow_evaluations = int(result["flow_evaluations"])
    if elapsed_seconds <= 0 or flow_evaluations <= 0:
        raise ValueError(f"{path} has invalid runtime/evaluation accounting")

    mock_root = path.parent / "mock"
    measurements_path = mock_root / "measurements.parquet"
    truth_path = mock_root / "truth.parquet"
    if not measurements_path.is_file() or not truth_path.is_file():
        raise ValueError(f"{path.parent} lacks the frozen analysis mock")
    actual_mock_hashes = {
        name: _sha256(mock_root / name) for name in ("measurements.parquet", "truth.parquet")
    }
    if payload.get("mock_sha256") != actual_mock_hashes:
        raise ValueError(f"{path.parent} frozen analysis mock hashes do not match")
    measurements = pd.read_parquet(measurements_path)
    truth = pd.read_parquet(truth_path)
    n_objects = int(result["n_objects"])
    if len(measurements) != n_objects or len(truth) != n_objects:
        raise ValueError(f"{path.parent} frozen analysis mock row count does not match n_objects")
    if "source_case" not in truth:
        raise ValueError(f"{truth_path} is not an image mock with source cases")
    cases = tuple(sorted(int(value) for value in truth["source_case"].unique()))
    if not cases:
        raise ValueError(f"{truth_path} contains no source cases")

    selection = payload.get("selection")
    r_blend = payload.get("r_blend")
    implementation = payload.get("implementation_sha256")
    if not isinstance(implementation, dict) or not implementation:
        raise ValueError(f"{path} lacks implementation hashes")
    proposal_cache = payload.get("proposal_cache_sha256")
    if not isinstance(proposal_cache, dict) or not proposal_cache:
        raise ValueError(f"{path} lacks proposal-cache hashes")
    model_cache = payload.get("model_cache_sha256")
    if not isinstance(model_cache, dict) or not model_cache:
        raise ValueError(f"{path} lacks model-cache hashes")
    config = payload.get("config")
    if not isinstance(config, dict):
        raise ValueError(f"{path} lacks run configuration")
    proposal_method = result.get(
        "proposal_method",
        config.get("proposal_method", "distance_kernel"),
    )
    if proposal_method not in {
        "initial_center_posterior_adapted",
        "distance_kernel",
    }:
        raise ValueError(f"{path} has an unsupported proposal method")
    proposal_reference = result.get("proposal_reference_shear")
    initial_likelihood_reused = bool(
        result.get("initial_likelihood_reused", False)
    )
    proposal_candidate_evaluations = int(
        result.get("proposal_candidate_flow_evaluations", 0)
    )
    proposal_reuse_evaluations = int(
        result.get("proposal_reuse_flow_evaluations", 0)
    )
    proposal_flow_evaluations = int(
        result.get("proposal_flow_evaluations", 0)
    )
    if proposal_flow_evaluations != (
        proposal_candidate_evaluations + proposal_reuse_evaluations
    ):
        raise ValueError(f"{path} has inconsistent proposal evaluation counts")
    if proposal_method == "initial_center_posterior_adapted":
        if proposal_reference is None or proposal_candidate_evaluations <= 0:
            raise ValueError(f"{path} lacks posterior-adapted proposal provenance")
    identity = {
        "model_sha256": payload.get("model_sha256"),
        "model_cache_sha256": model_cache,
        "scene_sha256": payload.get("scene_sha256"),
        "r_blend_cache_sha256": (r_blend.get("cache_sha256") if isinstance(r_blend, dict) else r_blend),
        "selection": (
            None
            if selection is None
            else {
                "cut_key": selection["cut_key"],
                "n_samples": selection["n_samples"],
                "seed": selection["seed"],
                "row_chunk": selection["row_chunk"],
                "cache_sha256": selection.get("cache_sha256"),
            }
        ),
        "proposal_cache_sha256": proposal_cache,
        "proposal_coordinate_config": {
            name: config.get(name)
            for name in (
                "proposal_flow_samples",
                "proposal_statistic",
                "proposal_row_chunk",
                "proposal_coordinate_seed",
            )
        },
        "optimizer_controls": {
            name: config.get(name)
            for name in (
                "initial",
                "max_iterations",
                "tolerance",
                "max_step",
                "shear_bound",
                "max_backtracks",
            )
        },
        "h": result["h"],
        "n_draws": result["n_draws"],
        "n_candidates": result["n_candidates"],
        "proposal_seed": result["proposal_seed"],
        "proposal_method": proposal_method,
        "proposal_reference_shear": proposal_reference,
        "initial_likelihood_reused": initial_likelihood_reused,
        "epsilon": result["epsilon"],
        "bandwidth": result["bandwidth"],
        "implementation_sha256": implementation,
    }
    return {
        "path": str(path),
        "cases": cases,
        "n_objects": n_objects,
        "injected": injected,
        "estimate": estimate,
        "covariance": covariance,
        "importance": importance,
        "elapsed_seconds": elapsed_seconds,
        "flow_evaluations": flow_evaluations,
        "proposal_flow_evaluations": proposal_flow_evaluations,
        "identity": identity,
        "mock_sha256": actual_mock_hashes,
    }


def summarize_paired_blocks(rows, *, injected_component: int = 0) -> dict:
    """Form one ± response per case block, then average blocks equally."""

    rows = list(rows)
    if not rows:
        raise ValueError("at least one recenter result is required")
    if injected_component not in (0, 1):
        raise ValueError("injected_component must be 0 or 1")
    identities = {json.dumps(row["identity"], sort_keys=True) for row in rows}
    if len(identities) != 1:
        raise ValueError("results mix models, selection, stencil, or sampler settings")
    mock_hashes = [json.dumps(row["mock_sha256"], sort_keys=True) for row in rows]
    if len(mock_hashes) != len(set(mock_hashes)):
        raise ValueError("the same frozen analysis mock appears more than once")

    grouped = {}
    for row in rows:
        injection = row["injected"]
        other = 1 - injected_component
        if abs(injection[other]) > 1e-14 or injection[injected_component] == 0:
            raise ValueError("each result must be a nonzero axis injection")
        sign = int(np.sign(injection[injected_component]))
        key = tuple(row["cases"])
        if sign in grouped.setdefault(key, {}):
            raise ValueError(f"case block {key} has duplicate shear arms")
        grouped[key][sign] = row

    case_owner = {}
    for cases in grouped:
        for case in cases:
            previous = case_owner.setdefault(case, cases)
            if previous != cases:
                raise ValueError(f"source case {case} appears in overlapping blocks {previous} and {cases}")

    blocks = []
    for cases, arms in sorted(grouped.items()):
        if set(arms) != {-1, 1}:
            raise ValueError(f"case block {cases} lacks a matched +/- pair")
        negative, positive = arms[-1], arms[1]
        amplitude = float(positive["injected"][injected_component])
        if not np.isclose(negative["injected"][injected_component], -amplitude, rtol=0, atol=1e-14):
            raise ValueError(f"case block {cases} has unequal injection amplitudes")
        slope = (positive["estimate"] - negative["estimate"]) / (2.0 * amplitude)
        intercept = 0.5 * (positive["estimate"] + negative["estimate"])
        slope_covariance = (positive["covariance"] + negative["covariance"]) / (4.0 * amplitude**2)
        intercept_covariance = 0.25 * (positive["covariance"] + negative["covariance"])
        other = 1 - injected_component
        blocks.append(
            {
                "cases": list(cases),
                "amplitude": amplitude,
                "n_objects_negative": negative["n_objects"],
                "n_objects_positive": positive["n_objects"],
                "importance_negative": negative["importance"],
                "importance_positive": positive["importance"],
                "elapsed_seconds_negative": negative["elapsed_seconds"],
                "elapsed_seconds_positive": positive["elapsed_seconds"],
                "flow_evaluations_negative": negative["flow_evaluations"],
                "flow_evaluations_positive": positive["flow_evaluations"],
                "proposal_flow_evaluations_negative": negative.get(
                    "proposal_flow_evaluations", 0
                ),
                "proposal_flow_evaluations_positive": positive.get(
                    "proposal_flow_evaluations", 0
                ),
                "additive_bias": float(intercept[injected_component]),
                "multiplicative_bias": float(slope[injected_component] - 1.0),
                "cross_additive": float(intercept[other]),
                "cross_response": float(slope[other]),
                "additive_variance": float(intercept_covariance[injected_component, injected_component]),
                "multiplicative_variance": float(slope_covariance[injected_component, injected_component]),
                "cross_additive_variance": float(intercept_covariance[other, other]),
                "cross_response_variance": float(slope_covariance[other, other]),
            }
        )

    def aggregate(name, variance_name):
        values = np.asarray([block[name] for block in blocks], dtype=float)
        curvature_se = math.sqrt(sum(block[variance_name] for block in blocks)) / len(blocks)
        case_se = float(values.std(ddof=1) / math.sqrt(len(values))) if len(values) >= 2 else None
        return {
            "estimate": float(values.mean()),
            "case_block_se": case_se,
            "curvature_se": float(curvature_se),
            "conservative_se": float(curvature_se if case_se is None else max(case_se, curvature_se)),
        }

    arm_importance = [
        block[name]
        for block in blocks
        for name in ("importance_negative", "importance_positive")
    ]

    return {
        "injected_component": f"g{injected_component + 1}",
        "n_case_blocks": len(blocks),
        "additive_bias": aggregate("additive_bias", "additive_variance"),
        "multiplicative_bias": aggregate("multiplicative_bias", "multiplicative_variance"),
        "cross_additive": aggregate("cross_additive", "cross_additive_variance"),
        "cross_response": aggregate("cross_response", "cross_response_variance"),
        "sampler_diagnostics": {
            "min_mean_ess": float(
                min(item["mean_ess"] for item in arm_importance)
            ),
            "min_p10_ess": float(
                min(item["p10_ess"] for item in arm_importance)
            ),
            "min_mean_ess_fraction": float(
                min(item["mean_ess_fraction"] for item in arm_importance)
            ),
            "max_p90_max_weight_fraction": float(
                max(item["p90_max_weight_fraction"] for item in arm_importance)
            ),
            "total_elapsed_seconds": float(
                sum(
                    block["elapsed_seconds_negative"]
                    + block["elapsed_seconds_positive"]
                    for block in blocks
                )
            ),
            "total_flow_evaluations": int(
                sum(
                    block["flow_evaluations_negative"]
                    + block["flow_evaluations_positive"]
                    for block in blocks
                )
            ),
            "total_proposal_flow_evaluations": int(
                sum(
                    block["proposal_flow_evaluations_negative"]
                    + block["proposal_flow_evaluations_positive"]
                    for block in blocks
                )
            ),
        },
        "identity": rows[0]["identity"],
        "blocks": blocks,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+")
    parser.add_argument("--injected-component", choices=("g1", "g2"), default="g1")
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)
    rows = [load_recenter_summary(path) for path in args.results]
    summary = summarize_paired_blocks(
        rows,
        injected_component=0 if args.injected_component == "g1" else 1,
    )
    for name in ("additive_bias", "multiplicative_bias", "cross_response"):
        item = summary[name]
        print(f"{name}: {item['estimate']:+.8f} +/- {item['conservative_se']:.8f}")
    print(f"case blocks: {summary['n_case_blocks']}")
    if args.output is not None:
        Path(args.output).write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
