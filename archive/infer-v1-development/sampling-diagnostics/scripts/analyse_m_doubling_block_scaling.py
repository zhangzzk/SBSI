#!/usr/bin/env python
# Archived Infer V1 sampling diagnostic.
"""Measure how a paired nested-M shift scales across disjoint object blocks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Completed K x M campaign root")
    parser.add_argument("--block-size", type=int, default=2000)
    parser.add_argument("--lower-m", type=int, default=32768)
    parser.add_argument("--upper-m", type=int, default=65536)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _estimate(center, score, information):
    score_sum = score.sum(axis=0, dtype=np.float64)
    information_sum = information.sum(axis=0, dtype=np.float64)
    information_sum = 0.5 * (information_sum + information_sum.T)
    eigenvalues = np.linalg.eigvalsh(information_sum)
    if not np.isfinite(eigenvalues).all() or eigenvalues[0] <= 0:
        raise RuntimeError(f"non-positive block information: {eigenvalues.tolist()}")
    return center + np.linalg.solve(information_sum, score_sum)


def _analyse_arm(arm_root: Path, block_size: int, lower_m: int, upper_m: int):
    result_path = arm_root / "result.json"
    payload = json.loads(result_path.read_text())
    moment_path = arm_root / payload["one_step_moments"]["path"]
    if _sha256(moment_path) != payload["one_step_moments"]["sha256"]:
        raise RuntimeError(f"moment hash mismatch in {arm_root}")

    ladder = tuple(int(value) for value in payload["config"]["adaptive_draw_ladder"])
    try:
        lower_index = ladder.index(lower_m)
        upper_index = ladder.index(upper_m)
    except ValueError as error:
        raise RuntimeError(f"requested M pair absent from {arm_root}") from error
    if lower_index >= upper_index:
        raise ValueError("lower M must precede upper M")

    center = np.asarray(payload["initial_center"]["center"], dtype=np.float64)
    with np.load(moment_path) as arrays:
        score = np.asarray(arrays["ladder_score"][[lower_index, upper_index]])
        information = np.asarray(
            arrays["ladder_information"][[lower_index, upper_index]]
        )
    n_objects = score.shape[1]
    if block_size <= 0 or n_objects % block_size:
        raise ValueError(f"block size {block_size} must divide {n_objects}")
    n_blocks = n_objects // block_size

    full_estimates = np.asarray(
        [_estimate(center, score[index], information[index]) for index in range(2)]
    )
    recorded = np.asarray(
        [payload["full_ladder"][str(m)]["estimate"] for m in (lower_m, upper_m)]
    )
    if not np.allclose(full_estimates, recorded, rtol=0.0, atol=1.0e-12):
        raise RuntimeError(f"full-sample estimate mismatch in {arm_root}")

    block_deltas = []
    for block_index in range(n_blocks):
        start = block_index * block_size
        stop = start + block_size
        estimates = np.asarray(
            [
                _estimate(
                    center,
                    score[index, start:stop],
                    information[index, start:stop],
                )
                for index in range(2)
            ]
        )
        block_deltas.append(estimates[1] - estimates[0])
    block_deltas = np.asarray(block_deltas)

    full_delta = full_estimates[1] - full_estimates[0]
    full_influence = []
    for index in range(2):
        information_sum = information[index].sum(axis=0, dtype=np.float64)
        information_sum = 0.5 * (information_sum + information_sum.T)
        step = full_estimates[index] - center
        residual = score[index] - np.einsum(
            "nij,j->ni", information[index], step
        )
        full_influence.append(
            np.linalg.solve(information_sum / n_objects, residual.T).T
        )
    influence_delta = full_influence[1] - full_influence[0]
    linearized_block_deltas = np.asarray(
        [
            full_delta
            + influence_delta[start : start + block_size].mean(axis=0)
            for start in range(0, n_objects, block_size)
        ]
    )
    block_mean = block_deltas.mean(axis=0)
    block_sd = block_deltas.std(axis=0, ddof=1)
    block_rms = np.sqrt(np.mean(np.square(block_deltas), axis=0))
    predicted_full_se = block_sd / np.sqrt(n_blocks)
    stored_full_se = np.asarray(
        payload["full_ladder"][str(upper_m)]["previous_rung_paired_standard_error"],
        dtype=np.float64,
    )
    expected_block_sd = stored_full_se * np.sqrt(n_objects / block_size)

    config = payload["config"]
    return {
        "arm": arm_root.name,
        "proposal_candidates": int(config["proposal_candidates"]),
        "proposal_seed": int(config["proposal_seed"]),
        "n_objects": n_objects,
        "block_size": block_size,
        "n_disjoint_blocks": n_blocks,
        "lower_m": lower_m,
        "upper_m": upper_m,
        "full_delta_g": full_delta.tolist(),
        "stored_full_paired_se": stored_full_se.tolist(),
        "full_delta_pull": np.divide(full_delta, stored_full_se).tolist(),
        "block_delta_g": block_deltas.tolist(),
        "block_mean_delta_g": block_mean.tolist(),
        "block_sd_delta_g": block_sd.tolist(),
        "block_rms_delta_g": block_rms.tolist(),
        "block_median_abs_delta_g": np.median(
            np.abs(block_deltas), axis=0
        ).tolist(),
        "predicted_full_se_from_blocks": predicted_full_se.tolist(),
        "expected_block_sd_from_full_paired_se": expected_block_sd.tolist(),
        "observed_to_expected_block_sd_ratio": np.divide(
            block_sd, expected_block_sd
        ).tolist(),
        "linearized_block_delta_g": linearized_block_deltas.tolist(),
        "linearized_block_sd_delta_g": linearized_block_deltas.std(
            axis=0, ddof=1
        ).tolist(),
        "linearized_to_expected_block_sd_ratio": np.divide(
            linearized_block_deltas.std(axis=0, ddof=1), expected_block_sd
        ).tolist(),
        "positive_block_counts": np.sum(block_deltas > 0, axis=0).tolist(),
    }


def main(argv=None):
    args = parse_args(argv)
    root = Path(args.root).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}")
    arms = sorted((root / "results").glob("*/result.json"))
    if not arms:
        raise RuntimeError(f"no completed arms under {root / 'results'}")
    reports = [
        _analyse_arm(path.parent, args.block_size, args.lower_m, args.upper_m)
        for path in arms
    ]
    payload = {
        "status": "complete",
        "method": "exact_disjoint_block_nested_m_scaling_v2",
        "interpretation": (
            "Each block re-sums retained per-object score and information; no "
            "likelihood evaluations are rerun. Exact block estimates can be "
            "nonlinear for influential small blocks, so full-sample influence-"
            "linearized block deltas are also reported for the 1/sqrt(N) check. "
            "Blocks are descriptive replicates, and arms sharing a proposal seed "
            "are correlated."
        ),
        "arms": reports,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
