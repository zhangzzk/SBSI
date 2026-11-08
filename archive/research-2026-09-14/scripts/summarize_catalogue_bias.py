#!/usr/bin/env python
"""Fit additive and multiplicative shear bias from catalogue profiles."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path

import numpy as np


def _hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def profile_summary(filename: str | Path, *, n_draws: int | None = None) -> dict:
    """Collapse proposal-seed replicates for one frozen mock profile."""

    path = Path(filename)
    payload = json.loads(path.read_text())
    arms = payload.get("profile") or []
    if not arms:
        raise ValueError(f"{path} contains no likelihood profiles")
    injected = {float(arm["injected_shear"]) for arm in arms}
    if len(injected) != 1:
        raise ValueError(f"{path} mixes injected directional shears")
    available = [set(int(rung["n_draws"]) for rung in arm["rungs"]) for arm in arms]
    common = set.intersection(*available)
    if not common:
        raise ValueError(f"{path} has no common draw rung across proposal seeds")
    selected_draws = max(common) if n_draws is None else int(n_draws)
    if selected_draws not in common:
        raise ValueError(f"{path} has no common M={selected_draws} rung")
    rungs = [
        next(rung for rung in arm["rungs"] if int(rung["n_draws"]) == selected_draws)
        for arm in arms
    ]
    if any(rung.get("quadratic_information") is None for rung in rungs):
        raise ValueError(f"{path} has a boundary/non-concave profile maximum")
    estimates = np.asarray([rung["estimated_shear"] for rung in rungs], dtype=float)
    information = np.asarray(
        [rung["quadratic_information"] for rung in rungs], dtype=float
    )
    if not np.isfinite(estimates).all() or not np.isfinite(information).all() or (
        information <= 0
    ).any():
        raise ValueError(f"{path} has invalid estimates or profile curvature")
    stat_se = float(np.mean(1.0 / np.sqrt(information)))
    mc_sd = float(np.std(estimates, ddof=1)) if len(estimates) > 1 else 0.0
    mc_se = mc_sd / math.sqrt(len(estimates))
    measurement = path.parent / "measurements.parquet"
    truth = path.parent / "truth.parquet"
    if not measurement.is_file() or not truth.is_file():
        raise ValueError(f"{path.parent} lacks its frozen mock files")
    direction = np.asarray(
        [payload["config"]["direction_g1"], payload["config"]["direction_g2"]],
        dtype=float,
    )
    direction /= np.linalg.norm(direction)
    selection = payload.get("selection")
    return {
        "result": str(path),
        "mock_sha256": {
            "measurements.parquet": _hash(measurement),
            "truth.parquet": _hash(truth),
        },
        "injected_shear": injected.pop(),
        "estimated_shear": float(estimates.mean()),
        "statistical_se": stat_se,
        "proposal_mc_sd": mc_sd,
        "proposal_mc_se": mc_se,
        "total_se": float(math.hypot(stat_se, mc_se)),
        "proposal_seeds": [int(arm["proposal_seed"]) for arm in arms],
        "n_draws": selected_draws,
        "n_detected": int(payload["config"]["n_detected"]),
        "mock_stream": {
            name: int(payload["config"][name])
            for name in ("scene_seed", "detection_seed", "flow_seed")
            if name in payload["config"]
        },
        "direction": direction.tolist(),
        "selection_key": None if selection is None else selection["cut_key"],
        "model_sha256": payload["model_sha256"],
    }


def fit_bias(summaries: list[dict]) -> dict:
    """Fit ``ghat = c + (1 + m) g`` with profile-curvature weights."""

    if len(summaries) < 3:
        raise ValueError("bias calibration needs at least three independent mocks")
    identities = {
        json.dumps(
            {
                "direction": row["direction"],
                "selection_key": row["selection_key"],
                "model_sha256": row["model_sha256"],
            },
            sort_keys=True,
        )
        for row in summaries
    }
    if len(identities) != 1:
        raise ValueError("profiles mix directions, selections, or model artifacts")
    hashes = [json.dumps(row["mock_sha256"], sort_keys=True) for row in summaries]
    if len(hashes) != len(set(hashes)):
        raise ValueError("the same frozen mock appears more than once")
    x = np.asarray([row["injected_shear"] for row in summaries], dtype=float)
    y = np.asarray([row["estimated_shear"] for row in summaries], dtype=float)
    sigma = np.asarray([row["total_se"] for row in summaries], dtype=float)
    if len(np.unique(x)) < 3:
        raise ValueError("bias calibration needs at least three injected shears")
    design = np.column_stack((np.ones(len(x)), x))
    precision = 1.0 / np.square(sigma)
    normal = design.T @ (precision[:, None] * design)
    covariance = np.linalg.inv(normal)
    coefficient = covariance @ design.T @ (precision * y)
    residual = y - design @ coefficient
    chi2 = float(np.sum(np.square(residual / sigma)))
    dof = len(x) - 2
    scale = max(1.0, chi2 / dof)
    robust_covariance = covariance * scale
    result = {
        "n_mocks": len(summaries),
        "n_shears": int(len(np.unique(x))),
        "additive_bias": float(coefficient[0]),
        "multiplicative_bias": float(coefficient[1] - 1.0),
        "additive_se": float(math.sqrt(robust_covariance[0, 0])),
        "multiplicative_se": float(math.sqrt(robust_covariance[1, 1])),
        "chi2": chi2,
        "dof": dof,
        "error_scale": scale,
        "mock_summaries": summaries,
    }
    streams: dict[str, list[dict]] = {}
    for row in summaries:
        stream = row.get("mock_stream") or {}
        if len(stream) == 3:
            streams.setdefault(json.dumps(stream, sort_keys=True), []).append(row)
    block_coefficients = []
    for stream, rows in sorted(streams.items()):
        bx = np.asarray([row["injected_shear"] for row in rows], dtype=float)
        if len(np.unique(bx)) < 3:
            continue
        by = np.asarray([row["estimated_shear"] for row in rows], dtype=float)
        bs = np.asarray([row["total_se"] for row in rows], dtype=float)
        bdesign = np.column_stack((np.ones(len(bx)), bx))
        bprecision = 1.0 / np.square(bs)
        bnormal = bdesign.T @ (bprecision[:, None] * bdesign)
        bcoefficient = np.linalg.inv(bnormal) @ bdesign.T @ (bprecision * by)
        block_coefficients.append(
            {
                "mock_stream": json.loads(stream),
                "additive_bias": float(bcoefficient[0]),
                "multiplicative_bias": float(bcoefficient[1] - 1.0),
            }
        )
    if len(block_coefficients) >= 2:
        block_c = np.asarray(
            [row["additive_bias"] for row in block_coefficients], dtype=float
        )
        block_m = np.asarray(
            [row["multiplicative_bias"] for row in block_coefficients], dtype=float
        )
        result["paired_blocks"] = {
            "n_blocks": len(block_coefficients),
            "additive_bias": float(block_c.mean()),
            "multiplicative_bias": float(block_m.mean()),
            "additive_se": float(block_c.std(ddof=1) / math.sqrt(len(block_c))),
            "multiplicative_se": float(block_m.std(ddof=1) / math.sqrt(len(block_m))),
            "blocks": block_coefficients,
        }
    pulls = np.asarray(
        [
            (row["estimated_shear"] - row["injected_shear"]) / row["total_se"]
            for row in summaries
        ],
        dtype=float,
    )
    result["coverage"] = {
        "mean_pull": float(pulls.mean()),
        "pull_std": float(pulls.std(ddof=1)),
        "within_1sigma": float(np.mean(np.abs(pulls) <= 1.0)),
        "within_2sigma": float(np.mean(np.abs(pulls) <= 2.0)),
    }
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+")
    parser.add_argument("--n-draws", type=int, default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)
    summaries = [profile_summary(path, n_draws=args.n_draws) for path in args.results]
    result = fit_bias(summaries)
    calibration = result.get("paired_blocks", result)
    print(
        f"c = {calibration['additive_bias']:+.6f} +/- "
        f"{calibration['additive_se']:.6f}\n"
        f"m = {calibration['multiplicative_bias']:+.4%} +/- "
        f"{calibration['multiplicative_se']:.4%}\n"
        f"chi2/dof = {result['chi2']:.2f}/{result['dof']}  "
        f"mocks={result['n_mocks']} shears={result['n_shears']}"
    )
    coverage = result["coverage"]
    print(
        f"pull mean/std = {coverage['mean_pull']:+.3f}/{coverage['pull_std']:.3f}; "
        f"coverage 1/2 sigma = {coverage['within_1sigma']:.1%}/"
        f"{coverage['within_2sigma']:.1%}"
    )
    if args.output is not None:
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
