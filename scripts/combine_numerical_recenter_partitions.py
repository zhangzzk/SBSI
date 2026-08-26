#!/usr/bin/env python
"""Combine disjoint one-step partitions by summing per-object moments exactly."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np


def _sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _summarize(center, score, information):
    center = np.asarray(center, dtype=np.float64)
    score_sum = score.sum(axis=0, dtype=np.float64)
    information_sum = information.sum(axis=0, dtype=np.float64)
    information_sum = 0.5 * (information_sum + information_sum.T)
    eigenvalues = np.linalg.eigvalsh(information_sum)
    if not np.isfinite(eigenvalues).all() or eigenvalues[0] <= 0:
        raise RuntimeError(f"non-positive combined information: {eigenvalues.tolist()}")
    step = np.linalg.solve(information_sum, score_sum)
    estimate = center + step
    residual = score - np.einsum("nij,j->ni", information, step)
    influence = np.linalg.solve(information_sum / len(score), residual.T).T
    robust_covariance = np.cov(influence, rowvar=False, ddof=1) / len(score)
    model_covariance = np.linalg.inv(information_sum)
    return {
        "center": center.tolist(),
        "estimate": estimate.tolist(),
        "step": step.tolist(),
        "score_sum": score_sum.tolist(),
        "information_sum": information_sum.tolist(),
        "information_eigenvalues": eigenvalues.tolist(),
        "robust_covariance": robust_covariance.tolist(),
        "model_covariance": model_covariance.tolist(),
        "robust_standard_error": np.sqrt(np.diag(robust_covariance)).tolist(),
        "model_standard_error": np.sqrt(np.diag(model_covariance)).tolist(),
        "quadratic_log_likelihood_gain": 0.5 * float(score_sum @ step),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", action="append", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    output = Path(args.output).resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    parts = []
    for root_value in args.part:
        root = Path(root_value).resolve()
        payload = json.loads((root / "result.json").read_text())
        moment_path = root / payload["one_step_moments"]["path"]
        if _sha256(moment_path) != payload["one_step_moments"]["sha256"]:
            raise RuntimeError(f"moment hash mismatch in {root}")
        arrays = np.load(moment_path)
        parts.append((payload["observation_partition"]["start"], root, payload, arrays))
    parts.sort(key=lambda item: item[0])
    reference = parts[0][2]
    identity_names = (
        "initial_center",
        "injected_shear",
        "model_sha256",
        "model_cache_sha256",
        "scene_sha256",
        "proposal_cache_sha256",
        "mock_input_sha256",
        "implementation_sha256",
    )
    expected_start = 0
    score_parts = []
    information_parts = []
    draw_parts = []
    unique_parts = []
    ladder_score_parts = []
    ladder_information_parts = []
    partition_reports = []
    for start, root, payload, arrays in parts:
        partition = payload["observation_partition"]
        if start != expected_start or partition["start"] != start:
            raise RuntimeError("observation partitions are not contiguous from zero")
        expected_start = int(partition["stop"])
        if partition["n_partition"] != len(arrays["score"]):
            raise RuntimeError(f"partition row count mismatch in {root}")
        for name in identity_names:
            if payload[name] != reference[name]:
                raise RuntimeError(f"partition identity mismatch for {name}: {root}")
        score_parts.append(arrays["score"])
        information_parts.append(arrays["information"])
        draw_parts.append(arrays["draw_counts"])
        unique_parts.append(arrays["unique_counts"])
        if "ladder_score" in arrays:
            ladder_score_parts.append(arrays["ladder_score"])
            ladder_information_parts.append(arrays["ladder_information"])
        partition_reports.append({
            "root": str(root),
            "start": start,
            "stop": expected_start,
            "result_sha256": _sha256(root / "result.json"),
            "moments_sha256": _sha256(root / "one_step_moments.npz"),
            "elapsed_seconds": payload["result"]["elapsed_seconds"],
        })
    if expected_start != reference["observation_partition"]["n_total"]:
        raise RuntimeError("partitions do not cover the complete source mock")

    score = np.concatenate(score_parts, axis=0)
    information = np.concatenate(information_parts, axis=0)
    draw_counts = np.concatenate(draw_parts)
    unique_counts = np.concatenate(unique_parts)
    arrays_out = {
        "score": score,
        "information": information,
        "draw_counts": draw_counts,
        "unique_counts": unique_counts,
    }
    ladder_report = None
    if ladder_score_parts:
        if len(ladder_score_parts) != len(parts):
            raise RuntimeError("only some partitions retain a full ladder")
        ladder_score = np.concatenate(ladder_score_parts, axis=1)
        ladder_information = np.concatenate(ladder_information_parts, axis=1)
        arrays_out["ladder_score"] = ladder_score
        arrays_out["ladder_information"] = ladder_information
        ladder = reference["config"]["adaptive_draw_ladder"]
        ladder_report = {
            str(n_draws): _summarize(
                reference["initial_center"]["center"],
                ladder_score[index],
                ladder_information[index],
            )
            for index, n_draws in enumerate(ladder)
        }
    moment_path = output / "one_step_moments.npz"
    np.savez_compressed(moment_path, **arrays_out)
    summary = _summarize(reference["initial_center"]["center"], score, information)
    injected = np.asarray(reference["injected_shear"], dtype=float)
    report = {
        "status": "complete",
        "method": "exact_sum_of_partitioned_one_step_score_information",
        "n_observations": int(len(score)),
        "injected_shear": injected.tolist(),
        "difference": (np.asarray(summary["estimate"]) - injected).tolist(),
        "summary": summary,
        "full_ladder": ladder_report,
        "draw_counts": {
            "min": int(draw_counts.min()),
            "median": float(np.median(draw_counts)),
            "max": int(draw_counts.max()),
        },
        "unique_counts": {
            "min": int(unique_counts.min()),
            "median": float(np.median(unique_counts)),
            "max": int(unique_counts.max()),
        },
        "partitions": partition_reports,
        "identity": {name: reference[name] for name in identity_names},
        "config": reference["config"],
        "one_step_moments": {
            "path": moment_path.name,
            "sha256": _sha256(moment_path),
        },
    }
    (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "status": report["status"],
        "n_observations": report["n_observations"],
        "estimate": summary["estimate"],
        "difference": report["difference"],
        "robust_standard_error": summary["robust_standard_error"],
    }, indent=2))


if __name__ == "__main__":
    main()
