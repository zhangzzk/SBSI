#!/usr/bin/env python
# Archived Infer V1 sampling diagnostic.
"""Summarize K=M=16,384 against the retained K=32,768 20k reference arms."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


SEEDS = (8701, 8702)
N_TARGET = 1_000_000


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_arm(path: Path):
    result_path = path / "result.json"
    payload = json.loads(result_path.read_text())
    moment_path = path / payload["one_step_moments"]["path"]
    if _sha256(moment_path) != payload["one_step_moments"]["sha256"]:
        raise RuntimeError(f"moment hash mismatch in {path}")
    with np.load(moment_path) as source:
        arrays = {name: np.asarray(source[name]) for name in source.files}
    return payload, arrays


def _estimate_and_influence(payload, arrays, m):
    ladder = tuple(int(value) for value in payload["config"]["adaptive_draw_ladder"])
    index = ladder.index(m)
    score = arrays["ladder_score"][index]
    information = arrays["ladder_information"][index]
    center = np.asarray(payload["initial_center"]["center"], dtype=np.float64)
    information_sum = information.sum(axis=0, dtype=np.float64)
    information_sum = 0.5 * (information_sum + information_sum.T)
    eigenvalues = np.linalg.eigvalsh(information_sum)
    if not np.isfinite(eigenvalues).all() or eigenvalues[0] <= 0:
        raise RuntimeError(f"non-positive information at M={m}")
    step = np.linalg.solve(information_sum, score.sum(axis=0, dtype=np.float64))
    estimate = center + step
    residual = score - np.einsum("nij,j->ni", information, step)
    influence = np.linalg.solve(information_sum / len(score), residual.T).T
    recorded = np.asarray(payload["full_ladder"][str(m)]["estimate"])
    if not np.allclose(estimate, recorded, rtol=0.0, atol=1.0e-12):
        raise RuntimeError(f"recorded estimate mismatch at M={m}")
    return estimate, influence


def _paired_delta(left, right):
    left_estimate, left_influence = left
    right_estimate, right_influence = right
    if left_influence.shape != right_influence.shape:
        raise RuntimeError("paired influence arrays have different shapes")
    delta = right_estimate - left_estimate
    paired_se = np.std(right_influence - left_influence, axis=0, ddof=1) / np.sqrt(
        len(left_influence)
    )
    return {
        "delta_g": delta.tolist(),
        "paired_standard_error": paired_se.tolist(),
        "pull": np.divide(delta, paired_se).tolist(),
    }


def _validate_identity(payloads):
    reference = payloads[0]
    fields = (
        "model_sha256",
        "model_cache_sha256",
        "scene_sha256",
        "proposal_cache_sha256",
        "mock_input_sha256",
        "initial_center",
        "observation_partition",
    )
    for payload in payloads[1:]:
        for field in fields:
            if payload[field] != reference[field]:
                raise RuntimeError(f"arm identity mismatch for {field}")


def main(argv=None):
    args = parse_args(argv)
    root = Path(args.root).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}")

    loaded = {}
    for seed in SEEDS:
        loaded[(16384, seed)] = _load_arm(
            root / "results" / f"k16384_e0p1_p{seed}_m512_16384"
        )
        loaded[(32768, seed)] = _load_arm(
            root / "results" / f"k32768_e0p1_p{seed}_m512_65536"
        )
    _validate_identity([value[0] for value in loaded.values()])

    n_objects = loaded[(16384, SEEDS[0])][0]["observation_partition"]["n_partition"]
    scale = float(np.sqrt(n_objects / N_TARGET))
    arms = {}
    k_comparisons = {}
    for seed in SEEDS:
        payload16, arrays16 = loaded[(16384, seed)]
        payload32, arrays32 = loaded[(32768, seed)]
        m8192 = _estimate_and_influence(payload16, arrays16, 8192)
        m16384 = _estimate_and_influence(payload16, arrays16, 16384)
        nested = _paired_delta(m8192, m16384)
        nested["projected_standard_error_at_1m"] = (
            np.asarray(nested["paired_standard_error"]) * scale
        ).tolist()
        nested["observed_shift_scaled_as_zero_mean_noise_to_1m"] = (
            np.asarray(nested["delta_g"]) * scale
        ).tolist()
        arms[str(seed)] = {
            "estimate_at_m16384": m16384[0].tolist(),
            "m8192_to_m16384": nested,
            "elapsed_seconds": float(payload16["result"]["elapsed_seconds"]),
            "flow_evaluations": int(payload16["result"]["flow_evaluations"]),
        }
        k_comparisons[str(seed)] = _paired_delta(
            m16384, _estimate_and_influence(payload32, arrays32, 16384)
        )

    seed_comparison = _paired_delta(
        _estimate_and_influence(*loaded[(16384, 8701)], 16384),
        _estimate_and_influence(*loaded[(16384, 8702)], 16384),
    )
    capture = json.loads(
        (root / "candidate_capture_kref131072" / "result.json").read_text()
    )["views"]["center"]["prefixes"]["16384"]
    report = {
        "status": "complete",
        "method": "paired_k16384_m16384_n20k_v1",
        "n_observations": n_objects,
        "target_observations": N_TARGET,
        "one_over_sqrt_n_scale_to_target": scale,
        "configuration": {
            "proposal_candidates": 16384,
            "maximum_draws": 16384,
            "proposal_flow_samples": 128,
            "proposal_epsilon": 0.1,
            "proposal_prefilter_candidates": 131072,
            "proposal_seeds": list(SEEDS),
        },
        "candidate_capture_against_kref131072_at_center": capture,
        "arms": arms,
        "paired_k16384_to_k32768_at_m16384": k_comparisons,
        "paired_seed8701_to_seed8702_at_k16384_m16384": seed_comparison,
        "interpretation_guardrail": (
            "Projected standard errors assume the empirically checked 1/sqrt(N) "
            "scaling of zero-mean finite-M noise. Candidate truncation and any "
            "nonzero finite-M expectation do not average away with more objects."
        ),
    }
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
