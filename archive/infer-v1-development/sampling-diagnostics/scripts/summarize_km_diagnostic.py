#!/usr/bin/env python
# Archived Infer V1 sampling diagnostic.
"""Summarize paired fixed-row K x M numerical diagnostics and select one seed check."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


K_VALUES = (8192, 32768, 65536)
FIRST_SEED = 8701
SECOND_SEED = 8702


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--mode", choices=("select", "final"), required=True)
    parser.add_argument("--tolerance", type=float, default=1.0e-4)
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _result_root(root: Path, k: int, seed: int) -> Path:
    return root / "results" / f"k{k}_e0p1_p{seed}_m512_65536"


def _load_arm(root: Path, k: int, seed: int) -> dict:
    arm_root = _result_root(root, k, seed)
    result_path = arm_root / "result.json"
    payload = json.loads(result_path.read_text())
    moments_path = arm_root / payload["one_step_moments"]["path"]
    if _sha256(moments_path) != payload["one_step_moments"]["sha256"]:
        raise RuntimeError(f"moment hash mismatch in {arm_root}")
    config = payload["config"]
    if config["proposal_candidates"] != k or config["proposal_seed"] != seed:
        raise RuntimeError(f"proposal identity mismatch in {arm_root}")
    if config["proposal_epsilon"] != 0.1:
        raise RuntimeError(f"epsilon mismatch in {arm_root}")
    if config["proposal_flow_samples"] != 128:
        raise RuntimeError(f"QMC sample mismatch in {arm_root}")
    if config["proposal_statistic"] != "mean" or config["proposal_dispersion_statistic"] != "std":
        raise RuntimeError(f"proposal-coordinate mismatch in {arm_root}")
    if not config["retain_full_ladder"]:
        raise RuntimeError(f"full ladder was not retained in {arm_root}")
    ladder = tuple(int(value) for value in config["adaptive_draw_ladder"])
    estimates = np.asarray(
        [payload["full_ladder"][str(value)]["estimate"] for value in ladder],
        dtype=np.float64,
    )
    with np.load(moments_path) as arrays:
        unique = np.asarray(arrays["unique_counts"], dtype=np.int64)
    return {
        "root": str(arm_root),
        "result_sha256": _sha256(result_path),
        "moments_sha256": _sha256(moments_path),
        "k": k,
        "seed": seed,
        "ladder": ladder,
        "estimates": estimates,
        "endpoint": estimates[-1],
        "nested_endpoint_change": estimates[-1] - estimates[-2],
        "elapsed_seconds": float(payload["result"]["elapsed_seconds"]),
        "flow_evaluations": int(payload["result"]["flow_evaluations"]),
        "unique_count_percentiles": np.percentile(unique, (0, 10, 50, 90, 100)).tolist(),
        "mean_unique_fraction": float(payload["result"]["mean_unique_fraction"]),
        "initial_center": payload["initial_center"],
        "observation_partition": payload["observation_partition"],
        "mock_input_sha256": payload["mock_input_sha256"],
        "proposal_cache_sha256": payload["proposal_cache_sha256"],
        "prefilter_candidates": config["proposal_prefilter_candidates"],
    }


def _public_arm(arm: dict) -> dict:
    return {
        "root": arm["root"],
        "result_sha256": arm["result_sha256"],
        "moments_sha256": arm["moments_sha256"],
        "k": arm["k"],
        "seed": arm["seed"],
        "draw_ladder": list(arm["ladder"]),
        "estimate_by_m": {
            str(m): arm["estimates"][index].tolist()
            for index, m in enumerate(arm["ladder"])
        },
        "endpoint": arm["endpoint"].tolist(),
        "nested_endpoint_change": arm["nested_endpoint_change"].tolist(),
        "elapsed_seconds": arm["elapsed_seconds"],
        "flow_evaluations": arm["flow_evaluations"],
        "unique_count_q00_q10_q50_q90_q100": arm["unique_count_percentiles"],
        "mean_unique_fraction": arm["mean_unique_fraction"],
        "prefilter_candidates": arm["prefilter_candidates"],
    }


def _validate_pairing(arms: list[dict]) -> None:
    reference = arms[0]
    fields = (
        "ladder",
        "initial_center",
        "observation_partition",
        "mock_input_sha256",
        "proposal_cache_sha256",
        "prefilter_candidates",
    )
    for arm in arms[1:]:
        for field in fields:
            if arm[field] != reference[field]:
                raise RuntimeError(f"paired-arm mismatch for {field}: {arm['root']}")
    partition = reference["observation_partition"]
    if partition["start"] != 0 or partition["stop"] != 20000 or partition["n_partition"] != 20000:
        raise RuntimeError("diagnostic arms must use the same frozen first 20,000 rows")


def _first_stage(root: Path, tolerance: float):
    arms = [_load_arm(root, k, FIRST_SEED) for k in K_VALUES]
    _validate_pairing(arms)
    comparisons = {}
    for left, right in zip(arms[:-1], arms[1:]):
        delta = right["estimates"] - left["estimates"]
        comparisons[f"k{left['k']}_to_k{right['k']}"] = {
            "delta_by_m": {
                str(m): delta[index].tolist()
                for index, m in enumerate(left["ladder"])
            },
            "endpoint_delta": delta[-1].tolist(),
            "endpoint_max_abs": float(np.max(np.abs(delta[-1]))),
            "passes_tolerance": bool(np.max(np.abs(delta[-1])) <= tolerance),
        }

    selected = None
    rationale = None
    for index, arm in enumerate(arms[:-1]):
        nested_pass = np.max(np.abs(arm["nested_endpoint_change"])) <= tolerance
        comparison = comparisons[f"k{arm['k']}_to_k{arms[index + 1]['k']}"]
        if nested_pass and comparison["passes_tolerance"]:
            selected = arm["k"]
            rationale = "smallest K passing both final-doubling and next-K endpoint gates"
            break
    if selected is None:
        for index, arm in enumerate(arms[:-1]):
            comparison = comparisons[f"k{arm['k']}_to_k{arms[index + 1]['k']}"]
            if comparison["passes_tolerance"]:
                selected = arm["k"]
                rationale = "smallest K passing the next-K gate; no arm passed both gates"
                break
    if selected is None:
        selected = arms[-1]["k"]
        rationale = "no smaller K passed the next-K gate; use largest tested support for seed check"

    report = {
        "status": "first_stage_complete",
        "method": "paired_fixed_rows_k_m_diagnostic_v1",
        "absolute_stability_tolerance": tolerance,
        "n_observations": 20000,
        "first_seed": FIRST_SEED,
        "arms": {str(arm["k"]): _public_arm(arm) for arm in arms},
        "k_expansion": comparisons,
        "selected_k_for_second_seed": selected,
        "selection_rationale": rationale,
        "outside_local_evidence_and_ess": "not persisted by retained-full-ladder production-prefix path",
    }
    return report, arms, selected


def main(argv=None):
    args = parse_args(argv)
    if args.tolerance <= 0 or not np.isfinite(args.tolerance):
        raise ValueError("tolerance must be finite and positive")
    root = Path(args.root).resolve()
    report, first_arms, selected = _first_stage(root, args.tolerance)
    if args.mode == "select":
        output = root / "first_stage_summary.json"
        if output.exists():
            raise SystemExit(f"refusing to overwrite {output}")
        output.write_text(json.dumps(report, indent=2) + "\n")
        (root / "selected_k.txt").write_text(f"{selected}\n")
        print(selected)
        return

    second = _load_arm(root, selected, SECOND_SEED)
    selected_first = next(arm for arm in first_arms if arm["k"] == selected)
    _validate_pairing([selected_first, second])
    seed_delta = second["estimates"] - selected_first["estimates"]
    ladder = selected_first["ladder"]
    smallest_stable_m = None
    for index, m in enumerate(ladder):
        first_tail = selected_first["estimates"][index:] - selected_first["endpoint"]
        second_tail = second["estimates"][index:] - second["endpoint"]
        if (
            np.max(np.abs(first_tail)) <= args.tolerance
            and np.max(np.abs(second_tail)) <= args.tolerance
            and np.max(np.abs(seed_delta[index])) <= args.tolerance
        ):
            smallest_stable_m = int(m)
            break
    report.update(
        {
            "status": "complete",
            "second_seed": SECOND_SEED,
            "second_seed_arm": _public_arm(second),
            "seed_delta_by_m": {
                str(m): seed_delta[index].tolist()
                for index, m in enumerate(ladder)
            },
            "seed_endpoint_max_abs": float(np.max(np.abs(seed_delta[-1]))),
            "seed_endpoint_passes_tolerance": bool(
                np.max(np.abs(seed_delta[-1])) <= args.tolerance
            ),
            "smallest_m_passing_nested_tail_and_seed_gate": smallest_stable_m,
            "material_m_reduction_supported": bool(
                smallest_stable_m is not None and smallest_stable_m < ladder[-1]
            ),
        }
    )
    output = root / "final_summary.json"
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}")
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "status": report["status"],
        "selected_k": selected,
        "seed_endpoint_max_abs": report["seed_endpoint_max_abs"],
        "smallest_stable_m": smallest_stable_m,
    }, indent=2))


if __name__ == "__main__":
    main()
