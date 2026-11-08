#!/usr/bin/env python3
"""Choose a direct-selected conditional-flow draw budget without response data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from sbsi.measurement_model import load_measurement_model
from sbsi.selection_model import load_selection_model
from scripts.train_minimal_selected_event_head import (
    FEATURES,
    MAG_MAX,
    MODEL_SEEDS,
    PHYSICAL_TARGETS,
    RADIUS_MIN_ARCSEC,
    TUNE_CASES,
    load_case,
)


BUDGETS = (64, 256, 1024)
QMC_SEEDS = (7301, 7302, 7303, 7304)
PLANNED_RESPONSE_H = 0.05
MAX_RESPONSE_SUPPORT_BOUND = 5.0e-5
PIXEL_SIZE_ARCSEC = 0.2


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


def _load_json(path: str | Path) -> dict:
    return json.loads(
        Path(path).read_text(),
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON constant {value}")
        ),
    )


def _selected_protocol(selected_root: Path) -> dict:
    protocol_path = selected_root / "protocol.json"
    result_path = selected_root / "development_result.json"
    protocol = _load_json(protocol_path)
    result = _load_json(result_path)
    if (
        protocol.get("version") != "minimal-selected-event-head-v2"
        or result.get("passed") is not True
        or result.get("response_or_test_cases_read") is not False
        or result.get("protocol_sha256") != sha256(protocol_path)
        or protocol["features"] != list(FEATURES)
        or protocol["event"]["radius_min_arcsec"] != RADIUS_MIN_ARCSEC
    ):
        raise RuntimeError("direct selected-event development gate is invalid")
    for rendered, expected in protocol["small_input_hashes"].items():
        if sha256(rendered) != expected:
            raise RuntimeError(f"selected-event input changed: {rendered}")
    return protocol


def prepare(args: argparse.Namespace) -> None:
    root = Path(args.output_root).resolve()
    selected_root = Path(args.selected_root).resolve()
    if root.exists():
        raise FileExistsError(f"preserving existing experiment: {root}")
    selected = _selected_protocol(selected_root)
    development_result = selected_root / "development_result.json"
    audit_job = Path(args.audit_job).resolve()
    review_job = Path(args.review_job).resolve()
    inputs = [
        Path(__file__).resolve(),
        audit_job,
        review_job,
        selected_root / "protocol.json",
        development_result,
        Path(selected["paths"]["flow_checkpoint"]),
        *(selected_root / f"models/seed{seed}/selected.pt" for seed in MODEL_SEEDS),
    ]
    for path in inputs:
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(path)

    root.mkdir(parents=False)
    (root / "support").mkdir()
    (root / "logs").mkdir()
    protocol = {
        "version": "minimal-selected-response-support-v3",
        "created_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "purpose": (
            "Choose the smallest prespecified QMC draw budget for the direct "
            "selected-event conditional-flow estimator using development support "
            "diagnostics only, before cases140--239 are opened."
        ),
        "estimator": (
            "sum_x p_S(x) E_flow[e | U,C,x] / sum_x p_S(x), where "
            "S=U&C and C is MAG_AUTO<25.8 and convolved FLUX_RADIUS>=0.75arcsec"
        ),
        "features": list(FEATURES),
        "flow_role": "conditional selected-shape mean only; no cut probability factor",
        "development_cases": list(TUNE_CASES),
        "response_cases_read": [],
        "candidate_draws_per_row": list(BUDGETS),
        "qmc_seeds": list(QMC_SEEDS),
        "planned_response_h": PLANNED_RESPONSE_H,
        "support_gate": {
            "metric": (
                "maximum over QMC seeds of the pooled component-1 response "
                "worst-case bound from zero-accepted rows"
            ),
            "maximum_response_bound": MAX_RESPONSE_SUPPORT_BOUND,
            "equivalent_m_percentage_points_near_unit_response": (
                100.0 * MAX_RESPONSE_SUPPORT_BOUND
            ),
            "selection_rule": "smallest prespecified draw budget passing the gate",
            "zero_row_completion": (
                "zero midpoint with exact component-wise bound |e_j| <= "
                "1 + |R_blend shear shift_j| from the physical disk flow"
            ),
        },
        "scope": {
            "existing_development_catalogues_only": True,
            "response_used": False,
            "new_image_simulation": False,
            "forbidden": [
                "response moments, response differences, or m",
                "new features, model parameters, training, or probability calibration",
                "changing the 0.75arcsec convolved FLUX_RADIUS sample",
            ],
        },
        "paths": {
            "output_root": str(root),
            "selected_root": str(selected_root),
            "flow_checkpoint": selected["paths"]["flow_checkpoint"],
            "source_root": selected["paths"]["source_root"],
            "audit_script": str(Path(__file__).resolve()),
            "audit_job": str(audit_job),
            "review_job": str(review_job),
        },
        "input_sha256": {str(path): sha256(path) for path in inputs},
    }
    write_json(root / "protocol.json", protocol)
    print(f"MINIMAL_SELECTED_SUPPORT_PREPARED root={root}", flush=True)


def load_protocol(output_root: str | Path) -> dict:
    root = Path(output_root).resolve()
    protocol_path = root / "protocol.json"
    protocol = _load_json(protocol_path)
    if (
        protocol.get("version") != "minimal-selected-response-support-v3"
        or Path(protocol["paths"]["output_root"]) != root
        or protocol["development_cases"] != list(TUNE_CASES)
        or protocol["response_cases_read"] != []
        or protocol["candidate_draws_per_row"] != list(BUDGETS)
        or protocol["qmc_seeds"] != list(QMC_SEEDS)
        or protocol["planned_response_h"] != PLANNED_RESPONSE_H
        or protocol["support_gate"]["maximum_response_bound"]
        != MAX_RESPONSE_SUPPORT_BOUND
    ):
        raise RuntimeError("support protocol changed")
    for rendered, expected in protocol["input_sha256"].items():
        if sha256(rendered) != expected:
            raise RuntimeError(f"support input changed: {rendered}")
    _selected_protocol(Path(protocol["paths"]["selected_root"]))
    return protocol


def direct_support_from_physical_samples(
    samples: np.ndarray,
    event_probability: np.ndarray,
    blend_shift: np.ndarray,
) -> dict:
    """Return support diagnostics without evaluating any response moment."""
    samples = np.asarray(samples, dtype=np.float64)
    probability = np.asarray(event_probability, dtype=np.float64)
    shift = np.asarray(blend_shift, dtype=np.float64)
    if (
        samples.ndim != 3
        or samples.shape[-1] != 4
        or probability.shape != (len(samples),)
        or shift.shape != (len(samples), 2)
        or not np.isfinite(samples).all()
        or not np.isfinite(probability).all()
        or not np.isfinite(shift).all()
        or np.any(samples[..., 2:] < 0.0)
        or np.any((probability < 0.0) | (probability > 1.0))
        or np.any(np.linalg.norm(samples[..., :2], axis=-1) > 1.0 + 1.0e-10)
    ):
        raise ValueError("invalid physical disk samples, probability, or shift")
    flux_min = 10.0 ** ((30.0 - MAG_MAX) / 2.5)
    passed = (samples[..., 3] > flux_min) & (
        samples[..., 2] >= RADIUS_MIN_ARCSEC / PIXEL_SIZE_ARCSEC
    )
    missing = passed.sum(axis=1) == 0
    total_mass = float(probability.sum())
    if total_mass <= 0.0:
        raise ValueError("direct selected-event probability has zero total mass")
    missing_mass = float(probability[missing].sum())
    missing_numerator_bound = np.sum(
        probability[missing, None] * (1.0 + np.abs(shift[missing])),
        axis=0,
        dtype=np.float64,
    )
    return {
        "zero_pass_row_count": int(missing.sum()),
        "total_row_count": int(len(samples)),
        "total_event_probability_mass": total_mass,
        "zero_pass_event_probability_mass": missing_mass,
        "zero_pass_event_probability_fraction": missing_mass / total_mass,
        "missing_numerator_abs_bound": missing_numerator_bound.tolist(),
        "leg_mean_abs_bound": (missing_numerator_bound / total_mass).tolist(),
    }


def audit_case(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("support audit requires CUDA")
    protocol = load_protocol(args.output_root)
    case = TUNE_CASES[args.index]
    destination = Path(args.output_root) / "support" / f"case{case:03d}.json"
    if destination.exists():
        raise FileExistsError(f"preserving existing support audit: {destination}")

    selected_root = Path(protocol["paths"]["selected_root"])
    selected_protocol = _selected_protocol(selected_root)
    ids, frames, _ = load_case(selected_protocol, case)
    case_key = "localization_root" if case < 90 else "replication_root"
    data_path = (
        Path(selected_protocol["paths"][case_key])
        / "prepared"
        / f"case{case:03d}"
        / "data.npz"
    )
    with np.load(data_path) as source:
        shift = np.asarray(source["shift"], dtype=np.float64)
    if shift.shape != (2, len(ids), 2) or not np.isfinite(shift).all():
        raise RuntimeError(f"invalid fixed blend shift: case{case:03d}")

    heads = [
        load_selection_model(
            selected_root / f"models/seed{seed}/selected.pt", device="cuda"
        )
        for seed in MODEL_SEEDS
    ]
    probability = np.stack(
        [
            np.mean([head.predict_proba(frames[leg]) for head in heads], axis=0)
            for leg in range(2)
        ]
    ).astype(np.float64)
    flow = load_measurement_model(protocol["paths"]["flow_checkpoint"], device="cuda")
    if (
        tuple(flow.target_transform.target_names) != PHYSICAL_TARGETS
        or tuple(flow.condition_preprocessor.feature_names) != FEATURES
    ):
        raise RuntimeError("frozen flow interface changed")

    records = {}
    for seed in QMC_SEEDS:
        seed_record = {}
        for leg in range(2):
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            samples = np.asarray(
                flow.sample(
                    frames[leg],
                    n_samples=max(BUDGETS),
                    batch_size=args.batch_size,
                    qmc=True,
                ),
                dtype=np.float64,
            )
            budget_record = {}
            for budget in BUDGETS:
                budget_record[str(budget)] = direct_support_from_physical_samples(
                    samples[:, :budget], probability[leg], shift[leg]
                )
            seed_record[f"leg{leg}"] = budget_record
            del samples
        records[str(seed)] = seed_record
        print(f"MINIMAL_SELECTED_SUPPORT case={case} seed={seed}", flush=True)

    output = {
        "checks_passed": True,
        "case": case,
        "n_objects": len(ids),
        "response_statistics_computed": False,
        "response_cases_read": False,
        "records": records,
        "protocol_sha256": sha256(Path(args.output_root) / "protocol.json"),
    }
    write_json(destination, output)
    print(f"MINIMAL_SELECTED_SUPPORT_CASE_COMPLETE case={case}", flush=True)


def review(args: argparse.Namespace) -> None:
    protocol = load_protocol(args.output_root)
    root = Path(args.output_root)
    case_records = []
    for case in TUNE_CASES:
        path = root / "support" / f"case{case:03d}.json"
        record = _load_json(path)
        if (
            record.get("checks_passed") is not True
            or record.get("case") != case
            or record.get("response_statistics_computed") is not False
            or record.get("response_cases_read") is not False
            or record.get("protocol_sha256") != sha256(root / "protocol.json")
        ):
            raise RuntimeError(f"invalid support audit: {path}")
        case_records.append(record)

    summaries = {}
    for budget in BUDGETS:
        per_seed = {}
        for seed in QMC_SEEDS:
            leg_bounds = []
            leg_zero_fraction = []
            for leg in range(2):
                total_mass = 0.0
                zero_mass = 0.0
                missing_bound = np.zeros(2, dtype=np.float64)
                for case_record in case_records:
                    item = case_record["records"][str(seed)][f"leg{leg}"][str(budget)]
                    total_mass += item["total_event_probability_mass"]
                    zero_mass += item["zero_pass_event_probability_mass"]
                    missing_bound += np.asarray(
                        item["missing_numerator_abs_bound"], dtype=np.float64
                    )
                leg_bounds.append(missing_bound / total_mass)
                leg_zero_fraction.append(zero_mass / total_mass)
            leg_bounds = np.asarray(leg_bounds)
            response_bound = (
                leg_bounds[0] + leg_bounds[1]
            ) / (2.0 * PLANNED_RESPONSE_H)
            per_seed[str(seed)] = {
                "pooled_zero_pass_event_probability_fraction_by_leg": (
                    leg_zero_fraction
                ),
                "pooled_leg_mean_abs_bound": leg_bounds.tolist(),
                "pooled_response_abs_bound": response_bound.tolist(),
            }
        max_component1 = max(
            value["pooled_response_abs_bound"][0] for value in per_seed.values()
        )
        summaries[str(budget)] = {
            "per_qmc_seed": per_seed,
            "maximum_component1_response_abs_bound": max_component1,
            "maximum_component1_m_percentage_points_near_unit_response": (
                100.0 * max_component1
            ),
            "passes_support_gate": bool(
                max_component1 <= MAX_RESPONSE_SUPPORT_BOUND
            ),
        }

    selected = next(
        (budget for budget in BUDGETS if summaries[str(budget)]["passes_support_gate"]),
        None,
    )
    result = {
        "checks_passed": True,
        "development_only": True,
        "response_statistics_computed": False,
        "response_cases_read": False,
        "summaries": summaries,
        "selected_draws_per_row": selected,
        "passed": selected is not None,
        "next_action": (
            "freeze this draw budget for one existing cases140--239 response test"
            if selected is not None
            else "do not open cases140--239; define a larger numerical budget only"
        ),
        "protocol_sha256": sha256(root / "protocol.json"),
    }
    destination = root / "support_decision.json"
    if destination.exists():
        raise FileExistsError(f"preserving existing support decision: {destination}")
    write_json(destination, result)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), flush=True)
    print("MINIMAL_SELECTED_SUPPORT_REVIEW_COMPLETE", flush=True)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--output-root", type=Path, required=True)
    prepare_parser.add_argument("--selected-root", type=Path, required=True)
    prepare_parser.add_argument("--audit-job", type=Path, required=True)
    prepare_parser.add_argument("--review-job", type=Path, required=True)
    case_parser = commands.add_parser("case")
    case_parser.add_argument("--output-root", type=Path, required=True)
    case_parser.add_argument("--index", type=int, choices=range(len(TUNE_CASES)), required=True)
    case_parser.add_argument("--batch-size", type=int, default=2048)
    review_parser = commands.add_parser("review")
    review_parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.command == "prepare":
        prepare(args)
    elif args.command == "case":
        audit_case(args)
    else:
        review(args)


if __name__ == "__main__":
    main()
