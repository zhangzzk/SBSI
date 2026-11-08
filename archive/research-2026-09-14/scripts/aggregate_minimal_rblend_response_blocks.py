#!/usr/bin/env python
"""Aggregate frozen minimal-R_blend response blocks at the simulation-case level."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from scripts.evaluate_constgold_complete_forward_response import (
    _case_bootstrap,
    _combine,
    _combine_measured,
    _matched_detection_selection_comparison,
    _measured_summary,
    _prediction_run_diagnostics,
    _response_summary,
    _serialize_sufficient,
    _truth_shape_case_bootstrap,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MEASURED_BRANCHES = (
    "matched_selected",
    "matched_detected",
    "matched_detected_selected",
    "complete",
    "detection_only",
)
PREDICTED_BRANCHES = (
    "matched_selected",
    "matched_detected",
    "matched_detected_selected",
    "complete",
    "detection_only",
    "selection_only",
    "unweighted",
    "truth_detection",
    "truth_complete",
)
COUNT_FIELDS = (
    "truth_population_count",
    "crossmatch_rows_without_shape",
    "shape_rows_without_crossmatch",
    "crossmatched_rows_outside_finite_prior_support",
    "actual_detected_count",
    "actual_usable_shape_count",
    "actual_selected_count",
    "dropped_nonfinite_measurements",
    "dropped_invalid_shape",
    "dropped_measured_cut",
)
IDENTITY_FIELDS = (
    "analysis",
    "shear_component",
    "h",
    "analysis_primary_re_min",
    "circularized_condition",
    "measured_output_cut",
    "prediction_definition",
    "response_ladder_definition",
    "detection_note",
    "truth_shape_population_definition",
    "models",
    "detection_model",
    "sampling",
)


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _repo_path(path: str | Path) -> Path:
    result = Path(path).expanduser()
    return result if result.is_absolute() else REPOSITORY_ROOT / result


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read JSON {path}: {error}") from error


def _validate_protocol(protocol_path: Path) -> tuple[dict, dict]:
    protocol = _load_json(protocol_path)
    if protocol.get("protocol") != "minimal-rblend-fresh-acceptance-v1":
        raise ValueError("wrong acceptance protocol")
    candidate_spec = protocol["candidate"]
    candidate_path = _repo_path(candidate_spec["likelihood_config"])
    if _sha256(candidate_path) != candidate_spec["likelihood_config_sha256"]:
        raise RuntimeError("candidate likelihood config hash mismatch")
    return protocol, _load_json(candidate_path)


def _validate_block_identity(
    payload: dict,
    *,
    reference: dict | None,
    protocol: dict,
    candidate: dict,
) -> None:
    if reference is not None:
        for field in IDENTITY_FIELDS:
            if payload.get(field) != reference.get(field):
                raise RuntimeError(f"response-block identity mismatch: {field}")
    response = protocol["response_evaluation"]
    population = protocol["population"]["measured_selection"]
    if payload.get("analysis") != "ConstGold complete forward likelihood response":
        raise RuntimeError("wrong block analysis")
    if payload.get("analysis_primary_re_min") is not None:
        raise RuntimeError("unexpected secondary primary-radius restriction")
    if payload.get("measured_output_cut") != {
        "ellipticity_norm_max": None,
        "measured_mag_auto_max": population["measured_mag_auto_max_exclusive"],
        "measured_radius_arcsec_min": population[
            "measured_radius_min_inclusive_arcsec"
        ],
        "measured_log_flux_radius_min": float(
            np.log(
                population["measured_radius_min_inclusive_arcsec"]
                / candidate["observing_conditions"]["pixel_size"]
            )
        ),
    }:
        raise RuntimeError("response block does not use the frozen no-|e| cuts")
    sampling = payload.get("sampling", {})
    if (
        sampling.get("draws_per_atom_per_flow_seed")
        != response["draws_per_qmc_group"]
        or sampling.get("seeds") != response["qmc_seeds"]
        or sampling.get("qmc") is not True
        or sampling.get("common_random_numbers_between_legs") is not True
    ):
        raise RuntimeError("response block does not use the frozen QMC design")
    model_runs = payload.get("models", [])
    expected_flow = candidate["measurement_model"]["sha256"]
    if (
        len(model_runs) != len(response["qmc_seeds"])
        or [item.get("sampling_seed") for item in model_runs]
        != response["qmc_seeds"]
        or any(item.get("sha256") != expected_flow for item in model_runs)
    ):
        raise RuntimeError("response block measurement-model run identity mismatch")
    detector = payload.get("detection_model", {})
    expected_detector = candidate["emulator"]["detection_model"]
    if (
        detector.get("aggregation") != "arithmetic_mean_probability"
        or detector.get("classifier_sha256")
        != [member["sha256"] for member in expected_detector["members"]]
        or len(detector.get("features", [])) != 9
    ):
        raise RuntimeError("response block detector identity mismatch")
    inputs = payload.get("inputs", {})
    if inputs.get("crowding_source") != "truth_scene" or inputs.get(
        "crowd_lookup"
    ) is not None:
        raise RuntimeError("response block did not use truth-derived crowding")


def _rebuild_sufficient(
    cases: tuple[int, ...], records: dict[int, dict], run_count: int
) -> tuple[dict, dict, list[dict]]:
    measured = {
        branch: {+1: {}, -1: {}} for branch in MEASURED_BRANCHES
    }
    truth = {
        "detection": {+1: {}, -1: {}},
        "detection_and_selection": {+1: {}, -1: {}},
    }
    predicted = [{+1: {}, -1: {}} for _ in range(run_count)]
    for case in cases:
        sufficient = records[case]["response_sufficient"]
        for branch in MEASURED_BRANCHES:
            for direction in (+1, -1):
                measured[branch][direction][case] = sufficient["measured"][branch][
                    str(direction)
                ]
        for branch in truth:
            for direction in (+1, -1):
                truth[branch][direction][case] = sufficient["truth_actual"][branch][
                    str(direction)
                ]
        runs = sufficient["predicted"]
        if len(runs) != run_count:
            raise RuntimeError(f"case {case} prediction-run count mismatch")
        for run_index, run in enumerate(runs):
            for direction in (+1, -1):
                predicted[run_index][direction][case] = run[str(direction)]
    return measured, truth, predicted


def _pooled_measured(cases: tuple[int, ...], measured: dict, h: float) -> dict:
    return {
        branch: _measured_summary(
            _combine_measured([measured[branch][+1][case] for case in cases]),
            _combine_measured([measured[branch][-1][case] for case in cases]),
            h,
        )
        for branch in MEASURED_BRANCHES
    }


def _pooled_predicted(
    cases: tuple[int, ...],
    predicted: list[dict],
    h: float,
    sampling_seed_count: int,
) -> dict:
    result = {}
    if len(predicted) % sampling_seed_count:
        raise RuntimeError("model-by-QMC prediction grid is not rectangular")
    model_count = len(predicted) // sampling_seed_count
    for branch in PREDICTED_BRANCHES:
        by_run = [
            _response_summary(
                _combine([run[+1][case] for case in cases], branch),
                _combine([run[-1][case] for case in cases], branch),
                h,
            )
            for run in predicted
        ]
        responses = np.asarray(
            [item["response_first_column"] for item in by_run], dtype=np.float64
        )
        result[branch] = {
            "plus_mean": np.mean([item["plus_mean"] for item in by_run], axis=0),
            "minus_mean": np.mean([item["minus_mean"] for item in by_run], axis=0),
            "response_first_column": responses.mean(axis=0),
            **_prediction_run_diagnostics(
                responses,
                model_count=model_count,
                sampling_seed_count=sampling_seed_count,
            ),
        }
    return result


def _row_accounting(cases: tuple[int, ...], records: dict[int, dict]) -> dict:
    result = {}
    for label in ("plus", "minus"):
        result[label] = {
            field: int(
                sum(records[case]["measured_population"][label][field] for case in cases)
            )
            for field in COUNT_FIELDS
        }
    return result


def aggregate(
    block_paths: list[Path], protocol_path: Path, *, bootstrap_replicates: int | None
) -> dict:
    """Validate and combine block JSONs without using their blockwise point estimates."""
    protocol, candidate = _validate_protocol(protocol_path)
    simulation = protocol["simulation"]
    expected_cases = tuple(range(simulation["case_first"], simulation["case_last"] + 1))
    if len(block_paths) != simulation["block_count"]:
        raise RuntimeError("response-block file count does not match the protocol")
    payloads = []
    records: dict[int, dict] = {}
    artifacts = []
    reference = None
    for path in block_paths:
        payload = _load_json(path)
        _validate_block_identity(
            payload, reference=reference, protocol=protocol, candidate=candidate
        )
        reference = payload if reference is None else reference
        block_cases = tuple(int(case) for case in payload.get("cases", []))
        if len(block_cases) != simulation["block_cases"]:
            raise RuntimeError(f"wrong case count in response block {path}")
        if set(payload.get("per_case", {})) != {str(case) for case in block_cases}:
            raise RuntimeError(f"per-case sufficient statistics mismatch in {path}")
        overlap = set(block_cases).intersection(records)
        if overlap:
            raise RuntimeError(f"duplicate response cases: {sorted(overlap)}")
        for case in block_cases:
            records[case] = payload["per_case"][str(case)]
        artifacts.append(
            {"path": str(path.resolve()), "sha256": _sha256(path), "cases": list(block_cases)}
        )
        payloads.append(payload)
    if tuple(sorted(records)) != expected_cases:
        missing = sorted(set(expected_cases) - set(records))
        extra = sorted(set(records) - set(expected_cases))
        raise RuntimeError(
            f"response cases do not equal the protocol: missing={missing}, extra={extra}"
        )

    reference = payloads[0]
    h = float(reference["h"])
    shear_component = reference["shear_component"]
    amplitudes = {
        abs(float(leg[shear_component]))
        for leg in simulation["legs"]
        if float(leg[shear_component]) != 0.0
    }
    if h not in amplitudes:
        raise RuntimeError("response amplitude is not a frozen protocol endpoint")
    diagonal_index = 0 if shear_component == "g1" else 1
    cross_index = 1 - diagonal_index
    response_component = "R11" if diagonal_index == 0 else "R22"
    cross_component = "R21" if diagonal_index == 0 else "R12"
    replicate_count = int(
        protocol["response_evaluation"]["bootstrap_replicates"]
        if bootstrap_replicates is None
        else bootstrap_replicates
    )
    if replicate_count < 2:
        raise ValueError("bootstrap replicates must be at least two")
    bootstrap_seed = int(protocol["response_evaluation"]["bootstrap_seed"])
    run_count = len(reference["models"])
    measured, truth, predicted = _rebuild_sufficient(
        expected_cases, records, run_count
    )
    measured_pooled = _pooled_measured(expected_cases, measured, h)
    predicted_pooled = _pooled_predicted(
        expected_cases,
        predicted,
        h,
        len(protocol["response_evaluation"]["qmc_seeds"]),
    )
    actual_truth = {
        branch: _measured_summary(
            _combine_measured([truth[branch][+1][case] for case in expected_cases]),
            _combine_measured([truth[branch][-1][case] for case in expected_cases]),
            h,
        )
        for branch in truth
    }
    bootstrap = _case_bootstrap(
        cases=expected_cases,
        measured_selected=measured["complete"],
        measured_detected=measured["detection_only"],
        measured_matched_selected=measured["matched_selected"],
        measured_matched_detected=measured["matched_detected"],
        predicted=predicted,
        h=h,
        replicates=replicate_count,
        seed=bootstrap_seed,
        component_index=diagonal_index,
    )
    truth_bootstrap = _truth_shape_case_bootstrap(
        cases=expected_cases,
        actual_detected=truth["detection"],
        actual_selected=truth["detection_and_selection"],
        predicted=predicted,
        h=h,
        replicates=replicate_count,
        seed=bootstrap_seed,
    )
    matched_selection = _matched_detection_selection_comparison(
        cases=expected_cases,
        measured=measured["matched_detected_selected"],
        predicted=predicted,
        h=h,
        replicates=replicate_count,
        seed=bootstrap_seed,
        component_index=diagonal_index,
    )

    measured_response = np.asarray(
        measured_pooled["complete"]["response_first_column"], dtype=np.float64
    )
    model_response = np.asarray(
        predicted_pooled["complete"]["response_first_column"], dtype=np.float64
    )
    if model_response[diagonal_index] <= 0:
        raise RuntimeError("nonpositive diagonal model response")
    m_percent = 100.0 * (
        measured_response[diagonal_index] / model_response[diagonal_index] - 1.0
    )
    m_bootstrap = bootstrap["multiplicative_bias_percent"]["complete"]
    case_ci95 = np.asarray(m_bootstrap["ci95"], dtype=np.float64)
    response_by_sampling_seed = np.asarray(
        predicted_pooled["complete"]["response_by_sampling_seed"],
        dtype=np.float64,
    )
    if (
        len(response_by_sampling_seed) < 2
        or (response_by_sampling_seed[:, diagonal_index] <= 0).any()
    ):
        raise RuntimeError("QMC diagnostics require at least two positive responses")
    m_by_sampling_seed = 100.0 * (
        measured_response[diagonal_index]
        / response_by_sampling_seed[:, diagonal_index]
        - 1.0
    )
    qmc_standard_error = float(
        m_by_sampling_seed.std(ddof=1) / np.sqrt(len(m_by_sampling_seed))
    )
    numerical_gate = protocol["response_evaluation"]["qmc_numerical_gate"]
    qmc_half_width = (
        float(numerical_gate["student_t_95_multiplier"]) * qmc_standard_error
    )
    expanded_ci95 = case_ci95 + np.asarray([-qmc_half_width, qmc_half_width])
    numerically_adequate = bool(
        qmc_standard_error <= numerical_gate["maximum_standard_error_percent"]
    )
    truth_actual_response = np.asarray(
        actual_truth["detection_and_selection"]["response_first_column"],
        dtype=np.float64,
    )
    truth_model_response = np.asarray(
        predicted_pooled["truth_complete"]["response_first_column"],
        dtype=np.float64,
    )
    total_error = model_response - measured_response
    truth_error = truth_model_response - truth_actual_response
    residual_error = total_error - truth_error
    denominator = model_response[diagonal_index]
    components = {
        "definition": (
            "model-minus-image response = truth-population weighting error + "
            "conditional measurement-residual error"
        ),
        "model_minus_image_response": total_error,
        "truth_population_response_error": truth_error,
        "measurement_residual_response_error": residual_error,
        "diagonal_m_contribution_percent": {
            "truth_population": -100.0 * truth_error[diagonal_index] / denominator,
            "measurement_residual": -100.0 * residual_error[diagonal_index] / denominator,
            "total": -100.0 * total_error[diagonal_index] / denominator,
        },
    }
    if not np.isclose(
        components["diagonal_m_contribution_percent"]["total"],
        m_percent,
        rtol=1e-12,
        atol=1e-12,
    ):
        raise RuntimeError("response-component decomposition does not close")
    component_limit = float(
        protocol["response_evaluation"]["component_robustness_gate"][
            "maximum_absolute_central_contribution_percent"
        ]
    )
    component_values = components["diagonal_m_contribution_percent"]
    observed_component_max = max(
        abs(float(component_values["truth_population"])),
        abs(float(component_values["measurement_residual"])),
    )
    components["robustness_gate"] = {
        "maximum_absolute_central_contribution_percent": component_limit,
        "observed_maximum_absolute_central_contribution_percent": (
            observed_component_max
        ),
        "opposite_sign_central_contributions": bool(
            component_values["truth_population"]
            * component_values["measurement_residual"]
            < 0.0
        ),
        "passes": bool(observed_component_max < component_limit),
    }

    return {
        "schema_version": 1,
        "analysis": "minimal-rblend blockwise response acceptance",
        "protocol": {
            "path": str(protocol_path.resolve()),
            "sha256": _sha256(protocol_path),
            "name": protocol["protocol"],
        },
        "endpoint": {
            "shear_component": shear_component,
            "h": h,
            "response_component": response_component,
            "cross_response_component": cross_component,
            "case_count": len(expected_cases),
        },
        "response_acceptance": {
            "measured_response": measured_response[diagonal_index],
            "model_response": model_response[diagonal_index],
            "m_percent": m_percent,
            "case_bootstrap_standard_error_percent": m_bootstrap["standard_error"],
            "case_bootstrap_ci95_percent": case_ci95,
            "m_by_qmc_seed_percent": m_by_sampling_seed,
            "qmc_standard_error_percent": qmc_standard_error,
            "qmc_standard_error_limit_percent": numerical_gate[
                "maximum_standard_error_percent"
            ],
            "qmc_student_t_95_multiplier": numerical_gate[
                "student_t_95_multiplier"
            ],
            "qmc_95_half_width_percent": qmc_half_width,
            "case_ci95_expanded_by_qmc_95_half_width_percent": expanded_ci95,
            "passes_qmc_numerical_gate": numerically_adequate,
            "equivalence_band_percent": [-0.3, 0.3],
            "passes_marginal_ci95_equivalence": bool(
                numerically_adequate
                and expanded_ci95[0] > -0.3
                and expanded_ci95[1] < 0.3
            ),
        },
        "cross_response": {
            "component": cross_component,
            "measured": measured_response[cross_index],
            "model": model_response[cross_index],
            "model_minus_measured": total_error[cross_index],
        },
        "response_components": components,
        "pooled": {
            "measured": measured_pooled,
            "predicted": predicted_pooled,
            "truth_actual": actual_truth,
        },
        "case_bootstrap": bootstrap,
        "truth_shape_case_bootstrap": truth_bootstrap,
        "matched_detection_separate_selection": matched_selection,
        "row_accounting": _row_accounting(expected_cases, records),
        "block_results": artifacts,
        "frozen_identity": {field: reference[field] for field in IDENTITY_FIELDS},
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block-result", type=Path, action="append", required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=REPOSITORY_ROOT / "configs/calibration_acceptance_minimal_rblend.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    result = aggregate(
        args.block_result,
        args.protocol,
        bootstrap_replicates=None,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(_serialize_sufficient(result), indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(_serialize_sufficient(result["response_acceptance"]), indent=2))


if __name__ == "__main__":
    main()
