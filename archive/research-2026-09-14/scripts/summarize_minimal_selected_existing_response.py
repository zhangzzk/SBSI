#!/usr/bin/env python3
"""Prepare and reduce the frozen direct-selected existing-catalogue response."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
from scipy.special import stdtrit


CASES = tuple(range(140, 240))
QMC_SEEDS = (7301, 7302, 7303, 7304)
MODEL_SEEDS = (20260913, 20260914, 20260915)
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
POPULATION_COUNT_FIELDS = (
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


def sha256(path: str | Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8_388_608), b""):
            value.update(block)
    return value.hexdigest()


def load_json(path: str | Path) -> dict:
    path = Path(path)
    return json.loads(
        path.read_text(),
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON constant {value} in {path}")
        ),
    )


def write_json(path: str | Path, value: dict) -> None:
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    os.replace(temporary, path)


def _validate_support_artifacts(
    support_root: Path, support_protocol: dict, support_result: dict
) -> list[Path]:
    protocol_hash = sha256(support_root / "protocol.json")
    cases = tuple(support_protocol["development_cases"])
    budgets = tuple(support_protocol["candidate_draws_per_row"])
    seeds = tuple(support_protocol["qmc_seeds"])
    h = float(support_protocol["planned_response_h"])
    maximum = float(support_protocol["support_gate"]["maximum_response_bound"])
    records = []
    paths = []
    for case in cases:
        path = support_root / "support" / f"case{case:03d}.json"
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(path)
        record = load_json(path)
        if (
            record.get("checks_passed") is not True
            or record.get("case") != case
            or record.get("response_statistics_computed") is not False
            or record.get("response_cases_read") is not False
            or record.get("protocol_sha256") != protocol_hash
        ):
            raise RuntimeError(f"invalid support case artifact: {path}")
        records.append(record)
        paths.append(path)

    passing = []
    for budget in budgets:
        seed_bounds = []
        for seed in seeds:
            leg_bounds = []
            for leg in range(2):
                total = sum(
                    record["records"][str(seed)][f"leg{leg}"][str(budget)][
                        "total_event_probability_mass"
                    ]
                    for record in records
                )
                missing = sum(
                    (
                        np.asarray(
                            record["records"][str(seed)][f"leg{leg}"][str(budget)][
                                "missing_numerator_abs_bound"
                            ],
                            dtype=np.float64,
                        )
                        for record in records
                    ),
                    start=np.zeros(2, dtype=np.float64),
                )
                leg_bounds.append(missing / total)
            seed_bounds.append((leg_bounds[0][0] + leg_bounds[1][0]) / (2.0 * h))
        reconstructed = float(max(seed_bounds))
        stored = support_result["summaries"][str(budget)]
        if (
            not np.isclose(
                reconstructed,
                stored["maximum_component1_response_abs_bound"],
                rtol=1.0e-13,
                atol=0.0,
            )
            or stored["passes_support_gate"] != (reconstructed <= maximum)
        ):
            raise RuntimeError(f"support decision does not reconstruct: draws={budget}")
        if reconstructed <= maximum:
            passing.append(budget)
    expected = passing[0] if passing else None
    if support_result.get("selected_draws_per_row") != expected:
        raise RuntimeError("stored support draw choice is not the smallest passing budget")
    return paths


def _validate_upstream(
    selected_root: Path, support_root: Path
) -> tuple[dict, dict, dict]:
    selected_protocol_path = selected_root / "protocol.json"
    selected_result_path = selected_root / "development_result.json"
    selected_protocol = load_json(selected_protocol_path)
    selected_result = load_json(selected_result_path)
    if (
        selected_protocol.get("version") != "minimal-selected-event-head-v2"
        or selected_protocol["features"] != list(FEATURES)
        or selected_protocol["event"]["radius_min_arcsec"] != 0.75
        or selected_result.get("passed") is not True
        or selected_result.get("response_or_test_cases_read") is not False
        or selected_result.get("protocol_sha256") != sha256(selected_protocol_path)
    ):
        raise RuntimeError("invalid selected-event development result")

    support_protocol_path = support_root / "protocol.json"
    support_result_path = support_root / "support_decision.json"
    support_protocol = load_json(support_protocol_path)
    support_result = load_json(support_result_path)
    draws = support_result.get("selected_draws_per_row")
    if (
        support_protocol.get("version") != "minimal-selected-response-support-v3"
        or support_result.get("passed") is not True
        or support_result.get("response_statistics_computed") is not False
        or support_result.get("response_cases_read") is not False
        or support_result.get("protocol_sha256") != sha256(support_protocol_path)
        or draws not in support_protocol["candidate_draws_per_row"]
        or support_protocol["qmc_seeds"] != list(QMC_SEEDS)
    ):
        raise RuntimeError("invalid response-blind support decision")
    _validate_support_artifacts(support_root, support_protocol, support_result)
    return selected_protocol, support_protocol, support_result


def prepare(args: argparse.Namespace) -> None:
    root = Path(args.output_root).resolve()
    selected_root = Path(args.selected_root).resolve()
    support_root = Path(args.support_root).resolve()
    if root.exists():
        raise FileExistsError(f"preserving existing experiment: {root}")
    selected_protocol, support_protocol, support_result = _validate_upstream(
        selected_root, support_root
    )
    paths = {
        "output_root": str(root),
        "selected_root": str(selected_root),
        "support_root": str(support_root),
        "catalogue_root": str(Path(args.catalogue_root).resolve()),
        "crowd_lookup": str(Path(args.crowd_lookup).resolve()),
        "source_root": selected_protocol["paths"]["source_root"],
        "flow_checkpoint": selected_protocol["paths"]["flow_checkpoint"],
        "emulator_model": str(Path(args.emulator_model).resolve()),
        "emulator_metadata": str(Path(args.emulator_metadata).resolve()),
        "blendemu_root": str(Path(args.blendemu_root).resolve()),
        "builder": str(Path(args.builder).resolve()),
        "evaluator": str(Path(args.evaluator).resolve()),
        "script": str(Path(__file__).resolve()),
        "lookup_job": str(Path(args.lookup_job).resolve()),
        "response_job": str(Path(args.response_job).resolve()),
        "review_job": str(Path(args.review_job).resolve()),
    }
    small_inputs = [
        selected_root / "protocol.json",
        selected_root / "development_result.json",
        support_root / "protocol.json",
        support_root / "support_decision.json",
        *(
            support_root / "support" / f"case{case:03d}.json"
            for case in support_protocol["development_cases"]
        ),
        Path(paths["flow_checkpoint"]),
        *(selected_root / f"models/seed{seed}/selected.pt" for seed in MODEL_SEEDS),
        Path(paths["crowd_lookup"]),
        Path(paths["emulator_model"]),
        Path(paths["emulator_metadata"]),
        Path(paths["builder"]),
        Path(paths["evaluator"]),
        Path(paths["script"]),
        Path(paths["lookup_job"]),
        Path(paths["response_job"]),
        Path(paths["review_job"]),
        *sorted((Path(paths["source_root"]) / "sbsi").glob("*.py")),
        *sorted((Path(paths["blendemu_root"]) / "blendemu").glob("*.py")),
    ]
    for path in small_inputs:
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(path)
    if not Path(paths["catalogue_root"]).is_dir():
        raise FileNotFoundError(paths["catalogue_root"])

    root.mkdir(parents=False)
    for name in ("lookup", "response", "logs"):
        (root / name).mkdir()
    protocol = {
        "version": "minimal-selected-existing-response-v1",
        "created_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "purpose": (
            "One frozen generalization response test of the response-blind-qualified "
            "direct selected-event head on existing cases140--239."
        ),
        "candidate": {
            "probability": (
                "equal three-seed p_S ensemble for S=usable and MAG_AUTO<25.8 "
                "and convolved FLUX_RADIUS>=0.75arcsec"
            ),
            "shape": "unchanged nine-input physical disk flow plus fixed R_blend shift",
            "features": list(FEATURES),
            "new_features": [],
            "probability_parameter_count_per_seed": 4865,
        },
        "sample": {
            "cases": list(CASES),
            "shear_component": "g1",
            "h": 0.05,
            "ellipticity_norm_max": None,
            "mag_auto_max_exclusive": 25.8,
            "flux_radius_min_inclusive_arcsec": 0.75,
            "flux_radius_semantics": (
                "SExtractor circular half-light radius on the PSF-convolved "
                "detection image; not deconvolved ngmix T"
            ),
        },
        "estimator": (
            "sum_x p_S(x) E_flow[e|U,C,x] / sum_x p_S(x); the cut probability "
            "is not multiplied into p_S a second time"
        ),
        "sampling": {
            "draws_per_row_per_qmc_seed": support_result[
                "selected_draws_per_row"
            ],
            "qmc_seeds": list(QMC_SEEDS),
            "common_random_numbers_between_legs": True,
            "zero_acceptance_completion": (
                "zero midpoint plus physical-disk worst-case bound"
            ),
        },
        "gates": {
            "equivalence_band_percent": [-0.3, 0.3],
            "case_ci95_expanded_by_student_t_qmc_and_support_strictly_inside": True,
            "central_abs_m_percent_below": 0.3,
            "central_abs_truth_population_contribution_percent_below": 0.3,
            "central_abs_measurement_residual_contribution_percent_below": 0.3,
            "qmc_standard_error_percent_at_most": 0.005,
            "qmc_confidence_level": 0.95,
            "qmc_interval": "Student-t with degrees of freedom n_qmc_seeds-1",
            "zero_acceptance_support_bound_percent_at_most": 0.005,
        },
        "scope": {
            "existing_catalogues_only": True,
            "new_image_simulation": False,
            "response_used_for_candidate_choice": False,
            "single_response_endpoint": True,
            "interpretation": (
                "A pass is existing-catalogue g1 generalization only and licenses "
                "production of missing independent g2/g0 acceptance catalogues; "
                "it is not fresh multi-axis 0.3% promotion."
            ),
        },
        "paths": paths,
        "input_sha256": {str(path): sha256(path) for path in small_inputs},
    }
    write_json(root / "protocol.json", protocol)
    print(f"MINIMAL_SELECTED_RESPONSE_PREPARED root={root}", flush=True)


def load_protocol(output_root: str | Path, *, verify_inputs: bool = True) -> dict:
    root = Path(output_root).resolve()
    protocol = load_json(root / "protocol.json")
    if (
        protocol.get("version") != "minimal-selected-existing-response-v1"
        or Path(protocol["paths"]["output_root"]) != root
        or protocol["sample"]["cases"] != list(CASES)
        or protocol["sample"]["h"] != 0.05
        or protocol["sample"]["flux_radius_min_inclusive_arcsec"] != 0.75
        or protocol["sampling"]["qmc_seeds"] != list(QMC_SEEDS)
    ):
        raise RuntimeError("existing-response protocol changed")
    if verify_inputs:
        for rendered, expected in protocol["input_sha256"].items():
            if sha256(rendered) != expected:
                raise RuntimeError(f"response input changed: {rendered}")
        _validate_upstream(
            Path(protocol["paths"]["selected_root"]),
            Path(protocol["paths"]["support_root"]),
        )
    return protocol


def _validate_response(result: dict, protocol: dict) -> None:
    sample = protocol["sample"]
    sampling = protocol["sampling"]
    if (
        result.get("analysis") != "ConstGold complete forward likelihood response"
        or result.get("probability_factorization") != "direct_selected"
        or result.get("cases") != list(CASES)
        or result.get("shear_component") != "g1"
        or result.get("h") != sample["h"]
        or result.get("analysis_primary_re_min") is not None
        or result.get("direct_selected_conditional_support") is None
    ):
        raise RuntimeError("wrong response identity or factorization")
    if result.get("measured_output_cut") != {
        "ellipticity_norm_max": None,
        "measured_mag_auto_max": sample["mag_auto_max_exclusive"],
        "measured_radius_arcsec_min": sample[
            "flux_radius_min_inclusive_arcsec"
        ],
        "measured_log_flux_radius_min": float(np.log(0.75 / 0.2)),
    }:
        raise RuntimeError("response cut changed")
    observed_sampling = result.get("sampling", {})
    if (
        observed_sampling.get("draws_per_atom_per_flow_seed")
        != sampling["draws_per_row_per_qmc_seed"]
        or observed_sampling.get("seeds") != list(QMC_SEEDS)
        or observed_sampling.get("qmc") is not True
        or observed_sampling.get("common_random_numbers_between_legs") is not True
    ):
        raise RuntimeError("response QMC design changed")

    flow_path = Path(protocol["paths"]["flow_checkpoint"])
    models = result.get("models", [])
    if (
        len(models) != len(QMC_SEEDS)
        or [item.get("sampling_seed") for item in models] != list(QMC_SEEDS)
        or any(Path(item.get("path", "")) != flow_path for item in models)
        or any(item.get("sha256") != sha256(flow_path) for item in models)
    ):
        raise RuntimeError("response flow identity changed")
    selected_root = Path(protocol["paths"]["selected_root"])
    head_paths = tuple(
        selected_root / f"models/seed{seed}/selected.pt" for seed in MODEL_SEEDS
    )
    detector = result.get("detection_model", {})
    if (
        detector.get("aggregation") != "arithmetic_mean_probability"
        or tuple(map(Path, detector.get("classifiers", []))) != head_paths
        or detector.get("classifier_sha256")
        != [sha256(path) for path in head_paths]
        or tuple(detector.get("features", [])) != FEATURES
    ):
        raise RuntimeError("direct selected-event ensemble identity changed")


def _catalogue_accounting(result: dict) -> dict:
    per_case = result.get("per_case", {})
    expected_cases = [str(case) for case in CASES]
    if list(per_case) != expected_cases:
        raise RuntimeError("response per-case coverage or order changed")

    by_leg = {}
    for leg in ("plus", "minus"):
        totals = {name: 0 for name in POPULATION_COUNT_FIELDS}
        for case in expected_cases:
            record = per_case[case].get("measured_population", {}).get(leg, {})
            for name in POPULATION_COUNT_FIELDS:
                value = record.get(name)
                if type(value) is not int or value < 0:
                    raise RuntimeError(
                        f"invalid catalogue accounting: case={case} leg={leg} field={name}"
                    )
                totals[name] += value
            if (
                record["actual_detected_count"]
                != record["actual_usable_shape_count"]
                + record["dropped_nonfinite_measurements"]
                + record["dropped_invalid_shape"]
                or record["actual_usable_shape_count"]
                != record["actual_selected_count"]
                + record["dropped_measured_cut"]
            ):
                raise RuntimeError(
                    f"catalogue drop accounting does not close: case={case} leg={leg}"
                )
        by_leg[leg] = totals

    matched = {"matched_detected_count": 0, "matched_selected_count": 0}
    for case in expected_cases:
        record = per_case[case]["measured_population"]
        for name in matched:
            value = record.get(name)
            if type(value) is not int or value < 0:
                raise RuntimeError(
                    f"invalid matched accounting: case={case} field={name}"
                )
            matched[name] += value
    return {"case_count": len(expected_cases), "by_shear_leg": by_leg, **matched}


def summarize(result: dict, protocol: dict) -> dict:
    _validate_response(result, protocol)
    measured = np.asarray(
        result["pooled"]["measured"]["complete"]["response_first_column"], float
    )
    model_record = result["pooled"]["predicted"]["complete"]
    model = np.asarray(model_record["response_first_column"], float)
    truth = result["pooled"]["truth_shape_population"]
    truth_actual = np.asarray(
        truth["actual_detection_and_selection"]["response_first_column"], float
    )
    truth_model = np.asarray(
        truth["predicted_detection_and_selection"]["response_first_column"], float
    )
    by_qmc = np.asarray(model_record["response_by_sampling_seed"], float)
    if (
        any(value.shape != (2,) for value in (measured, model, truth_actual, truth_model))
        or by_qmc.shape != (len(QMC_SEEDS), 2)
        or not np.isfinite(
            np.concatenate((measured, model, truth_actual, truth_model, by_qmc.ravel()))
        ).all()
        or model[0] <= 0.0
        or np.any(by_qmc[:, 0] <= 0.0)
    ):
        raise RuntimeError("malformed response arrays")

    total_error = model - measured
    truth_error = truth_model - truth_actual
    residual_error = total_error - truth_error
    m_percent = 100.0 * (measured[0] / model[0] - 1.0)
    contribution = {
        "truth_population": float(-100.0 * truth_error[0] / model[0]),
        "measurement_residual": float(-100.0 * residual_error[0] / model[0]),
        "total": float(-100.0 * total_error[0] / model[0]),
    }
    if not np.isclose(contribution["total"], m_percent, rtol=1e-12, atol=1e-12):
        raise RuntimeError("response decomposition does not close")

    acceptance = result["response_acceptance"]
    case_ci = np.asarray(acceptance["case_bootstrap_ci95_percent"], float)
    expanded_ci = np.asarray(
        acceptance["case_ci95_expanded_for_qmc_and_support_percent"], float
    )
    m_by_qmc = 100.0 * (measured[0] / by_qmc[:, 0] - 1.0)
    qmc_se = float(m_by_qmc.std(ddof=1) / np.sqrt(len(m_by_qmc)))
    qmc_confidence = float(protocol["gates"]["qmc_confidence_level"])
    qmc_degrees_of_freedom = len(QMC_SEEDS) - 1
    qmc_multiplier = float(
        stdtrit(qmc_degrees_of_freedom, 0.5 + qmc_confidence / 2.0)
    )
    qmc_half_width = qmc_multiplier * qmc_se
    support_bound = float(acceptance["zero_acceptance_support_bound_percent"])
    expected_expanded_ci = np.asarray(
        [
            case_ci[0] - qmc_half_width - support_bound,
            case_ci[1] + qmc_half_width + support_bound,
        ],
        dtype=float,
    )
    if (
        case_ci.shape != (2,)
        or expanded_ci.shape != (2,)
        or not np.isfinite(np.concatenate((case_ci, expanded_ci))).all()
        or not np.isclose(acceptance["m_percent"], m_percent)
        or not np.isclose(acceptance["qmc_seed_standard_error_percent"], qmc_se)
        or acceptance.get("qmc_seed_confidence_level") != qmc_confidence
        or acceptance.get("qmc_seed_degrees_of_freedom")
        != qmc_degrees_of_freedom
        or not np.isclose(
            acceptance.get("qmc_seed_student_t_multiplier", np.nan),
            qmc_multiplier,
        )
        or not np.isclose(
            acceptance.get("qmc_seed_ci95_half_width_percent", np.nan),
            qmc_half_width,
        )
        or not np.allclose(expanded_ci, expected_expanded_ci)
    ):
        raise RuntimeError("response acceptance does not reconstruct")

    gates = {
        "selected_event_proper_score_gate": True,
        "development_support_gate": True,
        "expanded_case_ci95_strictly_inside_0p3": bool(
            expanded_ci[0] > -0.3 and expanded_ci[1] < 0.3
        ),
        "central_abs_m_below_0p3": bool(abs(m_percent) < 0.3),
        "truth_component_abs_below_0p3": bool(
            abs(contribution["truth_population"]) < 0.3
        ),
        "residual_component_abs_below_0p3": bool(
            abs(contribution["measurement_residual"]) < 0.3
        ),
        "qmc_standard_error_at_most_0p005": bool(qmc_se <= 0.005),
        "zero_acceptance_support_bound_at_most_0p005": bool(
            support_bound <= 0.005
        ),
    }
    if acceptance["passes_expanded_ci95_equivalence"] is not gates[
        "expanded_case_ci95_strictly_inside_0p3"
    ]:
        raise RuntimeError("stored expanded equivalence gate disagrees")
    passed = all(gates.values())
    return {
        "schema_version": 1,
        "analysis": "minimal direct-selected existing g1 amplitude decision",
        "existing_catalogue_generalization_not_fresh_multi_axis_promotion": True,
        "sample": (
            "cases140--239; g1 +/-0.05; MAG_AUTO<25.8; convolved "
            "FLUX_RADIUS>=0.75arcsec; no measured-|e| cut"
        ),
        "response": {
            "measured_R11": float(measured[0]),
            "model_R11": float(model[0]),
            "m_percent": float(m_percent),
            "case_standard_error_percent": float(
                acceptance["case_bootstrap_standard_error_percent"]
            ),
            "case_ci95_percent": case_ci.tolist(),
            "qmc_standard_error_percent": qmc_se,
            "qmc_seed_degrees_of_freedom": qmc_degrees_of_freedom,
            "qmc_seed_student_t_multiplier": qmc_multiplier,
            "qmc_seed_ci95_half_width_percent": qmc_half_width,
            "zero_acceptance_support_bound_percent": support_bound,
            "expanded_case_ci95_percent": expanded_ci.tolist(),
            "m_by_qmc_seed_percent": m_by_qmc.tolist(),
        },
        "model_minus_image_response": total_error.tolist(),
        "truth_population_response_error": truth_error.tolist(),
        "measurement_residual_response_error": residual_error.tolist(),
        "m_contribution_percent": contribution,
        "catalogue_accounting": _catalogue_accounting(result),
        "gates": gates,
        "passes_existing_g1_endpoint": passed,
        "decision": (
            "survives_existing_g1_endpoint_only"
            if passed
            else "reject_direct_selected_candidate_without_new_simulation"
        ),
        "next_action": (
            "keep the candidate frozen and produce only the missing independent g2/g0 acceptance evidence"
            if passed
            else "use the prespecified truth/residual routing; do not add features unless minimal branches fail"
        ),
    }


def review(args: argparse.Namespace) -> None:
    root = Path(args.output_root).resolve()
    protocol = load_protocol(root)
    response_path = root / "response" / "result.json"
    destination = root / "response" / "decision.json"
    if destination.exists():
        raise FileExistsError(f"preserving existing decision: {destination}")
    result = summarize(load_json(response_path), protocol)
    result["input_artifacts"] = {
        "protocol": {"path": str(root / "protocol.json"), "sha256": sha256(root / "protocol.json")},
        "response": {"path": str(response_path), "sha256": sha256(response_path)},
    }
    write_json(destination, result)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), flush=True)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    for name in (
        "output_root", "selected_root", "support_root", "catalogue_root",
        "crowd_lookup", "emulator_model", "emulator_metadata", "blendemu_root",
        "builder", "evaluator", "lookup_job", "response_job", "review_job",
    ):
        prepare_parser.add_argument(f"--{name.replace('_', '-')}", type=Path, required=True)
    review_parser = commands.add_parser("review")
    review_parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.command == "prepare":
        prepare(args)
    else:
        review(args)


if __name__ == "__main__":
    main()
