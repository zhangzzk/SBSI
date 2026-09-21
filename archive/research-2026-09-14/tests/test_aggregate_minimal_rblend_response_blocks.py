import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.aggregate_minimal_rblend_response_blocks import aggregate


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _protocol(tmp_path: Path) -> Path:
    candidate = {
        "measurement_model": {"sha256": "flow-sha"},
        "emulator": {
            "detection_model": {
                "members": [
                    {"sha256": "detector-1"},
                    {"sha256": "detector-2"},
                    {"sha256": "detector-3"},
                ]
            }
        },
        "observing_conditions": {"pixel_size": 0.2},
    }
    candidate_path = tmp_path / "candidate.json"
    _write_json(candidate_path, candidate)
    candidate_sha = hashlib.sha256(candidate_path.read_bytes()).hexdigest()
    protocol = {
        "protocol": "minimal-rblend-fresh-acceptance-v1",
        "candidate": {
            "likelihood_config": str(candidate_path),
            "likelihood_config_sha256": candidate_sha,
        },
        "simulation": {
            "case_first": 0,
            "case_last": 1,
            "block_count": 2,
            "block_cases": 1,
            "legs": [
                {"g1": -0.02, "g2": 0.0},
                {"g1": 0.02, "g2": 0.0},
            ],
        },
        "population": {
            "measured_selection": {
                "measured_mag_auto_max_exclusive": 25.8,
                "measured_radius_min_inclusive_arcsec": 0.75,
            }
        },
        "response_evaluation": {
            "draws_per_qmc_group": 2,
            "qmc_seeds": [11, 12],
            "bootstrap_replicates": 100,
            "bootstrap_seed": 7,
            "qmc_numerical_gate": {
                "maximum_standard_error_percent": 0.5,
                "student_t_95_multiplier": 1.0,
            },
            "component_robustness_gate": {
                "maximum_absolute_central_contribution_percent": 0.3,
            },
        },
    }
    protocol_path = tmp_path / "protocol.json"
    _write_json(protocol_path, protocol)
    return protocol_path


def _prediction(direction: int) -> dict:
    model_numerator = [0.016 * direction, 0.002 * direction]
    truth_numerator = [0.018 * direction, 0.003 * direction]
    result = {
        branch: {"denominator": 1.0, "numerator": model_numerator}
        for branch in (
            "matched_selected",
            "matched_detected",
            "matched_detected_selected",
            "complete",
            "detection_only",
            "selection_only",
            "unweighted",
        )
    }
    result["truth_detection"] = {
        "denominator": 1.0,
        "numerator": truth_numerator,
    }
    result["truth_complete"] = {
        "denominator": 1.0,
        "numerator": truth_numerator,
    }
    return result


def _block(case: int) -> dict:
    measured_leg = {
        "1": {"count": 1, "sum": [0.02, 0.004]},
        "-1": {"count": 1, "sum": [-0.02, -0.004]},
    }
    truth_leg = {
        "1": {"count": 1, "sum": [0.02, 0.004]},
        "-1": {"count": 1, "sum": [-0.02, -0.004]},
    }
    report = {
        "truth_population_count": 1,
        "crossmatch_rows_without_shape": 0,
        "shape_rows_without_crossmatch": 1,
        "crossmatched_rows_outside_finite_prior_support": 0,
        "actual_detected_count": 1,
        "actual_usable_shape_count": 1,
        "actual_selected_count": 1,
        "dropped_nonfinite_measurements": 0,
        "dropped_invalid_shape": 0,
        "dropped_measured_cut": 0,
    }
    response_sufficient = {
        "measured": {
            branch: measured_leg
            for branch in (
                "matched_selected",
                "matched_detected",
                "matched_detected_selected",
                "complete",
                "detection_only",
            )
        },
        "truth_actual": {
            "detection": truth_leg,
            "detection_and_selection": truth_leg,
        },
        "predicted": [
            {"1": _prediction(+1), "-1": _prediction(-1)},
            {"1": _prediction(+1), "-1": _prediction(-1)},
        ],
    }
    return {
        "analysis": "ConstGold complete forward likelihood response",
        "shear_component": "g1",
        "h": 0.02,
        "analysis_primary_re_min": None,
        "circularized_condition": "fixed pre-shear circularized radius",
        "measured_output_cut": {
            "ellipticity_norm_max": None,
            "measured_mag_auto_max": 25.8,
            "measured_radius_arcsec_min": 0.75,
            "measured_log_flux_radius_min": float(np.log(0.75 / 0.2)),
        },
        "prediction_definition": "frozen",
        "response_ladder_definition": {"complete": "frozen"},
        "detection_note": "frozen",
        "truth_shape_population_definition": {"shape": "frozen"},
        "models": [
            {"sha256": "flow-sha", "sampling_seed": 11},
            {"sha256": "flow-sha", "sampling_seed": 12},
        ],
        "detection_model": {
            "aggregation": "arithmetic_mean_probability",
            "classifier_sha256": ["detector-1", "detector-2", "detector-3"],
            "features": [f"x{i}" for i in range(9)],
        },
        "sampling": {
            "draws_per_atom_per_flow_seed": 2,
            "seeds": [11, 12],
            "qmc": True,
            "common_random_numbers_between_legs": True,
        },
        "inputs": {"crowding_source": "truth_scene", "crowd_lookup": None},
        "cases": [case],
        "per_case": {
            str(case): {
                "response_sufficient": response_sufficient,
                "measured_population": {"plus": report, "minus": report},
            }
        },
    }


def test_aggregate_reconstructs_case_level_acceptance_and_components(tmp_path):
    protocol = _protocol(tmp_path)
    paths = []
    for case in (0, 1):
        path = tmp_path / f"block{case}.json"
        _write_json(path, _block(case))
        paths.append(path)

    result = aggregate(paths, protocol, bootstrap_replicates=20)

    acceptance = result["response_acceptance"]
    assert acceptance["measured_response"] == pytest.approx(1.0)
    assert acceptance["model_response"] == pytest.approx(0.8)
    assert acceptance["m_percent"] == pytest.approx(25.0)
    assert acceptance["qmc_standard_error_percent"] == 0.0
    np.testing.assert_allclose(
        acceptance["case_ci95_expanded_by_qmc_95_half_width_percent"],
        acceptance["case_bootstrap_ci95_percent"],
    )
    assert not acceptance["passes_marginal_ci95_equivalence"]
    contributions = result["response_components"][
        "diagonal_m_contribution_percent"
    ]
    assert contributions["truth_population"] == pytest.approx(12.5)
    assert contributions["measurement_residual"] == pytest.approx(12.5)
    assert contributions["total"] == pytest.approx(25.0)
    assert not result["response_components"]["robustness_gate"]["passes"]
    assert result["row_accounting"]["plus"]["shape_rows_without_crossmatch"] == 2
    assert len(result["block_results"]) == 2


def test_aggregate_rejects_identity_drift_between_blocks(tmp_path):
    protocol = _protocol(tmp_path)
    first = tmp_path / "block0.json"
    second = tmp_path / "block1.json"
    _write_json(first, _block(0))
    altered = _block(1)
    altered["sampling"]["seeds"] = [11, 99]
    _write_json(second, altered)

    with pytest.raises(RuntimeError, match="identity mismatch: sampling"):
        aggregate([first, second], protocol, bootstrap_replicates=20)


def test_aggregate_uses_all_qmc_seeds_and_gates_their_observed_spread(tmp_path):
    protocol = _protocol(tmp_path)
    paths = []
    for case in (0, 1):
        payload = _block(case)
        second_run = payload["per_case"][str(case)]["response_sufficient"][
            "predicted"
        ][1]
        second_run["1"]["complete"]["numerator"] = [0.020, 0.002]
        second_run["-1"]["complete"]["numerator"] = [-0.020, -0.002]
        path = tmp_path / f"block{case}.json"
        _write_json(path, payload)
        paths.append(path)

    result = aggregate(paths, protocol, bootstrap_replicates=20)
    acceptance = result["response_acceptance"]

    assert acceptance["model_response"] == pytest.approx(0.9)
    assert acceptance["m_percent"] == pytest.approx(100.0 * (1.0 / 0.9 - 1.0))
    np.testing.assert_allclose(acceptance["m_by_qmc_seed_percent"], [25.0, 0.0])
    assert acceptance["qmc_standard_error_percent"] == pytest.approx(12.5)
    assert not acceptance["passes_qmc_numerical_gate"]
    assert not acceptance["passes_marginal_ci95_equivalence"]
