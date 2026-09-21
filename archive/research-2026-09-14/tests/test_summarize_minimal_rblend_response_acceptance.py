import hashlib
import json
from pathlib import Path

import pytest

from scripts.summarize_minimal_rblend_response_acceptance import summarize


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _protocol(tmp_path: Path) -> Path:
    path = tmp_path / "protocol.json"
    _write(
        path,
        {
            "protocol": "minimal-rblend-fresh-acceptance-v1",
            "simulation": {
                "case_count": 10,
                "legs": [
                    {"g1": -0.02, "g2": 0.0},
                    {"g1": 0.02, "g2": 0.0},
                    {"g1": -0.04, "g2": 0.0},
                    {"g1": 0.04, "g2": 0.0},
                    {"g1": 0.0, "g2": -0.02},
                    {"g1": 0.0, "g2": 0.02},
                    {"g1": 0.0, "g2": -0.04},
                    {"g1": 0.0, "g2": 0.04},
                ],
            },
            "response_evaluation": {
                "qmc_numerical_gate": {
                    "maximum_standard_error_percent": 0.005,
                    "student_t_95_multiplier": 3.0,
                },
                "component_robustness_gate": {
                    "maximum_absolute_central_contribution_percent": 0.3,
                },
                "gate": "all four frozen gates pass",
            },
        },
    )
    return path


def _endpoint(
    protocol: Path,
    component: str,
    h: float,
    *,
    passed: bool = True,
    contributions: tuple[float, float] = (0.01, 0.01),
) -> dict:
    expanded = [-0.20, 0.20] if passed else [-0.31, 0.20]
    qmc_se = 0.001
    actual_pass = expanded[0] > -0.3 and expanded[1] < 0.3 and qmc_se <= 0.005
    return {
        "analysis": "minimal-rblend blockwise response acceptance",
        "protocol": {"sha256": hashlib.sha256(protocol.read_bytes()).hexdigest()},
        "endpoint": {"shear_component": component, "h": h, "case_count": 10},
        "response_acceptance": {
            "m_percent": 0.02,
            "case_bootstrap_standard_error_percent": 0.05,
            "case_bootstrap_ci95_percent": [-0.18, 0.18],
            "qmc_standard_error_percent": qmc_se,
            "qmc_standard_error_limit_percent": 0.005,
            "qmc_student_t_95_multiplier": 3.0,
            "qmc_95_half_width_percent": 0.003,
            "case_ci95_expanded_by_qmc_95_half_width_percent": expanded,
            "passes_qmc_numerical_gate": True,
            "equivalence_band_percent": [-0.3, 0.3],
            "passes_marginal_ci95_equivalence": actual_pass,
        },
        "cross_response": {"component": "cross", "measured": 0.0, "model": 0.0},
        "response_components": {
            "diagonal_m_contribution_percent": {
                "truth_population": contributions[0],
                "measurement_residual": contributions[1],
                "total": sum(contributions),
            },
            "robustness_gate": {
                "maximum_absolute_central_contribution_percent": 0.3,
                "observed_maximum_absolute_central_contribution_percent": max(
                    abs(contributions[0]), abs(contributions[1])
                ),
                "opposite_sign_central_contributions": (
                    contributions[0] * contributions[1] < 0.0
                ),
                "passes": max(abs(contributions[0]), abs(contributions[1])) < 0.3,
            },
        },
        "row_accounting": {"plus": {"actual_selected_count": 5}},
    }


def test_summary_makes_one_four_endpoint_decision(tmp_path):
    protocol = _protocol(tmp_path)
    paths = []
    for component, h in (("g1", 0.02), ("g1", 0.04), ("g2", 0.02), ("g2", 0.04)):
        path = tmp_path / f"{component}_{h}.json"
        _write(path, _endpoint(protocol, component, h))
        paths.append(path)

    result = summarize(paths[::-1], protocol)

    assert result["decision"] == "pass_to_full_inference"
    assert result["passes_all_four_endpoints"]
    assert result["failed_endpoints"] == []
    assert [(item["shear_component"], item["h"]) for item in result["endpoint_results"]] == [
        ("g1", 0.02), ("g1", 0.04), ("g2", 0.02), ("g2", 0.04)
    ]


def test_summary_rejects_and_detects_stored_decision_drift(tmp_path):
    protocol = _protocol(tmp_path)
    paths = []
    for component, h in (("g1", 0.02), ("g1", 0.04), ("g2", 0.02), ("g2", 0.04)):
        path = tmp_path / f"{component}_{h}.json"
        _write(path, _endpoint(protocol, component, h, passed=(component, h) != ("g2", 0.04)))
        paths.append(path)
    result = summarize(paths, protocol)
    assert result["decision"] == "reject_candidate"
    assert result["failed_endpoints"] == [{"shear_component": "g2", "h": 0.04}]

    altered = _endpoint(protocol, "g2", 0.04)
    altered["response_acceptance"]["passes_marginal_ci95_equivalence"] = False
    _write(paths[-1], altered)
    with pytest.raises(RuntimeError, match="stored equivalence decision mismatch"):
        summarize(paths, protocol)


def test_summary_rejects_out_of_band_component_cancellation(tmp_path):
    protocol = _protocol(tmp_path)
    paths = []
    for component, h in (("g1", 0.02), ("g1", 0.04), ("g2", 0.02), ("g2", 0.04)):
        path = tmp_path / f"{component}_{h}.json"
        component_terms = (0.40, -0.38) if (component, h) == ("g2", 0.04) else (0.01, 0.01)
        _write(
            path,
            _endpoint(protocol, component, h, contributions=component_terms),
        )
        paths.append(path)

    result = summarize(paths, protocol)

    assert result["decision"] == "reject_candidate"
    endpoint = result["endpoint_results"][-1]
    assert endpoint["passes_response_equivalence"]
    assert not endpoint["component_robustness_gate"]["passes"]
    assert not endpoint["passes"]
