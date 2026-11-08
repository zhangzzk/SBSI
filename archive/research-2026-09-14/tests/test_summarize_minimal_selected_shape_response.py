import json
from types import SimpleNamespace

import numpy as np

import scripts.summarize_minimal_selected_shape_response as reviewer
from scripts.summarize_minimal_selected_shape_response import CASES, SEEDS


def test_reused_response_protocol_constants_are_minimal_and_fixed():
    assert CASES == tuple(range(120, 140))
    assert SEEDS == (7301, 7302, 7303, 7304)
    assert np.sqrt(len(SEEDS)) == 2


def _response(*, model_path, model_hash, m, truth_contribution, support=0.0):
    measured = 1.0
    predicted = measured / (1.0 + m / 100.0)
    total_error = predicted - measured
    truth_error = -truth_contribution * predicted / 100.0
    return {
        "analysis": "ConstGold complete forward likelihood response",
        "cases": list(CASES),
        "shear_component": "g1",
        "h": 0.02,
        "probability_factorization": "direct_selected",
        "measured_output_cut": {
            "ellipticity_norm_max": None,
            "measured_mag_auto_max": 25.8,
            "measured_radius_arcsec_min": 0.75,
        },
        "pooled": {
            "measured": {
                "complete": {"response_first_column": [measured, 0.0]},
            },
            "predicted": {
                "complete": {
                    "response_first_column": [predicted, 0.0],
                    "response_by_sampling_seed": [
                        [predicted, 0.0] for _ in SEEDS
                    ],
                },
            },
            "truth_shape_population": {
                "actual_detection_and_selection": {
                    "response_first_column": [0.5, 0.0],
                },
                "predicted_detection_and_selection": {
                    "response_first_column": [0.5 + truth_error, 0.0],
                },
            },
        },
        "response_acceptance": {
            "case_bootstrap_standard_error_percent": 0.01,
            "case_bootstrap_ci95_percent": [m - 0.02, m + 0.02],
        },
        "direct_selected_conditional_support": {
            "pooled_ensemble_response_abs_bound": [support, 0.0],
        },
        "models": [
            {"path": model_path, "sha256": model_hash} for _ in SEEDS
        ],
    }


def _run_review(tmp_path, monkeypatch, *, baseline, candidate):
    (tmp_path / "response").mkdir()
    (tmp_path / "response/baseline.json").write_text(json.dumps(baseline))
    (tmp_path / "response/selected_shape.json").write_text(json.dumps(candidate))
    protocol = {
        "sample": {
            "ellipticity_norm_cut": None,
            "flux_radius": ">=0.75arcsec convolved SExtractor circular half-light radius",
            "mag_auto": "<25.8",
        },
        "arms": {"baseline": "/models/base.pt", "selected_shape": "/models/shape.pt"},
        "input_hashes": {"baseline_flow": "base-hash", "candidate_flow": "shape-hash"},
    }
    monkeypatch.setattr(reviewer, "load_protocol", lambda root, verify: protocol)
    reviewer.review(SimpleNamespace(root=tmp_path))
    return json.loads((tmp_path / "decision.json").read_text())


def test_reviewer_passes_only_as_a_reused_screen(tmp_path, monkeypatch):
    baseline = _response(
        model_path="/models/base.pt", model_hash="base-hash", m=0.5,
        truth_contribution=0.1,
    )
    candidate = _response(
        model_path="/models/shape.pt", model_hash="shape-hash", m=0.1,
        truth_contribution=0.05,
    )

    decision = _run_review(
        tmp_path, monkeypatch, baseline=baseline, candidate=candidate
    )

    assert decision["passes_reused_mechanism_screen"] is True
    assert all(decision["gates"].values())
    assert decision["screen_only_not_promotion"] is True
    assert decision["new_image_simulation"] is False
    assert "cannot establish fresh" in decision["limitations"][2]


def test_reviewer_rejects_favorable_total_from_component_cancellation(
    tmp_path, monkeypatch
):
    baseline = _response(
        model_path="/models/base.pt", model_hash="base-hash", m=0.5,
        truth_contribution=0.0,
    )
    candidate = _response(
        model_path="/models/shape.pt", model_hash="shape-hash", m=0.1,
        truth_contribution=0.4,
    )

    decision = _run_review(
        tmp_path, monkeypatch, baseline=baseline, candidate=candidate
    )

    assert abs(decision["selected_shape"]["m_percent"]) < 0.3
    assert decision["gates"]["truth_component_abs_below_0p3"] is False
    assert decision["passes_reused_mechanism_screen"] is False
    assert decision["decision"] == "stop_minimal_selected_shape_branch"
