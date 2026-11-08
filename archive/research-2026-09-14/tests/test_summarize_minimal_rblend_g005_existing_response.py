import copy

import pytest

from scripts.summarize_minimal_rblend_g005_existing_response import (
    CASES,
    CATALOGUE_ROOT,
    CROWD_LOOKUP,
    DETECTOR_HASHES,
    DETECTOR_PATHS,
    FLOW_FEATURES,
    FLOW_PATH,
    FLOW_SHA256,
    LOOKUP_PATHS,
    QMC_SEEDS,
    summarize,
)


def _payload(
    *,
    model_r=1.001,
    measured_r=1.0,
    truth_error=0.001,
    case_ci=(-0.2, 0.1),
):
    truth_actual = 0.99
    m = 100.0 * (measured_r / model_r - 1.0)
    ci_pass = case_ci[0] > -0.3 and case_ci[1] < 0.3
    result = {
        "analysis": "ConstGold complete forward likelihood response",
        "shear_component": "g1",
        "cases": list(CASES),
        "h": 0.05,
        "analysis_primary_re_min": None,
        "measured_output_cut": {
            "ellipticity_norm_max": None,
            "measured_mag_auto_max": 25.8,
            "measured_radius_arcsec_min": 0.75,
        },
        "sampling": {
            "qmc": True,
            "common_random_numbers_between_legs": True,
            "draws_per_atom_per_flow_seed": 64,
            "seeds": list(QMC_SEEDS),
        },
        "models": [
            {
                "path": str(FLOW_PATH),
                "sha256": FLOW_SHA256,
                "sampling_seed": seed,
            }
            for seed in QMC_SEEDS
        ],
        "detection_model": {
            "aggregation": "arithmetic_mean_probability",
            "classifiers": list(map(str, DETECTOR_PATHS)),
            "classifier_sha256": list(DETECTOR_HASHES),
            "features": list(FLOW_FEATURES),
            "neighbour_selection": "nearest",
            "impact_exponent": 2.0,
        },
        "inputs": {
            "constgold_root": str(CATALOGUE_ROOT),
            "crowd_lookup": str(CROWD_LOOKUP),
            "rblend_lookup": list(map(str, LOOKUP_PATHS)),
        },
        "response_acceptance": {
            "m_percent": m,
            "case_bootstrap_standard_error_percent": 0.08,
            "case_bootstrap_ci95_percent": list(case_ci),
            "passes_marginal_ci95_equivalence": ci_pass,
        },
        "pooled": {
            "measured": {
                "complete": {"response_first_column": [measured_r, 0.0]}
            },
            "predicted": {
                "complete": {
                    "response_first_column": [model_r, 0.0],
                    "response_by_sampling_seed": [
                        [model_r, 0.0] for _ in QMC_SEEDS
                    ],
                }
            },
            "truth_shape_population": {
                "actual_detection_and_selection": {
                    "response_first_column": [truth_actual, 0.0]
                },
                "predicted_detection_and_selection": {
                    "response_first_column": [truth_actual + truth_error, 0.0]
                },
            },
        },
    }
    prior = {
        "analysis": "minimal shared-R_blend reused response decision",
        "passes_reused_structural_screen": True,
        "decision": "survives_reused_screen_only",
        "response": {
            "m_percent": -0.1,
            "case_ci95_percent": [-0.5, 0.3],
        },
        "m_contribution_percent": {
            "truth_population": -0.1,
            "measurement_residual": 0.0,
        },
    }
    return result, prior


def test_existing_g005_accepts_only_full_equivalence_and_components():
    result, prior = _payload()
    summary = summarize(result, prior)
    assert summary["passes_existing_g1_endpoints"] is True
    assert summary["decision"] == "survives_existing_g1_endpoints_only"
    assert summary["response"]["qmc_standard_error_percent"] == 0.0
    assert summary["m_contribution_percent"]["total"] == pytest.approx(
        summary["response"]["m_percent"]
    )


def test_existing_g005_rejects_central_pass_when_ci_crosses_band():
    result, prior = _payload(case_ci=(-0.35, 0.1))
    summary = summarize(result, prior)
    assert abs(summary["response"]["m_percent"]) < 0.3
    assert summary["gates"]["case_ci95_strictly_inside_0p3"] is False
    assert summary["passes_existing_g1_endpoints"] is False


def test_existing_g005_rejects_component_cancellation():
    result, prior = _payload(truth_error=0.004)
    summary = summarize(result, prior)
    assert abs(summary["response"]["m_percent"]) < 0.3
    assert summary["gates"]["truth_component_abs_below_0p3"] is False
    assert summary["passes_existing_g1_endpoints"] is False


def test_existing_g005_rejects_identity_drift():
    result, prior = _payload()
    drifted = copy.deepcopy(result)
    drifted["measured_output_cut"]["measured_radius_arcsec_min"] = 0.365
    with pytest.raises(RuntimeError, match="not the frozen no-\\|e\\| sample"):
        summarize(drifted, prior)
