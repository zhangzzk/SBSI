import copy

import numpy as np
import pytest

from scripts.summarize_minimal_rblend_flow_existing_response import (
    CASES,
    CATALOGUE_ROOT,
    CROWD_LOOKUP,
    DETECTOR_HASHES,
    DETECTOR_PATHS,
    FLOW_FEATURES,
    FLOW_PATH,
    FLOW_SHA256,
    LOOKUP_ROOT,
    QMC_SEEDS,
    summarize,
)


def _payload(*, model_r=1.001, measured_r=1.0, truth_error=0.001):
    truth_actual = 0.99
    m = 100.0 * (measured_r / model_r - 1.0)
    result = {
        "analysis": "ConstGold complete forward likelihood response",
        "shear_component": "g1",
        "cases": list(CASES),
        "h": 0.02,
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
        },
        "inputs": {
            "constgold_root": str(CATALOGUE_ROOT),
            "crowd_lookup": str(CROWD_LOOKUP),
            "rblend_lookup": [
                str(LOOKUP_ROOT / f"case{case}/rblend.feather") for case in CASES
            ],
        },
        "response_acceptance": {
            "m_percent": m,
            "case_bootstrap_standard_error_percent": 0.2,
            "case_bootstrap_ci95_percent": [-0.4, 0.4],
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
    development = {
        "passed": True,
        "response_used": False,
        "arms": {"flow8_rblend": {"selected_sha256": FLOW_SHA256}},
    }
    baseline = {
        "screen_only_not_promotion": True,
        "cases": list(CASES),
        "candidate": {
            "m_percent": {"value": -0.372},
            "component_contribution_to_m_pp": {
                "truth": {"value": -0.549},
                "residual": {"value": 0.177},
            },
        },
    }
    return result, development, baseline


def test_reused_screen_accepts_small_non_cancelling_components():
    result, development, baseline = _payload()
    summary = summarize(result, development, baseline)
    assert summary["passes_reused_structural_screen"] is True
    assert summary["decision"] == "survives_reused_screen_only"
    assert summary["response"]["qmc_standard_error_percent"] == 0.0
    assert summary["m_contribution_percent"]["total"] == pytest.approx(
        summary["response"]["m_percent"]
    )


def test_reused_screen_rejects_cancelling_out_of_band_component():
    result, development, baseline = _payload(
        model_r=1.001, measured_r=1.0, truth_error=0.004
    )
    summary = summarize(result, development, baseline)
    assert abs(summary["response"]["m_percent"]) < 0.3
    assert summary["gates"]["truth_component_abs_below_0p3"] is False
    assert summary["passes_reused_structural_screen"] is False


def test_reused_screen_rejects_identity_drift():
    result, development, baseline = _payload()
    drifted = copy.deepcopy(result)
    drifted["sampling"]["seeds"][-1] = 9999
    with pytest.raises(RuntimeError, match="QMC protocol drifted"):
        summarize(drifted, development, baseline)
