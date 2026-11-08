import copy

import numpy as np
import pandas as pd
import pytest

from scripts import review_constgold_fixed_g0_response as review_module
from scripts.build_constgold_fixed_g0_blend_lookup import verify_anchor_truth
from scripts.evaluate_constgold_fixed_g0_response import (
    parse_subsets,
    response,
    sufficient,
)


def test_verify_anchor_truth_accepts_identity_compatible_source():
    truth = pd.DataFrame(
        {
            "index": [7, 4],
            "axis_ratio": [1.0, 1.0],
            "position_angle": [0.0, 30.0],
            "sersic_n": [1.5, 2.0],
            "r": [24.0, 25.0],
            "Re": [0.2, 0.4],
        }
    )
    ids = np.array([4, 7])
    context = np.array(
        [
            [0.0, 0.0, 2.0, 25.0, 0.4, 1.0, 2.0, 3.0],
            [0.0, 0.0, 1.5, 24.0, 0.2, 4.0, 5.0, 6.0],
        ]
    )
    aligned = verify_anchor_truth(truth, ids, context)
    assert aligned["index"].tolist() == [4, 7]


def test_sufficient_and_antithetic_response():
    plus = sufficient(np.array([[0.10, 0.0], [0.20, 0.0]]), np.array([1.0, 3.0]))
    minus = sufficient(np.array([[-0.10, 0.0], [0.00, 0.0]]), np.array([1.0, 3.0]))
    np.testing.assert_allclose(response(plus, minus, 0.02), [5.0, 0.0])


def test_parse_subsets_uses_half_open_case_bounds():
    cases = tuple(range(40, 90))
    parsed = parse_subsets(["ten=40:50", "twenty=40:60"], cases)
    assert parsed["ten"] == tuple(range(40, 50))
    assert parsed["twenty"] == tuple(range(40, 60))


def synthetic_review_inputs(tmp_path):
    cases = (40, 41)
    measured = {}
    predicted = {}
    reports = {}
    audit_reports = {}
    for case in cases:
        measured[str(case)] = {}
        predicted[str(case)] = {}
        for branch in review_module.BRANCHES:
            measured[str(case)][branch] = {
                "plus": {"denominator": 10.0, "numerator": [0.2, 0.0]},
                "minus": {"denominator": 10.0, "numerator": [-0.2, 0.0]},
            }
            predicted[str(case)][branch] = {
                "plus": {"denominator": 10.0, "numerator": [0.196, 0.0]},
                "minus": {"denominator": 10.0, "numerator": [-0.196, 0.0]},
            }
        reports[str(case)] = {
            "anchor_rows": 10,
            "matched_usable_rows": 10,
            "plus": {
                "anchor_raw_detected": 10,
                "anchor_usable": 10,
                "cross_without_shape": 0,
            },
            "minus": {
                "anchor_raw_detected": 10,
                "anchor_usable": 10,
                "cross_without_shape": 0,
            },
            "unmatched_usable_rows": {"plus_only": 0, "minus_only": 0},
        }
        audit_reports[str(case)] = {
            "anchor_rows": 10,
            "matched_usable_rows": 10,
            "plus_raw_detected": 10,
            "minus_raw_detected": 10,
            "plus_usable": 10,
            "minus_usable": 10,
            "plus_only_usable": 0,
            "minus_only_usable": 0,
            "plus_cross_without_shape": 0,
            "minus_cross_without_shape": 0,
        }
    provenance_paths = []
    for name in ("flow.pt", "rblend.feather", "c0.pt", "c1.pt", "c2.pt"):
        path = tmp_path / name
        path.touch()
        provenance_paths.append(path)
    branch_summary = {}
    for branch in review_module.BRANCHES:
        branch_summary[branch] = {
            "definition": branch,
            "measured_response": [1.0, 0.0],
            "predicted_response": [0.98, 0.0],
            "m_percent": 100.0 * (1.0 / 0.98 - 1.0),
            "bootstrap": {
                "unit": "ConstGold case",
                "replicates": 100,
                "seed": 1,
                "m_standard_error_percentage_points": 0.1,
                "m_ci95_percent": [1.8, 2.3],
                "predicted_minus_measured_standard_error": [0.01, 0.01],
                "predicted_minus_measured_ci95": [[-0.03, -0.01], [0.01, 0.03]],
            },
        }
    anchor_hashes = {str(case): f"anchor-{case}" for case in cases}
    result = {
        "format_version": 1,
        "domain": {
            "anchor": "g=0 detected, MAG_AUTO<25.8, FLUX_RADIUS>3.0 pixels",
            "truth_analysis_cut": None,
            "sheared_leg_magnitude_radius_recut": False,
        },
        "h": 0.02,
        "draws": 64,
        "sampling_seed": 7,
        "common_antithetic_latents_across_legs_and_models": True,
        "models": {
            "model": {
                "flow": str(provenance_paths[0]),
                "flow_sha256": "hash",
                "rblend": str(provenance_paths[1]),
                "rblend_sha256": "hash",
            }
        },
        "classifiers": [
            {"path": str(path), "sha256": "hash"}
            for path in provenance_paths[2:]
        ],
        "case_reports": reports,
        "anchor_sha256_by_case": anchor_hashes,
        "per_case_sufficient": {
            "measured": measured,
            "predicted": {"model": predicted},
        },
        "subsets": {
            "both": {
                "cases": list(cases),
                "n_cases": 2,
                "models": {"model": branch_summary},
                "paired_model_differences": {},
            }
        },
    }
    audit = {
        "h": 0.02,
        "case_reports": audit_reports,
        "anchor_sha256_by_case": anchor_hashes,
        "per_case_sufficient": {
            case: {
                branch: values
                for branch, values in branches.items()
                if branch != "modeled_usable"
            }
            for case, branches in measured.items()
        },
        "summaries": {
            "both": {
                "cases": list(cases),
                "branches": {
                    "matched_usable": {"measured_response": [1.0, 0.0]},
                    "actual_usable_flags": {"measured_response": [1.0, 0.0]},
                },
            }
        },
    }
    truth = {
        "h": 0.02,
        "truth_analysis_cut": None,
        "case_reports": {
            str(case): {
                "truth_rows": 20,
                "anchor_rows": 10,
                "missing_anchor_rows": 0,
                "invariant_truth_sha256": f"{case:064x}",
            }
            for case in cases
        },
    }
    return result, audit, truth


def test_review_cross_checks_result_against_independent_audit(tmp_path, monkeypatch):
    result, audit, truth = synthetic_review_inputs(tmp_path)
    monkeypatch.setattr(review_module, "file_sha256", lambda _: "hash")
    reviewed = review_module.review(result, audit, truth, ("model",))
    assert reviewed["validation"] == "passed"
    assert len(reviewed["rows"]) == 3

    corrupted = copy.deepcopy(result)
    corrupted["per_case_sufficient"]["measured"]["40"]["matched_usable"][
        "plus"
    ]["numerator"][0] += 1.0
    with pytest.raises(ValueError, match="numerator mismatch"):
        review_module.review(corrupted, audit, truth, ("model",))
