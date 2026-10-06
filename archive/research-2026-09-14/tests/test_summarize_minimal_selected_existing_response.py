import hashlib
import json

import numpy as np
import pytest

from scripts.summarize_minimal_selected_existing_response import (
    FEATURES,
    MODEL_SEEDS,
    QMC_SEEDS,
    _catalogue_accounting,
    _validate_support_artifacts,
    summarize,
)


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_support_decision_is_reconstructed_from_every_case_artifact(tmp_path):
    (tmp_path / "support").mkdir()
    (tmp_path / "protocol.json").write_text("{}\n")
    protocol_hash = _digest(tmp_path / "protocol.json")
    protocol = {
        "development_cases": [100, 101],
        "candidate_draws_per_row": [64, 256],
        "qmc_seeds": [1, 2],
        "planned_response_h": 0.05,
        "support_gate": {"maximum_response_bound": 5.0e-5},
    }
    for case in protocol["development_cases"]:
        records = {}
        for seed in protocol["qmc_seeds"]:
            records[str(seed)] = {}
            for leg in range(2):
                records[str(seed)][f"leg{leg}"] = {
                    "64": {
                        "total_event_probability_mass": 10.0,
                        "missing_numerator_abs_bound": [0.002, 0.0],
                    },
                    "256": {
                        "total_event_probability_mass": 10.0,
                        "missing_numerator_abs_bound": [0.00001, 0.0],
                    },
                }
        (tmp_path / "support" / f"case{case:03d}.json").write_text(
            json.dumps(
                {
                    "checks_passed": True,
                    "case": case,
                    "response_statistics_computed": False,
                    "response_cases_read": False,
                    "protocol_sha256": protocol_hash,
                    "records": records,
                }
            )
        )
    decision = {
        "selected_draws_per_row": 256,
        "summaries": {
            "64": {
                "maximum_component1_response_abs_bound": 0.004,
                "passes_support_gate": False,
            },
            "256": {
                "maximum_component1_response_abs_bound": 0.00002,
                "passes_support_gate": True,
            },
        },
    }

    paths = _validate_support_artifacts(tmp_path, protocol, decision)

    assert [path.name for path in paths] == ["case100.json", "case101.json"]


def test_direct_selected_existing_response_passes_only_expanded_gate(tmp_path):
    flow = tmp_path / "flow.pt"
    flow.write_bytes(b"flow")
    selected_root = tmp_path / "selected"
    head_paths = []
    for seed in MODEL_SEEDS:
        path = selected_root / f"models/seed{seed}/selected.pt"
        path.parent.mkdir(parents=True)
        path.write_bytes(str(seed).encode())
        head_paths.append(path)
    protocol = {
        "sample": {
            "h": 0.05,
            "mag_auto_max_exclusive": 25.8,
            "flux_radius_min_inclusive_arcsec": 0.75,
        },
        "sampling": {
            "draws_per_row_per_qmc_seed": 256,
            "qmc_seeds": list(QMC_SEEDS),
        },
        "gates": {"qmc_confidence_level": 0.95},
        "paths": {
            "flow_checkpoint": str(flow),
            "selected_root": str(selected_root),
        },
    }
    by_qmc = [[1.0, 0.0] for _ in QMC_SEEDS]
    leg_accounting = {
        "truth_population_count": 100,
        "crossmatch_rows_without_shape": 2,
        "shape_rows_without_crossmatch": 3,
        "crossmatched_rows_outside_finite_prior_support": 4,
        "actual_detected_count": 90,
        "actual_usable_shape_count": 88,
        "actual_selected_count": 80,
        "dropped_nonfinite_measurements": 1,
        "dropped_invalid_shape": 1,
        "dropped_measured_cut": 8,
    }
    result = {
        "analysis": "ConstGold complete forward likelihood response",
        "probability_factorization": "direct_selected",
        "cases": list(range(140, 240)),
        "shear_component": "g1",
        "h": 0.05,
        "analysis_primary_re_min": None,
        "direct_selected_conditional_support": {"bounded": True},
        "measured_output_cut": {
            "ellipticity_norm_max": None,
            "measured_mag_auto_max": 25.8,
            "measured_radius_arcsec_min": 0.75,
            "measured_log_flux_radius_min": float(np.log(0.75 / 0.2)),
        },
        "sampling": {
            "draws_per_atom_per_flow_seed": 256,
            "seeds": list(QMC_SEEDS),
            "qmc": True,
            "common_random_numbers_between_legs": True,
        },
        "models": [
            {
                "path": str(flow),
                "sha256": _digest(flow),
                "sampling_seed": seed,
            }
            for seed in QMC_SEEDS
        ],
        "detection_model": {
            "aggregation": "arithmetic_mean_probability",
            "classifiers": [str(path) for path in head_paths],
            "classifier_sha256": [_digest(path) for path in head_paths],
            "features": list(FEATURES),
        },
        "pooled": {
            "measured": {"complete": {"response_first_column": [1.0, 0.0]}},
            "predicted": {
                "complete": {
                    "response_first_column": [1.0, 0.0],
                    "response_by_sampling_seed": by_qmc,
                }
            },
            "truth_shape_population": {
                "actual_detection_and_selection": {
                    "response_first_column": [1.0, 0.0]
                },
                "predicted_detection_and_selection": {
                    "response_first_column": [1.0, 0.0]
                },
            },
        },
        "response_acceptance": {
            "m_percent": 0.0,
            "case_bootstrap_standard_error_percent": 0.02,
            "case_bootstrap_ci95_percent": [-0.1, 0.1],
            "qmc_seed_standard_error_percent": 0.0,
            "qmc_seed_confidence_level": 0.95,
            "qmc_seed_degrees_of_freedom": 3,
            "qmc_seed_student_t_multiplier": 3.182446305284263,
            "qmc_seed_ci95_half_width_percent": 0.0,
            "zero_acceptance_support_bound_percent": 0.001,
            "case_ci95_expanded_for_qmc_and_support_percent": [-0.101, 0.101],
            "passes_expanded_ci95_equivalence": True,
        },
        "per_case": {
            str(case): {
                "measured_population": {
                    "plus": leg_accounting,
                    "minus": leg_accounting,
                    "matched_detected_count": 85,
                    "matched_selected_count": 75,
                }
            }
            for case in range(140, 240)
        },
    }

    decision = summarize(result, protocol)

    assert decision["passes_existing_g1_endpoint"] is True
    assert decision["decision"] == "survives_existing_g1_endpoint_only"
    assert all(decision["gates"].values())
    assert decision["catalogue_accounting"]["by_shear_leg"]["plus"] == {
        name: value * 100 for name, value in leg_accounting.items()
    }
    assert decision["catalogue_accounting"]["matched_detected_count"] == 8500
    assert decision["catalogue_accounting"]["matched_selected_count"] == 7500


def test_catalogue_accounting_rejects_nonclosing_drop_counts():
    leg = {
        "truth_population_count": 100,
        "crossmatch_rows_without_shape": 0,
        "shape_rows_without_crossmatch": 0,
        "crossmatched_rows_outside_finite_prior_support": 0,
        "actual_detected_count": 90,
        "actual_usable_shape_count": 88,
        "actual_selected_count": 80,
        "dropped_nonfinite_measurements": 1,
        "dropped_invalid_shape": 1,
        "dropped_measured_cut": 8,
    }
    result = {
        "per_case": {
            str(case): {
                "measured_population": {
                    "plus": dict(leg),
                    "minus": dict(leg),
                    "matched_detected_count": 85,
                    "matched_selected_count": 75,
                }
            }
            for case in range(140, 240)
        }
    }
    result["per_case"]["140"]["measured_population"]["plus"][
        "dropped_invalid_shape"
    ] = 0

    with pytest.raises(RuntimeError, match="drop accounting does not close"):
        _catalogue_accounting(result)
