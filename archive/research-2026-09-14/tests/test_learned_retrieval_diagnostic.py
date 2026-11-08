from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from types import SimpleNamespace

import numpy as np
import pandas as pd

from sbsi.learned_retrieval_diagnostic import (
    LEARNED_RETRIEVAL_CENTER_PROTOCOL,
    learned_retrieval_center_summary,
    learned_retrieval_run_manifest_sha256,
)


def _digest(label: str, object_id: int) -> str:
    return sha256(f"{label}:{object_id}".encode()).hexdigest()


def _passing_diagnostic():
    protocol = deepcopy(LEARNED_RETRIEVAL_CENTER_PROTOCOL)
    object_ids = np.arange(
        protocol["universe_start"],
        protocol["universe_stop"],
        dtype=np.int64,
    )
    n_objects = len(object_ids)
    exact_selected = np.zeros(n_objects, dtype=bool)
    exact_selected[: protocol["exact_selection_count"]] = True

    exact_z = np.zeros(n_objects, dtype=np.float64)
    exact_z[: protocol["exact_selection_count"]] = np.resize(
        np.array([-1.0, 1.0]), protocol["exact_selection_count"]
    )
    tail_mass = np.ones(n_objects, dtype=np.float64)
    tail_se = np.full(n_objects, 1.0e-3, dtype=np.float64)
    exact_complement = tail_mass - exact_z * tail_se
    exact_total = np.full(n_objects, 10.0, dtype=np.float64)
    exact_stratum = exact_total - exact_complement

    support_hashes = [_digest("support", int(value)) for value in object_ids]
    frame = pd.DataFrame(
        {
            "object_index": object_ids,
            "learned_retrieval_count": np.full(
                n_objects, protocol["n_candidates"], dtype=np.int64
            ),
            "learned_retrieval_unique_count": np.full(
                n_objects, protocol["n_candidates"], dtype=np.int64
            ),
            "learned_retrieval_active_count": np.full(
                n_objects, protocol["n_candidates"], dtype=np.int64
            ),
            "tail_draws_in_learned_retrieval": np.zeros(
                n_objects, dtype=np.int64
            ),
            "draws_tail_proposal_complement": np.full(
                n_objects, protocol["n_global_draws"], dtype=np.int64
            ),
            "draws_tail_proposal_component_proxy_t1": np.full(
                n_objects, 8_192, dtype=np.int64
            ),
            "draws_tail_proposal_component_proxy_t2": np.full(
                n_objects, 6_144, dtype=np.int64
            ),
            "draws_tail_proposal_component_prior": np.full(
                n_objects, 2_048, dtype=np.int64
            ),
            "learned_retrieval_indices_sha256": support_hashes,
            "learned_retrieval_center_log_target_sha256": [
                _digest("log-target", int(value)) for value in object_ids
            ],
            "conditioned_exclusion_indices_sha256": support_hashes.copy(),
            "mass_prefilter_complement": np.zeros(n_objects),
            "mass_stratum": exact_stratum,
            "mass_complement_tail_proposal": tail_mass,
            "mass_complement_tail_proposal_standard_error": tail_se,
            "pareto_k_tail_proposal_complement": np.full(n_objects, 0.1),
            "exact_oracle_selected": exact_selected,
            "mass_stratum_exact_oracle": exact_stratum,
            "mass_stratum_exact_oracle_relative_error": np.zeros(n_objects),
            "mass_complement_exact_oracle": exact_complement,
            "mass_total_exact_oracle": exact_total,
            "delta_log_total_evidence_tail_minus_exact": np.zeros(n_objects),
            "mass_complement_tail_minus_exact_z": exact_z,
        }
    )

    learned = {
        "label": protocol["label"],
        "initial_strategy": protocol["initial_strategy"],
        "retrieval_object_chunk": protocol["retrieval_object_chunk"],
        "retrieval_atom_chunk": protocol["retrieval_atom_chunk"],
        "retrieval_matmul_precision": protocol["retrieval_matmul_precision"],
        "checkpoint_sha256": protocol["checkpoint_sha256"],
        "candidate_calls": protocol["candidate_calls"],
        "retrieval_batches": protocol["retrieval_batches"],
        "retrieval_objects": protocol["retrieval_objects"],
        "stratum_flow_evaluations": protocol["stratum_flow_evaluations"],
        "prior_flow_evaluations": protocol["prior_flow_evaluations"],
        "tail_flow_evaluations": protocol["tail_flow_evaluations"],
        "mock_input_sha256": deepcopy(protocol["mock_input_sha256"]),
        "proposal_initialization_seconds": 1.0,
        "run_manifest_sha256": learned_retrieval_run_manifest_sha256(frame),
    }
    exact = {
        "selection_manifest_sha256": protocol[
            "exact_selection_manifest_sha256"
        ],
        "selection_count": protocol["exact_selection_count"],
        "atom_chunk": protocol["exact_atom_chunk"],
        "object_chunk": protocol["exact_object_chunk"],
        "active_atoms": protocol["active_atoms"],
        "flow_evaluations": protocol["exact_flow_evaluations"],
    }
    metadata = {
        "observation_start": protocol["universe_start"],
        "observation_stop": protocol["universe_stop"],
        "center": deepcopy(protocol["center"]),
        "n_candidates": protocol["n_candidates"],
        "prefilter_candidates": protocol["prefilter_candidates"],
        "n_global_draws": protocol["n_global_draws"],
        "seed": protocol["production_seed"],
        "candidate_backend": protocol["candidate_backend"],
        "object_chunk": protocol["diagnostic_object_chunk"],
        "flow_atom_chunk": protocol["flow_atom_chunk"],
        "tail_proposal_atom_chunk": protocol["tail_proposal_atom_chunk"],
        "tail_proposal_object_chunk": protocol["tail_proposal_object_chunk"],
        "tail_proposal": deepcopy(protocol["tail_proposal_recipe"]),
        "flow_evaluations": protocol["run_flow_evaluations"],
        "phase_seconds": {
            "candidate_query": 1.0,
            "prefilter_flow": 1.0,
            "tail_proposal_draw": 1.0,
            "tail_proposal_flow": 1.0,
            "reduction": 1.0,
        },
        "learned_retrieval_center": learned,
        "exact_oracle": exact,
    }
    return SimpleNamespace(table=frame, metadata=metadata)


def test_learned_retrieval_center_summary_accepts_complete_protocol():
    summary = learned_retrieval_center_summary(_passing_diagnostic())

    assert summary["protocol"]["matched"] is True
    assert summary["gates"]["passed"] is True
    for family in (
        "integrity",
        "retrieval_capture",
        "tail_reliability",
        "exact_calibration",
        "efficiency",
    ):
        assert all(gate["passed"] for gate in summary["gates"][family].values())


def test_protocol_mutation_fails_integrity_family():
    diagnostic = _passing_diagnostic()
    diagnostic.metadata["learned_retrieval_center"]["label"] = "wrong-protocol"

    summary = learned_retrieval_center_summary(diagnostic)

    assert summary["protocol"]["matched"] is False
    assert not summary["gates"]["integrity"][
        "protocol_identity_matched"
    ]["passed"]
    assert summary["gates"]["passed"] is False


def test_float_count_column_fails_exact_integer_dtype_gate():
    diagnostic = _passing_diagnostic()
    diagnostic.table["learned_retrieval_count"] = diagnostic.table[
        "learned_retrieval_count"
    ].astype(np.float64)

    summary = learned_retrieval_center_summary(diagnostic)

    assert not summary["gates"]["integrity"][
        "integer_typed_count_columns"
    ]["passed"]
    assert summary["gates"]["passed"] is False


def test_support_exclusion_hash_mismatch_fails_with_fresh_manifest():
    diagnostic = _passing_diagnostic()
    diagnostic.table.loc[0, "conditioned_exclusion_indices_sha256"] = _digest(
        "different-exclusion", 25_000
    )
    diagnostic.metadata["learned_retrieval_center"][
        "run_manifest_sha256"
    ] = learned_retrieval_run_manifest_sha256(diagnostic.table)

    summary = learned_retrieval_center_summary(diagnostic)

    assert not summary["gates"]["integrity"][
        "support_and_exclusion_hashes_match"
    ]["passed"]
    assert summary["gates"]["integrity"]["run_manifest_recomputes"]["passed"]
    assert summary["gates"]["passed"] is False


def test_single_pareto_k_at_least_one_fails_tail_family():
    diagnostic = _passing_diagnostic()
    diagnostic.table.loc[0, "pareto_k_tail_proposal_complement"] = 1.0

    summary = learned_retrieval_center_summary(diagnostic)

    gate = summary["gates"]["tail_reliability"][
        "pareto_k_at_least_1_count"
    ]
    assert gate["observed"] == 1
    assert gate["passed"] is False
    assert summary["gates"]["passed"] is False


def test_single_low_capture_fails_minimum_capture_gate():
    diagnostic = _passing_diagnostic()
    diagnostic.table.loc[0, "mass_stratum_exact_oracle"] = 4.0

    summary = learned_retrieval_center_summary(diagnostic)

    gate = summary["gates"]["retrieval_capture"][
        "retrieval_capture_minimum"
    ]
    assert gate["observed"] == 0.4
    assert gate["passed"] is False
    assert summary["gates"]["passed"] is False


def test_miscalibrated_exact_z_fails_exact_calibration_family():
    diagnostic = _passing_diagnostic()
    diagnostic.table.loc[
        diagnostic.table["exact_oracle_selected"],
        "mass_complement_tail_minus_exact_z",
    ] = 0.0

    summary = learned_retrieval_center_summary(diagnostic)

    gate = summary["gates"]["exact_calibration"]["exact_z_rms"]
    assert gate["observed"] == 0.0
    assert gate["passed"] is False
    assert summary["gates"]["passed"] is False


def test_stratum_crosscheck_failure_fails_exact_calibration_family():
    diagnostic = _passing_diagnostic()
    diagnostic.table.loc[0, "mass_stratum_exact_oracle_relative_error"] = 2.0e-5

    summary = learned_retrieval_center_summary(diagnostic)

    gate = summary["gates"]["exact_calibration"][
        "stratum_relative_crosscheck_max"
    ]
    assert gate["observed"] == 2.0e-5
    assert gate["threshold"] == 1.0e-5
    assert gate["passed"] is False
    assert summary["gates"]["passed"] is False


def test_slow_method_fails_both_efficiency_gates():
    diagnostic = _passing_diagnostic()
    diagnostic.metadata["phase_seconds"]["candidate_query"] = 1.0e9

    summary = learned_retrieval_center_summary(diagnostic)

    assert not summary["gates"]["efficiency"][
        "mean_relative_variance_time_efficiency"
    ]["passed"]
    assert not summary["gates"]["efficiency"][
        "p90_higher_relative_variance_time_efficiency"
    ]["passed"]
    assert summary["gates"]["passed"] is False
