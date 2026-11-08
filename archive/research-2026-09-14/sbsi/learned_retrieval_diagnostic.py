"""Frozen stop/go gates for broad learned support plus conditioned DM3.

The learned score is used only to choose a deterministic support ``R``.  The
centre target is summed exactly on ``R`` and the existing fixed-component DM3
proposal estimates ``R``'s complement.  This module deliberately describes a
development diagnostic, not an inference release or a learned importance
probability.
"""

from __future__ import annotations

from hashlib import sha256
import struct
from typing import Any

import numpy as np
import pandas as pd

# The cont.303 restructure removed ``TailProposalRecipe`` and its
# ``TAIL_PROPOSAL_RECIPE`` instance from ``catalogue_sampling`` when the tilted
# and priority proposals replaced the four-component mixture.  The gated run
# recorded below genuinely used that mixture, so its recipe is kept here as a
# literal historical record.  Substituting today's proposal would misdescribe
# the run this protocol freezes.
TAIL_PROPOSAL_RECIPE_RECORD = {
    "name": "conditioned_dm_mis_t1_t2_prior_4_3_1_v2",
    "component_names": ("proxy_t1", "proxy_t2", "prior"),
    "component_weights": (0.5, 0.375, 0.125),
    "component_betas": (1.0, 0.5, None),
    "component_cycle": (0, 1, 0, 2, 1, 0, 1, 0),
}


LEARNED_RETRIEVAL_CENTER_LABEL = (
    "SBSI/complement-learned-retrieval-center-development/v1"
)
LEARNED_RETRIEVAL_CENTER_CHECKPOINT_SHA256 = (
    "1338cf30b2d500316b31304530a623674ad56ca94416cf29699b5645e366d39b"
)
LEARNED_RETRIEVAL_CENTER_MOCK_SHA256 = {
    "measurements.parquet": (
        "fdfecacc13adabd6005e21150ab0dc74767cad34da3d3ea95e362515ce38365b"
    ),
    "truth.parquet": (
        "78d1c8e02f43e1ac4673b2ca309b484d3565de3b9400bf736ca98bf584e5dc30"
    ),
    "likelihood_mock_manifest.json": (
        "5c097c98fa820177738156f6dd3290daaf68594dd90b2e80733f083b019d9a48"
    ),
}
LEARNED_RETRIEVAL_CENTER_BASELINE = {
    "mean_total_relative_error_squared": 4.6256941067948046e-4,
    "total_relative_error_squared_p90_higher": 4.45668696553694e-4,
    "method_seconds": 91.04022970038932,
}
LEARNED_RETRIEVAL_CENTER_GATE_THRESHOLDS = {
    "objects": 512,
    "pareto_k_below_0_5_count": 461,
    "pareto_k_below_0_7_count": 507,
    "pareto_k_at_least_1_count": 0,
    "complement_relative_error_median": 0.00882,
    "complement_relative_error_p90_higher": 0.0268,
    "retrieval_capture_p10": 0.70,
    "retrieval_capture_median": 0.85,
    "retrieval_capture_minimum": 0.50,
    "efficiency_mean": 0.90,
    "efficiency_p90_higher": 0.90,
    "absolute_delta_log_total_evidence_median": 0.01,
    "absolute_delta_log_total_evidence_p90_higher": 0.05,
    "stratum_relative_crosscheck_max": 1.0e-5,
    "absolute_signed_aggregate_z": 2.5,
    "exact_z_rms": [0.8, 1.2],
    "exact_z_within_1_96_count": 58,
}
LEARNED_RETRIEVAL_METHOD_TIME_COMPONENTS = (
    "candidate_query",
    "prefilter_flow",
    "tail_proposal_draw",
    "tail_proposal_flow",
    "reduction",
)
LEARNED_RETRIEVAL_CENTER_PROTOCOL = {
    "label": LEARNED_RETRIEVAL_CENTER_LABEL,
    "universe_start": 25_000,
    "universe_stop": 25_512,
    "center": [0.013610377120038044, 0.002740148300964851],
    "initial_strategy": "mean_observed_shape",
    "n_candidates": 524_288,
    "prefilter_candidates": 524_288,
    "n_global_draws": 16_384,
    "proposal_seed": 8_701,
    "production_seed": 7_008_704,
    "candidate_backend": "torch",
    "diagnostic_object_chunk": 8,
    "flow_atom_chunk": 4_096,
    "retrieval_object_chunk": 8,
    "retrieval_atom_chunk": 262_144,
    "retrieval_matmul_precision": "high",
    "tail_proposal_atom_chunk": 262_144,
    "tail_proposal_object_chunk": 32,
    "checkpoint_sha256": LEARNED_RETRIEVAL_CENTER_CHECKPOINT_SHA256,
    "tail_proposal_recipe": dict(TAIL_PROPOSAL_RECIPE_RECORD),
    "exact_selection_manifest_sha256": (
        "6abcc361aec9ada6010f2a91a68aa70440904873518182445c2a712ae69bad6a"
    ),
    "exact_selection_count": 64,
    "exact_atom_chunk": 65_536,
    "exact_object_chunk": 4,
    "active_atoms": 12_760_990,
    "candidate_calls": 64,
    "retrieval_batches": 64,
    "retrieval_objects": 512,
    "stratum_flow_evaluations": 268_435_456,
    "prior_flow_evaluations": 8_388_608,
    "tail_flow_evaluations": 8_388_608,
    "exact_flow_evaluations": 816_703_360,
    "run_flow_evaluations": 1_101_916_032,
    "mock_input_sha256": dict(LEARNED_RETRIEVAL_CENTER_MOCK_SHA256),
}


def _gate(observed: Any, operator: str, threshold: Any, passed: bool) -> dict:
    return {
        "observed": observed,
        "operator": operator,
        "threshold": threshold,
        "passed": bool(passed),
    }


def _higher_percentile(values: np.ndarray, percentile: float) -> float:
    return float(np.percentile(values, percentile, method="higher"))


def _is_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def learned_retrieval_run_manifest_sha256(frame: pd.DataFrame) -> str:
    """Hash per-object learned support, centre score, and exclusion identity."""

    required = (
        "object_index",
        "learned_retrieval_indices_sha256",
        "learned_retrieval_center_log_target_sha256",
        "conditioned_exclusion_indices_sha256",
    )
    if any(name not in frame for name in required):
        raise ValueError("learned-retrieval manifest columns are incomplete")
    rows = []
    for record in frame.loc[:, required].to_dict(orient="records"):
        object_id = record["object_index"]
        if (
            isinstance(object_id, (bool, np.bool_))
            or not isinstance(object_id, (int, np.integer))
            or object_id < 0
            or object_id > np.iinfo(np.int64).max
        ):
            raise ValueError("learned-retrieval manifest ids must be non-negative int64")
        hashes = tuple(record[name] for name in required[1:])
        if not all(_is_sha256(value) for value in hashes):
            raise ValueError("learned-retrieval manifest contains an invalid SHA-256")
        rows.append((int(object_id), hashes))
    rows.sort(key=lambda item: item[0])
    if len({object_id for object_id, _ in rows}) != len(rows):
        raise ValueError("learned-retrieval manifest object ids must be unique")
    digest = sha256()
    digest.update(b"SBSI/learned-retrieval-center-run/v1\0")
    digest.update(struct.pack(">Q", len(rows)))
    for object_id, hashes in rows:
        digest.update(struct.pack(">Q", object_id))
        for value in hashes:
            digest.update(bytes.fromhex(value))
    return digest.hexdigest()


def _exact_int64_vector(values) -> np.ndarray | None:
    """Return a non-negative, exactly integer-typed int64 vector."""

    try:
        if isinstance(values, pd.Series):
            if not pd.api.types.is_integer_dtype(values.dtype):
                return None
            raw = values.to_numpy()
        else:
            raw = np.asarray(values)
            if raw.dtype.kind not in "iu" or raw.ndim != 1:
                return None
        if raw.ndim != 1:
            return None
        scalars = raw.tolist()
        if any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or value < 0
            or value > np.iinfo(np.int64).max
            for value in scalars
        ):
            return None
        return np.asarray(scalars, dtype=np.int64)
    except (TypeError, ValueError, OverflowError):
        return None


def _observed_protocol(diagnostic) -> dict | None:
    metadata = diagnostic.metadata
    learned = metadata.get("learned_retrieval_center")
    exact = metadata.get("exact_oracle")
    if not isinstance(learned, dict) or not isinstance(exact, dict):
        return None
    try:
        return {
            "label": learned["label"],
            "universe_start": int(metadata["observation_start"]),
            "universe_stop": int(metadata["observation_stop"]),
            "center": [float(value) for value in metadata["center"]],
            "initial_strategy": learned["initial_strategy"],
            "n_candidates": int(metadata["n_candidates"]),
            "prefilter_candidates": int(metadata["prefilter_candidates"]),
            "n_global_draws": int(metadata["n_global_draws"]),
            "proposal_seed": int(metadata["seed"]) - 7_000_003,
            "production_seed": int(metadata["seed"]),
            "candidate_backend": metadata["candidate_backend"],
            "diagnostic_object_chunk": int(metadata["object_chunk"]),
            "flow_atom_chunk": int(metadata["flow_atom_chunk"]),
            "retrieval_object_chunk": int(learned["retrieval_object_chunk"]),
            "retrieval_atom_chunk": int(learned["retrieval_atom_chunk"]),
            "retrieval_matmul_precision": learned[
                "retrieval_matmul_precision"
            ],
            "tail_proposal_atom_chunk": int(
                metadata["tail_proposal_atom_chunk"]
            ),
            "tail_proposal_object_chunk": int(
                metadata["tail_proposal_object_chunk"]
            ),
            "checkpoint_sha256": learned["checkpoint_sha256"],
            "tail_proposal_recipe": metadata["tail_proposal"],
            "exact_selection_manifest_sha256": exact[
                "selection_manifest_sha256"
            ],
            "exact_selection_count": int(exact["selection_count"]),
            "exact_atom_chunk": int(exact["atom_chunk"]),
            "exact_object_chunk": int(exact["object_chunk"]),
            "active_atoms": int(exact["active_atoms"]),
            "candidate_calls": int(learned["candidate_calls"]),
            "retrieval_batches": int(learned["retrieval_batches"]),
            "retrieval_objects": int(learned["retrieval_objects"]),
            "stratum_flow_evaluations": int(
                learned["stratum_flow_evaluations"]
            ),
            "prior_flow_evaluations": int(learned["prior_flow_evaluations"]),
            "tail_flow_evaluations": int(learned["tail_flow_evaluations"]),
            "exact_flow_evaluations": int(exact["flow_evaluations"]),
            "run_flow_evaluations": int(metadata["flow_evaluations"]),
            "mock_input_sha256": learned["mock_input_sha256"],
        }
    except (KeyError, TypeError, ValueError):
        return None


def learned_retrieval_center_summary(diagnostic) -> dict:
    """Evaluate the predeclared 512-row learned-R centre stop/go protocol."""

    frame = diagnostic.table
    metadata = diagnostic.metadata
    learned = metadata.get("learned_retrieval_center")
    thresholds = LEARNED_RETRIEVAL_CENTER_GATE_THRESHOLDS
    baseline = LEARNED_RETRIEVAL_CENTER_BASELINE
    expected_objects = int(thresholds["objects"])

    if not isinstance(learned, dict):
        raise ValueError("diagnostic lacks learned-retrieval centre metadata")

    def values(name: str) -> np.ndarray:
        return frame[name].to_numpy(dtype=np.float64)

    observed_protocol = _observed_protocol(diagnostic)
    protocol_matched = bool(
        observed_protocol == LEARNED_RETRIEVAL_CENTER_PROTOCOL
    )
    object_ids = _exact_int64_vector(frame.get("object_index"))
    expected_ids = np.arange(25_000, 25_512, dtype=np.int64)
    ids_exact = bool(
        object_ids is not None and np.array_equal(object_ids, expected_ids)
    )

    count_names = (
        "learned_retrieval_count",
        "learned_retrieval_unique_count",
        "learned_retrieval_active_count",
        "tail_draws_in_learned_retrieval",
        "draws_tail_proposal_complement",
        "draws_tail_proposal_component_proxy_t1",
        "draws_tail_proposal_component_proxy_t2",
        "draws_tail_proposal_component_prior",
    )
    counts = {name: _exact_int64_vector(frame.get(name)) for name in count_names}
    integer_counts = all(value is not None for value in counts.values())
    widths_ok = bool(
        integer_counts
        and np.all(counts["learned_retrieval_count"] == 524_288)
        and np.all(counts["learned_retrieval_unique_count"] == 524_288)
        and np.all(counts["learned_retrieval_active_count"] == 524_288)
    )
    tail_exclusion_ok = bool(
        integer_counts
        and np.all(counts["tail_draws_in_learned_retrieval"] == 0)
    )
    component_counts_ok = bool(
        integer_counts
        and np.all(counts["draws_tail_proposal_complement"] == 16_384)
        and np.all(counts["draws_tail_proposal_component_proxy_t1"] == 8_192)
        and np.all(counts["draws_tail_proposal_component_proxy_t2"] == 6_144)
        and np.all(counts["draws_tail_proposal_component_prior"] == 2_048)
    )
    hash_names = (
        "learned_retrieval_indices_sha256",
        "learned_retrieval_center_log_target_sha256",
        "conditioned_exclusion_indices_sha256",
    )
    hashes_well_formed = bool(
        len(frame) == expected_objects
        and all(name in frame for name in hash_names)
        and all(_is_sha256(value) for name in hash_names for value in frame[name])
        and frame["learned_retrieval_indices_sha256"].astype(str).tolist()
        == frame["conditioned_exclusion_indices_sha256"].astype(str).tolist()
    )
    try:
        recomputed_run_manifest = learned_retrieval_run_manifest_sha256(frame)
    except (KeyError, TypeError, ValueError):
        recomputed_run_manifest = None
    run_manifest_matched = bool(
        _is_sha256(learned.get("run_manifest_sha256"))
        and learned.get("run_manifest_sha256") == recomputed_run_manifest
    )
    prefilter_remainder_zero = bool(
        len(frame) == expected_objects
        and np.array_equal(values("mass_prefilter_complement"), np.zeros(len(frame)))
    )

    total_mass = values("mass_stratum") + values(
        "mass_complement_tail_proposal"
    )
    tail_mass = values("mass_complement_tail_proposal")
    tail_se = values("mass_complement_tail_proposal_standard_error")
    finite_positive = bool(
        len(frame) == expected_objects
        and np.isfinite(total_mass).all()
        and np.isfinite(tail_mass).all()
        and np.isfinite(tail_se).all()
        and (total_mass > 0.0).all()
        and (tail_mass > 0.0).all()
        and (tail_se > 0.0).all()
    )

    integrity_gates = {
        "protocol_identity_matched": _gate(
            protocol_matched, "is", True, protocol_matched
        ),
        "absolute_object_ids_exact": _gate(ids_exact, "is", True, ids_exact),
        "integer_typed_count_columns": _gate(
            integer_counts, "is", True, integer_counts
        ),
        "learned_support_unique_active_width_all_objects": _gate(
            widths_ok, "is", True, widths_ok
        ),
        "conditioned_tail_excludes_learned_support": _gate(
            tail_exclusion_ok, "is", True, tail_exclusion_ok
        ),
        "fixed_dm3_component_counts_all_objects": _gate(
            component_counts_ok, "is", True, component_counts_ok
        ),
        "support_and_exclusion_hashes_match": _gate(
            hashes_well_formed, "is", True, hashes_well_formed
        ),
        "run_manifest_recomputes": _gate(
            run_manifest_matched, "is", True, run_manifest_matched
        ),
        "prefilter_remainder_is_zero": _gate(
            prefilter_remainder_zero, "is", True, prefilter_remainder_zero
        ),
        "finite_positive_masses_and_standard_errors": _gate(
            finite_positive, "is", True, finite_positive
        ),
    }

    tail_k = values("pareto_k_tail_proposal_complement")
    finite_k = np.isfinite(tail_k)
    below_0_5 = int(np.sum(finite_k & (tail_k < 0.5)))
    below_0_7 = int(np.sum(finite_k & (tail_k < 0.7)))
    at_least_1 = int(np.sum(finite_k & (tail_k >= 1.0)))
    relative_error = np.divide(
        tail_se,
        np.abs(tail_mass),
        out=np.full(len(frame), np.nan, dtype=np.float64),
        where=np.abs(tail_mass) > 0.0,
    )
    relative_error_finite = bool(
        len(relative_error) == expected_objects and np.isfinite(relative_error).all()
    )
    relative_error_median = (
        float(np.median(relative_error)) if relative_error_finite else float("nan")
    )
    relative_error_p90 = (
        _higher_percentile(relative_error, 90)
        if relative_error_finite
        else float("nan")
    )
    tail_gates = {
        "pareto_k_defined_all_objects": _gate(
            int((~finite_k).sum()),
            "undefined_count ==",
            0,
            len(tail_k) == expected_objects and finite_k.all(),
        ),
        "pareto_k_below_0_5_count": _gate(
            below_0_5,
            ">=",
            thresholds["pareto_k_below_0_5_count"],
            below_0_5 >= thresholds["pareto_k_below_0_5_count"],
        ),
        "pareto_k_below_0_7_count": _gate(
            below_0_7,
            ">=",
            thresholds["pareto_k_below_0_7_count"],
            below_0_7 >= thresholds["pareto_k_below_0_7_count"],
        ),
        "pareto_k_at_least_1_count": _gate(
            at_least_1,
            "==",
            thresholds["pareto_k_at_least_1_count"],
            at_least_1 == thresholds["pareto_k_at_least_1_count"],
        ),
        "complement_relative_error_median": _gate(
            relative_error_median,
            "<=",
            thresholds["complement_relative_error_median"],
            np.isfinite(relative_error_median)
            and relative_error_median
            <= thresholds["complement_relative_error_median"],
        ),
        "complement_relative_error_p90_higher": _gate(
            relative_error_p90,
            "<=",
            thresholds["complement_relative_error_p90_higher"],
            np.isfinite(relative_error_p90)
            and relative_error_p90
            <= thresholds["complement_relative_error_p90_higher"],
        ),
    }

    selected = frame.loc[frame["exact_oracle_selected"].astype(bool)]
    capture = (
        selected["mass_stratum_exact_oracle"].to_numpy(dtype=np.float64)
        / selected["mass_total_exact_oracle"].to_numpy(dtype=np.float64)
    )
    capture_finite = bool(
        len(capture) == 64 and np.isfinite(capture).all() and (capture > 0).all()
    )
    capture_p10 = (
        float(np.percentile(capture, 10)) if capture_finite else float("nan")
    )
    capture_median = (
        float(np.median(capture)) if capture_finite else float("nan")
    )
    capture_minimum = float(np.min(capture)) if capture_finite else float("nan")
    capture_gates = {
        "selected_exact_object_count": _gate(len(selected), "==", 64, len(selected) == 64),
        "retrieval_capture_p10": _gate(
            capture_p10,
            ">=",
            thresholds["retrieval_capture_p10"],
            np.isfinite(capture_p10)
            and capture_p10 >= thresholds["retrieval_capture_p10"],
        ),
        "retrieval_capture_median": _gate(
            capture_median,
            ">=",
            thresholds["retrieval_capture_median"],
            np.isfinite(capture_median)
            and capture_median >= thresholds["retrieval_capture_median"],
        ),
        "retrieval_capture_minimum": _gate(
            capture_minimum,
            ">=",
            thresholds["retrieval_capture_minimum"],
            np.isfinite(capture_minimum)
            and capture_minimum >= thresholds["retrieval_capture_minimum"],
        ),
    }

    delta_log = np.abs(
        selected["delta_log_total_evidence_tail_minus_exact"].to_numpy(
            dtype=np.float64
        )
    )
    exact_z = selected["mass_complement_tail_minus_exact_z"].to_numpy(
        dtype=np.float64
    )
    stratum_crosscheck = selected[
        "mass_stratum_exact_oracle_relative_error"
    ].to_numpy(dtype=np.float64)
    difference = (
        selected["mass_complement_tail_proposal"].to_numpy(dtype=np.float64)
        - selected["mass_complement_exact_oracle"].to_numpy(dtype=np.float64)
    )
    selected_se = selected[
        "mass_complement_tail_proposal_standard_error"
    ].to_numpy(dtype=np.float64)
    exact_finite = bool(
        len(selected) == 64
        and np.isfinite(delta_log).all()
        and np.isfinite(exact_z).all()
        and np.isfinite(stratum_crosscheck).all()
        and np.isfinite(difference).all()
        and np.isfinite(selected_se).all()
        and (selected_se > 0.0).all()
    )
    median_delta = float(np.median(delta_log)) if exact_finite else float("nan")
    p90_delta = (
        _higher_percentile(delta_log, 90) if exact_finite else float("nan")
    )
    aggregate_denominator = float(np.sqrt(np.sum(np.square(selected_se))))
    aggregate_z = (
        float(np.sum(difference) / aggregate_denominator)
        if exact_finite and aggregate_denominator > 0.0
        else float("nan")
    )
    z_rms = (
        float(np.sqrt(np.mean(np.square(exact_z))))
        if exact_finite
        else float("nan")
    )
    within_1_96 = int(np.sum(np.isfinite(exact_z) & (np.abs(exact_z) <= 1.96)))
    stratum_crosscheck_max = (
        float(np.max(stratum_crosscheck)) if exact_finite else float("nan")
    )
    exact_gates = {
        "stratum_relative_crosscheck_max": _gate(
            stratum_crosscheck_max,
            "<=",
            thresholds["stratum_relative_crosscheck_max"],
            np.isfinite(stratum_crosscheck_max)
            and stratum_crosscheck_max
            <= thresholds["stratum_relative_crosscheck_max"],
        ),
        "absolute_delta_log_total_evidence_median": _gate(
            median_delta,
            "<=",
            thresholds["absolute_delta_log_total_evidence_median"],
            np.isfinite(median_delta)
            and median_delta
            <= thresholds["absolute_delta_log_total_evidence_median"],
        ),
        "absolute_delta_log_total_evidence_p90_higher": _gate(
            p90_delta,
            "<=",
            thresholds["absolute_delta_log_total_evidence_p90_higher"],
            np.isfinite(p90_delta)
            and p90_delta
            <= thresholds["absolute_delta_log_total_evidence_p90_higher"],
        ),
        "absolute_signed_aggregate_z": _gate(
            abs(aggregate_z) if np.isfinite(aggregate_z) else float("nan"),
            "<=",
            thresholds["absolute_signed_aggregate_z"],
            np.isfinite(aggregate_z)
            and abs(aggregate_z) <= thresholds["absolute_signed_aggregate_z"],
        ),
        "exact_z_rms": _gate(
            z_rms,
            "in_closed_interval",
            thresholds["exact_z_rms"],
            np.isfinite(z_rms)
            and thresholds["exact_z_rms"][0]
            <= z_rms
            <= thresholds["exact_z_rms"][1],
        ),
        "exact_z_within_1_96_count": _gate(
            within_1_96,
            ">=",
            thresholds["exact_z_within_1_96_count"],
            within_1_96 >= thresholds["exact_z_within_1_96_count"],
        ),
    }

    total_relative_error = np.divide(
        tail_se,
        np.abs(total_mass),
        out=np.full(len(frame), np.nan, dtype=np.float64),
        where=np.abs(total_mass) > 0.0,
    )
    squared = np.square(total_relative_error)
    finite_efficiency = bool(
        len(squared) == expected_objects and np.isfinite(squared).all()
    )
    phase_seconds = metadata.get("phase_seconds", {})
    method_seconds = float(learned.get("proposal_initialization_seconds", np.nan))
    try:
        method_seconds += sum(
            float(phase_seconds[name])
            for name in LEARNED_RETRIEVAL_METHOD_TIME_COMPONENTS
        )
    except (KeyError, TypeError, ValueError):
        method_seconds = float("nan")
    efficiency_mean = (
        float(np.mean(squared))
        * method_seconds
        / (
            baseline["mean_total_relative_error_squared"]
            * baseline["method_seconds"]
        )
        if finite_efficiency and np.isfinite(method_seconds) and method_seconds > 0.0
        else float("nan")
    )
    efficiency_p90 = (
        _higher_percentile(squared, 90)
        * method_seconds
        / (
            baseline["total_relative_error_squared_p90_higher"]
            * baseline["method_seconds"]
        )
        if finite_efficiency and np.isfinite(method_seconds) and method_seconds > 0.0
        else float("nan")
    )
    efficiency_gates = {
        "mean_relative_variance_time_efficiency": _gate(
            efficiency_mean,
            "<=",
            thresholds["efficiency_mean"],
            np.isfinite(efficiency_mean)
            and efficiency_mean <= thresholds["efficiency_mean"],
        ),
        "p90_higher_relative_variance_time_efficiency": _gate(
            efficiency_p90,
            "<=",
            thresholds["efficiency_p90_higher"],
            np.isfinite(efficiency_p90)
            and efficiency_p90 <= thresholds["efficiency_p90_higher"],
        ),
    }

    families = {
        "integrity": integrity_gates,
        "retrieval_capture": capture_gates,
        "tail_reliability": tail_gates,
        "exact_calibration": exact_gates,
        "efficiency": efficiency_gates,
    }
    passed = all(
        gate["passed"]
        for family in families.values()
        for gate in family.values()
    )
    return {
        "protocol": {
            "expected": LEARNED_RETRIEVAL_CENTER_PROTOCOL,
            "observed": observed_protocol,
            "matched": protocol_matched,
        },
        "method_seconds": method_seconds,
        "proposal_initialization_seconds": learned.get(
            "proposal_initialization_seconds"
        ),
        "complement_relative_error_median": relative_error_median,
        "complement_relative_error_p90_higher": relative_error_p90,
        "retrieval_capture_p10": capture_p10,
        "retrieval_capture_median": capture_median,
        "retrieval_capture_minimum": capture_minimum,
        "absolute_delta_log_total_evidence_median": median_delta,
        "absolute_delta_log_total_evidence_p90_higher": p90_delta,
        "stratum_relative_crosscheck_max": stratum_crosscheck_max,
        "signed_aggregate_z": aggregate_z,
        "exact_z_rms": z_rms,
        "exact_z_within_1_96_count": within_1_96,
        "efficiency_mean": efficiency_mean,
        "efficiency_p90_higher": efficiency_p90,
        "gates": {
            "predeclared": protocol_matched,
            "thresholds_predeclared": True,
            **families,
            "passed": bool(passed),
            "policy": (
                "all integrity, retrieval, tail, exact-calibration, and fair-"
                "efficiency gates must pass before implementing the stencil bridge"
            ),
        },
    }


__all__ = [
    "LEARNED_RETRIEVAL_CENTER_BASELINE",
    "LEARNED_RETRIEVAL_CENTER_CHECKPOINT_SHA256",
    "LEARNED_RETRIEVAL_CENTER_GATE_THRESHOLDS",
    "LEARNED_RETRIEVAL_CENTER_LABEL",
    "LEARNED_RETRIEVAL_CENTER_PROTOCOL",
    "learned_retrieval_center_summary",
    "learned_retrieval_run_manifest_sha256",
]
