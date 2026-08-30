"""Measure complement mass and compare production tail proposals.

For every observation, the diagnostic evaluates the full prefilter exactly,
then pairs the prior-complement estimator with the production fixed
component-conditioned deterministic-MIS sampler.  Both sampled arms consume
the same per-object random record and use absolute object ids, so a partitioned
run is identical to the corresponding rows of an unsliced run.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import inspect
import struct
import time
from typing import Optional

import numpy as np
import pandas as pd
import torch

from .catalogue_sampling import (
    TAIL_PROPOSAL_RECIPE,
    pareto_tail_index,
)


EXACT_ORACLE_LABEL = "SBSI/complement-exact-oracle/v1"
EXACT_ORACLE_SELECTION_KEY_SHA256 = (
    "f02f535eab874dd6f9f6184d4040283493266074903fc6c20160b4ae88b5993e"
)
EXACT_ORACLE_UNIVERSE_SIZE = 512
EXACT_ORACLE_SELECTION_COUNT = 64
EXACT_ORACLE_ATOM_CHUNK = 65_536
EXACT_ORACLE_OBJECT_CHUNK = 4
EXACT_ORACLE_DIAGNOSTIC_SEED_OFFSET = 7_000_003
EXACT_ORACLE_CONFIRMATION = {
    "universe_start": 25_512,
    "universe_stop": 26_024,
    "center": [0.0, 0.0],
    "n_candidates": 16_384,
    "prefilter_candidates": 131_072,
    "n_global_draws": 131_072,
    "proposal_seed": 9_701,
    "diagnostic_seed": 7_009_704,
    "candidate_backend": "torch",
    "diagnostic_object_chunk": 64,
    "flow_atom_chunk": 4_096,
    "tail_proposal_atom_chunk": 262_144,
    "tail_proposal_object_chunk": 32,
    "exact_atom_chunk": 65_536,
    "exact_object_chunk": 4,
    "selection_key_sha256": (
        "f02f535eab874dd6f9f6184d4040283493266074903fc6c20160b4ae88b5993e"
    ),
    "selection_manifest_sha256": (
        "fc44a632ed4758440a3e241ffeab9fba087f58397f114cfbc38aecc7feedcb8b"
    ),
    "tail_proposal_recipe": {
        "name": "conditioned_dm_mis_t1_t2_prior_4_3_1_v2",
        "component_names": ("proxy_t1", "proxy_t2", "prior"),
        "component_weights": (0.5, 0.375, 0.125),
        "component_betas": (1.0, 0.5, None),
        "component_cycle": (0, 1, 0, 2, 1, 0, 1, 0),
        "conditioning": "component",
        "base_measure": "prior_times_detection",
        "temperature_scope": "proxy_likelihood_only",
        "denominator": "balance_mixture",
    },
    "mock_input_sha256": {
        "measurements.parquet": (
            "fdfecacc13adabd6005e21150ab0dc74767cad34da3d3ea95e362515ce38365b"
        ),
        "truth.parquet": (
            "78d1c8e02f43e1ac4673b2ca309b484d3565de3b9400bf736ca98bf584e5dc30"
        ),
        "likelihood_mock_manifest.json": (
            "5c097c98fa820177738156f6dd3290daaf68594dd90b2e80733f083b019d9a48"
        ),
    },
}


def exact_oracle_object_ids(
    universe_start: int, universe_stop: int
) -> tuple[np.ndarray, str]:
    """Return the frozen outcome-independent exact-oracle subset and its hash.

    An object's score is ``SHA256(key || uint64_be(absolute_id))``, where
    ``key = SHA256(EXACT_ORACLE_LABEL)``.  The 64 smallest scores are selected,
    and the returned ids are sorted so table order does not depend on score
    order.  The manifest hash is SHA-256 over those sorted big-endian uint64s.
    There are deliberately no seed or count arguments.
    """

    start = int(universe_start)
    stop = int(universe_stop)
    if start < 0 or stop - start != EXACT_ORACLE_UNIVERSE_SIZE:
        raise ValueError(
            "the exact oracle requires one declared 512-object universe"
        )
    if stop > np.iinfo(np.int64).max + 1:
        raise ValueError("exact-oracle object ids must fit non-negative int64")
    key = sha256(EXACT_ORACLE_LABEL.encode("utf-8")).digest()
    if key.hex() != EXACT_ORACLE_SELECTION_KEY_SHA256:
        raise RuntimeError("exact-oracle selection label no longer matches its key")
    scored = sorted(
        (
            sha256(key + struct.pack(">Q", object_id)).digest(),
            object_id,
        )
        for object_id in range(start, stop)
    )
    selected = np.sort(
        np.fromiter(
            (object_id for _, object_id in scored[:EXACT_ORACLE_SELECTION_COUNT]),
            dtype=np.int64,
            count=EXACT_ORACLE_SELECTION_COUNT,
        )
    )
    encoded = np.asarray(selected, dtype=">u8").tobytes()
    return selected, sha256(encoded).hexdigest()


@dataclass(frozen=True)
class _ExactOracleMass:
    """Streamed exact log masses for a selected set of observed objects."""

    log_stratum: np.ndarray
    log_complement: np.ndarray
    flow_evaluations: int
    active_atoms: int
    atoms_visited: int
    atom_blocks: int
    active_indices_sha256: str


def _stream_exact_oracle_mass(
    likelihood,
    observed,
    proposal,
    stratum_indices: np.ndarray,
    *,
    center: tuple[float, float],
    observed_targets: torch.Tensor,
    atom_chunk: int = EXACT_ORACLE_ATOM_CHUNK,
    object_chunk: int = EXACT_ORACLE_OBJECT_CHUNK,
) -> _ExactOracleMass:
    """Exactly sum ``S`` and ``S^c`` over active atoms in bounded memory.

    Production calls use the frozen 65,536-atom / four-object chunks.  The
    private chunk arguments exist only so tests can establish that the
    reduction is insensitive to how the same finite support is partitioned.
    Every target call uses proposal probability one, hence returns ``log c_j``.
    Block log-sums and their cross-block accumulation are both float64.
    """

    if atom_chunk <= 0 or object_chunk <= 0:
        raise ValueError("exact-oracle chunk sizes must be positive")
    active = np.asarray(proposal.active_indices, dtype=np.int64)
    strata = np.asarray(stratum_indices, dtype=np.int64)
    if active.ndim != 1 or not len(active):
        raise ValueError("exact oracle requires nonempty one-dimensional support")
    if (active < 0).any() or (np.diff(active) <= 0).any():
        raise ValueError("exact-oracle active support must be sorted and unique")
    if strata.ndim != 2 or strata.shape[0] != len(observed):
        raise ValueError("exact-oracle strata must align with observations")
    if len(observed) == 0:
        raise ValueError("exact oracle requires at least one selected object")

    stratum_positions = []
    for row in range(len(strata)):
        row_ids = strata[row]
        if len(np.unique(row_ids)) != len(row_ids):
            raise ValueError("exact-oracle stratum atom ids must be unique")
        positions = np.searchsorted(active, row_ids)
        valid = positions < len(active)
        if valid.any():
            valid[valid] &= active[positions[valid]] == row_ids[valid]
        if not valid.all():
            raise ValueError("exact-oracle stratum must lie in active support")
        stratum_positions.append(np.sort(positions))

    log_stratum = np.full(len(observed), -np.inf, dtype=np.float64)
    log_complement = np.full(len(observed), -np.inf, dtype=np.float64)
    atom_blocks = 0
    atoms_visited = 0
    device = observed_targets.device
    with torch.no_grad():
        for atom_start in range(0, len(active), int(atom_chunk)):
            atom_stop = min(atom_start + int(atom_chunk), len(active))
            atom_ids = active[atom_start:atom_stop]
            atoms_visited += len(atom_ids)
            atom_blocks += 1
            probability = np.ones(len(atom_ids), dtype=np.float64)
            for object_start in range(0, len(observed), int(object_chunk)):
                object_stop = min(
                    object_start + int(object_chunk), len(observed)
                )
                block_observed = observed.iloc[
                    object_start:object_stop
                ].reset_index(drop=True)
                log_c = likelihood.log_importance_weights_tensor(
                    block_observed,
                    float(center[0]),
                    float(center[1]),
                    atom_indices=atom_ids,
                    proposal_probability=probability,
                    observed_targets=observed_targets[object_start:object_stop],
                    object_chunk=int(object_chunk),
                    atom_chunk=int(atom_chunk),
                ).to(dtype=torch.float64)
                member = np.zeros(log_c.shape, dtype=bool)
                for local_row, positions in enumerate(
                    stratum_positions[object_start:object_stop]
                ):
                    left = np.searchsorted(positions, atom_start, side="left")
                    right = np.searchsorted(positions, atom_stop, side="left")
                    member[local_row, positions[left:right] - atom_start] = True
                member_tensor = torch.as_tensor(member, device=device)
                stratum_block = torch.logsumexp(
                    log_c.masked_fill(~member_tensor, -torch.inf), dim=1
                )
                complement_block = torch.logsumexp(
                    log_c.masked_fill(member_tensor, -torch.inf), dim=1
                )
                rows = slice(object_start, object_stop)
                log_stratum[rows] = np.logaddexp(
                    log_stratum[rows],
                    stratum_block.detach().cpu().numpy().astype(np.float64),
                )
                log_complement[rows] = np.logaddexp(
                    log_complement[rows],
                    complement_block.detach().cpu().numpy().astype(np.float64),
                )
    if atoms_visited != len(active):
        raise RuntimeError("exact-oracle support traversal was incomplete")
    active_hash = sha256(np.asarray(active, dtype=">u8").tobytes()).hexdigest()
    return _ExactOracleMass(
        log_stratum=log_stratum,
        log_complement=log_complement,
        flow_evaluations=int(len(observed) * len(active)),
        active_atoms=int(len(active)),
        atoms_visited=int(atoms_visited),
        atom_blocks=int(atom_blocks),
        active_indices_sha256=active_hash,
    )


@dataclass
class ComplementDiagnostic:
    """Per-object mass split and tail indices, plus the run's identity."""

    table: pd.DataFrame
    metadata: dict

    def summary(self) -> dict:
        frame = self.table
        total = (
            frame["mass_stratum"]
            + frame["mass_prefilter_complement"]
            + frame["mass_outside_prefilter"]
        )
        positive = total > 0

        def percentiles(values):
            values = np.asarray(values, dtype=np.float64)
            values = values[np.isfinite(values)]
            if values.size == 0:
                return None
            return [float(v) for v in np.percentile(values, (10, 50, 90))]

        tail_recipe = self.metadata.get("tail_proposal")
        component_mass = None
        component_target = None
        component_variance = None
        component_k = None
        component_top = None
        if tail_recipe is not None:
            names = tail_recipe["component_names"]
            component_mass = {
                name: percentiles(
                    frame[f"tail_proposal_component_complement_mass_{name}"]
                )
                for name in names
            }
            component_target = {
                name: percentiles(
                    frame[
                        f"tail_proposal_component_target_contribution_fraction_{name}"
                    ]
                )
                for name in names
            }
            component_variance = {
                name: percentiles(
                    frame[f"tail_proposal_component_variance_fraction_{name}"]
                )
                for name in names
            }
            component_k = {
                name: percentiles(
                    frame[f"pareto_k_tail_proposal_component_{name}"]
                )
                for name in names
            }
            component_top = {
                name: percentiles(
                    frame[f"tail_proposal_top_1_percent_source_fraction_{name}"]
                )
                for name in names
            }
        paired_z = None
        if "mass_complement_tail_minus_prior_z" in frame:
            paired_z = frame[
                "mass_complement_tail_minus_prior_z"
            ].to_numpy(dtype=np.float64)
            paired_z = paired_z[np.isfinite(paired_z)]
        tail_k = None
        if "pareto_k_tail_proposal_complement" in frame:
            tail_k = frame["pareto_k_tail_proposal_complement"].to_numpy(
                dtype=np.float64
            )

        def relative_error(standard_error, estimate):
            standard_error = np.asarray(standard_error, dtype=np.float64)
            denominator = np.abs(np.asarray(estimate, dtype=np.float64))
            result = np.full_like(denominator, np.nan)
            return np.divide(
                standard_error,
                denominator,
                out=result,
                where=denominator > 0.0,
            )

        result = {
            "objects": int(len(frame)),
            "objects_with_positive_mass": int(positive.sum()),
            "share_stratum_percentiles": percentiles(
                frame["mass_stratum"][positive] / total[positive]
            ),
            "share_prefilter_complement_percentiles": percentiles(
                frame["mass_prefilter_complement"][positive] / total[positive]
            ),
            "share_outside_prefilter_percentiles": percentiles(
                frame["mass_outside_prefilter"][positive] / total[positive]
            ),
            "outside_prefilter_relative_error_percentiles": percentiles(
                frame["mass_outside_prefilter_standard_error"][positive]
                / np.maximum(frame["mass_outside_prefilter"][positive], 0.0)
            ),
            "pareto_k_complement_percentiles": percentiles(
                frame["pareto_k_complement"]
            ),
            "pareto_k_prefilter_complement_percentiles": percentiles(
                frame["pareto_k_prefilter_complement"]
            ),
            "pareto_k_outside_prefilter_percentiles": percentiles(
                frame["pareto_k_outside_prefilter"]
            ),
            "pareto_k_complement_undefined": int(
                (~np.isfinite(frame["pareto_k_complement"])).sum()
            ),
            "pareto_k_tail_proposal_complement_percentiles": (
                percentiles(tail_k) if tail_k is not None else None
            ),
            "pareto_k_tail_proposal_complement_undefined": (
                int((~np.isfinite(tail_k)).sum()) if tail_k is not None else None
            ),
            "pareto_k_tail_proposal_below_0_5_fraction": (
                float(np.mean(np.isfinite(tail_k) & (tail_k < 0.5)))
                if tail_k is not None and len(tail_k)
                else None
            ),
            "pareto_k_tail_proposal_below_0_7_fraction": (
                float(np.mean(np.isfinite(tail_k) & (tail_k < 0.7)))
                if tail_k is not None and len(tail_k)
                else None
            ),
            "tail_proposal_relative_error_percentiles": (
                percentiles(
                    relative_error(
                        frame["mass_complement_tail_proposal_standard_error"],
                        frame["mass_complement_tail_proposal"],
                    )
                )
                if "mass_complement_tail_proposal" in frame
                else None
            ),
            "prior_complement_relative_error_percentiles": (
                percentiles(
                    relative_error(
                        frame["mass_complement_prior_standard_error"],
                        frame["mass_complement_prior"],
                    )
                )
                if "mass_complement_prior" in frame
                else None
            ),
            "tail_proposal_over_prior_relative_error_percentiles": (
                percentiles(
                    relative_error(
                        frame["mass_complement_tail_proposal_standard_error"],
                        frame["mass_complement_tail_proposal"],
                    )
                    / relative_error(
                        frame["mass_complement_prior_standard_error"],
                        frame["mass_complement_prior"],
                    )
                )
                if "mass_complement_tail_proposal" in frame
                else None
            ),
            "tail_proposal_minus_prior_z_mean": (
                float(paired_z.mean()) if paired_z is not None and len(paired_z) else None
            ),
            "tail_proposal_minus_prior_z_rms": (
                float(np.sqrt(np.mean(np.square(paired_z))))
                if paired_z is not None and len(paired_z)
                else None
            ),
            "tail_proposal_minus_prior_z_percentiles": (
                percentiles(paired_z) if paired_z is not None else None
            ),
            "tail_proposal_component_complement_mass_percentiles": component_mass,
            "tail_proposal_component_target_contribution_fraction_percentiles": (
                component_target
            ),
            "tail_proposal_component_variance_fraction_percentiles": (
                component_variance
            ),
            "pareto_k_tail_proposal_component_percentiles": component_k,
            "tail_proposal_top_1_percent_source_fraction_percentiles": component_top,
            "flow_evaluations": int(self.metadata["flow_evaluations"]),
        }
        if self.metadata.get("exact_oracle") is not None:
            result["exact_oracle"] = self._exact_oracle_summary()
            result["gates"] = result["exact_oracle"]["gates"]
        return result

    def _exact_oracle_summary(self) -> dict:
        """Reduce the frozen exact subset and all-512 predeclared gates."""

        frame = self.table
        protocol = self.metadata["exact_oracle"]
        selected_mask = frame["exact_oracle_selected"].to_numpy(dtype=bool)
        selected = frame.loc[selected_mask]

        def values(name):
            return selected[name].to_numpy(dtype=np.float64)

        def gate(observed, operator, threshold, passed):
            return {
                "observed": observed,
                "operator": operator,
                "threshold": threshold,
                "passed": bool(passed),
            }

        tail_k = frame["pareto_k_tail_proposal_complement"].to_numpy(
            dtype=np.float64
        )
        tail_relative_error = (
            frame["mass_complement_tail_proposal_standard_error"].to_numpy(
                dtype=np.float64
            )
            / np.abs(
                frame["mass_complement_tail_proposal"].to_numpy(dtype=np.float64)
            )
        )
        finite_tail = np.isfinite(tail_k)
        finite_relative_error = np.isfinite(tail_relative_error)
        fraction_below_0_5 = float(np.mean(finite_tail & (tail_k < 0.5)))
        fraction_below_0_7 = float(np.mean(finite_tail & (tail_k < 0.7)))
        relative_error_median = (
            float(np.median(tail_relative_error))
            if finite_relative_error.all() and len(tail_relative_error)
            else float("nan")
        )
        relative_error_p90 = (
            float(np.percentile(tail_relative_error, 90, method="higher"))
            if finite_relative_error.all() and len(tail_relative_error)
            else float("nan")
        )
        proposal_gates = {
            "pareto_k_defined_all_objects": gate(
                int((~finite_tail).sum()), "undefined_count ==", 0, finite_tail.all()
            ),
            "pareto_k_below_0_5_fraction": gate(
                fraction_below_0_5, ">=", 0.90, fraction_below_0_5 >= 0.90
            ),
            "pareto_k_below_0_7_fraction": gate(
                fraction_below_0_7, ">=", 0.99, fraction_below_0_7 >= 0.99
            ),
            "complement_relative_error_median": gate(
                relative_error_median,
                "<=",
                0.00882,
                np.isfinite(relative_error_median)
                and relative_error_median <= 0.00882,
            ),
            "complement_relative_error_p90_higher": gate(
                relative_error_p90,
                "<=",
                0.0268,
                np.isfinite(relative_error_p90) and relative_error_p90 <= 0.0268,
            ),
        }

        exact_complement = values("mass_complement_exact_oracle")
        exact_stratum = values("mass_stratum_exact_oracle")
        exact_total = values("mass_total_exact_oracle")
        tail_mass = values("mass_complement_tail_proposal")
        tail_se = values("mass_complement_tail_proposal_standard_error")
        stratum_relative_error = values(
            "mass_stratum_exact_oracle_relative_error"
        )
        delta_log_total = values("delta_log_total_evidence_tail_minus_exact")
        exact_z = values("mass_complement_tail_minus_exact_z")
        differences = tail_mass - exact_complement

        finite_positive = bool(
            len(selected) == EXACT_ORACLE_SELECTION_COUNT
            and np.isfinite(exact_complement).all()
            and np.isfinite(exact_stratum).all()
            and np.isfinite(exact_total).all()
            and np.isfinite(tail_mass).all()
            and np.isfinite(tail_se).all()
            and (exact_complement > 0).all()
            and (exact_stratum > 0).all()
            and (exact_total > 0).all()
            and (tail_mass > 0).all()
            and (tail_se > 0).all()
        )
        crosscheck_max = (
            float(np.max(stratum_relative_error))
            if len(stratum_relative_error)
            and np.isfinite(stratum_relative_error).all()
            else float("nan")
        )
        absolute_delta = np.abs(delta_log_total)
        median_absolute_delta = (
            float(np.median(absolute_delta))
            if len(absolute_delta) and np.isfinite(absolute_delta).all()
            else float("nan")
        )
        p90_absolute_delta = (
            float(np.percentile(absolute_delta, 90, method="higher"))
            if len(absolute_delta) and np.isfinite(absolute_delta).all()
            else float("nan")
        )
        aggregate_denominator = float(np.sqrt(np.sum(np.square(tail_se))))
        aggregate_z = (
            float(np.sum(differences) / aggregate_denominator)
            if np.isfinite(differences).all()
            and np.isfinite(aggregate_denominator)
            and aggregate_denominator > 0.0
            else float("nan")
        )
        exact_z_rms = (
            float(np.sqrt(np.mean(np.square(exact_z))))
            if len(exact_z) and np.isfinite(exact_z).all()
            else float("nan")
        )
        within_1_96 = int(np.sum(np.isfinite(exact_z) & (np.abs(exact_z) <= 1.96)))
        selected_count = int(selected_mask.sum())
        support_visited_once = bool(
            protocol["active_atoms_visited_per_selected_object"]
            == protocol["active_atoms"]
            and protocol["selected_objects_evaluated"]
            == EXACT_ORACLE_SELECTION_COUNT
        )
        confirmation_matched = bool(
            protocol["confirmation_protocol"]["matched"]
        )
        exact_gates = {
            "confirmation_protocol_matched": gate(
                confirmation_matched, "is", True, confirmation_matched
            ),
            "selected_object_count": gate(
                selected_count,
                "==",
                EXACT_ORACLE_SELECTION_COUNT,
                selected_count == EXACT_ORACLE_SELECTION_COUNT,
            ),
            "active_support_visited_once_per_selected_object": gate(
                support_visited_once, "is", True, support_visited_once
            ),
            "finite_positive_exact_masses_and_standard_errors": gate(
                finite_positive, "is", True, finite_positive
            ),
            "stratum_relative_crosscheck_max": gate(
                crosscheck_max,
                "<=",
                1.0e-5,
                np.isfinite(crosscheck_max) and crosscheck_max <= 1.0e-5,
            ),
            "absolute_delta_log_total_evidence_median": gate(
                median_absolute_delta,
                "<=",
                0.01,
                np.isfinite(median_absolute_delta)
                and median_absolute_delta <= 0.01,
            ),
            "absolute_delta_log_total_evidence_p90_higher": gate(
                p90_absolute_delta,
                "<=",
                0.05,
                np.isfinite(p90_absolute_delta) and p90_absolute_delta <= 0.05,
            ),
            "absolute_signed_aggregate_z": gate(
                abs(aggregate_z) if np.isfinite(aggregate_z) else float("nan"),
                "<=",
                2.5,
                np.isfinite(aggregate_z) and abs(aggregate_z) <= 2.5,
            ),
            "exact_z_rms": gate(
                exact_z_rms,
                "in_closed_interval",
                [0.8, 1.2],
                np.isfinite(exact_z_rms) and 0.8 <= exact_z_rms <= 1.2,
            ),
            "exact_z_within_1_96_count": gate(
                within_1_96, ">=", 58, within_1_96 >= 58
            ),
        }
        proposal_passed = all(item["passed"] for item in proposal_gates.values())
        exact_passed = all(item["passed"] for item in exact_gates.values())
        return {
            "protocol": protocol,
            "selected_objects": selected_count,
            "mass_complement_exact_percentiles": [
                float(value)
                for value in np.percentile(exact_complement, (10, 50, 90))
            ],
            "stratum_relative_crosscheck_max": crosscheck_max,
            "absolute_delta_log_total_evidence_median": median_absolute_delta,
            "absolute_delta_log_total_evidence_p90_higher": p90_absolute_delta,
            "signed_aggregate_z": aggregate_z,
            "exact_z_rms": exact_z_rms,
            "exact_z_within_1_96_count": within_1_96,
            "gates": {
                "predeclared": confirmation_matched,
                "thresholds_predeclared": True,
                "confirmation_protocol_matched": confirmation_matched,
                "proposal_all_512_objects": proposal_gates,
                "exact_oracle_64_object_subset": exact_gates,
                "proposal_passed": proposal_passed,
                "exact_oracle_passed": exact_passed,
                "passed": bool(proposal_passed and exact_passed),
                "policy": (
                    "the exact subset cannot relax the all-object proposal gates"
                ),
            },
        }


def tail_of(values: np.ndarray) -> tuple[float, float]:
    """Pareto tail index and maximum of one weight sample, or NaN if degenerate."""

    values = np.asarray(values, dtype=np.float64)
    if values.size < 20:
        return float("nan"), float("nan")
    index = float(
        pareto_tail_index(torch.as_tensor(values[None, :], dtype=torch.float64))[0]
    )
    return index, float(values.max())


def _fixed_component_standard_error(
    values: np.ndarray, component_id: np.ndarray
) -> float:
    """Standard error of a fixed-allocation component-mixture mean.

    The estimator is the ordinary mean of all draws, but its source counts are
    fixed. Its variance is therefore ``sum_c n_c s_c^2 / M^2``. Applying the
    same reduction to paired per-draw differences preserves covariance from
    the shared inverse-CDF uniform.
    """

    values = np.asarray(values, dtype=np.float64)
    component_id = np.asarray(component_id, dtype=np.int16)
    if values.ndim != 1 or component_id.shape != values.shape:
        raise ValueError("values and component ids must be aligned vectors")
    if not np.isfinite(values).all():
        raise ValueError("fixed-component values must be finite")
    variance = 0.0
    for component in range(len(TAIL_PROPOSAL_RECIPE.component_names)):
        selected = values[component_id == component]
        if len(selected) < 2:
            return float("nan")
        variance += len(selected) * float(selected.var(ddof=1))
    return float(np.sqrt(variance) / len(values))


def _membership(lookup: np.ndarray, marked: np.ndarray, queried: np.ndarray) -> np.ndarray:
    """Return which ``queried`` atom ids appear in ``marked``.

    ``lookup`` is a reusable zero-initialised table over all atom ids, restored
    before returning, so no per-row sort of a 131,072-atom set is needed.
    """

    lookup[marked] = True
    member = lookup[queried].copy()
    lookup[marked] = False
    return member


def diagnose_complement(
    likelihood,
    mock,
    proposal,
    *,
    center: tuple[float, float],
    n_candidates: int,
    prefilter_candidates: int,
    n_global_draws: int,
    seed: int,
    observation_start: int = 0,
    observation_stop: Optional[int] = None,
    object_id_offset: int = 0,
    object_chunk: int = 32,
    atom_chunk: int = 4096,
    candidate_backend: str = "torch",
    include_tail_proposal: bool = False,
    include_exact_oracle: bool = False,
    confirmation_mock_sha256: Optional[dict[str, str]] = None,
) -> ComplementDiagnostic:
    """Split target mass and pair prior versus fixed DM-MIS tail draws.

    ``observation_start`` and ``observation_stop`` are local bounds into
    ``mock``.  ``object_id_offset`` maps those rows back to their absolute ids;
    those ids seed both samplers and are also written to ``object_index``.
    """

    if prefilter_candidates < n_candidates:
        raise ValueError("prefilter cannot be smaller than the exact stratum")
    if n_global_draws <= 0:
        raise ValueError("n_global_draws must be positive")
    if object_id_offset < 0:
        raise ValueError("object_id_offset must be non-negative")
    if (
        include_tail_proposal
        and n_global_draws % TAIL_PROPOSAL_RECIPE.cycle_length
    ):
        raise ValueError(
            "tail-proposal draws must be a multiple of "
            f"{TAIL_PROPOSAL_RECIPE.cycle_length}"
        )
    if include_exact_oracle and not include_tail_proposal:
        raise ValueError("the exact oracle requires the tail-proposal diagnostic")
    if include_exact_oracle and getattr(likelihood, "selection", None) is not None:
        raise ValueError(
            "the exact oracle requires measured selection to be disabled so "
            "frozen absolute row ids are preserved"
        )

    measurements = mock.measurements
    stop = len(measurements) if observation_stop is None else int(observation_stop)
    stop = min(stop, len(measurements))
    start = int(observation_start)
    if start < 0 or start >= stop:
        raise ValueError("empty observation window")
    exact_selected_ids = None
    exact_selection_manifest_sha256 = None
    exact_selected_id_set = set()
    if include_exact_oracle:
        exact_selected_ids, exact_selection_manifest_sha256 = (
            exact_oracle_object_ids(
                object_id_offset + start,
                object_id_offset + stop,
            )
        )
        exact_selected_id_set = set(map(int, exact_selected_ids))

    torch_device = (
        likelihood.flow_model.device if candidate_backend == "torch" else None
    )
    flow_device = str(likelihood.flow_model.device)
    tail_proposal_atom_chunk = None
    tail_proposal_object_chunk = None
    if include_tail_proposal:
        parameters = inspect.signature(proposal.draw_tilted_mis).parameters
        atom_chunk_parameter = parameters.get("atom_chunk")
        if (
            atom_chunk_parameter is not None
            and atom_chunk_parameter.default is not inspect.Parameter.empty
        ):
            tail_proposal_atom_chunk = int(atom_chunk_parameter.default)
        object_chunk_parameter = parameters.get("object_chunk")
        if (
            object_chunk_parameter is not None
            and object_chunk_parameter.default is not inspect.Parameter.empty
        ):
            tail_proposal_object_chunk = int(object_chunk_parameter.default)
    lookup = np.zeros(len(proposal.prior_weights), dtype=bool)
    rows = []
    flow_evaluations = 0
    started = time.perf_counter()
    phase_seconds = {
        "candidate_query": 0.0,
        "prefilter_flow": 0.0,
        "prior_draw": 0.0,
        "prior_flow": 0.0,
        "tail_proposal_draw": 0.0,
        "tail_proposal_flow": 0.0,
        "reduction": 0.0,
    }
    if include_exact_oracle:
        phase_seconds["exact_oracle_flow"] = 0.0
    exact_records = {}
    exact_flow_evaluations = 0
    exact_active_atoms = None
    exact_atoms_visited = None
    exact_atom_blocks = None
    exact_active_indices_sha256 = None

    for object_start in range(start, stop, object_chunk):
        object_stop = min(object_start + object_chunk, stop)
        observed = measurements.iloc[object_start:object_stop].reset_index(drop=True)
        absolute_ids = object_id_offset + np.arange(
            object_start, object_stop, dtype=np.int64
        )
        observed_targets = likelihood.observed_target_tensor(observed)

        phase_started = time.perf_counter()
        prefilter = proposal.candidates(
            observed,
            n_candidates=prefilter_candidates,
            torch_device=torch_device,
        )
        stratum = proposal.candidates(
            observed,
            n_candidates=n_candidates,
            prefilter_candidates=prefilter_candidates,
            torch_device=torch_device,
        )
        phase_seconds["candidate_query"] += time.perf_counter() - phase_started

        phase_started = time.perf_counter()
        ones = np.ones_like(prefilter.indices, dtype=np.float64)
        prefilter_log_c = (
            likelihood.log_importance_weights_tensor(
                observed,
                float(center[0]),
                float(center[1]),
                atom_indices=prefilter.indices,
                proposal_probability=ones,
                observed_targets=observed_targets,
                object_chunk=len(observed),
                atom_chunk=atom_chunk,
            )
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        flow_evaluations += int(prefilter.indices.size)
        phase_seconds["prefilter_flow"] += time.perf_counter() - phase_started

        # Both arms use production samplers, the same absolute object ids, the
        # same seed, and the same fixed-width record's atom-uniform column.
        # Their estimates are therefore paired by common random numbers even
        # when this call covers only a slice of a larger catalogue.
        phase_started = time.perf_counter()
        prior_draw = proposal.draw_stratified(
            stratum,
            n_draws=n_global_draws,
            seed=seed,
            object_ids=absolute_ids,
        )
        phase_seconds["prior_draw"] += time.perf_counter() - phase_started
        phase_started = time.perf_counter()
        prior_log_ratio = (
            likelihood.log_importance_weights_tensor(
                observed,
                float(center[0]),
                float(center[1]),
                atom_indices=prior_draw.indices,
                proposal_probability=prior_draw.probability,
                observed_targets=observed_targets,
                object_chunk=len(observed),
                atom_chunk=atom_chunk,
            )
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        flow_evaluations += int(prior_draw.indices.size)
        phase_seconds["prior_flow"] += time.perf_counter() - phase_started

        tail_draw = None
        tail_ratio = None
        if include_tail_proposal:
            phase_started = time.perf_counter()
            tail_draw = proposal.draw_tilted_mis(
                stratum,
                observed,
                n_draws=n_global_draws,
                seed=seed,
                object_ids=absolute_ids,
                device=likelihood.flow_model.device,
            )
            phase_seconds["tail_proposal_draw"] += (
                time.perf_counter() - phase_started
            )
            phase_started = time.perf_counter()
            tail_log_ratio = (
                likelihood.log_importance_weights_tensor(
                    observed,
                    float(center[0]),
                    float(center[1]),
                    atom_indices=tail_draw.indices,
                    proposal_probability=tail_draw.probability,
                    observed_targets=observed_targets,
                    object_chunk=len(observed),
                    atom_chunk=atom_chunk,
                )
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )
            flow_evaluations += int(tail_draw.indices.size)
            tail_ratio = np.exp(tail_log_ratio)
            phase_seconds["tail_proposal_flow"] += (
                time.perf_counter() - phase_started
            )

        if include_exact_oracle:
            selected_positions = np.fromiter(
                (
                    position
                    for position, object_id in enumerate(absolute_ids)
                    if int(object_id) in exact_selected_id_set
                ),
                dtype=np.int64,
            )
            if len(selected_positions):
                phase_started = time.perf_counter()
                exact = _stream_exact_oracle_mass(
                    likelihood,
                    observed.iloc[selected_positions].reset_index(drop=True),
                    proposal,
                    stratum.indices[selected_positions],
                    center=center,
                    observed_targets=observed_targets[
                        selected_positions.tolist()
                    ],
                )
                phase_seconds["exact_oracle_flow"] += (
                    time.perf_counter() - phase_started
                )
                exact_flow_evaluations += exact.flow_evaluations
                flow_evaluations += exact.flow_evaluations
                if exact_active_atoms is None:
                    exact_active_atoms = exact.active_atoms
                    exact_atoms_visited = exact.atoms_visited
                    exact_atom_blocks = exact.atom_blocks
                    exact_active_indices_sha256 = exact.active_indices_sha256
                elif (
                    exact_active_atoms != exact.active_atoms
                    or exact_atoms_visited != exact.atoms_visited
                    or exact_atom_blocks != exact.atom_blocks
                    or exact_active_indices_sha256 != exact.active_indices_sha256
                ):
                    raise RuntimeError(
                        "exact-oracle active-support identity changed within a run"
                    )
                for selected_row, position in enumerate(selected_positions):
                    exact_records[int(absolute_ids[position])] = (
                        float(exact.log_stratum[selected_row]),
                        float(exact.log_complement[selected_row]),
                    )

        phase_started = time.perf_counter()
        for row in range(len(observed)):
            prefilter_ids = prefilter.indices[row]
            stratum_ids = stratum.indices[row]
            drawn_ids = prior_draw.indices[row]

            in_stratum = _membership(lookup, stratum_ids, prefilter_ids)
            c_prefilter = np.exp(prefilter_log_c[row])
            mass_stratum = float(c_prefilter[in_stratum].sum())
            mass_prefilter_complement = float(c_prefilter[~in_stratum].sum())

            drawn_in_stratum = prior_draw.local_member[row]
            drawn_in_prefilter = _membership(lookup, prefilter_ids, drawn_ids)
            ratio = np.exp(prior_log_ratio[row])

            outside = ~drawn_in_prefilter
            contribution = np.where(outside, ratio, 0.0)
            mass_outside = float(contribution.mean())
            mass_outside_se = (
                float(contribution.std(ddof=1) / np.sqrt(len(contribution)))
                if len(contribution) > 1
                else float("nan")
            )

            complement = ~drawn_in_stratum
            between = drawn_in_prefilter & complement

            k_complement, max_complement = tail_of(ratio[complement])
            k_between, max_between = tail_of(ratio[between])
            k_outside, max_outside = tail_of(ratio[outside])

            record = {}
            if tail_ratio is not None:
                valid_tail = ~tail_draw.local_member[row]
                selected = tail_ratio[row, valid_tail]
                source_component = tail_draw.component_id[row]
                k_tail, max_tail = tail_of(selected)
                tail_contribution = np.where(valid_tail, tail_ratio[row], 0.0)
                mass_tail = float(tail_contribution.mean())
                mass_tail_se = _fixed_component_standard_error(
                    tail_contribution, source_component
                )
                prior_contribution = np.where(complement, ratio, 0.0)
                mass_prior = float(prior_contribution.mean())
                paired_difference = tail_contribution - prior_contribution
                mass_difference = float(paired_difference.mean())
                mass_difference_se = _fixed_component_standard_error(
                    paired_difference, source_component
                )
                mass_difference_z = (
                    mass_difference / mass_difference_se
                    if np.isfinite(mass_difference_se) and mass_difference_se > 0.0
                    else float("nan")
                )
                component_records = {}
                component_variance = []
                top_count = max(1, int(np.ceil(0.01 * len(tail_contribution))))
                top_positions = np.argpartition(
                    tail_contribution, len(tail_contribution) - top_count
                )[-top_count:]
                for component, name in enumerate(
                    TAIL_PROPOSAL_RECIPE.component_names
                ):
                    component_values = tail_contribution[
                        source_component == component
                    ]
                    count = len(component_values)
                    alpha = count / len(tail_contribution)
                    sample_variance = (
                        float(component_values.var(ddof=1))
                        if count > 1
                        else float("nan")
                    )
                    variance_contribution = (
                        alpha**2 * sample_variance / count
                        if count > 1
                        else float("nan")
                    )
                    component_variance.append(variance_contribution)
                    component_k, component_maximum = tail_of(component_values)
                    component_records.update(
                        {
                            f"mass_tail_proposal_component_mean_{name}": float(
                                component_values.mean()
                            ),
                            f"mass_tail_proposal_component_weighted_mean_{name}": (
                                float(alpha * component_values.mean())
                            ),
                            f"mass_tail_proposal_component_sample_variance_{name}": (
                                sample_variance
                            ),
                            f"mass_tail_proposal_component_variance_{name}": (
                                variance_contribution
                            ),
                            f"pareto_k_tail_proposal_component_{name}": component_k,
                            f"maximum_ratio_tail_proposal_component_{name}": (
                                component_maximum
                            ),
                            f"p99_ratio_tail_proposal_component_{name}": float(
                                np.percentile(component_values, 99)
                            ),
                            f"tail_proposal_top_1_percent_source_fraction_{name}": (
                                float(
                                    np.mean(source_component[top_positions] == component)
                                )
                            ),
                        }
                    )
                total_component_variance = float(np.sum(component_variance))
                for component, name in enumerate(
                    TAIL_PROPOSAL_RECIPE.component_names
                ):
                    weighted_mean = component_records[
                        f"mass_tail_proposal_component_weighted_mean_{name}"
                    ]
                    component_records[
                        f"tail_proposal_component_target_contribution_fraction_{name}"
                    ] = weighted_mean / mass_tail if mass_tail > 0.0 else float("nan")
                    component_records[
                        f"tail_proposal_component_variance_fraction_{name}"
                    ] = (
                        component_variance[component] / total_component_variance
                        if total_component_variance > 0.0
                        else float("nan")
                    )
                record.update(
                    {
                        **component_records,
                        "pareto_k_tail_proposal_complement": k_tail,
                        "maximum_ratio_tail_proposal_complement": max_tail,
                        "mass_complement_tail_proposal": mass_tail,
                        "mass_complement_tail_proposal_standard_error": mass_tail_se,
                        "draws_tail_proposal_complement": int(valid_tail.sum()),
                        "mass_complement_tail_minus_prior": mass_difference,
                        "mass_complement_tail_minus_prior_standard_error": (
                            mass_difference_se
                        ),
                        "mass_complement_tail_minus_prior_z": mass_difference_z,
                    }
                )
                record["mass_complement_prior"] = mass_prior
                record["mass_complement_prior_standard_error"] = (
                    float(
                        prior_contribution.std(ddof=1)
                        / np.sqrt(len(prior_contribution))
                    )
                    if len(prior_contribution) > 1
                    else float("nan")
                )
                for component, name in enumerate(
                    TAIL_PROPOSAL_RECIPE.component_names
                ):
                    record[
                        f"tail_proposal_component_complement_mass_{name}"
                    ] = float(tail_draw.component_complement_mass[row, component])
                    record[f"draws_tail_proposal_component_{name}"] = int(
                        np.sum(source_component == component)
                    )

                if include_exact_oracle:
                    object_id = int(absolute_ids[row])
                    record["exact_oracle_selected"] = (
                        object_id in exact_selected_id_set
                    )
                    if object_id in exact_records:
                        exact_log_stratum, exact_log_complement = exact_records[
                            object_id
                        ]
                        exact_log_total = float(
                            np.logaddexp(exact_log_stratum, exact_log_complement)
                        )
                        exact_stratum = float(np.exp(exact_log_stratum))
                        exact_complement = float(np.exp(exact_log_complement))
                        exact_total = float(np.exp(exact_log_total))
                        stratum_relative_error = (
                            abs(mass_stratum - exact_stratum)
                            / abs(exact_stratum)
                            if exact_stratum != 0.0
                            else float("nan")
                        )
                        tail_minus_exact = mass_tail - exact_complement
                        tail_minus_exact_z = (
                            tail_minus_exact / mass_tail_se
                            if np.isfinite(mass_tail_se) and mass_tail_se > 0.0
                            else float("nan")
                        )
                        tail_total = mass_stratum + mass_tail
                        record.update(
                            {
                                "log_mass_stratum_exact_oracle": (
                                    exact_log_stratum
                                ),
                                "mass_stratum_exact_oracle": exact_stratum,
                                "mass_stratum_exact_oracle_relative_error": (
                                    stratum_relative_error
                                ),
                                "log_mass_complement_exact_oracle": (
                                    exact_log_complement
                                ),
                                "mass_complement_exact_oracle": (
                                    exact_complement
                                ),
                                "log_mass_total_exact_oracle": exact_log_total,
                                "mass_total_exact_oracle": exact_total,
                                "mass_complement_tail_minus_exact": (
                                    tail_minus_exact
                                ),
                                "mass_complement_tail_minus_exact_z": (
                                    tail_minus_exact_z
                                ),
                                "delta_log_total_evidence_tail_minus_exact": (
                                    float(np.log(tail_total) - exact_log_total)
                                    if tail_total > 0.0
                                    else float("nan")
                                ),
                            }
                        )

            rows.append(
                {
                    **record,
                    "object_index": int(absolute_ids[row]),
                    "mass_stratum": mass_stratum,
                    "mass_prefilter_complement": mass_prefilter_complement,
                    "mass_outside_prefilter": mass_outside,
                    "mass_outside_prefilter_standard_error": mass_outside_se,
                    "pareto_k_complement": k_complement,
                    "pareto_k_prefilter_complement": k_between,
                    "pareto_k_outside_prefilter": k_outside,
                    "maximum_ratio_complement": max_complement,
                    "maximum_ratio_prefilter_complement": max_between,
                    "maximum_ratio_outside_prefilter": max_outside,
                    "draws_prefilter_complement": int(between.sum()),
                    "draws_outside_prefilter": int(outside.sum()),
                }
            )
        phase_seconds["reduction"] += time.perf_counter() - phase_started

    exact_oracle_metadata = None
    if include_exact_oracle:
        if len(exact_records) != EXACT_ORACLE_SELECTION_COUNT:
            raise RuntimeError(
                "exact-oracle selection was not completely evaluated"
            )
        confirmation_observed = {
            "universe_start": int(object_id_offset + start),
            "universe_stop": int(object_id_offset + stop),
            "center": [float(center[0]), float(center[1])],
            "n_candidates": int(n_candidates),
            "prefilter_candidates": int(prefilter_candidates),
            "n_global_draws": int(n_global_draws),
            "proposal_seed": int(seed) - EXACT_ORACLE_DIAGNOSTIC_SEED_OFFSET,
            "diagnostic_seed": int(seed),
            "candidate_backend": candidate_backend,
            "diagnostic_object_chunk": int(object_chunk),
            "flow_atom_chunk": int(atom_chunk),
            "tail_proposal_atom_chunk": tail_proposal_atom_chunk,
            "tail_proposal_object_chunk": tail_proposal_object_chunk,
            "exact_atom_chunk": EXACT_ORACLE_ATOM_CHUNK,
            "exact_object_chunk": EXACT_ORACLE_OBJECT_CHUNK,
            "selection_key_sha256": EXACT_ORACLE_SELECTION_KEY_SHA256,
            "selection_manifest_sha256": exact_selection_manifest_sha256,
            "tail_proposal_recipe": asdict(TAIL_PROPOSAL_RECIPE),
            "mock_input_sha256": (
                None
                if confirmation_mock_sha256 is None
                else dict(confirmation_mock_sha256)
            ),
        }
        exact_oracle_metadata = {
            "label": EXACT_ORACLE_LABEL,
            "selection_rule": "64 smallest SHA256(key || uint64_be(object_id))",
            "selection_key_sha256": EXACT_ORACLE_SELECTION_KEY_SHA256,
            "universe_size": EXACT_ORACLE_UNIVERSE_SIZE,
            "universe_start": int(object_id_offset + start),
            "universe_stop": int(object_id_offset + stop),
            "selection_count": EXACT_ORACLE_SELECTION_COUNT,
            "selected_object_ids": exact_selected_ids.tolist(),
            "selection_manifest_sha256": exact_selection_manifest_sha256,
            "atom_chunk": EXACT_ORACLE_ATOM_CHUNK,
            "object_chunk": EXACT_ORACLE_OBJECT_CHUNK,
            "proposal_probability": 1.0,
            "reduction": "direct_stratum_and_complement_logsumexp_float64",
            "active_atoms": int(exact_active_atoms),
            "active_atoms_visited_per_selected_object": int(
                exact_atoms_visited
            ),
            "selected_objects_evaluated": int(len(exact_records)),
            "atom_blocks_per_selected_object": int(exact_atom_blocks),
            "active_indices_sha256": exact_active_indices_sha256,
            "flow_evaluations": int(exact_flow_evaluations),
            "confirmation_protocol": {
                "expected": dict(EXACT_ORACLE_CONFIRMATION),
                "observed": confirmation_observed,
                "matched": bool(
                    confirmation_observed == EXACT_ORACLE_CONFIRMATION
                ),
            },
        }
    metadata = {
        "center": [float(center[0]), float(center[1])],
        "n_candidates": int(n_candidates),
        "prefilter_candidates": int(prefilter_candidates),
        "n_global_draws": int(n_global_draws),
        "seed": int(seed),
        "observation_start": int(object_id_offset + start),
        "observation_stop": int(object_id_offset + stop),
        "local_observation_start": start,
        "local_observation_stop": stop,
        "object_id_offset": int(object_id_offset),
        "candidate_backend": candidate_backend,
        "candidate_device": None if torch_device is None else str(torch_device),
        "flow_device": flow_device,
        "object_chunk": int(object_chunk),
        "flow_atom_chunk": int(atom_chunk),
        "tail_proposal_device": flow_device if include_tail_proposal else None,
        "tail_proposal_atom_chunk": tail_proposal_atom_chunk,
        "tail_proposal_object_chunk": tail_proposal_object_chunk,
        "flow_evaluations": int(flow_evaluations),
        "tail_proposal": (
            asdict(TAIL_PROPOSAL_RECIPE) if include_tail_proposal else None
        ),
        "tail_proposal_common_random_numbers": bool(include_tail_proposal),
        "elapsed_seconds": float(time.perf_counter() - started),
        "phase_seconds": {
            name: float(value) for name, value in phase_seconds.items()
        },
    }
    if exact_oracle_metadata is not None:
        metadata["exact_oracle"] = exact_oracle_metadata
    return ComplementDiagnostic(pd.DataFrame(rows), metadata)
