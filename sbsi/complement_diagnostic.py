"""Measure complement mass and compare production tail proposals.

For every observation, the diagnostic evaluates the full prefilter exactly,
then pairs the prior-complement estimator with the production fixed
component-conditioned deterministic-MIS sampler.  Both sampled arms consume
the same per-object random record and use absolute object ids, so a partitioned
run is identical to the corresponding rows of an unsliced run.
"""

from __future__ import annotations

from copy import deepcopy
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
    PILOT_STRATUM_HASH_CONTRACT,
    PILOT_STRATUM_RECIPE,
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
COMPLEMENT_DIAGNOSTIC_SEED_OFFSET = 7_000_003
PILOT_STRATUM_DIAGNOSTIC_SEED_OFFSET = 1_000_003
PILOT_STRATUM_REPLICATION_SEED_OFFSET = 2_000_003
# Kept as an oracle-specific alias because the frozen v1 protocol serializes
# the production diagnostic seed through this name.
EXACT_ORACLE_DIAGNOSTIC_SEED_OFFSET = COMPLEMENT_DIAGNOSTIC_SEED_OFFSET
PILOT_STRATUM_DEVELOPMENT_LABEL = (
    "SBSI/complement-pilot-stratum-development/v1"
)
PILOT_STRATUM_DEVELOPMENT_BASELINE = {
    "total_relative_error_median": 0.013214266498501442,
    "total_relative_error_p90_higher": 0.021110866788308195,
    "mean_total_relative_error_squared": 4.6256941067948046e-4,
    "total_relative_error_squared_p90_higher": 4.45668696553694e-4,
    "method_seconds": 91.04022970038932,
}
PILOT_STRATUM_DEVELOPMENT_MOCK_SHA256 = {
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
PILOT_STRATUM_DEVELOPMENT_GATE_THRESHOLDS = {
    "objects": 512,
    "pareto_k_below_0_5_count": 461,
    "pareto_k_below_0_7_count": 507,
    "complement_relative_error_median": 0.00882,
    "complement_relative_error_p90_higher": 0.0268,
    "efficiency_mean": 0.90,
    "efficiency_p90_higher": 0.90,
}
PILOT_STRATUM_METHOD_TIME_COMPONENTS = (
    "candidate_query",
    "prefilter_flow",
    "pilot_proposal_draw",
    "pilot_proposal_flow",
    "pilot_promotion",
    "pilot_stratum_exact_flow",
    "tail_proposal_draw",
    "tail_proposal_flow",
    "reduction",
)
PILOT_STRATUM_DEVELOPMENT_PROTOCOL = {
    "universe_start": 25_000,
    "universe_stop": 25_512,
    "center": [0.013610377120038044, 0.002740148300964851],
    "base_stratum_size": 16_384,
    "promotions": 4_096,
    "final_stratum_size": 20_480,
    "prefilter_candidates": 131_072,
    "n_global_draws": 16_384,
    "proposal_seed": 8_701,
    "production_seed": 7_008_704,
    "pilot_seed": 1_008_704,
    "candidate_backend": "torch",
    "diagnostic_object_chunk": 64,
    "flow_atom_chunk": 4_096,
    "tail_proposal_atom_chunk": 262_144,
    "tail_proposal_object_chunk": 32,
    "pilot_atom_chunk": 262_144,
    "pilot_object_chunk": 32,
    "tail_proposal_recipe": asdict(TAIL_PROPOSAL_RECIPE),
    "pilot_stratum_recipe": asdict(PILOT_STRATUM_RECIPE),
    "pilot_stratum_recipe_sha256": PILOT_STRATUM_RECIPE.sha256,
    "hash_contract": PILOT_STRATUM_HASH_CONTRACT,
    "pilot_flow_evaluations": 4_194_304,
    "final_stratum_flow_evaluations": 10_485_760,
    "run_flow_evaluations": 98_566_144,
    "mock_input_sha256": dict(PILOT_STRATUM_DEVELOPMENT_MOCK_SHA256),
}
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
    "mock_input_sha256": dict(PILOT_STRATUM_DEVELOPMENT_MOCK_SHA256),
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


def _pilot_stratum_run_manifest_sha256(
    object_hashes: list[tuple[int, str]],
) -> str:
    """Hash final-stratum identities independently of execution batching.

    Promotion constructs one canonical final-stratum hash per absolute object
    id.  The diagnostic run identity is the sorted sequence of those pairs,
    rather than any batch-shaped pilot array, so identical per-object results
    compose across calls and slices.  Numerical reproduction separately fixes
    the proposal atom/object chunks that generated those object hashes.
    """

    ordered = sorted(
        (int(object_id), str(value)) for object_id, value in object_hashes
    )
    if len({object_id for object_id, _ in ordered}) != len(ordered):
        raise ValueError("pilot-stratum manifest object ids must be unique")
    digest = sha256()
    digest.update(b"SBSI/pilot-stratum-run/v1\0")
    digest.update(struct.pack(">Q", len(ordered)))
    for object_id, value in ordered:
        if object_id < 0 or object_id > np.iinfo(np.int64).max:
            raise ValueError("pilot-stratum manifest ids must fit non-negative int64")
        if len(value) != 64 or any(
            character not in "0123456789abcdef" for character in value
        ):
            raise ValueError("pilot-stratum object hashes must be lowercase SHA-256")
        digest.update(struct.pack(">Q", object_id))
        digest.update(bytes.fromhex(value))
    return digest.hexdigest()


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
        if self.metadata.get("pilot_stratum") is not None:
            result["pilot_stratum"] = self._pilot_stratum_summary()
            result["gates"] = result["pilot_stratum"]["gates"]
        return result

    def _pilot_stratum_summary(self) -> dict:
        """Reduce the frozen 512-row development protocol and its hard gates."""

        frame = self.table
        protocol = self.metadata["pilot_stratum"]
        thresholds = PILOT_STRATUM_DEVELOPMENT_GATE_THRESHOLDS
        baseline = PILOT_STRATUM_DEVELOPMENT_BASELINE
        expected_objects = int(thresholds["objects"])

        def values(name):
            return frame[name].to_numpy(dtype=np.float64)

        def percentiles(data):
            data = np.asarray(data, dtype=np.float64)
            data = data[np.isfinite(data)]
            if not len(data):
                return None
            return [float(value) for value in np.percentile(data, (10, 50, 90))]

        def gate(observed, operator, threshold, passed):
            return {
                "observed": observed,
                "operator": operator,
                "threshold": threshold,
                "passed": bool(passed),
            }

        def passing_count(name, predicate) -> int:
            data = values(name)
            return int(np.sum(predicate(data)))

        def exact_int64_vector(data) -> Optional[np.ndarray]:
            """Return integer-typed, exactly int64-compatible values only."""

            try:
                if isinstance(data, pd.Series):
                    if not pd.api.types.is_integer_dtype(data.dtype):
                        return None
                    raw = data.to_numpy()
                else:
                    raw = np.asarray(data)
                    if raw.dtype.kind not in "iu":
                        return None
                if raw.ndim != 1:
                    return None
                scalar_values = raw.tolist()
                if any(
                    isinstance(value, (bool, np.bool_))
                    or not isinstance(value, (int, np.integer))
                    or value < 0
                    or value > np.iinfo(np.int64).max
                    for value in scalar_values
                ):
                    return None
                return np.asarray(scalar_values, dtype=np.int64)
            except (TypeError, ValueError, OverflowError):
                return None

        development = protocol.get("development_protocol", {})
        expected_identity = development.get("expected")
        observed_identity = development.get("observed")
        try:
            pilot_flow_evaluations = int(
                protocol["flow_evaluations"]["pilot_proposal"]
            )
            final_stratum_flow_evaluations = int(
                protocol["flow_evaluations"]["final_stratum_exact"]
            )
            pilot_arm_flow_evaluations = int(
                protocol["flow_evaluations"]["total"]
            )
            metadata_identity = {
                "universe_start": int(self.metadata["observation_start"]),
                "universe_stop": int(self.metadata["observation_stop"]),
                "center": [
                    float(self.metadata["center"][0]),
                    float(self.metadata["center"][1]),
                ],
                "base_stratum_size": int(protocol["base_stratum_size"]),
                "promotions": int(
                    protocol["final_stratum_size"]
                    - protocol["base_stratum_size"]
                ),
                "final_stratum_size": int(protocol["final_stratum_size"]),
                "prefilter_candidates": int(
                    self.metadata["prefilter_candidates"]
                ),
                "n_global_draws": int(self.metadata["n_global_draws"]),
                "proposal_seed": int(protocol["production_seed"])
                - COMPLEMENT_DIAGNOSTIC_SEED_OFFSET,
                "production_seed": int(protocol["production_seed"]),
                "pilot_seed": int(protocol["pilot_seed"]),
                "candidate_backend": self.metadata["candidate_backend"],
                "diagnostic_object_chunk": int(self.metadata["object_chunk"]),
                "flow_atom_chunk": int(self.metadata["flow_atom_chunk"]),
                "tail_proposal_atom_chunk": int(
                    self.metadata["tail_proposal_atom_chunk"]
                ),
                "tail_proposal_object_chunk": int(
                    self.metadata["tail_proposal_object_chunk"]
                ),
                "pilot_atom_chunk": int(protocol["pilot_atom_chunk"]),
                "pilot_object_chunk": int(protocol["pilot_object_chunk"]),
                "tail_proposal_recipe": self.metadata["tail_proposal"],
                "pilot_stratum_recipe": protocol["recipe"],
                "pilot_stratum_recipe_sha256": protocol["recipe_sha256"],
                "hash_contract": protocol["hash_contract"],
                "pilot_flow_evaluations": pilot_flow_evaluations,
                "final_stratum_flow_evaluations": (
                    final_stratum_flow_evaluations
                ),
                "run_flow_evaluations": int(
                    self.metadata["flow_evaluations"]
                ),
                "mock_input_sha256": protocol["mock_input_sha256"],
            }
            pilot_flow_counts_matched = bool(
                pilot_arm_flow_evaluations
                == pilot_flow_evaluations + final_stratum_flow_evaluations
            )
        except (KeyError, TypeError, ValueError):
            metadata_identity = None
            pilot_flow_counts_matched = False
        identity_matched = bool(
            development.get("label") == PILOT_STRATUM_DEVELOPMENT_LABEL
            and expected_identity == PILOT_STRATUM_DEVELOPMENT_PROTOCOL
            and observed_identity == PILOT_STRATUM_DEVELOPMENT_PROTOCOL
            and metadata_identity == PILOT_STRATUM_DEVELOPMENT_PROTOCOL
            and pilot_flow_counts_matched
            and development.get("matched") is True
        )
        validated_object_ids = exact_int64_vector(frame["object_index"])
        object_ids = (
            validated_object_ids
            if validated_object_ids is not None
            else np.asarray([], dtype=np.int64)
        )
        expected_ids = np.arange(
            PILOT_STRATUM_DEVELOPMENT_PROTOCOL["universe_start"],
            PILOT_STRATUM_DEVELOPMENT_PROTOCOL["universe_stop"],
            dtype=np.int64,
        )
        ids_exact = bool(
            validated_object_ids is not None
            and np.array_equal(object_ids, expected_ids)
        )

        total_mass = values("mass_stratum") + values(
            "mass_complement_tail_proposal"
        )
        tail_standard_error = values(
            "mass_complement_tail_proposal_standard_error"
        )
        finite_positive_mass_and_se = int(
            np.sum(
                np.isfinite(total_mass)
                & (total_mass > 0.0)
                & np.isfinite(tail_standard_error)
                & (tail_standard_error > 0.0)
            )
        )
        base_count = passing_count(
            "pilot_stratum_base_count", lambda data: data == 16_384
        )
        promoted_count = passing_count(
            "pilot_stratum_promoted_count", lambda data: data == 4_096
        )
        final_count = int(
            np.sum(
                values("pilot_stratum_base_count")
                + values("pilot_stratum_promoted_count")
                == 20_480
            )
        )
        anchor_count = passing_count(
            "pilot_stratum_anchor_count", lambda data: data == 256
        )
        tail_draw_count = passing_count(
            "draws_tail_proposal_complement", lambda data: data == 16_384
        )
        finite_score_count = passing_count(
            "pilot_stratum_finite_score_atom_count", lambda data: data >= 256
        )
        fallback_count = passing_count(
            "pilot_stratum_fallback_fill_count", lambda data: data == 0
        )

        hash_fields = (
            "pilot_draw_indices_sha256",
            "pilot_probability_sha256",
            "pilot_scores_sha256",
            "base_stratum_indices_sha256",
            "anchor_indices_sha256",
            "promoted_indices_sha256",
            "final_stratum_indices_sha256",
            "conditioned_exclusion_indices_sha256",
        )
        call_hash_fields = tuple(
            name for name in hash_fields if name != "conditioned_exclusion_indices_sha256"
        )

        def is_sha256(value) -> bool:
            return bool(
                isinstance(value, str)
                and len(value) == 64
                and all(character in "0123456789abcdef" for character in value)
            )

        object_provenance = protocol.get("object_provenance", [])
        provenance_records_well_formed = bool(
            isinstance(object_provenance, list)
            and all(isinstance(record, dict) for record in object_provenance)
        )
        provenance_ids = exact_int64_vector(
            [record.get("object_index", -1) for record in object_provenance]
        )
        object_hashes_well_formed = bool(
            provenance_records_well_formed
            and len(object_provenance) == len(frame)
            and provenance_ids is not None
            and np.array_equal(provenance_ids, object_ids)
            and all(
                is_sha256(record.get(name))
                for record in object_provenance
                for name in hash_fields
            )
        )
        table_hashes_match = bool(
            object_hashes_well_formed
            and frame["pilot_stratum_final_indices_sha256"].astype(str).tolist()
            == [
                record["final_stratum_indices_sha256"]
                for record in object_provenance
            ]
        )
        try:
            recomputed_manifest = _pilot_stratum_run_manifest_sha256(
                [
                    (
                        record["object_index"],
                        record["final_stratum_indices_sha256"],
                    )
                    for record in object_provenance
                ]
            )
        except (KeyError, TypeError, ValueError):
            recomputed_manifest = None
        run_manifest_matches = bool(
            recomputed_manifest is not None
            and protocol.get("run_manifest_sha256") == recomputed_manifest
        )
        execution_calls = protocol.get("execution_call_provenance", [])
        covered_objects = 0
        calls_well_formed = isinstance(execution_calls, list)
        for record in execution_calls if calls_well_formed else ():
            if not isinstance(record, dict):
                calls_well_formed = False
                break
            try:
                call_identity = exact_int64_vector(
                    [
                        record["object_start"],
                        record["object_stop"],
                        record["objects"],
                    ]
                )
            except KeyError:
                calls_well_formed = False
                break
            if call_identity is None:
                calls_well_formed = False
                break
            call_start, call_stop, call_objects = call_identity.tolist()
            if (
                call_start < 0
                or call_stop <= call_start
                or call_objects != call_stop - call_start
                or covered_objects + call_objects > len(object_ids)
                or not np.array_equal(
                    object_ids[
                        covered_objects : covered_objects + call_objects
                    ],
                    np.arange(call_start, call_stop, dtype=np.int64),
                )
                or not all(is_sha256(record.get(name)) for name in call_hash_fields)
            ):
                calls_well_formed = False
                break
            covered_objects += call_objects
        calls_well_formed = bool(
            calls_well_formed
            and covered_objects == len(object_ids)
            and protocol.get("hash_contract") == PILOT_STRATUM_HASH_CONTRACT
            and protocol.get("recipe_sha256") == PILOT_STRATUM_RECIPE.sha256
        )
        hash_manifest_matched = bool(
            object_hashes_well_formed
            and table_hashes_match
            and run_manifest_matches
            and calls_well_formed
        )

        integrity_gates = {
            "absolute_object_ids_exact": gate(ids_exact, "is", True, ids_exact),
            "development_protocol_identity_matched": gate(
                identity_matched, "is", True, identity_matched
            ),
            "finite_positive_total_masses_and_tail_standard_errors": gate(
                finite_positive_mass_and_se,
                "==",
                expected_objects,
                finite_positive_mass_and_se == expected_objects,
            ),
            "base_stratum_count_all_objects": gate(
                base_count, "==", expected_objects, base_count == expected_objects
            ),
            "promoted_count_all_objects": gate(
                promoted_count,
                "==",
                expected_objects,
                promoted_count == expected_objects,
            ),
            "final_stratum_count_all_objects": gate(
                final_count, "==", expected_objects, final_count == expected_objects
            ),
            "anchor_count_all_objects": gate(
                anchor_count, "==", expected_objects, anchor_count == expected_objects
            ),
            "tail_draw_count_all_objects": gate(
                tail_draw_count,
                "==",
                expected_objects,
                tail_draw_count == expected_objects,
            ),
            "finite_score_count_at_least_256_all_objects": gate(
                finite_score_count,
                "==",
                expected_objects,
                finite_score_count == expected_objects,
            ),
            "zero_fallback_all_objects": gate(
                fallback_count,
                "==",
                expected_objects,
                fallback_count == expected_objects,
            ),
            "hashes_and_run_manifest_internally_matched": gate(
                hash_manifest_matched, "is", True, hash_manifest_matched
            ),
        }

        tail_k = values("pareto_k_tail_proposal_complement")
        finite_tail_k = np.isfinite(tail_k)
        below_0_5_count = int(np.sum(finite_tail_k & (tail_k < 0.5)))
        below_0_7_count = int(np.sum(finite_tail_k & (tail_k < 0.7)))
        tail_mass = values("mass_complement_tail_proposal")
        complement_relative_error = np.divide(
            tail_standard_error,
            np.abs(tail_mass),
            out=np.full(len(frame), np.nan, dtype=np.float64),
            where=np.abs(tail_mass) > 0.0,
        )
        relative_error_finite = bool(
            len(complement_relative_error) == expected_objects
            and np.isfinite(complement_relative_error).all()
        )
        complement_relative_error_median = (
            float(np.median(complement_relative_error))
            if relative_error_finite
            else float("nan")
        )
        complement_relative_error_p90 = (
            float(
                np.percentile(
                    complement_relative_error, 90, method="higher"
                )
            )
            if relative_error_finite
            else float("nan")
        )
        tail_gates = {
            "pareto_k_defined_all_objects": gate(
                int((~finite_tail_k).sum()),
                "undefined_count ==",
                0,
                len(tail_k) == expected_objects and finite_tail_k.all(),
            ),
            "pareto_k_below_0_5_count": gate(
                below_0_5_count,
                ">=",
                thresholds["pareto_k_below_0_5_count"],
                below_0_5_count >= thresholds["pareto_k_below_0_5_count"],
            ),
            "pareto_k_below_0_7_count": gate(
                below_0_7_count,
                ">=",
                thresholds["pareto_k_below_0_7_count"],
                below_0_7_count >= thresholds["pareto_k_below_0_7_count"],
            ),
            "complement_relative_error_median": gate(
                complement_relative_error_median,
                "<=",
                thresholds["complement_relative_error_median"],
                np.isfinite(complement_relative_error_median)
                and complement_relative_error_median
                <= thresholds["complement_relative_error_median"],
            ),
            "complement_relative_error_p90_higher": gate(
                complement_relative_error_p90,
                "<=",
                thresholds["complement_relative_error_p90_higher"],
                np.isfinite(complement_relative_error_p90)
                and complement_relative_error_p90
                <= thresholds["complement_relative_error_p90_higher"],
            ),
        }

        total_relative_error = np.divide(
            tail_standard_error,
            np.abs(total_mass),
            out=np.full(len(frame), np.nan, dtype=np.float64),
            where=np.abs(total_mass) > 0.0,
        )
        finite_total_relative_error = bool(
            len(total_relative_error) == expected_objects
            and np.isfinite(total_relative_error).all()
        )
        if finite_total_relative_error:
            total_relative_error_median = float(np.median(total_relative_error))
            total_relative_error_p90 = float(
                np.percentile(total_relative_error, 90, method="higher")
            )
            total_relative_error_squared = np.square(total_relative_error)
            mean_total_relative_error_squared = float(
                np.mean(total_relative_error_squared)
            )
            total_relative_error_squared_p90 = float(
                np.percentile(
                    total_relative_error_squared, 90, method="higher"
                )
            )
        else:
            total_relative_error_median = float("nan")
            total_relative_error_p90 = float("nan")
            mean_total_relative_error_squared = float("nan")
            total_relative_error_squared_p90 = float("nan")

        phase_seconds = self.metadata.get("phase_seconds", {})
        method_phase_values = []
        for name in PILOT_STRATUM_METHOD_TIME_COMPONENTS:
            try:
                value = float(phase_seconds[name])
            except (KeyError, TypeError, ValueError):
                value = float("nan")
            method_phase_values.append(value)
        method_phase_values_array = np.asarray(
            method_phase_values, dtype=np.float64
        )
        method_seconds = (
            float(np.sum(method_phase_values_array))
            if np.isfinite(method_phase_values_array).all()
            and np.all(method_phase_values_array >= 0.0)
            and float(np.sum(method_phase_values_array)) > 0.0
            else float("nan")
        )
        method_time_ratio = (
            method_seconds / baseline["method_seconds"]
            if np.isfinite(method_seconds)
            else float("nan")
        )
        total_elapsed_seconds = float(
            self.metadata.get("elapsed_seconds", float("nan"))
        )
        total_elapsed_ratio = (
            total_elapsed_seconds / baseline["method_seconds"]
            if np.isfinite(total_elapsed_seconds) and total_elapsed_seconds >= 0.0
            else float("nan")
        )
        efficiency_mean = (
            method_time_ratio
            * mean_total_relative_error_squared
            / baseline["mean_total_relative_error_squared"]
            if np.isfinite(method_time_ratio)
            and np.isfinite(mean_total_relative_error_squared)
            else float("nan")
        )
        efficiency_p90 = (
            method_time_ratio
            * total_relative_error_squared_p90
            / baseline["total_relative_error_squared_p90_higher"]
            if np.isfinite(method_time_ratio)
            and np.isfinite(total_relative_error_squared_p90)
            else float("nan")
        )
        efficiency_gates = {
            "mean_relative_variance_time_efficiency": gate(
                efficiency_mean,
                "<=",
                thresholds["efficiency_mean"],
                np.isfinite(efficiency_mean)
                and efficiency_mean <= thresholds["efficiency_mean"],
            ),
            "p90_higher_relative_variance_time_efficiency": gate(
                efficiency_p90,
                "<=",
                thresholds["efficiency_p90_higher"],
                np.isfinite(efficiency_p90)
                and efficiency_p90 <= thresholds["efficiency_p90_higher"],
            ),
        }
        integrity_passed = all(item["passed"] for item in integrity_gates.values())
        tail_passed = all(item["passed"] for item in tail_gates.values())
        efficiency_passed = all(item["passed"] for item in efficiency_gates.values())

        promoted_share = np.divide(
            values("mass_stratum_promoted_exact"),
            values("mass_stratum"),
            out=np.full(len(frame), np.nan, dtype=np.float64),
            where=values("mass_stratum") > 0.0,
        )
        return {
            "protocol": protocol,
            "development_protocol_matched": identity_matched,
            "quantile_method": "higher",
            "promoted_count_percentiles": percentiles(
                frame["pilot_stratum_promoted_count"]
            ),
            "anchor_count_percentiles": percentiles(
                frame["pilot_stratum_anchor_count"]
            ),
            "neighbor_count_percentiles": percentiles(
                frame["pilot_stratum_neighbor_count"]
            ),
            "fallback_fill_count_percentiles": percentiles(
                frame["pilot_stratum_fallback_fill_count"]
            ),
            "max_neighbor_query_width_percentiles": percentiles(
                frame["pilot_stratum_max_neighbor_query_width"]
            ),
            "promoted_outside_prefilter_count_percentiles": percentiles(
                frame["pilot_stratum_promoted_outside_prefilter_count"]
            ),
            "mass_promoted_exact_percentiles": percentiles(
                frame["mass_stratum_promoted_exact"]
            ),
            "promoted_share_of_exact_stratum_percentiles": percentiles(
                promoted_share
            ),
            "complement_relative_error_median": complement_relative_error_median,
            "complement_relative_error_p90_higher": (
                complement_relative_error_p90
            ),
            "total_relative_error_median": total_relative_error_median,
            "total_relative_error_p90_higher": total_relative_error_p90,
            "mean_total_relative_error_squared": (
                mean_total_relative_error_squared
            ),
            "total_relative_error_squared_p90_higher": (
                total_relative_error_squared_p90
            ),
            "baseline": dict(baseline),
            "method_time_components": list(PILOT_STRATUM_METHOD_TIME_COMPONENTS),
            "method_seconds": method_seconds,
            "method_time_ratio": method_time_ratio,
            "total_elapsed_seconds": total_elapsed_seconds,
            "total_elapsed_ratio": total_elapsed_ratio,
            "efficiency_mean": efficiency_mean,
            "efficiency_p90_higher": efficiency_p90,
            "gates": {
                "predeclared": identity_matched,
                "thresholds_predeclared": True,
                "integrity": integrity_gates,
                "tail_reliability": tail_gates,
                "efficiency": efficiency_gates,
                "integrity_passed": integrity_passed,
                "tail_reliability_passed": tail_passed,
                "efficiency_passed": efficiency_passed,
                "passed": bool(
                    integrity_passed and tail_passed and efficiency_passed
                ),
                "policy": (
                    "integrity, tail reliability, and fair efficiency must all pass"
                ),
            },
        }

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
    include_pilot_stratum: bool = False,
    pilot_seed: Optional[int] = None,
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
    if include_pilot_stratum and not include_tail_proposal:
        raise ValueError(
            "the pilot stratum requires the tail-proposal diagnostic"
        )
    if include_pilot_stratum and include_exact_oracle:
        raise ValueError(
            "the pilot stratum cannot be combined with exact-oracle v1"
        )
    if include_pilot_stratum and pilot_seed is None:
        raise ValueError("the pilot stratum requires an explicit pilot seed")
    if not include_pilot_stratum and pilot_seed is not None:
        raise ValueError("a pilot seed requires the pilot-stratum diagnostic")
    if include_pilot_stratum and int(pilot_seed) < 0:
        raise ValueError("the pilot seed must be non-negative")
    if include_pilot_stratum and int(pilot_seed) == int(seed):
        raise ValueError("pilot and production seeds must be different")
    if include_exact_oracle and getattr(likelihood, "selection", None) is not None:
        raise ValueError(
            "the exact oracle requires measured selection to be disabled so "
            "frozen absolute row ids are preserved"
        )
    if include_pilot_stratum and getattr(likelihood, "selection", None) is not None:
        raise ValueError(
            "the pilot stratum requires measured selection to be disabled so "
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
    if include_pilot_stratum:
        phase_seconds.update(
            {
                "pilot_proposal_draw": 0.0,
                "pilot_proposal_flow": 0.0,
                "pilot_promotion": 0.0,
                "pilot_stratum_exact_flow": 0.0,
            }
        )
    exact_records = {}
    exact_flow_evaluations = 0
    exact_active_atoms = None
    exact_atoms_visited = None
    exact_atom_blocks = None
    exact_active_indices_sha256 = None
    pilot_flow_evaluations = 0
    pilot_stratum_flow_evaluations = 0
    pilot_object_provenance = []
    pilot_call_provenance = []
    pilot_base_stratum_size = None
    pilot_final_stratum_size = None

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

        pilot_promotion = None
        pilot_stratum_log_c = None
        if include_pilot_stratum:
            # The discovery stream is conditioned only on the deterministic
            # base stratum S0 and is discarded after selecting S.  Its seed is
            # deliberately distinct from both production reference streams.
            phase_started = time.perf_counter()
            pilot_draw = proposal.draw_tilted_mis(
                stratum,
                observed,
                n_draws=PILOT_STRATUM_RECIPE.pilot_draws,
                seed=int(pilot_seed),
                object_ids=absolute_ids,
                device=likelihood.flow_model.device,
                recipe=PILOT_STRATUM_RECIPE.pilot_proposal,
                atom_chunk=PILOT_STRATUM_RECIPE.atom_chunk,
                object_chunk=PILOT_STRATUM_RECIPE.object_chunk,
            )
            phase_seconds["pilot_proposal_draw"] += (
                time.perf_counter() - phase_started
            )
            phase_started = time.perf_counter()
            pilot_log_ratio = (
                likelihood.log_importance_weights_tensor(
                    observed,
                    float(center[0]),
                    float(center[1]),
                    atom_indices=pilot_draw.indices,
                    proposal_probability=pilot_draw.probability,
                    observed_targets=observed_targets,
                    object_chunk=len(observed),
                    atom_chunk=atom_chunk,
                )
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )
            if np.isnan(pilot_log_ratio).any() or np.isposinf(
                pilot_log_ratio
            ).any():
                raise RuntimeError(
                    "pilot importance log ratios may be finite or -inf only"
                )
            pilot_count = int(pilot_draw.indices.size)
            pilot_flow_evaluations += pilot_count
            flow_evaluations += pilot_count
            phase_seconds["pilot_proposal_flow"] += (
                time.perf_counter() - phase_started
            )

            # Keep the explicit recipe score itself in provenance.  This stays
            # in the log domain, so no exponential or per-object centering is
            # needed: log(c^2/qbar) = 2 log(c/qbar) + log(qbar).
            pilot_scores = 2.0 * pilot_log_ratio + np.log(
                pilot_draw.probability
            )
            phase_started = time.perf_counter()
            pilot_promotion = proposal.promote_exact_stratum(
                stratum,
                observed,
                pilot_draw,
                pilot_scores,
                recipe=PILOT_STRATUM_RECIPE,
            )
            phase_seconds["pilot_promotion"] += (
                time.perf_counter() - phase_started
            )
            if pilot_promotion.provenance.pilot_seed != int(pilot_seed):
                raise RuntimeError("pilot promotion recorded the wrong seed")
            if (
                pilot_promotion.provenance.recipe_sha256
                != PILOT_STRATUM_RECIPE.sha256
            ):
                raise RuntimeError("pilot promotion recorded the wrong recipe")
            pilot_provenance = pilot_promotion.provenance
            if (
                pilot_provenance.pilot_atom_chunk
                != PILOT_STRATUM_RECIPE.atom_chunk
                or pilot_provenance.pilot_object_chunk
                != PILOT_STRATUM_RECIPE.object_chunk
            ):
                raise RuntimeError("pilot promotion recorded the wrong chunks")
            current_base_size = int(pilot_promotion.base_size)
            current_final_size = int(
                pilot_promotion.candidates.indices.shape[1]
            )
            if pilot_base_stratum_size is None:
                pilot_base_stratum_size = current_base_size
                pilot_final_stratum_size = current_final_size
            elif (
                pilot_base_stratum_size != current_base_size
                or pilot_final_stratum_size != current_final_size
            ):
                raise RuntimeError(
                    "pilot-stratum widths changed between object batches"
                )

            object_hash_fields = {
                "pilot_draw_indices_sha256": (
                    pilot_provenance.object_pilot_draw_indices_sha256
                ),
                "pilot_probability_sha256": (
                    pilot_provenance.object_pilot_probability_sha256
                ),
                "pilot_scores_sha256": (
                    pilot_provenance.object_pilot_scores_sha256
                ),
                "base_stratum_indices_sha256": (
                    pilot_provenance.object_base_stratum_indices_sha256
                ),
                "anchor_indices_sha256": (
                    pilot_provenance.object_anchor_indices_sha256
                ),
                "promoted_indices_sha256": (
                    pilot_provenance.object_promoted_indices_sha256
                ),
                "final_stratum_indices_sha256": (
                    pilot_provenance.object_final_stratum_indices_sha256
                ),
                "conditioned_exclusion_indices_sha256": (
                    pilot_draw.object_excluded_candidate_indices_sha256
                ),
            }
            if any(
                len(values) != len(absolute_ids)
                for values in object_hash_fields.values()
            ):
                raise RuntimeError(
                    "pilot promotion recorded incomplete per-object hashes"
                )
            for local_row, object_id in enumerate(absolute_ids):
                pilot_object_provenance.append(
                    {
                        "object_index": int(object_id),
                        **{
                            name: values[local_row]
                            for name, values in object_hash_fields.items()
                        },
                    }
                )
            pilot_call_provenance.append(
                {
                    "object_start": int(absolute_ids[0]),
                    "object_stop": int(absolute_ids[-1]) + 1,
                    "objects": int(len(absolute_ids)),
                    "pilot_draw_indices_sha256": (
                        pilot_provenance.call_pilot_draw_indices_sha256
                    ),
                    "pilot_probability_sha256": (
                        pilot_provenance.call_pilot_probability_sha256
                    ),
                    "pilot_scores_sha256": (
                        pilot_provenance.call_pilot_scores_sha256
                    ),
                    "base_stratum_indices_sha256": (
                        pilot_provenance.call_base_stratum_indices_sha256
                    ),
                    "anchor_indices_sha256": (
                        pilot_provenance.call_anchor_indices_sha256
                    ),
                    "promoted_indices_sha256": (
                        pilot_provenance.call_promoted_indices_sha256
                    ),
                    "final_stratum_indices_sha256": (
                        pilot_provenance.call_final_stratum_indices_sha256
                    ),
                }
            )

            stratum = pilot_promotion.candidates
            phase_started = time.perf_counter()
            pilot_stratum_log_c = (
                likelihood.log_importance_weights_tensor(
                    observed,
                    float(center[0]),
                    float(center[1]),
                    atom_indices=stratum.indices,
                    proposal_probability=np.ones_like(
                        stratum.indices, dtype=np.float64
                    ),
                    observed_targets=observed_targets,
                    object_chunk=len(observed),
                    atom_chunk=atom_chunk,
                )
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )
            stratum_count = int(stratum.indices.size)
            pilot_stratum_flow_evaluations += stratum_count
            flow_evaluations += stratum_count
            phase_seconds["pilot_stratum_exact_flow"] += (
                time.perf_counter() - phase_started
            )

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
            mass_prefilter_complement = float(c_prefilter[~in_stratum].sum())

            pilot_record = {}
            if pilot_promotion is None:
                mass_stratum = float(c_prefilter[in_stratum].sum())
            else:
                stratum_c = np.exp(pilot_stratum_log_c[row])
                base_size = int(pilot_promotion.base_size)
                promoted_ids = pilot_promotion.promoted_indices[row]
                promoted_c = stratum_c[base_size:]
                valid_anchors = pilot_promotion.anchor_indices[row][
                    pilot_promotion.anchor_valid[row]
                ]
                anchor_member = np.isin(promoted_ids, valid_anchors)
                fallback_count = int(
                    pilot_promotion.fallback_fill_counts[row]
                )
                fallback_member = np.zeros(len(promoted_ids), dtype=bool)
                if fallback_count:
                    fallback_member[-fallback_count:] = True
                neighbor_member = ~(anchor_member | fallback_member)
                promoted_in_prefilter = _membership(
                    lookup, prefilter_ids, promoted_ids
                )

                mass_stratum = float(stratum_c.sum())
                pilot_record = {
                    "pilot_stratum_base_count": base_size,
                    "pilot_stratum_promoted_count": int(len(promoted_ids)),
                    "pilot_stratum_anchor_count": int(anchor_member.sum()),
                    "pilot_stratum_neighbor_count": int(neighbor_member.sum()),
                    "pilot_stratum_fallback_fill_count": fallback_count,
                    "pilot_stratum_max_neighbor_query_width": int(
                        pilot_promotion.max_neighbor_query_width[row]
                    ),
                    "pilot_stratum_unique_drawn_atom_count": int(
                        pilot_promotion.pilot_unique_counts[row]
                    ),
                    "pilot_stratum_finite_score_atom_count": int(
                        pilot_promotion.pilot_finite_score_counts[row]
                    ),
                    "pilot_stratum_promoted_outside_prefilter_count": int(
                        (~promoted_in_prefilter).sum()
                    ),
                    "pilot_stratum_anchor_outside_prefilter_count": int(
                        (anchor_member & ~promoted_in_prefilter).sum()
                    ),
                    "pilot_stratum_neighbor_outside_prefilter_count": int(
                        (neighbor_member & ~promoted_in_prefilter).sum()
                    ),
                    "pilot_stratum_fallback_outside_prefilter_count": int(
                        (fallback_member & ~promoted_in_prefilter).sum()
                    ),
                    "mass_stratum_base_exact": float(
                        stratum_c[:base_size].sum()
                    ),
                    "mass_stratum_promoted_exact": float(promoted_c.sum()),
                    "mass_stratum_pilot_anchor_exact": float(
                        promoted_c[anchor_member].sum()
                    ),
                    "mass_stratum_pilot_neighbor_exact": float(
                        promoted_c[neighbor_member].sum()
                    ),
                    "mass_stratum_pilot_fallback_exact": float(
                        promoted_c[fallback_member].sum()
                    ),
                    "pilot_stratum_final_indices_sha256": (
                        pilot_promotion.provenance.object_final_stratum_indices_sha256[
                            row
                        ]
                    ),
                }

            drawn_in_stratum = prior_draw.local_member[row]
            drawn_in_prefilter = _membership(lookup, prefilter_ids, drawn_ids)
            ratio = np.exp(prior_log_ratio[row])

            if pilot_promotion is None:
                # Preserve the ordinary diagnostic's exact mask and streams.
                outside = ~drawn_in_prefilter
            else:
                # A promoted atom can lie beyond the deterministic prefilter.
                # It belongs to the exact final stratum and must not leak back
                # into the outside-prefilter diagnostic.
                outside = ~drawn_in_prefilter & ~drawn_in_stratum
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
                    **pilot_record,
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
    pilot_stratum_metadata = None
    if include_pilot_stratum:
        ordered_object_provenance = sorted(
            pilot_object_provenance, key=lambda item: item["object_index"]
        )
        ordered_object_hashes = [
            (item["object_index"], item["final_stratum_indices_sha256"])
            for item in ordered_object_provenance
        ]
        development_observed = {
            "universe_start": int(object_id_offset + start),
            "universe_stop": int(object_id_offset + stop),
            "center": [float(center[0]), float(center[1])],
            "base_stratum_size": int(pilot_base_stratum_size),
            "promotions": int(
                pilot_final_stratum_size - pilot_base_stratum_size
            ),
            "final_stratum_size": int(pilot_final_stratum_size),
            "prefilter_candidates": int(prefilter_candidates),
            "n_global_draws": int(n_global_draws),
            "proposal_seed": int(seed) - COMPLEMENT_DIAGNOSTIC_SEED_OFFSET,
            "production_seed": int(seed),
            "pilot_seed": int(pilot_seed),
            "candidate_backend": candidate_backend,
            "diagnostic_object_chunk": int(object_chunk),
            "flow_atom_chunk": int(atom_chunk),
            "tail_proposal_atom_chunk": tail_proposal_atom_chunk,
            "tail_proposal_object_chunk": tail_proposal_object_chunk,
            "pilot_atom_chunk": int(PILOT_STRATUM_RECIPE.atom_chunk),
            "pilot_object_chunk": int(PILOT_STRATUM_RECIPE.object_chunk),
            "tail_proposal_recipe": asdict(TAIL_PROPOSAL_RECIPE),
            "pilot_stratum_recipe": asdict(PILOT_STRATUM_RECIPE),
            "pilot_stratum_recipe_sha256": PILOT_STRATUM_RECIPE.sha256,
            "hash_contract": PILOT_STRATUM_HASH_CONTRACT,
            "pilot_flow_evaluations": int(pilot_flow_evaluations),
            "final_stratum_flow_evaluations": int(
                pilot_stratum_flow_evaluations
            ),
            "run_flow_evaluations": int(flow_evaluations),
            "mock_input_sha256": (
                None
                if confirmation_mock_sha256 is None
                else dict(confirmation_mock_sha256)
            ),
        }
        pilot_stratum_metadata = {
            "mode": "diagnostic_only_discarded_independent_pilot",
            "recipe": asdict(PILOT_STRATUM_RECIPE),
            "recipe_sha256": PILOT_STRATUM_RECIPE.sha256,
            "pilot_proposal_recipe": asdict(
                PILOT_STRATUM_RECIPE.pilot_proposal
            ),
            "pilot_seed": int(pilot_seed),
            "production_seed": int(seed),
            "pilot_seed_offset": PILOT_STRATUM_DIAGNOSTIC_SEED_OFFSET,
            "production_seed_offset": COMPLEMENT_DIAGNOSTIC_SEED_OFFSET,
            "mock_input_sha256": (
                None
                if confirmation_mock_sha256 is None
                else dict(confirmation_mock_sha256)
            ),
            "streams_independent": bool(int(pilot_seed) != int(seed)),
            "score_formula": "2 * log_target_over_qbar + log(qbar)",
            "score_centering": None,
            "pilot_draws_enter_estimate": False,
            "final_stratum_proposal_probability": 1.0,
            "pilot_atom_chunk": PILOT_STRATUM_RECIPE.atom_chunk,
            "pilot_object_chunk": PILOT_STRATUM_RECIPE.object_chunk,
            "base_stratum_size": int(pilot_base_stratum_size),
            "final_stratum_size": int(pilot_final_stratum_size),
            "flow_evaluations": {
                "pilot_proposal": int(pilot_flow_evaluations),
                "final_stratum_exact": int(
                    pilot_stratum_flow_evaluations
                ),
                "total": int(
                    pilot_flow_evaluations
                    + pilot_stratum_flow_evaluations
                ),
            },
            "hash_contract": PILOT_STRATUM_HASH_CONTRACT,
            "manifest_contract": (
                "sha256(domain || count || sorted(uint64_be(object_id), "
                "object_final_stratum_sha256))"
            ),
            "object_provenance": ordered_object_provenance,
            "execution_call_provenance": pilot_call_provenance,
            "run_manifest_sha256": _pilot_stratum_run_manifest_sha256(
                ordered_object_hashes
            ),
            "development_protocol": {
                "label": PILOT_STRATUM_DEVELOPMENT_LABEL,
                "expected": deepcopy(PILOT_STRATUM_DEVELOPMENT_PROTOCOL),
                "observed": development_observed,
                "matched": bool(
                    development_observed
                    == PILOT_STRATUM_DEVELOPMENT_PROTOCOL
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
    if pilot_stratum_metadata is not None:
        metadata["pilot_stratum"] = pilot_stratum_metadata
    return ComplementDiagnostic(pd.DataFrame(rows), metadata)
