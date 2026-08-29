"""Measure complement mass and compare production tail proposals.

For every observation, the diagnostic evaluates the full prefilter exactly,
then pairs the prior-complement estimator with the production fixed
component-conditioned deterministic-MIS sampler.  Both sampled arms consume
the same per-object random record and use absolute object ids, so a partitioned
run is identical to the corresponding rows of an unsliced run.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import inspect
from typing import Optional

import numpy as np
import pandas as pd
import torch

from .catalogue_sampling import (
    TAIL_PROPOSAL_RECIPE,
    pareto_tail_index,
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
        if tail_recipe is not None:
            component_mass = {
                name: percentiles(
                    frame[f"tail_proposal_component_complement_mass_{name}"]
                )
                for name in tail_recipe["component_names"]
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

        return {
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
            "flow_evaluations": int(self.metadata["flow_evaluations"]),
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

    measurements = mock.measurements
    stop = len(measurements) if observation_stop is None else int(observation_stop)
    stop = min(stop, len(measurements))
    start = int(observation_start)
    if start < 0 or start >= stop:
        raise ValueError("empty observation window")

    torch_device = (
        likelihood.flow_model.device if candidate_backend == "torch" else None
    )
    flow_device = str(likelihood.flow_model.device)
    tail_proposal_atom_chunk = None
    if include_tail_proposal:
        atom_chunk_parameter = inspect.signature(
            proposal.draw_tilted_mis
        ).parameters.get("atom_chunk")
        if (
            atom_chunk_parameter is not None
            and atom_chunk_parameter.default is not inspect.Parameter.empty
        ):
            tail_proposal_atom_chunk = int(atom_chunk_parameter.default)
    lookup = np.zeros(len(proposal.prior_weights), dtype=bool)
    rows = []
    flow_evaluations = 0

    for object_start in range(start, stop, object_chunk):
        object_stop = min(object_start + object_chunk, stop)
        observed = measurements.iloc[object_start:object_stop].reset_index(drop=True)
        absolute_ids = object_id_offset + np.arange(
            object_start, object_stop, dtype=np.int64
        )
        observed_targets = likelihood.observed_target_tensor(observed)

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

        # Both arms use production samplers, the same absolute object ids, the
        # same seed, and the same fixed-width record's atom-uniform column.
        # Their estimates are therefore paired by common random numbers even
        # when this call covers only a slice of a larger catalogue.
        prior_draw = proposal.draw_stratified(
            stratum,
            n_draws=n_global_draws,
            seed=seed,
            object_ids=absolute_ids,
        )
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

        tail_draw = None
        tail_ratio = None
        if include_tail_proposal:
            tail_draw = proposal.draw_tilted_mis(
                stratum,
                observed,
                n_draws=n_global_draws,
                seed=seed,
                object_ids=absolute_ids,
                device=likelihood.flow_model.device,
            )
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
                record.update(
                    {
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
        "flow_evaluations": int(flow_evaluations),
        "tail_proposal": (
            asdict(TAIL_PROPOSAL_RECIPE) if include_tail_proposal else None
        ),
        "tail_proposal_common_random_numbers": bool(include_tail_proposal),
    }
    return ComplementDiagnostic(pd.DataFrame(rows), metadata)
