"""Locate the target mass and the heavy weight tail outside the exact stratum.

The stratified estimator (``doc/WORKLOG.md`` cont.313) sums the candidate
stratum ``S`` exactly and estimates its complement by sampling atoms from the
detected prior ``pi``.  That complement estimator is what still carries a
Pareto tail index above the 0.7 reliability threshold for 54% of objects, and
its ratio ``c_j / pi_j`` is unbounded because ``pi`` knows nothing about the
observation.

Any better tail proposal has to know *where* the leftover mass sits.  Two
candidate designs make opposite predictions:

  * If most of the complement mass and most of the heavy tail live in
    ``P \\ S`` -- inside the 131,072-atom location prefilter but outside the
    16,384 atoms kept for exact evaluation -- then a third stratum sampled with
    the cheap diagonal Gaussian proxy, which is already computed over ``P`` and
    then discarded, would cap the ratio where it actually bites.

  * If the complement mass is instead spread over the ~12.6 million atoms
    outside ``P`` and the tail there is just as heavy, no prefilter-restricted
    proposal can help, and the tail proposal has to be defined over the whole
    catalogue.

This module measures the split rather than assuming it.  For each object it
evaluates the exact unnormalised target ``c_j = pi_j Pdet_j L_j`` on every atom
of ``P``, so ``sum_S c`` and ``sum_{P\\S} c`` are exact, and estimates the
remaining mass outside ``P`` from an independent prior sample.  It then reports
the Pareto tail index of the prior-sampled complement separately on its
``P \\ S`` and outside-``P`` parts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import torch

from .catalogue_sampling import (
    WholeCatalogueProxy as _WholeCatalogueProxy,
    pareto_tail_index,
)


@dataclass
class ComplementDiagnostic:
    """Per-object mass split and tail indices, plus the run's identity."""

    table: pd.DataFrame
    metadata: dict

    def summary(self) -> dict:
        frame = self.table
        # A requested measurement that produced no column means the code path
        # never ran.  That is a defect and must not be reported as an absent
        # result: jobs 16122607/08/09 burned three GPU allocations returning
        # nulls because the proxy construction had been made unreachable.
        for requested, column in (
            (self.metadata.get("tilt_delta") is not None, "pareto_k_tilted_complement"),
            (bool(self.metadata.get("exact_top")), "pareto_k_tilted_exact_top"),
        ):
            if requested and column not in frame:
                raise ValueError(
                    f"{column} was requested but the diagnostic recorded none"
                )
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
            "pareto_k_tilted_complement_percentiles": (
                percentiles(frame["pareto_k_tilted_complement"])
                if "pareto_k_tilted_complement" in frame
                else None
            ),
            "tilted_over_prior_relative_error_percentiles": (
                percentiles(
                    (
                        frame["mass_complement_tilted_standard_error"]
                        / np.maximum(frame["mass_complement_tilted"], 0.0)
                    )
                    / (
                        frame["mass_complement_prior_standard_error"]
                        / np.maximum(frame["mass_complement_prior"], 0.0)
                    )
                )
                if "mass_complement_tilted" in frame
                else None
            ),
            "pareto_k_tilted_exact_top_percentiles": (
                percentiles(frame["pareto_k_tilted_exact_top"])
                if "pareto_k_tilted_exact_top" in frame
                else None
            ),
            "tilted_exact_top_relative_error_percentiles": (
                percentiles(
                    frame["mass_complement_tilted_exact_top_standard_error"]
                    / np.maximum(frame["mass_complement_tilted_exact_top"], 0.0)
                )
                if "mass_complement_tilted_exact_top" in frame
                else None
            ),
            "atoms_exact_top_new_percentiles": (
                percentiles(frame["atoms_exact_top_new"])
                if "atoms_exact_top_new" in frame
                else None
            ),
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
    object_chunk: int = 32,
    atom_chunk: int = 4096,
    candidate_backend: str = "torch",
    tilt_delta: Optional[float] = None,
    tilt_temperature: float = 1.0,
    exact_top: int = 0,
) -> ComplementDiagnostic:
    """Split the target mass into stratum, prefilter complement, and beyond."""

    if prefilter_candidates < n_candidates:
        raise ValueError("prefilter cannot be smaller than the exact stratum")
    if n_global_draws <= 0:
        raise ValueError("n_global_draws must be positive")

    measurements = mock.measurements
    stop = len(measurements) if observation_stop is None else int(observation_stop)
    stop = min(stop, len(measurements))
    start = int(observation_start)
    if start >= stop:
        raise ValueError("empty observation window")

    torch_device = (
        likelihood.flow_model.device if candidate_backend == "torch" else None
    )
    lookup = np.zeros(len(proposal.prior_weights), dtype=bool)
    prior = proposal.prior_weights
    proxy = None
    if exact_top < 0:
        raise ValueError("exact_top must be non-negative")
    if exact_top and tilt_delta is None:
        raise ValueError("exact_top needs a tilted proposal to rank atoms by")
    if tilt_delta is not None:
        if not 0.0 < tilt_delta < 1.0:
            raise ValueError("tilt_delta must lie strictly between zero and one")
        proxy = _WholeCatalogueProxy(proposal, likelihood.flow_model.device)
    rows = []
    flow_evaluations = 0

    for object_start in range(start, stop, object_chunk):
        object_stop = min(object_start + object_chunk, stop)
        observed = measurements.iloc[object_start:object_stop].reset_index(drop=True)
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

        # Independent prior sample.  This is deliberately not the estimator's
        # own draw stream: the question is where the complement mass is, not
        # how one particular seed happened to land.
        global_indices = np.empty(
            (len(observed), n_global_draws), dtype=np.int64
        )
        for row in range(len(observed)):
            rng = np.random.default_rng(
                np.random.SeedSequence([int(seed), int(object_start + row)])
            )
            global_indices[row] = np.searchsorted(
                proposal.global_cdf, rng.random(n_global_draws), side="right"
            )
        global_log_c = (
            likelihood.log_importance_weights_tensor(
                observed,
                float(center[0]),
                float(center[1]),
                atom_indices=global_indices,
                proposal_probability=np.ones_like(global_indices, dtype=np.float64),
                observed_targets=observed_targets,
                object_chunk=len(observed),
                atom_chunk=atom_chunk,
            )
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        flow_evaluations += int(global_indices.size)

        tilted_ratio = None
        if proxy is not None:
            tilted_indices = np.empty(
                (len(observed), n_global_draws), dtype=np.int64
            )
            tilted_probability = np.empty(
                (len(observed), n_global_draws), dtype=np.float64
            )
            observation_values = proposal._observed_values(observed)
            top_atoms = [None] * len(observed)
            for row in range(len(observed)):
                rng = np.random.default_rng(
                    np.random.SeedSequence(
                        [int(seed) + 1, int(object_start + row)]
                    )
                )
                ids, probability = proxy.draw(
                    observation_values[row],
                    n_draws=n_global_draws,
                    delta=float(tilt_delta),
                    temperature=float(tilt_temperature),
                    rng=rng,
                )
                tilted_indices[row] = ids
                tilted_probability[row] = probability
                if exact_top:
                    top_atoms[row] = proxy.top_atoms(
                        observation_values[row],
                        exact_top,
                        delta=float(tilt_delta),
                        temperature=float(tilt_temperature),
                    )
            tilted_log_c = (
                likelihood.log_importance_weights_tensor(
                    observed,
                    float(center[0]),
                    float(center[1]),
                    atom_indices=tilted_indices,
                    proposal_probability=np.ones_like(
                        tilted_indices, dtype=np.float64
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
            flow_evaluations += int(tilted_indices.size)
            tilted_ratio = np.exp(tilted_log_c) / tilted_probability

        for row in range(len(observed)):
            prefilter_ids = prefilter.indices[row]
            stratum_ids = stratum.indices[row]
            drawn_ids = global_indices[row]

            in_stratum = _membership(lookup, stratum_ids, prefilter_ids)
            c_prefilter = np.exp(prefilter_log_c[row])
            mass_stratum = float(c_prefilter[in_stratum].sum())
            mass_prefilter_complement = float(c_prefilter[~in_stratum].sum())

            drawn_in_stratum = _membership(lookup, stratum_ids, drawn_ids)
            drawn_in_prefilter = _membership(lookup, prefilter_ids, drawn_ids)
            ratio = np.exp(global_log_c[row]) / prior[drawn_ids]

            outside = ~drawn_in_prefilter
            contribution = np.where(outside, ratio, 0.0)
            mass_outside = float(contribution.mean())
            mass_outside_se = float(
                contribution.std(ddof=1) / np.sqrt(len(contribution))
            )

            complement = ~drawn_in_stratum
            between = drawn_in_prefilter & complement

            k_complement, max_complement = tail_of(ratio[complement])
            k_between, max_between = tail_of(ratio[between])
            k_outside, max_outside = tail_of(ratio[outside])

            record = {}
            if tilted_ratio is not None:
                tilted_ids = tilted_indices[row]
                tilted_complement = ~_membership(lookup, stratum_ids, tilted_ids)
                selected = tilted_ratio[row]
                contribution_tilted = np.where(tilted_complement, selected, 0.0)
                k_tilted, max_tilted = tail_of(selected[tilted_complement])
                record.update(
                    {
                        "pareto_k_tilted_complement": k_tilted,
                        "maximum_ratio_tilted_complement": max_tilted,
                        "mass_complement_tilted": float(contribution_tilted.mean()),
                        "mass_complement_tilted_standard_error": float(
                            contribution_tilted.std(ddof=1)
                            / np.sqrt(len(contribution_tilted))
                        ),
                        "draws_tilted_complement": int(tilted_complement.sum()),
                    }
                )
                if exact_top:
                    # What the tilted complement looks like once the atoms the
                    # whole-catalogue score ranks highest are summed exactly
                    # instead of sampled.  Only the stratum changes; the draws,
                    # their probabilities and their exact c are the same ones.
                    extended = np.concatenate([stratum_ids, top_atoms[row]])
                    outside_extended = ~_membership(lookup, extended, tilted_ids)
                    kept = selected[outside_extended]
                    k_extended, max_extended = tail_of(kept)
                    contribution_extended = np.where(
                        outside_extended, selected, 0.0
                    )
                    record.update(
                        {
                            "pareto_k_tilted_exact_top": k_extended,
                            "maximum_ratio_tilted_exact_top": max_extended,
                            "mass_complement_tilted_exact_top": float(
                                contribution_extended.mean()
                            ),
                            "mass_complement_tilted_exact_top_standard_error": float(
                                contribution_extended.std(ddof=1)
                                / np.sqrt(len(contribution_extended))
                            ),
                            "draws_tilted_exact_top": int(outside_extended.sum()),
                            "atoms_exact_top_new": int(
                                np.setdiff1d(top_atoms[row], stratum_ids).size
                            ),
                        }
                    )
                prior_contribution = np.where(complement, ratio, 0.0)
                record["mass_complement_prior"] = float(prior_contribution.mean())
                record["mass_complement_prior_standard_error"] = float(
                    prior_contribution.std(ddof=1) / np.sqrt(len(prior_contribution))
                )

            rows.append(
                {
                    **record,
                    "object_index": int(object_start + row),
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
        "observation_start": start,
        "observation_stop": stop,
        "flow_evaluations": int(flow_evaluations),
        "tilt_delta": None if tilt_delta is None else float(tilt_delta),
        "exact_top": int(exact_top),
        "tilt_temperature": float(tilt_temperature),
    }
    return ComplementDiagnostic(pd.DataFrame(rows), metadata)
