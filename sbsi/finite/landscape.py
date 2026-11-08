"""Exact whole-catalogue integrand for a handful of observations.

Every convergence diagnostic in this repository measures the *sampler*.  The
Pareto tail index, the effective sample size and the maximum weight fraction
are all properties of ``c_j / q_j``, so a heavy tail can mean either an awkward
target or a badly matched proposal, and no amount of staring at those numbers
separates the two.  cont.331 needs that separation: the failing objects are
overwhelmingly the bright ones (Spearman ``rho = -0.726`` between the tail
index and measured magnitude), and "bright galaxies are hard to sample" and
"bright galaxies have a target that no sampler can integrate well" call for
opposite responses.

This module removes the proposal from the question.  For one observation it
evaluates the exact unnormalised target ``c_j = pi_j Pdet_j L_j`` on *every*
active atom -- no draws, no candidate stratum, no prefilter -- and reports how
that mass is distributed across atoms:

  * ``n_eff``, the effective number of contributing atoms, ``1 / sum_j p_j^2``
    with ``p = c / sum c``.  This is the reciprocal participation ratio, and it
    is the honest answer to "how many of the 12.76 million atoms actually
    matter for this galaxy".
  * ``atoms_for_50/90/99_percent``, the rank at which the sorted mass reaches
    each share.  ``n_eff`` compresses the whole distribution into one number;
    these say whether the mass is a point, a plateau, or a long ramp.
  * the sorted head of the distribution itself, so the fall-off can be plotted
    rather than summarised.

The two outcomes look nothing alike.  A bright galaxy whose mass sits on three
atoms out of 12.76 million is not a sampling failure at all -- it says the
finite scene prior is too coarse to describe that galaxy, and the fix belongs
to the catalogue, not the estimator.  Mass spread over thousands of atoms in a
ragged, multi-lobed shape *is* a proposal problem and justifies more sampler
work.

Cost is one flow evaluation per atom per observation, so this is affordable for
a dozen observations (12.76 million each, about 2% of one 25,000-object
inference run) and for no more than that.  It is a measurement, not a release:
nothing here feeds an estimate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch


@dataclass
class LikelihoodLandscape:
    """Per-observation mass distribution over atoms, plus the run's identity."""

    table: pd.DataFrame
    head_rank: np.ndarray
    head_share: np.ndarray
    head_atom: np.ndarray
    head_coordinates: np.ndarray
    observed: pd.DataFrame
    metadata: dict

    def summary(self) -> dict:
        frame = self.table

        def percentiles(values: pd.Series) -> list:
            usable = np.asarray(values, dtype=np.float64)
            usable = usable[np.isfinite(usable)]
            if usable.size == 0:
                return []
            return [float(v) for v in np.percentile(usable, [0, 10, 50, 90, 100])]

        return {
            "n_observations": int(len(frame)),
            "n_atoms": int(self.metadata["n_atoms"]),
            "n_eff_percentiles": percentiles(frame["n_eff"]),
            "top1_share_percentiles": percentiles(frame["top1_share"]),
            "atoms_for_50_percent_percentiles": percentiles(
                frame["atoms_for_50_percent"]
            ),
            "atoms_for_90_percent_percentiles": percentiles(
                frame["atoms_for_90_percent"]
            ),
            "atoms_for_99_percent_percentiles": percentiles(
                frame["atoms_for_99_percent"]
            ),
        }


def _mass_statistics(share: np.ndarray) -> dict:
    """Reduce one normalised, descending mass profile to scalars."""

    cumulative = np.cumsum(share)
    # `searchsorted` on the cumulative profile gives the smallest rank whose
    # running total reaches the threshold; +1 turns a zero-based position into
    # a count of atoms.
    def rank_for(threshold: float) -> int:
        return int(np.searchsorted(cumulative, threshold, side="left")) + 1

    return {
        "n_eff": float(1.0 / np.square(share).sum()),
        "top1_share": float(share[0]),
        "top10_share": float(cumulative[min(9, cumulative.size - 1)]),
        "top100_share": float(cumulative[min(99, cumulative.size - 1)]),
        "atoms_for_50_percent": rank_for(0.5),
        "atoms_for_90_percent": rank_for(0.9),
        "atoms_for_99_percent": rank_for(0.99),
    }


def likelihood_landscape(
    likelihood,
    mock,
    proposal,
    *,
    center: Sequence[float],
    rows: Sequence[int],
    observation_start: int = 0,
    head: int = 4096,
    atom_chunk: int = 1 << 20,
    coordinate_columns: Optional[Sequence[str]] = None,
    shortlist_sizes: Sequence[int] = (),
    prefilter_candidates: Optional[int] = None,
    candidate_backend: str = "torch",
) -> LikelihoodLandscape:
    """Score every active atom against a few observations, exactly.

    ``rows`` are positions within ``mock`` -- the already-sliced observation
    window -- and ``observation_start`` only labels them with their absolute
    identity in the output so a landscape can be matched back to the estimator's
    per-object diagnostics.
    """

    selected = np.asarray(rows, dtype=np.int64)
    if selected.ndim != 1 or selected.size == 0:
        raise ValueError("rows must be a non-empty one-dimensional sequence")
    if (selected < 0).any() or (selected >= len(mock.measurements)).any():
        raise ValueError("rows must index the observation window")
    if head <= 0:
        raise ValueError("head must be positive")
    if atom_chunk <= 0:
        raise ValueError("atom_chunk must be positive")

    active = np.ascontiguousarray(proposal.active_indices, dtype=np.int64)
    n_atoms = int(active.size)
    if n_atoms == 0:
        raise ValueError("the proposal has no active atoms")
    kept = int(min(head, n_atoms))

    values = proposal.coordinates.values
    if coordinate_columns is None:
        coordinate_columns = [f"coordinate_{i}" for i in range(values.shape[1])]
    if len(coordinate_columns) != values.shape[1]:
        raise ValueError("coordinate_columns must name every proposal coordinate")

    shortlist_sizes = tuple(int(v) for v in shortlist_sizes)
    if any(v <= 0 for v in shortlist_sizes):
        raise ValueError("shortlist sizes must be positive")
    if shortlist_sizes and prefilter_candidates is None:
        prefilter_candidates = max(shortlist_sizes)
    if prefilter_candidates is not None and prefilter_candidates < max(
        shortlist_sizes, default=0
    ):
        raise ValueError("prefilter cannot be smaller than the largest shortlist")

    # Only the shortlist comparison queries the proposal, so a landscape asked
    # for without one must not reach into the flow model for a device it will
    # never use.
    torch_device = (
        likelihood.flow_model.device
        if shortlist_sizes and candidate_backend == "torch"
        else None
    )
    measurements = mock.measurements
    records = []
    head_share = np.empty((selected.size, kept), dtype=np.float64)
    head_atom = np.empty((selected.size, kept), dtype=np.int64)
    head_coordinates = np.empty(
        (selected.size, kept, values.shape[1]), dtype=np.float64
    )
    # One atom index row is reused for every observation: it is the whole
    # active set every time, and at 12.76 million int64 entries rebuilding it
    # per observation would cost more than the arithmetic it feeds.
    atom_row = active[None, :]
    ones = np.ones_like(atom_row, dtype=np.float64)

    for position, row in enumerate(selected):
        observed = measurements.iloc[int(row) : int(row) + 1].reset_index(drop=True)
        log_c = (
            likelihood.log_importance_weights_tensor(
                observed,
                float(center[0]),
                float(center[1]),
                atom_indices=atom_row,
                proposal_probability=ones,
                observed_targets=likelihood.observed_target_tensor(observed),
                object_chunk=1,
                atom_chunk=atom_chunk,
            )
            .detach()
            .to(dtype=torch.float64)
            .reshape(-1)
        )
        # Subtracting the maximum before exponentiating is what makes this
        # representable: `log_c` spans hundreds of nats across 12.76 million
        # atoms and `exp` of the raw values underflows to zero everywhere but
        # the peak.  The shift cancels in the normalised share.
        finite = torch.isfinite(log_c)
        if not bool(finite.any()):
            raise ValueError(f"observation {int(row)} has no finite atom weight")
        peak = log_c[finite].max()
        weight = torch.where(finite, torch.exp(log_c - peak), torch.zeros((), dtype=torch.float64, device=log_c.device))
        total = weight.sum()
        share = (weight / total).cpu().numpy()

        order = np.argsort(share)[::-1]
        top = order[:kept]
        head_share[position] = share[top]
        head_atom[position] = active[top]
        head_coordinates[position] = values[active[top]]

        record = {
            "observation": int(observation_start + int(row)),
            "window_row": int(row),
            "log_total_mass": float((torch.log(total) + peak).cpu()),
            "n_atoms_positive": int((share > 0).sum()),
        }
        record.update(_mass_statistics(np.sort(share)[::-1]))
        # The estimator sums a proxy-ranked shortlist exactly and samples the
        # rest.  Whether that shortlist contains the atoms this object's mass
        # actually sits on is the difference between an essentially exact
        # answer and a heavy-tailed correction, and only the exact profile
        # computed above can say.
        for size in shortlist_sizes:
            candidates = proposal.candidates(
                observed,
                n_candidates=int(size),
                prefilter_candidates=int(prefilter_candidates),
                torch_device=torch_device,
            )
            listed = np.asarray(candidates.indices[0], dtype=np.int64)
            # `share` is indexed by position in the active set, not by absolute
            # atom id, so the shortlist has to be mapped through the same order
            # before its mass can be read off.
            position = np.searchsorted(active, listed)
            position = position[
                (position < active.size) & (active[np.minimum(position, active.size - 1)] == listed)
            ]
            record[f"shortlist_{size}_mass"] = float(share[position].sum())
            record[f"shortlist_{size}_holds_top1"] = bool(
                np.isin(int(np.argmax(share)), position)
            )
        records.append(record)

    table = pd.DataFrame.from_records(records)
    observed = measurements.iloc[selected].reset_index(drop=True)
    metadata = {
        "center": [float(center[0]), float(center[1])],
        "n_atoms": n_atoms,
        "head": kept,
        "atom_chunk": int(atom_chunk),
        "observation_start": int(observation_start),
        "rows": [int(v) for v in selected],
        "coordinate_columns": list(coordinate_columns),
        # `coordinates.values` is already in measured units -- the flow-predicted
        # location of each atom -- so `head_coordinates` needs no transform to
        # plot against an observation.  The centre and scale are the proposal's
        # own standardization and are carried only so a reader can reproduce the
        # proxy score without reopening the proposal cache; applying them to the
        # coordinates is a mistake and put atoms at magnitude 45 once.
        "coordinate_center": [float(v) for v in np.asarray(proposal.coordinates.center)],
        "coordinate_scale": [float(v) for v in np.asarray(proposal.coordinates.scale)],
        "shortlist_sizes": list(shortlist_sizes),
        "prefilter_candidates": (
            None if prefilter_candidates is None else int(prefilter_candidates)
        ),
        "flow_evaluations": int(n_atoms) * int(selected.size),
    }
    return LikelihoodLandscape(
        table=table,
        head_rank=np.arange(1, kept + 1, dtype=np.int64),
        head_share=head_share,
        head_atom=head_atom,
        head_coordinates=head_coordinates,
        observed=observed,
        metadata=metadata,
    )
