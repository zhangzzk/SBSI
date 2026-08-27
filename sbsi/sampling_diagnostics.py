"""Inspect how an adapted proposal represents per-observation evidence."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

from .catalogue_likelihood import CatalogueLikelihood
from .catalogue_sampling import DefensiveLocalProposal, ProposalCandidates


@dataclass(frozen=True)
class ImportanceSamplingDiagnostic:
    """Draw-level quantities needed to audit an importance proposal."""

    object_ids: np.ndarray
    center: tuple[float, float]
    conditional_log_likelihood: np.ndarray
    log_importance_weight: np.ndarray
    atom_indices: np.ndarray
    proposal_probability: np.ndarray
    local_member: np.ndarray
    global_component: np.ndarray
    population_log_normalization: float
    n_candidates: int
    prefilter_candidates: int | None
    epsilon: float
    proposal_seed: int

    def take(self, rows: Sequence[int]) -> "ImportanceSamplingDiagnostic":
        rows = np.asarray(rows, dtype=np.int64)
        if rows.ndim != 1 or ((rows < 0) | (rows >= len(self.object_ids))).any():
            raise ValueError("diagnostic row selection is invalid")
        return ImportanceSamplingDiagnostic(
            object_ids=self.object_ids[rows],
            center=self.center,
            conditional_log_likelihood=self.conditional_log_likelihood[rows],
            log_importance_weight=self.log_importance_weight[rows],
            atom_indices=self.atom_indices[rows],
            proposal_probability=self.proposal_probability[rows],
            local_member=self.local_member[rows],
            global_component=self.global_component[rows],
            population_log_normalization=self.population_log_normalization,
            n_candidates=self.n_candidates,
            prefilter_candidates=self.prefilter_candidates,
            epsilon=self.epsilon,
            proposal_seed=self.proposal_seed,
        )


@dataclass(frozen=True)
class ExactProposalTargetComparison:
    """Exact finite-catalogue target and its production proposal for one object."""

    object_id: int
    center: tuple[float, float]
    atom_indices: np.ndarray
    conditional_log_likelihood: np.ndarray
    log_target: np.ndarray
    proposal_probability: np.ndarray
    candidate_member: np.ndarray
    candidate_indices: np.ndarray
    candidate_score_gap: np.ndarray
    population_log_normalization: float
    epsilon: float
    n_candidates: int
    prefilter_candidates: int | None

    @property
    def target_probability(self) -> np.ndarray:
        return np.exp(self.log_target - logsumexp(self.log_target))


def evaluate_importance_sampling(
    likelihood: CatalogueLikelihood,
    observed,
    proposal: DefensiveLocalProposal,
    *,
    object_ids: Sequence[int],
    center: Sequence[float],
    n_draws: int,
    n_candidates: int,
    prefilter_candidates: int | None,
    epsilon: float,
    proposal_seed: int,
    candidate_backend: str = "torch",
    object_chunk: int = 128,
    atom_chunk: int = 4096,
) -> ImportanceSamplingDiagnostic:
    """Draw once at the adapted centre and retain raw and weighted evidence.

    ``conditional_log_likelihood`` is the flow-only ``log L`` at each sampled
    atom.  ``log_importance_weight`` is ``log[pi Pdet L / q]``.  Plotting the
    former with equal draw mass and then with normalized importance mass makes
    proposal coverage and evidence concentration directly comparable.
    """

    center = np.asarray(center, dtype=np.float64)
    object_ids = np.asarray(object_ids, dtype=np.int64)
    if center.shape != (2,) or not np.isfinite(center).all():
        raise ValueError("diagnostic center must be a finite two-vector")
    if object_ids.shape != (len(observed),) or (object_ids < 0).any():
        raise ValueError("object_ids must align with observations")
    if n_draws <= 0 or n_candidates <= 0:
        raise ValueError("diagnostic draw and candidate counts must be positive")
    if candidate_backend not in {"scipy", "torch"}:
        raise ValueError("candidate_backend must be scipy or torch")
    if not likelihood.tensor_native_available:
        raise TypeError("sampling diagnostics require a tensor-native Torch flow")

    observed = observed.reset_index(drop=True)
    observed_targets = likelihood.observed_target_tensor(observed)
    candidates = proposal.candidates(
        observed,
        n_candidates=n_candidates,
        prefilter_candidates=prefilter_candidates,
        torch_device=(
            likelihood.flow_model.device if candidate_backend == "torch" else None
        ),
    )
    candidate_target_tensor = likelihood.log_importance_weights_tensor(
        observed,
        float(center[0]),
        float(center[1]),
        atom_indices=candidates.indices,
        proposal_probability=np.ones_like(candidates.indices, dtype=np.float64),
        observed_targets=observed_targets,
        object_chunk=object_chunk,
        atom_chunk=atom_chunk,
    )
    candidate_target = candidate_target_tensor.detach().cpu().numpy().astype(np.float64)
    draw = proposal.draw_adapted(
        candidates,
        candidate_target,
        n_draws=n_draws,
        epsilon=epsilon,
        seed=proposal_seed,
        object_ids=object_ids,
    )
    log_weight = likelihood.log_importance_weights_tensor(
        observed,
        float(center[0]),
        float(center[1]),
        atom_indices=draw.indices,
        proposal_probability=draw.probability,
        observed_targets=observed_targets,
        object_chunk=object_chunk,
        atom_chunk=atom_chunk,
    ).detach().cpu().numpy().astype(np.float64)

    view = likelihood.cache.get(float(center[0]), float(center[1]))
    detected_mass = (
        likelihood.cache.prior.weights[draw.indices]
        * view.detection_probability[draw.indices]
    )
    conditional = np.full_like(log_weight, -np.inf)
    positive = detected_mass > 0
    conditional[positive] = (
        log_weight[positive]
        + np.log(draw.probability[positive])
        - np.log(detected_mass[positive])
    )
    return ImportanceSamplingDiagnostic(
        object_ids=object_ids,
        center=(float(center[0]), float(center[1])),
        conditional_log_likelihood=conditional,
        log_importance_weight=log_weight,
        atom_indices=draw.indices,
        proposal_probability=draw.probability,
        local_member=draw.local_member,
        global_component=draw.global_component,
        population_log_normalization=likelihood.log_population_normalization(
            float(center[0]), float(center[1])
        ),
        n_candidates=int(n_candidates),
        prefilter_candidates=(
            None if prefilter_candidates is None else int(prefilter_candidates)
        ),
        epsilon=float(epsilon),
        proposal_seed=int(proposal_seed),
    )


def evaluate_exact_proposal_target(
    likelihood: CatalogueLikelihood,
    observed,
    proposal: DefensiveLocalProposal,
    *,
    object_id: int,
    center: Sequence[float],
    n_candidates: int,
    prefilter_candidates: int | None,
    epsilon: float,
    candidate_backend: str = "torch",
    atom_chunk: int = 65536,
    candidates: ProposalCandidates | None = None,
) -> ExactProposalTargetComparison:
    """Evaluate every positive-prior atom and reconstruct the production ``q``."""

    center = np.asarray(center, dtype=np.float64)
    if center.shape != (2,) or not np.isfinite(center).all():
        raise ValueError("exact comparison center must be a finite two-vector")
    if len(observed) != 1 or object_id < 0:
        raise ValueError("exact comparison requires one observation and its object id")
    if not 0 < epsilon <= 1:
        raise ValueError("epsilon must lie in (0, 1]")
    if candidate_backend not in {"scipy", "torch"}:
        raise ValueError("candidate_backend must be scipy or torch")
    if not likelihood.tensor_native_available:
        raise TypeError("exact comparison requires a tensor-native Torch flow")

    observed = observed.reset_index(drop=True)
    observed_targets = likelihood.observed_target_tensor(observed)
    if candidates is None:
        candidates = proposal.candidates(
            observed,
            n_candidates=n_candidates,
            prefilter_candidates=prefilter_candidates,
            torch_device=(
                likelihood.flow_model.device if candidate_backend == "torch" else None
            ),
        )
    elif candidates.indices.shape != (1, int(n_candidates)):
        raise ValueError("precomputed candidates do not match the exact comparison")
    candidate_target = likelihood.log_importance_weights_tensor(
        observed,
        float(center[0]),
        float(center[1]),
        atom_indices=candidates.indices,
        proposal_probability=np.ones_like(candidates.indices, dtype=np.float64),
        observed_targets=observed_targets,
        object_chunk=1,
        atom_chunk=atom_chunk,
    )[0].detach().cpu().numpy().astype(np.float64)
    candidate_normalizer = logsumexp(candidate_target)
    if not np.isfinite(candidate_normalizer):
        raise RuntimeError("candidate target has zero finite mass")
    local_probability = np.exp(candidate_target - candidate_normalizer)

    prior_probability = likelihood.cache.prior.weights
    active = np.flatnonzero(prior_probability > 0).astype(np.int64)
    proposal_probability = epsilon * prior_probability[active].astype(
        np.float64, copy=True
    )
    candidate_position = np.searchsorted(active, candidates.indices[0])
    if (
        (candidate_position >= len(active)).any()
        or not np.array_equal(active[candidate_position], candidates.indices[0])
    ):
        raise RuntimeError("candidate support is not contained in active prior atoms")
    proposal_probability[candidate_position] += (1.0 - epsilon) * local_probability
    if not np.isclose(proposal_probability.sum(), 1.0, rtol=0, atol=1e-12):
        raise RuntimeError("reconstructed proposal probability does not sum to one")

    log_target = likelihood.log_importance_weights_tensor(
        observed,
        float(center[0]),
        float(center[1]),
        atom_indices=active,
        proposal_probability=np.ones(len(active), dtype=np.float64),
        observed_targets=observed_targets,
        object_chunk=1,
        atom_chunk=atom_chunk,
    )[0].detach().cpu().numpy().astype(np.float64)
    view = likelihood.cache.get(float(center[0]), float(center[1]))
    detected_mass = (
        prior_probability[active] * view.detection_probability[active]
    )
    conditional = np.full_like(log_target, -np.inf)
    positive = detected_mass > 0
    conditional[positive] = log_target[positive] - np.log(detected_mass[positive])
    candidate_member = np.zeros(len(active), dtype=bool)
    candidate_member[candidate_position] = True
    return ExactProposalTargetComparison(
        object_id=int(object_id),
        center=(float(center[0]), float(center[1])),
        atom_indices=active,
        conditional_log_likelihood=conditional,
        log_target=log_target,
        proposal_probability=proposal_probability,
        candidate_member=candidate_member,
        candidate_indices=candidates.indices[0].copy(),
        candidate_score_gap=candidates.distances[0].copy(),
        population_log_normalization=likelihood.log_population_normalization(
            float(center[0]), float(center[1])
        ),
        epsilon=float(epsilon),
        n_candidates=int(n_candidates),
        prefilter_candidates=(
            None if prefilter_candidates is None else int(prefilter_candidates)
        ),
    )


def defensive_proposal_probabilities(
    target_probability: np.ndarray,
    prior_probability: np.ndarray,
    candidate_positions: np.ndarray,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Construct ``epsilon*pi + (1-epsilon)*p(.|candidate)`` exactly."""

    target = np.asarray(target_probability, dtype=np.float64)
    prior = np.asarray(prior_probability, dtype=np.float64)
    positions = np.asarray(candidate_positions, dtype=np.int64)
    if target.ndim != 1 or prior.shape != target.shape or not len(target):
        raise ValueError("target and prior must be aligned one-dimensional arrays")
    if (target < 0).any() or (prior <= 0).any():
        raise ValueError("target must be non-negative and active prior must be positive")
    if not np.isclose(target.sum(), 1.0) or not np.isclose(prior.sum(), 1.0):
        raise ValueError("target and prior must each sum to one")
    if (
        positions.ndim != 1
        or not len(positions)
        or (positions < 0).any()
        or (positions >= len(target)).any()
        or len(np.unique(positions)) != len(positions)
    ):
        raise ValueError("candidate positions must be unique valid target positions")
    if not 0 < epsilon <= 1:
        raise ValueError("epsilon must lie in (0, 1]")
    candidate_mass = float(target[positions].sum())
    if candidate_mass <= 0:
        raise ValueError("candidate support has zero exact target mass")
    local = target[positions] / candidate_mass
    proposal = epsilon * prior.copy()
    proposal[positions] += (1.0 - epsilon) * local
    if not np.isclose(proposal.sum(), 1.0, rtol=0, atol=2e-12):
        raise RuntimeError("defensive proposal does not sum to one")
    return proposal, local, candidate_mass


def summarize_defensive_proposal(
    target_probability: np.ndarray,
    prior_probability: np.ndarray,
    candidate_positions: np.ndarray,
    epsilon: float,
) -> dict:
    """Return exact mismatch metrics for one candidate support and epsilon."""

    proposal, _, candidate_mass = defensive_proposal_probabilities(
        target_probability,
        prior_probability,
        candidate_positions,
        epsilon,
    )
    target = np.asarray(target_probability, dtype=np.float64)
    positive = target > 0
    ratio = np.zeros_like(target)
    ratio[positive] = target[positive] / proposal[positive]
    second_moment = float(np.sum(target[positive] * ratio[positive]))
    candidate_proposal_mass = float(proposal[candidate_positions].sum())
    return {
        "epsilon": float(epsilon),
        "n_candidates": int(len(candidate_positions)),
        "candidate_target_mass": candidate_mass,
        "outside_candidate_target_mass": float(1.0 - candidate_mass),
        "candidate_proposal_mass": candidate_proposal_mass,
        "outside_candidate_proposal_mass": float(1.0 - candidate_proposal_mass),
        "asymptotic_ess_fraction": float(1.0 / second_moment),
        "chi_square_target_vs_proposal": float(second_moment - 1.0),
        "kl_target_vs_proposal": float(
            np.sum(target[positive] * np.log(ratio[positive]))
        ),
        "total_variation": float(0.5 * np.abs(target - proposal).sum()),
        "maximum_target_to_proposal_ratio": float(ratio.max()),
    }


def optimize_defensive_epsilon(
    target_probability: np.ndarray,
    prior_probability: np.ndarray,
    candidate_positions: np.ndarray,
    *,
    lower: float = 1.0e-3,
) -> tuple[float, float]:
    """Minimize the exact importance-weight second moment over epsilon."""

    target = np.asarray(target_probability, dtype=np.float64)
    prior = np.asarray(prior_probability, dtype=np.float64)
    positions = np.asarray(candidate_positions, dtype=np.int64)
    # Validation and the candidate-conditional target come from the shared
    # constructor.  Its epsilon value is immaterial here.
    _, local, candidate_mass = defensive_proposal_probabilities(
        target, prior, positions, 1.0
    )
    if not 0 < lower < 1:
        raise ValueError("epsilon lower bound must lie in (0, 1)")
    positive = target > 0
    global_second = float(np.sum(np.square(target[positive]) / prior[positive]))
    candidate_global_second = float(
        np.sum(np.square(target[positions]) / prior[positions])
    )
    outside_second = max(0.0, global_second - candidate_global_second)
    target_candidate = target[positions]
    prior_candidate = prior[positions]

    def objective(epsilon: float) -> float:
        proposal_candidate = (
            epsilon * prior_candidate + (1.0 - epsilon) * local
        )
        inside = np.sum(np.square(target_candidate) / proposal_candidate)
        return float(outside_second / epsilon + inside)

    result = minimize_scalar(
        objective,
        bounds=(float(lower), 1.0),
        method="bounded",
        options={"xatol": 1.0e-5},
    )
    candidates = ((float(result.x), float(result.fun)), (1.0, objective(1.0)))
    epsilon, second_moment = min(candidates, key=lambda item: item[1])
    return float(epsilon), float(1.0 / second_moment)


def simulate_defensive_evidence_errors(
    target_probability: np.ndarray,
    prior_probability: np.ndarray,
    candidate_positions: np.ndarray,
    epsilon: float,
    *,
    component_uniform: np.ndarray,
    global_positions: np.ndarray,
    local_uniform: np.ndarray,
    ladder: Sequence[int],
) -> list[dict]:
    """Simulate paired finite-M log-evidence errors from an exact target.

    The returned errors are relative to the exact evidence, so the normalized
    importance ratio ``p/q`` is sufficient and no additional flow calls are
    needed.  Supplying shared uniforms makes comparisons across settings use
    common random numbers.
    """

    proposal, local, _ = defensive_proposal_probabilities(
        target_probability,
        prior_probability,
        candidate_positions,
        epsilon,
    )
    component = np.asarray(component_uniform, dtype=np.float64)
    global_position = np.asarray(global_positions, dtype=np.int64)
    local_u = np.asarray(local_uniform, dtype=np.float64)
    if component.ndim != 2 or global_position.shape != component.shape:
        raise ValueError("component uniforms and global positions must align")
    if local_u.shape != component.shape:
        raise ValueError("local uniforms must align with component uniforms")
    if (
        (component < 0).any()
        or (component >= 1).any()
        or (local_u < 0).any()
        or (local_u >= 1).any()
    ):
        raise ValueError("uniform variates must lie in [0, 1)")
    if (global_position < 0).any() or (global_position >= len(proposal)).any():
        raise ValueError("global draw positions lie outside active prior support")
    sizes = tuple(sorted({int(value) for value in ladder}))
    if not sizes or sizes[0] <= 0 or sizes[-1] > component.shape[1]:
        raise ValueError("evidence ladder lies outside the simulated draws")
    local_cdf = np.cumsum(local)
    local_cdf[-1] = 1.0
    local_position = candidate_positions[
        np.searchsorted(local_cdf, local_u, side="right")
    ]
    selected = np.where(component < epsilon, global_position, local_position)
    ratio = target_probability[selected] / proposal[selected]
    rows = []
    for n_draws in sizes:
        relative = ratio[:, :n_draws].mean(axis=1)
        log_error = np.log(relative)
        absolute = np.abs(log_error)
        rows.append(
            {
                "n_draws": int(n_draws),
                "n_replicates": int(len(relative)),
                "mean_relative_evidence": float(relative.mean()),
                "relative_evidence_std": float(relative.std(ddof=1)),
                "log_evidence_error_percentiles": np.percentile(
                    log_error, [0, 10, 50, 90, 100]
                ).tolist(),
                "absolute_log_evidence_error_percentiles": np.percentile(
                    absolute, [50, 90, 95, 99, 100]
                ).tolist(),
                "fraction_within_0p01": float(np.mean(absolute <= 0.01)),
                "fraction_within_0p05": float(np.mean(absolute <= 0.05)),
            }
        )
    return rows


def normalized_importance_weights(log_weight: np.ndarray) -> np.ndarray:
    """Normalize finite log importance weights without underflow."""

    log_weight = np.asarray(log_weight, dtype=np.float64)
    normalizer = logsumexp(log_weight)
    if not np.isfinite(normalizer):
        return np.zeros_like(log_weight)
    return np.exp(log_weight - normalizer)


def summarize_importance_sampling(
    diagnostic: ImportanceSamplingDiagnostic,
    ladder: Sequence[int],
) -> list[dict]:
    """Return per-object, per-prefix evidence and concentration metrics."""

    ladder = tuple(sorted({int(value) for value in ladder}))
    width = diagnostic.log_importance_weight.shape[1]
    if not ladder or ladder[0] <= 0 or ladder[-1] > width:
        raise ValueError("diagnostic ladder lies outside the retained draws")
    rows = []
    for row, object_id in enumerate(diagnostic.object_ids):
        for n_draws in ladder:
            log_weight = diagnostic.log_importance_weight[row, :n_draws]
            weight = normalized_importance_weights(log_weight)
            positive = weight > 0
            ess = (
                float(1.0 / np.square(weight).sum())
                if positive.any()
                else 0.0
            )
            rows.append(
                {
                    "object_id": int(object_id),
                    "n_draws": int(n_draws),
                    "log_evidence": float(
                        logsumexp(log_weight)
                        - np.log(n_draws)
                        - diagnostic.population_log_normalization
                    ),
                    "ess": ess,
                    "ess_fraction": float(ess / n_draws),
                    "max_weight_fraction": (
                        float(weight.max()) if weight.size else 0.0
                    ),
                    "outside_local_evidence_fraction": float(
                        weight[~diagnostic.local_member[row, :n_draws]].sum()
                    ),
                    "global_draw_evidence_fraction": float(
                        weight[diagnostic.global_component[row, :n_draws]].sum()
                    ),
                    "unique_atoms": int(
                        np.unique(diagnostic.atom_indices[row, :n_draws]).size
                    ),
                }
            )
    return rows


def summarize_exact_proposal_target(
    comparison: ExactProposalTargetComparison,
) -> dict:
    """Summarize exact proposal mismatch and target-mass capture."""

    target = comparison.target_probability
    proposal = comparison.proposal_probability
    if target.shape != proposal.shape or not np.isclose(target.sum(), 1.0):
        raise ValueError("exact target and proposal must be aligned probabilities")
    if (proposal <= 0).any():
        raise ValueError("proposal must have positive support on every target atom")
    ratio = target / proposal
    candidate = comparison.candidate_member
    order = np.argsort(-proposal, kind="stable")
    cumulative_target = np.cumsum(target[order])
    cumulative_proposal = np.cumsum(proposal[order])
    ranks = (4096, 8192, 16384, 32768, 65536, 131072, 1048576)
    rank_capture = {
        str(rank): float(cumulative_target[min(rank, len(order)) - 1])
        for rank in ranks
        if rank <= len(order)
    }
    proposal_mass_for_target = {}
    rank_for_target = {}
    for fraction in (0.5, 0.9, 0.99):
        position = min(
            int(np.searchsorted(cumulative_target, fraction, side="left")),
            len(order) - 1,
        )
        proposal_mass_for_target[str(fraction)] = float(
            cumulative_proposal[position]
        )
        rank_for_target[str(fraction)] = int(position + 1)
    second_moment = float(np.sum(np.square(target) / proposal))
    positive_target = target > 0
    return {
        "object_id": comparison.object_id,
        "n_active_atoms": int(len(target)),
        "center": list(comparison.center),
        "epsilon": comparison.epsilon,
        "n_candidates": comparison.n_candidates,
        "prefilter_candidates": comparison.prefilter_candidates,
        "exact_log_evidence": float(
            logsumexp(comparison.log_target)
            - comparison.population_log_normalization
        ),
        "candidate_target_mass": float(target[candidate].sum()),
        "outside_candidate_target_mass": float(target[~candidate].sum()),
        "candidate_proposal_mass": float(proposal[candidate].sum()),
        "outside_candidate_proposal_mass": float(proposal[~candidate].sum()),
        "asymptotic_ess_fraction": float(1.0 / second_moment),
        "chi_square_target_vs_proposal": float(second_moment - 1.0),
        "kl_target_vs_proposal": float(
            np.sum(target[positive_target] * np.log(ratio[positive_target]))
        ),
        "total_variation": float(0.5 * np.abs(target - proposal).sum()),
        "maximum_target_to_proposal_ratio": float(ratio.max()),
        "target_mass_by_proposal_rank": rank_capture,
        "proposal_mass_needed_for_target_mass": proposal_mass_for_target,
        "proposal_rank_needed_for_target_mass": rank_for_target,
    }


def _weighted_quantile(values: np.ndarray, probability: np.ndarray, quantile: float) -> float:
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(probability[order])
    position = min(int(np.searchsorted(cumulative, quantile, side="left")), len(order) - 1)
    return float(values[order[position]])


def plot_exact_proposal_target(
    comparison: ExactProposalTargetComparison,
    output: str | Path,
    *,
    bins: int = 70,
) -> tuple[Path, Path]:
    """Plot exact proposal mass against exact posterior target mass."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    target = comparison.target_probability
    proposal = comparison.proposal_probability
    log_likelihood = comparison.conditional_log_likelihood
    summary = summarize_exact_proposal_target(comparison)
    finite = np.isfinite(log_likelihood)
    left = min(
        _weighted_quantile(log_likelihood[finite], proposal[finite], 0.01),
        _weighted_quantile(log_likelihood[finite], target[finite], 0.001),
    )
    right = max(
        _weighted_quantile(log_likelihood[finite], proposal[finite], 0.999),
        _weighted_quantile(log_likelihood[finite], target[finite], 0.999),
    )
    shown = np.clip(log_likelihood, left, right)
    edges = np.linspace(left, right, bins + 1)
    order = np.argsort(-proposal, kind="stable")
    cumulative_proposal = np.cumsum(proposal[order])
    cumulative_target = np.cumsum(target[order])
    ratio = target / proposal

    figure, axes = plt.subplots(2, 2, figsize=(8.2, 6.2), constrained_layout=True)
    axes[0, 0].hist(
        shown,
        bins=edges,
        weights=proposal,
        histtype="step",
        linewidth=1.5,
        color="#0072B2",
        label="Proposal q",
    )
    axes[0, 0].hist(
        shown,
        bins=edges,
        weights=target,
        histtype="step",
        linewidth=1.5,
        color="#D55E00",
        label="Exact target p",
    )
    axes[0, 0].set_xlabel(r"Conditional log likelihood  $\log p(x_i\mid z_j,g)$")
    axes[0, 0].set_ylabel("Probability mass per bin")
    axes[0, 0].legend(frameon=False, fontsize=8)
    axes[0, 0].text(
        0.02,
        0.96,
        (
            f"left edge: q={100 * proposal[log_likelihood < left].sum():.2f}%, "
            f"p={100 * target[log_likelihood < left].sum():.2f}%\n"
            f"right edge: q={100 * proposal[log_likelihood > right].sum():.2f}%, "
            f"p={100 * target[log_likelihood > right].sum():.2f}%"
        ),
        transform=axes[0, 0].transAxes,
        ha="left",
        va="top",
        fontsize=7,
    )

    axes[0, 1].plot(
        cumulative_proposal,
        cumulative_target,
        color="#009E73",
        linewidth=1.5,
    )
    axes[0, 1].plot([0, 1], [0, 1], "--", color="0.5", linewidth=1, label="Ideal q=p")
    axes[0, 1].set_xlabel("Cumulative proposal mass")
    axes[0, 1].set_ylabel("Cumulative exact target mass")
    axes[0, 1].legend(frameon=False, fontsize=8)

    ranks = np.arange(1, len(order) + 1)
    axes[1, 0].semilogx(ranks, cumulative_target, color="#CC79A7", linewidth=1.5)
    axes[1, 0].axvline(
        comparison.n_candidates,
        linestyle="--",
        color="0.35",
        linewidth=1,
        label=f"K={comparison.n_candidates:,}",
    )
    axes[1, 0].set_xlabel("Atoms ranked by proposal probability")
    axes[1, 0].set_ylabel("Exact target mass captured")
    axes[1, 0].set_ylim(0, 1.01)
    axes[1, 0].legend(frameon=False, fontsize=8)

    log_ratio = np.log10(ratio)
    ratio_left = _weighted_quantile(log_ratio, target, 0.001)
    ratio_right = _weighted_quantile(log_ratio, target, 0.999)
    ratio_shown = np.clip(log_ratio, ratio_left, ratio_right)
    axes[1, 1].hist(
        ratio_shown,
        bins=70,
        weights=target,
        histtype="stepfilled",
        color="#E69F00",
        alpha=0.7,
    )
    axes[1, 1].axvline(0, linestyle="--", color="0.35", linewidth=1)
    axes[1, 1].set_xlabel(r"Atom mismatch  $\log_{10}[p_i(j)/q_i(j)]$")
    axes[1, 1].set_ylabel("Exact target mass per bin")
    axes[1, 1].text(
        0.98,
        0.96,
        (
            f"candidate target mass={100 * summary['candidate_target_mass']:.2f}%\n"
            f"ESS/M ceiling={100 * summary['asymptotic_ess_fraction']:.3f}%\n"
            f"TV={summary['total_variation']:.3f}"
        ),
        transform=axes[1, 1].transAxes,
        ha="right",
        va="top",
        fontsize=7,
    )
    for axis in axes.flat:
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(labelsize=7)
    figure.suptitle(
        (
            f"Observation {comparison.object_id:,}: production proposal versus "
            f"exact {len(target):,}-atom target"
        ),
        fontsize=10,
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    png = output / f"exact_proposal_target_{comparison.object_id:07d}.png"
    pdf = output / f"exact_proposal_target_{comparison.object_id:07d}.pdf"
    figure.savefig(png, dpi=300, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    return png, pdf


def save_exact_proposal_target(
    comparison: ExactProposalTargetComparison,
    output: str | Path,
    *,
    metadata: Mapping | None = None,
) -> Path:
    """Save exact comparison metrics and compact top-atom diagnostics."""

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    target = comparison.target_probability
    proposal = comparison.proposal_probability
    ratio = target / proposal
    top = np.argsort(-target, kind="stable")[:10000]
    np.savez_compressed(
        output / "top_target_atoms.npz",
        atom_indices=comparison.atom_indices[top],
        conditional_log_likelihood=comparison.conditional_log_likelihood[top],
        target_probability=target[top],
        proposal_probability=proposal[top],
        target_to_proposal_ratio=ratio[top],
        candidate_member=comparison.candidate_member[top],
    )
    payload = summarize_exact_proposal_target(comparison)
    payload["metadata"] = dict(metadata or {})
    result = output / "result.json"
    result.write_text(json.dumps(payload, indent=2) + "\n")
    return result


def select_example_rows(
    diagnostic: ImportanceSamplingDiagnostic,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Select typical, low-ESS, and worst-ESS distinct examples."""

    ess = np.asarray(
        [
            (
                1.0 / denominator
                if (denominator := np.square(normalized_importance_weights(row)).sum())
                > 0
                else 0.0
            )
            for row in diagnostic.log_importance_weight
        ],
        dtype=np.float64,
    )
    minimum = int(np.argmin(ess))
    targets = (
        ("typical ESS", float(np.median(ess))),
        ("low ESS (p10)", float(np.percentile(ess, 10))),
    )
    selected = []
    labels = []
    for label, target in targets:
        order = np.argsort(np.abs(ess - target), kind="stable")
        choice = next(
            (
                int(row)
                for row in order
                if int(row) not in selected and int(row) != minimum
            ),
            None,
        )
        if choice is None:
            break
        selected.append(choice)
        labels.append(label)
    if minimum not in selected:
        selected.append(minimum)
        labels.append("minimum ESS")
    return np.asarray(selected, dtype=np.int64), tuple(labels)


def save_importance_sampling_diagnostic(
    diagnostic: ImportanceSamplingDiagnostic,
    output: str | Path,
    *,
    ladder: Sequence[int],
    labels: Sequence[str],
    pool_summary: Mapping | None = None,
    metadata: Mapping | None = None,
) -> Path:
    """Save draw arrays and a human-readable summary without overwriting."""

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"diagnostic output is not empty: {output}")
    np.savez_compressed(
        output / "samples.npz",
        object_ids=diagnostic.object_ids,
        conditional_log_likelihood=diagnostic.conditional_log_likelihood,
        log_importance_weight=diagnostic.log_importance_weight,
        atom_indices=diagnostic.atom_indices,
        proposal_probability=diagnostic.proposal_probability,
        local_member=diagnostic.local_member,
        global_component=diagnostic.global_component,
    )
    payload = {
        "center": list(diagnostic.center),
        "object_ids": diagnostic.object_ids.tolist(),
        "labels": list(labels),
        "ladder": list(map(int, ladder)),
        "n_candidates": diagnostic.n_candidates,
        "prefilter_candidates": diagnostic.prefilter_candidates,
        "epsilon": diagnostic.epsilon,
        "proposal_seed": diagnostic.proposal_seed,
        "population_log_normalization": diagnostic.population_log_normalization,
        "metrics": summarize_importance_sampling(diagnostic, ladder),
        "pool_summary": dict(pool_summary or {}),
        "metadata": dict(metadata or {}),
    }
    summary = output / "result.json"
    summary.write_text(json.dumps(payload, indent=2) + "\n")
    return summary


def plot_importance_sampling_diagnostic(
    diagnostic: ImportanceSamplingDiagnostic,
    output: str | Path,
    *,
    ladder: Sequence[int],
    labels: Sequence[str],
    bins: int = 55,
    lower_proposal_percentile: float = 5.0,
) -> tuple[Path, ...]:
    """Plot proposal-draw mass beside importance-weighted evidence mass.

    Extreme flow failures can stretch ``log L`` by thousands while carrying
    negligible evidence.  Values below ``lower_proposal_percentile`` are
    collected into the first visible bin, and both the censored proposal mass
    and censored evidence mass are printed on every panel.
    """

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    if not 0 <= lower_proposal_percentile < 50:
        raise ValueError("lower proposal percentile must lie in [0, 50)")
    metrics = summarize_importance_sampling(diagnostic, ladder)
    palette = {"proposal": "#0072B2", "evidence": "#D55E00"}
    paths = []
    for row, (object_id, label) in enumerate(zip(diagnostic.object_ids, labels)):
        values = diagnostic.conditional_log_likelihood[row]
        finite = values[np.isfinite(values)]
        if not finite.size:
            raise RuntimeError(f"object {object_id} has no finite likelihood draws")
        left = float(np.percentile(finite, lower_proposal_percentile))
        right = float(finite.max())
        if left == right:
            left -= 0.5
            right += 0.5
        edges = np.linspace(left, right, bins + 1)
        figure, axes = plt.subplots(
            len(ladder),
            2,
            figsize=(8.0, 2.25 * len(ladder)),
            sharex=True,
            constrained_layout=True,
        )
        axes = np.atleast_2d(axes)
        for rung_row, n_draws in enumerate(ladder):
            x = values[:n_draws]
            shown_x = np.maximum(x, left)
            weight = normalized_importance_weights(
                diagnostic.log_importance_weight[row, :n_draws]
            )
            before = np.full(n_draws, 1.0 / n_draws)
            below = x < left
            axes[rung_row, 0].hist(
                shown_x,
                bins=edges,
                weights=before,
                histtype="stepfilled",
                color=palette["proposal"],
                alpha=0.65,
            )
            axes[rung_row, 1].hist(
                shown_x,
                bins=edges,
                weights=weight,
                histtype="stepfilled",
                color=palette["evidence"],
                alpha=0.65,
            )
            entry = next(
                item
                for item in metrics
                if item["object_id"] == int(object_id)
                and item["n_draws"] == int(n_draws)
            )
            axes[rung_row, 0].set_ylabel(f"M={n_draws:,}\nprobability/bin")
            axes[rung_row, 1].text(
                0.98,
                0.94,
                (
                    f"log evidence={entry['log_evidence']:.3f}\n"
                    f"ESS={entry['ess']:.0f} "
                    f"({100 * entry['ess_fraction']:.3f}% of M)\n"
                    f"max weight={100 * entry['max_weight_fraction']:.1f}%\n"
                    "outside local="
                    f"{100 * entry['outside_local_evidence_fraction']:.1f}%"
                ),
                transform=axes[rung_row, 1].transAxes,
                ha="right",
                va="top",
                fontsize=7,
            )
            axes[rung_row, 0].text(
                0.02,
                0.94,
                f"{100 * before[below].sum():.1f}% draws at left edge",
                transform=axes[rung_row, 0].transAxes,
                ha="left",
                va="top",
                fontsize=7,
            )
            axes[rung_row, 1].text(
                0.02,
                0.94,
                f"{100 * weight[below].sum():.2f}% evidence at left edge",
                transform=axes[rung_row, 1].transAxes,
                ha="left",
                va="top",
                fontsize=7,
            )
            for axis in axes[rung_row]:
                axis.spines[["top", "right"]].set_visible(False)
                axis.tick_params(labelsize=7)
        axes[0, 0].set_title("Proposal draws (equal mass)", fontsize=9)
        axes[0, 1].set_title("After importance weighting (evidence mass)", fontsize=9)
        for axis in axes[-1]:
            axis.set_xlabel(r"Conditional log likelihood  $\log p(x_i\mid z_j,g)$")
        figure.suptitle(
            (
                f"Observation {int(object_id):,} — {label}; "
                f"center=({diagnostic.center[0]:.5f}, {diagnostic.center[1]:.5f})"
            ),
            fontsize=10,
        )
        stem = output / f"sampling_observation_{int(object_id):07d}"
        png = stem.with_suffix(".png")
        pdf = stem.with_suffix(".pdf")
        figure.savefig(png, dpi=300, bbox_inches="tight")
        figure.savefig(pdf, bbox_inches="tight")
        plt.close(figure)
        paths.extend((png, pdf))
    return tuple(paths)


__all__ = [
    "ExactProposalTargetComparison",
    "ImportanceSamplingDiagnostic",
    "defensive_proposal_probabilities",
    "evaluate_exact_proposal_target",
    "evaluate_importance_sampling",
    "normalized_importance_weights",
    "optimize_defensive_epsilon",
    "plot_importance_sampling_diagnostic",
    "plot_exact_proposal_target",
    "save_exact_proposal_target",
    "save_importance_sampling_diagnostic",
    "select_example_rows",
    "simulate_defensive_evidence_errors",
    "summarize_defensive_proposal",
    "summarize_importance_sampling",
    "summarize_exact_proposal_target",
]
