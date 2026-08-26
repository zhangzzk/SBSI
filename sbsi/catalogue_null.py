"""Streamed Section 5 catalogue-prior null estimator and go/no-go gates.

This module evaluates the local finite-difference expansion of the *marginal*
detected-population log likelihood at zero shear.  Importance atoms are fixed
across every stencil view and draw ladders are exact prefixes.  Only one object
batch owns an ``N_batch x M`` tensor at a time.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import time
from typing import Callable, Mapping, Optional, Sequence

import numpy as np
from scipy.special import logsumexp
import torch

from .catalogue_closure import MockCatalogue
from .catalogue_likelihood import CatalogueLikelihood
from .catalogue_sampling import (
    CoalescedProposalDraw,
    DefensiveLocalProposal,
    ProposalDraw,
    select_adaptive_draw_counts,
    select_independent_pilot_draw_counts,
)


@dataclass(frozen=True)
class TailDiagnostics:
    max_abs_score: float
    p99_abs_score: float
    p999_abs_score: float
    excess_kurtosis: float
    hill_tail_index: float
    hill_k: int


@dataclass(frozen=True)
class StreamedImportanceDiagnostics:
    mean_ess: float
    median_ess: float
    p10_ess: float
    mean_ess_fraction: float
    p90_max_weight_fraction: float
    mean_outside_local_contribution: float
    mean_global_draw_contribution: float


@dataclass(frozen=True)
class Section5Estimate:
    component: str
    h: float
    n_draws: int
    n_objects: int
    estimated_shear: float
    robust_standard_error: float
    model_standard_error: float
    score_mean: float
    score_mean_standard_error: float
    score_z: float
    score_variance: float
    information_mean: float
    information_identity_ratio: float
    information_identity_standard_error: float
    tail: TailDiagnostics
    importance: Optional[StreamedImportanceDiagnostics]


@dataclass(frozen=True)
class Section5SeedResult:
    proposal_seed: Optional[int]
    steps: tuple[float, ...]
    ladder: tuple[int, ...]
    n_candidates: Optional[int]
    n_objects: int
    n_views: int
    flow_evaluations: int
    elapsed_seconds: float
    estimates: tuple[Section5Estimate, ...]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Section5StreamResult:
    results: tuple[Section5SeedResult, ...]
    independent_banks: Mapping[str, Section5SeedResult]
    object_moments: Mapping[int, "Section5ObjectMoments"] = field(default_factory=dict)


@dataclass(frozen=True)
class Section5ObjectMoments:
    """Retained per-object derivatives for paired closure diagnostics.

    Production inference normally needs only the streamed sums.  Nonzero-
    shear closure is different: common-random-number ``+g`` and ``-g`` mocks
    must retain their aligned influence functions to measure a precise paired
    response.  The keys are ``(component, h, n_draws)``.
    """

    proposal_seed: int
    score: Mapping[tuple[str, float, int], np.ndarray]
    information: Mapping[tuple[str, float, int], np.ndarray]


@dataclass(frozen=True)
class PairedSection5Estimate:
    """Response of one estimated component to a paired shear injection."""

    injected_component: str
    estimated_component: str
    amplitude: float
    h: float
    n_draws: int
    n_objects: int
    n_blocks: int
    positive_estimated_shear: float
    negative_estimated_shear: float
    response: float
    expected_response: float
    response_bias: float
    robust_standard_error: float
    response_pull: float
    symmetric_offset: float
    symmetric_offset_standard_error: float
    score_correlation: float
    influence_correlation: float
    response_tail: TailDiagnostics


@dataclass(frozen=True)
class Section5GateAssessment:
    passed: bool
    checks: Mapping[str, bool]
    metrics: Mapping[str, float]
    failures: tuple[str, ...]


@dataclass(frozen=True)
class Section5AutogradResult:
    """Exact derivatives of the finite-catalogue marginal likelihood.

    ``score`` has shape ``(N, 2)`` and ``information`` has shape
    ``(N, 2, 2)``.  This diagnostic is intentionally restricted to the
    shape-only, no-cut, no-external-response model used by the null closure.
    """

    score: np.ndarray
    information: np.ndarray
    atom_indices: np.ndarray
    elapsed_seconds: float

    def __post_init__(self):
        score = np.asarray(self.score, dtype=np.float64)
        information = np.asarray(self.information, dtype=np.float64)
        atoms = np.asarray(self.atom_indices, dtype=np.int64)
        if score.ndim != 2 or score.shape[1] != 2:
            raise ValueError("autograd score must have shape (N, 2)")
        if information.shape != (len(score), 2, 2):
            raise ValueError("autograd information must have shape (N, 2, 2)")
        if atoms.ndim != 1 or not len(atoms):
            raise ValueError("autograd atom indices must be a non-empty vector")
        if not np.isfinite(score).all() or not np.isfinite(information).all():
            raise ValueError("autograd derivatives contain non-finite values")
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "information", information)
        object.__setattr__(self, "atom_indices", atoms)


@dataclass(frozen=True)
class AdaptiveSection5Result:
    """Per-object local moments from a zero-view adaptive draw ladder."""

    center: tuple[float, float]
    h: float
    draw_ladder: tuple[int, ...]
    n_candidates: int
    score: np.ndarray
    information: np.ndarray
    draw_counts: np.ndarray
    unique_counts: np.ndarray
    flow_evaluations: int
    elapsed_seconds: float
    ladder_score: Optional[np.ndarray] = field(default=None, repr=False)
    ladder_information: Optional[np.ndarray] = field(default=None, repr=False)
    proposal_prefilter_candidates: Optional[int] = None
    allocation_method: str = "production_prefix"
    pilot_draws: Optional[int] = None
    pilot_seed: Optional[int] = None
    pilot_ess_fraction: Optional[np.ndarray] = field(default=None, repr=False)
    pilot_max_weight_fraction: Optional[np.ndarray] = field(default=None, repr=False)
    phase_seconds: Mapping[str, float] = field(default_factory=dict)
    bias_correction: str = "none"
    candidate_backend: str = "scipy"

    def __post_init__(self):
        score = np.asarray(self.score, dtype=np.float64)
        information = np.asarray(self.information, dtype=np.float64)
        counts = np.asarray(self.draw_counts, dtype=np.int64)
        unique = np.asarray(self.unique_counts, dtype=np.int64)
        if score.ndim != 2 or score.shape[1] != 2:
            raise ValueError("adaptive score must have shape (N, 2)")
        if information.shape not in (score.shape, (len(score), 2, 2)):
            raise ValueError(
                "adaptive information must be diagonal (N, 2) or full (N, 2, 2)"
            )
        if counts.shape != (len(score),) or unique.shape != counts.shape:
            raise ValueError("adaptive draw counts must contain one value per object")
        if not np.isfinite(score).all() or not np.isfinite(information).all():
            raise ValueError("adaptive moments contain non-finite values")
        if (counts <= 0).any() or (unique <= 0).any() or (unique > counts).any():
            raise ValueError("adaptive draw/unique counts are invalid")
        if (
            self.proposal_prefilter_candidates is not None
            and self.proposal_prefilter_candidates < self.n_candidates
        ):
            raise ValueError("proposal prefilter cannot be smaller than final support")
        if self.allocation_method not in ("production_prefix", "independent_pilot"):
            raise ValueError("unknown adaptive allocation method")
        if self.bias_correction not in ("none", "richardson_1_over_m"):
            raise ValueError("unknown adaptive finite-draw bias correction")
        if self.candidate_backend not in ("scipy", "torch"):
            raise ValueError("unknown candidate-query backend")
        pilot_ess = self.pilot_ess_fraction
        pilot_maximum = self.pilot_max_weight_fraction
        if self.allocation_method == "independent_pilot":
            if self.pilot_draws is None or self.pilot_draws <= 0:
                raise ValueError("independent-pilot allocation requires pilot draws")
            if self.pilot_seed is None:
                raise ValueError("independent-pilot allocation requires a pilot seed")
            if pilot_ess is None or pilot_maximum is None:
                raise ValueError("independent-pilot diagnostics are missing")
            pilot_ess = np.asarray(pilot_ess, dtype=np.float64)
            pilot_maximum = np.asarray(pilot_maximum, dtype=np.float64)
            if pilot_ess.shape != counts.shape or pilot_maximum.shape != counts.shape:
                raise ValueError("pilot diagnostics must contain one value per object")
            if (
                not np.isfinite(pilot_ess).all()
                or not np.isfinite(pilot_maximum).all()
                or (pilot_ess < 0).any()
                or (pilot_ess > 1).any()
                or (pilot_maximum < 0).any()
                or (pilot_maximum > 1).any()
            ):
                raise ValueError("pilot diagnostics are invalid")
            object.__setattr__(self, "pilot_ess_fraction", pilot_ess)
            object.__setattr__(self, "pilot_max_weight_fraction", pilot_maximum)
        elif pilot_ess is not None or pilot_maximum is not None:
            raise ValueError("production-prefix allocation cannot carry pilot diagnostics")
        phases = {str(name): float(value) for name, value in self.phase_seconds.items()}
        if any(not np.isfinite(value) or value < 0 for value in phases.values()):
            raise ValueError("adaptive phase timings must be finite and non-negative")
        ladder_score = self.ladder_score
        ladder_information = self.ladder_information
        if (ladder_score is None) != (ladder_information is None):
            raise ValueError("full-ladder score and information must be retained together")
        if ladder_score is not None:
            ladder_score = np.asarray(ladder_score, dtype=np.float64)
            ladder_information = np.asarray(ladder_information, dtype=np.float64)
            expected_score = (len(self.draw_ladder), len(score), 2)
            if ladder_score.shape != expected_score:
                raise ValueError(
                    f"full-ladder score must have shape {expected_score}"
                )
            expected_information = (
                (len(self.draw_ladder), len(score), 2, 2)
                if information.ndim == 3
                else expected_score
            )
            if ladder_information.shape != expected_information:
                raise ValueError(
                    "full-ladder information has the wrong shape: "
                    f"expected {expected_information}"
                )
            if not np.isfinite(ladder_score).all() or not np.isfinite(
                ladder_information
            ).all():
                raise ValueError("full-ladder moments contain non-finite values")
            object.__setattr__(self, "ladder_score", ladder_score)
            object.__setattr__(self, "ladder_information", ladder_information)
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "information", information)
        object.__setattr__(self, "draw_counts", counts)
        object.__setattr__(self, "unique_counts", unique)
        object.__setattr__(self, "phase_seconds", phases)


@dataclass(frozen=True)
class AdaptiveOneStepResult:
    """One full-2D likelihood Newton step from a cheap external centre."""

    center: tuple[float, float]
    estimate: tuple[float, float]
    step: tuple[float, float]
    information_eigenvalues: tuple[float, float]
    robust_covariance: tuple[tuple[float, float], tuple[float, float]]
    model_covariance: tuple[tuple[float, float], tuple[float, float]]
    robust_standard_error: tuple[float, float]
    model_standard_error: tuple[float, float]
    quadratic_log_likelihood_gain: float
    moments: AdaptiveSection5Result = field(repr=False)

    def to_dict(self) -> dict:
        return {
            "center": list(self.center),
            "estimate": list(self.estimate),
            "step": list(self.step),
            "information_eigenvalues": list(self.information_eigenvalues),
            "robust_covariance": [list(row) for row in self.robust_covariance],
            "model_covariance": [list(row) for row in self.model_covariance],
            "robust_standard_error": list(self.robust_standard_error),
            "model_standard_error": list(self.model_standard_error),
            "quadratic_log_likelihood_gain": self.quadratic_log_likelihood_gain,
            "n_objects": int(len(self.moments.score)),
            "h": self.moments.h,
            "draw_ladder": list(self.moments.draw_ladder),
            "n_candidates": self.moments.n_candidates,
            "proposal_prefilter_candidates": (
                self.moments.proposal_prefilter_candidates
            ),
            "draw_count_percentiles": np.percentile(
                self.moments.draw_counts, [0, 25, 50, 75, 90, 100]
            ).tolist(),
            "mean_unique_fraction": float(
                np.mean(self.moments.unique_counts / self.moments.draw_counts)
            ),
            "flow_evaluations": self.moments.flow_evaluations,
            "elapsed_seconds": self.moments.elapsed_seconds,
            "phase_seconds": dict(self.moments.phase_seconds),
            "retained_full_ladder": self.moments.ladder_score is not None,
            "allocation_method": self.moments.allocation_method,
            "bias_correction": self.moments.bias_correction,
            "candidate_backend": self.moments.candidate_backend,
            "pilot_draws": self.moments.pilot_draws,
            "pilot_seed": self.moments.pilot_seed,
            "pilot_ess_fraction_percentiles": (
                None
                if self.moments.pilot_ess_fraction is None
                else np.percentile(
                    self.moments.pilot_ess_fraction, [0, 10, 50, 90, 100]
                ).tolist()
            ),
            "pilot_max_weight_fraction_percentiles": (
                None
                if self.moments.pilot_max_weight_fraction is None
                else np.percentile(
                    self.moments.pilot_max_weight_fraction, [0, 10, 50, 90, 100]
                ).tolist()
            ),
        }


@dataclass(frozen=True)
class ImportanceAutogradResult:
    """Batched two-component gradient/Hessian of the sampled evidence."""

    center: tuple[float, float]
    score: np.ndarray
    information: np.ndarray
    draw_counts: np.ndarray
    unique_counts: np.ndarray
    flow_evaluations: int
    elapsed_seconds: float

    def __post_init__(self):
        score = np.asarray(self.score, dtype=np.float64)
        information = np.asarray(self.information, dtype=np.float64)
        if score.ndim != 2 or score.shape[1] != 2:
            raise ValueError("autograd importance score must have shape (N, 2)")
        if information.shape != (len(score), 2, 2):
            raise ValueError("autograd importance information must have shape (N, 2, 2)")
        if not np.isfinite(score).all() or not np.isfinite(information).all():
            raise ValueError("autograd importance moments contain non-finite values")
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "information", information)


@dataclass(frozen=True)
class AutogradShearOptimizationResult:
    """Newton recentering history using full sampled 2x2 information."""

    estimate: tuple[float, float]
    converged: bool
    iterations: tuple[ImportanceAutogradResult, ...]


@dataclass(frozen=True)
class NumericalLikelihoodPoint:
    """One fixed-draw population log-likelihood evaluation."""

    shear: tuple[float, float]
    log_likelihood_sum: float


@dataclass(frozen=True)
class NumericalShearIteration:
    """One safeguarded local numerical expansion and accepted update."""

    center: tuple[float, float]
    log_likelihood_sum: float
    score: tuple[float, float]
    information: tuple[tuple[float, float], tuple[float, float]]
    information_eigenvalues: tuple[float, float]
    step_method: str
    raw_step: tuple[float, float]
    accepted_step: tuple[float, float]
    next_center: tuple[float, float]
    next_log_likelihood_sum: float
    accepted: bool


@dataclass(frozen=True)
class NumericalShearOptimizationResult:
    """Fixed-draw, safeguarded two-component catalogue-likelihood maximum."""

    estimate: tuple[float, float]
    converged: bool
    reason: str
    h: float
    n_objects: int
    n_draws: int
    n_candidates: int
    proposal_method: str
    proposal_reference_shear: Optional[tuple[float, float]]
    proposal_seed: int
    epsilon: float
    bandwidth: Optional[float]
    initial_likelihood_reused: bool
    proposal_candidate_flow_evaluations: int
    proposal_reuse_flow_evaluations: int
    proposal_flow_evaluations: int
    numerator_flow_evaluations: int
    selection_flow_evaluations: int
    diagnostic_flow_evaluations: int
    flow_evaluations: int
    elapsed_seconds: float
    importance: StreamedImportanceDiagnostics
    iterations: tuple[NumericalShearIteration, ...]
    evaluations: tuple[NumericalLikelihoodPoint, ...]

    def to_dict(self) -> dict:
        return asdict(self)


def _torch_shear_ellipticity(e1, e2, g1, g2):
    """Differentiable real form of the exact reduced-shear Mobius map."""

    denominator_real = 1.0 + g1 * e1 + g2 * e2
    denominator_imag = g1 * e2 - g2 * e1
    numerator_real = e1 + g1
    numerator_imag = e2 + g2
    denominator = denominator_real.square() + denominator_imag.square()
    out1 = (
        numerator_real * denominator_real
        + numerator_imag * denominator_imag
    ) / denominator
    out2 = (
        numerator_imag * denominator_real
        - numerator_real * denominator_imag
    ) / denominator
    return out1, out2


def autograd_exact_section5(
    likelihood: CatalogueLikelihood,
    mock: MockCatalogue,
    *,
    atom_indices: Optional[Sequence[int]] = None,
    max_objects: Optional[int] = None,
    center: Sequence[float] = (0.0, 0.0),
) -> Section5AutogradResult:
    """Differentiate the exact finite-catalogue evidence with Torch autograd.

    This is a derivative cross-check, not a production implementation.  It
    processes one object at a time so second-order graphs never span an object
    batch.  Detection is allowed only when its declared inputs are spin-0;
    its detected-population normalizer is then exactly shear invariant under
    the project's shape-only shear convention.
    """

    if likelihood.selection is not None:
        raise ValueError("autograd Section 5 diagnostic requires measured cuts disabled")
    if likelihood.cache.blend_response is not None:
        raise ValueError("autograd Section 5 diagnostic requires external R_blend=0")
    center = np.asarray(center, dtype=np.float64)
    if center.shape != (2,) or not np.isfinite(center).all():
        raise ValueError("center must be a finite two-vector")
    likelihood.cache.validate_detection_shear_invariance()
    bundle = likelihood.flow_model
    required = ("model", "condition_preprocessor", "target_transform", "device")
    if any(not hasattr(bundle, name) for name in required):
        raise TypeError("autograd Section 5 requires a Torch measurement-model bundle")

    features = tuple(bundle.condition_preprocessor.feature_names)
    if "e1_input_p" not in features or "e2_input_p" not in features:
        raise ValueError("flow conditions lack the primary intrinsic ellipticity")
    first = features.index("e1_input_p")
    second = features.index("e2_input_p")
    active = np.flatnonzero(likelihood.cache.prior.weights > 0)
    atoms = active if atom_indices is None else np.asarray(atom_indices, dtype=np.int64)
    if atoms.ndim != 1 or not len(atoms):
        raise ValueError("atom_indices must be a non-empty vector")
    if ((atoms < 0) | (atoms >= len(likelihood.cache.prior.weights))).any():
        raise ValueError("atom_indices contain an out-of-range row")
    mass = (
        likelihood.cache.prior.weights[atoms]
        * likelihood.cache.get(0.0, 0.0).detection_probability[atoms]
    )
    if (mass <= 0).any() or not np.isfinite(mass).all():
        raise ValueError("autograd atoms must all have positive finite detected mass")

    observed = mock.measurements
    if max_objects is not None:
        if int(max_objects) <= 0:
            raise ValueError("max_objects must be positive")
        observed = observed.iloc[: int(max_objects)]
    device = bundle.device
    zero = likelihood.cache.get(0.0, 0.0).flow.iloc[atoms]
    raw_zero = bundle.condition_preprocessor.raw_tensor_from_frame(zero, device)
    log_mass = torch.as_tensor(
        np.log(mass), dtype=raw_zero.dtype, device=device
    )
    score = np.empty((len(observed), 2), dtype=np.float64)
    information = np.empty((len(observed), 2, 2), dtype=np.float64)
    started = time.perf_counter()
    for object_index in range(len(observed)):
        target = bundle._targets_from_frame(observed.iloc[[object_index]])
        target = target.expand(len(atoms), -1)
        shear = torch.tensor(
            center, dtype=raw_zero.dtype, device=device, requires_grad=True
        )
        e1, e2 = _torch_shear_ellipticity(
            raw_zero[:, first], raw_zero[:, second], shear[0], shear[1]
        )
        raw = raw_zero.clone()
        raw[:, first] = e1
        raw[:, second] = e2
        context = bundle.condition_preprocessor.transform_tensor(raw)
        log_terms = bundle.model.log_prob(target, context) + log_mass
        log_evidence = torch.logsumexp(log_terms, dim=0)
        gradient = torch.autograd.grad(
            log_evidence, shear, create_graph=True
        )[0]
        rows = []
        for component in range(2):
            rows.append(
                torch.autograd.grad(
                    gradient[component],
                    shear,
                    retain_graph=component == 0,
                )[0]
            )
        score[object_index] = gradient.detach().cpu().numpy()
        information[object_index] = -torch.stack(rows).detach().cpu().numpy()
    return Section5AutogradResult(
        score=score,
        information=information,
        atom_indices=atoms,
        elapsed_seconds=float(time.perf_counter() - started),
    )


def _tail_diagnostics(score: np.ndarray) -> TailDiagnostics:
    score = np.asarray(score, dtype=np.float64)
    absolute = np.abs(score[np.isfinite(score)])
    if not len(absolute):
        return TailDiagnostics(*(float("nan"),) * 5, hill_k=0)
    centered = score - np.mean(score)
    variance = float(np.mean(centered**2))
    excess = (
        float(np.mean(centered**4) / variance**2 - 3.0)
        if variance > 0
        else float("nan")
    )
    positive = np.sort(absolute[absolute > 0])[::-1]
    k = min(256, max(5, len(positive) // 20)) if len(positive) >= 6 else 0
    hill = float("nan")
    if k and len(positive) > k and positive[k] > 0:
        denominator = float(np.log(positive[:k] / positive[k]).sum())
        if denominator > 0:
            hill = float(k / denominator)
    return TailDiagnostics(
        max_abs_score=float(np.max(absolute)),
        p99_abs_score=float(np.percentile(absolute, 99)),
        p999_abs_score=float(np.percentile(absolute, 99.9)),
        excess_kurtosis=excess,
        hill_tail_index=hill,
        hill_k=int(k),
    )


def _summarize_importance(parts: Sequence[tuple[np.ndarray, ...]], n_draws: int):
    if not parts:
        return None
    ess, maximum, outside, global_draw = (
        np.concatenate([part[index] for part in parts]) for index in range(4)
    )
    return StreamedImportanceDiagnostics(
        mean_ess=float(np.mean(ess)),
        median_ess=float(np.median(ess)),
        p10_ess=float(np.percentile(ess, 10)),
        mean_ess_fraction=float(np.mean(ess) / n_draws),
        p90_max_weight_fraction=float(np.percentile(maximum, 90)),
        mean_outside_local_contribution=float(np.mean(outside)),
        mean_global_draw_contribution=float(np.mean(global_draw)),
    )


def _diagnostic_arrays(log_weights: np.ndarray, draw: ProposalDraw):
    peak = np.max(log_weights, axis=1, keepdims=True)
    finite = np.isfinite(peak[:, 0])
    normalized = np.zeros_like(log_weights)
    normalized[finite] = np.exp(log_weights[finite] - peak[finite])
    total = normalized.sum(axis=1, keepdims=True)
    good = finite & (total[:, 0] > 0)
    normalized[good] /= total[good]
    ess = np.zeros(len(log_weights), dtype=np.float64)
    ess[good] = 1.0 / np.square(normalized[good]).sum(axis=1)
    return (
        ess,
        normalized.max(axis=1),
        np.sum(normalized * (~draw.local_member), axis=1),
        np.sum(normalized * draw.global_component, axis=1),
    )


def _diagnostic_tensors(log_weights: torch.Tensor, draw: ProposalDraw):
    """Device-side importance diagnostics with only four vectors returned."""

    peak = torch.max(log_weights, dim=1, keepdim=True).values
    finite = torch.isfinite(peak[:, 0])
    normalized = torch.zeros_like(log_weights)
    normalized[finite] = torch.exp(log_weights[finite] - peak[finite])
    total = normalized.sum(dim=1, keepdim=True)
    good = finite & (total[:, 0] > 0)
    normalized[good] /= total[good]
    ess = torch.zeros(len(log_weights), dtype=log_weights.dtype, device=log_weights.device)
    ess[good] = 1.0 / normalized[good].square().sum(dim=1)
    local = torch.as_tensor(draw.local_member, device=log_weights.device)
    global_component = torch.as_tensor(
        draw.global_component, device=log_weights.device
    )
    values = (
        ess,
        normalized.max(dim=1).values,
        (normalized * (~local)).sum(dim=1),
        (normalized * global_component).sum(dim=1),
    )
    return tuple(value.detach().cpu().numpy().astype(np.float64) for value in values)


def _pilot_concentration(log_weights: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Return per-object ESS fraction and maximum weight for a pilot draw."""

    weights = log_weights.detach().to(dtype=torch.float64)
    peak = torch.max(weights, dim=1, keepdim=True).values
    finite = torch.isfinite(peak[:, 0])
    shifted = torch.zeros_like(weights)
    shifted[finite] = torch.exp(weights[finite] - peak[finite])
    total = shifted.sum(dim=1, keepdim=True)
    good = finite & (total[:, 0] > 0)
    normalized = torch.zeros_like(weights)
    normalized[good] = shifted[good] / total[good]
    ess = torch.zeros(len(weights), dtype=weights.dtype, device=weights.device)
    ess[good] = 1.0 / normalized[good].square().sum(dim=1)
    maximum = normalized.max(dim=1).values
    return (
        (ess / weights.shape[1]).cpu().numpy(),
        maximum.cpu().numpy(),
    )


def _coalesced_logsumexp(
    log_weights: torch.Tensor,
    draw: CoalescedProposalDraw,
    n_draws: int,
) -> torch.Tensor:
    counts = torch.as_tensor(
        draw.counts(n_draws), dtype=log_weights.dtype, device=log_weights.device
    )
    terms = torch.where(
        counts > 0,
        log_weights + torch.log(counts.clamp_min(1)),
        torch.full_like(log_weights, -torch.inf),
    )
    return torch.logsumexp(terms, dim=1) - np.log(n_draws)


def _coalesced_diagnostic_tensors(
    log_weights: torch.Tensor,
    draw: CoalescedProposalDraw,
    n_draws: int,
):
    counts_np = draw.counts(n_draws)
    outside_np, global_np = draw.category_counts(n_draws)
    counts = torch.as_tensor(
        counts_np, dtype=log_weights.dtype, device=log_weights.device
    )
    peak = torch.max(
        torch.where(counts > 0, log_weights, torch.full_like(log_weights, -torch.inf)),
        dim=1,
        keepdim=True,
    ).values
    weights = torch.where(
        counts > 0, torch.exp(log_weights - peak), torch.zeros_like(log_weights)
    )
    total = (counts * weights).sum(dim=1)
    good = torch.isfinite(peak[:, 0]) & (total > 0)
    ess = torch.zeros(len(log_weights), dtype=log_weights.dtype, device=log_weights.device)
    ess[good] = total[good].square() / (
        counts[good] * weights[good].square()
    ).sum(dim=1)
    maximum = torch.zeros_like(ess)
    maximum[good] = weights[good].max(dim=1).values / total[good]
    outside = torch.as_tensor(
        outside_np, dtype=log_weights.dtype, device=log_weights.device
    )
    global_component = torch.as_tensor(
        global_np, dtype=log_weights.dtype, device=log_weights.device
    )
    outside_fraction = torch.zeros_like(ess)
    global_fraction = torch.zeros_like(ess)
    outside_fraction[good] = (outside[good] * weights[good]).sum(dim=1) / total[good]
    global_fraction[good] = (
        global_component[good] * weights[good]
    ).sum(dim=1) / total[good]
    values = (ess, maximum, outside_fraction, global_fraction)
    return tuple(value.detach().cpu().numpy().astype(np.float64) for value in values)


def _reuse_adapted_reference_weights(
    likelihood: CatalogueLikelihood,
    observed,
    observed_targets: torch.Tensor,
    candidate_target: torch.Tensor,
    draw: ProposalDraw,
    *,
    center: Sequence[float] = (0.0, 0.0),
    object_chunk: int,
    atom_chunk: int,
) -> torch.Tensor:
    """Reuse reference likelihoods and evaluate only global draws outside them."""

    device = candidate_target.device
    position = torch.as_tensor(draw.local_position, dtype=torch.long, device=device)
    safe = position.clamp_min(0)
    rows = torch.arange(len(draw.indices), device=device)[:, None]
    proposal = torch.as_tensor(
        np.ascontiguousarray(draw.probability),
        dtype=candidate_target.dtype,
        device=device,
    )
    output = candidate_target[rows, safe] - torch.log(proposal)
    outside = ~draw.local_member
    if not outside.any():
        return output

    counts = outside.sum(axis=1)
    width = int(counts.max())
    packed_indices = np.empty((len(draw.indices), width), dtype=np.int64)
    packed_probability = np.ones((len(draw.indices), width), dtype=np.float64)
    for row, count in enumerate(counts):
        count = int(count)
        selected = np.flatnonzero(outside[row])
        if count:
            packed_indices[row, :count] = draw.indices[row, selected]
            packed_probability[row, :count] = draw.probability[row, selected]
        if count < width:
            packed_indices[row, count:] = draw.indices[row, 0]
            packed_probability[row, count:] = draw.probability[row, 0]
    evaluated = likelihood.log_importance_weights_tensor(
        observed,
        float(center[0]),
        float(center[1]),
        atom_indices=packed_indices,
        proposal_probability=packed_probability,
        observed_targets=observed_targets,
        object_chunk=object_chunk,
        atom_chunk=atom_chunk,
    )
    for row, count in enumerate(counts):
        count = int(count)
        if count:
            selected = torch.as_tensor(
                np.flatnonzero(outside[row]), dtype=torch.long, device=device
            )
            output[row, selected] = evaluated[row, :count]
    return output


def summarize_section5(
    score: np.ndarray,
    information: np.ndarray,
    *,
    component: str,
    h: float,
    n_draws: int,
    importance: Optional[StreamedImportanceDiagnostics] = None,
) -> Section5Estimate:
    """Summarize one per-object score/information pair."""

    score = np.asarray(score, dtype=np.float64)
    information = np.asarray(information, dtype=np.float64)
    if score.ndim != 1 or information.shape != score.shape or len(score) < 2:
        raise ValueError("score and information must be aligned vectors with at least two rows")
    if not np.isfinite(score).all() or not np.isfinite(information).all():
        raise ValueError("score or information contains non-finite values")
    n = len(score)
    score_mean = float(np.mean(score))
    score_variance = float(np.var(score, ddof=1))
    score_mean_se = float(np.sqrt(score_variance / n))
    information_mean = float(np.mean(information))
    information_sum = float(np.sum(information))
    estimate = float(np.sum(score) / information_sum)
    residual = score - estimate * information
    robust_se = float(
        np.sqrt(n / (n - 1) * np.square(residual).sum() / information_sum**2)
    )
    model_se = (
        float(1.0 / np.sqrt(information_sum))
        if information_sum > 0
        else float("nan")
    )

    # Delta-method influence function for E[I] / Var(s).  The population-form
    # variance is used inside the influence function; the reported ratio uses
    # the usual unbiased sample variance.
    centered_square = np.square(score - score_mean)
    variance_population = float(np.mean(centered_square))
    ratio = (
        float(information_mean / score_variance)
        if score_variance > 0
        else float("nan")
    )
    ratio_se = float("nan")
    if variance_population > 0:
        ratio_population = information_mean / variance_population
        influence = (
            (information - information_mean) / variance_population
            - ratio_population
            * (centered_square - variance_population)
            / variance_population
        )
        ratio_se = float(np.std(influence, ddof=1) / np.sqrt(n))
    return Section5Estimate(
        component=str(component),
        h=float(h),
        n_draws=int(n_draws),
        n_objects=n,
        estimated_shear=estimate,
        robust_standard_error=robust_se,
        model_standard_error=model_se,
        score_mean=score_mean,
        score_mean_standard_error=score_mean_se,
        score_z=(
            float(score_mean / score_mean_se) if score_mean_se > 0 else float("nan")
        ),
        score_variance=score_variance,
        information_mean=information_mean,
        information_identity_ratio=ratio,
        information_identity_standard_error=ratio_se,
        tail=_tail_diagnostics(score),
        importance=importance,
    )


def summarize_paired_section5(
    positive_score: np.ndarray,
    positive_information: np.ndarray,
    negative_score: np.ndarray,
    negative_information: np.ndarray,
    *,
    injected_component: str,
    estimated_component: str,
    amplitude: float,
    h: float,
    n_draws: int,
    block_ids: Optional[np.ndarray] = None,
) -> PairedSection5Estimate:
    """Summarize a common-random-number ``+g/-g`` response.

    The response is the symmetric derivative of the *population* Section 5
    estimator.  Since ``E[s]=0`` at the expansion point, differentiating
    ``E[s]/E[I]`` leaves ``dE[s]/dg / E[I]``; fluctuations of each arm's
    denominator do not enter at first order.  Estimating this as the paired
    ratio ``mean[(s+ - s-)/(2g)] / mean[(I+ + I-)/2]`` is both the direct
    target and much less noisy than differencing two finite-sample ratios.
    """

    arrays = tuple(
        np.asarray(value, dtype=np.float64)
        for value in (
            positive_score,
            positive_information,
            negative_score,
            negative_information,
        )
    )
    positive_score, positive_information, negative_score, negative_information = arrays
    if positive_score.ndim != 1 or any(value.shape != positive_score.shape for value in arrays):
        raise ValueError("paired scores and information must be aligned vectors")
    if len(positive_score) < 2 or not all(np.isfinite(value).all() for value in arrays):
        raise ValueError("paired moments must contain at least two finite rows")
    amplitude = float(amplitude)
    if not np.isfinite(amplitude) or amplitude <= 0:
        raise ValueError("paired injection amplitude must be positive and finite")
    if injected_component not in {"g1", "g2"} or estimated_component not in {"g1", "g2"}:
        raise ValueError("paired component names must be 'g1' or 'g2'")

    positive_information_mean = float(np.mean(positive_information))
    negative_information_mean = float(np.mean(negative_information))
    if positive_information_mean == 0 or negative_information_mean == 0:
        raise ValueError("paired mean information must be nonzero")
    positive_estimate = float(np.sum(positive_score) / np.sum(positive_information))
    negative_estimate = float(np.sum(negative_score) / np.sum(negative_information))
    positive_influence = (
        positive_score - positive_estimate * positive_information
    ) / positive_information_mean
    negative_influence = (
        negative_score - negative_estimate * negative_information
    ) / negative_information_mean
    response_numerator = (positive_score - negative_score) / (2.0 * amplitude)
    symmetric_information = (positive_information + negative_information) / 2.0
    information_mean = float(np.mean(symmetric_information))
    if information_mean == 0:
        raise ValueError("paired symmetric mean information must be nonzero")
    response = float(np.mean(response_numerator) / information_mean)
    response_influence = (
        response_numerator - response * symmetric_information
    ) / information_mean
    offset_numerator = (positive_score + negative_score) / 2.0
    symmetric_offset = float(np.mean(offset_numerator) / information_mean)
    offset_influence = (
        offset_numerator - symmetric_offset * symmetric_information
    ) / information_mean
    n = len(positive_score)
    if block_ids is None:
        response_units = response_influence
        offset_units = offset_influence
    else:
        blocks = np.asarray(block_ids)
        if blocks.shape != (n,):
            raise ValueError("paired block_ids must align with the object moments")
        labels, inverse = np.unique(blocks, return_inverse=True)
        counts = np.bincount(inverse)
        if len(labels) < 2 or not np.all(counts == counts[0]):
            raise ValueError("paired blocks must be at least two equal-size groups")
        response_units = np.bincount(
            inverse, weights=response_influence
        ) / counts
        offset_units = np.bincount(inverse, weights=offset_influence) / counts
    expected = float(injected_component == estimated_component)
    n_units = len(response_units)
    response_se = float(np.std(response_units, ddof=1) / np.sqrt(n_units))
    offset_se = float(np.std(offset_units, ddof=1) / np.sqrt(n_units))
    bias = float(response - expected)
    def correlation(first, second):
        if np.std(first) == 0 or np.std(second) == 0:
            return float("nan")
        return float(np.corrcoef(first, second)[0, 1])

    return PairedSection5Estimate(
        injected_component=injected_component,
        estimated_component=estimated_component,
        amplitude=amplitude,
        h=float(h),
        n_draws=int(n_draws),
        n_objects=n,
        n_blocks=n_units,
        positive_estimated_shear=positive_estimate,
        negative_estimated_shear=negative_estimate,
        response=float(response),
        expected_response=expected,
        response_bias=bias,
        robust_standard_error=response_se,
        response_pull=(float(bias / response_se) if response_se > 0 else float("nan")),
        symmetric_offset=symmetric_offset,
        symmetric_offset_standard_error=offset_se,
        score_correlation=correlation(positive_score, negative_score),
        influence_correlation=correlation(positive_influence, negative_influence),
        response_tail=_tail_diagnostics(response_units),
    )


def _view_normalizer(likelihood: CatalogueLikelihood, g1: float, g2: float) -> float:
    # In the shape-only, no-cut closure, detection and prior mass are exactly
    # shear-invariant.  Reuse the zero view instead of materializing a full
    # pandas catalogue at every stencil point.
    if likelihood.selection is None and likelihood.tensor_shape_only_available:
        likelihood.cache.validate_detection_shear_invariance()
        view = likelihood.cache.get(0.0, 0.0)
    else:
        view = likelihood.cache.get(g1, g2)
    if likelihood.selection is None:
        mass = likelihood.cache.prior.weights * view.detection_probability
        total = float(np.sum(mass))
        if total <= 0:
            raise RuntimeError("catalogue prior has zero detected probability")
        return float(np.log(total))
    return likelihood.log_population_normalization(g1, g2)


def run_streamed_section5(
    likelihood: CatalogueLikelihood,
    mock: MockCatalogue,
    proposal: DefensiveLocalProposal,
    *,
    steps: Sequence[float] = (0.005, 0.01, 0.02),
    ladder: Sequence[int] = (8192, 32768, 65536),
    n_candidates: int = 131072,
    epsilon: float,
    bandwidth: float,
    proposal_seeds: Sequence[int],
    object_chunk: int = 16,
    atom_chunk: int = 4096,
    independent_bank_weights: Optional[Mapping[str, np.ndarray]] = None,
    bank_h: float = 0.005,
    posterior_adapt_proposal: bool = False,
    require_null_truth: bool = True,
    retain_object_moments: bool = False,
    proposal_observed=None,
    use_tensor_native: bool = True,
) -> Section5StreamResult:
    """Evaluate both Section 5 components without an object-by-atom tensor.

    The derivatives are always evaluated at zero shear.  By default this is
    the historical null validator and therefore requires ``g_true=(0,0)``.
    Set ``require_null_truth=False`` only for a nonzero closure mock whose
    local response is intentionally being measured about zero.
    """

    if likelihood.selection is not None:
        raise ValueError("Section 5 null closure requires measured-output cuts disabled")
    if likelihood.cache.blend_response is not None:
        raise ValueError("Section 5 null closure requires external R_blend=0")
    injected = mock.truth[["injected_g1", "injected_g2"]].drop_duplicates()
    if len(injected) != 1 or not np.isfinite(injected.to_numpy(dtype=float)).all():
        raise ValueError("Section 5 closure requires one finite injected shear per mock")
    if require_null_truth and not np.allclose(injected.iloc[0], 0.0, rtol=0, atol=1e-14):
        raise ValueError("Section 5 null closure requires g_true=(0,0)")
    steps = tuple(sorted({float(value) for value in steps}))
    ladder = tuple(sorted({int(value) for value in ladder}))
    seeds = tuple(dict.fromkeys(int(value) for value in proposal_seeds))
    if not steps or steps[0] <= 0 or not np.isfinite(steps).all():
        raise ValueError("finite-difference steps must be positive and finite")
    if not ladder or ladder[0] <= 0:
        raise ValueError("draw ladder must be positive")
    if not seeds or object_chunk <= 0 or atom_chunk <= 0:
        raise ValueError("seeds and chunk sizes must be non-empty/positive")
    proposal_matches_observed = proposal_observed is None
    if proposal_observed is None:
        proposal_observed = mock.measurements
    elif len(proposal_observed) != len(mock.measurements):
        raise ValueError("proposal observations must align one-to-one with the mock")

    views = {(0.0, 0.0)}
    for h in steps:
        views.update({(h, 0.0), (-h, 0.0), (0.0, h), (0.0, -h)})
    normalizers = {view: _view_normalizer(likelihood, *view) for view in sorted(views)}
    bank_h = float(bank_h)
    bank_views = {
        (0.0, 0.0),
        (bank_h, 0.0),
        (-bank_h, 0.0),
        (0.0, bank_h),
        (0.0, -bank_h),
    }
    if independent_bank_weights and bank_h not in steps:
        raise ValueError("bank_h must be one of the evaluated finite-difference steps")
    full_prior = likelihood.cache.prior.weights
    bank_log_ratio = {}
    bank_normalizers = {}
    for label, values in (independent_bank_weights or {}).items():
        weights = np.asarray(values, dtype=np.float64)
        if weights.shape != full_prior.shape or not np.isfinite(weights).all() or (weights < 0).any():
            raise ValueError(f"independent bank {label!r} has invalid weights")
        total = float(weights.sum())
        if total <= 0:
            raise ValueError(f"independent bank {label!r} is empty")
        weights = weights / total
        if np.any((weights > 0) & (full_prior <= 0)):
            raise ValueError("independent bank has support outside the full prior")
        ratio = np.full(len(weights), -np.inf, dtype=np.float64)
        positive = weights > 0
        ratio[positive] = np.log(weights[positive]) - np.log(full_prior[positive])
        bank_log_ratio[str(label)] = ratio
        bank_normalizers[str(label)] = {
            view: float(
                np.log(
                    np.sum(
                        weights
                        * likelihood.cache.get(*view).detection_probability
                    )
                )
            )
            for view in bank_views
        }
    n_objects = len(mock.measurements)
    results = []
    bank_results: dict[str, Section5SeedResult] = {}
    retained_moments: dict[int, Section5ObjectMoments] = {}
    for seed in seeds:
        started = time.perf_counter()
        flow_evaluation_count = 0
        scores: dict[tuple[str, float, int], list[np.ndarray]] = {}
        infos: dict[tuple[str, float, int], list[np.ndarray]] = {}
        diagnostics: dict[int, list[tuple[np.ndarray, ...]]] = {
            n_draws: [] for n_draws in ladder
        }
        collect_banks = seed == seeds[0] and bool(bank_log_ratio)
        bank_scores: dict[tuple[str, str, int], list[np.ndarray]] = {}
        bank_infos: dict[tuple[str, str, int], list[np.ndarray]] = {}
        bank_diagnostics: dict[tuple[str, int], list[tuple[np.ndarray, ...]]] = {}
        for object_start in range(0, n_objects, object_chunk):
            object_stop = min(object_start + object_chunk, n_objects)
            observed = mock.measurements.iloc[object_start:object_stop].reset_index(drop=True)
            proposal_frame = proposal_observed.iloc[
                object_start:object_stop
            ].reset_index(drop=True)
            tensor_native = bool(
                use_tensor_native and likelihood.tensor_native_available
            )
            observed_targets = (
                likelihood.observed_target_tensor(observed)
                if tensor_native
                else None
            )
            candidate_target_tensor = None
            if posterior_adapt_proposal:
                candidates = proposal.candidates(
                    proposal_frame, n_candidates=n_candidates
                )
                if tensor_native:
                    proposal_targets = (
                        observed_targets
                        if proposal_matches_observed
                        else likelihood.observed_target_tensor(proposal_frame)
                    )
                    candidate_target_tensor = likelihood.log_importance_weights_tensor(
                        proposal_frame,
                        0.0,
                        0.0,
                        atom_indices=candidates.indices,
                        proposal_probability=np.ones_like(
                            candidates.indices, dtype=np.float64
                        ),
                        observed_targets=proposal_targets,
                        object_chunk=len(observed),
                        atom_chunk=atom_chunk,
                    )
                    candidate_target = (
                        candidate_target_tensor.detach().cpu().numpy().astype(np.float64)
                    )
                    flow_evaluation_count += int(candidates.indices.size)
                else:
                    candidate_target = likelihood.log_importance_weights(
                        proposal_frame,
                        0.0,
                        0.0,
                        atom_indices=candidates.indices,
                        proposal_probability=np.ones_like(
                            candidates.indices, dtype=np.float64
                        ),
                        object_chunk=len(observed),
                        atom_chunk=atom_chunk,
                    )
                    flow_evaluation_count += int(candidates.indices.size)
                draw = proposal.draw_adapted(
                    candidates,
                    candidate_target,
                    n_draws=ladder[-1],
                    epsilon=epsilon,
                    seed=seed,
                    object_offset=object_start,
                )
                del candidate_target, candidates
            else:
                draw = proposal.draw(
                    proposal_frame,
                    n_draws=ladder[-1],
                    n_candidates=n_candidates,
                    epsilon=epsilon,
                    bandwidth=bandwidth,
                    seed=seed,
                    object_offset=object_start,
                )
            coalesced = draw.coalesce() if tensor_native else None
            log_likelihood: dict[tuple[tuple[float, float], int], np.ndarray] = {}
            bank_log_likelihood: dict[
                tuple[str, tuple[float, float], int], np.ndarray
            ] = {}
            for view in sorted(views):
                if tensor_native:
                    if (
                        view == (0.0, 0.0)
                        and posterior_adapt_proposal
                        and proposal_matches_observed
                        and candidate_target_tensor is not None
                    ):
                        full_zero_weights = _reuse_adapted_reference_weights(
                            likelihood,
                            observed,
                            observed_targets,
                            candidate_target_tensor,
                            draw,
                            object_chunk=len(observed),
                            atom_chunk=atom_chunk,
                        )
                        outside_width = int((~draw.local_member).sum(axis=1).max())
                        flow_evaluation_count += len(draw.indices) * outside_width
                        first_position = torch.as_tensor(
                            coalesced.first_position,
                            dtype=torch.long,
                            device=full_zero_weights.device,
                        )
                        rows = torch.arange(
                            len(draw.indices), device=full_zero_weights.device
                        )[:, None]
                        log_weights_tensor = full_zero_weights[rows, first_position]
                        del full_zero_weights, first_position, rows
                    else:
                        log_weights_tensor = likelihood.log_importance_weights_tensor(
                            observed,
                            *view,
                            atom_indices=coalesced.indices,
                            proposal_probability=coalesced.probability,
                            observed_targets=observed_targets,
                            object_chunk=len(observed),
                            atom_chunk=atom_chunk,
                        )
                        flow_evaluation_count += int(coalesced.indices.size)
                    for n_draws in ladder:
                        reduced = (
                            _coalesced_logsumexp(
                                log_weights_tensor, coalesced, n_draws
                            )
                            - normalizers[view]
                        )
                        log_likelihood[(view, n_draws)] = (
                            reduced.detach().cpu().numpy().astype(np.float64)
                        )
                        if view == (0.0, 0.0):
                            diagnostics[n_draws].append(
                                _coalesced_diagnostic_tensors(
                                    log_weights_tensor,
                                    coalesced,
                                    n_draws,
                                )
                            )
                    if collect_banks and view in bank_views:
                        draw_indices = torch.as_tensor(
                            np.ascontiguousarray(coalesced.indices),
                            dtype=torch.long,
                            device=log_weights_tensor.device,
                        )
                        for label, ratio in bank_log_ratio.items():
                            ratio_tensor = torch.as_tensor(
                                ratio,
                                dtype=log_weights_tensor.dtype,
                                device=log_weights_tensor.device,
                            )
                            adjusted = log_weights_tensor + ratio_tensor[draw_indices]
                            for n_draws in ladder[-2:]:
                                reduced = (
                                    _coalesced_logsumexp(
                                        adjusted, coalesced, n_draws
                                    )
                                    - bank_normalizers[label][view]
                                )
                                bank_log_likelihood[(label, view, n_draws)] = (
                                    reduced.detach().cpu().numpy().astype(np.float64)
                                )
                                if view == (0.0, 0.0):
                                    bank_diagnostics.setdefault((label, n_draws), []).append(
                                        _coalesced_diagnostic_tensors(
                                            adjusted, coalesced, n_draws
                                        )
                                    )
                            del adjusted, ratio_tensor
                        del draw_indices
                    del log_weights_tensor
                else:
                    log_weights = likelihood.log_importance_weights(
                        observed,
                        *view,
                        atom_indices=draw.indices,
                        proposal_probability=draw.probability,
                        object_chunk=len(observed),
                        atom_chunk=atom_chunk,
                    )
                    flow_evaluation_count += int(draw.indices.size)
                    for n_draws in ladder:
                        log_likelihood[(view, n_draws)] = (
                            logsumexp(log_weights[:, :n_draws], axis=1)
                            - np.log(n_draws)
                            - normalizers[view]
                        )
                        if view == (0.0, 0.0):
                            diagnostics[n_draws].append(
                                _diagnostic_arrays(
                                    log_weights[:, :n_draws], draw.prefix(n_draws)
                                )
                            )
                    if collect_banks and view in bank_views:
                        for label, ratio in bank_log_ratio.items():
                            adjusted = log_weights + ratio[draw.indices]
                            for n_draws in ladder[-2:]:
                                bank_log_likelihood[(label, view, n_draws)] = (
                                    logsumexp(adjusted[:, :n_draws], axis=1)
                                    - np.log(n_draws)
                                    - bank_normalizers[label][view]
                                )
                                if view == (0.0, 0.0):
                                    bank_diagnostics.setdefault((label, n_draws), []).append(
                                        _diagnostic_arrays(
                                            adjusted[:, :n_draws], draw.prefix(n_draws)
                                        )
                                    )
                            del adjusted
                    del log_weights
            del candidate_target_tensor, observed_targets, coalesced

            for h in steps:
                for component, plus, minus in (
                    ("g1", (h, 0.0), (-h, 0.0)),
                    ("g2", (0.0, h), (0.0, -h)),
                ):
                    for n_draws in ladder:
                        zero = log_likelihood[((0.0, 0.0), n_draws)]
                        upper = log_likelihood[(plus, n_draws)]
                        lower = log_likelihood[(minus, n_draws)]
                        key = (component, h, n_draws)
                        scores.setdefault(key, []).append((upper - lower) / (2.0 * h))
                        infos.setdefault(key, []).append(
                            -(upper - 2.0 * zero + lower) / h**2
                        )
            if collect_banks:
                for label in bank_log_ratio:
                    for component, plus, minus in (
                        ("g1", (bank_h, 0.0), (-bank_h, 0.0)),
                        ("g2", (0.0, bank_h), (0.0, -bank_h)),
                    ):
                        for n_draws in ladder[-2:]:
                            zero = bank_log_likelihood[
                                (label, (0.0, 0.0), n_draws)
                            ]
                            upper = bank_log_likelihood[(label, plus, n_draws)]
                            lower = bank_log_likelihood[(label, minus, n_draws)]
                            key = (label, component, n_draws)
                            bank_scores.setdefault(key, []).append(
                                (upper - lower) / (2.0 * bank_h)
                            )
                            bank_infos.setdefault(key, []).append(
                                -(upper - 2.0 * zero + lower) / bank_h**2
                            )

        estimates = []
        retained_score: dict[tuple[str, float, int], np.ndarray] = {}
        retained_information: dict[tuple[str, float, int], np.ndarray] = {}
        for component in ("g1", "g2"):
            for h in steps:
                for n_draws in ladder:
                    key = (component, h, n_draws)
                    score_values = np.concatenate(scores[key])
                    information_values = np.concatenate(infos[key])
                    estimates.append(
                        summarize_section5(
                            score_values,
                            information_values,
                            component=component,
                            h=h,
                            n_draws=n_draws,
                            importance=_summarize_importance(
                                diagnostics[n_draws], n_draws
                            ),
                        )
                    )
                    if retain_object_moments:
                        retained_score[key] = score_values
                        retained_information[key] = information_values
        results.append(
            Section5SeedResult(
                proposal_seed=seed,
                steps=steps,
                ladder=ladder,
                n_candidates=int(n_candidates),
                n_objects=n_objects,
                n_views=len(views),
                flow_evaluations=int(flow_evaluation_count),
                elapsed_seconds=float(time.perf_counter() - started),
                estimates=tuple(estimates),
            )
        )
        if retain_object_moments:
            retained_moments[seed] = Section5ObjectMoments(
                proposal_seed=seed,
                score=retained_score,
                information=retained_information,
            )
        if collect_banks:
            for label in bank_log_ratio:
                estimates = []
                for component in ("g1", "g2"):
                    for n_draws in ladder[-2:]:
                        key = (label, component, n_draws)
                        estimates.append(
                            summarize_section5(
                                np.concatenate(bank_scores[key]),
                                np.concatenate(bank_infos[key]),
                                component=component,
                                h=bank_h,
                                n_draws=n_draws,
                                importance=_summarize_importance(
                                    bank_diagnostics[(label, n_draws)], n_draws
                                ),
                            )
                        )
                bank_results[label] = Section5SeedResult(
                    proposal_seed=seed,
                    steps=(bank_h,),
                    ladder=tuple(ladder[-2:]),
                    n_candidates=int(n_candidates),
                    n_objects=n_objects,
                    n_views=5,
                    flow_evaluations=0,
                    elapsed_seconds=float(time.perf_counter() - started),
                    estimates=tuple(estimates),
                )
    return Section5StreamResult(tuple(results), bank_results, retained_moments)


def run_adaptive_section5(
    likelihood: CatalogueLikelihood,
    mock: MockCatalogue,
    proposal: DefensiveLocalProposal,
    *,
    center: Sequence[float] = (0.0, 0.0),
    h: float = 0.00125,
    draw_ladder: Sequence[int] = (2048, 4096, 8192, 16384),
    n_candidates: int = 16384,
    proposal_prefilter_candidates: Optional[int] = None,
    epsilon: float = 0.1,
    proposal_seed: int = 8701,
    min_ess: float = 256.0,
    max_weight_fraction: float = 0.25,
    allocation_method: str = "production_prefix",
    pilot_draws: int = 512,
    pilot_seed: Optional[int] = None,
    pilot_safety_factor: float = 1.0,
    bias_correction: str = "none",
    candidate_backend: str = "scipy",
    full_information: bool = False,
    retain_full_ladder: bool = False,
    object_id_offset: int = 0,
    object_chunk: int = 16,
    atom_chunk: int = 4096,
) -> AdaptiveSection5Result:
    """Fast local Section 5 path with object-specific nested draw stopping.

    With ``allocation_method='production_prefix'``, the zero/centre production
    weights choose the first acceptable nested prefix (the historical path).
    With ``allocation_method='independent_pilot'``, a separate random stream
    chooses a fixed production rung before the production draw is generated.
    Objects are grouped by that rung, duplicate atoms are evaluated once, and
    the same production draws are held fixed across all stencil views.  With
    ``full_information=True``, four corner views add the mixed derivative and
    return an ``(N,2,2)`` information matrix suitable for a one-step 2D update.
    """

    if not likelihood.tensor_native_available:
        raise TypeError("adaptive Section 5 requires a tensor-native Torch flow")
    if likelihood.selection is not None:
        raise ValueError("adaptive Section 5 currently requires measured cuts disabled")
    if likelihood.cache.blend_response is not None:
        raise ValueError("adaptive Section 5 currently requires external R_blend=0")
    likelihood.cache.validate_detection_shear_invariance()
    center = np.asarray(center, dtype=np.float64)
    ladder = tuple(sorted({int(value) for value in draw_ladder}))
    if center.shape != (2,) or not np.isfinite(center).all():
        raise ValueError("center must be a finite two-vector")
    if not np.isfinite(h) or h <= 0:
        raise ValueError("h must be finite and positive")
    if not ladder or ladder[0] <= 0:
        raise ValueError("adaptive draw ladder must be positive")
    if n_candidates <= 0 or object_chunk <= 0 or atom_chunk <= 0:
        raise ValueError("candidate and chunk sizes must be positive")
    if object_id_offset < 0:
        raise ValueError("object_id_offset must be non-negative")
    if allocation_method not in ("production_prefix", "independent_pilot"):
        raise ValueError("unknown adaptive allocation method")
    if bias_correction not in ("none", "richardson_1_over_m"):
        raise ValueError("unknown adaptive finite-draw bias correction")
    if candidate_backend not in ("scipy", "torch"):
        raise ValueError("unknown candidate-query backend")
    if bias_correction != "none" and retain_full_ladder:
        raise ValueError(
            "finite-draw bias correction and retained raw ladder are separate modes"
        )
    if bias_correction == "richardson_1_over_m" and any(
        value < 2 or value % 2 for value in ladder
    ):
        raise ValueError("Richardson correction requires positive even draw rungs")
    if pilot_draws <= 0:
        raise ValueError("pilot draws must be positive")
    if not np.isfinite(pilot_safety_factor) or pilot_safety_factor < 1.0:
        raise ValueError("pilot safety factor must be finite and at least one")
    if pilot_seed is None:
        pilot_seed = int(proposal_seed) + 1_000_003
    if allocation_method == "independent_pilot" and int(pilot_seed) == int(proposal_seed):
        raise ValueError("pilot and production proposal seeds must differ")
    if (
        proposal_prefilter_candidates is not None
        and proposal_prefilter_candidates < n_candidates
    ):
        raise ValueError("proposal prefilter cannot be smaller than final support")

    views = {
        "zero": (float(center[0]), float(center[1])),
        "g1_plus": (float(center[0] + h), float(center[1])),
        "g1_minus": (float(center[0] - h), float(center[1])),
        "g2_plus": (float(center[0]), float(center[1] + h)),
        "g2_minus": (float(center[0]), float(center[1] - h)),
    }
    if full_information:
        views.update(
            {
                "pp": (float(center[0] + h), float(center[1] + h)),
                "pm": (float(center[0] + h), float(center[1] - h)),
                "mp": (float(center[0] - h), float(center[1] + h)),
                "mm": (float(center[0] - h), float(center[1] - h)),
            }
        )
    normalizers = {name: _view_normalizer(likelihood, *view) for name, view in views.items()}
    n_objects = len(mock.measurements)
    score = np.empty((n_objects, 2), dtype=np.float64)
    information = np.empty(
        (n_objects, 2, 2) if full_information else (n_objects, 2),
        dtype=np.float64,
    )
    selected_counts = np.empty(n_objects, dtype=np.int64)
    unique_counts = np.empty(n_objects, dtype=np.int64)
    pilot_ess_fraction = (
        np.empty(n_objects, dtype=np.float64)
        if allocation_method == "independent_pilot"
        else None
    )
    pilot_maximum = (
        np.empty(n_objects, dtype=np.float64)
        if allocation_method == "independent_pilot"
        else None
    )
    ladder_score = (
        np.empty((len(ladder), n_objects, 2), dtype=np.float64)
        if retain_full_ladder
        else None
    )
    ladder_information = (
        np.empty(
            (len(ladder), n_objects, 2, 2)
            if full_information
            else (len(ladder), n_objects, 2),
            dtype=np.float64,
        )
        if retain_full_ladder
        else None
    )
    flow_evaluations = 0
    started = time.perf_counter()
    phase_seconds = {
        "candidate_query": 0.0,
        "candidate_flow": 0.0,
        "allocation_and_center_weights": 0.0,
        "stencil_views_and_reduction": 0.0,
    }

    for object_start in range(0, n_objects, object_chunk):
        object_stop = min(object_start + object_chunk, n_objects)
        observed = mock.measurements.iloc[object_start:object_stop].reset_index(drop=True)
        observed_targets = likelihood.observed_target_tensor(observed)
        phase_started = time.perf_counter()
        candidates = proposal.candidates(
            observed,
            n_candidates=n_candidates,
            prefilter_candidates=proposal_prefilter_candidates,
            torch_device=(
                likelihood.flow_model.device if candidate_backend == "torch" else None
            ),
        )
        phase_seconds["candidate_query"] += time.perf_counter() - phase_started
        phase_started = time.perf_counter()
        candidate_target_tensor = likelihood.log_importance_weights_tensor(
            observed,
            *views["zero"],
            atom_indices=candidates.indices,
            proposal_probability=np.ones_like(candidates.indices, dtype=np.float64),
            observed_targets=observed_targets,
            object_chunk=len(observed),
            atom_chunk=atom_chunk,
        )
        flow_evaluations += int(candidates.indices.size)
        candidate_target = candidate_target_tensor.detach().cpu().numpy().astype(np.float64)
        phase_seconds["candidate_flow"] += time.perf_counter() - phase_started
        phase_started = time.perf_counter()
        if allocation_method == "independent_pilot":
            pilot_draw = proposal.draw_adapted(
                candidates,
                candidate_target,
                n_draws=int(pilot_draws),
                epsilon=epsilon,
                seed=int(pilot_seed),
                object_offset=object_id_offset + object_start,
            )
            pilot_weights = _reuse_adapted_reference_weights(
                likelihood,
                observed,
                observed_targets,
                candidate_target_tensor,
                pilot_draw,
                center=center,
                object_chunk=len(observed),
                atom_chunk=atom_chunk,
            )
            flow_evaluations += len(pilot_draw.indices) * int(
                (~pilot_draw.local_member).sum(axis=1).max()
            )
            pilot_ess, pilot_peak = _pilot_concentration(pilot_weights)
            pilot_ess_fraction[object_start:object_stop] = pilot_ess
            pilot_maximum[object_start:object_stop] = pilot_peak
            if retain_full_ladder:
                counts = np.full(len(observed), ladder[-1], dtype=np.int64)
            else:
                counts = select_independent_pilot_draw_counts(
                    pilot_weights.detach().cpu().numpy(),
                    ladder,
                    min_ess=min_ess,
                    max_weight_fraction=max_weight_fraction,
                    safety_factor=pilot_safety_factor,
                )
            del pilot_draw, pilot_weights, pilot_ess, pilot_peak

        draw = None
        zero_weights = None
        if allocation_method == "production_prefix":
            draw = proposal.draw_adapted(
                candidates,
                candidate_target,
                n_draws=ladder[-1],
                epsilon=epsilon,
                seed=proposal_seed,
                object_offset=object_id_offset + object_start,
            )
            zero_weights = _reuse_adapted_reference_weights(
                likelihood,
                observed,
                observed_targets,
                candidate_target_tensor,
                draw,
                center=center,
                object_chunk=len(observed),
                atom_chunk=atom_chunk,
            )
            flow_evaluations += len(draw.indices) * int(
                (~draw.local_member).sum(axis=1).max()
            )
            if retain_full_ladder:
                # Diagnostic calibration evaluates the maximum prefix once and
                # reduces those same unique-atom weights at every nested rung.
                # It therefore cannot save work in this calibration run, but it
                # supplies common-draw convergence without extra flow calls.
                counts = np.full(len(observed), ladder[-1], dtype=np.int64)
            else:
                counts = select_adaptive_draw_counts(
                    zero_weights.detach().cpu().numpy(),
                    ladder,
                    min_ess=min_ess,
                    max_weight_fraction=max_weight_fraction,
                )
        phase_seconds["allocation_and_center_weights"] += (
            time.perf_counter() - phase_started
        )
        selected_counts[object_start:object_stop] = counts

        phase_started = time.perf_counter()
        for n_draws in np.unique(counts):
            local_rows = np.flatnonzero(counts == n_draws)
            global_rows = object_start + local_rows
            absolute_rows = object_id_offset + global_rows
            group_observed = observed.iloc[local_rows].reset_index(drop=True)
            group_targets = observed_targets[torch.as_tensor(
                local_rows, dtype=torch.long, device=observed_targets.device
            )]
            if allocation_method == "independent_pilot":
                group_draw = proposal.draw_adapted(
                    candidates.take(local_rows),
                    candidate_target[local_rows],
                    n_draws=int(n_draws),
                    epsilon=epsilon,
                    seed=proposal_seed,
                    object_ids=absolute_rows,
                )
                group_candidate_target = candidate_target_tensor[
                    torch.as_tensor(
                        local_rows,
                        dtype=torch.long,
                        device=candidate_target_tensor.device,
                    )
                ]
                group_zero_weights = _reuse_adapted_reference_weights(
                    likelihood,
                    group_observed,
                    group_targets,
                    group_candidate_target,
                    group_draw,
                    center=center,
                    object_chunk=len(group_observed),
                    atom_chunk=atom_chunk,
                )
                flow_evaluations += len(group_draw.indices) * int(
                    (~group_draw.local_member).sum(axis=1).max()
                )
            else:
                group_draw = draw.take(local_rows).prefix(int(n_draws))
                group_zero_weights = zero_weights
            coalesced = group_draw.coalesce()
            unique_counts[global_rows] = coalesced.valid.sum(axis=1)
            first = torch.as_tensor(
                coalesced.first_position,
                dtype=torch.long,
                device=group_zero_weights.device,
            )
            if allocation_method == "independent_pilot":
                row_tensor = torch.arange(
                    len(local_rows), device=group_zero_weights.device
                )[:, None]
            else:
                row_tensor = torch.as_tensor(
                    local_rows,
                    dtype=torch.long,
                    device=group_zero_weights.device,
                )[:, None]
            zero_unique = group_zero_weights[row_tensor, first]
            log_likelihood = {
                "zero": (
                    _coalesced_logsumexp(zero_unique, coalesced, int(n_draws))
                    - normalizers["zero"]
                )
            }
            half_draws = int(n_draws) // 2
            half_log_likelihood = (
                {
                    "zero": (
                        _coalesced_logsumexp(
                            zero_unique, coalesced, half_draws
                        )
                        - normalizers["zero"]
                    )
                }
                if bias_correction == "richardson_1_over_m"
                else None
            )
            rung_log_likelihood = None
            if retain_full_ladder:
                rung_log_likelihood = {
                    ("zero", rung): (
                        _coalesced_logsumexp(zero_unique, coalesced, rung)
                        - normalizers["zero"]
                    )
                    for rung in ladder
                }
            for name in tuple(views)[1:]:
                weights = likelihood.log_importance_weights_tensor(
                    group_observed,
                    *views[name],
                    atom_indices=coalesced.indices,
                    proposal_probability=coalesced.probability,
                    observed_targets=group_targets,
                    object_chunk=len(group_observed),
                    atom_chunk=atom_chunk,
                )
                flow_evaluations += int(coalesced.indices.size)
                log_likelihood[name] = (
                    _coalesced_logsumexp(weights, coalesced, int(n_draws))
                    - normalizers[name]
                )
                if half_log_likelihood is not None:
                    half_log_likelihood[name] = (
                        _coalesced_logsumexp(weights, coalesced, half_draws)
                        - normalizers[name]
                    )
                if retain_full_ladder:
                    for rung in ladder:
                        rung_log_likelihood[(name, rung)] = (
                            _coalesced_logsumexp(weights, coalesced, rung)
                            - normalizers[name]
                        )
                del weights
            zero = log_likelihood["zero"]
            for component, plus, minus in (
                (0, "g1_plus", "g1_minus"),
                (1, "g2_plus", "g2_minus"),
            ):
                upper = log_likelihood[plus]
                lower = log_likelihood[minus]
                component_score = (upper - lower) / (2.0 * h)
                diagonal_tensor = -(upper - 2.0 * zero + lower) / h**2
                if half_log_likelihood is not None:
                    half_upper = half_log_likelihood[plus]
                    half_lower = half_log_likelihood[minus]
                    half_zero = half_log_likelihood["zero"]
                    component_score = 2.0 * component_score - (
                        (half_upper - half_lower) / (2.0 * h)
                    )
                    diagonal_tensor = 2.0 * diagonal_tensor - (
                        -(half_upper - 2.0 * half_zero + half_lower) / h**2
                    )
                score[global_rows, component] = (
                    component_score.detach().cpu().numpy()
                )
                diagonal = diagonal_tensor.detach().cpu().numpy()
                if full_information:
                    information[global_rows, component, component] = diagonal
                else:
                    information[global_rows, component] = diagonal
            if full_information:
                mixed_hessian = (
                    log_likelihood["pp"]
                    - log_likelihood["pm"]
                    - log_likelihood["mp"]
                    + log_likelihood["mm"]
                ) / (4.0 * h**2)
                mixed_information = (-mixed_hessian).detach().cpu().numpy()
                if half_log_likelihood is not None:
                    half_mixed_hessian = (
                        half_log_likelihood["pp"]
                        - half_log_likelihood["pm"]
                        - half_log_likelihood["mp"]
                        + half_log_likelihood["mm"]
                    ) / (4.0 * h**2)
                    mixed_information = (
                        -2.0 * mixed_hessian + half_mixed_hessian
                    ).detach().cpu().numpy()
                information[global_rows, 0, 1] = mixed_information
                information[global_rows, 1, 0] = mixed_information
            if retain_full_ladder:
                for rung_index, rung in enumerate(ladder):
                    rung_zero = rung_log_likelihood[("zero", rung)]
                    for component, plus, minus in (
                        (0, "g1_plus", "g1_minus"),
                        (1, "g2_plus", "g2_minus"),
                    ):
                        upper = rung_log_likelihood[(plus, rung)]
                        lower = rung_log_likelihood[(minus, rung)]
                        ladder_score[rung_index, global_rows, component] = (
                            (upper - lower) / (2.0 * h)
                        ).detach().cpu().numpy()
                        diagonal = (
                            -(upper - 2.0 * rung_zero + lower) / h**2
                        ).detach().cpu().numpy()
                        if full_information:
                            ladder_information[
                                rung_index, global_rows, component, component
                            ] = diagonal
                        else:
                            ladder_information[
                                rung_index, global_rows, component
                            ] = diagonal
                    if full_information:
                        rung_mixed_hessian = (
                            rung_log_likelihood[("pp", rung)]
                            - rung_log_likelihood[("pm", rung)]
                            - rung_log_likelihood[("mp", rung)]
                            + rung_log_likelihood[("mm", rung)]
                        ) / (4.0 * h**2)
                        rung_mixed_information = (
                            -rung_mixed_hessian
                        ).detach().cpu().numpy()
                        ladder_information[rung_index, global_rows, 0, 1] = (
                            rung_mixed_information
                        )
                        ladder_information[rung_index, global_rows, 1, 0] = (
                            rung_mixed_information
                        )
            del coalesced, group_draw, log_likelihood, zero_unique, first, row_tensor
            del half_log_likelihood
            if allocation_method == "independent_pilot":
                del group_candidate_target, group_zero_weights
            del rung_log_likelihood
        phase_seconds["stencil_views_and_reduction"] += (
            time.perf_counter() - phase_started
        )
        del candidates, candidate_target_tensor, candidate_target, draw, zero_weights

    return AdaptiveSection5Result(
        center=(float(center[0]), float(center[1])),
        h=float(h),
        draw_ladder=ladder,
        n_candidates=int(n_candidates),
        score=score,
        information=information,
        draw_counts=selected_counts,
        unique_counts=unique_counts,
        flow_evaluations=int(flow_evaluations),
        elapsed_seconds=float(time.perf_counter() - started),
        ladder_score=ladder_score,
        ladder_information=ladder_information,
        proposal_prefilter_candidates=(
            None
            if proposal_prefilter_candidates is None
            else int(proposal_prefilter_candidates)
        ),
        allocation_method=allocation_method,
        pilot_draws=(
            int(pilot_draws) if allocation_method == "independent_pilot" else None
        ),
        pilot_seed=(
            int(pilot_seed) if allocation_method == "independent_pilot" else None
        ),
        pilot_ess_fraction=pilot_ess_fraction,
        pilot_max_weight_fraction=pilot_maximum,
        phase_seconds=phase_seconds,
        bias_correction=bias_correction,
        candidate_backend=candidate_backend,
    )


def estimate_one_step_adaptive_section5(
    likelihood: CatalogueLikelihood,
    mock: MockCatalogue,
    proposal: DefensiveLocalProposal,
    *,
    center: Sequence[float],
    **kwargs,
) -> AdaptiveOneStepResult:
    """Take one full-2D catalogue-likelihood Newton step from ``center``.

    The centre can be a cheap statistic such as the raw mean measured shape.
    It is not the final estimator: the returned correction solves the catalogue
    score using the full numerical 2x2 information.  Adaptive proposal prefixes
    remain fixed across all nine stencil views.
    """

    center_array = np.asarray(center, dtype=np.float64)
    if center_array.shape != (2,) or not np.isfinite(center_array).all():
        raise ValueError("one-step center must be a finite two-vector")
    if "full_information" in kwargs:
        raise TypeError("full_information is fixed by the one-step estimator")
    moments = run_adaptive_section5(
        likelihood,
        mock,
        proposal,
        center=center_array,
        full_information=True,
        **kwargs,
    )
    return _summarize_one_step(center_array, moments)


def _summarize_one_step(
    center_array: np.ndarray,
    moments: AdaptiveSection5Result,
) -> AdaptiveOneStepResult:
    """Convert per-object score/information into one full-2D Newton step."""

    score_sum = moments.score.sum(axis=0, dtype=np.float64)
    information_sum = moments.information.sum(axis=0, dtype=np.float64)
    information_sum = 0.5 * (information_sum + information_sum.T)
    eigenvalues = np.linalg.eigvalsh(information_sum)
    if not np.isfinite(eigenvalues).all() or eigenvalues[0] <= 0:
        raise RuntimeError(
            "one-step catalogue information is not positive definite: "
            f"eigenvalues={eigenvalues.tolist()}"
        )
    step = np.linalg.solve(information_sum, score_sum)
    estimate = center_array + step

    residual = moments.score - np.einsum("nij,j->ni", moments.information, step)
    mean_information = information_sum / len(residual)
    influence = np.linalg.solve(mean_information, residual.T).T
    robust_covariance = np.cov(influence, rowvar=False, ddof=1) / len(residual)
    model_covariance = np.linalg.inv(information_sum)
    return AdaptiveOneStepResult(
        center=tuple(map(float, center_array)),
        estimate=tuple(map(float, estimate)),
        step=tuple(map(float, step)),
        information_eigenvalues=tuple(map(float, eigenvalues)),
        robust_covariance=tuple(
            tuple(map(float, row)) for row in robust_covariance
        ),
        model_covariance=tuple(tuple(map(float, row)) for row in model_covariance),
        robust_standard_error=tuple(
            map(float, np.sqrt(np.diag(robust_covariance)))
        ),
        model_standard_error=tuple(map(float, np.sqrt(np.diag(model_covariance)))),
        quadratic_log_likelihood_gain=0.5 * float(score_sum @ step),
        moments=moments,
    )


def run_stratified_section5(
    likelihood: CatalogueLikelihood,
    mock: MockCatalogue,
    proposal: DefensiveLocalProposal,
    *,
    center: Sequence[float] = (0.0, 0.0),
    h: float = 0.001,
    complement_draw_ladder: Sequence[int] = (128, 256, 512, 1024),
    n_candidates: int = 4096,
    proposal_prefilter_candidates: Optional[int] = 16384,
    proposal_seed: int = 8701,
    retain_full_ladder: bool = True,
    object_chunk: int = 16,
    atom_chunk: int = 4096,
) -> AdaptiveSection5Result:
    """Sum the candidate evidence exactly and sample only its complement.

    For each object and shear view this estimates the unchanged catalogue
    numerator as

    ``sum[j in C] pi_j Pdet_j L_j + mean[j~pi](Pdet_j L_j 1[j not in C])``.

    Candidate support is selected once at the expansion centre and every prior
    draw is fixed across all nine stencil views.  Unlike the self-normalized
    adapted proposal, already-known candidate atoms are never resampled.
    """

    if not likelihood.tensor_native_available:
        raise TypeError("stratified Section 5 requires a tensor-native Torch flow")
    if likelihood.selection is not None:
        raise ValueError("stratified Section 5 currently requires measured cuts disabled")
    if likelihood.cache.blend_response is not None:
        raise ValueError("stratified Section 5 currently requires external R_blend=0")
    likelihood.cache.validate_detection_shear_invariance()
    center = np.asarray(center, dtype=np.float64)
    ladder = tuple(sorted({int(value) for value in complement_draw_ladder}))
    if center.shape != (2,) or not np.isfinite(center).all():
        raise ValueError("center must be a finite two-vector")
    if not np.isfinite(h) or h <= 0:
        raise ValueError("h must be finite and positive")
    if not ladder or ladder[0] <= 0:
        raise ValueError("complement draw ladder must be positive")
    if n_candidates <= 0 or object_chunk <= 0 or atom_chunk <= 0:
        raise ValueError("candidate and chunk sizes must be positive")

    views = {
        "zero": (float(center[0]), float(center[1])),
        "g1_plus": (float(center[0] + h), float(center[1])),
        "g1_minus": (float(center[0] - h), float(center[1])),
        "g2_plus": (float(center[0]), float(center[1] + h)),
        "g2_minus": (float(center[0]), float(center[1] - h)),
        "pp": (float(center[0] + h), float(center[1] + h)),
        "pm": (float(center[0] + h), float(center[1] - h)),
        "mp": (float(center[0] - h), float(center[1] + h)),
        "mm": (float(center[0] - h), float(center[1] - h)),
    }
    normalizers = {
        name: _view_normalizer(likelihood, *view) for name, view in views.items()
    }
    n_objects = len(mock.measurements)
    score = np.empty((n_objects, 2), dtype=np.float64)
    information = np.empty((n_objects, 2, 2), dtype=np.float64)
    ladder_score = np.empty((len(ladder), n_objects, 2), dtype=np.float64)
    ladder_information = np.empty(
        (len(ladder), n_objects, 2, 2), dtype=np.float64
    )
    unique_counts = np.empty(n_objects, dtype=np.int64)
    flow_evaluations = 0
    started = time.perf_counter()

    for object_start in range(0, n_objects, object_chunk):
        object_stop = min(object_start + object_chunk, n_objects)
        observed = mock.measurements.iloc[object_start:object_stop].reset_index(drop=True)
        observed_targets = likelihood.observed_target_tensor(observed)
        candidates = proposal.candidates(
            observed,
            n_candidates=n_candidates,
            prefilter_candidates=proposal_prefilter_candidates,
        )
        global_draw = proposal.draw_global(
            candidates,
            n_draws=ladder[-1],
            seed=proposal_seed,
            object_offset=object_start,
        )
        unique_counts[object_start:object_stop] = np.asarray(
            [len(np.unique(row)) for row in global_draw.indices], dtype=np.int64
        )
        membership = torch.as_tensor(
            global_draw.local_member,
            dtype=torch.bool,
            device=observed_targets.device,
        )
        log_likelihood = {}
        for name, view in views.items():
            candidate_weight = likelihood.log_importance_weights_tensor(
                observed,
                *view,
                atom_indices=candidates.indices,
                proposal_probability=np.ones_like(
                    candidates.indices, dtype=np.float64
                ),
                observed_targets=observed_targets,
                object_chunk=len(observed),
                atom_chunk=atom_chunk,
            )
            global_weight = likelihood.log_importance_weights_tensor(
                observed,
                *view,
                atom_indices=global_draw.indices,
                proposal_probability=global_draw.probability,
                observed_targets=observed_targets,
                object_chunk=len(observed),
                atom_chunk=atom_chunk,
            ).masked_fill(membership, -torch.inf)
            flow_evaluations += int(candidates.indices.size + global_draw.indices.size)
            candidate_sum = torch.logsumexp(candidate_weight, dim=1)
            for rung in ladder:
                complement_mean = (
                    torch.logsumexp(global_weight[:, :rung], dim=1)
                    - np.log(rung)
                )
                log_likelihood[(name, rung)] = (
                    torch.logaddexp(candidate_sum, complement_mean)
                    - normalizers[name]
                )
            del candidate_weight, global_weight, candidate_sum

        rows = slice(object_start, object_stop)
        for rung_index, rung in enumerate(ladder):
            zero = log_likelihood[("zero", rung)]
            for component, plus, minus in (
                (0, "g1_plus", "g1_minus"),
                (1, "g2_plus", "g2_minus"),
            ):
                upper = log_likelihood[(plus, rung)]
                lower = log_likelihood[(minus, rung)]
                ladder_score[rung_index, rows, component] = (
                    (upper - lower) / (2.0 * h)
                ).detach().cpu().numpy()
                ladder_information[rung_index, rows, component, component] = (
                    -(upper - 2.0 * zero + lower) / h**2
                ).detach().cpu().numpy()
            mixed = -(
                log_likelihood[("pp", rung)]
                - log_likelihood[("pm", rung)]
                - log_likelihood[("mp", rung)]
                + log_likelihood[("mm", rung)]
            ) / (4.0 * h**2)
            mixed = mixed.detach().cpu().numpy()
            ladder_information[rung_index, rows, 0, 1] = mixed
            ladder_information[rung_index, rows, 1, 0] = mixed
        del observed_targets, candidates, global_draw, membership, log_likelihood

    score[:] = ladder_score[-1]
    information[:] = ladder_information[-1]
    return AdaptiveSection5Result(
        center=(float(center[0]), float(center[1])),
        h=float(h),
        draw_ladder=ladder,
        n_candidates=int(n_candidates),
        score=score,
        information=information,
        draw_counts=np.full(n_objects, ladder[-1], dtype=np.int64),
        unique_counts=unique_counts,
        flow_evaluations=int(flow_evaluations),
        elapsed_seconds=float(time.perf_counter() - started),
        ladder_score=ladder_score if retain_full_ladder else None,
        ladder_information=ladder_information if retain_full_ladder else None,
        proposal_prefilter_candidates=proposal_prefilter_candidates,
    )


def estimate_one_step_stratified_section5(
    likelihood: CatalogueLikelihood,
    mock: MockCatalogue,
    proposal: DefensiveLocalProposal,
    *,
    center: Sequence[float],
    **kwargs,
) -> AdaptiveOneStepResult:
    """Take one full-2D step using exact candidates plus complement sampling."""

    center_array = np.asarray(center, dtype=np.float64)
    moments = run_stratified_section5(
        likelihood, mock, proposal, center=center_array, **kwargs
    )
    return _summarize_one_step(center_array, moments)


def autograd_importance_section5(
    likelihood: CatalogueLikelihood,
    mock: MockCatalogue,
    proposal: DefensiveLocalProposal,
    *,
    center: Sequence[float] = (0.0, 0.0),
    draw_ladder: Sequence[int] = (2048, 4096, 8192, 16384),
    n_candidates: int = 16384,
    epsilon: float = 0.1,
    proposal_seed: int = 8701,
    min_ess: float = 256.0,
    max_weight_fraction: float = 0.25,
    object_chunk: int = 8,
    atom_chunk: int = 4096,
) -> ImportanceAutogradResult:
    """Differentiate an adaptively sampled catalogue likelihood in one view.

    A separate two-component shear variable is assigned to every observed
    object.  Because object evidences are independent, differentiating the sum
    yields all per-object gradients and all diagonal 2x2 Hessian blocks in two
    reverse passes, including the off-diagonal information.
    """

    if not likelihood.tensor_native_available:
        raise TypeError("importance autograd requires a tensor-native Torch flow")
    if likelihood.selection is not None or likelihood.cache.blend_response is not None:
        raise ValueError("importance autograd requires no measured cuts and R_blend=0")
    likelihood.cache.validate_detection_shear_invariance()
    center = np.asarray(center, dtype=np.float64)
    ladder = tuple(sorted({int(value) for value in draw_ladder}))
    if center.shape != (2,) or not np.isfinite(center).all():
        raise ValueError("center must be a finite two-vector")
    if not ladder or ladder[0] <= 0:
        raise ValueError("autograd draw ladder must be positive")
    if object_chunk <= 0 or atom_chunk <= 0 or n_candidates <= 0:
        raise ValueError("candidate and chunk sizes must be positive")

    bundle = likelihood.flow_model
    features = tuple(bundle.condition_preprocessor.feature_names)
    if "e1_input_p" not in features or "e2_input_p" not in features:
        raise ValueError("flow conditions lack intrinsic ellipticity")
    first = features.index("e1_input_p")
    second = features.index("e2_input_p")
    raw_zero = likelihood.raw_zero_context_tensor()
    log_mass = likelihood._tensor_view(0.0, 0.0).log_detected_mass
    n_objects = len(mock.measurements)
    score = np.empty((n_objects, 2), dtype=np.float64)
    information = np.empty((n_objects, 2, 2), dtype=np.float64)
    selected_counts = np.empty(n_objects, dtype=np.int64)
    unique_counts = np.empty(n_objects, dtype=np.int64)
    flow_evaluations = 0
    started = time.perf_counter()

    for object_start in range(0, n_objects, object_chunk):
        object_stop = min(object_start + object_chunk, n_objects)
        observed = mock.measurements.iloc[object_start:object_stop].reset_index(drop=True)
        observed_targets = likelihood.observed_target_tensor(observed)
        candidates = proposal.candidates(observed, n_candidates=n_candidates)
        candidate_target_tensor = likelihood.log_importance_weights_tensor(
            observed,
            float(center[0]),
            float(center[1]),
            atom_indices=candidates.indices,
            proposal_probability=np.ones_like(candidates.indices, dtype=np.float64),
            observed_targets=observed_targets,
            object_chunk=len(observed),
            atom_chunk=atom_chunk,
        )
        flow_evaluations += int(candidates.indices.size)
        draw = proposal.draw_adapted(
            candidates,
            candidate_target_tensor.detach().cpu().numpy().astype(np.float64),
            n_draws=ladder[-1],
            epsilon=epsilon,
            seed=proposal_seed,
            object_offset=object_start,
        )
        center_weights = _reuse_adapted_reference_weights(
            likelihood,
            observed,
            observed_targets,
            candidate_target_tensor,
            draw,
            center=center,
            object_chunk=len(observed),
            atom_chunk=atom_chunk,
        )
        flow_evaluations += len(draw.indices) * int(
            (~draw.local_member).sum(axis=1).max()
        )
        counts = select_adaptive_draw_counts(
            center_weights.detach().cpu().numpy(),
            ladder,
            min_ess=min_ess,
            max_weight_fraction=max_weight_fraction,
        )
        selected_counts[object_start:object_stop] = counts

        for n_draws in np.unique(counts):
            local_rows = np.flatnonzero(counts == n_draws)
            group_draw = draw.take(local_rows).prefix(int(n_draws))
            coalesced = group_draw.coalesce()
            global_rows = object_start + local_rows
            unique_counts[global_rows] = coalesced.valid.sum(axis=1)
            selected = torch.as_tensor(
                np.ascontiguousarray(coalesced.indices),
                dtype=torch.long,
                device=raw_zero.device,
            )
            raw = raw_zero.index_select(0, selected.reshape(-1)).reshape(
                len(local_rows), coalesced.indices.shape[1], -1
            ).clone()
            shear = torch.as_tensor(
                np.broadcast_to(center, (len(local_rows), 2)).copy(),
                dtype=raw.dtype,
                device=raw.device,
            ).requires_grad_(True)
            intrinsic_e1 = raw[:, :, first].clone()
            intrinsic_e2 = raw[:, :, second].clone()
            e1, e2 = _torch_shear_ellipticity(
                intrinsic_e1,
                intrinsic_e2,
                shear[:, 0, None],
                shear[:, 1, None],
            )
            raw[:, :, first] = e1
            raw[:, :, second] = e2
            context = bundle.condition_preprocessor.transform_tensor(
                raw.reshape(-1, raw.shape[-1])
            )
            target_rows = torch.as_tensor(
                local_rows, dtype=torch.long, device=observed_targets.device
            )
            targets = observed_targets.index_select(0, target_rows)
            targets = targets[:, None, :].expand(
                len(local_rows), coalesced.indices.shape[1], len(likelihood.target_names)
            ).reshape(-1, len(likelihood.target_names))
            log_flow = bundle.log_prob_tensor(
                targets, context, batch_size=65536
            ).reshape(len(local_rows), coalesced.indices.shape[1])
            proposal_probability = torch.as_tensor(
                coalesced.probability,
                dtype=log_flow.dtype,
                device=log_flow.device,
            )
            unique_log_weight = (
                log_flow
                + log_mass.index_select(0, selected.reshape(-1)).reshape_as(log_flow)
                - torch.log(proposal_probability)
            )
            log_evidence = _coalesced_logsumexp(
                unique_log_weight, coalesced, int(n_draws)
            )
            gradient = torch.autograd.grad(
                log_evidence.sum(), shear, create_graph=True
            )[0]
            hessian_rows = []
            for component in range(2):
                hessian_rows.append(
                    torch.autograd.grad(
                        gradient[:, component].sum(),
                        shear,
                        retain_graph=component == 0,
                    )[0]
                )
            score[global_rows] = gradient.detach().cpu().numpy()
            information[global_rows] = -torch.stack(
                hessian_rows, dim=1
            ).detach().cpu().numpy()
            flow_evaluations += int(coalesced.indices.size)
            del raw, shear, context, targets, log_flow, unique_log_weight, log_evidence
        del candidates, candidate_target_tensor, draw, center_weights

    return ImportanceAutogradResult(
        center=(float(center[0]), float(center[1])),
        score=score,
        information=information,
        draw_counts=selected_counts,
        unique_counts=unique_counts,
        flow_evaluations=int(flow_evaluations),
        elapsed_seconds=float(time.perf_counter() - started),
    )


def optimize_shear_autograd(
    likelihood: CatalogueLikelihood,
    mock: MockCatalogue,
    proposal: DefensiveLocalProposal,
    *,
    initial: Sequence[float] = (0.0, 0.0),
    max_iterations: int = 5,
    tolerance: float = 1e-4,
    max_step: float = 0.03,
    **kwargs,
) -> AutogradShearOptimizationResult:
    """Iteratively recenter the full two-component sampled likelihood."""

    center = np.asarray(initial, dtype=np.float64)
    if center.shape != (2,) or not np.isfinite(center).all():
        raise ValueError("initial shear must be a finite two-vector")
    if max_iterations <= 0 or tolerance <= 0 or max_step <= 0:
        raise ValueError("optimizer controls must be positive")
    history = []
    converged = False
    for _ in range(int(max_iterations)):
        result = autograd_importance_section5(
            likelihood, mock, proposal, center=center, **kwargs
        )
        history.append(result)
        score_sum = result.score.sum(axis=0)
        information_sum = result.information.sum(axis=0)
        step = np.linalg.solve(information_sum, score_sum)
        scale = max(1.0, float(np.max(np.abs(step))) / float(max_step))
        step = step / scale
        center = center + step
        if float(np.max(np.abs(step))) < tolerance:
            converged = True
            break
    return AutogradShearOptimizationResult(
        estimate=(float(center[0]), float(center[1])),
        converged=converged,
        iterations=tuple(history),
    )


def evaluate_fixed_draw_log_likelihood(
    likelihood: CatalogueLikelihood,
    mock: MockCatalogue,
    draw: ProposalDraw,
    shears: Sequence[Sequence[float]],
    *,
    object_chunk: int = 16,
    atom_chunk: int = 4096,
) -> Mapping[tuple[float, float], float]:
    """Evaluate the same importance atoms at arbitrary two-component shears.

    This is deliberately the same detected-and-selected catalogue likelihood
    as the exact path.  Only the expansion centre changes.  An external fixed
    ``R_blend`` shifts each atom's measured-shape likelihood, while a declared
    measured cut changes only the shared population normalization for retained
    observations.
    """

    likelihood.cache.validate_detection_shear_invariance()
    points = tuple(
        dict.fromkeys(
            (float(value[0]), float(value[1]))
            for value in shears
        )
    )
    if not points or any(len(value) != 2 for value in shears):
        raise ValueError("shears must contain at least one two-component point")
    if not np.isfinite(np.asarray(points, dtype=np.float64)).all():
        raise ValueError("shear points must be finite")
    if object_chunk <= 0 or atom_chunk <= 0:
        raise ValueError("chunk sizes must be positive")
    if draw.indices.ndim != 2 or draw.indices.shape[0] != len(mock.measurements):
        raise ValueError("fixed proposal draw must contain one row per mock object")

    log_draws = float(np.log(draw.indices.shape[1]))
    sums = {point: 0.0 for point in points}

    for start in range(0, len(mock.measurements), object_chunk):
        stop = min(start + object_chunk, len(mock.measurements))
        observed = mock.measurements.iloc[start:stop].reset_index(drop=True)
        indices = np.ascontiguousarray(draw.indices[start:stop])
        probability = np.ascontiguousarray(draw.probability[start:stop])
        observed_targets = (
            likelihood.observed_target_tensor(observed)
            if likelihood.tensor_native_available
            else None
        )
        for point in points:
            if likelihood.tensor_native_available:
                weights = likelihood.log_importance_weights_tensor(
                    observed,
                    point[0],
                    point[1],
                    atom_indices=indices,
                    proposal_probability=probability,
                    observed_targets=observed_targets,
                    object_chunk=len(observed),
                    atom_chunk=atom_chunk,
                )
                # The population objective sums thousands of per-object log
                # evidences, while recentering can compare trial points that
                # differ by only ~1e-4 in total log likelihood.  Reducing the
                # importance weights in float32 creates a visibly corrugated
                # line-search surface at that scale even though the flow
                # itself is evaluated in float32.  Promote only the cheap
                # log-sum-exp reduction; this leaves the likelihood target and
                # flow evaluations unchanged.
                evidence = torch.logsumexp(weights.double(), dim=1) - log_draws
                sums[point] += float(evidence.sum().item())
                del weights, evidence
            else:
                weights = likelihood.log_importance_weights(
                    observed,
                    point[0],
                    point[1],
                    atom_indices=indices,
                    proposal_probability=probability,
                    object_chunk=len(observed),
                    atom_chunk=atom_chunk,
                )
                evidence = logsumexp(weights, axis=1) - log_draws
                sums[point] += float(evidence.sum())
                del weights, evidence
        del observed, indices, probability, observed_targets

    return {
        point: value
        - len(mock.measurements)
        * likelihood.log_population_normalization(point[0], point[1])
        for point, value in sums.items()
    }


def evaluate_fixed_draw_importance_diagnostics(
    likelihood: CatalogueLikelihood,
    mock: MockCatalogue,
    draw: ProposalDraw,
    shear: Sequence[float],
    *,
    object_chunk: int = 16,
    atom_chunk: int = 4096,
) -> StreamedImportanceDiagnostics:
    """Summarize final-shear importance concentration without retaining weights."""

    shear = tuple(shear)
    if len(shear) != 2:
        raise ValueError("diagnostic shear must be a finite two-vector")
    point = (float(shear[0]), float(shear[1]))
    if not np.isfinite(point).all():
        raise ValueError("diagnostic shear must be a finite two-vector")
    if object_chunk <= 0 or atom_chunk <= 0:
        raise ValueError("chunk sizes must be positive")
    if draw.indices.ndim != 2 or draw.indices.shape[0] != len(mock.measurements):
        raise ValueError("fixed proposal draw must contain one row per mock object")

    likelihood.cache.validate_detection_shear_invariance()
    parts = []
    for start in range(0, len(mock.measurements), object_chunk):
        stop = min(start + object_chunk, len(mock.measurements))
        observed = mock.measurements.iloc[start:stop].reset_index(drop=True)
        indices = np.ascontiguousarray(draw.indices[start:stop])
        probability = np.ascontiguousarray(draw.probability[start:stop])
        chunk_draw = ProposalDraw(
            indices=indices,
            probability=probability,
            local_member=np.ascontiguousarray(draw.local_member[start:stop]),
            global_component=np.ascontiguousarray(
                draw.global_component[start:stop]
            ),
            candidate_radius=np.ascontiguousarray(draw.candidate_radius[start:stop]),
            seed=draw.seed,
            local_position=np.ascontiguousarray(draw.local_position[start:stop]),
        )
        if likelihood.tensor_native_available:
            observed_targets = likelihood.observed_target_tensor(observed)
            weights = likelihood.log_importance_weights_tensor(
                observed,
                point[0],
                point[1],
                atom_indices=indices,
                proposal_probability=probability,
                observed_targets=observed_targets,
                object_chunk=len(observed),
                atom_chunk=atom_chunk,
            )
            parts.append(_diagnostic_tensors(weights, chunk_draw))
            del observed_targets, weights
        else:
            weights = likelihood.log_importance_weights(
                observed,
                point[0],
                point[1],
                atom_indices=indices,
                proposal_probability=probability,
                object_chunk=len(observed),
                atom_chunk=atom_chunk,
            )
            parts.append(_diagnostic_arrays(weights, chunk_draw))
            del weights
        del observed, indices, probability, chunk_draw
    diagnostics = _summarize_importance(parts, draw.indices.shape[1])
    if diagnostics is None:
        raise RuntimeError("importance diagnostics received no mock objects")
    return diagnostics


def _concatenate_proposal_draws(draws: Sequence[ProposalDraw]) -> ProposalDraw:
    """Join object-chunk draws without changing any object's random stream."""

    draws = tuple(draws)
    if not draws:
        raise ValueError("at least one proposal draw chunk is required")
    seeds = {int(draw.seed) for draw in draws}
    widths = {int(draw.indices.shape[1]) for draw in draws}
    if len(seeds) != 1 or len(widths) != 1:
        raise ValueError("proposal draw chunks must share their seed and width")
    return ProposalDraw(
        indices=np.concatenate([draw.indices for draw in draws], axis=0),
        probability=np.concatenate([draw.probability for draw in draws], axis=0),
        local_member=np.concatenate([draw.local_member for draw in draws], axis=0),
        global_component=np.concatenate(
            [draw.global_component for draw in draws], axis=0
        ),
        candidate_radius=np.concatenate(
            [draw.candidate_radius for draw in draws], axis=0
        ),
        seed=draws[0].seed,
        local_position=np.concatenate(
            [draw.local_position for draw in draws], axis=0
        ),
    )


def _draw_initial_center_posterior_adapted(
    likelihood: CatalogueLikelihood,
    mock: MockCatalogue,
    proposal: DefensiveLocalProposal,
    *,
    reference: Sequence[float],
    n_draws: int,
    n_candidates: int,
    epsilon: float,
    proposal_seed: int,
    object_chunk: int,
    atom_chunk: int,
) -> tuple[ProposalDraw, Optional[float], int, int]:
    """Build one exact initial-centre proposal and optionally reuse its density.

    The local proposal is proportional to the exact unnormalised numerator
    ``pi * Pdet * L(reference)`` on the deterministic candidate support.  It is
    mixed with the full catalogue prior and retains the exact ``pi/q``
    correction.  The resulting draw is assembled once and is never adapted
    again during recentering.

    Tensor-native likelihoods also reuse the candidate likelihoods to evaluate
    the reference point, calling the flow only for defensive global draws that
    fall outside the candidate support.  The dataframe fallback keeps the same
    adapted proposal but lets the ordinary evaluator recompute the reference.
    """

    reference = (float(reference[0]), float(reference[1]))
    log_draws = float(np.log(n_draws))
    draw_chunks = []
    candidate_evaluations = 0
    reuse_evaluations = 0
    reference_log_sum = 0.0
    reused_reference = bool(likelihood.tensor_native_available)

    for start in range(0, len(mock.measurements), object_chunk):
        stop = min(start + object_chunk, len(mock.measurements))
        observed = mock.measurements.iloc[start:stop].reset_index(drop=True)
        candidates = proposal.candidates(observed, n_candidates=n_candidates)
        ones = np.ones_like(candidates.indices, dtype=np.float64)
        if likelihood.tensor_native_available:
            observed_targets = likelihood.observed_target_tensor(observed)
            candidate_target_tensor = likelihood.log_importance_weights_tensor(
                observed,
                reference[0],
                reference[1],
                atom_indices=candidates.indices,
                proposal_probability=ones,
                observed_targets=observed_targets,
                object_chunk=len(observed),
                atom_chunk=atom_chunk,
            )
            candidate_target = (
                candidate_target_tensor.detach().cpu().numpy().astype(np.float64)
            )
        else:
            observed_targets = None
            candidate_target_tensor = None
            candidate_target = likelihood.log_importance_weights(
                observed,
                reference[0],
                reference[1],
                atom_indices=candidates.indices,
                proposal_probability=ones,
                object_chunk=len(observed),
                atom_chunk=atom_chunk,
            )
        candidate_evaluations += int(candidates.indices.size)
        chunk_draw = proposal.draw_adapted(
            candidates,
            candidate_target,
            n_draws=n_draws,
            epsilon=epsilon,
            seed=proposal_seed,
            object_offset=start,
        )
        draw_chunks.append(chunk_draw)

        if reused_reference:
            reference_weights = _reuse_adapted_reference_weights(
                likelihood,
                observed,
                observed_targets,
                candidate_target_tensor,
                chunk_draw,
                center=reference,
                object_chunk=len(observed),
                atom_chunk=atom_chunk,
            )
            outside_width = int((~chunk_draw.local_member).sum(axis=1).max())
            reuse_evaluations += len(observed) * outside_width
            reference_evidence = (
                torch.logsumexp(reference_weights.double(), dim=1) - log_draws
            )
            reference_log_sum += float(reference_evidence.sum().item())
            del reference_weights, reference_evidence

        del (
            observed,
            candidates,
            ones,
            observed_targets,
            candidate_target_tensor,
            candidate_target,
            chunk_draw,
        )

    draw = _concatenate_proposal_draws(draw_chunks)
    reference_value = None
    if reused_reference:
        reference_value = reference_log_sum - len(mock.measurements) * (
            likelihood.log_population_normalization(*reference)
        )
    return draw, reference_value, candidate_evaluations, reuse_evaluations


def _numerical_stencil(
    evaluate: Callable[
        [Sequence[tuple[float, float]]], Mapping[tuple[float, float], float]
    ],
    center: np.ndarray,
    h: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return population log likelihood, score, and full observed information."""

    x, y = map(float, center)
    points = (
        (x, y),
        (x + h, y),
        (x - h, y),
        (x, y + h),
        (x, y - h),
        (x + h, y + h),
        (x + h, y - h),
        (x - h, y + h),
        (x - h, y - h),
    )
    values = evaluate(points)
    zero, xp, xm, yp, ym, pp, pm, mp, mm = (
        float(values[point]) for point in points
    )
    score = np.array(
        [(xp - xm) / (2.0 * h), (yp - ym) / (2.0 * h)],
        dtype=np.float64,
    )
    information = np.array(
        [
            [-(xp - 2.0 * zero + xm) / h**2, 0.0],
            [0.0, -(yp - 2.0 * zero + ym) / h**2],
        ],
        dtype=np.float64,
    )
    cross_hessian = (pp - pm - mp + mm) / (4.0 * h**2)
    information[0, 1] = information[1, 0] = -cross_hessian
    return zero, score, information


def _safeguarded_numerical_recenter(
    evaluate: Callable[
        [Sequence[tuple[float, float]]], Mapping[tuple[float, float], float]
    ],
    *,
    initial: Sequence[float],
    h: float,
    max_iterations: int,
    tolerance: float,
    max_step: float,
    shear_bound: float,
    max_backtracks: int,
) -> tuple[tuple[float, float], bool, str, tuple[NumericalShearIteration, ...]]:
    """Safeguarded Newton iteration for a deterministic likelihood evaluator."""

    center = np.asarray(initial, dtype=np.float64)
    if center.shape != (2,) or not np.isfinite(center).all():
        raise ValueError("initial shear must be a finite two-vector")
    controls = (h, tolerance, max_step, shear_bound)
    if not np.isfinite(controls).all() or any(value <= 0 for value in controls):
        raise ValueError("optimizer controls must be finite and positive")
    if max_iterations <= 0 or max_backtracks <= 0:
        raise ValueError("iteration counts must be positive")
    if np.max(np.abs(center)) > shear_bound:
        raise ValueError("initial shear lies outside the optimizer guard bound")

    history = []
    converged = False
    reason = "maximum_iterations"
    for _ in range(int(max_iterations)):
        value, score, information = _numerical_stencil(evaluate, center, h)
        eigenvalues = np.linalg.eigvalsh(information)
        positive = bool(eigenvalues[0] > 0)
        raw_newton = None
        if positive:
            try:
                raw_newton = np.linalg.solve(information, score)
            except np.linalg.LinAlgError:
                raw_newton = None

        if raw_newton is not None and np.max(np.abs(raw_newton)) <= tolerance:
            history.append(
                NumericalShearIteration(
                    center=tuple(map(float, center)),
                    log_likelihood_sum=float(value),
                    score=tuple(map(float, score)),
                    information=tuple(tuple(map(float, row)) for row in information),
                    information_eigenvalues=tuple(map(float, eigenvalues)),
                    step_method="converged_newton",
                    raw_step=tuple(map(float, raw_newton)),
                    accepted_step=(0.0, 0.0),
                    next_center=tuple(map(float, center)),
                    next_log_likelihood_sum=float(value),
                    accepted=False,
                )
            )
            converged = True
            reason = "newton_step_below_tolerance"
            break

        gradient_scale = float(np.max(np.abs(score)))
        gradient_step = (
            score * (max_step / gradient_scale)
            if gradient_scale > 0
            else np.zeros(2, dtype=np.float64)
        )
        directions = []
        if raw_newton is not None and float(score @ raw_newton) > 0:
            newton_scale = max(1.0, float(np.max(np.abs(raw_newton))) / max_step)
            directions.append(("newton", raw_newton / newton_scale, raw_newton))
        directions.append(("gradient", gradient_step, gradient_step))

        accepted = False
        accepted_method = "stalled"
        accepted_step = np.zeros(2, dtype=np.float64)
        accepted_center = center.copy()
        accepted_value = float(value)
        raw_step = directions[0][2]
        for method, direction, unscaled in directions:
            raw_step = unscaled
            for backtrack in range(int(max_backtracks)):
                step = direction * (0.5**backtrack)
                candidate = center + step
                if np.max(np.abs(candidate)) > shear_bound:
                    continue
                point = tuple(map(float, candidate))
                trial = float(evaluate((point,))[point])
                if trial > value + 1.0e-10:
                    accepted = True
                    accepted_method = method
                    accepted_step = step
                    accepted_center = candidate
                    accepted_value = trial
                    break
            if accepted:
                break

        history.append(
            NumericalShearIteration(
                center=tuple(map(float, center)),
                log_likelihood_sum=float(value),
                score=tuple(map(float, score)),
                information=tuple(tuple(map(float, row)) for row in information),
                information_eigenvalues=tuple(map(float, eigenvalues)),
                step_method=accepted_method,
                raw_step=tuple(map(float, raw_step)),
                accepted_step=tuple(map(float, accepted_step)),
                next_center=tuple(map(float, accepted_center)),
                next_log_likelihood_sum=float(accepted_value),
                accepted=accepted,
            )
        )
        if not accepted:
            reason = "no_likelihood_increasing_step"
            break
        center = accepted_center

    return tuple(map(float, center)), converged, reason, tuple(history)


def optimize_shear_numerical(
    likelihood: CatalogueLikelihood,
    mock: MockCatalogue,
    proposal: DefensiveLocalProposal,
    *,
    initial: Sequence[float] = (0.0, 0.0),
    h: float = 0.001,
    n_draws: int = 16384,
    n_candidates: int = 32768,
    epsilon: float = 0.1,
    bandwidth: float = 1.0,
    proposal_seed: int = 8701,
    proposal_method: str = "initial_center_posterior_adapted",
    max_iterations: int = 10,
    tolerance: float = 1e-4,
    max_step: float = 0.01,
    shear_bound: float = 0.1,
    max_backtracks: int = 8,
    object_chunk: int = 16,
    atom_chunk: int = 4096,
) -> NumericalShearOptimizationResult:
    """Maximize the fixed catalogue likelihood by local numerical recentering.

    Proposal atoms and probabilities are drawn once and reused for every
    stencil and line-search point.  This preserves common random numbers and
    the exact importance correction throughout the optimization.
    """

    if n_draws <= 0 or n_candidates <= 0:
        raise ValueError("draw and candidate counts must be positive")
    if not (0 < epsilon <= 1):
        raise ValueError("proposal epsilon must satisfy 0 < epsilon <= 1")
    proposal_method = str(proposal_method)
    if proposal_method not in {
        "initial_center_posterior_adapted",
        "distance_kernel",
    }:
        raise ValueError(
            "proposal_method must be initial_center_posterior_adapted or distance_kernel"
        )
    if proposal_method == "distance_kernel" and bandwidth <= 0:
        raise ValueError("distance-kernel bandwidth must be positive")
    initial_array = np.asarray(initial, dtype=np.float64)
    if initial_array.shape != (2,) or not np.isfinite(initial_array).all():
        raise ValueError("initial shear must be a finite two-vector")
    started = time.perf_counter()
    selection_initial_shears = (
        set(likelihood.selection.available_shears)
        if likelihood.selection is not None
        and hasattr(likelihood.selection, "available_shears")
        else set()
    )
    proposal_candidate_evaluations = 0
    proposal_reuse_evaluations = 0
    reference_value = None
    if proposal_method == "initial_center_posterior_adapted":
        (
            draw,
            reference_value,
            proposal_candidate_evaluations,
            proposal_reuse_evaluations,
        ) = _draw_initial_center_posterior_adapted(
            likelihood,
            mock,
            proposal,
            reference=initial_array,
            n_draws=int(n_draws),
            n_candidates=int(n_candidates),
            epsilon=float(epsilon),
            proposal_seed=int(proposal_seed),
            object_chunk=int(object_chunk),
            atom_chunk=int(atom_chunk),
        )
        result_bandwidth = None
        proposal_reference = tuple(map(float, initial_array))
    else:
        draw = proposal.draw(
            mock.measurements,
            n_draws=int(n_draws),
            n_candidates=int(n_candidates),
            epsilon=float(epsilon),
            bandwidth=float(bandwidth),
            seed=int(proposal_seed),
        )
        result_bandwidth = float(bandwidth)
        proposal_reference = None

    initial_key = tuple(map(float, initial_array))
    cache: dict[tuple[float, float], float] = {}
    evaluation_order: list[tuple[float, float]] = []
    general_evaluation_order: list[tuple[float, float]] = []
    if reference_value is not None:
        cache[initial_key] = float(reference_value)
        evaluation_order.append(initial_key)
        if initial_key != (0.0, 0.0):
            likelihood.discard_tensor_views((initial_key,))
            likelihood.cache.discard_views((initial_key,))

    def evaluate(
        points: Sequence[tuple[float, float]],
    ) -> Mapping[tuple[float, float], float]:
        normalized = tuple(
            dict.fromkeys((float(point[0]), float(point[1])) for point in points)
        )
        missing = tuple(point for point in normalized if point not in cache)
        if missing:
            values = evaluate_fixed_draw_log_likelihood(
                likelihood,
                mock,
                draw,
                missing,
                object_chunk=object_chunk,
                atom_chunk=atom_chunk,
            )
            for point in missing:
                cache[point] = float(values[point])
                evaluation_order.append(point)
                general_evaluation_order.append(point)
            # With an external response the general tensor path materializes
            # one full catalogue view per arbitrary shear.  The scalar value
            # above is deterministic and cached, so retain only the zero base;
            # measured-selection probabilities live in their own compact
            # cache and are unaffected by releasing these frames/tensors.
            releasable = tuple(point for point in missing if point != (0.0, 0.0))
            if releasable:
                likelihood.discard_tensor_views(releasable)
                likelihood.cache.discard_views(releasable)
        return {point: cache[point] for point in normalized}

    estimate, converged, reason, iterations = _safeguarded_numerical_recenter(
        evaluate,
        initial=initial_array,
        h=float(h),
        max_iterations=int(max_iterations),
        tolerance=float(tolerance),
        max_step=float(max_step),
        shear_bound=float(shear_bound),
        max_backtracks=int(max_backtracks),
    )
    evaluations = tuple(
        NumericalLikelihoodPoint(point, cache[point]) for point in evaluation_order
    )
    numerator_evaluations = (
        len(mock.measurements)
        * int(n_draws)
        * len(general_evaluation_order)
    )
    proposal_evaluations = (
        int(proposal_candidate_evaluations) + int(proposal_reuse_evaluations)
    )
    selection_evaluations = 0
    if likelihood.selection is not None and hasattr(likelihood.selection, "n_samples"):
        new_selection_shears = (
            set(likelihood.selection.available_shears) - selection_initial_shears
        )
        n_active = int(np.count_nonzero(likelihood.cache.prior.weights > 0))
        selection_evaluations = (
            len(new_selection_shears)
            * n_active
            * int(likelihood.selection.n_samples)
        )
    importance = evaluate_fixed_draw_importance_diagnostics(
        likelihood,
        mock,
        draw,
        estimate,
        object_chunk=object_chunk,
        atom_chunk=atom_chunk,
    )
    diagnostic_evaluations = len(mock.measurements) * int(n_draws)
    if estimate != (0.0, 0.0):
        likelihood.discard_tensor_views((estimate,))
        likelihood.cache.discard_views((estimate,))
    return NumericalShearOptimizationResult(
        estimate=estimate,
        converged=converged,
        reason=reason,
        h=float(h),
        n_objects=len(mock.measurements),
        n_draws=int(n_draws),
        n_candidates=int(n_candidates),
        proposal_method=proposal_method,
        proposal_reference_shear=proposal_reference,
        proposal_seed=int(proposal_seed),
        epsilon=float(epsilon),
        bandwidth=result_bandwidth,
        initial_likelihood_reused=reference_value is not None,
        proposal_candidate_flow_evaluations=int(
            proposal_candidate_evaluations
        ),
        proposal_reuse_flow_evaluations=int(proposal_reuse_evaluations),
        proposal_flow_evaluations=int(proposal_evaluations),
        numerator_flow_evaluations=int(numerator_evaluations),
        selection_flow_evaluations=int(selection_evaluations),
        diagnostic_flow_evaluations=int(diagnostic_evaluations),
        flow_evaluations=int(
            proposal_evaluations
            + numerator_evaluations
            + selection_evaluations
            + diagnostic_evaluations
        ),
        elapsed_seconds=float(time.perf_counter() - started),
        importance=importance,
        iterations=iterations,
        evaluations=evaluations,
    )


def run_exact_section5(
    likelihood: CatalogueLikelihood,
    mock: MockCatalogue,
    *,
    steps: Sequence[float] = (0.005, 0.01, 0.02),
    object_chunk: int = 32,
    atom_chunk: int = 4096,
) -> Section5SeedResult:
    """Exhaustive finite-catalogue oracle for a deliberately small prior."""

    steps = tuple(sorted({float(value) for value in steps}))
    started = time.perf_counter()
    estimates = []
    for component, direction in (("g1", (1.0, 0.0)), ("g2", (0.0, 1.0))):
        for h in steps:
            score = likelihood.score_and_information(
                mock.measurements,
                center=(0.0, 0.0),
                direction=direction,
                delta=h,
                richardson=False,
                object_chunk=object_chunk,
                atom_chunk=atom_chunk,
            )
            estimates.append(
                summarize_section5(
                    score.score,
                    score.information,
                    component=component,
                    h=h,
                    n_draws=int(np.count_nonzero(likelihood.cache.prior.weights)),
                )
            )
    n_atoms = int(np.count_nonzero(likelihood.cache.prior.weights))
    return Section5SeedResult(
        proposal_seed=None,
        steps=steps,
        ladder=(n_atoms,),
        n_candidates=None,
        n_objects=len(mock.measurements),
        n_views=1 + 4 * len(steps),
        flow_evaluations=len(mock.measurements) * n_atoms * (1 + 4 * len(steps)),
        elapsed_seconds=float(time.perf_counter() - started),
        estimates=tuple(estimates),
    )


def _index(result: Section5SeedResult):
    return {
        (estimate.component, estimate.h, estimate.n_draws): estimate
        for estimate in result.estimates
    }


def assess_section5_null(
    results: Sequence[Section5SeedResult],
    *,
    primary_h: float = 0.005,
    exact_oracle: Optional[Section5SeedResult] = None,
    oracle_importance: Optional[Section5SeedResult] = None,
    independent_banks: Sequence[Section5SeedResult] = (),
    agreement_fraction_se: float = 0.25,
    centring_sigma: float = 3.0,
    identity_fraction: float = 0.05,
    identity_sigma: float = 3.0,
    stability_fraction: float = 0.05,
    bank_stability_fraction: float = 0.10,
    min_hill_tail_index: float = 2.0,
) -> Section5GateAssessment:
    """Apply the predeclared Section 5 numerical, sampling, and tail gates."""

    results = tuple(results)
    if len(results) < 2:
        raise ValueError("two independent proposal seeds are required")
    if any(len(result.ladder) < 2 for result in results):
        raise ValueError("at least two draw rungs are required")
    primary_h = float(primary_h)
    metrics: dict[str, float] = {}
    checks: dict[str, bool] = {}
    indices = [_index(result) for result in results]
    max_draw = min(result.ladder[-1] for result in results)
    previous_draw = min(result.ladder[-2] for result in results)

    numerical_scaled = []
    rung_scaled = []
    seed_scaled = []
    centring_pull = []
    identity_fractional = []
    identity_pull = []
    stability = []
    hill = []
    for component in ("g1", "g2"):
        primary = [index[(component, primary_h, max_draw)] for index in indices]
        for index, reference in zip(indices, primary):
            for h in results[0].steps:
                current = index[(component, h, max_draw)]
                scale = max(reference.robust_standard_error, current.robust_standard_error)
                numerical_scaled.append(
                    abs(current.estimated_shear - reference.estimated_shear) / scale
                )
            previous = index[(component, primary_h, previous_draw)]
            scale = max(reference.robust_standard_error, previous.robust_standard_error)
            rung_scaled.append(
                abs(reference.estimated_shear - previous.estimated_shear) / scale
            )
            for field_name in ("score_variance", "information_mean"):
                upper = getattr(reference, field_name)
                lower = getattr(previous, field_name)
                stability.append(abs(upper - lower) / max(abs(upper), abs(lower)))
            centring_pull.extend(
                [
                    abs(reference.score_z),
                    abs(reference.estimated_shear / reference.robust_standard_error),
                ]
            )
            identity_fractional.append(
                abs(reference.information_identity_ratio - 1.0)
            )
            identity_pull.append(
                abs(reference.information_identity_ratio - 1.0)
                / reference.information_identity_standard_error
            )
            hill.append(reference.tail.hill_tail_index)
        scale = max(value.robust_standard_error for value in primary)
        seed_scaled.append(
            np.ptp([value.estimated_shear for value in primary]) / scale
        )
        for field_name in ("score_variance", "information_mean"):
            values = [getattr(value, field_name) for value in primary]
            stability.append(np.ptp(values) / max(abs(value) for value in values))

    metrics.update(
        {
            "max_h_change_in_se": float(max(numerical_scaled)),
            "max_rung_change_in_se": float(max(rung_scaled)),
            "max_seed_spread_in_se": float(max(seed_scaled)),
            "max_centring_pull": float(max(centring_pull)),
            "max_information_identity_fraction": float(max(identity_fractional)),
            "max_information_identity_pull": float(max(identity_pull)),
            "max_tail_relative_change": float(max(stability)),
            "min_hill_tail_index": float(np.nanmin(hill)),
        }
    )
    checks.update(
        {
            "numerical_derivative": bool(max(numerical_scaled) <= agreement_fraction_se),
            "draw_ladder": bool(max(rung_scaled) <= agreement_fraction_se),
            "independent_proposal_seeds": bool(max(seed_scaled) <= agreement_fraction_se),
            "score_centring": bool(max(centring_pull) <= centring_sigma),
            "information_identity": bool(
                max(identity_fractional) <= identity_fraction
                and max(identity_pull) <= identity_sigma
            ),
            "tail_stability": bool(
                max(stability) <= stability_fraction
                and np.isfinite(hill).all()
                and min(hill) > min_hill_tail_index
            ),
        }
    )

    oracle_scaled = float("inf")
    if exact_oracle is not None and oracle_importance is not None:
        exact_index = {
            (estimate.component, estimate.h): estimate
            for estimate in exact_oracle.estimates
        }
        sampled_index = {
            (estimate.component, estimate.h): estimate
            for estimate in oracle_importance.estimates
            if estimate.n_draws == oracle_importance.ladder[-1]
        }
        shared = set(exact_index) & set(sampled_index)
        differences = []
        for key in shared:
            exact = exact_index[key]
            sampled = sampled_index[key]
            scale = max(exact.robust_standard_error, sampled.robust_standard_error)
            differences.append(abs(exact.estimated_shear - sampled.estimated_shear) / scale)
        if differences:
            oracle_scaled = float(max(differences))
    metrics["max_oracle_difference_in_se"] = oracle_scaled
    checks["exact_oracle"] = bool(oracle_scaled <= agreement_fraction_se)

    bank_change = float("inf")
    bank_hill = float("nan")
    if len(independent_banks) >= 2:
        bank_estimates = [
            _index(result)[(component, primary_h, result.ladder[-1])]
            for result in independent_banks
            for component in ("g1", "g2")
        ]
        changes = []
        for component in ("g1", "g2"):
            component_values = [
                _index(result)[(component, primary_h, result.ladder[-1])]
                for result in independent_banks
            ]
            for field in ("score_variance", "information_mean"):
                values = [getattr(value, field) for value in component_values]
                changes.append(np.ptp(values) / max(abs(value) for value in values))
        bank_change = float(max(changes))
        bank_hill = float(
            np.nanmin([value.tail.hill_tail_index for value in bank_estimates])
        )
    metrics["max_independent_bank_relative_change"] = bank_change
    metrics["min_independent_bank_hill_tail_index"] = bank_hill
    checks["independent_banks"] = bool(
        bank_change <= bank_stability_fraction
        and np.isfinite(bank_hill)
        and bank_hill > min_hill_tail_index
    )

    failures = tuple(name for name, passed in checks.items() if not passed)
    return Section5GateAssessment(
        passed=not failures,
        checks=checks,
        metrics=metrics,
        failures=failures,
    )


__all__ = [
    "AdaptiveOneStepResult",
    "AdaptiveSection5Result",
    "AutogradShearOptimizationResult",
    "ImportanceAutogradResult",
    "PairedSection5Estimate",
    "Section5AutogradResult",
    "Section5Estimate",
    "Section5GateAssessment",
    "Section5ObjectMoments",
    "Section5SeedResult",
    "Section5StreamResult",
    "autograd_exact_section5",
    "autograd_importance_section5",
    "StreamedImportanceDiagnostics",
    "TailDiagnostics",
    "assess_section5_null",
    "estimate_one_step_adaptive_section5",
    "estimate_one_step_stratified_section5",
    "run_exact_section5",
    "run_adaptive_section5",
    "run_stratified_section5",
    "run_streamed_section5",
    "optimize_shear_autograd",
    "summarize_paired_section5",
    "summarize_section5",
]
