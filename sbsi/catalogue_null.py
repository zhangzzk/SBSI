"""Streamed Section 5 catalogue-prior null estimator and go/no-go gates.

This module evaluates the local finite-difference expansion of the *marginal*
detected-population log likelihood at zero shear.  Importance atoms are fixed
across every stencil view and draw ladders are exact prefixes.  Only one object
batch owns an ``N_batch x M`` tensor at a time.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import time
from typing import Mapping, Optional, Sequence

import numpy as np
from scipy.special import logsumexp
import torch

from .catalogue_closure import MockCatalogue
from .catalogue_likelihood import CatalogueLikelihood
from .catalogue_sampling import (
    CoalescedProposalDraw,
    DefensiveLocalProposal,
    ProposalDraw,
    pareto_tail_index,
    select_adaptive_draw_counts,
    select_independent_pilot_draw_counts,
)


# Both stratified modes sum the candidate stratum exactly and estimate only
# its complement; they differ solely in the proposal that complement is drawn
# from.  Every downstream branch treats them identically.
STRATIFIED_MODES = ("stratified", "tilted_stratified")
CANDIDATE_SOURCES = ("location_prefilter", "whole_catalogue_gaussian_proxy")


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
    # Per-object weight diagnostics of the zero-shear view, one row per prefix
    # in `weight_diagnostic_draws`: the retained ladder when it is kept, and
    # the production draw count alone otherwise.  They answer whether the
    # estimator's weights admit a root-M rate at all, which no other saved
    # quantity does.
    weight_diagnostic_draws: Optional[tuple[int, ...]] = None
    weight_ess: Optional[np.ndarray] = field(default=None, repr=False)
    weight_max_fraction: Optional[np.ndarray] = field(default=None, repr=False)
    weight_relative_error: Optional[np.ndarray] = field(default=None, repr=False)
    weight_pareto_k: Optional[np.ndarray] = field(default=None, repr=False)
    proposal_prefilter_candidates: Optional[int] = None
    allocation_method: str = "production_prefix"
    pilot_draws: Optional[int] = None
    pilot_seed: Optional[int] = None
    pilot_ess_fraction: Optional[np.ndarray] = field(default=None, repr=False)
    pilot_max_weight_fraction: Optional[np.ndarray] = field(default=None, repr=False)
    phase_seconds: Mapping[str, float] = field(default_factory=dict)
    bias_correction: str = "none"
    candidate_backend: str = "scipy"
    candidate_source: str = "location_prefilter"
    estimator_mode: str = "mixture"
    # Atom slots the stencil actually pushed through the flow, against the slots
    # that carry a nonzero draw count.  `coalesce` pads every row in an object
    # chunk out to that chunk's widest unique-atom count, and the padded slots are
    # scored and then discarded by the reduction, so the ratio is pure waste.
    stencil_atom_slots: int = 0
    stencil_valid_atom_slots: int = 0

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
        # Under the mixture every draw contributes, so a row with no unique
        # atom is a defect.  Under stratification `unique` counts only the
        # retained complement draws, and zero is the legitimate answer for an
        # object whose candidate support already covers the catalogue.
        floor = 0 if self.estimator_mode in STRATIFIED_MODES else 1
        if (counts <= 0).any() or (unique < floor).any() or (unique > counts).any():
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
        if self.candidate_source not in CANDIDATE_SOURCES:
            raise ValueError("unknown candidate source")
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
        diagnostic_draws = self.weight_diagnostic_draws
        diagnostics = {
            "weight_ess": self.weight_ess,
            "weight_max_fraction": self.weight_max_fraction,
            "weight_relative_error": self.weight_relative_error,
            "weight_pareto_k": self.weight_pareto_k,
        }
        if diagnostic_draws is None:
            if any(value is not None for value in diagnostics.values()):
                raise ValueError("weight diagnostics need their draw counts")
        else:
            diagnostic_draws = tuple(int(value) for value in diagnostic_draws)
            if not diagnostic_draws or any(value <= 0 for value in diagnostic_draws):
                raise ValueError("weight-diagnostic draw counts must be positive")
            expected = (len(diagnostic_draws), len(score))
            for name, value in diagnostics.items():
                if value is None:
                    raise ValueError(f"weight diagnostics are incomplete: {name}")
                value = np.asarray(value, dtype=np.float64)
                if value.shape != expected:
                    raise ValueError(f"{name} must have shape {expected}")
                object.__setattr__(self, name, value)
            # `weight_pareto_k` is NaN wherever the tail is undefined -- a row
            # whose weights are flat above the fitting threshold -- so only the
            # two exact quantities are required to be finite.
            if (
                not np.isfinite(self.weight_ess).all()
                or not np.isfinite(self.weight_max_fraction).all()
                or not np.isfinite(self.weight_relative_error).all()
            ):
                raise ValueError("weight diagnostics contain non-finite values")
            object.__setattr__(self, "weight_diagnostic_draws", diagnostic_draws)
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "information", information)
        object.__setattr__(self, "draw_counts", counts)
        object.__setattr__(self, "unique_counts", unique)
        object.__setattr__(self, "phase_seconds", phases)

    def weight_diagnostic_summary(self) -> Optional[list]:
        """One JSON-friendly row per diagnosed prefix.

        Percentiles rather than means, because these distributions are the
        heavy-tailed thing under investigation and a mean ESS says very little
        about the worst objects.  `pareto_k_undefined` counts the rows whose
        tail could not be fitted, so a small `pareto_k` sample cannot be
        mistaken for a clean one.
        """

        if self.weight_diagnostic_draws is None:
            return None
        percentiles = [0, 10, 50, 90, 100]
        rows = []
        for index, n_draws in enumerate(self.weight_diagnostic_draws):
            khat = self.weight_pareto_k[index]
            fitted = khat[np.isfinite(khat)]
            rows.append(
                {
                    "draws": int(n_draws),
                    "ess_percentiles": np.percentile(
                        self.weight_ess[index], percentiles
                    ).tolist(),
                    "ess_fraction_percentiles": np.percentile(
                        self.weight_ess[index] / n_draws, percentiles
                    ).tolist(),
                    "max_weight_fraction_percentiles": np.percentile(
                        self.weight_max_fraction[index], percentiles
                    ).tolist(),
                    # The one row that compares across estimator modes.
                    "relative_standard_error_percentiles": np.percentile(
                        self.weight_relative_error[index], percentiles
                    ).tolist(),
                    "pareto_k_percentiles": (
                        None
                        if fitted.size == 0
                        else np.percentile(fitted, percentiles).tolist()
                    ),
                    "pareto_k_undefined": int(khat.size - fitted.size),
                    # The PSIS reliability threshold is sample-size dependent;
                    # above it the weight variance is effectively unusable.
                    "pareto_k_threshold": float(
                        min(1.0 - 1.0 / np.log10(max(n_draws, 11)), 0.7)
                    ),
                    "pareto_k_above_threshold": (
                        None
                        if fitted.size == 0
                        else float(
                            np.mean(
                                fitted
                                > min(1.0 - 1.0 / np.log10(max(n_draws, 11)), 0.7)
                            )
                        )
                    ),
                }
            )
        return rows


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
            "stencil_atom_slots": int(self.moments.stencil_atom_slots),
            "stencil_valid_atom_slots": int(self.moments.stencil_valid_atom_slots),
            "stencil_padding_fraction": (
                None
                if self.moments.stencil_atom_slots == 0
                else 1.0
                - self.moments.stencil_valid_atom_slots
                / self.moments.stencil_atom_slots
            ),
            "elapsed_seconds": self.moments.elapsed_seconds,
            "phase_seconds": dict(self.moments.phase_seconds),
            "retained_full_ladder": self.moments.ladder_score is not None,
            "weight_diagnostics": self.moments.weight_diagnostic_summary(),
            "allocation_method": self.moments.allocation_method,
            "bias_correction": self.moments.bias_correction,
            "candidate_backend": self.moments.candidate_backend,
            "candidate_source": self.moments.candidate_source,
            "estimator_mode": self.moments.estimator_mode,
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


def _stratified_logsumexp(
    exact_log_terms: torch.Tensor,
    tail_log_weights: torch.Tensor,
    draw: CoalescedProposalDraw,
    n_draws: int,
) -> torch.Tensor:
    """Log of an exact candidate stratum plus its sampled complement.

    ``exact_log_terms`` is ``log(pi Pdet L)`` on the deterministic candidate
    support, summed with no proposal correction and therefore no variance.
    ``tail_log_weights`` is ``log(pi Pdet L / pi)`` on the unique prior draws;
    slots whose atom already lies inside the candidate support carry no
    contribution, because that stratum is accounted for exactly.
    """

    counts = torch.as_tensor(
        draw.counts(n_draws),
        dtype=tail_log_weights.dtype,
        device=tail_log_weights.device,
    )
    outside = torch.as_tensor(
        (draw.local_position < 0) & draw.valid,
        device=tail_log_weights.device,
    )
    tail = torch.where(
        outside & (counts > 0),
        tail_log_weights + torch.log(counts.clamp_min(1)) - np.log(n_draws),
        torch.full_like(tail_log_weights, -torch.inf),
    )
    return torch.logsumexp(torch.cat((exact_log_terms, tail), dim=1), dim=1)


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


def _weight_tail_diagnostics(
    log_weights: torch.Tensor,
    draw: CoalescedProposalDraw,
    n_draws: int,
    *,
    atom_valid: Optional[np.ndarray] = None,
    exact_log_terms: Optional[torch.Tensor] = None,
):
    """Weight diagnostics of one nested prefix, at the zero-shear view.

    `ess` counts effective draws out of `n_draws`; `maximum` is the largest
    single draw's share of the sampled sum; `khat` is the generalized-Pareto
    shape of the weight tail, so `khat >= 0.5` means the weight variance is
    infinite and no number of extra draws buys a root-M rate.

    `atom_valid` restricts the weight set to the atoms the estimator actually
    reduces.  Under stratification the draws that land in the exact stratum
    carry weight zero; they contribute nothing to either sum, so they lower the
    effective count exactly as they should.

    `relative` is the estimator's own relative standard error, and it is the
    only one of these that compares across estimator modes.  ESS/M does not:
    the stratified arm's exact stratum contributes no variance and no draws, so
    its ESS fraction is mechanically lower while its estimate can be far
    better.  Passing the exact stratum's log terms as `exact_log_terms` folds
    in the share of the total that is sampled at all,

    ```text
    relvar = (1/ESS - 1/M) (T / (E + T))^2
    ```

    with `E` the exact stratum and `T` the sampled mean.  For the mixture the
    stratum is empty, the share is one, and this reduces to the usual
    `sqrt(1/ESS - 1/M)`.
    """

    counts = torch.as_tensor(
        draw.counts(n_draws), dtype=log_weights.dtype, device=log_weights.device
    )
    live = counts > 0
    if atom_valid is not None:
        live = live & torch.as_tensor(atom_valid, device=log_weights.device)
    peak = torch.max(
        torch.where(live, log_weights, torch.full_like(log_weights, -torch.inf)),
        dim=1,
        keepdim=True,
    ).values
    finite_peak = torch.isfinite(peak)
    weights = torch.where(
        live & finite_peak,
        torch.exp(log_weights - torch.where(finite_peak, peak, torch.zeros_like(peak))),
        torch.zeros_like(log_weights),
    )
    total = (counts * weights).sum(dim=1)
    good = finite_peak[:, 0] & (total > 0)
    ess = torch.zeros(len(log_weights), dtype=log_weights.dtype, device=log_weights.device)
    ess[good] = total[good].square() / (counts[good] * weights[good].square()).sum(dim=1)
    maximum = torch.zeros_like(ess)
    maximum[good] = weights[good].max(dim=1).values / total[good]

    sampled_mean = total / float(n_draws)
    if exact_log_terms is None:
        share = torch.ones_like(ess)
    else:
        # Rescaled by the same per-row peak, so the two strata stay comparable.
        exact_total = torch.exp(
            exact_log_terms - torch.where(finite_peak, peak, torch.zeros_like(peak))
        ).sum(dim=1)
        denominator = exact_total + sampled_mean
        share = torch.where(
            denominator > 0, sampled_mean / denominator, torch.zeros_like(ess)
        )
    relative = torch.zeros_like(ess)
    relative[good] = share[good] * torch.sqrt(
        (1.0 / ess[good] - 1.0 / float(n_draws)).clamp_min(0.0)
    )

    # The tail fit needs the draw sample itself, not the unique atoms: a weight
    # drawn ten times is ten draws of that weight.
    inverse = torch.as_tensor(
        draw.inverse[:, :n_draws].astype(np.int64, copy=False),
        device=log_weights.device,
    )
    khat = pareto_tail_index(torch.gather(weights, 1, inverse))
    khat[~good.detach().cpu().numpy()] = np.nan
    return (
        ess.detach().cpu().numpy().astype(np.float64),
        maximum.detach().cpu().numpy().astype(np.float64),
        relative.detach().cpu().numpy().astype(np.float64),
        khat,
    )


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

    # Left-pack each row's outside draws into a rectangle as wide as the widest
    # row.  `np.nonzero` walks the mask in row-major order, so subtracting each
    # row's start offset turns its flat position into its rank within the row --
    # the same packing the per-row loop this replaced did one row at a time.
    counts = outside.sum(axis=1)
    width = int(counts.max())
    rows, columns = np.nonzero(outside)
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    packed_position = np.arange(rows.size, dtype=np.int64) - starts[rows]
    packed_indices = np.repeat(draw.indices[:, :1], width, axis=1)
    packed_probability = np.repeat(draw.probability[:, :1], width, axis=1)
    packed_valid = np.zeros(packed_indices.shape, dtype=bool)
    packed_indices[rows, packed_position] = draw.indices[rows, columns]
    packed_probability[rows, packed_position] = draw.probability[rows, columns]
    packed_valid[rows, packed_position] = True
    evaluated = likelihood.log_importance_weights_tensor(
        observed,
        float(center[0]),
        float(center[1]),
        atom_indices=packed_indices,
        proposal_probability=packed_probability,
        atom_valid=packed_valid,
        observed_targets=observed_targets,
        object_chunk=object_chunk,
        atom_chunk=atom_chunk,
    )
    row_index = torch.as_tensor(rows, dtype=torch.long, device=device)
    output[row_index, torch.as_tensor(columns, dtype=torch.long, device=device)] = (
        evaluated[row_index, torch.as_tensor(packed_position, dtype=torch.long, device=device)]
    )
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
    surrogate = getattr(likelihood, "population_normalization", None)
    if surrogate is not None:
        if likelihood.selection is None:
            raise ValueError("a population-normalization cache requires measured selection")
        return surrogate.log_mass(g1, g2)
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
                        # Padded slots of the packed rectangle are no longer
                        # scored, so the count is the outside draws themselves.
                        flow_evaluation_count += int((~draw.local_member).sum())
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


def _reduce_stencil(
    log_likelihood,
    half_log_likelihood,
    rung_log_likelihood,
    *,
    score: np.ndarray,
    information: np.ndarray,
    ladder_score: Optional[np.ndarray],
    ladder_information: Optional[np.ndarray],
    global_rows: np.ndarray,
    ladder: Sequence[int],
    h: float,
    full_information: bool,
    retain_full_ladder: bool,
) -> None:
    """Turn one group's stencil log likelihoods into score and information.

    The estimators differ only in how each view's log likelihood is formed;
    the finite-difference algebra below is identical for all of them and is
    shared so that no mode can drift away from another.
    """

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
    candidate_source: str = "location_prefilter",
    estimator_mode: str = "mixture",
    tilt_delta: float = 0.1,
    tilt_temperature: float = 1.0,
    full_information: bool = False,
    retain_full_ladder: bool = False,
    object_id_offset: int = 0,
    object_ids: Optional[np.ndarray] = None,
    object_chunk: int = 16,
    atom_chunk: int = 4096,
    progress=None,
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
    # Measured selection enters only through the scalar population normalizer,
    # while an external R_blend disables the zero-view shortcut and forces each
    # stencil view through the ordinary tensor path.  Both are already handled
    # below; neither changes the proposal or complement algebra.
    # The zero-view tensor shortcut requires spin-0 detection inputs.  A
    # shape-sensitive detector is valid on the general path: CatalogueModelCache
    # builds and scores its detection table independently at every stencil view.
    if likelihood.tensor_shape_only_available:
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
    # cont.345: a bright-only recompute evaluates a non-contiguous subset of
    # the mock, so the per-object draw seed can no longer be derived from a
    # scalar offset.  Passing the absolute row ids explicitly keeps every
    # object's atoms identical to the ones it would draw inside a full pass,
    # which is what makes the hybrid comparable to the exact chain under
    # common random numbers (doc/CONVENTIONS.md section 6d).
    if object_ids is None:
        resolved_object_ids = None
    else:
        resolved_object_ids = np.asarray(object_ids, dtype=np.int64)
        if resolved_object_ids.ndim != 1:
            raise ValueError("object_ids must be one-dimensional")
        if (resolved_object_ids < 0).any():
            raise ValueError("object_ids must be non-negative")
        if object_id_offset:
            raise ValueError("object_ids and object_id_offset are exclusive")
    if allocation_method not in ("production_prefix", "independent_pilot"):
        raise ValueError("unknown adaptive allocation method")
    if bias_correction not in ("none", "richardson_1_over_m"):
        raise ValueError("unknown adaptive finite-draw bias correction")
    if candidate_backend not in ("scipy", "torch"):
        raise ValueError("unknown candidate-query backend")
    if candidate_source not in CANDIDATE_SOURCES:
        raise ValueError("unknown candidate source")
    if estimator_mode not in ("mixture",) + STRATIFIED_MODES:
        raise ValueError("unknown estimator mode")
    if estimator_mode in STRATIFIED_MODES:
        # The stratified estimator has no mixture weights, so the mixture ESS
        # and maximum-weight rules that drive adaptive stopping do not define
        # its draw budget.  Screening it therefore requires the fixed ladder.
        if allocation_method != "production_prefix" or not retain_full_ladder:
            raise ValueError(
                "stratified estimation requires the production prefix "
                "allocation and a retained full ladder"
            )
        if bias_correction != "none":
            raise ValueError(
                "stratified estimation and finite-draw bias correction are "
                "separate modes"
            )
    if estimator_mode == "tilted_stratified" and not (0.0 < tilt_delta < 1.0):
        raise ValueError("tilt_delta must lie strictly between zero and one")
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
    # Weight diagnostics follow the retained ladder for the same reason the
    # ladder moments do: only there does every object share one prefix, so a
    # rung is a comparable quantity across objects.  Adaptive allocation gives
    # each object its own draw count and there is no common rung to report.
    weight_ess = (
        np.empty((len(ladder), n_objects), dtype=np.float64)
        if retain_full_ladder
        else None
    )
    weight_max_fraction = (
        np.empty((len(ladder), n_objects), dtype=np.float64)
        if retain_full_ladder
        else None
    )
    weight_relative_error = (
        np.empty((len(ladder), n_objects), dtype=np.float64)
        if retain_full_ladder
        else None
    )
    weight_pareto_k = (
        np.empty((len(ladder), n_objects), dtype=np.float64)
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
        "weight_diagnostics": 0.0,
    }
    stencil_atom_slots = 0
    stencil_valid_atom_slots = 0
    last_progress = started

    # Every draw already accepts explicit ids and resolves a bare offset to
    # `offset + arange`, so passing the array unconditionally is identical to
    # the previous offset path for a contiguous partition and correct for a
    # non-contiguous one.
    if resolved_object_ids is None:
        all_object_ids = object_id_offset + np.arange(n_objects, dtype=np.int64)
    elif len(resolved_object_ids) != n_objects:
        raise ValueError("object_ids must carry one id per observation")
    else:
        all_object_ids = resolved_object_ids

    for object_start in range(0, n_objects, object_chunk):
        object_stop = min(object_start + object_chunk, n_objects)
        chunk_object_ids = all_object_ids[object_start:object_stop]
        observed = mock.measurements.iloc[object_start:object_stop].reset_index(drop=True)
        observed_targets = likelihood.observed_target_tensor(observed)
        phase_started = time.perf_counter()
        candidate_device = (
            likelihood.flow_model.device if candidate_backend == "torch" else None
        )
        draw = None
        if (
            candidate_source == "whole_catalogue_gaussian_proxy"
            and estimator_mode == "tilted_stratified"
        ):
            # Production direct-top-K path: rank the pure proxy, then transform
            # that same dense score matrix in place into the tilted proposal.
            # Candidate membership, uniforms, and q probabilities are
            # identical to the former two-pass implementation.
            candidates, draw = (
                proposal.whole_catalogue_proxy_candidates_and_tilted_draw(
                    observed,
                    n_candidates=n_candidates,
                    n_draws=ladder[-1],
                    delta=tilt_delta,
                    temperature=tilt_temperature,
                    seed=proposal_seed,
                    object_ids=chunk_object_ids,
                    device=candidate_device,
                )
            )
        elif candidate_source == "whole_catalogue_gaussian_proxy":
            candidates = proposal.whole_catalogue_proxy_candidates(
                observed,
                n_candidates=n_candidates,
                device=candidate_device,
            )
        else:
            candidates = proposal.candidates(
                observed,
                n_candidates=n_candidates,
                prefilter_candidates=proposal_prefilter_candidates,
                torch_device=candidate_device,
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
        candidate_target = (
            candidate_target_tensor.detach().cpu().numpy().astype(np.float64)
        )
        phase_seconds["candidate_flow"] += time.perf_counter() - phase_started
        phase_started = time.perf_counter()
        if allocation_method == "independent_pilot":
            pilot_draw = proposal.draw_adapted(
                candidates,
                candidate_target,
                n_draws=int(pilot_draws),
                epsilon=epsilon,
                seed=int(pilot_seed),
                object_ids=chunk_object_ids,
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
            flow_evaluations += int((~pilot_draw.local_member).sum())
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

        zero_weights = None
        if estimator_mode in STRATIFIED_MODES:
            # No mixture weights exist here: the candidate stratum is summed
            # exactly and every draw estimates only its complement.  The whole
            # budget therefore reaches the tail that carries the mass the
            # mixture proposal covers with epsilon of its draws.
            if estimator_mode == "tilted_stratified":
                # cont.317: drawing that complement from the flat prior is what
                # leaves the tail index above 0.7.  The tilted proposal uses
                # the same uniform stream, so the arms stay paired.
                if draw is None:
                    draw = proposal.draw_tilted(
                        candidates,
                        observed,
                        n_draws=ladder[-1],
                        delta=tilt_delta,
                        temperature=tilt_temperature,
                        seed=proposal_seed,
                        object_ids=chunk_object_ids,
                        device=(
                            likelihood.flow_model.device
                            if candidate_backend == "torch"
                            else None
                        ),
                    )
            else:
                draw = proposal.draw_stratified(
                    candidates,
                    n_draws=ladder[-1],
                    seed=proposal_seed,
                    object_ids=chunk_object_ids,
                )
            counts = np.full(len(observed), ladder[-1], dtype=np.int64)
        elif allocation_method == "production_prefix":
            draw = proposal.draw_adapted(
                candidates,
                candidate_target,
                n_draws=ladder[-1],
                epsilon=epsilon,
                seed=proposal_seed,
                object_ids=chunk_object_ids,
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
            flow_evaluations += int((~draw.local_member).sum())
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
            absolute_rows = all_object_ids[global_rows]
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
                flow_evaluations += int((~group_draw.local_member).sum())
            else:
                group_draw = draw.take(local_rows).prefix(int(n_draws))
                group_zero_weights = zero_weights
            coalesced = group_draw.coalesce()

            def _record_weight_diagnostics(
                log_weights, group_draw, rows, atom_valid=None, exact_log_terms=None
            ):
                """Fill the ladder diagnostic rows for this object group."""

                if not retain_full_ladder:
                    return
                started_diagnostics = time.perf_counter()
                for rung_index, rung in enumerate(ladder):
                    (
                        weight_ess[rung_index, rows],
                        weight_max_fraction[rung_index, rows],
                        weight_relative_error[rung_index, rows],
                        weight_pareto_k[rung_index, rows],
                    ) = _weight_tail_diagnostics(
                        log_weights,
                        group_draw,
                        rung,
                        atom_valid=atom_valid,
                        exact_log_terms=exact_log_terms,
                    )
                phase_seconds["weight_diagnostics"] += (
                    time.perf_counter() - started_diagnostics
                )

            if estimator_mode in STRATIFIED_MODES:
                # Slots whose atom already lies in the candidate stratum are
                # never handed to the flow: that stratum is summed exactly.
                outside_valid = coalesced.valid & (coalesced.local_position < 0)
                unique_counts[global_rows] = outside_valid.sum(axis=1)
                exact_indices = candidates.indices[local_rows]
                exact_ones = np.ones(exact_indices.shape, dtype=np.float64)
                log_likelihood = {}
                half_log_likelihood = None
                rung_log_likelihood = {} if retain_full_ladder else None
                for name in views:
                    if name == "zero":
                        exact = candidate_target_tensor[
                            torch.as_tensor(
                                local_rows,
                                dtype=torch.long,
                                device=candidate_target_tensor.device,
                            )
                        ]
                    else:
                        exact = likelihood.log_importance_weights_tensor(
                            group_observed,
                            *views[name],
                            atom_indices=exact_indices,
                            proposal_probability=exact_ones,
                            observed_targets=group_targets,
                            object_chunk=len(group_observed),
                            atom_chunk=atom_chunk,
                        )
                        flow_evaluations += int(exact_indices.size)
                        stencil_atom_slots += int(exact_indices.size)
                        stencil_valid_atom_slots += int(exact_indices.size)
                    tail = likelihood.log_importance_weights_tensor(
                        group_observed,
                        *views[name],
                        atom_indices=coalesced.indices,
                        proposal_probability=coalesced.probability,
                        atom_valid=outside_valid,
                        observed_targets=group_targets,
                        object_chunk=len(group_observed),
                        atom_chunk=atom_chunk,
                    )
                    flow_evaluations += int(outside_valid.sum())
                    stencil_atom_slots += int(coalesced.indices.size)
                    stencil_valid_atom_slots += int(outside_valid.sum())
                    if name == "zero":
                        _record_weight_diagnostics(
                            tail,
                            coalesced,
                            global_rows,
                            atom_valid=outside_valid,
                            exact_log_terms=exact,
                        )
                    log_likelihood[name] = (
                        _stratified_logsumexp(
                            exact, tail, coalesced, int(n_draws)
                        )
                        - normalizers[name]
                    )
                    if retain_full_ladder:
                        for rung in ladder:
                            rung_log_likelihood[(name, rung)] = (
                                _stratified_logsumexp(
                                    exact, tail, coalesced, rung
                                )
                                - normalizers[name]
                            )
                    del exact, tail
                _reduce_stencil(
                    log_likelihood,
                    half_log_likelihood,
                    rung_log_likelihood,
                    score=score,
                    information=information,
                    ladder_score=ladder_score,
                    ladder_information=ladder_information,
                    global_rows=global_rows,
                    ladder=ladder,
                    h=h,
                    full_information=full_information,
                    retain_full_ladder=retain_full_ladder,
                )
                continue
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
            _record_weight_diagnostics(zero_unique, coalesced, global_rows)
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
                    atom_valid=coalesced.valid,
                    observed_targets=group_targets,
                    object_chunk=len(group_observed),
                    atom_chunk=atom_chunk,
                )
                flow_evaluations += int(coalesced.valid.sum())
                stencil_atom_slots += int(coalesced.indices.size)
                stencil_valid_atom_slots += int(coalesced.valid.sum())
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
            _reduce_stencil(
                log_likelihood,
                half_log_likelihood,
                rung_log_likelihood,
                score=score,
                information=information,
                ladder_score=ladder_score,
                ladder_information=ladder_information,
                global_rows=global_rows,
                ladder=ladder,
                h=h,
                full_information=full_information,
                retain_full_ladder=retain_full_ladder,
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
        now = time.perf_counter()
        if progress is not None and (
            object_start == 0 or object_stop == n_objects or now - last_progress >= 60.0
        ):
            progress(object_stop, n_objects, now - started)
            last_progress = now

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
        weight_diagnostic_draws=(tuple(ladder) if retain_full_ladder else None),
        weight_ess=weight_ess,
        weight_max_fraction=weight_max_fraction,
        weight_relative_error=weight_relative_error,
        weight_pareto_k=weight_pareto_k,
        proposal_prefilter_candidates=(
            None
            if candidate_source == "whole_catalogue_gaussian_proxy"
            or proposal_prefilter_candidates is None
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
        candidate_source=candidate_source,
        estimator_mode=estimator_mode,
        stencil_atom_slots=int(stencil_atom_slots),
        stencil_valid_atom_slots=int(stencil_valid_atom_slots),
    )


def estimate_one_step_adaptive_section5(
    likelihood: CatalogueLikelihood,
    mock: MockCatalogue,
    proposal: DefensiveLocalProposal,
    *,
    center: Sequence[float],
    require_positive_definite: bool = True,
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
    return _summarize_one_step(
        center_array,
        moments,
        require_positive_definite=bool(require_positive_definite),
    )


def _summarize_one_step(
    center_array: np.ndarray,
    moments: AdaptiveSection5Result,
    *,
    require_positive_definite: bool = True,
) -> AdaptiveOneStepResult:
    """Convert per-object score/information into one full-2D Newton step."""

    score_sum = moments.score.sum(axis=0, dtype=np.float64)
    information_sum = moments.information.sum(axis=0, dtype=np.float64)
    information_sum = 0.5 * (information_sum + information_sum.T)
    eigenvalues = np.linalg.eigvalsh(information_sum)
    if not np.isfinite(eigenvalues).all() or (
        require_positive_definite and eigenvalues[0] <= 0
    ):
        raise RuntimeError(
            "one-step catalogue information is not positive definite: "
            f"eigenvalues={eigenvalues.tolist()}"
        )
    if np.any(np.isclose(eigenvalues, 0.0, rtol=0, atol=np.finfo(float).eps)):
        raise RuntimeError(
            "one-step catalogue information is singular: "
            f"eigenvalues={eigenvalues.tolist()}"
        )
    step = np.linalg.solve(information_sum, score_sum)
    estimate = center_array + step

    residual = moments.score - np.einsum("nij,j->ni", moments.information, step)
    mean_information = information_sum / len(residual)
    influence = np.linalg.solve(mean_information, residual.T).T
    robust_covariance = np.cov(influence, rowvar=False, ddof=1) / len(residual)
    model_covariance = np.linalg.inv(information_sum)
    robust_diagonal = np.diag(robust_covariance)
    model_diagonal = np.diag(model_covariance)
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
            map(float, np.sqrt(np.maximum(robust_diagonal, 0.0)))
        ),
        # An indefinite observation shard is never a standalone estimator; it
        # exists only to persist additive moments for the positive-definite
        # catalogue-level hybrid solve.  Keep its diagnostic covariance finite
        # without pretending a negative diagonal is a variance.
        model_standard_error=tuple(
            map(float, np.sqrt(np.maximum(model_diagonal, 0.0)))
        ),
        quadratic_log_likelihood_gain=0.5 * float(score_sum @ step),
        moments=moments,
    )


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
    yields all per-object gradients and all diagonal 2x2 Hessian blocks,
    including the off-diagonal information.
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
        flow_evaluations += int((~draw.local_member).sum())
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
            hessian = torch.stack(hessian_rows, dim=1)
            score[global_rows] = gradient.detach().cpu().numpy()
            information[global_rows] = -hessian.detach().cpu().numpy()
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
    "run_exact_section5",
    "run_adaptive_section5",
    "run_streamed_section5",
    "summarize_paired_section5",
    "summarize_section5",
]
