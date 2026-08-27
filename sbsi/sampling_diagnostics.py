"""Inspect how an adapted proposal represents per-observation evidence."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy.special import logsumexp

from .catalogue_likelihood import CatalogueLikelihood
from .catalogue_sampling import DefensiveLocalProposal


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
) -> tuple[Path, ...]:
    """Plot proposal-draw mass beside importance-weighted evidence mass."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    metrics = summarize_importance_sampling(diagnostic, ladder)
    palette = {"proposal": "#0072B2", "evidence": "#D55E00"}
    paths = []
    for row, (object_id, label) in enumerate(zip(diagnostic.object_ids, labels)):
        values = diagnostic.conditional_log_likelihood[row]
        finite = values[np.isfinite(values)]
        if not finite.size:
            raise RuntimeError(f"object {object_id} has no finite likelihood draws")
        left, right = float(finite.min()), float(finite.max())
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
            weight = normalized_importance_weights(
                diagnostic.log_importance_weight[row, :n_draws]
            )
            before = np.full(n_draws, 1.0 / n_draws)
            axes[rung_row, 0].hist(
                x,
                bins=edges,
                weights=before,
                histtype="stepfilled",
                color=palette["proposal"],
                alpha=0.65,
            )
            axes[rung_row, 1].hist(
                x,
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
                    f"ESS={entry['ess']:.0f} ({entry['ess_fraction']:.3f}M)\n"
                    f"max weight={entry['max_weight_fraction']:.3f}\n"
                    f"outside local={entry['outside_local_evidence_fraction']:.3f}"
                ),
                transform=axes[rung_row, 1].transAxes,
                ha="right",
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
            f"Observation {int(object_id):,} — {label}; center={diagnostic.center}",
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
    "ImportanceSamplingDiagnostic",
    "evaluate_importance_sampling",
    "normalized_importance_weights",
    "plot_importance_sampling_diagnostic",
    "save_importance_sampling_diagnostic",
    "select_example_rows",
    "summarize_importance_sampling",
]
