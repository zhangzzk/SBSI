#!/usr/bin/env python
"""Compare flatter score-selection forms through epsilon-zero likelihood histograms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from plot_infer_v1_sampling import _candidate_positions, _sha256
from test_infer_v1_proposal_diversification import _load_pipeline, _sync_cuda
from sbsi.catalogue_sampling import select_weighted_candidates
from sbsi.sampling_diagnostics import evaluate_exact_proposal_target


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-result", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--object-id", type=int, default=514716)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--score-pool", type=int, default=4194304)
    parser.add_argument("--score-prefilter", type=int, default=8388608)
    parser.add_argument("--k", type=int, default=16384)
    parser.add_argument(
        "--weight-forms",
        nargs="+",
        default=("exponential", "gaussian", "logistic", "cauchy"),
    )
    parser.add_argument("--temperatures", nargs="+", type=float, default=(0.5, 1.0, 1.5))
    parser.add_argument(
        "--candidate-seeds", nargs="+", type=int, default=(7301, 7302, 7303, 7304, 7305)
    )
    parser.add_argument("--bins", type=int, default=70)
    return parser.parse_args(argv)


def _weighted_quantile(values, weights, quantile):
    weights = np.asarray(weights, dtype=np.float64)
    weights /= weights.sum()
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order])
    position = min(
        int(np.searchsorted(cumulative, quantile, side="left")), len(order) - 1
    )
    return float(values[order[position]])


def _histogram(values, weights, edges):
    return np.histogram(np.clip(values, edges[0], edges[-1]), edges, weights=weights)[0]


def _percentiles(values):
    return np.percentile(np.asarray(values, dtype=np.float64), [10, 50, 90]).tolist()


def _plot_histograms(frame, target_hist, edges, output, forms, temperatures, baseline):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"target": "#D55E00", "proposal": "#0072B2"}
    centers = 0.5 * (edges[:-1] + edges[1:])
    figure, axes = plt.subplots(
        len(forms), len(temperatures), figsize=(10.2, 9.0), sharex=True, sharey=True
    )
    axes = np.atleast_2d(axes)
    panel = 0
    for row, weight_form in enumerate(forms):
        for column, temperature in enumerate(temperatures):
            axis = axes[row, column]
            selected = frame[
                (frame["weight_form"] == weight_form)
                & np.isclose(frame["temperature"], temperature)
            ]
            proposal = np.percentile(
                np.stack(selected["proposal_hist"].to_numpy()), [10, 50, 90], axis=0
            )
            capture = 100 * np.percentile(
                selected["candidate_target_mass"], [10, 50, 90]
            )
            axis.fill_between(
                centers,
                0,
                target_hist,
                step="mid",
                color=colors["target"],
                alpha=0.2,
                label="Exact target p",
            )
            axis.step(
                centers,
                target_hist,
                where="mid",
                color=colors["target"],
                linewidth=1.3,
            )
            axis.fill_between(
                centers,
                proposal[0],
                proposal[2],
                step="mid",
                color=colors["proposal"],
                alpha=0.16,
                label="q seed p10-p90",
            )
            axis.step(
                centers,
                proposal[1],
                where="mid",
                color=colors["proposal"],
                linewidth=1.4,
                label=r"Proposal q ($\epsilon=0$)",
            )
            axis.text(
                0.02,
                0.96,
                (
                    f"{chr(ord('A') + panel)}  {weight_form}, T={temperature:g}\n"
                    f"capture={capture[1]:.2f}% [{capture[0]:.2f}, {capture[2]:.2f}]"
                ),
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=7,
            )
            if panel == 0:
                axis.legend(frameon=False, fontsize=7)
            axis.spines[["top", "right"]].set_visible(False)
            axis.tick_params(labelsize=7)
            panel += 1
    for axis in axes[-1]:
        axis.set_xlabel(r"Conditional log likelihood  $\log L_j$", fontsize=8)
    for axis in axes[:, 0]:
        axis.set_ylabel("Probability mass / bin", fontsize=8)
    figure.suptitle(
        (
            "Observation 514,716: flatter candidate selection, local proposal only\n"
            f"K=16,384, epsilon=0; deterministic deep top-K capture={100 * baseline:.2f}%"
        ),
        fontsize=10,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    png = output / "flatter_selection_likelihood_histograms.png"
    pdf = output / "flatter_selection_likelihood_histograms.pdf"
    figure.savefig(png, dpi=300, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    return png, pdf


def _plot_capture(frame, output, baseline):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "exponential": "#0072B2",
        "gaussian": "#D55E00",
        "logistic": "#009E73",
        "cauchy": "#CC79A7",
    }
    figure, axis = plt.subplots(figsize=(5.0, 3.2), constrained_layout=True)
    for weight_form, group in frame.groupby("weight_form", sort=False):
        summary = group.groupby("temperature")["candidate_target_mass"].agg(list)
        temperatures = np.asarray(summary.index)
        percentiles = np.asarray([_percentiles(values) for values in summary])
        axis.plot(
            temperatures,
            100 * percentiles[:, 1],
            marker="o",
            linewidth=1.4,
            color=colors[weight_form],
            label=weight_form,
        )
        axis.fill_between(
            temperatures,
            100 * percentiles[:, 0],
            100 * percentiles[:, 2],
            color=colors[weight_form],
            alpha=0.15,
        )
    axis.axhline(
        100 * baseline,
        color="0.35",
        linestyle="--",
        label="Deterministic deep top-K",
    )
    axis.set_xlabel("Temperature T")
    axis.set_ylabel("Exact target capture (%)")
    axis.legend(frameon=False, fontsize=7, ncol=2)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(labelsize=7)
    png = output / "flatter_selection_capture.png"
    pdf = output / "flatter_selection_capture.pdf"
    figure.savefig(png, dpi=300, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    return png, pdf


def main(argv=None):
    args = _parse_args(argv)
    output = Path(args.output).resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}")
    allowed = {"exponential", "gaussian", "logistic", "cauchy"}
    if not args.weight_forms or not set(args.weight_forms) <= allowed:
        raise RuntimeError("unknown or empty weight-form list")
    if args.k <= 0 or args.k > args.score_pool:
        raise RuntimeError("K lies outside the score pool")
    if any(value <= 0 for value in args.temperatures):
        raise RuntimeError("temperatures must be positive")

    reference_path = Path(args.reference_result).resolve()
    _, identity, likelihood, proposal, mock, metadata = _load_pipeline(
        reference_path, args.device
    )
    observed = mock.measurements.iloc[[args.object_id]].reset_index(drop=True)
    score_pool = proposal.candidates(
        observed,
        n_candidates=int(args.score_pool),
        prefilter_candidates=int(args.score_prefilter),
        torch_device=likelihood.flow_model.device,
    )
    comparison = evaluate_exact_proposal_target(
        likelihood,
        observed,
        proposal,
        object_id=int(args.object_id),
        center=identity["initial_center"]["center"],
        n_candidates=int(args.score_pool),
        prefilter_candidates=int(args.score_prefilter),
        epsilon=0.1,
        candidate_backend="torch",
        atom_chunk=65536,
        candidates=score_pool,
    )
    _sync_cuda()

    active = comparison.atom_indices
    target = comparison.target_probability
    log_likelihood = comparison.conditional_log_likelihood
    ranked = comparison.candidate_indices
    score_gap = comparison.candidate_score_gap
    baseline_positions = _candidate_positions(active, ranked[: args.k])
    baseline_capture = float(target[baseline_positions].sum())
    baseline_q = target[baseline_positions] / baseline_capture
    finite = np.isfinite(log_likelihood)
    left = min(
        _weighted_quantile(
            log_likelihood[baseline_positions], baseline_q, 0.01
        ),
        _weighted_quantile(log_likelihood[finite], target[finite], 0.001),
    )
    right = max(
        _weighted_quantile(
            log_likelihood[baseline_positions], baseline_q, 0.999
        ),
        _weighted_quantile(log_likelihood[finite], target[finite], 0.999),
    )
    edges = np.linspace(left, right, int(args.bins) + 1)
    target_hist = _histogram(log_likelihood, target, edges)

    rows = []
    for weight_form in args.weight_forms:
        for temperature in args.temperatures:
            for seed in args.candidate_seeds:
                selected = select_weighted_candidates(
                    ranked,
                    score_gap,
                    n_candidates=int(args.k),
                    temperature=float(temperature),
                    weight_form=weight_form,
                    seed=int(seed),
                )
                positions = _candidate_positions(active, selected)
                capture = float(target[positions].sum())
                local = target[positions] / capture
                rows.append(
                    {
                        "weight_form": weight_form,
                        "temperature": float(temperature),
                        "candidate_seed": int(seed),
                        "candidate_target_mass": capture,
                        "proposal_hist": _histogram(
                            log_likelihood[positions], local, edges
                        ),
                    }
                )
    frame = pd.DataFrame(rows)
    aggregate = []
    for (weight_form, temperature), group in frame.groupby(
        ["weight_form", "temperature"], sort=False
    ):
        aggregate.append(
            {
                "weight_form": weight_form,
                "temperature": float(temperature),
                "candidate_target_mass_percentiles": _percentiles(
                    group["candidate_target_mass"]
                ),
            }
        )

    output.mkdir(parents=True, exist_ok=False)
    frame.drop(columns=["proposal_hist"]).to_csv(output / "settings.csv", index=False)
    long_rows = []
    for _, row in frame.iterrows():
        for bin_index, proposal_mass in enumerate(row["proposal_hist"]):
            long_rows.append(
                {
                    "weight_form": row["weight_form"],
                    "temperature": row["temperature"],
                    "candidate_seed": row["candidate_seed"],
                    "candidate_target_mass": row["candidate_target_mass"],
                    "bin": bin_index,
                    "bin_left": edges[bin_index],
                    "bin_right": edges[bin_index + 1],
                    "proposal_mass": proposal_mass,
                    "target_mass": target_hist[bin_index],
                }
            )
    pd.DataFrame(long_rows).to_csv(output / "histograms.csv.gz", index=False)
    files = [
        *_plot_histograms(
            frame,
            target_hist,
            edges,
            output,
            args.weight_forms,
            args.temperatures,
            baseline_capture,
        ),
        *_plot_capture(frame, output, baseline_capture),
    ]
    result = {
        "status": "complete",
        "object_id": int(args.object_id),
        "exact_log_evidence": float(
            logsumexp(comparison.log_target)
            - comparison.population_log_normalization
        ),
        "design": {
            "k": int(args.k),
            "score_pool": int(args.score_pool),
            "score_prefilter": int(args.score_prefilter),
            "weight_forms": list(args.weight_forms),
            "temperatures": list(map(float, args.temperatures)),
            "candidate_seeds": list(map(int, args.candidate_seeds)),
            "epsilon": 0.0,
            "selection": "full Gumbel top-K without replacement",
            "exact_target_used_for_selection": False,
        },
        "baseline_deep_topk_target_mass": baseline_capture,
        "aggregate": aggregate,
        "semantics": {
            "proposal_q": "exact target restricted and renormalized on selected support",
            "epsilon_zero_warning": "q has zero mass outside selected support; evidence importance sampling would be biased",
            "candidate_target_mass": "sum of exact normalized posterior target probability over selected atoms",
        },
        "identity": {
            **metadata,
            "diagnostic_sha256": _sha256(Path(__file__)),
        },
        "files": [str(path) for path in files],
        "infer_v1_defaults_changed": False,
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": "complete", "output": str(output)}))


if __name__ == "__main__":
    main()
