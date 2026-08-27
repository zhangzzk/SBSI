#!/usr/bin/env python
"""Test fully probabilistic tempered top-K support on one exact target."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from plot_infer_v1_sampling import _candidate_positions, _sha256
from test_infer_v1_proposal_diversification import _load_pipeline, _sync_cuda
from sbsi.catalogue_sampling import select_tempered_candidates
from sbsi.sampling_diagnostics import (
    evaluate_exact_proposal_target,
    summarize_defensive_proposal,
)


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-result", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--object-id", type=int, default=514716)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--score-pool", type=int, default=4194304)
    parser.add_argument("--score-prefilter", type=int, default=8388608)
    parser.add_argument("--k", type=int, default=65536)
    parser.add_argument("--temperatures", nargs="+", type=float, default=(0.5, 1.0, 1.5))
    parser.add_argument(
        "--candidate-seeds", nargs="+", type=int, default=(7301, 7302, 7303, 7304, 7305)
    )
    parser.add_argument("--epsilon", type=float, default=0.3)
    return parser.parse_args(argv)


def _percentiles(values):
    return np.percentile(np.asarray(values, dtype=np.float64), [10, 50, 90]).tolist()


def _plot(rows, baseline, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame = pd.DataFrame(rows)
    temperatures = sorted(frame["temperature"].unique())
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
    for seed, group in frame.groupby("candidate_seed"):
        group = group.sort_values("temperature")
        axes[0].plot(
            group["temperature"],
            100 * group["candidate_target_mass"],
            color="#56B4E9",
            alpha=0.45,
            marker="o",
            markersize=3,
            linewidth=0.8,
        )
        axes[1].plot(
            group["temperature"],
            100 * group["asymptotic_ess_fraction"],
            color="#009E73",
            alpha=0.45,
            marker="o",
            markersize=3,
            linewidth=0.8,
        )
    axes[0].axhline(
        100 * baseline["candidate_target_mass"],
        color="#D55E00",
        linestyle="--",
        label="Deterministic top-K",
    )
    axes[1].axhline(
        100 * baseline["asymptotic_ess_fraction"],
        color="#D55E00",
        linestyle="--",
        label="Deterministic top-K",
    )
    axes[0].set_ylabel("Exact target capture (%)")
    axes[1].set_ylabel("Exact ESS/M (%)")
    for axis in axes:
        axis.set_xlabel("Temperature T")
        axis.set_xticks(temperatures)
        axis.legend(frameon=False, fontsize=7)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(labelsize=7)
    figure.suptitle(
        "Observation 514,716: fully probabilistic soft top-K",
        fontsize=10,
    )
    png = output / "soft_topk.png"
    pdf = output / "soft_topk.pdf"
    figure.savefig(png, dpi=300, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    return png, pdf


def main(argv=None):
    args = _parse_args(argv)
    output = Path(args.output).resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}")
    if args.k <= 0 or args.k > args.score_pool:
        raise RuntimeError("K lies outside the score pool")
    if not 0 < args.epsilon <= 1:
        raise RuntimeError("epsilon must lie in (0, 1]")
    if any(value <= 0 for value in args.temperatures):
        raise RuntimeError("temperatures must be positive")

    reference_path = Path(args.reference_result).resolve()
    _, identity, likelihood, proposal, mock, metadata = _load_pipeline(
        reference_path, args.device
    )
    observed = mock.measurements.iloc[[args.object_id]].reset_index(drop=True)
    _sync_cuda()
    start = perf_counter()
    score_pool = proposal.candidates(
        observed,
        n_candidates=int(args.score_pool),
        prefilter_candidates=int(args.score_prefilter),
        torch_device=likelihood.flow_model.device,
    )
    _sync_cuda()
    score_pool_seconds = perf_counter() - start
    start = perf_counter()
    comparison = evaluate_exact_proposal_target(
        likelihood,
        observed,
        proposal,
        object_id=int(args.object_id),
        center=identity["initial_center"]["center"],
        n_candidates=int(args.score_pool),
        prefilter_candidates=int(args.score_prefilter),
        epsilon=float(args.epsilon),
        candidate_backend="torch",
        atom_chunk=65536,
        candidates=score_pool,
    )
    _sync_cuda()
    exact_seconds = perf_counter() - start

    active = comparison.atom_indices
    target = comparison.target_probability
    prior = likelihood.cache.prior.weights[active].astype(np.float64, copy=True)
    prior /= prior.sum()
    ranked = comparison.candidate_indices
    score_gap = comparison.candidate_score_gap
    baseline_positions = _candidate_positions(active, ranked[: args.k])
    baseline = summarize_defensive_proposal(
        target, prior, baseline_positions, float(args.epsilon)
    )
    rows = []
    for temperature in args.temperatures:
        for seed in args.candidate_seeds:
            start = perf_counter()
            selected = select_tempered_candidates(
                ranked,
                score_gap,
                n_candidates=int(args.k),
                temperature=float(temperature),
                seed=int(seed),
            )
            selection_seconds = perf_counter() - start
            positions = _candidate_positions(active, selected)
            rows.append(
                {
                    "temperature": float(temperature),
                    "candidate_seed": int(seed),
                    "selection_seconds": float(selection_seconds),
                    **summarize_defensive_proposal(
                        target, prior, positions, float(args.epsilon)
                    ),
                }
            )

    aggregate = []
    frame = pd.DataFrame(rows)
    for temperature, group in frame.groupby("temperature", sort=True):
        aggregate.append(
            {
                "temperature": float(temperature),
                "n_candidate_seeds": int(len(group)),
                "candidate_target_mass_percentiles": _percentiles(
                    group["candidate_target_mass"]
                ),
                "asymptotic_ess_fraction_percentiles": _percentiles(
                    group["asymptotic_ess_fraction"]
                ),
                "maximum_target_to_proposal_ratio_percentiles": _percentiles(
                    group["maximum_target_to_proposal_ratio"]
                ),
                "selection_seconds_percentiles": _percentiles(
                    group["selection_seconds"]
                ),
            }
        )
    robust_winner = max(
        aggregate,
        key=lambda row: (
            row["asymptotic_ess_fraction_percentiles"][0],
            row["candidate_target_mass_percentiles"][0],
        ),
    )
    robust_beats_topk = bool(
        robust_winner["asymptotic_ess_fraction_percentiles"][0]
        > baseline["asymptotic_ess_fraction"]
    )

    output.mkdir(parents=True, exist_ok=False)
    frame.to_csv(output / "settings.csv", index=False)
    figures = _plot(rows, baseline, output)
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
            "temperatures": list(map(float, args.temperatures)),
            "candidate_seeds": list(map(int, args.candidate_seeds)),
            "epsilon": float(args.epsilon),
            "selection": "full Gumbel top-K without replacement",
            "selection_log_weight": "(score-score_max)/temperature",
            "deterministic_core": 0,
            "exact_target_used_for_selection": False,
        },
        "baseline_topk": baseline,
        "aggregate": aggregate,
        "robust_winner": robust_winner,
        "robust_winner_beats_topk_ess": robust_beats_topk,
        "runtime": {
            "score_pool_seconds": float(score_pool_seconds),
            "exact_target_seconds": float(exact_seconds),
        },
        "identity": {
            **metadata,
            "soft_topk_script_sha256": _sha256(Path(__file__)),
        },
        "files": [str(path) for path in figures],
        "infer_v1_defaults_changed": False,
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": "complete", "output": str(output)}))


if __name__ == "__main__":
    main()
