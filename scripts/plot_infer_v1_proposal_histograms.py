#!/usr/bin/env python
"""Compare diversified proposal log-likelihood marginals with the exact target."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from plot_infer_v1_sampling import _candidate_positions, _sha256
from test_infer_v1_proposal_diversification import (
    _load_pipeline,
    _qmc_query_frame,
    _sync_cuda,
)
from sbsi.catalogue_sampling import (
    select_score_diversified_candidates,
    union_candidate_sets,
)
from sbsi.sampling_diagnostics import evaluate_exact_proposal_target


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-result", required=True)
    parser.add_argument("--screen-result", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--object-id", type=int, default=514716)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bins", type=int, default=90)
    return parser.parse_args(argv)


def _weighted_quantile(values, weights, quantile: float) -> float:
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / weights.sum()
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order])
    position = min(
        int(np.searchsorted(cumulative, quantile, side="left")), len(order) - 1
    )
    return float(values[order[position]])


def _mass_histogram(values, weights, edges):
    return np.histogram(np.clip(values, edges[0], edges[-1]), edges, weights=weights)[0]


def _build_supports(observed, comparison, proposal, design):
    active = comparison.atom_indices
    ranked = comparison.candidate_indices
    score_gap = comparison.candidate_score_gap
    supports = []

    def add(indices, **metadata):
        supports.append(
            {
                "positions": _candidate_positions(
                    active, np.asarray(indices, dtype=np.int64)
                ),
                **metadata,
            }
        )

    for k in design["k"]:
        add(
            ranked[:k],
            method_key="top_deep",
            method_label="Deep top-K",
            family="top",
            n_candidates=k,
            candidate_seed=None,
        )
        for core_fraction in design["core_fractions"]:
            fraction = (
                f"{int(round(100 * core_fraction))}/"
                f"{int(round(100 * (1 - core_fraction)))}"
            )
            variants = [("log_stratified", None)] + [
                ("tempered", float(temperature))
                for temperature in design["temperatures"]
            ]
            for method, temperature in variants:
                for seed in design["candidate_seeds"]:
                    indices = select_score_diversified_candidates(
                        ranked,
                        score_gap,
                        n_candidates=k,
                        core_fraction=float(core_fraction),
                        tail_method=method,
                        temperature=temperature,
                        seed=int(seed),
                        n_log_strata=int(design["log_strata"]),
                    )
                    key = (
                        f"log_{fraction}"
                        if method == "log_stratified"
                        else f"temp{temperature:g}_{fraction}"
                    )
                    label = (
                        f"Log tail {fraction}"
                        if method == "log_stratified"
                        else f"T={temperature:g} {fraction}"
                    )
                    add(
                        indices,
                        method_key=key,
                        method_label=label,
                        family=method,
                        n_candidates=k,
                        candidate_seed=int(seed),
                    )

    proposal._build_uncertainty_mips_tree()
    local_scale = np.median(proposal.coordinates.dispersion[ranked[:4096]], axis=0)
    for k in design["k"]:
        fallback = proposal.uncertainty_candidates(
            observed, n_candidates=int(k)
        ).indices[0]
        add(
            fallback,
            method_key="top_global",
            method_label="Global uncertainty top-K",
            family="global_top",
            n_candidates=k,
            candidate_seed=None,
        )
        for n_queries in design["query_counts"]:
            per_query = int(k // int(n_queries))
            for seed in design["candidate_seeds"]:
                queries = _qmc_query_frame(
                    observed,
                    target_names=proposal.coordinates.target_names,
                    scale=local_scale,
                    n_queries=int(n_queries),
                    seed=int(seed),
                )
                queried = proposal.uncertainty_candidates(
                    queries, n_candidates=per_query
                )
                indices, _ = union_candidate_sets(
                    queried.indices,
                    n_candidates=k,
                    fallback_indices=fallback,
                )
                add(
                    indices,
                    method_key=f"union_q{int(n_queries)}",
                    method_label=f"QMC union Q={int(n_queries)}",
                    family="query_union",
                    n_candidates=k,
                    candidate_seed=int(seed),
                )
    return supports


def _aggregate_histograms(frame, *, method_key, k, epsilon, column):
    chosen = frame[
        (frame["method_key"] == method_key)
        & (frame["n_candidates"] == k)
        & np.isclose(frame["epsilon"], epsilon)
    ]
    if chosen.empty:
        raise RuntimeError(f"missing histogram group {method_key}, K={k}, eps={epsilon}")
    matrix = np.stack(chosen[column].to_numpy())
    return np.percentile(matrix, [10, 50, 90], axis=0), chosen


def _draw_histogram_panel(
    axis,
    frame,
    target_hist,
    centers,
    *,
    method_key,
    method_label,
    k,
    epsilon,
    log_y,
    legend=False,
):
    proposal_hist, chosen = _aggregate_histograms(
        frame,
        method_key=method_key,
        k=k,
        epsilon=epsilon,
        column="proposal_hist",
    )
    local_hist, _ = _aggregate_histograms(
        frame,
        method_key=method_key,
        k=k,
        epsilon=epsilon,
        column="local_hist",
    )
    axis.fill_between(
        centers,
        0,
        target_hist,
        step="mid",
        color="#D55E00",
        alpha=0.22,
        label="Exact posterior target p",
    )
    axis.step(
        centers,
        target_hist,
        where="mid",
        color="#D55E00",
        linewidth=1.5,
    )
    if len(chosen) > 1:
        axis.fill_between(
            centers,
            proposal_hist[0],
            proposal_hist[2],
            step="mid",
            color="#0072B2",
            alpha=0.16,
            label="q seed p10-p90",
        )
    axis.step(
        centers,
        proposal_hist[1],
        where="mid",
        color="#0072B2",
        linewidth=1.5,
        label="Full proposal q",
    )
    axis.step(
        centers,
        local_hist[1],
        where="mid",
        color="#009E73",
        linestyle="--",
        linewidth=1.15,
        label=r"Local component $q_{local}$",
    )
    if len(chosen) > 1:
        axis.fill_between(
            centers,
            local_hist[0],
            local_hist[2],
            step="mid",
            color="#009E73",
            alpha=0.09,
        )
    capture = 100 * np.percentile(chosen["candidate_target_mass"], [10, 50, 90])
    axis.text(
        0.02,
        0.96,
        (
            f"{method_label}\n"
            f"target capture={capture[1]:.2f}%"
            + (
                f" [{capture[0]:.2f}, {capture[2]:.2f}]" if len(chosen) > 1 else ""
            )
        ),
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=7.5,
    )
    if log_y:
        axis.set_yscale("log")
        axis.set_ylim(bottom=1.0e-7)
    else:
        axis.set_ylim(bottom=0)
    if legend:
        axis.legend(frameon=False, fontsize=7, loc="upper right")
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(labelsize=7)


def _plot_selected(frame, target_hist, edges, output, *, k, epsilon, log_y):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    centers = 0.5 * (edges[:-1] + edges[1:])
    selected = [
        ("top_deep", "Deep top-K"),
        ("log_75/25", "Log tail 75/25"),
        ("temp1.5_75/25", "T=1.5 tail 75/25"),
        ("temp1.5_50/50", "T=1.5 tail 50/50"),
        ("union_q8", "QMC union Q=8"),
        ("union_q16", "QMC union Q=16"),
    ]
    figure, axes = plt.subplots(3, 2, figsize=(9.0, 8.2), sharex=True)
    for number, (axis, (key, label)) in enumerate(zip(axes.flat, selected)):
        _draw_histogram_panel(
            axis,
            frame,
            target_hist,
            centers,
            method_key=key,
            method_label=label,
            k=k,
            epsilon=epsilon,
            log_y=log_y,
            legend=number == 0,
        )
        axis.text(
            -0.09,
            1.03,
            chr(ord("A") + number),
            transform=axis.transAxes,
            fontweight="bold",
            fontsize=10,
        )
    for axis in axes[-1]:
        axis.set_xlabel(r"Conditional log likelihood  $\log L_j$", fontsize=8)
    for axis in axes[:, 0]:
        axis.set_ylabel("Probability mass per bin", fontsize=8)
    scale = "logarithmic y scale" if log_y else "linear y scale"
    figure.suptitle(
        (
            f"Observation 514,716: proposal vs exact-target log-likelihood mass\n"
            rf"$K={k:,}$, $\epsilon={epsilon:g}$; {scale}"
        ),
        fontsize=11,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    suffix = "logy" if log_y else "linear"
    png = output / f"proposal_loglikelihood_histograms_{suffix}.png"
    pdf = output / f"proposal_loglikelihood_histograms_{suffix}.pdf"
    figure.savefig(png, dpi=300, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    return png, pdf


def _plot_all_pages(frame, target_hist, edges, output, *, methods, k_values, epsilons):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    centers = 0.5 * (edges[:-1] + edges[1:])
    path = output / "proposal_loglikelihood_histograms_all_settings.pdf"
    with PdfPages(path) as pdf:
        for method_key, method_label in methods:
            figure, axes = plt.subplots(
                len(k_values), len(epsilons), figsize=(10.2, 5.8), sharex=True, sharey=True
            )
            axes = np.atleast_2d(axes)
            for row, k in enumerate(k_values):
                for column, epsilon in enumerate(epsilons):
                    axis = axes[row, column]
                    _draw_histogram_panel(
                        axis,
                        frame,
                        target_hist,
                        centers,
                        method_key=method_key,
                        method_label=f"K={k:,}, eps={epsilon:g}",
                        k=k,
                        epsilon=epsilon,
                        log_y=True,
                        legend=row == 0 and column == 0,
                    )
                    if row == len(k_values) - 1:
                        axis.set_xlabel(r"Conditional $\log L_j$", fontsize=8)
                    if column == 0:
                        axis.set_ylabel("Mass / bin", fontsize=8)
            figure.suptitle(
                f"Observation 514,716: {method_label} versus exact target",
                fontsize=11,
            )
            figure.tight_layout(rect=(0, 0, 1, 0.95))
            pdf.savefig(figure, bbox_inches="tight")
            plt.close(figure)
    return path


def main(argv=None):
    args = _parse_args(argv)
    reference_path = Path(args.reference_result).resolve()
    screen_path = Path(args.screen_result).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}")
    screen = json.loads(screen_path.read_text())
    if screen.get("status") != "complete":
        raise RuntimeError("diversification screen is not complete")
    design = screen["design"]
    design["temperatures"] = [1.5, 2.0, 4.0]
    design["log_strata"] = 16

    _, identity, likelihood, proposal, mock, metadata = _load_pipeline(
        reference_path, args.device
    )
    object_id = int(args.object_id)
    observed = mock.measurements.iloc[[object_id]].reset_index(drop=True)
    score_pool = proposal.candidates(
        observed,
        n_candidates=int(design["score_pool"]),
        prefilter_candidates=int(design["score_prefilter"]),
        torch_device=likelihood.flow_model.device,
    )
    comparison = evaluate_exact_proposal_target(
        likelihood,
        observed,
        proposal,
        object_id=object_id,
        center=identity["initial_center"]["center"],
        n_candidates=int(design["score_pool"]),
        prefilter_candidates=int(design["score_prefilter"]),
        epsilon=0.1,
        candidate_backend="torch",
        atom_chunk=65536,
        candidates=score_pool,
    )
    _sync_cuda()
    supports = _build_supports(observed, comparison, proposal, design)

    active = comparison.atom_indices
    log_likelihood = comparison.conditional_log_likelihood
    target = comparison.target_probability
    prior = likelihood.cache.prior.weights[active].astype(np.float64, copy=True)
    prior /= prior.sum()
    baseline = next(
        support
        for support in supports
        if support["method_key"] == "top_deep" and support["n_candidates"] == 32768
    )
    baseline_mass = target[baseline["positions"]].sum()
    baseline_local = target[baseline["positions"]] / baseline_mass
    baseline_q = 0.1 * prior
    baseline_q[baseline["positions"]] += 0.9 * baseline_local
    finite = np.isfinite(log_likelihood)
    left = min(
        _weighted_quantile(log_likelihood[finite], baseline_q[finite], 0.01),
        _weighted_quantile(log_likelihood[finite], target[finite], 0.001),
    )
    right = max(
        _weighted_quantile(log_likelihood[finite], baseline_q[finite], 0.999),
        _weighted_quantile(log_likelihood[finite], target[finite], 0.999),
    )
    edges = np.linspace(left, right, int(args.bins) + 1)
    target_hist = _mass_histogram(log_likelihood, target, edges)
    prior_hist = _mass_histogram(log_likelihood, prior, edges)

    rows = []
    for support in supports:
        positions = support["positions"]
        candidate_mass = float(target[positions].sum())
        local = target[positions] / candidate_mass
        local_hist = _mass_histogram(log_likelihood[positions], local, edges)
        local_left = float(local[log_likelihood[positions] < left].sum())
        local_right = float(local[log_likelihood[positions] > right].sum())
        for epsilon in design["epsilons"]:
            proposal_hist = epsilon * prior_hist + (1.0 - epsilon) * local_hist
            rows.append(
                {
                    **{key: value for key, value in support.items() if key != "positions"},
                    "epsilon": float(epsilon),
                    "candidate_target_mass": candidate_mass,
                    "proposal_left_edge_mass": float(
                        epsilon * prior[log_likelihood < left].sum()
                        + (1.0 - epsilon) * local_left
                    ),
                    "proposal_right_edge_mass": float(
                        epsilon * prior[log_likelihood > right].sum()
                        + (1.0 - epsilon) * local_right
                    ),
                    "proposal_hist": proposal_hist,
                    "local_hist": local_hist,
                }
            )
    frame = pd.DataFrame(rows)

    output.mkdir(parents=True, exist_ok=False)
    long_rows = []
    for _, row in frame.iterrows():
        for bin_index, (proposal_mass, local_mass) in enumerate(
            zip(row["proposal_hist"], row["local_hist"])
        ):
            long_rows.append(
                {
                    **{
                        key: row[key]
                        for key in (
                            "method_key",
                            "method_label",
                            "family",
                            "n_candidates",
                            "candidate_seed",
                            "epsilon",
                            "candidate_target_mass",
                            "proposal_left_edge_mass",
                            "proposal_right_edge_mass",
                        )
                    },
                    "bin": bin_index,
                    "bin_left": edges[bin_index],
                    "bin_right": edges[bin_index + 1],
                    "target_mass": target_hist[bin_index],
                    "proposal_mass": proposal_mass,
                    "local_mass": local_mass,
                    "prior_mass": prior_hist[bin_index],
                }
            )
    pd.DataFrame(long_rows).to_csv(output / "histograms.csv.gz", index=False)

    selected_k = max(int(value) for value in design["k"])
    selected_epsilon = max(float(value) for value in design["epsilons"])
    files = []
    for log_y in (False, True):
        files.extend(
            _plot_selected(
                frame,
                target_hist,
                edges,
                output,
                k=selected_k,
                epsilon=selected_epsilon,
                log_y=log_y,
            )
        )
    methods = list(
        frame[["method_key", "method_label"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    files.append(
        _plot_all_pages(
            frame,
            target_hist,
            edges,
            output,
            methods=methods,
            k_values=design["k"],
            epsilons=design["epsilons"],
        )
    )
    payload = {
        "status": "complete",
        "object_id": object_id,
        "exact_log_evidence": float(
            logsumexp(comparison.log_target)
            - comparison.population_log_normalization
        ),
        "semantics": {
            "x": "conditional atom log likelihood log L_j",
            "target": "exact normalized posterior atom mass p(j) proportional to pi_j Pdet_j L_j",
            "local": "exact target restricted and renormalized on each candidate support",
            "proposal": "epsilon times active prior plus (1-epsilon) times local",
            "seed_summary": "selected plots show median and p10-p90 across candidate seeds",
            "edge_bins": "values outside the displayed range are clipped into the first or last bin",
        },
        "display": {
            "bins": int(args.bins),
            "left": left,
            "right": right,
            "selected_k": selected_k,
            "selected_epsilon": selected_epsilon,
            "target_left_edge_mass": float(target[log_likelihood < left].sum()),
            "target_right_edge_mass": float(target[log_likelihood > right].sum()),
        },
        "design": design,
        "n_supports": len(supports),
        "n_settings": len(frame),
        "identity": {
            **metadata,
            "screen_result": str(screen_path),
            "screen_result_sha256": _sha256(screen_path),
            "histogram_script_sha256": _sha256(Path(__file__)),
        },
        "files": [str(path) for path in files],
        "infer_v1_defaults_changed": False,
    }
    (output / "result.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"status": "complete", "output": str(output), "files": payload["files"]}))


if __name__ == "__main__":
    main()
