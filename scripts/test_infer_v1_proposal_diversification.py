#!/usr/bin/env python
"""Screen diversified Infer V1 candidate supports against one exact target."""

from __future__ import annotations

import argparse
import json
from math import ceil, log2
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import norm, qmc
import torch

from plot_infer_v1_sampling import _candidate_positions, _sha256, _verify_hashes
from sbsi.catalogue_closure import MockCatalogue
from sbsi.catalogue_likelihood import CatalogueLikelihood, CatalogueModelCache
from sbsi.catalogue_sampling import (
    DefensiveLocalProposal,
    ProposalCoordinateTable,
    select_score_diversified_candidates,
    union_candidate_sets,
)
from sbsi.measurement_model import load_measurement_model
from sbsi.models import ModelPaths, load_emulator
from sbsi.sampling_diagnostics import (
    evaluate_exact_proposal_target,
    simulate_defensive_evidence_errors,
    summarize_defensive_proposal,
)
from sbsi.scene_prior import ScenePrior


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-result", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--object-id", type=int, default=514716)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--score-pool", type=int, default=4194304)
    parser.add_argument("--score-prefilter", type=int, default=8388608)
    parser.add_argument("--k", nargs="+", type=int, default=(32768, 65536))
    parser.add_argument(
        "--core-fractions", nargs="+", type=float, default=(0.75, 0.5)
    )
    parser.add_argument(
        "--temperatures", nargs="+", type=float, default=(1.5, 2.0, 4.0)
    )
    parser.add_argument("--epsilons", nargs="+", type=float, default=(0.1, 0.2, 0.3))
    parser.add_argument(
        "--candidate-seeds", nargs="+", type=int, default=(7301, 7302, 7303, 7304, 7305)
    )
    parser.add_argument("--query-counts", nargs="+", type=int, default=(8, 16))
    parser.add_argument("--log-strata", type=int, default=16)
    parser.add_argument("--mc-replicates", type=int, default=256)
    parser.add_argument("--mc-draws", type=int, default=16384)
    parser.add_argument("--mc-seed", type=int, default=9917)
    parser.add_argument("--holdout-mc-seed", type=int, default=9918)
    return parser.parse_args(argv)


def _sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _load_pipeline(reference_path: Path, device: str):
    reference = json.loads(reference_path.read_text())
    if reference.get("status") != "complete":
        raise RuntimeError("reference inference result is not complete")
    config = reference["config"]
    identity = reference["identity"]
    unsupported = {
        "blend_response_cache": config.get("blend_response_cache"),
        "selection_cache": config.get("selection_cache"),
        "cut_abs_ehat": config.get("cut_abs_ehat"),
        "cut_bound": config.get("cut_bound"),
    }
    if any(bool(value) for value in unsupported.values()):
        raise RuntimeError("diversification diagnostic requires no measured cut")
    _verify_hashes(config["scene_store"], identity["scene_sha256"], "scene")
    _verify_hashes(config["model_cache"], identity["model_cache_sha256"], "model cache")
    _verify_hashes(
        config["proposal_cache"], identity["proposal_cache_sha256"], "proposal cache"
    )
    _verify_hashes(config["mock_input"], identity["mock_input_sha256"], "likelihood mock")
    model_files = {
        "measurement": config["measurement_model"],
        "emulator": config["emulator_model"],
        "emulator_metadata": config["emulator_metadata"],
    }
    if {name: _sha256(path) for name, path in model_files.items()} != identity["model_sha256"]:
        raise RuntimeError("model hashes do not match the reference result")
    conditions = {
        name: config[name]
        for name in ("pixel_size", "zero_point", "psf_fwhm", "moffat_beta", "pixel_rms")
    }
    prior = ScenePrior.load(config["scene_store"])
    flow = load_measurement_model(config["measurement_model"], device=device)
    detector = load_emulator(
        ModelPaths(
            flow_checkpoints=(Path(config["measurement_model"]),),
            emulator_model=Path(config["emulator_model"]),
            emulator_metadata=Path(config["emulator_metadata"]),
        ),
        conditions=conditions,
        device=device,
    )
    cache = CatalogueModelCache.load(config["model_cache"], prior=prior)
    cache.attach_detector(detector)
    cache.validate_model_features(flow)
    cache.validate_detection_shear_invariance()
    likelihood = CatalogueLikelihood(flow, cache)
    coordinates = ProposalCoordinateTable.load(config["proposal_cache"])
    proposal = DefensiveLocalProposal(
        coordinates,
        prior.weights,
        local_base_weights=cache.get(0.0, 0.0).detection_probability,
    )
    mock = MockCatalogue.load(config["mock_input"])
    if config.get("compile_flow", False):
        flow.compile_log_prob(mode=None, dynamic=True)
    metadata = {
        "inference_version": config["inference_version"],
        "reference_result": str(reference_path),
        "reference_result_sha256": _sha256(reference_path),
        "diagnostic_sha256": _sha256(Path(__file__)),
        "model_sha256": identity["model_sha256"],
        "model_cache_sha256": identity["model_cache_sha256"],
        "proposal_cache_sha256": identity["proposal_cache_sha256"],
        "mock_input_sha256": identity["mock_input_sha256"],
        "implementation_sha256": identity["implementation_sha256"],
    }
    return config, identity, likelihood, proposal, mock, metadata


def _qmc_query_frame(
    observed: pd.DataFrame,
    *,
    target_names: tuple[str, ...],
    scale: np.ndarray,
    n_queries: int,
    seed: int,
) -> pd.DataFrame:
    if n_queries <= 1 or n_queries & (n_queries - 1):
        raise ValueError("query count must be a power of two greater than one")
    sampler = qmc.Sobol(d=len(target_names), scramble=True, seed=int(seed))
    points = sampler.random_base2(m=int(ceil(log2(n_queries))))[: n_queries - 1]
    offsets = np.zeros((n_queries, len(target_names)), dtype=np.float64)
    offsets[1:] = norm.ppf(np.clip(points, 1.0e-6, 1.0 - 1.0e-6)) * scale
    queries = pd.concat([observed] * n_queries, ignore_index=True)
    center = observed.loc[:, target_names].to_numpy(dtype=np.float64)[0]
    queries.loc[:, list(target_names)] = (center[None, :] + offsets).astype(
        observed.loc[:, target_names].dtypes.iloc[0]
    )
    return queries


def _common_mc(prior: np.ndarray, *, n_replicates: int, n_draws: int, seed: int):
    rng = np.random.default_rng(int(seed))
    shape = (int(n_replicates), int(n_draws))
    component = rng.random(shape)
    global_uniform = rng.random(shape)
    local = rng.random(shape)
    cdf = np.cumsum(prior)
    cdf[-1] = 1.0
    global_position = np.searchsorted(cdf, global_uniform, side="right")
    return component, global_position, local


def _percentiles(values) -> list[float]:
    return np.percentile(np.asarray(values, dtype=np.float64), [0, 10, 50, 90, 100]).tolist()


def _aggregate(settings: list[dict]) -> list[dict]:
    fields = (
        "method_key",
        "method_label",
        "family",
        "n_candidates",
        "core_fraction",
        "temperature",
        "query_count",
        "epsilon",
    )
    metrics = (
        "candidate_target_mass",
        "asymptotic_ess_fraction",
        "maximum_target_to_proposal_ratio",
        "normal_p90_absolute_log_evidence_error",
        "empirical_p90_absolute_log_evidence_error",
        "proposal_runtime_seconds",
        "union_prefill_fraction",
    )
    groups: dict[tuple, list[dict]] = {}
    for row in settings:
        key = tuple(row.get(field) for field in fields)
        groups.setdefault(key, []).append(row)
    result = []
    for key, rows in groups.items():
        entry = dict(zip(fields, key))
        entry["n_candidate_seeds"] = len(rows)
        for metric in metrics:
            entry[f"{metric}_percentiles"] = _percentiles(
                [row[metric] for row in rows]
            )
        result.append(entry)
    return result


def _plot(payload: dict, output: Path) -> tuple[Path, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    aggregate = payload["aggregate"]
    methods = sorted(
        {row["method_key"]: row["method_label"] for row in aggregate}.items(),
        key=lambda item: item[1],
    )
    method_keys = [item[0] for item in methods]
    method_labels = [item[1] for item in methods]
    k_values = payload["design"]["k"]
    epsilons = payload["design"]["epsilons"]
    columns = [(k, epsilon) for k in k_values for epsilon in epsilons]

    capture = np.full((len(methods), len(k_values)), np.nan)
    ess = np.full((len(methods), len(columns)), np.nan)
    maximum = np.full_like(ess, np.nan)
    for row in aggregate:
        m = method_keys.index(row["method_key"])
        k = k_values.index(row["n_candidates"])
        capture[m, k] = row["candidate_target_mass_percentiles"][2]
        column = columns.index((row["n_candidates"], row["epsilon"]))
        ess[m, column] = 100 * row["asymptotic_ess_fraction_percentiles"][2]
        maximum[m, column] = row["maximum_target_to_proposal_ratio_percentiles"][2]

    figure, axes = plt.subplots(2, 2, figsize=(10.2, 7.4), constrained_layout=True)
    image = axes[0, 0].imshow(capture, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    axes[0, 0].set_xticks(range(len(k_values)), [f"{k // 1024}k" for k in k_values])
    axes[0, 0].set_yticks(range(len(methods)), method_labels)
    axes[0, 0].set_xlabel("Candidate count K")
    axes[0, 0].set_title("Median exact target capture")
    figure.colorbar(image, ax=axes[0, 0], shrink=0.85)

    labels = [f"{k // 1024}k\n{epsilon:.1f}" for k, epsilon in columns]
    image = axes[0, 1].imshow(
        ess,
        aspect="auto",
        cmap="viridis",
        norm=LogNorm(vmin=np.nanmin(ess), vmax=np.nanmax(ess)),
    )
    axes[0, 1].set_xticks(range(len(columns)), labels)
    axes[0, 1].set_yticks(range(len(methods)), method_labels)
    axes[0, 1].set_xlabel("K and epsilon")
    axes[0, 1].set_title("Median exact ESS/M (%)")
    figure.colorbar(image, ax=axes[0, 1], shrink=0.85)

    image = axes[1, 0].imshow(
        maximum,
        aspect="auto",
        cmap="cividis_r",
        norm=LogNorm(vmin=np.nanmin(maximum), vmax=np.nanmax(maximum)),
    )
    axes[1, 0].set_xticks(range(len(columns)), labels)
    axes[1, 0].set_yticks(range(len(methods)), method_labels)
    axes[1, 0].set_xlabel("K and epsilon")
    axes[1, 0].set_title("Median maximum p/q")
    figure.colorbar(image, ax=axes[1, 0], shrink=0.85)

    colors = {0.1: "#0072B2", 0.2: "#E69F00", 0.3: "#009E73"}
    markers = {32768: "o", 65536: "s"}
    for row in aggregate:
        axes[1, 1].scatter(
            row["proposal_runtime_seconds_percentiles"][2],
            row["normal_p90_absolute_log_evidence_error_percentiles"][2],
            color=colors[row["epsilon"]],
            marker=markers[row["n_candidates"]],
            alpha=0.65,
            s=24,
        )
    best = payload["best_group"]
    axes[1, 1].scatter(
        best["proposal_runtime_seconds_percentiles"][2],
        best["normal_p90_absolute_log_evidence_error_percentiles"][2],
        facecolor="none",
        edgecolor="black",
        linewidth=1.2,
        s=90,
        label="Best robust group",
    )
    axes[1, 1].set_xscale("log")
    axes[1, 1].set_yscale("log")
    axes[1, 1].axhline(0.05, color="0.45", linestyle="--", linewidth=0.8)
    axes[1, 1].set_xlabel("Warm proposal runtime per observation (s)")
    axes[1, 1].set_ylabel(r"Normal p90 $|\Delta\log Z|$ at M=16k")
    axes[1, 1].legend(frameon=False, fontsize=7)
    axes[1, 1].text(
        0.98,
        0.04,
        "color: epsilon = 0.1 / 0.2 / 0.3\nmarker: K = 32k / 64k",
        transform=axes[1, 1].transAxes,
        ha="right",
        va="bottom",
        fontsize=7,
    )

    for label, axis in zip("ABCD", axes.flat):
        axis.text(-0.12, 1.04, label, transform=axis.transAxes, fontweight="bold", fontsize=10)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(labelsize=7)
    figure.suptitle(
        f"Observation {payload['object_id']:,}: diversified proposal screen",
        fontsize=11,
    )
    png = output / "proposal_diversification.png"
    pdf = output / "proposal_diversification.pdf"
    figure.savefig(png, dpi=300, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    return png, pdf


def main(argv=None):
    args = _parse_args(argv)
    reference_path = Path(args.reference_result).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}")
    k_values = sorted({int(value) for value in args.k})
    epsilons = sorted({float(value) for value in args.epsilons})
    if not k_values or k_values[0] <= 0 or k_values[-1] > args.score_pool:
        raise RuntimeError("K values lie outside the score pool")
    if not all(0 < value <= 1 for value in epsilons):
        raise RuntimeError("epsilons must lie in (0, 1]")
    config, identity, likelihood, proposal, mock, metadata = _load_pipeline(
        reference_path, args.device
    )
    object_id = int(args.object_id)
    if not 0 <= object_id < len(mock.measurements):
        raise RuntimeError("object id lies outside the frozen mock")
    observed = mock.measurements.iloc[[object_id]].reset_index(drop=True)
    center = identity["initial_center"]["center"]

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
        object_id=object_id,
        center=center,
        n_candidates=int(args.score_pool),
        prefilter_candidates=int(args.score_prefilter),
        epsilon=float(config["proposal_epsilon"]),
        candidate_backend=config["candidate_backend"],
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

    supports: list[dict] = []

    def add_support(indices: np.ndarray, **fields):
        positions = _candidate_positions(active, np.asarray(indices, dtype=np.int64))
        supports.append({"positions": positions, **fields})

    for k in k_values:
        add_support(
            ranked[:k],
            method_key="top_deep",
            method_label="Deep top-K",
            family="top",
            n_candidates=k,
            core_fraction=None,
            temperature=None,
            query_count=None,
            candidate_seed=None,
            warm_runtime_seconds=0.0,
            union_prefill_fraction=1.0,
            runtime_basis="shared deep-score pool",
        )
        for core_fraction in args.core_fractions:
            for method, temperature in (
                [("log_stratified", None)]
                + [("tempered", float(value)) for value in args.temperatures]
            ):
                for seed in args.candidate_seeds:
                    start = perf_counter()
                    indices = select_score_diversified_candidates(
                        ranked,
                        score_gap,
                        n_candidates=k,
                        core_fraction=float(core_fraction),
                        tail_method=method,
                        temperature=temperature,
                        seed=int(seed),
                        n_log_strata=int(args.log_strata),
                    )
                    elapsed = perf_counter() - start
                    fraction_label = f"{int(round(100 * core_fraction))}/{int(round(100 * (1-core_fraction)))}"
                    method_key = (
                        f"log_{fraction_label}"
                        if method == "log_stratified"
                        else f"temp{temperature:g}_{fraction_label}"
                    )
                    method_label = (
                        f"Log tail {fraction_label}"
                        if method == "log_stratified"
                        else f"T={temperature:g} {fraction_label}"
                    )
                    add_support(
                        indices,
                        method_key=method_key,
                        method_label=method_label,
                        family=method,
                        n_candidates=k,
                        core_fraction=float(core_fraction),
                        temperature=temperature,
                        query_count=None,
                        candidate_seed=int(seed),
                        warm_runtime_seconds=elapsed,
                        union_prefill_fraction=1.0,
                        runtime_basis="shared deep-score pool",
                    )

    start = perf_counter()
    proposal._build_uncertainty_mips_tree()
    uncertainty_index_seconds = perf_counter() - start
    local_scale = np.median(
        proposal.coordinates.dispersion[ranked[:4096]], axis=0
    )
    central_fallback: dict[int, tuple[np.ndarray, float]] = {}
    for k in k_values:
        start = perf_counter()
        central = proposal.uncertainty_candidates(observed, n_candidates=k)
        elapsed = perf_counter() - start
        central_fallback[k] = (central.indices[0], elapsed)
        add_support(
            central.indices[0],
            method_key="top_global",
            method_label="Global uncertainty top-K",
            family="global_top",
            n_candidates=k,
            core_fraction=None,
            temperature=None,
            query_count=1,
            candidate_seed=None,
            warm_runtime_seconds=elapsed,
            union_prefill_fraction=1.0,
            runtime_basis="cached uncertainty index",
        )
    for k in k_values:
        fallback, fallback_seconds = central_fallback[k]
        for n_queries in args.query_counts:
            per_query = int(k // int(n_queries))
            for seed in args.candidate_seeds:
                start = perf_counter()
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
                indices, prefill = union_candidate_sets(
                    queried.indices,
                    n_candidates=k,
                    fallback_indices=fallback,
                )
                elapsed = perf_counter() - start + fallback_seconds
                add_support(
                    indices,
                    method_key=f"union_q{int(n_queries)}",
                    method_label=f"QMC union Q={int(n_queries)}",
                    family="query_union",
                    n_candidates=k,
                    core_fraction=None,
                    temperature=None,
                    query_count=int(n_queries),
                    candidate_seed=int(seed),
                    warm_runtime_seconds=elapsed,
                    union_prefill_fraction=float(prefill / k),
                    runtime_basis="cached uncertainty index",
                )

    component, global_position, local_uniform = _common_mc(
        prior,
        n_replicates=int(args.mc_replicates),
        n_draws=int(args.mc_draws),
        seed=int(args.mc_seed),
    )
    settings = []
    metric_start = perf_counter()
    for support_id, support in enumerate(supports):
        for epsilon in epsilons:
            exact = summarize_defensive_proposal(
                target, prior, support["positions"], epsilon
            )
            mc = simulate_defensive_evidence_errors(
                target,
                prior,
                support["positions"],
                epsilon,
                component_uniform=component,
                global_positions=global_position,
                local_uniform=local_uniform,
                ladder=(int(args.mc_draws),),
            )[0]
            relative_se = float(
                np.sqrt(exact["chi_square_target_vs_proposal"] / args.mc_draws)
            )
            row = {
                "support_id": support_id,
                **{key: value for key, value in support.items() if key != "positions"},
                **exact,
                "epsilon": epsilon,
                "proposal_runtime_seconds": float(
                    support["warm_runtime_seconds"]
                    + (score_pool_seconds if support["runtime_basis"] == "shared deep-score pool" else 0.0)
                ),
                "projected_relative_evidence_se": relative_se,
                "normal_p90_absolute_log_evidence_error": float(
                    1.6448536269514722 * relative_se
                ),
                "empirical_p90_absolute_log_evidence_error": float(
                    mc["absolute_log_evidence_error_percentiles"][1]
                ),
                "mc": mc,
            }
            settings.append(row)
    metric_seconds = perf_counter() - metric_start
    aggregate = _aggregate(settings)
    best_group = min(
        aggregate,
        key=lambda row: (
            row["normal_p90_absolute_log_evidence_error_percentiles"][3],
            row["n_candidates"],
            row["proposal_runtime_seconds_percentiles"][2],
        ),
    )

    hold_component, hold_global, hold_local = _common_mc(
        prior,
        n_replicates=int(args.mc_replicates),
        n_draws=int(args.mc_draws),
        seed=int(args.holdout_mc_seed),
    )
    holdout = []
    for row in settings:
        same_group = all(
            row.get(field) == best_group.get(field)
            for field in (
                "method_key",
                "n_candidates",
                "core_fraction",
                "temperature",
                "query_count",
                "epsilon",
            )
        )
        if not same_group:
            continue
        support = supports[row["support_id"]]
        mc = simulate_defensive_evidence_errors(
            target,
            prior,
            support["positions"],
            row["epsilon"],
            component_uniform=hold_component,
            global_positions=hold_global,
            local_uniform=hold_local,
            ladder=(int(args.mc_draws),),
        )[0]
        holdout.append(
            {
                "candidate_seed": row["candidate_seed"],
                "mean_relative_evidence": mc["mean_relative_evidence"],
                "log_evidence_error_percentiles": mc[
                    "log_evidence_error_percentiles"
                ],
                "absolute_log_evidence_error_percentiles": mc[
                    "absolute_log_evidence_error_percentiles"
                ],
                "fraction_within_0p05": mc["fraction_within_0p05"],
            }
        )

    payload = {
        "status": "complete",
        "object_id": object_id,
        "exact_log_evidence": float(
            logsumexp(comparison.log_target)
            - comparison.population_log_normalization
        ),
        "design": {
            "k": k_values,
            "core_fractions": list(map(float, args.core_fractions)),
            "tail_methods": ["log_stratified"]
            + [f"tempered_T{value:g}" for value in args.temperatures],
            "candidate_seeds": list(map(int, args.candidate_seeds)),
            "epsilons": epsilons,
            "query_counts": list(map(int, args.query_counts)),
            "score_pool": int(args.score_pool),
            "score_prefilter": int(args.score_prefilter),
            "mc_replicates": int(args.mc_replicates),
            "mc_draws": int(args.mc_draws),
            "mc_seed": int(args.mc_seed),
            "holdout_mc_seed": int(args.holdout_mc_seed),
            "common_random_numbers": True,
            "exact_target_used_for_selection": False,
            "infer_v1_defaults_changed": False,
        },
        "runtime": {
            "deep_score_pool_seconds": float(score_pool_seconds),
            "exact_target_evaluation_seconds": float(exact_seconds),
            "uncertainty_index_build_seconds": float(uncertainty_index_seconds),
            "all_exact_and_mc_metrics_seconds": float(metric_seconds),
        },
        "n_supports": len(supports),
        "n_settings": len(settings),
        "best_group": best_group,
        "holdout_best_group": holdout,
        "aggregate": aggregate,
        "settings": settings,
        "metadata": metadata,
    }
    output.mkdir(parents=True, exist_ok=False)
    result = output / "result.json"
    result.write_text(json.dumps(payload, indent=2) + "\n")
    pd.DataFrame(settings).drop(columns=["mc"]).to_csv(
        output / "settings.csv", index=False
    )
    pd.DataFrame(aggregate).to_csv(output / "aggregate.csv", index=False)
    figures = _plot(payload, output)
    print(
        json.dumps(
            {
                "status": "complete",
                "result": str(result),
                "n_supports": len(supports),
                "n_settings": len(settings),
                "best_group": best_group,
                "figures": [str(path) for path in figures],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
