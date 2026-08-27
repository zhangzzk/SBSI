#!/usr/bin/env python
"""Visualize nested Infer V1 importance draws for representative observations."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from sbsi.catalogue_closure import MockCatalogue
from sbsi.catalogue_likelihood import CatalogueLikelihood, CatalogueModelCache
from sbsi.catalogue_sampling import DefensiveLocalProposal, ProposalCoordinateTable
from sbsi.measurement_model import load_measurement_model
from sbsi.models import ModelPaths, load_emulator
from sbsi.sampling_diagnostics import (
    evaluate_exact_proposal_target,
    evaluate_importance_sampling,
    optimize_defensive_epsilon,
    plot_exact_proposal_target,
    plot_importance_sampling_diagnostic,
    save_exact_proposal_target,
    save_importance_sampling_diagnostic,
    select_example_rows,
    simulate_defensive_evidence_errors,
    summarize_defensive_proposal,
    summarize_importance_sampling,
)
from sbsi.scene_prior import ScenePrior


def _sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_hashes(root: str | Path, expected: dict[str, str], label: str) -> None:
    root = Path(root)
    actual = {name: _sha256(root / name) for name in expected}
    if actual != expected:
        raise RuntimeError(f"{label} hashes do not match the reference result")


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-result", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pool-size", type=int, default=64)
    parser.add_argument("--pool-seed", type=int, default=9101)
    parser.add_argument("--examples", type=int, default=3, choices=(1, 2, 3))
    parser.add_argument(
        "--exact-object-id",
        type=int,
        default=None,
        help="scan every active prior atom for this one observation instead of a pool",
    )
    parser.add_argument(
        "--proposal-sweep",
        action="store_true",
        help="compare nested K and epsilon settings against the exact target",
    )
    parser.add_argument(
        "--proposal-k-ladder",
        nargs="+",
        type=int,
        default=(16384, 32768, 65536, 131072, 262144, 524288),
    )
    parser.add_argument("--proposal-prefilter", type=int, default=4194304)
    parser.add_argument(
        "--proposal-epsilon-ladder",
        nargs="+",
        type=float,
        default=(0.1, 0.2, 0.3, 0.4, 0.6, 1.0),
    )
    parser.add_argument(
        "--proposal-m-ladder",
        nargs="+",
        type=int,
        default=(4096, 8192, 16384),
    )
    parser.add_argument("--proposal-replicates", type=int, default=256)
    parser.add_argument("--proposal-mc-seed", type=int, default=9917)
    parser.add_argument(
        "--ladder", nargs="+", type=int, default=(4096, 8192, 16384)
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def _rung_pool_summary(metrics: list[dict], ladder: tuple[int, ...]) -> dict:
    summary = {}
    fields = (
        "ess",
        "ess_fraction",
        "max_weight_fraction",
        "outside_local_evidence_fraction",
        "global_draw_evidence_fraction",
        "unique_atoms",
    )
    for n_draws in ladder:
        rows = [row for row in metrics if row["n_draws"] == n_draws]
        summary[str(n_draws)] = {
            f"{field}_percentiles": np.percentile(
                [row[field] for row in rows], [0, 10, 50, 90, 100]
            ).tolist()
            for field in fields
        }
    return summary


def _candidate_positions(active: np.ndarray, atom_indices: np.ndarray) -> np.ndarray:
    positions = np.searchsorted(active, np.asarray(atom_indices, dtype=np.int64))
    if (
        (positions >= len(active)).any()
        or not np.array_equal(active[positions], atom_indices)
    ):
        raise RuntimeError("candidate atoms are not contained in active prior support")
    return positions


def _plot_proposal_sweep(payload: dict, output: Path) -> tuple[Path, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = payload["settings"]
    k_ladder = payload["design"]["k_ladder"]
    epsilon_ladder = payload["design"]["epsilon_ladder"]
    m_max = max(payload["design"]["m_ladder"])
    deep = [row for row in rows if row["ranking"] == "deep_gaussian"]
    current = [row for row in rows if row["ranking"] == "production"]
    colors = plt.get_cmap("viridis")(
        np.linspace(0.12, 0.9, len(k_ladder))
    )
    figure, axes = plt.subplots(2, 2, figsize=(8.2, 6.4), constrained_layout=True)

    optimized = [row for row in deep if row["epsilon_kind"] == "optimized"]
    optimized.sort(key=lambda row: row["n_candidates"])
    axes[0, 0].semilogx(
        k_ladder,
        [payload["ideal_target_mass_by_k"][str(k)] for k in k_ladder],
        "--",
        color="0.25",
        marker="o",
        label="Ideal exact-target rank",
    )
    axes[0, 0].semilogx(
        [row["n_candidates"] for row in optimized],
        [row["candidate_target_mass"] for row in optimized],
        color="#0072B2",
        marker="o",
        label="Deep Gaussian rank",
    )
    if current:
        axes[0, 0].scatter(
            [current[0]["n_candidates"]],
            [current[0]["candidate_target_mass"]],
            color="#D55E00",
            marker="s",
            label="Infer V1 production",
            zorder=4,
        )
    axes[0, 0].set_xlabel("Candidate count K")
    axes[0, 0].set_ylabel("Exact target mass captured")
    axes[0, 0].set_ylim(0, 1.01)
    axes[0, 0].legend(frameon=False, fontsize=7)

    for color, k in zip(colors, k_ladder):
        selected = [
            row
            for row in deep
            if row["n_candidates"] == k and row["epsilon_kind"] == "fixed"
        ]
        selected.sort(key=lambda row: row["epsilon"])
        axes[0, 1].plot(
            [row["epsilon"] for row in selected],
            [100 * row["asymptotic_ess_fraction"] for row in selected],
            color=color,
            marker="o",
            markersize=3,
            label=f"K={k // 1024}k",
        )
        best = next(
            row
            for row in optimized
            if row["n_candidates"] == k
        )
        axes[0, 1].scatter(
            best["epsilon"],
            100 * best["asymptotic_ess_fraction"],
            color=color,
            edgecolor="black",
            linewidth=0.4,
            s=24,
            zorder=4,
        )
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_xlabel(r"Defensive fraction $\epsilon$")
    axes[0, 1].set_ylabel("Exact ESS/M (%)")
    axes[0, 1].legend(frameon=False, fontsize=6, ncol=2)

    heat = np.full((len(k_ladder), len(epsilon_ladder)), np.nan)
    for row in deep:
        if row["epsilon_kind"] != "fixed":
            continue
        k_index = k_ladder.index(row["n_candidates"])
        epsilon_index = epsilon_ladder.index(row["epsilon"])
        heat[k_index, epsilon_index] = row[
            "normal_p90_absolute_log_evidence_error_at_max_m"
        ]
    image = axes[1, 0].imshow(heat, aspect="auto", cmap="cividis_r")
    axes[1, 0].set_xticks(range(len(epsilon_ladder)), epsilon_ladder)
    axes[1, 0].set_yticks(
        range(len(k_ladder)), [f"{k // 1024}k" for k in k_ladder]
    )
    axes[1, 0].set_xlabel(r"Defensive fraction $\epsilon$")
    axes[1, 0].set_ylabel("Candidate count K")
    colorbar = figure.colorbar(image, ax=axes[1, 0], shrink=0.9)
    colorbar.set_label(r"Exact-variance normal p90 $|\Delta\log Z|$ at M=16k")

    x = np.asarray([row["n_candidates"] for row in optimized])
    quantiles = np.asarray(
        [
            next(item for item in row["mc"] if item["n_draws"] == m_max)[
                "log_evidence_error_percentiles"
            ]
            for row in optimized
        ]
    )
    axes[1, 1].fill_between(
        x,
        quantiles[:, 1],
        quantiles[:, 3],
        color="#56B4E9",
        alpha=0.35,
        label="p10–p90",
    )
    axes[1, 1].semilogx(
        x,
        quantiles[:, 2],
        color="#0072B2",
        marker="o",
        label="Median",
    )
    axes[1, 1].axhline(0, color="0.25", linewidth=1)
    axes[1, 1].axhline(0.05, color="0.5", linestyle="--", linewidth=0.8)
    axes[1, 1].axhline(-0.05, color="0.5", linestyle="--", linewidth=0.8)
    axes[1, 1].set_xlabel("Candidate count K")
    axes[1, 1].set_ylabel(r"$\Delta\log Z$ at optimized $\epsilon$, M=16k")
    axes[1, 1].legend(frameon=False, fontsize=7)

    for label, axis in zip("ABCD", axes.flat):
        axis.text(
            -0.14,
            1.04,
            label,
            transform=axis.transAxes,
            fontweight="bold",
            fontsize=10,
            va="top",
        )
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(labelsize=7)
    figure.suptitle(
        (
            f"Observation {payload['object_id']:,}: paired proposal search "
            f"against exact log Z={payload['exact_log_evidence']:.4f}"
        ),
        fontsize=10,
    )
    png = output / "proposal_sweep.png"
    pdf = output / "proposal_sweep.pdf"
    figure.savefig(png, dpi=300, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    return png, pdf


def _run_proposal_sweep(
    *,
    comparison,
    likelihood,
    proposal,
    observed,
    config: dict,
    args,
    output: Path,
    metadata: dict,
) -> dict:
    k_ladder = sorted({int(value) for value in args.proposal_k_ladder})
    epsilon_ladder = sorted({float(value) for value in args.proposal_epsilon_ladder})
    m_ladder = sorted({int(value) for value in args.proposal_m_ladder})
    if (
        not k_ladder
        or k_ladder[0] <= 0
        or k_ladder[-1] > len(comparison.candidate_indices)
    ):
        raise RuntimeError("proposal K ladder lies outside the deep candidate support")
    if not epsilon_ladder or not all(0 < value <= 1 for value in epsilon_ladder):
        raise RuntimeError("proposal epsilons must lie in (0, 1]")
    if not m_ladder or m_ladder[0] <= 0 or args.proposal_replicates <= 1:
        raise RuntimeError("proposal MC ladder and replicate count must be positive")

    active = comparison.atom_indices
    target = comparison.target_probability
    prior = likelihood.cache.prior.weights[active].astype(np.float64, copy=True)
    prior /= prior.sum()
    deep_position = _candidate_positions(active, comparison.candidate_indices)
    production_candidates = proposal.candidates(
        observed,
        n_candidates=int(config["proposal_candidates"]),
        prefilter_candidates=config["proposal_prefilter_candidates"],
        torch_device=likelihood.flow_model.device,
    )
    production_position = _candidate_positions(
        active, production_candidates.indices[0]
    )

    rng = np.random.default_rng(int(args.proposal_mc_seed))
    shape = (int(args.proposal_replicates), m_ladder[-1])
    component_uniform = rng.random(shape)
    global_uniform = rng.random(shape)
    local_uniform = rng.random(shape)
    prior_cdf = np.cumsum(prior)
    prior_cdf[-1] = 1.0
    global_position = np.searchsorted(prior_cdf, global_uniform, side="right")

    settings = []

    def add_setting(ranking: str, positions: np.ndarray, epsilon: float, kind: str):
        exact = summarize_defensive_proposal(target, prior, positions, epsilon)
        mc = simulate_defensive_evidence_errors(
            target,
            prior,
            positions,
            epsilon,
            component_uniform=component_uniform,
            global_positions=global_position,
            local_uniform=local_uniform,
            ladder=m_ladder,
        )
        relative_se = float(
            np.sqrt(exact["chi_square_target_vs_proposal"] / m_ladder[-1])
        )
        normal_p90 = float(1.6448536269514722 * relative_se)
        settings.append(
            {
                "ranking": ranking,
                "epsilon_kind": kind,
                **exact,
                "mc": mc,
                "projected_relative_evidence_se_at_max_m": relative_se,
                "normal_p90_absolute_log_evidence_error_at_max_m": normal_p90,
                "passes_m16k_evidence_goal": bool(normal_p90 <= 0.05),
            }
        )

    for epsilon in epsilon_ladder:
        add_setting("production", production_position, epsilon, "fixed")
    optimum, _ = optimize_defensive_epsilon(target, prior, production_position)
    add_setting("production", production_position, optimum, "optimized")

    for k in k_ladder:
        positions = deep_position[:k]
        for epsilon in epsilon_ladder:
            add_setting("deep_gaussian", positions, epsilon, "fixed")
        optimum, _ = optimize_defensive_epsilon(target, prior, positions)
        add_setting("deep_gaussian", positions, optimum, "optimized")

    ideal_order = np.argsort(-target, kind="stable")
    successful = [row for row in settings if row["passes_m16k_evidence_goal"]]
    if successful:
        best = min(
            successful,
            key=lambda row: (
                row["n_candidates"],
                -row["asymptotic_ess_fraction"],
            ),
        )
    else:
        best = min(
            settings,
            key=lambda row: row[
                "normal_p90_absolute_log_evidence_error_at_max_m"
            ],
        )
    payload = {
        "status": "complete",
        "object_id": comparison.object_id,
        "exact_log_evidence": float(
            logsumexp(comparison.log_target)
            - comparison.population_log_normalization
        ),
        "design": {
            "k_ladder": k_ladder,
            "deep_prefilter": int(args.proposal_prefilter),
            "epsilon_ladder": epsilon_ladder,
            "m_ladder": m_ladder,
            "replicates": int(args.proposal_replicates),
            "mc_seed": int(args.proposal_mc_seed),
            "success_rule": "exact-variance normal p90 |Delta log Z| <= 0.05 at max M",
            "common_random_numbers": True,
        },
        "ideal_target_mass_by_k": {
            str(k): float(target[ideal_order[:k]].sum()) for k in k_ladder
        },
        "settings": settings,
        "best_setting": best,
        "metadata": metadata,
    }
    output.mkdir(parents=True, exist_ok=False)
    result = output / "result.json"
    result.write_text(json.dumps(payload, indent=2) + "\n")
    figures = _plot_proposal_sweep(payload, output)
    return {
        "status": "complete",
        "object_id": comparison.object_id,
        "result": str(result),
        "figures": [str(path) for path in figures],
        "best_setting": best,
    }


def main(argv=None):
    args = _parse_args(argv)
    reference_path = Path(args.reference_result).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}")
    if args.proposal_sweep and args.exact_object_id is None:
        raise SystemExit("--proposal-sweep requires --exact-object-id")
    if args.exact_object_id is None and args.pool_size < args.examples:
        raise SystemExit("--pool-size must be at least --examples")
    ladder = tuple(sorted({int(value) for value in args.ladder}))
    if not ladder or ladder[0] <= 0:
        raise SystemExit("--ladder must contain positive draw counts")

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
        raise RuntimeError(
            "this focused diagnostic currently requires R_blend=0 and no measured cut"
        )

    _verify_hashes(config["scene_store"], identity["scene_sha256"], "scene")
    _verify_hashes(
        config["model_cache"], identity["model_cache_sha256"], "model cache"
    )
    _verify_hashes(
        config["proposal_cache"],
        identity["proposal_cache_sha256"],
        "proposal cache",
    )
    _verify_hashes(
        config["mock_input"], identity["mock_input_sha256"], "likelihood mock"
    )
    model_files = {
        "measurement": config["measurement_model"],
        "emulator": config["emulator_model"],
        "emulator_metadata": config["emulator_metadata"],
    }
    actual_model_hashes = {
        name: _sha256(path) for name, path in model_files.items()
    }
    if actual_model_hashes != identity["model_sha256"]:
        raise RuntimeError("model hashes do not match the reference result")

    conditions = {
        name: config[name]
        for name in (
            "pixel_size",
            "zero_point",
            "psf_fwhm",
            "moffat_beta",
            "pixel_rms",
        )
    }
    prior = ScenePrior.load(config["scene_store"])
    flow = load_measurement_model(config["measurement_model"], device=args.device)
    detector = load_emulator(
        ModelPaths(
            flow_checkpoints=(Path(config["measurement_model"]),),
            emulator_model=Path(config["emulator_model"]),
            emulator_metadata=Path(config["emulator_metadata"]),
        ),
        conditions=conditions,
        device=args.device,
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
    pool_size = min(int(args.pool_size), len(mock.measurements))
    rng = np.random.default_rng(int(args.pool_seed))
    pool_ids = np.sort(
        rng.choice(len(mock.measurements), size=pool_size, replace=False)
    ).astype(np.int64)
    center = identity["initial_center"]["center"]
    if config.get("compile_flow", False):
        flow.compile_log_prob(mode=None, dynamic=True)

    common_metadata = {
        "inference_version": config["inference_version"],
        "reference_result": str(reference_path),
        "reference_result_sha256": _sha256(reference_path),
        "sampling_diagnostic_sha256": _sha256(Path(__file__)),
        "mock_input": config["mock_input"],
        "mock_input_sha256": identity["mock_input_sha256"],
        "model_sha256": identity["model_sha256"],
        "model_cache_sha256": identity["model_cache_sha256"],
        "proposal_cache_sha256": identity["proposal_cache_sha256"],
        "reference_implementation_sha256": identity["implementation_sha256"],
    }
    if args.exact_object_id is not None:
        object_id = int(args.exact_object_id)
        if not 0 <= object_id < len(mock.measurements):
            raise RuntimeError("--exact-object-id lies outside the saved mock")
        exact_candidates = (
            max(args.proposal_k_ladder)
            if args.proposal_sweep
            else int(config["proposal_candidates"])
        )
        exact_prefilter = (
            int(args.proposal_prefilter)
            if args.proposal_sweep
            else config["proposal_prefilter_candidates"]
        )
        if exact_prefilter is not None and exact_prefilter < exact_candidates:
            raise RuntimeError("proposal prefilter must be at least the largest K")
        comparison = evaluate_exact_proposal_target(
            likelihood,
            mock.measurements.iloc[[object_id]],
            proposal,
            object_id=object_id,
            center=center,
            n_candidates=exact_candidates,
            prefilter_candidates=exact_prefilter,
            epsilon=float(config["proposal_epsilon"]),
            candidate_backend=config["candidate_backend"],
            atom_chunk=65536,
        )
        if args.proposal_sweep:
            result = _run_proposal_sweep(
                comparison=comparison,
                likelihood=likelihood,
                proposal=proposal,
                observed=mock.measurements.iloc[[object_id]],
                config=config,
                args=args,
                output=output,
                metadata={
                    **common_metadata,
                    "target": "normalized pi_j Pdet_j L_ij over all active atoms",
                    "deep_ranking": "Gaussian mean/std reranking within one fixed deep location prefilter",
                    "ideal_ranking": "diagnostic upper bound from exact target order; not implementable",
                },
            )
            print(json.dumps(result, indent=2))
            return
        result = save_exact_proposal_target(
            comparison,
            output,
            metadata={
                **common_metadata,
                "target": "normalized pi_j Pdet_j L_ij over all active atoms",
                "proposal": "epsilon*pi + (1-epsilon)*candidate-local target",
            },
        )
        figures = plot_exact_proposal_target(comparison, output)
        print(
            json.dumps(
                {
                    "status": "complete",
                    "object_id": object_id,
                    "result": str(result),
                    "figures": [str(path) for path in figures],
                },
                indent=2,
            )
        )
        return

    pool = evaluate_importance_sampling(
        likelihood,
        mock.measurements.iloc[pool_ids],
        proposal,
        object_ids=pool_ids,
        center=center,
        n_draws=ladder[-1],
        n_candidates=int(config["proposal_candidates"]),
        prefilter_candidates=config["proposal_prefilter_candidates"],
        epsilon=float(config["proposal_epsilon"]),
        proposal_seed=int(config["proposal_seed"]),
        candidate_backend=config["candidate_backend"],
        object_chunk=int(config["object_chunk"]),
        atom_chunk=int(config["atom_chunk"]),
    )
    pool_metrics = summarize_importance_sampling(pool, ladder)
    selected_rows, labels = select_example_rows(pool)
    selected_rows = selected_rows[: args.examples]
    labels = labels[: args.examples]
    selected = pool.take(selected_rows)
    save_importance_sampling_diagnostic(
        selected,
        output,
        ladder=ladder,
        labels=labels,
        pool_summary={
            "n_objects": int(pool_size),
            "pool_seed": int(args.pool_seed),
            "percentile_order": [0, 10, 50, 90, 100],
            "rungs": _rung_pool_summary(pool_metrics, ladder),
        },
        metadata={
            **common_metadata,
            "conditional_log_likelihood": "flow-only log p(x_i | z_j, g_center)",
            "importance_weight": "pi_j Pdet_j L_ij / q_i(j)",
        },
    )
    figures = plot_importance_sampling_diagnostic(
        selected,
        output,
        ladder=ladder,
        labels=labels,
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "pool_size": int(pool_size),
                "object_ids": selected.object_ids.tolist(),
                "labels": list(labels),
                "figures": [str(path) for path in figures],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
