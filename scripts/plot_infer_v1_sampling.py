#!/usr/bin/env python
"""Visualize nested Infer V1 importance draws for representative observations."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from sbsi.catalogue_closure import MockCatalogue
from sbsi.catalogue_likelihood import CatalogueLikelihood, CatalogueModelCache
from sbsi.catalogue_sampling import DefensiveLocalProposal, ProposalCoordinateTable
from sbsi.measurement_model import load_measurement_model
from sbsi.models import ModelPaths, load_emulator
from sbsi.sampling_diagnostics import (
    evaluate_exact_proposal_target,
    evaluate_importance_sampling,
    plot_exact_proposal_target,
    plot_importance_sampling_diagnostic,
    save_exact_proposal_target,
    save_importance_sampling_diagnostic,
    select_example_rows,
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


def main(argv=None):
    args = _parse_args(argv)
    reference_path = Path(args.reference_result).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}")
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
        comparison = evaluate_exact_proposal_target(
            likelihood,
            mock.measurements.iloc[[object_id]],
            proposal,
            object_id=object_id,
            center=center,
            n_candidates=int(config["proposal_candidates"]),
            prefilter_candidates=config["proposal_prefilter_candidates"],
            epsilon=float(config["proposal_epsilon"]),
            candidate_backend=config["candidate_backend"],
            atom_chunk=65536,
        )
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
