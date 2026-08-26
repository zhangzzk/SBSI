#!/usr/bin/env python
# Archived combined null-validation experiment.
"""Run the streamed two-component Section 5 catalogue-prior null closure."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import torch

from sbsi.catalogue_closure import generate_mock_catalogue
from sbsi.catalogue_likelihood import CatalogueLikelihood, CatalogueModelCache
from sbsi.catalogue_null import (
    assess_section5_null,
    run_exact_section5,
    run_streamed_section5,
)
from sbsi.catalogue_sampling import DefensiveLocalProposal, ProposalCoordinateTable
from sbsi.measurement_model import load_measurement_model
from sbsi.models import ModelPaths, load_emulator
from sbsi.scene_prior import ScenePrior


def _sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _csv(text, cast):
    return tuple(cast(value) for value in str(text).split(",") if value.strip())


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-store", required=True)
    parser.add_argument("--measurement-model", required=True)
    parser.add_argument("--emulator-metadata", required=True)
    parser.add_argument("--emulator-model", required=True)
    parser.add_argument("--model-cache", required=True)
    parser.add_argument("--proposal-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", default="0.005,0.01,0.02")
    parser.add_argument("--importance-ladder", default="8192,32768,65536")
    parser.add_argument("--proposal-candidates", type=int, default=131072)
    parser.add_argument("--proposal-seeds", default="8701,8702")
    parser.add_argument("--proposal-epsilon", type=float, default=0.1)
    parser.add_argument("--proposal-bandwidth", type=float, default=1.0)
    parser.add_argument(
        "--proposal-targets",
        default=(
            "measured_ngmix_g1,measured_ngmix_g2,measured_mag_auto,"
            "measured_log_flux_radius"
        ),
    )
    parser.add_argument("--proposal-flow-samples", type=int, default=16)
    parser.add_argument("--proposal-statistic", choices=("mean", "median"), default="median")
    parser.add_argument("--proposal-row-chunk", type=int, default=4096)
    parser.add_argument("--proposal-coordinate-seed", type=int, default=8201)
    parser.add_argument("--n-detected", type=int, default=1024)
    parser.add_argument("--scene-seed", type=int, default=1901)
    parser.add_argument("--detection-seed", type=int, default=2901)
    parser.add_argument("--flow-seed", type=int, default=3901)
    parser.add_argument("--oracle-atoms", type=int, default=4096)
    parser.add_argument("--oracle-n-detected", type=int, default=256)
    parser.add_argument("--oracle-seed", type=int, default=4901)
    parser.add_argument("--bank-column", default="property_seed")
    parser.add_argument("--max-independent-banks", type=int, default=3)
    parser.add_argument("--powered-n-detected", type=int, default=10000)
    parser.add_argument("--run-powered-on-pass", action="store_true")
    parser.add_argument("--object-chunk", type=int, default=16)
    parser.add_argument("--atom-chunk", type=int, default=4096)
    parser.add_argument("--detection-radius-arcsec", type=float, default=3.0)
    parser.add_argument("--flow-neighbour-radius-arcsec", type=float, default=7.0)
    parser.add_argument("--crowding-near-arcsec", type=float, default=3.0)
    parser.add_argument("--crowding-far-arcsec", type=float, default=7.0)
    parser.add_argument("--pixel-size", type=float, default=0.2)
    parser.add_argument("--zero-point", type=float, default=30.0)
    parser.add_argument("--psf-fwhm", type=float, default=0.73)
    parser.add_argument("--moffat-beta", type=float, default=2.224)
    parser.add_argument("--pixel-rms", type=float, default=0.312)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args(argv)


def _proposal(
    likelihood,
    coordinates,
    *,
    prior_weights=None,
):
    weights = likelihood.cache.prior.weights if prior_weights is None else prior_weights
    return DefensiveLocalProposal(
        coordinates,
        weights,
        local_base_weights=likelihood.cache.get(0.0, 0.0).detection_probability,
    )


def main(argv=None):
    args = parse_args(argv)
    output = Path(args.output).resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    steps = _csv(args.steps, float)
    ladder = _csv(args.importance_ladder, int)
    proposal_seeds = _csv(args.proposal_seeds, int)
    proposal_targets = _csv(args.proposal_targets, str)
    if 0.005 not in steps:
        raise SystemExit("--steps must include the primary h=0.005 stencil")
    if len(ladder) < 3 or tuple(sorted(ladder)) != (8192, 32768, 65536):
        raise SystemExit("validation ladder must be exactly 8192,32768,65536")
    if len(proposal_seeds) != 2:
        raise SystemExit("exactly two independent proposal seeds are required")

    conditions = {
        "pixel_size": args.pixel_size,
        "zero_point": args.zero_point,
        "psf_fwhm": args.psf_fwhm,
        "moffat_beta": args.moffat_beta,
        "pixel_rms": args.pixel_rms,
    }
    prior = ScenePrior.load(args.scene_store)
    flow = load_measurement_model(args.measurement_model, device=args.device)
    model_paths = ModelPaths(
        flow_checkpoints=(Path(args.measurement_model),),
        emulator_model=Path(args.emulator_model),
        emulator_metadata=Path(args.emulator_metadata),
    )
    detector = load_emulator(model_paths, conditions=conditions, device=args.device)
    model_hashes = {
        "measurement": _sha256(args.measurement_model),
        "emulator": _sha256(args.emulator_model),
        "emulator_metadata": _sha256(args.emulator_metadata),
    }
    scene_root = Path(args.scene_store)
    scene_hashes = {
        name: _sha256(scene_root / name)
        for name in ("manifest.json", "galaxies.parquet", "neighbours.npz")
    }
    cache_identity = {
        "purpose": "section5_null_shape_only_rblend0",
        "model_sha256": model_hashes,
        "scene_sha256": scene_hashes,
        "flow_features": list(flow.condition_preprocessor.feature_names),
        "selection": None,
        "r_blend": 0.0,
    }
    cache_path = Path(args.model_cache)
    if (cache_path / "manifest.json").is_file():
        cache = CatalogueModelCache.load(cache_path, prior=prior)
        if cache.metadata != cache_identity:
            raise RuntimeError("model cache provenance does not match this null closure")
        if tuple(cache.flow_features or ()) != tuple(flow.condition_preprocessor.feature_names):
            raise RuntimeError("model cache flow features do not match the checkpoint")
        expected_geometry = (
            conditions,
            args.detection_radius_arcsec,
            args.flow_neighbour_radius_arcsec,
            (args.crowding_near_arcsec, args.crowding_far_arcsec),
        )
        actual_geometry = (
            cache.conditions,
            cache.detection_radius_arcsec,
            cache.flow_neighbour_radius_arcsec,
            cache.crowding_radii_arcsec,
        )
        if actual_geometry != expected_geometry:
            raise RuntimeError("model cache observing geometry does not match this run")
        cache.attach_detector(detector)
    else:
        cache = CatalogueModelCache(
            prior,
            detector=detector,
            conditions=conditions,
            detection_radius_arcsec=args.detection_radius_arcsec,
            flow_neighbour_radius_arcsec=args.flow_neighbour_radius_arcsec,
            crowding_radii_arcsec=(args.crowding_near_arcsec, args.crowding_far_arcsec),
            flow_features=flow.condition_preprocessor.feature_names,
        )
        cache.metadata = cache_identity
    detection_features = cache.validate_detection_shear_invariance()
    likelihood = CatalogueLikelihood(flow, cache)

    stencil = {(0.0, 0.0)}
    for h in steps:
        stencil.update({(h, 0.0), (-h, 0.0), (0.0, h), (0.0, -h)})
    cache.precompute(sorted(stencil))
    if not (cache_path / "manifest.json").is_file():
        cache.save(cache_path)

    proposal_identity = {
        **cache_identity,
        "conditions": conditions,
        "coordinate_config": {
            "n_flow_samples": args.proposal_flow_samples,
            "statistic": args.proposal_statistic,
            "seed": args.proposal_coordinate_seed,
        },
    }
    proposal_path = Path(args.proposal_cache)
    if (proposal_path / "manifest.json").is_file():
        coordinates = ProposalCoordinateTable.load(proposal_path)
        if coordinates.metadata != proposal_identity or coordinates.target_names != proposal_targets:
            raise RuntimeError("proposal cache identity does not match this null closure")
    else:
        coordinates = ProposalCoordinateTable.from_flow(
            likelihood,
            target_names=proposal_targets,
            n_flow_samples=args.proposal_flow_samples,
            statistic=args.proposal_statistic,
            row_chunk=args.proposal_row_chunk,
            seed=args.proposal_coordinate_seed,
            metadata=proposal_identity,
        )
        coordinates.save(proposal_path)

    mock = generate_mock_catalogue(
        likelihood,
        n_detected=args.n_detected,
        g1=0.0,
        g2=0.0,
        scene_seed=args.scene_seed,
        detection_seed=args.detection_seed,
        flow_seed=args.flow_seed,
    )
    mock.save(output / "validation_mock")
    active = np.flatnonzero(prior.weights > 0)
    if args.bank_column not in prior.galaxies:
        raise RuntimeError(
            f"scene prior lacks independent-bank column {args.bank_column!r}"
        )
    labels = sorted(prior.galaxies.loc[active, args.bank_column].drop_duplicates())
    labels = labels[: args.max_independent_banks]
    if len(labels) < 2:
        raise RuntimeError("at least two independent prior-bank labels are required")
    bank_weights = {
        str(label): prior.weights
        * (prior.galaxies[args.bank_column].to_numpy() == label)
        for label in labels
    }
    streamed = run_streamed_section5(
        likelihood,
        mock,
        _proposal(likelihood, coordinates),
        steps=steps,
        ladder=ladder,
        n_candidates=args.proposal_candidates,
        epsilon=args.proposal_epsilon,
        bandwidth=args.proposal_bandwidth,
        proposal_seeds=proposal_seeds,
        object_chunk=args.object_chunk,
        atom_chunk=args.atom_chunk,
        independent_bank_weights=bank_weights,
        posterior_adapt_proposal=True,
    )
    main_results = streamed.results
    bank_results = list(streamed.independent_banks.values())
    bank_payload = [
        {"label": label, "result": result.to_dict()}
        for label, result in streamed.independent_banks.items()
    ]

    if args.oracle_atoms <= 1 or args.oracle_atoms > len(active):
        raise RuntimeError("--oracle-atoms must be between 2 and the active prior size")
    rng = np.random.default_rng(args.oracle_seed)
    oracle_rows = rng.choice(active, size=args.oracle_atoms, replace=False)
    oracle_mass = np.zeros(len(prior.weights), dtype=np.float64)
    oracle_mass[oracle_rows] = prior.weights[oracle_rows]
    oracle_prior = prior.reweighted(
        oracle_mass,
        metadata={"oracle_seed": args.oracle_seed, "oracle_atoms": args.oracle_atoms},
    )
    oracle_likelihood = CatalogueLikelihood(flow, cache.with_prior(oracle_prior))
    oracle_mock = generate_mock_catalogue(
        oracle_likelihood,
        n_detected=args.oracle_n_detected,
        g1=0.0,
        g2=0.0,
        scene_seed=args.oracle_seed + 1,
        detection_seed=args.oracle_seed + 2,
        flow_seed=args.oracle_seed + 3,
    )
    oracle_mock.save(output / "oracle_mock")
    exact_oracle = run_exact_section5(
        oracle_likelihood,
        oracle_mock,
        steps=steps,
        object_chunk=args.object_chunk,
        atom_chunk=args.atom_chunk,
    )
    oracle_importance = run_streamed_section5(
        oracle_likelihood,
        oracle_mock,
        _proposal(oracle_likelihood, coordinates),
        steps=steps,
        ladder=ladder[-2:],
        n_candidates=args.proposal_candidates,
        epsilon=args.proposal_epsilon,
        bandwidth=args.proposal_bandwidth,
        proposal_seeds=(proposal_seeds[0],),
        object_chunk=args.object_chunk,
        atom_chunk=args.atom_chunk,
        posterior_adapt_proposal=True,
    ).results[0]

    assessment = assess_section5_null(
        main_results,
        primary_h=0.005,
        exact_oracle=exact_oracle,
        oracle_importance=oracle_importance,
        independent_banks=bank_results,
    )
    powered = None
    if assessment.passed and args.run_powered_on_pass:
        powered_mock = generate_mock_catalogue(
            likelihood,
            n_detected=args.powered_n_detected,
            g1=0.0,
            g2=0.0,
            scene_seed=args.scene_seed + 100,
            detection_seed=args.detection_seed + 100,
            flow_seed=args.flow_seed + 100,
        )
        powered_mock.save(output / "powered_mock")
        powered = run_streamed_section5(
            likelihood,
            powered_mock,
            _proposal(likelihood, coordinates),
            steps=(0.005,),
            ladder=(ladder[-1],),
            n_candidates=args.proposal_candidates,
            epsilon=args.proposal_epsilon,
            bandwidth=args.proposal_bandwidth,
            proposal_seeds=(proposal_seeds[0],),
            object_chunk=args.object_chunk,
            atom_chunk=args.atom_chunk,
            posterior_adapt_proposal=True,
        ).results[0]

    root = Path(__file__).resolve().parents[1]
    implementation = {
        relative: _sha256(root / relative)
        for relative in (
            "scripts/run_section5_null.py",
            "sbsi/catalogue_null.py",
            "sbsi/catalogue_likelihood.py",
            "sbsi/catalogue_sampling.py",
            "sbsi/catalogue_closure.py",
            "sbsi/scene_prior.py",
            "sbsi/shear_map.py",
        )
    }
    validation_evaluations = int(
        sum(result.flow_evaluations for result in main_results)
    )
    validation_elapsed = float(
        sum(result.elapsed_seconds for result in main_results)
    )
    equivalent_throughput = (
        validation_evaluations / validation_elapsed
        if validation_elapsed > 0
        else float("nan")
    )
    production_candidate_evaluations = min(
        args.proposal_candidates, int(np.count_nonzero(prior.weights))
    )
    production_per_object = production_candidate_evaluations + 5 * ladder[-1]
    production_per_billion = 1_000_000_000 * production_per_object
    payload = {
        "method": "MATH.md Section 5 local numerical marginal-likelihood expansion",
        "proposal_method": "zero_shear_posterior_adapted_defensive_importance_v1",
        "injected_shear": [0.0, 0.0],
        "selection": None,
        "r_blend": 0.0,
        "shear_transform": "intrinsic_ellipticity_only_v1",
        "detection_features": list(detection_features),
        "results": [result.to_dict() for result in main_results],
        "exact_oracle": exact_oracle.to_dict(),
        "oracle_importance": oracle_importance.to_dict(),
        "independent_banks": bank_payload,
        "assessment": asdict(assessment),
        "powered": None if powered is None else powered.to_dict(),
        "config": vars(args),
        "conditions": conditions,
        "model_sha256": model_hashes,
        "scene_sha256": scene_hashes,
        "implementation_sha256": implementation,
        "scalability": {
            "validation_flow_evaluations": validation_evaluations,
            "flow_evaluations_per_object": int(
                main_results[0].flow_evaluations / main_results[0].n_objects
            ),
            "projected_flow_evaluations_per_billion_objects": int(
                1_000_000_000
                * main_results[0].flow_evaluations
                / main_results[0].n_objects
            ),
            "equivalent_flow_evaluations_per_second": equivalent_throughput,
            "production_primary_views": 5,
            "production_proposal_candidate_evaluations_per_object": production_candidate_evaluations,
            "production_primary_flow_evaluations_per_object_at_fixed_M": production_per_object,
            "production_primary_flow_evaluations_per_billion_at_fixed_M": production_per_billion,
            "projected_gpu_hours_per_billion_at_fixed_M": (
                production_per_billion / equivalent_throughput / 3600.0
                if equivalent_throughput > 0
                else float("nan")
            ),
        },
    }
    (output / "result.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"assessment": payload["assessment"], "powered": powered is not None}, indent=2))
    print(f"Section 5 null closure -> {output}")
    if not assessment.passed:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
