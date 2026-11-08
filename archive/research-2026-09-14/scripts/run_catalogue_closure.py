#!/usr/bin/env python
"""Run exact and/or importance-sampled finite-catalogue closure."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sbsi.catalogue_blend import CatalogueBlendResponse
from sbsi.catalogue_closure import (
    MockCatalogue,
    generate_mock_catalogue,
    summarize_closure_score,
)
from sbsi.catalogue_likelihood import (
    CatalogueLikelihood,
    CatalogueModelCache,
    CatalogueSelection,
    OutputCut,
)
from sbsi.catalogue_sampling import (
    DefensiveLocalProposal,
    ProposalCoordinateTable,
    assess_importance_convergence,
    run_importance_ladder,
    run_importance_profiles,
)
from sbsi.measurement_model import load_measurement_model
from sbsi.models import ModelPaths, load_detection_classifier, load_emulator
from sbsi.scene_prior import ScenePrior
from sbsi.scene_prior import SHEAR_TRANSFORM


def _sha256(path):
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _implementation_hashes() -> dict[str, str]:
    """Hash the executable inference path for dirty-worktree provenance."""

    root = Path(__file__).resolve().parents[1]
    relative_paths = (
        "scripts/run_catalogue_closure.py",
        "sbsi/catalogue_closure.py",
        "sbsi/catalogue_blend.py",
        "sbsi/catalogue_likelihood.py",
        "sbsi/catalogue_sampling.py",
        "sbsi/scene_prior.py",
        "sbsi/shear_map.py",
        "sbsi/measurement_model.py",
        "sbsi/selection_model.py",
        "sbsi/forward_catalogue.py",
        "sbsi/models.py",
    )
    return {name: _sha256(root / name) for name in relative_paths}


def _optional_float(text):
    """Argparse type accepting a float or ``none`` for an absent cut."""

    return None if str(text).strip().lower() in {"none", "off", ""} else float(text)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-store", required=True)
    parser.add_argument("--measurement-model", required=True)
    parser.add_argument("--emulator-metadata", required=True)
    parser.add_argument(
        "--emulator-model",
        required=True,
        help="companion response-emulator artifact; also supplies the legacy detector",
    )
    parser.add_argument(
        "--detection-model",
        default=None,
        help="optional explicit SBSI detection-classifier checkpoint",
    )
    parser.add_argument(
        "--inject-r-blend",
        action="store_true",
        help="inject and infer the fixed external catalogue blend response",
    )
    parser.add_argument(
        "--blend-response-cache",
        default=None,
        help="cache built by build_catalogue_blend_response.py",
    )
    parser.add_argument("--output", required=True, help="output directory")
    parser.add_argument("--sampler", choices=("exact", "importance", "both"), default="exact")
    parser.add_argument("--model-cache", default=None, help="optional directory for stencil model views")
    parser.add_argument(
        "--selection-cache",
        default=None,
        help="optional cache for per-atom measured-cut pass probabilities",
    )
    parser.add_argument(
        "--cut-abs-ehat",
        type=_optional_float,
        default=None,
        help="keep measurements with |measured shape| below this value; default none",
    )
    parser.add_argument(
        "--cut-bound",
        action="append",
        default=[],
        metavar="NAME:LO:HI",
        help=(
            "keep LO <= flow output NAME < HI; empty LO/HI is unbounded; "
            "repeatable and ANDed with --cut-abs-ehat"
        ),
    )
    parser.add_argument("--selection-samples", type=int, default=64)
    parser.add_argument("--selection-seed", type=int, default=8101)
    parser.add_argument("--selection-row-chunk", type=int, default=8192)
    parser.add_argument("--proposal-cache", default=None)
    parser.add_argument(
        "--proposal-targets",
        default=None,
        help="comma-separated measured flow outputs used by the local proposal",
    )
    parser.add_argument("--proposal-flow-samples", type=int, default=32)
    parser.add_argument("--proposal-statistic", choices=("mean", "median"), default="median")
    parser.add_argument("--proposal-row-chunk", type=int, default=8192)
    parser.add_argument("--proposal-coordinate-seed", type=int, default=7101)
    parser.add_argument("--importance-ladder", default=None, help="comma-separated nested draw counts")
    parser.add_argument(
        "--proposal-candidates",
        default=None,
        help="comma-separated local candidate counts for support-radius expansion",
    )
    parser.add_argument("--proposal-epsilon", type=float, default=None)
    parser.add_argument("--proposal-bandwidth", type=float, default=None)
    parser.add_argument("--proposal-seeds", default=None, help="comma-separated independent proposal seeds")
    parser.add_argument(
        "--profile-shears",
        default=None,
        help=(
            "comma-separated scalar shears along the inference direction; when "
            "set, evaluate a fixed-draw likelihood profile instead of the score ladder"
        ),
    )
    parser.add_argument("--gate-max-exact-error", type=float, default=None)
    parser.add_argument("--gate-max-rung-change", type=float, default=None)
    parser.add_argument("--gate-max-seed-spread", type=float, default=None)
    parser.add_argument("--gate-max-candidate-change", type=float, default=None)
    parser.add_argument("--gate-min-ess-fraction", type=float, default=None)
    parser.add_argument("--gate-max-weight-fraction", type=float, default=None)
    parser.add_argument("--n-detected", type=int, default=1000)
    parser.add_argument(
        "--mock-input",
        default=None,
        help="directory containing frozen measurements.parquet and truth.parquet",
    )
    parser.add_argument("--injected-g1", type=float, default=0.02)
    parser.add_argument("--injected-g2", type=float, default=0.0)
    parser.add_argument("--direction-g1", type=float, default=1.0)
    parser.add_argument("--direction-g2", type=float, default=0.0)
    parser.add_argument("--score-center-g1", type=float, default=0.0)
    parser.add_argument("--score-center-g2", type=float, default=0.0)
    parser.add_argument("--fd-delta", type=float, default=0.01)
    parser.add_argument("--no-richardson", action="store_true")
    parser.add_argument("--detection-radius-arcsec", type=float, required=True)
    parser.add_argument(
        "--detection-neighbour-selection",
        choices=("nearest", "impact"),
        default="nearest",
    )
    parser.add_argument("--detection-impact-exponent", type=float, default=2.0)
    parser.add_argument("--flow-neighbour-radius-arcsec", type=float, required=True)
    parser.add_argument("--crowding-near-arcsec", type=float, required=True)
    parser.add_argument("--crowding-far-arcsec", type=float, required=True)
    parser.add_argument("--pixel-size", type=float, required=True)
    parser.add_argument("--zero-point", type=float, required=True)
    parser.add_argument("--psf-fwhm", type=float, required=True)
    parser.add_argument("--moffat-beta", type=float, required=True)
    parser.add_argument("--pixel-rms", type=float, required=True)
    parser.add_argument("--scene-seed", type=int, default=1001)
    parser.add_argument("--detection-seed", type=int, default=2001)
    parser.add_argument("--flow-seed", type=int, default=3001)
    parser.add_argument("--object-chunk", type=int, default=64)
    parser.add_argument("--atom-chunk", type=int, default=4096)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.inject_r_blend and args.blend_response_cache is None:
        raise SystemExit("--inject-r-blend requires --blend-response-cache")
    if args.blend_response_cache is not None and not args.inject_r_blend:
        raise SystemExit("--blend-response-cache requires --inject-r-blend")
    implementation_hashes = _implementation_hashes()
    use_importance = args.sampler in {"importance", "both"}
    use_profile = use_importance and args.profile_shears is not None
    if use_importance:
        required = {
            "--proposal-targets": args.proposal_targets,
            "--importance-ladder": args.importance_ladder,
            "--proposal-candidates": args.proposal_candidates,
            "--proposal-epsilon": args.proposal_epsilon,
            "--proposal-bandwidth": args.proposal_bandwidth,
            "--proposal-seeds": args.proposal_seeds,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise SystemExit("importance sampling requires " + ", ".join(missing))
    gate_values = (
        args.gate_max_exact_error,
        args.gate_max_rung_change,
        args.gate_max_seed_spread,
        args.gate_max_candidate_change,
        args.gate_min_ess_fraction,
        args.gate_max_weight_fraction,
    )
    if any(value is not None for value in gate_values):
        if not all(value is not None for value in gate_values):
            raise SystemExit("all six --gate-* thresholds must be supplied together")
        if args.sampler != "both":
            raise SystemExit("convergence gates require --sampler both for exact-oracle comparison")
        if use_profile:
            raise SystemExit("exact-oracle convergence gates apply to score ladders, not profiles")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    conditions = {
        "pixel_size": args.pixel_size,
        "zero_point": args.zero_point,
        "psf_fwhm": args.psf_fwhm,
        "moffat_beta": args.moffat_beta,
        "pixel_rms": args.pixel_rms,
    }
    prior = ScenePrior.load(args.scene_store)
    flow = load_measurement_model(args.measurement_model, device=args.device)
    output_cut = None
    if args.cut_abs_ehat is not None or args.cut_bound:
        try:
            output_cut = OutputCut.from_specs(
                flow.target_transform.target_names,
                abs_shape=args.cut_abs_ehat,
                specs=args.cut_bound,
            )
        except (KeyError, ValueError) as error:
            raise SystemExit(f"invalid measured selection: {error}") from error
    elif args.selection_cache is not None:
        raise SystemExit("--selection-cache requires a measured cut")
    model_hashes = {
        "measurement": _sha256(args.measurement_model),
        "emulator": _sha256(args.emulator_model),
        "emulator_metadata": _sha256(args.emulator_metadata),
        "detection": (
            None if args.detection_model is None else _sha256(args.detection_model)
        ),
    }
    scene_root = Path(args.scene_store)
    scene_hashes = {
        name: _sha256(scene_root / name)
        for name in ("manifest.json", "galaxies.parquet", "neighbours.npz")
    }
    blend_response = None
    blend_cache_hashes = None
    if args.inject_r_blend:
        blend_root = Path(args.blend_response_cache)
        blend_response = CatalogueBlendResponse.load(blend_root)
        blend_cache_hashes = {
            name: _sha256(blend_root / name)
            for name in ("manifest.json", "r_blend.npy")
        }
        expected_blend_metadata = {
            "scene_sha256": scene_hashes,
            "emulator_sha256": {
                "model": model_hashes["emulator"],
                "metadata": model_hashes["emulator_metadata"],
            },
            "conditions": conditions,
        }
        actual_blend_metadata = {
            name: blend_response.metadata.get(name)
            for name in expected_blend_metadata
        }
        if actual_blend_metadata != expected_blend_metadata:
            raise RuntimeError(
                "blend-response cache identity does not match the supplied scene, "
                "emulator, or observing conditions"
            )
    cache_identity = {
        "model_sha256": model_hashes,
        "scene_sha256": scene_hashes,
    }
    if blend_response is not None:
        cache_identity["blend_response"] = {
            "enabled": True,
            "cache_sha256": blend_cache_hashes,
        }

    cache_path = None if args.model_cache is None else Path(args.model_cache)
    if cache_path is not None and (cache_path / "manifest.json").is_file():
        cache = CatalogueModelCache.load(
            cache_path, prior=prior, blend_response=blend_response
        )
        expected_geometry = (
            conditions,
            args.detection_radius_arcsec,
            args.detection_neighbour_selection,
            args.detection_impact_exponent,
            args.flow_neighbour_radius_arcsec,
            (args.crowding_near_arcsec, args.crowding_far_arcsec),
            tuple(flow.condition_preprocessor.feature_names),
        )
        actual_geometry = (
            cache.conditions,
            cache.detection_radius_arcsec,
            cache.detection_neighbour_selection,
            cache.detection_impact_exponent,
            cache.flow_neighbour_radius_arcsec,
            cache.crowding_radii_arcsec,
            tuple(cache.flow_features or flow.condition_preprocessor.feature_names),
        )
        if cache.metadata != cache_identity or actual_geometry != expected_geometry:
            raise RuntimeError(
                "model cache identity/configuration does not match the supplied "
                "catalogue, models, observing conditions, or apertures"
            )
        if use_profile:
            model_paths = ModelPaths(
                flow_checkpoints=(Path(args.measurement_model),),
                emulator_model=Path(args.emulator_model),
                emulator_metadata=Path(args.emulator_metadata),
                detection_classifier=(
                    None
                    if args.detection_model is None
                    else Path(args.detection_model)
                ),
            )
            cache.attach_detector(
                (
                    load_emulator(
                        model_paths,
                        conditions=conditions,
                        device=args.device,
                    )
                    if args.detection_model is None
                    else load_detection_classifier(model_paths, device=args.device)
                )
            )
    else:
        model_paths = ModelPaths(
            flow_checkpoints=(Path(args.measurement_model),),
            emulator_model=Path(args.emulator_model),
            emulator_metadata=Path(args.emulator_metadata),
            detection_classifier=(
                None if args.detection_model is None else Path(args.detection_model)
            ),
        )
        detector = (
            load_emulator(model_paths, conditions=conditions, device=args.device)
            if args.detection_model is None
            else load_detection_classifier(model_paths, device=args.device)
        )
        cache = CatalogueModelCache(
            prior,
            detector=detector,
            conditions=conditions,
            detection_radius_arcsec=args.detection_radius_arcsec,
            detection_neighbour_selection=args.detection_neighbour_selection,
            detection_impact_exponent=args.detection_impact_exponent,
            flow_neighbour_radius_arcsec=args.flow_neighbour_radius_arcsec,
            crowding_radii_arcsec=(args.crowding_near_arcsec, args.crowding_far_arcsec),
            blend_response=blend_response,
            flow_features=flow.condition_preprocessor.feature_names,
        )
    selection = None
    selection_initial_shears = ()
    selection_path = None if args.selection_cache is None else Path(args.selection_cache)
    selection_identity = {
        **cache_identity,
        "selection_samples": args.selection_samples,
        "selection_seed": args.selection_seed,
        "selection_row_chunk": args.selection_row_chunk,
    }
    if output_cut is not None:
        if selection_path is not None and (selection_path / "manifest.json").is_file():
            selection = CatalogueSelection.load(selection_path, output_cut=output_cut)
            requested_sampling = (
                args.selection_samples,
                args.selection_seed,
                args.selection_row_chunk,
            )
            cached_sampling = (
                selection.n_samples,
                selection.seed,
                selection.row_chunk,
            )
            if selection.metadata != selection_identity or cached_sampling != requested_sampling:
                raise RuntimeError(
                    "selection cache identity/configuration does not match the supplied "
                    "catalogue, model, cut, or sampling settings"
                )
        else:
            selection = CatalogueSelection(
                output_cut,
                n_samples=args.selection_samples,
                seed=args.selection_seed,
                row_chunk=args.selection_row_chunk,
            )
            selection.metadata = selection_identity
        selection_initial_shears = selection.available_shears
    likelihood = CatalogueLikelihood(flow, cache, selection=selection)

    direction = (args.direction_g1, args.direction_g2)
    center = (args.score_center_g1, args.score_center_g2)
    norm = (args.direction_g1**2 + args.direction_g2**2) ** 0.5
    if norm == 0:
        raise SystemExit("--direction-g1 and --direction-g2 must not both be zero")
    stencil = [
        (0.0, 0.0),
        (args.injected_g1, args.injected_g2),
        center,
    ]
    for scale in (args.fd_delta, -args.fd_delta):
        stencil.append(
            (
                center[0] + scale * args.direction_g1 / norm,
                center[1] + scale * args.direction_g2 / norm,
            )
        )
        if not args.no_richardson:
            stencil.append(
                (
                    center[0] + 0.5 * scale * args.direction_g1 / norm,
                    center[1] + 0.5 * scale * args.direction_g2 / norm,
                )
            )
    cache.precompute(stencil)
    if cache_path is not None and not (cache_path / "manifest.json").is_file():
        cache.save(cache_path, metadata=cache_identity)

    mock_input_hashes = None
    if args.mock_input is None:
        mock = generate_mock_catalogue(
            likelihood,
            n_detected=args.n_detected,
            g1=args.injected_g1,
            g2=args.injected_g2,
            scene_seed=args.scene_seed,
            detection_seed=args.detection_seed,
            flow_seed=args.flow_seed,
        )
    else:
        mock_root = Path(args.mock_input)
        mock = MockCatalogue.load(mock_root)
        if mock.shear_transform != SHEAR_TRANSFORM:
            raise RuntimeError(
                "frozen mock does not declare the current shape-only shear transform; "
                "rebuild the mock catalogue from its existing source products"
            )
        if len(mock.measurements) != args.n_detected:
            raise RuntimeError(
                f"frozen mock has {len(mock.measurements)} rows, expected {args.n_detected}"
            )
        missing_targets = sorted(set(likelihood.target_names) - set(mock.measurements))
        if missing_targets:
            raise RuntimeError(f"frozen mock lacks flow targets: {missing_targets}")
        injected = mock.truth[["injected_g1", "injected_g2"]].drop_duplicates()
        expected_injected = np.asarray([args.injected_g1, args.injected_g2], dtype=float)
        if len(injected) != 1 or not np.allclose(
            injected.iloc[0].to_numpy(float), expected_injected, rtol=0, atol=1e-14
        ):
            raise RuntimeError(
                "frozen mock injection does not match --injected-g1/--injected-g2"
            )
        mock_input_hashes = {
            name: _sha256(mock_root / name)
            for name in ("measurements.parquet", "truth.parquet")
        }
        image_manifest = mock_root / "image_mock_manifest.json"
        if image_manifest.is_file():
            mock_input_hashes[image_manifest.name] = _sha256(image_manifest)
        if mock.kind == "image":
            required_image_truth = {"source_case", "source_input_index"}
            missing_image_truth = sorted(required_image_truth - set(mock.truth))
            if missing_image_truth:
                raise RuntimeError(
                    "image mock lacks source identity columns: "
                    f"{missing_image_truth}"
                )
        if output_cut is not None:
            selected = np.asarray(
                output_cut(mock.measurements[list(likelihood.target_names)].to_numpy(float)),
                dtype=bool,
            )
            if selected.shape != (len(mock.measurements),) or not selected.all():
                failed = int((~selected).sum()) if selected.shape == (len(mock.measurements),) else -1
                raise RuntimeError(
                    f"frozen mock contains {failed} rows outside the requested measured cut"
                )
        if blend_response is not None and mock.kind == "likelihood":
            required_truth = {
                "scene_row",
                "r_blend",
                "blend_shift_g1",
                "blend_shift_g2",
            }
            missing_truth = sorted(required_truth - set(mock.truth))
            if missing_truth:
                raise RuntimeError(
                    "frozen R_blend mock lacks truth columns: "
                    f"{missing_truth}"
                )
            rows = mock.truth["scene_row"].to_numpy(dtype=np.int64)
            expected_response = blend_response.values[rows]
            expected_shift = cache.get(
                args.injected_g1, args.injected_g2
            ).blend_shift[rows]
            if not np.allclose(
                mock.truth["r_blend"], expected_response, rtol=0, atol=0
            ) or not np.allclose(
                mock.truth[["blend_shift_g1", "blend_shift_g2"]],
                expected_shift,
                rtol=0,
                atol=1e-14,
            ):
                raise RuntimeError(
                    "frozen mock R_blend truth does not match the supplied response cache"
                )
    reference_score = None
    exact_result = None
    if args.sampler in {"exact", "both"}:
        reference_score = likelihood.score_and_information(
            mock.measurements,
            center=center,
            direction=direction,
            delta=args.fd_delta,
            richardson=not args.no_richardson,
            object_chunk=args.object_chunk,
            atom_chunk=args.atom_chunk,
        )
        direction_array = np.asarray(direction, dtype=float)
        direction_array /= (direction_array @ direction_array) ** 0.5
        exact_result = summarize_closure_score(
            mock,
            reference_score,
            direction=direction,
            evaluation_shear=float(
                np.asarray(center, dtype=float) @ direction_array
            ),
        )

    importance_results = []
    profile_results = []
    importance_assessment = None
    if use_importance:
        proposal_targets = tuple(
            name.strip() for name in args.proposal_targets.split(",") if name.strip()
        )
        proposal_identity = {
            **cache_identity,
            "conditions": conditions,
            "flow_neighbour_radius_arcsec": args.flow_neighbour_radius_arcsec,
            "crowding_radii_arcsec": [
                args.crowding_near_arcsec,
                args.crowding_far_arcsec,
            ],
            "coordinate_config": {
                "n_flow_samples": args.proposal_flow_samples,
                "statistic": args.proposal_statistic,
                "seed": args.proposal_coordinate_seed,
            },
        }
        proposal_path = None if args.proposal_cache is None else Path(args.proposal_cache)
        if proposal_path is not None and (proposal_path / "manifest.json").is_file():
            coordinates = ProposalCoordinateTable.load(proposal_path)
            if coordinates.metadata != proposal_identity or coordinates.target_names != proposal_targets:
                raise RuntimeError(
                    "proposal cache identity or target names do not match the supplied scene/models"
                )
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
            if proposal_path is not None:
                coordinates.save(proposal_path)
        proposal = DefensiveLocalProposal(
            coordinates,
            prior.weights,
            local_base_weights=cache.get(0.0, 0.0).detection_probability,
        )
        ladder = tuple(int(value) for value in args.importance_ladder.split(",") if value)
        candidate_ladder = tuple(
            int(value) for value in args.proposal_candidates.split(",") if value
        )
        proposal_seeds = tuple(int(value) for value in args.proposal_seeds.split(",") if value)
        if not ladder or not candidate_ladder or not proposal_seeds:
            raise SystemExit(
                "importance ladder, candidate counts, and proposal seeds must be non-empty"
            )
        if use_profile:
            # Mock generation and proposal construction have already consumed
            # the loaded views.  A profile needs only one large catalogue view
            # at a time, and the attached detector can rebuild it lazily.
            cache.discard_views(cache.available_shears)
            gc.collect()
        for n_candidates in candidate_ladder:
            if use_profile:
                profile_shears = tuple(
                    float(value)
                    for value in args.profile_shears.split(",")
                    if value
                )
                if not profile_shears:
                    raise SystemExit("profile shear list must be non-empty")
                profile_results.extend(
                    run_importance_profiles(
                        likelihood,
                        mock,
                        proposal,
                        shears=profile_shears,
                        ladder=ladder,
                        n_candidates=n_candidates,
                        epsilon=args.proposal_epsilon,
                        bandwidth=args.proposal_bandwidth,
                        proposal_seeds=proposal_seeds,
                        direction=direction,
                        object_chunk=args.object_chunk,
                    )
                )
            else:
                for proposal_seed in proposal_seeds:
                    importance_results.append(
                        run_importance_ladder(
                            likelihood,
                            mock,
                            proposal,
                            ladder=ladder,
                            n_candidates=n_candidates,
                            epsilon=args.proposal_epsilon,
                            bandwidth=args.proposal_bandwidth,
                            proposal_seed=proposal_seed,
                            center=center,
                            direction=direction,
                            delta=args.fd_delta,
                            richardson=not args.no_richardson,
                            compare_exact=args.sampler == "both",
                            reference_score=reference_score,
                            object_chunk=args.object_chunk,
                            atom_chunk=args.atom_chunk,
                        )
                    )
        if all(value is not None for value in gate_values):
            importance_assessment = assess_importance_convergence(
                importance_results,
                max_exact_shear_error=args.gate_max_exact_error,
                max_rung_change=args.gate_max_rung_change,
                max_seed_spread=args.gate_max_seed_spread,
                max_candidate_change=args.gate_max_candidate_change,
                min_ess_fraction=args.gate_min_ess_fraction,
                max_p90_weight_fraction=args.gate_max_weight_fraction,
            )
    mock.save(output)
    mock_output_hashes = {
        name: _sha256(output / name)
        for name in ("measurements.parquet", "truth.parquet")
    }
    selection_summary = None
    if selection is not None:
        if (
            selection_path is not None
            and selection.available_shears != selection_initial_shears
        ):
            selection.save(selection_path)
        probability_summary = []
        for g1, g2 in selection.available_shears:
            probability = selection.cached_probability(g1, g2)
            probability_summary.append(
                {
                    "g1": g1,
                    "g2": g2,
                    "prior_mean_p_pass": float(np.sum(prior.weights * probability)),
                }
            )
        selection_summary = {
            "cut_key": output_cut.key(),
            "description": output_cut.describe(),
            "n_samples": selection.n_samples,
            "seed": selection.seed,
            "row_chunk": selection.row_chunk,
            "probabilities": probability_summary,
        }
    if reference_score is not None:
        pd.DataFrame(
            {
                "log_likelihood": reference_score.log_likelihood,
                "score": reference_score.score,
                "information": reference_score.information,
            }
        ).to_parquet(output / "exact_score.parquet", index=False)
    manifest = {
        "result": None if exact_result is None else asdict(exact_result),
        "importance": [result.to_dict() for result in importance_results],
        "profile": [result.to_dict() for result in profile_results],
        "importance_assessment": (
            None if importance_assessment is None else asdict(importance_assessment)
        ),
        "config": vars(args),
        "conditions": conditions,
        "selection": selection_summary,
        "r_blend": {
            "enabled": blend_response is not None,
            "cache": args.blend_response_cache,
            "cache_sha256": blend_cache_hashes,
            "metadata": None if blend_response is None else blend_response.metadata,
            "report": None if blend_response is None else blend_response.report,
        },
        "model_sha256": model_hashes,
        "mock_sha256": mock_output_hashes,
        "mock_input_sha256": mock_input_hashes,
        "mock_kind": mock.kind,
        "implementation_sha256": implementation_hashes,
        "flow_targets": list(likelihood.target_names),
        "flow_conditions": list(flow.condition_preprocessor.feature_names),
    }
    (output / "result.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                "result": manifest["result"],
                "importance": manifest["importance"],
                "profile": manifest["profile"],
                "importance_assessment": manifest["importance_assessment"],
            },
            indent=2,
        )
    )
    print(f"closure output -> {output}")


if __name__ == "__main__":
    main()
