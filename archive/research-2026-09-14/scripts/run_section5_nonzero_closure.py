#!/usr/bin/env python
"""Paired nonzero-shear closure for the local Section 5 estimator."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import time

import numpy as np
import torch

from sbsi.catalogue_closure import generate_mock_catalogue, generate_ring_mock_catalogue
from sbsi.catalogue_likelihood import CatalogueLikelihood, CatalogueModelCache
from sbsi.catalogue_null import run_streamed_section5, summarize_paired_section5
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


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-store", required=True)
    parser.add_argument("--measurement-model", required=True)
    parser.add_argument("--emulator-metadata", required=True)
    parser.add_argument("--emulator-model", required=True)
    parser.add_argument("--model-cache", required=True)
    parser.add_argument("--proposal-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-detected", type=int, default=10000)
    parser.add_argument("--injection", type=float, default=0.02)
    parser.add_argument("--components", nargs="+", choices=("g1", "g2"), default=("g1", "g2"))
    parser.add_argument("--h", type=float, default=0.00125)
    parser.add_argument("--draws", type=int, default=65536)
    parser.add_argument("--previous-draws", type=int, default=32768)
    parser.add_argument("--proposal-candidates", type=int, default=131072)
    parser.add_argument("--proposal-seed", type=int, default=8701)
    parser.add_argument("--proposal-epsilon", type=float, default=0.1)
    parser.add_argument("--scene-seed", type=int, default=2101)
    parser.add_argument("--detection-seed", type=int, default=3101)
    parser.add_argument("--flow-seed", type=int, default=4101)
    parser.add_argument("--orientation-seed", type=int, default=5101)
    parser.add_argument(
        "--ring-pairs", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="save the paired mocks and provenance without evaluating their likelihoods",
    )
    parser.add_argument("--object-chunk", type=int, default=16)
    parser.add_argument("--atom-chunk", type=int, default=4096)
    parser.add_argument("--pixel-size", type=float, default=0.2)
    parser.add_argument("--zero-point", type=float, default=30.0)
    parser.add_argument("--psf-fwhm", type=float, default=0.73)
    parser.add_argument("--moffat-beta", type=float, default=2.224)
    parser.add_argument("--pixel-rms", type=float, default=0.312)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args(argv)


def _moment_key(component: str, h: float, n_draws: int):
    return (component, float(h), int(n_draws))


def _pair_hash(mock) -> str:
    values = mock.truth[
        ["scene_row", "detection_uniform", "scene_seed", "detection_seed", "flow_seed"]
    ].to_numpy()
    return sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def main(argv=None):
    args = parse_args(argv)
    if args.injection <= 0 or args.h <= 0:
        raise SystemExit("--injection and --h must be positive")
    if args.previous_draws <= 0 or args.draws <= args.previous_draws:
        raise SystemExit("require 0<previous-draws<draws")
    components = tuple(dict.fromkeys(args.components))
    output = Path(args.output).resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    started = time.perf_counter()
    conditions = {
        "pixel_size": args.pixel_size,
        "zero_point": args.zero_point,
        "psf_fwhm": args.psf_fwhm,
        "moffat_beta": args.moffat_beta,
        "pixel_rms": args.pixel_rms,
    }
    prior = ScenePrior.load(args.scene_store)
    flow = load_measurement_model(args.measurement_model, device=args.device)
    detector = load_emulator(
        ModelPaths(
            flow_checkpoints=(Path(args.measurement_model),),
            emulator_model=Path(args.emulator_model),
            emulator_metadata=Path(args.emulator_metadata),
        ),
        conditions=conditions,
        device=args.device,
    )
    cache = CatalogueModelCache.load(args.model_cache, prior=prior)
    cache.attach_detector(detector)
    cache.validate_model_features(flow)
    detection_features = cache.validate_detection_shear_invariance()
    likelihood = CatalogueLikelihood(flow, cache)
    coordinates = ProposalCoordinateTable.load(args.proposal_cache)
    proposal = DefensiveLocalProposal(
        coordinates,
        prior.weights,
        local_base_weights=cache.get(0.0, 0.0).detection_probability,
    )

    base_mock = generate_mock_catalogue(
        likelihood,
        n_detected=args.n_detected,
        g1=0.0,
        g2=0.0,
        scene_seed=args.scene_seed,
        detection_seed=args.detection_seed,
        flow_seed=args.flow_seed,
    )
    base_mock.save(output / "mock_base_detected")
    proposal_mock = (
        generate_ring_mock_catalogue(
            likelihood,
            base_mock,
            g1=0.0,
            g2=0.0,
            orientation_seed=args.orientation_seed,
            flow_seed=args.flow_seed,
        )
        if args.ring_pairs
        else base_mock
    )
    proposal_mock.save(output / "mock_proposal_zero")
    proposal_hash = _pair_hash(proposal_mock)

    mocks = {}
    pairing = {}
    for injected_component in components:
        pair = {}
        for sign_name, sign in (("positive", 1.0), ("negative", -1.0)):
            g1 = sign * args.injection if injected_component == "g1" else 0.0
            g2 = sign * args.injection if injected_component == "g2" else 0.0
            mock = (
                generate_ring_mock_catalogue(
                    likelihood,
                    base_mock,
                    g1=g1,
                    g2=g2,
                    orientation_seed=args.orientation_seed,
                    flow_seed=args.flow_seed,
                )
                if args.ring_pairs
                else generate_mock_catalogue(
                    likelihood,
                    n_detected=args.n_detected,
                    g1=g1,
                    g2=g2,
                    scene_seed=args.scene_seed,
                    detection_seed=args.detection_seed,
                    flow_seed=args.flow_seed,
                )
            )
            mock.save(output / f"mock_{injected_component}_{sign_name}")
            pair[sign_name] = mock
            leg_name = f"{injected_component}_{sign_name}"
            mocks[leg_name] = mock

        positive_hash = _pair_hash(pair["positive"])
        negative_hash = _pair_hash(pair["negative"])
        if positive_hash != negative_hash or positive_hash != proposal_hash:
            raise RuntimeError(f"{injected_component} mocks do not share random-stream identities")
        if not np.array_equal(
            pair["positive"].truth["scene_row"].to_numpy(),
            pair["negative"].truth["scene_row"].to_numpy(),
        ):
            raise RuntimeError(f"{injected_component} paired mocks selected different scene rows")
        pairing[injected_component] = {
            "shared_truth_stream_sha256": positive_hash,
            "same_scene_rows": True,
            "same_detection_uniforms": bool(
                np.array_equal(
                    pair["positive"].truth["detection_uniform"].to_numpy(),
                    pair["negative"].truth["detection_uniform"].to_numpy(),
                )
            ),
            "shared_zero_shear_proposal": True,
        }

    if args.generate_only:
        payload = {
            "method": "paired nonzero-shear Section 5 mock generation only",
            "config": vars(args),
            "conditions": conditions,
            "detection_features": list(detection_features),
            "selection": None,
            "r_blend": 0.0,
            "pairing": pairing,
            "n_measurements_per_arm": int(len(proposal_mock.measurements)),
            "elapsed_seconds": float(time.perf_counter() - started),
            "model_sha256": {
                "measurement": _sha256(args.measurement_model),
                "emulator": _sha256(args.emulator_model),
                "emulator_metadata": _sha256(args.emulator_metadata),
            },
        }
        (output / "result.json").write_text(json.dumps(payload, indent=2) + "\n")
        print(json.dumps(payload, indent=2))
        print(f"paired nonzero Section 5 mocks -> {output}")
        return

    legs = {}
    leg_results = {}
    retained = {}
    for leg_name, mock in mocks.items():
        streamed = run_streamed_section5(
            likelihood,
            mock,
            proposal,
            steps=(args.h,),
            ladder=(args.previous_draws, args.draws),
            n_candidates=args.proposal_candidates,
            epsilon=args.proposal_epsilon,
            bandwidth=1.0,
            proposal_seeds=(args.proposal_seed,),
            object_chunk=args.object_chunk,
            atom_chunk=args.atom_chunk,
            posterior_adapt_proposal=True,
            require_null_truth=False,
            retain_object_moments=True,
            proposal_observed=proposal_mock.measurements,
        )
        leg_results[leg_name] = streamed.results[0]
        legs[leg_name] = leg_results[leg_name].to_dict()
        retained[leg_name] = streamed.object_moments[args.proposal_seed]

    moment_arrays = {}
    paired = []
    paired_objects = {}
    for injected_component in components:
        positive = retained[f"{injected_component}_positive"]
        negative = retained[f"{injected_component}_negative"]
        block_ids = (
            mocks[f"{injected_component}_positive"].truth["ring_id"].to_numpy()
            if args.ring_pairs
            else None
        )
        for estimated_component in ("g1", "g2"):
            for n_draws in (args.previous_draws, args.draws):
                key = _moment_key(estimated_component, args.h, n_draws)
                summary = summarize_paired_section5(
                    positive.score[key],
                    positive.information[key],
                    negative.score[key],
                    negative.information[key],
                    injected_component=injected_component,
                    estimated_component=estimated_component,
                    amplitude=args.injection,
                    h=args.h,
                    n_draws=n_draws,
                    block_ids=block_ids,
                )
                paired.append(asdict(summary))
                paired_objects[(injected_component, estimated_component, n_draws)] = summary
                prefix = f"{injected_component}_{estimated_component}_m{n_draws}"
                moment_arrays[f"{prefix}_positive_score"] = positive.score[key]
                moment_arrays[f"{prefix}_positive_information"] = positive.information[key]
                moment_arrays[f"{prefix}_negative_score"] = negative.score[key]
                moment_arrays[f"{prefix}_negative_information"] = negative.information[key]
    np.savez_compressed(output / "object_moments.npz", **moment_arrays)

    checks = {}
    metrics = {}
    for injected_component in components:
        final = paired_objects[(injected_component, injected_component, args.draws)]
        previous = paired_objects[(injected_component, injected_component, args.previous_draws)]
        cross_component = "g2" if injected_component == "g1" else "g1"
        cross_final = paired_objects[(injected_component, cross_component, args.draws)]
        cross_previous = paired_objects[
            (injected_component, cross_component, args.previous_draws)
        ]
        rung_scale = max(final.robust_standard_error, previous.robust_standard_error, 1.0e-12)
        cross_rung_scale = max(
            cross_final.robust_standard_error,
            cross_previous.robust_standard_error,
            1.0e-12,
        )
        point_error = abs(final.response_bias)
        arm_sampler_stable = True
        arm_tail_stable = True
        arm_information_positive = True
        for sign_name in ("positive", "negative"):
            result = leg_results[f"{injected_component}_{sign_name}"]
            index = {
                (estimate.component, estimate.n_draws): estimate
                for estimate in result.estimates
            }
            arm_final = index[(injected_component, args.draws)]
            arm_previous = index[(injected_component, args.previous_draws)]
            arm_sampler_stable = arm_sampler_stable and (
                abs(arm_final.score_variance - arm_previous.score_variance)
                / max(abs(arm_final.score_variance), 1.0e-12)
                <= 0.05
                and abs(arm_final.information_mean - arm_previous.information_mean)
                / max(abs(arm_final.information_mean), 1.0e-12)
                <= 0.05
            )
            arm_tail_stable = arm_tail_stable and (
                np.isfinite(arm_final.tail.hill_tail_index)
                and arm_final.tail.hill_tail_index > 2.0
            )
            arm_information_positive = (
                arm_information_positive and arm_final.information_mean > 0
            )
        metrics[injected_component] = {
            "multiplicative_bias": final.response_bias,
            "multiplicative_standard_error": final.robust_standard_error,
            "multiplicative_pull": abs(final.response_pull),
            "rung_change_in_se": abs(final.response - previous.response) / rung_scale,
            "three_sigma_bound": point_error + 3.0 * final.robust_standard_error,
            "paired_influence_correlation": final.influence_correlation,
            "symmetric_offset": final.symmetric_offset,
            "symmetric_offset_standard_error": final.symmetric_offset_standard_error,
            "arm_sampler_stable": bool(arm_sampler_stable),
            "arm_tail_stable": bool(arm_tail_stable),
            "arm_information_positive": bool(arm_information_positive),
            "paired_response_hill_tail_index": final.response_tail.hill_tail_index,
            "cross_estimated_component": cross_component,
            "cross_response": cross_final.response,
            "cross_response_standard_error": cross_final.robust_standard_error,
            "cross_response_pull": abs(cross_final.response_pull),
            "cross_rung_change_in_se": abs(
                cross_final.response - cross_previous.response
            )
            / cross_rung_scale,
            "cross_three_sigma_bound": abs(cross_final.response)
            + 3.0 * cross_final.robust_standard_error,
            "cross_response_hill_tail_index": cross_final.response_tail.hill_tail_index,
        }
        checks[injected_component] = {
            "statistically_consistent": abs(final.response_pull) <= 3.0,
            "draw_ladder": metrics[injected_component]["rung_change_in_se"] <= 0.25,
            "symmetric_offset": abs(final.symmetric_offset) <= 3.0 * final.symmetric_offset_standard_error,
            "arm_sampler_stability": bool(arm_sampler_stable),
            "arm_tail_stability": bool(arm_tail_stable),
            "arm_information_positive": bool(arm_information_positive),
            "paired_response_tail": bool(
                np.isfinite(final.response_tail.hill_tail_index)
                and final.response_tail.hill_tail_index > 2.0
            ),
            "cross_statistically_consistent": abs(cross_final.response_pull) <= 3.0,
            "cross_draw_ladder": metrics[injected_component][
                "cross_rung_change_in_se"
            ]
            <= 0.25,
            "cross_point_within_0p2_percent": abs(cross_final.response) <= 0.002,
            "cross_demonstrates_0p2_percent_at_3sigma": metrics[
                injected_component
            ]["cross_three_sigma_bound"]
            <= 0.002,
            "cross_response_tail": bool(
                np.isfinite(cross_final.response_tail.hill_tail_index)
                and cross_final.response_tail.hill_tail_index > 2.0
            ),
            "point_within_0p2_percent": point_error <= 0.002,
            "demonstrates_0p2_percent_at_3sigma": metrics[injected_component]["three_sigma_bound"] <= 0.002,
        }

    closure_passed = bool(
        all(
            item["statistically_consistent"]
            and item["draw_ladder"]
            and item["symmetric_offset"]
            and item["arm_sampler_stability"]
            and item["arm_tail_stability"]
            and item["arm_information_positive"]
            and item["paired_response_tail"]
            and item["cross_statistically_consistent"]
            and item["cross_draw_ladder"]
            and item["cross_response_tail"]
            for item in checks.values()
        )
    )
    target_demonstrated = bool(
        all(
            item["demonstrates_0p2_percent_at_3sigma"]
            and item["cross_demonstrates_0p2_percent_at_3sigma"]
            for item in checks.values()
        )
    )
    payload = {
        "method": "paired nonzero-shear local Section 5 closure about g=0",
        "config": vars(args),
        "conditions": conditions,
        "detection_features": list(detection_features),
        "selection": None,
        "r_blend": 0.0,
        "pairing": pairing,
        "n_measurements_per_arm": int(len(proposal_mock.measurements)),
        "legs": legs,
        "paired_estimates": paired,
        "checks": checks,
        "metrics": metrics,
        "closure_passed": closure_passed,
        "target_demonstrated": target_demonstrated,
        "elapsed_seconds": float(time.perf_counter() - started),
        "model_sha256": {
            "measurement": _sha256(args.measurement_model),
            "emulator": _sha256(args.emulator_model),
            "emulator_metadata": _sha256(args.emulator_metadata),
        },
    }
    (output / "result.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(
        json.dumps(
            {
                "closure_passed": closure_passed,
                "target_demonstrated": target_demonstrated,
                "metrics": metrics,
            },
            indent=2,
        )
    )
    print(f"paired nonzero Section 5 closure -> {output}")


if __name__ == "__main__":
    main()
