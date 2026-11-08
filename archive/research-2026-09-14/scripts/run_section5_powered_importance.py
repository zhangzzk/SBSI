#!/usr/bin/env python
"""Powered full-catalogue Section 5 null after exact matched-prior validation."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import time

import torch

from sbsi.catalogue_closure import generate_mock_catalogue
from sbsi.catalogue_likelihood import CatalogueLikelihood, CatalogueModelCache
from sbsi.catalogue_null import run_streamed_section5
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
    parser.add_argument("--h", type=float, default=0.00125)
    parser.add_argument("--draws", type=int, default=65536)
    parser.add_argument("--previous-draws", type=int, default=32768)
    parser.add_argument("--proposal-candidates", type=int, default=131072)
    parser.add_argument("--proposal-seed", type=int, default=8701)
    parser.add_argument("--proposal-epsilon", type=float, default=0.1)
    parser.add_argument("--scene-seed", type=int, default=2001)
    parser.add_argument("--detection-seed", type=int, default=3001)
    parser.add_argument("--flow-seed", type=int, default=4001)
    parser.add_argument("--object-chunk", type=int, default=16)
    parser.add_argument("--atom-chunk", type=int, default=4096)
    parser.add_argument("--pixel-size", type=float, default=0.2)
    parser.add_argument("--zero-point", type=float, default=30.0)
    parser.add_argument("--psf-fwhm", type=float, default=0.73)
    parser.add_argument("--moffat-beta", type=float, default=2.224)
    parser.add_argument("--pixel-rms", type=float, default=0.312)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.h <= 0 or args.previous_draws <= 0 or args.draws <= args.previous_draws:
        raise SystemExit("require h>0 and 0<previous_draws<draws")
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
    mock = generate_mock_catalogue(
        likelihood,
        n_detected=args.n_detected,
        g1=0.0,
        g2=0.0,
        scene_seed=args.scene_seed,
        detection_seed=args.detection_seed,
        flow_seed=args.flow_seed,
    )
    mock.save(output / "mock")
    result = run_streamed_section5(
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
    ).results[0]
    index = {
        (estimate.component, estimate.n_draws): estimate
        for estimate in result.estimates
    }
    checks = {}
    metrics = {}
    for component in ("g1", "g2"):
        final = index[(component, args.draws)]
        previous = index[(component, args.previous_draws)]
        metrics[component] = {
            "centring_pull": abs(final.estimated_shear / final.robust_standard_error),
            "identity_deviation": abs(final.information_identity_ratio - 1.0),
            "identity_pull": abs(final.information_identity_ratio - 1.0)
            / final.information_identity_standard_error,
            "rung_change_in_se": abs(final.estimated_shear - previous.estimated_shear)
            / max(final.robust_standard_error, previous.robust_standard_error),
            "score_variance_rung_change": abs(
                final.score_variance - previous.score_variance
            )
            / max(abs(final.score_variance), 1.0e-12),
            "information_mean_rung_change": abs(
                final.information_mean - previous.information_mean
            )
            / max(abs(final.information_mean), 1.0e-12),
            "hill_tail_index": final.tail.hill_tail_index,
        }
        checks[component] = {
            "score_centring": metrics[component]["centring_pull"] <= 3.0,
            "information_identity": (
                metrics[component]["identity_deviation"] <= 0.05
                and metrics[component]["identity_pull"] <= 3.0
            ),
            "draw_ladder": metrics[component]["rung_change_in_se"] <= 0.25,
            "tail_stability": (
                metrics[component]["score_variance_rung_change"] <= 0.05
                and metrics[component]["information_mean_rung_change"] <= 0.05
                and metrics[component]["hill_tail_index"] > 2.0
            ),
        }
    payload = {
        "method": "powered full-catalogue posterior-adapted Section 5 null",
        "config": vars(args),
        "conditions": conditions,
        "detection_features": list(detection_features),
        "selection": None,
        "r_blend": 0.0,
        "injected_shear": [0.0, 0.0],
        "result": result.to_dict(),
        "checks": checks,
        "metrics": metrics,
        "passed": bool(all(all(values.values()) for values in checks.values())),
        "elapsed_seconds": float(time.perf_counter() - started),
        "model_sha256": {
            "measurement": _sha256(args.measurement_model),
            "emulator": _sha256(args.emulator_model),
            "emulator_metadata": _sha256(args.emulator_metadata),
        },
    }
    (output / "result.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"passed": payload["passed"], "checks": checks}, indent=2))
    print(f"powered Section 5 importance null -> {output}")


if __name__ == "__main__":
    main()
