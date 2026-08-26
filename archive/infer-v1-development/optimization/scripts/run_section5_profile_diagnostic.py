#!/usr/bin/env python
# Archived nonlinear profile diagnostic.
"""Profile saved Section 5 mocks to diagnose finite-shear nonlinearity."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import time

import numpy as np
import torch

from sbsi.catalogue_closure import MockCatalogue
from sbsi.catalogue_likelihood import CatalogueLikelihood, CatalogueModelCache
from sbsi.catalogue_sampling import (
    DefensiveLocalProposal,
    ProposalCoordinateTable,
    run_importance_profiles,
)
from sbsi.measurement_model import load_measurement_model
from sbsi.models import ModelPaths, load_emulator
from sbsi.scene_prior import SHEAR_TRANSFORM, ScenePrior


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
    parser.add_argument("--mock-input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--profile-shears", required=True)
    parser.add_argument("--direction", choices=("g1", "g2"), required=True)
    parser.add_argument("--draws", type=int, default=16384)
    parser.add_argument("--previous-draws", type=int, default=8192)
    parser.add_argument("--proposal-candidates", type=int, default=32768)
    parser.add_argument("--proposal-seed", type=int, default=8701)
    parser.add_argument("--proposal-epsilon", type=float, default=0.1)
    parser.add_argument("--object-chunk", type=int, default=16)
    parser.add_argument("--pixel-size", type=float, default=0.2)
    parser.add_argument("--zero-point", type=float, default=30.0)
    parser.add_argument("--psf-fwhm", type=float, default=0.73)
    parser.add_argument("--moffat-beta", type=float, default=2.224)
    parser.add_argument("--pixel-rms", type=float, default=0.312)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args(argv)


def _quadratic_at_zero(points, h: float) -> dict:
    by_shear = {float(point.shear): float(point.log_likelihood_sum) for point in points}
    required = (-h, 0.0, h)
    if not all(value in by_shear for value in required):
        return {"h": float(h), "available": False}
    minus, zero, plus = (by_shear[value] for value in required)
    score = (plus - minus) / (2.0 * h)
    information = -(plus - 2.0 * zero + minus) / h**2
    return {
        "h": float(h),
        "available": True,
        "score_sum": float(score),
        "information_sum": float(information),
        "one_step_shear": float(score / information) if information != 0 else None,
    }


def main(argv=None):
    args = parse_args(argv)
    if args.previous_draws <= 0 or args.draws <= args.previous_draws:
        raise SystemExit("require 0 < --previous-draws < --draws")
    shears = tuple(sorted({float(value) for value in args.profile_shears.split(",")}))
    if not shears or not np.isfinite(shears).all():
        raise SystemExit("--profile-shears must contain finite values")
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
    mock = MockCatalogue.load(args.mock_input)
    if mock.shear_transform != SHEAR_TRANSFORM:
        raise RuntimeError("saved mock does not use the current shape-only shear map")
    direction = (1.0, 0.0) if args.direction == "g1" else (0.0, 1.0)
    injected_vector = mock.truth[["injected_g1", "injected_g2"]].drop_duplicates()
    if len(injected_vector) != 1:
        raise RuntimeError("saved mock contains more than one injected shear")
    injected = injected_vector.iloc[0].to_numpy(float)
    orthogonal = injected[1] if args.direction == "g1" else injected[0]
    if abs(orthogonal) > 1.0e-14:
        raise RuntimeError("saved mock injection is not aligned with --direction")

    result = run_importance_profiles(
        likelihood,
        mock,
        proposal,
        shears=shears,
        ladder=(args.previous_draws, args.draws),
        n_candidates=args.proposal_candidates,
        epsilon=args.proposal_epsilon,
        bandwidth=1.0,
        proposal_seeds=(args.proposal_seed,),
        direction=direction,
        object_chunk=args.object_chunk,
    )[0]
    final = result.rungs[-1]
    zero_steps = tuple(
        h for h in (0.00125, 0.0025, 0.005, 0.01, 0.02) if -h in shears and h in shears
    )
    payload = {
        "method": "saved-mock finite-shear likelihood-profile diagnostic",
        "config": vars(args),
        "conditions": conditions,
        "selection": None,
        "r_blend": 0.0,
        "detection_features": list(detection_features),
        "injected_shear": injected.tolist(),
        "n_objects": len(mock.measurements),
        "profile": result.to_dict(),
        "zero_expansions": [
            _quadratic_at_zero(final.points, h) for h in zero_steps
        ],
        "elapsed_seconds": float(time.perf_counter() - started),
        "model_sha256": {
            "measurement": _sha256(args.measurement_model),
            "emulator": _sha256(args.emulator_model),
            "emulator_metadata": _sha256(args.emulator_metadata),
        },
        "mock_sha256": {
            name: _sha256(Path(args.mock_input) / name)
            for name in ("measurements.parquet", "truth.parquet")
        },
    }
    (output / "result.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(
        json.dumps(
            {
                "injected_shear": payload["injected_shear"],
                "rungs": [
                    {
                        "n_draws": rung.n_draws,
                        "grid_estimated_shear": rung.grid_estimated_shear,
                        "estimated_shear": rung.estimated_shear,
                        "quadratic_information": rung.quadratic_information,
                    }
                    for rung in result.rungs
                ],
                "zero_expansions": payload["zero_expansions"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
