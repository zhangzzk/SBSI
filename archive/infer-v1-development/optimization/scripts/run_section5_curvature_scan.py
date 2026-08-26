#!/usr/bin/env python
# Archived Infer V1 optimization experiment.
"""Diagnose finite-shear curvature in saved catalogue-prior mocks."""

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
    run_importance_curvature_scan,
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


def _floats(text: str) -> tuple[float, ...]:
    values = tuple(sorted({float(value) for value in text.split(",")}))
    if not values or not np.isfinite(values).all():
        raise argparse.ArgumentTypeError("expected a comma-separated finite list")
    return values


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
    parser.add_argument("--direction", choices=("g1", "g2"), required=True)
    parser.add_argument("--centers", type=_floats, required=True)
    parser.add_argument("--h", type=float, default=0.00125)
    parser.add_argument("--zero-derivative-steps", type=_floats, default=(0.00125, 0.0025))
    parser.add_argument("--draws", type=int, default=16384)
    parser.add_argument("--previous-draws", type=int, default=8192)
    parser.add_argument("--proposal-candidates", type=int, default=32768)
    parser.add_argument("--proposal-seed", type=int, default=8701)
    parser.add_argument("--proposal-epsilon", type=float, default=0.1)
    parser.add_argument("--object-chunk", type=int, default=16)
    parser.add_argument("--decomposition-chunk", type=int, default=32)
    parser.add_argument("--pixel-size", type=float, default=0.2)
    parser.add_argument("--zero-point", type=float, default=30.0)
    parser.add_argument("--psf-fwhm", type=float, default=0.73)
    parser.add_argument("--moffat-beta", type=float, default=2.224)
    parser.add_argument("--pixel-rms", type=float, default=0.312)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args(argv)


def _array_summary(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "mean_standard_error": float(np.std(values, ddof=1) / np.sqrt(len(values))),
        "median": float(np.median(values)),
        "fraction_negative": float(np.mean(values < 0)),
        "quantiles": {
            str(q): float(np.quantile(values, q))
            for q in (0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999)
        },
    }


def _higher_derivatives(shears, profile, step: float) -> dict:
    index = {round(float(value), 14): i for i, value in enumerate(shears)}
    keys = tuple(round(multiplier * step, 14) for multiplier in (-2, -1, 0, 1, 2))
    if not all(key in index for key in keys):
        return {"step": float(step), "available": False}
    minus2, minus1, zero, plus1, plus2 = (profile[index[key]] for key in keys)
    third = (plus2 - 2.0 * plus1 + 2.0 * minus1 - minus2) / (2.0 * step**3)
    fourth = (minus2 - 4.0 * minus1 + 6.0 * zero - 4.0 * plus1 + plus2) / step**4
    return {
        "step": float(step),
        "available": True,
        "information_first_derivative": _array_summary(-third),
        "information_second_derivative": _array_summary(-fourth),
    }


def main(argv=None):
    args = parse_args(argv)
    if args.previous_draws <= 0 or args.draws <= args.previous_draws:
        raise SystemExit("require 0 < --previous-draws < --draws")
    if args.h <= 0 or any(step <= 0 for step in args.zero_derivative_steps):
        raise SystemExit("finite-difference steps must be positive")
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
    injected_vectors = mock.truth[["injected_g1", "injected_g2"]].drop_duplicates()
    if len(injected_vectors) != 1:
        raise RuntimeError("saved mock contains more than one injected shear")
    injected_vector = injected_vectors.iloc[0].to_numpy(float)
    orthogonal = injected_vector[1] if args.direction == "g1" else injected_vector[0]
    if abs(orthogonal) > 1.0e-14:
        raise RuntimeError("saved mock injection is not aligned with --direction")

    profile_shears = {0.0}
    for step in args.zero_derivative_steps:
        profile_shears.update((-2.0 * step, -step, step, 2.0 * step))
    scan = run_importance_curvature_scan(
        likelihood,
        mock,
        proposal,
        centers=args.centers,
        h=args.h,
        profile_shears=tuple(profile_shears),
        ladder=(args.previous_draws, args.draws),
        n_candidates=args.proposal_candidates,
        epsilon=args.proposal_epsilon,
        bandwidth=1.0,
        proposal_seed=args.proposal_seed,
        direction=direction,
        truth_atom_indices=mock.truth["scene_row"].to_numpy(np.int64),
        object_chunk=args.object_chunk,
        decomposition_chunk=args.decomposition_chunk,
    )

    arrays = {
        "shears": np.asarray(scan.shears),
        "centers": np.asarray(args.centers),
        "scene_row": mock.truth["scene_row"].to_numpy(np.int64),
    }
    for n_draws, profile in scan.marginalized_log_likelihood.items():
        arrays[f"m{n_draws}_marginal_log_likelihood"] = profile
    arrays["true_atom_log_likelihood"] = scan.true_atom_log_likelihood
    rung_summaries = []
    for rung in scan.rungs:
        prefix = f"m{rung.n_draws}"
        arrays[f"{prefix}_marginal_score"] = rung.marginalized_score
        arrays[f"{prefix}_marginal_information"] = rung.marginalized_information
        arrays[f"{prefix}_posterior_conditional_information"] = rung.posterior_conditional_information
        arrays[f"{prefix}_posterior_score_variance"] = rung.posterior_score_variance
        arrays[f"{prefix}_decomposition_residual"] = rung.decomposition_residual
        arrays[f"{prefix}_true_atom_score"] = rung.true_atom_score
        arrays[f"{prefix}_true_atom_information"] = rung.true_atom_information
        centers = []
        for index, center in enumerate(rung.centers):
            marginal = rung.marginalized_information[index]
            conditional = rung.posterior_conditional_information[index]
            variance = rung.posterior_score_variance[index]
            residual = rung.decomposition_residual[index]
            centers.append(
                {
                    "center": float(center),
                    "marginalized_score": _array_summary(rung.marginalized_score[index]),
                    "marginalized_information": _array_summary(marginal),
                    "posterior_conditional_information": _array_summary(conditional),
                    "posterior_score_variance": _array_summary(variance),
                    "true_atom_information": _array_summary(rung.true_atom_information[index]),
                    "decomposition": {
                        "mean_reconstructed_information": float(np.mean(conditional - variance)),
                        "mean_residual": float(np.mean(residual)),
                        "max_absolute_residual": float(np.max(np.abs(residual))),
                        "mean_residual_fraction": float(
                            np.mean(residual) / max(abs(np.mean(marginal)), 1.0e-12)
                        ),
                    },
                }
            )
        rung_summaries.append({"n_draws": rung.n_draws, "centers": centers})
    np.savez_compressed(output / "object_curvature.npz", **arrays)

    final_profile = scan.marginalized_log_likelihood[args.draws]
    payload = {
        "method": "saved-mock per-object curvature and posterior-mixture decomposition",
        "config": {
            **vars(args),
            "centers": list(args.centers),
            "zero_derivative_steps": list(args.zero_derivative_steps),
        },
        "conditions": conditions,
        "selection": None,
        "r_blend": 0.0,
        "detection_features": list(detection_features),
        "injected_shear": injected_vector.tolist(),
        "n_objects": len(mock.measurements),
        "profile_shears": list(scan.shears),
        "rungs": rung_summaries,
        "zero_higher_derivatives": {
            "marginalized": [
                _higher_derivatives(scan.shears, final_profile, step)
                for step in args.zero_derivative_steps
            ],
            "true_atom": [
                _higher_derivatives(scan.shears, scan.true_atom_log_likelihood, step)
                for step in args.zero_derivative_steps
            ],
        },
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
    print(json.dumps({"output": str(output), "rungs": rung_summaries}, indent=2))


if __name__ == "__main__":
    main()
