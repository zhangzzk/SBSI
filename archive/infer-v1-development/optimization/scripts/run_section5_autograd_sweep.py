#!/usr/bin/env python
# Archived Infer V1 optimization experiment.
"""Compare exact finite differences with autograd over a step-size sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from sbsi.catalogue_closure import MockCatalogue
from sbsi.catalogue_likelihood import CatalogueLikelihood, CatalogueModelCache
from sbsi.catalogue_null import autograd_exact_section5
from sbsi.measurement_model import load_measurement_model
from sbsi.models import ModelPaths, load_emulator
from sbsi.scene_prior import ScenePrior


def _csv(text):
    return tuple(float(value) for value in str(text).split(",") if value.strip())


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-store", required=True)
    parser.add_argument("--measurement-model", required=True)
    parser.add_argument("--emulator-metadata", required=True)
    parser.add_argument("--emulator-model", required=True)
    parser.add_argument("--model-cache", required=True)
    parser.add_argument("--mock", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--exact-atoms", type=int, default=4096)
    parser.add_argument("--exact-seed", type=int, default=5201)
    parser.add_argument("--objects", type=int, default=32)
    parser.add_argument("--steps", default="0.01,0.005,0.0025,0.00125,0.000625")
    parser.add_argument("--object-chunk", type=int, default=32)
    parser.add_argument("--atom-chunk", type=int, default=4096)
    parser.add_argument("--pixel-size", type=float, default=0.2)
    parser.add_argument("--zero-point", type=float, default=30.0)
    parser.add_argument("--psf-fwhm", type=float, default=0.73)
    parser.add_argument("--moffat-beta", type=float, default=2.224)
    parser.add_argument("--pixel-rms", type=float, default=0.312)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args(argv)


def _metric(reference, current):
    difference = current - reference
    return {
        "relative_rms": float(
            np.sqrt(np.mean(difference**2))
            / max(float(np.std(reference, ddof=1)), 1.0e-12)
        ),
        "max_absolute": float(np.max(np.abs(difference))),
        "correlation": float(np.corrcoef(reference, current)[0, 1]),
    }


def main(argv=None):
    args = parse_args(argv)
    steps = _csv(args.steps)
    if not steps or min(steps) <= 0:
        raise SystemExit("--steps must contain positive values")
    output = Path(args.output).resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
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
    cache.validate_detection_shear_invariance()
    active = np.flatnonzero(prior.weights > 0)
    probability = prior.weights[active]
    probability = probability / probability.sum()
    rows = np.random.default_rng(args.exact_seed).choice(
        active, size=args.exact_atoms, replace=False, p=probability
    )
    mass = np.zeros(len(prior.weights), dtype=np.float64)
    mass[rows] = prior.weights[rows]
    exact_prior = prior.reweighted(
        mass,
        metadata={"purpose": "autograd_step_sweep", "seed": args.exact_seed},
    )
    likelihood = CatalogueLikelihood(flow, cache.with_prior(exact_prior))
    full_mock = MockCatalogue.load(args.mock)
    mock = MockCatalogue(
        full_mock.measurements.iloc[: args.objects].reset_index(drop=True),
        full_mock.truth.iloc[: args.objects].reset_index(drop=True),
    )
    autograd = autograd_exact_section5(likelihood, mock)
    results = []
    for h in steps:
        components = []
        for index, (name, direction) in enumerate(
            (("g1", (1.0, 0.0)), ("g2", (0.0, 1.0)))
        ):
            finite = likelihood.score_and_information(
                mock.measurements,
                center=(0.0, 0.0),
                direction=direction,
                delta=h,
                richardson=False,
                object_chunk=args.object_chunk,
                atom_chunk=args.atom_chunk,
            )
            score_metric = _metric(autograd.score[:, index], finite.score)
            info_metric = _metric(
                autograd.information[:, index, index], finite.information
            )
            # Information can have a much smaller spread than its nonzero mean;
            # retain the mean-normalized RMS used by the powered diagnostic too.
            info_difference = finite.information - autograd.information[:, index, index]
            info_metric["relative_rms_to_mean"] = float(
                np.sqrt(np.mean(info_difference**2))
                / max(abs(float(np.mean(autograd.information[:, index, index]))), 1.0e-12)
            )
            components.append(
                {"component": name, "score": score_metric, "information": info_metric}
            )
        results.append({"h": h, "components": components})
    payload = {
        "method": "exact finite-difference versus autograd step sweep",
        "config": vars(args),
        "autograd_elapsed_seconds": autograd.elapsed_seconds,
        "results": results,
    }
    (output / "result.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(results, indent=2))
    print(f"Section 5 autograd sweep -> {output}")


if __name__ == "__main__":
    main()
