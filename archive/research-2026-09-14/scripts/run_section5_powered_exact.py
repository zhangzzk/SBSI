#!/usr/bin/env python
"""Powered matched-prior Section 5 identity and autograd diagnostics."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import time

import numpy as np
import torch

from sbsi.catalogue_closure import generate_mock_catalogue
from sbsi.catalogue_likelihood import CatalogueLikelihood, CatalogueModelCache
from sbsi.catalogue_null import (
    autograd_exact_section5,
    run_exact_section5,
    summarize_section5,
)
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
    parser.add_argument("--output", required=True)
    parser.add_argument("--exact-atoms", type=int, default=4096)
    parser.add_argument("--exact-objects", type=int, default=32768)
    parser.add_argument("--matched-bank-atoms", type=int, default=4096)
    parser.add_argument("--matched-bank-objects", type=int, default=8192)
    parser.add_argument("--autograd-objects", type=int, default=32)
    parser.add_argument("--h", type=float, default=0.00125)
    parser.add_argument("--seed", type=int, default=5201)
    parser.add_argument("--bank-column", default="property_seed")
    parser.add_argument("--max-independent-banks", type=int, default=3)
    parser.add_argument("--object-chunk", type=int, default=32)
    parser.add_argument("--atom-chunk", type=int, default=4096)
    parser.add_argument("--pixel-size", type=float, default=0.2)
    parser.add_argument("--zero-point", type=float, default=30.0)
    parser.add_argument("--psf-fwhm", type=float, default=0.73)
    parser.add_argument("--moffat-beta", type=float, default=2.224)
    parser.add_argument("--pixel-rms", type=float, default=0.312)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args(argv)


def _sample_prior(prior, eligible, n_atoms, seed, label):
    eligible = np.asarray(eligible, dtype=np.int64)
    if n_atoms <= 1 or n_atoms > len(eligible):
        raise ValueError(f"{label}: n_atoms must be in [2, {len(eligible)}]")
    probability = prior.weights[eligible]
    probability = probability / probability.sum()
    rng = np.random.default_rng(seed)
    rows = rng.choice(eligible, size=n_atoms, replace=False, p=probability)
    mass = np.zeros(len(prior.weights), dtype=np.float64)
    mass[rows] = prior.weights[rows]
    return prior.reweighted(
        mass,
        metadata={"purpose": label, "seed": int(seed), "n_atoms": int(n_atoms)},
    ), np.sort(rows)


def _identity_payload(result):
    estimates = []
    passed = True
    for estimate in result.estimates:
        deviation = abs(estimate.information_identity_ratio - 1.0)
        pull = deviation / estimate.information_identity_standard_error
        component_passed = deviation <= 0.05 and pull <= 3.0
        passed = passed and component_passed
        estimates.append(
            {
                "component": estimate.component,
                "ratio": estimate.information_identity_ratio,
                "ratio_standard_error": estimate.information_identity_standard_error,
                "deviation": deviation,
                "pull": pull,
                "passed_5_percent_and_3sigma": component_passed,
            }
        )
    return {"passed": bool(passed), "estimates": estimates}


def _derivative_comparison(autograd, finite, h):
    components = []
    for index, (name, finite_result) in enumerate(finite):
        auto_score = autograd.score[:, index]
        auto_info = autograd.information[:, index, index]
        score_difference = finite_result.score - auto_score
        info_difference = finite_result.information - auto_info
        score_scale = max(float(np.std(auto_score, ddof=1)), 1.0e-12)
        info_scale = max(abs(float(np.mean(auto_info))), 1.0e-12)
        components.append(
            {
                "component": name,
                "h": float(h),
                "score_relative_rms": float(np.sqrt(np.mean(score_difference**2)) / score_scale),
                "information_relative_rms": float(np.sqrt(np.mean(info_difference**2)) / info_scale),
                "score_max_absolute": float(np.max(np.abs(score_difference))),
                "information_max_absolute": float(np.max(np.abs(info_difference))),
                "score_correlation": float(np.corrcoef(auto_score, finite_result.score)[0, 1]),
                "information_correlation": float(
                    np.corrcoef(auto_info, finite_result.information)[0, 1]
                ),
            }
        )
    return {
        "n_objects": int(len(autograd.score)),
        "n_atoms": int(len(autograd.atom_indices)),
        "autograd_elapsed_seconds": autograd.elapsed_seconds,
        "max_hessian_asymmetry": float(
            np.max(np.abs(autograd.information[:, 0, 1] - autograd.information[:, 1, 0]))
        ),
        "mean_cross_information": float(
            np.mean(0.5 * (autograd.information[:, 0, 1] + autograd.information[:, 1, 0]))
        ),
        "components": components,
        "passed": bool(
            max(item["score_relative_rms"] for item in components) <= 0.01
            and max(item["information_relative_rms"] for item in components) <= 0.05
        ),
    }


def main(argv=None):
    args = parse_args(argv)
    if args.h <= 0:
        raise SystemExit("--h must be positive")
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
    active = np.flatnonzero(prior.weights > 0)

    exact_prior, exact_rows = _sample_prior(
        prior, active, args.exact_atoms, args.seed, "powered_exact_full_bank"
    )
    exact_likelihood = CatalogueLikelihood(flow, cache.with_prior(exact_prior))
    exact_mock = generate_mock_catalogue(
        exact_likelihood,
        n_detected=args.exact_objects,
        g1=0.0,
        g2=0.0,
        scene_seed=args.seed + 1,
        detection_seed=args.seed + 2,
        flow_seed=args.seed + 3,
    )
    exact_mock.save(output / "exact_mock")
    exact_result = run_exact_section5(
        exact_likelihood,
        exact_mock,
        steps=(args.h,),
        object_chunk=args.object_chunk,
        atom_chunk=args.atom_chunk,
    )

    autograd = autograd_exact_section5(
        exact_likelihood,
        exact_mock,
        max_objects=args.autograd_objects,
    )
    finite_mock = type(exact_mock)(
        exact_mock.measurements.iloc[: args.autograd_objects].reset_index(drop=True),
        exact_mock.truth.iloc[: args.autograd_objects].reset_index(drop=True),
    )
    finite = []
    for name, direction in (("g1", (1.0, 0.0)), ("g2", (0.0, 1.0))):
        finite.append(
            (
                name,
                exact_likelihood.score_and_information(
                    finite_mock.measurements,
                    center=(0.0, 0.0),
                    direction=direction,
                    delta=args.h,
                    richardson=False,
                    object_chunk=args.object_chunk,
                    atom_chunk=args.atom_chunk,
                ),
            )
        )
    derivative_comparison = _derivative_comparison(autograd, finite, args.h)
    autograd_summaries = []
    for index, name in enumerate(("g1", "g2")):
        autograd_summaries.append(
            asdict(
                summarize_section5(
                    autograd.score[:, index],
                    autograd.information[:, index, index],
                    component=name,
                    h=0.0,
                    n_draws=len(exact_rows),
                )
            )
        )

    if args.bank_column not in prior.galaxies:
        raise RuntimeError(f"scene prior lacks bank column {args.bank_column!r}")
    labels = sorted(prior.galaxies.loc[active, args.bank_column].drop_duplicates())
    labels = labels[: args.max_independent_banks]
    matched_banks = []
    bank_values = prior.galaxies[args.bank_column].to_numpy()
    for bank_index, label in enumerate(labels):
        eligible = active[bank_values[active] == label]
        bank_prior, bank_rows = _sample_prior(
            prior,
            eligible,
            args.matched_bank_atoms,
            args.seed + 100 * (bank_index + 1),
            f"matched_bank_{label}",
        )
        bank_likelihood = CatalogueLikelihood(flow, cache.with_prior(bank_prior))
        bank_mock = generate_mock_catalogue(
            bank_likelihood,
            n_detected=args.matched_bank_objects,
            g1=0.0,
            g2=0.0,
            scene_seed=args.seed + 100 * (bank_index + 1) + 1,
            detection_seed=args.seed + 100 * (bank_index + 1) + 2,
            flow_seed=args.seed + 100 * (bank_index + 1) + 3,
        )
        bank_mock.save(output / f"matched_bank_{label}_mock")
        bank_result = run_exact_section5(
            bank_likelihood,
            bank_mock,
            steps=(args.h,),
            object_chunk=args.object_chunk,
            atom_chunk=args.atom_chunk,
        )
        matched_banks.append(
            {
                "label": str(label),
                "atom_rows_sha256": sha256(bank_rows.tobytes()).hexdigest(),
                "result": bank_result.to_dict(),
                "identity": _identity_payload(bank_result),
            }
        )

    exact_identity = _identity_payload(exact_result)
    matched_statistical_compatibility = all(
        all(item["pull"] <= 3.0 for item in bank["identity"]["estimates"])
        for bank in matched_banks
    )
    payload = {
        "method": "powered matched-prior exhaustive Section 5 null",
        "config": vars(args),
        "conditions": conditions,
        "detection_features": list(detection_features),
        "selection": None,
        "r_blend": 0.0,
        "injected_shear": [0.0, 0.0],
        "exact_atom_rows_sha256": sha256(exact_rows.tobytes()).hexdigest(),
        "exact_result": exact_result.to_dict(),
        "exact_identity": exact_identity,
        "autograd_summaries": autograd_summaries,
        "derivative_comparison": derivative_comparison,
        "matched_banks": matched_banks,
        "assessment": {
            "powered_exact_identity": exact_identity["passed"],
            "autograd_finite_difference": derivative_comparison["passed"],
            "matched_banks_statistically_compatible": matched_statistical_compatibility,
        },
        "elapsed_seconds": float(time.perf_counter() - started),
        "model_sha256": {
            "measurement": _sha256(args.measurement_model),
            "emulator": _sha256(args.emulator_model),
            "emulator_metadata": _sha256(args.emulator_metadata),
        },
    }
    (output / "result.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["assessment"], indent=2))
    print(f"powered Section 5 exact diagnostics -> {output}")


if __name__ == "__main__":
    main()
