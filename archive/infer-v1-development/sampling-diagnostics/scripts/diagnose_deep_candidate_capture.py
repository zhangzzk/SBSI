#!/usr/bin/env python
# Archived Infer V1 sampling diagnostic.
"""Audit exact posterior mass captured by Gaussian-ranked candidate prefixes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.special import logsumexp
import torch

from sbsi.catalogue_closure import MockCatalogue
from sbsi.catalogue_likelihood import CatalogueLikelihood, CatalogueModelCache
from sbsi.catalogue_sampling import DefensiveLocalProposal, ProposalCoordinateTable
from sbsi.measurement_model import load_measurement_model
from sbsi.models import ModelPaths, load_emulator
from sbsi.scene_prior import ScenePrior


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
    parser.add_argument("--n-objects", type=int, default=1024)
    parser.add_argument("--center", nargs=2, type=float, required=True)
    parser.add_argument("--h", type=float, default=0.001)
    parser.add_argument("--full-stencil", action="store_true")
    parser.add_argument(
        "--candidate-prefixes",
        nargs="+",
        type=int,
        default=(8192, 16384, 32768, 65536),
    )
    parser.add_argument("--reference-candidates", type=int, default=65536)
    parser.add_argument("--extension-trigger-prefix", type=int, default=32768)
    parser.add_argument("--extension-min-q01", type=float, default=0.999)
    parser.add_argument("--object-chunk", type=int, default=16)
    parser.add_argument("--atom-chunk", type=int, default=4096)
    parser.add_argument("--compile-flow", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _summary(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    quantiles = (0.1, 1, 5, 10, 50, 90, 99)
    return {
        "q0p1_q01_q05_q10_q50_q90_q99": np.percentile(values, quantiles).tolist(),
        "mean": float(np.mean(values)),
        "minimum": float(np.min(values)),
        "fraction_ge_0p9": float(np.mean(values >= 0.9)),
        "fraction_ge_0p99": float(np.mean(values >= 0.99)),
        "fraction_ge_0p999": float(np.mean(values >= 0.999)),
    }


def _views(center: tuple[float, float], h: float, full: bool):
    g1, g2 = center
    result = {"center": (g1, g2)}
    if full:
        result.update(
            {
                "g1_plus": (g1 + h, g2),
                "g1_minus": (g1 - h, g2),
                "g2_plus": (g1, g2 + h),
                "g2_minus": (g1, g2 - h),
                "pp": (g1 + h, g2 + h),
                "pm": (g1 + h, g2 - h),
                "mp": (g1 - h, g2 + h),
                "mm": (g1 - h, g2 - h),
            }
        )
    return result


def main(argv=None):
    args = parse_args(argv)
    output = Path(args.output).resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}")
    if args.n_objects <= 0 or args.object_chunk <= 0 or args.atom_chunk <= 0:
        raise ValueError("object and chunk counts must be positive")
    if args.h <= 0 or not np.isfinite(args.h):
        raise ValueError("h must be finite and positive")
    prefixes = tuple(sorted({int(value) for value in args.candidate_prefixes}))
    reference = int(args.reference_candidates)
    if not prefixes or prefixes[0] <= 0 or prefixes[-1] > reference:
        raise ValueError("candidate prefixes must be positive and within the reference")
    if args.extension_trigger_prefix not in prefixes:
        raise ValueError("extension trigger prefix must be one of the reported prefixes")
    output.mkdir(parents=True)

    conditions = {
        "pixel_size": 0.2,
        "zero_point": 30.0,
        "psf_fwhm": 0.73,
        "moffat_beta": 2.224,
        "pixel_rms": 0.312,
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
    likelihood = CatalogueLikelihood(flow, cache)
    coordinates = ProposalCoordinateTable.load(args.proposal_cache)
    if coordinates.dispersion is None:
        raise ValueError("Gaussian uncertainty ranking requires cached dispersion")
    proposal = DefensiveLocalProposal(
        coordinates,
        prior.weights,
        local_base_weights=cache.get(0.0, 0.0).detection_probability,
    )
    source = MockCatalogue.load(args.mock_input)
    count = min(int(args.n_objects), len(source.measurements))
    mock = MockCatalogue(
        source.measurements.iloc[:count].reset_index(drop=True),
        source.truth.iloc[:count].reset_index(drop=True),
    )
    if args.compile_flow:
        flow.compile_log_prob(mode=None, dynamic=True)

    view_coordinates = _views(tuple(map(float, args.center)), args.h, args.full_stencil)
    masses = {
        name: [] for name in view_coordinates
    }
    effective = {name: [] for name in view_coordinates}
    maximum = {name: [] for name in view_coordinates}
    query_seconds = 0.0
    flow_seconds = {name: 0.0 for name in view_coordinates}
    started = time.perf_counter()

    for start in range(0, count, args.object_chunk):
        stop = min(start + args.object_chunk, count)
        observed = mock.measurements.iloc[start:stop].reset_index(drop=True)
        phase = time.perf_counter()
        candidates = proposal.candidates(observed, n_candidates=reference)
        query_seconds += time.perf_counter() - phase

        values = observed.loc[:, coordinates.target_names].to_numpy(dtype=np.float64)
        atom_dispersion = coordinates.dispersion[candidates.indices]
        residual = (values[:, None, :] - coordinates.values[candidates.indices]) / atom_dispersion
        base = proposal.local_base_weights[candidates.indices]
        proxy = np.full(base.shape, -np.inf, dtype=np.float64)
        positive = base > 0
        gaussian = -np.log(atom_dispersion).sum(axis=2) - 0.5 * np.square(residual).sum(axis=2)
        proxy[positive] = np.log(base[positive]) + gaussian[positive]
        order = np.argsort(-proxy, axis=1)
        ranked_indices = np.take_along_axis(candidates.indices, order, axis=1)
        observed_targets = likelihood.observed_target_tensor(observed)

        for name, (g1, g2) in view_coordinates.items():
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            phase = time.perf_counter()
            target = likelihood.log_importance_weights_tensor(
                observed,
                g1,
                g2,
                atom_indices=ranked_indices,
                proposal_probability=np.ones_like(ranked_indices, dtype=np.float64),
                observed_targets=observed_targets,
                object_chunk=len(observed),
                atom_chunk=args.atom_chunk,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            flow_seconds[name] += time.perf_counter() - phase
            target = target.detach().cpu().numpy().astype(np.float64)
            normalizer = logsumexp(target, axis=1)
            if not np.isfinite(normalizer).all():
                raise ValueError(f"non-finite reference mass in view {name}")
            posterior = np.exp(target - normalizer[:, None])
            cumulative = np.cumsum(posterior, axis=1)
            masses[name].append(
                np.column_stack([cumulative[:, size - 1] for size in prefixes])
            )
            positive_mass = posterior > 0
            entropy = -np.sum(
                np.where(
                    positive_mass,
                    posterior * np.log(np.where(positive_mass, posterior, 1.0)),
                    0.0,
                ),
                axis=1,
            )
            effective[name].append(np.exp(entropy))
            maximum[name].append(np.max(posterior, axis=1))
            del target, posterior, cumulative
        del candidates, ranked_indices, order, proxy, observed_targets

    arrays = {"prefix_sizes": np.asarray(prefixes, dtype=np.int64)}
    report_views = {}
    for name in view_coordinates:
        view_mass = np.concatenate(masses[name], axis=0)
        view_effective = np.concatenate(effective[name])
        view_maximum = np.concatenate(maximum[name])
        arrays[f"mass__{name}"] = view_mass
        arrays[f"effective_atoms__{name}"] = view_effective
        arrays[f"max_mass_fraction__{name}"] = view_maximum
        report_views[name] = {
            "shear": list(view_coordinates[name]),
            "prefixes": {
                str(size): _summary(view_mass[:, column])
                for column, size in enumerate(prefixes)
            },
            "reference_effective_atoms": _summary(view_effective),
            "reference_max_mass_fraction": _summary(view_maximum),
            "flow_seconds": float(flow_seconds[name]),
        }
    npz_path = output / "candidate_capture.npz"
    np.savez_compressed(npz_path, **arrays)

    trigger_column = prefixes.index(int(args.extension_trigger_prefix))
    trigger_q01 = min(
        float(np.percentile(np.concatenate(masses[name], axis=0)[:, trigger_column], 1))
        for name in view_coordinates
    )
    report = {
        "status": "complete",
        "method": "deep_location_reference_gaussian_mean_std_ranking_v1",
        "normalization": "exact_posterior_mass_within_deep_location_reference",
        "n_objects": count,
        "reference_candidates": reference,
        "candidate_prefixes": list(prefixes),
        "center": list(map(float, args.center)),
        "h": float(args.h),
        "full_stencil": bool(args.full_stencil),
        "qmc_coordinate_config": {
            "samples": int(coordinates.n_flow_samples),
            "statistic": coordinates.statistic,
            "dispersion_statistic": coordinates.dispersion_statistic,
        },
        "extension_rule": {
            "trigger_prefix": int(args.extension_trigger_prefix),
            "minimum_q01_across_views": trigger_q01,
            "required_q01": float(args.extension_min_q01),
            "recommend_extend_reference": bool(trigger_q01 < args.extension_min_q01),
        },
        "timing": {
            "total_seconds": float(time.perf_counter() - started),
            "candidate_query_seconds": float(query_seconds),
            "flow_seconds_by_view": {key: float(value) for key, value in flow_seconds.items()},
        },
        "views": report_views,
        "arrays": {"path": npz_path.name, "sha256": _sha256(npz_path)},
        "config": vars(args),
    }
    (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
