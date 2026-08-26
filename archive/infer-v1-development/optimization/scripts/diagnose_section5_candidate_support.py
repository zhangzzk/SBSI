#!/usr/bin/env python
# Archived Infer V1 optimization experiment.
"""Measure how efficiently distance-ranked candidates cover local posterior mass."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.special import logsumexp
import torch

from sbsi.catalogue_closure import MockCatalogue
from sbsi.catalogue_likelihood import CatalogueLikelihood, CatalogueModelCache
from sbsi.catalogue_sampling import (
    DefensiveLocalProposal,
    ProposalCoordinateTable,
    candidate_support_diagnostics,
)
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
    parser.add_argument("--center", nargs=2, type=float, default=(0.0, 0.0))
    parser.add_argument(
        "--candidate-prefixes",
        nargs="+",
        type=int,
        default=(128, 256, 512, 1024, 2048, 4096, 8192, 16384),
    )
    parser.add_argument("--object-chunk", type=int, default=16)
    parser.add_argument("--atom-chunk", type=int, default=4096)
    parser.add_argument("--compile-flow", action="store_true")
    parser.add_argument("--final-candidates", type=int)
    parser.add_argument("--prefilter-prefixes", nargs="+", type=int)
    parser.add_argument("--direct-uncertainty-candidates", type=int)
    parser.add_argument(
        "--location-backend", choices=("scipy", "torch"), default="scipy"
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args(argv)


def _mass_summary(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "q01_q10_q50_q90_q99": np.percentile(
            values, (1, 10, 50, 90, 99)
        ).tolist(),
        "mean": float(np.mean(values)),
        "fraction_ge_0p9": float(np.mean(values >= 0.9)),
        "fraction_ge_0p99": float(np.mean(values >= 0.99)),
    }


def main(argv=None):
    args = parse_args(argv)
    output = Path(args.output).resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    if args.n_objects <= 0 or args.object_chunk <= 0 or args.atom_chunk <= 0:
        raise ValueError("object count and chunk sizes must be positive")
    prefixes = tuple(sorted({int(value) for value in args.candidate_prefixes}))
    if not prefixes or prefixes[0] <= 0:
        raise ValueError("candidate prefixes must be positive")

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

    distance_rows = []
    optimal_rows = []
    uncertainty_rows = []
    prefilter_rows = {}
    direct_mass_ratio_rows = []
    direct_overlap_rows = []
    direct_query_seconds = 0.0
    direct_flow_seconds = 0.0
    effective_rows = []
    maximum_rows = []
    query_seconds = 0.0
    flow_seconds = 0.0
    center = tuple(map(float, args.center))
    started = time.perf_counter()
    for start in range(0, count, args.object_chunk):
        stop = min(start + args.object_chunk, count)
        observed = mock.measurements.iloc[start:stop].reset_index(drop=True)
        query_started = time.perf_counter()
        candidates = proposal.candidates(
            observed,
            n_candidates=prefixes[-1],
            torch_device=(flow.device if args.location_backend == "torch" else None),
        )
        query_seconds += time.perf_counter() - query_started
        observed_targets = likelihood.observed_target_tensor(observed)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        flow_started = time.perf_counter()
        target = likelihood.log_importance_weights_tensor(
            observed,
            *center,
            atom_indices=candidates.indices,
            proposal_probability=np.ones_like(candidates.indices, dtype=np.float64),
            observed_targets=observed_targets,
            object_chunk=len(observed),
            atom_chunk=args.atom_chunk,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        flow_seconds += time.perf_counter() - flow_started
        diagnostics = candidate_support_diagnostics(
            target.detach().cpu().numpy().astype(np.float64), prefixes
        )
        if args.direct_uncertainty_candidates is not None:
            direct_started = time.perf_counter()
            direct = proposal.uncertainty_candidates(
                observed, n_candidates=args.direct_uncertainty_candidates
            )
            direct_query_seconds += time.perf_counter() - direct_started
            direct_started = time.perf_counter()
            direct_target = likelihood.log_importance_weights_tensor(
                observed,
                *center,
                atom_indices=direct.indices,
                proposal_probability=np.ones_like(direct.indices, dtype=np.float64),
                observed_targets=observed_targets,
                object_chunk=len(observed),
                atom_chunk=args.atom_chunk,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            direct_flow_seconds += time.perf_counter() - direct_started
            reference_numpy = target.detach().cpu().numpy().astype(np.float64)
            direct_numpy = direct_target.detach().cpu().numpy().astype(np.float64)
            direct_mass_ratio_rows.append(
                np.exp(
                    logsumexp(direct_numpy, axis=1)
                    - logsumexp(reference_numpy, axis=1)
                )
            )
            direct_overlap_rows.append(
                np.array(
                    [
                        np.mean(np.isin(direct.indices[row], candidates.indices[row]))
                        for row in range(len(observed))
                    ]
                )
            )
            del direct, direct_target
        if coordinates.dispersion is not None:
            observed_values = observed.loc[
                :, coordinates.target_names
            ].to_numpy(dtype=np.float64)
            atom_dispersion = coordinates.dispersion[candidates.indices]
            residual = (
                observed_values[:, None, :] - coordinates.values[candidates.indices]
            ) / atom_dispersion
            base = proposal.local_base_weights[candidates.indices]
            approximate_log_target = np.full(
                base.shape, -np.inf, dtype=np.float64
            )
            positive = base > 0
            gaussian = (
                -np.log(atom_dispersion).sum(axis=2)
                - 0.5 * np.square(residual).sum(axis=2)
            )
            approximate_log_target[positive] = (
                np.log(base[positive]) + gaussian[positive]
            )
            uncertainty_order = np.argsort(-approximate_log_target, axis=1)
            target_numpy = target.detach().cpu().numpy().astype(np.float64)
            uncertainty_target = np.take_along_axis(
                target_numpy, uncertainty_order, axis=1
            )
            uncertainty_diagnostics = candidate_support_diagnostics(
                uncertainty_target, prefixes
            )
            uncertainty_rows.append(
                uncertainty_diagnostics.distance_prefix_mass
            )
            if args.final_candidates is not None and args.prefilter_prefixes:
                final = int(args.final_candidates)
                target_numpy = target.detach().cpu().numpy().astype(np.float64)
                posterior = np.exp(
                    target_numpy - logsumexp(target_numpy, axis=1)[:, None]
                )
                for prefilter in sorted(set(args.prefilter_prefixes)):
                    prefilter = int(prefilter)
                    if final <= 0 or prefilter < final or prefilter > prefixes[-1]:
                        raise ValueError(
                            "prefilter prefixes must lie between final and reference support"
                        )
                    selected = np.argpartition(
                        approximate_log_target[:, :prefilter],
                        kth=prefilter - final,
                        axis=1,
                    )[:, -final:]
                    retained = np.take_along_axis(
                        posterior[:, :prefilter], selected, axis=1
                    ).sum(axis=1)
                    prefilter_rows.setdefault(prefilter, []).append(retained)
        distance_rows.append(diagnostics.distance_prefix_mass)
        optimal_rows.append(diagnostics.optimal_prefix_mass)
        effective_rows.append(diagnostics.reference_effective_atoms)
        maximum_rows.append(diagnostics.reference_max_mass_fraction)
        del candidates, observed_targets, target, diagnostics

    distance = np.concatenate(distance_rows, axis=0)
    optimal = np.concatenate(optimal_rows, axis=0)
    uncertainty = (
        np.concatenate(uncertainty_rows, axis=0) if uncertainty_rows else None
    )
    effective = np.concatenate(effective_rows)
    maximum = np.concatenate(maximum_rows)
    arrays = {
        "prefix_sizes": np.asarray(prefixes, dtype=np.int64),
        "distance_prefix_mass": distance,
        "optimal_prefix_mass": optimal,
        "reference_effective_atoms": effective,
        "reference_max_mass_fraction": maximum,
    }
    if uncertainty is not None:
        arrays["uncertainty_prefix_mass"] = uncertainty
    for prefilter, rows in prefilter_rows.items():
        arrays[f"prefilter_{prefilter}_mass"] = np.concatenate(rows)
    np.savez_compressed(output / "candidate_support.npz", **arrays)
    report = {
        "method": "section5_candidate_support_v1",
        "config": vars(args),
        "n_objects": count,
        "reference_candidates": prefixes[-1],
        "normalization": "within_reference_candidates",
        "candidate_flow_evaluations": int(count * prefixes[-1]),
        "timing": {
            "total_seconds": float(time.perf_counter() - started),
            "candidate_query_seconds": float(query_seconds),
            "flow_seconds": float(flow_seconds),
        },
        "reference": {
            "effective_atoms_q10_q50_q90": np.percentile(
                effective, (10, 50, 90)
            ).tolist(),
            "max_mass_fraction_q10_q50_q90": np.percentile(
                maximum, (10, 50, 90)
            ).tolist(),
        },
        "prefixes": {},
        "prefilter_support": {
            str(prefilter): _mass_summary(np.concatenate(rows))
            for prefilter, rows in sorted(prefilter_rows.items())
        },
        "direct_uncertainty": (
            None
            if not direct_mass_ratio_rows
            else {
                "candidates": int(args.direct_uncertainty_candidates),
                "mass_ratio_to_location_reference": _mass_summary(
                    np.concatenate(direct_mass_ratio_rows)
                ),
                "candidate_overlap_fraction": _mass_summary(
                    np.concatenate(direct_overlap_rows)
                ),
                "query_seconds": float(direct_query_seconds),
                "flow_seconds": float(direct_flow_seconds),
            }
        ),
    }
    for column, size in enumerate(prefixes):
        item = {
            "distance_ranked": _mass_summary(distance[:, column]),
            "optimal_within_reference": _mass_summary(optimal[:, column]),
        }
        if uncertainty is not None:
            item["uncertainty_ranked"] = _mass_summary(
                uncertainty[:, column]
            )
        report["prefixes"][str(size)] = item
    (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
