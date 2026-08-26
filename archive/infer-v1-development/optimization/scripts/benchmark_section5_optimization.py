#!/usr/bin/env python
# Archived Infer V1 optimization experiment.
"""Benchmark legacy, tensor-native, adaptive, and autograd Section 5 paths."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from sbsi.catalogue_closure import generate_mock_catalogue
from sbsi.catalogue_likelihood import CatalogueLikelihood, CatalogueModelCache
from sbsi.catalogue_null import (
    autograd_importance_section5,
    run_adaptive_section5,
    run_streamed_section5,
)
from sbsi.catalogue_sampling import DefensiveLocalProposal, ProposalCoordinateTable
from sbsi.measurement_model import load_measurement_model
from sbsi.models import ModelPaths, load_emulator
from sbsi.precision import MODES as PRECISION_MODES, precision_region
from sbsi.scene_prior import ScenePrior


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-store", required=True)
    parser.add_argument("--measurement-model", required=True)
    parser.add_argument("--emulator-metadata", required=True)
    parser.add_argument("--emulator-model", required=True)
    parser.add_argument("--model-cache", required=True)
    parser.add_argument("--proposal-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-detected", type=int, default=512)
    parser.add_argument("--mock-g1", type=float, default=0.0)
    parser.add_argument("--mock-g2", type=float, default=0.0)
    parser.add_argument("--center-g1", type=float, default=0.0)
    parser.add_argument("--center-g2", type=float, default=0.0)
    parser.add_argument("--h", type=float, default=0.005)
    parser.add_argument("--draw-ladder", nargs="+", type=int, default=(2048, 4096, 8192))
    parser.add_argument("--proposal-candidates", type=int, default=16384)
    parser.add_argument("--proposal-prefilter-candidates", type=int)
    parser.add_argument("--proposal-seed", type=int, default=8701)
    parser.add_argument("--proposal-epsilon", type=float, default=0.1)
    parser.add_argument("--min-ess", type=float, default=256.0)
    parser.add_argument("--max-weight-fraction", type=float, default=0.25)
    parser.add_argument(
        "--adaptive-allocation",
        choices=("production_prefix", "independent_pilot"),
        default="production_prefix",
    )
    parser.add_argument("--pilot-draws", type=int, default=512)
    parser.add_argument("--pilot-seed", type=int, default=18701)
    parser.add_argument("--pilot-safety-factor", type=float, default=1.0)
    parser.add_argument(
        "--adaptive-bias-correction",
        choices=("none", "richardson_1_over_m"),
        default="none",
    )
    parser.add_argument(
        "--candidate-backend", choices=("scipy", "torch"), default="scipy"
    )
    parser.add_argument("--object-chunk", type=int, default=128)
    parser.add_argument("--autograd-object-chunk", type=int, default=8)
    parser.add_argument("--atom-chunk", type=int, default=4096)
    parser.add_argument("--precision", choices=PRECISION_MODES, default="fp32")
    parser.add_argument("--compile-flow", action="store_true")
    parser.add_argument(
        "--compile-mode",
        choices=("default", "reduce-overhead", "max-autotune"),
        default="default",
    )
    parser.add_argument(
        "--compile-dynamic",
        action="store_true",
        help="Compile one dynamic-shape graph instead of specializing fixed batch shapes.",
    )
    parser.add_argument(
        "--warmup-rows",
        type=int,
        default=65536,
        help="Representative flow rows evaluated before timing; also pays compile cost.",
    )
    parser.add_argument("--skip-legacy", action="store_true")
    parser.add_argument("--skip-autograd", action="store_true")
    parser.add_argument(
        "--full-information",
        action="store_true",
        help="Benchmark the full nine-view 2x2 information used by one-step inference.",
    )
    parser.add_argument(
        "--retain-fixed-ladder",
        action="store_true",
        help=(
            "retain score and information at every nested fixed-draw rung; "
            "flow evaluations are unchanged because the maximum-rung atoms are "
            "already evaluated once"
        ),
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args(argv)


def _estimate(score, information):
    score = np.asarray(score, dtype=np.float64)
    information = np.asarray(information, dtype=np.float64)
    if information.ndim == 2:
        return np.sum(score, axis=0) / np.sum(information, axis=0)
    return np.linalg.solve(np.sum(information, axis=0), np.sum(score, axis=0))


def _diagonal_influence(score, information):
    estimate = _estimate(score, information)
    if information.ndim == 3:
        residual = score - np.einsum("nij,j->ni", information, estimate)
        return residual @ np.linalg.inv(np.mean(information, axis=0)).T
    return (score - estimate[None, :] * information) / np.mean(information, axis=0)


def _timed(callable_):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    result = callable_()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        peak = int(torch.cuda.max_memory_allocated())
    else:
        peak = 0
    return result, float(time.perf_counter() - started), peak


def main(argv=None):
    args = parse_args(argv)
    output = Path(args.output).resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}")
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
    target_dim = int(flow.model.target_dim)
    context_dim = int(flow.model.context_dim)
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

    def make_proposal():
        return DefensiveLocalProposal(
            coordinates,
            prior.weights,
            local_base_weights=cache.get(0.0, 0.0).detection_probability,
        )

    mock = generate_mock_catalogue(
        likelihood,
        n_detected=args.n_detected,
        g1=args.mock_g1,
        g2=args.mock_g2,
        scene_seed=2201,
        detection_seed=3201,
        flow_seed=4201,
    )
    mock.save(output / "mock")
    # The mock is always generated by the original eager fp32 checkpoint.  Compilation
    # and reduced precision are inference-only treatments in this benchmark, so every
    # execution mode sees byte-identical measurements rather than its own mock draw.
    if args.compile_flow:
        compile_mode = None if args.compile_mode == "default" else args.compile_mode
        flow.compile_log_prob(
            mode=compile_mode,
            dynamic=bool(args.compile_dynamic),
        )
    if args.warmup_rows <= 0:
        raise ValueError("warmup rows must be positive")
    warm_target = torch.zeros(
        (args.warmup_rows, target_dim), dtype=torch.float32, device=flow.device
    )
    warm_context = torch.zeros(
        (args.warmup_rows, context_dim), dtype=torch.float32, device=flow.device
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    warm_started = time.perf_counter()
    with torch.no_grad(), precision_region(args.precision, args.device):
        flow.model.log_prob(warm_target, warm_context)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    warmup_seconds = float(time.perf_counter() - warm_started)
    del warm_target, warm_context

    def timed(callable_):
        def execute():
            with precision_region(args.precision, args.device):
                return callable_()

        return _timed(execute)

    ladder = tuple(sorted(set(args.draw_ladder)))
    center = (float(args.center_g1), float(args.center_g2))
    if center != (0.0, 0.0) and (not args.full_information or not args.skip_legacy):
        raise ValueError(
            "a nonzero benchmark centre requires --full-information and --skip-legacy"
        )
    retain_fixed_ladder = bool(args.retain_fixed_ladder or args.full_information)
    fixed_ladder = ladder if retain_fixed_ladder else (ladder[-1],)
    common = dict(
        steps=(args.h,),
        ladder=fixed_ladder,
        n_candidates=args.proposal_candidates,
        epsilon=args.proposal_epsilon,
        bandwidth=1.0,
        proposal_seeds=(args.proposal_seed,),
        atom_chunk=args.atom_chunk,
        posterior_adapt_proposal=True,
        retain_object_moments=True,
    )
    results = {}
    if args.full_information and not args.skip_legacy:
        raise ValueError("the legacy path has no full 2x2 nine-view implementation")
    if not args.skip_legacy:
        likelihood.discard_tensor_views()
        legacy, elapsed, peak = timed(
            lambda: run_streamed_section5(
                likelihood,
                mock,
                make_proposal(),
                object_chunk=4,
                use_tensor_native=False,
                **common,
            )
        )
        moments = legacy.object_moments[args.proposal_seed]
        score = np.column_stack(
            [moments.score[(name, args.h, ladder[-1])] for name in ("g1", "g2")]
        )
        information = np.column_stack(
            [moments.information[(name, args.h, ladder[-1])] for name in ("g1", "g2")]
        )
        results["legacy_fixed"] = {
            "elapsed_seconds": elapsed,
            "peak_gpu_bytes": peak,
            "estimate": _estimate(score, information).tolist(),
            "flow_evaluations": legacy.results[0].flow_evaluations,
        }

    likelihood.discard_tensor_views()
    if args.full_information:
        fixed, elapsed, peak = timed(
            lambda: run_adaptive_section5(
                likelihood,
                mock,
                make_proposal(),
                center=center,
                h=args.h,
                draw_ladder=fixed_ladder,
                n_candidates=args.proposal_candidates,
                proposal_prefilter_candidates=args.proposal_prefilter_candidates,
                epsilon=args.proposal_epsilon,
                proposal_seed=args.proposal_seed,
                min_ess=np.inf,
                max_weight_fraction=1.0,
                allocation_method="production_prefix",
                full_information=True,
                retain_full_ladder=retain_fixed_ladder,
                candidate_backend=args.candidate_backend,
                object_chunk=args.object_chunk,
                atom_chunk=args.atom_chunk,
            )
        )
        moments = None
        fixed_score = fixed.score
        fixed_information = fixed.information
        fixed_flow_evaluations = fixed.flow_evaluations
    else:
        fixed, elapsed, peak = timed(
            lambda: run_streamed_section5(
                likelihood,
                mock,
                make_proposal(),
                object_chunk=args.object_chunk,
                use_tensor_native=True,
                **common,
            )
        )
        moments = fixed.object_moments[args.proposal_seed]
        fixed_score = np.column_stack(
            [moments.score[(name, args.h, ladder[-1])] for name in ("g1", "g2")]
        )
        fixed_information = np.column_stack(
            [moments.information[(name, args.h, ladder[-1])] for name in ("g1", "g2")]
        )
        fixed_flow_evaluations = fixed.results[0].flow_evaluations
    results["tensor_fixed"] = {
        "elapsed_seconds": elapsed,
        "peak_gpu_bytes": peak,
        "estimate": _estimate(fixed_score, fixed_information).tolist(),
        "flow_evaluations": fixed_flow_evaluations,
        "robust_standard_error": (
            np.std(_diagonal_influence(fixed_score, fixed_information), axis=0, ddof=1)
            / np.sqrt(len(fixed_score))
        ).tolist(),
    }
    fixed_elapsed_seconds = elapsed
    if retain_fixed_ladder:
        fixed_ladder_report = {}
        previous_score = None
        previous_information = None
        for rung_index, n_draws in enumerate(ladder):
            if args.full_information:
                rung_score = fixed.ladder_score[rung_index]
                rung_information = fixed.ladder_information[rung_index]
            else:
                rung_score = np.column_stack(
                    [
                        moments.score[(name, args.h, n_draws)]
                        for name in ("g1", "g2")
                    ]
                )
                rung_information = np.column_stack(
                    [
                        moments.information[(name, args.h, n_draws)]
                        for name in ("g1", "g2")
                    ]
                )
            rung_estimate = _estimate(rung_score, rung_information)
            rung_report = {
                "estimate": rung_estimate.tolist(),
                "robust_standard_error": (
                    np.std(
                        _diagonal_influence(rung_score, rung_information),
                        axis=0,
                        ddof=1,
                    )
                    / np.sqrt(len(rung_score))
                ).tolist(),
            }
            if previous_score is not None:
                previous_estimate = _estimate(
                    previous_score, previous_information
                )
                paired = _diagonal_influence(
                    rung_score, rung_information
                ) - _diagonal_influence(
                    previous_score, previous_information
                )
                paired_se = np.std(paired, axis=0, ddof=1) / np.sqrt(len(paired))
                shift = rung_estimate - previous_estimate
                rung_report["previous_rung_shift"] = shift.tolist()
                rung_report["previous_rung_paired_standard_error"] = paired_se.tolist()
                rung_report["previous_rung_pull"] = np.divide(
                    shift,
                    paired_se,
                    out=np.full(2, np.nan),
                    where=paired_se > 0,
                ).tolist()
            fixed_ladder_report[str(n_draws)] = rung_report
            previous_score = rung_score
            previous_information = rung_information
        results["fixed_ladder"] = fixed_ladder_report

    adaptive_kwargs = dict(
        draw_ladder=ladder,
        n_candidates=args.proposal_candidates,
        epsilon=args.proposal_epsilon,
        proposal_seed=args.proposal_seed,
        min_ess=args.min_ess,
        max_weight_fraction=args.max_weight_fraction,
        atom_chunk=args.atom_chunk,
    )
    likelihood.discard_tensor_views()
    adaptive, elapsed, peak = timed(
        lambda: run_adaptive_section5(
            likelihood,
            mock,
            make_proposal(),
            center=center,
            h=args.h,
            object_chunk=args.object_chunk,
            proposal_prefilter_candidates=args.proposal_prefilter_candidates,
            allocation_method=args.adaptive_allocation,
            pilot_draws=args.pilot_draws,
            pilot_seed=args.pilot_seed,
            pilot_safety_factor=args.pilot_safety_factor,
            bias_correction=args.adaptive_bias_correction,
            candidate_backend=args.candidate_backend,
            full_information=args.full_information,
            **adaptive_kwargs,
        )
    )
    results["tensor_adaptive"] = {
        "elapsed_seconds": elapsed,
        "peak_gpu_bytes": peak,
        "estimate": _estimate(adaptive.score, adaptive.information).tolist(),
        "flow_evaluations": adaptive.flow_evaluations,
        "draw_count_percentiles": np.percentile(
            adaptive.draw_counts, [0, 25, 50, 75, 90, 100]
        ).tolist(),
        "mean_unique_fraction": float(np.mean(adaptive.unique_counts / adaptive.draw_counts)),
        "allocation_method": adaptive.allocation_method,
        "bias_correction": adaptive.bias_correction,
        "candidate_backend": adaptive.candidate_backend,
        "pilot_draws": adaptive.pilot_draws,
        "pilot_seed": adaptive.pilot_seed,
        "pilot_ess_fraction_percentiles": (
            None
            if adaptive.pilot_ess_fraction is None
            else np.percentile(
                adaptive.pilot_ess_fraction, [0, 10, 50, 90, 100]
            ).tolist()
        ),
        "pilot_max_weight_fraction_percentiles": (
            None
            if adaptive.pilot_max_weight_fraction is None
            else np.percentile(
                adaptive.pilot_max_weight_fraction, [0, 10, 50, 90, 100]
            ).tolist()
        ),
        "phase_seconds": dict(adaptive.phase_seconds),
        "robust_standard_error": (
            np.std(_diagonal_influence(adaptive.score, adaptive.information), axis=0, ddof=1)
            / np.sqrt(len(adaptive.score))
        ).tolist(),
        "speedup_vs_fixed": float(fixed_elapsed_seconds / elapsed),
    }

    paired_influence = _diagonal_influence(
        adaptive.score, adaptive.information
    ) - _diagonal_influence(fixed_score, fixed_information)
    difference = _estimate(adaptive.score, adaptive.information) - _estimate(
        fixed_score, fixed_information
    )
    difference_se = np.std(paired_influence, axis=0, ddof=1) / np.sqrt(len(paired_influence))
    results["fixed_vs_adaptive"] = {
        "estimate_difference": difference.tolist(),
        "paired_standard_error": difference_se.tolist(),
        "paired_pull": np.divide(
            difference,
            difference_se,
            out=np.full(2, np.nan),
            where=difference_se > 0,
        ).tolist(),
        "passed_025_paired_se": bool(
            np.all(
                np.abs(difference)
                <= 0.25 * difference_se
            )
        ),
        "maximum_absolute_shift": float(np.max(np.abs(difference))),
    }

    if not args.skip_autograd:
        likelihood.discard_tensor_views()
        autograd, elapsed, peak = timed(
            lambda: autograd_importance_section5(
                likelihood,
                mock,
                make_proposal(),
                object_chunk=args.autograd_object_chunk,
                **adaptive_kwargs,
            )
        )
        results["autograd_adaptive"] = {
            "elapsed_seconds": elapsed,
            "peak_gpu_bytes": peak,
            "estimate": _estimate(autograd.score, autograd.information).tolist(),
            "diagonal_estimate": _estimate(
                autograd.score,
                np.diagonal(autograd.information, axis1=1, axis2=2),
            ).tolist(),
            "flow_evaluations": autograd.flow_evaluations,
            "draw_count_percentiles": np.percentile(
                autograd.draw_counts, [0, 25, 50, 75, 90, 100]
            ).tolist(),
            "mean_unique_fraction": float(np.mean(autograd.unique_counts / autograd.draw_counts)),
            "mean_off_diagonal_information": float(
                np.mean((autograd.information[:, 0, 1] + autograd.information[:, 1, 0]) / 2)
            ),
        }
    moment_arrays = {
        "fixed_score": fixed_score,
        "fixed_information": fixed_information,
        "adaptive_score": adaptive.score,
        "adaptive_information": adaptive.information,
        "adaptive_draw_counts": adaptive.draw_counts,
        "adaptive_unique_counts": adaptive.unique_counts,
    }
    if adaptive.pilot_ess_fraction is not None:
        moment_arrays.update(
            {
                "adaptive_pilot_ess_fraction": adaptive.pilot_ess_fraction,
                "adaptive_pilot_max_weight_fraction": (
                    adaptive.pilot_max_weight_fraction
                ),
            }
        )
    if retain_fixed_ladder:
        for rung_index, n_draws in enumerate(ladder):
            if args.full_information:
                moment_arrays[f"fixed_score_m{n_draws}"] = fixed.ladder_score[
                    rung_index
                ]
                moment_arrays[f"fixed_information_m{n_draws}"] = (
                    fixed.ladder_information[rung_index]
                )
            else:
                moment_arrays[f"fixed_score_m{n_draws}"] = np.column_stack(
                    [
                        moments.score[(name, args.h, n_draws)]
                        for name in ("g1", "g2")
                    ]
                )
                moment_arrays[f"fixed_information_m{n_draws}"] = np.column_stack(
                    [
                        moments.information[(name, args.h, n_draws)]
                        for name in ("g1", "g2")
                    ]
                )
    if not args.skip_autograd:
        moment_arrays.update(
            {
                "autograd_score": autograd.score,
                "autograd_information": autograd.information,
                "autograd_draw_counts": autograd.draw_counts,
            }
        )
    np.savez_compressed(output / "moments.npz", **moment_arrays)
    payload = {
        "method": "section5_optimization_benchmark_v1",
        "config": vars(args),
        "execution": {
            "torch_version": torch.__version__,
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            "precision": args.precision,
            "compiled": bool(args.compile_flow),
            "compile_target": "model.log_prob" if args.compile_flow else None,
            "compile_mode": args.compile_mode if args.compile_flow else None,
            "compile_dynamic": bool(args.compile_dynamic) if args.compile_flow else None,
            "warmup_rows": int(args.warmup_rows),
            "warmup_seconds": warmup_seconds,
        },
        "n_objects": len(mock.measurements),
        "moments": "moments.npz",
        "results": results,
        "retained_fixed_ladder": retain_fixed_ladder,
    }
    (output / "result.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
