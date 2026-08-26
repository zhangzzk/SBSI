#!/usr/bin/env python
# Archived Infer V1 optimization experiment.
"""Compare Torch and JAX execution of one frozen mean-affine flow kernel."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from sbsi.jax_flow import build_jax_log_prob_from_torch
from sbsi.measurement_model import load_measurement_model


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurement-model", required=True)
    parser.add_argument(
        "--flow-context-parquet",
        help=(
            "optional cached catalogue flow view; when supplied, benchmark "
            "real standardized conditions and targets sampled from the flow"
        ),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--rows-per-view", type=int, default=65536)
    parser.add_argument("--views", type=int, default=9)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--seed", type=int, default=19101)
    parser.add_argument("--skip-torch-compile", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def _summarize(times, rows):
    values = np.asarray(times, dtype=np.float64)
    median = float(np.median(values))
    return {
        "seconds": values.tolist(),
        "median_seconds": median,
        "rows_per_second": float(rows / median),
    }


def _difference(actual, expected):
    residual = np.asarray(actual, dtype=np.float64) - np.asarray(expected, dtype=np.float64)
    return {
        "maximum_absolute": float(np.max(np.abs(residual))),
        "rms": float(np.sqrt(np.mean(np.square(residual)))),
        "mean": float(np.mean(residual)),
    }


def main(argv=None):
    args = parse_args(argv)
    if args.rows_per_view <= 0 or args.views <= 0 or args.repeats <= 0:
        raise ValueError("rows, views, and repeats must be positive")
    output = Path(args.output).resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}")
    output.mkdir(parents=True)

    # JAX otherwise reserves most accelerator memory, which prevents a fair
    # same-process comparison with Torch on smaller cards.
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import jax
    import jax.numpy as jnp

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        if not args.allow_cpu:
            raise RuntimeError("CUDA is unavailable; submit this benchmark through SLURM")
        args.device = "cpu"
    jax_devices = jax.devices()
    jax_gpu_available = any(
        device.platform in ("gpu", "cuda") for device in jax_devices
    )

    eager = load_measurement_model(args.measurement_model, device=args.device)
    model = eager.model
    target_dim = int(model.target_dim)
    context_dim = int(model.context_dim)
    rng = np.random.default_rng(args.seed)
    if args.flow_context_parquet:
        names = list(eager.condition_preprocessor.feature_names)
        frame = pd.read_parquet(args.flow_context_parquet, columns=names)
        if not len(frame):
            raise ValueError("flow context parquet is empty")
        selected = rng.choice(len(frame), size=args.rows_per_view, replace=True)
        context_base = eager.context_tensor(
            frame.iloc[selected].reset_index(drop=True)
        )
        torch.manual_seed(args.seed + 1)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed + 1)
        with torch.no_grad():
            target_torch = model.sample(
                context_base,
                n_samples=args.views,
            ).permute(1, 0, 2).contiguous()
        context_torch = (
            context_base[None, :, :]
            .expand(args.views, args.rows_per_view, context_dim)
            .contiguous()
        )
        target = target_torch.detach().cpu().numpy()
        context = context_torch.detach().cpu().numpy()
        input_kind = "catalogue_context_flow_draws"
    else:
        target = rng.normal(
            size=(args.views, args.rows_per_view, target_dim)
        ).astype(np.float32)
        context = rng.normal(
            size=(args.views, args.rows_per_view, context_dim)
        ).astype(np.float32)
        target_torch = torch.as_tensor(target, device=eager.device)
        context_torch = torch.as_tensor(context, device=eager.device)
        input_kind = "independent_standard_normal_stress"
    flat_target = target_torch.reshape(-1, target_dim)
    flat_context = context_torch.reshape(-1, context_dim)
    total_rows = int(args.views * args.rows_per_view)

    def synchronize_torch():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def torch_sequential():
        return torch.stack(
            [model.log_prob(target_torch[index], context_torch[index]) for index in range(args.views)]
        )

    def torch_fused():
        return model.log_prob(flat_target, flat_context).reshape(
            args.views, args.rows_per_view
        )

    with torch.no_grad():
        reference_torch = torch_sequential()
        torch_fused_output = torch_fused()
    synchronize_torch()
    reference = reference_torch.detach().cpu().numpy()
    fused_reference = torch_fused_output.detach().cpu().numpy()
    correctness = {
        "torch_fused_vs_sequential": _difference(fused_reference, reference)
    }

    timings = {}
    for name, function in (("torch_sequential", torch_sequential), ("torch_fused", torch_fused)):
        samples = []
        with torch.no_grad():
            function()
            synchronize_torch()
            for _ in range(args.repeats):
                started = time.perf_counter()
                function()
                synchronize_torch()
                samples.append(time.perf_counter() - started)
        timings[name] = _summarize(samples, total_rows)

    if not args.skip_torch_compile:
        compiled = load_measurement_model(args.measurement_model, device=args.device)
        compiled.compile_log_prob(mode=None, dynamic=False)
        compiled_target = target_torch
        compiled_context = context_torch

        def torch_compiled_fused():
            return compiled.model.log_prob(
                compiled_target.reshape(-1, target_dim),
                compiled_context.reshape(-1, context_dim),
            ).reshape(args.views, args.rows_per_view)

        with torch.no_grad():
            compile_started = time.perf_counter()
            compiled_output = torch_compiled_fused()
            synchronize_torch()
            compile_seconds = float(time.perf_counter() - compile_started)
            samples = []
            for _ in range(args.repeats):
                started = time.perf_counter()
                torch_compiled_fused()
                synchronize_torch()
                samples.append(time.perf_counter() - started)
        timings["torch_compiled_fused"] = _summarize(samples, total_rows)
        timings["torch_compiled_fused"]["compile_seconds"] = compile_seconds
        correctness["torch_compiled_fused_vs_eager_fused"] = _difference(
            compiled_output.detach().cpu().numpy(), fused_reference
        )

    if not args.device.startswith("cuda") or jax_gpu_available:
        jax_log_prob = build_jax_log_prob_from_torch(model)
        target_jax = jnp.asarray(target)
        context_jax = jnp.asarray(context)
        compile_started = time.perf_counter()
        jax_output = jax_log_prob(target_jax, context_jax)
        jax.block_until_ready(jax_output)
        jax_compile_seconds = float(time.perf_counter() - compile_started)
        samples = []
        for _ in range(args.repeats):
            started = time.perf_counter()
            value = jax_log_prob(target_jax, context_jax)
            jax.block_until_ready(value)
            samples.append(time.perf_counter() - started)
        timings["jax_fused"] = _summarize(samples, total_rows)
        timings["jax_fused"]["compile_seconds"] = jax_compile_seconds
        correctness["jax_fused_vs_torch_fused"] = _difference(
            np.asarray(jax_output), fused_reference
        )

    eager_median = timings["torch_sequential"]["median_seconds"]
    for report in timings.values():
        if "median_seconds" in report:
            report["speedup_vs_torch_sequential"] = float(
                eager_median / report["median_seconds"]
            )
    best_torch_fused = min(
        report["median_seconds"]
        for name, report in timings.items()
        if name.startswith("torch_") and name != "torch_sequential"
    )
    jax_correctness = correctness.get("jax_fused_vs_torch_fused")
    jax_timing = timings.get("jax_fused")
    payload = {
        "method": "jax_mean_affine_hot_kernel_benchmark_v1",
        "config": vars(args),
        "execution": {
            "torch_version": torch.__version__,
            "jax_version": jax.__version__,
            "torch_device": str(eager.device),
            "torch_gpu": (
                torch.cuda.get_device_name() if torch.cuda.is_available() else None
            ),
            "jax_devices": [str(device) for device in jax_devices],
            "jax_gpu_available": bool(jax_gpu_available),
            "input_kind": input_kind,
            "total_rows_per_repeat": total_rows,
        },
        "correctness": correctness,
        "timings": timings,
        "gates": {
            "jax_kernel_max_abs_below_0p01": (
                None
                if jax_correctness is None
                else bool(jax_correctness["maximum_absolute"] < 0.01)
            ),
            "jax_kernel_rms_below_0p001": (
                None
                if jax_correctness is None
                else bool(jax_correctness["rms"] < 0.001)
            ),
            "jax_incremental_speedup_at_least_1p5": (
                None
                if jax_timing is None
                else bool(jax_timing["median_seconds"] <= best_torch_fused / 1.5)
            ),
        },
    }
    (output / "result.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
