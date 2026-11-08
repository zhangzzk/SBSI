#!/usr/bin/env python
"""Synthetic, full-size proposal throughput probe; never inference data."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from sbsi.catalogue_sampling import WholeCatalogueProxy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atoms", type=int, default=139936000)
    parser.add_argument("--chunk", type=int, default=2)
    parser.add_argument("--cdf-block-size", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.atoms < 1 or args.chunk < 1:
        raise ValueError("new output, positive atoms and chunk required")
    torch.set_num_threads(2)
    # Allocate the exact production tensor layouts directly on the GPU. This
    # avoids making large synthetic host catalogues that carry no science.
    proxy = WholeCatalogueProxy.__new__(WholeCatalogueProxy)
    proxy.score_dtype = torch.float32
    proxy.cdf_block_size = args.cdf_block_size
    proxy.a1 = torch.full((args.atoms, 4), .2, device="cuda", dtype=torch.float32)
    proxy.a2 = torch.full((args.atoms, 4), .3, device="cuda", dtype=torch.float32)
    proxy.constant = torch.full((args.atoms,), -2., device="cuda", dtype=torch.float64)
    proxy.active = torch.arange(args.atoms, device="cuda", dtype=torch.int64)
    proxy.prior = torch.full((args.atoms,), 1/args.atoms, device="cuda", dtype=torch.float64)
    rng = np.random.default_rng(20260920)
    times = []
    for i in range(5):
        observations = rng.normal(size=(args.chunk, 4))
        uniforms = rng.uniform(size=(args.chunk, 16384))
        torch.cuda.synchronize()
        start = time.monotonic()
        atoms, probability = proxy.draw_uniforms_batch(observations, uniforms, delta=.1)
        torch.cuda.synchronize()
        seconds = time.monotonic()-start
        np.testing.assert_allclose(probability, 1/args.atoms, rtol=1e-12)
        assert np.all((atoms >= 0) & (atoms < args.atoms))
        times.append(seconds)
        print(f"TILTED atoms={args.atoms} objects={args.chunk} iteration={i} seconds={seconds:.5f}", flush=True)
    result = dict(status="benchmark_complete", synthetic=True, atoms=args.atoms, chunk=args.chunk,
        cdf_block_size=args.cdf_block_size,
        seconds=times, gpu=torch.cuda.get_device_name(), peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        four_gpu_500k_proposal_hours=float(np.median(times[1:]))/args.chunk*500000/4/3600,
        limitations=["Synthetic uniform proxy at full size; no scientific catalogue or estimate.",
                     "Isolates tilted sampling; excludes local candidate retrieval, data preparation and likelihood evaluation.",
                     "No flow/context tensors resident: production must budget additional GPU memory."])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(f"TILTED_BENCHMARK_COMPLETE {args.output}", flush=True)


if __name__ == "__main__":
    main()
