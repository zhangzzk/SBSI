#!/usr/bin/env python
"""Light synthetic benchmark for the conditioned DM-MIS proposal scan.

The production benchmark belongs on a scheduled A40 and the already-used
proposal-development rows.  This small harness is intentionally safe on a
login-node CPU: it measures the same two-pass constructor at several fixed
proposal object chunks, verifies common streams, and reports numerical drift
and CUDA peak allocation when a GPU is requested.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch

from sbsi.catalogue_sampling import (
    DefensiveLocalProposal,
    ProposalCandidates,
    ProposalCoordinateTable,
    TAIL_PROPOSAL_RECIPE,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--atoms", type=int, default=32768)
    parser.add_argument("--objects", type=int, default=16)
    parser.add_argument("--dimensions", type=int, default=4)
    parser.add_argument("--candidates", type=int, default=256)
    parser.add_argument("--draws", type=int, default=256)
    parser.add_argument("--atom-chunk", type=int, default=8192)
    parser.add_argument("--object-chunks", default="1,8,16,32")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260830)
    return parser.parse_args()


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    args = _arguments()
    device = torch.device(args.device)
    if args.atoms <= 1 or args.objects <= 0 or args.dimensions <= 0:
        raise ValueError("atoms, objects, and dimensions must be positive")
    if not 0 <= args.candidates < args.atoms:
        raise ValueError("candidates must lie in [0, atoms)")
    if args.draws <= 0 or args.draws % TAIL_PROPOSAL_RECIPE.cycle_length:
        raise ValueError(
            "draws must be a positive multiple of the proposal cycle length"
        )
    object_chunks = tuple(
        int(value) for value in args.object_chunks.split(",") if value.strip()
    )
    if not object_chunks or min(object_chunks) <= 0 or args.repeats <= 0:
        raise ValueError("object chunks must be positive")

    rng = np.random.default_rng(args.seed)
    names = tuple(f"x{dimension}" for dimension in range(args.dimensions))
    values = rng.normal(size=(args.atoms, args.dimensions))
    dispersion = np.exp(
        rng.normal(loc=-0.5, scale=0.35, size=values.shape)
    )
    prior = rng.random(args.atoms)
    prior /= prior.sum()
    detection = 0.1 + 0.9 * rng.random(args.atoms)
    observations = rng.normal(size=(args.objects, args.dimensions))
    coordinates = ProposalCoordinateTable(
        values=values,
        target_names=names,
        center=np.zeros(args.dimensions),
        scale=np.ones(args.dimensions),
        dispersion=dispersion,
        statistic="mean",
    )
    proposal = DefensiveLocalProposal(
        coordinates, prior, local_base_weights=detection
    )
    excluded = np.stack(
        [
            rng.choice(args.atoms, size=args.candidates, replace=False)
            for _ in range(args.objects)
        ]
    ).astype(np.int64)
    candidates = ProposalCandidates(
        indices=excluded,
        distances=np.zeros(excluded.shape, dtype=np.float64),
        radius=np.ones(args.objects, dtype=np.float64),
    )
    observed = pd.DataFrame(observations, columns=names)
    # Build and transfer immutable proxy coefficients outside every timing.
    proposal._tilted_proxy(device)

    def timed_draw(object_chunk: int):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        _synchronize(device)
        started = time.perf_counter()
        draw = proposal.draw_tilted_mis(
            candidates,
            observed,
            n_draws=args.draws,
            seed=args.seed + 1,
            device=device,
            atom_chunk=args.atom_chunk,
            object_chunk=object_chunk,
        )
        _synchronize(device)
        return (
            draw,
            time.perf_counter() - started,
            (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else None
            ),
        )

    # Warm every shape before timing so chunk one does not pay one-off Torch
    # initialization on behalf of the other arms.
    warmed = {chunk: timed_draw(chunk)[0] for chunk in object_chunks}
    reference = warmed[object_chunks[0]]
    records = []
    for object_chunk in object_chunks:
        timings = []
        peaks = []
        draw = None
        for _ in range(args.repeats):
            draw, seconds, peak = timed_draw(object_chunk)
            timings.append(seconds)
            if peak is not None:
                peaks.append(peak)
        record = {
            "object_chunk": object_chunk,
            "repeats": args.repeats,
            "seconds_median": float(np.median(timings)),
            "seconds_minimum": float(np.min(timings)),
            "cuda_peak_allocated_bytes": (
                max(peaks) if peaks else None
            ),
        }
        if object_chunk == object_chunks[0]:
            reference = draw
            record.update(
                {
                    "index_mismatch_fraction_vs_chunk1": 0.0,
                    "probability_max_abs_vs_chunk1": 0.0,
                    "probability_max_rel_vs_chunk1": 0.0,
                    "complement_mass_max_abs_vs_chunk1": 0.0,
                }
            )
        else:
            absolute = np.abs(draw.probability - reference.probability)
            relative = absolute / np.maximum(
                np.abs(reference.probability), np.finfo(np.float64).tiny
            )
            record.update(
                {
                    "index_mismatch_fraction_vs_chunk1": float(
                        np.mean(draw.indices != reference.indices)
                    ),
                    "probability_max_abs_vs_chunk1": float(np.max(absolute)),
                    "probability_max_rel_vs_chunk1": float(np.max(relative)),
                    "complement_mass_max_abs_vs_chunk1": float(
                        np.max(
                            np.abs(
                                draw.component_complement_mass
                                - reference.component_complement_mass
                            )
                        )
                    ),
                }
            )
        records.append(record)

    print(
        json.dumps(
            {
                "device": str(device),
                "atoms": args.atoms,
                "objects": args.objects,
                "dimensions": args.dimensions,
                "candidates": args.candidates,
                "draws": args.draws,
                "atom_chunk": args.atom_chunk,
                "repeats": args.repeats,
                "recipe": TAIL_PROPOSAL_RECIPE.name,
                "results": records,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
