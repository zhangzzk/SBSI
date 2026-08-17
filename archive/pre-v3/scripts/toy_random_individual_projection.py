"""Compare random one-neighbour projections with coherent toy-scene response.

For every random draw, each neighbour receives an independent spin-2 shear
direction, but neighbours are rendered one at a time.  The measured primary
response is projected onto the active neighbour's direction and the projections
are summed.  This removes the cross-neighbour label terms present when several
neighbours are sheared in the same image.

The reference is the direction-averaged coherent central response, not only the
fixed g1 component.  The two are both reported, because a random projection
estimates half the response trace whereas a fixed-g1 coherent render estimates a
particular matrix component in an anisotropic scene.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from scripts.toy_random_half_response import (
    Measurer,
    directional_response,
    isotropic_response,
    scenarios,
    unit,
)


def summarize(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("summary requires at least two finite one-dimensional values")
    return {
        "mean": float(values.mean()),
        "direction_sem": float(values.std(ddof=1) / np.sqrt(len(values))),
        "direction_sd": float(values.std(ddof=1)),
        "n_draw": int(len(values)),
        "percentiles_2p5_50_97p5": [
            float(x) for x in np.percentile(values, [2.5, 50.0, 97.5])
        ],
    }


def one_direction_sum(measurer: Measurer, directions: np.ndarray, g: float) -> float:
    """Render neighbours separately and sum their matching projections."""
    n_nbr = len(measurer.scene) - 1
    directions = np.asarray(directions, dtype=float)
    if directions.shape != (n_nbr, 2):
        raise ValueError(f"directions shape {directions.shape} != {(n_nbr, 2)}")
    total = 0.0
    for j, direction in enumerate(directions):
        active = np.zeros(n_nbr, dtype=bool)
        active[j] = True
        response = directional_response(measurer, active, directions, g, central=True)
        total += float(response @ direction)
    return total


def random_individual_protocol(
    measurer: Measurer, g: float, n_draw: int, seed: int,
) -> tuple[dict, dict]:
    """Return raw and orthogonal-paired random one-at-a-time sums.

    The raw estimator uses one independent random direction per neighbour.  The
    paired estimator averages that projection with the projection in the
    orthogonal spin-2 direction.  At linear order the paired value equals half
    the trace exactly for every neighbour, so its remaining variation isolates
    finite-amplitude/direction-dependent measurement effects.
    """
    n_nbr = len(measurer.scene) - 1
    rng = np.random.RandomState(seed)
    raw = []
    paired = []
    failures = 0
    for _ in range(n_draw):
        directions = unit(rng.uniform(0.0, 2.0 * np.pi, size=n_nbr))
        orthogonal = np.column_stack([-directions[:, 1], directions[:, 0]])
        try:
            value = one_direction_sum(measurer, directions, g)
            value_orthogonal = one_direction_sum(measurer, orthogonal, g)
        except Exception:
            failures += 1
            continue
        raw.append(value)
        paired.append(0.5 * (value + value_orthogonal))
    if len(raw) < 0.95 * n_draw:
        raise RuntimeError(f"too many failed random-direction draws: {failures}/{n_draw}")
    raw_stat = summarize(np.asarray(raw))
    paired_stat = summarize(np.asarray(paired))
    raw_stat["n_failure"] = int(failures)
    paired_stat["n_failure"] = int(failures)
    return raw_stat, paired_stat


def coherent_fixed_direction(measurer: Measurer, g: float) -> float:
    n_nbr = len(measurer.scene) - 1
    direction = np.array([1.0, 0.0])
    directions = np.repeat(direction[None, :], n_nbr, axis=0)
    response = directional_response(
        measurer, np.ones(n_nbr, dtype=bool), directions, g, central=True,
    )
    return float(response @ direction)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--g", type=float, default=0.05)
    ap.add_argument("--n-draw", type=int, default=256)
    ap.add_argument("--n-angles", type=int, default=32)
    ap.add_argument("--stamp", type=int, default=112)
    ap.add_argument("--sky", type=float, default=0.0)
    ap.add_argument("--noise-seed", type=int, default=773)
    ap.add_argument("--fit-seed", type=int, default=42)
    ap.add_argument("--seed", type=int, default=8675309)
    ap.add_argument("--scenario", choices=["all", *scenarios().keys()], default="all")
    ap.add_argument("--output-json", required=True)
    args = ap.parse_args()
    if args.g <= 0.0:
        raise ValueError("--g must be positive")
    if args.n_draw < 32:
        raise ValueError("--n-draw must be at least 32")
    if args.n_angles < 8 or args.n_angles % 4:
        raise ValueError("--n-angles must be a multiple of four and at least eight")
    if os.path.exists(args.output_json):
        raise FileExistsError(args.output_json)

    selected = scenarios()
    if args.scenario != "all":
        selected = {args.scenario: selected[args.scenario]}
    payload = {
        "design": {
            "g": float(args.g), "n_draw": int(args.n_draw),
            "n_angles": int(args.n_angles), "sky": float(args.sky),
            "seed": int(args.seed), "noise_seed": int(args.noise_seed),
            "primary_sheared": False, "positions_sheared": False,
            "randomization_unit": "one independent spin-2 direction per neighbour and draw",
            "rendering": "one active neighbour per image arm; all other sources present unsheared",
            "reference": "coherent central response averaged over spin-2 directions",
        },
        "scenarios": {},
    }
    print("RANDOM-DIRECTION INDIVIDUAL-NEIGHBOUR TOY", flush=True)
    for index, (name, (target, neighbours)) in enumerate(selected.items()):
        measurer = Measurer(
            [target, *neighbours], args.stamp, args.sky,
            args.noise_seed, args.fit_seed,
        )
        coherent = isotropic_response(
            measurer, args.g, args.n_angles, per_neighbour=False,
        )
        individual_quadrature = isotropic_response(
            measurer, args.g, args.n_angles, per_neighbour=True,
        )
        raw, paired = random_individual_protocol(
            measurer, args.g, args.n_draw, args.seed + 100000 * index,
        )
        coherent_mean = float(coherent["mean"])
        raw["minus_coherent"] = float(raw["mean"] - coherent_mean)
        paired["minus_coherent"] = float(paired["mean"] - coherent_mean)
        item = {
            "n_neighbours": int(len(neighbours)),
            "coherent_central_isotropic": coherent,
            "coherent_central_fixed_g1": coherent_fixed_direction(measurer, args.g),
            "individual_central_isotropic_quadrature": individual_quadrature,
            "random_individual_projection_sum": raw,
            "random_individual_projection_sum_orthogonal_paired": paired,
            "exact_additivity_contrast": float(
                coherent_mean - individual_quadrature["mean"]
            ),
        }
        payload["scenarios"][name] = item
        print(
            f"{name}: Nnbr={len(neighbours)} coherent_iso={coherent_mean:+.6f} "
            f"random_sum={raw['mean']:+.6f} +/- {raw['direction_sem']:.6f} "
            f"delta={raw['minus_coherent']:+.6f}; "
            f"orth_pair={paired['mean']:+.6f} +/- {paired['direction_sem']:.6f} "
            f"delta={paired['minus_coherent']:+.6f}",
            flush=True,
        )

    output = os.path.abspath(args.output_json)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(f"wrote {output}")
    print("TOY_RANDOM_INDIVIDUAL_DONE", flush=True)


if __name__ == "__main__":
    main()
