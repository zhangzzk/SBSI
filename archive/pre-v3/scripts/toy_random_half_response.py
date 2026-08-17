"""Toy tomography of BlendEMU's random-half, random-direction response labels.

The deployed pair emulator is trained on a forward ``g=0 -> 0.2`` image pair.  A
random half of the input galaxies are secondaries and are sheared in independent
spin-2 directions.  The *same measured primary shape change* is projected onto
each designated secondary direction to form the pair labels.  At first order,
averaging those labels must recover the sum of the individual-neighbour response
traces.  This script tests that assumption with the actual GalSim + ngmix toy
measurement, before another full simulation is attempted.

For each fixed scene it compares:

* coherent, antithetic, direction-averaged neighbour response (anchor estimand);
* the sum of one-neighbour-at-a-time antithetic responses (additivity control);
* random-half labels, using the production forward convention;
* the same random-half draws with central differences (forward-term control);
* all neighbours active in independent random directions (participation control).

Positions are fixed under shear, matching the production response simulation.
The primary is never sheared.  This is a mechanism toy, not a population estimate:
even a positive result only says that the mechanism can carry the required sign
and scale in controlled scenes.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass

import galsim
import numpy as np

SBSI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SBSI_ROOT not in sys.path:
    sys.path.insert(0, SBSI_ROOT)

from archive.toy_blend_linearity import BETA, PIX, PSF_FWHM, _PSF, measure  # noqa: E402


ZERO_POINT = 30.0
PIXEL_RMS = 0.312


def source(mag, re_arcsec, n, x, y, e1=0.0, e2=0.0):
    return {
        "mag": float(mag), "re": float(re_arcsec), "n": float(n),
        "x": float(x), "y": float(y), "e1": float(e1), "e2": float(e2),
    }


def polar_source(mag, re_arcsec, n, distance, angle_deg, e=0.0, e_angle_deg=0.0):
    angle = np.deg2rad(angle_deg)
    e_angle = np.deg2rad(e_angle_deg)
    return source(
        mag, re_arcsec, n, distance * np.cos(angle), distance * np.sin(angle),
        e * np.cos(2.0 * e_angle), e * np.sin(2.0 * e_angle),
    )


def scenarios():
    target = source(24.5, 0.70, 1.0, 0.0, 0.0, e1=0.18, e2=-0.08)
    return {
        "one_control": (
            target,
            [polar_source(25.0, 0.55, 1.0, 1.0, 35, e=0.22, e_angle_deg=71)],
        ),
        "four_mixed": (
            target,
            [polar_source(24.0, 0.80, 1.5, 0.7, 20, e=0.25, e_angle_deg=5),
             polar_source(25.0, 0.55, 1.0, 1.0, 110, e=0.18, e_angle_deg=63),
             polar_source(26.0, 0.40, 0.8, 1.6, 205, e=0.30, e_angle_deg=117),
             polar_source(27.0, 0.30, 1.0, 2.5, 300, e=0.12, e_angle_deg=151)],
        ),
        "eight_faint_close": (
            target,
            [polar_source(
                26.2 + 0.1 * (j % 3), 0.32 + 0.03 * (j % 2), 1.0,
                0.75 + 0.09 * j, 17 + 43 * j, e=0.12 + 0.025 * j,
                e_angle_deg=11 + 29 * j,
            ) for j in range(8)],
        ),
        "six_asymmetric": (
            target,
            [polar_source(24.3, 0.70, 1.2, 0.62, 12, e=0.28, e_angle_deg=84),
             polar_source(25.1, 0.48, 0.9, 0.88, 58, e=0.15, e_angle_deg=12),
             polar_source(25.6, 0.42, 1.0, 1.05, 133, e=0.32, e_angle_deg=101),
             polar_source(26.4, 0.35, 1.0, 1.38, 196, e=0.22, e_angle_deg=37),
             polar_source(26.8, 0.30, 0.8, 1.85, 246, e=0.18, e_angle_deg=149),
             polar_source(27.2, 0.28, 1.0, 2.70, 319, e=0.27, e_angle_deg=73)],
        ),
    }


def render(scene, shears, stamp):
    """Render with an arbitrary ``(g1,g2)`` per source; catalogue positions stay fixed."""
    if len(scene) != len(shears):
        raise ValueError("scene/shears length mismatch")
    image = galsim.ImageF(stamp, stamp, scale=PIX)
    for gal, (g1, g2) in zip(scene, shears):
        flux = 10.0 ** (-0.4 * (gal["mag"] - ZERO_POINT))
        obj = galsim.Sersic(n=gal["n"], half_light_radius=gal["re"], flux=flux)
        if gal["e1"] or gal["e2"]:
            obj = obj.shear(g1=gal["e1"], g2=gal["e2"])
        if g1 or g2:
            obj = obj.shear(g1=float(g1), g2=float(g2))
        galsim.Convolve([obj, _PSF]).drawImage(
            image=image, add_to_image=True,
            offset=(gal["x"] / PIX, gal["y"] / PIX),
        )
    return image.array


@dataclass
class Measurer:
    scene: list[dict]
    stamp: int
    sky: float
    noise_seed: int
    fit_seed: int

    def __post_init__(self):
        zero = np.zeros((len(self.scene), 2), dtype=float)
        self.base_image = render(self.scene, zero, self.stamp)
        self.noise = np.random.RandomState(self.noise_seed).normal(
            0.0, self.sky, self.base_image.shape,
        )
        self.base_shape = self._measure(self.base_image)

    def _measure(self, image):
        return np.asarray(measure(image + self.noise, self.fit_seed), dtype=float)

    def shape(self, shears):
        return self._measure(render(self.scene, shears, self.stamp))


def unit(alpha):
    """A unit vector in spin-2 component space (alpha already spans 0..2pi)."""
    alpha = np.asarray(alpha, dtype=float)
    return np.stack([np.cos(alpha), np.sin(alpha)], axis=-1)


def directional_response(measurer, active, directions, g, central):
    """Measured two-component response for one multi-source shear draw."""
    shears = np.zeros((len(measurer.scene), 2), dtype=float)
    shears[1 + np.flatnonzero(active)] = g * directions[active]
    plus = measurer.shape(shears)
    if central:
        minus = measurer.shape(-shears)
        return (plus - minus) / (2.0 * g)
    return (plus - measurer.base_shape) / g


def isotropic_response(measurer, g, n_angles, per_neighbour=False):
    """Direction-average coherent or summed one-at-a-time central response."""
    n_nbr = len(measurer.scene) - 1
    directions = unit(2.0 * np.pi * np.arange(n_angles) / n_angles)
    totals = []
    for u in directions:
        if per_neighbour:
            value = 0.0
            for j in range(n_nbr):
                active = np.zeros(n_nbr, dtype=bool)
                active[j] = True
                vec = directional_response(
                    measurer, active, np.repeat(u[None, :], n_nbr, axis=0), g, True,
                )
                value += float(vec @ u)
        else:
            active = np.ones(n_nbr, dtype=bool)
            vec = directional_response(
                measurer, active, np.repeat(u[None, :], n_nbr, axis=0), g, True,
            )
            value = float(vec @ u)
        totals.append(value)
    values = np.asarray(totals, dtype=float)
    return {
        "mean": float(values.mean()),
        "direction_sem": float(values.std(ddof=1) / np.sqrt(len(values))),
        "n_angles": int(len(values)),
        "values": values.tolist(),
    }


def component_jacobians(measurer, g):
    """Per-neighbour 2x2 central response matrices along the component axes."""
    n_nbr = len(measurer.scene) - 1
    matrices = np.zeros((n_nbr, 2, 2), dtype=float)
    axes = np.eye(2)
    for j in range(n_nbr):
        active = np.zeros(n_nbr, dtype=bool)
        active[j] = True
        for component, u in enumerate(axes):
            directions = np.repeat(u[None, :], n_nbr, axis=0)
            matrices[j, :, component] = directional_response(
                measurer, active, directions, g, True,
            )
    return matrices


def sum_conditional_pair_means(labels, active):
    """Sum the per-pair means conditional on that neighbour being active."""
    labels = np.asarray(labels, dtype=float)
    active = np.asarray(active, dtype=bool)
    if labels.shape != active.shape or labels.ndim != 2:
        raise ValueError("labels and active must be matching 2D arrays")
    counts = active.sum(axis=0)
    if np.any(counts == 0):
        raise ValueError("every neighbour needs at least one active draw")
    means = np.where(active, labels, 0.0).sum(axis=0) / counts
    return float(means.sum()), means, counts


def bootstrap_pair_sum(labels, active, n_boot, seed):
    """Draw-block bootstrap for the summed conditional pair-label mean."""
    labels = np.asarray(labels, dtype=float)
    active = np.asarray(active, dtype=bool)
    rng = np.random.RandomState(seed)
    estimates = []
    for _ in range(n_boot):
        idx = rng.randint(0, len(labels), size=len(labels))
        try:
            estimate, _, _ = sum_conditional_pair_means(labels[idx], active[idx])
        except ValueError:
            continue
        estimates.append(estimate)
    if len(estimates) < 0.9 * n_boot:
        raise RuntimeError(f"too many invalid bootstrap samples: {len(estimates)}/{n_boot}")
    estimates = np.asarray(estimates, dtype=float)
    return float(estimates.std(ddof=1)), [float(x) for x in np.percentile(estimates, [2.5, 97.5])]


def random_protocol(
        measurer, g, n_draw, active_probability, central, seed, n_boot, jacobians):
    """Emulate pair labels from simultaneous independent random-direction shears."""
    n_nbr = len(measurer.scene) - 1
    rng = np.random.RandomState(seed)
    active_rows = []
    label_rows = []
    residual_rows = []
    failures = 0
    for _ in range(n_draw):
        active = rng.uniform(size=n_nbr) < active_probability
        directions = unit(rng.uniform(0.0, 2.0 * np.pi, size=n_nbr))
        if not active.any():
            active_rows.append(active)
            label_rows.append(np.zeros(n_nbr, dtype=float))
            residual_rows.append(np.zeros(n_nbr, dtype=float))
            continue
        try:
            response = directional_response(measurer, active, directions, g, central)
        except Exception:
            failures += 1
            continue
        labels = directions @ response
        linear_response = np.einsum(
            "nij,nj->i", jacobians[active], directions[active], optimize=True,
        )
        residual_labels = directions @ (response - linear_response)
        active_rows.append(active)
        label_rows.append(labels)
        residual_rows.append(residual_labels)
    active = np.asarray(active_rows, dtype=bool)
    labels = np.asarray(label_rows, dtype=float)
    residuals = np.asarray(residual_rows, dtype=float)
    raw_estimate, raw_pair_means, counts = sum_conditional_pair_means(labels, active)
    residual_estimate, residual_pair_means, residual_counts = sum_conditional_pair_means(
        residuals, active,
    )
    if not np.array_equal(counts, residual_counts):
        raise RuntimeError("control-variate active counts changed")
    linear_pair_means = 0.5 * np.trace(jacobians, axis1=1, axis2=2)
    pair_means = linear_pair_means + residual_pair_means
    estimate = float(pair_means.sum())
    sem, residual_ci95 = bootstrap_pair_sum(residuals, active, n_boot, seed + 9173)
    ci95 = [float(estimate + x - residual_estimate) for x in residual_ci95]
    raw_sem, raw_ci95 = bootstrap_pair_sum(labels, active, n_boot, seed + 19173)
    return {
        "mean": estimate,
        "bootstrap_sem": sem,
        "bootstrap_ci95": ci95,
        "pair_means": pair_means.tolist(),
        "linear_pair_means": linear_pair_means.tolist(),
        "nonlinear_pair_residual_means": residual_pair_means.tolist(),
        "raw_mean": raw_estimate,
        "raw_bootstrap_sem": raw_sem,
        "raw_bootstrap_ci95": raw_ci95,
        "raw_pair_means": raw_pair_means.tolist(),
        "active_counts": counts.astype(int).tolist(),
        "n_success": int(len(labels)),
        "n_failure": int(failures),
        "active_probability": float(active_probability),
        "central": bool(central),
    }


def contrast(a, b, sem=0.0):
    return {"difference": float(a - b), "reference": float(b), "sem": float(sem)}


def run_scene(name, target, neighbours, args):
    scene = [target, *neighbours]
    measurer = Measurer(scene, args.stamp, args.sky, args.noise_seed, args.fit_seed)
    output = {"n_neighbours": len(neighbours), "amplitudes": {}}
    print(f"\n{name}: N_neighbour={len(neighbours)}", flush=True)
    for g in args.g:
        coherent = isotropic_response(measurer, g, args.n_angles, per_neighbour=False)
        individual = isotropic_response(measurer, g, args.n_angles, per_neighbour=True)
        jacobians = component_jacobians(measurer, g)
        jacobian_trace_sum = float(0.5 * np.trace(jacobians, axis1=1, axis2=2).sum())
        half_fwd = random_protocol(
            measurer, g, args.n_draw, 0.5, False, args.seed + round(1000 * g), args.n_boot,
            jacobians,
        )
        half_ctr = random_protocol(
            measurer, g, args.n_draw, 0.5, True, args.seed + round(1000 * g), args.n_boot,
            jacobians,
        )
        all_fwd = random_protocol(
            measurer, g, args.n_draw, 1.0, False, args.seed + 10000 + round(1000 * g), args.n_boot,
            jacobians,
        )
        item = {
            "coherent_central_isotropic": coherent,
            "individual_central_isotropic_sum": individual,
            "component_jacobian_trace_sum": jacobian_trace_sum,
            "random_half_forward": half_fwd,
            "random_half_central": half_ctr,
            "random_all_forward": all_fwd,
            "contrasts": {
                "additivity_coherent_minus_individual": contrast(
                    coherent["mean"], individual["mean"]),
                "half_forward_minus_coherent": contrast(
                    half_fwd["mean"], coherent["mean"], half_fwd["bootstrap_sem"]),
                "half_central_minus_coherent": contrast(
                    half_ctr["mean"], coherent["mean"], half_ctr["bootstrap_sem"]),
                "half_forward_minus_half_central": contrast(
                    half_fwd["mean"], half_ctr["mean"],
                    max(half_fwd["bootstrap_sem"], half_ctr["bootstrap_sem"])),
                "all_forward_minus_half_forward": contrast(
                    all_fwd["mean"], half_fwd["mean"],
                    np.hypot(all_fwd["bootstrap_sem"], half_fwd["bootstrap_sem"])),
            },
        }
        output["amplitudes"][str(g)] = item
        print(
            f"  g={g:.3f}: coherent={coherent['mean']:+.6f}; "
            f"individual={individual['mean']:+.6f}; "
            f"half-forward={half_fwd['mean']:+.6f} +/- {half_fwd['bootstrap_sem']:.6f}; "
            f"half-central={half_ctr['mean']:+.6f} +/- {half_ctr['bootstrap_sem']:.6f}; "
            f"all-forward={all_fwd['mean']:+.6f} +/- {all_fwd['bootstrap_sem']:.6f} "
            f"[raw half SEM={half_fwd['raw_bootstrap_sem']:.6f}]",
            flush=True,
        )
    lo, hi = [str(x) for x in sorted(args.g)]
    low = output["amplitudes"][lo]
    high = output["amplitudes"][hi]
    output["cross_amplitude_contrasts"] = {
        "production_half_forward_hi_minus_anchor_coherent_lo": contrast(
            high["random_half_forward"]["mean"],
            low["coherent_central_isotropic"]["mean"],
            high["random_half_forward"]["bootstrap_sem"],
        ),
        "coherent_hi_minus_lo": contrast(
            high["coherent_central_isotropic"]["mean"],
            low["coherent_central_isotropic"]["mean"],
        ),
    }
    decisive = output["cross_amplitude_contrasts"][
        "production_half_forward_hi_minus_anchor_coherent_lo"
    ]
    print(
        f"  production-label(g={hi}) - anchor-truth(g={lo}) = "
        f"{decisive['difference']:+.6f} +/- {decisive['sem']:.6f}", flush=True,
    )
    return output


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--g", type=float, nargs=2, default=[0.05, 0.2])
    ap.add_argument("--n-angles", type=int, default=16)
    ap.add_argument("--n-draw", type=int, default=256)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--stamp", type=int, default=112)
    ap.add_argument("--sky", type=float, default=0.0)
    ap.add_argument("--noise-seed", type=int, default=773)
    ap.add_argument("--fit-seed", type=int, default=42)
    ap.add_argument("--seed", type=int, default=314159)
    ap.add_argument("--scenario", choices=["all", *scenarios().keys()], default="all")
    ap.add_argument("--output-json", required=True)
    args = ap.parse_args()
    if args.n_angles < 8 or args.n_angles % 4:
        raise ValueError("n-angles must be a multiple of four and at least eight")
    if args.n_draw < 32:
        raise ValueError("n-draw must be at least 32")
    if sorted(args.g) != list(args.g) or args.g[0] <= 0:
        raise ValueError("--g must contain two increasing positive amplitudes")
    selected = scenarios()
    if args.scenario != "all":
        selected = {args.scenario: selected[args.scenario]}

    print("RANDOM-HALF RESPONSE-LABEL TOMOGRAPHY")
    print(
        f"g={args.g}, angles={args.n_angles}, random draws={args.n_draw}, "
        f"sky={args.sky}, fixed positions=True",
    )
    result = {
        "design": {
            "g": args.g, "n_angles": args.n_angles, "n_draw": args.n_draw,
            "n_boot": args.n_boot, "stamp": args.stamp, "sky": args.sky,
            "noise_seed": args.noise_seed, "fit_seed": args.fit_seed, "seed": args.seed,
            "primary_sheared": False, "positions_sheared": False,
            "production_protocol": "forward g0->g, Bernoulli-half active, independent spin-2 directions",
        },
        "predeclared_interpretation": {
            "needed_sign": "random_half_forward(g=0.2) below coherent(g=0.05)",
            "carrier_scale": 0.005,
            "support_rule": "difference <= -0.005 and >3 bootstrap SEM in at least two crowded scenes",
            "single_neighbour_control_tolerance": 0.002,
            "scope": "mechanism capability only; not survey prevalence",
        },
        "scenarios": {},
    }
    for index, (name, (target, neighbours)) in enumerate(selected.items()):
        args.seed += 100000 * index
        result["scenarios"][name] = run_scene(name, target, neighbours, args)

    out = os.path.abspath(args.output_json)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if os.path.exists(out):
        raise FileExistsError(out)
    with open(out, "x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"\nwrote {out}")
    print("TOY_RANDOM_HALF_DONE", flush=True)


if __name__ == "__main__":
    main()
