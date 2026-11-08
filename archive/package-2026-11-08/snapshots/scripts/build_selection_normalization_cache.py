#!/usr/bin/env python
"""Build a validated local-quadratic ``log B_W(g)`` cache from exact stencils."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from sbsi.selection_normalization import load_population_normalization


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _result_path(path: str | Path) -> Path:
    path = Path(path)
    return path / "result.json" if path.is_dir() else path


def _identity(result: dict) -> dict:
    config = result["config"]
    selection = result["selection"]
    return {
        "scene_sha256": result["scene_sha256"],
        "model_sha256": result["model_sha256"],
        "blend_response_sha256": result["r_blend"]["cache_sha256"],
        "conditions": result["conditions"],
        "detection_radius_arcsec": float(config["detection_radius_arcsec"]),
        "flow_neighbour_radius_arcsec": float(config["flow_neighbour_radius_arcsec"]),
        "crowding_radii_arcsec": [
            float(config["crowding_near_arcsec"]),
            float(config["crowding_far_arcsec"]),
        ],
        "selection_samples": int(selection["n_samples"]),
        "selection_seed": int(selection["seed"]),
        "selection_row_chunk": int(selection["row_chunk"]),
        "cut_key": selection["cut_key"],
    }


def _derivatives(result: dict):
    center = np.asarray(result["initial_center"]["center"], dtype=np.float64)
    h = float(result["config"]["h"])
    reported = result["selection"].get("probabilities") or []
    if reported:
        points = {
            (round(float(row["g1"]), 10), round(float(row["g2"]), 10)):
            float(np.log(row["detected_and_selected_mass"]))
            for row in reported
        }
    else:
        cache_report = result["selection"].get("normalization_cache") or {}
        if cache_report.get("method") != "exact_distributed_detected_selected_mass":
            raise ValueError("result lacks an exact selection-normalization stencil")
        cache_path = Path(cache_report["path"])
        if _sha256(cache_path) != cache_report["sha256"]:
            raise ValueError("exact selection-normalization cache hash has changed")
        normalization = load_population_normalization(cache_path)
        points = {
            (round(float(g1), 10), round(float(g2), 10)): normalization.log_mass(g1, g2)
            for g1, g2 in normalization.available_shears
        }

    def value(dx, dy):
        key = (round(float(center[0] + dx), 10), round(float(center[1] + dy), 10))
        if key not in points:
            raise ValueError(f"result lacks exact selection-normalization stencil point {key}")
        return points[key]

    zero = value(0, 0)
    gradient = np.array(
        [(value(h, 0) - value(-h, 0)) / (2 * h),
         (value(0, h) - value(0, -h)) / (2 * h)]
    )
    hessian = np.array(
        [
            [
                (value(h, 0) - 2 * zero + value(-h, 0)) / h**2,
                (
                    value(h, h) - value(h, -h) - value(-h, h) + value(-h, -h)
                ) / (4 * h**2),
            ],
            [0.0, (value(0, h) - 2 * zero + value(0, -h)) / h**2],
        ]
    )
    hessian[1, 0] = hessian[0, 1]
    return center, h, zero, gradient, hessian


def _stencil_derivatives(points, center, h):
    """Finite-difference derivatives from one exact full-2D nine-point stencil."""

    center = np.asarray(center, dtype=np.float64)
    keyed = {
        (round(float(g1), 10), round(float(g2), 10)): float(log_mass)
        for (g1, g2), log_mass in points.items()
    }

    def value(dx, dy):
        key = (round(float(center[0] + dx), 10), round(float(center[1] + dy), 10))
        if key not in keyed:
            raise ValueError(f"exact cache lacks full stencil point {key}")
        return keyed[key]

    zero = value(0, 0)
    gradient = np.array(
        [(value(h, 0) - value(-h, 0)) / (2 * h),
         (value(0, h) - value(0, -h)) / (2 * h)]
    )
    hessian = np.array(
        [
            [
                (value(h, 0) - 2 * zero + value(-h, 0)) / h**2,
                (
                    value(h, h) - value(h, -h) - value(-h, h) + value(-h, -h)
                ) / (4 * h**2),
            ],
            [0.0, (value(0, h) - 2 * zero + value(0, -h)) / h**2],
        ]
    )
    hessian[1, 0] = hessian[0, 1]
    return zero, gradient, hessian


def _joint_fit_exact_caches(
    paths,
    *,
    reference_center=(0.0, 0.0),
    max_log_error,
    max_gradient_error,
    max_hessian_error,
):
    """Fit one quadratic to all points from two or more exact stencils.

    A Taylor expansion from only the first centre can be a poor extrapolator
    even when one quadratic represents the complete corridor well.  This fit
    estimates all six coefficients jointly from the exact masses, then checks
    value and derivative residuals at every contributing stencil.
    """

    if len(paths) < 2:
        raise ValueError("joint fitting requires at least two --joint-fit-exact-cache values")
    loaded = []
    identity = None
    h = None
    for name in paths:
        path = Path(name)
        normalization = load_population_normalization(path)
        if not hasattr(normalization, "available_shears"):
            raise ValueError(f"joint-fit input is not an exact cache: {path}")
        if identity is None:
            identity = dict(normalization.identity)
            h = float(normalization.finite_difference_step)
        elif dict(normalization.identity) != identity:
            raise ValueError(f"joint-fit cache has a different likelihood identity: {path}")
        elif not np.isclose(normalization.finite_difference_step, h, rtol=0, atol=1e-15):
            raise ValueError(f"joint-fit cache uses a different stencil step: {path}")
        shears = np.asarray(normalization.available_shears, dtype=np.float64)
        if shears.shape != (9, 2):
            raise ValueError(f"joint-fit cache must contain one nine-point stencil: {path}")
        center = shears.mean(axis=0)
        log_points = {
            tuple(point): normalization.log_mass(*point) for point in shears
        }
        # This also verifies that the nine values form the expected stencil.
        exact = _stencil_derivatives(log_points, center, h)
        loaded.append((path, normalization, shears, center, log_points, exact))

    all_shears = np.concatenate([row[2] for row in loaded], axis=0)
    all_log_mass = np.concatenate(
        [np.asarray(list(row[4].values()), dtype=np.float64) for row in loaded]
    )
    center = np.asarray(reference_center, dtype=np.float64)
    if center.shape != (2,) or not np.isfinite(center).all():
        raise ValueError("joint-fit reference center must be a finite shear pair")
    scale = np.maximum(np.max(np.abs(all_shears - center), axis=0), h)
    offset = (all_shears - center) / scale
    design = np.column_stack(
        (
            np.ones(len(offset)),
            offset[:, 0],
            offset[:, 1],
            0.5 * offset[:, 0] ** 2,
            offset[:, 0] * offset[:, 1],
            0.5 * offset[:, 1] ** 2,
        )
    )
    coefficients, _, rank, singular_values = np.linalg.lstsq(
        design, all_log_mass, rcond=None
    )
    if rank != 6:
        raise ValueError(f"joint-fit quadratic design is rank deficient: rank={rank}")
    fitted = design @ coefficients
    value_error = fitted - all_log_mass
    gradient = coefficients[1:3] / scale
    hessian = np.array(
        [
            [
                coefficients[3] / scale[0] ** 2,
                coefficients[4] / (scale[0] * scale[1]),
            ],
            [
                coefficients[4] / (scale[0] * scale[1]),
                coefficients[5] / scale[1] ** 2,
            ],
        ]
    )

    stencil_validation = []
    for path, _, _, stencil_center, _, exact in loaded:
        exact_zero, exact_gradient, exact_hessian = exact
        delta = stencil_center - center
        predicted_zero = (
            coefficients[0] + gradient @ delta + 0.5 * delta @ hessian @ delta
        )
        predicted_gradient = gradient + hessian @ delta
        gradient_error = predicted_gradient - exact_gradient
        hessian_error = hessian - exact_hessian
        stencil_validation.append(
            {
                "cache": str(path.resolve()),
                "sha256": _sha256(path),
                "center": stencil_center.tolist(),
                "center_log_mass_error": float(predicted_zero - exact_zero),
                "gradient_error": gradient_error.tolist(),
                "maximum_absolute_gradient_error": float(np.max(np.abs(gradient_error))),
                "maximum_absolute_hessian_error": float(np.max(np.abs(hessian_error))),
            }
        )

    maximum_log = float(np.max(np.abs(value_error)))
    maximum_gradient = max(
        row["maximum_absolute_gradient_error"] for row in stencil_validation
    )
    maximum_hessian = max(
        row["maximum_absolute_hessian_error"] for row in stencil_validation
    )
    if (
        maximum_log > max_log_error
        or maximum_gradient > max_gradient_error
        or maximum_hessian > max_hessian_error
    ):
        raise ValueError(
            "joint selection-normalization fit fails residual checks: "
            f"log mass {maximum_log:g} (limit {max_log_error:g}), "
            f"gradient {maximum_gradient:g} (limit {max_gradient_error:g}), "
            f"Hessian {maximum_hessian:g} (limit {max_hessian_error:g})"
        )

    return {
        "version": 1,
        "method": "local_quadratic_log_detected_selected_mass",
        "center": center.tolist(),
        "finite_difference_step": h,
        "log_mass_at_center": float(coefficients[0]),
        "gradient": gradient.tolist(),
        "hessian": hessian.tolist(),
        "trust_box": {
            "minimum": np.minimum(all_shears.min(axis=0), center - h).tolist(),
            "maximum": np.maximum(all_shears.max(axis=0), center + h).tolist(),
            "padding_steps": 0.0,
            "policy": "envelope of exact fit points and the canonical reference stencil",
        },
        "identity": identity,
        "validation": {
            "method": "joint least-squares residuals at all exact points and stencil derivatives",
            "n_exact_points": int(len(all_shears)),
            "maximum_absolute_log_mass_error": maximum_log,
            "root_mean_square_log_mass_error": float(np.sqrt(np.mean(value_error**2))),
            "maximum_absolute_gradient_error": maximum_gradient,
            "maximum_absolute_hessian_error": maximum_hessian,
            "log_mass_error_limit": float(max_log_error),
            "gradient_error_limit": float(max_gradient_error),
            "hessian_error_limit": float(max_hessian_error),
            "design_condition_number": float(singular_values[0] / singular_values[-1]),
            "stencils": stencil_validation,
        },
        "source": {
            "method": "joint_fit_of_exact_distributed_stencils",
            "caches": [
                {"path": str(path.resolve()), "sha256": _sha256(path)}
                for path, *_ in loaded
            ],
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--source-result")
    source_group.add_argument(
        "--joint-fit-exact-cache",
        action="append",
        help="exact nine-point cache; repeat for two or more centres",
    )
    parser.add_argument("--validation-result", action="append", default=[])
    parser.add_argument(
        "--reference-center",
        default="0,0",
        help="canonical g1,g2 origin for a joint-fit cache (default: 0,0)",
    )
    parser.add_argument("--max-log-mass-error", type=float, default=2.0e-6)
    parser.add_argument("--max-gradient-error", type=float, default=0.02)
    parser.add_argument("--max-hessian-error", type=float, default=5.0)
    parser.add_argument(
        "--trust-padding-steps",
        type=float,
        default=1.25,
        help="pad the envelope of validated centers by this many stencil steps",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    if args.joint_fit_exact_cache is not None:
        if args.validation_result:
            parser.error("--validation-result cannot be combined with a joint exact-cache fit")
        payload = _joint_fit_exact_caches(
            args.joint_fit_exact_cache,
            reference_center=tuple(float(value) for value in args.reference_center.split(",")),
            max_log_error=float(args.max_log_mass_error),
            max_gradient_error=float(args.max_gradient_error),
            max_hessian_error=float(args.max_hessian_error),
        )
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2) + "\n")
        print(json.dumps(payload, indent=2))
        return 0

    source_path = _result_path(args.source_result)
    source = json.loads(source_path.read_text())
    if source.get("selection", {}).get("normalization_surrogate") is not None:
        raise ValueError("source result must use an exact selection normalization")
    identity = _identity(source)
    center, h, zero, gradient, hessian = _derivatives(source)

    validation = []
    centers = [center]
    for name in args.validation_result:
        path = _result_path(name)
        result = json.loads(path.read_text())
        if _identity(result) != identity:
            raise ValueError(f"validation result has a different likelihood identity: {path}")
        other_center, other_h, _, other_gradient, other_hessian = _derivatives(result)
        if not np.isclose(other_h, h, rtol=0, atol=1e-15):
            raise ValueError(f"validation result uses a different stencil step: {path}")
        predicted_gradient = gradient + hessian @ (other_center - center)
        gradient_error = predicted_gradient - other_gradient
        hessian_error = hessian - other_hessian
        validation.append(
            {
                "result": str(path.resolve()),
                "sha256": _sha256(path),
                "center": other_center.tolist(),
                "gradient_error": gradient_error.tolist(),
                "maximum_absolute_gradient_error": float(np.max(np.abs(gradient_error))),
                "maximum_absolute_hessian_error": float(np.max(np.abs(hessian_error))),
            }
        )
        centers.append(other_center)
    if not validation:
        raise ValueError("at least one cross-center --validation-result is required")
    if not np.isfinite(args.trust_padding_steps) or args.trust_padding_steps < 1.0:
        raise ValueError("--trust-padding-steps must be finite and at least one")
    max_gradient = max(row["maximum_absolute_gradient_error"] for row in validation)
    max_hessian = max(row["maximum_absolute_hessian_error"] for row in validation)
    if max_gradient > args.max_gradient_error or max_hessian > args.max_hessian_error:
        raise ValueError(
            "selection-normalization surrogate fails validation: "
            f"gradient {max_gradient:g} (limit {args.max_gradient_error:g}), "
            f"Hessian {max_hessian:g} (limit {args.max_hessian_error:g})"
        )

    centers = np.asarray(centers)
    payload = {
        "version": 1,
        "method": "local_quadratic_log_detected_selected_mass",
        "center": center.tolist(),
        "finite_difference_step": h,
        "log_mass_at_center": zero,
        "gradient": gradient.tolist(),
        "hessian": hessian.tolist(),
        "trust_box": {
            "minimum": (centers.min(axis=0) - args.trust_padding_steps * h).tolist(),
            "maximum": (centers.max(axis=0) + args.trust_padding_steps * h).tolist(),
            "padding_steps": float(args.trust_padding_steps),
            "policy": "axis-aligned envelope of exact validation centers padded in stencil-step units",
        },
        "identity": identity,
        "validation": {
            "maximum_absolute_gradient_error": max_gradient,
            "maximum_absolute_hessian_error": max_hessian,
            "gradient_error_limit": float(args.max_gradient_error),
            "hessian_error_limit": float(args.max_hessian_error),
            "points": validation,
        },
        "source": {
            "result": str(source_path.resolve()),
            "sha256": _sha256(source_path),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
