"""Frozen physical proposal metric from independent zero-shear pilot gradients.

The metric is an uncentred second moment of likelihood gradients, plus a fixed
broad-proposal precision. It is a proposal heuristic, not an estimated posterior
information matrix. Non-equilibrated pilot states can overstate sensitivity and
make the result too stiff. The native prior is deliberately absent from this
heuristic; it remains present in every production acceptance ratio.

At g=0 the response contributes only through b*g=0, so its physical derivatives
vanish. Holding b fixed is therefore valid here. Nonzero shear is rejected:
there, ignoring the physical derivative of b would not be the exact chain rule.
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np
from scipy.special import expit

from .particle_geometry import BROAD_WIDTHS, PHYSICAL_COLUMNS, PILOT_SCALE, FrozenParticleGeometry


def physical_likelihood_gradients(
    coordinates: np.ndarray,
    summaries: np.ndarray,
    context_gradients: np.ndarray,
    *,
    sersic_n_min: float,
    sersic_n_max: float,
    ell_beta_emax: float,
    shear=(0.0, 0.0),
) -> np.ndarray:
    """Map raw eight-context log-K gradients to physical columns [1,6,7,8,9].

    Inputs are aligned row matrices with respectively10,9,8 columns. The other
    target coordinates and the entire neighbour scene stay fixed. The first
    five saved context columns must replay from the supplied coordinates, and
    crowding must be nonnegative. Gradients include F8 and C8 as supplied by the
    diagnostic evaluator; neither native-prior nor Jacobian terms are added.
    """
    x = np.asarray(coordinates, dtype=np.float64)
    values = np.asarray(summaries, dtype=np.float64)
    gradient = np.asarray(context_gradients, dtype=np.float64)
    centre = np.asarray(shear, dtype=np.float64)
    if centre.shape != (2,) or not np.array_equal(centre, np.zeros(2)):
        raise ValueError("physical likelihood-gradient geometry requires exactly zero shear")
    if (x.ndim != 2 or x.shape[1] != 10 or not len(x)
            or values.shape != (len(x), 9) or gradient.shape != (len(x), 8)
            or not all(np.isfinite(v).all() for v in (x, values, gradient))):
        raise ValueError("finite aligned coordinate, summary, context-gradient matrices required")
    if (not np.isfinite([sersic_n_min, sersic_n_max, ell_beta_emax]).all()
            or not 0 < sersic_n_min < sersic_n_max or not 0 < ell_beta_emax < 1):
        raise ValueError("valid positive Sersic bounds and distortion maximum below one required")
    if np.any(values[:, 5:8] < 0):
        raise ValueError("physical crowding summaries must be nonnegative")
    n_fraction, a_fraction = expit(x[:, 7]), expit(x[:, 8])
    a = ell_beta_emax * a_fraction
    rendered = a > 1e-7
    square_root = np.sqrt(1 - a*a)
    rho = np.where(rendered, a / (1 + square_root), 0.0)
    cosine, sine = np.cos(x[:, 9]), np.sin(x[:, 9])
    with np.errstate(over="ignore", under="ignore"):
        radius = np.exp(x[:, 6])
    mapped = np.column_stack((rho*cosine, rho*sine,
                              sersic_n_min + (sersic_n_max-sersic_n_min)*n_fraction,
                              x[:, 1], radius))
    if (not np.isfinite(mapped).all() or np.any(radius <= 0)
            or not np.allclose(mapped, values[:, :5], atol=5e-10, rtol=5e-11)):
        raise ValueError("saved physical summaries do not replay from pilot coordinates")
    result = np.empty((len(x), 5), dtype=np.float64)
    crowding_derivative = 0.4 * np.log(10) * -np.expm1(-values[:, 5:8])
    result[:, 0] = gradient[:, 3] + np.sum(gradient[:, 5:8] * crowding_derivative, axis=1)
    result[:, 1] = gradient[:, 4] * radius
    result[:, 2] = gradient[:, 2] * (sersic_n_max-sersic_n_min) * n_fraction * expit(-x[:, 7])
    drho_da = 1 / (square_root * (1 + square_root))
    drho_dx = np.where(rendered, drho_da * ell_beta_emax * a_fraction * expit(-x[:, 8]), 0.0)
    result[:, 3] = (gradient[:, 0]*cosine + gradient[:, 1]*sine) * drho_dx
    result[:, 4] = -gradient[:, 0]*mapped[:, 1] + gradient[:, 1]*mapped[:, 0]
    if not np.isfinite(result).all():
        raise FloatingPointError("physical likelihood gradient overflowed")
    return result


def fit_gradient_geometry(
    gradient_runs: list[np.ndarray], metadata: dict, *, phase_centres: np.ndarray | None = None
) -> FrozenParticleGeometry:
    """Freeze equal-run mean GGᵀ plus broad precision, with no mean subtraction.

    Each gradient array has shape (objects, particles, 5). Work in coordinates
    whitened by the broad proposal widths to compute the same regularized
    metric stably. Every run has equal weight regardless of its particle count.
    The existing geometry object supplies fixed-beta precision interpolation,
    symmetric increments, artifact persistence, and production seed checks.
    """
    if not gradient_runs:
        raise ValueError("at least one independent pilot-gradient run is required")
    arrays, moments, means, quantiles = [], [], [], []
    objects = None
    for raw in gradient_runs:
        gradient = np.asarray(raw, dtype=np.float64)
        if (gradient.ndim != 3 or gradient.shape[-1] != 5 or not gradient.shape[1]
                or not gradient.shape[0] or not np.isfinite(gradient).all()
                or (objects is not None and gradient.shape[0] != objects)):
            raise ValueError("finite gradient arrays require shape (same objects, particles, 5)")
        objects = gradient.shape[0]
        arrays.append(gradient)
        scaled = gradient * BROAD_WIDTHS
        with np.errstate(over="ignore", invalid="ignore"):
            moments.append(np.einsum("oni,onj->oij", scaled, scaled) / gradient.shape[1])
        means.append(gradient.mean(axis=1))
        quantiles.append(np.moveaxis(np.quantile(np.abs(gradient), [0.5, 0.9, 0.99, 1], axis=1), 0, 1))
    metric_scaled = np.mean(moments, axis=0) + np.eye(5)
    if not np.isfinite(metric_scaled).all():
        raise FloatingPointError("pilot gradient metric overflowed")
    try:
        cholesky = np.linalg.cholesky(metric_scaled)
        inverse_lower = np.linalg.solve(cholesky, np.broadcast_to(np.eye(5), cholesky.shape))
    except np.linalg.LinAlgError as exc:
        raise FloatingPointError("regularized pilot metric lost positive definiteness") from exc
    inverse_scaled = np.einsum("oki,okj->oij", inverse_lower, inverse_lower)
    covariance = PILOT_SCALE * inverse_scaled * BROAD_WIDTHS[None, :, None] * BROAD_WIDTHS[None, None, :]
    details = deepcopy(metadata)
    details.update({
        "fit": "equal-run uncentred likelihood-gradient second moment plus fixed broad precision",
        "physical_columns": list(PHYSICAL_COLUMNS),
        "pilot_particle_counts": [x.shape[1] for x in arrays],
        "likelihood_gradient_shear": [0.0, 0.0],
        "native_prior_in_metric": False,
        "metric_regularizer": "diag(1/broad_widths**2)",
        "metric_scaled_by_broad_widths": metric_scaled.tolist(),
        "metric_scaled_eigenvalues": np.linalg.eigvalsh(metric_scaled).tolist(),
        "metric_scaled_condition_numbers": np.linalg.cond(metric_scaled).tolist(),
        "gradient_mean_equal_run": np.mean(means, axis=0).tolist(),
        "gradient_mean_by_run": np.asarray(means).tolist(),
        "gradient_abs_quantile_levels": [0.5, 0.9, 0.99, 1.0],
        "gradient_abs_quantiles_by_run": np.asarray(quantiles).tolist(),
        "pilot_covariance_factor": PILOT_SCALE,
        "temperature_rule": "inverse(beta*inverse(pilot_scaled)+(1-beta)*inverse(broad))",
        "limitation": "Non-equilibrated pilot gradients may overstate local sensitivity; fixed proposal heuristic, not posterior information or convergence evidence.",
    })
    centres = np.zeros(objects) if phase_centres is None else phase_centres
    return FrozenParticleGeometry(covariance, centres, details)
