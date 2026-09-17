"""Guard-band shear-response supervision for full-domain measurement flows.

The flow remains a density for all valid measured outputs.  This module adds
no measurement to its condition.  Instead it evaluates differentiable test
functions of generated ``FLUX_RADIUS`` and flux, then constrains the induced
mean-shape response globally and around prospective catalogue cuts.

For a soft guard ``w_c(y)``, baseline pass mass ``D_c`` and baseline selected
mean ``mu_c``, the influence function

``v_c(y) = w_c(y) * (e - mu_c) / D_c``

has the same first-order shear response as the guarded mean shape.  Its
response is additive over objects, so random shear directions can be handled
by an ordinary two-component forward regression without introducing a paired
output likelihood.
"""

from __future__ import annotations

import math
from typing import Iterable, Mapping

import numpy as np
import torch


SHAPE_RESPONSE_COMPONENTS = ("R11", "R12", "R21", "R22")


def make_guard_cuts(
    radius_thresholds: Iterable[float],
    magnitude_thresholds: Iterable[float],
) -> tuple[dict, ...]:
    """Build global, marginal and joint lower-radius/upper-magnitude guards."""

    radius = tuple(float(value) for value in radius_thresholds)
    magnitude = tuple(float(value) for value in magnitude_thresholds)
    if (
        not radius
        or not magnitude
        or not np.isfinite(radius).all()
        or not np.isfinite(magnitude).all()
        or len(set(radius)) != len(radius)
        or len(set(magnitude)) != len(magnitude)
    ):
        raise ValueError("distinct finite radius and magnitude thresholds required")
    cuts = [{"name": "global", "radius_min": None, "magnitude_max": None}]
    cuts.extend(
        {
            "name": f"radius_gt_{value:g}",
            "radius_min": value,
            "magnitude_max": None,
        }
        for value in radius
    )
    cuts.extend(
        {
            "name": f"magnitude_lt_{value:g}",
            "radius_min": None,
            "magnitude_max": value,
        }
        for value in magnitude
    )
    cuts.extend(
        {
            "name": f"radius_gt_{r:g}_magnitude_lt_{m:g}",
            "radius_min": r,
            "magnitude_max": m,
        }
        for r in radius
        for m in magnitude
    )
    return tuple(cuts)


def _validate_cuts(cuts: Iterable[Mapping]) -> tuple[dict, ...]:
    result = tuple(dict(cut) for cut in cuts)
    if not result:
        raise ValueError("at least one guard cut is required")
    names = []
    for cut in result:
        if set(cut) != {"name", "radius_min", "magnitude_max"}:
            raise ValueError("guard cuts require name, radius_min and magnitude_max")
        names.append(str(cut["name"]))
        for key in ("radius_min", "magnitude_max"):
            value = cut[key]
            if value is not None and not math.isfinite(float(value)):
                raise ValueError(f"nonfinite guard threshold: {key}")
    if len(set(names)) != len(names):
        raise ValueError("guard names must be unique")
    return result


def _numpy_sigmoid(value):
    return np.exp(-np.logaddexp(0.0, -np.asarray(value, dtype=np.float64)))


def numpy_guard_weights(
    values,
    cuts,
    *,
    radius_softness: float,
    magnitude_softness: float,
    zero_point: float = 30.0,
    hard: bool = False,
):
    """Evaluate catalogue guards on physical flow outputs.

    With ``hard=False`` each guard is the smooth sigmoid band used for
    training.  With ``hard=True`` the same guard becomes the exact catalogue
    indicator ``radius > radius_min`` and ``magnitude < magnitude_max``.  The
    softness arguments are still validated so that a caller cannot silently
    switch conventions, but they do not enter the hard weights.
    """

    values = np.asarray(values, dtype=np.float64)
    cuts = _validate_cuts(cuts)
    if (
        values.ndim != 2
        or values.shape[1] != 4
        or not np.isfinite(values).all()
        or np.any(values[:, 2:] <= 0.0)
        or min(radius_softness, magnitude_softness) <= 0.0
        or not np.isfinite([radius_softness, magnitude_softness, zero_point]).all()
    ):
        raise ValueError("finite physical outputs and positive guard softness required")
    radius = values[:, 2]
    magnitude = float(zero_point) - 2.5 * np.log10(values[:, 3])
    result = np.ones((len(values), len(cuts)), dtype=np.float64)
    for index, cut in enumerate(cuts):
        if cut["radius_min"] is not None:
            if hard:
                result[:, index] *= (radius > float(cut["radius_min"])).astype(
                    np.float64
                )
            else:
                result[:, index] *= _numpy_sigmoid(
                    (radius - float(cut["radius_min"])) / radius_softness
                )
        if cut["magnitude_max"] is not None:
            if hard:
                result[:, index] *= (magnitude < float(cut["magnitude_max"])).astype(
                    np.float64
                )
            else:
                result[:, index] *= _numpy_sigmoid(
                    (float(cut["magnitude_max"]) - magnitude) / magnitude_softness
                )
    return result


def torch_guard_weights(
    values,
    cuts,
    *,
    radius_softness: float,
    magnitude_softness: float,
    zero_point: float = 30.0,
    hard: bool = False,
):
    """Differentiable counterpart of :func:`numpy_guard_weights`.

    ``hard=True`` returns the exact catalogue indicator.  It has zero gradient
    almost everywhere and is intended for evaluation only, never for guard
    training.
    """

    cuts = _validate_cuts(cuts)
    if (
        values.ndim < 2
        or values.shape[-1] != 4
        or min(radius_softness, magnitude_softness) <= 0.0
    ):
        raise ValueError("physical outputs and positive guard softness required")
    radius = values[..., 2]
    flux = values[..., 3]
    # ``forward_physical`` uses a softplus positive map.  Its mathematical
    # limit is positive, but sufficiently negative float64 coordinates can
    # round to exactly zero.  log10(0) has the desired forward limit for a
    # faint-end guard, yet its autograd derivative is NaN (zero sigmoid
    # derivative times an infinite log derivative).  Below the smallest
    # representable positive value the guard is already identically zero, so
    # clamp only that numerical underflow tail and give it the correct zero
    # gradient.
    safe_flux = flux.clamp_min(torch.finfo(flux.dtype).tiny)
    magnitude = float(zero_point) - 2.5 * torch.log10(safe_flux)
    weights = []
    for cut in cuts:
        value = torch.ones_like(radius)
        if cut["radius_min"] is not None:
            if hard:
                value = value * (radius > float(cut["radius_min"])).to(radius.dtype)
            else:
                value = value * torch.sigmoid(
                    (radius - float(cut["radius_min"])) / radius_softness
                )
        if cut["magnitude_max"] is not None:
            if hard:
                value = value * (magnitude < float(cut["magnitude_max"])).to(
                    radius.dtype
                )
            else:
                value = value * torch.sigmoid(
                    (float(cut["magnitude_max"]) - magnitude) / magnitude_softness
                )
        weights.append(value)
    return torch.stack(weights, dim=-1)


def fit_guard_response(
    measured_zero,
    measured_shear,
    gamma,
    cuts,
    *,
    radius_softness: float,
    magnitude_softness: float,
    zero_point: float = 30.0,
    chunk_size: int = 1_000_000,
    hard: bool = False,
):
    """Fit guarded two-by-two forward responses from measured paired rows.

    ``hard=True`` selects each leg with the exact catalogue indicator instead
    of the smooth training band.  Both legs are always weighted by their own
    measurement, so the fitted response includes the selection response and is
    a functional of the per-leg marginals alone.
    """

    zero = np.asarray(measured_zero, dtype=np.float64)
    shear = np.asarray(measured_shear, dtype=np.float64)
    gamma = np.asarray(gamma, dtype=np.float64)
    cuts = _validate_cuts(cuts)
    if (
        zero.ndim != 2
        or zero.shape[1] != 4
        or shear.shape != zero.shape
        or gamma.shape != (len(zero), 2)
        or not len(zero)
        or not np.isfinite(zero).all()
        or not np.isfinite(shear).all()
        or not np.isfinite(gamma).all()
        or chunk_size < 1
    ):
        raise ValueError("finite aligned measured pairs and two-component shear required")
    normal = gamma.T @ gamma
    if not np.isfinite(normal).all() or np.linalg.cond(normal) > 1.0e8:
        raise ValueError("guard response requires a stable two-component shear design")
    mass_sum = np.zeros(len(cuts), dtype=np.float64)
    shape_sum = np.zeros((len(cuts), 2), dtype=np.float64)
    for start in range(0, len(zero), chunk_size):
        block = zero[start : start + chunk_size]
        weight = numpy_guard_weights(
            block,
            cuts,
            radius_softness=radius_softness,
            magnitude_softness=magnitude_softness,
            zero_point=zero_point,
            hard=hard,
        )
        mass_sum += weight.sum(axis=0)
        shape_sum += np.einsum("nc,no->co", weight, block[:, :2])
    mass = mass_sum / len(zero)
    if np.any(mass <= 0.0) or not np.isfinite(mass).all():
        raise ValueError("every guard requires positive finite baseline mass")
    mean = shape_sum / mass_sum[:, None]
    cross = np.zeros((len(cuts), 2, 2), dtype=np.float64)
    for start in range(0, len(zero), chunk_size):
        stop = start + chunk_size
        block_zero = zero[start:stop]
        block_shear = shear[start:stop]
        weight_zero = numpy_guard_weights(
            block_zero,
            cuts,
            radius_softness=radius_softness,
            magnitude_softness=magnitude_softness,
            zero_point=zero_point,
            hard=hard,
        )
        weight_shear = numpy_guard_weights(
            block_shear,
            cuts,
            radius_softness=radius_softness,
            magnitude_softness=magnitude_softness,
            zero_point=zero_point,
            hard=hard,
        )
        influence_zero = (
            weight_zero[:, :, None]
            * (block_zero[:, None, :2] - mean[None, :, :])
            / mass[None, :, None]
        )
        influence_shear = (
            weight_shear[:, :, None]
            * (block_shear[:, None, :2] - mean[None, :, :])
            / mass[None, :, None]
        )
        cross += np.einsum(
            "nco,ni->coi", influence_shear - influence_zero, gamma[start:stop]
        )
    response = cross @ np.linalg.inv(normal)
    return {
        "target": response.reshape(len(cuts), len(SHAPE_RESPONSE_COMPONENTS)),
        "baseline_mass": mass,
        "baseline_mean_shape": mean,
        "shape_normal": normal,
        "pairs": int(len(zero)),
        "cuts": cuts,
        "radius_softness": float(radius_softness),
        "magnitude_softness": float(magnitude_softness),
        "zero_point": float(zero_point),
        "hard": bool(hard),
    }


def physical_context_draws(model, contexts, draws, seed):
    """Differentiable physical draws with antithetic CRN across contexts."""

    if not callable(getattr(model, "forward_physical", None)):
        raise ValueError("guard response requires an explicit physical flow")
    if not contexts or draws < 2 or draws % 2:
        raise ValueError("at least one context and an even draw count >=2 are required")
    batch = len(contexts[0])
    if any(
        context.ndim != 2
        or len(context) != batch
        or context.shape[1] != model.context_dim
        for context in contexts
    ):
        raise ValueError("all contexts must align with the flow context width")
    generator = torch.Generator(device=contexts[0].device).manual_seed(int(seed))
    half = torch.randn(
        batch,
        draws // 2,
        model.target_dim,
        dtype=contexts[0].dtype,
        device=contexts[0].device,
        generator=generator,
    )
    base = torch.cat((half, -half), dim=1).reshape(batch * draws, model.target_dim)
    result = []
    for context in contexts:
        expanded = context[:, None, :].expand(batch, draws, model.context_dim)
        sampled, _ = model.forward_physical(
            base, expanded.reshape(batch * draws, model.context_dim)
        )
        result.append(sampled.reshape(batch, draws, model.target_dim))
    return result


class GuardResponsePopulation:
    """Matched full-domain rows and fixed measured guard-response targets."""

    def __init__(
        self,
        context,
        pairs,
        measured_zero,
        measured_shear,
        gamma,
        cuts,
        *,
        radius_softness: float,
        magnitude_softness: float,
        response_scale,
        zero_point: float = 30.0,
        hard: bool = False,
    ):
        cuts = _validate_cuts(cuts)
        pairs = np.asarray(pairs)
        gamma = np.asarray(gamma, dtype=np.float64)
        scale = np.asarray(response_scale, dtype=np.float64)
        if scale.ndim == 0:
            scale = np.full(len(cuts), float(scale), dtype=np.float64)
        if (
            context.ndim != 2
            or pairs.ndim != 2
            or pairs.shape[1] != 2
            or not np.issubdtype(pairs.dtype, np.integer)
            or np.any(pairs < 0)
            or np.any(pairs >= len(context))
            or gamma.shape != pairs.shape
            or scale.shape != (len(cuts),)
            or not np.isfinite(scale).all()
            or np.any(scale <= 0.0)
        ):
            raise ValueError("valid contexts, pairs, shear and response scale required")
        statistics = fit_guard_response(
            measured_zero,
            measured_shear,
            gamma,
            cuts,
            radius_softness=radius_softness,
            magnitude_softness=magnitude_softness,
            zero_point=zero_point,
            hard=hard,
        )
        device = context.device
        self.context = context
        self.pairs = torch.as_tensor(pairs, dtype=torch.long, device=device)
        self.cuts = statistics["cuts"]
        self.radius_softness = float(radius_softness)
        self.magnitude_softness = float(magnitude_softness)
        self.zero_point = float(zero_point)
        self.hard = bool(hard)
        self.target = torch.as_tensor(
            statistics["target"], dtype=torch.float64, device=device
        )
        self.baseline_mass = torch.as_tensor(
            statistics["baseline_mass"], dtype=torch.float64, device=device
        )
        self.baseline_mean = torch.as_tensor(
            statistics["baseline_mean_shape"], dtype=torch.float64, device=device
        )
        normal = np.asarray(statistics["shape_normal"], dtype=np.float64)
        self.design = torch.as_tensor(
            gamma @ np.linalg.inv(normal) * len(gamma),
            dtype=torch.float64,
            device=device,
        )
        self.response_scale = torch.as_tensor(
            scale[:, None], dtype=torch.float64, device=device
        )
        self.statistics = statistics

    def contributions(self, model, indices, draws, seed):
        pair = self.pairs[indices]
        generated = physical_context_draws(
            model,
            [self.context[pair[:, 0]], self.context[pair[:, 1]]],
            draws,
            seed,
        )
        influences = []
        for values in generated:
            weights = torch_guard_weights(
                values,
                self.cuts,
                radius_softness=self.radius_softness,
                magnitude_softness=self.magnitude_softness,
                zero_point=self.zero_point,
                hard=self.hard,
            )
            influence = (
                weights[..., None]
                * (values[..., None, :2] - self.baseline_mean[None, None, :, :])
                / self.baseline_mass[None, None, :, None]
            ).mean(dim=1)
            influences.append(influence)
        delta = influences[1] - influences[0]
        response = torch.einsum("bco,bi->bcoi", delta, self.design[indices])
        response = response.reshape(
            len(indices), len(self.cuts), len(SHAPE_RESPONSE_COMPONENTS)
        )
        if not bool(torch.isfinite(response).all()):
            raise ValueError("nonfinite generated guard-response contribution")
        return response


def independent_guard_score(mean_a, mean_c, target, response_scale):
    """Independent-replica cross quadratic averaged over guards/components."""

    if mean_a.shape != mean_c.shape or mean_a.shape != target.shape:
        raise ValueError("aligned guard-response means and target required")
    left = (mean_a - target) / response_scale
    right = (mean_c - target) / response_scale
    return 0.5 * (left * right).mean()


def sampled_guard_response_backward(
    model,
    population,
    pairs_per_step,
    draws,
    *,
    chunk_size,
    seed,
    weight=1.0,
):
    """Backpropagate one sampled guard-response cross score by replay."""

    if (
        pairs_per_step < 1
        or chunk_size < 1
        or draws < 2
        or draws % 2
        or seed < 0
        or not math.isfinite(weight)
        or weight < 0.0
    ):
        raise ValueError("positive sizes, even draws and finite nonnegative weight required")
    rng = np.random.default_rng(seed)
    indices = torch.as_tensor(
        rng.integers(len(population.pairs), size=int(pairs_per_step)),
        dtype=torch.long,
        device=population.context.device,
    )

    def chunks(stream):
        for block, start in enumerate(range(0, pairs_per_step, chunk_size)):
            yield (
                indices[start : start + chunk_size],
                seed + 100_000_000 * stream + block,
            )

    means = []
    with torch.no_grad():
        for stream in range(2):
            total = torch.zeros_like(population.target)
            for rows, latent_seed in chunks(stream):
                total += population.contributions(
                    model, rows, draws, latent_seed
                ).sum(dim=0)
            means.append(total / pairs_per_step)
        score = independent_guard_score(
            means[0], means[1], population.target, population.response_scale
        )
        coefficient = [
            0.5
            * (means[1] - population.target)
            / population.response_scale.square()
            / population.target.numel(),
            0.5
            * (means[0] - population.target)
            / population.response_scale.square()
            / population.target.numel(),
        ]
    if not bool(torch.isfinite(score)):
        raise ValueError("nonfinite guard-response score")
    if weight:
        with torch.enable_grad():
            for stream in range(2):
                for rows, latent_seed in chunks(stream):
                    values = population.contributions(model, rows, draws, latent_seed)
                    surrogate = (
                        values * coefficient[stream][None, :, :]
                    ).sum() * weight / pairs_per_step
                    if not bool(torch.isfinite(surrogate)):
                        raise ValueError("nonfinite guard-response gradient replay")
                    surrogate.backward()
    return {
        "loss": float(score),
        "pairs_per_group": int(pairs_per_step),
        "group_means": torch.stack(means).cpu().numpy(),
    }


@torch.no_grad()
def evaluate_guard_response(
    model,
    population,
    indices,
    *,
    draws=16,
    groups=4,
    batch_size=2048,
    seed=811_000_000_001,
):
    """Evaluate fixed pairs with independent latent groups."""

    if groups < 2 or draws < 2 or draws % 2 or batch_size < 1 or len(indices) < 1:
        raise ValueError("valid groups, draws, batch size and indices required")
    indices = torch.as_tensor(
        indices, dtype=torch.long, device=population.context.device
    )
    means = []
    for group in range(groups):
        total = torch.zeros_like(population.target)
        for block, start in enumerate(range(0, len(indices), batch_size)):
            rows = indices[start : start + batch_size]
            total += population.contributions(
                model,
                rows,
                draws,
                seed + group * 1_000_000_000 + block,
            ).sum(dim=0)
        means.append(total / len(indices))
    stacked = torch.stack(means)
    pair_scores = []
    for left in range(groups):
        for right in range(left + 1, groups):
            pair_scores.append(
                independent_guard_score(
                    stacked[left],
                    stacked[right],
                    population.target,
                    population.response_scale,
                )
            )
    return {
        "loss": float(torch.stack(pair_scores).mean()),
        "group_means": stacked.cpu().numpy(),
        "target": population.target.cpu().numpy(),
        "mean_response": stacked.mean(dim=0).cpu().numpy(),
        "mean_residual": (stacked.mean(dim=0) - population.target).cpu().numpy(),
        "draws": int(draws),
        "groups": int(groups),
        "pairs": int(len(indices)),
    }


__all__ = [
    "GuardResponsePopulation",
    "SHAPE_RESPONSE_COMPONENTS",
    "evaluate_guard_response",
    "fit_guard_response",
    "independent_guard_score",
    "make_guard_cuts",
    "numpy_guard_weights",
    "physical_context_draws",
    "sampled_guard_response_backward",
    "torch_guard_weights",
]
