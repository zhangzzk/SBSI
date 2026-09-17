"""Grid-free uniformly pair-weighted supervision of physical mean changes.

For measured ``d = y_g - y_0`` and ``h = |gamma|``, two independent latent
replicas on the same object pairs estimate

``E_pair[sum((Delta_mu-d)^2 / output_scale^2) / (4 h^2)]``.

Each replica uses common antithetic base draws between the two shear legs.
The optional third component is physical ``FLUX_RADIUS`` in a declared fixed
pixel unit.  Flux remains density-only.
"""

from __future__ import annotations

import math

import numpy as np
import torch


def positive_shear_pair_mask(gamma):
    """Return response-eligible rows without inventing a shear-amplitude floor."""

    gamma = np.asarray(gamma)
    if gamma.ndim != 2 or gamma.shape[1] != 2 or not np.isfinite(gamma).all():
        raise ValueError("finite two-component shear vectors required")
    return np.linalg.norm(gamma, axis=1) > 0.0


def physical_context_means(model, contexts, draws, seed):
    """Differentiable physical means with antithetic CRN across contexts."""

    if not callable(getattr(model, "forward_physical", None)):
        raise ValueError("paired physical response requires an explicit physical flow")
    if not contexts or draws < 2 or draws % 2:
        raise ValueError("at least one context and an even draw count >=2 are required")
    batch = len(contexts[0])
    if any(
        context.ndim != 2
        or len(context) != batch
        or context.shape[1] != model.context_dim
        for context in contexts
    ):
        raise ValueError("all contexts must be aligned with the flow context width")
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
    means = []
    for context in contexts:
        expanded = context[:, None, :].expand(batch, draws, model.context_dim)
        sampled, _ = model.forward_physical(
            base, expanded.reshape(batch * draws, model.context_dim)
        )
        means.append(sampled.reshape(batch, draws, model.target_dim).mean(dim=1))
    return means


class PairedResponsePopulation:
    """Matched shape, optionally plus physical-radius, response rows."""

    def __init__(self, context, pairs, measured_delta, gamma, *, radius_scale=None):
        device = context.device
        pairs = np.asarray(pairs)
        delta = np.asarray(measured_delta)
        gamma = np.asarray(gamma)
        components = 2 if radius_scale is None else 3
        if radius_scale is not None and (
            not math.isfinite(radius_scale) or radius_scale <= 0
        ):
            raise ValueError("positive finite physical radius scale required")
        if (
            context.ndim != 2
            or pairs.ndim != 2
            or pairs.shape[1] != 2
            or not len(pairs)
            or not np.issubdtype(pairs.dtype, np.integer)
            or np.any(pairs < 0)
            or np.any(pairs >= len(context))
            or delta.shape != (len(pairs), components)
            or gamma.shape != pairs.shape
            or not np.isfinite(delta).all()
            or not np.isfinite(gamma).all()
        ):
            raise ValueError(
                "finite aligned physical differences, shear vectors and paired rows required"
            )
        amplitude = np.linalg.norm(gamma, axis=1)
        if np.any(amplitude <= 0) or not np.isfinite(amplitude).all():
            raise ValueError("nonzero finite shear amplitude required; no flooring")
        self.context = context
        self.pairs = torch.as_tensor(pairs, device=device, dtype=torch.long)
        self.measured_delta = torch.as_tensor(delta, device=device, dtype=torch.float64)
        self.amplitude = torch.as_tensor(amplitude, device=device, dtype=torch.float64)
        self.output_scale = torch.as_tensor(
            [1.0, 1.0] + ([] if radius_scale is None else [radius_scale]),
            device=device,
            dtype=torch.float64,
        )

    def residual(self, model, indices, draws, seed):
        pair = self.pairs[indices]
        means = physical_context_means(
            model,
            [self.context[pair[:, 0]], self.context[pair[:, 1]]],
            draws,
            seed,
        )
        components = len(self.output_scale)
        value = (
            means[1][:, :components]
            - means[0][:, :components]
            - self.measured_delta[indices]
        ) / self.amplitude[indices, None] / self.output_scale
        if not bool(torch.isfinite(value).all()):
            raise ValueError("nonfinite paired physical response residual")
        return value


def paired_cross_score(a, c):
    """Per-pair independent-replica cross score with fixed divisor four."""

    if a.ndim != 2 or a.shape[-1] not in (2, 3) or c.shape != a.shape:
        raise ValueError("aligned two- or three-component residuals required")
    return (a * c).sum(-1) / 4.0


def paired_response_backward(
    model,
    population,
    pairs_per_step,
    draws,
    *,
    chunk_size,
    seed,
    weight=1.0,
):
    """Accumulate a sampled response gradient without clearing NLL gradients."""

    if (
        min(pairs_per_step, chunk_size) < 1
        or draws < 2
        or draws % 2
        or seed < 0
        or not math.isfinite(weight)
        or weight < 0
    ):
        raise ValueError(
            "positive batch sizes, even draws, and nonnegative finite weight/seed required"
        )
    device = population.context.device
    indices = torch.as_tensor(
        np.random.default_rng(seed).integers(
            len(population.pairs), size=int(pairs_per_step)
        ),
        device=device,
        dtype=torch.long,
    )
    total = 0.0
    component_losses = torch.zeros(
        len(population.output_scale), device=device, dtype=torch.float64
    )
    for block, start in enumerate(range(0, pairs_per_step, chunk_size)):
        rows = indices[start : start + chunk_size]
        with torch.set_grad_enabled(weight > 0):
            a = population.residual(model, rows, draws, seed + 100000 + 2 * block)
            c = population.residual(model, rows, draws, seed + 100001 + 2 * block)
            loss = paired_cross_score(a, c).sum() / pairs_per_step
            if not bool(torch.isfinite(loss)):
                raise ValueError("nonfinite paired response score")
            if weight:
                (weight * loss).backward()
        total += float(loss.detach())
        component_losses += (a.detach() * c.detach()).sum(0) / (4 * pairs_per_step)
    return {
        "loss": total,
        "pairs_per_group": int(pairs_per_step),
        "component_losses": component_losses.cpu().numpy(),
    }


def summarize_paired_residuals(residuals):
    """Use all distinct latent-group products and a group jackknife."""

    residuals = np.asarray(residuals, dtype=np.float64)
    if (
        residuals.ndim != 3
        or residuals.shape[0] < 4
        or residuals.shape[1] < 1
        or residuals.shape[2] not in (2, 3)
        or not np.isfinite(residuals).all()
    ):
        raise ValueError(
            "at least four finite aligned latent groups of response residuals required"
        )

    def scores(values):
        count = len(values)
        return (
            (values.sum(0) ** 2 - (values**2).sum(0)).sum(-1)
            / (4 * count * (count - 1))
        )

    per_pair = scores(residuals)
    deleted = np.array(
        [scores(np.delete(residuals, k, axis=0)).mean() for k in range(len(residuals))]
    )
    se = math.sqrt(
        (len(deleted) - 1)
        / len(deleted)
        * np.sum((deleted - deleted.mean()) ** 2)
    )
    naive = np.mean(np.sum(residuals.mean(0) ** 2, axis=-1)) / 4
    component_scores = (
        (residuals.sum(0) ** 2 - (residuals**2).sum(0)).mean(0)
        / (4 * len(residuals) * (len(residuals) - 1))
    )
    component_deleted = np.array(
        [
            (values.sum(0) ** 2 - (values**2).sum(0)).mean(0)
            / (4 * len(values) * (len(values) - 1))
            for values in (
                np.delete(residuals, k, axis=0) for k in range(len(residuals))
            )
        ]
    )

    def component_se(values):
        return float(
            np.sqrt(
                (len(values) - 1)
                / len(values)
                * np.sum((values - values.mean()) ** 2)
            )
        )

    result = {
        "response_loss": float(per_pair.mean()),
        "shape_loss": float(component_scores[:2].sum()),
        "per_pair": per_pair,
        "component_losses": component_scores,
        "shape_mc_jackknife_se": component_se(component_deleted[:, :2].sum(1)),
        "mc_jackknife_se": se,
        "mc_deleted_scores": deleted,
        "mean_quadratic": float(naive),
        "integration_quadratic": float(naive - per_pair.mean()),
    }
    if residuals.shape[2] == 3:
        result.update(
            radius_loss=float(component_scores[2]),
            radius_mc_jackknife_se=component_se(component_deleted[:, 2]),
        )
    else:
        result["shape_loss"] = result["response_loss"]
    return result


def evaluate_paired_response(
    model,
    population,
    indices,
    *,
    draws=64,
    groups=4,
    batch_size=1024,
    seed=3180000000001,
):
    """Evaluate a fixed matched-pair subset with independent latent groups."""

    if (
        groups < 4
        or groups > 100
        or batch_size < 1
        or len(indices) < 1
        or len(indices) >= 1_000_000_000
    ):
        raise ValueError("valid group/batch sizes and nonempty pair indices required")
    model.eval()
    indices = torch.as_tensor(indices, dtype=torch.long, device=population.context.device)
    values = []
    with torch.no_grad():
        for group in range(groups):
            chunks = [
                population.residual(
                    model,
                    indices[start : start + batch_size],
                    draws,
                    seed + group * 1_000_000_000 + start,
                )
                .cpu()
                .numpy()
                for start in range(0, len(indices), batch_size)
            ]
            values.append(np.concatenate(chunks))
    residuals = np.stack(values)
    return {
        **summarize_paired_residuals(residuals),
        "group_residuals": residuals,
        "draws": int(draws),
        "groups": int(groups),
        "seed": int(seed),
        "pairs": int(len(indices)),
    }


__all__ = [
    "PairedResponsePopulation",
    "evaluate_paired_response",
    "paired_cross_score",
    "paired_response_backward",
    "physical_context_means",
    "summarize_paired_residuals",
]
