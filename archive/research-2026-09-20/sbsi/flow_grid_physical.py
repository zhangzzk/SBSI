"""Cell-balanced physical response supervision for conditional flows.

The grid is defined by fixed baseline-leg properties.  Within each cell the
shape response is a two-by-two forward regression on the sky shear.  The
scalar radius response is a two-component regression in the baseline shape
frame.  Case-jackknife covariances, rather than row scatter, set the metric.

This module deliberately contains no catalogue selection.  Callers must pass
an already fixed population and aligned zero/sheared pairs.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from .flow_paired_shape import physical_context_means


RESPONSE_COMPONENTS = (
    "R11",
    "R12",
    "R21",
    "R22",
    "radius_parallel",
    "radius_cross",
)


def quantile_grid_edges(features, bins=6):
    """Return distinct marginal quantile boundaries for a three-axis grid."""

    values = np.asarray(features, dtype=np.float64)
    if (
        values.ndim != 2
        or values.shape[1] != 3
        or len(values) < bins
        or bins < 2
        or not np.isfinite(values).all()
    ):
        raise ValueError("finite three-axis features and at least two bins required")
    edges = np.quantile(values, np.linspace(0.0, 1.0, bins + 1), axis=0).T
    if np.any(np.diff(edges, axis=1) <= 0.0):
        raise ValueError("grid quantiles do not define distinct intervals")
    return edges


def assign_grid_cells(features, edges):
    """Assign finite features to cells, retaining tails in the outer cells."""

    values = np.asarray(features, dtype=np.float64)
    boundaries = np.asarray(edges, dtype=np.float64)
    if (
        values.ndim != 2
        or values.shape[1] != 3
        or boundaries.ndim != 2
        or boundaries.shape[0] != 3
        or boundaries.shape[1] < 3
        or not np.isfinite(values).all()
        or not np.isfinite(boundaries).all()
        or np.any(np.diff(boundaries, axis=1) <= 0.0)
    ):
        raise ValueError("finite aligned three-axis features and boundaries required")
    bins = boundaries.shape[1] - 1
    indices = [
        np.searchsorted(boundaries[axis, 1:-1], values[:, axis], side="right")
        for axis in range(3)
    ]
    return np.ravel_multi_index(indices, (bins, bins, bins)).astype(np.int64)


def baseline_frame_gamma(gamma, ellipticity):
    """Express shear in the fixed baseline spin-2 frame.

    Exactly circular baseline shapes use the sky axes.  The returned first
    component is parallel to the baseline ellipticity and the second is the
    cross component.
    """

    gamma = np.asarray(gamma, dtype=np.float64)
    ellipticity = np.asarray(ellipticity, dtype=np.float64)
    if (
        gamma.ndim != 2
        or gamma.shape[1] != 2
        or ellipticity.shape != gamma.shape
        or not np.isfinite(gamma).all()
        or not np.isfinite(ellipticity).all()
    ):
        raise ValueError("finite aligned two-component shear and ellipticity required")
    norm = np.linalg.norm(ellipticity, axis=1)
    unit = np.zeros_like(ellipticity)
    unit[:, 0] = 1.0
    np.divide(ellipticity, norm[:, None], out=unit, where=norm[:, None] > 0.0)
    return np.column_stack(
        (
            np.sum(unit * gamma, axis=1),
            -unit[:, 1] * gamma[:, 0] + unit[:, 0] * gamma[:, 1],
        )
    )


def _fit_cell_regression(delta, gamma, cell_ids, cases, n_cells):
    """Fit per-cell forward regressions and delete-one-case replicates."""

    delta = np.asarray(delta, dtype=np.float64)
    gamma = np.asarray(gamma, dtype=np.float64)
    cell_ids = np.asarray(cell_ids)
    cases = np.asarray(cases)
    if (
        delta.ndim != 2
        or delta.shape[1] < 1
        or gamma.shape != (len(delta), 2)
        or cell_ids.shape != (len(delta),)
        or cases.shape != cell_ids.shape
        or not len(delta)
        or n_cells < 1
        or not np.isfinite(delta).all()
        or not np.isfinite(gamma).all()
        or not np.issubdtype(cell_ids.dtype, np.integer)
        or np.any(cell_ids < 0)
        or np.any(cell_ids >= n_cells)
    ):
        raise ValueError("invalid aligned cell regression arrays")
    case_labels, case_ids = np.unique(cases, return_inverse=True)
    n_cases = len(case_labels)
    if n_cases < 8:
        raise ValueError("grid response covariance requires at least eight cases")
    grouped = case_ids * n_cells + cell_ids
    counts_by_case = np.bincount(
        grouped, minlength=n_cases * n_cells
    ).reshape(n_cases, n_cells)
    counts = counts_by_case.sum(axis=0)
    coverage = (counts_by_case > 0).sum(axis=0)
    if np.any(counts < 8) or np.any(coverage < 8):
        raise ValueError("every grid cell requires at least eight rows and eight cases")

    normal = np.empty((n_cases, n_cells, 2, 2), dtype=np.float64)
    cross = np.empty(
        (n_cases, n_cells, delta.shape[1], 2), dtype=np.float64
    )
    for left in range(2):
        for right in range(2):
            normal[..., left, right] = np.bincount(
                grouped,
                weights=gamma[:, left] * gamma[:, right],
                minlength=n_cases * n_cells,
            ).reshape(n_cases, n_cells)
        for output in range(delta.shape[1]):
            cross[..., output, left] = np.bincount(
                grouped,
                weights=delta[:, output] * gamma[:, left],
                minlength=n_cases * n_cells,
            ).reshape(n_cases, n_cells)

    total_normal = normal.sum(axis=0)
    total_cross = cross.sum(axis=0)
    deleted_normal = total_normal[None] - normal
    if np.any(np.linalg.cond(total_normal) > 1.0e8) or np.any(
        np.linalg.cond(deleted_normal) > 1.0e8
    ):
        raise ValueError("unstable full or case-deleted cell shear design")
    inverse = np.linalg.inv(total_normal)
    target = (total_cross @ inverse).reshape(n_cells, -1)
    deleted = (
        (total_cross[None] - cross) @ np.linalg.inv(deleted_normal)
    ).reshape(n_cases, n_cells, -1)
    return {
        "target": target,
        "jackknife": deleted,
        "counts": counts,
        "case_coverage": coverage,
        "case_labels": case_labels,
        "normal": total_normal,
    }


def _equilibrated_precision(covariance):
    covariance = np.asarray(covariance, dtype=np.float64)
    if (
        covariance.ndim != 3
        or covariance.shape[1] != covariance.shape[2]
        or not np.isfinite(covariance).all()
    ):
        raise ValueError("finite square response covariance required")
    scales = np.sqrt(np.diagonal(covariance, axis1=1, axis2=2))
    if np.any(scales <= 0.0) or not np.isfinite(scales).all():
        raise ValueError("strictly positive response variances required")
    correlation = covariance / scales[:, :, None] / scales[:, None, :]
    eigenvalues = np.linalg.eigvalsh(correlation)
    if (
        not np.isfinite(eigenvalues).all()
        or np.any(eigenvalues <= 0.0)
        or np.any(eigenvalues[:, -1] / eigenvalues[:, 0] > 1.0e10)
    ):
        raise ValueError("singular or ill-conditioned joint response covariance")
    precision = (
        np.linalg.inv(correlation) / scales[:, :, None] / scales[:, None, :]
    )
    return precision, eigenvalues


def fit_physical_grid_response(
    measured_delta,
    gamma,
    scalar_gamma,
    cell_ids,
    cases,
    n_cells,
):
    """Fit the joint four-shape plus two-radius response in each cell."""

    delta = np.asarray(measured_delta, dtype=np.float64)
    if delta.ndim != 2 or delta.shape[1] != 3 or not np.isfinite(delta).all():
        raise ValueError("shape1, shape2 and physical-radius differences required")
    shape = _fit_cell_regression(delta[:, :2], gamma, cell_ids, cases, n_cells)
    radius_scale = float(np.std(delta[:, 2], ddof=1))
    if not math.isfinite(radius_scale) or radius_scale <= 0.0:
        raise ValueError("nondegenerate physical-radius differences required")
    radius = _fit_cell_regression(
        delta[:, 2:3] / radius_scale,
        scalar_gamma,
        cell_ids,
        cases,
        n_cells,
    )
    if not np.array_equal(shape["counts"], radius["counts"]) or not np.array_equal(
        shape["case_labels"], radius["case_labels"]
    ):
        raise RuntimeError("shape and radius response populations differ")
    target = np.concatenate(
        (shape["target"], radius["target"] * radius_scale), axis=1
    )
    jackknife = np.concatenate(
        (shape["jackknife"], radius["jackknife"] * radius_scale), axis=2
    )
    centered = jackknife - jackknife.mean(axis=0)
    n_cases = len(jackknife)
    covariance = (n_cases - 1) / n_cases * np.einsum(
        "cbi,cbj->bij", centered, centered
    )
    precision, correlation_eigenvalues = _equilibrated_precision(covariance)
    return {
        "target": target,
        "covariance": covariance,
        "precision": precision,
        "jackknife": jackknife,
        "counts": shape["counts"],
        "case_coverage": np.minimum(
            shape["case_coverage"], radius["case_coverage"]
        ),
        "case_labels": shape["case_labels"],
        "shape_normal": shape["normal"],
        "scalar_normal": radius["normal"],
        "radius_numerical_scale": np.asarray(radius_scale),
        "correlation_eigenvalues": correlation_eigenvalues,
    }


def independent_grid_score(mean_a, mean_c, target, precision):
    """Return the half cross-quadratic for each cell."""

    if (
        mean_a.shape != mean_c.shape
        or mean_a.shape != target.shape
        or mean_a.ndim != 2
        or mean_a.shape[1] != len(RESPONSE_COMPONENTS)
        or precision.shape
        != (len(target), len(RESPONSE_COMPONENTS), len(RESPONSE_COMPONENTS))
    ):
        raise ValueError("aligned six-component grid means and precision required")
    left = mean_a - target
    right = mean_c - target
    return 0.5 * torch.einsum("bi,bij,bj->b", left, precision, right)


class PhysicalGridPopulation:
    """A cell-sorted response population referencing an existing context tensor."""

    def __init__(
        self,
        context,
        pairs,
        cell_ids,
        gamma,
        scalar_gamma,
        statistics,
        *,
        precision=None,
    ):
        pairs = np.asarray(pairs)
        cell_ids = np.asarray(cell_ids)
        gamma = np.asarray(gamma, dtype=np.float64)
        scalar_gamma = np.asarray(scalar_gamma, dtype=np.float64)
        target = np.asarray(statistics["target"], dtype=np.float64)
        fixed_precision = np.asarray(
            statistics["precision"] if precision is None else precision,
            dtype=np.float64,
        )
        n_cells = len(target)
        if (
            context.ndim != 2
            or pairs.ndim != 2
            or pairs.shape[1] != 2
            or not len(pairs)
            or not np.issubdtype(pairs.dtype, np.integer)
            or np.any(pairs < 0)
            or np.any(pairs >= len(context))
            or cell_ids.shape != (len(pairs),)
            or not np.issubdtype(cell_ids.dtype, np.integer)
            or np.any(cell_ids < 0)
            or np.any(cell_ids >= n_cells)
            or gamma.shape != (len(pairs), 2)
            or scalar_gamma.shape != gamma.shape
            or not np.isfinite(gamma).all()
            or not np.isfinite(scalar_gamma).all()
            or target.shape != (n_cells, len(RESPONSE_COMPONENTS))
            or fixed_precision.shape
            != (n_cells, len(RESPONSE_COMPONENTS), len(RESPONSE_COMPONENTS))
        ):
            raise ValueError("invalid aligned physical grid population")
        order = np.argsort(cell_ids, kind="stable")
        sorted_cells = cell_ids[order].astype(np.int64, copy=False)
        counts = np.bincount(sorted_cells, minlength=n_cells)
        if np.any(counts == 0):
            raise ValueError("every response cell must be populated")
        if not np.array_equal(counts, np.asarray(statistics["counts"])):
            raise ValueError("population counts differ from response statistics")

        device = context.device
        self.context = context
        self.pairs = torch.as_tensor(pairs[order], dtype=torch.long, device=device)
        self.bin_ids = torch.as_tensor(
            sorted_cells, dtype=torch.long, device=device
        )
        self.gamma = torch.as_tensor(gamma[order], dtype=torch.float64, device=device)
        self.scalar_gamma = torch.as_tensor(
            scalar_gamma[order], dtype=torch.float64, device=device
        )
        self.target = torch.as_tensor(target, dtype=torch.float64, device=device)
        self.precision = torch.as_tensor(
            fixed_precision, dtype=torch.float64, device=device
        )
        self.offsets = np.r_[0, np.cumsum(counts)].astype(np.int64)
        self.n_cells = n_cells

        shape_normal = torch.as_tensor(
            np.asarray(statistics["shape_normal"], dtype=np.float64),
            dtype=torch.float64,
            device=device,
        )
        scalar_normal = torch.as_tensor(
            np.asarray(statistics["scalar_normal"], dtype=np.float64),
            dtype=torch.float64,
            device=device,
        )
        if shape_normal.shape != (n_cells, 2, 2) or scalar_normal.shape != (
            n_cells,
            2,
            2,
        ):
            raise ValueError("one shape and scalar shear normal per cell required")
        cell_counts = torch.as_tensor(counts, dtype=torch.float64, device=device)
        self.shape_design = (
            torch.einsum(
                "ni,nij->nj",
                self.gamma,
                torch.linalg.inv(shape_normal)[self.bin_ids],
            )
            * cell_counts[self.bin_ids, None]
        )
        self.scalar_design = (
            torch.einsum(
                "ni,nij->nj",
                self.scalar_gamma,
                torch.linalg.inv(scalar_normal)[self.bin_ids],
            )
            * cell_counts[self.bin_ids, None]
        )
        if not bool(
            torch.isfinite(self.target).all()
            and torch.isfinite(self.precision).all()
            and torch.isfinite(self.shape_design).all()
            and torch.isfinite(self.scalar_design).all()
        ):
            raise ValueError("nonfinite physical grid statistics")

    def contributions(self, model, indices, draws, seed):
        pair = self.pairs[indices]
        means = physical_context_means(
            model,
            [self.context[pair[:, 0]], self.context[pair[:, 1]]],
            draws,
            seed,
        )
        delta = means[1] - means[0]
        shape = (
            delta[:, :2, None] * self.shape_design[indices, None, :]
        ).reshape(-1, 4)
        radius = delta[:, 2:3, None] * self.scalar_design[indices, None, :]
        return torch.cat((shape, radius.reshape(-1, 2)), dim=1)


def _response_plan(population, cells_per_step, pairs_per_cell, seed):
    if (
        cells_per_step < 1
        or cells_per_step > population.n_cells
        or pairs_per_cell < 1
        or seed < 0
    ):
        raise ValueError("valid positive cell/pair sample sizes and seed required")
    rng = np.random.default_rng(seed)
    cells = rng.permutation(population.n_cells)[:cells_per_step]
    local = torch.arange(
        cells_per_step, device=population.context.device
    ).repeat_interleave(pairs_per_cell)
    groups = []
    for _ in range(2):
        groups.append(
            torch.as_tensor(
                np.concatenate(
                    [
                        rng.integers(
                            population.offsets[cell],
                            population.offsets[cell + 1],
                            pairs_per_cell,
                        )
                        for cell in cells
                    ]
                ),
                dtype=torch.long,
                device=population.context.device,
            )
        )
    return cells, local, groups


def sampled_grid_response_backward(
    model,
    population,
    cells_per_step,
    pairs_per_cell,
    draws,
    *,
    chunk_size,
    seed,
    weight=1.0,
):
    """Backpropagate an unbiased cell-mean cross quadratic in chunks."""

    if (
        draws < 2
        or draws % 2
        or chunk_size < 1
        or not math.isfinite(weight)
        or weight < 0.0
    ):
        raise ValueError("even draws, positive chunk size and nonnegative weight required")
    cells, local, groups = _response_plan(
        population, cells_per_step, pairs_per_cell, seed
    )
    cell_tensor = torch.as_tensor(
        cells, dtype=torch.long, device=population.context.device
    )

    def chunks(stream):
        for block, start in enumerate(range(0, len(local), chunk_size)):
            stop = min(start + chunk_size, len(local))
            yield (
                groups[stream][start:stop],
                local[start:stop],
                seed + 100_000 + 2 * block + stream,
            )

    divisor = torch.bincount(local, minlength=len(cells)).to(torch.float64)
    means = []
    with torch.no_grad():
        for stream in range(2):
            total = torch.zeros(
                (len(cells), len(RESPONSE_COMPONENTS)),
                dtype=torch.float64,
                device=population.context.device,
            )
            for rows, bins, latent_seed in chunks(stream):
                values = population.contributions(model, rows, draws, latent_seed)
                if not bool(torch.isfinite(values).all()):
                    raise ValueError("nonfinite sampled grid response")
                total.index_add_(0, bins, values)
            means.append(total / divisor[:, None])
        target = population.target[cell_tensor]
        precision = population.precision[cell_tensor]
        score = independent_grid_score(*means, target, precision).mean() / len(
            RESPONSE_COMPONENTS
        )
        coefficients = [
            0.5
            * torch.einsum("bij,bj->bi", precision, means[1] - target)
            / (len(RESPONSE_COMPONENTS) * len(cells)),
            0.5
            * torch.einsum("bji,bj->bi", precision, means[0] - target)
            / (len(RESPONSE_COMPONENTS) * len(cells)),
        ]
    if not bool(torch.isfinite(score)):
        raise ValueError("nonfinite sampled grid score")
    if weight:
        with torch.enable_grad():
            for stream in range(2):
                for rows, bins, latent_seed in chunks(stream):
                    values = population.contributions(
                        model, rows, draws, latent_seed
                    )
                    surrogate = (
                        values
                        * coefficients[stream][bins]
                        / divisor[bins, None]
                    ).sum() * weight
                    if not bool(torch.isfinite(surrogate)):
                        raise ValueError("nonfinite grid-response gradient replay")
                    surrogate.backward()
    return {
        "loss": float(score),
        "cells": cells.tolist(),
        "pairs_per_group": int(len(local)),
    }


def summarize_grid_means(group_means, target, precision):
    """All-distinct-group cross score and finite-integration diagnostics."""

    means = np.asarray(group_means, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    precision = np.asarray(precision, dtype=np.float64)
    if (
        means.ndim != 3
        or means.shape[0] < 4
        or means.shape[2] != len(RESPONSE_COMPONENTS)
        or target.shape != means.shape[1:]
        or precision.shape
        != (means.shape[1], means.shape[2], means.shape[2])
        or not np.isfinite(means).all()
        or not np.isfinite(target).all()
        or not np.isfinite(precision).all()
    ):
        raise ValueError("at least four finite aligned grid-mean groups required")

    def scores(values):
        residual = values - target
        center = residual.mean(axis=0)
        deviations = residual - center
        mc_covariance = np.einsum(
            "gbi,gbj->bij", deviations, deviations
        ) / (len(values) * (len(values) - 1))
        quadratic = (
            0.5
            * np.einsum("bi,bij,bj->b", center, precision, center)
            / means.shape[2]
        )
        integration = (
            0.5
            * np.einsum("bij,bji->b", precision, mc_covariance)
            / means.shape[2]
        )
        return quadratic - integration, quadratic, integration, mc_covariance

    per_cell, quadratic, integration, mc_covariance = scores(means)
    deleted = np.asarray(
        [scores(np.delete(means, index, axis=0))[0].mean() for index in range(len(means))]
    )
    mc_se = math.sqrt(
        (len(deleted) - 1)
        / len(deleted)
        * np.square(deleted - deleted.mean()).sum()
    )
    return {
        "loss": float(per_cell.mean()),
        "per_cell": per_cell,
        "group_means": means,
        "mean_quadratic": float(quadratic.mean()),
        "integration_quadratic": float(integration.mean()),
        "integration_covariance_of_mean": mc_covariance,
        "mc_jackknife_se": mc_se,
        "mc_deleted_scores": deleted,
    }


@torch.no_grad()
def evaluate_grid_response(
    model,
    population,
    *,
    draws=64,
    groups=4,
    batch_size=1024,
    seed=7301,
):
    """Evaluate complete cell means for several independent latent groups."""

    if groups < 4 or draws < 2 or draws % 2 or batch_size < 1:
        raise ValueError("four groups, even draws and a positive batch size required")
    model.eval()
    counts = torch.as_tensor(
        np.diff(population.offsets),
        dtype=torch.float64,
        device=population.context.device,
    )
    means = []
    for group in range(groups):
        total = torch.zeros_like(population.target)
        for block, start in enumerate(range(0, len(population.pairs), batch_size)):
            rows = torch.arange(
                start,
                min(start + batch_size, len(population.pairs)),
                dtype=torch.long,
                device=population.context.device,
            )
            values = population.contributions(
                model,
                rows,
                draws,
                seed + group * 1_000_000_000 + block,
            )
            if not bool(torch.isfinite(values).all()):
                raise ValueError("nonfinite validation grid response")
            total.index_add_(0, population.bin_ids[rows], values)
        means.append((total / counts[:, None]).cpu().numpy())
    return summarize_grid_means(
        np.stack(means),
        population.target.cpu().numpy(),
        population.precision.cpu().numpy(),
    )


__all__ = [
    "PhysicalGridPopulation",
    "RESPONSE_COMPONENTS",
    "assign_grid_cells",
    "baseline_frame_gamma",
    "evaluate_grid_response",
    "fit_physical_grid_response",
    "independent_grid_score",
    "quantile_grid_edges",
    "sampled_grid_response_backward",
    "summarize_grid_means",
]
