"""Covariance-weighted cell responses with independent stochastic mean estimates.

The matrix is the forward paired regression delta_e = R gamma on each
population, in row-major order R11,R12,R21,R22. Covariance is that of the
estimated cell matrix, using a delete-one-case jackknife, not row scatter.
"""

import numpy as np
import torch


def fit_grid_response(delta_e, gamma, bin_ids, cases, n_cells):
    """Return cell WLS matrices, full 4x4 jackknife covariances and design vectors.

    Unit pair weights. The model uses the identical fixed design: mean of
    delta_e_model outer (gamma @ inverse(mean(gamma outer gamma))).
    No fallback, outlier clipping, covariance floor or fitted correction.
    """
    delta_e, gamma = np.asarray(delta_e, float), np.asarray(gamma, float)
    bin_ids, cases = np.asarray(bin_ids), np.asarray(cases)
    if (delta_e.ndim != 2 or delta_e.shape[1] != 2 or gamma.shape != delta_e.shape
            or bin_ids.shape != (len(gamma),) or cases.shape != bin_ids.shape
            or not len(gamma) or n_cells < 1
            or not np.isfinite(delta_e).all() or not np.isfinite(gamma).all()
            or not np.issubdtype(bin_ids.dtype, np.integer)
            or np.any(bin_ids < 0) or np.any(bin_ids >= n_cells)):
        raise ValueError("invalid finite aligned grid-response data")
    unique_cases, case_ids = np.unique(cases, return_inverse=True)
    n_cases = len(unique_cases)
    if n_cases < 8:
        raise ValueError("grid covariance requires at least eight independent cases")
    group = case_ids * n_cells + bin_ids
    counts_by_case = np.bincount(group, minlength=n_cases*n_cells).reshape(n_cases, n_cells)
    counts = counts_by_case.sum(axis=0)
    coverage = (counts_by_case > 0).sum(axis=0)
    if np.any(counts < 8) or np.any(coverage < 8):
        raise ValueError("every grid cell needs at least eight rows and eight cases")
    normal = np.empty((n_cases, n_cells, 2, 2))
    cross = np.empty_like(normal)
    for a in range(2):
        for b in range(2):
            normal[..., a, b] = np.bincount(
                group, weights=gamma[:, a]*gamma[:, b], minlength=n_cases*n_cells
            ).reshape(n_cases, n_cells)
            cross[..., a, b] = np.bincount(
                group, weights=delta_e[:, a]*gamma[:, b], minlength=n_cases*n_cells
            ).reshape(n_cases, n_cells)
    total_normal, total_cross = normal.sum(axis=0), cross.sum(axis=0)
    leave_normal = total_normal[None] - normal
    if (np.any(np.linalg.cond(total_normal) > 1e8)
            or np.any(np.linalg.cond(leave_normal) > 1e8)):
        raise ValueError("unstable cell shear design, including case deletions")
    inverse = np.linalg.inv(total_normal)
    target = (total_cross @ inverse).reshape(n_cells, 4)
    leave = ((total_cross[None]-cross) @ np.linalg.inv(leave_normal)).reshape(n_cases, n_cells, 4)
    centered = leave - leave.mean(axis=0)
    covariance = (n_cases-1)/n_cases * np.einsum("cbi,cbj->bij", centered, centered)
    eigenvalues = np.linalg.eigvalsh(covariance)
    if (not np.isfinite(covariance).all() or np.any(eigenvalues <= 0)
            or np.any(eigenvalues[:, -1]/eigenvalues[:, 0] > 1e10)):
        raise ValueError("singular/ill-conditioned response covariance; use a coarser grid")
    precision = np.linalg.inv(covariance)
    # Normalizing by the *full cell* count keeps sampled contributions unbiased.
    design = np.einsum("ni,nij->nj", gamma, inverse[bin_ids]) * counts[bin_ids, None]
    return dict(target=target, covariance=covariance, precision=precision,
                counts=counts, case_coverage=coverage, cases=unique_cases,
                jackknife=leave, eigenvalues=eigenvalues, design=design)


def independent_grid_score(mean_a, mean_c, target, precision):
    """One half cross-quadratic per cell; A and C must be independent estimates.

    Includes the fixed target-only constant. Scores can be negative. There is
    no occupancy weighting or division by the number of cells in the objective.
    """
    if (mean_a.shape != mean_c.shape or mean_a.shape != target.shape
            or mean_a.ndim != 2 or mean_a.shape[1] not in (4,6,8)
            or precision.shape != (len(target), target.shape[1], target.shape[1])):
        raise ValueError("grid score needs aligned four/six/eight-component means and precision")
    a, c = mean_a-target, mean_c-target
    return 0.5 * torch.einsum("bi,bij,bj->b", a, precision, c)


class GridResponsePopulation:
    """Matched finite population referencing the existing NLL context tensor."""

    def __init__(self, context, payload, shape_scales):
        self.context = context
        device = context.device
        self.pairs = torch.as_tensor(payload["pairs"], dtype=torch.long, device=device)
        self.design = torch.as_tensor(payload["design"], dtype=torch.float64, device=device)
        self.target = torch.as_tensor(payload["target"], dtype=torch.float64, device=device)
        self.precision = torch.as_tensor(payload["precision"], dtype=torch.float64, device=device)
        self.bin_ids = torch.as_tensor(payload["bin_ids"], dtype=torch.long, device=device)
        self.offsets = np.r_[0, np.cumsum(payload["counts"])].astype(int)
        self.scales = torch.as_tensor(shape_scales, dtype=context.dtype, device=device)
        self.n_cells = len(self.target)
        if (len(self.pairs) != self.offsets[-1]
                or not torch.equal(self.bin_ids, self.bin_ids.sort().values)):
            raise ValueError("population must contain all pairs sorted by cell")

    def contributions(self, model, indices, draws, seed=None):
        from .flow_training import complete_flow_context_means
        pair = self.pairs[indices]
        means = complete_flow_context_means(
            model, [self.context[pair[:, 0]], self.context[pair[:, 1]]], draws, seed=seed)
        delta = ((means[1][:, :2]-means[0][:, :2])*self.scales).double()
        return (delta[:, :, None]*self.design[indices, None, :]).reshape(-1, 4)

    def sampled_score(self, model, cells_per_step, pairs_per_cell, draws):
        """Unbiased estimate of 0.5*SUM_cells residual^T V^-1 residual.

        Independent with-replacement galaxy samples AND independent latent groups
        avoid adding either minibatch galaxy variance or latent MC variance.
        """
        k = min(cells_per_step, self.n_cells)
        if k < 1 or pairs_per_cell < 1:
            raise ValueError("positive cell and pair batch sizes required")
        cells = torch.randperm(self.n_cells, device=self.context.device)[:k]
        estimates = []
        for _ in range(2):
            indices = torch.cat([
                torch.randint(int(self.offsets[b]), int(self.offsets[b+1]),
                              (pairs_per_cell,), device=self.context.device)
                for b in cells.cpu().tolist()
            ])
            estimates.append(self.contributions(model, indices, draws).reshape(k, pairs_per_cell, 4).mean(1))
        return self.n_cells/k * independent_grid_score(
            *estimates, self.target[cells], self.precision[cells]).sum()

    @torch.no_grad()
    def evaluate(self, model, draws, batch_size=1024, seed=7301):
        """Full finite-population means; fixed independent CRN streams, no row cap."""
        means = []
        for stream in range(2):
            sums = torch.zeros_like(self.target)
            for index, start in enumerate(range(0, len(self.pairs), batch_size)):
                rows = torch.arange(start, min(start+batch_size, len(self.pairs)), device=self.context.device)
                # Distinct even/odd seeds keep streams disjoint for any population size.
                values = self.contributions(model, rows, draws, seed+2*index+stream)
                sums.index_add_(0, self.bin_ids[rows], values)
            counts = torch.as_tensor(np.diff(self.offsets), device=self.context.device)
            means.append(sums/counts[:, None])
        scores = independent_grid_score(*means, self.target, self.precision)
        if not torch.isfinite(scores).all():
            raise ValueError("nonfinite held-out grid response")
        return dict(loss=float(scores.sum()), per_cell=scores.cpu().numpy(),
                    mean_a=means[0].cpu().numpy(), mean_c=means[1].cpu().numpy())
