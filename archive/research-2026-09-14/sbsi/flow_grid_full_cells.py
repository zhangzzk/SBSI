"""Exact cell-mean gradients with bounded activation memory and full pair coverage.

The loss is unchanged. Two independent latent streams evaluate every galaxy
in each selected cell. A second pass replays identical latent draws and
accumulates the gradient of those *whole-cell* means. No model update occurs
between passes. This avoids a separate galaxy-subsampling noise term.
"""

import math

import numpy as np
import torch

from .flow_grid_response import independent_grid_score


def _validate(model, population, cells, draws, chunk_size, seed):
    cells = [int(c) for c in cells]
    if (not cells or len(set(cells)) != len(cells)
            or min(cells) < 0 or max(cells) >= population.n_cells
            or chunk_size < 1 or draws < 2 or draws % 2 or seed < 0):
        raise ValueError("distinct valid cells, positive chunks, even draws and nonnegative seed required")
    counts = np.diff(population.offsets)
    if np.any(counts[cells] <= 0):
        raise ValueError("empty cells cannot define mean responses")
    # The explicit latent seed does not replay unrelated stochastic/stateful layers.
    for module in model.modules():
        if module.training and isinstance(module, (torch.nn.modules.dropout._DropoutNd,
                                                   torch.nn.modules.batchnorm._BatchNorm)):
            raise ValueError("cell replay requires deterministic context transforms without training dropout/batchnorm")
    return cells, counts


def _chunks(population, cell, chunk_size):
    first, last = population.offsets[cell:cell+2]
    for start in range(int(first), int(last), chunk_size):
        yield start, torch.arange(start, min(start+chunk_size, int(last)), device=population.context.device)


def _cell_mean(model, population, cell, count, draws, chunk_size, seed, stream):
    total = torch.zeros(population.target.shape[1], dtype=torch.float64, device=population.context.device)
    for start, rows in _chunks(population, cell, chunk_size):
        contribution = population.contributions(model, rows, draws, seed=seed+2*start+stream)
        if not bool(torch.isfinite(contribution).all()):
            raise ValueError("nonfinite model response contribution")
        total = total+contribution.sum(0)
    return total/int(count)


def full_cell_response(model, population, cells, draws, *, chunk_size=256,
                       seed=7301, weight=1., backward=False):
    """Normalized response score for complete selected cells, plus optional gradient.

    The returned score is 1/(2*4*k) SUM of the k cross-quadratics. ``weight``
    multiplies only the accumulated gradient so callers can report raw loss.
    Existing gradients are retained, allowing accumulation with NLL gradients.
    Parameters and optimizer must stay fixed until this function returns.
    """
    cells, counts = _validate(model, population, cells, draws, chunk_size, seed)
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("response gradient weight must be finite and nonnegative")
    ids = torch.as_tensor(cells, device=population.context.device)
    components = population.target.shape[1]
    with torch.no_grad():
        estimates = [torch.stack([
            _cell_mean(model,population,cell,counts[cell],draws,chunk_size,seed,stream)
            for cell in cells]) for stream in range(2)]
        precision, target = population.precision[ids], population.target[ids]
        per_cell = independent_grid_score(*estimates,target,precision)/components
        score = per_cell.mean()
        a, c = estimates[0]-target, estimates[1]-target
        if not bool(torch.isfinite(score)):
            raise ValueError("nonfinite full-cell response score")
        # dL/dA and dL/dC. Include the transpose for exact finite-precision
        # agreement with the cross-quadratic even if P is slightly asymmetric.
        coefficients = [
            .5*torch.einsum("bij,bj->bi",precision,c)/(components*len(cells)),
            .5*torch.einsum("bji,bj->bi",precision,a)/(components*len(cells)),
        ]
    if backward and weight:
        with torch.enable_grad():
            for stream in range(2):
                for local,cell in enumerate(cells):
                    for start,rows in _chunks(population,cell,chunk_size):
                        values=population.contributions(model,rows,draws,seed=seed+2*start+stream)
                        surrogate=(values*coefficients[stream][local]).sum()*weight/int(counts[cell])
                        if not bool(torch.isfinite(surrogate)):
                            raise ValueError("nonfinite full-cell replay contribution")
                        surrogate.backward()
    return dict(loss=float(score),mean_a=estimates[0].detach(),mean_c=estimates[1].detach(),
                cells=cells,per_cell=per_cell.detach(),pairs_per_group=int(counts[cells].sum()))


def epoch_full_cells(model, values, context, population, optimizer, *, weight,
                     cells_per_step=4, draws=16, response_chunk_size=256,
                     nll_chunk_size=4096, seed=7301, max_grad_norm=5.):
    """Visit all NLL rows and all response pairs once per group in an epoch.

    Shuffle cells, divide them into approximately equal groups, and split an
    independent NLL row permutation across the same optimizer steps. Each step
    accumulates gradients for its complete NLL block and complete cell means,
    then clips/updates once. This is cell reshuffling, not iid cell sampling.
    """
    if (cells_per_step < 1 or nll_chunk_size < 1 or len(values) != len(context)
            or not len(values) or optimizer is None or not math.isfinite(weight) or weight < 0
            or not math.isfinite(max_grad_norm) or max_grad_norm <= 0):
        raise ValueError("invalid full-cell epoch options or aligned NLL data")
    counts = torch.bincount(population.bin_ids, minlength=population.n_cells)
    expected = torch.as_tensor(np.diff(population.offsets), device=counts.device)
    if not torch.equal(counts,expected):
        raise ValueError("pair cell indices/counts disagree")
    n_steps = min(math.ceil(population.n_cells/cells_per_step),len(values))
    rng = np.random.default_rng(seed)
    cell_groups = np.array_split(rng.permutation(population.n_cells),n_steps)
    generator = torch.Generator(device=context.device)
    generator.manual_seed(int(seed))
    order = torch.randperm(len(values),device=context.device,generator=generator)
    nll_groups = torch.tensor_split(order,n_steps)
    model.train()
    nll_total = response_total = 0.
    nll_seen = pairs_seen = cells_seen = 0
    norms = []
    for cells,rows in zip(cell_groups,nll_groups):
        optimizer.zero_grad(set_to_none=True)
        block_nll = 0.
        for start in range(0,len(rows),nll_chunk_size):
            indices=rows[start:start+nll_chunk_size]
            contribution=-model.log_prob(values[indices],context[indices]).sum()/len(rows)
            if not bool(torch.isfinite(contribution)):
                raise ValueError("nonfinite full-cell epoch NLL")
            contribution.backward()
            block_nll+=float(contribution.detach())
        result=full_cell_response(model,population,cells,draws,chunk_size=response_chunk_size,
                                  seed=seed,weight=weight,backward=True)
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),max_grad_norm)
        if not bool(torch.isfinite(norm)):
            raise ValueError("nonfinite combined gradient")
        optimizer.step()
        norms.append(float(norm))
        nll_total+=block_nll*len(rows)
        response_total+=result["loss"]*len(cells)
        nll_seen+=len(rows)
        pairs_seen+=result["pairs_per_group"]
        cells_seen+=len(cells)
    if nll_seen != len(values) or pairs_seen != len(population.pairs) or cells_seen != population.n_cells:
        raise RuntimeError("incomplete full-cell epoch coverage")
    return dict(nll=nll_total/nll_seen,grid=response_total/cells_seen,
                nll_rows=nll_seen,response_pairs_per_group=pairs_seen,cells=cells_seen,
                optimizer_steps=n_steps,gradient_norms=norms)
