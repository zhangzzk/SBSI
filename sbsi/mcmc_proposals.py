"""Small experimental Gaussian proposal helpers; adaptation is warmup-only."""
import numpy as np


def complementary_stretch_proposal(values, columns, chains_per_observation, rng, *, stretch=2.):
    """Update a random half-ensemble using only frozen complementary walkers.

    Rows are grouped by observation. The caller must use the returned active
    mask and add (dimension-1)*log(z) to the log MH ratio. This is a conditional
    move: nuisance coordinates and each walker's environment stay fixed.
    Columns must be ordinary Euclidean coordinates, never wrapped angles.
    Local full-support moves should accompany a small ensemble.
    """
    x = np.asarray(values, float)
    columns = np.asarray(columns)
    n = chains_per_observation
    if (x.ndim != 2 or not np.isfinite(x).all() or not isinstance(n, (int, np.integer))
            or n < 4 or n % 2 or len(x) % n or not len(x)
            or columns.ndim != 1 or not len(columns) or columns.dtype.kind not in 'iu'
            or len(np.unique(columns)) != len(columns) or np.any(columns < 0)
            or np.any(columns >= x.shape[1]) or not np.isfinite(stretch) or stretch <= 1):
        raise ValueError('finite grouped states, distinct Euclidean columns, even ensemble>=4 and stretch>1 required')
    groups = len(x)//n
    half = n//2
    chosen_half = rng.integers(0, 2, size=groups)
    base = np.arange(groups)*n
    active_rows = (base[:, None]+chosen_half[:, None]*half+np.arange(half)).ravel()
    partner_rows = (base[:, None]+(1-chosen_half[:, None])*half
                    +rng.integers(0, half, size=(groups, half))).ravel()
    z = ((stretch-1)*rng.random(len(active_rows))+1)**2/stretch
    proposed = x.copy()
    active_block = np.ix_(active_rows, columns)
    partner = x[np.ix_(partner_rows, columns)]
    proposed[active_block] = partner+z[:, None]*(x[active_block]-partner)
    correction = np.zeros(len(x))
    correction[active_rows] = (len(columns)-1)*np.log(z)
    active = np.zeros(len(x), dtype=bool)
    active[active_rows] = True
    return proposed, correction, active


def empirical_block_cholesky(history, baseline_scales, *, shrinkage=.05, circular_indices=()):
    """Per-chain covariance of retained states, including MH rejections.

    History has shape (draw, chain, coordinate). Shrink toward the initial
    diagonal proposal to retain full support when a chain has barely moved.
    Call only during discarded warmup and freeze the returned factors for
    posterior sampling. This changes proposal efficiency, never the target.
    """
    values, scales = np.asarray(history, float), np.asarray(baseline_scales, float)
    if (values.ndim != 3 or len(values) < 2 or scales.shape != values.shape[2:]
            or not np.isfinite(values).all() or not np.isfinite(scales).all()
            or np.any(scales <= 0) or not 0 < shrinkage <= 1):
        raise ValueError('finite draw/chain/coordinate history, positive scales and shrinkage required')
    if circular_indices:
        values = values.copy()
        for column in circular_indices:
            if not isinstance(column, (int, np.integer)) or not 0 <= column < values.shape[2]:
                raise ValueError('in-range circular coordinate index required')
            reference = values[-1:, :, column]
            values[:, :, column] = reference+(values[:, :, column]-reference+np.pi) % (2*np.pi)-np.pi
    centered = values-values.mean(axis=0, keepdims=True)
    covariance = np.einsum('tni,tnj->nij', centered, centered)/(len(values)-1)
    covariance = (1-shrinkage)*covariance + shrinkage*np.diag(scales**2)[None]
    return np.linalg.cholesky(covariance)


def gaussian_block_delta(cholesky, scale, rng):
    """Symmetric increments from frozen per-chain covariance and scalar scales."""
    factor, scale = np.asarray(cholesky, float), np.asarray(scale, float)
    if (factor.ndim != 3 or factor.shape[1] != factor.shape[2]
            or scale.shape != factor.shape[:1] or not np.isfinite(factor).all()
            or not np.isfinite(scale).all() or np.any(scale <= 0)):
        raise ValueError('aligned finite covariance factors and positive chain scales required')
    return np.einsum('nij,nj->ni', factor, rng.normal(size=factor.shape[:2]))*scale[:, None]


def adjacent_swap_permutation(log_likelihood, betas, slots, parity, uniforms):
    """Replica-exchange swaps between adjacent likelihood temperatures.

    Rows are level-major, row = level*slots + slot, and level l targets
    prior*likelihood**betas[l]. Pair levels (l, l+1) with l % 2 == parity, as in
    deterministic even/odd swapping; each pass leaves the product target
    invariant. Accept with exp((b_l-b_{l+1})*(L_{l+1}-L_l)); the prior cancels.
    Returns perm (new row r holds old row perm[r]) and the accepted swaps,
    shape (levels-1, slots), False for pairs not attempted this pass.
    Warmup adaptation state belongs to rows (levels), not to swapped states.
    """
    ll, b, u = (np.asarray(v, float) for v in (log_likelihood, betas, uniforms))
    levels = len(b)
    if (b.ndim != 1 or levels < 2 or not np.isfinite(b).all() or b[0] != 1 or np.any(np.diff(b) >= 0)
            or b[-1] <= 0 or not isinstance(slots, (int, np.integer)) or slots < 1
            or ll.shape != (levels*slots,) or not np.isfinite(ll).all() or parity not in (0, 1)
            or u.shape != (levels-1, slots) or not np.isfinite(u).all() or np.any((u <= 0) | (u >= 1))):
        raise ValueError('decreasing betas from 1, finite level-major likelihoods and open uniforms required')
    grid = ll.reshape(levels, slots)
    accepted = np.zeros((levels-1, slots), dtype=bool)
    attempted = np.arange(levels-1) % 2 == parity
    accepted[attempted] = (np.log(u[attempted])
        < (b[:-1]-b[1:])[attempted, None]*(grid[1:]-grid[:-1])[attempted])
    # Same-parity pairs are disjoint, so the swaps cannot conflict.
    rows = np.arange(levels*slots).reshape(levels, slots)
    perm = rows.copy()
    perm[:-1][accepted] = rows[1:][accepted]
    perm[1:][accepted] = rows[:-1][accepted]
    return perm.ravel(), accepted


def pooled_block_cholesky(history, baseline_scales, chains_per_observation, *, shrinkage=.05):
    """Pool discarded warmup states only within each observation.

    Share the resulting fixed proposal geometry across that observation's
    chains. Pooling includes between-chain displacement; observations never
    share covariance. Freeze before a separate scalar-tuning warmup stage.
    Circular coordinates are intentionally unsupported by this helper.
    """
    values = np.asarray(history, float)
    n = chains_per_observation
    if (values.ndim != 3 or not isinstance(n, (int, np.integer)) or n < 1
            or values.shape[1] < n or values.shape[1] % n):
        raise ValueError('aligned draw/chain/coordinate history and positive chain grouping required')
    t, total, d = values.shape
    grouped = values.reshape(t, total//n, n, d).transpose(0, 2, 1, 3).reshape(t*n, total//n, d)
    factors = empirical_block_cholesky(grouped, baseline_scales, shrinkage=shrinkage)
    return np.repeat(factors, n, axis=0)
