"""Archived catalogue-draw experiment for the earlier §5B score estimator.

STATUS.  This module is retained to reproduce the finite-M quadrature study and
its tests.  It is not the operational catalogue-prior likelihood.  New work
uses ``scene_prior``, ``catalogue_likelihood``, ``catalogue_sampling``, and
``catalogue_closure``, which keep complete neighbour scenes, detection, fixed
Rblend, and measured-selection normalization in one finite-prior likelihood.

WHAT THIS IS FOR.  `PosteriorShapeEstimator.log_likelihood` integrates over the primary's
true shape on a 2-D grid and holds every OTHER conditioning variable at the scored row's
own catalogue value.  That is correct for a closure test and wrong for deployment: the
true magnitude, size, Sersic index and neighbour fluxes are latent, and `INFERENCE.md`
§5B.1(i) requires them marginalised over a scene prior rather than substituted from
measured quantities.  This module supplies that marginalisation by DRAWING THE SCENE FROM
THE INPUT CATALOGUE -- the empirical joint, so the correlations and the clustering come
along for free and no 6-D parametric prior has to be fitted.

    log p(xhat_i | e_k) = logsumexp_m [ log w_im + log p(xhat_i | e_k, theta_m) ]
                          - logsumexp_m log w_im

with `theta_m` drawn from a pool of catalogue rows.  `log_likelihood_marginal` in
`posterior_shape.py` is the same identity for the three neighbour-flux columns with
uniform weights; this module generalises it to an arbitrary feature subset, adds
importance weights, and -- the point of the exercise -- feeds every prefix of the draw
sequence straight into `scores_from_loglike`, so one pass measures the whole convergence
ladder in the draw count.

WHY ATOMS ARE LEGITIMATE HERE.  The generator needs `grad log p0`, and a point cloud
cannot be differentiated.  But the prior is differentiated ONLY in the directions that
move with gamma -- the 2-D shape subspace, where the smooth fitted `SmoothRadialPrior`
still supplies the density.  The remaining dimensions are INTEGRATED and never
differentiated, and empirical atoms are exact for integration.  The argument stops at
kappa = 0: under magnification the size and flux channels move with the parameter too, and
a catalogue draw would no longer suffice there (`INFERENCE.md` §6).

A FINITE POOL IS NOT AN APPROXIMATION HERE.  The obvious objection to an empirical
prior is that `P` atoms only approximate the population they were drawn from, so the model
density is wrong at finite `P`.  That objection does not apply to the closure test, because
the TARGET is defined as the empirical distribution over those `P` rows and the data are
generated from exactly it: every pool row supplies the conditioning of exactly one scored
object.  Using each row once rather than drawing row indices iid has the same expectation
for both halves of (2.6) -- the per-object expectation `h(theta)` enters each sum linearly,
so `sum_i h(theta_i)` and `N * mean_j h(theta_j)` agree -- and lower variance.  What a
finite pool does cost is the width of the prior the answer is a statement ABOUT, which is a
question for deployment and not an error in the estimator.

THE FINITE-M ERROR IS A BIAS, NOT NOISE.  `(1/M) sum_m p(xhat|e,theta_m)` is an unbiased
estimate of the marginal likelihood, but the estimator consumes its LOGARITHM, and the
posterior weights are nonlinear in it.  The leading term of `E[log Lhat] - log L` is
`-Var/(2 M L^2)`, so the induced shift in `m` falls as `1/M` and does NOT average away
over galaxies.  That is a quadrature error of exactly the same standing as the shape
grid's, and it gets the same treatment: a ladder, plus independent replicates for the
error bar (`pass_fraction_by_node`'s docstring records what happens when a nested prefix
is mistaken for an independent one).  Prefixes here are deliberately NESTED, which makes
consecutive ladder points PAIRED -- the right thing for measuring the difference, and the
wrong thing for calling a ladder point's error bar independent.  Both are reported.

THE PROPOSAL MATTERS MORE THAN THE COUNT.  A measured magnitude and size nearly pin the
true magnitude and size, so a uniform catalogue draw lands almost every atom at
essentially zero weight and the effective sample size collapses far below M.
`draw_indices` therefore also offers a defensive mixture proposal: with probability
`beta` draw from pool rows in the galaxy's own MEASURED cell, otherwise draw uniformly.
The target stays the uniform-over-pool distribution and the weights correct exactly,

    w = 1 / ( beta * 1[cell(m) == cell(i)] / f_cell + (1 - beta) )   <= 1 / (1 - beta),

which is BOUNDED -- no heavy tail, unlike a bare nearest-neighbour proposal.  A proposal
may depend on the data; the target may not, and does not.
"""

from __future__ import annotations

import numpy as np
import torch

from .score_inference import scores_from_loglike


# --------------------------------------------------------------------------------------
# strata in measured space
# --------------------------------------------------------------------------------------

def quantile_edges(values, n_bins):
    """`n_bins+1` edges at equal-count quantiles of `values`, open at both ends.

    Equal count rather than equal width: the proposal wants cells that are equally
    POPULATED, so that `f_cell` never collapses and the bounded weight above stays near 1.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        raise ValueError("no finite values to build quantile edges from")
    q = np.quantile(v, np.linspace(0.0, 1.0, n_bins + 1))
    q[0], q[-1] = -np.inf, np.inf
    return np.unique(q)


def cell_index(v1, v2, edges1, edges2):
    """Flat 2-D cell id for each row; NaN coordinates land in cell -1 (no stratum)."""
    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)
    n2 = len(edges2) - 1
    i1 = np.clip(np.digitize(v1, edges1[1:-1]), 0, len(edges1) - 2)
    i2 = np.clip(np.digitize(v2, edges2[1:-1]), 0, n2 - 1)
    out = i1 * n2 + i2
    return np.where(np.isfinite(v1) & np.isfinite(v2), out, -1).astype(np.int64)


class _CellIndex:
    """CSR-style membership table: `members[start[c]:start[c+1]]` are pool rows in cell c."""

    def __init__(self, cell_of_pool, n_cells):
        c = np.asarray(cell_of_pool, dtype=np.int64)
        self.n_cells = int(n_cells)
        self.count = np.bincount(c[c >= 0], minlength=n_cells).astype(np.int64)
        self.start = np.concatenate([[0], np.cumsum(self.count)])
        order = np.argsort(np.where(c >= 0, c, n_cells), kind="stable")
        self.members = order[:self.start[-1]].astype(np.int64)
        self.frac = self.count / max(len(c), 1)


def pool_shard_indices(n_pool, shard, n_shards, seed=4242):
    """Sorted indices of one disjoint 1/`n_shards` shard of a pool of `n_pool` rows.

    `shard` is 0-based.  The shards partition the pool exactly: their union is every index,
    their pairwise intersections are empty, and sizes differ by at most one row.

    Halves answer "does WHICH half matter".  Quarters exist for a different reason: the
    irreducible finite-catalogue term `a` is read off the shard difference under the
    assumption that it scales as 1/N_pool, and a single shard size cannot test that
    assumption.  Two shard sizes give two points and turn it into a measurement.
    """
    n_pool, n_shards, shard = int(n_pool), int(n_shards), int(shard)
    if n_shards < 2:
        raise ValueError(f"n_shards must be >= 2; got {n_shards}")
    if not 0 <= shard < n_shards:
        raise ValueError(f"shard must lie in [0,{n_shards}); got {shard}")
    if n_pool < n_shards:
        raise ValueError(f"cannot split {n_pool} row(s) into {n_shards} shards")
    perm = np.random.default_rng(seed).permutation(n_pool)
    return np.sort(np.array_split(perm, n_shards)[shard])


def pool_half_indices(n_pool, half, seed=4242):
    """Sorted indices of one disjoint half of a draw pool of `n_pool` rows.

    `half` is `"A"`, `"B"`, or `"none"` (which returns `None`, meaning no restriction).
    A and B are EXACT COMPLEMENTS: their union is every index and their intersection is
    empty, so a run on each asks whether the answer depends on which half of the catalogue
    supplies the prior, holding the scored rows fixed.

    `seed` is deliberately separate from the draw seed.  The two arms must share their
    draw randomness for the difference to be paired, so the split has to stay put when the
    draw seed moves; folding them together would silently unpair the comparison.

    On an odd `n_pool` the extra row goes to A, so `len(A) + len(B) == n_pool` always.
    """
    if half == "none":
        return None
    if half not in ("A", "B"):
        raise ValueError(f"half must be 'A', 'B', or 'none'; got {half!r}")
    n_pool = int(n_pool)
    if n_pool < 2:
        raise ValueError(f"cannot halve a pool of {n_pool} row(s)")
    return pool_shard_indices(n_pool, 0 if half == "A" else 1, 2, seed)


def draw_indices(n_rows, m_draws, pool_size, rng, cell_of_row=None,
                 cell_of_pool=None, n_cells=0, beta=0.0):
    """`(idx (N,M) int64, log_w (N,M) float32)` for the marginalisation.

    `beta = 0` (or no strata) gives the plain uniform draw with unit weights, which is the
    empirical scene prior sampled directly.  With `beta > 0` the proposal is the defensive
    mixture described in the module docstring and the returned weights make the target
    uniform-over-pool again.  Rows whose cell is empty in the pool fall back to `beta = 0`
    individually rather than being dropped.
    """
    n_rows, m_draws, pool_size = int(n_rows), int(m_draws), int(pool_size)
    if beta <= 0.0 or cell_of_row is None or cell_of_pool is None:
        idx = rng.integers(0, pool_size, size=(n_rows, m_draws), dtype=np.int64)
        return idx, np.zeros((n_rows, m_draws), dtype=np.float32)
    if not (0.0 < beta < 1.0):
        raise ValueError("beta must lie strictly inside (0,1); w = 1/(1-beta) at beta=1")

    ci = _CellIndex(cell_of_pool, n_cells)
    cr = np.asarray(cell_of_row, dtype=np.int64)
    # a galaxy whose measured cell holds no pool row has no stratum to draw from
    ok = (cr >= 0) & (ci.count[np.clip(cr, 0, ci.n_cells - 1)] > 0)
    beta_i = np.where(ok, beta, 0.0)

    take_cell = rng.random((n_rows, m_draws)) < beta_i[:, None]
    idx = rng.integers(0, pool_size, size=(n_rows, m_draws), dtype=np.int64)
    if take_cell.any():
        rr, mm = np.nonzero(take_cell)
        c = cr[rr]
        off = (rng.random(len(rr)) * ci.count[c]).astype(np.int64)
        idx[rr, mm] = ci.members[ci.start[c] + np.minimum(off, ci.count[c] - 1)]

    # w = 1 / (beta * 1[same cell] / f_cell + (1-beta)); computed for EVERY drawn atom,
    # including the uniform ones that happen to land in the galaxy's own cell -- the
    # mixture density is what it is regardless of which component produced the draw.
    same = np.asarray(cell_of_pool, dtype=np.int64)[idx] == cr[:, None]
    f = np.where(ok, ci.frac[np.clip(cr, 0, ci.n_cells - 1)], 1.0)[:, None]
    q_ratio = beta_i[:, None] * same / np.maximum(f, 1e-12) + (1.0 - beta_i[:, None])
    return idx, (-np.log(np.maximum(q_ratio, 1e-30))).astype(np.float32)


# --------------------------------------------------------------------------------------
# the marginal score pass
# --------------------------------------------------------------------------------------

def standardized_pool(est, frame, feature_names):
    """`(pool (P,F) float32 standardized, feat_idx)` straight off the trained preprocessor.

    Values are pulled from the SAME rescaled frame the scoring pass uses, so whatever
    `rescale()` did to a column is already baked in and cannot drift between the pool and
    the conditioning it replaces.  Standardized rather than raw for the same reason: it is
    the representation `_grid_tiled_context` writes into.
    """
    pre = est.bundle.condition_preprocessor
    feats = list(pre.feature_names)
    missing = [n for n in feature_names if n not in feats]
    if missing:
        raise KeyError(f"feature(s) {missing} not in the flow's conditioning set {feats}")
    jj = [feats.index(n) for n in feature_names]
    if set(jj) & {est.i1, est.i2}:
        raise ValueError("refusing to marginalise the shape dims: they ARE the grid")
    ctx = np.asarray(pre.transform_frame(frame), dtype=np.float32)   # (P,D)
    return ctx[:, jj].copy(), jj


@torch.no_grad()
def marginal_score_pass(est, nodes, frame, ehat_raw, pool_std, feat_idx, draw_idx,
                        log_w, ladder, chunk=512, tag="", include_pin=True,
                        progress_every=20):
    """`(s_i, I_i)` per ladder point, from ONE sweep over the draws.

    Returns `(results, ess, ess_rung)`:
      * `results`: dict keyed by `"pin"` (the conditional likelihood at the row's own true
        properties -- the existing closure estimator, evaluated on the same rows and the
        same `xhat`, so the comparison is exactly paired) and by each integer `M` in
        `ladder`, each holding `(s (N,2), info (N,2,2))`.
      * `ess`: `(N,)` effective sample size of the importance weights for the EVIDENCE at
        the full draw count -- the number that says whether the proposal is working.
      * `ess_rung`: `(N, len(ladder))` the same quantity at EVERY rung, column order
        following `ladder`.  Needed because the finite-draw error is governed by the
        effective number of scenes carrying weight rather than by `M`, and those differ:
        measured `ESS ~ M^0.90` (nbr, beta=0) down to `M^0.29` (struct, beta=0.8).

    Prefixes are nested, so the whole ladder costs `max(ladder)` flow evaluations rather
    than their sum, and consecutive points are paired.  Nothing of size `N x G` is held.
    """
    model = est.bundle.model
    tstd = est.bundle.target_transform
    pre = est.bundle.condition_preprocessor
    dev = est.device
    ladder = sorted({int(v) for v in ladder})
    m_max = max(ladder)
    if draw_idx.shape[1] < m_max:
        raise ValueError(f"draw_idx supplies {draw_idx.shape[1]} draws, ladder needs {m_max}")

    n = len(frame)
    ehat_std = tstd.transform_array(np.asarray(ehat_raw, dtype=np.float32))
    pool_t = torch.as_tensor(np.ascontiguousarray(pool_std), dtype=torch.float32, device=dev)
    idx_t = torch.as_tensor(np.ascontiguousarray(draw_idx[:, :m_max]), device=dev)
    lw_t = torch.as_tensor(np.ascontiguousarray(log_w[:, :m_max]), dtype=torch.float32,
                           device=dev)
    lp_t = torch.as_tensor(np.where(nodes.support, nodes.log_prior, -np.inf),
                           dtype=torch.float32, device=dev)                     # (G,)

    keys = (["pin"] if include_pin else []) + ladder
    out = {k: (np.empty((n, 2)), np.empty((n, 2, 2))) for k in keys}
    ess = np.empty(n)
    ess_rung = np.empty((n, len(ladder)))
    t0 = __import__("time").time()
    done = 0
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        base = est._grid_tiled_context(frame.iloc[start:stop])                  # (B,G,D)
        b = base.shape[0]
        xh = torch.as_tensor(ehat_std[start:stop], dtype=torch.float32, device=dev)
        x = xh[:, None, :].expand(b, est.G, est.target_dim).reshape(b * est.G, est.target_dim)

        if include_pin:
            flat = base.view(b * est.G, -1)
            ll = model.flow.log_prob(x - model._mu(flat), model._flow_ctx(flat)).view(b, est.G)
            s, i, _ = scores_from_loglike(_snap(ll), nodes, device=dev)
            out["pin"][0][start:stop], out["pin"][1][start:stop] = s, i

        acc = torch.full((b, est.G), float("-inf"), device=dev)
        lw_run = torch.full((b,), float("-inf"), device=dev)
        ev = torch.empty((b, m_max), device=dev)      # per-draw log evidence, for the ESS
        for m in range(m_max):
            rep = base.clone()
            vals = pool_t[idx_t[start:stop, m]]                                 # (B,F)
            for f, j in enumerate(feat_idx):
                rep[:, :, j] = vals[:, f:f + 1]
                if pre.add_missing_indicators:
                    rep[:, :, est.n_feat + j] = 0.0
            flat = rep.view(b * est.G, -1)
            ll = model.flow.log_prob(x - model._mu(flat), model._flow_ctx(flat)).view(b, est.G)
            lw = lw_t[start:stop, m]
            acc = torch.logaddexp(acc, ll + lw[:, None])
            lw_run = torch.logaddexp(lw_run, lw)
            ev[:, m] = lw + torch.logsumexp(ll + lp_t[None, :], dim=1)
            if (m + 1) in out:
                s, i, _ = scores_from_loglike(_snap(acc - lw_run[:, None]), nodes, device=dev)
                out[m + 1][0][start:stop], out[m + 1][1][start:stop] = s, i
                # ESS OF THE PREFIX, not only of the full draw set.  The finite-draw error
                # is governed by the effective number of scenes actually carrying weight,
                # and that is NOT M: at beta=0 the proposal weights are uniform yet the
                # measured ESS/M is 61% (nbr) and 9% (struct), because this ESS reflects how
                # much the drawn scenes differ in LIKELIHOOD.  Recording it per rung is what
                # lets the ladder be refitted against 1/ESS instead of 1/M.
                # NOT named `pre`: that is the condition preprocessor, live in this scope
                # and read on every draw a few lines above.  Shadowing it made three GPU
                # jobs die 21s in with 'Tensor' object has no attribute
                # 'add_missing_indicators'.
                ev_pre = ev[:, :m + 1]
                ess_rung[start:stop, ladder.index(m + 1)] = torch.exp(
                    2.0 * torch.logsumexp(ev_pre, dim=1)
                    - torch.logsumexp(2.0 * ev_pre, dim=1)).cpu().numpy()
        # ESS of the self-normalised weights for the evidence: (sum c)^2 / sum c^2, in logs
        e2 = torch.logsumexp(2.0 * ev, dim=1)
        ess[start:stop] = torch.exp(2.0 * torch.logsumexp(ev, dim=1) - e2).cpu().numpy()

        done = stop
        if progress_every and (start // chunk) % progress_every == 0 or stop == n:
            el = __import__("time").time() - t0
            print(f"  [{tag}] {done:,}/{n:,}  {el:.0f}s  ETA {el / done * (n - done):.0f}s",
                  flush=True)
    return out, ess, ess_rung


def _snap(ll):
    """Row-max-shifted float32 numpy copy of a `(B,G)` device tensor.

    The shift is exact for everything downstream (softmax over the grid is shift
    invariant, and `scores_from_loglike` reweights only), and it is what keeps a marginal
    log-likelihood -- which is NOT row-max normalised the way `log_likelihood`'s output is
    -- inside float32 after `M` logaddexps.
    """
    return (ll - ll.max(dim=1, keepdim=True).values).cpu().numpy().astype(np.float32)
