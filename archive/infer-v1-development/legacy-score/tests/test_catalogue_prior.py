"""Archived catalogue-drawn conditioning-prior tests.

The one property everything rests on is that the MIXTURE PROPOSAL LEAVES THE TARGET
ALONE.  Drawing preferentially from a galaxy's own measured cell is only legitimate if the
weights put the estimator back on the uniform-over-pool distribution, so that is tested
directly and as an EXACT identity rather than a tolerance: the unnormalised estimator
`(1/M) sum_m w_im f(theta_im)` has expectation `mean_j f_j` for any `f`, with no `O(1/M)`
bias term to hide behind.  (The self-normalised form used in the likelihood does carry an
`O(1/M)` bias -- that is the thing `eval_score_catprior.py` measures, not a defect.)
"""

import numpy as np
import pytest

from sbsi.catalogue_prior import _CellIndex, cell_index, draw_indices, quantile_edges


def test_quantile_edges_are_open_and_increasing():
    v = np.random.default_rng(0).normal(size=10_000)
    e = quantile_edges(v, 8)
    assert e[0] == -np.inf and e[-1] == np.inf
    assert np.all(np.diff(e) > 0)
    assert len(e) == 9
    # interior edges are equal-count quantiles, so the bins are near-equally populated
    counts = np.histogram(v, bins=np.r_[v.min() - 1, e[1:-1], v.max() + 1])[0]
    assert counts.max() / counts.min() < 1.2


def test_quantile_edges_survive_ties_and_nans():
    v = np.r_[np.zeros(500), np.ones(500), np.full(10, np.nan)]
    e = quantile_edges(v, 8)              # np.unique collapses the degenerate quantiles
    assert np.all(np.diff(e) > 0) and e[0] == -np.inf and e[-1] == np.inf


def test_cell_index_ranges_and_nan_handling():
    rng = np.random.default_rng(1)
    a, b = rng.normal(size=1000), rng.normal(size=1000)
    a[:7] = np.nan
    e1, e2 = quantile_edges(a, 4), quantile_edges(b, 5)
    c = cell_index(a, b, e1, e2)
    assert (c[:7] == -1).all()
    n_cells = (len(e1) - 1) * (len(e2) - 1)
    assert c[7:].min() >= 0 and c[7:].max() < n_cells


def test_cell_membership_table_is_a_partition():
    rng = np.random.default_rng(2)
    cells = rng.integers(0, 12, size=500)
    cells[:20] = -1                                  # unassignable rows
    ci = _CellIndex(cells, 12)
    assert len(ci.members) == 480
    for c in range(12):
        got = ci.members[ci.start[c]:ci.start[c + 1]]
        assert (cells[got] == c).all()
        assert len(got) == ci.count[c]
    assert np.array_equal(np.sort(ci.members), np.sort(np.flatnonzero(cells >= 0)))


def test_uniform_draw_has_unit_weights():
    rng = np.random.default_rng(3)
    idx, lw = draw_indices(64, 8, 500, rng)
    assert idx.shape == (64, 8) and lw.shape == (64, 8)
    assert idx.min() >= 0 and idx.max() < 500
    assert np.all(lw == 0.0)


def test_beta_zero_and_no_strata_agree():
    a = draw_indices(32, 4, 100, np.random.default_rng(4))[0]
    b = draw_indices(32, 4, 100, np.random.default_rng(4), cell_of_row=None,
                     cell_of_pool=None, beta=0.0)[0]
    assert np.array_equal(a, b)


def test_beta_outside_the_open_unit_interval_is_refused():
    rng = np.random.default_rng(5)
    cells = np.zeros(10, dtype=np.int64)
    with pytest.raises(ValueError):
        draw_indices(10, 2, 10, rng, cell_of_row=cells, cell_of_pool=cells,
                     n_cells=1, beta=1.0)


def _stratified(n_rows, m, pool, beta, seed=7, n_cells=16):
    """A pool whose values correlate with its cell, and rows that ask for their own cell."""
    rng = np.random.default_rng(seed)
    cell_pool = rng.integers(0, n_cells, size=pool)
    f = cell_pool.astype(float) ** 2 + rng.normal(size=pool)      # strongly cell-dependent
    cell_row = rng.integers(0, n_cells, size=n_rows)
    idx, lw = draw_indices(n_rows, m, pool, rng, cell_of_row=cell_row,
                           cell_of_pool=cell_pool, n_cells=n_cells, beta=beta)
    return idx, np.exp(lw), f, cell_row, cell_pool


def test_importance_weights_are_bounded_by_one_over_one_minus_beta():
    for beta in (0.3, 0.8, 0.95):
        _, w, _, _, _ = _stratified(2000, 16, 3000, beta)
        # float32 log-weights, so the bound holds to that precision and not tighter
        assert w.max() <= (1.0 / (1.0 - beta)) * (1.0 + 1e-6)
        assert w.min() > 0.0


def test_mixture_proposal_is_unbiased_for_the_pool_mean():
    """`E[(1/M) sum_m w_im f(theta_im)] = mean_j f_j` -- the target is untouched.

    Unbiased EXACTLY, so this is a statistical test with a real error bar and not a
    tolerance dialled until it passes: the estimator's own scatter across rows gives the
    sigma, and the check is that the deviation is inside 4 of them.
    """
    for beta in (0.0, 0.5, 0.9):
        idx, w, f, _, _ = _stratified(20_000, 8, 5000, beta, seed=11)
        per_row = (w * f[idx]).mean(axis=1)
        est, sigma = per_row.mean(), per_row.std(ddof=1) / np.sqrt(len(per_row))
        assert abs(est - f.mean()) < 4 * sigma, (beta, est, f.mean(), sigma)


def test_weights_average_to_one():
    """`f == 1` in the identity above: the proposal integrates to the pool, exactly."""
    for beta in (0.4, 0.85):
        _, w, _, _, _ = _stratified(20_000, 8, 5000, beta, seed=12)
        per_row = w.mean(axis=1)
        assert abs(per_row.mean() - 1.0) < 4 * per_row.std(ddof=1) / np.sqrt(len(per_row))


def test_rows_with_an_empty_cell_fall_back_to_uniform():
    """A galaxy whose measured cell holds no pool row keeps unit weights, not a divide-by-zero."""
    rng = np.random.default_rng(13)
    # two populated cells, so stratifying on cell 0 is a REAL reweighting (f_0 = 0.75)
    # and the empty-cell fallback is not confounded with a degenerate single-cell pool
    pool_cells = np.r_[np.zeros(150), np.ones(50)].astype(np.int64)
    row_cells = np.r_[np.zeros(50), np.full(50, 3)].astype(np.int64)
    idx, lw = draw_indices(100, 6, 200, rng, cell_of_row=row_cells,
                           cell_of_pool=pool_cells, n_cells=8, beta=0.8)
    assert np.isfinite(lw).all()
    assert np.all(lw[50:] == 0.0)                          # empty cell -> beta_i = 0
    assert np.any(lw[:50] != 0.0)                          # populated cell -> reweighted


def test_stratification_concentrates_the_draws():
    """The proposal must actually do something: `beta` raises the in-cell hit rate."""
    _, _, _, cell_row, cell_pool = _stratified(4000, 16, 4000, 0.0, seed=14)
    idx0 = _stratified(4000, 16, 4000, 0.0, seed=14)[0]
    idx8 = _stratified(4000, 16, 4000, 0.8, seed=14)[0]
    hit = lambda i: float((cell_pool[i] == cell_row[:, None]).mean())
    assert hit(idx0) < 0.12          # ~1/16 by chance
    assert hit(idx8) > 0.75          # beta plus the chance hits
