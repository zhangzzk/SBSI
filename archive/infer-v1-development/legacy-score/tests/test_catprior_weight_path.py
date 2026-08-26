"""Archived tests for the legacy beta-weight path.

`eval_score_catprior.py`'s `--self-pool-check` passes `np.zeros(...)` as `log_w`, so it
certifies the marginal path only where every weight is one.  Everything below is about the
weighted path, and it exists because the beta=0.8 ladders came back non-monotone
(d(m) = +0.03, +7.35, -0.52, -2.56, ... % at M = 1, 2, 4, 8) with chi2/dof = 10.8 -- a
shape no 1/M law can take.

The headline here is `test_at_M1_the_weights_cancel_identically`: self-normalised
importance sampling at a single draw divides the weight straight back out, so the M=1 rung
of a weighted ladder estimates the PROPOSAL rather than the target.  That is not a small-M
inaccuracy that a 1/M fit absorbs -- it is a different estimator sitting on the same axis.
"""

import numpy as np
import pytest

from sbsi.catalogue_prior import draw_indices


def snapshot(ll, lw, ladder):
    """The exact arithmetic `marginal_score_pass` performs, on log-likelihoods directly.

    Mirrors the accumulator there: `acc = logaddexp(acc, ll + lw)`, `lw_run =
    logaddexp(lw_run, lw)`, and the rung-M value is `acc - lw_run`.  Working on `ll`
    instead of on scores keeps this a test of the WEIGHTING, not of the flow.
    """
    acc = np.full(ll.shape[0], -np.inf)
    lw_run = np.full(ll.shape[0], -np.inf)
    out = {}
    for m in range(ll.shape[1]):
        acc = np.logaddexp(acc, ll[:, m] + lw[:, m])
        lw_run = np.logaddexp(lw_run, lw[:, m])
        if (m + 1) in ladder:
            out[m + 1] = acc - lw_run
    return out


def test_at_M1_the_weights_cancel_identically():
    """M=1 returns `ll` whatever the weight is -- so it is not on the same ladder."""
    rng = np.random.default_rng(0)
    ll = rng.normal(size=(500, 4))
    # `acc - lw_run` is `(ll + lw) - lw`, which cancels ALGEBRAICALLY but not bit-exactly:
    # adding a large `lw` to `ll` and subtracting it again costs the low bits of `ll`.  The
    # point of the test is the cancellation, so the tolerance tracks that round-off rather
    # than pretending it is absent.  Production runs this in float32, where it is coarser.
    for scale in (0.0, 1.0, 12.0, -7.0):
        lw = scale * rng.normal(size=(500, 4))
        got = snapshot(ll, lw, {1})[1]
        tol = 1e-12 * max(abs(scale), 1.0)
        assert np.allclose(got, ll[:, 0], atol=tol, rtol=0), \
            "M=1 must reduce to the bare log-likelihood of the single drawn atom"
    # and at zero weight the cancellation IS exact, which fixes the arithmetic above
    got0 = snapshot(ll, np.zeros_like(ll), {1})[1]
    assert np.array_equal(got0, ll[:, 0])


def test_a_constant_weight_changes_nothing_at_any_rung():
    """Self-normalisation is invariant to a constant shift of `log_w`; a leak breaks it."""
    rng = np.random.default_rng(1)
    ll = rng.normal(size=(300, 8))
    lw = rng.normal(size=(300, 8))
    ladder = {1, 2, 4, 8}
    base = snapshot(ll, lw, ladder)
    for c in (-5.0, 0.5, 9.0):
        shifted = snapshot(ll, lw + c, ladder)
        for m in ladder:
            assert np.allclose(base[m], shifted[m], atol=1e-10)


def test_duplicated_atoms_reproduce_the_single_atom_answer():
    """M identical draws must give the M=1 answer for that atom, for ANY weights."""
    rng = np.random.default_rng(2)
    one = rng.normal(size=(200, 1))
    ll = np.repeat(one, 8, axis=1)
    lw = rng.normal(size=(200, 8)) * 3.0
    got = snapshot(ll, lw, {1, 2, 4, 8})
    for m in (1, 2, 4, 8):
        assert np.allclose(got[m], one[:, 0], atol=1e-9)


@pytest.mark.parametrize("beta,ncell", [(0.8, 64), (0.5, 16), (0.2, 9)])
def test_the_drawn_weights_match_the_defensive_mixture_density(beta, ncell):
    """`w = 1/(beta/f + 1-beta)` in-cell and `1/(1-beta)` out, as the mixture requires."""
    rng = np.random.default_rng(3)
    # 20,160 is divisible by 64, 16 and 9, so every cell holds exactly the same count and
    # the EMPIRICAL fraction the code uses really is 1/ncell.  With a pool that does not
    # divide evenly the cells differ by one row and `f` takes two nearby values -- correct
    # behaviour, but it would make the closed form below wrong for the wrong reason.
    pool = 20160
    cell_of_pool = np.arange(pool) % ncell          # exactly equal cells, so f = 1/ncell
    cell_of_row = rng.integers(0, ncell, size=4000)
    idx, logw = draw_indices(len(cell_of_row), 6, pool, rng,
                             cell_of_row=cell_of_row, cell_of_pool=cell_of_pool,
                             n_cells=ncell, beta=beta)
    w = np.exp(logw)
    same = cell_of_pool[idx] == cell_of_row[:, None]
    f = 1.0 / ncell
    assert np.allclose(w[same], 1.0 / (beta / f + 1 - beta), rtol=1e-5)
    assert np.allclose(w[~same], 1.0 / (1 - beta), rtol=1e-5)
    # `logw` is stored float32, so `exp(-log(0.2))` comes back as 5.0000005 rather than 5.
    # The bound is a design guarantee about the mixture, not about float32, so it is
    # checked to single precision.
    assert w.max() <= (1.0 / (1 - beta)) * (1 + 1e-6), \
        "the bound w <= 1/(1-beta) is the whole point of the defensive mixture"


def test_the_own_cell_share_of_the_TARGET_is_f_regardless_of_beta():
    """A sanity anchor on what beta can and cannot do.

    Importance sampling re-weights to uniform-over-pool, so the row's own cell contributes
    its pool fraction `f` to the ANSWER at every beta.  What beta buys is variance, via the
    likelihood concentration: oversampling the own cell pays off exactly when the own-cell
    likelihood is large enough that `f * L_hi` is a big share of the integral.  This test
    exists to stop the earlier wrong reading -- that a small own-cell share of the
    NORMALISER meant beta was useless -- from coming back.
    """
    beta, ncell = 0.8, 64
    f = 1.0 / ncell
    w_same, w_diff = 1.0 / (beta / f + 1 - beta), 1.0 / (1 - beta)
    p_same = beta + (1 - beta) * f
    share_norm = p_same * w_same / (p_same * w_same + (1 - p_same) * w_diff)
    assert share_norm == pytest.approx(f, rel=1e-6)
    # the proposal is optimal when p_same matches f*L_hi / (f*L_hi + (1-f)*L_lo)
    ratio_needed = (p_same / (1 - p_same)) * (1 - f) / f
    assert 200 < ratio_needed < 320       # beta=0.8/64 cells suits L_hi/L_lo ~ 257


def test_the_all_same_cell_regime_persists_to_M_of_order_ten():
    """How deep the M=1 pathology reaches: while every atom is own-cell, weights cancel.

    With P(atom in own cell) = 0.803 at beta=0.8, a majority of rows have ALL atoms in
    their own cell out to M=4, so those rows are still running the proposal-estimator
    rather than the target one.  The ladder cannot be read as 1/M until this is rare.
    """
    beta, ncell = 0.8, 64
    p = beta + (1 - beta) / ncell
    frac = {M: p ** M for M in (1, 2, 4, 8, 16, 32)}
    assert frac[1] > 0.8 and frac[2] > 0.6 and frac[4] > 0.4
    assert frac[16] < 0.05 and frac[32] < 0.01


def test_the_draw_loop_does_not_shadow_names_it_reads():
    """A static guard on `marginal_score_pass`, paid for by three dead GPU jobs.

    The per-rung ESS patch bound `pre = ev[:, :m+1]` inside the draw loop, shadowing
    `pre = est.bundle.condition_preprocessor`, which the same loop reads on every draw.
    Nothing in `tests/` runs `marginal_score_pass` against a real flow bundle, so the whole
    suite stayed green and three jobs died 21 seconds in with `'Tensor' object has no
    attribute 'add_missing_indicators'`.

    A full smoke test would need a stub flow, a stub preprocessor and a real node bank; this
    checks the one property that actually failed -- that a name bound once before the loop
    is never rebound inside it -- and costs no GPU and no fixtures.
    """
    import ast
    import inspect

    import sbsi.catalogue_prior as cp

    tree = ast.parse(inspect.getsource(cp))
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "marginal_score_pass")
    loops = [n for n in ast.walk(fn)
             if isinstance(n, ast.For) and isinstance(n.target, ast.Name)
             and n.target.id == "m"]
    assert loops, "expected a `for m in range(m_max)` draw loop to guard"
    loop = loops[0]

    def bound(node):
        out = set()
        for n in ast.walk(node):
            if isinstance(n, ast.Assign):
                out |= {t.id for t in n.targets if isinstance(t, ast.Name)}
                for t in n.targets:                       # tuple unpacking
                    if isinstance(t, ast.Tuple):
                        out |= {e.id for e in t.elts if isinstance(e, ast.Name)}
        return out

    inside = bound(loop)
    # "before" means EARLIER IN THE SOURCE, by line -- not `fn.body`, because the draw loop
    # is nested inside the chunk loop, so iterating `fn.body` would count the draw loop's
    # own bindings as prior ones and the guard would flag every local it introduces.
    before = set()
    for n in ast.walk(fn):
        if isinstance(n, ast.Assign) and n.lineno < loop.lineno:
            before |= {t.id for t in n.targets if isinstance(t, ast.Name)}
            for t in n.targets:
                if isinstance(t, ast.Tuple):
                    before |= {e.id for e in t.elts if isinstance(e, ast.Name)}
    read_in_loop = {n.id for n in ast.walk(loop)
                    if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    # a name set up before the loop, read inside it, and ALSO reassigned inside it
    clash = (before & inside & read_in_loop) - {"acc", "lw_run", "ev", "rep", "flat", "ll",
                                                "lw", "s", "i", "vals", "x", "b"}
    assert not clash, (
        f"the draw loop rebinds {sorted(clash)}, which it also reads from the setup above "
        f"it -- this is the shadowing that killed jobs 15844986/87/88")
