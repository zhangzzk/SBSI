"""The blocked jackknife that replaces the Fisher error bar once objects are paired.

`1/sqrt(sum_i I_i)` is the Cramer-Rao bound for INDEPENDENT objects.  `eval_score_select.py`
now draws true shapes in 90-degree-rotated ring pairs on purpose, so its objects are
anticorrelated and that bound is no longer the estimator's variance -- it overstates it by
exactly the variance reduction the pairing bought, which is the number we are trying to
measure.  These tests pin down the three properties the replacement must have:

  1. on unpaired rows it REPRODUCES the Fisher bar (so it is not simply optimistic),
  2. on ring-paired rows it reports a genuinely smaller error, and that error matches the
     scatter of repeated independent realisations (the only ground truth for an error bar),
  3. differencing two runs' replicates block by block gives the error on the DIFFERENCE,
     which is smaller than either error bar when the two share their galaxies.

The toy is A.7's shift family again (`s = y/nu^2`, `I = 1/nu^2`), where the closed forms are
known, plus an explicit `e -> -e` ring so the cancellation is real rather than asserted.
"""

import numpy as np
import pytest

from sbsi.score_inference import (
    blocked_sums,
    full_shear_estimate,
    jackknife_blocks,
    jackknife_shear,
    jackknife_sigma,
)

TAU, SIGMA = 0.30, 0.10  # intrinsic shape scatter and measurement noise
NU2 = TAU**2 + SIGMA**2
NBLOCK = 100


def draw(n_pair, g, rng, ring):
    """A.7-shaped `(s, I, block)`: `s = y/nu^2`, `I = 1/nu^2`, optionally ring-paired.

    `y = e + g + noise` with `e ~ N(0, tau^2)` per component.  With `ring=True` each drawn
    `e` appears twice, as `+e` and `-e`, so the intrinsic term cancels within the pair and
    only `g` (plus the measurement noise) survives -- the toy analogue of the 90-degree
    rotation, whose whole point is that it leaves the signal alone.
    """
    if ring:
        e = TAU * rng.standard_normal((n_pair, 2))
        e = np.concatenate([e, -e], axis=0)
        pair = np.concatenate([np.arange(n_pair), np.arange(n_pair)])
    else:
        e = TAU * rng.standard_normal((2 * n_pair, 2))
        pair = np.arange(2 * n_pair)
    y = e + np.array([g, 0.0]) + SIGMA * rng.standard_normal(e.shape)
    s = y / NU2
    info = np.tile((np.eye(2) / NU2)[None], (len(y), 1, 1))
    return s, info, pair % NBLOCK


def test_jackknife_from_cached_block_sums_is_the_same_estimator():
    """The cached path must be the row path, exactly -- not merely close.

    `eval_score_select.py` now caches `blocked_sums` so the population block can be
    re-estimated without a fresh multi-hour score pass.  That is only safe if dropping the
    rows changes nothing, so compare central value, error bar AND the per-block
    replicates, with and without a population correction.
    """
    rng = np.random.default_rng(11)
    s, info, block = draw(4000, 0.02, rng, ring=True)
    s_sel = np.array([-0.002, 0.011])
    i_sel = np.array([[1.01, 0.008], [0.008, 1.02]])
    for ss, ii in ((None, None), (s_sel, None), (s_sel, i_sel)):
        rows = jackknife_shear(s, info, block, NBLOCK, ss, ii)
        cached = jackknife_blocks(*blocked_sums(s, info, block, NBLOCK), ss, ii)
        for a, b in zip(rows, cached):
            assert np.array_equal(a, b)


def test_blocked_sums_reproduce_the_totals():
    rng = np.random.default_rng(0)
    s, info, block = draw(500, 0.02, rng, ring=False)
    cnt, ns, ni = blocked_sums(s, info, block, NBLOCK)
    assert cnt.sum() == len(s)
    assert np.allclose(ns.sum(axis=0), s.sum(axis=0))
    assert np.allclose(ni.sum(axis=0), info.sum(axis=0))


def test_blocked_sums_rejects_out_of_range_ids():
    rng = np.random.default_rng(0)
    s, info, block = draw(10, 0.02, rng, ring=False)
    with pytest.raises(ValueError):
        blocked_sums(s, info, block, 5)
    with pytest.raises(ValueError):
        blocked_sums(s, info, block[:3], NBLOCK)


def test_central_value_matches_full_shear_estimate():
    """The jackknife must not perturb the estimate itself, only bar it."""
    rng = np.random.default_rng(1)
    s, info, block = draw(2000, 0.02, rng, ring=False)
    s_sel, i_sel = np.array([0.01, 0.0]), 0.1 * np.eye(2)
    for ss, ii in ((None, None), (s_sel, None), (s_sel, i_sel)):
        gh_j, _, _ = jackknife_shear(s, info, block, NBLOCK, ss, ii)
        gh_f, _, _ = full_shear_estimate(s, info, ss, ii)
        assert np.allclose(gh_j, gh_f, rtol=1e-12, atol=1e-14)


def test_unpaired_jackknife_reproduces_the_fisher_bar():
    """With independent objects the jackknife must agree with `1/sqrt(sum I)`.

    This is the test that stops the new error bar from being merely smaller: where the
    Cramer-Rao bound IS the answer, the jackknife has to return it.

    Averaged over realisations, because a single jackknife bar built from `B` blocks is
    itself only determined to about `1/sqrt(2(B-1))` -- 7% at `B = 100`, which is larger
    than the agreement worth asserting.
    """
    bars, fishers = [], []
    for k in range(12):
        rng = np.random.default_rng(200 + k)
        s, info, block = draw(20_000, 0.02, rng, ring=False)
        _, sig, _ = jackknife_shear(s, info, block, NBLOCK)
        _, _, den = full_shear_estimate(s, info)
        bars.append(sig[0])
        fishers.append(1.0 / np.sqrt(den[0, 0]))
    assert float(np.mean(bars)) == pytest.approx(float(np.mean(fishers)), rel=0.04)


def test_ring_pairing_beats_the_fisher_bar():
    """Ring pairs cancel the intrinsic term, so the true error falls well below Cramer-Rao.

    The expected factor is `nu/sigma = sqrt(tau^2+sigma^2)/sigma`, about 3.2 here: what
    survives a pair average is the measurement noise alone.
    """
    rng = np.random.default_rng(3)
    s, info, block = draw(20_000, 0.02, rng, ring=True)
    _, sig, _ = jackknife_shear(s, info, block, NBLOCK)
    _, _, den = full_shear_estimate(s, info)
    fisher = 1.0 / np.sqrt(den[0, 0])
    assert sig[0] < 0.5 * fisher
    assert fisher / sig[0] == pytest.approx(np.sqrt(NU2) / SIGMA, rel=0.15)


@pytest.mark.parametrize("ring", [False, True])
def test_jackknife_matches_the_scatter_of_repeated_realisations(ring):
    """The only honest check of an error bar: does it predict the spread of repeats?

    40 independent catalogues, each barred by the jackknife; the mean bar must match the
    realised standard deviation across them.  Run for BOTH pairings, so the paired case is
    verified rather than assumed.
    """
    ghats, bars = [], []
    for k in range(40):
        rng = np.random.default_rng(100 + k)
        s, info, block = draw(4000, 0.02, rng, ring=ring)
        gh, sig, _ = jackknife_shear(s, info, block, NBLOCK)
        ghats.append(gh[0])
        bars.append(sig[0])
    realised = float(np.std(ghats, ddof=1))
    predicted = float(np.mean(bars))
    assert predicted == pytest.approx(realised, rel=0.25)


def test_paired_difference_is_tighter_than_either_bar():
    """Two estimates sharing galaxies: the block-by-block difference kills the common noise.

    `eval_score_select.py` asks exactly this question -- does the corrected CUT estimate sit
    where the UNCUT one does -- and the two run on the same rows.  Differencing replicates
    within a block is what makes that comparison sharp; adding the two bars in quadrature
    would instead be strictly worse than either.
    """
    rng = np.random.default_rng(7)
    s, info, block = draw(20_000, 0.02, rng, ring=False)
    keep = np.hypot(s[:, 0], s[:, 1]) * NU2 < 0.55  # a cut on the same objects
    _, sig_a, reps_a = jackknife_shear(s, info, block, NBLOCK)
    _, sig_b, reps_b = jackknife_shear(s[keep], info[keep], block[keep], NBLOCK)
    d_sig = float(jackknife_sigma((reps_b[:, 0] - reps_a[:, 0])[:, None])[0])
    assert d_sig < min(sig_a[0], sig_b[0])
    assert d_sig < 0.7 * np.hypot(sig_a[0], sig_b[0])


def test_paired_difference_hits_its_analytic_size_for_a_random_subset():
    """Pin the pairing gain to a number, not to an inequality.

    For a subset chosen at RANDOM with keep fraction `f`, the two estimators are means over
    a set and a subset, so `Cov = Var_full` exactly and
        `Var(sub - full) = V/N_sub - V/N = Var_full (1/f - 1)`,
    i.e. the difference bar is `sqrt(1/f - 1)` times the full-sample bar -- 0.53 at
    `f = 0.78`, a factor 1.9 better than the full bar and 2.9 better than adding the two
    bars in quadrature.  A magnitude cut does worse than this (it preferentially removes
    the highest-weight objects), which is why the inequality test above is loose; this
    test is the one that says the machinery is right.

    Averaged over realisations: one draw determines this ratio only to about 10%, so a
    single-seed assertion at the 15% level fails roughly a third of the time on noise.
    """
    f, ratios = 0.78, []
    for k in range(15):
        rng = np.random.default_rng(900 + k)
        s, info, block = draw(40_000, 0.02, rng, ring=False)
        keep = rng.random(len(s)) < f
        _, sig_a, reps_a = jackknife_shear(s, info, block, NBLOCK)
        _, _, reps_b = jackknife_shear(s[keep], info[keep], block[keep], NBLOCK)
        d_sig = float(jackknife_sigma((reps_b[:, 0] - reps_a[:, 0])[:, None])[0])
        ratios.append(d_sig / sig_a[0])
    assert float(np.mean(ratios)) == pytest.approx(np.sqrt(1.0 / f - 1.0), rel=0.08)


def test_jackknife_sigma_needs_two_replicates():
    assert np.all(np.isnan(jackknife_sigma(np.zeros((1, 2)))))


def test_empty_blocks_keep_the_replicate_array_aligned():
    """Replicates must be one per block even when a block is empty, or pairing misaligns."""
    rng = np.random.default_rng(11)
    s, info, block = draw(500, 0.02, rng, ring=False)
    sub = block < NBLOCK - 10  # last 10 blocks empty
    _, _, reps = jackknife_shear(s[sub], info[sub], block[sub], NBLOCK)
    assert len(reps) == NBLOCK
    assert np.allclose(reps[-1], reps[-2])  # both delete nothing
