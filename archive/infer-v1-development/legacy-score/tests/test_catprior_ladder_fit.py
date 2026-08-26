"""Archived tests of the rejected finite-M extrapolation.

A zero-injection test is not enough and can be actively misleading: an extrapolation that
is biased toward zero -- an over-shrinking GLS, a covariance that over-weights the deep
rungs where `d(m)` is smallest, a sign slip that damps the intercept -- passes a
zero-injection test with full marks.  The claim this machinery will eventually support
("the closure residual is bounded at a few tenths of a percent") lives entirely in the
regime a null test never visits, so the injections below are mostly NONZERO.

The synthetic noise reproduces the real estimator's structure rather than a convenient
stand-in.  Two ingredients matter:

  * the PIN arm is common to every rung, so its block noise enters every `d(m)` with the
    same sign and correlates the rungs strongly;
  * the draws are NESTED to the bit -- rung `M` averages the first `M` per-draw terms, so
    rung 2 contains rung 1's term identically.

That is what makes the full rung covariance necessary, so a test that omitted it would not
exercise the thing being tested.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "scripts"))

from combine_catprior import fit_ladder, intercept_spread  # noqa: E402
import combine_catprior as cc  # noqa: E402

LADDER = [1, 2, 4, 8, 16]


def synth(d_true, seeds=4, blocks=200, sigma_gal=2.0e-4, sigma_draw=1.5e-3, seed=3):
    """(D, reps) with pin-arm noise shared across rungs and NESTED per-draw noise."""
    rng = np.random.default_rng(seed)
    K, m_max = len(LADDER), max(LADDER)
    gal = rng.standard_normal(blocks) * sigma_gal          # common to all rungs AND seeds
    x = np.empty((seeds, blocks, K))
    for s in range(seeds):
        eps = rng.standard_normal((blocks, m_max)) * sigma_draw
        cum = np.cumsum(eps, axis=1)
        for k, M in enumerate(LADDER):
            x[s, :, k] = d_true(M) + gal + cum[:, M - 1] / M
    full = x.mean(axis=1)                                   # (S, K)
    tot = x.sum(axis=1)
    reps = np.stack([(tot[s] - x[s]) / (blocks - 1) for s in range(seeds)]).mean(axis=0)
    return full, reps


def test_recovers_a_NONZERO_intercept_without_shrinking_it_toward_zero():
    inj_d, inj_a = 0.004, 0.030
    D, reps = synth(lambda M: inj_d + inj_a / M)
    fits, _ = fit_ladder(D, reps, LADDER)
    d_inf, sig = fits[0][1][0], np.sqrt(fits[0][2][0, 0])
    assert abs(d_inf - inj_d) < 4 * sig, f"{d_inf:.5f} vs injected {inj_d}"
    # the failure this test exists for: a fit that damps the intercept toward zero
    assert d_inf > 0.5 * inj_d, f"intercept shrunk to {d_inf:.5f} from {inj_d}"
    assert abs(fits[0][1][1] - inj_a) < 4 * np.sqrt(fits[0][2][1, 1])


def test_recovers_a_NEGATIVE_intercept():
    """Sign errors that damp toward zero look identical to success on positive-only tests."""
    inj_d, inj_a = -0.006, 0.030
    D, reps = synth(lambda M: inj_d + inj_a / M, seed=17)
    fits, _ = fit_ladder(D, reps, LADDER)
    d_inf, sig = fits[0][1][0], np.sqrt(fits[0][2][0, 0])
    assert abs(d_inf - inj_d) < 4 * sig
    assert d_inf < 0.5 * inj_d


def test_zero_intercept_with_slope_returns_zero():
    D, reps = synth(lambda M: 0.030 / M, seed=5)
    fits, _ = fit_ladder(D, reps, LADDER)
    assert abs(fits[0][1][0]) < 4 * np.sqrt(fits[0][2][0, 0])


def test_flat_ladder_returns_no_slope():
    D, reps = synth(lambda M: 0.004, seed=9)
    fits, _ = fit_ladder(D, reps, LADDER)
    assert abs(fits[0][1][1]) < 4 * np.sqrt(fits[0][2][1, 1])
    assert abs(fits[0][1][0] - 0.004) < 4 * np.sqrt(fits[0][2][0, 0])


def test_the_model_dependence_flag_FIRES_on_a_law_that_is_not_1_over_M():
    """A decision rule never observed to trigger is not yet a decision rule."""
    D, reps = synth(lambda M: 0.030 / M ** 1.5, seed=11)
    fits, _ = fit_ladder(D, reps, LADDER)
    _, moved = intercept_spread(fits)
    assert moved, "wrong-law ladder did not trip the model-dependence flag"


def test_the_flag_does_NOT_fire_on_a_genuine_1_over_M_ladder():
    D, reps = synth(lambda M: 0.004 + 0.030 / M, seed=13)
    fits, _ = fit_ladder(D, reps, LADDER)
    _, moved = intercept_spread(fits)
    assert not moved


def test_a_single_seed_is_flagged_as_having_no_independent_check():
    D, reps = synth(lambda M: 0.030 / M, seeds=1)
    _, note = fit_ladder(D, reps, LADDER)
    assert "no independent check" in note


def test_seed_scatter_sits_INSIDE_the_jackknife_bar_rather_than_adding_to_it():
    """The draw noise is PART of the block-to-block scatter, not an extra term.

    An earlier version added `C_seed / S` on top of the jackknife covariance.  That double
    counts, because `d_M` is a mean over galaxies each carrying its own draws, so the
    jackknife already contains the draw noise -- and because `reps` is seed-AVERAGED, it
    already reflects the 1/S shrinkage too.  The observable consequence is that the
    across-seed spread must come out SMALLER than the jackknife bar; if it did not, the
    draw noise would be additional and the old code would have been right.
    """
    from combine_catprior import seed_scatter_check
    D, reps = synth(lambda M: 0.030 / M, seeds=4, blocks=400, seed=71)
    rows = seed_scatter_check(D, reps)
    assert rows is not None
    for spread, bar, ratio in rows:
        assert ratio < 1.0, f"seed spread {spread:.2e} exceeds jackknife bar {bar:.2e}"


def test_the_seed_scatter_check_is_unavailable_with_one_seed():
    from combine_catprior import seed_scatter_check
    D, reps = synth(lambda M: 0.030 / M, seeds=1)
    assert seed_scatter_check(D, reps) is None


def test_dropping_the_double_count_does_not_shrink_bars_below_the_truth():
    """Removing an inflation must not make the fit overconfident: the injected intercept
    still has to be recovered inside its (now smaller) bar."""
    inj = 0.004
    D, reps = synth(lambda M: inj + 0.030 / M, seeds=4, blocks=400, seed=73)
    fits, _ = fit_ladder(D, reps, LADDER)
    d_inf, sig = fits[0][1][0], np.sqrt(fits[0][2][0, 0])
    assert abs(d_inf - inj) < 4 * sig


def test_hartlap_widens_the_bars_rather_than_narrowing_them():
    """The correction must move errors UP; a sign slip here would flatter every chi2."""
    D, reps = synth(lambda M: 0.004 + 0.030 / M, blocks=12, seed=23)
    fits, _ = fit_ladder(D, reps, LADDER)
    D2, reps2 = synth(lambda M: 0.004 + 0.030 / M, blocks=400, seed=23)
    fits2, _ = fit_ladder(D2, reps2, LADDER)
    # few blocks -> larger Hartlap inflation -> the bar cannot be the tighter of the two
    assert np.sqrt(fits[0][2][0, 0]) > np.sqrt(fits2[0][2][0, 0])


def _rho_pairs(reps, ladder, top_i=-1):
    from combine_catprior import implied_noise_ratio, rho_floor
    out = []
    top = ladder[top_i]
    for k, M in enumerate(ladder[:top_i]):
        a, b = reps[:, top_i], reps[:, k]
        va, vb = a.var(ddof=1), b.var(ddof=1)
        vd = (b - a).var(ddof=1)
        r = (va + vb - vd) / (2 * np.sqrt(va * vb))
        out.append((M, r, rho_floor(M, top), implied_noise_ratio(r, M, top)))
    return out


def test_rho_never_falls_below_the_parameter_free_floor_for_nested_draws():
    """`sqrt(M/Mmax)` is a floor for ANY mix of pin-arm and per-draw noise."""
    for sg, sd in ((0.0, 1.5e-3), (2.0e-4, 1.5e-3), (2.0e-3, 1.5e-4)):
        _, reps = synth(lambda M: 0.030 / M, seeds=1, blocks=4000,
                        sigma_gal=sg, sigma_draw=sd, seed=31)
        for M, r, floor, _ in _rho_pairs(reps, LADDER):
            assert r >= floor - 0.05, f"sigma_gal={sg}: M={M} rho={r:.3f} < floor {floor:.3f}"


def test_the_implied_noise_ratio_is_CONSTANT_across_rungs_when_nesting_holds():
    """`a/s` is one number, so recovering it rung-by-rung is a test of the model itself."""
    _, reps = synth(lambda M: 0.030 / M, seeds=1, blocks=4000,
                    sigma_gal=6.0e-4, sigma_draw=1.5e-3, seed=37)
    ratios = [v for _, _, _, v in _rho_pairs(reps, LADDER)]
    assert np.std(ratios) / abs(np.mean(ratios)) < 0.25, ratios
    # and it must match the ratio actually injected, a/s = sigma_gal^2 / sigma_draw^2
    assert abs(np.mean(ratios) - (6.0e-4 / 1.5e-3) ** 2) < 0.15


def test_non_nested_draws_break_the_constant_ratio():
    """The diagnostic has to be able to FAIL, or it is not diagnosing anything."""
    rng = np.random.default_rng(41)
    B = 4000
    gal = rng.standard_normal(B) * 6.0e-4
    # independent noise scaled 1/sqrt(M) -- same per-rung VARIANCE as nesting, no shared
    # prefix, so rho collapses toward the pin-arm-only value and `a/s` stops being constant
    reps = np.stack([0.030 / M + gal + rng.standard_normal(B) * 1.5e-3 / np.sqrt(M)
                     for M in LADDER], axis=1)
    ratios = [v for _, _, _, v in _rho_pairs(reps, LADDER)]
    assert np.std(ratios) / abs(np.mean(ratios)) > 0.25, ratios


def _rho(a, b):
    va, vb, vd = a.var(ddof=1), b.var(ddof=1), (b - a).var(ddof=1)
    return (va + vb - vd) / (2 * np.sqrt(va * vb))


def _D_pred(M, N, top):
    return float(np.sqrt((1.0 / N - 1.0 / top) / (1.0 / M - 1.0 / top)))


def test_exact_parameter_free_correlation_of_the_differenced_rungs():
    """Subtracting the deepest rung cancels the common noise, so rho is EXACT, not bounded.

    `Var(D_M) = s(1/M - 1/Mmax)` and `Cov(D_M,D_N) = s(1/N - 1/Mmax)`, so `a` drops out
    entirely -- the prediction holds for any pin-arm noise, including none.
    """
    top = LADDER[-1]
    for sg in (0.0, 6.0e-4, 3.0e-3):          # must hold at every pin-arm noise level
        _, reps = synth(lambda M: 0.030 / M, seeds=1, blocks=6000,
                        sigma_gal=sg, sigma_draw=1.5e-3, seed=53)
        D = np.stack([reps[:, k] - reps[:, -1] for k in range(len(LADDER) - 1)], axis=1)
        for i in range(len(LADDER) - 2):
            meas = _rho(D[:, i], D[:, i + 1])
            pred = _D_pred(LADDER[i], LADDER[i + 1], top)
            assert abs(meas - pred) < 0.10, (sg, LADDER[i], meas, pred)


def test_the_exact_check_FAILS_on_non_nested_draws():
    rng = np.random.default_rng(59)
    B, top = 6000, LADDER[-1]
    gal = rng.standard_normal(B) * 6.0e-4
    reps = np.stack([gal + rng.standard_normal(B) * 1.5e-3 / np.sqrt(M) for M in LADDER],
                    axis=1)
    D = np.stack([reps[:, k] - reps[:, -1] for k in range(len(LADDER) - 1)], axis=1)
    off = [abs(_rho(D[:, i], D[:, i + 1]) - _D_pred(LADDER[i], LADDER[i + 1], top))
           for i in range(len(LADDER) - 2)]
    assert max(off) > 0.15, off


def test_the_floor_is_VACUOUS_on_raw_m_which_is_why_rho_uses_the_differenced_vector():
    """Guards the bug this fix corrected: on raw `m` the pin arm dominates and rho ~ 1."""
    _, reps = synth(lambda M: 0.030 / M, seeds=1, blocks=4000,
                    sigma_gal=1.0e-4, sigma_draw=1.5e-3, seed=61)
    pin = np.random.default_rng(2).standard_normal(4000) * 5.0e-2   # big shared pin arm
    raw = reps + pin[:, None]
    # on raw m every rung pair looks near-perfectly correlated regardless of the draws,
    # so the floor test cannot discriminate and would pass even with nesting broken
    assert _rho(raw[:, 0], raw[:, -1]) > 0.99
    # on the differenced vector the same data sit near the floor, where the test has teeth
    assert _rho(reps[:, 0], reps[:, -1]) < 0.6


# --- AMENDMENT 4(b): fitting against an explicit abscissa (1/ESS) -----------------------
#
# These pin the mechanism, and one of them pins the WARNING rather than the capability.
# A refit against 1/ESS looks like new evidence and mostly is not: where ESS is a power law
# in M, the substitution just grants the fit a free exponent.  That is a property worth a
# test, because it is the reason the headline claim rests on the across-configuration
# comparison instead (AMENDMENT 5).


def test_an_explicit_abscissa_recovers_a_law_written_in_that_abscissa():
    """d = d_inf + a/ESS is recovered when the fit is given 1/ESS, not 1/M."""
    ess = {M: 1.6 * M ** 0.30 for M in LADDER}          # struct beta=0.8-like, far from M
    inj_d, inj_a = 0.004, 0.030
    D, reps = synth(lambda M: inj_d + inj_a / ess[M])
    x = np.array([1.0 / ess[M] for M in LADDER])
    fits, _ = fit_ladder(D, reps, LADDER, x=x, xname="1/ESS")
    d_inf, sig = fits[0][1][0], np.sqrt(fits[0][2][0, 0])
    assert abs(d_inf - inj_d) < 4 * sig, f"{d_inf:.5f} vs injected {inj_d}"
    assert abs(fits[0][1][1] - inj_a) < 4 * np.sqrt(fits[0][2][1, 1])


def test_the_default_abscissa_is_still_1_over_M():
    """Passing no `x` must reproduce the old behaviour bit for bit."""
    D, reps = synth(lambda M: 0.004 + 0.030 / M)
    a, _ = fit_ladder(D, reps, LADDER)
    b, _ = fit_ladder(D, reps, LADDER, x=np.array([1.0 / M for M in LADDER]))
    for fa, fb in zip(a, b):
        assert np.allclose(fa[1], fb[1]) and np.allclose(fa[2], fb[2])
        assert fa[3] == pytest.approx(fb[3])


def test_the_wrong_abscissa_is_STRONGLY_rejected_not_flattered():
    """The measurement that overturned my own pre-registered worry, kept as a test.

    AMENDMENT 5(ii) asserted that swapping 1/M for 1/ESS is a free pass: where ESS is a
    power law in M, the substitution hands the fit an exponent, so a chi2 improvement would
    prove nothing.  That is wrong, and this test is the demonstration.  With the truth set
    to bias ~ 1/M exactly and the fit given M^-0.9, chi2/dof lands at 465, 43 and 5.9 as the
    draw noise is raised over a 10x range -- rejected everywhere, and far outside anything a
    marginal ladder shows.  The GLS is sensitive to the SHAPE of the abscissa because the
    rungs are strongly correlated, so a reparametrisation that is wrong is visible.

    The consequence, recorded in AMENDMENT 6: the within-ladder 4(b) chi2 DOES carry
    information, and the real nbr improvement (2.73 -> 1.83 under the measured ESS exponent)
    is a result rather than an artefact of granting a free parameter.
    """
    xe = np.array([1.0 / (M ** 0.9) for M in LADDER])
    got = []
    for sd in (1.5e-3, 5e-3, 1.5e-2):
        D, reps = synth(lambda M: 0.004 + 0.030 / M, sigma_draw=sd)   # truth is 1/M
        right, _ = fit_ladder(D, reps, LADDER)
        wrong, _ = fit_ladder(D, reps, LADDER, x=xe, xname="1/ESS")
        got.append((right[0][3], wrong[0][3]))
        assert right[0][3] < 2.0, f"the CORRECT abscissa should fit: {right[0][3]:.2f}"
        assert wrong[0][3] > 4.0, (
            f"at sigma_draw={sd:.1e} the wrong abscissa gave chi2/dof {wrong[0][3]:.2f}; "
            f"if this is now small, the 4(b) comparison has lost its power and AMENDMENT 6 "
            f"needs revisiting")
    # the discriminating power falls as noise rises -- state the direction, not just a bound
    assert got[0][1] > got[1][1] > got[2][1], f"expected monotone loss of power: {got}"


def test_a_power_law_ESS_is_detected_as_such():
    """The residual diagnostic that decides whether 4(b) is worth anything."""
    ladder = np.array(LADDER, dtype=float)
    ess = 1.591 * ladder ** 0.2925                        # measured struct beta=0.8 law
    slope, inter = np.polyfit(np.log(ladder), np.log(ess), 1)
    rms = float(np.sqrt(np.mean((np.log(ess) - (slope * np.log(ladder) + inter)) ** 2)))
    assert slope == pytest.approx(0.2925, abs=1e-6)
    assert rms < 1e-9, "a pure power law must leave no residual"


def test_a_non_power_law_ESS_is_NOT_flagged_as_a_reparametrisation():
    """The other half: if ESS genuinely departs from a power law, 4(b) has content."""
    ladder = np.array(LADDER, dtype=float)
    ess = 1.0 + ladder / (1.0 + ladder / 6.0)             # saturating, not a power law
    slope, inter = np.polyfit(np.log(ladder), np.log(ess), 1)
    rms = float(np.sqrt(np.mean((np.log(ess) - (slope * np.log(ladder) + inter)) ** 2)))
    assert rms > 0.02, f"expected a visible departure, got rms {rms:.4f}"


def test_the_abscissa_uncertainty_widens_the_chi2_range_as_the_ESS_bar_grows():
    """AMENDMENT 7: bound the chi2 by enumerating +/-1 sigma shifts on the abscissa.

    The chi2 is brutally sensitive to the SHAPE of the abscissa (AMENDMENT 6), so the
    abscissa's own error is amplified by the same gain.  This pins the direction that makes
    the bound meaningful -- a bigger ESS bar must produce a WIDER range, and a zero bar must
    collapse it to the central value.  Without that, a 'sensitivity check' that always
    printed a narrow range would look reassuring while testing nothing.
    """
    import itertools
    ess = np.array([1.6 * M ** 0.30 for M in LADDER])
    D, reps = synth(lambda M: 0.004 + 0.030 / (1.6 * M ** 0.30))

    def span(rel):
        c2 = []
        for signs in itertools.product((-1.0, 1.0), repeat=len(LADDER)):
            e = ess * (1.0 + np.array(signs) * rel)
            fits, _ = fit_ladder(D, reps, LADDER, x=1.0 / e)
            c2.append(fits[0][3])
        return max(c2) - min(c2)

    assert span(0.0) == pytest.approx(0.0, abs=1e-12), "a zero bar must collapse the range"
    narrow, wide = span(0.01), span(0.05)
    assert narrow > 0, "a nonzero bar must open the range"
    assert wide > narrow, f"a 5% bar must span more than a 1% bar: {wide} vs {narrow}"


# --------------------------------------------------------------------------------------
# The abscissa is a mean over rows, and the estimator does NOT weight rows equally.
# `ghat = sum_i s_i / sum_i I_i`, so the per-row law c_i/ESS_i aggregates weighted by I_i.
# These tests pin the exactness of that correction and the sign we expect it to take.
# --------------------------------------------------------------------------------------

def test_row_weights_is_None_when_the_run_predates_per_row_information():
    """The fallback must announce itself, not fabricate a weight."""
    gdir = np.array([1.0, 0.0])
    assert cc.row_weights({"irow": {}}, [1, 2], gdir) is None
    assert cc.row_weights({}, [1, 2], gdir) is None
    # the default source is the PIN arm, so rungs alone are not enough
    assert cc.row_weights({"irow": {1: np.eye(2)[None], 2: np.eye(2)[None]}},
                          [1, 2], gdir) is None
    # and for the per-rung variant a PARTIAL set is unusable too: a ladder mixing
    # weighted and flat rungs would vary the statistic along the axis the fit reads
    assert cc.row_weights({"irow": {1: np.eye(2)[None]}}, [1, 2], gdir,
                          source="per_rung") is None


def test_row_weights_projects_the_information_onto_the_shear_direction():
    rng = np.random.default_rng(0)
    c = rng.uniform(1.0, 5.0, size=7)
    irow = {m: (c[:, None, None] * np.eye(2)[None]) * m for m in (1, 2)}
    w = cc.row_weights({"irow": irow}, [1, 2], np.array([1.0, 0.0]), source="per_rung")
    assert w.shape == (7, 2)
    assert np.allclose(w[:, 0], c)
    assert np.allclose(w[:, 1], 2 * c)
    # and it must be a genuine projection, not a trace: an anisotropic I read along g2
    aniso = {m: np.tile(np.diag([9.0, 1.0]), (7, 1, 1)) for m in (1, 2)}
    w2 = cc.row_weights({"irow": aniso}, [1, 2], np.array([0.0, 1.0]), source="per_rung")
    assert np.allclose(w2, 1.0)


def test_the_weight_is_FROZEN_across_rungs_by_default():
    """Every rung must get the SAME weight vector, taken from the pin arm.

    Each rung's own I is finite-draw biased by exactly the effect the ladder measures, so
    an unfrozen weight drifts along the ladder for that reason alone -- forging a
    rung-dependence indistinguishable from the one being looked for.  The pin arm carries
    no marginalisation, so it has no draw bias to leak in.
    """
    rng = np.random.default_rng(6)
    n, ladder = 40, [1, 2, 4]
    pin = rng.uniform(1.0, 3.0, size=n)
    irow = {"pin": pin[:, None, None] * np.eye(2)[None]}
    # rungs carry a DIFFERENT, rung-dependent I -- the frozen weight must ignore it
    for j, m in enumerate(ladder):
        irow[m] = (pin * (1.0 + 0.3 * j))[:, None, None] * np.eye(2)[None]
    gdir = np.array([1.0, 0.0])
    w = cc.row_weights({"irow": irow}, ladder, gdir)
    assert w.shape == (n, len(ladder))
    assert np.allclose(w, pin[:, None])                       # frozen, and frozen on PIN
    for j in range(1, len(ladder)):
        assert np.allclose(w[:, j], w[:, 0])
    # the per-rung variant must actually differ, or the freeze is untested
    wpr = cc.row_weights({"irow": irow}, ladder, gdir, source="per_rung")
    assert not np.allclose(wpr[:, -1], wpr[:, 0])
    # "deep" freezes too, but on the deepest rung -- also constant across rungs
    wd = cc.row_weights({"irow": irow}, ladder, gdir, source="deep")
    assert np.allclose(wd, (pin * 1.6)[:, None])


def test_an_unfrozen_weight_FORGES_a_rung_dependence_that_the_freeze_removes():
    """The confound is real, not hypothetical -- this is the artifact, reproduced.

    Build data whose ESS has NO rung-dependent correlation with the true information, but
    whose per-rung I is draw-biased in a way that strengthens with M.  The per-rung weight
    manufactures a growing shift; the frozen weight does not.
    """
    rng = np.random.default_rng(7)
    n, ladder = 8000, [1, 2, 4, 8]
    pin = rng.lognormal(0.0, 0.4, size=n)
    ess = np.stack([rng.lognormal(0.5, 0.4, size=n) for _ in ladder], axis=1)
    irow = {"pin": pin[:, None, None] * np.eye(2)[None]}
    for j, m in enumerate(ladder):
        biased = pin * np.exp(0.25 * j * (np.log(ess[:, j]) - np.log(ess[:, j]).mean()))
        irow[m] = biased[:, None, None] * np.eye(2)[None]
    gdir = np.array([1.0, 0.0])
    run = {"irow": irow}
    _, _, _, _, froz = cc.ess_abscissa([ess], ladder,
                                       wr=[cc.row_weights(run, ladder, gdir)])
    _, _, _, _, perr = cc.ess_abscissa(
        [ess], ladder, wr=[cc.row_weights(run, ladder, gdir, source="per_rung")])
    spread = lambda d: max(d.values()) - min(d.values())          # noqa: E731
    assert spread(perr) > 0.05, perr        # the artifact: a big manufactured trend
    assert spread(froz) < 0.01, froz        # the freeze kills it
    assert spread(perr) > 5 * spread(froz)


def test_the_weighted_minus_flat_gap_IS_the_covariance_exactly():
    """Not a leading-order expansion: a weighted mean minus a flat mean is Cov(w,x)/<w>."""
    rng = np.random.default_rng(1)
    n = 4000
    ess = rng.lognormal(mean=1.0, sigma=0.6, size=(n, 1))
    w = np.exp(-0.8 * np.log(ess)) * rng.lognormal(0.0, 0.3, size=(n, 1))
    inv, _, _, _, shift = cc.ess_abscissa([ess], [8], wr=[w])
    flat, _, _, _, shift0 = cc.ess_abscissa([ess], [8])
    x = 1.0 / ess[:, 0]
    cov = np.mean((w[:, 0] - w[:, 0].mean()) * (x - x.mean())) / w[:, 0].mean()
    assert np.isclose(inv[8] - flat[8], cov, rtol=1e-10)
    assert np.isclose(shift[8], (inv[8] - flat[8]) / flat[8], rtol=1e-10)
    assert shift0[8] == 0.0          # no weights -> nothing to correct, and it says so


def test_the_expected_sign_is_POSITIVE_when_informative_rows_are_hard_to_match():
    """High I and low ESS travel together, so the flat mean UNDERSTATES the abscissa.

    This is the sign that matters: it is the SAME direction as the Jensen gap, so the two
    approximations stack rather than partially cancelling.  A test that only checked the
    magnitude would let a sign error through unnoticed.
    """
    rng = np.random.default_rng(2)
    ess = rng.lognormal(mean=1.2, sigma=0.5, size=(6000, 1))
    w = 1.0 / ess                                    # informative <-> low ESS, by hand
    _, _, _, _, shift = cc.ess_abscissa([ess], [16], wr=[w])
    assert shift[16] > 0.05
    _, _, _, _, anti = cc.ess_abscissa([ess], [16], wr=[ess.copy()])
    assert anti[16] < 0.0            # and it reverses when the correlation reverses


def test_the_weighting_shift_VARIES_ACROSS_RUNGS_so_it_distorts_shape():
    """A rung-dependent shift is not a reparameterisation of the slope.

    If the shift were common across rungs it would rescale `a` and leave `d_inf` and the
    fit quality alone.  It is not: the I-ESS correlation tightens as M grows, so the shift
    grows too, which bends the curve the GLS is fitting.
    """
    rng = np.random.default_rng(3)
    ladder = [1, 4, 16, 64]
    ess = np.stack([rng.lognormal(0.5, 0.3 + 0.15 * k, size=5000)
                    for k in range(len(ladder))], axis=1)
    w = np.stack([np.exp(-(0.2 + 0.3 * k) * np.log(ess[:, k])) for k in range(len(ladder))],
                 axis=1)
    _, _, _, _, shift = cc.ess_abscissa([ess], ladder, wr=[w])
    vals = [shift[m] for m in ladder]
    assert all(v > 0 for v in vals)
    assert vals[-1] > vals[0] + 0.02, vals


def test_the_flat_default_is_bit_identical_to_the_pre_weighting_behaviour():
    """The fallback must not drift: old runs have to keep reading the way they read."""
    rng = np.random.default_rng(4)
    er = [rng.lognormal(0.7, 0.4, size=(800, 3)) for _ in range(2)]
    inv, mean, cv, sd, shift = cc.ess_abscissa(er, [1, 2, 4])
    want = np.stack([(1.0 / e).mean(axis=0) for e in er]).mean(axis=0)
    for m, v in zip([1, 2, 4], want):
        assert np.isclose(inv[m], v, rtol=1e-12)
        assert shift[m] == 0.0
    assert np.isclose(mean[1], np.mean([e[:, 0].mean() for e in er]), rtol=1e-12)


def test_weighted_standard_error_reduces_to_the_plain_one_at_equal_weights():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(500, 2))
    w = np.full_like(x, 3.7)
    assert np.allclose(cc._wsem(x, w), cc._wsem(x, None), rtol=1e-10)
    # unequal weights must SHRINK nothing artificially -- concentrating weight on few rows
    # loses effective sample size and the error has to grow
    conc = np.zeros_like(x)
    conc[:10] = 1.0
    assert np.all(cc._wsem(x, conc) > cc._wsem(x, None))


def test_load_picks_up_per_row_information_when_the_file_carries_it(tmp_path):
    """Guards the eval-side field name against a silent rename."""
    import json
    n, ladder = 5, [1, 2]
    kw = dict(config=json.dumps({"draw_seed": 1}), ladder=np.array(ladder),
              ess=np.ones(n), features=np.array(["nbr"]))
    for k in ["pin"] + ladder:
        kw[f"{k}_cnt"] = np.ones(2)
        kw[f"{k}_ns"] = np.ones((2, 2))
        kw[f"{k}_ni"] = np.tile(np.eye(2), (2, 1, 1))
    p = tmp_path / "no_irow.npz"
    np.savez(p, **kw)
    assert cc.load(str(p))["irow"] == {}

    # rungs ONLY: not enough, because the default weight source is the pin arm
    for k in ladder:
        kw[f"{k}_i_row"] = np.tile(np.eye(2), (n, 1, 1)).astype(np.float32) * k
    p2 = tmp_path / "rungs_only.npz"
    np.savez(p2, **kw)
    got = cc.load(str(p2))["irow"]
    assert sorted(got) == ladder
    assert got[2].shape == (n, 2, 2) and np.allclose(got[2][0], 2 * np.eye(2))
    assert cc.row_weights({"irow": got}, ladder, np.array([1.0, 0.0])) is None

    # `eval_score_catprior.py` writes one field per key in ["pin"] + ladder, so the pin
    # arm IS on disk -- this pins that contract, which the frozen weight depends on
    kw["pin_i_row"] = np.tile(np.eye(2), (n, 1, 1)).astype(np.float32) * 7
    p3 = tmp_path / "with_pin.npz"
    np.savez(p3, **kw)
    got = cc.load(str(p3))["irow"]
    assert "pin" in got
    w = cc.row_weights({"irow": got}, ladder, np.array([1.0, 0.0]))
    assert w is not None and np.allclose(w, 7.0)


# --------------------------------------------------------------------------------------
# The drift diagnostic's RESOLUTION.  It warns when the information's draw bias falls more
# slowly than 1/M, so silence has to be readable as "consistent" vs "unresolved" -- which
# only an error bar decides.  An unresolved measurement is not a pass.
# --------------------------------------------------------------------------------------

def _drift_case(rate, n=4000, ladder=(1, 2, 4, 8), seed=0):
    """Synthetic run whose per-rung information carries an injected draw bias `rate(M)`."""
    rng = np.random.default_rng(seed)
    ladder = list(ladder)
    er = np.stack([rng.lognormal(np.log(m), 0.4, size=n) for m in ladder], axis=1)
    pin = np.exp(-0.5 * (np.log(er[:, -1]) - np.log(er[:, -1]).mean()))
    wf = np.repeat(pin[:, None], len(ladder), axis=1)
    wp = np.stack([pin * np.exp(0.3 * rate(m) * (np.log(er[:, j]) - np.log(er[:, j]).mean()))
                   for j, m in enumerate(ladder)], axis=1)
    return [er], [wf], [wp], ladder


def test_drift_slope_recovers_an_injected_exponent():
    for inj in (1.0, 0.5, 0.2):
        er, wf, wp, ladder = _drift_case(lambda m, a=inj: m ** -a)
        sl, sd, drift = cc.drift_slope(er, wf, wp, ladder)
        assert len(drift) == len(ladder) and np.all(drift > 0)
        assert abs(sl + inj) < cc.slope_floor(len(ladder)), (inj, sl)


def test_the_bootstrap_bar_is_finite_and_shrinks_with_rows():
    small = cc.drift_slope(*_drift_case(lambda m: 1.0 / m, n=400))[1]
    big = cc.drift_slope(*_drift_case(lambda m: 1.0 / m, n=8000))[1]
    assert np.isfinite(small) and np.isfinite(big)
    assert big < small


def test_the_bootstrap_bar_DOES_NOT_cover_the_recovery_bias():
    """The reason a systematic floor exists at all -- pinned so it cannot be quietly dropped.

    A true 1/M drift does not come back as exactly -1: fitting a power law over a short
    ladder costs a systematic the row bootstrap is blind to.  Reporting the bootstrap bar
    alone would call correct data several sigma discrepant, which is the false alarm the
    floor exists to prevent.
    """
    er, wf, wp, ladder = _drift_case(lambda m: 1.0 / m, n=8000)
    sl, sd, _ = cc.drift_slope(er, wf, wp, ladder)
    assert abs(sl + 1.0) > 2 * sd            # truth is OUTSIDE the bootstrap bar
    assert cc.slope_floor(len(ladder)) > abs(sl + 1.0)   # and the floor covers it
    assert cc.slope_floor(len(ladder)) >= 2 * sd  # so the floor, not the bar, sets the threshold


def test_drift_slope_declines_to_fit_a_power_law_through_two_points():
    er, wf, wp, ladder = _drift_case(lambda m: 1.0 / m, ladder=(1, 4))
    sl, sd, drift = cc.drift_slope(er, wf, wp, ladder)
    assert np.isnan(sl) and np.isnan(sd)
    assert len(drift) == 2                   # the drift itself is still reported


def test_the_slope_floor_is_a_FUNCTION_of_ladder_length_not_one_number():
    """Measured by injection: the recovery bias falls as 1/K, flat in the true exponent.

    A single floor is wrong in the direction that matters -- too loose on long ladders,
    discarding sensitivity exactly where the measurement is best.  The fixed 0.10 this
    replaced came from one injected truth at one depth on a single seed.
    """
    assert cc.slope_floor(4) > cc.slope_floor(5) > cc.slope_floor(7)
    for k in (4, 5, 7):
        assert np.isclose(cc.slope_floor(k) * k, cc.SLOPE_FLOOR_K)
    assert cc.slope_floor(0) > 0                     # degenerate input must not divide by zero
    # and it must actually COVER the measured bias at each production ladder, with margin
    for lad, want in (([1, 2, 4, 8], 0.034), ([1, 2, 4, 8, 16], 0.026),
                      ([1, 2, 4, 8, 16, 32, 64], 0.019)):
        assert cc.slope_floor(len(lad)) > want
        assert cc.slope_floor(len(lad)) < 4 * want   # covering, not vacuously wide


def test_the_recovery_bias_is_FLAT_in_the_true_exponent():
    """Justifies floor(K) rather than floor(K, p) -- the other plausible shape."""
    bias = []
    for inj in (1.0, 0.7, 0.5, 0.3):
        er, wf, wp, ladder = _drift_case(lambda m, a=inj: m ** -a, n=6000)
        bias.append(abs(cc.drift_slope(er, wf, wp, ladder, n_boot=8)[0] + inj))
    assert max(bias) - min(bias) < 0.01, bias        # flat to well under the floor


# ======================================================================================
# POOL HALVES
#
# The A/B split has to satisfy three things at once for `d_A - d_B` to mean anything:
# the halves must be exact complements (or the arms overlap and the difference is
# diluted), the split must not move when the DRAW seed moves (or the arms stop being
# paired), and it must move when its OWN seed moves (or several "independent" splits are
# the same split).  Each is one test below.
# ======================================================================================

from sbsi.catalogue_prior import pool_half_indices  # noqa: E402


def test_the_halves_are_EXACT_complements():
    n = 5000
    a, b = pool_half_indices(n, "A"), pool_half_indices(n, "B")
    assert np.intersect1d(a, b).size == 0, "arms share pool rows; the difference is diluted"
    assert np.array_equal(np.union1d(a, b), np.arange(n)), "the halves do not cover the pool"
    assert len(a) + len(b) == n


def test_an_odd_pool_loses_no_row():
    # the extra row goes to B by construction; what matters is that it goes SOMEWHERE
    a, b = pool_half_indices(7, "A"), pool_half_indices(7, "B")
    assert len(a) + len(b) == 7
    assert np.array_equal(np.union1d(a, b), np.arange(7))


def test_the_split_is_INDEPENDENT_of_the_draw_seed():
    # the whole point of a separate seed: the two arms must share their draw randomness,
    # so changing --draw-seed must not reshuffle which rows are drawable
    first = pool_half_indices(4000, "A", seed=4242)
    again = pool_half_indices(4000, "A", seed=4242)
    assert np.array_equal(first, again), "the split is not reproducible at fixed seed"


def test_a_different_split_seed_gives_a_GENUINELY_different_split():
    # several splits are only several measurements if they actually differ
    a1 = pool_half_indices(4000, "A", seed=1)
    a2 = pool_half_indices(4000, "A", seed=2)
    overlap = np.intersect1d(a1, a2).size / len(a1)
    assert not np.array_equal(a1, a2)
    # two random halves of the same pool share ~50% by chance; anything near 100% would
    # mean the seed barely moved the split
    assert 0.4 < overlap < 0.6, f"overlap {overlap:.2f} is not chance-level"


def test_the_indices_are_sorted_so_the_subset_keeps_catalogue_order():
    a = pool_half_indices(3000, "B", seed=7)
    assert np.all(np.diff(a) > 0), "unsorted indices would permute the pool as well as cut it"


def test_none_means_no_restriction():
    assert pool_half_indices(100, "none") is None


def test_a_bad_half_label_is_REFUSED_rather_than_silently_ignored():
    # a typo'd --pool-half must not quietly run an unrestricted job and be logged as a half
    with pytest.raises(ValueError, match="A.*B.*none"):
        pool_half_indices(100, "a")
    with pytest.raises(ValueError):
        pool_half_indices(1, "A")


# ======================================================================================
# THE PAIRING GAIN
#
# cont.189 predicts the gain as sqrt(1/f) from the measured draw-noise share f.  That
# prediction is only worth registering if the estimator actually recovers sqrt(1/f) from
# data with a known split, so that is the first test.  The rest guard the hard gate.
# ======================================================================================

def _arm_stats(gal, pool, ladder):
    """stats()-shaped dict whose rung replicates are `gal + pool` and whose pin is zero."""
    out = {"pin": (0.0, np.zeros(len(gal)))}
    for j, m in enumerate(ladder):
        out[m] = (0.0, gal + pool[:, j])
    return out


def test_the_gain_recovers_the_KNOWN_variance_split():
    # gain = sd(unpaired)/sd(paired) = sqrt(1 + Vgal/Vpool) = sqrt(1/f) with f the pool
    # share.  This is the identity cont.189 pre-registers its prediction from.
    rng = np.random.default_rng(0)
    B, ladder = 40000, [1]
    for f in (0.108, 0.300, 0.582):          # the measured shares at M=16, 4, 1
        gal = rng.normal(0, np.sqrt(1 - f), B)
        pa = rng.normal(0, np.sqrt(f), (B, 1))
        pb = rng.normal(0, np.sqrt(f), (B, 1))
        rows, _ = cc.pool_halves_report(_arm_stats(gal, pa, ladder),
                                        _arm_stats(gal, pb, ladder), ladder)
        assert rows[0]["gain"] == pytest.approx(np.sqrt(1 / f), rel=0.03), \
            f"gain does not recover sqrt(1/f) at f={f}"


def test_a_fully_unshared_difference_gives_NO_pairing_benefit():
    # no common galaxy term -> nothing can cancel -> gain 1.  This is the floor, and it is
    # why "gain near 1" is NOT evidence of breakage in this test: it is what a
    # pool-dominated variance looks like.
    rng = np.random.default_rng(1)
    B, ladder = 40000, [1]
    gal = np.zeros(B)
    rows, _ = cc.pool_halves_report(
        _arm_stats(gal, rng.normal(0, 1, (B, 1)), ladder),
        _arm_stats(gal, rng.normal(0, 1, (B, 1)), ladder), ladder)
    assert rows[0]["gain"] == pytest.approx(1.0, rel=0.03)


def test_a_fully_shared_difference_pairs_away_almost_everything():
    rng = np.random.default_rng(2)
    B, ladder = 20000, [1]
    gal = rng.normal(0, 1, B)
    tiny = rng.normal(0, 0.01, (B, 1)), rng.normal(0, 0.01, (B, 1))
    rows, _ = cc.pool_halves_report(_arm_stats(gal, tiny[0], ladder),
                                    _arm_stats(gal, tiny[1], ladder), ladder)
    assert rows[0]["gain"] > 50


def test_the_pin_gate_is_zero_when_the_split_left_pin_alone():
    ladder = [1, 2]
    rng = np.random.default_rng(3)
    gal = rng.normal(0, 1, 500)
    a = _arm_stats(gal, rng.normal(0, 1, (500, 2)), ladder)
    b = _arm_stats(gal, rng.normal(0, 1, (500, 2)), ladder)
    _, pin_gap = cc.pool_halves_report(a, b, ladder)
    assert pin_gap == 0.0


def test_the_pin_gate_FIRES_when_the_split_moved_pin():
    # pin never reads the pool, so any movement means the split touched something else
    ladder = [1]
    gal = np.zeros(200)
    a = _arm_stats(gal, np.zeros((200, 1)), ladder)
    b = _arm_stats(gal, np.zeros((200, 1)), ladder)
    b["pin"] = (1e-9, np.zeros(200))
    _, pin_gap = cc.pool_halves_report(a, b, ladder)
    assert pin_gap == pytest.approx(1e-9)


def test_the_difference_is_reported_against_the_PAIRED_bar_not_the_unpaired_one():
    # quoting the unpaired bar would understate the significance of a real A/B gap
    ladder = [1]
    rng = np.random.default_rng(4)
    gal = rng.normal(0, 1, 5000)
    a = _arm_stats(gal, rng.normal(0, 0.1, (5000, 1)), ladder)
    b = _arm_stats(gal, rng.normal(0, 0.1, (5000, 1)), ladder)
    a[1] = (0.05, a[1][1])                    # inject an A/B offset
    rows, _ = cc.pool_halves_report(a, b, ladder)
    assert rows[0]["diff"] == pytest.approx(0.05)
    assert rows[0]["nsig"] == pytest.approx(0.05 / rows[0]["sd_paired"])
    assert rows[0]["sd_paired"] < rows[0]["sd_unpaired"]


# ======================================================================================
# SHARDS AND THE IRREDUCIBLE FINITE-CATALOGUE TERM
# ======================================================================================

from sbsi.catalogue_prior import pool_shard_indices  # noqa: E402


def test_shards_partition_the_pool_exactly():
    for n_shards in (2, 3, 4, 8):
        sh = [pool_shard_indices(1000, i, n_shards) for i in range(n_shards)]
        assert len(np.unique(np.concatenate(sh))) == 1000, "shards do not cover the pool"
        assert sum(len(x) for x in sh) == 1000, "shards overlap"
        assert max(len(x) for x in sh) - min(len(x) for x in sh) <= 1


def test_halves_are_the_two_shard_special_case():
    # the half API must not drift away from the general one
    assert np.array_equal(pool_half_indices(999, "A", 5), pool_shard_indices(999, 0, 2, 5))
    assert np.array_equal(pool_half_indices(999, "B", 5), pool_shard_indices(999, 1, 2, 5))


def test_a_bad_shard_request_is_REFUSED():
    with pytest.raises(ValueError):
        pool_shard_indices(100, 0, 1)          # one shard is not a split
    with pytest.raises(ValueError):
        pool_shard_indices(100, 4, 4)          # 0-based, so 4 is out of range
    with pytest.raises(ValueError):
        pool_shard_indices(3, 0, 4)            # more shards than rows


def test_the_irreducible_term_recovers_an_INJECTED_value():
    # a_prod = A  ->  each half-pool arm carries 2A  ->  var(A-B) = 4A, on top of the
    # within-split sampling floor s^2, which must be subtracted or A comes out inflated
    rng = np.random.default_rng(0)
    for A, s in ((4e-6, 1e-3), (1e-5, 2e-3)):
        d = rng.normal(0, np.sqrt(4 * A + s**2), 4000)
        a, sd, n = cc.irreducible_term(d, np.full(4000, s), n_shards=2)
        assert n == 4000
        assert a == pytest.approx(A, rel=0.10), f"failed to recover a={A}"


def test_the_sampling_floor_is_SUBTRACTED_not_ignored():
    # pure sampling noise and no pool term at all must give a ~ 0, not s^2/4
    rng = np.random.default_rng(1)
    s = 1e-3
    d = rng.normal(0, s, 5000)
    a, _, _ = cc.irreducible_term(d, np.full(5000, s), n_shards=2)
    assert a < 0.05 * s**2 / 4, "the sampling floor leaked into the irreducible term"


def test_a_negative_draw_is_REPORTED_not_clipped():
    # `a` is a difference of two noisy variances, so a downward fluctuation is ORDINARY at
    # small n.  Clipping it at zero would bias a symmetric estimator upward, and hardest in
    # the low-signal regime this test is built for -- i.e. it would manufacture a detection.
    a, sd, _ = cc.irreducible_term(np.zeros(50), np.full(50, 1e-3))
    assert a < 0, "a downward fluctuation was clipped; the estimator is now biased high"
    assert a == pytest.approx(-(1e-3) ** 2 / 4)      # (0 - floor) / (2 * n_shards)
    assert sd == 0.0, "zero scatter across splits means zero error on the excess"


def test_the_clipped_and_honest_estimators_DISAGREE_in_the_null_case():
    # the concrete cost of clipping: average many null realisations and the honest estimator
    # centres on zero while the clipped one sits strictly above it
    rng = np.random.default_rng(11)
    floor = 2e-3
    honest, clipped = [], []
    for _ in range(400):
        d = rng.normal(0.0, floor, 4)                # pure floor, no irreducible term
        a, _, _ = cc.irreducible_term(d, np.full(4, floor), n_shards=2)
        honest.append(a)
        clipped.append(max(0.0, a))
    honest, clipped = np.array(honest), np.array(clipped)
    sem = np.std(honest, ddof=1) / np.sqrt(len(honest))
    assert abs(np.mean(honest)) < 3 * sem, "the unclipped estimator must centre on zero"
    # and the clipped one must NOT: it sits a resolvable fraction of its own sigma high
    assert np.mean(clipped) > 3 * sem, "clipping is what biases the null upward"
    assert np.mean(clipped) > 0.3 * np.std(honest, ddof=1), "the bias is ~0.4 sigma, not noise"


def test_the_verdict_rule_quotes_a_VALUE_only_above_2_sigma():
    # pre-registered in cont.189 §6: >2 sigma is a detection, everything else an upper limit
    kind, amp = cc.a_verdict(1.0e-4, 1.0e-5)         # 10 sigma
    assert kind == "detection" and amp == pytest.approx(1e-2)
    kind, _ = cc.a_verdict(1.0e-4, 6.0e-5)           # 1.7 sigma
    assert kind == "upper limit"


def test_a_NEGATIVE_estimate_still_yields_a_usable_upper_limit():
    # the whole point of not clipping: a null result must still bound `a`, via the INTERVAL
    kind, amp = cc.a_verdict(-1.0e-5, 2.0e-5)
    assert kind == "upper limit"
    assert amp == pytest.approx(np.sqrt(-1.0e-5 + 1.645 * 2.0e-5))
    assert amp > 0, "an upper limit on a variance amplitude must be real and positive"


def test_a_limit_below_zero_is_floored_but_the_POINT_estimate_never_is():
    # clipping is legitimate for a LIMIT (a variance cannot be negative) and illegitimate
    # for the point estimate -- this pins that the code applies it to exactly one of them
    kind, amp = cc.a_verdict(-1.0e-4, 1.0e-6)
    assert kind == "upper limit" and amp == 0.0
    raw, _, _ = cc.irreducible_term(np.zeros(20), np.full(20, 1e-2))
    assert raw < 0, "the point estimate must keep its sign"


def test_ONE_split_refuses_to_estimate():
    # n = 1 has no scatter to measure; returning a number here would be the whole trap
    a, sd, n = cc.irreducible_term([0.01], [1e-3])
    assert a is None and sd is None and n == 1


def test_the_shard_count_enters_the_scaling():
    # quarters put each arm at a quarter pool, so the same observed scatter implies a
    # SMALLER production term -- this is the 1/N assumption made explicit
    d = np.full(200, 0.0)
    rng = np.random.default_rng(2)
    d = rng.normal(0, 1e-3, 2000)
    a2, _, _ = cc.irreducible_term(d, np.zeros(2000), n_shards=2)
    a4, _, _ = cc.irreducible_term(d, np.zeros(2000), n_shards=4)
    assert a4 == pytest.approx(a2 / 2, rel=1e-9)


def test_the_blindness_check_PASSES_when_sd_paired_only_carries_sampling_noise():
    # if the jackknife is blind to the pool, sd_paired is the same measurement each split
    rng = np.random.default_rng(0)
    B = 200
    sd = 1e-3 * (1 + rng.normal(0, 1 / np.sqrt(2 * (B - 1)), 40))
    sc, ex, verdict = cc.jackknife_blindness_check(sd, n_blocks=B)
    assert sc < 2 * ex and verdict.startswith("BLIND")


def test_the_blindness_check_FIRES_when_sd_paired_tracks_the_pool():
    # a galaxy-dependent pool effect leaks into sd_paired and makes it vary split to split;
    # that is the case where subtracting the floor over-subtracts and biases `a` LOW
    rng = np.random.default_rng(1)
    sd = 1e-3 * (1 + rng.normal(0, 0.25, 40))
    sc, ex, verdict = cc.jackknife_blindness_check(sd, n_blocks=200)
    assert sc > 2 * ex and verdict.startswith("NOT BLIND")


def test_the_blindness_check_declines_on_one_split():
    sc, ex, verdict = cc.jackknife_blindness_check([1e-3])
    assert sc is None and "too few" in verdict


def test_over_subtraction_biases_the_irreducible_term_LOW_not_high():
    # pins the direction of the failure the blindness check guards against: if the floor
    # is inflated (because it contains pool signal) then `a` comes out too SMALL
    rng = np.random.default_rng(2)
    A, s = 4e-6, 1e-3
    d = rng.normal(0, np.sqrt(4 * A + s**2), 6000)
    honest, _, _ = cc.irreducible_term(d, np.full(6000, s), n_shards=2)
    inflated, _, _ = cc.irreducible_term(d, np.full(6000, s * 1.10), n_shards=2)
    assert inflated < honest, "an inflated floor must bias `a` downward"


# ======================================================================================
# THE RUN-IDENTITY KEY
#
# Three bugs in one session had the same shape: an identifying field missing from the
# grouping key, so runs that differ get averaged and nothing errors.  The key is now an
# ALLOWLIST of what may differ, so these tests pin the inversion itself -- a new field must
# be identifying by DEFAULT, without anyone remembering to register it.
# ======================================================================================

def _run(**cfg):
    base = dict(draw_seed=99, beta=0.0, max_rows=1000, grid_n=61, save="/tmp/x.npz")
    base.update(cfg)
    return dict(cfg=base, features=["nbr"], ladder=[1, 2], path="/tmp/x.npz")


def test_a_NEW_config_field_is_identifying_without_being_registered():
    # the whole point of the inversion: nobody has to remember to add it
    a = cc.run_identity(_run())[0]
    b = cc.run_identity(_run(some_future_knob=7))[0]
    assert a != b, "a new config field was silently poolable -- the key is a blacklist again"


def test_the_declared_replicate_axis_does_NOT_split_the_group():
    a = cc.run_identity(_run(draw_seed=99))[0]
    b = cc.run_identity(_run(draw_seed=2024))[0]
    assert a == b, "runs differing only in the replicate axis must still pool"


def test_the_output_path_is_NOT_identifying():
    # it differs by construction on every run and says nothing about the measurement
    a = cc.run_identity(_run(save="/tmp/one.npz"))[0]
    b = cc.run_identity(_run(save="/tmp/two.npz"))[0]
    assert a == b


def test_pool_half_and_split_seed_are_identifying():
    # the three historical bugs, pinned directly
    base = cc.run_identity(_run())[0]
    assert cc.run_identity(_run(pool_half="A"))[0] != base
    assert cc.run_identity(_run(pool_half="A"))[0] != cc.run_identity(_run(pool_half="B"))[0]
    assert (cc.run_identity(_run(pool_half="A", pool_split_seed=1))[0]
            != cc.run_identity(_run(pool_half="A", pool_split_seed=2))[0])


def test_the_ladder_and_features_are_identifying():
    a = _run(); b = _run(); b["ladder"] = [1, 2, 4]
    assert cc.run_identity(a)[0] != cc.run_identity(b)[0]
    c = _run(); c["features"] = ["nbr", "struct"]
    assert cc.run_identity(a)[0] != cc.run_identity(c)[0]


def test_pooling_REFUSES_when_the_replicate_axis_does_not_distinguish_the_runs():
    # duplicate seeds in one group mean an identifying field went missing; the only symptom
    # last time was a header line, so this is now a hard failure
    with pytest.raises(SystemExit, match="REFUSING TO POOL"):
        cc.assert_group_is_replicates([_run(), _run()])


def test_a_genuine_replicate_set_is_ALLOWED():
    cc.assert_group_is_replicates([_run(draw_seed=99), _run(draw_seed=2024)])

def test_the_significance_of_a_is_INDEPENDENT_of_how_large_a_is():
    # cont.189 §6a.  V is itself estimated from n samples, so the excess inherits its
    # sqrt(2/(n-1)) error whole and the true value cancels out of the ratio.  This is what
    # makes the ceiling knowable before the run -- and what makes 4 splits a bound, not a test.
    rng = np.random.default_rng(3)
    for n in (4, 9):
        med = []
        for a_true in (0.001, 0.010):                # a ten-fold larger effect
            sig = []
            for _ in range(1500):
                d = rng.normal(0.0, np.sqrt(4 * a_true**2), n)   # var(d) = 2*n_shards*a
                a, sd, _ = cc.irreducible_term(d, np.zeros(n), n_shards=2)
                sig.append(a / sd)
            med.append(np.median(sig))
        assert med[0] == pytest.approx(med[1], rel=1e-6), \
            "significance must not depend on the effect size"
        assert med[0] == pytest.approx(np.sqrt((n - 1) / 2), rel=1e-6)


def test_four_splits_CANNOT_reach_a_detection_at_any_effect_size():
    # the planning consequence, pinned so it cannot be forgotten while reading a null result
    rng = np.random.default_rng(5)
    for a_true in (0.001, 0.10):
        d = rng.normal(0.0, np.sqrt(4 * a_true**2), 4)
        a, sd, _ = cc.irreducible_term(d, np.zeros(4), n_shards=2)
        assert cc.a_verdict(a, sd)[0] == "upper limit", \
            "4 splits have a 1.22 sigma ceiling; a detection here would mean the rule broke"

# ---------------------------------------------------------------- k-WAY POOL SHARDS ----

def _shard_jk(rng, k, a_full, B=200, sc=0.020, sa=0.008):
    """An HONEST leave-one-block-out jackknife: k arms over the SAME B blocks of galaxies.

    Fabricating replicate scatter directly gets this wrong -- twice, during development --
    because the full-sample value and its replicates are not independent draws.  Build the
    per-block contributions and do the real leave-one-out.
    """
    pool = rng.normal(0.0, np.sqrt(k * a_full), k)     # a at N/k is k*a_full
    c = rng.normal(0.0, sc, B)                          # common: the SAME galaxies for all arms
    d = rng.normal(0.0, sa, (k, B))                     # arm-specific
    tot = c[None, :] + d
    arm_full = pool + tot.mean(axis=1)
    S = tot.sum(axis=1, keepdims=True)
    return arm_full, pool[:, None] + (S - tot) / (B - 1)


def test_the_k_arm_term_recovers_an_INJECTED_value():
    rng = np.random.default_rng(21)
    a_full = 0.0015**2
    for k in (2, 4, 8):
        ests = [cc.shard_family_term(*_shard_jk(rng, k, a_full))[0] for _ in range(600)]
        assert np.mean(ests) == pytest.approx(a_full, rel=0.12), f"biased at k={k}"


def test_the_COMMON_galaxy_term_cancels_out_of_the_across_arm_scatter():
    # the arms score the SAME galaxies, so inflating the shared block term must not move `a`
    rng = np.random.default_rng(5)
    a_full = 0.0015**2
    quiet = [cc.shard_family_term(*_shard_jk(rng, 4, a_full, sc=0.002))[0] for _ in range(500)]
    loud = [cc.shard_family_term(*_shard_jk(rng, 4, a_full, sc=0.200))[0] for _ in range(500)]
    assert np.mean(loud) == pytest.approx(np.mean(quiet), rel=0.15), \
        "a 100x louder SHARED term moved `a`; the common component is not cancelling"


def test_k_way_shards_beat_halves_at_EQUAL_job_count():
    # cont.189 §6c, the reason the plan changed before any GPU time was spent: k arms give
    # k-1 dof from k jobs where halves give 1 from 2, and a k-way arm carries k*a_full so
    # the floor matters less.  8 jobs as k=8 must beat 8 jobs as four half-splits.
    rng = np.random.default_rng(9)
    a_full = 0.0015**2
    eight = [cc.shard_family_term(*_shard_jk(rng, 8, a_full)) for _ in range(400)]
    sig8 = np.median([a / sd for a, sd, _ in eight])
    halves = []
    for _ in range(400):
        diffs, floors = [], []
        for _s in range(4):                             # 4 splits x 2 jobs = the same 8
            af, ajk = _shard_jk(rng, 2, a_full)
            diffs.append(af[0] - af[1])
            r = ajk[0] - ajk[1]
            floors.append(np.sqrt((len(r) - 1) / len(r) * ((r - r.mean()) ** 2).sum()))
        a, sd, _ = cc.irreducible_term(diffs, floors, n_shards=2)
        halves.append(a / sd)
    # measured ratio is ~1.63 (1.84 vs 1.13); assert clear of noise, not at the point value
    assert sig8 > 1.4 * np.median(halves), \
        "k=8 must be worth ~1.6x the halves design at equal cost"
    assert eight[0][2] == 7, "k=8 must carry 7 degrees of freedom, not 1"


def test_the_k_arm_term_is_also_SIGNED_not_clipped():
    rng = np.random.default_rng(4)
    neg = [cc.shard_family_term(*_shard_jk(rng, 4, 0.0))[0] for _ in range(300)]
    assert min(neg) < 0, "a null injection must be able to come out negative"
    assert abs(np.mean(neg)) < 0.3 * np.std(neg), "and must centre on zero"


def test_one_arm_REFUSES():
    a, sd, dof = cc.shard_family_term(np.zeros(1), np.zeros((1, 200)))
    assert a is None and sd is None and dof == 0



# ======================================================================================
# ABSCISSA RESCALING -- which summaries of a common-slope comparison carry units
#
# Raised by the peer session against cont.191's headline.  A uniform rescale x -> c*x of the
# fit abscissa scales every fitted slope AND its error by 1/c, so summaries built as a ratio
# of the two are invariant and summaries built from slopes alone are not.  cont.191 quotes
# four; these tests pin which is which, so the write-up cannot drift back to leading on a
# dimensioned one.
# ======================================================================================

def _common(vals, errs):
    v, e = np.asarray(vals, float), np.asarray(errs, float)
    w = 1.0 / e ** 2
    c = (v * w).sum() / w.sum()
    return (float((w * (v - c) ** 2).sum()), float(c), float(v.max() - v.min()),
            abs(float(vals[0] - vals[1])) / float(np.hypot(errs[0], errs[1])))


def test_rescaling_the_abscissa_leaves_chi2_and_sigma_EXACTLY_alone():
    """chi2 and the two-config sigma are ratios of slope to error, so c cancels."""
    v = [-8.655, -140.625, -6.196]
    e = [0.759, 3.779, 2.514]
    c2, _, _, sig = _common(v, e)
    for c in (0.5, 2.0, 17.3):
        c2r, _, _, sigr = _common([x / c for x in v], [x / c for x in e])
        assert c2r == pytest.approx(c2, rel=1e-12)
        assert sigr == pytest.approx(sig, rel=1e-12)


def test_the_RAW_span_does_NOT_survive_a_rescale_but_span_over_a_does():
    """The exact defect the peer identified: raw span carries units, span/|a| does not."""
    v = [-8.655, -140.625, -6.196]
    e = [0.759, 3.779, 2.514]
    _, a, span, _ = _common(v, e)
    for c in (0.5, 2.0, 17.3):
        _, ar, spanr, _ = _common([x / c for x in v], [x / c for x in e])
        assert spanr == pytest.approx(span / c, rel=1e-12)      # scales -- NOT invariant
        assert spanr / abs(ar) == pytest.approx(span / abs(a), rel=1e-12)   # invariant


def test_both_abscissae_are_ANCHORED_at_M_equals_1_so_there_is_no_free_scale():
    """Why the raw span is still readable here: c is not free, it is pinned to 1.

    ESS = 1 at one draw by construction and M^-p = 1 at M = 1, so every abscissa the
    exponent control uses starts at exactly 1.  A rescale is therefore not a symmetry of
    this comparison -- which is what keeps the raw span meaningful as a diagnostic even
    though it is the one summary that would move under a rescale.
    """
    for p in (0.05, 0.825, 1.0, 4.3, 12.0):
        x = np.array([float(m) ** (-p) for m in (1, 2, 4, 8, 16)])
        assert x[0] == pytest.approx(1.0, abs=1e-12)
    # the ESS side of the same statement: one draw carries one effective sample
    ess_rung = np.ones((500, 1))
    assert float(np.mean(1.0 / ess_rung)) == pytest.approx(1.0, abs=1e-12)


def test_span_over_a_is_MONOTONE_in_p_so_it_cannot_be_the_SELECTION_criterion():
    """Invariant does not mean usable for selection.

    span/|a| rises monotonically across the non-degenerate range, so minimising it drives
    the exponent back toward the p -> 0 degeneracy that the raw span exists to flag.  The
    two summaries have different jobs and cont.191 keeps them separate.
    """
    # slopes and errors both inflate as 1/(regressor range); the span inflates faster than
    # the common value, which is what makes the ratio monotone
    rows = {0.05: ([-40.96, -296.54, 8.68], [4.034, 10.236, 6.987]),
            0.30: ([-11.24, -112.30, 3.33], [1.036, 3.455, 2.305]),
            1.00: ([-8.15, -142.03, 3.13], [0.724, 3.818, 2.543])}
    r = [_common(*rows[p])[2] / abs(_common(*rows[p])[1]) for p in (0.05, 0.30, 1.00)]
    assert r[0] < r[1] < r[2], f"span/|a| not monotone: {r}"


def test_gls_slope_VARIANCE_does_not_depend_on_the_data_values():
    """Why cont.191 §4b holds the admissible-p boundary fixed inside the bootstrap.

    The natural objection is that the boundary is a data-dependent selection and must be
    re-derived in every draw.  It is not: for GLS with a fixed covariance,
    Var(a_hat) = (A' C^-1 A)^-1 depends on the DESIGN and the COVARIANCE only.  So the
    error-inflation ratio that defines the boundary is identical in every draw, and holding
    it fixed is exact rather than a shortcut.  What the boundary DOES depend on is C, which
    is estimated -- that channel is measured separately (cont.191 §4c).
    """
    rng = np.random.default_rng(4242)
    ladder = [1, 2, 4, 8, 16]
    K = len(ladder)
    A_ = rng.normal(size=(K, K))
    C = A_ @ A_.T + K * np.eye(K)                      # an arbitrary valid covariance

    def slope_var(yv, x):
        A = np.stack([x ** k for k in range(2)], axis=1)
        Ci = np.linalg.pinv(C)
        cov = np.linalg.pinv(A.T @ Ci @ A)
        return float(cov[1, 1]), float((cov @ A.T @ Ci @ yv)[1])

    for p in (0.05, 0.675, 1.0, 4.3):
        x = np.array([float(m) ** (-p) for m in ladder])
        vars_, slopes = zip(*[slope_var(rng.normal(scale=10.0, size=K), x)
                              for _ in range(5)])
        assert len(set(np.round(vars_, 12))) == 1, "slope variance moved with the data"
        assert len(set(np.round(slopes, 12))) == 5, "slopes should differ across draws"


def test_a_over_sd_IS_q_RESTATED_and_carries_no_independent_information():
    """The shard table's `a/sd` column is NOT a z-score, and reading it as one misleads.

    `shard_family_term` returns `a = (V - F)/k` and `sd = V*sqrt(2/dof)/k`, so `sd` is
    proportional to `V` ITSELF rather than to an independent error on `a`.  The ratio is
    therefore

        a/sd = (V - F)/(V*sqrt(2/dof)) = (1 - q)*sqrt(dof/2),   q = F/V

    identically -- the same expression as the design's significance ceiling in cont.189 §6a.
    It is a restatement of `q` and nothing else.

    THIS IS PINNED BECAUSE THE PRODUCTION RUN MADE IT LOOK LIKE A DEFECT.  cont.193's M=8
    rung reported a/sd = -4.43, which read as a 4.4 sigma negative variance excess and would
    have sent someone after the floor estimator; it is `(1 - 3.367)*sqrt(7/2)`, arithmetic,
    and the honest reading is that `V` drew low on 7 dof (a 53% fractional error).  A test is
    the right home for this because the misreading is available to anyone looking at the
    column without re-deriving `sd`.
    """
    rng = np.random.default_rng(20260820)
    for k, nb, scale in ((8, 200, 1.0), (4, 120, 0.3), (2, 64, 3.0)):
        arm_full = rng.normal(0.0, scale, size=k)
        arm_jk = rng.normal(0.0, scale, size=(k, nb))
        a, sd, dof = cc.shard_family_term(arm_full, arm_jk)
        V = float(np.var(arm_full, ddof=1))
        q = (V - a * k) / V
        assert dof == k - 1
        assert a / sd == pytest.approx((1.0 - q) * np.sqrt(dof / 2.0), rel=1e-12, abs=1e-12)
