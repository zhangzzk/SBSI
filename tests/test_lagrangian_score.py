"""`INFERENCE.md` §5C against the closed forms of A.7, and the two traps of `MATH.md` §7.

A.7 is the whole document in a Gaussian toy: truth `z ~ N(0, tau^2)`, shear as a shift
`S_g z = z + g`, one-dimensional flow `p(y | x) = N(x, sigma^2)`, `nu^2 = tau^2 + sigma^2`.
Every §5C quantity then has a closed form, so these are exact assertions rather than
regression values:

    s_i = y_i / nu^2                I_i = 1 / nu^2
    <s>_sel = lambda / nu           I_sel = lambda (lambda - a) / nu^2
    I - I_sel = Var[y | y > c] / nu^4

with `a = c/nu` and `lambda = phi(a)/Phibar(a)` the inverse Mills ratio.  Note the
Lagrangian route reaches them WITHOUT ever differentiating the prior: `phi_k(g) =
log N(y_i; z_k + g, sigma^2)` has `phi' = (y-z)/sigma^2` and `phi'' = -1/sigma^2`, and the
posterior average of those two is what produces `y/nu^2` and `1/nu^2`.

The node bank here is a QUADRATURE grid, so `log_prior` / `weights` are passed explicitly.
A bank sampled from `p_0` (§5C.5) carries the prior already and must not pass them.
"""

import numpy as np
import pytest
from scipy.integrate import trapezoid
from scipy.stats import norm

from sbs_shear.lagrangian_score import (
    curve_derivatives,
    denominator_consistency,
    drift_5_9b,
    population_curve,
    posterior_weights,
    score_and_information,
    shear_estimate_bartlett,
    shear_estimate_louis,
)

TAU, SIGMA = 1.0, 0.7
NU2 = TAU ** 2 + SIGMA ** 2
NU = np.sqrt(NU2)
CUT = 0.3                      # `MATH.md` §7(b) uses this cut on y


@pytest.fixture(scope="module")
def bank():
    """Quadrature node bank over the intrinsic truth `z`, plus its prior weights."""
    z = np.linspace(-8.0 * TAU, 8.0 * TAU, 1601)
    log_prior = norm.logpdf(z, 0.0, TAU)
    w = np.exp(log_prior - log_prior.max())
    return z, log_prior, w / w.sum()


def phi_curves(y, z, delta=0.01, det=None):
    """`(phi(0), phi'(0), phi''(0))` for the A.7 flow, optionally with a detection factor.

    `det(x)` is `P(detected | truth = x)`.  It is inside the curve because gamma moves the
    sample THROUGH it (§5C.2) -- the whole point of `test_pdet_channel_is_not_optional`.
    """
    def f(t):
        x = z[None, :] + t
        out = norm.logpdf(y[:, None], x, SIGMA)
        return out if det is None else out + np.log(det(x))

    return curve_derivatives(f, delta=delta)


# --------------------------------------------------------------------------------------
# the stencil itself
# --------------------------------------------------------------------------------------

def test_curve_derivatives_on_an_analytic_curve():
    """Richardson-extrapolated central differences on `log(2 + sin t)`."""
    f = lambda t: np.log(2.0 + np.sin(t))
    v, d1, d2 = curve_derivatives(f, delta=0.05)
    assert v == pytest.approx(np.log(2.0), abs=1e-12)
    assert float(d1) == pytest.approx(0.5, abs=1e-7)          # cos0/(2+sin0)
    assert float(d2) == pytest.approx(-0.25, abs=1e-7)        # -sin0/2 - (cos0/2)^2


# --------------------------------------------------------------------------------------
# A.7: the per-object block
# --------------------------------------------------------------------------------------

def test_score_and_information_match_a7_closed_form(bank):
    """`s_i = y_i/nu^2` and `I_i = 1/nu^2`, from the Lagrangian curve alone."""
    z, log_prior, _ = bank
    y = np.array([-1.5, -0.5, 0.0, 0.4, 1.2, 2.0])
    p0, d1, d2 = phi_curves(y, z)
    s, info = score_and_information(p0, d1, d2, log_prior=log_prior)

    assert s == pytest.approx(y / NU2, rel=1e-6, abs=1e-9)
    assert info == pytest.approx(np.full_like(y, 1.0 / NU2), rel=1e-6)


def test_posterior_weights_reproduce_the_analytic_posterior(bank):
    """Fisher's identity needs the right posterior: `N(y tau^2/nu^2, tau^2 sigma^2/nu^2)`."""
    z, log_prior, _ = bank
    y = np.array([-0.5, 1.2])
    p0, _, _ = phi_curves(y, z)
    w = posterior_weights(p0, log_prior=log_prior)

    mean = w @ z
    var = w @ z ** 2 - mean ** 2
    assert mean == pytest.approx(y * TAU ** 2 / NU2, rel=1e-6)
    assert var == pytest.approx(np.full_like(y, TAU ** 2 * SIGMA ** 2 / NU2), rel=1e-5)


def test_sampled_bank_needs_no_log_prior(bank):
    """§5C.5: `z_k ~ p_0` carries the prior, so passing `log_prior` would double-count."""
    z, log_prior, _ = bank
    y = np.array([0.4, 1.2])
    rng = np.random.default_rng(0)
    zs = rng.normal(0.0, TAU, 200_000)
    p0, d1, d2 = phi_curves(y, zs)
    s, info = score_and_information(p0, d1, d2)               # no log_prior -- sampled

    assert s == pytest.approx(y / NU2, rel=2e-2)
    assert info == pytest.approx(np.full_like(y, 1.0 / NU2), rel=2e-2)

    wrong, _ = score_and_information(p0, d1, d2, log_prior=norm.logpdf(zs, 0.0, TAU))
    assert np.all(np.abs(wrong - y / NU2) > 0.05)             # double-counted prior


# --------------------------------------------------------------------------------------
# A.7: the population block
# --------------------------------------------------------------------------------------

def test_selection_terms_match_a7_closed_form(bank):
    """`<s>_sel = lambda/nu` and `I_sel = lambda(lambda-a)/nu^2` -- (A.7), (A.7b)."""
    z, _, w = bank
    # P_pass(z; g) = P(y > c | truth z+g) = Phibar((c - z - g)/sigma).  Same nodes for
    # every g, which is the common-random-numbers requirement of §5C.5 point 2.
    s_sel, i_sel = population_curve(lambda t: norm.sf((CUT - z - t) / SIGMA), weights=w)

    a = CUT / NU
    lam = norm.pdf(a) / norm.sf(a)
    assert s_sel == pytest.approx(lam / NU, rel=1e-6)
    assert i_sel == pytest.approx(lam * (lam - a) / NU2, rel=1e-5)


def test_selected_information_is_the_truncated_normal_variance(bank):
    """A.7b's own check: `I - I_sel = Var[y | y > c] / nu^4`."""
    z, _, w = bank
    _, i_sel = population_curve(lambda t: norm.sf((CUT - z - t) / SIGMA), weights=w)

    a = CUT / NU
    lam = norm.pdf(a) / norm.sf(a)
    var_trunc = NU2 * (1.0 + a * lam - lam ** 2)              # truncated-normal variance
    assert (1.0 / NU2) - i_sel == pytest.approx(var_trunc / NU2 ** 2, rel=1e-5)


# --------------------------------------------------------------------------------------
# the estimator under a cut -- `MATH.md` §7(b)
# --------------------------------------------------------------------------------------

def kept_sample(truth, n=20_000):
    """Quantile-stratified draw from `N(truth, nu^2)` truncated to `y > CUT`.

    Stratifying instead of sampling makes every population average quadrature-accurate at
    small `n`, so these tests assert converged values rather than Monte-Carlo ones.  The
    sweep behind the constants below (node count 601 -> 2401, span 8 -> 10, `delta`
    0.01 -> 0.002, `n` 20k -> 200k) moves them by less than 0.03%.
    """
    u = (np.arange(n) + 0.5) / n
    lo = norm.cdf((CUT - truth) / NU)
    return truth + NU * norm.ppf(lo + u * (1.0 - lo))


def a7_selection_closed_form():
    a = CUT / NU
    lam = norm.pdf(a) / norm.sf(a)
    return lam / NU, lam * (lam - a) / NU2


def test_estimator_under_a_cut_matches_the_closed_form(bank):
    """(5.8) and (5.9) reproduce the estimator built from A.7's analytic ingredients.

    This is the strongest available check: the reference uses `s = y/nu^2`, `I = 1/nu^2`
    and (A.7)/(A.7b) directly, with no node bank, no finite differences and no quadrature,
    so agreement means the whole §5C machinery is exact on this model.

    `m` is not zero because (5.8) is a SINGLE Newton step from gamma = 0 (`MATH.md` A2);
    at gamma = 0.05 the exact step lands at +1.224% for (5.8) and -1.303% for (5.9).
    """
    z, log_prior, w = bank
    truth = 0.05
    y = kept_sample(truth)

    p0, d1, d2 = phi_curves(y, z)
    s, info = score_and_information(p0, d1, d2, log_prior=log_prior)
    s_sel, i_sel = population_curve(lambda t: norm.sf((CUT - z - t) / SIGMA), weights=w)

    s_ref, i_ref = y / NU2, np.full_like(y, 1.0 / NU2)
    s_sel_ref, i_sel_ref = a7_selection_closed_form()

    louis = shear_estimate_louis(s, info, s_sel, i_sel)
    bart = shear_estimate_bartlett(s, s_sel)
    assert louis == pytest.approx(
        shear_estimate_louis(s_ref, i_ref, s_sel_ref, i_sel_ref), rel=1e-5)
    assert bart == pytest.approx(shear_estimate_bartlett(s_ref, s_sel_ref), rel=1e-5)
    assert louis / truth - 1.0 == pytest.approx(+0.01224, abs=2e-4)
    assert bart / truth - 1.0 == pytest.approx(-0.01303, abs=2e-4)


def test_centring_is_not_optional_in_either_moment(bank):
    """Dropping `<s>_sel`, or dropping `I_sel` alone, are both O(1) errors."""
    z, log_prior, w = bank
    truth = 0.05
    y = kept_sample(truth)
    p0, d1, d2 = phi_curves(y, z)
    s, info = score_and_information(p0, d1, d2, log_prior=log_prior)
    s_sel, i_sel = population_curve(lambda t: norm.sf((CUT - z - t) / SIGMA), weights=w)

    # No centring at all: returns the truncated mean E[y|y>c]/nu^2 instead of the shear.
    # `MATH.md` §7(b) measures m = +1770%; the converged value is +1773%.
    uncentred = shear_estimate_bartlett(s, 0.0)
    assert uncentred / truth - 1.0 == pytest.approx(17.73, rel=0.01)

    # Numerator centred, denominator left alone -- the error A.7 closes with, worth
    # m = -I_sel/I exactly.  At a cut on the median that is -2/pi; here it is -68.7%.
    half = shear_estimate_louis(s, info, s_sel, 0.0)
    assert half / truth - 1.0 == pytest.approx(-i_sel * NU2, rel=0.01)
    assert half / truth - 1.0 < -0.5


def test_denominator_consistency_and_the_5_9b_drift(bank):
    """§5C.5 cross-check (iii), and (5.9b) predicting the gap between the two estimators."""
    z, log_prior, w = bank
    s_sel, i_sel = population_curve(lambda t: norm.sf((CUT - z - t) / SIGMA), weights=w)

    # At gamma = 0 the two denominators are equal by the information equality.
    p0, d1, d2 = phi_curves(kept_sample(0.0), z)
    s0, info0 = score_and_information(p0, d1, d2, log_prior=log_prior)
    _, _, ratio = denominator_consistency(s0, info0, s_sel, i_sel)
    assert ratio == pytest.approx(1.0, rel=0.01)

    # Away from it they drift, and (5.9b) says by how much.
    truth = 0.05
    p0, d1, d2 = phi_curves(kept_sample(truth), z)
    s, info = score_and_information(p0, d1, d2, log_prior=log_prior)
    measured = (shear_estimate_bartlett(s, s_sel) / truth
                - shear_estimate_louis(s, info, s_sel, i_sel) / truth)
    assert drift_5_9b(s, info, s_sel, i_sel, truth) == pytest.approx(measured, rel=0.02)


# --------------------------------------------------------------------------------------
# the detection channel -- `MATH.md` §7(a)
# --------------------------------------------------------------------------------------

def test_pdet_channel_is_not_optional(bank):
    """`P_det` carries gamma; keeping it in the weights but not differentiating it is O(1).

    `MATH.md` §7(a) reports the omitted channel at 17-760% of the score and reversing its
    sign at one of its four test points.  Its exact digits come from an unspecified
    sigmoid, so this reproduces the CLAIM with a stated sigmoid: the full curve matches
    the direct `d_gamma log A` to quadrature precision, and the truncated one does not.
    """
    z, log_prior, _ = bank
    det = lambda x: 1.0 / (1.0 + np.exp(-x))
    y = np.array([-0.5, 0.4, 1.2, 2.0])

    # reference: differentiate log A = log Integral pi(z) L(y|z+g) Pdet(z+g) dz directly
    prior = np.exp(log_prior)

    def log_a(t):
        x = z[None, :] + t
        integrand = prior[None, :] * norm.pdf(y[:, None], x, SIGMA) * det(x)
        return np.log(trapezoid(integrand, z, axis=1))

    _, direct, _ = curve_derivatives(log_a, delta=0.01)

    p0, d1, d2 = phi_curves(y, z, det=det)
    s, _ = score_and_information(p0, d1, d2, log_prior=log_prior)
    assert s == pytest.approx(direct, rel=1e-6, abs=1e-9)

    # The trap: same weights (which DO include Pdet), derivative of the flow only.
    w_full = posterior_weights(p0, log_prior=log_prior)
    _, d1_flow, _ = phi_curves(y, z)                          # no det inside the curve
    dropped = np.sum(w_full * d1_flow, axis=1)
    assert np.max(np.abs(dropped - direct)) > 0.1             # O(1), not a refinement
    assert np.any(np.sign(dropped) != np.sign(direct))        # and it flips a sign
