"""Lagrangian score inference -- the executable form of `INFERENCE.md` §5C.

`score_inference.py` implements §5B, the EULERIAN form: shear acts on the prior density,
so the generator `u_k = -(v . grad log p_0 + div v)` needs `grad log p_0` in closed
differentiable form.  That requirement is what caps that module at a 2-D isotropic shape
prior with a hand-built radial spline.

§5C is the same estimator reparametrized (§5C.1: "(1.1) and (5.4) are the same integral").
Shear acts on the prior SAMPLES instead, so `p_0` is needed only as a sampler:

    phi_k(gamma) = log [ p_flow( xhat_i | S_gamma z_k ) * P_det( S_gamma z_k ) ]   (5.5b)
    ut_k  = phi_k'(0)                       the score channel
    dUt_k = phi_k''(0)                      the Louis channel -- SAME curve
    w_k  propto exp( phi_k(0) )             posterior weights (5.2); cut cancels
    s_i   = E_w[ phi' ],   I_i = -E_w[ phi'' ] - Var_w( phi' )                     (5.5b)

    P(gamma) = E_{p_0(z)}[ Ppass( S_gamma z ) * Pdet( S_gamma z ) ]                (5.5c)
    <s>_sel  =  (log P)'(0),    I_sel = -(log P)''(0)

    ghat = ( sum_i s_i - N <s>_sel ) / ( sum_i I_i - N I_sel )                     (5.8)

Everything reduces to first and second derivatives of ONE scalar curve per (object, node)
and ONE for the population, which is the entire content of §5C.  This module supplies
those derivatives and the assembly; it knows nothing about flows, catalogues or shear
maps -- the caller provides `phi` as a callable of gamma.

Three traps this module is built to keep the caller out of, each with a live test:

  `P_det` carries gamma too.  In the Eulerian form `P_det(x, n)` is a fixed function of
      truth and only weights.  Here gamma moves the sample THROUGH it, so it must be
      inside `phi`.  `MATH.md` §7(a) measures the omitted channel at 17-760% of the
      score, reversing its sign at one of four test points.

  `P_pass` does NOT carry gamma in `phi`.  By (5.2) the cut evaluates to
      `W(xhat_i) = 1` for a galaxy in hand and cancels from the per-object posterior.
      It reappears only in the population curve (5.5c).  Detection does not cancel; the
      cut does.

  Centring is not optional, and it applies to BOTH moments.  A truncated sample has
      `E_0[s] != 0`.  Dropping `I_sel` while keeping `<s>_sel` is the classic error:
      A.7 evaluates it at `I_sel/I = 2/pi` for a cut at the median, i.e. `m = -64%`.
      For a spin-2 shear the numerator term averages to zero by orientation and `I_sel`
      is the ONLY surviving selection term (§5B.2), so it is the one that matters here.

Finite differences rather than a JVP: §5C.5 point 1 blesses "a central second difference
in gamma at fixed node", because holding the node fixed is common random numbers -- the
structural advantage this parametrization has over the data side.  Both derivatives come
off the same curve, so no `grad^2 log p_flow` is ever formed.
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------------------
# the scalar curve: phi(0), phi'(0), phi''(0)
# --------------------------------------------------------------------------------------

def curve_derivatives(f, delta=0.01, richardson=True):
    """Value, first and second derivative at 0 of a scalar curve `f(t)`.

    `f` maps a float to an array of any shape (one entry per object-node, or a scalar for
    the population curve).  Central differences, so both derivatives are O(delta^2)
    accurate; `richardson` extrapolates with `(4 g(delta/2) - g(delta))/3` and removes
    that term, at the cost of two extra evaluations.

    The curve must be evaluated with COMMON RANDOM NUMBERS across `t` -- same node, same
    Monte-Carlo draws inside any integral.  Otherwise `f` is a noisy step function and its
    second derivative is meaningless; `I_sel` is the term that dies first (§5C.5 point 2).
    """
    f0 = np.asarray(f(0.0), dtype=np.float64)

    def raw(d):
        fp = np.asarray(f(+d), dtype=np.float64)
        fm = np.asarray(f(-d), dtype=np.float64)
        return (fp - fm) / (2.0 * d), (fp - 2.0 * f0 + fm) / d ** 2

    d1, d2 = raw(delta)
    if richardson:
        h1, h2 = raw(0.5 * delta)
        d1 = (4.0 * h1 - d1) / 3.0
        d2 = (4.0 * h2 - d2) / 3.0
    return f0, d1, d2


# --------------------------------------------------------------------------------------
# per-object block
# --------------------------------------------------------------------------------------

def posterior_weights(phi0, log_prior=None):
    """Self-normalised `w_k propto exp(phi_k(0)) * p_0(z_k)`, row-wise over nodes.

    `phi0` is `(N, K)`: `log[p_flow(xhat_i | z_k) P_det(z_k)]` at gamma = 0, up to any
    per-object constant (it cancels).  `log_prior` is `(K,)` and is needed only for a
    QUADRATURE node bank, where nodes sit on a grid rather than being drawn from `p_0`;
    for a sampled bank the prior is already carried by the sampling and this must be None
    (§5C.5, "z_k ~ p_0(z)").  Passing it for a sampled bank double-counts the prior.
    """
    ll = np.asarray(phi0, dtype=np.float64)
    if log_prior is not None:
        ll = ll + np.asarray(log_prior, dtype=np.float64)[None, :]
    ll = ll - ll.max(axis=1, keepdims=True)
    w = np.exp(ll)
    return w / w.sum(axis=1, keepdims=True)


def score_and_information(phi0, dphi, ddphi, log_prior=None):
    """`(s_i, I_i)` from the three curve quantities -- (5.5b).

        s_i = E_w[phi'],    I_i = -E_w[phi''] - Var_w(phi')

    All inputs `(N, K)`.  Individual `I_i` may be negative; only the SUM is the Fisher
    information, which is why (5.8) sums numerator and denominator separately instead of
    averaging per-object ratios.
    """
    w = posterior_weights(phi0, log_prior)
    s = np.sum(w * dphi, axis=1)
    mean_dd = np.sum(w * ddphi, axis=1)
    var_d = np.sum(w * dphi ** 2, axis=1) - s ** 2
    return s, -mean_dd - var_d


# --------------------------------------------------------------------------------------
# population block
# --------------------------------------------------------------------------------------

def selection_terms(log_p0, dlog_p, ddlog_p):
    """`(<s>_sel, I_sel)` from the population curve `log P(keep | gamma)` -- (5.5c).

    A single number each, subtracted from EVERY galaxy in the catalogue: the double
    integral in `P(keep | gamma)` leaves no `xhat_i` and no `z`, so it comes out of the
    posterior expectation entirely (`MATH.md` §4).
    """
    del log_p0  # value is not used; kept in the signature to mirror curve_derivatives
    return float(dlog_p), float(-ddlog_p)


def population_curve(pi_of_gamma, weights=None, delta=0.01, richardson=True):
    """`(<s>_sel, I_sel)` directly from `Pi(S_gamma z)` evaluated on the node bank.

    `pi_of_gamma(t)` must return the per-node `Ppass * Pdet` at shear `t`, on the SAME
    nodes for every `t` -- that is the common-random-numbers requirement, and it is what
    makes the second derivative meaningful.  `weights` are normalised prior weights over
    the nodes, needed only for a QUADRATURE bank; leave None for a bank sampled from
    `p_0`, where `P(gamma) = mean_k Pi_k(gamma)` already carries the prior.
    """
    w = None if weights is None else np.asarray(weights, dtype=np.float64)

    def log_p(t):
        pi = np.asarray(pi_of_gamma(t), dtype=np.float64)
        return np.log(np.mean(pi) if w is None else float(np.sum(w * pi) / np.sum(w)))

    return selection_terms(*curve_derivatives(log_p, delta, richardson))


# --------------------------------------------------------------------------------------
# the estimator
# --------------------------------------------------------------------------------------

def shear_estimate_louis(s, info, s_sel=0.0, i_sel=0.0):
    """(5.8) -- slope over curvature, both centred by the population terms."""
    s = np.asarray(s, dtype=np.float64)
    info = np.asarray(info, dtype=np.float64)
    n = s.size
    num = s.sum() - n * float(s_sel)
    den = info.sum() - n * float(i_sel)
    return num / den


def shear_estimate_bartlett(s, s_sel=0.0):
    """(5.9) -- the same numerator over the variance of the centred score.

    Needs no second derivatives at all: Bartlett applied to `p_keep` gives its information
    as the variance of its own score.  Legitimate, and a free cross-check on (5.8) --
    §5B.2's consistency test is exactly the statement that the two denominators agree in
    expectation.

    Two cautions, both from §5C.5.  Centring must be applied in BOTH places: dropping
    `s_sel` from the denominator as well as the numerator leaves an O(1) bias, since a
    truncated sample has `E_0[s] != 0`.  And the two denominators, equal AT gamma = 0 by
    the information equality, drift apart away from it by (5.9b); use this once as a cheap
    check, or iterate it, but not once and uniterated on a tight bias budget.
    """
    c = np.asarray(s, dtype=np.float64) - float(s_sel)
    return c.sum() / np.sum(c ** 2)


def denominator_consistency(s, info, s_sel=0.0, i_sel=0.0):
    """§5C.5 cross-check (iii): the two denominators, per object, should agree.

    Returns `(bartlett, louis, ratio)` with
    `bartlett = N^-1 sum (s_i - <s>_sel)^2` and `louis = N^-1 sum I_i - I_sel`.
    They are equal at gamma = 0 by the information equality, so `ratio` near 1 is the
    test; the residual is the (5.9b) drift plus Monte-Carlo error.
    """
    s = np.asarray(s, dtype=np.float64)
    info = np.asarray(info, dtype=np.float64)
    bart = float(np.mean((s - float(s_sel)) ** 2))
    lou = float(np.mean(info) - float(i_sel))
    return bart, lou, bart / lou


def drift_5_9b(s, info, s_sel=0.0, i_sel=0.0, gamma=0.0):
    """(5.9b): the predicted `m_(5.9) - m_(5.8)` from the same per-object arrays.

        -gamma * ( mu_3 - Cov_0(I_keep, s_keep) ) / I_keep

    `mu_3` is the third central moment of the kept-sample score.  The covariance term is
    not optional -- it vanishes only when `I_i` is identical for every object, a Gaussian
    accident; `MATH.md` §7(c) measures dropping it as a 60% overprediction.
    """
    sk = np.asarray(s, dtype=np.float64) - float(s_sel)
    ik = np.asarray(info, dtype=np.float64) - float(i_sel)
    mu3 = float(np.mean((sk - sk.mean()) ** 3))
    cov = float(np.mean((ik - ik.mean()) * (sk - sk.mean())))
    return -float(gamma) * (mu3 - cov) / float(np.mean(ik))
