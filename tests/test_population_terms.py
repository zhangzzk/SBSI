"""Certify §5B's population block against `INFERENCE.md` A.7's closed forms.

A.7 is the one model where every object in the document is elementary: truth
`x ~ N(0, tau^2)`, the parameter SHIFTS the prior (`p_theta = N(theta, tau^2)`, the additive
stand-in for `S_gamma`), and the measurement is `y = x + noise`, `p(y|x) = N(x, sigma^2)`.
Writing `nu^2 = tau^2 + sigma^2`:

    s_i = y_i / nu^2                    I_i = 1 / nu^2
    <s>_sel = phi(a) / (nu Phibar(a))   I_sel = lambda (lambda - a) / nu^2
    with a = c/nu and lambda = phi(a)/Phibar(a) for a cut y > c.

These tests drive the PRODUCTION functions (`scores_from_loglike`, `population_terms`,
`full_shear_estimate`) through a node bank that implements the same interface as
`ShapeScoreNodes` but carries A.7's shift family instead of the Mobius shear.  That is
deliberate: a reimplementation of the algebra would test nothing.  The toy is made 2-D by
taking two independent copies, so the 2-vector / 2x2-matrix shapes the real code uses are
exercised, and the second (uncut) component doubles as a null.

The headline test is `test_omitting_i_sel_costs_64_percent`: A.7 states that at a cut on
the median an estimator which centres the score but leaves the denominator alone reports
`m = -64%`.  That is the defect `WORKLOG.md` cont.164 lists against this module, and the
test reproduces the number from the production code.

Everything runs on a grid in float32 (the production dtype), so tolerances are ~1e-3
relative rather than machine precision.
"""

import numpy as np
import pytest
from scipy.stats import norm

from sbsi.score_inference import (
    full_shear_estimate, population_terms, scores_from_loglike,
)

TAU, SIGMA = 1.0, 1.0
NU2 = TAU ** 2 + SIGMA ** 2
NU = np.sqrt(NU2)


class ShiftNodes:
    """A.7's shift family on a flat 2-D grid, in `ShapeScoreNodes`' interface.

    `p_theta(x) = N(theta, tau^2 I)`, so
        u_a      = d_theta_a log p_theta(x)|_0 = x_a / tau^2
        du[a,b]  = d_theta_b u_a               = -delta_ab / tau^2
    """

    def __init__(self, span=6.0, n=161, tau=TAU, info_delta=0.01):
        ax = np.linspace(-span * tau, span * tau, n)
        g1, g2 = np.meshgrid(ax, ax, indexing="ij")
        self.grid = np.stack([g1.ravel(), g2.ravel()], axis=1)
        self.tau = float(tau)
        self.info_delta = float(info_delta)
        self.support = np.ones(len(self.grid), dtype=bool)
        self.log_prior = self._logp(self.grid, (0.0, 0.0))
        self.u = self.grid / tau ** 2
        self.du = np.tile((-np.eye(2) / tau ** 2)[None], (len(self.grid), 1, 1))
        d = self.info_delta
        self.shifted = {}
        for key, th in ((("g1", +1), (+d, 0.0)), (("g1", -1), (-d, 0.0)),
                        (("g2", +1), (0.0, +d)), (("g2", -1), (0.0, -d))):
            self.shifted[key] = (self._logp(self.grid, th),
                                 (self.grid - np.asarray(th)) / tau ** 2)

    def _logp(self, x, th):
        d = x - np.asarray(th, float)
        return -0.5 * np.sum(d ** 2, axis=1) / self.tau ** 2

    def prior_weights(self):
        w = np.exp(self.log_prior - self.log_prior.max())
        return w / w.sum()


def loglike(y, nodes, sigma=SIGMA):
    """`log p(y_i | x_k)` up to a per-row constant, which cancels.  `(N,G)`."""
    d = y[:, None, :] - nodes.grid[None, :, :]
    return -0.5 * np.sum(d ** 2, axis=2) / sigma ** 2


def log_pi(nodes, c, sigma=SIGMA):
    """`log Pi_k = log P(y_1 > c | x_k)` -- the cut acts on component 1 only."""
    return norm.logsf((c - nodes.grid[:, 0]) / sigma)


def scores_batched(y, nodes, batch=4096, **kw):
    """`scores_from_loglike` over batches: the `(N,G)` likelihood never exists in full."""
    ss, ii = [], []
    for i in range(0, len(y), batch):
        s, info, _ = scores_from_loglike(loglike(y[i:i + batch], nodes), nodes, **kw)
        ss.append(s)
        ii.append(info)
    return np.concatenate(ss), np.concatenate(ii)


@pytest.fixture(scope="module")
def nodes():
    """Fine grid for the closed-form checks, which need only a handful of rows."""
    return ShiftNodes(n=161)


@pytest.fixture(scope="module")
def sample():
    """One simulated catalogue, shared by both estimator tests (they are the slow ones).

    Coarser grid: quadrature error at step 0.2 tau is far below the sampling error, and
    the cost is linear in the node count.
    """
    nd = ShiftNodes(n=61)
    theta, c, n = np.array([0.05, 0.0]), 0.0, 150_000
    rng = np.random.default_rng(3)
    x = theta + TAU * rng.standard_normal((n, 2))
    y = x + SIGMA * rng.standard_normal((n, 2))
    y = y[y[:, 0] > c]
    s, info = scores_batched(y, nd)
    s_sel, i_sel = population_terms(nd, log_pi(nd, c))
    return theta, s, info, s_sel, i_sel


# ---------------------------------------------------------------------------------------
# per-object block: Fisher's identity and Louis, against A.7
# ---------------------------------------------------------------------------------------

def test_per_object_score_and_information(nodes):
    """`s_i = y_i / nu^2` and `I_i = 1 / nu^2`, the first two rows of A.7's table."""
    y = np.array([[0.0, 0.0], [1.0, -0.5], [2.0, 1.5], [-1.7, 0.3]])
    s, info, _ = scores_from_loglike(loglike(y, nodes), nodes)
    assert np.allclose(s, y / NU2, rtol=2e-3, atol=2e-4)
    eye = np.tile(np.eye(2)[None], (len(y), 1, 1)) / NU2
    assert np.allclose(info, eye, rtol=3e-3, atol=3e-4)


def test_detection_channel_reweights(nodes):
    """`log_det` must move the weights; a tilt along x1 shifts `s` toward it.

    Detection is a function of the node's TRUE properties and does not cancel
    (§5B.1(iii)).  A linear tilt `log Pdet = k x1` on a Gaussian prior is an exact
    conjugate shift: the posterior mean of `x1` moves by `k tau^2 sigma^2 / nu^2`, so
    `s_1 = E_post[x_1]/tau^2` moves by exactly `k sigma^2 / nu^2`.
    """
    y = np.array([[0.4, -0.2], [1.1, 0.7]])
    k = 0.3
    base, _, _ = scores_from_loglike(loglike(y, nodes), nodes)
    tilt, _, _ = scores_from_loglike(loglike(y, nodes), nodes,
                                     log_det=k * nodes.grid[:, 0])
    shift = k * SIGMA ** 2 / NU2
    assert np.allclose(tilt[:, 0] - base[:, 0], shift, rtol=5e-3, atol=5e-4)
    assert np.allclose(tilt[:, 1] - base[:, 1], 0.0, atol=5e-4)


def test_detection_accepts_per_galaxy_array(nodes):
    """`(N,G)` `log_det` must agree with `(G,)` when every row is the same."""
    y = np.array([[0.4, -0.2], [1.1, 0.7]])
    v = 0.3 * nodes.grid[:, 0]
    a, _, _ = scores_from_loglike(loglike(y, nodes), nodes, log_det=v)
    b, _, _ = scores_from_loglike(loglike(y, nodes), nodes,
                                  log_det=np.tile(v[None], (len(y), 1)))
    assert np.allclose(a, b, rtol=1e-4, atol=1e-5)


# ---------------------------------------------------------------------------------------
# population block: A.7 and A.7b
# ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("c", [-0.5, 0.0, 0.75, 1.5])
def test_population_terms_match_A7(nodes, c):
    """`<s>_sel = lambda/nu` and `I_sel = lambda(lambda-a)/nu^2`, exactly A.7 / A.7b."""
    s_sel, i_sel = population_terms(nodes, log_pi(nodes, c))
    a = c / NU
    lam = norm.pdf(a) / norm.sf(a)
    assert s_sel[0] == pytest.approx(lam / NU, rel=3e-3)
    assert s_sel[1] == pytest.approx(0.0, abs=3e-4)           # uncut component: null
    assert i_sel[0, 0] == pytest.approx(lam * (lam - a) / NU2, rel=6e-3)
    assert i_sel[1, 1] == pytest.approx(0.0, abs=1e-3)
    assert i_sel[0, 1] == pytest.approx(0.0, abs=1e-3)


def test_A7b_identity(nodes):
    """A.7b's own cross-check: `I - I_sel = Var[y | y > c] / nu^4`."""
    c = 0.4
    _, i_sel = population_terms(nodes, log_pi(nodes, c))
    a = c / NU
    lam = norm.pdf(a) / norm.sf(a)
    var_trunc = NU2 * (1.0 + a * lam - lam ** 2)             # truncated-normal variance
    assert (1.0 / NU2 - i_sel[0, 0]) == pytest.approx(var_trunc / NU2 ** 2, rel=6e-3)


def test_i_sel_over_i_is_two_over_pi_at_the_median(nodes):
    """A.7's headline ratio: a cut at the median destroys `2/pi` of the information."""
    _, i_sel = population_terms(nodes, log_pi(nodes, 0.0))
    assert i_sel[0, 0] * NU2 == pytest.approx(2.0 / np.pi, rel=6e-3)


# ---------------------------------------------------------------------------------------
# the full estimator (5.3)
# ---------------------------------------------------------------------------------------

def test_full_estimator_is_unbiased_under_a_cut(sample):
    """(5.3) with BOTH corrections returns theta; the uncorrected form does not."""
    theta, s, info, s_sel, i_sel = sample
    full, _, _ = full_shear_estimate(s, info, s_sel, i_sel)
    naive, _, _ = full_shear_estimate(s, info)                # no centring at all
    assert full[0] == pytest.approx(theta[0], abs=0.02)
    assert full[1] == pytest.approx(0.0, abs=0.02)
    assert abs(naive[0] - theta[0]) > 10 * abs(full[0] - theta[0])


def test_omitting_i_sel_costs_64_percent(sample):
    """A.7: centring the numerator but not the denominator reports `m = -64%`.

    This is cont.164 defect 2 measured on the production code.  For a spin-2 shear the
    numerator term averages away by orientation, so `I_sel` is the term that survives --
    i.e. the missing one was precisely the one that matters for us.

    Nearly noise-free even though it rides on a simulated catalogue: both estimates share
    a numerator, so the ratio is `1 - N I_sel / sum_i I_i` and the `I_i` are almost
    constant in this toy.
    """
    _, s, info, s_sel, i_sel = sample
    full, _, _ = full_shear_estimate(s, info, s_sel, i_sel)
    half, _, _ = full_shear_estimate(s, info, s_sel, None)    # numerator only
    m = half[0] / full[0] - 1.0
    assert m == pytest.approx(-2.0 / np.pi, abs=0.02)


def test_population_terms_rejects_wrong_shape(nodes):
    with pytest.raises(ValueError):
        population_terms(nodes, np.zeros((3, len(nodes.grid))))
