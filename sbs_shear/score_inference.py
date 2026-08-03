"""Score-based shear inference -- the executable form of `INFERENCE.md` §5B.

`posterior_shape.py` already answers "what is this galaxy's shape?"  This module answers
the different question "what is the shear of this catalogue?", from the SAME ingredients
(the e-grid node bank, the exact Mobius-pullback sheared prior, and the flow's per-node
log-likelihood) but with a different read-out.

The chain, with `INFERENCE.md` equation tags:

    u_k       = -( v . grad log p_0 + div v )(e_k)        generator on TRUTH        (2.7)
    w_k       propto L_k pi_0(e_k),  L_k = p_flow(ehat_i | e_k, rest)   posterior weights
    s_i       = sum_k w_k u_k                             Fisher's identity         (2.2)
    I_i       = -sum_k w_k du_k - Var_w(u)                Louis                     (2.5)
    ghat      = sum_i s_i / sum_i I_i                     the estimator             (2.6)
    R         = Cov_0(ehat, s) ~ (1/N) sum_i ehat_i s_i   the response              (2.3)

Everything here is the SHAPE channel only: the latent is the primary's true (lensed)
ellipticity, all other conditioning of the flow is held at its catalogue value.  Per
§5B.2 that is the honest scope of a 2-D shape grid -- it yields `R_self`'s shape part,
not the size/flux channel (§4.3) and not the blend channel (§3, identically zero for a
geometry-blind flow).  §5C's external-`R_blend` injection is `blend_injection_term`.

Sign conventions follow `shear_map.py`: `eps' = (eps + g)/(1 + conj(g) eps)`, so
`v = d eps'/dg|_0` has components `v_1 = 1 - eps^2`, `v_2 = i (1 + eps^2)` and
`div v_1 = -4 e1`, `div v_2 = -4 e2`.  With an isotropic `p_0` this collapses to the
closed form used as an independent cross-check of the finite differences:

    u_a(e) = e_a * [ 4 - phi'(r) (1 - r^2) / r ],      phi(r) = log p_0(e) at |e| = r.

Numerically the only delicate ingredient is `phi'`, the radial derivative of an
EMPIRICAL log-density.  `SmoothRadialPrior` therefore replaces the piecewise-linear
interpolation of `RadialShapePrior` with a C2 quantile-knot spline in `|eps|^2`, and
continues it smoothly past the last populated bin rather than cutting it off -- a hard
support edge would inject a spurious boundary term into the score, because the sheared
prior's support boundary moves with gamma.

Two cheap diagnostics catch every prior-fitting failure seen while building this, and
both are asserted in `tests/test_score_inference.py`:

  `ShapeScoreNodes.bartlett()`         `E_0[u] = 0` and `E_0[du] + Var_0(u) = 0`; the
                                       second is the information of a galaxy with a flat
                                       likelihood, so any residual is a spurious
                                       information floor biasing `ghat` by
                                       `residual / <I>`.
  `ShapeScoreNodes.closed_form_residual()`  finite-difference `u` against the closed
                                       form; sensitive to kinks in `psi` that the
                                       Bartlett average washes out (it caught a
                                       clamped, upturning spline tail).
"""

from __future__ import annotations

import numpy as np
import torch

from .posterior_shape import RadialShapePrior


# --------------------------------------------------------------------------------------
# prior
# --------------------------------------------------------------------------------------

class SmoothRadialPrior(RadialShapePrior):
    """`RadialShapePrior` with a C2 log-density fitted in `t = |eps|^2`.

    Two things are wrong with the parent class for score work, and both are fixed here.

    *Smoothness.*  The parent interpolates `log_dens` linearly between bin centres, so
    its radial derivative is piecewise constant and its second derivative a comb of
    deltas.  That is harmless for a posterior MEAN (an integral) and useless for a SCORE
    (a derivative).  Here the binned log-density is fitted with a weighted cubic
    smoothing spline (Poisson weights `sqrt(counts)`).

    *The origin.*  The parent bins uniformly in `r` and forms the 2-D density as
    `f(r) / (2 pi r)`, which divides by a vanishing radius exactly where the counts are
    scarcest.  The resulting `phi'(0) != 0` puts a spurious `1/r` term into the
    generator: with `u_a = e_a [4 - phi'(r)(1-r^2)/r]` and `e_a ~ r`, a non-zero
    `phi'(0)` leaves `u` finite but direction-discontinuous at the origin and `du`
    divergent, which shows up as a `1/delta` blow-up of the Bartlett curvature.  A
    genuinely smooth isotropic density is a function of `r^2`, so binning in `t = r^2`
    -- equal-AREA annuli, no `1/r`, uniform Poisson errors -- imposes `phi'(0) = 0` by
    construction:

        phi(r) = psi(r^2),      phi'(r) = 2 r psi'(r^2),
        u_a(e) = e_a [ 4 - 2 psi'(r^2) (1 - r^2) ]      -- manifestly regular at e = 0.

    Beyond the last populated bin `psi` continues with its end slope, i.e. a Gaussian
    tail in `r`, rather than the parent's hard `-inf` edge.  A hard edge is not merely
    inconvenient: the sheared prior's support boundary MOVES with gamma, so a truncated
    density contributes a delta-function boundary term to `u` that the real population
    does not have.

    `sheared_log_prob` (the exact Mobius pullback) is inherited and picks up the new
    `log_prob` automatically; `sample` is overridden to draw from the *fitted* density,
    so a closure test generating shapes from this prior is estimating under exactly the
    density it drew from.
    """

    def __init__(self, e1_samples, e2_samples, n_bins=120, min_samples=10_000,
                 n_knots=8, knot_margin=0.10, r_hard=0.999):
        super().__init__(e1_samples, e2_samples, n_bins=n_bins, min_samples=min_samples)
        from scipy.interpolate import LSQUnivariateSpline

        r = np.hypot(np.asarray(e1_samples, float), np.asarray(e2_samples, float))
        r = r[np.isfinite(r) & (r < 1.0)]
        t = r ** 2
        self.t_max = float(t.max())
        t_edges = np.linspace(0.0, self.t_max * (1.0 + 1e-9), n_bins + 1)
        counts, _ = np.histogram(t, bins=t_edges)
        dt = np.diff(t_edges)
        # equal-area annuli: the area between t and t+dt is pi dt, so the 2-D density is
        # counts / (N pi dt) -- no 1/r anywhere, and uniform Poisson errors.
        good = counts > 0
        if good.sum() < 8:
            raise ValueError(f"only {int(good.sum())} populated bins; need >= 8")
        x = (0.5 * (t_edges[:-1] + t_edges[1:]))[good]
        y = np.log(counts[good] / (len(t) * np.pi * dt[good]))
        w = np.sqrt(counts[good].astype(float))          # sigma(log density) ~ 1/sqrt(N)
        # Fixed knots at DATA quantiles, not a smoothing penalty.  A smoothing spline
        # with a per-bin chi^2 target chases Poisson noise in the sparse |eps| tail --
        # measured, that made psi' swing to +34 near r = 0.9 and blew the Bartlett
        # curvature up to ~7% of Var(u).  Quantile knots crowd where the data are (99%
        # of the mass sits at t < 0.5), leaving the sparse tail spanned by a single
        # cubic, which is both stable and honest about what the data constrain there.
        #
        # The knots are kept a `knot_margin` fraction of the bins away from both ends.
        # A knot hard against the left boundary leaves the first polynomial span with
        # almost no data: on a Gaussian test prior, whose psi' is exactly constant, that
        # made psi'(0) read -4.5 instead of -8.68 and put a 2%-of-Var(u) floor into the
        # Bartlett curvature.  Quantile LEVELS are shrunk to the interior rather than the
        # knots being clipped, so raising n_knots adds knots instead of piling them up on
        # the boundary and losing them to `unique`.
        m = max(2, int(len(x) * knot_margin))
        q_lo = float(np.mean(t <= x[m]))
        q_hi = float(np.mean(t <= x[-1 - m]))
        knots = np.unique(np.quantile(t, np.linspace(q_lo, q_hi, int(n_knots))))
        knots = knots[(knots > x[0]) & (knots < x[-1])]
        self._spl = LSQUnivariateSpline(x, y, t=knots, w=w, k=3)
        self._dspl = self._spl.derivative()
        self.fit_chi2_dof = float(np.sum((w * (y - self._spl(x))) ** 2)
                                  / max(len(x) - len(knots) - 4, 1))
        self.n_knots = len(knots)
        self._t0, self._t1 = float(x[0]), float(x[-1])
        self._y0, self._d0 = float(self._spl(self._t0)), float(self._dspl(self._t0))
        self._y1, self._d1 = float(self._spl(self._t1)), float(self._dspl(self._t1))
        self._d1 = min(self._d1, -1e-3)                  # decaying tail only
        self.r_hard = float(r_hard)
        self._build_sampler()

    # -- the fitted density, as a function of t = r^2 ----------------------------------

    def _psi(self, t):
        t = np.asarray(t, dtype=float)
        out = np.empty_like(t)
        lo, hi = t < self._t0, t > self._t1
        mid = ~(lo | hi)
        out[mid] = self._spl(t[mid])
        out[lo] = self._y0 + self._d0 * (t[lo] - self._t0)
        out[hi] = self._y1 + self._d1 * (t[hi] - self._t1)
        return out

    def _dpsi(self, t):
        t = np.asarray(t, dtype=float)
        out = np.empty_like(t)
        lo, hi = t < self._t0, t > self._t1
        mid = ~(lo | hi)
        out[mid] = self._dspl(t[mid])
        out[lo] = self._d0
        out[hi] = self._d1
        return out

    def _dphi(self, r):
        """`d log p_0 / dr = 2 r psi'(r^2)` -- what the closed-form generator needs."""
        r = np.asarray(r, dtype=float)
        return 2.0 * r * self._dpsi(r ** 2)

    def log_prob(self, e1, e2):
        """Smooth `log p_0`; `-inf` only at the unphysical `|eps| >= 1`."""
        t = np.asarray(e1, dtype=float) ** 2 + np.asarray(e2, dtype=float) ** 2
        return np.where(t < self.r_hard ** 2, self._psi(np.minimum(t, self.r_hard ** 2)),
                        -np.inf)

    # -- sampling from the FITTED density (so closure tests have no prior mismatch) -----

    def _build_sampler(self, n=4096):
        t = np.linspace(0.0, self.r_hard ** 2, n)
        pdf = np.pi * np.exp(self._psi(t))               # d(mass)/dt
        cdf = np.concatenate([[0.0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(t))])
        self._t_grid, self._t_cdf = t, cdf / cdf[-1]
        self.norm_error = float(cdf[-1] - 1.0)

    def sample(self, n, rng):
        u = rng.random(int(n))
        t = np.interp(u, self._t_cdf, self._t_grid)
        r = np.sqrt(t)
        phi = rng.random(int(n)) * 2.0 * np.pi
        return r * np.cos(phi), r * np.sin(phi)


# --------------------------------------------------------------------------------------
# node bank: the generator u and its gamma-derivative on the shape grid
# --------------------------------------------------------------------------------------

def generator_closed_form(prior, grid):
    """`u` from the closed form `u_a = e_a [4 - phi'(r)(1-r^2)/r]` (isotropic `p_0`).

    Independent of `generator_finite_difference`, which differentiates the Mobius
    pullback numerically; the two agreeing is a real check that the prior, the shear map
    and the divergence term are mutually consistent."""
    e1, e2 = grid[:, 0], grid[:, 1]
    r = np.hypot(e1, e2)
    rs = np.maximum(r, 1e-9)
    amp = 4.0 - prior._dphi(rs) * (1.0 - r ** 2) / rs
    return np.stack([e1 * amp, e2 * amp], axis=1)


def generator_finite_difference(prior, grid, delta=0.01, richardson=True):
    """`u_a = d_a log p_gamma(e)|_0` and `du_ab = d_a d_b log p_gamma(e)|_0`.

    Central differences of `prior.sheared_log_prob`, i.e. of the EXACT Mobius pullback,
    so the shear map is never linearised by hand.  With `richardson`, `u` is refined by
    the standard `(4 f(d/2) - f(d))/3` extrapolation, killing the O(delta^2) term.

    Returns `(u (G,2), du (G,2,2))`.  Nodes where any stencil point falls outside the
    prior's support come back as zeros and are flagged by `~support`.
    """
    e1, e2 = grid[:, 0], grid[:, 1]

    def f(g1, g2):
        return prior.sheared_log_prob(e1, e2, g1, g2)

    def stencil(d):
        f0 = f(0.0, 0.0)
        fp1, fm1 = f(+d, 0.0), f(-d, 0.0)
        fp2, fm2 = f(0.0, +d), f(0.0, -d)
        u = np.stack([(fp1 - fm1) / (2 * d), (fp2 - fm2) / (2 * d)], axis=1)
        h11 = (fp1 - 2 * f0 + fm1) / d ** 2
        h22 = (fp2 - 2 * f0 + fm2) / d ** 2
        h12 = (f(+d, +d) - f(+d, -d) - f(-d, +d) + f(-d, -d)) / (4 * d ** 2)
        du = np.stack([np.stack([h11, h12], -1), np.stack([h12, h22], -1)], axis=1)
        return u, du

    u, du = stencil(delta)
    if richardson:
        u_half, du_half = stencil(0.5 * delta)
        u = (4.0 * u_half - u) / 3.0
        du = (4.0 * du_half - du) / 3.0
    support = np.isfinite(u).all(axis=1) & np.isfinite(du).all(axis=(1, 2))
    u = np.where(support[:, None], np.nan_to_num(u), 0.0)
    du = np.where(support[:, None, None], np.nan_to_num(du), 0.0)
    return u, du, support


def _nodes_at(prior, grid, gamma, delta, richardson=True):
    """`(log p_gamma, u_gamma)` on the grid, where `u_gamma = d_g' log p_g'|_{g'=gamma}`.

    The gamma-family of node banks is what turns the information into a finite
    difference of the score (see `ShapeScoreNodes`), and it costs nothing: the flow's
    likelihood does not depend on gamma, so the SAME cached `L_k` serves every offset.
    """
    e1, e2 = grid[:, 0], grid[:, 1]
    g1, g2 = float(gamma[0]), float(gamma[1])

    def f(a, b):
        return prior.sheared_log_prob(e1, e2, g1 + a, g2 + b)

    def grad(d):
        return np.stack([(f(+d, 0) - f(-d, 0)) / (2 * d),
                         (f(0, +d) - f(0, -d)) / (2 * d)], axis=1)

    u = grad(delta)
    if richardson:
        u = (4.0 * grad(0.5 * delta) - u) / 3.0
    return f(0.0, 0.0), u


class ShapeScoreNodes:
    """The shared node bank of `INFERENCE.md` §5B: grid, log-prior, `u`, `du`.

    Built once and reused for every galaxy -- nothing here depends on the data.

    Two information estimators are carried, and they are not redundant:

    * `du` gives the ANALYTIC `I = -E_w[du] - Var_w(u)` (Louis, 2.5).  Exact, cheap, and
      validated by `bartlett()` -- but only for the plain model, because it knows about
      `gamma` solely through the prior.
    * `shifted` holds the node bank at `gamma = +-delta_I` along each axis, so the
      information can instead be read as `I = -d_gamma s_gamma|_0`, a finite difference
      of the score itself.  That form stays exact when the likelihood ALSO carries a
      gamma-dependence -- which is exactly what §5C.3's `R_blend` injection introduces,
      and where the analytic Louis expression would silently drop a term of order
      `R_blend / (R_flow + R_blend)`.

    The two agreeing on the plain model is the check that the finite-difference route is
    trustworthy before it is used on the injected one.
    """

    def __init__(self, grid, prior, delta=0.01, richardson=True, info_delta=0.01):
        self.grid = np.asarray(grid, dtype=np.float64)
        self.prior = prior
        self.log_prior = prior.log_prob(self.grid[:, 0], self.grid[:, 1])
        u, du, support = generator_finite_difference(prior, self.grid, delta, richardson)
        self.u, self.du = u, du
        self.support = support & np.isfinite(self.log_prior)
        self.log_prior = np.where(self.support, self.log_prior, -np.inf)
        self.delta = delta
        self.info_delta = float(info_delta)
        self.u_closed = (generator_closed_form(prior, self.grid)
                         if hasattr(prior, "_dphi") else None)
        d = self.info_delta
        self.shifted = {}
        for key, gam in ((("g1", +1), (+d, 0.0)), (("g1", -1), (-d, 0.0)),
                         (("g2", +1), (0.0, +d)), (("g2", -1), (0.0, -d))):
            lp, uu = _nodes_at(prior, self.grid, gam, delta, richardson)
            lp = np.where(np.isfinite(lp) & self.support, lp, -np.inf)
            self.shifted[key] = (lp, np.where(self.support[:, None],
                                              np.nan_to_num(uu), 0.0))

    # -- diagnostics -------------------------------------------------------------------

    def prior_weights(self):
        """Normalised prior weights over the node bank (flat-grid quadrature)."""
        lp = self.log_prior - np.max(self.log_prior[self.support])
        w = np.where(self.support, np.exp(lp), 0.0)
        return w / w.sum()

    def bartlett(self):
        """The two prior-only identities that must hold for ANY normalised `p_gamma`.

        Because `integral p_gamma = 1` for every gamma, differentiating twice under the
        integral gives `E_0[u] = 0` and `E_0[du] + Var_0(u) = 0` -- (2.4) with a flat
        likelihood, where the evidence is constant so its information vanishes.  They
        test the prior, the generator, the shear map and the grid quadrature at once,
        with no flow and no data involved.
        """
        w = self.prior_weights()
        mean_u = w @ self.u
        mean_du = np.einsum("k,kab->ab", w, self.du)
        cov_u = np.einsum("k,ka,kb->ab", w, self.u, self.u) - np.outer(mean_u, mean_u)
        return dict(mean_u=mean_u, curvature=mean_du + cov_u,
                    scale=float(np.sqrt(np.mean(np.diag(cov_u)))))

    def closed_form_residual(self):
        """max |u_fd - u_closed| / rms(u) over supported nodes (0 if no closed form)."""
        if self.u_closed is None:
            return float("nan")
        m = self.support
        num = np.abs(self.u[m] - self.u_closed[m]).max()
        return float(num / np.sqrt(np.mean(self.u[m] ** 2)))


# --------------------------------------------------------------------------------------
# per-object score and information
# --------------------------------------------------------------------------------------

def _weighted_score(ll_t, log_prior, u, extra, gamma):
    """`s(gamma) = E_w[u_gamma + extra]` for one node bank, on a slab already on device.

    `extra` (the §5C.3 injection) enters twice, and both are needed.  It is part of the
    generator, and it is also part of the LIKELIHOOD's gamma-dependence: to first order
    the injected model has `log L_k(gamma) = log L_k(0) + gamma . extra_k`, since `extra`
    is by construction that derivative.  Carrying the `gamma . extra` reweighting is what
    makes the finite-difference information include the blend channel instead of
    silently dropping it.
    """
    ll = ll_t + log_prior
    if extra is not None:
        if gamma[0]:
            ll = ll + float(gamma[0]) * extra[:, :, 0]
        if gamma[1]:
            ll = ll + float(gamma[1]) * extra[:, :, 1]
    mx = ll.max(dim=1, keepdim=True).values
    w = torch.exp(ll - mx)
    norm = w.sum(dim=1, keepdim=True)
    w = w / norm
    if extra is None:
        s = w @ u
    else:
        s = (w[:, :, None] * (u[None, :, :] + extra)).sum(dim=1)
    return s, w, torch.log(norm[:, 0]) + mx[:, 0]


def scores_from_loglike(loglike, nodes, chunk=65536, device=None, extra=None,
                        extra_hess=None, analytic_info=False, diag=None, log_det=None):
    """Turn per-node log-likelihoods into `(s_i, I_i)`.

    `loglike`: `(N,G)` array holding `log p_flow(ehat_i | e_k, rest)` up to a per-row
    constant, which cancels.  `extra`: optional `(N,G,2)` galaxy-dependent addition to
    the generator -- §5C.3's external-`R_blend` injection.

    Information is `I = -d_gamma s_gamma|_0`, evaluated by re-weighting the SAME
    likelihood under the node bank at `gamma = +-delta_I` (`nodes.shifted`).  With
    `analytic_info` the Louis form `-E_w[du] - Var_w(u)` is returned instead; the two
    agree on the plain model and the finite difference is the one that survives the
    injection.

    `log_det`: optional `(G,)` or `(N,G)` log detection probability at each node --
    §5B.1(iii).  Detection is NOT determined by `xhat`, so unlike the cut it does NOT
    cancel from the per-object posterior and must multiply the weights.  It is a function
    of the node's TRUE properties, which do not move with gamma in the Eulerian picture
    (the nodes are fixed and the density slides over them), so the same vector is added to
    the base AND the shifted log-priors.  Omitting it was the first of the three defects
    `WORKLOG.md` cont.164 lists against this module; `MATH.md` §7(a) measures the missing
    channel at 17-760% of the score, with a sign reversal at one of four test points.

    Returns `(s (N,2), info (N,2,2), log_evidence (N,))`.  Individual `info` entries may
    be negative; only their sum is the Fisher information (2.4), which is why (2.6) sums
    numerator and denominator separately rather than averaging per-object ratios.
    """
    dev = torch.device(device) if device is not None else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    T = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float32, device=dev)
    u = T(nodes.u)                                                             # (G,2)
    du = T(nodes.du.reshape(-1, 4))
    lp = T(np.where(nodes.support, nodes.log_prior, -np.inf))                  # (G,)
    sh = {k: (T(v[0]), T(v[1])) for k, v in nodes.shifted.items()}
    # detection multiplies the weights everywhere the prior does, and is gamma-independent
    ld_rows = None
    if log_det is not None:
        ld = np.asarray(log_det, dtype=np.float64)
        if ld.ndim == 1:
            ldt = T(ld)
            lp = lp + ldt
            sh = {k: (v[0] + ldt, v[1]) for k, v in sh.items()}
        elif ld.ndim == 2:
            ld_rows = ld                                   # (N,G): folded in per chunk
        else:
            raise ValueError("log_det must be (G,) or (N,G)")
    d = nodes.info_delta
    n = loglike.shape[0]
    s_out = np.empty((n, 2), dtype=np.float64)
    i_out = np.empty((n, 2, 2), dtype=np.float64)
    z_out = np.empty(n, dtype=np.float64)
    if extra is not None:                # (B,G,2) fp32 is 2x the log-likelihood slab
        chunk = min(chunk, 16384)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        arr = np.ascontiguousarray(loglike[start:stop])
        if not arr.flags.writeable:
            arr = arr.copy()
        ll_t = torch.as_tensor(arr, device=dev).float()
        if ld_rows is not None:
            # a per-(galaxy, node) constant shifts the log-weight identically whether it
            # is carried on the likelihood or the prior side, and this way the SAME shift
            # is seen by all four shifted-bank evaluations below.
            ll_t = ll_t + T(ld_rows[start:stop])
        ex = T(extra[start:stop]) if extra is not None else None
        exh = T(extra_hess[start:stop]) if extra_hess is not None else None
        s, w, logz = _weighted_score(ll_t, lp, u, ex, (0.0, 0.0))
        if analytic_info:
            m2 = ((w[:, :, None] * (u[None] + ex)).transpose(1, 2) @ (u[None] + ex)
                  if ex is not None else torch.einsum("bk,ka,kc->bac", w, u, u))
            cov = m2 - s[:, :, None] * s[:, None, :]
            info = -((w @ du).view(-1, 2, 2) + cov)
        else:
            cols = []
            for axis, gp, gm in (("g1", (+d, 0.0), (-d, 0.0)),
                                 ("g2", (0.0, +d), (0.0, -d))):
                lp_p, u_p = sh[(axis, +1)]
                lp_m, u_m = sh[(axis, -1)]
                s_p, _, _ = _weighted_score(ll_t, lp_p, u_p, ex, gp)
                s_m, _, _ = _weighted_score(ll_t, lp_m, u_m, ex, gm)
                cols.append(-(s_p - s_m) / (2 * d))                            # (B,2)
            info = torch.stack(cols, dim=2)                                    # I[:,a,b]
            if exh is not None:
                # I = -d_gamma s = -(E[d_gamma u] + E[d^2_gamma log L] + Var(u+extra));
                # the finite difference above supplies every term but the middle one.
                corr = torch.einsum("bk,bkac->bac", w, exh)
                if diag is not None:
                    diag.setdefault("fd", []).append(float(info[:, 0, 0].mean()))
                    diag.setdefault("hess", []).append(float(corr[:, 0, 0].mean()))
                    diag.setdefault("extra", []).append(
                        float((w * ex[:, :, 0]).sum(1).mean()))
                    diag.setdefault("n", []).append(int(info.shape[0]))
                info = info - corr
        s_out[start:stop] = s.double().cpu().numpy()
        i_out[start:stop] = info.double().cpu().numpy()
        z_out[start:stop] = logz.double().cpu().numpy()
    return s_out, i_out, z_out


# --------------------------------------------------------------------------------------
# the population block -- (5.3)'s second half
# --------------------------------------------------------------------------------------

def population_terms(nodes, log_pi, **kw):
    """`(<s>_sel, I_sel)` -- the selection corrections subtracted from EVERY galaxy.

    `log_pi`: `(G,)` log `Pi_k = P_pass * P_det` on the node bank, a function of the
    node's TRUE properties only.  Returns `(s_sel (2,), i_sel (2,2))`.

    The whole point is that this needs no new machinery.  The population survival factor
    is

        P(keep | gamma) = Integral p_gamma(x) Pi(x) dx,

    which is the SAME shape of object as a galaxy's evidence with `Pi` playing the part of
    the likelihood -- so Fisher's identity and Louis's identity apply verbatim,

        <s>_sel = E_Pi[u],      I_sel = -E_Pi[d_gamma u] - Var_Pi(u),

    with `E_Pi` the average under weights proportional to `p_0 * Pi`.  Both are therefore
    `scores_from_loglike` evaluated on a SINGLE pseudo-galaxy whose log-likelihood is
    `log Pi`, which also inherits its finite-difference information route and its
    validated chunking for free.  `A.7` gives the closed forms this is tested against.

    WHY IT MATTERS FOR US.  For a spin-2 shear the numerator term `<s>_sel` averages away
    by orientation, so `I_sel` is the ONLY selection term that survives -- and it is the
    one the previous implementation lacked.  A.7 puts it at `I_sel / I = 2/pi` for a cut on
    the median, i.e. an estimator that centres the score but leaves the denominator alone
    reports `m = -64%`.  Nothing about that is a small correction.
    """
    lp = np.asarray(log_pi, dtype=np.float64)
    if lp.ndim != 1 or lp.shape[0] != nodes.grid.shape[0]:
        raise ValueError(f"log_pi must be (G,) with G={nodes.grid.shape[0]}, "
                         f"got {lp.shape}")
    s, info, _ = scores_from_loglike(lp[None, :], nodes, **kw)
    return s[0], info[0]


def population_log_pi(pi_per_galaxy):
    """Collapse a `(N,G)` per-galaxy `Pi` to the `(G,)` the population block wants.

    Needed because this implementation tiles the shape grid against each galaxy's OWN
    non-shape true properties, so `Pi` is per `(galaxy, node)` rather than per node.  The
    scene prior is then (shape prior) x (empirical distribution of the other true
    properties across the catalogue), and marginalising that empirical factor is the plain
    mean over galaxies at fixed shape node:

        Pi_k^eff = (1/N) sum_i Pi_ik,     P(keep | gamma) = Integral p_gamma(e) Pi^eff(e) de.

    Note this is exactly where §5B.1(i)'s "the node bank is shared across the catalogue"
    is violated by the V1-shaped bank: with a genuinely shared bank there would be nothing
    to average.  See `WORKLOG.md` cont.164 defect 3.
    """
    pi = np.asarray(pi_per_galaxy, dtype=np.float64)
    if pi.ndim != 2:
        raise ValueError("pi_per_galaxy must be (N,G)")
    return np.log(np.maximum(pi.mean(axis=0), 1e-300))


# --------------------------------------------------------------------------------------
# read-outs
# --------------------------------------------------------------------------------------

def full_shear_estimate(s, info, s_sel=None, i_sel=None):
    """(5.3) in full: `ghat = (sum_i s_i - N <s>_sel) . (sum_i I_i - N I_sel)^-1`.

    The 2-D solve, for a shear whose direction is not known per object.  `s` is `(N,2)`,
    `info` `(N,2,2)`; `s_sel` `(2,)` and `i_sel` `(2,2)` come from `population_terms` and
    default to zero, which reproduces the previous uncorrected `sum s / sum I`.

    Returns `(ghat (2,), num (2,), den (2,2))` so the caller can report the pieces --
    which is worth doing, because the selection correction lands almost entirely in `den`.
    """
    s = np.asarray(s, dtype=np.float64)
    info = np.asarray(info, dtype=np.float64)
    n = s.shape[0]
    num = s.sum(axis=0) - (n * np.asarray(s_sel, float) if s_sel is not None else 0.0)
    den = info.sum(axis=0) - (n * np.asarray(i_sel, float) if i_sel is not None else 0.0)
    return np.linalg.solve(den, num), num, den


def blocked_sums(s, info, block, n_blocks):
    """Partial sums of `(count, sum s, sum I)` within each block.  `O(N)`, one pass.

    Everything (5.3) needs from the catalogue is a SUM, so a delete-one-block jackknife
    costs one subtraction per block once these are in hand -- no re-scan of the rows.
    """
    b = np.asarray(block, dtype=np.int64)
    if b.ndim != 1 or b.shape[0] != s.shape[0]:
        raise ValueError("block must be (N,) matching s")
    if b.size and (b.min() < 0 or b.max() >= n_blocks):
        raise ValueError("block ids must lie in [0, n_blocks)")
    cnt = np.bincount(b, minlength=n_blocks).astype(np.float64)
    ns = np.stack([np.bincount(b, weights=s[:, a], minlength=n_blocks) for a in range(2)],
                  axis=1)
    ni = np.stack([np.stack([np.bincount(b, weights=info[:, a, c], minlength=n_blocks)
                             for c in range(2)], axis=1) for a in range(2)], axis=1)
    return cnt, ns, ni


def jackknife_shear(s, info, block, n_blocks, s_sel=None, i_sel=None):
    """(5.3) plus a delete-one-block jackknife covariance.

    WHY NOT THE FISHER ERROR BAR.  `1/sqrt(sum_i I_i)` is the Cramer-Rao bound for a
    sample of INDEPENDENT objects.  The moment the catalogue contains ring pairs -- two
    orientations of the same galaxy, deliberately anticorrelated so the intrinsic shape
    noise cancels -- that bound is no longer the estimator's variance, and it overstates
    it by whatever the variance reduction achieved.  Reporting it would hide the very
    thing the pairing was for.  The jackknife measures the realised scatter instead, and
    is correct either way; with unpaired rows it simply reproduces the Fisher bar.

    Blocks must respect the pairing: both members of a ring pair belong to the SAME
    block, or deleting one member while keeping the other breaks the cancellation and
    the jackknife reports the unpaired variance again.

    `s_sel` / `i_sel` are treated as FIXED (their own uncertainty is reported separately
    by the population block's replicates); the jackknife here is over the catalogue only.

    Returns `(ghat (2,), sigma (2,), reps (n_blocks, 2))`.  `reps` is returned so a
    caller can jackknife a DIFFERENCE of two estimates that share a blocking -- e.g. cut
    versus uncut on the same rows, where the difference is far better determined than
    either estimate, because the two share their shape noise.
    """
    s = np.asarray(s, dtype=np.float64)
    info = np.asarray(info, dtype=np.float64)
    return jackknife_blocks(*blocked_sums(s, info, block, n_blocks), s_sel, i_sel)


def jackknife_blocks(cnt, ns, ni, s_sel=None, i_sel=None):
    """`jackknife_shear` straight from `blocked_sums` output, without the rows.

    Same contract and same numbers; the split exists because the two halves of (5.3) cost
    wildly different amounts.  Scoring the catalogue is hours on a GPU and depends on
    nothing about the population block; `<s>_sel` and `I_sel` are minutes and are the
    term whose convergence is actually in question.  Caching `(cnt, ns, ni)` lets the
    population term be re-estimated as often as needed against a FIXED catalogue -- which
    also makes those re-runs paired, so a change in `Pi` is not confounded with a change
    in the galaxies.
    """
    cnt = np.asarray(cnt, dtype=np.float64)
    ns = np.asarray(ns, dtype=np.float64)
    ni = np.asarray(ni, dtype=np.float64)
    zs = np.zeros(2) if s_sel is None else np.asarray(s_sel, float)
    zi = np.zeros((2, 2)) if i_sel is None else np.asarray(i_sel, float)

    def est(n, sum_s, sum_i):
        return np.linalg.solve(sum_i - n * zi, sum_s - n * zs)

    full = est(cnt.sum(), ns.sum(axis=0), ni.sum(axis=0))
    # One replicate per block ALWAYS, including any empty ones (which simply reproduce
    # `full` and contribute nothing to the scatter).  Length `n_blocks` regardless of
    # occupancy is what lets a caller subtract two runs' replicates block by block; a
    # ragged array silently misaligns the pairing it is there to exploit.
    reps = np.stack([est(cnt.sum() - cnt[b], ns.sum(axis=0) - ns[b], ni.sum(axis=0) - ni[b])
                     for b in range(len(cnt))])
    return full, jackknife_sigma(reps), reps


def jackknife_sigma(reps):
    """Delete-one jackknife standard error from the leave-one-out replicates."""
    reps = np.asarray(reps, dtype=np.float64)
    b = len(reps)
    if b < 2:
        return np.full(reps.shape[1:], np.nan)
    d = reps - reps.mean(axis=0)
    return np.sqrt((b - 1) / b * np.sum(d ** 2, axis=0))


def project(s, info, ghat1, ghat2):
    """Project `(s, I)` onto the per-object applied-shear direction.

    Each constant-shear case has its own shear direction, so the scalar shear along it
    is the only 1-D parameter with a common meaning across the catalogue.
    """
    g = np.stack([np.asarray(ghat1, float), np.asarray(ghat2, float)], axis=1)
    s_p = np.einsum("na,na->n", s, g)
    i_p = np.einsum("na,nab,nb->n", g, info, g)
    return s_p, i_p


def shear_estimate(s_proj, i_proj):
    """`ghat = sum_i s_i / sum_i I_i` -- (2.6), one Newton step at gamma = 0."""
    den = float(np.sum(i_proj))
    return float(np.sum(s_proj)) / den, den


def response_from_score(ehat_proj, s_proj):
    """`R = Cov_0(ehat, s)` -- (2.3) with `f = ehat`, the response WITHOUT paired sims.

    The mean subtraction matters when the sample is not exactly at gamma = 0: at applied
    |gamma| = 0.02 the `<ehat><s>` product is a ~1% contamination of `R`.
    """
    ehat_proj = np.asarray(ehat_proj, float)
    s_proj = np.asarray(s_proj, float)
    return float(np.mean(ehat_proj * s_proj) - np.mean(ehat_proj) * np.mean(s_proj))


def bootstrap_by_case(values, cases, n_boot=200, seed=0, reducer=np.mean):
    """Per-case bootstrap error, matching `validate_constant_with_blend`'s convention."""
    cases = np.asarray(cases)
    uc = np.unique(cases)
    idx = {c: np.flatnonzero(cases == c) for c in uc}
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        pick = rng.choice(uc, size=len(uc), replace=True)
        sel = np.concatenate([idx[c] for c in pick])
        out.append(reducer(values[sel]))
    return float(np.std(out))


# --------------------------------------------------------------------------------------
# §5C.3 -- injecting an external R_blend
# --------------------------------------------------------------------------------------

def shear_velocity_jacobian(grid):
    """`J[k,b,a] = d eps'_b / d gamma_a` at gamma = 0 on the node grid (Mobius, §2.4)."""
    e1, e2 = grid[:, 0], grid[:, 1]
    j = np.empty((len(grid), 2, 2))
    j[:, 0, 0] = 1.0 - (e1 ** 2 - e2 ** 2)
    j[:, 0, 1] = -2.0 * e1 * e2
    j[:, 1, 0] = -2.0 * e1 * e2
    j[:, 1, 1] = 1.0 + (e1 ** 2 - e2 ** 2)
    return j


def blend_injection_term(mean_grad_ehat, grid, r_blend):
    """The `- R_b(theta_b) grad_ehat log p_flow . v_eps` term of (5.7).

    A geometry-blind flow -- ours conditions on the scalar `nbr_flux_*` only -- has
    `Cov(ehat, s_nbr) = 0` identically (§3), so §5B recovers the SELF response and
    nothing else.  §5C.3 restores the neighbour channel by shifting the likelihood's
    data argument, `p_flow(ehat - R_b * delta_e | ...)`, whose gamma-derivative is this
    term.  It is an injection, not a calibration: `R_b` comes from BlendEMU.

    `mean_grad_ehat`: `(N,G,2)` or `(G,2)` gradient of `log p_flow` w.r.t. the measured
    shape, evaluated at each node.  `grid`: `(G,2)` nodes.  `r_blend`: `(N,)` per-object
    blend response.  Returns `(N,G,2)` to be passed as `scores_from_loglike(extra=...)`.

    SUPERSEDED by `eval_score_response.blend_stencil_on_grid`, which returns this term
    AND its second derivative from one stencil.  Kept because it is the plain reading of
    (5.9) and the two agree, so it serves as the cross-check.

    History worth keeping: using this function alone leaves the INFORMATION incomplete.
    `scores_from_loglike` then treats the injection as gamma-independent, so its finite
    difference sees the reweighting `log L(gamma) = log L(0) + gamma . extra` but not the
    curvature of the injected likelihood, `d^2_gamma log L = w^T grad^2 log p_flow w`
    with `w = R_b v`.  Measured consequence on constgold: `<I>` FELL 3.49 -> 3.19 when
    the injection was switched on (adding model response must RAISE it) and `ghat/g` read
    1.390 instead of 1.013.  Pass `extra_hess` to close it.
    """
    e1, e2 = grid[:, 0], grid[:, 1]
    # v_eps = d eps'/d gamma at gamma = 0: v_1 = 1 - eps^2, v_2 = i (1 + eps^2)
    v = np.empty((len(grid), 2, 2))
    v[:, 0, 0] = 1.0 - (e1 ** 2 - e2 ** 2)
    v[:, 0, 1] = -2.0 * e1 * e2
    v[:, 1, 0] = -2.0 * e1 * e2
    v[:, 1, 1] = 1.0 + (e1 ** 2 - e2 ** 2)
    g = np.asarray(mean_grad_ehat, float)
    if g.ndim == 2:
        g = g[None, :, :]
    term = -np.einsum("n,ngb,gba->nga", np.asarray(r_blend, float), g, v)
    return term
