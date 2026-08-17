"""Grid-based latent-shape posterior for an intrinsic-shape-blind flow.

Computes, per detected galaxy, the posterior-mean corrected shape

    etilde = E[e | ehat, theta_hat]
           = [ integral de  e * p(ehat | e, theta_hat, theta_b) * pi(e) ]
           / [ integral de      p(ehat | e, theta_hat, theta_b) * pi(e) ]

for the flow's supplied neighbour context. Downstream analyses consume ``etilde``
like ordinary ellipticities. By the tower rule its expectation recovers the shear
signal when the lensed-shape prior and likelihood are correctly specified.

Why this is cheap (no MCMC): the accepted measurement flows are ConditionalMeanFlow
with the intrinsic-shape features e1/e2_input_p in `flow_drop_indices`, i.e. the
residual flow is BLIND to e and only the mean head mu(context) moves with e:

    log p(ehat | e, rest) = flow.log_prob(ehat_std - mu(ctx(e)), flow_ctx_fixed)

-- a location family in e.  The posterior is evaluated exactly on a 2-D e-grid:
softmax(log pi + log p) over grid points, then a weighted sum.  The observed vector may
contain shape only or shape plus photometry; current four-output flows use
``(g1, g2, magnitude, log radius)`` in the likelihood. The estimator asserts the
e-flow-blind structure at load time. See ``doc/INFERENCE.md`` for the supported boundary.
"""

from __future__ import annotations

import numpy as np
import torch

from .measurement_model import ConditionalMeanFlow, ConditionalMeanFlowRA, MeasurementModelBundle
from .shear_map import inverse_shear_to_ellipticity


# Engineered features that depend on the primary shape OTHER than the raw components.
# The grid injection below overwrites only e1/e2_input_p (+ their missing flags); if a
# feature set conditioned on any of these, the injected context would be inconsistent.
_E_DERIVED_FEATURES = {
    "e_abs_p", "e_parallel_p", "e_cross_p",
    "e_pframe_parallel_p", "e_pframe_cross_p",
    # secondary shapes are projected into the PRIMARY frame -> depend on primary e too
    "e_pframe_parallel_s", "e_pframe_cross_s",
    "gamma_pframe_parallel_p", "gamma_pframe_cross_p",
    "gamma_pframe_parallel_s", "gamma_pframe_cross_s",
    "pair_pframe_cos2", "pair_pframe_sin2",
}
_E_DERIVED_FEATURES |= {f"{n}_blend" for n in _E_DERIVED_FEATURES}


def shape_target_indices(names):
    """Locate the (e1,e2)-like shape target pair among the flow's target names
    (same conventions as response_ratio_diagnostic._shape_target_indices)."""
    for c1, c2 in (("measured_e1_image", "measured_e2_image"),
                   ("measured_ngmix_g1", "measured_ngmix_g2"),
                   ("measured_galsim_g1", "measured_galsim_g2")):
        if c1 in names and c2 in names:
            return names.index(c1), names.index(c2)
    raise KeyError(f"No known shape target pair in {names}")


def make_e_grid(n=41, emax=0.96, rmax=0.95):
    """Disk-masked square grid of candidate ellipticities.

    Returns (grid, cell_area): grid is (G,2) float64 with |e| <= rmax (epsilon
    convention keeps |e| < 1; the empirical population tops out at ~0.90).  The cell
    area cancels between the posterior numerator and denominator, but is returned for
    diagnostics (evidence normalization)."""
    ax = np.linspace(-emax, emax, n)
    e1, e2 = np.meshgrid(ax, ax, indexing="ij")
    pts = np.stack([e1.ravel(), e2.ravel()], axis=1)
    keep = np.hypot(pts[:, 0], pts[:, 1]) <= rmax
    step = ax[1] - ax[0]
    return pts[keep].copy(), float(step * step)


class RadialShapePrior:
    """Empirical isotropic prior over the intrinsic ellipticity, plus its EXACT
    sheared (lensed) version via the Mobius pullback.

    Intrinsic: the detected/source-selected population is isotropic (checked:
    mean e1/e2 ~ 2e-4, std_e1 == std_e2 ~ 0.238), so pi0(e) = f(|e|) / (2 pi |e|)
    with f the binned radial density of |e| from a sample.

    Sheared at known g: the lensed population is the intrinsic one pushed through the
    Mobius map e' = (e+g)/(1+conj(g) e).  Rather than KDE-ing a sheared sample (a
    bandwidth-induced prior-variance inflation would bias the shrinkage at the ~0.5%
    level), pull grid points back exactly: the map is holomorphic in e with
    de'/de = (1-|g|^2)/(1+conj(g) e)^2, so the 2-D area Jacobian is |de/de'|^2 and

        log pi_g(e') = log pi0(S_{-g}(e')) + 2*log|1-|g|^2| - 4*log|1 - conj(g) e'| .
    """

    def __init__(self, e1_samples, e2_samples, n_bins=60, min_samples=10_000):
        e1 = np.asarray(e1_samples, dtype=float)
        e2 = np.asarray(e2_samples, dtype=float)
        good = np.isfinite(e1) & np.isfinite(e2)
        r = np.hypot(e1[good], e2[good])
        r = r[r < 1.0]
        if len(r) < min_samples:
            raise ValueError(f"Prior sample too small ({len(r)}); need >={min_samples} shapes")
        self.r_max = float(r.max())
        self.edges = np.linspace(0.0, self.r_max * (1.0 + 1e-6), n_bins + 1)
        counts, _ = np.histogram(r, bins=self.edges)
        # radial pdf f(r) (integrates to 1 in r); 2-D density = f(r)/(2 pi r)
        f = counts / (len(r) * np.diff(self.edges))
        self.centers = 0.5 * (self.edges[:-1] + self.edges[1:])
        dens2d = f / (2.0 * np.pi * np.maximum(self.centers, 1e-6))
        # log-density with a tiny floor so empty interior bins interpolate, not -inf
        floor = dens2d[dens2d > 0].min() * 1e-3
        self.log_dens = np.log(np.maximum(dens2d, floor))
        self.n_samples = int(len(r))
        # cache the radial CDF for sampling
        self._cdf = np.concatenate([[0.0], np.cumsum(counts / counts.sum())])

    def log_prob(self, e1, e2):
        """log pi0 at points (e1,e2); -inf outside the sampled support."""
        r = np.hypot(np.asarray(e1, dtype=float), np.asarray(e2, dtype=float))
        out = np.interp(r, self.centers, self.log_dens)
        return np.where(r <= self.r_max, out, -np.inf)

    def sheared_log_prob(self, e1, e2, g1, g2):
        """log pi_g at lensed-shape points (e1,e2): exact Mobius pullback (see class doc)."""
        e1 = np.asarray(e1, dtype=float)
        e2 = np.asarray(e2, dtype=float)
        gsq = float(g1) ** 2 + float(g2) ** 2
        if gsq >= 1.0:  # unphysical |g|>=1 (e.g. a diverged selfcal iterate): no support
            return np.full(np.broadcast(e1, e2).shape, -np.inf)
        i1, i2 = inverse_shear_to_ellipticity(e1, e2, g1, g2)
        denom = np.abs(1.0 - (g1 - 1j * g2) * (e1 + 1j * e2))  # |1 - conj(g) e'|
        log_jac = 2.0 * np.log(1.0 - gsq) - 4.0 * np.log(np.maximum(denom, 1e-12))
        return self.log_prob(i1, i2) + log_jac

    def sample(self, n, rng):
        """Draw intrinsic shapes from the fitted prior itself (inverse-CDF radius,
        uniform angle).  Used by the closure test so the generative model matches the
        estimation prior exactly (tower rule holds by construction)."""
        u = rng.random(n)
        r = np.interp(u, self._cdf, self.edges)
        phi = rng.random(n) * 2.0 * np.pi
        return r * np.cos(phi), r * np.sin(phi)


class PosteriorShapeEstimator:
    """Grid posterior-mean shape estimator for an e-flow-blind ConditionalMeanFlow."""

    def __init__(self, bundle: MeasurementModelBundle, grid, device=None):
        self.bundle = bundle
        self.device = torch.device(device) if device is not None else bundle.device
        model = bundle.model
        if not isinstance(model, ConditionalMeanFlow):
            raise TypeError("PosteriorShapeEstimator requires a ConditionalMeanFlow "
                            f"(got {type(model).__name__}); the grid trick needs the "
                            "location-family structure p(x|c)=p_resid(x-mu(c)|flow_ctx)")
        # FENCE (realisation-aware head). This class hand-rolls `x - mu(c)` in log_likelihood and
        # log_likelihood_marginal instead of calling model.log_prob. For ConditionalMeanFlowRA the
        # residual is `x - mu(c) - A(c, u)`, so the hand-rolled form would be WRONG (not merely
        # stale) and would fail SILENTLY. Raise instead; porting the grid trick to the RA head
        # needs A evaluated per grid point.
        if isinstance(model, ConditionalMeanFlowRA):
            raise NotImplementedError(
                "PosteriorShapeEstimator does not support ConditionalMeanFlowRA: its residual is "
                "x - mu(c) - A(c, u) and this class hand-rolls x - mu(c). Use a 'mean_affine' "
                "checkpoint, or extend the grid trick to evaluate A per grid point.")
        pre = bundle.condition_preprocessor
        feats = pre.feature_names
        for name in ("e1_input_p", "e2_input_p"):
            if name not in feats:
                raise KeyError(f"Feature set lacks {name}; nothing to integrate over")
        self.i1 = feats.index("e1_input_p")
        self.i2 = feats.index("e2_input_p")
        n_feat = len(feats)
        # location-family guarantee: the residual flow must be blind to e (+ flags)
        keep = set(int(i) for i in model.keep_indices.tolist())
        e_dims = {self.i1, self.i2}
        if pre.add_missing_indicators:
            e_dims |= {n_feat + self.i1, n_feat + self.i2}
        overlap = e_dims & keep
        if overlap:
            raise ValueError(f"Residual flow SEES e-context dims {sorted(overlap)} -> "
                             "not a location family in e; grid trick invalid")
        bad = _E_DERIVED_FEATURES & set(feats)
        if bad:
            raise ValueError(f"Feature set contains e-derived features {sorted(bad)}; "
                             "grid injection of e1/e2_input_p alone would be inconsistent")
        self.j1, self.j2 = shape_target_indices(bundle.target_transform.target_names)
        self.target_dim = int(bundle.target_transform.dim)
        if self.target_dim < 2:
            raise ValueError(
                f"Expected at least two measured outputs; got "
                f"{bundle.target_transform.target_names}"
            )

        grid = np.asarray(grid, dtype=np.float64)
        self.grid_raw = torch.as_tensor(grid, dtype=torch.float32, device=self.device)
        # standardize grid e-values exactly as the trained preprocessor would
        g_std = np.stack([
            (grid[:, 0] - pre.means[self.i1]) / pre.scales[self.i1],
            (grid[:, 1] - pre.means[self.i2]) / pre.scales[self.i2],
        ], axis=1)
        self.grid_std = torch.as_tensor(g_std, dtype=torch.float32, device=self.device)
        self.G = grid.shape[0]
        self.n_feat = n_feat
        self.grid_np = grid

    def _grid_tiled_context(self, frame):
        """(B,G,D) standardized context tile with the grid e-values injected."""
        pre = self.bundle.condition_preprocessor
        ctx = torch.as_tensor(pre.transform_frame(frame), dtype=torch.float32,
                              device=self.device)  # (B,D)
        b = ctx.shape[0]
        rep = ctx[:, None, :].expand(b, self.G, ctx.shape[1]).contiguous()
        rep[:, :, self.i1] = self.grid_std[:, 0]
        rep[:, :, self.i2] = self.grid_std[:, 1]
        if pre.add_missing_indicators:
            rep[:, :, self.n_feat + self.i1] = 0.0   # grid e is never missing
            rep[:, :, self.n_feat + self.i2] = 0.0
        return rep

    def set_mu_correction(self, table_raw, cell_idx):
        """Arm a location correction: likelihood location becomes mu + U(e_grid; cell).

        table_raw: (K, G, 2) offsets in RAW target units, evaluated on THIS grid --
        the g0-measured location-vs-e misfit U (population-mean-free per cell, so the
        residual flow's own learned mean stays aligned; WORKLOG cont.33).
        cell_idx: (N,) int per-object cell assignment (e.g. r-mag bin).

        NOTE: only the conditional `log_likelihood` consumes the armed correction;
        `log_likelihood_marginal` ignores it (the driver refuses the combination)."""
        t = np.asarray(table_raw, dtype=np.float32)
        if t.ndim != 3 or t.shape[1] != self.G or t.shape[2] != 2:
            raise ValueError(f"mu-correction table shape {t.shape} != (K, {self.G}, 2)")
        scales = self.bundle.target_transform.scales
        shape_scales = np.asarray(scales)[[self.j1, self.j2]]
        self._mu_corr = torch.as_tensor(t / shape_scales[None, None, :],
                                        dtype=torch.float32, device=self.device)
        self._mu_corr_cells = np.asarray(cell_idx, dtype=np.int64)

    @torch.no_grad()
    def log_likelihood(self, frame, ehat_raw, chunk=1024, out_dtype=np.float16,
                       row_offset=0):
        """log p(ehat | e_grid, rest) for every (galaxy, grid point).

        frame: rescale()d dataframe carrying all conditioning columns (the e columns
        are placeholders -- they are overwritten by grid values).  ehat_raw: (N,2)
        measured outputs in raw target units.  Returns an (N,G) array (fp16 by default
        so the gold run can hold both signs in RAM and reweight under many priors
        without re-evaluating the flow).  row_offset: index of frame's first row in
        the full-run row order -- needed to look up armed per-object state (the
        mu-correction cells) when the caller slabs the frame."""
        model = self.bundle.model
        tstd = self.bundle.target_transform
        ehat_std = tstd.transform_array(np.asarray(ehat_raw, dtype=np.float32))
        n = len(frame)
        corr = getattr(self, "_mu_corr", None)
        out = np.empty((n, self.G), dtype=out_dtype)
        for start in range(0, n, chunk):
            stop = min(start + chunk, n)
            rep = self._grid_tiled_context(frame.iloc[start:stop])
            b = rep.shape[0]
            flat = rep.view(b * self.G, -1)
            mu = model._mu(flat)                                        # (B*G,2)
            if corr is not None:
                cells = self._mu_corr_cells[row_offset + start:row_offset + stop]
                delta = corr[torch.as_tensor(cells, device=self.device)].view(b * self.G, 2)
                mu = mu.clone()
                mu[:, self.j1] += delta[:, 0]
                mu[:, self.j2] += delta[:, 1]
            xh = torch.as_tensor(ehat_std[start:stop], dtype=torch.float32,
                                 device=self.device)
            x = xh[:, None, :].expand(b, self.G, self.target_dim).reshape(
                b * self.G, self.target_dim
            ) - mu
            ll = model.flow.log_prob(x, model._flow_ctx(flat)).view(b, self.G)
            # fp16 range guard (same as log_likelihood_marginal): softmax over the grid
            # is shift-invariant per row, so store row-max-shifted values.  Stored rows
            # are only meaningful up to a per-row constant (log_evidence too).
            ll = ll - ll.max(dim=1, keepdim=True).values
            out[start:stop] = ll.cpu().numpy().astype(out_dtype)
        return out

    @torch.no_grad()
    def log_likelihood_marginal(self, frame, ehat_raw, pool, sample_idx,
                                feature_names=("nbr_flux_near", "nbr_flux_far",
                                               "nbr_flux_max"),
                                chunk=256, out_dtype=np.float16):
        """Marginal log p(ehat | e_grid, theta_hat): the blend conditioning theta_b
        (`feature_names`) integrated over an empirical prior -- the cont.22 dtheta_b
        stage, i.e. the deployment estimator when the TRUE neighbour fluxes are not
        available:

            log p(ehat | e, rest) = logsumexp_m log p(ehat | e, rest, theta_b_m) - log M

        pool: (P,F) raw values of the theta_b features (a population sample -- e.g.
        the detected+selected rows themselves, so pi(theta_b) is the detected
        population's).  sample_idx: (N,M) per-galaxy row indices into pool; PER-GALAXY
        draws, so the finite-M quadrature noise averages out of <etilde> instead of
        biasing it coherently.  Costs M x log_likelihood."""
        model = self.bundle.model
        pre = self.bundle.condition_preprocessor
        feats = pre.feature_names
        jj = [feats.index(n) for n in feature_names]
        if set(jj) & {self.i1, self.i2}:
            raise ValueError("theta_b features overlap the shape dims")
        pool = np.asarray(pool, dtype=np.float64)
        if pool.shape[1] != len(jj):
            raise ValueError(f"pool has {pool.shape[1]} columns for {len(jj)} features")
        pool_std = np.stack([(pool[:, f] - pre.means[j]) / pre.scales[j]
                             for f, j in enumerate(jj)], axis=1).astype(np.float32)
        pool_t = torch.as_tensor(pool_std, device=self.device)
        sidx = torch.as_tensor(np.asarray(sample_idx, dtype=np.int64), device=self.device)
        m = int(sidx.shape[1])
        tstd = self.bundle.target_transform
        ehat_std = tstd.transform_array(np.asarray(ehat_raw, dtype=np.float32))
        n = len(frame)
        out = np.empty((n, self.G), dtype=out_dtype)
        logm = float(np.log(m))
        for start in range(0, n, chunk):
            stop = min(start + chunk, n)
            base = self._grid_tiled_context(frame.iloc[start:stop])   # (B,G,D)
            b = base.shape[0]
            xh = torch.as_tensor(ehat_std[start:stop], dtype=torch.float32,
                                 device=self.device)
            x = xh[:, None, :].expand(b, self.G, self.target_dim).reshape(
                b * self.G, self.target_dim
            )
            acc = torch.full((b, self.G), float("-inf"), device=self.device)
            for k in range(m):
                rep = base.clone()
                vals = pool_t[sidx[start:stop, k]]                    # (B,F)
                for f, j in enumerate(jj):
                    rep[:, :, j] = vals[:, f:f + 1]
                    if pre.add_missing_indicators:
                        rep[:, :, self.n_feat + j] = 0.0
                flat = rep.view(b * self.G, -1)
                mu = model._mu(flat)
                ll = model.flow.log_prob(x - mu, model._flow_ctx(flat)).view(b, self.G)
                acc = torch.logaddexp(acc, ll)
            res = acc - logm
            # fp16 range guard: extreme draws push loglike past -65504 -> -inf in the
            # cast (job 15064545 stage C warned here); a full -inf row would NaN the
            # posterior.  Softmax over the grid is shift-invariant per row, so store
            # row-max-shifted values (row max becomes 0) -- exact for all reweighting.
            res = res - res.max(dim=1, keepdim=True).values
            out[start:stop] = res.cpu().numpy().astype(out_dtype)
        return out

    @torch.no_grad()
    def posterior_mean(self, loglike, log_prior, chunk=65536, case_idx=None):
        """etilde from stored log-likelihoods under a (new) prior -- reweighting only,
        no flow evaluation.  Runs chunked on self.device, so a full 8M x G pass takes
        seconds, not minutes (prior modes and selfcal passes stay cheap at full scale).

        log_prior: (G,) shared row, (N,G) per-object, or (K,G) per-case together with
        case_idx (N,) mapping each row to its prior row (avoids expanding/fancy-indexing
        the big array per case).  loglike may be an fp16 np.memmap (--loglike-cache).
        Returns (etilde (N,2) float64, log_evidence (N,) float64 up to the flat-grid
        constant)."""
        lp = np.asarray(log_prior, dtype=np.float32)
        n = loglike.shape[0]
        et = np.empty((n, 2), dtype=np.float64)
        lz = np.empty(n, dtype=np.float64)
        lp_t = torch.as_tensor(lp, device=self.device)
        ci = None
        if case_idx is not None:
            if lp_t.ndim != 2:
                raise ValueError("case_idx requires a (K,G) log_prior matrix")
            ci = torch.as_tensor(np.asarray(case_idx, dtype=np.int64), device=self.device)
        for start in range(0, n, chunk):
            stop = min(start + chunk, n)
            arr = np.ascontiguousarray(loglike[start:stop])
            if not arr.flags.writeable:  # read-only mmap slice -> torch wants a copy
                arr = arr.copy()
            ll = torch.as_tensor(arr, device=self.device).float()
            if lp_t.ndim == 1:
                ll = ll + lp_t
            elif ci is not None:
                ll = ll + lp_t[ci[start:stop]]
            else:
                ll = ll + lp_t[start:stop]
            mx = ll.max(dim=1, keepdim=True).values
            w = torch.exp(ll - mx)
            s = w.sum(dim=1)
            e = (w @ self.grid_raw) / s[:, None]
            et[start:stop] = e.double().cpu().numpy()
            lz[start:stop] = (torch.log(s) + mx[:, 0]).double().cpu().numpy()
        return et, lz

    def estimate(self, frame, ehat_raw, log_prior, chunk=1024):
        """Convenience one-shot: log_likelihood + posterior_mean."""
        ll = self.log_likelihood(frame, ehat_raw, chunk=chunk)
        return self.posterior_mean(ll, log_prior)


def shear_prior_matrix(prior: RadialShapePrior, grid, g1_arr, g2_arr):
    """Stack sheared-prior rows: (K,G) log pi_{g_k}(grid) for K shear vectors."""
    g1_arr = np.atleast_1d(np.asarray(g1_arr, dtype=float))
    g2_arr = np.atleast_1d(np.asarray(g2_arr, dtype=float))
    rows = [prior.sheared_log_prob(grid[:, 0], grid[:, 1], float(a), float(b))
            for a, b in zip(g1_arr, g2_arr)]
    return np.stack(rows, axis=0)
