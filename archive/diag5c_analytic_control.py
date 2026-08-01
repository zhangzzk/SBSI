"""ANALYTIC CONTROL for the INFERENCE.md 5C Lagrangian score estimator.

Question this answers
---------------------
The 5C closure on the trained Gold-V2 flow fails the Bartlett information equality
`Var(s - <s>_sel) = <I> - I_sel` by factors of 7-30.  Is that a property of the 5C
MACHINERY + SCENE BANK (importance sampling a ~18-D latent scene with ESS ~ 40-100), or a
property of the TRAINED FLOW?

This script keeps EVERYTHING except the flow:

  * the real catalogue scenes (`load_rows`, same checkpoint `true_cut`),
  * the same Mobius primary-only shear map (`apply_shear_to_ellipticity`, verified below
    against a closed-form derivative),
  * the same bank construction (nodes = first K in-domain rows; galaxies pinned at a fixed
    offset so a K-ladder does NOT move the galaxy sample),
  * the same assembly helpers `posterior_weights` / `score_and_information` /
    `selection_terms` / `shear_estimate_*` / `denominator_consistency`,
  * the same central-difference stencil in gamma,
  * data DRAWN FROM THE MODEL at `g_true`, then Bernoulli-thinned by Pdet.

and replaces `log p_flow(xhat | ctx(z))` by a Gaussian emission
`xhat ~ N(m(z, gamma), diag(sigma^2))` whose derivatives in gamma are available in CLOSED
FORM.  The closed forms are checked against the finite differences that the real script
uses, so an FD/analytic mismatch cannot hide inside the result.

The sweep that matters is the CONDITIONING DIMENSION `d` of `m` at fixed shear
information: dims beyond the sheared shape carry (almost) no shear signal, they only force
the bank to match more of the scene, which sharpens the posterior and lowers ESS.  If the
mixture/curse-of-dimensionality story is right, the ratio must degrade with `d` at fixed K.

An `oracle` row is printed for every configuration: the same estimator with each galaxy's
OWN scene as a one-node bank.  There the mixture is exact, so the Bartlett ratio must be 1
(up to Monte-Carlo error on N galaxies).  It localises any failure: oracle != 1 means the
assembly or the population terms are wrong; oracle == 1 with bank != 1 means the bank is.

Usage
-----
    python scripts/diag5c_analytic_control.py --n-gal 2000 --nodes 500,2000,10000 \
        --gammas 0.0,0.05 --dimsets d2,d4,d8,d14
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

SBSI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (SBSI_ROOT, os.path.join(SBSI_ROOT, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from closure_v2_lagrangian import load_rows  # noqa: E402
from sbs_shear.lagrangian_score import (  # noqa: E402
    denominator_consistency,
    posterior_weights,
    score_and_information,
    selection_terms,
    shear_estimate_bartlett,
    shear_estimate_louis,
)
from sbs_shear.shear_map import apply_shear_to_ellipticity  # noqa: E402

from train_joint_forward import intrinsic_shape  # noqa: E402

CAT = ("/project/ls-gruen/users/zekang.zhang/sbsi_catalogues/"
       "det_meas_ngmix_ap7_g0.0_train.feather")
TRUE_CUT = (0.3, 26.0)          # Re_input_p > 0.3, r_input_p < 26.0 (checkpoint metadata)


# ----------------------------------------------------------------------------------------
# the shear map and its closed-form gamma-derivatives
# ----------------------------------------------------------------------------------------

def sheared_shape(e1, e2, t):
    """`eps' = (eps + i t)/(1 - i t eps)` and its first two derivatives in `t`.

    `t` is gamma_2 (shear applied on axis 1, exactly as `closure_v2_lagrangian` does with
    `g_true = (0.0, gamma)`).  With `u = eps + i t`, `v = 1 - i t eps`:

        f   = u / v
        f'  = i (v + eps u) / v^2                (the numerator has zero t-derivative)
        f'' = -2 eps (v + eps u) / v^3

    At t = 0 this reproduces `shear_map.shear_jacobian_at_zero`'s gamma_2 column,
    `de1'/dg2 = -2 e1 e2`, `de2'/dg2 = 1 + e1^2 - e2^2`.
    """
    eps = np.asarray(e1, float) + 1j * np.asarray(e2, float)
    u = eps + 1j * t
    v = 1.0 - 1j * t * eps
    f = u / v
    fp = 1j * (v + eps * u) / v ** 2
    fpp = -2.0 * eps * (v + eps * u) / v ** 3
    return f, fp, fpp


# ----------------------------------------------------------------------------------------
# scene arrays
# ----------------------------------------------------------------------------------------

# `nbg` (= `neighbored`) is NOT usable as an emission dim: every in-domain row of this
# catalogue has neighbored=True, so its population std is exactly 0.  Keeping it gave
# sigma = 1e-8, |z|^2 ~ 1e16, and the expanded quadratic form lost all precision -- which
# showed up as finite differences disagreeing with the closed form by ~1e3 sd.  Dropped.
DIMSETS = {
    "d2":  ["e1", "e2"],
    "d4":  ["e1", "e2", "mag", "logT"],
    "d8":  ["e1", "e2", "mag", "logT", "sersic_p", "z_p", "dist", "dmag_s"],
    "d13": ["e1", "e2", "mag", "logT", "sersic_p", "z_p", "dist", "dmag_s",
            "e1s", "e2s", "Re_s", "sersic_s", "z_s"],
}
SHEARED_DIMS = {"e1", "e2", "logT"}          # everything else is gamma-independent


def scene_arrays(df):
    """Everything the emission mean needs, as plain numpy, from raw catalogue columns."""
    e1, e2 = intrinsic_shape(df, "p")
    out = dict(e1_0=np.asarray(e1, float), e2_0=np.asarray(e2, float))
    g = lambda c: df[c].to_numpy(float)                                   # noqa: E731
    out["mag"] = g("r_input_p")
    out["logT0"] = np.log(g("Re_input_p"))
    out["sersic_p"] = g("sersic_n_input_p")
    out["z_p"] = g("redshift_input_p")
    nbg = df["neighbored"].to_numpy(bool)
    out["nbg"] = nbg.astype(float)
    out["dist"] = np.where(nbg, g("distance"), np.nan)
    e1s, e2s = intrinsic_shape(df, "s")
    out["e1s"] = np.where(nbg, np.asarray(e1s, float), np.nan)
    out["e2s"] = np.where(nbg, np.asarray(e2s, float), np.nan)
    out["dmag_s"] = np.where(nbg, g("r_input_s") - g("r_input_p"), np.nan)
    out["Re_s"] = np.where(nbg, g("Re_input_s"), np.nan)
    out["sersic_s"] = np.where(nbg, g("sersic_n_input_s"), np.nan)
    out["z_s"] = np.where(nbg, g("redshift_input_s"), np.nan)
    return out


def fill_and_scale(sc, fills):
    """NaN-fill the neighbour columns with the population mean (what `nbr_std` does)."""
    out = dict(sc)
    for k, v in fills.items():
        if k in out:
            a = out[k].copy()
            a[~np.isfinite(a)] = v
            out[k] = a
    return out


def emission(sc, t, names, sigma):
    """`(M, dM/dt, d2M/dt2)` as `(K, d)` arrays for the requested dim names.

    `logT(t) = logT0 + 2 * (e(t) . gamma) = logT0 + 2 t Im f(t)` -- the standard
    `dlogT = 2 e.g` size response, evaluated on the SHEARED shape so it carries curvature.
    """
    f, fp, fpp = sheared_shape(sc["e1_0"], sc["e2_0"], t)
    cols, dcols, ddcols = [], [], []
    for nm in names:
        if nm == "e1":
            cols.append(np.real(f)); dcols.append(np.real(fp)); ddcols.append(np.real(fpp))
        elif nm == "e2":
            cols.append(np.imag(f)); dcols.append(np.imag(fp)); ddcols.append(np.imag(fpp))
        elif nm == "logT":
            cols.append(sc["logT0"] + 2.0 * t * np.imag(f))
            dcols.append(2.0 * np.imag(f) + 2.0 * t * np.imag(fp))
            ddcols.append(4.0 * np.imag(fp) + 2.0 * t * np.imag(fpp))
        else:
            cols.append(sc[nm]); dcols.append(np.zeros_like(sc[nm]))
            ddcols.append(np.zeros_like(sc[nm]))
    M = np.stack(cols, axis=1)
    dM = np.stack(dcols, axis=1)
    ddM = np.stack(ddcols, axis=1)
    del sigma
    return M, dM, ddM


def logT_curve(sc, t):
    """`logT(t)` and its two derivatives -- needed by the detection channel."""
    f, fp, fpp = sheared_shape(sc["e1_0"], sc["e2_0"], t)
    L = sc["logT0"] + 2.0 * t * np.imag(f)
    Lp = 2.0 * np.imag(f) + 2.0 * t * np.imag(fp)
    Lpp = 4.0 * np.imag(fp) + 2.0 * t * np.imag(fpp)
    return L, Lp, Lpp


def detection(sc, t, a, b, c):
    """`Pdet = sigmoid(a - b*mag + c*logT(t))` with closed-form `log Pdet` derivatives.

    The size channel is what gives detection a genuine gamma-dependence, so the population
    terms `<s>_sel` / `I_sel` are actually exercised rather than being identically zero.
    """
    L, Lp, Lpp = logT_curve(sc, t)
    h = a - b * sc["mag"] + c * L
    P = 1.0 / (1.0 + np.exp(-h))
    hp, hpp = c * Lp, c * Lpp
    log_p = -np.logaddexp(0.0, -h)
    dlog = (1.0 - P) * hp
    ddlog = (1.0 - P) * hpp - P * (1.0 - P) * hp ** 2
    return P, log_p, dlog, ddlog


# ----------------------------------------------------------------------------------------
# phi and its derivatives
# ----------------------------------------------------------------------------------------

def phi_analytic(x, M, dM, ddM, sig, log_p, dlog, ddlog, ctr=None):
    """`(phi, dphi, ddphi)` as `(N, K)`, all closed form.

        phi_ik   = -0.5 |(x_i - M_k)/sig|^2 - sum log(sig sqrt(2 pi)) + log Pdet_k
        dphi_ik  =  sum_j (z_ij - mu_kj) dmu_kj                 + dlogPdet_k
        ddphi_ik =  sum_j [(z_ij - mu_kj) ddmu_kj - dmu_kj^2]   + ddlogPdet_k
    with z = x/sig, mu = M/sig.
    """
    # CENTRE before scaling.  `q` is formed by the |z|^2 - 2 z.mu + |mu|^2 expansion (a
    # matmul, which is what makes an (N, K) block affordable); that expansion cancels
    # catastrophically when |z| >> spread, so an uncentred dim with a small sigma destroys
    # phi to O(1).  Subtracting the same constant from x and M leaves q exactly invariant.
    c = 0.0 if ctr is None else ctr[None, :]
    z = (x - c) / sig
    mu = (M - c) / sig
    dmu = dM / sig
    ddmu = ddM / sig
    const = -0.5 * x.shape[1] * np.log(2 * np.pi) - np.log(sig).sum()
    q = ((z ** 2).sum(1)[:, None] - 2.0 * (z @ mu.T) + (mu ** 2).sum(1)[None, :])
    phi = -0.5 * q + const + log_p[None, :]
    d1 = (z @ dmu.T) - (mu * dmu).sum(1)[None, :] + dlog[None, :]
    d2 = (z @ ddmu.T) - (mu * ddmu).sum(1)[None, :] - (dmu ** 2).sum(1)[None, :] + ddlog[None, :]
    return phi, d1, d2


def phi_value(x, M, sig, log_p, ctr=None):
    c = 0.0 if ctr is None else ctr[None, :]
    z = (x - c) / sig
    mu = (M - c) / sig
    const = -0.5 * x.shape[1] * np.log(2 * np.pi) - np.log(sig).sum()
    q = ((z ** 2).sum(1)[:, None] - 2.0 * (z @ mu.T) + (mu ** 2).sum(1)[None, :])
    return -0.5 * q + const + log_p[None, :]


# ----------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalogue", default=CAT)
    ap.add_argument("--n-gal", type=int, default=2000)
    ap.add_argument("--nodes", default="500,2000,10000")
    ap.add_argument("--gammas", default="0.0,0.05")
    ap.add_argument("--dimsets", default="d2,d4,d8,d14")
    ap.add_argument("--gal-offset", type=int, default=None,
                    help="fixed row offset for the galaxy sample; defaults to max(nodes) so "
                         "the K ladder never moves the galaxies")
    ap.add_argument("--delta", type=float, default=0.01)
    ap.add_argument("--sigma-e", default="0.3",
                    help="comma list: emission noise on the shape dims, natural units.  "
                         "Shrinking it sharpens the posterior WITHOUT changing d, which is "
                         "the control that separates dimension from sharpness")
    ap.add_argument("--sigma-nuis", default="0.5",
                    help="comma list: emission noise on every non-shape dim, in units of "
                         "that dim's population std.  <1 makes the dim discriminative for "
                         "the bank, which is the knob that drives ESS down at fixed d")
    ap.add_argument("--det-b", type=float, default=1.0, help="Pdet magnitude slope")
    ap.add_argument("--det-c", type=float, default=1.0, help="Pdet log-size slope")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--raw-rows", type=int, default=None,
                    help="how many raw catalogue rows to read before the true_cut")
    ap.add_argument("--dedupe-bank", action="store_true",
                    help="keep one row per `input_index`.  The catalogue repeats each "
                         "primary ~8 times with different neighbours/cases, so a K-row bank "
                         "holds only ~K/8 DISTINCT primaries -- and S_g shears the primary "
                         "only, so that is the bank's real resolution in the shear "
                         "coordinate.  This flag separates 'more rows' from 'more support'.")
    ap.add_argument("--no-fd-check", action="store_true")
    args = ap.parse_args()

    node_list = [int(v) for v in args.nodes.split(",")]
    gam_list = [float(v) for v in args.gammas.split(",")]
    dim_list = [v.strip() for v in args.dimsets.split(",")]
    sig_list = [float(v) for v in str(args.sigma_nuis).split(",")]
    sige_list = [float(v) for v in str(args.sigma_e).split(",")]
    off = args.gal_offset if args.gal_offset is not None else max(node_list)
    kmax = max(node_list)
    if off < kmax:
        raise SystemExit("--gal-offset must be >= max(--nodes) or the bank overlaps the galaxies")

    need = off + args.n_gal
    rows = load_rows(args.catalogue, args.raw_rows or need * 5)
    keep = ((rows["Re_input_p"].to_numpy(float) > TRUE_CUT[0])
            & (rows["r_input_p"].to_numpy(float) < TRUE_CUT[1]))
    print(f"true_cut Re>{TRUE_CUT[0]} & mag<{TRUE_CUT[1]}: "
          f"{int(keep.sum()):,}/{len(rows):,} = {keep.mean():.1%} kept")
    rows = rows[keep].reset_index(drop=True)
    if args.dedupe_bank:
        n_before = len(rows)
        rows = rows.drop_duplicates(subset="input_index").reset_index(drop=True)
        print(f"dedupe-bank: {len(rows):,}/{n_before:,} = {len(rows) / n_before:.1%} rows "
              f"survive as DISTINCT primaries")
    if len(rows) < need:
        raise SystemExit(f"only {len(rows):,} in-domain rows, need {need:,}")
    print(f"rows: bank from [0,{kmax}), galaxies from [{off},{off + args.n_gal})  "
          f"(pinned: the K ladder does NOT move the galaxy sample)")

    bank_all = scene_arrays(rows.iloc[:kmax].reset_index(drop=True))
    gal_sc_raw = scene_arrays(rows.iloc[off:off + args.n_gal].reset_index(drop=True))

    # NaN fills and per-dim scales come from the FULL bank, once, so the d-ladder is nested
    # and the K-ladder shares the same units.
    fills = {}
    for k, v in bank_all.items():
        m = np.isfinite(v)
        fills[k] = float(v[m].mean()) if m.any() else 0.0
    bank_all = fill_and_scale(bank_all, fills)
    gal_sc = fill_and_scale(gal_sc_raw, fills)

    scales = {}
    for nm in set(sum(DIMSETS.values(), [])):
        if nm in ("e1", "e2"):
            continue
        key = "logT0" if nm == "logT" else nm
        scales[nm] = float(np.std(bank_all[key]))
    # detection intercept: <Pdet> ~ 0.5 on the bank population
    a0 = args.det_b * float(np.median(bank_all["mag"])) - args.det_c * float(
        np.median(bank_all["logT0"]))
    P0, _, _, _ = detection(bank_all, 0.0, a0, args.det_b, args.det_c)
    print(f"detection: Pdet = sigmoid({a0:.3f} - {args.det_b}*mag + {args.det_c}*logT),  "
          f"<Pdet> on the bank = {P0.mean():.4f}")

    def sigma_for(names, snuis, sige):
        s = []
        for nm in names:
            s.append(sige if nm in ("e1", "e2") else snuis * max(scales[nm], 1e-8))
        return np.asarray(s, float)

    def ratio_ci(s, info, s_sel, i_sel, nboot=400, seed=7):
        """Bootstrap over galaxies: the ratio's own Monte-Carlo error bar.

        `Var(s)` on ~1e3 objects with a heavy-tailed score is not a precise number, so a
        ratio of 1.1 and a ratio of 30 are not the same kind of statement.  Without this
        the d-ladder cannot be read.
        """
        rng = np.random.default_rng(seed)
        n = s.size
        out = np.empty(nboot)
        for b in range(nboot):
            j = rng.integers(0, n, n)
            bd = float(np.mean((s[j] - s_sel) ** 2))
            ld = float(np.mean(info[j]) - i_sel)
            out[b] = bd / ld if ld != 0 else np.nan
        return float(np.nanpercentile(out, 16)), float(np.nanpercentile(out, 84))

    hdr = (f"{'dims':>5} {'sig_e':>6} {'sig_n':>6} {'K':>6} {'g_true':>7} {'bank':>7} "
           f"{'ESS':>8} {'<s>':>9} {'<s>_sel':>9} {'Var(s-c)':>10} {'<I>-Isel':>10} "
           f"{'RATIO':>8} {'[16,84]%':>17} {'RATIO_fd':>9} {'ghat5.8':>9} {'ghat5.9':>9}")
    print("\n" + hdr)
    print("-" * len(hdr))

    for dname, snuis, sige in [(dn, sn, se) for dn in dim_list
                               for sn in sig_list for se in sige_list]:
        names = DIMSETS[dname]
        sig = sigma_for(names, snuis, sige)
        bad = [nm for nm in names if nm not in ("e1", "e2") and scales[nm] < 1e-6]
        if bad:
            raise SystemExit(f"zero-variance emission dim(s) {bad}: they carry no "
                             f"information and destroy the conditioning of phi")
        ctr = emission({k: v[:2000] for k, v in bank_all.items()}, 0.0, names,
                       sig)[0].mean(axis=0)
        d = len(names)
        for gtrue in gam_list:
            # ---- data drawn FROM the model at g_true, then Bernoulli detection ----------
            rng = np.random.default_rng(args.seed + int(round(1000 * gtrue)))
            Mg, _, _ = emission(gal_sc, gtrue, names, sig)
            xhat_all = Mg + rng.normal(size=Mg.shape) * sig[None, :]
            Pg, _, _, _ = detection(gal_sc, gtrue, a0, args.det_b, args.det_c)
            det = rng.random(len(Pg)) < Pg
            xhat = xhat_all[det]
            n_keep = int(det.sum())

            # ---- population terms, by the SAME finite differences as the real script ----
            def pop_terms(sc_bank, dl):
                lp = {}
                for t in (0.0, +dl, -dl):
                    P, _, _, _ = detection(sc_bank, t, a0, args.det_b, args.det_c)
                    lp[t] = float(np.log(P.mean()))
                return selection_terms(lp[0.0], (lp[+dl] - lp[-dl]) / (2 * dl),
                                       (lp[+dl] - 2 * lp[0.0] + lp[-dl]) / dl ** 2)

            for K in node_list:
                bank = {k: v[:K] for k, v in bank_all.items()}
                s_sel, i_sel = pop_terms(bank, args.delta)
                M0, dM0, ddM0 = emission(bank, 0.0, names, sig)
                _, lp0, dlp0, ddlp0 = detection(bank, 0.0, a0, args.det_b, args.det_c)
                phi0, d1, d2 = phi_analytic(xhat, M0, dM0, ddM0, sig, lp0, dlp0, ddlp0,
                                            ctr=ctr)

                rfd = np.nan
                if not args.no_fd_check:
                    # the SAME central difference the real script uses, on the SAME phi.
                    # Printing it next to the closed form separates "FD is not converged"
                    # from "the estimator is broken".
                    dl = args.delta
                    pp_ = phi_value(xhat, emission(bank, +dl, names, sig)[0], sig,
                                    detection(bank, +dl, a0, args.det_b, args.det_c)[1],
                                    ctr=ctr)
                    pm_ = phi_value(xhat, emission(bank, -dl, names, sig)[0], sig,
                                    detection(bank, -dl, a0, args.det_b, args.det_c)[1],
                                    ctr=ctr)
                    d1f = (pp_ - pm_) / (2 * dl)
                    d2f = (pp_ - 2 * phi0 + pm_) / dl ** 2
                    r1 = np.max(np.abs(d1f - d1)) / max(np.std(d1), 1e-30)
                    r2 = np.max(np.abs(d2f - d2)) / max(np.std(d2), 1e-30)
                    sf, iff = score_and_information(phi0, d1f, d2f)
                    _, _, rfd = denominator_consistency(sf, iff, s_sel, i_sel)
                    del pp_, pm_, d1f, d2f, sf, iff
                    print(f"  [{dname} sig_e={sige:g} sig_n={snuis:g} K={K} g={gtrue:g}] FD vs analytic "
                          f"(delta={dl}): max|d1_fd-d1|/sd(d1)={r1:.2e}  "
                          f"max|d2_fd-d2|/sd(d2)={r2:.2e}")

                s, info = score_and_information(phi0, d1, d2)
                bd, ld, ratio = denominator_consistency(s, info, s_sel, i_sel)
                lo, hi = ratio_ci(s, info, s_sel, i_sel)
                w = posterior_weights(phi0)
                ess = float(np.mean(1.0 / np.sum(w ** 2, axis=1)))
                lou = shear_estimate_louis(s, info, s_sel, i_sel)
                bar = shear_estimate_bartlett(s, s_sel)
                print(f"{d:>5d} {sige:>6.3f} {snuis:>6.2f} {K:>6d} {gtrue:>7.3f} {'sample':>7} "
                      f"{ess:>8.1f} {np.mean(s):>9.4f} {s_sel:>9.4f} {bd:>10.3f} "
                      f"{ld:>10.3f} {ratio:>+8.3f} [{lo:>+7.3f},{hi:>+7.3f}] "
                      f"{rfd:>+9.3f} {lou:>+9.5f} {bar:>+9.5f}")
                del phi0, d1, d2, w

            # ---- ORACLE: each galaxy's own scene as a one-node bank ---------------------
            gsel = {k: v[det] for k, v in gal_sc.items()}
            Mo, dMo, ddMo = emission(gsel, 0.0, names, sig)
            _, lpo, dlpo, ddlpo = detection(gsel, 0.0, a0, args.det_b, args.det_c)
            zz = (xhat - Mo) / sig
            dmu, ddmu = dMo / sig, ddMo / sig
            s_o = (zz * dmu).sum(1) + dlpo
            d2_o = (zz * ddmu).sum(1) - (dmu ** 2).sum(1) + ddlpo
            info_o = -d2_o                       # Var_w(phi') = 0 for a one-node bank
            s_sel_o, i_sel_o = pop_terms(bank_all, args.delta)
            bd, ld, ratio = denominator_consistency(s_o, info_o, s_sel_o, i_sel_o)
            lo, hi = ratio_ci(s_o, info_o, s_sel_o, i_sel_o)
            lou = shear_estimate_louis(s_o, info_o, s_sel_o, i_sel_o)
            bar = shear_estimate_bartlett(s_o, s_sel_o)
            print(f"{d:>5d} {sige:>6.3f} {snuis:>6.2f} {n_keep:>6d} {gtrue:>7.3f} {'ORACLE':>7} "
                  f"{1.0:>8.1f} {np.mean(s_o):>9.4f} {s_sel_o:>9.4f} {bd:>10.3f} "
                  f"{ld:>10.3f} {ratio:>+8.3f} [{lo:>+7.3f},{hi:>+7.3f}] {np.nan:>+9.3f} "
                  f"{lou:>+9.5f} {bar:>+9.5f}")
            print(f"      (detected {n_keep:,}/{args.n_gal:,} = {n_keep / args.n_gal:.1%})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
