"""CHANNEL SPLIT of the shear derivative of the 5C log-density (hypothesis B).

`ConditionalMeanFlow.log_prob(x, c) = flow.log_prob(x - mu(c), flow_ctx(c))`, so gamma
enters the per-node curve `phi_k(g)` through THREE separate doors:

  (i)   MEAN  channel: mu(ctx(S_g z)) moves the residual x - mu.  This is the only door
        the training loss supervises through a shear-response term.
  (ii)  SHAPE channel: flow_ctx(ctx(S_g z)) changes the residual density itself.  The
        shear derivative of the density's SHAPE is unsupervised.
  (iii) DET   channel: log Pdet(ctx(S_g z)).  Per-node, no dependence on xhat.

Isolate (i) and (ii) by freezing one at its g=0 value while differencing the other:

    L[a][b] = flow.log_prob( xhat_i - mu(ctx(S_a z_k)),  flow_ctx(ctx(S_b z_k)) )

    phi_mean(g)  = L[g][0]      (shape frozen)
    phi_shape(g) = L[0][g]      (mean frozen)
    phi_flow(g)  = L[g][g]      (the real thing, minus the detection term)

All nine (a, b) in {0, +d, -d}^2 are evaluated, which additionally gives the MIXED second
derivative by the 4-point stencil, so the second-order accounting is closed exactly:

    phi_flow'    = phi_mean' + phi_shape'                     (+ O(d^2))
    phi_flow''   = phi_mean'' + phi_shape'' + 2 phi_cross''    (+ O(d^2))

and the residual of each identity is reported rather than assumed.

Blocks:
  (0) sanity: the (0,0) block reproduces `closure_v2_lagrangian.phi_block`; the two
      channel identities above hold to their stencil order.
  (1) posterior-weighted magnitudes of |phi_mean'| vs |phi_shape'| vs |phi_det'|.
  (2) the I decomposition -E_w[phi''] and Var_w(phi') split by channel, including the
      cross terms.  If the O(75) near-cancellation lives in the SHAPE channel, that is a
      direct hit on hypothesis (B).
  (3) s_i by channel and each channel's correlation with the full s_i.
  (4) how far off-manifold the weight-carrying nodes are: latent norm ||z|| (the flow's
      OWN conditional scale, since the flow pushes the residual to a standard normal),
      ||xhat - mu|| in standardized target units, and the log-weight deficit.
  (5) MATCHED-PAIR CONTROL: the same galaxies scored under their OWN scene, i.e. the only
      configuration the training loss ever saw.  Channel behaviour that is tame there and
      wild in the bank is the cleanest statement of (B).

Error bars: bootstrap over GALAXIES (200 resamples) on every aggregate that carries a
conclusion.  Nothing here is imported by other modules.
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

SBSI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (SBSI_ROOT, os.path.join(SBSI_ROOT, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from closure_v2_lagrangian import (  # noqa: E402
    CAT, CKPT, load_rows, phi_block, rebuild, scene_context,
)
from sbs_shear.lagrangian_score import (  # noqa: E402
    denominator_consistency, posterior_weights, selection_terms,
)
from train_joint_forward import intrinsic_shape  # noqa: E402

LOG2PI = float(np.log(2.0 * np.pi))


# ---------------------------------------------------------------------------------------
# the (a, b) block: mean context a, shape context b
# ---------------------------------------------------------------------------------------

@torch.no_grad()
def ll_block(model, xhat, mu_a, fctx_b, chunk, max_pairs=500_000, want_z=False):
    """`L[a][b]_ik = log p_flow( xhat_i - mu_a[k] | fctx_b[k] )`, an (N, K) array.

    Same chunking discipline as `closure_v2_lagrangian.phi_block`: the cap is on the
    PRODUCT chunk*K because the expanded context is (chunk*K, ctx_dim).
    """
    n, k = xhat.shape[0], mu_a.shape[0]
    chunk = max(1, min(chunk, max_pairs // max(k, 1)))
    out = np.empty((n, k), dtype=np.float64)
    zn = np.empty((n, k), dtype=np.float32) if want_z else None
    zmx = np.empty((n, k), dtype=np.float32) if want_z else None
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        b = e - s
        resid = (xhat[s:e, None, :] - mu_a[None, :, :]).reshape(b * k, -1)
        cc = fctx_b[None, :, :].expand(b, k, fctx_b.shape[1]).reshape(b * k, -1)
        z, logdet = model.mean_flow.flow.inverse(resid, cc)
        ll = -0.5 * (z.pow(2).sum(-1) + z.shape[-1] * LOG2PI) + logdet
        out[s:e] = ll.view(b, k).double().cpu().numpy()
        if want_z:
            zn[s:e] = z.pow(2).sum(-1).sqrt().view(b, k).cpu().numpy()
            zmx[s:e] = z.abs().max(-1).values.view(b, k).cpu().numpy()
    return (out, zn, zmx) if want_z else out


@torch.no_grad()
def ll_diag(model, xhat, mu_a, fctx_b, want_z=False):
    """The MATCHED case: galaxy i under its own scene i.  Returns (N,) not (N, K)."""
    z, logdet = model.mean_flow.flow.inverse(xhat - mu_a, fctx_b)
    ll = -0.5 * (z.pow(2).sum(-1) + z.shape[-1] * LOG2PI) + logdet
    if want_z:
        return (ll.double().cpu().numpy(),
                z.pow(2).sum(-1).sqrt().cpu().numpy(),
                z.abs().max(-1).values.cpu().numpy())
    return ll.double().cpu().numpy()


# ---------------------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------------------

def boot(x, rng, nboot=200):
    """(mean, se) of `x` over galaxies, bootstrap se."""
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    idx = rng.integers(0, n, size=(nboot, n))
    return float(x.mean()), float(np.std(x[idx].mean(axis=1), ddof=1))


def boot_med(x, rng, nboot=200):
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    idx = rng.integers(0, n, size=(nboot, n))
    return float(np.median(x)), float(np.std(np.median(x[idx], axis=1), ddof=1))


def row_wq(vals, w, qs):
    """Per-galaxy quantiles of `vals` (N, K) under the row measure `w` (N, K)."""
    o = np.argsort(vals, axis=1)
    v = np.take_along_axis(vals, o, 1)
    ww = np.take_along_axis(w, o, 1)
    c = np.cumsum(ww, axis=1)
    c = c / c[:, -1:]
    out = np.empty((vals.shape[0], len(qs)), dtype=np.float64)
    ar = np.arange(vals.shape[0])
    for j, q in enumerate(qs):
        out[:, j] = v[ar, np.argmax(c >= q, axis=1)]
    return out


def wmean(w, a):
    return np.sum(w * a, axis=1)


def wcov(w, a, b, ma=None, mb=None):
    ma = wmean(w, a) if ma is None else ma
    mb = wmean(w, b) if mb is None else mb
    return np.sum(w * a * b, axis=1) - ma * mb


def qline(tag, q, rng, qs):
    """Print per-galaxy weighted quantiles, median over galaxies with bootstrap se."""
    cells = []
    for j in range(len(qs)):
        m, se = boot_med(q[:, j], rng)
        cells.append(f"{m:>10.3f}+-{se:<7.3f}")
    print(f"    {tag:>18} " + " ".join(cells))


# ---------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=CKPT)
    ap.add_argument("--catalogue", default=CAT)
    ap.add_argument("--n-gal", type=int, default=4000)
    ap.add_argument("--n-node", type=int, default=2000)
    ap.add_argument("--gal-offset", type=int, default=20000)
    ap.add_argument("--gamma", type=float, default=0.0, help="true shear of the drawn data")
    ap.add_argument("--deltas", default="0.01,0.005")
    ap.add_argument("--chunk", type=int, default=250)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--nboot", type=int, default=200)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(args.device)
    rng = np.random.default_rng(args.seed + 777)
    print(f"device={args.device}  torch={torch.__version__}")

    model, pre, nbr_std, target_std, meta = rebuild(args.checkpoint, dev)
    keep_idx = model.mean_flow.keep_indices
    print(f"checkpoint: {os.path.basename(args.checkpoint)}")
    print(f"  target_dim={model.target_dim}  context_dim={model.context_dim}  "
          f"flow_ctx_dim={len(keep_idx)}  (flow sees "
          f"{'ALL' if len(keep_idx) == model.context_dim else 'A SUBSET OF'} the context)")

    span = max(args.n_gal + args.n_node, args.gal_offset + args.n_gal)
    rows = load_rows(args.catalogue, span * 4)
    tc = meta.get("true_cut")
    if tc is not None:
        keep_true = ((rows["Re_input_p"].to_numpy(float) > float(tc[0]))
                     & (rows["r_input_p"].to_numpy(float) < float(tc[1])))
        rows = rows[keep_true].reset_index(drop=True)
    if args.gal_offset < args.n_node:
        raise SystemExit("--gal-offset must be >= --n-node")
    if len(rows) < args.gal_offset + args.n_gal:
        raise SystemExit(f"only {len(rows):,} in-domain rows")
    node_df = rows.iloc[:args.n_node].reset_index(drop=True)
    gal_df = rows.iloc[args.gal_offset:args.gal_offset + args.n_gal].reset_index(drop=True)
    print(f"rows: {len(node_df):,} nodes + {len(gal_df):,} galaxies "
          f"(gal_offset={args.gal_offset}, PINNED)")

    def intr_of(df):
        d = {}
        d["e1p"], d["e2p"] = intrinsic_shape(df, "p")
        d["e1s"], d["e2s"] = intrinsic_shape(df, "s")
        return d

    node_intr, gal_intr = intr_of(node_df), intr_of(gal_df)

    # ---- data drawn FROM the model at the true shear ----------------------------------
    g_true = (0.0, float(args.gamma))
    ctx_g, pdet_g = scene_context(model, gal_df, gal_intr, g_true, pre, nbr_std, dev)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    with torch.no_grad():
        xhat = model.mean_flow.sample(ctx_g, n_samples=1)
        if xhat.dim() == 3:
            xhat = xhat[:, 0, :]
    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    keep = (torch.rand(len(gal_df), generator=gen).to(dev) < pdet_g)
    kept = torch.nonzero(keep).squeeze(-1).cpu().numpy()
    xhat = xhat[keep]
    n_keep = int(xhat.shape[0])
    print(f"detection: kept {n_keep:,}/{len(gal_df):,}  (<Pdet>={float(pdet_g.mean()):.4f})")

    deltas = [float(x) for x in args.deltas.split(",")]

    # ---- node-bank contexts at every stencil point ------------------------------------
    ctxs, pdets = {}, {}
    for t in sorted({0.0} | {s * d for d in deltas for s in (+1, -1)}):
        c, p = scene_context(model, node_df, node_intr, (0.0, t), pre, nbr_std, dev)
        ctxs[t], pdets[t] = c, p
    with torch.no_grad():
        mus = {t: model.mu(c) for t, c in ctxs.items()}
        fcx = {t: model.mean_flow._flow_ctx(c) for t, c in ctxs.items()}

    # ---- the shared (0,0) block, plus off-manifold geometry ---------------------------
    t0 = time.time()
    L00, zn00, zmx00 = ll_block(model, xhat, mus[0.0], fcx[0.0], args.chunk, want_z=True)
    logpd0 = torch.log(pdets[0.0]).double().cpu().numpy()
    p0 = L00 + logpd0[None, :]
    w = posterior_weights(p0)
    print(f"(0,0) block in {time.time() - t0:.0f}s")

    # SANITY: the (0,0) block must reproduce the production phi_block exactly.
    ref = phi_block(model, xhat[:64], ctxs[0.0], torch.log(pdets[0.0]), args.chunk)
    dd = np.abs(ref - p0[:64])
    print(f"  sanity vs closure_v2_lagrangian.phi_block: max|diff| = {dd.max():.3e}  "
          f"(rms {np.sqrt((dd ** 2).mean()):.3e})")

    print("\n" + "=" * 92)
    print("(4) OFF-MANIFOLD GEOMETRY OF THE BANK  (g = 0 contexts)")
    print("=" * 92)
    with torch.no_grad():
        mu0 = mus[0.0].double().cpu().numpy()
    xh = xhat.double().cpu().numpy()
    dist = np.sqrt(((xh[:, None, :] - mu0[None, :, :]) ** 2).sum(-1))
    deficit = p0 - p0.max(axis=1, keepdims=True)
    order = np.argsort(-w, axis=1)
    print(f"  target_dim = {model.target_dim}; a MATCHED residual has E||z||^2 = "
          f"{model.target_dim} i.e. ||z|| ~ {np.sqrt(model.target_dim):.2f}")
    print(f"  {'node set':>22} {'||z||':>16} {'max|z_d|':>16} {'||xhat-mu||':>16} "
          f"{'log-w deficit':>16}")
    for tag, k in (("top-1 weight", 1), ("top-10 weight", 10), ("top-100 weight", 100)):
        k = min(k, w.shape[1])
        sel = order[:, :k]
        zz = np.take_along_axis(zn00.astype(np.float64), sel, 1).mean(axis=1)
        zm = np.take_along_axis(zmx00.astype(np.float64), sel, 1).mean(axis=1)
        dsel = np.take_along_axis(dist, sel, 1).mean(axis=1)
        fsel = np.take_along_axis(deficit, sel, 1).mean(axis=1)
        cells = []
        for arr in (zz, zm, dsel, fsel):
            m, se = boot(arr, rng, args.nboot)
            cells.append(f"{m:>9.3f}+-{se:<6.3f}")
        print(f"  {tag:>22} " + " ".join(cells))
    # the whole bank, weighted by w
    fw = (w / w.shape[0]).ravel()
    def pooled_wq(v, qs=(0.5, 0.9, 0.99)):
        o = np.argsort(v)
        c = np.cumsum(fw[o]); c = c / c[-1]
        return np.interp(qs, c, v[o])
    print(f"  posterior-weighted over the WHOLE bank: ||z|| p50/p90/p99 = "
          + "/".join(f"{v:.2f}" for v in pooled_wq(zn00.ravel().astype(np.float64))))
    print(f"  uniform over the whole bank:            ||z|| p50/p90/p99 = "
          + "/".join(f"{v:.2f}" for v in np.percentile(zn00, [50, 90, 99])))
    print(f"  log-weight deficit: min {deficit.min():.1f}   "
          f"posterior-weighted p50/p90/p99 = "
          + "/".join(f"{v:.2f}" for v in pooled_wq(deficit.ravel())))
    ess = 1.0 / np.sum(w ** 2, axis=1)
    print(f"  ESS per galaxy: mean {ess.mean():.1f} of K={args.n_node}")

    # ==================================================================================
    # the channel split, per stencil width
    # ==================================================================================
    for d in deltas:
        print("\n" + "=" * 92)
        print(f"CHANNEL SPLIT at delta = {d}   (K={args.n_node}, N_det={n_keep}, "
              f"g_true={args.gamma})")
        print("=" * 92)
        t0 = time.time()
        L = {}
        for a in (+d, -d):
            L[(a, 0.0)] = ll_block(model, xhat, mus[a], fcx[0.0], args.chunk)
            L[(0.0, a)] = ll_block(model, xhat, mus[0.0], fcx[a], args.chunk)
        for a in (+d, -d):
            for b in (+d, -d):
                L[(a, b)] = ll_block(model, xhat, mus[a], fcx[b], args.chunk)
        print(f"  8 blocks in {time.time() - t0:.0f}s")

        lpd = {t: torch.log(pdets[t]).double().cpu().numpy() for t in (0.0, +d, -d)}
        d1_det = (lpd[+d] - lpd[-d]) / (2 * d)                       # (K,)
        d2_det = (lpd[+d] - 2 * lpd[0.0] + lpd[-d]) / d ** 2

        d1_mean = (L[(+d, 0.0)] - L[(-d, 0.0)]) / (2 * d)
        d2_mean = (L[(+d, 0.0)] - 2 * L00 + L[(-d, 0.0)]) / d ** 2
        d1_shape = (L[(0.0, +d)] - L[(0.0, -d)]) / (2 * d)
        d2_shape = (L[(0.0, +d)] - 2 * L00 + L[(0.0, -d)]) / d ** 2
        d1_flow = (L[(+d, +d)] - L[(-d, -d)]) / (2 * d)
        d2_flow = (L[(+d, +d)] - 2 * L00 + L[(-d, -d)]) / d ** 2
        d2_cross = (L[(+d, +d)] - L[(+d, -d)] - L[(-d, +d)] + L[(-d, -d)]) / (4 * d ** 2)
        del L

        # ---- (0) closure of the decomposition ----------------------------------------
        r1 = d1_flow - (d1_mean + d1_shape)
        r2 = d2_flow - (d2_mean + d2_shape + 2 * d2_cross)
        print("\n  (0) DECOMPOSITION CLOSURE  (residual should be O(d^2), NOT assumed)")
        print(f"      phi' : rms residual {np.sqrt((r1 ** 2).mean()):.3e}  vs rms|phi'| "
              f"{np.sqrt((d1_flow ** 2).mean()):.3e}   -> "
              f"{np.sqrt((r1 ** 2).mean()) / np.sqrt((d1_flow ** 2).mean()):.2e}")
        print(f"      phi'': rms residual {np.sqrt((r2 ** 2).mean()):.3e}  vs rms|phi''| "
              f"{np.sqrt((d2_flow ** 2).mean()):.3e}   -> "
              f"{np.sqrt((r2 ** 2).mean()) / np.sqrt((d2_flow ** 2).mean()):.2e}")
        # weighted versions (what actually matters)
        print(f"      E_w residual: phi' {np.abs(wmean(w, r1)).mean():.3e}   "
              f"phi'' {np.abs(wmean(w, r2)).mean():.3e}")

        # ---- (1) first-derivative magnitudes ------------------------------------------
        qs = (0.5, 0.9, 0.99)
        print("\n  (1) |d phi / d gamma| BY CHANNEL, posterior-weighted per galaxy")
        print(f"    {'quantile':>18} " + " ".join(f"{'p' + str(int(100 * q)):>18}"
                                                  for q in qs))
        qm = row_wq(np.abs(d1_mean), w, qs)
        qsh = row_wq(np.abs(d1_shape), w, qs)
        qfl = row_wq(np.abs(d1_flow), w, qs)
        qdt = row_wq(np.abs(np.broadcast_to(d1_det[None, :], w.shape)).copy(), w, qs)
        qline("MEAN channel", qm, rng, qs)
        qline("SHAPE channel", qsh, rng, qs)
        qline("DET channel", qdt, rng, qs)
        qline("FULL phi'", qfl, rng, qs)
        for j, q in enumerate(qs):
            r = qsh[:, j] / np.maximum(qm[:, j], 1e-30)
            m, se = boot_med(r, rng, args.nboot)
            print(f"      ratio SHAPE/MEAN at p{int(100 * q):<3}: {m:.3f} +- {se:.3f}")
        rms_m = np.sqrt(wmean(w, d1_mean ** 2))
        rms_s = np.sqrt(wmean(w, d1_shape ** 2))
        m_, se_ = boot_med(rms_s / np.maximum(rms_m, 1e-30), rng, args.nboot)
        print(f"      weighted rms ratio SHAPE/MEAN: median over galaxies "
              f"{m_:.3f} +- {se_:.3f}")
        # the closest analogue of the MATCHED case inside the bank: the best-matching node
        print("    at the TOP-WEIGHT nodes (mean over galaxies of the per-node mean):")
        for tag, kk in (("top-1", 1), ("top-10", 10)):
            kk = min(kk, w.shape[1])
            sel = order[:, :kk]
            cells = []
            for nm, arr in (("MEAN", np.abs(d1_mean)), ("SHAPE", np.abs(d1_shape)),
                            ("|MEAN2|", np.abs(d2_mean)), ("|SHAPE2|", np.abs(d2_shape)),
                            ("|CROSS2|", np.abs(d2_cross))):
                v = np.take_along_axis(arr, sel, 1).mean(axis=1)
                m2, se2 = boot(v, rng, args.nboot)
                cells.append(f"{nm}={m2:.3f}+-{se2:.3f}")
            print(f"      {tag:>7}: " + "  ".join(cells))

        # ---- (2) the I decomposition ---------------------------------------------------
        sm = wmean(w, d1_mean)
        ss = wmean(w, d1_shape)
        sd = wmean(w, np.broadcast_to(d1_det[None, :], w.shape))
        s_full = wmean(w, d1_flow) + sd
        vmm = wcov(w, d1_mean, d1_mean, sm, sm)
        vss = wcov(w, d1_shape, d1_shape, ss, ss)
        vdd = wcov(w, np.broadcast_to(d1_det[None, :], w.shape),
                   np.broadcast_to(d1_det[None, :], w.shape), sd, sd)
        cms = wcov(w, d1_mean, d1_shape, sm, ss)
        cmd = wcov(w, d1_mean, np.broadcast_to(d1_det[None, :], w.shape), sm, sd)
        csd = wcov(w, d1_shape, np.broadcast_to(d1_det[None, :], w.shape), ss, sd)
        emm = -wmean(w, d2_mean)
        ess_ = -wmean(w, d2_shape)
        exx = -2.0 * wmean(w, d2_cross)
        edd = -wmean(w, np.broadcast_to(d2_det[None, :], w.shape))
        var_tot = vmm + vss + vdd + 2 * (cms + cmd + csd)
        e_tot = emm + ess_ + exx + edd
        info = e_tot - var_tot

        print("\n  (2) THE TWO O(75) TERMS, SPLIT BY CHANNEL  (galaxy means +- bootstrap se)")
        print(f"    {'term':>28} {'value':>22}")
        for tag, arr in (("-E_w[phi''] TOTAL", e_tot),
                         ("   from MEAN  -E[phi''_mm]", emm),
                         ("   from SHAPE -E[phi''_ss]", ess_),
                         ("   from CROSS -2E[phi''_ms]", exx),
                         ("   from DET   -E[phi''_dd]", edd),
                         ("Var_w(phi') TOTAL", var_tot),
                         ("   Var(MEAN)", vmm),
                         ("   Var(SHAPE)", vss),
                         ("   Var(DET)", vdd),
                         ("   2Cov(MEAN,SHAPE)", 2 * cms),
                         ("   2Cov(MEAN,DET)", 2 * cmd),
                         ("   2Cov(SHAPE,DET)", 2 * csd),
                         ("I = -E[phi''] - Var(phi')", info)):
            m, se = boot(arr, rng, args.nboot)
            print(f"    {tag:>28} {m:>14.4f} +- {se:<8.4f}")
        print("\n    per-channel PARTIAL information (that channel alone):")
        for tag, e_, v_ in (("MEAN ", emm, vmm), ("SHAPE", ess_, vss), ("DET  ", edd, vdd)):
            me, see = boot(e_, rng, args.nboot)
            mv, sev = boot(v_, rng, args.nboot)
            mi, sei = boot(e_ - v_, rng, args.nboot)
            print(f"      {tag}: -E[phi'']={me:>10.4f}+-{see:<8.4f}  "
                  f"Var={mv:>10.4f}+-{sev:<8.4f}  I={mi:>10.4f}+-{sei:<8.4f}")

        # population terms, for the identity in the same units
        lp = {t: float(np.log(float(pdets[t].mean()))) for t in (0.0, +d, -d)}
        s_sel, i_sel = selection_terms(lp[0.0], (lp[+d] - lp[-d]) / (2 * d),
                                       (lp[+d] - 2 * lp[0.0] + lp[-d]) / d ** 2)
        bd, ld, ratio = denominator_consistency(s_full, info, s_sel, i_sel)
        print(f"\n    for reference: <s>_sel={s_sel:+.5f} I_sel={i_sel:+.5f}  "
              f"Var(s-<s>_sel)={bd:.3f} vs <I>-I_sel={ld:.3f}  ratio={ratio:+.3f}")
        print("    (round 1: <I> is a cancellation of two O(75) terms and is consistent "
              "with zero;\n     do NOT read the ratio at the digit level -- the channel "
              "SPLIT is what this run measures)")

        # ---- (3) s_i by channel -------------------------------------------------------
        print("\n  (3) s_i BY CHANNEL")
        print(f"    {'channel':>14} {'<s>':>20} {'sd over galaxies':>20} {'corr with full s':>18}")
        for tag, arr in (("MEAN", sm), ("SHAPE", ss), ("DET", sd), ("FULL", s_full)):
            m, se = boot(arr, rng, args.nboot)
            sdv, sdse = boot((arr - arr.mean()) ** 2, rng, args.nboot)
            cc = np.corrcoef(arr, s_full)[0, 1] if np.std(arr) > 0 else np.nan
            print(f"    {tag:>14} {m:>12.5f}+-{se:<7.5f} {np.sqrt(max(sdv, 0)):>20.5f} "
                  f"{cc:>18.4f}")
        print(f"    corr(s_MEAN, s_SHAPE) = {np.corrcoef(sm, ss)[0, 1]:+.4f}")
        print(f"    Var over galaxies: MEAN {np.var(sm):.5f}  SHAPE {np.var(ss):.5f}  "
              f"FULL {np.var(s_full):.5f}")

        # ---- (3b) CHANNEL-RESTRICTED COUNTERFACTUAL ESTIMATORS ------------------------
        # NOT deployable estimators -- the model's score really is the SUM of the
        # channels.  The narrow question they answer: if the UNSUPERVISED shape channel
        # is frozen out, does the estimator behave?
        print("\n  (3b) COUNTERFACTUAL ESTIMATORS WITH CHANNELS FROZEN OUT")
        print("       (diagnostic only; the real phi' is the sum.  <s>_sel/I_sel are kept "
              "in every\n        variant because the sample IS selected.)")
        det1 = np.broadcast_to(d1_det[None, :], w.shape)
        det2 = np.broadcast_to(d2_det[None, :], w.shape)
        variants = [
            ("FULL (exact)", d1_flow + det1, d2_flow + det2),
            ("MEAN+SHAPE+CROSS+DET", d1_mean + d1_shape + det1,
             d2_mean + d2_shape + 2 * d2_cross + det2),
            ("MEAN + DET", d1_mean + det1, d2_mean + det2),
            ("SHAPE + DET", d1_shape + det1, d2_shape + det2),
            ("MEAN only", d1_mean, d2_mean),
            ("SHAPE only", d1_shape, d2_shape),
        ]
        print(f"    {'variant':>22} {'<s>':>19} {'<I>':>19} {'Var(s-sel)':>12} "
              f"{'eq ratio':>10} {'ghat(5.9)':>21} {'ghat(5.8)':>12}")
        for tag, a1, a2 in variants:
            sv = wmean(w, a1)
            iv = -wmean(w, a2) - (wmean(w, a1 ** 2) - sv ** 2)
            bdv, ldv, rv = denominator_consistency(sv, iv, s_sel, i_sel)
            cs = sv - s_sel
            gb = float(cs.sum() / np.sum(cs ** 2))
            gl = float((sv.sum() - sv.size * s_sel) / (iv.sum() - iv.size * i_sel))
            idxb = rng.integers(0, sv.size, size=(args.nboot, sv.size))
            csb = cs[idxb]
            gbb = csb.sum(axis=1) / np.sum(csb ** 2, axis=1)
            ms, ses = boot(sv, rng, args.nboot)
            mi, sei = boot(iv, rng, args.nboot)
            print(f"    {tag:>22} {ms:>10.4f}+-{ses:<8.4f} {mi:>10.3f}+-{sei:<8.3f} "
                  f"{bdv:>12.3f} {rv:>10.3f} {gb:>12.6f}+-{np.std(gbb, ddof=1):<8.6f} "
                  f"{gl:>12.6f}")

        # ---- (4b) channel magnitude vs off-manifold distance --------------------------
        print("\n  (4b) CHANNEL MAGNITUDE vs LOG-WEIGHT DEFICIT (pooled over all (i,k))")
        edges = [-1e9, -100, -30, -10, -3, -1, 0.0]
        lab = ["<-100", "-100..-30", "-30..-10", "-10..-3", "-3..-1", "-1..0"]
        bi = np.digitize(deficit.ravel(), edges[1:-1])
        am = np.abs(d1_mean).ravel()
        ash = np.abs(d1_shape).ravel()
        a2m = np.abs(d2_mean).ravel()
        a2s = np.abs(d2_shape).ravel()
        zf = zn00.ravel().astype(np.float64)
        print(f"    {'deficit bin':>14} {'frac':>8} {'weight':>9} {'med||z||':>9} "
              f"{'med|MEAN1|':>11} {'med|SHAPE1|':>12} {'S/M':>7} "
              f"{'med|MEAN2|':>11} {'med|SHAPE2|':>12}")
        for b in range(len(lab)):
            m_ = bi == b
            if m_.sum() == 0:
                continue
            mm, ms = np.median(am[m_]), np.median(ash[m_])
            print(f"    {lab[b]:>14} {m_.mean():>8.2%} {fw[m_].sum():>9.4f} "
                  f"{np.median(zf[m_]):>9.2f} {mm:>11.4f} {ms:>12.4f} "
                  f"{ms / max(mm, 1e-30):>7.2f} {np.median(a2m[m_]):>11.3f} "
                  f"{np.median(a2s[m_]):>12.3f}")
        del am, ash, a2m, a2s

        # ---- (5) MATCHED-PAIR CONTROL -------------------------------------------------
        print("\n  (5) MATCHED-PAIR CONTROL: each galaxy under its OWN scene")
        gctx, gpdet = {}, {}
        for t in (0.0, +d, -d):
            c, p = scene_context(model, gal_df, gal_intr, (0.0, t), pre, nbr_std, dev)
            gctx[t] = c[keep]
            gpdet[t] = p[keep]
        with torch.no_grad():
            gmu = {t: model.mu(c) for t, c in gctx.items()}
            gfc = {t: model.mean_flow._flow_ctx(c) for t, c in gctx.items()}
        M00, mzn, mzmx = ll_diag(model, xhat, gmu[0.0], gfc[0.0], want_z=True)
        M = {}
        for a in (+d, -d):
            M[(a, 0.0)] = ll_diag(model, xhat, gmu[a], gfc[0.0])
            M[(0.0, a)] = ll_diag(model, xhat, gmu[0.0], gfc[a])
            M[(a, a)] = ll_diag(model, xhat, gmu[a], gfc[a])
        m1_mean = (M[(+d, 0.0)] - M[(-d, 0.0)]) / (2 * d)
        m2_mean = (M[(+d, 0.0)] - 2 * M00 + M[(-d, 0.0)]) / d ** 2
        m1_shape = (M[(0.0, +d)] - M[(0.0, -d)]) / (2 * d)
        m2_shape = (M[(0.0, +d)] - 2 * M00 + M[(0.0, -d)]) / d ** 2
        m1_flow = (M[(+d, +d)] - M[(-d, -d)]) / (2 * d)
        mz, mzse = boot(mzn.astype(np.float64), rng, args.nboot)
        print(f"      ||z|| on matched pairs: {mz:.3f} +- {mzse:.3f}   "
              f"(bank top-1-weight node gave the number above)")
        print(f"      max|z_d| matched: {np.mean(mzmx):.3f}   "
              f"||xhat-mu|| matched: "
              f"{np.mean(np.sqrt(((xh - gmu[0.0].double().cpu().numpy()) ** 2).sum(-1))):.3f}")
        print(f"      {'channel':>14} {'med|d1|':>18} {'p90|d1|':>18} {'p99|d1|':>18}")
        for tag, arr in (("MEAN", np.abs(m1_mean)), ("SHAPE", np.abs(m1_shape)),
                         ("FULL", np.abs(m1_flow))):
            cells = []
            for q in (50, 90, 99):
                v = np.percentile(arr, q)
                idx = rng.integers(0, arr.shape[0], size=(args.nboot, arr.shape[0]))
                se = float(np.std(np.percentile(arr[idx], q, axis=1), ddof=1))
                cells.append(f"{v:>11.4f}+-{se:<6.4f}")
            print(f"      {tag:>14} " + " ".join(cells))
        rr = np.abs(m1_shape) / np.maximum(np.abs(m1_mean), 1e-30)
        m_, se_ = boot_med(rr, rng, args.nboot)
        print(f"      matched ratio |SHAPE|/|MEAN|: median {m_:.4f} +- {se_:.4f}")
        for tag, arr in (("MEAN", m2_mean), ("SHAPE", m2_shape)):
            m, se = boot(arr, rng, args.nboot)
            print(f"      matched <phi''_{tag}> = {m:+.4f} +- {se:.4f}   "
                  f"med|.| {np.median(np.abs(arr)):.4f}")

    print("\nDONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
