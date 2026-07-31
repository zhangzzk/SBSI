"""phi-ANATOMY for the INFERENCE.md 5C Lagrangian score on the Gold-V2 joint model.

The information equality Var(s - <s>_sel) = <I> - I_sel fails by a factor 7-30 on the real
model while being exact in the 1-D toy.  Winsorising phi' did not restore it.  The stated
(but until now UNMEASURED) reason was "the extreme phi' sit on negligible-weight nodes".
This script measures that, and separates it from the competing explanation that the
derivatives are round-off dominated.

Four blocks, all at g_true = 0 where the Bartlett identity is exact:

  (1) WHERE THE VARIANCE LIVES.  Nodes sorted by posterior weight; cumulative share of
      |w phi'| and of the Var_w(phi') terms as a function of the cumulative WEIGHT
      captured (not the node count).  Plus WEIGHTED quantiles of |phi'| and the total
      weight carried by the largest-|phi'| nodes.  A weight-truncated re-estimate of the
      identity is the direct test of "drop the tail".

  (2) IS phi' NUMERICAL.  phi', phi'' recomputed on the SAME nodes at four stencil widths;
      scatter across widths reported for the HIGH-WEIGHT nodes only.  phi is produced in
      float32, so there is a round-off floor ~ sqrt(6) sigma_phi / delta^2 on phi''; if
      |phi''| tracks that floor as delta shrinks the second derivative is noise.

  (3) PRECISION CONTROL.  phi recomputed in float64 on CPU for a subset of galaxies and
      compared elementwise with the float32 GPU values.  That difference IS sigma_phi --
      it is not predicted, it is measured -- and it propagates into the (2) floor.  s_i,
      I_i, Var_w(phi'), E_w[phi''] compared between the two precisions.

  (4) WHAT MAKES phi' BIG.  |phi'| against the log-weight deficit phi_k(0) - max_k phi_k(0)
      and against ||xhat_i - mu(ctx_k)|| in standardized target units.

Nothing here is imported by other modules; the shared machinery comes from
`closure_v2_lagrangian` and `sbs_shear.lagrangian_score` unchanged.
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
    denominator_consistency, posterior_weights, score_and_information, selection_terms,
)
from train_joint_forward import (  # noqa: E402
    NEIGHBOR_FEATURES, intrinsic_shape, neighbor_padded, shifted_feature_frame,
)

EPS32 = float(np.finfo(np.float32).eps)          # 1.19e-7


# ---------------------------------------------------------------------------------------
# float64 / quantization-controlled re-implementation of the context path
# ---------------------------------------------------------------------------------------

def _primary_std64(pre, frame, quantize):
    """`pre.transform_frame` in float64.  `quantize=True` reproduces the float32 rounding
    of the production path exactly, so the ONLY remaining difference is the arithmetic."""
    if quantize:
        return pre.transform_frame(frame).astype(np.float64)
    raw = frame[pre.feature_names].to_numpy(dtype=np.float64, copy=True)
    finite = np.isfinite(raw)
    filled = np.where(finite, raw, np.asarray(pre.fill_values, dtype=np.float64))
    scaled = (filled - np.asarray(pre.means, np.float64)) / np.asarray(pre.scales, np.float64)
    if pre.add_missing_indicators:
        scaled = np.concatenate([scaled, (~finite).astype(np.float64)], axis=1)
    return scaled


def _neighbor_std64(nbr_std, frame, nbg, quantize):
    if quantize:
        npad, mask = neighbor_padded(frame, nbg, nbr_std)
        return npad.astype(np.float64), mask.astype(np.float64)
    raw = frame[NEIGHBOR_FEATURES].to_numpy(dtype=np.float64)
    finite = np.isfinite(raw)
    filled = np.where(finite, raw, np.asarray(nbr_std.fill_values, np.float64))
    std = (filled - np.asarray(nbr_std.means, np.float64)) / np.asarray(nbr_std.scales, np.float64)
    std[~nbg] = 0.0
    return std[:, None, :], nbg.astype(np.float64)[:, None]


@torch.no_grad()
def scene_context64(model, frame, intr, gamma_vec, pre, nbr_std, device, quantize):
    """`scene_context` with float64 tensors.  `model` must already be `.double()`."""
    mag = float(np.hypot(*gamma_vec))
    gdir = (gamma_vec[0] / mag, gamma_vec[1] / mag) if mag > 0 else (1.0, 0.0)
    f, nbg = shifted_feature_frame(frame, intr, gdir, mag, primary_only=True)
    p = torch.as_tensor(_primary_std64(pre, f, quantize), dtype=torch.float64, device=device)
    npad, mask = _neighbor_std64(nbr_std, f, nbg, quantize)
    ctx = model.context(p,
                        torch.as_tensor(npad, dtype=torch.float64, device=device),
                        torch.as_tensor(mask, dtype=torch.float64, device=device))
    return ctx, model.detection_prob(ctx)


@torch.no_grad()
def phi_block64(model, xhat, ctx, log_pdet, chunk=10):
    n, k = xhat.shape[0], ctx.shape[0]
    out = np.empty((n, k), dtype=np.float64)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        b = e - s
        tgt = xhat[s:e, None, :].expand(b, k, xhat.shape[1]).reshape(b * k, -1)
        cc = ctx[None, :, :].expand(b, k, ctx.shape[1]).reshape(b * k, -1)
        ll = model.log_prob_obs(tgt, cc).view(b, k)
        out[s:e] = (ll + log_pdet[None, :]).cpu().numpy()
    return out


# ---------------------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------------------

def weighted_quantiles(vals, w, qs):
    """Quantiles of `vals` under the measure `w` (both flat, w need not be normalised)."""
    o = np.argsort(vals)
    v, ww = vals[o], w[o]
    c = np.cumsum(ww)
    c = c / c[-1]
    return np.interp(qs, c, v)


def eq_ratio(p0, d1, d2, s_sel, i_sel, w=None):
    if w is None:
        s, info = score_and_information(p0, d1, d2)
    else:
        s = np.sum(w * d1, axis=1)
        info = -np.sum(w * d2, axis=1) - (np.sum(w * d1 ** 2, axis=1) - s ** 2)
    return denominator_consistency(s, info, s_sel, i_sel)


# ---------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=CKPT)
    ap.add_argument("--catalogue", default=CAT)
    ap.add_argument("--n-gal", type=int, default=2000)
    ap.add_argument("--n-node", type=int, default=2000)
    ap.add_argument("--gamma", type=float, default=0.0)
    ap.add_argument("--gal-offset", type=int, default=2000)
    ap.add_argument("--deltas", default="0.02,0.01,0.005,0.0025")
    ap.add_argument("--ref-delta", type=float, default=0.01)
    ap.add_argument("--chunk", type=int, default=250)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--n-f64", type=int, default=50, help="galaxies for the CPU float64 control")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(args.device)
    torch.manual_seed(args.seed)
    print(f"device={args.device}  torch={torch.__version__}  float32 eps={EPS32:.3e}")
    print(f"  matmul allow_tf32={torch.backends.cuda.matmul.allow_tf32}  "
          f"cudnn allow_tf32={torch.backends.cudnn.allow_tf32}  "
          f"(TF32 would cost ~3 decimal digits and is the first thing to check)")

    model, pre, nbr_std, target_std, meta = rebuild(args.checkpoint, dev)
    print(f"checkpoint: {os.path.basename(args.checkpoint)}")

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

    # ---- data drawn FROM the model at the true shear -----------------------------------
    g_true = (0.0, float(args.gamma))
    ctx_g, pdet_g = scene_context(model, gal_df, gal_intr, g_true, pre, nbr_std, dev)
    with torch.no_grad():
        xhat = model.mean_flow.sample(ctx_g, n_samples=1)
        if xhat.dim() == 3:
            xhat = xhat[:, 0, :]
    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    keep = (torch.rand(len(gal_df), generator=gen).to(dev) < pdet_g)
    xhat = xhat[keep]
    n_keep = int(xhat.shape[0])
    print(f"detection: kept {n_keep:,}/{len(gal_df):,}  (<Pdet>={float(pdet_g.mean()):.4f})")

    deltas = [float(x) for x in args.deltas.split(",")]
    dref = args.ref_delta
    if dref not in deltas:
        deltas.append(dref)

    # ---- phi on the node bank, all stencil widths, common nodes ------------------------
    t0 = time.time()
    ctx0, pdet0 = scene_context(model, node_df, node_intr, (0.0, 0.0), pre, nbr_std, dev)
    p0 = phi_block(model, xhat, ctx0, torch.log(pdet0), args.chunk)
    pdet_mean = {0.0: float(pdet0.mean())}
    D1, D2 = {}, {}
    for d in deltas:
        blk = {}
        for t in (+d, -d):
            c, pd_ = scene_context(model, node_df, node_intr, (0.0, t), pre, nbr_std, dev)
            blk[t] = phi_block(model, xhat, c, torch.log(pd_), args.chunk)
            pdet_mean[t] = float(pd_.mean())
        D1[d] = (blk[+d] - blk[-d]) / (2 * d)
        D2[d] = (blk[+d] - 2 * p0 + blk[-d]) / d ** 2
    print(f"phi blocks done in {time.time() - t0:.0f}s   |phi(0)|: "
          f"median {np.median(np.abs(p0)):.2f}  p99 {np.percentile(np.abs(p0), 99):.1f}  "
          f"max {np.abs(p0).max():.1f}")

    lp = {t: np.log(v) for t, v in pdet_mean.items()}
    s_sel, i_sel = selection_terms(lp[0.0], (lp[+dref] - lp[-dref]) / (2 * dref),
                                   (lp[+dref] - 2 * lp[0.0] + lp[-dref]) / dref ** 2)

    w = posterior_weights(p0)
    d1, d2 = D1[dref], D2[dref]
    s = np.sum(w * d1, axis=1)
    ew_dd = np.sum(w * d2, axis=1)
    var_w = np.sum(w * d1 ** 2, axis=1) - s ** 2
    info = -ew_dd - var_w
    bd, ld, ratio = denominator_consistency(s, info, s_sel, i_sel)
    ess = 1.0 / np.sum(w ** 2, axis=1)

    print("\n" + "=" * 88)
    print(f"BASELINE  (delta={dref}, K={args.n_node}, g_true={args.gamma})")
    print("=" * 88)
    print(f"  <s>={np.mean(s):+.4f}  <s>_sel={s_sel:+.5f}  I_sel={i_sel:+.5f}")
    print(f"  INFORMATION EQUALITY: Var(s-<s>_sel)={bd:.3f}  vs  <I>-I_sel={ld:.3f}"
          f"   ratio={ratio:+.3f}   (must be 1)")
    print(f"  I decomposition: <-E_w[phi'']>={-np.mean(ew_dd):+.4f}   "
          f"<Var_w(phi')>={np.mean(var_w):+.4f}   -> <I>={np.mean(info):+.4f}")
    print(f"    per-galaxy medians: -E_w[phi'']={-np.median(ew_dd):+.3f}  "
          f"Var_w(phi')={np.median(var_w):.3f}   "
          f"p99 Var_w={np.percentile(var_w, 99):.1f}  max Var_w={var_w.max():.1f}")
    print(f"  ESS: mean {ess.mean():.1f}  median {np.median(ess):.1f}  min {ess.min():.1f}")

    # =====================================================================================
    # (1) WHERE THE VARIANCE LIVES
    # =====================================================================================
    print("\n" + "=" * 88)
    print("(1) WHERE THE VARIANCE LIVES  -- nodes ordered by posterior weight")
    print("=" * 88)
    order = np.argsort(-w, axis=1)
    ws = np.take_along_axis(w, order, 1)
    cw = np.cumsum(ws, axis=1)
    a1 = np.take_along_axis(np.abs(w * d1), order, 1)
    c1 = np.cumsum(a1, 1); c1 = c1 / c1[:, -1:]
    vterm = w * (d1 - s[:, None]) ** 2
    av = np.take_along_axis(vterm, order, 1)
    cv = np.cumsum(av, 1); cv = cv / cv[:, -1:]
    add = np.take_along_axis(np.abs(w * d2), order, 1)
    c2 = np.cumsum(add, 1); c2 = c2 / c2[:, -1:]
    idx_all = np.arange(w.shape[0])
    print(f"  {'cum weight':>10} {'#nodes':>8} {'share|w phi1|':>14} {'share Var_w':>12} "
          f"{'share|w phi2|':>14}")
    qmask = {}
    for q in (0.5, 0.9, 0.99, 0.999):
        j = np.argmax(cw >= q, axis=1)
        qmask[q] = j
        print(f"  {q:>10.3f} {np.mean(j + 1):>8.1f} {np.mean(c1[idx_all, j]):>14.3%} "
              f"{np.mean(cv[idx_all, j]):>12.3%} {np.mean(c2[idx_all, j]):>14.3%}")
    print("  (#nodes = mean number of nodes needed to reach that cumulative weight, of "
          f"{args.n_node})")

    qs = [0.5, 0.9, 0.99, 0.999, 1.0]
    flat_w = (w / w.shape[0]).ravel()
    aflat = np.abs(d1).ravel()
    wq = weighted_quantiles(aflat, flat_w, qs)
    uq = np.percentile(aflat, [100 * q for q in qs])
    print("\n  |phi'| quantiles     " + "".join(f"{100*q:>10.1f}%" for q in qs))
    print("    under w (posterior)" + "".join(f"{v:>11.2f}" for v in wq))
    print("    under uniform nodes" + "".join(f"{v:>11.2f}" for v in uq))
    aflat2 = np.abs(d2).ravel()
    print("  |phi''| quantiles    " + "".join(f"{100*q:>10.1f}%" for q in qs))
    print("    under w (posterior)"
          + "".join(f"{v:>11.2f}" for v in weighted_quantiles(aflat2, flat_w, qs)))
    print("    under uniform nodes"
          + "".join(f"{v:>11.2f}" for v in np.percentile(aflat2, [100 * q for q in qs])))

    # how much POSTERIOR WEIGHT sits on the biggest-|phi'| nodes
    o1 = np.argsort(-np.abs(d1), axis=1)
    wsorted_by_d1 = np.take_along_axis(w, o1, 1)
    print("\n  posterior weight carried by the largest-|phi'| nodes:")
    for k in (1, 2, 20, 200):
        print(f"    top {k:>4} nodes by |phi'| ({k / args.n_node:>6.1%} of the bank): "
              f"mean weight {wsorted_by_d1[:, :k].sum(1).mean():.4f}   "
              f"median {np.median(wsorted_by_d1[:, :k].sum(1)):.4f}")

    # weight-truncated re-estimate: the direct 'drop the tail' test
    print("\n  weight-TRUNCATED re-estimate (keep only the top-q-weight nodes, renormalise):")
    for q in (0.999, 0.99, 0.9):
        j = qmask[q]
        keepmask = np.zeros_like(w, dtype=bool)
        rank = np.empty_like(order)
        np.put_along_axis(rank, order, np.arange(w.shape[1])[None, :].repeat(w.shape[0], 0), 1)
        keepmask = rank <= j[:, None]
        wt = np.where(keepmask, w, 0.0)
        wt = wt / wt.sum(1, keepdims=True)
        b2, l2, r2 = eq_ratio(p0, d1, d2, s_sel, i_sel, w=wt)
        print(f"    q={q:<6}: Var(s)={b2:9.3f}  <I>-I_sel={l2:8.3f}  ratio={r2:+8.3f}")

    # =====================================================================================
    # (2) IS phi' NUMERICAL
    # =====================================================================================
    print("\n" + "=" * 88)
    print("(2) STENCIL-WIDTH SCATTER  -- HIGH-WEIGHT nodes only (top 90% of weight)")
    print("=" * 88)
    jh = qmask[0.9]
    rank = np.empty_like(order)
    np.put_along_axis(rank, order, np.arange(w.shape[1])[None, :].repeat(w.shape[0], 0), 1)
    hi = rank <= jh[:, None]
    print(f"  high-weight node set: {hi.sum() / hi.shape[0]:.1f} nodes per galaxy "
          f"({hi.mean():.2%} of the bank)")
    dl = sorted(deltas, reverse=True)
    print(f"\n  {'delta':>8} {'med|phi1|':>11} {'med|phi2|':>11} {'p99|phi2|':>11} "
          f"{'<Var_w(phi1)>':>14} {'<-E_w[phi2]>':>13} {'eq ratio':>9}")
    for d in dl:
        h1, h2 = D1[d][hi], D2[d][hi]
        sd = np.sum(w * D1[d], axis=1)
        vd = np.sum(w * D1[d] ** 2, axis=1) - sd ** 2
        ed = np.sum(w * D2[d], axis=1)
        _, _, rr = eq_ratio(p0, D1[d], D2[d], s_sel, i_sel)
        print(f"  {d:>8.4f} {np.median(np.abs(h1)):>11.3f} {np.median(np.abs(h2)):>11.3f} "
              f"{np.percentile(np.abs(h2), 99):>11.2f} {np.mean(vd):>14.4f} "
              f"{-np.mean(ed):>13.4f} {rr:>+9.3f}")

    stack1 = np.stack([D1[d][hi] for d in dl])
    stack2 = np.stack([D2[d][hi] for d in dl])
    rel1 = np.std(stack1, axis=0) / np.maximum(np.abs(np.mean(stack1, axis=0)), 1e-12)
    rel2 = np.std(stack2, axis=0) / np.maximum(np.abs(np.mean(stack2, axis=0)), 1e-12)
    print(f"\n  relative scatter ACROSS the 4 widths (high-weight nodes):")
    print(f"    phi' : median {np.median(rel1):.3e}  p90 {np.percentile(rel1, 90):.3e}")
    print(f"    phi'': median {np.median(rel2):.3e}  p90 {np.percentile(rel2, 90):.3e}")
    fine, coarse = min(dl), max(dl)
    r1p = np.abs(D1[fine][hi] - D1[coarse][hi]) / np.maximum(np.abs(D1[coarse][hi]), 1e-12)
    r2p = np.abs(D2[fine][hi] - D2[coarse][hi]) / np.maximum(np.abs(D2[coarse][hi]), 1e-12)
    print(f"    |d({fine})-d({coarse})|/|d({coarse})|:  phi' median {np.median(r1p):.3e}   "
          f"phi'' median {np.median(r2p):.3e}")

    med_absphi = float(np.median(np.abs(p0)))
    print(f"\n  PREDICTED round-off floor from float32 phi (|phi| median {med_absphi:.2f}):")
    print(f"  {'delta':>8} {'floor phi1':>12} {'floor phi2':>12}   "
          f"(sigma_phi = eps32*|phi| = {EPS32 * med_absphi:.2e})")
    sig_pred = EPS32 * med_absphi
    for d in dl:
        print(f"  {d:>8.4f} {np.sqrt(2) * sig_pred / (2 * d):>12.3e} "
              f"{np.sqrt(6) * sig_pred / d ** 2:>12.3e}")

    # =====================================================================================
    # (3) PRECISION CONTROL: float64 on CPU
    # =====================================================================================
    print("\n" + "=" * 88)
    print(f"(3) PRECISION CONTROL -- float64 CPU on {args.n_f64} galaxies x {args.n_node} nodes")
    print("=" * 88)
    t0 = time.time()
    cpu = torch.device("cpu")
    m64, pre64, nbr64, _, _ = rebuild(args.checkpoint, cpu)
    m64 = m64.double()
    sub = slice(0, args.n_f64)
    xh64 = xhat[sub].double().cpu()
    for quant, tag in ((True, "f64 arith, f32-quantized inputs"),
                       (False, "f64 arith, f64 inputs")):
        blk64 = {}
        for t in (0.0, +dref, -dref):
            c, pd_ = scene_context64(m64, node_df, node_intr, (0.0, t), pre64, nbr64, cpu, quant)
            blk64[t] = phi_block64(m64, xh64, c, torch.log(pd_))
        q0 = blk64[0.0]
        q1 = (blk64[+dref] - blk64[-dref]) / (2 * dref)
        q2 = (blk64[+dref] - 2 * q0 + blk64[-dref]) / dref ** 2
        g0, g1, g2 = p0[sub], d1[sub], d2[sub]
        dphi = q0 - g0
        print(f"\n  [{tag}]   ({time.time() - t0:.0f}s)")
        print(f"    MEASURED float32 error in phi: rms {np.sqrt(np.mean(dphi ** 2)):.3e}  "
              f"p99 {np.percentile(np.abs(dphi), 99):.3e}  max {np.abs(dphi).max():.3e}")
        sig = float(np.sqrt(np.mean(dphi ** 2)))
        print(f"    -> implied floors at delta={dref}: phi' {np.sqrt(2)*sig/(2*dref):.3e}   "
              f"phi'' {np.sqrt(6)*sig/dref**2:.3e}")
        print(f"    phi'  : rms diff {np.sqrt(np.mean((q1 - g1) ** 2)):.3e}  "
              f"vs rms|phi'| {np.sqrt(np.mean(g1 ** 2)):.3e}")
        print(f"    phi'' : rms diff {np.sqrt(np.mean((q2 - g2) ** 2)):.3e}  "
              f"vs rms|phi''| {np.sqrt(np.mean(g2 ** 2)):.3e}")
        w64 = posterior_weights(q0)
        s64 = np.sum(w64 * q1, axis=1)
        v64 = np.sum(w64 * q1 ** 2, axis=1) - s64 ** 2
        e64 = np.sum(w64 * q2, axis=1)
        wg = posterior_weights(g0)
        sg = np.sum(wg * g1, axis=1)
        vg = np.sum(wg * g1 ** 2, axis=1) - sg ** 2
        eg = np.sum(wg * g2, axis=1)
        print(f"    s_i        : f32 <{np.mean(sg):+.5f}>  f64 <{np.mean(s64):+.5f}>  "
              f"rms diff {np.sqrt(np.mean((s64 - sg) ** 2)):.3e}  "
              f"corr {np.corrcoef(s64, sg)[0, 1]:.8f}")
        print(f"    Var_w(phi'): f32 <{np.mean(vg):+.5f}>  f64 <{np.mean(v64):+.5f}>  "
              f"rms diff {np.sqrt(np.mean((v64 - vg) ** 2)):.3e}")
        print(f"    E_w[phi'']  : f32 <{np.mean(eg):+.5f}>  f64 <{np.mean(e64):+.5f}>  "
              f"rms diff {np.sqrt(np.mean((e64 - eg) ** 2)):.3e}")
        i32, i64 = -eg - vg, -e64 - v64
        print(f"    I_i        : f32 <{np.mean(i32):+.5f}>  f64 <{np.mean(i64):+.5f}>")
        print(f"    identity on this subset: f32 ratio "
              f"{denominator_consistency(sg, i32, s_sel, i_sel)[2]:+.3f}   f64 ratio "
              f"{denominator_consistency(s64, i64, s_sel, i_sel)[2]:+.3f}")

    # =====================================================================================
    # (4) WHAT MAKES phi' BIG
    # =====================================================================================
    print("\n" + "=" * 88)
    print("(4) WHAT MAKES phi' BIG")
    print("=" * 88)
    with torch.no_grad():
        mu0 = model.mu(ctx0).double().cpu().numpy()             # (K, target_dim)
    xh = xhat.double().cpu().numpy()                            # (N, target_dim)
    dist = np.sqrt(((xh[:, None, :] - mu0[None, :, :]) ** 2).sum(-1))   # std target units
    deficit = p0 - p0.max(axis=1, keepdims=True)                # <= 0
    la = np.log10(np.maximum(np.abs(d1), 1e-12))
    print(f"  pooled corr(log10|phi'|, log-weight deficit) = "
          f"{np.corrcoef(la.ravel(), deficit.ravel())[0, 1]:+.4f}")
    print(f"  pooled corr(log10|phi'|, ||xhat-mu||)        = "
          f"{np.corrcoef(la.ravel(), dist.ravel())[0, 1]:+.4f}")
    pc_def = np.array([np.corrcoef(la[i], deficit[i])[0, 1] for i in range(la.shape[0])])
    pc_dis = np.array([np.corrcoef(la[i], dist[i])[0, 1] for i in range(la.shape[0])])
    print(f"  per-galaxy corr(log10|phi'|, deficit): median {np.median(pc_def):+.4f}  "
          f"IQR [{np.percentile(pc_def, 25):+.3f}, {np.percentile(pc_def, 75):+.3f}]")
    print(f"  per-galaxy corr(log10|phi'|, dist)   : median {np.median(pc_dis):+.4f}  "
          f"IQR [{np.percentile(pc_dis, 25):+.3f}, {np.percentile(pc_dis, 75):+.3f}]")

    print("\n  binned by log-weight deficit (pooled over all (i,k)):")
    edges = [-1e9, -100, -30, -10, -3, -1, 0.0]
    bi = np.digitize(deficit.ravel(), edges[1:-1])
    print(f"    {'deficit bin':>16} {'frac nodes':>11} {'tot weight':>11} "
          f"{'med|phi1|':>11} {'p99|phi1|':>11} {'med|phi2|':>11}")
    fw = (w / w.shape[0]).ravel()
    ad = np.abs(d1).ravel(); ad2 = np.abs(d2).ravel()
    lab = ["<-100", "-100..-30", "-30..-10", "-10..-3", "-3..-1", "-1..0"]
    for b in range(len(lab)):
        m_ = bi == b
        if m_.sum() == 0:
            continue
        print(f"    {lab[b]:>16} {m_.mean():>11.3%} {fw[m_].sum():>11.4f} "
              f"{np.median(ad[m_]):>11.3f} {np.percentile(ad[m_], 99):>11.2f} "
              f"{np.median(ad2[m_]):>11.3f}")

    print("\n  binned by ||xhat - mu(ctx)|| (standardized target units):")
    edges2 = [0, 1, 2, 4, 8, 16, 1e9]
    bi2 = np.digitize(dist.ravel(), edges2[1:-1])
    lab2 = ["<1", "1-2", "2-4", "4-8", "8-16", ">16"]
    for b in range(len(lab2)):
        m_ = bi2 == b
        if m_.sum() == 0:
            continue
        print(f"    {lab2[b]:>16} {m_.mean():>11.3%} {fw[m_].sum():>11.4f} "
              f"{np.median(ad[m_]):>11.3f} {np.percentile(ad[m_], 99):>11.2f} "
              f"{np.median(ad2[m_]):>11.3f}")

    # the reverse view: what do the WORST-|phi'| nodes look like
    top = o1[:, :20]
    print(f"\n  top-20 |phi'| nodes per galaxy: mean deficit "
          f"{np.take_along_axis(deficit, top, 1).mean():+.2f}   mean dist "
          f"{np.take_along_axis(dist, top, 1).mean():.2f}   mean weight "
          f"{np.take_along_axis(w, top, 1).sum(1).mean():.3e}")
    hiw = order[:, :20]
    print(f"  top-20 WEIGHT nodes per galaxy: mean deficit "
          f"{np.take_along_axis(deficit, hiw, 1).mean():+.2f}   mean dist "
          f"{np.take_along_axis(dist, hiw, 1).mean():.2f}   mean |phi'| "
          f"{np.abs(np.take_along_axis(d1, hiw, 1)).mean():.2f}   mean |phi''| "
          f"{np.abs(np.take_along_axis(d2, hiw, 1)).mean():.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
