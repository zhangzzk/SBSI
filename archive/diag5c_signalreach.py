"""SIGNAL REACH: how much shear signal actually enters the V2 joint forward model.

Four measurements, all on the real checkpoint and real scenes, under primary-only shear:
  1. which INPUT features move with gamma (standardised),
  2. how far the CONTEXT moves (flow-visible dims vs mu-only dims),
  3. how far the MEAN HEAD moves -- including a reproduction of the certified <R_flow>,
  4. the shear-induced change in phi per node against the node-to-node spread of phi.

Nothing here uses the 5C estimator; (3) is an independent check of the model against a
number certified by a completely different route (Gold-v1 <R_flow> = 0.2930).
"""

import argparse
import os
import sys

import numpy as np
import torch

ROOT = "/home/z/Zekang.Zhang/SBSI/.claude/worktrees/inference-5b"
for _p in (ROOT, os.path.join(ROOT, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from closure_v2_lagrangian import rebuild, load_rows, phi_block  # noqa: E402
from train_joint_forward import (  # noqa: E402
    intrinsic_shape, neighbor_padded, shifted_feature_frame,
)

CAT = ("/project/ls-gruen/users/zekang.zhang/sbsi_catalogues/"
       "det_meas_ngmix_ap7_g0.0_train.feather")
CKPT = ("/project/ls-gruen/users/zekang.zhang/sbsi_caches/forward_proto/"
        "forward_ens_lr250_swa8_seed421_joint.pt")


def boot(x, f=np.mean, n=400, seed=0):
    """Bootstrap standard error of statistic f over the galaxy axis."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    N = x.shape[0]
    vals = [f(x[rng.integers(0, N, N)]) for _ in range(n)]
    return float(f(x)), float(np.std(vals))


@torch.no_grad()
def build(model, df, intr, gdir, delta, pre, nbr_std, dev, primary_only=True):
    f, nbg = shifted_feature_frame(df, intr, gdir, delta, primary_only=primary_only)
    p = pre.transform_frame(f)
    npad, mask = neighbor_padded(f, nbg, nbr_std)
    pt = torch.as_tensor(p, dtype=torch.float32, device=dev)
    nt = torch.as_tensor(npad, dtype=torch.float32, device=dev)
    mt = torch.as_tensor(mask, dtype=torch.float32, device=dev)
    ctx = model.context(pt, nt, mt)
    return dict(p=np.asarray(p, dtype=np.float64),
                n=np.asarray(npad, dtype=np.float64)[:, 0, :],
                nbg=nbg, ctx=ctx,
                mu=model.mu(ctx), pdet=model.detection_prob(ctx))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=CKPT)
    ap.add_argument("--catalogue", default=CAT)
    ap.add_argument("--n-gal", type=int, default=20000)
    ap.add_argument("--gammas", default="0.01,0.05")
    ap.add_argument("--n-node", type=int, default=2000)
    ap.add_argument("--n-gal-phi", type=int, default=400)
    ap.add_argument("--n-scale-samples", type=int, default=64)
    ap.add_argument("--rsim-npz", default=None,
                    help="R_sim (flux x size x blend) target grid the model was supervised "
                         "against; used as the truth side re-weighted to this population")
    ap.add_argument("--apply-true-cut", type=int, default=1,
                    help="0 = keep the FULL population, so the true-cut mean and the "
                         "full-population mean of R can be read off the SAME array. The "
                         "certified <R_flow>=0.2930 is a full-population number; the 5C "
                         "estimator runs on the true-cut subset.")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    dev = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"device={dev}  torch={torch.__version__}", flush=True)

    model, pre, nbr_std, target_std, meta = rebuild(args.checkpoint, dev)
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    delta_tr = float(ck["delta"])
    pfeat = list(ck["primary_preprocessor"]["feature_names"])
    nfeat = list(ck["neighbor_features"])
    tnames = list(ck["target_transform"]["target_names"])
    tscale = np.asarray(ck["target_transform"]["scales"], dtype=np.float64)
    keep = model.mean_flow.keep_indices.cpu().numpy()
    print(f"metadata: {meta}")
    print(f"training delta = {delta_tr};  target scales = {tscale.round(4).tolist()}")
    print(f"primary_dim={ck['model_config']['primary_dim']} "
          f"(= {len(pfeat)} features + {ck['model_config']['primary_dim'] - len(pfeat)} "
          f"missing indicators);  neighbor_dim={len(nfeat)}")
    print(f"flow keep_indices: {len(keep)} of {model.context_dim} context dims "
          f"({'ALL dims -> the residual flow sees the whole context' if len(keep) == model.context_dim else 'SOME DIMS DROPPED'})")

    rows = load_rows(args.catalogue, args.n_gal * 6)
    re_min, mag_max = [float(v) for v in meta["true_cut"]]
    kt = ((rows["Re_input_p"].to_numpy(float) > re_min)
          & (rows["r_input_p"].to_numpy(float) < mag_max))
    print(f"true_cut Re>{re_min}, mag<{mag_max} keeps {int(kt.sum()):,}/{len(rows):,} "
          f"= {kt.mean():.1%} of rows read")
    if args.apply_true_cut:
        rows = rows[kt].reset_index(drop=True)
    else:
        rows = rows.reset_index(drop=True)
        print("  --apply-true-cut 0: keeping the FULL population "
              "(note: OUT OF DOMAIN for the density; the mean head is still evaluated)")
    df = rows.iloc[:args.n_gal].reset_index(drop=True)
    N = len(df)
    intr = {}
    intr["e1p"], intr["e2p"] = intrinsic_shape(df, "p")
    intr["e1s"], intr["e2s"] = intrinsic_shape(df, "s")
    print(f"population: {N:,} scenes; neighboured fraction "
          f"{df['neighbored'].astype(bool).mean():.1%}")

    gammas = [float(x) for x in args.gammas.split(",")]
    base = build(model, df, intr, (0.0, 1.0), 0.0, pre, nbr_std, dev)

    # ---------------------------------------------------------------- 3. MEAN HEAD ----
    print("\n" + "=" * 78)
    print("3. MEAN-HEAD SHEAR RESPONSE  (the certified cross-check)")
    print("=" * 78)
    print("R = 0.5*[ (mu_e1(+d)-mu_e1(-d))*s1 + (mu_e2(+d)-mu_e2(-d))*s2 ] / (2d)")
    print("   exactly scripts/harvest_joint_rflow.rflow_chunk; certified Gold-v1 <R_flow>=0.2930")
    for po in (True, False):
        for d in sorted(set([delta_tr] + gammas)):
            mus = {}
            for nm, gd, sg in [("e1p", (1., 0.), +1.), ("e1m", (1., 0.), -1.),
                               ("e2p", (0., 1.), +1.), ("e2m", (0., 1.), -1.)]:
                mus[nm] = build(model, df, intr, gd, sg * d, pre, nbr_std, dev,
                                primary_only=po)["mu"].cpu().numpy().astype(np.float64)
            r1 = (mus["e1p"][:, 0] - mus["e1m"][:, 0]) * tscale[0] / (2 * d)
            r2 = (mus["e2p"][:, 1] - mus["e2m"][:, 1]) * tscale[1] / (2 * d)
            r = 0.5 * (r1 + r2)
            m, se = boot(r, seed=args.seed)
            m1, se1 = boot(r1, seed=args.seed)
            m2, se2 = boot(r2, seed=args.seed)
            print(f"  primary_only={str(po):5s} d={d:.3f}: <R>={m:+.4f} +/- {se:.4f}   "
                  f"(e1 {m1:+.4f}+/-{se1:.4f}, e2 {m2:+.4f}+/-{se2:.4f})   "
                  f"sd_obj={r.std():.4f}")
            if po and abs(d - delta_tr) < 1e-12:
                # WHERE the response lives: the certified 0.2930 is a FULL-population mean,
                # the 5C estimator runs on the true-cut (bright/large) subset.
                mag = df["r_input_p"].to_numpy(float)
                re_ = df["Re_input_p"].to_numpy(float)
                intc = (re_ > re_min) & (mag < mag_max)
                for lab, msk in [("true_cut (Re>0.3,mag<26)", intc),
                                 ("NOT true_cut", ~intc)]:
                    if msk.sum() > 20:
                        mm, ss = boot(r[msk], seed=args.seed)
                        print(f"      {lab:26s} n={int(msk.sum()):6d}  <R>={mm:+.4f}+/-{ss:.4f}")
                for lo, hi in [(0, 21), (21, 23), (23, 24), (24, 25), (25, 26), (26, 99)]:
                    msk = (mag >= lo) & (mag < hi)
                    if msk.sum() > 20:
                        mm, ss = boot(r[msk], seed=args.seed)
                        print(f"      mag [{lo},{hi}) n={int(msk.sum()):6d}  "
                              f"<R>={mm:+.4f}+/-{ss:.4f}")
                if args.rsim_npz:
                    # TRUTH SIDE on THIS population: the R_sim (flux x size x blend) grid the
                    # model was supervised against, marginalised over the blend axis with the
                    # grid's own counts and re-weighted by the 5C population's (mag, size)
                    # histogram.  CONFOUND: the blend marginal is the TRAINING catalogue's,
                    # not this catalogue's, and the two have very different neighbour
                    # separations -- so this is an approximate truth, not an exact one.
                    z = np.load(args.rsim_npz)
                    R3, C3 = z["Rsim"], z["counts"].astype(float)
                    with np.errstate(invalid="ignore"):
                        R2 = np.nansum(np.where(np.isfinite(R3), R3 * C3, 0.0), axis=2) \
                            / np.maximum(np.nansum(np.where(np.isfinite(R3), C3, 0.0), axis=2), 1e-9)
                    ef_, es_ = z["edges_flux"], z["edges_size"]
                    fi = np.clip(np.digitize(mag, ef_) - 1, 0, len(ef_) - 2)
                    si = np.clip(np.digitize(re_, es_) - 1, 0, len(es_) - 2)
                    rt = R2[fi, si]
                    ok = np.isfinite(rt)
                    mm, ss = boot(rt[ok], seed=args.seed)
                    mo, so = boot(r[ok], seed=args.seed)
                    print(f"      R_sim GRID re-weighted to this population: <R_sim>={mm:+.4f}"
                          f"+/-{ss:.4f}  vs <R_model>={mo:+.4f}+/-{so:.4f}  "
                          f"-> model/sim = {mo / mm:.3f}   (n={int(ok.sum()):,})")
                    for lo, hi in [(0, 21), (21, 23), (23, 24), (24, 25), (25, 26)]:
                        msk = ok & (mag >= lo) & (mag < hi)
                        if msk.sum() > 20:
                            a, _ = boot(r[msk], seed=args.seed)
                            b_, _ = boot(rt[msk], seed=args.seed)
                            print(f"        mag [{lo},{hi}) model {a:+.4f} vs sim {b_:+.4f}"
                                  f"  ratio {a / b_ if b_ else float('nan'):.3f}")
                for lo, hi in [(0, 0.15), (0.15, 0.3), (0.3, 0.5), (0.5, 1.0), (1.0, 99)]:
                    msk = (re_ >= lo) & (re_ < hi)
                    if msk.sum() > 20:
                        mm, ss = boot(r[msk], seed=args.seed)
                        print(f"      Re  [{lo},{hi}) n={int(msk.sum()):6d}  "
                              f"<R>={mm:+.4f}+/-{ss:.4f}")
                # cross-response and the flux/size rows of the same Jacobian
                for j, nmj in enumerate(tnames):
                    a = (mus["e2p"][:, j] - mus["e2m"][:, j]) * tscale[j] / (2 * d)
                    mm, ss = boot(a, seed=args.seed)
                    print(f"      dmu[{nmj}]/dgamma2 = {mm:+.4f} +/- {ss:.4f}")

    # conditional scale of the flow, per target dim
    sub = min(2000, N)
    with torch.no_grad():
        s = model.mean_flow.sample(base["ctx"][:sub], n_samples=args.n_scale_samples)
    resid = (s - base["mu"][:sub, None, :]).cpu().numpy().astype(np.float64)
    sig = resid.std(axis=1).mean(axis=0)              # per-object sd, averaged over objects
    print(f"\n  flow conditional sd (standardised units, {args.n_scale_samples} draws x "
          f"{sub} objects): {np.round(sig, 4).tolist()}")
    print(f"  in physical units: {np.round(sig * tscale, 4).tolist()}  for {tnames}")

    for g in gammas:
        mg = build(model, df, intr, (0.0, 1.0), g, pre, nbr_std, dev)["mu"].cpu().numpy()
        dmu = (mg - base["mu"].cpu().numpy()).astype(np.float64)
        print(f"\n  gamma2 = {g}:  |<dmu>| and rms(dmu) in units of the conditional sd")
        for j, nmj in enumerate(tnames):
            mm, ss = boot(dmu[:, j], seed=args.seed)
            print(f"    {nmj:24s} <dmu>={mm:+.5f}+/-{ss:.5f}  "
                  f"= {mm / sig[j]:+.4f} sigma_cond   rms(dmu)/sigma_cond="
                  f"{np.sqrt((dmu[:, j]**2).mean()) / sig[j]:.4f}")

    # ------------------------------------------------------------- 1. INPUT FEATURES --
    print("\n" + "=" * 78)
    print("1. WHICH INPUTS MOVE  (standardised |df|/sd over the population)")
    print("=" * 78)
    for g in gammas:
        b = build(model, df, intr, (0.0, 1.0), g, pre, nbr_std, dev)
        print(f"\n  gamma2 = {g}")
        sdp = base["p"].std(axis=0)
        dp = b["p"] - base["p"]
        names = pfeat + [f"missing[{c}]" for c in pfeat]
        nmoved = 0
        for j in range(dp.shape[1]):
            rms = np.sqrt((dp[:, j] ** 2).mean())
            rel = rms / max(sdp[j], 1e-12)
            tag = "  <-- MOVES" if rel > 1e-8 else ""
            nmoved += rel > 1e-8
            nm = names[j] if j < len(names) else f"col{j}"
            print(f"    P{j:2d} {nm:28s} sd={sdp[j]:.4f}  rms(df)={rms:.5f}  "
                  f"rms/sd={rel:.5f}{tag}")
        nb = base["nbg"]
        sdn = base["n"][nb].std(axis=0)
        dn = (b["n"] - base["n"])[nb]
        nmoved_n = 0
        for j in range(dn.shape[1]):
            rms = np.sqrt((dn[:, j] ** 2).mean())
            rel = rms / max(sdn[j], 1e-12)
            nmoved_n += rel > 1e-8
            print(f"    N{j:2d} {nfeat[j]:28s} sd={sdn[j]:.4f}  rms(df)={rms:.5f}  "
                  f"rms/sd={rel:.5f}{'  <-- MOVES' if rel > 1e-8 else ''}")
        print(f"    => {nmoved} of {dp.shape[1]} primary inputs move, "
              f"{nmoved_n} of {dn.shape[1]} neighbour inputs move")

    # ------------------------------------------------------------------ 2. CONTEXT ----
    print("\n" + "=" * 78)
    print("2. HOW FAR THE CONTEXT MOVES")
    print("=" * 78)
    c0 = base["ctx"].cpu().numpy().astype(np.float64)
    sdc = c0.std(axis=0)
    for g in gammas:
        cg = build(model, df, intr, (0.0, 1.0), g, pre, nbr_std, dev)["ctx"]
        dc = (cg.cpu().numpy().astype(np.float64) - c0)
        rel = np.sqrt((dc ** 2).mean(axis=0)) / np.maximum(sdc, 1e-12)
        # overall: Euclidean displacement vs population spread
        disp = np.sqrt((dc ** 2).sum(axis=1))
        spread = np.sqrt(((c0 - c0.mean(axis=0)) ** 2).sum(axis=1))
        md, sd_ = boot(disp, seed=args.seed)
        print(f"\n  gamma2 = {g}: per-dim rms(dctx)/sd(ctx): median {np.median(rel):.5f}, "
              f"p90 {np.percentile(rel, 90):.5f}, max {rel.max():.5f} (dim {int(rel.argmax())})")
        print(f"    overall |dctx| = {md:.4f} +/- {sd_:.4f} vs population spread "
              f"|ctx-<ctx>| = {spread.mean():.4f}  ->  ratio {md / spread.mean():.5f}")
        inkeep = np.isin(np.arange(len(rel)), keep)
        print(f"    flow-visible dims ({inkeep.sum()}): median rel {np.median(rel[inkeep]):.5f}"
              + (f";  mu-only dims ({(~inkeep).sum()}): median rel "
                 f"{np.median(rel[~inkeep]):.5f}" if (~inkeep).sum() else
                 ";  mu-only dims: NONE (keep_indices covers the whole context)"))
        top = np.argsort(-rel)[:8]
        print(f"    top-8 moving dims: "
              + ", ".join(f"{int(t)}:{rel[t]:.4f}" for t in top))
        print(f"    fraction of dctx norm carried by those 8 dims: "
              f"{(dc[:, top] ** 2).sum() / (dc ** 2).sum():.3f}")

    # ------------------------------------------------------- 4. phi SIGNAL VS NOISE ----
    print("\n" + "=" * 78)
    print("4. SIGNAL VS NOISE IN phi, AT NODE LEVEL")
    print("=" * 78)
    ng, nk = args.n_gal_phi, args.n_node
    node_df = rows.iloc[:nk].reset_index(drop=True)
    gal_df = rows.iloc[nk:nk + ng].reset_index(drop=True)
    nintr, gintr = {}, {}
    for d_, dd in ((node_df, nintr), (gal_df, gintr)):
        dd["e1p"], dd["e2p"] = intrinsic_shape(d_, "p")
        dd["e1s"], dd["e2s"] = intrinsic_shape(d_, "s")
    gb = build(model, gal_df, gintr, (0.0, 1.0), 0.0, pre, nbr_std, dev)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    with torch.no_grad():
        xhat = model.mean_flow.sample(gb["ctx"], n_samples=1)[:, 0, :]
    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    kp = (torch.rand(len(gal_df), generator=gen).to(dev) < gb["pdet"])
    xhat = xhat[kp]
    print(f"  bank K={nk}, detected galaxies {int(kp.sum())}/{ng}")
    nb0 = build(model, node_df, nintr, (0.0, 1.0), 0.0, pre, nbr_std, dev)
    phi0 = phi_block(model, xhat, nb0["ctx"], torch.log(nb0["pdet"]), 256)
    sd_node = phi0.std(axis=1)                      # node-to-node spread, per galaxy
    for g in gammas:
        nbg_ = build(model, node_df, nintr, (0.0, 1.0), g, pre, nbr_std, dev)
        phig = phi_block(model, xhat, nbg_["ctx"], torch.log(nbg_["pdet"]), 256)
        dphi = phig - phi0
        rms_dphi = np.sqrt((dphi ** 2).mean(axis=1))
        mean_dphi = dphi.mean(axis=1)
        ratio = rms_dphi / sd_node
        m, se = boot(ratio, np.median, seed=args.seed)
        m2, se2 = boot(rms_dphi, np.median, seed=args.seed)
        m3, se3 = boot(sd_node, np.median, seed=args.seed)
        print(f"\n  gamma2 = {g}:")
        print(f"    node-to-node spread sd_k(phi_k(0))     median {m3:.3f} +/- {se3:.3f}")
        print(f"    shear-induced rms_k(phi_k(g)-phi_k(0)) median {m2:.4f} +/- {se2:.4f}")
        print(f"    per-node signal/noise ratio            median {m:.5f} +/- {se:.5f}")
        mm, ss = boot(mean_dphi, seed=args.seed)
        print(f"    mean_k dphi (per galaxy) = {mm:+.4f} +/- {ss:.4f}")
        # posterior-weighted: how much the log-evidence moves
        from scipy.special import logsumexp
        dlz = logsumexp(phig, axis=1) - logsumexp(phi0, axis=1)
        mm, ss = boot(dlz, seed=args.seed)
        print(f"    d log Z per galaxy       = {mm:+.5f} +/- {ss:.5f}  "
              f"(sd over galaxies {dlz.std():.4f})")
    print("\nDONE_SIGNALREACH", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
