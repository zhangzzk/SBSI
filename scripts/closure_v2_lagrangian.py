"""`INFERENCE.md` §5C closure on the Gold-V2 joint forward model.

The one test of (5.8) that cannot be quadrature-limited: DRAW THE DATA FROM THE MODEL at a
known shear, then ask the estimator to return it.  Model and data agree by construction, so
any deviation is the estimator, not the physics.  It exercises the two pieces the V1 shape
route could not have -- the detection channel and the population terms -- because the V2
checkpoint carries `detection_prob` in the same network as the likelihood.

    phi_k(g) = log p_obs( xhat_i | ctx(S_g z_k) ) + log Pdet( ctx(S_g z_k) )        (5.5b)
    s_i = E_w[phi'],  I_i = -E_w[phi''] - Var_w(phi'),   w_k propto exp(phi_k(0))
    P(g) = mean_k Pdet( ctx(S_g z_k) ),  <s>_sel = (log P)'(0),  I_sel = -(log P)''(0)
    ghat = ( sum_i s_i - N <s>_sel ) / ( sum_i I_i - N I_sel )                       (5.8)

`P_pass = 1` here: no cut is applied on the measured vector, so the population term is
detection alone.  Adding a measured cut is the next increment and needs `P_pass` by MC over
the flow with common random numbers (§5C.5 point 2).

SHEAR.  `S_g` is applied with `primary_only=True`, matching the checkpoint's own
`primary_only_shear` metadata -- the model was TRAINED with only the primary sheared, so
that is the gamma-family it actually represents.  Shearing the neighbours too is a
different family and is a later increment (it is what would let §3's blend channel be
non-zero; the conditioning has the pair-angle features for it).

DERIVATIVES.  Central differences on gamma, but the stencil width is SWEPT and every width
is reported: on the V1 flow finite differences did not converge (WORKLOG cont.164), so a
single width proves nothing.  Stability across `--deltas` is the evidence; if it drifts,
the fix is autograd, as in `PosteriorShapeEstimator.log_likelihood_shear_derivatives`.

Usage:
    python scripts/closure_v2_lagrangian.py --n-gal 4000 --n-node 400 --gamma 0.1
"""

import argparse
import os
import sys

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import torch

SBSI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (SBSI_ROOT, os.path.join(SBSI_ROOT, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sbs_shear.forward_model import SetConditionedForwardModel  # noqa: E402
from sbs_shear.lagrangian_score import (  # noqa: E402
    posterior_weights,
    score_and_information,
    selection_terms,
    shear_estimate_bartlett,
    shear_estimate_louis,
)
from sbs_shear.measurement_model import TargetStandardizer  # noqa: E402
from sbs_shear.scene_model import SetFeatureStandardizer  # noqa: E402
from sbs_shear.selection_model import TabularPreprocessor  # noqa: E402

from train_joint_forward import (  # noqa: E402
    intrinsic_shape,
    neighbor_padded,
    shifted_feature_frame,
)

CAT = ("/project/ls-gruen/users/zekang.zhang/sbsi_catalogues/"
       "det_meas_ngmix_ap7_g0.0_train.feather")
CKPT = ("/project/ls-gruen/users/zekang.zhang/sbsi_caches/forward_proto/"
        "forward_ens_lr250_swa8_seed421_joint.pt")


def rebuild(ckpt_path, device):
    """Same reconstruction as `harvest_joint_rflow.rebuild`, kept in step with it."""
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ck["model_config"]
    model = SetConditionedForwardModel(
        target_dim=cfg["target_dim"], primary_dim=cfg["primary_dim"],
        neighbor_dim=cfg["neighbor_dim"], context_dim=cfg["context_dim"],
        set_hidden_dim=cfg["set_hidden_dim"], set_neighbor_layers=2, set_context_layers=2,
        flow_hidden_dim=cfg["flow_hidden_dim"], flow_layers=cfg["flow_layers"],
        n_flows=cfg["n_flows"], mean_hidden=cfg["mean_hidden"], det_hidden=cfg["det_hidden"],
        det_layers=2, activation="silu", pooling="sum", base_flow="affine",
    ).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return (model,
            TabularPreprocessor.from_state(ck["primary_preprocessor"]),
            SetFeatureStandardizer.from_state(ck["neighbor_preprocessor"]),
            TargetStandardizer.from_state(ck["target_transform"]),
            ck.get("metadata", {}))


def load_rows(path, n):
    """First `n` rows, by record batch -- reading the whole feather is far too slow."""
    with pa.memory_map(path) as src:
        rd = ipc.open_file(src)
        batches, got = [], 0
        for i in range(rd.num_record_batches):
            b = rd.get_batch(i)
            batches.append(b)
            got += b.num_rows
            if got >= n:
                break
    return pa.Table.from_batches(batches).to_pandas().head(n).reset_index(drop=True)


@torch.no_grad()
def scene_context(model, frame, intr, gamma_vec, pre, nbr_std, device):
    """`ctx(S_gamma z)` and `Pdet(S_gamma z)` for a whole frame of true scenes."""
    mag = float(np.hypot(*gamma_vec))
    gdir = (gamma_vec[0] / mag, gamma_vec[1] / mag) if mag > 0 else (1.0, 0.0)
    f, nbg = shifted_feature_frame(frame, intr, gdir, mag, primary_only=True)
    p = torch.as_tensor(pre.transform_frame(f), dtype=torch.float32, device=device)
    npad, mask = neighbor_padded(f, nbg, nbr_std)
    ctx = model.context(p,
                        torch.as_tensor(npad, dtype=torch.float32, device=device),
                        torch.as_tensor(mask, dtype=torch.float32, device=device))
    return ctx, model.detection_prob(ctx)


@torch.no_grad()
def phi_block(model, xhat_std, ctx, log_pdet, chunk, max_pairs=500_000):
    """`phi_ik = log p_obs(xhat_i | ctx_k) + log Pdet_k`, as an (N_gal, N_node) array.

    The galaxy chunk is capped on the PRODUCT `chunk * n_node`, because the expanded
    context is `(chunk*n_node, context_dim)` -- at context_dim 128 a naive chunk of 256
    with 40k nodes would ask for 5 GB.
    """
    n = xhat_std.shape[0]
    k = ctx.shape[0]
    chunk = max(1, min(chunk, max_pairs // max(k, 1)))
    out = np.empty((n, k), dtype=np.float64)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        b = e - s
        tgt = xhat_std[s:e, None, :].expand(b, k, xhat_std.shape[1]).reshape(b * k, -1)
        cc = ctx[None, :, :].expand(b, k, ctx.shape[1]).reshape(b * k, -1)
        ll = model.log_prob_obs(tgt, cc).view(b, k)
        out[s:e] = (ll + log_pdet[None, :]).double().cpu().numpy()
    return out


def weight_diagnostics(p0, self_idx=None):
    """Do the posterior weights actually TRACK the galaxy, or are they the same for all?

    A numerator of ~0 in (5.8) means `E[s_i] ~ <s>_sel` whatever the data's gamma, and the
    simplest way that happens is `w_k` barely depending on `i`: then `s_i = E_w[phi']` is
    the same constant for every galaxy and the shear signal cancels in the centring.

    `tv` is the total-variation distance of each galaxy's weight row from the population
    mean row, in [0, 1].  tv ~ 0 means the posterior is galaxy-INDEPENDENT (the failure);
    tv ~ 1 means each galaxy picks out its own scenes.  `self_rank` is available only when
    the bank contains each galaxy's own scene: the rank of that scene by weight, which
    should be at or very near 0 if the likelihood discriminates at all.
    """
    w = posterior_weights(p0)
    wbar = w.mean(axis=0)
    tv = 0.5 * np.abs(w - wbar[None, :]).sum(axis=1)
    out = dict(tv_mean=float(tv.mean()), tv_med=float(np.median(tv)),
               ess_mean=float(np.mean(1.0 / np.sum(w ** 2, axis=1))),
               ess_pop=float(1.0 / np.sum(wbar ** 2)))
    if self_idx is not None:
        order = np.argsort(-w, axis=1)
        rank = np.array([int(np.flatnonzero(order[j] == self_idx[j])[0])
                         for j in range(len(self_idx))])
        out["self_rank_med"] = float(np.median(rank))
        out["self_top1"] = float(np.mean(rank == 0))
        out["self_top1pct"] = float(np.mean(rank < max(1, w.shape[1] // 100)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=CKPT)
    ap.add_argument("--catalogue", default=CAT)
    ap.add_argument("--n-gal", type=int, default=4000)
    ap.add_argument("--n-node", type=int, default=400)
    ap.add_argument("--gamma", type=float, default=0.1, help="true shear, applied on axis 1")
    ap.add_argument("--deltas", default="0.02,0.01,0.005",
                    help="stencil half-widths to sweep; stability across them is the evidence")
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--self-bank", action="store_true",
                    help="DIAGNOSTIC: use the galaxies' OWN true scenes as the node bank, so "
                         "every galaxy has exact support.  Not a deployable configuration -- "
                         "it separates 'is the estimator right given support?' from 'is "
                         "sampling p_0 an adequate proposal?'")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(args.device)
    print(f"device={args.device}  torch={torch.__version__}")

    model, pre, nbr_std, target_std, meta = rebuild(args.checkpoint, dev)
    print(f"checkpoint: {os.path.basename(args.checkpoint)}")
    print(f"  metadata: {meta}")
    print(f"  target_dim={model.target_dim}  context_dim={model.context_dim}")

    rows = load_rows(args.catalogue, (args.n_gal + args.n_node) * 4)
    # The checkpoint's own metadata carries `true_cut`: the model was TRAINED only on rows
    # passing it, so evaluating the density outside it is out of domain, where log_prob and
    # its gamma-derivatives are unconstrained.  Applying it is not tuning -- it is staying
    # inside the model's support.
    tc = meta.get("true_cut")
    if tc is not None:
        re_min, mag_max = float(tc[0]), float(tc[1])
        keep_true = ((rows["Re_input_p"].to_numpy(float) > re_min)
                     & (rows["r_input_p"].to_numpy(float) < mag_max))
        print(f"true_cut Re>{re_min} & mag<{mag_max}: "
              f"{int(keep_true.sum()):,}/{len(rows):,} = {keep_true.mean():.1%} kept")
        rows = rows[keep_true].reset_index(drop=True)
    need = args.n_gal + (0 if args.self_bank else args.n_node)
    if len(rows) < need:
        raise SystemExit(f"only {len(rows):,} in-domain rows, need {need:,}")
    if args.self_bank:
        gal_df = rows.iloc[:args.n_gal].reset_index(drop=True)
        node_df = gal_df.copy()
    else:
        node_df = rows.iloc[:args.n_node].reset_index(drop=True)
        gal_df = rows.iloc[args.n_node:args.n_node + args.n_gal].reset_index(drop=True)
    print(f"rows: {len(node_df):,} node scenes + {len(gal_df):,} galaxy scenes")
    print("  NOTE: the latent is the FULL true scene, so the node bank is importance")
    print("  sampling in ~18 dimensions.  Watch ESS, not n_node (5B.3 item 5).")

    def intr_of(df):
        d = {}
        d["e1p"], d["e2p"] = intrinsic_shape(df, "p")
        d["e1s"], d["e2s"] = intrinsic_shape(df, "s")
        return d

    node_intr, gal_intr = intr_of(node_df), intr_of(gal_df)

    # ---- generate synthetic data FROM the model at the true shear ---------------------
    g_true = (0.0, float(args.gamma))
    ctx_g, pdet_g = scene_context(model, gal_df, gal_intr, g_true, pre, nbr_std, dev)
    with torch.no_grad():
        xhat = model.mean_flow.sample(ctx_g, n_samples=1)
        if xhat.dim() == 3:
            xhat = xhat[:, 0, :]
    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    keep = (torch.rand(len(gal_df), generator=gen).to(dev) < pdet_g)
    kept_idx = torch.nonzero(keep).squeeze(-1).cpu().numpy()
    xhat, n_keep = xhat[keep], int(keep.sum())
    print(f"detection: kept {n_keep:,}/{len(gal_df):,} = {n_keep / len(gal_df):.1%}  "
          f"(<Pdet> = {float(pdet_g.mean()):.4f})")
    if n_keep < 100:
        raise SystemExit("too few detected rows to estimate a shear")

    # ---- the estimator, at several stencil widths ------------------------------------
    print(f"\ntrue gamma2 = {args.gamma:+.4f}   (P_pass = 1: no measured cut applied)")
    print(f"{'delta':>8}  {'ghat (5.8)':>12} {'m':>9}  {'ghat (5.9)':>12} {'m':>9}  "
          f"{'<s>_sel':>10} {'I_sel':>9} {'<I>':>9} {'ESS':>7}")
    for d in [float(x) for x in args.deltas.split(",")]:
        phis, pdets = {}, {}
        for t in (0.0, +d, -d):
            c, pd_ = scene_context(model, node_df, node_intr, (0.0, t), pre, nbr_std, dev)
            phis[t] = phi_block(model, xhat, c, torch.log(pd_), args.chunk)
            pdets[t] = float(pd_.mean())
        p0, pp, pm = phis[0.0], phis[+d], phis[-d]
        d1 = (pp - pm) / (2 * d)
        d2 = (pp - 2 * p0 + pm) / d ** 2
        s, info = score_and_information(p0, d1, d2)          # sampled bank: no log_prior

        # DIRECT check of Fisher's identity (2.2): s_i must equal d_gamma log Z_i, where
        # Z_i(g) = mean_k exp(phi_k(g)) is the evidence.  This differences the EVIDENCE
        # rather than averaging the per-node derivative, so it shares no arithmetic with
        # `score_and_information` beyond phi itself.  Agreement => the estimator faithfully
        # computes the model's score and any failure is the model/bank; disagreement => an
        # implementation bug in the weighting or the derivative.
        from scipy.special import logsumexp as _lse
        lz = {t: _lse(v, axis=1) for t, v in (("0", p0), ("+", pp), ("-", pm))}
        s_direct = (lz["+"] - lz["-"]) / (2 * d)
        i_direct = -(lz["+"] - 2 * lz["0"] + lz["-"]) / d ** 2
        print(f"   Fisher check: <s>_Ew={np.mean(s):+.5f} vs <s>_direct={np.mean(s_direct):+.5f}"
              f"   corr={np.corrcoef(s, s_direct)[0, 1]:.6f}"
              f"   rms|ds|/sd={np.sqrt(np.mean((s - s_direct) ** 2)) / np.std(s):.2e}")
        print(f"   Louis check : <I>_Louis={np.mean(info):+.4f} vs "
              f"<I>_direct={np.mean(i_direct):+.4f}")


        lp = {t: np.log(v) for t, v in pdets.items()}         # P(g) = mean_k Pdet(S_g z_k)
        s_sel, i_sel = selection_terms(lp[0.0], (lp[+d] - lp[-d]) / (2 * d),
                                       (lp[+d] - 2 * lp[0.0] + lp[-d]) / d ** 2)
        print(f"   ghat from DIRECT evidence derivatives: (5.8) "
              f"{shear_estimate_louis(s_direct, i_direct, s_sel, i_sel):+.6f}   (5.9) "
              f"{shear_estimate_bartlett(s_direct, s_sel):+.6f}")
        louis = shear_estimate_louis(s, info, s_sel, i_sel)
        bart = shear_estimate_bartlett(s, s_sel)
        ess = float(np.mean(1.0 / np.sum(posterior_weights(p0) ** 2, axis=1)))
        if d == float(args.deltas.split(",")[0]):
            wd = weight_diagnostics(p0, kept_idx if args.self_bank else None)
            print("   weights: TV from population mean row: mean "
                  f"{wd['tv_mean']:.4f}, median {wd['tv_med']:.4f}   "
                  f"(0 = galaxy-INDEPENDENT posterior, 1 = fully galaxy-specific)")
            print(f"            ESS per galaxy {wd['ess_mean']:.1f}, "
                  f"ESS of the mean row {wd['ess_pop']:.1f}")
            if "self_rank_med" in wd:
                print(f"            own-scene rank: median {wd['self_rank_med']:.0f}, "
                      f"top-1 {wd['self_top1']:.1%}, top-1% {wd['self_top1pct']:.1%}")
        print(f"{d:>8.4f}  {louis:>12.6f} {louis / args.gamma - 1:>+8.2%}  "
              f"{bart:>12.6f} {bart / args.gamma - 1:>+8.2%}  "
              f"{s_sel:>10.5f} {i_sel:>9.4f} {float(np.mean(info)):>9.4f} {ess:>7.1f}")

    print("\n  Read the SPREAD across delta, not any single row: on the V1 flow finite")
    print("  differences did not converge (cont.164).  If these drift, switch to autograd.")
    print("  m is a single Newton step, so it is not expected to be exactly 0 at this gamma")
    print("  (MATH.md A2); the A.7 toy lands at +1.2% for (5.8) and -1.3% for (5.9) at 0.05.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
