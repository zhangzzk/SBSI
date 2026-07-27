#!/usr/bin/env python
"""R_theta held-out readout for the Gold-V2 joint forward model (Stage-2 gate, cont.120 NEXT-b).

Loads one or more TRAINED joint-forward checkpoints and reads out, on a HELD-OUT half-shear val
sample, the flow's MEASURED shear responses that the ``--lam-theta`` pin targets -- WITHOUT any
retraining. It answers the question the trainer's validation never prints:

  did the theta pin actually move the *held-out* measured-size response toward the target,
  while keeping the shape response R_shape ~ +0.71?

For each checkpoint we push the +/-delta intrinsic-ellipticity contexts through the (frozen) mean
head and read cols 0..3 = (measured e1, e2, mag, log flux_radius):

  R_shape  = 0.5*<(mu_e1p-mu_e1m)*sc0 + (mu_e2p-mu_e2m)*sc1> / (2 delta)      (dims 0,1; target ~+0.71)
  dS1,dS2  = (mu_.p - mu_.m)[:,3]*sc3 / (2 delta)                             (measured log-size response)
  dM1,dM2  = (mu_.p - mu_.m)[:,2]*sc2 / (2 delta)                             (measured mag response)

  b_size (flow) = spin-2 coupling slope, regress [dS1;dS2] on [e1_int;e2_int] through the origin
                = (sum dS1*e1 + sum dS2*e2) / (sum e1^2 + sum e2^2)           (theta target: global +0.490 log)
  b_mag  (flow) = same for the mag response                                   (theta target: global +0.025)
  R_size_coh    = <dS1>,<dS2>  (even/coherent part; the spin-2 part averages ~0)

This is EXACTLY the quantity ``theta_coupling_residual`` pins in the loss (it enforces dS1=bS*e1,
dS2=bS*e2 per cell), read back out on held-out cases. FIREWALL: reads the HALF-SHEAR g0 flow leg
only (never constgold); this is evaluation, nothing trains.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
SBSI_ROOT = os.path.dirname(SCRIPTS_DIR)
for _p in (SBSI_ROOT, SCRIPTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# reuse the trainer's data pipeline + model builder verbatim (no reimplementation)
import train_joint_forward as T  # noqa: E402
from sbs_shear.measurement_model import TargetStandardizer  # noqa: E402
from sbs_shear.scene_model import SetFeatureStandardizer  # noqa: E402
from sbs_shear.selection_model import TabularPreprocessor  # noqa: E402


def load_ckpt_model(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = ck["model_config"]
    args = SimpleNamespace(**{k: cfg[k] for k in
                             ("context_dim", "set_hidden_dim", "flow_hidden_dim", "flow_layers",
                              "n_flows", "mean_hidden", "det_hidden")})
    model = T.build_model(cfg["primary_dim"], cfg["neighbor_dim"], args, device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    primary_pre = TabularPreprocessor.from_state(ck["primary_preprocessor"])
    nbr_std = SetFeatureStandardizer.from_state(ck["neighbor_preprocessor"])
    target_std = TargetStandardizer.from_state(ck["target_transform"])
    return model, primary_pre, nbr_std, target_std, ck


@torch.no_grad()
def readout(model, D, sc, delta, bs=32768):
    """Return per-object arrays of the measured-shape/mag/size responses, and e_int."""
    r_sh, dS1, dS2, dM1, dM2 = [], [], [], [], []
    sc0, sc1, sc2, sc3 = sc
    inv = 1.0 / (2.0 * delta)
    for idx in T.batches(D["_n"], bs, D["mask"].device, shuffle=False):
        m = D["mask"][idx]
        mu1p = model.mu(model.context(D["p_e1p"][idx], D["n_e1p"][idx], m))
        mu1m = model.mu(model.context(D["p_e1m"][idx], D["n_e1m"][idx], m))
        mu2p = model.mu(model.context(D["p_e2p"][idx], D["n_e2p"][idx], m))
        mu2m = model.mu(model.context(D["p_e2m"][idx], D["n_e2m"][idx], m))
        r_sh.append((0.5 * ((mu1p[:, 0] - mu1m[:, 0]) * sc0 + (mu2p[:, 1] - mu2m[:, 1]) * sc1) * inv).cpu())
        dM1.append(((mu1p[:, 2] - mu1m[:, 2]) * sc2 * inv).cpu())
        dM2.append(((mu2p[:, 2] - mu2m[:, 2]) * sc2 * inv).cpu())
        dS1.append(((mu1p[:, 3] - mu1m[:, 3]) * sc3 * inv).cpu())
        dS2.append(((mu2p[:, 3] - mu2m[:, 3]) * sc3 * inv).cpu())
    cat = lambda xs: torch.cat(xs).numpy()
    return dict(r_shape=cat(r_sh), dS1=cat(dS1), dS2=cat(dS2), dM1=cat(dM1), dM2=cat(dM2),
                e1=D["e1_int"].cpu().numpy(), e2=D["e2_int"].cpu().numpy(),
                binid=D["binid"].cpu().numpy())


def spin2_slope(d1, d2, e1, e2):
    """Origin regression of the response on intrinsic e (the spin-2 coupling the pin enforces)."""
    num = float(np.dot(d1, e1) + np.dot(d2, e2))
    den = float(np.dot(e1, e1) + np.dot(e2, e2))
    return num / den if den > 0 else np.nan


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpts", nargs="+", required=True,
                    help="checkpoint .pt paths (or bare tags resolved under --outdir).")
    ap.add_argument("--outdir", default="/project/ls-gruen/users/zekang.zhang/sbsi_caches/forward_proto")
    ap.add_argument("--flow-catalogue",
                    default="/project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_ngmix_g0.0_train.feather")
    ap.add_argument("--target-npz",
                    default="/home/z/Zekang.Zhang/SBSI/results/response_target_isoblend_RAWfine_c0-99_6x9x5.npz",
                    help="the SHAPE target grid (for flux/size/dist edges = the training binning).")
    ap.add_argument("--theta-target-npz",
                    default="/home/z/Zekang.Zhang/SBSI/results/response_target_theta_coupling_c0-99_6x9x5.npz",
                    help="the R_theta coupling target (per-cell b_size/b_mag to compare against).")
    ap.add_argument("--max-case", type=int, default=200)
    ap.add_argument("--train-case-max", type=int, default=160)
    ap.add_argument("--max-rows-val", type=int, default=500000)
    ap.add_argument("--max-rows-flow", type=int, default=2400000)
    ap.add_argument("--seed", type=int, default=421)
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--true-re-min", type=float, default=0.3)
    ap.add_argument("--true-mag-max", type=float, default=26.0)
    ap.add_argument("--output", default="/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/rtheta_readout.npz")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    true_cut = (args.true_re_min, args.true_mag_max)
    t0 = time.time()

    # ---- edges + coupling grids (define the per-cell binning; same as training) ----
    tz = np.load(args.target_npz)
    flux_edges = tz["edges_flux"].astype(float)
    size_edges = tz["edges_size"].astype(float)
    dist_edges = tz["edges_dist"].astype(float)
    tt = np.load(args.theta_target_npz)
    theta_size_grid = tt["coupling_size"].astype(np.float32)
    theta_mag_grid = tt["coupling_mag"].astype(np.float32)
    b_size_target = float(tt["b_size_global"])
    b_mag_target = float(tt["b_mag_global"])
    print(f"targets: b_size_global(log)={b_size_target:+.4f}  b_mag_global={b_mag_target:+.4f}  "
          f"grid={theta_size_grid.shape}", flush=True)

    # ---- held-out val stream (same seed/catalogue/cases as training -> identical val split) ----
    print("Loading half-shear g0 flow stream (val split)...", flush=True)
    _, Fva_df = T.load_stream(args.flow_catalogue, "flow", args.max_case, args.train_case_max,
                              args.max_rows_flow, args.max_rows_val, args.seed, true_cut=true_cut)
    print(f"  val rows={len(Fva_df):,}  ({time.time()-t0:.1f}s)", flush=True)

    ckpts = []
    for c in args.ckpts:
        ckpts.append(c if os.path.isabs(c) or c.endswith(".pt")
                     else os.path.join(args.outdir, f"forward_{c}_joint.pt"))

    rows = []
    for cpath in ckpts:
        tag = os.path.basename(cpath).replace("forward_", "").replace("_joint.pt", "")
        if not os.path.exists(cpath):
            print(f"[skip] {tag}: missing {cpath}", flush=True)
            continue
        model, primary_pre, nbr_std, target_std, ck = load_ckpt_model(cpath, device)
        sc = tuple(float(target_std.scales[i]) for i in range(4))
        # build D with THIS checkpoint's preprocessors (theta grids -> e_int populated in D)
        D = T.precompute_stream(Fva_df, "flow", primary_pre, nbr_std, args.delta, device,
                                target_std=target_std, flux_edges=flux_edges, size_edges=size_edges,
                                dist_edges=dist_edges, primary_only=True,
                                theta_size_grid=theta_size_grid, theta_mag_grid=theta_mag_grid)
        o = readout(model, D, sc, args.delta)
        Rsh = float(np.mean(o["r_shape"]))
        bS = spin2_slope(o["dS1"], o["dS2"], o["e1"], o["e2"])
        bM = spin2_slope(o["dM1"], o["dM2"], o["e1"], o["e2"])
        RS_coh = 0.5 * float(np.mean(o["dS1"]) + np.mean(o["dS2"]))
        RM_coh = 0.5 * float(np.mean(o["dM1"]) + np.mean(o["dM2"]))
        # per-cell b_size (flow) vs target, on occupied cells with >=200 obj
        binid = o["binid"]; tgtflat = theta_size_grid.reshape(-1)
        cells = []
        for k in np.unique(binid):
            sel = binid == k
            if int(sel.sum()) < 200 or not np.isfinite(tgtflat[k]):
                continue
            bk = spin2_slope(o["dS1"][sel], o["dS2"][sel], o["e1"][sel], o["e2"][sel])
            cells.append((bk, float(tgtflat[k])))
        cells = np.array(cells) if cells else np.zeros((0, 2))
        cell_rmse = float(np.sqrt(np.mean((cells[:, 0] - cells[:, 1]) ** 2))) if len(cells) else np.nan
        rows.append(dict(tag=tag, R_shape=Rsh, b_size=bS, b_mag=bM,
                         R_size_coh=RS_coh, R_mag_coh=RM_coh, cell_rmse=cell_rmse, n=len(o["r_shape"])))
        print(f"[{tag:22s}] R_shape={Rsh:+.4f}  b_size(log)={bS:+.4f} (tgt {b_size_target:+.3f})  "
              f"b_mag={bM:+.4f} (tgt {b_mag_target:+.3f})  R_size_coh={RS_coh:+.4f}  "
              f"per-cell_rmse={cell_rmse:.3f}  N={len(o['r_shape']):,}", flush=True)

    print("\n===== R_theta READOUT (held-out val; shape target R_shape~+0.71, "
          f"b_size target {b_size_target:+.3f}) =====", flush=True)
    print(f"  {'tag':<22} {'R_shape':>8} {'b_size':>8} {'b_mag':>8} {'R_sizecoh':>10} {'cellRMSE':>9}")
    for r in rows:
        print(f"  {r['tag']:<22} {r['R_shape']:>+8.4f} {r['b_size']:>+8.4f} {r['b_mag']:>+8.4f} "
              f"{r['R_size_coh']:>+10.4f} {r['cell_rmse']:>9.3f}")
    print(f"\n  TARGET                  {'~+0.71':>8} {b_size_target:>+8.3f} {b_mag_target:>+8.3f} "
          f"{'~0':>10} {'0':>9}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.savez(args.output, tags=np.array([r["tag"] for r in rows]),
             R_shape=np.array([r["R_shape"] for r in rows]),
             b_size=np.array([r["b_size"] for r in rows]),
             b_mag=np.array([r["b_mag"] for r in rows]),
             R_size_coh=np.array([r["R_size_coh"] for r in rows]),
             cell_rmse=np.array([r["cell_rmse"] for r in rows]),
             b_size_target=b_size_target, b_mag_target=b_mag_target)
    print(f"\nsaved {args.output}   ({time.time()-t0:.1f}s)", flush=True)
    print("RTHETA_READOUT_DONE", flush=True)


if __name__ == "__main__":
    main()
