"""Flux/size shear-response eval for the 4D true-conditioned measurement flow (ablation S2).

Harvests the mean head's dims 2,3 (measured_mag_auto, measured_log_flux_radius) response to a
shear applied along EACH OBJECT'S OWN shear direction ghat, and compares to the matched-pair
half-shear TRUTH. Magnitude and size are SCALARS (not spin-2), so:
  * OVERALL means are ~0 (shear conserves flux/area) -- a sanity check, not the signal;
  * the informative quantity is the spin-2 ORIENTATION-COUPLING slope
        b = sum(R * x) / sum(x^2),   x = e_int . ghat
    i.e. how the measured mag/size response depends on galaxy tilt relative to the shear.

Reports truth vs flow for OVERALL, per-true-size bins, and the coupling slope b, on the
ISOLATED acceptance set and ALL objects. Reuses eval_selfresp_gap's ruler/harvest machinery.

Units match the flow targets: dim2 = measured_mag_auto (linear mag), dim3 = measured_log_flux_radius
(so the size TRUTH is d log(flux_radius)/dg, not linear). Difference read out with the
target-standardizer scales[2],[3].

NOTE: for the S2 ckpts, dims 2,3 were shaped by NLL only (no response pin lives on them in the
V1-format trainer), so this tests whether the density fit ALONE reproduces the flux/size response.
"""
from __future__ import annotations
import argparse, glob, os, sys, time
import numpy as np
import pandas as pd
import pyarrow.feather as pf
import torch

SBSI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SBSI_ROOT not in sys.path:
    sys.path.insert(0, SBSI_ROOT)

from eval_selfresp_gap import (read_leg, domain_cut, build_shifted_context,  # noqa: E402
                               CAT, CROWD, NN, NGMIX, GAMMA, FLOW_COLS, SIZE_EDGES)
from sbs_shear.measurement_model import load_measurement_model  # noqa: E402

MS = ["measured_mag_auto", "measured_flux_radius"]


def load_ruler_fs(g0_leg, gS_leg, max_case, re_min, mag_max, iso_radius, crowd, nn, t0=None):
    """Ruler carrying BOTH-leg mag/size so a matched-pair truth for R_mag/R_size can be built,
    plus the intrinsic-shape-onto-ghat projection x for the coupling slope."""
    tick = (lambda: (time.time() - t0)) if t0 is not None else (lambda: 0.0)
    g0 = domain_cut(read_leg(g0_leg,
                             ["case", "input_index", "detected", "Re_input_p", "r_input_p",
                              "neighbored", "distance"] + NGMIX + MS, max_case),
                    re_min, mag_max).drop_duplicates(["case", "input_index"])
    gS = domain_cut(read_leg(gS_leg, FLOW_COLS + NGMIX + GAMMA, max_case), re_min, mag_max)
    gp = np.hypot(gS["gamma1_input_p"].to_numpy(float), gS["gamma2_input_p"].to_numpy(float))
    gS = gS[gp > 1e-6].reset_index(drop=True).drop_duplicates(["case", "input_index"])
    base = gS.merge(g0[["case", "input_index"] + NGMIX + MS],
                    on=["case", "input_index"], suffixes=("_g", "_0"))
    print(f"matched both-detected true-cut: N={len(base):,}  cases={base['case'].nunique()}  ({tick():.1f}s)",
          flush=True)

    gp = np.hypot(base["gamma1_input_p"].to_numpy(float), base["gamma2_input_p"].to_numpy(float))
    gmed = float(np.median(gp))
    gh1 = base["gamma1_input_p"].to_numpy(float) / gp
    gh2 = base["gamma2_input_p"].to_numpy(float) / gp

    # shape truth (sanity; identical to eval_selfresp_gap)
    de1 = base["measured_ngmix_g1_g"].to_numpy(float) - base["measured_ngmix_g1_0"].to_numpy(float)
    de2 = base["measured_ngmix_g2_g"].to_numpy(float) - base["measured_ngmix_g2_0"].to_numpy(float)
    R_shape = (de1 * gh1 + de2 * gh2) / gmed

    # mag truth (linear) and size truth (LOG) -- match flow dims 2,3
    R_mag = (base["measured_mag_auto_g"].to_numpy(float)
             - base["measured_mag_auto_0"].to_numpy(float)) / gmed
    fr_g = base["measured_flux_radius_g"].to_numpy(float)
    fr_0 = base["measured_flux_radius_0"].to_numpy(float)
    ok = np.isfinite(fr_g) & np.isfinite(fr_0) & (fr_g > 0) & (fr_0 > 0)
    R_size = np.full(len(base), np.nan)
    R_size[ok] = (np.log(fr_g[ok]) - np.log(fr_0[ok])) / gmed

    # spin-2 coupling variable x = e_int . ghat  (e1/e2_input_rot0_p = intrinsic pre-shear shape)
    ei1 = base["e1_input_rot0_p"].to_numpy(float)
    ei2 = base["e2_input_rot0_p"].to_numpy(float)
    x = ei1 * gh1 + ei2 * gh2

    keys = base[["case", "input_index"]]
    nbf = pf.read_table(crowd).to_pandas()[["case", "input_index", "nbr_flux_near",
                                            "nbr_flux_far", "nbr_flux_max"]]
    base = base.merge(nbf, on=["case", "input_index"], how="left")
    nnl = pf.read_table(nn).to_pandas()[["case", "input_index", "nn_dist_bright"]]
    j = keys.merge(nnl, on=["case", "input_index"], how="left")
    nnb = j["nn_dist_bright"].to_numpy(float)
    iso = (~np.isfinite(nnb)) | (nnb > iso_radius)
    size = base["Re_input_p"].to_numpy(float)
    print(f"g_med={gmed:.4f}  <R_shape>={np.nanmean(R_shape):+.4f}  "
          f"<R_mag>={np.nanmean(R_mag):+.5f}  <R_size(log)>={np.nanmean(R_size):+.5f}  "
          f"isolated frac={iso.mean():.1%}", flush=True)
    return dict(base=base, gh1=gh1, gh2=gh2, gmed=gmed, x=x,
                R_shape=R_shape, R_mag=R_mag, R_size=R_size, iso=iso, size=size)


@torch.no_grad()
def model_fluxsize_resp(base, gh1, gh2, bundle, delta, difference, device, chunk=300_000):
    """Per-object flow R_mag, R_size from mean-head dims 2,3, shearing each object along its ghat."""
    model = bundle.model.to(device)
    model.eval()
    # FENCE (realisation-aware head). The R_mag / R_size readout uses the MEAN head on dims 2,3
    # only, and the RA shift A is zero on those channels by construction, so mu-only stays exact
    # for them -- but only while ra_targets == [0, 1]. Assert that.
    if hasattr(model, "ra_targets"):
        _rt = [int(i) for i in model.ra_targets.tolist()]
        if _rt != [0, 1]:
            raise NotImplementedError(
                f"realisation-aware head has ra_targets={_rt}; this script reads mu only on dims "
                "2,3 and is exact only when A leaves those channels untouched (ra_targets=[0,1])")
        # ... but `Rshape` below reads dims 0,1, which A DOES write. mu-only is the wrong shape
        # response for an RA checkpoint (silently: the caller prints flow/truth-1 from it). Refuse.
        raise NotImplementedError(
            "eval_fluxsize_response also computes the Rshape CONTROL from the mean head (dims 0,1), "
            "which a realisation-aware head modifies through A(c,u); mu-only would report a wrong "
            "shape response with no warning. R_mag/R_size themselves stay exact -- to unblock, drop "
            "the Rshape control or read it from common-random-number draws of model.sample.")
    pp = bundle.condition_preprocessor
    cond = list(bundle.metadata.get("condition_features", pp.feature_names))
    scales = bundle.target_transform.scales
    sc0 = float(scales[0]); sc1 = float(scales[1])
    sc2 = float(scales[2]); sc3 = float(scales[3])
    rk = dict(pixel_rms=0.312, pixel_size=0.2, zero_mag=30.0, psf_fwhm=0.73, moffat_beta=2.224)

    def mu(ctx_np):
        t = torch.as_tensor(ctx_np, dtype=torch.float32, device=device)
        return model._mu(t).detach().cpu().numpy()

    n = len(base)
    Rmag = np.empty(n, dtype=np.float64)
    Rsize = np.empty(n, dtype=np.float64)
    Rshape = np.empty(n, dtype=np.float64)  # CONTROL: shape response through the SAME ghat leg
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        fr = base.iloc[s:e].reset_index(drop=True)
        g1 = gh1[s:e]; g2 = gh2[s:e]
        gd = (g1, g2)
        mu0 = mu(build_shifted_context(fr, gd, 0.0, pp, cond, rk))
        mup = mu(build_shifted_context(fr, gd, +delta, pp, cond, rk))
        if difference == "central":
            mum = mu(build_shifted_context(fr, gd, -delta, pp, cond, rk))
            Rmag[s:e] = 0.5 * (mup[:, 2] - mum[:, 2]) * sc2 / delta
            Rsize[s:e] = 0.5 * (mup[:, 3] - mum[:, 3]) * sc3 / delta
            Rshape[s:e] = 0.5 * ((mup[:, 0] - mum[:, 0]) * sc0 * g1
                                 + (mup[:, 1] - mum[:, 1]) * sc1 * g2) / delta
        else:
            Rmag[s:e] = (mup[:, 2] - mu0[:, 2]) * sc2 / delta
            Rsize[s:e] = (mup[:, 3] - mu0[:, 3]) * sc3 / delta
            Rshape[s:e] = ((mup[:, 0] - mu0[:, 0]) * sc0 * g1
                           + (mup[:, 1] - mu0[:, 1]) * sc1 * g2) / delta
    return Rmag, Rsize, Rshape


def _slope(R, x, mask):
    m = mask & np.isfinite(R) & np.isfinite(x)
    xx = x[m]; rr = R[m]
    d = float(np.sum(xx * xx))
    return float(np.sum(rr * xx) / d) if d > 0 else np.nan


def print_fs_table(name, R_truth, R_flow, x, size, sel, label):
    good = np.isfinite(R_truth) & np.isfinite(R_flow)
    print(f"\n[{label}]  {name}  N={int((sel & good).sum()):,}")
    print(f"  {'bin':>16} {'truth':>12} {'flow':>12} {'N':>10}")

    def row(tag, m):
        m = m & good
        if m.sum() < 30:
            print(f"  {tag:>16} {'-':>12} {'-':>12} {int(m.sum()):>10,}")
            return
        print(f"  {tag:>16} {np.mean(R_truth[m]):+12.5f} {np.mean(R_flow[m]):+12.5f} {int(m.sum()):>10,}")

    row("OVERALL", sel)
    for i in range(len(SIZE_EDGES) - 1):
        lo, hi = SIZE_EDGES[i], SIZE_EDGES[i + 1]
        row(f"sz[{lo:.2f},{hi:.2f})", sel & (size >= lo) & (size < hi))
    bt = _slope(R_truth, x, sel & good)
    bf = _slope(R_flow, x, sel & good)
    rel = (bf / bt - 1) * 100 if (bt and np.isfinite(bt)) else np.nan
    print(f"  {'COUPLING b':>16} {bt:+12.5f} {bf:+12.5f}   flow/truth-1 = {rel:+.1f}%   <- the signal")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", nargs="+", default=None)
    ap.add_argument("--ckpt-glob", default=None)
    ap.add_argument("--g0-leg", default=CAT + "det_meas_ngmix_g0.0_train.feather")
    ap.add_argument("--gS-leg", default=CAT + "det_meas_ngmix_g0.05_val.feather")
    ap.add_argument("--max-case", type=int, default=39)
    ap.add_argument("--true-re-min", type=float, default=0.3)
    ap.add_argument("--true-mag-max", type=float, default=26.0)
    ap.add_argument("--iso-radius", type=float, default=7.0)
    ap.add_argument("--response-difference", choices=["forward", "central"], default="forward")
    ap.add_argument("--response-delta", type=float, default=0.05)
    ap.add_argument("--crowd", default=CROWD)
    ap.add_argument("--nn", default=NN)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    ckpts = list(args.ckpt or [])
    if args.ckpt_glob:
        ckpts += sorted(glob.glob(args.ckpt_glob))
    if not ckpts:
        ap.error("give --ckpt and/or --ckpt-glob")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    print(f"device={device}  ckpts={len(ckpts)}  diff={args.response_difference}  delta={args.response_delta}",
          flush=True)
    for c in ckpts:
        print("   ", os.path.basename(c))

    ru = load_ruler_fs(args.g0_leg, args.gS_leg, args.max_case, args.true_re_min, args.true_mag_max,
                       args.iso_radius, args.crowd, args.nn, t0=t0)
    base, gh1, gh2, x = ru["base"], ru["gh1"], ru["gh2"], ru["x"]
    R_mag_hs, R_size_hs, iso, size = ru["R_mag"], ru["R_size"], ru["iso"], ru["size"]

    Rmag_seeds, Rsize_seeds, Rshape_seeds = [], [], []
    for c in ckpts:
        bundle = load_measurement_model(c, device=device)
        Rm, Rs, Rsh = model_fluxsize_resp(base, gh1, gh2, bundle, args.response_delta,
                                          args.response_difference, device)
        Rmag_seeds.append(Rm); Rsize_seeds.append(Rs); Rshape_seeds.append(Rsh)
        print(f"  {os.path.basename(c)}  b_mag={_slope(Rm, x, iso):+.5f}  b_size={_slope(Rs, x, iso):+.5f}  "
              f"({time.time()-t0:.1f}s)", flush=True)
    Rmag = np.nanmean(np.stack(Rmag_seeds, 0), 0)
    Rsize = np.nanmean(np.stack(Rsize_seeds, 0), 0)
    Rshape = np.nanmean(np.stack(Rshape_seeds, 0), 0)

    # ---- CONTROL: shape response via the SAME per-object-ghat leg must reproduce truth R_shape.
    #      If it does, the ghat harvest is sound and any mag/size miss is a real model property.
    R_shape_hs = ru["R_shape"]
    gI = iso & np.isfinite(R_shape_hs) & np.isfinite(Rshape)
    print(f"\n[CONTROL ghat-leg SHAPE, ISO]  truth <R_shape>={np.mean(R_shape_hs[gI]):+.4f}  "
          f"flow <R_shape>={np.mean(Rshape[gI]):+.4f}  flow/truth-1={(np.mean(Rshape[gI])/np.mean(R_shape_hs[gI])-1)*100:+.2f}%"
          f"   (must be ~0% for the harvest to be trusted)", flush=True)

    allobj = np.ones(len(base), bool)
    print_fs_table("R_mag (measured_mag_auto, linear)", R_mag_hs, Rmag, x, size, iso,
                   "ISOLATED nn_bright>%.0f\"" % args.iso_radius)
    print_fs_table("R_size (measured_log_flux_radius)", R_size_hs, Rsize, x, size, iso,
                   "ISOLATED nn_bright>%.0f\"" % args.iso_radius)
    print_fs_table("R_mag (measured_mag_auto, linear)", R_mag_hs, Rmag, x, size, allobj, "ALL objects")
    print_fs_table("R_size (measured_log_flux_radius)", R_size_hs, Rsize, x, size, allobj, "ALL objects")

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        np.savez(args.output, R_mag_hs=R_mag_hs, R_size_hs=R_size_hs, R_mag=Rmag, R_size=Rsize,
                 R_mag_seeds=np.stack(Rmag_seeds, 0), R_size_seeds=np.stack(Rsize_seeds, 0),
                 x=x, size=size, iso=iso, gmed=ru["gmed"],
                 case=base["case"].to_numpy(int), input_index=base["input_index"].to_numpy(int),
                 delta=args.response_delta, difference=args.response_difference, nseeds=len(ckpts))
        print(f"\nsaved {args.output}")
    print("FLUXSIZE_RESP_DONE", flush=True)


if __name__ == "__main__":
    main()
