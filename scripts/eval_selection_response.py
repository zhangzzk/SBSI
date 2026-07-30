"""Stage-2 SELECTION-response harness for the 4D true-conditioned measurement flow.

The flow now emits the JOINT measured (shape, mag, log-size) conditioned on TRUE props, with the
correct shear response on all three (shape pin + theta-coupling pin). Stage 2 pushes a moving
MEASURED cut through that joint and asks whether the flow reproduces the sim's SELECTION response
-- the extra response that appears because a measured cut lets objects across its boundary as the
measured observable itself responds to shear.

Estimator (the selected-CATALOGUE mean, which carries the moving-boundary term):

  For a measured cut C on measured observable x (size = flux_radius px, mag = mag_auto):
    <e.ghat>_S(leg) = sum over objects passing C at that leg of (e.ghat)  /  (# passing at that leg)
    R_C = ( <e.ghat>_S(gS)  -  <e.ghat>_S(g0) ) / g_med          (forward g0 -> g0.05, per-object ghat)

  This is NOT a per-object secant: pass(g0) != pass(gS) for a MEASURED cut, so the two-means form
  is required and it is exactly what the sim's selected catalogue does. For a TRUE cut pass(g0) ==
  pass(gS) (true props don't shear) -> the two-means form collapses to the per-object self-response
  on a fixed subset -> the selection term VANISHES (built-in null test).

TRUTH R_C : from the det_meas half-shear catalogue, matched both-detected pairs (g0<->g0.05),
            true-property acceptance, ISOLATED (no brighter true nbr within 7"; R_blend~0 so this
            is a pure flow-vs-truth selection test, firewall-clean -- no emulator, no constgold).
MODEL R_C : sample the flow's joint (shape,mag,logsize) at s=0 and s=+g_med (CRN, per-object ghat),
            apply the SAME cut to the samples, form the same selected-catalogue mean.

m_C = R_sim,C / R_model,C - 1  (isolated -> R_blend ~ 0). Gate: |m_C| within target across the
measured size (2.5/2.9/3.5/4.4 px) and mag (24/24.5/25) cut suite, AND the true-cut nulls ~ no-cut.

FIREWALL: nothing trains/fits here; scores existing ckpts against the det_meas truth only.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.feather as pf
import torch

SBSI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
for _p in (SBSI_ROOT, SCRIPTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sbs_shear.measurement_model import load_measurement_model  # noqa: E402
from sbs_shear.preprocessing import rescale  # noqa: E402
from sbs_shear.shear_map import apply_shear_to_ellipticity  # noqa: E402
# shared truth-ruler machinery (identical acceptance base as the self-response eval)
from eval_selfresp_gap import (  # noqa: E402
    CROWD, NN, read_leg, domain_cut,
)

CAT = "/project/ls-gruen/users/zekang.zhang/sbsi_catalogues/"
NGMIX = ["measured_ngmix_g1", "measured_ngmix_g2"]
GAMMA = ["gamma1_input_p", "gamma2_input_p"]
MEAS = ["measured_mag_auto", "measured_flux_radius"]
# raw columns rescale()/the true-conditioning touch on the g leg (superset of eval_selfresp FLOW_COLS)
FLOW_COLS = [
    "case", "input_index", "detected", "neighbored", "distance", "polarization_angle",
    "shear_component_convention",
    "Re_input_p", "Re_input_s", "r_input_p", "r_input_s",
    "sersic_n_input_p", "sersic_n_input_s", "redshift_input_p", "redshift_input_s",
    "e1_input_rot0_p", "e2_input_rot0_p", "e1_input_rot0_s", "e2_input_rot0_s",
    "axis_ratio_input_p", "position_angle_input_p", "axis_ratio_input_s", "position_angle_input_s",
]
RK = dict(pixel_rms=0.312, pixel_size=0.2, zero_mag=30.0, psf_fwhm=0.73, moffat_beta=2.224)


def build_base(g0_leg, gS_leg, max_case, re_min, mag_max, iso_radius, crowd, nn, t0):
    """Matched both-detected, true-cut base carrying measured (shape,mag,size) at BOTH legs,
    plus per-object ghat, gmed, isolation mask, and the flow conditioners (from the gS leg)."""
    tick = lambda: time.time() - t0
    g0 = domain_cut(read_leg(g0_leg,
                             ["case", "input_index", "detected", "Re_input_p", "r_input_p",
                              "neighbored", "distance"] + NGMIX + MEAS, max_case),
                    re_min, mag_max).drop_duplicates(["case", "input_index"])
    gS = domain_cut(read_leg(gS_leg, FLOW_COLS + NGMIX + MEAS + GAMMA, max_case), re_min, mag_max)
    gp = np.hypot(gS["gamma1_input_p"].to_numpy(float), gS["gamma2_input_p"].to_numpy(float))
    gS = gS[gp > 1e-6].reset_index(drop=True).drop_duplicates(["case", "input_index"])
    base = gS.merge(g0[["case", "input_index"] + NGMIX + MEAS], on=["case", "input_index"],
                    suffixes=("_g", "_0"))
    print(f"matched both-detected true-cut: N={len(base):,}  cases={base['case'].nunique()}  ({tick():.1f}s)",
          flush=True)

    gp = np.hypot(base["gamma1_input_p"].to_numpy(float), base["gamma2_input_p"].to_numpy(float))
    gmed = float(np.median(gp))
    gh1 = base["gamma1_input_p"].to_numpy(float) / gp
    gh2 = base["gamma2_input_p"].to_numpy(float) / gp

    # merge nbr_flux conditioners (isolated -> ~0) and the truth isolation distance
    nbf = pf.read_table(crowd).to_pandas()[["case", "input_index", "nbr_flux_near",
                                            "nbr_flux_far", "nbr_flux_max"]]
    base = base.merge(nbf, on=["case", "input_index"], how="left")
    for c in ("nbr_flux_near", "nbr_flux_far", "nbr_flux_max"):
        base[c] = base[c].fillna(0.0)
    nnl = pf.read_table(nn).to_pandas()[["case", "input_index", "nn_dist_bright"]]
    j = base[["case", "input_index"]].merge(nnl, on=["case", "input_index"], how="left")
    nnb = j["nn_dist_bright"].to_numpy(float)
    iso = (~np.isfinite(nnb)) | (nnb > iso_radius)
    print(f"nbr_flux matched {base['nbr_flux_near'].gt(0).mean():.1%}  iso frac={iso.mean():.1%}  "
          f"g_med={gmed:.4f}  ({tick():.1f}s)", flush=True)

    # zero the catalogue half-shear (we apply our OWN +/-s to the intrinsic shape rot0)
    base["gamma1_input_p"] = 0.0
    base["gamma2_input_p"] = 0.0
    return dict(base=base, gh1=gh1, gh2=gh2, gmed=gmed, iso=iso)


def truth_selected_response(base, gh1, gh2, gmed, sel, xcol, thr, keep_high):
    """R_C from the sim: two-means selected-catalogue response under a measured cut on xcol.
    keep_high=True keeps x>thr (size), False keeps x<thr (bright mag). Returns (R, frac, R_nocut, R_err)
    where R_err is the analytic standard error of R from the two selected-leg means."""
    e1_0 = base["measured_ngmix_g1_0"].to_numpy(float); e2_0 = base["measured_ngmix_g2_0"].to_numpy(float)
    e1_g = base["measured_ngmix_g1_g"].to_numpy(float); e2_g = base["measured_ngmix_g2_g"].to_numpy(float)
    p0 = e1_0 * gh1 + e2_0 * gh2
    pg = e1_g * gh1 + e2_g * gh2
    x0 = base[xcol + "_0"].to_numpy(float); xg = base[xcol + "_g"].to_numpy(float)
    pass0 = sel & (x0 > thr if keep_high else x0 < thr) & np.isfinite(p0)
    passg = sel & (xg > thr if keep_high else xg < thr) & np.isfinite(pg)
    n0 = int(pass0.sum()); ng = int(passg.sum())
    mbar0 = float(np.mean(p0[pass0])) if n0 else np.nan
    mbarg = float(np.mean(pg[passg])) if ng else np.nan
    R = (mbarg - mbar0) / gmed
    # error: the two legs share the SAME intrinsic galaxy, so on the both-pass set the per-object
    # difference (pg-p0) cancels intrinsic-shape variance -> matched-pair SE (much smaller than the
    # two-independent-means SE). Objects that switch pass across legs (the moving boundary) add a small
    # extra term ~ their fraction * <p>; we fold it in via the count mismatch.
    both = pass0 & passg
    nb = int(both.sum())
    if nb:
        d = pg[both] - p0[both]
        R_err = float(np.std(d)) / np.sqrt(nb) / gmed
    else:
        var0 = float(np.var(p0[pass0])) / n0 if n0 else np.nan
        varg = float(np.var(pg[passg])) / ng if ng else np.nan
        R_err = float(np.sqrt(var0 + varg)) / gmed
    frac = 0.5 * (pass0.mean() / max(sel.mean(), 1e-9) + passg.mean() / max(sel.mean(), 1e-9))
    # no-cut baseline on the same sel (per-object == two-means since pass identical)
    good = sel & np.isfinite(p0) & np.isfinite(pg)
    R_nc = (float(np.mean(pg[good])) - float(np.mean(p0[good]))) / gmed
    return R, frac, R_nc, R_err


@torch.no_grad()
def model_selected_response(bundle, base, gh1, gh2, gmed, sel, cuts, n_samples, batch_size,
                            flow_seed, device, chunk=100_000):
    """R_C from the flow: sample joint (shape,mag,logsize) at s=0 and s=+gmed (CRN, per-object ghat),
    apply each cut to the samples, accumulate the selected-catalogue mean per leg.

    cuts: list of dicts {name, dim, thr_raw, keep_high}. Returns {name: (R_model, frac_model)} and
    also the no-cut R_model (key '__nocut__'). Accumulates sum(proj) & count of passing draws only,
    so memory is O(chunk * n_samples) transient."""
    model = bundle.model.to(device); model.eval()
    idx_all = np.where(sel)[0]
    intr1 = base["e1_input_rot0_p"].to_numpy(float).copy()
    intr2 = base["e2_input_rot0_p"].to_numpy(float).copy()

    # accumulators: per cut per leg -> [sum_proj, count]; plus no-cut per leg
    acc = {c["name"]: {0: [0.0, 0], 1: [0.0, 0]} for c in cuts}
    acc["__nocut__"] = {0: [0.0, 0], 1: [0.0, 0]}

    def leg_draws(fr_idx, s, seed):
        fr = base.iloc[fr_idx].reset_index(drop=True).copy()
        e1s, e2s = apply_shear_to_ellipticity(intr1[fr_idx], intr2[fr_idx],
                                              s * gh1[fr_idx], s * gh2[fr_idx])
        fr["e1_input_rot0_p"] = e1s; fr["e2_input_rot0_p"] = e2s
        fr = rescale(fr, **RK)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        return bundle.sample(fr, n_samples=n_samples, batch_size=batch_size)  # (n, ns, 4)

    for cs in range(0, len(idx_all), chunk):
        ci = idx_all[cs:cs + chunk]
        g1c = gh1[ci][:, None]; g2c = gh2[ci][:, None]
        seed = flow_seed + cs  # same seed for both legs of this chunk (CRN); varies across chunks
        for leg, s in ((0, 0.0), (1, +gmed)):
            d = leg_draws(ci, s, seed)                       # (n, ns, 4)
            proj = d[:, :, 0] * g1c + d[:, :, 1] * g2c       # (n, ns)
            mag = d[:, :, 2]; logsz = d[:, :, 3]
            fin = np.isfinite(proj)
            acc["__nocut__"][leg][0] += float(proj[fin].sum()); acc["__nocut__"][leg][1] += int(fin.sum())
            for c in cuts:
                xv = logsz if c["dim"] == 3 else mag
                pm = fin & ((xv > c["thr"]) if c["keep_high"] else (xv < c["thr"]))
                acc[c["name"]][leg][0] += float(proj[pm].sum()); acc[c["name"]][leg][1] += int(pm.sum())

    out = {}
    n_all = acc["__nocut__"][0][1] / max(len(idx_all), 1)  # ~n_samples
    for name, a in acc.items():
        mb0 = a[0][0] / a[0][1] if a[0][1] else np.nan
        mbg = a[1][0] / a[1][1] if a[1][1] else np.nan
        R = (mbg - mb0) / gmed
        frac = 0.5 * (a[0][1] + a[1][1]) / max(acc["__nocut__"][0][1], 1)
        out[name] = (R, frac)
    return out


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
    ap.add_argument("--size-cuts", type=float, nargs="+", default=[2.5, 2.9, 3.5, 4.4],
                    help="measured_flux_radius (px) lower cuts; keep x>thr")
    ap.add_argument("--mag-cuts", type=float, nargs="+", default=[24.0, 24.5, 25.0],
                    help="measured_mag_auto upper cuts; keep x<thr (bright)")
    ap.add_argument("--true-size-cuts", type=float, nargs="+", default=[0.5, 0.65],
                    help="TRUE Re_input_p lower cuts for the null test (selection must ~vanish)")
    ap.add_argument("--n-samples", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=16384)
    ap.add_argument("--flow-seed", type=int, default=12345)
    ap.add_argument("--all-too", action="store_true", help="also print the ALL-objects (diagnostic) table")
    ap.add_argument("--output", default=None)
    ap.add_argument("--tf32", action="store_true",
                    help="enable TF32 matmuls (Ampere+; no-op on v100/cpu). The flow is 10 coupling "
                         "layers x 3 conditioner layers x hidden 256 -- i.e. almost pure GEMM -- so "
                         "this is the single biggest GPU lever here. OPT-IN, not default: TF32 keeps "
                         "10 mantissa bits vs fp32's 23, and although the effect on a 128-draw mean "
                         "should be ~1e-3 relative, that must be MEASURED against an fp32 run before "
                         "any science number relies on it (jobs/job_selresp_tf32_check.sh).")
    args = ap.parse_args()

    ckpts = list(args.ckpt or [])
    if args.ckpt_glob:
        ckpts += sorted(glob.glob(args.ckpt_glob))
    if not ckpts:
        ap.error("give --ckpt and/or --ckpt-glob")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tf32_effective = False
    if args.tf32:
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # TF32 is Ampere+ (compute capability >= 8.0). On a V100 (7.0) these flags are a silent
        # NO-OP: results come out bit-identical to fp32 and nothing runs faster. Say so loudly --
        # a TF32-vs-fp32 comparison run on a V100 "passes" for the trivial reason that both legs
        # are fp32, which is a vacuous pass, not a validation.
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability()
            tf32_effective = cap[0] >= 8
            if not tf32_effective:
                print(f"!! --tf32 requested but device is compute capability {cap[0]}.{cap[1]} "
                      f"({torch.cuda.get_device_name()}); TF32 needs >= 8.0 (Ampere). "
                      f"THE FLAG IS A NO-OP HERE -- this run is plain fp32.", flush=True)
    t0 = time.time()
    print(f"device={device}  ckpts={len(ckpts)}  n_samples={args.n_samples}  "
          f"tf32={'ON (effective)' if tf32_effective else ('ON (NO-OP)' if args.tf32 else 'off')}",
          flush=True)
    for c in ckpts:
        print("   ", os.path.basename(c))

    ru = build_base(args.g0_leg, args.gS_leg, args.max_case, args.true_re_min, args.true_mag_max,
                    args.iso_radius, CROWD, NN, t0)
    base, gh1, gh2, gmed, iso = ru["base"], ru["gh1"], ru["gh2"], ru["gmed"], ru["iso"]

    # log-space thresholds for the flow's measured_log_flux_radius (dim 3); mag is dim 2 raw
    size_cuts = [dict(name=f"size>{c:g}", dim=3, thr=float(np.log(c)), keep_high=True, raw=c, kind="size")
                 for c in args.size_cuts]
    mag_cuts = [dict(name=f"mag<{c:g}", dim=2, thr=float(c), keep_high=False, raw=c, kind="mag")
                for c in args.mag_cuts]
    model_cuts = size_cuts + mag_cuts

    scopes = [("ISOLATED (nn_bright>%.0f\")" % args.iso_radius, iso)]
    if args.all_too:
        scopes.append(("ALL objects (diagnostic)", np.ones(len(base), bool)))

    saved = {}
    for scope_name, sel in scopes:
        print(f"\n================ {scope_name}  N={int(sel.sum()):,} ================", flush=True)
        # ---- MODEL: ensemble R_model over ckpts ----
        per_ckpt = []
        for c in ckpts:
            bundle = load_measurement_model(c, device=device)
            res = model_selected_response(bundle, base, gh1, gh2, gmed, sel, model_cuts,
                                          args.n_samples, args.batch_size, args.flow_seed, device)
            per_ckpt.append(res)
            print(f"  [model {os.path.basename(c)}]  nocut R_model={res['__nocut__'][0]:+.4f}  "
                  f"({time.time()-t0:.1f}s)", flush=True)
        # ensemble = mean over ckpts of R_model, frac
        def ens(name):
            Rs = [p[name][0] for p in per_ckpt]; Fs = [p[name][1] for p in per_ckpt]
            return float(np.mean(Rs)), float(np.mean(Fs)), float(np.std(Rs))
        Rm_nc, _, Rm_nc_sd = ens("__nocut__")

        # ---- TRUTH: no-cut + each measured cut ----
        # no-cut truth self-response
        _, _, Rsim_nc, Rsim_nc_err = truth_selected_response(base, gh1, gh2, gmed, sel,
                                                             "measured_flux_radius", -1e9, True)
        # selection SHIFT = R(cut)/R(nocut)-1, i.e. how much the moving cut shifts the response.
        # We report it for BOTH truth and model: if the model reproduces the shift, it recovers the
        # selection bias. m=R_sim/R_model-1 is the residual (absolute recovery of the selected response).
        print(f"\n  {'cut':>12} {'fracS':>6} {'fracM':>6} | {'R_sim':>9} {'R_model':>9} | "
              f"{'shift_sim':>9} {'shift_mod':>9} | {'m=Rsim/Rmod-1':>14}")
        print(f"  {'NO CUT':>12} {1.0:>6.2f} {1.0:>6.2f} | {Rsim_nc:+9.4f} {Rm_nc:+9.4f} | "
              f"{'--':>9} {'--':>9} | {(Rsim_nc/Rm_nc-1)*100 if Rm_nc else np.nan:+13.2f}%")
        rows = [dict(cut="NO CUT", frac=1.0, fracM=1.0, R_sim=Rsim_nc, R_sim_err=Rsim_nc_err,
                     R_model=Rm_nc, R_model_sd=Rm_nc_sd, raw=np.nan, kind="nocut", m=(Rsim_nc/Rm_nc-1))]
        for c in model_cuts:
            xcol = "measured_flux_radius" if c["kind"] == "size" else "measured_mag_auto"
            R_sim, frac_s, _, R_sim_err = truth_selected_response(base, gh1, gh2, gmed, sel, xcol,
                                                                 c["raw"], c["keep_high"])
            R_mod, frac_m, R_mod_sd = ens(c["name"])
            m = R_sim / R_mod - 1 if R_mod else np.nan
            shift_sim = (R_sim / Rsim_nc - 1) if Rsim_nc else np.nan  # truth selection shift
            shift_mod = (R_mod / Rm_nc - 1) if Rm_nc else np.nan      # model selection shift
            print(f"  {c['name']:>12} {frac_s:>6.2f} {frac_m:>6.2f} | {R_sim:+9.4f} {R_mod:+9.4f} | "
                  f"{shift_sim*100:+8.2f}% {shift_mod*100:+8.2f}% | {m*100:+13.2f}%")
            rows.append(dict(cut=c["name"], frac=frac_s, fracM=frac_m, R_sim=R_sim, R_sim_err=R_sim_err,
                             R_model=R_mod, R_model_sd=R_mod_sd, raw=c["raw"], kind=c["kind"], m=m,
                             shift_sim=shift_sim, shift_mod=shift_mod))

        # ---- NULL TEST: TRUE-property cuts (selection must ~vanish: R_sim(trueC) == self-resp on subset) ----
        print(f"\n  --- NULL TEST (TRUE-Re cuts: moving-boundary term must vanish) ---")
        print(f"  {'true cut':>12} {'frac':>6} | {'R_sim(2mean)':>12} {'R_sim(perobj)':>13} {'boundary':>9}")
        Re = base["Re_input_p"].to_numpy(float)
        e1_0 = base["measured_ngmix_g1_0"].to_numpy(float); e2_0 = base["measured_ngmix_g2_0"].to_numpy(float)
        e1_g = base["measured_ngmix_g1_g"].to_numpy(float); e2_g = base["measured_ngmix_g2_g"].to_numpy(float)
        p0 = e1_0 * gh1 + e2_0 * gh2; pg = e1_g * gh1 + e2_g * gh2
        for tc in args.true_size_cuts:
            keep = sel & (Re > tc) & np.isfinite(p0) & np.isfinite(pg)
            if keep.sum() < 100:
                continue
            R_2m = (float(np.mean(pg[keep])) - float(np.mean(p0[keep]))) / gmed  # two-means (pass same both legs)
            R_po = float(np.mean((pg[keep] - p0[keep]))) / gmed                  # per-object secant
            print(f"  {'Re>%.2f'%tc:>12} {keep.mean()/max(sel.mean(),1e-9):>6.2f} | {R_2m:+12.4f} "
                  f"{R_po:+13.4f} {R_2m-R_po:+9.4f}")
        saved[scope_name] = rows

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        out = dict(gmed=gmed, ckpts=[os.path.basename(c) for c in ckpts],
                   scope_names=[s[0] for s in scopes], pixel_size=0.2)
        for si, (scope_name, _) in enumerate(scopes):
            rows = saved[scope_name]
            def col(k, default=np.nan):
                return np.array([r.get(k, default) for r in rows])
            out[f"s{si}_cut"] = np.array([r["cut"] for r in rows])
            out[f"s{si}_kind"] = np.array([r.get("kind", "") for r in rows])
            for k in ("raw", "R_sim", "R_sim_err", "R_model", "R_model_sd",
                      "shift_sim", "shift_mod", "m", "frac", "fracM"):
                out[f"s{si}_{k}"] = col(k)
        np.savez(args.output, **out)
        print(f"\nsaved {args.output}")
    print("SELECTION_RESPONSE_DONE", flush=True)


if __name__ == "__main__":
    main()
