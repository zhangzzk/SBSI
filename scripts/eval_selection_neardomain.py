"""Does the flow predict selection bias for cuts NEAR ITS TRAINING DOMAIN? (the regime that matters)

OWNER'S POINT, which reframes the 30u result. The flow is trained on mag < 26 and Re > 0.3", and it
is a KNOWN limitation that it is under-resolved outside that box. Cuts like "keep the top 30% by S/N"
walk deep inside the domain and are not what a real analysis applies. The question worth answering is
whether the flow predicts selection for cuts placed AROUND THE TRAINING EDGE. The existing constgold
table already hints the answer is yes there: at keep 0.95 it gave model +2.555% vs sim +2.599%,
agreeing to 0.04 pts, and only diverged as the cut tightened.

WHY INTRINSIC SHAPES HERE. In intrinsic mode both sim and model project the SAME exact sheared
intrinsic shape, so the only thing that can differ is WHICH OBJECTS PASS THE CUT. That does two
things at once: it isolates selection (no measurement-error term), and it removes the blend confound
that made the constgold model column non-comparable (30u) -- there is no R_blend to be missing when
the shape is identical on both sides. So `m_flow` below is a clean model-bias number, not a
composite.

THE TWO NUMBERS (owner's definitions, unchanged):
    m_sel  = R_sim(cut)   / R_sim(no cut) - 1      the shift the SIM has
    m_flow = R_model(cut) / R_sim(cut)   - 1       the bias the MODEL has predicting it

CUT AXES. All on MEASURED quantities, which is what an analysis actually cuts on:
  * magnitude   -- at and just inside the training edge (26.0, 25.5, 25.0)
  * size        -- around the PSF half-light radius R50 = 0.527", which is where the size-axis
                   selection effect was measured to switch on
  * proxy S/N   -- at MILD keep-fractions only (0.98/0.95/0.90), never the aggressive tail
  * JOINT       -- magnitude AND size, and S/N AND size. This is the realistic case and it is why
                   `truth_selected_response` / `model_selected_response` gained a list/`conds`
                   branch; single-variable calls are untouched.

S/N IS THE PROXY ON BOTH SIDES, self-consistently: the flow does not output flux_auto/fluxerr_auto,
so it can only represent a mag+size proxy. The proxy-vs-real error is REPORTED SEPARATELY below on
the sim, at these same mild thresholds, so it is never silently folded into m_flow.

DETECTION IS SEPARATED, as requested: build_base keeps only objects detected in BOTH legs.
FIREWALL: half-shear legs. No constgold, no emulator, nothing trained or selected here.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

SBSI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
for _p in (SBSI_ROOT, SCRIPTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sbs_shear.measurement_model import load_measurement_model  # noqa: E402
import eval_selection_response as ESR  # noqa: E402
from eval_selection_response import (  # noqa: E402
    CAT, CROWD, NN, apply_shear_to_ellipticity, model_selected_response, truth_selected_response,
)

PX = 0.2            # arcsec per pixel
PSF_R50 = 0.5268    # arcsec, Moffat FWHM 0.73" beta 2.224
SIZE_COL = "measured_flux_radius"
MAG_COL = "measured_mag_auto"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--g0-leg", default=CAT + "det_meas_ngmix_g0.0_train.feather")
    ap.add_argument("--gS-leg", default=CAT + "det_meas_ngmix_g0.05_val.feather")
    ap.add_argument("--max-case", type=int, default=39)
    ap.add_argument("--true-re-min", type=float, default=0.3)
    ap.add_argument("--true-mag-max", type=float, default=26.0)
    ap.add_argument("--iso-radius", type=float, default=7.0)
    ap.add_argument("--sn-a", type=float, default=-0.3676)
    ap.add_argument("--sn-b", type=float, default=-0.7736)
    ap.add_argument("--mag-cuts", type=float, nargs="+", default=[26.0, 25.5, 25.0])
    ap.add_argument("--size-cuts", type=float, nargs="+", default=[0.50, 0.55, 0.60, 0.70],
                    help="arcsec; PSF R50 = 0.527\". 0.70 is a REGRESSION ANCHOR: this run edits "
                         "shared harness code (the list/`conds` joint-cut branch), and the dense "
                         "grid already published m_sel = +5.40% at R>0.70\". If that row does not "
                         "come back, the edit disturbed the existing single-cut path.")
    ap.add_argument("--sn-keeps", type=float, nargs="+", default=[0.98, 0.95, 0.90])
    ap.add_argument("--n-samples", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=16384)
    ap.add_argument("--flow-seed", type=int, default=12345)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    # The REAL S/N needs the flux columns; they are only used for the proxy-vs-real block.
    ESR.MEAS = list(dict.fromkeys(ESR.MEAS + ["measured_flux_auto", "measured_fluxerr_auto"]))
    ru = ESR.build_base(args.g0_leg, args.gS_leg, args.max_case, args.true_re_min,
                        args.true_mag_max, args.iso_radius, CROWD, NN, t0)
    base, gh1, gh2, gmed, iso = ru["base"], ru["gh1"], ru["gh2"], ru["gmed"], ru["iso"]
    print(f"device={device}  ckpts={len(args.ckpt)}  n_samples={args.n_samples}", flush=True)
    print(f"training domain: true mag < {args.true_mag_max}, true Re > {args.true_re_min}\"; "
          f"PSF R50 = {PSF_R50:.3f}\"", flush=True)

    lin = ("lin", args.sn_a, args.sn_b)
    mag0 = base[MAG_COL + "_0"].to_numpy(float)
    rad0 = base[SIZE_COL + "_0"].to_numpy(float)
    u0 = args.sn_a * mag0 + args.sn_b * np.log10(np.maximum(rad0, 1e-6))
    ok = iso & np.isfinite(u0)
    sn_thr = {kf: float(np.quantile(u0[ok], 1.0 - kf)) for kf in args.sn_keeps}

    # ---- the cut list. Each entry carries BOTH the sim spec (xcol/thr/keep_high or a list of
    # conditions) and the model spec (dim/thr/keep_high or `conds`), so the two sides cannot drift.
    cuts = []
    for c in args.mag_cuts:
        cuts.append(dict(name=f"mag<{c:g}", sim=(MAG_COL, c, False),
                         mod=dict(dim=2, thr=float(c), keep_high=False, lin=None)))
    for a in args.size_cuts:
        px = a / PX
        cuts.append(dict(name=f'R>{a:.2f}"', sim=(SIZE_COL, px, True),
                         mod=dict(dim=3, thr=float(np.log(px)), keep_high=True, lin=None)))
    for kf in args.sn_keeps:
        cuts.append(dict(name=f"SNproxy k{kf:.2f}", sim=(lin, sn_thr[kf], True),
                         mod=dict(dim=None, thr=sn_thr[kf], keep_high=True,
                                  lin=(args.sn_a, args.sn_b))))
    # JOINT: what a real analysis applies.
    joints = [
        (f'mag<26.0 & R>0.55"', [(MAG_COL, 26.0, False), (SIZE_COL, 0.55 / PX, True)],
         [dict(dim=2, thr=26.0, keep_high=False),
          dict(dim=3, thr=float(np.log(0.55 / PX)), keep_high=True)]),
        (f'mag<25.5 & R>0.55"', [(MAG_COL, 25.5, False), (SIZE_COL, 0.55 / PX, True)],
         [dict(dim=2, thr=25.5, keep_high=False),
          dict(dim=3, thr=float(np.log(0.55 / PX)), keep_high=True)]),
        (f'SNproxy k0.95 & R>0.55"', [(lin, sn_thr[0.95], True), (SIZE_COL, 0.55 / PX, True)],
         [dict(lin=(args.sn_a, args.sn_b), thr=sn_thr[0.95], keep_high=True),
          dict(dim=3, thr=float(np.log(0.55 / PX)), keep_high=True)]),
    ] if 0.95 in sn_thr else []
    for nm, simspec, modspec in joints:
        cuts.append(dict(name=nm,
                         sim=[dict(var=v, thr=t, keep_high=k) for v, t, k in simspec],
                         mod=dict(conds=modspec)))

    # ---- sim, exact intrinsic shapes -> PURE selection -----------------------------------------
    sim, frac_s = {}, {}
    R_nc, _, _, _ = truth_selected_response(base, gh1, gh2, gmed, iso, SIZE_COL, -1e9, True,
                                            intrinsic=True)
    for c in cuts:
        if isinstance(c["sim"], list):
            R, fr, _, _ = truth_selected_response(base, gh1, gh2, gmed, iso, c["sim"], 0.0, True,
                                                  intrinsic=True)
        else:
            xcol, thr, kh = c["sim"]
            R, fr, _, _ = truth_selected_response(base, gh1, gh2, gmed, iso, xcol, thr, kh,
                                                  intrinsic=True)
        sim[c["name"]], frac_s[c["name"]] = R, fr

    # ---- model ---------------------------------------------------------------------------------
    mcuts = []
    for c in cuts:
        d = dict(c["mod"]); d["name"] = c["name"]
        d.setdefault("lin", None); d.setdefault("conds", None); d.setdefault("dim", None)
        mcuts.append(d)
    names = ["__nocut__"] + [c["name"] for c in cuts]
    per = []
    for ck in args.ckpt:
        b = load_measurement_model(ck, device=device)
        per.append(model_selected_response(b, base, gh1, gh2, gmed, iso, mcuts, args.n_samples,
                                           args.batch_size, args.flow_seed, device, intrinsic=True))
        print(f"  scored {os.path.basename(ck)} ({time.time()-t0:.0f}s)", flush=True)
    mod = {k: float(np.mean([p[k][0] for p in per])) for k in names}
    sem = {k: (float(np.std([p[k][0] for p in per], ddof=1) / np.sqrt(len(per)))
               if len(per) > 1 else np.nan) for k in names}

    print("\n" + "=" * 100)
    print("SELECTION AT / NEAR THE TRAINING DOMAIN EDGE  (intrinsic shapes = PURE selection;")
    print("                                               isolated, both-detected)")
    print("=" * 100)
    print(f"  R_sim(no cut) = {R_nc:+.5f}    R_model(no cut) = {mod['__nocut__']:+.5f}")
    print(f"\n  {'cut':>26} {'keep':>7} | {'m_sel [%]':>11} {'m_flow [%]':>18}")
    for c in cuts:
        k = c["name"]
        m_sel = (sim[k] / R_nc - 1.0) * 100.0
        m_flow = (mod[k] / sim[k] - 1.0) * 100.0
        e = sem[k] / abs(sim[k]) * 100.0 if np.isfinite(sem[k]) else np.nan
        print(f"  {k:>26} {frac_s[k]:>7.3f} | {m_sel:>+10.3f}% {m_flow:>+11.3f} +- {e:<.3f}")

    # ---- proxy vs real S/N on the SIM, at these same mild thresholds ---------------------------
    # Reported separately so it is never folded into m_flow above.
    f0 = base["measured_flux_auto_0"].to_numpy(float)
    fe0 = base["measured_fluxerr_auto_0"].to_numpy(float)
    fg = base["measured_flux_auto_g"].to_numpy(float)
    feg = base["measured_fluxerr_auto_g"].to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        sn_0 = np.where(fe0 > 0, f0 / fe0, np.nan)
        sn_g = np.where(feg > 0, fg / feg, np.nan)
    i1 = base["e1_input_rot0_p"].to_numpy(float)
    i2 = base["e2_input_rot0_p"].to_numpy(float)
    s1, s2 = apply_shear_to_ellipticity(i1, i2, gmed * gh1, gmed * gh2)
    p_int, p_shr = i1 * gh1 + i2 * gh2, s1 * gh1 + s2 * gh2
    okr = iso & np.isfinite(sn_0) & np.isfinite(sn_g) & (sn_0 > 0) & (sn_g > 0) \
        & np.isfinite(p_int) & np.isfinite(p_shr)
    Rr_nc = (float(p_shr[okr].mean()) - float(p_int[okr].mean())) / gmed
    print("\n" + "=" * 100)
    print("SIM ONLY: REAL S/N vs the PROXY at the same MILD keep-fractions (kept out of m_flow)")
    print("=" * 100)
    print(f"  {'keep':>7} {'real':>11} {'proxy':>11} {'gap':>10}")
    for kf in args.sn_keeps:
        tr = float(np.quantile(sn_0[okr], 1.0 - kf))
        pr = okr & (sn_0 > tr), okr & (sn_g > tr)
        Rr = (float(p_shr[pr[1]].mean()) - float(p_int[pr[0]].mean())) / gmed
        tp = float(np.quantile(u0[okr], 1.0 - kf))
        pp = okr & (u0 > tp), okr & ((args.sn_a * base[MAG_COL + "_g"].to_numpy(float)
                                      + args.sn_b * np.log10(np.maximum(
                                          base[SIZE_COL + "_g"].to_numpy(float), 1e-6))) > tp)
        Rp = (float(p_shr[pp[1]].mean()) - float(p_int[pp[0]].mean())) / gmed
        mr, mp = (Rr / Rr_nc - 1) * 100, (Rp / Rr_nc - 1) * 100
        print(f"  {kf:>7.2f} {mr:>+10.3f}% {mp:>+10.3f}% {mp-mr:>+10.3f}")

    print("\n  m_sel  = R_sim(cut)/R_sim(no cut) - 1   (the shift the SIM has)")
    print("  m_flow = R_model(cut)/R_sim(cut) - 1    (the bias the MODEL has predicting it)")
    print("  Intrinsic shapes on BOTH sides -> no measurement-error term and NO missing blend term,")
    print("  so m_flow here is a clean model bias, unlike the constgold model column (see 30u).")
    print("NEARDOMAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
