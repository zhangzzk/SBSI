"""constgold selection table at NEAR-DOMAIN cuts, on REAL measured mag/size -- no proxy anywhere.

WHAT CHANGED AND WHY IT MATTERS. constgold previously stored only ONE per-leg measured quantity
(S/N), so the model had to cut on a mag+size PROXY for S/N while the sim cut on the real thing, and
thresholds had to be matched by KEEP-FRACTION rather than by value. That proxy carried its own error
(-0.16 to -0.26 pts at mild cuts). The catalogue now carries `measured_mag_auto_{plus,minus}` and
`measured_flux_radius_{plus,minus}` (job 15366166, verified to reproduce all 40 pre-existing columns
exactly), so for magnitude and size cuts BOTH SIDES CUT ON THE SAME REAL QUANTITY AT THE SAME
ABSOLUTE THRESHOLD. No proxy, no quantile matching.

Rows marked (*) are the exception: real-S/N cuts, which the flow still cannot represent because it
does not output FLUX_AUTO/FLUXERR_AUTO. Those keep the proxy on the model side and are flagged.

COLUMNS (owner's spec: the old (2) sheared-intrinsic column dropped, m_flow added)
  (1) pure sel  R_unsheared(cut) / R_sheared(no cut). The unsheared column projects the SAME raw
                shape in both legs, so its shape response is identically 0 and its no-cut R is 0 --
                whatever survives is the pure MOVING-BOUNDARY term. Normalised by the sheared no-cut
                R because its own would be 0/0.
  (3) measured  R_meas(cut)/R_meas(no cut) - 1, the shift the SIM has in measured shapes.
  (4) MODEL m   R_model(cut)/R_model(no cut) - 1, the flow's counterpart of (3).
  m_flow        R_model(cut)/R_meas(cut) - 1, the residual bias.

READ m_flow WITH THE BLEND CAVEAT (see WORKLOG 30u). This reports the FULL population, where column
(3) contains the neighbour response while the flow's R is self-response only. So m_flow here is NOT a
pure model error -- it inherits a missing term worth ~15% of R at no cut, which the certified
pipeline supplies from a separate blend emulator. Judge the flow on how m_flow VARIES with the cut,
not on its absolute value.

FIREWALL: constgold is EVALUATION only. Nothing is trained, fitted or selected here.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pyarrow.feather as pf
import torch

SBSI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (SBSI_ROOT, os.path.join(SBSI_ROOT, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sbs_shear.measurement_model import load_measurement_model  # noqa: E402
from sbs_shear.preprocessing import (  # noqa: E402
    DEFAULT_SELECTION_CUTS, rescale, source_select_selection,
)
from sbs_shear.coordinates import ellipticity_from_axis_ratio_angle  # noqa: E402
from sbs_shear.shear_map import apply_shear_to_ellipticity  # noqa: E402

NEED = ["case", "input_index", "neighbored", "distance", "polarization_angle",
        "Re_input_p", "Re_input_s", "axis_ratio_input_p", "axis_ratio_input_s",
        "position_angle_input_p", "position_angle_input_s", "r_input_p", "r_input_s",
        "redshift_input_p", "redshift_input_s", "sersic_n_input_p", "sersic_n_input_s",
        "applied_g1", "applied_g2", "shear_magnitude", "shear_angle",
        "measured_e1_plus", "measured_e2_plus", "measured_e1_minus", "measured_e2_minus",
        "et_plus", "S/N_plus", "S/N_minus",
        "measured_mag_auto_plus", "measured_mag_auto_minus",
        "measured_flux_radius_plus", "measured_flux_radius_minus"]

CG = ("/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/"
      "constant_response_catalogue_train.feather")
CROWD = "/home/z/Zekang.Zhang/SBSI/results/crowd_flux_conc_c0-199.feather"
RK = dict(pixel_rms=0.312, pixel_size=0.2, zero_mag=30.0, psf_fwhm=0.73, moffat_beta=2.224)
PX = 0.2


def leg_avg(ap, am, g):
    return (ap - am) / (2.0 * g)


@torch.no_grad()
def model_selected(bundle, df, g, gh1, gh2, intr, cuts, groups, n_samples, batch_size,
                   seed, device, sign=1.0, chunk=200_000):
    """Score the flow once; accumulate selected means for every (group, cut) pair.

    Cuts are given as ABSOLUTE thresholds on the flow's own sampled measured magnitude and size --
    the same physical quantities and the same numbers the sim uses. Only the (*) S/N rows fall back
    to the mag+size proxy, which the flow can form from its outputs.
    """
    names = bundle.target_transform.target_names
    i1, i2 = names.index("measured_ngmix_g1"), names.index("measured_ngmix_g2")
    imag, ilr = names.index("measured_mag_auto"), names.index("measured_log_flux_radius")
    ln10 = np.log(10.0)
    n = len(df)
    keys = ["__nocut__"] + [c["name"] for c in cuts]
    acc = {gname: {k: [[0.0, 0], [0.0, 0]] for k in keys} for gname in groups}

    for li, s in ((0, +g), (1, -g)):
        for lo in range(0, n, chunk):
            hi = min(lo + chunk, n)
            fr = df.iloc[lo:hi].reset_index(drop=True).copy()
            e1s, e2s = apply_shear_to_ellipticity(intr[0][lo:hi], intr[1][lo:hi],
                                                  s * gh1[lo:hi], s * gh2[lo:hi])
            fr["e1_input_rot0_p"], fr["e2_input_rot0_p"] = e1s, e2s
            fr = rescale(fr, **RK)
            torch.manual_seed(seed)                      # CRN: identical latents in both legs
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            d = bundle.sample(fr, n_samples=n_samples, batch_size=batch_size)
            proj = sign * (d[:, :, i1] * gh1[lo:hi, None] + d[:, :, i2] * gh2[lo:hi, None])
            mag = d[:, :, imag]
            # Compare size in LOG space. Exponentiating first overflowed to +inf on extreme flow
            # draws, and `inf > thr` is True, so those draws were KEPT by every size cut -- inflating
            # the model's selected set on the size rows only (the sim side was never affected).
            # `log_size > log(thr)` is exactly equivalent for thr > 0 and cannot overflow.
            logsz = d[:, :, ilr]                         # flow emits NATURAL log of radius in px
            fin = np.isfinite(proj)
            for gname, gmask in groups.items():
                gm = gmask[lo:hi][:, None]
                base = fin & gm
                acc[gname]["__nocut__"][li][0] += float(np.where(base, proj, 0.0).sum())
                acc[gname]["__nocut__"][li][1] += int(base.sum())
                for c in cuts:
                    pm = base
                    for sc in c["conds"]:
                        if sc["var"] == "mag":
                            xv = mag
                        elif sc["var"] == "size":
                            xv = logsz                   # threshold already stored as log(px)
                        else:                            # proxy S/N, (*) rows only
                            xv = sc["a"] * mag + sc["b"] * (logsz / ln10)
                        pm = pm & ((xv > sc["thr"]) if sc["keep_high"] else (xv < sc["thr"]))
                    acc[gname][c["name"]][li][0] += float(np.where(pm, proj, 0.0).sum())
                    acc[gname][c["name"]][li][1] += int(pm.sum())

    out = {}
    for gname in groups:
        out[gname] = {}
        for k in keys:
            a = acc[gname][k]
            mp = a[0][0] / a[0][1] if a[0][1] else np.nan
            mm = a[1][0] / a[1][1] if a[1][1] else np.nan
            out[gname][k] = leg_avg(mp, mm, g)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--cat", default=CG)
    ap.add_argument("--crowd", default=CROWD)
    ap.add_argument("--min-case", type=int, default=40)
    ap.add_argument("--max-rows", type=int, default=4_000_000)
    ap.add_argument("--mag-cuts", type=float, nargs="+", default=[26.0, 25.5, 25.0])
    ap.add_argument("--size-cuts", type=float, nargs="+", default=[0.50, 0.55, 0.60, 0.70])
    ap.add_argument("--sn-keeps", type=float, nargs="+", default=[0.98, 0.95, 0.90])
    ap.add_argument("--sn-a", type=float, default=-0.3676)
    ap.add_argument("--sn-b", type=float, default=-0.7736)
    ap.add_argument("--n-samples", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=16384)
    ap.add_argument("--flow-seed", type=int, default=12345)
    ap.add_argument("--dom-mag-max", type=float, default=26.0)
    ap.add_argument("--dom-re-min", type=float, default=0.3)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    df = pf.read_table(args.cat, columns=NEED, memory_map=True).to_pandas()
    n_raw = len(df)
    df = df[df["case"] >= args.min_case].reset_index(drop=True)
    print(f"constgold(+meas): {n_raw:,} -> {len(df):,} with case>={args.min_case} "
          f"({time.time()-t0:.0f}s)", flush=True)
    if len(df) == 0:
        raise SystemExit("REFUSING: no rows after the case cut.")
    e1i, e2i = ellipticity_from_axis_ratio_angle(df["axis_ratio_input_p"].to_numpy(float),
                                                df["position_angle_input_p"].to_numpy(float))
    df["e1_input_rot0_p"], df["e2_input_rot0_p"] = e1i, e2i
    df["gamma1_input_p"], df["gamma2_input_p"] = 0.0, 0.0
    df = source_select_selection(df, cuts=DEFAULT_SELECTION_CUTS).reset_index(drop=True)
    # DOMAIN CUT is mandatory: the dom6x6 flow was trained with primary_mag_max=26.0 /
    # primary_re_min=0.3, and scoring it outside that box put R_model(no cut) at +0.173 vs ~0.29,
    # inflating every model entry ~4x (run 15365425). Sim and model must share one population.
    dom = (df["r_input_p"].to_numpy(float) < args.dom_mag_max) & \
          (df["Re_input_p"].to_numpy(float) > args.dom_re_min)
    df = df[dom].reset_index(drop=True)
    print(f"  after selection + DOMAIN cut: {len(df):,}", flush=True)
    if args.max_rows and len(df) > args.max_rows:
        sel = np.random.default_rng(0).choice(len(df), size=args.max_rows, replace=False)
        sel.sort()
        df = df.iloc[sel].reset_index(drop=True)
        print(f"  subsampled to {len(df):,} (seed 0)", flush=True)

    cf = pf.read_table(args.crowd).to_pandas()
    fcols = [c for c in ("nbr_flux_near", "nbr_flux_far", "nbr_flux_max") if c in cf.columns]
    mm = df[["case", "input_index"]].merge(cf[["case", "input_index", *fcols]],
                                           on=["case", "input_index"], how="left")
    for c in fcols:
        df[c] = mm[c].to_numpy(float)

    g = float(np.median(df.shear_magnitude.to_numpy(float)))
    sa = df.shear_angle.to_numpy(float)
    c2, s2 = np.cos(2 * sa), np.sin(2 * sa)
    me1p, me2p = df.measured_e1_plus.to_numpy(float), df.measured_e2_plus.to_numpy(float)
    me1m, me2m = df.measured_e1_minus.to_numpy(float), df.measured_e2_minus.to_numpy(float)
    myp, st = me1p * c2 + me2p * s2, df.et_plus.to_numpy(float)
    okp = np.isfinite(myp) & np.isfinite(st)
    sign = 1.0 if np.nanmean((myp * st)[okp]) > 0 else -1.0
    proj = lambda a, b: sign * (a * c2 + b * s2)

    i1, i2 = df["e1_input_rot0_p"].to_numpy(float), df["e2_input_rot0_p"].to_numpy(float)
    gh1, gh2 = c2, s2
    e1p, e2p = apply_shear_to_ellipticity(i1, i2, g * gh1, g * gh2)
    e1m, e2m = apply_shear_to_ellipticity(i1, i2, -g * gh1, -g * gh2)
    kinds = [("unsheared", proj(i1, i2), proj(i1, i2)),
             ("sheared", proj(e1p, e2p), proj(e1m, e2m)),
             ("measured", proj(me1p, me2p), proj(me1m, me2m))]

    magp = df["measured_mag_auto_plus"].to_numpy(float)
    magm = df["measured_mag_auto_minus"].to_numpy(float)
    szp = df["measured_flux_radius_plus"].to_numpy(float)
    szm = df["measured_flux_radius_minus"].to_numpy(float)
    snp, snm = df["S/N_plus"].to_numpy(float), df["S/N_minus"].to_numpy(float)
    fin = np.isfinite(snp) & np.isfinite(snm) & np.isfinite(magp) & np.isfinite(magm) \
        & np.isfinite(szp) & np.isfinite(szm)
    for _, a, b in kinds:
        fin &= np.isfinite(a) & np.isfinite(b)
    print(f"finite: {int(fin.sum()):,}   g={g:.4f}", flush=True)
    groups = {"ALL": fin}

    # ---- cut list: sim (per-leg arrays) and model (absolute thresholds) built together ----------
    up = args.sn_a * magp + args.sn_b * np.log10(np.maximum(szp, 1e-6))
    cuts = []
    for c in args.mag_cuts:
        cuts.append(dict(name=f"mag<{c:g}", proxy=False,
                         sim=[(magp, magm, c, False)],
                         conds=[dict(var="mag", thr=float(c), keep_high=False)]))
    for a in args.size_cuts:
        px = a / PX
        cuts.append(dict(name=f'R>{a:.2f}"', proxy=False,
                         sim=[(szp, szm, px, True)],
                         conds=[dict(var="size", thr=float(np.log(px)), keep_high=True)]))
    for mc, sc_ in ((26.0, 0.55), (25.5, 0.55)):
        cuts.append(dict(name=f'mag<{mc:g} & R>{sc_:.2f}"', proxy=False,
                         sim=[(magp, magm, mc, False), (szp, szm, sc_ / PX, True)],
                         conds=[dict(var="mag", thr=float(mc), keep_high=False),
                                dict(var="size", thr=float(np.log(sc_ / PX)), keep_high=True)]))
    for kf in args.sn_keeps:
        thr = float(np.quantile(snp[fin], 1.0 - kf))
        # (*) real S/N on the sim; the flow cannot form it, so the MODEL uses the mag+size proxy at
        # the matching quantile of its own proxy distribution -- the only proxy left in this table.
        pthr = float(np.quantile(up[fin], 1.0 - kf))
        cuts.append(dict(name=f"S/N>{thr:.1f} (*)", proxy=True,
                         sim=[(snp, snm, thr, True)],
                         conds=[dict(var="proxy", a=args.sn_a, b=args.sn_b, thr=pthr,
                                     keep_high=True)]))

    # ---- sim ------------------------------------------------------------------------------------
    sim = {gn: {} for gn in groups}
    for gn, gm in groups.items():
        sim[gn]["__nocut__"] = {k: leg_avg(float(a[gm].mean()), float(b[gm].mean()), g)
                                for k, a, b in kinds}
        for c in cuts:
            pp, pm = gm.copy(), gm.copy()
            for xp, xm, thr, kh in c["sim"]:
                pp &= (xp > thr) if kh else (xp < thr)
                pm &= (xm > thr) if kh else (xm < thr)
            sim[gn][c["name"]] = {k: leg_avg(float(a[pp].mean()), float(b[pm].mean()), g)
                                  for k, a, b in kinds}
            sim[gn][c["name"]]["keep"] = 0.5 * (pp.sum() + pm.sum()) / max(gm.sum(), 1)

    # ---- model ----------------------------------------------------------------------------------
    per = []
    for ck in args.ckpt:
        bundle = load_measurement_model(ck, device=device)
        per.append(model_selected(bundle, df, g, gh1, gh2, (i1, i2), cuts, groups,
                                  args.n_samples, args.batch_size, args.flow_seed, device,
                                  sign=sign))
        print(f"  scored {os.path.basename(ck)} ({time.time()-t0:.0f}s)", flush=True)
    keys = ["__nocut__"] + [c["name"] for c in cuts]
    mod = {gn: {k: float(np.mean([p[gn][k] for p in per])) for k in keys} for gn in groups}
    sem = {gn: {k: (float(np.std([p[gn][k] for p in per], ddof=1) / np.sqrt(len(per)))
                    if len(per) > 1 else np.nan) for k in keys} for gn in groups}

    for gn in groups:
        R0 = sim[gn]["__nocut__"]
        M0 = mod[gn]["__nocut__"]
        print("\n" + "=" * 104)
        print(f"CONSTGOLD SELECTION, NEAR-DOMAIN CUTS -- {gn}")
        print("=" * 104)
        print(f"  sim R(no cut): unsheared={R0['unsheared']:+.5f} sheared={R0['sheared']:+.5f} "
              f"measured={R0['measured']:+.5f}   model R(no cut)={M0:+.5f}")
        print(f"\n  {'cut':>22} {'keep':>6} | {'(1) pure sel':>13} {'(3) measured':>13} "
              f"{'(4) MODEL m':>16} | {'m_flow':>10}")
        for c in cuts:
            k = c["name"]
            s = sim[gn][k]
            c1 = s["unsheared"] / R0["sheared"] * 100.0
            c3 = (s["measured"] / R0["measured"] - 1.0) * 100.0
            c4 = (mod[gn][k] / M0 - 1.0) * 100.0
            mf = (mod[gn][k] / s["measured"] - 1.0) * 100.0
            e = sem[gn][k] / abs(M0) * 100.0 if np.isfinite(sem[gn][k]) else np.nan
            print(f"  {k:>22} {s['keep']:>6.3f} | {c1:>+12.3f}% {c3:>+12.3f}% "
                  f"{c4:>+11.3f} +- {e:<.3f} | {mf:>+9.3f}%")

    print("\n  (1) pure sel = R_unsheared(cut)/R_sheared(no cut): same raw shape both legs, so the")
    print("      shape response is 0 and this is the pure moving-boundary term.")
    print("  (3) measured = R_meas(cut)/R_meas(no cut) - 1     (4) MODEL m = same ratio for the flow")
    print("  m_flow = R_model(cut)/R_meas(cut) - 1  -- the residual bias.")
    print("  Rows WITHOUT (*) cut sim and model on the SAME real measured quantity at the SAME")
    print("  absolute threshold: no proxy, no quantile matching. (*) rows are real S/N, which the")
    print("  flow cannot form (no FLUX_AUTO/FLUXERR_AUTO output), so those keep the mag+size proxy.")
    print("CG_NEARDOMAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
