"""Estimator-responsivity match + firewall-clean R_blend for the reframe deliverable (loop cont.114).

The deliverable m is assembled on constgold (measured with constgold's `measured_e`), but the flow's
R_self is trained on the ngmix half-shear leg (measured with `measured_ngmix_g`). Before those two can
be compared we must know the ngmix and constgold shape estimators have the SAME shear responsivity --
otherwise the flow's R_self is off by an unknown scale in constgold units. The two intrinsic-e
distributions differ ~7% in std (0.355 vs 0.33), which does NOT tell us the response scale.

Clean test using ISOLATED primaries (no neighbour -> R_blend=0 -> pure self-response), true-cut:
  * ngmix isolated R_self  = <e_ngmix . ghat_p>/g   on gp>0 & ~neighboured rows   (ngmix units)
  * constgold isolated R1  = <(e_+ - e_-).ghat>/2g   on ~neighboured primaries    (constgold units)
  If these AGREE, the estimators share responsivity and the flow R_self is valid on constgold.

Then the coherent constgold response on BLENDED objects gives R_blend firewall-cleanly:
  * constgold coherent(blended) = R_self(blended) + R_blend   [constgold units]
  * ngmix R_self(blended)       = <e_ngmix.ghat_p>/g on gp>0 & neighboured & gs=0 [ngmix units]
  => R_blend  ~  constgold_coherent(blended) - ngmix_R_self(blended)   (unit-matched if the isolated
     test passes). This is the number the emulator must reproduce; constgold is read for VALIDATION only.

Reads the ngmix half-shear leg (train-safe) + constgold (held-out, eval-only). Trains nothing.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc

SBSI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SBSI_ROOT not in sys.path:
    sys.path.insert(0, SBSI_ROOT)
from scripts.eval_joint_triad import _stream as cg_stream, _ghat as cg_ghat  # noqa: E402
from sbs_shear.measurement_model import add_measurement_target_features  # noqa: E402

MAG_EDGES = np.array([18.0, 24.0, 25.0, 26.0])
SIZE_EDGES = np.array([0.30, 0.38, 0.50, 1.50])


def _boot_ratio(ssum, scnt, g, nb=300, seed=12345):
    """Case-level bootstrap std of (sum/cnt/g)."""
    rng = np.random.RandomState(seed)
    n = len(ssum)
    draws = np.empty(nb)
    for k in range(nb):
        idx = rng.randint(0, n, n)
        c = scnt[idx].sum()
        draws[k] = ssum[idx].sum() / c / g if c > 0 else np.nan
    return float(np.nanstd(draws))


def _R_case(proj, case, g):
    """R=<proj>/g with case-level bootstrap error."""
    uc, inv = np.unique(case, return_inverse=True)
    ssum = np.bincount(inv, weights=proj, minlength=len(uc))
    scnt = np.bincount(inv, minlength=len(uc)).astype(float)
    R = ssum.sum() / scnt.sum() / g
    return R, _boot_ratio(ssum, scnt, g), int(scnt.sum())


def _binned_R(proj, mag, size, case, g):
    out = {}
    for tag, edges, val in [("mag", MAG_EDGES, mag), ("size", SIZE_EDGES, size)]:
        rows = []
        for i in range(len(edges) - 1):
            m = (val >= edges[i]) & (val < edges[i + 1])
            n = int(m.sum())
            R = proj[m].sum() / n / g if n else np.nan
            rows.append((f"{tag}[{edges[i]:.2f},{edges[i+1]:.2f})", R, n))
        out[tag] = rows
    return out


# ---------------- ngmix half-shear reader ----------------
def ngmix_stream(path, true_cut, max_rows, e_cols):
    re_min, mag_max = true_cut
    cols = [*e_cols, "gamma1_input_p", "gamma2_input_p", "gamma1_input_s", "gamma2_input_s",
            "neighbored", "detected", "Re_input_p", "r_input_p", "case",
            "measured_a_image", "measured_b_image", "measured_theta_image"]  # for MOMENTS e (=constgold estimator)
    parts, nread = [], 0
    with ipc.open_file(path) as r:
        avail = set(r.schema.names)
        use = [c for c in cols if c in avail]
        for bi in range(r.num_record_batches):
            if max_rows and nread >= max_rows:
                break
            b = pa.Table.from_batches([r.get_batch(bi)]).select(use).to_pandas()
            nread += len(b)
            if "detected" in b.columns:
                b = b[b["detected"].astype(bool)]
            b = b[(b["Re_input_p"] > re_min) & (b["r_input_p"] < mag_max)]
            if len(b):
                parts.append(b.reset_index(drop=True))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=use)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ngmix-cat",
                    default="/project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_ngmix_g0.05_val.feather")
    ap.add_argument("--e-cols", nargs=2, default=["measured_ngmix_g1", "measured_ngmix_g2"])
    ap.add_argument("--ngmix-max-rows", type=int, default=0)
    ap.add_argument("--cg-max-case", type=int, default=None)
    ap.add_argument("--true-re-min", type=float, default=0.3)
    ap.add_argument("--true-mag-max", type=float, default=26.0)
    ap.add_argument("--output", default="/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/estimator_match.npz")
    args = ap.parse_args()
    true_cut = (args.true_re_min, args.true_mag_max)
    t0 = time.time()
    res = {}

    # =============== NGMIX half-shear: isolated & blended R_self ===============
    print(f"[ngmix] {args.ngmix_cat}\n  e-cols={args.e_cols} true_cut=Re>{true_cut[0]}&mag<{true_cut[1]}", flush=True)
    dn = ngmix_stream(args.ngmix_cat, true_cut, args.ngmix_max_rows, args.e_cols)
    e1 = dn[args.e_cols[0]].to_numpy(float); e2 = dn[args.e_cols[1]].to_numpy(float)
    gp = dn[["gamma1_input_p", "gamma2_input_p"]].to_numpy(float)
    gs = np.hypot(dn["gamma1_input_s"].to_numpy(float), dn["gamma2_input_s"].to_numpy(float))
    gpm = np.hypot(gp[:, 0], gp[:, 1])
    nbr = dn["neighbored"].astype(bool).to_numpy()
    magn = dn["r_input_p"].to_numpy(float); sizen = dn["Re_input_p"].to_numpy(float); casen = dn["case"].to_numpy(np.int64)
    fin = np.isfinite(e1) & np.isfinite(e2)
    g_ng = float(np.median(gpm[gpm > 1e-6]))
    sheared = fin & (gpm > 1e-6)
    gh1 = np.where(gpm > 1e-6, gp[:, 0] / np.where(gpm > 1e-6, gpm, 1), 0.0)
    gh2 = np.where(gpm > 1e-6, gp[:, 1] / np.where(gpm > 1e-6, gpm, 1), 0.0)
    projn = e1 * gh1 + e2 * gh2

    m_iso = sheared & (~nbr)                                # isolated primary
    m_bl = sheared & nbr & (gs < 1e-6)                      # neighbour present, neighbour UNsheared
    print(f"  rows: sheared={int(sheared.sum()):,}  isolated={int(m_iso.sum()):,}  blended(gs=0)={int(m_bl.sum()):,}", flush=True)
    for name, m in [("ngmix_iso_Rself", m_iso), ("ngmix_bl_Rself", m_bl)]:
        R, se, n = _R_case(projn[m], casen[m], g_ng)
        res[name] = R; res[name + "_err"] = se; res[name + "_n"] = n
        print(f"  {name:20s}: R = {R:+.4f} +- {se:.4f}  (N={n:,})", flush=True)
    bn_iso = _binned_R(projn[m_iso], magn[m_iso], sizen[m_iso], casen[m_iso], g_ng)
    bn_bl = _binned_R(projn[m_bl], magn[m_bl], sizen[m_bl], casen[m_bl], g_ng)

    # =============== MOMENTS decomposition (constgold's OWN estimator, available in ALL rows) ===============
    # measured_e_image = SExtractor a/b/theta ellipticity == constgold's measured_e estimator (no cross-estimator
    # conversion). Available in blend rows too, so R_blend is measured DIRECTLY on the half-shear leg, firewall-clean.
    res["moments"] = 1.0
    have_mom = {"measured_a_image", "measured_b_image", "measured_theta_image"}.issubset(dn.columns)
    if have_mom:
        mom = add_measurement_target_features(dn.copy())
        me1 = mom["measured_e1_image"].to_numpy(float); me2 = mom["measured_e2_image"].to_numpy(float)
        finm = np.isfinite(me1) & np.isfinite(me2)
        gsr = dn[["gamma1_input_s", "gamma2_input_s"]].to_numpy(float)
        ghs1 = np.where(gs > 1e-6, gsr[:, 0] / np.where(gs > 1e-6, gs, 1), 0.0)
        ghs2 = np.where(gs > 1e-6, gsr[:, 1] / np.where(gs > 1e-6, gs, 1), 0.0)
        proj_self_m = me1 * gh1 + me2 * gh2            # response to PRIMARY shear
        proj_blend_m = me1 * ghs1 + me2 * ghs2         # response to NEIGHBOUR shear
        g_gs = float(np.median(gs[gs > 1e-6]))
        mm_iso = finm & (gpm > 1e-6) & (~nbr)                       # isolated self
        mm_bl = finm & (gpm > 1e-6) & nbr & (gs < 1e-6)            # blended self (neighbour unsheared)
        mm_blend = finm & (gpm < 1e-6) & (gs > 1e-6) & nbr        # neighbour-sheared -> R_blend
        print("\n[moments] measured_e_image (=constgold estimator) on half-shear leg", flush=True)
        print(f"  rows: iso={int(mm_iso.sum()):,}  self_bl={int(mm_bl.sum()):,}  blend={int(mm_blend.sum()):,}", flush=True)
        for name, m, proj, gg, cc in [("mom_iso_Rself", mm_iso, proj_self_m, g_ng, casen),
                                      ("mom_bl_Rself", mm_bl, proj_self_m, g_ng, casen),
                                      ("mom_Rblend", mm_blend, proj_blend_m, g_gs, casen)]:
            R, se, n = _R_case(proj[m], cc[m], gg)
            res[name] = R; res[name + "_err"] = se; res[name + "_n"] = n
            print(f"  {name:16s}: R = {R:+.4f} +- {se:.4f}  (N={n:,})", flush=True)
        mom_bn_self = _binned_R(proj_self_m[mm_bl], magn[mm_bl], sizen[mm_bl], casen[mm_bl], g_ng)
        mom_bn_blend = _binned_R(proj_blend_m[mm_blend], magn[mm_blend], sizen[mm_blend], casen[mm_blend], g_gs)
    else:
        print("\n[moments] a/b/theta columns absent -> skipping moments decomposition", flush=True)

    # =============== CONSTGOLD: isolated & coherent R1 ===============
    CDIR = "/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant"
    rcols = ["measured_e1_plus", "measured_e2_plus", "measured_e1_minus", "measured_e2_minus",
             "applied_g1", "applied_g2", "axis_ratio_input_p", "position_angle_input_p",
             "r_input_p", "Re_input_p", "neighbored", "distance", "input_index", "case"]
    print(f"\n[constgold] {CDIR}/constant_response_catalogue_train.feather", flush=True)
    Rc = cg_stream(CDIR + "/constant_response_catalogue_train.feather", rcols, None, args.cg_max_case, true_cut)
    gR, ghc1, ghc2 = cg_ghat(Rc)
    e1p = Rc["measured_e1_plus"].to_numpy(float); e2p = Rc["measured_e2_plus"].to_numpy(float)
    e1m = Rc["measured_e1_minus"].to_numpy(float); e2m = Rc["measured_e2_minus"].to_numpy(float)
    proj1 = 0.5 * ((e1p - e1m) * ghc1 + (e2p - e2m) * ghc2)           # /g -> R1 (coherent)
    magc = Rc["r_input_p"].to_numpy(float); sizec = Rc["Re_input_p"].to_numpy(float); casec = Rc["case"].to_numpy(np.int64)
    nbrc = Rc["neighbored"].astype(bool).to_numpy()
    print(f"  loaded={len(Rc):,} cases={Rc['case'].nunique()}  isolated={int((~nbrc).sum()):,} blended={int(nbrc.sum()):,}", flush=True)
    for name, m in [("cg_iso_R1", ~nbrc), ("cg_bl_R1", nbrc), ("cg_all_R1", np.ones(len(Rc), bool))]:
        R, se, n = _R_case(proj1[m], casec[m], gR)
        res[name] = R; res[name + "_err"] = se; res[name + "_n"] = n
        print(f"  {name:20s}: R = {R:+.4f} +- {se:.4f}  (N={n:,})", flush=True)
    cg_iso_bin = _binned_R(proj1[~nbrc], magc[~nbrc], sizec[~nbrc], casec[~nbrc], gR)
    cg_bl_bin = _binned_R(proj1[nbrc], magc[nbrc], sizec[nbrc], casec[nbrc], gR)

    # =============== VERDICT ===============
    print("\n===== ESTIMATOR MATCH (isolated self-response) =====", flush=True)
    ratio = res["cg_iso_R1"] / res["ngmix_iso_Rself"] if res["ngmix_iso_Rself"] else np.nan
    print(f"  ngmix isolated R_self  = {res['ngmix_iso_Rself']:+.4f} +- {res['ngmix_iso_Rself_err']:.4f}")
    print(f"  constgold isolated R1  = {res['cg_iso_R1']:+.4f} +- {res['cg_iso_R1_err']:.4f}")
    print(f"  responsivity ratio cg/ngmix = {ratio:.4f}   (1.000 => estimators matched)")
    res["resp_ratio_iso"] = ratio

    print("\n===== IMPLIED R_blend (firewall: constgold read for validation only) =====", flush=True)
    # unit-CONSISTENT: bridge the ngmix R_self into constgold units BEFORE subtracting from constgold coherent.
    ng_bl_cgunits = res["ngmix_bl_Rself"] * ratio
    Rblend = res["cg_bl_R1"] - ng_bl_cgunits
    print(f"  constgold coherent (blended)      = {res['cg_bl_R1']:+.4f}  [constgold units]")
    print(f"  ngmix R_self(blended) x bridge     = {res['ngmix_bl_Rself']:+.4f} x {ratio:.4f} = {ng_bl_cgunits:+.4f}  [constgold units]")
    print(f"  => R_blend (constgold units)       = {Rblend:+.4f}   (target the emulator must hit, in constgold units)")
    Rblend_mixed = res["cg_bl_R1"] - res["ngmix_bl_Rself"]
    print(f"     (unit-INCONSISTENT cg_bl-ng_bl  = {Rblend_mixed:+.4f}, shown for reference only)")
    res["R_blend_implied"] = Rblend; res["R_blend_mixed"] = Rblend_mixed
    print(f"  neighbour dilution of R_self (ngmix iso-bl) = {res['ngmix_iso_Rself']-res['ngmix_bl_Rself']:+.4f}")

    if have_mom:
        print("\n===== MOMENTS CLOSURE (same estimator as constgold, no conversion) =====", flush=True)
        msum = res["mom_bl_Rself"] + res["mom_Rblend"]
        print(f"  moments R_self(blended)   = {res['mom_bl_Rself']:+.4f} +- {res['mom_bl_Rself_err']:.4f}")
        print(f"  moments R_blend           = {res['mom_Rblend']:+.4f} +- {res['mom_Rblend_err']:.4f}")
        print(f"  moments R_self + R_blend  = {msum:+.4f}")
        print(f"  constgold coherent(bl) R1 = {res['cg_bl_R1']:+.4f} +- {res['cg_bl_R1_err']:.4f}")
        print(f"  closure ratio (sum/cg_bl) = {msum/res['cg_bl_R1']:.4f}   (1.000 => decomposition closes in constgold's estimator)")
        print(f"  moments iso R_self        = {res['mom_iso_Rself']:+.4f}  vs constgold iso R1 = {res['cg_iso_R1']:+.4f}"
              f"  (ratio {res['cg_iso_R1']/res['mom_iso_Rself']:.4f})")
        res["mom_closure_ratio"] = msum / res["cg_bl_R1"]
        print("\n  per-bin moments: R_self  R_blend  sum  vs cg_bl_R1", flush=True)
        print(f"  {'bin':22s} {'m_self':>8} {'m_blend':>8} {'sum':>8} {'cg_bl':>8}")
        for key in ("mag", "size"):
            for (lab, rs, _), (_, rb, _), (_, rcb, _) in zip(mom_bn_self[key], mom_bn_blend[key], cg_bl_bin[key]):
                print(f"  {lab:22s} {rs:+8.4f} {rb:+8.4f} {rs+rb:+8.4f} {rcb:+8.4f}")

    print("\n----- per-bin: ngmix R_self(iso/bl) vs constgold R1(iso/bl) -----", flush=True)
    print(f"  {'bin':22s} {'ng_iso':>8} {'cg_iso':>8} | {'ng_bl':>8} {'cg_bl':>8} {'Rblend':>8}")
    for key in ("mag", "size"):
        for (lab, rni, _), (_, rci, _), (_, rnb, _), (_, rcb, _) in zip(
                bn_iso[key], cg_iso_bin[key], bn_bl[key], cg_bl_bin[key]):
            print(f"  {lab:22s} {rni:+8.4f} {rci:+8.4f} | {rnb:+8.4f} {rcb:+8.4f} {rcb-rnb:+8.4f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.savez(args.output, **{k: (v if np.isscalar(v) else v) for k, v in res.items()})
    print(f"\nsaved {args.output}   ({time.time()-t0:.1f}s)", flush=True)
    print("ESTIMATOR_MATCH_DONE", flush=True)


if __name__ == "__main__":
    main()
