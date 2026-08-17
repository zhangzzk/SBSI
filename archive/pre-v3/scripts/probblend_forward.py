"""Probabilistic blending — forward model for the undetected-neighbour R_blend.

Two levels, per case of the main set (fs2_25876, has detection truth):

LEVEL A  (classifier as thinning kernel — still uses true neighbour positions):
  Over DETECTED primaries, compare the undetected-neighbour blend response computed with the
  HARD detection truth  Sum (1 - detected_j) f_reg   against the SOFT classifier
  Sum (1 - p_det_classifier_j) f_reg. If they agree, the trained classifier is a faithful
  neighbour-thinning kernel — the object p(undetected|nbr) needed by the forward model.

LEVEL B  (population forward model — NO true neighbour positions):
  Replace the true neighbour set by a Poisson population draw:
    - property marginal  Phi(m,s,n)  bootstrapped from the field (a fair draw from the LF),
    - surface density    rho = N_field / A  (A = bounding-box area, arcsec^2),
    - separations        theta ~ 2 theta / R_max^2  on [theta_min, R_max]  (area-uniform),
    - thinned by the classifier  (1 - p_det(nbr as primary, real primary as its neighbour, theta)).
  R_blend^undet_pop(i) = rho * pi (R_max^2 - theta_min^2) * mean_k[(1-p_det) f_reg(x_i; nbr_k, theta_k)]
  Its MEAN over primaries is compared to the truth undetected mean. A second variant injects the
  measured neighbour separation/brightness distribution (empirical clustering xi) instead of the
  Poisson area-uniform draw, to size the clustering term.
"""
import argparse, sys
import numpy as np, pandas as pd, pyarrow.feather as pf
from sbs_shear.paths import SIM_BASE as BASE

DET = f"{BASE}/detection_catalogue_train.feather"
BLEND_MODELS = "/home/z/Zekang.Zhang/blendemu/models"
COND = dict(pixel_size=0.2, zero_point=30.0, psf_fwhm=0.73, moffat_beta=2.224, pixel_rms=0.312)
TILE = "tile180.0_-0.5"
R_TOTAL = 0.462
R_MAX = 10.0        # regressor aperture (arcsec)
THETA_MIN = 0.05    # inner cutoff (arcsec); truth census has signal down to ~0.05"


def field_path(case):
    return f"{BASE}/case{case}_0.0/real0/catalogues/input/gals_info_{TILE}.feather"


def bbox_area_arcsec2(ra, dec):
    dec0 = np.deg2rad(np.median(dec))
    dra = (ra.max() - ra.min()) * np.cos(dec0) * 3600.0
    ddec = (dec.max() - dec.min()) * 3600.0
    return dra * ddec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--tag", default="lsst_r_extnbr_ho")
    ap.add_argument("--n-prim", type=int, default=30000, help="detected primaries sampled for level B")
    ap.add_argument("--n-syn", type=int, default=40, help="synthetic neighbours per primary (level B)")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    sys.path.insert(0, "/home/z/Zekang.Zhang/blendemu")
    from blendemu.inference import BlendingPredictor
    pred = BlendingPredictor.load(BLEND_MODELS, tag=args.tag, conditions=COND, device="cpu")

    det = pf.read_table(DET, columns=["case", "input_index", "detected"]).to_pandas()
    det = det[det.case.isin(args.cases)].drop_duplicates(["case", "input_index"])
    det_map = {c: dict(zip(g.input_index.to_numpy(), g.detected.to_numpy().astype(bool)))
               for c, g in det.groupby("case")}

    A_hard = dict(full=[], undet=[], det=[], undet_soft=[], det_soft=[], n=[])
    B_rows = []
    for c in args.cases:
        fld = pf.read_table(field_path(c)).to_pandas()
        fld = fld.rename(columns={col: col.replace("_input", "") for col in fld.columns})
        area = bbox_area_arcsec2(fld.RA.to_numpy(), fld.DEC.to_numpy())
        rho = len(fld) / area
        # per-pair regression on true field
        reg = pred.predict_response(fld, fld)
        pk = [x for x in reg.columns if x.startswith("index") and x.endswith("_p")][0]
        sk = [x for x in reg.columns if x.startswith("index") and x.endswith("_s")][0]
        pr = reg[[pk, sk, "Re_input_p", "r_input_p", "sersic_n_input_p",
                  "Re_input_s", "r_input_s", "sersic_n_input_s", "distance", "response"]].copy()
        pr.columns = ["ip", "is_", "Rep", "rp", "np_", "Res", "rs", "ns", "dist", "resp"]
        # soft detection prob per index (classifier on true field)
        dfd = pred.predict_detection(fld, warn_extrapolation=False)
        # dfd rows align to fld rows; index column is index_input_p
        ik = [x for x in dfd.columns if x.startswith("index") and x.endswith("_p")][0]
        pdet_map = dict(zip(dfd[ik].to_numpy().astype(np.int64), dfd["detection_prob"].to_numpy()))
        dm = det_map[c]
        pr["det_s"] = pr.is_.map(dm).fillna(False).astype(bool).to_numpy()
        pr["det_p"] = pr.ip.map(dm).fillna(False).astype(bool).to_numpy()
        pr["pdet_s"] = pr.is_.map(pdet_map).fillna(0.0).to_numpy()

        dp = pr[pr.det_p]            # detected primaries = observed objects
        g = dp.groupby("ip")
        full = g.resp.sum()
        undet = dp.assign(w=(1 - dp.det_s) * dp.resp).groupby("ip").w.sum().reindex(full.index).fillna(0)
        detc = full - undet
        undet_soft = dp.assign(w=(1 - dp.pdet_s) * dp.resp).groupby("ip").w.sum().reindex(full.index).fillna(0)
        det_soft = dp.assign(w=dp.pdet_s * dp.resp).groupby("ip").w.sum().reindex(full.index).fillna(0)
        for k, v in [("full", full), ("undet", undet), ("det", detc),
                     ("undet_soft", undet_soft), ("det_soft", det_soft)]:
            A_hard[k].append(v.mean())
        A_hard["n"].append(len(full))
        print(f"case{c}: rho={rho:.4f}/arcsec^2 area={area:.3e}  Ndet-pri={len(full):,}\n"
              f"   <full>={full.mean():.4f} <undet_hard>={undet.mean():.4f} "
              f"<undet_soft>={undet_soft.mean():.4f}  (soft/hard={undet_soft.mean()/undet.mean():.2f})",
              flush=True)

        # ---------- LEVEL B: CONDITIONED population forward model ----------
        # Undetected-neighbour intensity integral with NO true neighbour positions:
        #   * neighbour props drawn from the field LF `pop`; separations area-uniform (xi(theta)=1,
        #     verified Poisson field); expected count rho*pi*(Rmax^2-thmin^2).
        #   * neighbour detection thinned by (1 - p_det) with the icat2cla-faithful context
        #     (nearest of {primary@theta, Poisson field gal@d_f}, isolated NaN beyond 3").
        #   * CONDITIONING weight p_det(primary|nbr)/p_det(primary|isolated): down-weights bright
        #     close neighbours that would have destroyed the primary's own detection (design sec.5).
        R_MAX_CLA = 3.0
        fld_idx = fld["index"].to_numpy().astype(np.int64)   # 'index_input' -> 'index' after rename
        is_det_pri = pd.Series(fld_idx).map(dm).fillna(False).astype(bool).to_numpy()
        det_ip = fld_idx[is_det_pri]
        det_props = fld.loc[is_det_pri, ["Re", "r", "sersic_n"]].to_numpy()
        if len(det_ip) > args.n_prim:
            sel = rng.choice(len(det_ip), args.n_prim, replace=False)
            det_ip, det_props = det_ip[sel], det_props[sel]
        prim, prim_ip = det_props, det_ip
        npri = len(prim)
        pop = fld[["Re", "r", "sersic_n"]].to_numpy()
        prim_rep = np.repeat(prim, args.n_syn, axis=0)
        M = npri * args.n_syn
        area_factor = rho * np.pi * (R_MAX**2 - THETA_MIN**2)

        snb = pop[rng.integers(0, len(pop), size=M)]
        theta = np.sqrt(THETA_MIN**2 + rng.random(M) * (R_MAX**2 - THETA_MIN**2))
        freg = pred.predict_on_pairs(pd.DataFrame({
            "Re_input_p": prim_rep[:, 0], "r_input_p": prim_rep[:, 1], "sersic_n_input_p": prim_rep[:, 2],
            "Re_input_s": snb[:, 0], "r_input_s": snb[:, 1], "sersic_n_input_s": snb[:, 2],
            "distance": theta}), task="response", warn_extrapolation=False)["response"].to_numpy()
        # neighbour detection: faithful context
        d_f = np.sqrt(-np.log(np.clip(rng.random(M), 1e-9, 1)) / (np.pi * rho))
        fgal = pop[rng.integers(0, len(pop), size=M)]
        use_p = theta <= d_f
        near_d = np.where(use_p, theta, d_f); near_p = np.where(use_p[:, None], prim_rep, fgal)
        ctx = np.column_stack([near_p, near_d]).astype(float)
        ctx[near_d > R_MAX_CLA, :] = np.nan
        pdn = pred.predict_on_pairs(pd.DataFrame({
            "Re_input_p": snb[:, 0], "r_input_p": snb[:, 1], "sersic_n_input_p": snb[:, 2],
            "Re_input_s": ctx[:, 0], "r_input_s": ctx[:, 1], "sersic_n_input_s": ctx[:, 2],
            "distance": ctx[:, 3]}), task="detection", warn_extrapolation=False)["detection_prob"].to_numpy()
        # conditioning: primary detection given this neighbour / isolated
        pdp_nbr = pred.predict_on_pairs(pd.DataFrame({
            "Re_input_p": prim_rep[:, 0], "r_input_p": prim_rep[:, 1], "sersic_n_input_p": prim_rep[:, 2],
            "Re_input_s": snb[:, 0], "r_input_s": snb[:, 1], "sersic_n_input_s": snb[:, 2],
            "distance": np.where(theta > R_MAX_CLA, np.nan, theta)}),
            task="detection", warn_extrapolation=False)["detection_prob"].to_numpy()
        nanc = np.full(npri, np.nan)
        pdp_iso = pred.predict_on_pairs(pd.DataFrame({
            "Re_input_p": prim[:, 0], "r_input_p": prim[:, 1], "sersic_n_input_p": prim[:, 2],
            "Re_input_s": nanc, "r_input_s": nanc, "sersic_n_input_s": nanc, "distance": nanc}),
            task="detection", warn_extrapolation=False)["detection_prob"].to_numpy()
        cond_w = np.clip(pdp_nbr / np.clip(np.repeat(pdp_iso, args.n_syn), 1e-3, None), 0, 1)

        contrib = cond_w * (1 - pdn) * freg
        per_prim = contrib.reshape(npri, args.n_syn).mean(axis=1) * area_factor   # R_undet forward, per primary
        B_rows.append(dict(case=c, r_undet_fwd=float(np.mean(per_prim)),
                           r_undet_hard=float(undet.mean()), r_undet_soft=float(undet_soft.mean())))
        print(f"   [B/conditioned] <R_undet_fwd>={np.mean(per_prim):.4f}  vs hard={undet.mean():.4f}"
              f" (r={np.mean(per_prim)/undet.mean():.3f})  soft={undet_soft.mean():.4f}"
              f" (r={np.mean(per_prim)/undet_soft.mean():.3f})", flush=True)

        # ---- FULL R_blend reconstruction per primary magnitude ----
        t_full = full.reindex(prim_ip).fillna(0.0).to_numpy()
        t_det = detc.reindex(prim_ip).fillna(0.0).to_numpy()
        t_undet = undet.reindex(prim_ip).fillna(0.0).to_numpy()      # truth undetected term
        t_undet_soft = undet_soft.reindex(prim_ip).fillna(0.0).to_numpy()
        r_fwd = t_det + per_prim                                     # detected nbrs (known) + forward undet
        condw_per = cond_w.reshape(npri, args.n_syn).mean(axis=1)    # mean conditioning weight per primary
        pdp_iso_v = pdp_iso                                          # primary isolated detection prob
        rmag = prim[:, 1]
        for lo, hi in [(18, 24), (24, 25), (25, 26), (26, 27), (27, 28.1)]:
            m = (rmag >= lo) & (rmag < hi)
            if m.sum() < 200:
                continue
            print(f"      r_p {lo:.0f}-{hi:.0f}: full T={t_full[m].mean():.4f} fwd={r_fwd[m].mean():.4f}"
                  f" dm={(r_fwd[m].mean()-t_full[m].mean())/R_TOTAL:+.2%}  |  undet T_hard={t_undet[m].mean():.4f}"
                  f" T_soft={t_undet_soft[m].mean():.4f} fwd={per_prim[m].mean():.4f}"
                  f" (f/soft={per_prim[m].mean()/max(t_undet_soft[m].mean(),1e-9):.2f})"
                  f"  <condw>={condw_per[m].mean():.2f} <pdet_iso>={pdp_iso_v[m].mean():.2f}", flush=True)

    print("\n============ LEVEL A summary (classifier thinning) ============")
    for k in ["full", "undet", "det", "undet_soft", "det_soft"]:
        print(f"  <{k:>11}> = {np.mean(A_hard[k]):.4f}")
    print(f"  undet_soft/undet_hard = {np.mean(A_hard['undet_soft'])/np.mean(A_hard['undet']):.3f}  "
          f"(classifier vs detection-truth census; 1.0 = perfectly calibrated)")
    print(f"  dropping undetected nbrs entirely: delta_m ~ {np.mean(A_hard['undet'])/R_TOTAL:+.2%}")
    print("\n============ LEVEL B summary (CONDITIONED population forward model) ============")
    Bdf = pd.DataFrame(B_rows)
    rf, rh, rs = Bdf.r_undet_fwd.mean(), Bdf.r_undet_hard.mean(), Bdf.r_undet_soft.mean()
    print(f"  <R_undet_fwd>={rf:.4f}  vs hard={rh:.4f} (r={rf/rh:.3f})  vs soft={rs:.4f} (r={rf/rs:.3f})")
    print(f"  residual multiplicative bias after forward model: delta_m ~ {(rh-rf)/R_TOTAL:+.2%}")
    print(f"  (of which model geometry ~{(rs-rf)/R_TOTAL:+.2%}, classifier calibration ~{(rh-rs)/R_TOTAL:+.2%})")
    print("PROBBLEND_FWD_DONE", flush=True)


if __name__ == "__main__":
    main()
