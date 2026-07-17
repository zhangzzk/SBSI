"""Decisive per-theta-shell diagnostic: where does the population p_det context model diverge from
truth? For case 0, over detected primaries, compare the undetected blend response per separation
shell computed three ways:
  TRUTH-soft : Sum over TRUE neighbours (1 - p_det_classifier(real context)) f_reg   [target]
  primctx    : population MC, neighbour detection context = primary only (isolated beyond 3")
  faithful   : population MC, context = nearest of {primary@theta, Poisson field gal@d_f} (<3" else iso)
The shell where TRUTH sits between the two tells us the correct context model.
"""
import sys
import numpy as np, pandas as pd, pyarrow.feather as pf

BASE = "/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876"
DET = f"{BASE}/detection_catalogue_train.feather"
BLEND_MODELS = "/home/z/Zekang.Zhang/blendemu/models"
COND = dict(pixel_size=0.2, zero_point=30.0, psf_fwhm=0.73, moffat_beta=2.224, pixel_rms=0.312)
TILE = "tile180.0_-0.5"
R_MAX, THETA_MIN, R_MAX_CLA = 10.0, 0.05, 3.0
SHELLS = [0.05, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
CASE = 0
N_SYN = 60


def main():
    sys.path.insert(0, "/home/z/Zekang.Zhang/blendemu")
    from blendemu.inference import BlendingPredictor
    pred = BlendingPredictor.load(BLEND_MODELS, tag="lsst_r_extnbr_ho", conditions=COND, device="cpu")

    det = pf.read_table(DET, columns=["case", "input_index", "detected"]).to_pandas()
    det = det[det.case == CASE].drop_duplicates(["input_index"])
    dm = dict(zip(det.input_index.to_numpy(), det.detected.to_numpy().astype(bool)))

    fld = pf.read_table(f"{BASE}/case{CASE}_0.0/real0/catalogues/input/gals_info_{TILE}.feather").to_pandas()
    fld = fld.rename(columns={col: col.replace("_input", "") for col in fld.columns})
    fld_idx = fld["index"].to_numpy().astype(np.int64)
    ra0, dec0 = np.median(fld.RA), np.median(fld.DEC)
    area = ((fld.RA.max()-fld.RA.min())*np.cos(np.deg2rad(dec0))*3600.0) * ((fld.DEC.max()-fld.DEC.min())*3600.0)
    rho = len(fld) / area

    # --- TRUTH per shell ---
    reg = pred.predict_response(fld, fld)
    pk = [x for x in reg.columns if x.startswith("index") and x.endswith("_p")][0]
    sk = [x for x in reg.columns if x.startswith("index") and x.endswith("_s")][0]
    pr = reg[[pk, sk, "distance", "response"]].copy(); pr.columns = ["ip", "is_", "dist", "resp"]
    dfd = pred.predict_detection(fld, warn_extrapolation=False)
    ik = [x for x in dfd.columns if x.startswith("index") and x.endswith("_p")][0]
    pdmap = dict(zip(dfd[ik].to_numpy().astype(np.int64), dfd["detection_prob"].to_numpy()))
    pr["det_p"] = pr.ip.map(dm).fillna(False).astype(bool).to_numpy()
    pr["pdet_s"] = pr.is_.map(pdmap).fillna(0.0).to_numpy()
    dp = pr[pr.det_p]
    n_pri = dp.ip.nunique()
    dp = dp.assign(w=(1 - dp.pdet_s) * dp.resp)
    truth_shell = {}
    for lo, hi in zip(SHELLS[:-1], SHELLS[1:]):
        s = dp[(dp.dist >= lo) & (dp.dist < hi)]
        truth_shell[(lo, hi)] = s.w.sum() / n_pri

    # --- population MC per shell ---
    rng = np.random.default_rng(1)
    prim = fld.loc[pd.Series(fld_idx).map(dm).fillna(False).astype(bool).to_numpy(),
                   ["Re", "r", "sersic_n"]].to_numpy()
    prim = prim[rng.choice(len(prim), min(30000, len(prim)), replace=False)]
    npri = len(prim); pop = fld[["Re", "r", "sersic_n"]].to_numpy()
    prim_rep = np.repeat(prim, N_SYN, axis=0); M = npri * N_SYN
    snb = pop[rng.integers(0, len(pop), size=M)]
    theta = np.sqrt(THETA_MIN**2 + rng.random(M) * (R_MAX**2 - THETA_MIN**2))
    freg = pred.predict_on_pairs(pd.DataFrame({
        "Re_input_p": prim_rep[:, 0], "r_input_p": prim_rep[:, 1], "sersic_n_input_p": prim_rep[:, 2],
        "Re_input_s": snb[:, 0], "r_input_s": snb[:, 1], "sersic_n_input_s": snb[:, 2],
        "distance": theta}), task="response", warn_extrapolation=False)["response"].to_numpy()
    d_f = np.sqrt(-np.log(np.clip(rng.random(M), 1e-9, 1)) / (np.pi * rho))
    fgal = pop[rng.integers(0, len(pop), size=M)]
    # ---- conditioning weight: reweight neighbours by the PRIMARY's detection prob given the nbr ----
    # p_det(primary | nbr@theta) / p_det(primary | isolated).  Down-weights bright close neighbours
    # that would have destroyed the primary's clean detection (selection informed by blending).
    pdp_nbr = pred.predict_on_pairs(pd.DataFrame({
        "Re_input_p": prim_rep[:, 0], "r_input_p": prim_rep[:, 1], "sersic_n_input_p": prim_rep[:, 2],
        "Re_input_s": snb[:, 0], "r_input_s": snb[:, 1], "sersic_n_input_s": snb[:, 2],
        "distance": np.where(theta > R_MAX_CLA, np.nan, theta)}),  # icat2cla: neighbour ignored beyond 3"
        task="detection", warn_extrapolation=False)["detection_prob"].to_numpy()
    nan_col = np.full(npri, np.nan)
    pdp_iso = pred.predict_on_pairs(pd.DataFrame({
        "Re_input_p": prim[:, 0], "r_input_p": prim[:, 1], "sersic_n_input_p": prim[:, 2],
        "Re_input_s": nan_col, "r_input_s": nan_col, "sersic_n_input_s": nan_col,
        "distance": nan_col}), task="detection", warn_extrapolation=False)["detection_prob"].to_numpy()
    pdp_iso_rep = np.repeat(pdp_iso, N_SYN)
    cond_w = np.clip(pdp_nbr / np.clip(pdp_iso_rep, 1e-3, None), 0, 1)
    models = {}
    for variant in ["primctx", "faithful", "conditioned"]:
        if variant == "primctx":
            ctx = np.column_stack([prim_rep, theta]).astype(float); iso = theta > R_MAX_CLA
        else:
            use_p = theta <= d_f
            near_d = np.where(use_p, theta, d_f); near_p = np.where(use_p[:, None], prim_rep, fgal)
            ctx = np.column_stack([near_p, near_d]).astype(float); iso = near_d > R_MAX_CLA
        ctx[iso, :] = np.nan
        pdn = pred.predict_on_pairs(pd.DataFrame({
            "Re_input_p": snb[:, 0], "r_input_p": snb[:, 1], "sersic_n_input_p": snb[:, 2],
            "Re_input_s": ctx[:, 0], "r_input_s": ctx[:, 1], "sersic_n_input_s": ctx[:, 2],
            "distance": ctx[:, 3]}), task="detection", warn_extrapolation=False)["detection_prob"].to_numpy()
        contrib = (1 - pdn) * freg
        if variant == "conditioned":
            contrib = contrib * cond_w                    # faithful context + primary-detection reweight
        shellv = {}
        for lo, hi in zip(SHELLS[:-1], SHELLS[1:]):
            m = (theta >= lo) & (theta < hi)
            shellv[(lo, hi)] = rho * np.pi * (hi**2 - lo**2) * (contrib[m].mean() if m.any() else 0.0)
        models[variant] = shellv

    print(f"case{CASE}: rho={rho:.4f}, n_pri={n_pri:,}")
    print(f"\n{'shell':>12} {'TRUTH':>10} {'primctx':>9} {'faithful':>9} {'condtn':>9}  "
          f"{'p/t':>5} {'f/t':>5} {'c/t':>5}")
    tt = pt = ft = ct = 0.0
    for k in truth_shell:
        t, p, f, cc = truth_shell[k], models["primctx"][k], models["faithful"][k], models["conditioned"][k]
        tt += t; pt += p; ft += f; ct += cc
        print(f"  {k[0]:4.2f}-{k[1]:4.2f}\" {t:10.5f} {p:9.5f} {f:9.5f} {cc:9.5f}  "
              f"{p/t if t else 0:5.2f} {f/t if t else 0:5.2f} {cc/t if t else 0:5.2f}")
    print(f"  {'TOTAL':>10} {tt:10.5f} {pt:9.5f} {ft:9.5f} {ct:9.5f}  {pt/tt:5.2f} {ft/tt:5.2f} {ct/tt:5.2f}")
    print(f"\n  vs hard truth (0.0152): conditioned/hard = {ct/0.0152:.3f}")
    print("CTX_DIAG_DONE")


if __name__ == "__main__":
    main()
