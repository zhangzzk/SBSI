#!/usr/bin/env python
"""Direct sim-vs-sim check (owner 2026-07-23): the HALF-SHEAR (training) isolated self-response is
+0.49 (faint) while constgold's is +0.60 -- but the IMAGE SIM CONFIG IS IDENTICAL (same tile,
rms=0.3115, seeing=0.73, Moffat beta=2.224, psf_e=0, same NGMIX estimator, same fs2_25876 galaxies).
So the gap is NOT depth/PSF/noise; it must be in how the response is EXTRACTED:
  half-shear (forward, g0 vs +g legs, matched per (case,input_index)):
      R_self = [(e_g - e_0).ghat] / g          (e = measured_ngmix; e_int & additive-c cancel per obj)
  constgold (antithetic +/-g pairs):
      R_self = 0.5 (e_+ - e_-).ghat / g
This script ALSO measures, per mag bin per sim:
  std(dproj) = scatter of the LEG DIFFERENCE along shear -> tests whether the two differenced legs
               share a noise seed (small -> shared/cancels; ~sqrt2*sigma_e -> independent seeds),
  std(e0proj)= single-leg shape+noise scatter, and median S/N.
IDENTICAL cuts on both: detected, neighbored=False (ISOLATED), Re_input_p>0.3, r_input_p<26 (true cut).
FIREWALL: constgold read for comparison only; nothing trains.
"""
from __future__ import annotations

import pathlib as _pathlib, sys as _sys  # noqa: E402  -- make `sbs_shear` importable
_sys.path.insert(0, str(next(p for p in _pathlib.Path(__file__).resolve().parents
                             if (p / 'sbs_shear').is_dir())))
from sbs_shear import paths  # noqa: E402
import argparse, os, sys, time
import numpy as np, pandas as pd
import pyarrow as pa, pyarrow.ipc as ipc

CAT = f"{paths.CATALOGUE_DIR}/"
CDIR = f"{paths.CONST_SIM_DIR}"
MAG_EDGES = np.array([18.0, 24.0, 25.0, 26.0])


def _read(path, cols, max_case):
    parts = []
    with ipc.open_file(path) as r:
        avail = set(r.schema.names); use = [c for c in cols if c in avail]
        for bi in range(r.num_record_batches):
            b = pa.Table.from_batches([r.get_batch(bi)]).select(use).to_pandas()
            if max_case is not None:
                if int(b["case"].min()) > max_case:
                    break
                b = b[b["case"] <= max_case]
            if len(b):
                parts.append(b)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cols)


def _truecut_iso(df):
    det = df["detected"].astype(bool).to_numpy() if "detected" in df.columns else np.ones(len(df), bool)
    m = (det
         & (~df["neighbored"].astype(bool).to_numpy())
         & (df["Re_input_p"].to_numpy(float) > 0.3)
         & (df["r_input_p"].to_numpy(float) < 26.0))
    return df[m].reset_index(drop=True)


def _perbin(resp, mag, case, nboot=300, seed=0):
    rng = np.random.default_rng(seed)
    uniq = np.unique(case)
    out = {}
    mi = np.clip(np.digitize(mag, MAG_EDGES) - 1, 0, len(MAG_EDGES) - 2)
    for k in range(len(MAG_EDGES) - 1):
        sel = mi == k
        tag = f"mag[{MAG_EDGES[k]:g},{MAG_EDGES[k+1]:g})"
        if sel.sum() < 5:
            out[tag] = (np.nan, np.nan, int(sel.sum())); continue
        mean = float(np.mean(resp[sel]))
        bs = []; cs = case[sel]; rs = resp[sel]
        for _ in range(nboot):
            draw = rng.choice(uniq, size=len(uniq), replace=True)
            m = np.isin(cs, draw)
            if m.sum() > 2:
                bs.append(float(np.mean(rs[m])))
        out[tag] = (mean, float(np.nanstd(bs)), int(sel.sum()))
    return out


def _auxbin(mag, aux):
    """Per mag bin: std(dproj) [leg-diff scatter], std(e0proj) [single-leg], median S/N, R=mean(dproj)/g."""
    dproj, e0proj, sn, g = aux["dproj"], aux["e0proj"], aux["sn"], aux["gmed"]
    mi = np.clip(np.digitize(mag, MAG_EDGES) - 1, 0, len(MAG_EDGES) - 2)
    out = {}
    for k in range(len(MAG_EDGES) - 1):
        sel = mi == k
        tag = f"mag[{MAG_EDGES[k]:g},{MAG_EDGES[k+1]:g})"
        if sel.sum() < 5:
            out[tag] = None; continue
        dd = dproj[sel]; ee = e0proj[sel]; ss = sn[sel]
        out[tag] = dict(R=float(np.nanmean(dd)) / g, std_d=float(np.nanstd(dd)),
                        std_e0=float(np.nanstd(ee)), sn=float(np.nanmedian(ss[np.isfinite(ss)]))
                        if np.isfinite(ss).any() else np.nan, n=int(sel.sum()))
    return out


def halfshear(g0path, gpath, max_case, gname):
    cols = ["measured_ngmix_g1", "measured_ngmix_g2", "gamma1_input_p", "gamma2_input_p",
            "neighbored", "detected", "Re_input_p", "r_input_p", "sersic_n_input_p",
            "measured_flux_radius", "measured_flux_auto", "measured_fluxerr_auto",
            "case", "input_index"]
    t0 = time.time()
    G0 = _truecut_iso(_read(g0path, cols, max_case))
    G = _truecut_iso(_read(gpath, cols, max_case))
    gp = np.hypot(G["gamma1_input_p"].to_numpy(float), G["gamma2_input_p"].to_numpy(float))
    G = G[gp > 1e-6].reset_index(drop=True)
    G0 = G0.drop_duplicates(["case", "input_index"])
    G = G.drop_duplicates(["case", "input_index"])
    keep0 = ["case", "input_index", "measured_ngmix_g1", "measured_ngmix_g2"]
    m = G.merge(G0[keep0], on=["case", "input_index"], suffixes=("_g", "_0"))
    gp = np.hypot(m["gamma1_input_p"].to_numpy(float), m["gamma2_input_p"].to_numpy(float))
    gmed = float(np.median(gp))
    gh1 = m["gamma1_input_p"].to_numpy(float) / gp; gh2 = m["gamma2_input_p"].to_numpy(float) / gp
    de1 = m["measured_ngmix_g1_g"].to_numpy(float) - m["measured_ngmix_g1_0"].to_numpy(float)
    de2 = m["measured_ngmix_g2_g"].to_numpy(float) - m["measured_ngmix_g2_0"].to_numpy(float)
    resp = (de1 * gh1 + de2 * gh2) / gmed
    mag = m["r_input_p"].to_numpy(float); case = m["case"].to_numpy(int)
    dproj = de1 * gh1 + de2 * gh2
    e0proj = m["measured_ngmix_g1_0"].to_numpy(float) * gh1 + m["measured_ngmix_g2_0"].to_numpy(float) * gh2
    if "measured_flux_auto" in m.columns and "measured_fluxerr_auto" in m.columns:
        sn = m["measured_flux_auto"].to_numpy(float) / m["measured_fluxerr_auto"].to_numpy(float)
    else:
        sn = np.full(len(m), np.nan)
    good = np.isfinite(resp)
    nbad = int((~good).sum())
    resp, mag, case = resp[good], mag[good], case[good]
    aux = {"dproj": dproj[good], "e0proj": e0proj[good], "sn": sn[good], "gmed": gmed}
    print(f"[half-shear {gname}] matched N={len(m):,}  finite N={len(resp):,} (dropped {nbad:,} nan-ngmix)  "
          f"g_med={gmed:.4f}  ({time.time()-t0:.1f}s)", flush=True)
    dcols = [c for c in ["Re_input_p", "r_input_p", "sersic_n_input_p", "measured_flux_radius"] if c in m.columns]
    dist = m[dcols].copy(); dist["mag"] = m["r_input_p"].to_numpy(float)
    return _perbin(resp, mag, case), dist, (mag, aux)


def constgold(max_case):
    cols = ["measured_e1_plus", "measured_e2_plus", "measured_e1_minus", "measured_e2_minus",
            "applied_g1", "applied_g2", "neighbored", "detected", "S/N_plus", "S/N_minus",
            "Re_input_p", "r_input_p", "sersic_n_input_p", "measured_flux_radius", "case", "input_index"]
    t0 = time.time()
    R = _truecut_iso(_read(CDIR + "/constant_response_catalogue_train.feather", cols, max_case))
    g = np.hypot(R["applied_g1"].to_numpy(float), R["applied_g2"].to_numpy(float))
    gmed = float(np.median(g)); gh1 = R["applied_g1"].to_numpy(float) / g; gh2 = R["applied_g2"].to_numpy(float) / g
    e1p = R["measured_e1_plus"].to_numpy(float); e2p = R["measured_e2_plus"].to_numpy(float)
    e1m = R["measured_e1_minus"].to_numpy(float); e2m = R["measured_e2_minus"].to_numpy(float)
    resp = 0.5 * ((e1p - e1m) * gh1 + (e2p - e2m) * gh2) / gmed
    mag = R["r_input_p"].to_numpy(float); case = R["case"].to_numpy(int)
    # dproj = (e+ - e-).ghat ; for constgold this is a 2g-span central difference (note factor 0.5 in R)
    dproj = (e1p - e1m) * gh1 + (e2p - e2m) * gh2
    e0proj = e1m * gh1 + e2m * gh2                                    # single (minus) leg
    sn = R["S/N_plus"].to_numpy(float) if "S/N_plus" in R.columns else np.full(len(R), np.nan)
    good = np.isfinite(resp)
    resp, mag, case = resp[good], mag[good], case[good]
    aux = {"dproj": 0.5 * dproj[good], "e0proj": e0proj[good], "sn": sn[good], "gmed": gmed}
    print(f"[constgold] N={len(R):,}  finite N={len(resp):,}  g_med={gmed:.4f}  ({time.time()-t0:.1f}s)", flush=True)
    dcols = [c for c in ["Re_input_p", "r_input_p", "sersic_n_input_p", "measured_flux_radius"] if c in R.columns]
    dist = R[dcols].copy(); dist["mag"] = R["r_input_p"].to_numpy(float)
    return _perbin(resp, mag, case), dist, (mag, aux)


def _fmt(d):
    return "  ".join(f"{k.split('[')[1][:-1]}:{v[0]:+.3f}±{v[1]:.3f}(n{v[2]//1000}k)" for k, v in d.items())


def _distsummary(dist, label):
    faint = dist[(dist["mag"] >= 25.0) & (dist["mag"] < 26.0)]
    print(f"\n  [{label}] FAINT-isolated (mag[25,26)) N={len(faint):,}")
    for c in ["Re_input_p", "sersic_n_input_p", "measured_flux_radius", "mag"]:
        if c in faint and faint[c].notna().any():
            v = faint[c].to_numpy(float); v = v[np.isfinite(v)]
            print(f"    {c:20s} mean={v.mean():.3f} med={np.median(v):.3f} "
                  f"q[10,50,90]={np.round(np.quantile(v,[.1,.5,.9]),3).tolist()}")


def _auxtable(label, mag, aux):
    ab = _auxbin(mag, aux)
    print(f"\n  [{label}]  (std_d=leg-diff scatter; std_e0=single-leg; ratio=std_d/(sqrt2*std_e0): ~1 indep seeds, <<1 shared)")
    for tag, v in ab.items():
        if v is None:
            print(f"    {tag:14s} (n<5)"); continue
        ratio = v["std_d"] / (np.sqrt(2) * v["std_e0"]) if v["std_e0"] > 0 else np.nan
        print(f"    {tag:14s} R={v['R']:+.3f}  std_d={v['std_d']:.3f}  std_e0={v['std_e0']:.3f}  "
              f"ratio={ratio:.2f}  medS/N={v['sn']:.1f}  n={v['n']//1000}k")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="ngmix",
                    help="half-shear measurement variant: ngmix | ngmix_ap7 | ngmix_np7")
    ap.add_argument("--hs-max-case", type=int, default=19)
    ap.add_argument("--cg-max-case", type=int, default=39)
    ap.add_argument("--output", default="/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/selfresp_sims.npz")
    args = ap.parse_args()
    t0 = time.time()
    v = args.variant
    print(f"HALF-SHEAR VARIANT = {v}", flush=True)
    hs02, d_hs02, a_hs02 = halfshear(CAT + f"det_meas_{v}_g0.0_train.feather",
                                     CAT + f"det_meas_{v}_g0.02_test.feather", args.hs_max_case, f"{v} g=0.02")
    hs05, d_hs05, a_hs05 = halfshear(CAT + f"det_meas_{v}_g0.0_train.feather",
                                     CAT + f"det_meas_{v}_g0.05_val.feather", args.hs_max_case, f"{v} g=0.05")
    cg, d_cg, a_cg = constgold(args.cg_max_case)

    print("\n" + "=" * 92)
    print("ISOLATED (no true neighbour) TRUE-CUT (Re>0.3, mag<26) self-response, per mag bin  [mean +- case-bootstrap]")
    print("=" * 92)
    print(f"  half-shear g=0.02 : {_fmt(hs02)}")
    print(f"  half-shear g=0.05 : {_fmt(hs05)}   <- what the flow/target learned")
    print(f"  constgold  g=0.02 : {_fmt(cg)}     <- the acceptance truth")
    print("\n  FAINT [25,26) head-to-head (same cuts, isolated, ngmix):")
    for lab, d in [("HS g=0.02", hs02), ("HS g=0.05", hs05), ("constgold", cg)]:
        mv, e, n = d["mag[25,26)"]
        print(f"    {lab:11s} R_self = {mv:+.4f} +- {e:.4f}  (N={n:,})")

    print("\n" + "=" * 92)
    print("NOISE-SEED / LEG-CONSTRUCTION diagnostic (images are identical config -> gap must be here)")
    print("=" * 92)
    _auxtable("half-shear g=0.02", *a_hs02)
    _auxtable("half-shear g=0.05", *a_hs05)
    _auxtable("constgold +/-0.02", *a_cg)

    print("\n" + "=" * 92)
    print("DISTRIBUTIONS of the FAINT-isolated populations (population/measurement confound check)")
    print("=" * 92)
    _distsummary(d_hs02, "half-shear g=0.02")
    _distsummary(d_hs05, "half-shear g=0.05")
    _distsummary(d_cg, "constgold g=0.02")
    np.savez(args.output, hs02=str(hs02), hs05=str(hs05), cg=str(cg))
    print(f"\nsaved {args.output}   ({time.time()-t0:.1f}s)")
    print("SELFRESP_SIMS_DONE", flush=True)


if __name__ == "__main__":
    main()
