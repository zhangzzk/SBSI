"""Detection bias RESOLVED BY BLENDING SEVERITY on constgold (Stage-3, truth-only).

Extends scripts/eval_detection_constgold.py: instead of the 3-way ALL/ISOLATED/BLENDED split
plus a mag ladder, this bins the BLENDED population by how severe the blend is and measures the
detection bias per severity bin (optionally split by true primary magnitude).

For every (severity bin [x mag band]) we take the DETECTED galaxies in each leg, average their
INTRINSIC sheared shape along the fixed +g axis, and form the leg-average response:
  R_both = (<et>_+[both] - <et>_-[both]) / (2 g)     # detection-clean reference (both-detected)
  R_full = (<et>_+[all +leg] - <et>_-[all -leg]) / (2 g)  # real observed catalogue in that bin
  DETECTION BIAS = R_full / R_both - 1
All binning is on TRUE (input) neighbour properties -> shear-invariant -> adds no selection term.

SEVERITY DEFINITIONS (all from the same truth columns the framework's blend features use):
  * blendedness   beta = f_s K / (f_p + f_s K),  K = exp(-1/2 (distance/post_re_p)^2)
      = fraction of the primary-aperture flux contributed by the neighbour (Bosch 2018 style),
        built from the framework's own distance_scaled + flux ratio -> headline axis.
  * distance_scaled = distance / post_re_p  (separation in convolved primary sizes) -- kernel-free.
  * flux_ratio_nbr  = log10(f_s / f_p)      (neighbour-over-primary; bigger = brighter neighbour)
      -- kernel-free; note framework stores log10(f_p/f_s), we flip sign so "up = more severe".
  post_re_p = sqrt(Re_p^2 + psf_size^2), psf_size = moffat_fwhm2re(0.73, 2.224).

FIREWALL: nothing trains or fits. constgold is read only for validation/diagnosis.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pyarrow.feather as pf

SBSI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SBSI_ROOT not in sys.path:
    sys.path.insert(0, SBSI_ROOT)
from scripts.eval_detection_constgold import apply_shear  # noqa: E402  reduced-shear addition

D = "/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/"
PLUS = D + "constant_shear_catalogue_0.02_train.feather"
MINUS = D + "constant_shear_catalogue_-0.02_train.feather"
COLS = ["case", "input_index", "neighbored", "Re_input_p", "r_input_p",
        "axis_ratio_input_p", "position_angle_input_p", "applied_g1", "applied_g2",
        "measured_e1", "measured_e2", "S/N", "Re_input_s", "r_input_s", "distance"]

PSF_FWHM, MOFFAT_BETA = 0.73, 2.224


def moffat_fwhm2re(fwhm, beta):
    factor = np.sqrt((2 ** (1 / (beta - 1)) - 1) / (2 ** (1 / beta) - 1)) / 2
    return fwhm * factor


PSF_SIZE = moffat_fwhm2re(PSF_FWHM, MOFFAT_BETA)


def load_leg(path):
    t = pf.read_table(path, columns=COLS, memory_map=True).to_pandas()
    d = {}
    d["case"] = t.case.to_numpy(np.int64)
    d["input_index"] = t.input_index.to_numpy(np.int64)
    d["neighbored"] = t.neighbored.to_numpy(bool)
    d["Re"] = t.Re_input_p.to_numpy(float)
    d["mag"] = t.r_input_p.to_numpy(float)
    q = t.axis_ratio_input_p.to_numpy(float)
    pa = np.deg2rad(t.position_angle_input_p.to_numpy(float))
    e = (1.0 - q) / (1.0 + q)
    d["e1i"] = e * np.cos(2 * pa)
    d["e2i"] = e * np.sin(2 * pa)
    d["g1"] = t.applied_g1.to_numpy(float)
    d["g2"] = t.applied_g2.to_numpy(float)
    d["me1"] = t.measured_e1.to_numpy(float)
    d["me2"] = t.measured_e2.to_numpy(float)
    d["sn"] = t["S/N"].to_numpy(float)
    d["Re_s"] = t.Re_input_s.to_numpy(float)
    d["mag_s"] = t.r_input_s.to_numpy(float)
    d["dist"] = t.distance.to_numpy(float)
    return d


def severities(d):
    """Per-row blend severities (only meaningful where neighbored)."""
    post_re_p = np.sqrt(d["Re"] ** 2 + PSF_SIZE ** 2)
    dscaled = d["dist"] / post_re_p
    fp = 10.0 ** (-0.4 * d["mag"])
    fs = 10.0 ** (-0.4 * d["mag_s"])
    with np.errstate(over="ignore", invalid="ignore"):
        K = np.exp(-0.5 * dscaled ** 2)
        beta = (fs * K) / (fp + fs * K)
        fratio_nbr = np.log10(fs / fp)  # >0 => neighbour brighter than primary
    return {"beta": beta, "distance_scaled": dscaled, "flux_ratio_nbr": fratio_nbr}


def build_pair(dp, dm, sign):
    """Return dict of projected et arrays for both legs (sheared-intrinsic + measured)."""
    out = {}
    for tag, d, dirsign in (("p", dp, +1.0), ("m", dm, -1.0)):
        gm = np.hypot(d["g1"], d["g2"])
        c2 = dirsign * d["g1"] / gm
        s2 = dirsign * d["g2"] / gm
        e1s, e2s = apply_shear(d["e1i"], d["e2i"], d["g1"], d["g2"])
        out[f"shint_{tag}"] = sign * (e1s * c2 + e2s * s2)
        out[f"meas_{tag}"] = sign * (d["me1"] * c2 + d["me2"] * s2)
    return out


def leg_mean(et, mask):
    v = et[mask]
    v = v[np.isfinite(v)]
    return v.mean() if v.size else np.nan, v


def bias_in_mask(etp, etm, allp, allm, bp, bm, op, om, g):
    """R_both, R_full, det-bias, and an SE from the single-leg tail decomposition."""
    pF, _ = leg_mean(etp, allp)
    mF, _ = leg_mean(etm, allm)
    pB, _ = leg_mean(etp, bp)
    mB, _ = leg_mean(etm, bm)
    R_full = (pF - mF) / (2 * g)
    R_both = (pB - mB) / (2 * g)
    detbias = (R_full / R_both - 1.0) if np.isfinite(R_both) and R_both != 0 else np.nan
    # SE: <et>_all - <et>_both = f_only (<et>_only - <et>_both); variance from the single-leg means
    ep_o = etp[op]; ep_o = ep_o[np.isfinite(ep_o)]
    em_o = etm[om]; em_o = em_o[np.isfinite(em_o)]
    npall = max(int(np.isfinite(etp[allp]).sum()), 1)
    nmall = max(int(np.isfinite(etm[allm]).sum()), 1)
    f_p = ep_o.size / npall
    f_m = em_o.size / nmall
    vp = ep_o.var() / ep_o.size if ep_o.size else 0.0
    vm = em_o.var() / em_o.size if em_o.size else 0.0
    se_delta = np.sqrt(f_p ** 2 * vp + f_m ** 2 * vm) / (2 * g)
    se_bias = se_delta / abs(R_both) if np.isfinite(R_both) and R_both else np.nan
    return R_both, R_full, detbias, se_bias, npall, nmall


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plus", default=PLUS)
    ap.add_argument("--minus", default=MINUS)
    ap.add_argument("--min-case", type=int, default=40)
    ap.add_argument("--nbins", type=int, default=6, help="severity bins (equal-count)")
    ap.add_argument("--kind", default="shint", choices=["shint", "meas"],
                    help="shape flavour for the response (shint=sheared-intrinsic; meas=measured)")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    print(f"loading +leg {args.plus}", flush=True)
    dp = load_leg(args.plus)
    print(f"loading -leg {args.minus}", flush=True)
    dm = load_leg(args.minus)

    if args.min_case > 0:
        kp = dp["case"] >= args.min_case
        km = dm["case"] >= args.min_case
        dp = {k: v[kp] for k, v in dp.items()}
        dm = {k: v[km] for k, v in dm.items()}
    print(f"after min-case>={args.min_case}: +leg {dp['case'].size:,}  -leg {dm['case'].size:,}",
          flush=True)

    g = float(np.median(np.hypot(dp["g1"], dp["g2"])))
    print(f"g = {g:.4f}", flush=True)

    BIG = np.int64(1) << 34
    keyp = dp["case"] * BIG + dp["input_index"]
    keym = dm["case"] * BIG + dm["input_index"]
    _, ip_idx, im_idx = np.intersect1d(keyp, keym, assume_unique=True, return_indices=True)
    both_p = np.zeros(keyp.size, bool); both_p[ip_idx] = True
    both_m = np.zeros(keym.size, bool); both_m[im_idx] = True
    only_p = ~both_p
    only_m = ~both_m
    print(f"both={int(both_p.sum()):,}  +only={int(only_p.sum()):,}  -only={int(only_m.sum()):,}",
          flush=True)

    # lock projection sign so measured R_both > 0
    sh0 = build_pair(dp, dm, 1.0)
    R_try = (np.nanmean(sh0["meas_p"][both_p]) - np.nanmean(sh0["meas_m"][both_m])) / (2 * g)
    sign = 1.0 if R_try > 0 else -1.0
    sh = build_pair(dp, dm, sign) if sign < 0 else sh0
    etp, etm = sh[f"{args.kind}_p"], sh[f"{args.kind}_m"]
    print(f"projection sign={sign:+.0f}; using kind={args.kind}", flush=True)

    sev_p = severities(dp)
    sev_m = severities(dm)

    # magnitude bands (true primary r-mag; shear-invariant)
    mag_bands = [("all", -np.inf, np.inf),
                 ("bright r<25.5", -np.inf, 25.5),
                 ("faint r>=25.5", 25.5, np.inf)]
    sev_axes = ["beta", "distance_scaled", "flux_ratio_nbr"]

    # isolated reference (severity = 0), sheared-intrinsic, per mag band
    out = dict(g=g, kind=args.kind, nbins=args.nbins,
               mag_band_labels=np.array([b[0] for b in mag_bands]),
               sev_axes=np.array(sev_axes))
    print("\n==== ISOLATED reference (neighbored=False) ====")
    for bl, lo, hi in mag_bands:
        mp = (~dp["neighbored"]) & (dp["mag"] >= lo) & (dp["mag"] < hi)
        mm = (~dm["neighbored"]) & (dm["mag"] >= lo) & (dm["mag"] < hi)
        Rb, Rf, db, se, npall, nmall = bias_in_mask(
            etp, etm, mp, mm, both_p & mp, both_m & mm, only_p & mp, only_m & mm, g)
        out[f"iso_{bl}"] = np.array([Rb, Rf, db, se], float)
        print(f"  {bl:>14}: N={npall:,}  R_both={Rb:+.4f} R_full={Rf:+.4f} "
              f"det-bias={100*db:+.3f}% +-{100*se:.3f}%")

    for ax in sev_axes:
        # equal-count edges from the BLENDED both-detected population (pooled legs)
        pool = np.concatenate([sev_p[ax][dp["neighbored"] & both_p],
                               sev_m[ax][dm["neighbored"] & both_m]])
        pool = pool[np.isfinite(pool)]
        qs = np.linspace(0, 1, args.nbins + 1)
        edges = np.quantile(pool, qs)
        edges[0] -= 1e-9; edges[-1] += 1e-9
        print(f"\n==== severity axis = {ax} ====  edges={np.round(edges, 4)}")
        for bl, lo, hi in mag_bands:
            centers, Rbs, Rfs, dbs, ses, Ns = [], [], [], [], [], []
            print(f"  -- mag band: {bl} --")
            print(f"     {'bin':>18} {'sev_med':>8} {'R_both':>8} {'R_full':>8} "
                  f"{'det-bias':>9} {'N_+leg':>10}")
            magp = (dp["mag"] >= lo) & (dp["mag"] < hi)
            magm = (dm["mag"] >= lo) & (dm["mag"] < hi)
            for i in range(args.nbins):
                inp = (dp["neighbored"] & magp & np.isfinite(sev_p[ax])
                       & (sev_p[ax] >= edges[i]) & (sev_p[ax] < edges[i + 1]))
                inm = (dm["neighbored"] & magm & np.isfinite(sev_m[ax])
                       & (sev_m[ax] >= edges[i]) & (sev_m[ax] < edges[i + 1]))
                if int(inp.sum()) < 2000 or int(inm.sum()) < 2000:
                    continue
                Rb, Rf, db, se, npall, nmall = bias_in_mask(
                    etp, etm, inp, inm, both_p & inp, both_m & inm,
                    only_p & inp, only_m & inm, g)
                med = float(np.nanmedian(np.concatenate(
                    [sev_p[ax][inp], sev_m[ax][inm]])))
                centers.append(med); Rbs.append(Rb); Rfs.append(Rf)
                dbs.append(db); ses.append(se); Ns.append(npall)
                print(f"     [{edges[i]:>7.4f},{edges[i+1]:>7.4f}) {med:>8.4f} {Rb:>8.4f} "
                      f"{Rf:>8.4f} {100*db:>+8.3f}% {npall:>10,}")
            key = f"{ax}__{bl}"
            out[f"{key}__center"] = np.array(centers, float)
            out[f"{key}__Rboth"] = np.array(Rbs, float)
            out[f"{key}__Rfull"] = np.array(Rfs, float)
            out[f"{key}__db"] = np.array(dbs, float)
            out[f"{key}__se"] = np.array(ses, float)
            out[f"{key}__N"] = np.array(Ns, float)
        out[f"{ax}__edges"] = edges

    if args.output:
        np.savez(args.output, **out)
        print(f"\nsaved {args.output}", flush=True)
    print("\nDETECTION_SEVERITY_DONE", flush=True)


if __name__ == "__main__":
    main()
