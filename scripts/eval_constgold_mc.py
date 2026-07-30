"""m and c on constgold straight from the two legs, in three shape columns.

Owner's request: instead of the shift ratio R(cut)/R(no cut) - 1, use the textbook two-leg estimator.
constgold stores the SAME galaxy at +g and -g, so with <e> = (1+m) g + c:

    <e>_plus  = (1+m)(+g) + c
    <e>_minus = (1+m)(-g) + c
  =>   m = ( <e>_plus - <e>_minus ) / (2g) - 1        (difference kills c)
       c = ( <e>_plus + <e>_minus ) / 2               (sum kills the shear term)

Shapes are projected onto each case's shear direction, so `c` here is the additive bias ALONG the
shear direction. `c1`/`c2` in the fixed sky frame are printed once as a symmetry check: with random
shear angles they must average to ~0 regardless of anything else.

THREE SHAPE COLUMNS, the same ladder as eval_selection_constgold_3col.py:
  (1) UNSHEARED INTRINSIC -- both legs carry the SAME raw intrinsic shape. There is no shear signal
      at all, so m == -1 EXACTLY at no cut. Under a moving cut, (1 + m) is precisely the pure
      selection response: read `1+m`, not `m`, in this column.
  (2) SHEARED INTRINSIC -- analytic +/-g on the intrinsic shape. Response is 1 for isotropic
      orientations, so m is directly the multiplicative bias, and m ~ 0 at no cut is the sanity check.
  (3) MEASURED -- the stored ngmix shapes. m is large and negative because RAW ngmix ellipticity is
      not responsivity-corrected (R ~ 0.45, so m ~ -55%). That is the raw response, NOT the
      calibrated pipeline m; do not compare it to the +-0.3% deliverable.

ERRORS. The two legs are the SAME galaxy, so per-object combinations are the right unit:
    d = et_plus - et_minus  -> intrinsic shape cancels  -> se(m) = sd(d)/sqrt(n)/(2g)   [tight]
    a = (et_plus + et_minus)/2 -> intrinsic shape ADDS  -> se(c) = sd(a)/sqrt(n)        [shape-noise limited]
Both are evaluated on the BOTH-PASS set; under a moving boundary the legs differ slightly, so these
are the matched-pair errors and ignore the small mismatch term.

FIREWALL: constgold is read for EVALUATION only. Nothing is trained, fitted or selected on it.
"""
from __future__ import annotations

import argparse

import numpy as np
import pyarrow.feather as pf

CG = ("/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/"
      "constant_response_catalogue_train.feather")
COLS = ["case", "Re_input_p", "axis_ratio_input_p", "position_angle_input_p",
        "applied_g1", "applied_g2", "shear_magnitude", "shear_angle",
        "measured_e1_plus", "measured_e2_plus", "measured_e1_minus", "measured_e2_minus",
        "et_plus", "response", "S/N_plus", "S/N_minus"]


def apply_shear(e1, e2, g1, g2):
    e = e1 + 1j * e2
    g = g1 + 1j * g2
    es = (e + g) / (1.0 + np.conj(g) * e)
    return es.real, es.imag


def mc_two_leg(etp, etm, g, passp, passm, both):
    """m and c from the two leg means, with matched-pair errors on the both-pass set."""
    np_, nm_, nb = int(passp.sum()), int(passm.sum()), int(both.sum())
    if np_ == 0 or nm_ == 0:
        return (np.nan,) * 4
    mp, mm = float(np.mean(etp[passp])), float(np.mean(etm[passm]))
    m = (mp - mm) / (2.0 * g) - 1.0
    c = 0.5 * (mp + mm)
    if nb > 1:
        d = etp[both] - etm[both]
        a = 0.5 * (etp[both] + etm[both])
        se_m = float(np.std(d, ddof=1)) / np.sqrt(nb) / (2.0 * g)
        se_c = float(np.std(a, ddof=1)) / np.sqrt(nb)
    else:
        se_m = se_c = np.nan
    return m, se_m, c, se_c


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cat", default=CG)
    ap.add_argument("--min-case", type=int, default=40)
    ap.add_argument("--sn-cuts", type=float, nargs="+", default=[5, 7, 10, 15, 20, 30])
    ap.add_argument("--re-null", type=float, nargs="+", default=[0.3, 0.5])
    args = ap.parse_args()

    print(f"loading {args.cat}", flush=True)
    t = pf.read_table(args.cat, columns=COLS, memory_map=True).to_pandas()
    t = t[t.case >= args.min_case].reset_index(drop=True)
    g = float(np.median(t.shear_magnitude.to_numpy(float)))
    print(f"N={len(t):,}  cases={t.case.min()}..{t.case.max()}  g={g:.4f}", flush=True)

    q = t.axis_ratio_input_p.to_numpy(float)
    pa = np.deg2rad(t.position_angle_input_p.to_numpy(float))
    e = (1.0 - q) / (1.0 + q)
    e1i, e2i = e * np.cos(2 * pa), e * np.sin(2 * pa)
    ag1, ag2 = t.applied_g1.to_numpy(float), t.applied_g2.to_numpy(float)
    e1p, e2p = apply_shear(e1i, e2i, ag1, ag2)
    e1m, e2m = apply_shear(e1i, e2i, -ag1, -ag2)
    me1p, me2p = t.measured_e1_plus.to_numpy(float), t.measured_e2_plus.to_numpy(float)
    me1m, me2m = t.measured_e1_minus.to_numpy(float), t.measured_e2_minus.to_numpy(float)

    sa = t.shear_angle.to_numpy(float)
    c2, s2 = np.cos(2 * sa), np.sin(2 * sa)
    myp = me1p * c2 + me2p * s2
    st = t.et_plus.to_numpy(float)
    ok = np.isfinite(myp) & np.isfinite(st)
    sign = 1.0 if np.nanmean((myp * st)[ok]) > 0 else -1.0
    proj = lambda a, b: sign * (a * c2 + b * s2)

    pu = proj(e1i, e2i)
    kinds = [("(1) unsheared intrinsic", pu, pu),
             ("(2) sheared intrinsic", proj(e1p, e2p), proj(e1m, e2m)),
             ("(3) measured", proj(me1p, me2p), proj(me1m, me2m))]

    fin = np.ones(len(t), bool)
    for _, a, b in kinds:
        fin &= np.isfinite(a) & np.isfinite(b)
    snp, snm = t["S/N_plus"].to_numpy(float), t["S/N_minus"].to_numpy(float)
    fin &= np.isfinite(snp) & np.isfinite(snm)
    Re = t.Re_input_p.to_numpy(float)
    print(f"finite in all three shape kinds: {int(fin.sum()):,}", flush=True)

    # sky-frame additive check: with random shear angles these must be ~0
    for lab, a1, a2, b1, b2 in (("measured", me1p, me2p, me1m, me2m),
                                ("sheared intrinsic", e1p, e2p, e1m, e2m)):
        c1 = 0.5 * (np.mean(a1[fin]) + np.mean(b1[fin]))
        cc2 = 0.5 * (np.mean(a2[fin]) + np.mean(b2[fin]))
        print(f"  sky-frame additive check [{lab}]: c1={c1:+.2e}  c2={cc2:+.2e}")

    def block(title, sel_rows):
        print("\n" + "=" * 112)
        print(title)
        print("=" * 112)
        print(f"  {'cut':>9} {'frac':>6} | " +
              " | ".join(f"{k:^30}" for k, _, _ in kinds))
        print(f"  {'':>9} {'':>6} | " +
              " | ".join(f"{'m [%]':>15}{'c':>15}" for _ in kinds))
        for lab, pp, pm in sel_rows:
            both = pp & pm
            cells = []
            for _, a, b in kinds:
                m, se_m, c, se_c = mc_two_leg(a, b, g, pp, pm, both)
                cells.append(f"{100*m:>+9.3f}+-{100*se_m:<4.3f}{c:>+10.2e}+-{se_c:<.0e}")
            fr = 0.5 * (pp.mean() + pm.mean()) / max(fin.mean(), 1e-9)
            print(f"  {lab:>9} {fr:>6.3f} | " + " | ".join(cells))

    rows = [("NO CUT", fin, fin)]
    for thr in args.sn_cuts:
        rows.append((f"S/N>{thr:g}", fin & (snp > thr), fin & (snm > thr)))
    block("m AND c FROM THE TWO LEGS   m=(<e>_+ - <e>_-)/2g - 1   c=(<e>_+ + <e>_-)/2", rows)

    null = []
    for thr in args.re_null:
        k = fin & (Re > thr)
        null.append((f"Re>{thr:.2f}", k, k))
    block("NULL: cut on TRUE Re (boundary cannot move) -> column (1) must give m = -1 EXACTLY", null)

    print("\n  Column (1): read 1+m, not m. m=-1 exactly means ZERO selection response.")
    print("  Column (2): m ~ 0 at no cut is the sanity check; deviations under a cut are the bias.")
    print("  Column (3): m ~ -55% at no cut is the RAW ngmix response (not responsivity-corrected).")
    print("              It is NOT the calibrated pipeline m and must not be read against +-0.3%.")
    print("CONSTGOLD_MC_DONE", flush=True)


if __name__ == "__main__":
    main()
