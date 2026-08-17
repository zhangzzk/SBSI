"""PHASE 0b -- is the FORWARD-vs-ANTITHETIC extraction difference SIZE-DEPENDENT?

WHY THIS GATES PHASE 2. `scripts/eval_blend_vs_flow_perbin.py` splits the per-bin response error
into "flow" and "emulator" parts by pasting a fig5 column (half-shear, FORWARD `0 -> +g`) next to
constgold numbers (ANTITHETIC `+-g`). The extraction gap between those two conventions is a KNOWN
real effect in this project (+0.49 vs +0.60 at the faint end, on image-identical sims). If that gap
is FLAT in true size it shifts both attribution columns together and the size-axis split survives.
If it is SIZE-DEPENDENT it forges exactly the signal Phase 2 is chasing, and the "flow explains 2.07 /
emulator remainder 2.41" split has to be recomputed before any emulator work starts.

WHY IT CANNOT BE MEASURED THE OBVIOUS WAY. The direct test -- extract both ways on one population --
is impossible: constgold has ONLY `+-0.02` legs (no `g=0`), so it cannot be extracted forward, and the
half-shear legs have no `-g`, so they cannot be extracted antithetically. Checked, not assumed:
`lsst_sims_fs2_25876_constant/` contains `case*_0.02` and `case*_-0.02` and nothing else.

WHAT IS ACTUALLY AVAILABLE, AND WHAT IT LICENSES. Write the projected response as a series in the
applied shear magnitude,

    <de . ghat>(g) = R1 g + R2 g^2 + R3 g^3 + ...

    forward     R_fwd(g)  = <de.ghat>(g) / g              = R1 + R2 g + R3 g^2 + ...
    antithetic  R_anti(g) = [<de>(+g) - <de>(-g)] / (2g)  = R1        + R3 g^2 + ...

so the extraction difference is `R_fwd(g) - R_anti(g) = R2 g + O(g^3)` -- it is carried by the EVEN
term, and it is measurable from two forward legs at different |g| WITHOUT ever needing a `-g` leg:

    R2 ~= [R_fwd(0.05) - R_fwd(0.02)] / (0.05 - 0.02)

The half-shear family supplies exactly that: `g0.02_test` (cases 0-19) and `g0.05_val` (cases 0-199)
share the SAME `g0.0_train` reference leg, so both forward responses are measured on the same objects
against the same baseline. Cases 0-19 is the overlap and is what this script uses.

READ THE RESULT AS A SLOPE, NOT AS A CALIBRATION. This yields dR/dg per size bin. It does NOT license
subtracting a correction from anything -- AGENTS.md forbids silent empirical corrections, and this
number would be one. It answers one yes/no question: is dR/dg flat in true size?

TWO POPULATIONS ARE REPORTED, AND THE DIFFERENCE BETWEEN THEM IS THE POINT.

  own-set     each |g| on its OWN matched both-detected set (what fig5 actually does)
  common-set  only objects detected in ALL THREE legs

If the two agree, the g-dependence is a genuine response nonlinearity. If they disagree, part of what
looks like an extraction difference is SELECTION -- which legs an object survives -- and that is the
same mechanism as Phase 0d's leg-matching bias, not a Taylor-series effect. This script does not
assume which it is; it prints both and lets the numbers say.

An a-priori note kept deliberately, because it makes the result falsifiable: for an ISOTROPIC
intrinsic ellipticity distribution `<de.ghat>(g)` is ODD in g, so `R2 = 0` and the two extractions
agree exactly. A large measured `R2` therefore cannot be pure nonlinearity -- it would have to come
from an isotropy-breaking mechanism, and selection is the obvious candidate. The own-set vs common-set
comparison is what distinguishes them.

FIREWALL: nothing is trained, fit or tuned. No constgold is read. Half-shear only.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

SBSI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (SBSI_ROOT, os.path.join(SBSI_ROOT, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from eval_selfresp_gap import NGMIX, GAMMA, domain_cut, read_leg  # noqa: E402
from sbs_shear.paths import CATALOGUES as CAT

LEG0 = CAT + "det_meas_ngmix_g0.0_train.feather"
LEGS = [(0.02, CAT + "det_meas_ngmix_g0.02_test.feather"),
        (0.05, CAT + "det_meas_ngmix_g0.05_val.feather")]
BASE_COLS = ["case", "input_index", "detected", "Re_input_p", "r_input_p", "neighbored", "distance"]


def load_leg(path, max_case, extra=()):
    df = read_leg(path, BASE_COLS + NGMIX + list(extra), max_case)
    return domain_cut(df).drop_duplicates(["case", "input_index"]).reset_index(drop=True)


def resp(base, gmag_col=True):
    """Forward projected response and the per-object |g| actually applied."""
    gp = np.hypot(base["gamma1_input_p"].to_numpy(float), base["gamma2_input_p"].to_numpy(float))
    gh1 = base["gamma1_input_p"].to_numpy(float) / gp
    gh2 = base["gamma2_input_p"].to_numpy(float) / gp
    de1 = base["measured_ngmix_g1_g"].to_numpy(float) - base["measured_ngmix_g1_0"].to_numpy(float)
    de2 = base["measured_ngmix_g2_g"].to_numpy(float) - base["measured_ngmix_g2_0"].to_numpy(float)
    return (de1 * gh1 + de2 * gh2) / gp, gp


def binned(x, y, edges):
    idx = np.digitize(x, edges) - 1
    out = []
    for b in range(len(edges) - 1):
        s = (idx == b) & np.isfinite(y)
        n = int(s.sum())
        if n < 200:
            out.append((np.nan, n, np.nan))
            continue
        out.append((float(np.mean(y[s])), n, float(np.std(y[s], ddof=1) / np.sqrt(n))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-case", type=int, default=19,
                    help="the g0.02 test leg only covers cases 0-19; that is the hard overlap")
    ap.add_argument("--size-edges", type=float, nargs="*",
                    default=[0.30, 0.38, 0.46, 0.56, 0.70, 0.95, 1.50])
    ap.add_argument("--save-npz", default="results/extraction_gdep.npz")
    args = ap.parse_args()
    t0 = time.time()
    ed = np.array(args.size_edges, float)

    print(f"reading g=0 reference leg (cases 0-{args.max_case}) ...", flush=True)
    g0 = load_leg(LEG0, args.max_case)
    print(f"  g0: N={len(g0):,}  ({time.time()-t0:.0f}s)", flush=True)

    legs = {}
    for gval, path in LEGS:
        print(f"reading g={gval} leg ...", flush=True)
        d = load_leg(path, args.max_case, extra=GAMMA)
        gp = np.hypot(d["gamma1_input_p"].to_numpy(float), d["gamma2_input_p"].to_numpy(float))
        d = d[gp > 1e-6].reset_index(drop=True)
        legs[gval] = d
        print(f"  g={gval}: N={len(d):,}  ({time.time()-t0:.0f}s)", flush=True)

    # ---- do the two sheared legs even apply the SAME direction to the same object? ----------
    # If they do, the leg-to-leg difference is nearly noise-free (common intrinsic shape) and the
    # slope below is far better determined than a naive error bar would suggest. If they do not,
    # each leg is an independent draw. Printed rather than assumed either way.
    k = ["case", "input_index"]
    both = legs[0.02][k + GAMMA].merge(legs[0.05][k + GAMMA], on=k, suffixes=("_a", "_b"))
    if len(both):
        ha = np.arctan2(both["gamma2_input_p_a"], both["gamma1_input_p_a"])
        hb = np.arctan2(both["gamma2_input_p_b"], both["gamma1_input_p_b"])
        dth = np.abs(np.angle(np.exp(1j * (ha - hb))))
        print(f"\nshear DIRECTION agreement between the 0.02 and 0.05 legs on {len(both):,} shared "
              f"objects: median |dtheta| = {np.degrees(np.median(dth)):.2f} deg, "
              f"frac within 1 deg = {np.mean(dth < np.radians(1)):.3f}")
        print("  (same direction -> the two legs share the intrinsic shape and the slope is "
              "differential; different -> independent draws, larger error on the slope)")

    # ---- build both populations ------------------------------------------------------------
    common = None
    for gval in legs:
        ks = legs[gval][k]
        common = ks if common is None else common.merge(ks, on=k)
    common = common.merge(g0[k], on=k)
    print(f"\ncommon set (detected + in-domain in ALL THREE legs): N={len(common):,}")

    store, res = {}, {}
    for popname in ("own", "common"):
        print(f"\n{'='*104}\nPOPULATION: {popname}-set\n{'='*104}")
        curves = {}
        for gval in sorted(legs):
            d = legs[gval]
            if popname == "common":
                d = d.merge(common, on=k)
            base = d.merge(g0[k + NGMIX], on=k, suffixes=("_g", "_0"))
            r, gp = resp(base)
            size = base["Re_input_p"].to_numpy(float)
            gm = float(np.median(gp))
            ok = np.isfinite(r)
            print(f"  g={gval:<5} N={len(base):>10,}  g_med={gm:.4f}  "
                  f"<R_fwd>={np.nanmean(r[ok]):+.4f} +- {np.nanstd(r[ok])/np.sqrt(ok.sum()):.4f}")
            curves[gval] = (size, r, binned(size, r, ed))
        gs = sorted(curves)
        ga, gb = gs[0], gs[-1]

        print(f"\n  {'size bin':>14}{'n(0.02)':>10}{'R_fwd(0.02)':>13}{'n(0.05)':>10}"
              f"{'R_fwd(0.05)':>13}{'diff':>10}{'dR/dg':>10}{'sigma':>8}")
        rows = []
        for b in range(len(ed) - 1):
            ma, na, ea = curves[ga][2][b]
            mb, nb_, eb = curves[gb][2][b]
            diff = mb - ma
            sig = np.hypot(ea, eb)
            slope = diff / (gb - ga)
            print(f"  {f'{ed[b]:.2f}-{ed[b+1]:.2f}':>14}{na:>10,}{ma:>13.4f}{nb_:>10,}"
                  f"{mb:>13.4f}{diff:>+10.4f}{slope:>+10.3f}{diff/sig if sig else np.nan:>8.1f}")
            rows.append((0.5 * (ed[b] + ed[b + 1]), na, ma, ea, nb_, mb, eb, diff, sig, slope))
        a = np.array(rows, float)
        store[popname] = a
        sl = a[:, 9]
        ok = np.isfinite(sl)
        if ok.sum() > 1:
            spread = float(np.nanmax(sl[ok]) - np.nanmin(sl[ok]))
            # Is the slope CONSTANT across size? chi2 against its inverse-variance-weighted mean.
            w = 1.0 / (a[ok, 8] / (gb - ga)) ** 2
            mu = float(np.sum(w * sl[ok]) / np.sum(w))
            chi2 = float(np.sum(w * (sl[ok] - mu) ** 2))
            dof = int(ok.sum() - 1)
            print(f"\n  -> dR/dg: weighted mean {mu:+.3f}, spread across size bins {spread:.3f}")
            print(f"  -> FLATNESS TEST  chi2/dof = {chi2:.1f}/{dof} "
                  f"({'CONSISTENT with flat' if chi2 < 2*dof + 3 else 'NOT flat'})")
            res[popname] = (mu, spread, chi2, dof)

    # ---- what it means for the attribution -------------------------------------------------
    print(f"\n{'='*104}\nPHASE 0b GATE\n{'='*104}")
    print("  Gate as written in PLAN_resolution.md: 'flat in true size -> attribution stands.")
    print("  Size-dependent -> the flow explains 2.07 / emulator remainder 2.41 split must be")
    print("  recomputed with the extraction term removed.'")
    for p, (mu, spread, chi2, dof) in res.items():
        print(f"    {p:>7}-set: dR/dg = {mu:+.3f} (spread {spread:.3f} across size), "
              f"chi2/dof {chi2:.1f}/{dof}")
    if len(res) == 2:
        d_own, d_com = res["own"][0], res["common"][0]
        print(f"\n  own-set vs common-set slope: {d_own:+.3f} vs {d_com:+.3f} "
              f"(difference {d_own - d_com:+.3f})")
        print("  A large difference means the apparent extraction effect is SELECTION -- which legs")
        print("  an object survives -- not response nonlinearity. That routes it to Phase 0d, not")
        print("  to a nonlinearity correction.")
    print(f"\n  Size of the effect at the extractions actually paired in "
          f"eval_blend_vs_flow_perbin.py (fig5 forward at g=0.05 vs constgold antithetic at g->0,")
    print(f"  so the expected offset in R is dR/dg * 0.05):")
    for p, a in store.items():
        sl, sg = a[:, 9], a[:, 8] / (0.05 - 0.02)
        ok = np.isfinite(sl)
        off, offe = sl[ok] * 0.05, sg[ok] * 0.05
        # The raw max-min of the per-bin slopes is NOT a measured size variation -- every bin here
        # sits within ~1 sigma of the mean, so that spread is dominated by noise. Quote the
        # measured mean offset and a 1-sigma UPPER BOUND on any bin-to-bin variation instead.
        print(f"    {p:>7}-set: mean offset {np.mean(off):+.4f}; "
              f"largest per-bin |offset| {np.max(np.abs(off)):.4f}; "
              f"1-sigma bound on any single bin's offset {np.max(offe):.4f}")
    print("\n  Reference: the required-blend span Phase 2 is built on is 0.071-0.160, RANGE 0.089.")
    print("  READ IT AS A NON-DETECTION WITH A WEAK BOUND, NOT AS A CLEARANCE. The flatness chi2")
    print("  above finds no size-dependence, but the per-bin errors are not small compared with")
    print("  0.089, so this test cannot EXCLUDE a contaminating size trend -- it only fails to find")
    print("  one. The smallest-size bin carries the largest uncertainty and is where any real trend")
    print("  would most likely hide.")

    np.savez(args.save_npz, edges=ed, **{f"{p}": v for p, v in store.items()})
    print(f"\nsaved -> {args.save_npz}   ({time.time()-t0:.0f}s total)")
    print("LIMIT: 'antithetic = the g->0 limit' is exact only to O(g^2). The slope is measured from "
          "TWO |g| points, so a quadratic term in R_fwd(g) cannot be separated from the linear one; "
          "a third leg (g=0.2 exists, val only) would test that and is not used here because it is "
          "far outside the constgold regime.")


if __name__ == "__main__":
    main()
