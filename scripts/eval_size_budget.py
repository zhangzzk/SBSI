"""Does the SIZE-AXIS per-bin residual CLOSE once the emulator is exonerated?

WHERE THIS COMES FROM. 2026-08-01s attributed the true-size residual to the EMULATOR, on the strength
of `required_blend = r_sim - R_flow` implying the emulator's `R_blend` was "3.1x too flat" in true
size. 2026-08-02c tested that on the firewall-clean per-pair ruler and it did not survive: summed
`R_blend` agrees with ruler truth in EVERY true-size bin within ~1.9 sigma (overall +1.73% +- 3.25).

If the emulator is right, the arithmetic reassigns the whole residual:

    R_model - r_sim = (R_flow + R_blend) - (R_self + R_blend_true)
                    = (R_flow - R_self)        [emulator terms cancel]

so the model's per-bin residual IS the flow's per-bin error, up to the emulator's own (now small)
error. That is a PREDICTION, and it is falsifiable, because the flow's error is independently measured
by fig5 against half-shear. This script tests the closure:

    implied flow error (from constgold)  =?=  fig5 flow error  +  known target-construction terms

TWO TARGET-CONSTRUCTION TERMS ARE AVAILABLE AND ARE APPLIED AS EXPLICIT, LABELLED COLUMNS -- never
folded into a headline number (AGENTS.md: no silent corrections).

  * LEG-MATCHING (Phase 0d). `r_sim` is measured on the BOTH-DETECTED intersection, so it is `R_both`.
    The realistically-selected response is `R_full = R_both (1 + db)` with `db` measured per size bin
    (-0.843% to -1.717%). Judging the model against `R_full` instead of `R_both` changes the residual
    by roughly `-db`, i.e. shifts it UP by 0.84-1.72 pt with a 0.93 pt size-dependent part.
  * EXTRACTION (Phase 0b). fig5 is FORWARD at g=0.05, constgold is ANTITHETIC. 0b found no
    size-dependence (chi2/dof 2.2/5) but with a weak bound, so NO correction is applied for it; it is
    carried as a stated uncertainty on the closure, not as a column.

READ THE CLOSURE, NOT THE COLUMNS. If `implied - fig5 - legmatch` is small and unstructured, the
size-axis budget is closed and the resolution problem on this axis is the FLOW's self-response, fully
accounted. If a large structured remainder survives, something else is in play -- the first suspects
being additivity (still UNTESTED, Phase 0a is blocked) and the emulator's residual error.

CAVEATS THAT LIMIT WHAT A CLOSURE WOULD PROVE:
  * fig5 lives on a different population (2.36M half-shear rows) and a different extraction. The
    pairing is indicative; this script does not pretend the subtraction is exact.
  * the 0d det-bias uses SExtractor shapes, not ngmix.
  * additivity `R_total = R_self + R_blend` is assumed throughout and remains untested.
No fit, no tuning, no constgold-derived correction is applied to any model. Read of existing npz only.
"""
from __future__ import annotations

import argparse

import numpy as np

# column layout of results/blend_vs_flow_perbin.npz[<axis>]
CX, N, ST, FL, BL, TOT, RES, REQ, F5R, EXPL, REMD, SHARE = range(12)
# column layout of results/detection_perbin.npz[<scope>_<axis>]
D_CX, D_N, D_FO, D_RBOTH, D_RFULL, D_DB, D_SE = range(7)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perbin", default="results/blend_vs_flow_perbin.npz")
    ap.add_argument("--detbin", default="results/detection_perbin.npz")
    ap.add_argument("--axis", default="Re_input_p")
    ap.add_argument("--det-key", default="ALL_Re")
    ap.add_argument("--save-npz", default="results/size_budget.npz")
    args = ap.parse_args()

    a = np.load(args.perbin, allow_pickle=True)[args.axis]
    d = np.load(args.detbin, allow_pickle=True)[args.det_key]

    cx, st, fl, bl, tot = a[:, CX], a[:, ST], a[:, FL], a[:, BL], a[:, TOT]
    res, f5r = a[:, RES], a[:, F5R]

    # det-bias is measured on 6 coarse bins; the residual on 12. It is smooth and monotonic in size,
    # so interpolate onto the residual's centres. Extrapolation is CLAMPED to the end values rather
    # than extended linearly -- an extrapolated det-bias is not measured and must not be invented.
    db = np.interp(cx, d[:, D_CX], d[:, D_DB] * 100.0,
                   left=d[0, D_DB] * 100.0, right=d[-1, D_DB] * 100.0)
    dbe = np.interp(cx, d[:, D_CX], d[:, D_SE] * 100.0,
                    left=d[0, D_SE] * 100.0, right=d[-1, D_SE] * 100.0)
    clamped = (cx < d[0, D_CX]) | (cx > d[-1, D_CX])

    # implied flow error, in PERCENT OF R_flow: (R_model - r_sim)/R_flow, valid iff R_blend is right
    implied = (tot - st) / fl * 100.0
    # same, but judging against the realistically-selected target R_full = R_both (1 + db/100)
    implied_full = (tot - st * (1.0 + db / 100.0)) / fl * 100.0
    unexpl = implied - f5r                 # constgold-implied minus the independently measured flow
    closure = implied_full - f5r           # after the leg-matching term is applied

    print(f"\n{'='*128}")
    print("SIZE-AXIS BUDGET -- is the residual the FLOW, once the emulator is exonerated by the ruler?")
    print(f"{'='*128}")
    print(f"  {'Re':>8}{'n':>11}{'r_sim':>8}{'R_flow':>8}{'R_bl':>7}{'model resid%':>13}"
          f"{'implied flow%':>14}{'fig5 flow%':>12}{'unexpl%':>10} | {'0d db%':>8}"
          f"{'vs R_full%':>11}{'CLOSURE%':>10}")
    for i in range(len(cx)):
        flag = "*" if clamped[i] else " "
        print(f"  {cx[i]:>8.3f}{int(a[i,N]):>11,}{st[i]:>8.4f}{fl[i]:>8.4f}{bl[i]:>7.4f}"
              f"{res[i]:>+13.2f}{implied[i]:>+14.2f}{f5r[i]:>+12.2f}{unexpl[i]:>+10.2f} | "
              f"{db[i]:>+7.2f}{flag}{implied_full[i]:>+11.2f}{closure[i]:>+10.2f}")
    print("  * = det-bias clamped to the nearest measured bin (outside the 0d binning range)")

    ok = np.isfinite(f5r)
    def rms(v):
        return float(np.sqrt(np.nanmean(v[ok] ** 2)))
    def span(v):
        return float(np.nanmax(v[ok]) - np.nanmin(v[ok]))

    print(f"\n  over {int(ok.sum())} bins with a fig5 measurement:")
    print(f"    model residual              rms {rms(res):6.2f} pt   span {span(res):6.2f} pt")
    print(f"    implied flow error          rms {rms(implied):6.2f} pt   span {span(implied):6.2f} pt")
    print(f"    fig5 measured flow error    rms {rms(f5r):6.2f} pt   span {span(f5r):6.2f} pt")
    print(f"    UNEXPLAINED (implied-fig5)  rms {rms(unexpl):6.2f} pt   span {span(unexpl):6.2f} pt")
    print(f"    CLOSURE (after 0d leg-match)rms {rms(closure):6.2f} pt   span {span(closure):6.2f} pt")
    if rms(unexpl):
        print(f"\n    the leg-matching term accounts for "
              f"{100*(1 - rms(closure)/rms(unexpl)):.0f}% of the unexplained rms "
              f"and {100*(1 - span(closure)/span(unexpl)):.0f}% of its span")

    # SPLIT OFFSET FROM SCATTER, and check the smallest-size bin's leverage. The rms alone hides
    # both: an offset is a different physical statement from bin-to-bin scatter, and the first bin
    # sits at the flow's training-domain edge (Re > 0.3) where a boundary effect is expected.
    print(f"\n  OFFSET vs SCATTER (the rms conflates them):")
    for lab, v in (("unexplained", unexpl), ("closure", closure)):
        w = v[ok]
        print(f"    {lab:>12}: all {int(ok.sum())} bins rms {np.sqrt(np.mean(w**2)):5.2f} | "
              f"dropping the smallest-size bin: mean {np.mean(w[1:]):+5.2f} pt, "
              f"scatter about it {np.std(w[1:]):5.2f} pt, rms {np.sqrt(np.mean(w[1:]**2)):5.2f} pt")
    print(f"    smallest-size bin (Re ~ {cx[0]:.3f}) alone: unexplained {unexpl[0]:+.2f} pt, "
          f"closure {closure[0]:+.2f} pt")
    print("    That bin is the flow's TRAINING-DOMAIN EDGE (trained on Re > 0.3), so a boundary")
    print("    effect there is expected rather than surprising -- but it is not thereby explained.")

    print(f"\n{'='*128}\nVERDICT\n{'='*128}")
    print(f"  correlation(implied flow error, fig5 flow error) over the {int(ok.sum())} bins = "
          f"{np.corrcoef(implied[ok], f5r[ok])[0,1]:+.3f}")
    print("  A strongly POSITIVE correlation is the signature this script exists to look for: it")
    print("  means the constgold size residual and the independently-measured half-shear flow error")
    print("  are the SAME defect seen twice, which is what 'the size axis belongs to the flow'")
    print("  predicts. A near-zero correlation would mean the reassignment does not hold up and")
    print("  something not yet identified carries the size axis.")
    print("\n  REMEMBER WHAT IS STILL ASSUMED: additivity (untested, Phase 0a blocked), the emulator's")
    print("  residual per-bin error (+-6-10% on the ruler, not zero), and a size-independent")
    print("  extraction gap (0b non-detection, weak bound). A closure here is CONSISTENCY, not proof.")

    np.savez(args.save_npz, cx=cx, res=res, implied=implied, implied_full=implied_full,
             f5r=f5r, unexpl=unexpl, closure=closure, db=db, dbe=dbe, clamped=clamped)
    print(f"\nsaved -> {args.save_npz}")


if __name__ == "__main__":
    main()
