"""Measure R_blend DIRECTLY from the half-shear sim, with no emulator in the loop.

THE QUESTION. 2026-08-07d/e left a -2.39% level gap between the half-shear self-response and what
constgold demands, after the (mag, size) population term (+0.28%) was removed:

    half-shear self-response, ghat_p projection      0.8146 +- 0.0035
    constgold demand, R_sim - R_blend(emulator)      0.8369 +- 0.0012

Exactly one of two things is true, and the emulator sits inside only one of them:

  (A) the emulator's R_blend (0.1257) is TOO SMALL, so the demand is inflated. The independent
      per-pair ruler already reports the emulator low (v22 -9.78%, `_ho` -6.89%) -- right sign,
      neither significant alone.
  (B) the two sims genuinely disagree about the self response at the ~2.4% level, and no emulator
      change can reconcile them.

HOW THIS TELLS THEM APART. In the half-shear legs every galaxy carries an INDEPENDENT random shear
direction, so the same rows that give the self response under a ghat_p projection give the BLEND
response under a ghat_s projection onto the NEIGHBOUR's shear direction:

    R_self (per object)  = (e - e_0) . ghat_p / g            -- identical across an object's rows
    R_blend(per object)  = SUM over its pairs of (e - e_0) . ghat_s / g

Note the asymmetry, and it is not cosmetic: self is AVERAGED over an object's rows (they are the
same detection seen once per neighbour, so every row carries the same value) while blend is SUMMED
(each neighbour contributes its own, additively). Averaging the blend term would divide the answer
by the mean neighbour count.

This yields a sim-measured R_blend that never touches the emulator. If
`R_self + R_blend_sim ~= constgold R_sim`, the sims agree and (A) holds -- the emulator is the
carrier. If it falls short by ~0.022, (B) holds and the emulator is exonerated.

THE APERTURE CAVEAT, WHICH IS THE WHOLE DIFFICULTY. R_blend is a SUM over neighbours, so its value
depends entirely on which pairs are in the list -- AGENTS.md records a case where restricting the
pair list moved a verdict by a factor of nine, and insists a number always name the list it came
from. This script therefore reports R_blend_sim as a CUMULATIVE CURVE in pair separation rather than
as one number, so the comparison can be read at whatever aperture is being asked about, and so a
total that only closes by summing out to an implausible radius is visible as such.

USE AN ALL-PAIRS CATALOGUE. This bites harder than the aperture: the `det_meas_crowd_*` legs
annotate only the NEAREST neighbour (15.70M rows for 15.70M objects), so summing "over pairs" there
sums over exactly one and understates the blend response ~10x. Job 15596566 ran that way and
returned 0.0145 against a needed 0.148 -- an artefact of the pair list, not a measurement. The
`det_meas_ngmix_ap7_*` legs are the all-pairs build (25.66M rows, ~1.6 per object, 7" cap) and are
the default here. The printed rows-per-object is the guard: if it is ~1.0 the catalogue is a
nearest-neighbour build and the summed number must be discarded.

WHAT REMAINS CROSS-CATALOGUE. The final comparison of `R_self + R_blend_sim` against constgold's
R_sim crosses both catalogues AND conventions (half-shear is forward, constgold antithetic;
CONVENTIONS.md 6c). That step is a flag, not a proof, and the known convention term
(-0.55% +- 0.61, 2026-08-05u) is printed beside it rather than applied. Nothing here is corrected.

FIREWALL / SEEDS. Half-shear only; no model, no training, no tuning. The constgold reference numbers
are seed-independent (R_sim) or read from one dump. No `m` is reported.
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from sbs_shear.halfshear import (KEYMUL, LEG_PATHS, case_blocked, load_g0_lookup, load_leg,
                                 pack_key, verify_independent_shear_directions)

# Reference levels this is being compared against, all measured elsewhere and quoted for context
# only -- none of them is applied to anything computed here.
CONSTGOLD_RSIM = 0.96256          # 16-seed V2.2-box constgold R_sim (seed-independent)
EMULATOR_RBLEND = 0.12567         # v22 emulator, same rows (job 15596152)
CONVENTION_PCT = -0.55            # forward vs antithetic at |g|=0.02, +- 0.61 (WORKLOG 2026-08-05u)

# All-pairs legs (~1.6 rows/object, 7" build). The `crowd` legs in LEG_PATHS annotate only the
# nearest neighbour and CANNOT carry a summed blend response -- see the module docstring.
ALLPAIR_LEGS = {
    0.02: "/project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_ngmix_ap7_g0.02_test.feather",
    0.05: "/project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_ngmix_ap7_g0.05_val.feather",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--g", type=float, default=0.02, choices=sorted(LEG_PATHS))
    ap.add_argument("--pair-catalogue", default=None,
                    help="override the all-pairs leg; must be an ALL-PAIRS build (see docstring)")
    ap.add_argument("--mag-max", type=float, default=25.8)
    ap.add_argument("--re-min", type=float, default=0.5)
    ap.add_argument("--max-case", type=int, default=99)
    args = ap.parse_args()
    t0 = time.time()

    print(f"V2.2 box: true PRIMARY mag < {args.mag_max}, Re > {args.re_min}; "
          f"cases 0-{args.max_case}, |g| = {args.g}")
    print("neighbours are NOT cut on -- a blend response comes from whatever is nearby.\n")

    path = args.pair_catalogue or ALLPAIR_LEGS[args.g]
    print(f"pair catalogue: {path.split('/')[-1]}")
    g0 = load_g0_lookup()
    df = load_leg(path, args.mag_max, args.re_min, args.max_case,
                  extra_cols=("gamma1_input_s", "gamma2_input_s", "distance", "neighbored"))
    print(f"loaded {len(df):,} pair-rows, cases "
          f"{df['case'].min()}-{df['case'].max()} ({time.time()-t0:.0f}s)", flush=True)

    conc_p = verify_independent_shear_directions(df)
    theta2 = 2 * (np.arctan2(df["gamma2_input_s"].to_numpy(float),
                             df["gamma1_input_s"].to_numpy(float))
                  - np.arctan2(df["gamma2_input_p"].to_numpy(float),
                               df["gamma1_input_p"].to_numpy(float)))
    align = float(np.hypot(np.nanmean(np.cos(theta2)), np.nanmean(np.sin(theta2))))
    print(f"within-case direction concentration (primary) = {conc_p:.4f}")
    print(f"primary-vs-neighbour direction alignment      = {align:.4f}")
    print("  both ~0 is required: it is what makes the two projections separate cleanly, so that")
    print("  ghat_p picks up no neighbour shear and ghat_s picks up no self shear.\n")

    gp1, gp2 = df["gamma1_input_p"].to_numpy(float), df["gamma2_input_p"].to_numpy(float)
    gs1, gs2 = df["gamma1_input_s"].to_numpy(float), df["gamma2_input_s"].to_numpy(float)
    pmag, smag = np.hypot(gp1, gp2), np.hypot(gs1, gs2)
    key = pack_key(df["case"], df["input_index"])
    pos = np.clip(np.searchsorted(g0["key"], key), 0, len(g0["key"]) - 1)
    matched = g0["key"][pos] == key
    de1 = df["measured_ngmix_g1"].to_numpy(float) - np.where(matched, g0["e1"][pos], np.nan)
    de2 = df["measured_ngmix_g2"].to_numpy(float) - np.where(matched, g0["e2"][pos], np.nan)

    ok = matched & np.isfinite(de1) & np.isfinite(de2) & (pmag > 1e-6)
    nbr = df["neighbored"].to_numpy(bool) if "neighbored" in df.columns else np.ones(len(df), bool)
    dist = df["distance"].to_numpy(float)
    print(f"SNC matched {matched.mean():.2%}; {nbr.mean():.2%} of pair-rows have a rendered "
          f"neighbour")

    proj_p = np.where(ok, (de1 * gp1 + de2 * gp2) / np.where(pmag > 0, pmag, 1) / args.g, np.nan)
    good_s = ok & nbr & (smag > 1e-6) & np.isfinite(dist)
    proj_s = np.where(good_s, (de1 * gs1 + de2 * gs2) / np.where(smag > 0, smag, 1) / args.g, 0.0)

    uk, first, inv = np.unique(key[ok], return_index=True, return_inverse=True)
    case = (uk // KEYMUL).astype(np.int64)
    R_self = np.bincount(inv, weights=proj_p[ok]) / np.bincount(inv)   # rows agree; mean is exact
    s_ok = good_s[ok]

    m_self, e_self, ncase = case_blocked(R_self, case)
    print(f"\nR_self  (ghat_p, averaged over an object's rows) = {m_self:.5f} +- {e_self:.5f}"
          f"   ({len(uk):,} objects, {ncase} cases)")

    rows_per_obj = ok.sum() / len(uk)
    print(f"\nPAIR-LIST DEPTH: {rows_per_obj:.2f} rows per object")
    if rows_per_obj < 1.2:
        print("  *** NEAREST-NEIGHBOUR BUILD -- the summed blend response below is an ARTEFACT of")
        print("  *** the pair list, roughly an order of magnitude low, and must NOT be quoted.")
        print("  *** Re-run against an all-pairs (ap7) leg.")

    print("\nR_blend SUMMED OVER PAIRS, cumulative in separation")
    print("  the number is meaningless without the radius beside it -- read the row you need")
    print(f"\n  {'r_max':>8}  {'R_blend_sim':>12}  {'+-':>8}  {'pairs/obj':>10}  "
          f"{'self+blend':>11}  {'vs R_sim':>9}")
    d_ok = dist[ok]
    for rmax in [1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, np.inf]:
        sel = s_ok & (d_ok <= rmax)
        Rb = np.bincount(inv, weights=np.where(sel, proj_s[ok], 0.0), minlength=len(uk))
        m_b, e_b, _ = case_blocked(Rb, case)
        npair = sel.sum() / len(uk)
        tot = m_self + m_b
        lbl = "  all" if not np.isfinite(rmax) else f"{rmax:6.1f}\""
        print(f"  {lbl:>8}  {m_b:12.5f}  {e_b:8.5f}  {npair:10.2f}  {tot:11.5f}  "
              f"{100*(tot-CONSTGOLD_RSIM)/CONSTGOLD_RSIM:+8.2f}%")

    print(f"\n  emulator R_blend on the constgold rows, for scale: {EMULATOR_RBLEND:.5f}")
    print(f"  constgold R_sim (the total to be matched):        {CONSTGOLD_RSIM:.5f}")
    print(f"  blend needed to close it from R_self:             {CONSTGOLD_RSIM-m_self:.5f}")

    print("\nHOW TO READ THIS")
    print("  If some plausible aperture makes self+blend land on R_sim, the SIMS AGREE and the")
    print("  emulator's R_blend is the carrier -- hypothesis (A). If no aperture gets there, the")
    print("  sims disagree about the response itself -- hypothesis (B) -- and no emulator change")
    print("  fixes V2.2.")
    print(f"  The forward-vs-antithetic convention term ({CONVENTION_PCT:+.2f}% +- 0.61) is NOT")
    print("  applied above; it is the size of the residual that may legitimately remain.")
    print("  Beware the pair list: a radius the emulator never sums over is not evidence about")
    print("  the emulator, only about the sim.")
    print("\nDIAG_RBLEND_FROM_SIM_DONE", flush=True)


if __name__ == "__main__":
    main()
