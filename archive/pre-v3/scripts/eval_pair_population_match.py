"""Does the constgold neighbour population match the one FLOW #2 was trained on?

WHY THIS EXISTS. A per-pair blending model is only usable end to end if the set of pairs it is SUMMED
over at evaluation resembles the set it was trained on. The first flow-#2 lookup attempt
(job 15478515) showed it does not:

    ap7 half-shear training set : <k> = 4.16 neighbours per primary within 7"
    constgold input field       : <k> = 8.29 neighbours per primary within 7"
    resulting <R_blend>         : 0.468, against the fiducial emulator's 0.136 on the same cases

A factor ~2 of that is simply twice as many pairs. The rest has to be the per-pair response being
evaluated on neighbours unlike any in training. Either way the summed number is not comparable, and
an `m` computed from it would be meaningless -- so it was not computed (AGENTS.md: if a quantity
cannot be computed properly, do not substitute a number).

WHAT THIS MEASURES. The two neighbour populations side by side, so the mismatch can be attributed
rather than guessed:
  * counts within the aperture, and how they build up with separation;
  * the neighbour MAGNITUDE distribution -- the obvious candidate, since the training catalogue's
    annotation may be flux-limited while a raw input field is not;
  * the neighbour SIZE distribution;
  * what `<k>` becomes under a matched magnitude cut, which is the direct test of whether a flux
    limit explains the whole gap.

If a magnitude cut reconciles the two, the lookup builder should apply exactly that cut and the
end-to-end path reopens. If it does not, the two suites have genuinely different galaxy densities and
the per-pair model has to be re-trained on, or re-weighted to, the evaluation population.

FIREWALL: reads constgold INPUT (true properties/positions) and the half-shear training pair set. No
constgold measurement is read and nothing is fitted.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.feather as pf
from scipy.spatial import cKDTree

SBSI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
for _p in (SBSI_ROOT, SCRIPTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from build_blend_lookup_flow import CBASE, TILE, input_feather, pairs_within  # noqa: E402

QUANTILES = [1, 5, 25, 50, 75, 95, 99]


def describe(name, v):
    q = np.nanpercentile(v, QUANTILES)
    print(f"  {name:>28}  n={len(v):>12,}  " + "  ".join(
        f"p{p}={x:7.3f}" for p, x in zip(QUANTILES, q)))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairset", required=True)
    ap.add_argument("--case", type=int, default=40)
    ap.add_argument("--sign", default="0.02")
    ap.add_argument("--base", default=CBASE)
    ap.add_argument("--aperture", type=float, default=7.0)
    ap.add_argument("--mag-max", type=float, default=26.0)
    ap.add_argument("--re-min", type=float, default=0.3)
    args = ap.parse_args()

    # ---- training population ----
    tr = pd.read_feather(args.pairset, columns=["r_input_s", "Re_input_s", "distance",
                                                "r_input_p", "Re_input_p", "pid"])
    ntr_prim = tr["pid"].nunique()
    print(f"\nTRAINING (half-shear ap7): {len(tr):,} pairs over {ntr_prim:,} primaries, "
          f"<k> = {len(tr)/ntr_prim:.3f}")

    # ---- constgold population ----
    fp = input_feather(args.case, args.sign, args.base)
    t = pf.read_table(fp).to_pandas()
    ra, dec = t["RA_input"].to_numpy(float), t["DEC_input"].to_numpy(float)
    i, j, d = pairs_within(ra, dec, args.aperture)
    mag, re_ = t["r_input"].to_numpy(float), t["Re_input"].to_numpy(float)
    prim_in = (mag[i] < args.mag_max) & (re_[i] > args.re_min)
    ncg_prim = int(((np.bincount(i, minlength=len(t)) > 0)
                    & (mag < args.mag_max) & (re_ > args.re_min)).sum())
    cg = pd.DataFrame({"r_input_s": mag[j], "Re_input_s": re_[j], "distance": d})[prim_in]
    print(f"CONSTGOLD case{args.case}: {len(t):,} galaxies in the field; "
          f"{len(cg):,} pairs on {ncg_prim:,} in-domain primaries, <k> = {len(cg)/max(ncg_prim,1):.3f}")

    print(f"\n{'='*118}\nNEIGHBOUR PROPERTY DISTRIBUTIONS (percentiles)\n{'='*118}")
    for nm, a, b in (("neighbour mag  r_input_s", tr["r_input_s"], cg["r_input_s"]),
                     ("neighbour size Re_input_s", tr["Re_input_s"], cg["Re_input_s"]),
                     ("separation      distance", tr["distance"], cg["distance"])):
        print(f"\n{nm}")
        describe("TRAINING", a.to_numpy(float))
        describe("CONSTGOLD", b.to_numpy(float))

    print(f"\n{'='*118}")
    print("DOES A NEIGHBOUR MAGNITUDE CUT RECONCILE <k>?")
    print(f"{'='*118}")
    print(f"  {'nbr mag <':>12}{'training <k>':>16}{'constgold <k>':>16}{'ratio':>10}")
    for mm in (24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 99.0):
        kt = (tr["r_input_s"].to_numpy(float) < mm).sum() / ntr_prim
        kc = (cg["r_input_s"].to_numpy(float) < mm).sum() / max(ncg_prim, 1)
        print(f"  {mm:>12.1f}{kt:>16.3f}{kc:>16.3f}{kc/max(kt,1e-9):>10.2f}")

    print(f"\n  Read the RATIO column. A value near 1.0 at some magnitude means the training")
    print("  annotation is flux-limited there and the lookup should apply the same cut. A ratio")
    print("  that stays ~2 at every cut means the two suites differ in galaxy DENSITY, which no")
    print("  cut can fix -- the model would then have to be applied with an evaluation-population")
    print("  reweighting, or retrained on pairs drawn from the constgold field's own statistics.")
    print("\nPAIR_POPULATION_MATCH_DONE", flush=True)


if __name__ == "__main__":
    main()
