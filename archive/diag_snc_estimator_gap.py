"""DO THE TWO SNC RESPONSE ESTIMATORS AGREE **REGION BY REGION**?

WHY THIS EXISTS -- it is the last link in a chain that otherwise cannot all be true at once.
Measured, all on small size (true Re <= 0.386):

  (1) the trained flow matches ITS OWN TARGET to +0.33 +- 0.10%   (03t, on the ruler rows)
  (2) the flow misses the DUMP TRUTH `r_sim_self` by -2.68%       (03s, central extraction)
  (3) an oracle fit on that dump truth REACHES it, +0.29 +- 0.82% (03q, D6 arm)

(1) and (2) can only both hold if the flow's TARGET disagrees with the dump truth at small size.
(3) says the disagreement is not an information limit -- the same features reach the dump truth when
fit against it. So the suspect is the LABEL the target was built from.

There are two SNC estimators of the same self-response in this project, and the per-object target and
the scoring truth use DIFFERENT ones:

  ruler `R_snc`      = measured_ngmix - `g0_lookup` (raw secondaries Shapes catalogues),
                       built by build_perobj_oracle.load_ruler; THIS is what the pin is fit to.
  dump `r_sim_self`  = e(g)-e(0) from a both-detected merge against the `g0.0_train` leg,
                       built by dump_halfshear_selfresp; THIS is what every score is graded on.

WORKLOG 2026-08-03q compared them and found means agreeing to 0.351% with correlation 0.525, and
concluded "same quantity, independent measurement noise". **That comparison was AGGREGATE ONLY.** A
region-dependent offset is invisible in a global mean and is exactly what would produce (1)+(2)+(3).
This script therefore repeats the comparison RESOLVED BY REGION.

IF THEY DISAGREE AT SMALL SIZE, the reading flips completely: the flow is faithfully reproducing a
biased supervision label, every "the flow fails at small size" result in 03k-03t is a statement about
the LABEL, and the fix is upstream in the estimator -- not in the flow, the pin, or the architecture.
IF THEY AGREE, then (1), (2) and (3) are genuinely inconsistent and something in the comparison chain
is wrong instead; that would need to be found before any of those numbers are used again.

FIREWALL. Half-shear only. constgold is never read and no `m` is computed. Nothing is fit here --
this compares two existing measurements of the same quantity on the same objects.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pyarrow.feather as pf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_perobj_oracle import load_ruler  # noqa: E402

DUMP = "results/halfshear_selfresp.feather"


def region_stats(a, b, keep, case, rng, nboot=300):
    """mean(a), mean(b) and 100*(a/b - 1) in a region, bootstrapped over CASES."""
    ok = keep & np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 100:
        return None
    ma, mb = float(np.mean(a[ok])), float(np.mean(b[ok]))
    cen = 100.0 * (ma / mb - 1.0)
    uc = np.unique(case[ok])
    bs = []
    for _ in range(nboot):
        pick = rng.choice(uc, size=len(uc), replace=True)
        m = np.isin(case, pick) & ok
        if m.sum() > 100:
            bs.append(100.0 * (float(np.mean(a[m])) / float(np.mean(b[m])) - 1.0))
    return ma, mb, cen, float(np.std(bs)), int(ok.sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ruler", required=True)
    ap.add_argument("--snc-lookup", required=True)
    ap.add_argument("--dump", default=DUMP)
    ap.add_argument("--nominal-g", type=float, default=0.05)
    ap.add_argument("--max-case", type=int, default=39)
    ap.add_argument("--primary-mag-max", type=float, default=26.0)
    ap.add_argument("--primary-re-min", type=float, default=0.3)
    ap.add_argument("--small-re", type=float, default=0.386)
    ap.add_argument("--faint-sn", type=float, default=13.09)
    args = ap.parse_args()

    ruler = load_ruler(args.ruler, args.snc_lookup, args.nominal_g, args.max_case,
                       args.primary_mag_max, args.primary_re_min)
    dump = pf.read_table(args.dump, memory_map=True).to_pandas()
    dump = dump[dump["case"] <= args.max_case]
    print(f"dump rows (case <= {args.max_case}): {len(dump):,}")

    mg = dump[["case", "input_index", "r_sim_self", "Re_input_p", "SN"]].merge(
        ruler[["case", "input_index", "R_snc"]], on=["case", "input_index"], how="inner")
    print(f"merged on (case, input_index): {len(mg):,} rows\n")

    a = mg["R_snc"].to_numpy(float)              # what the TARGET is fit to
    b = mg["r_sim_self"].to_numpy(float)         # what every SCORE is graded on
    re = mg["Re_input_p"].to_numpy(float)
    sn = mg["SN"].to_numpy(float)
    case = mg["case"].to_numpy(np.int64)
    rng = np.random.default_rng(0)

    print("=== ruler `R_snc` (the pin's label) vs dump `r_sim_self` (the scoring truth) ===")
    print("    positive = the LABEL sits ABOVE the truth, which would make a faithful model\n"
          "    read HIGH against the truth; negative = the label sits BELOW it.\n")
    for lab, keep in (("ALL", np.ones(len(mg), bool)),
                      (f"small size Re <= {args.small_re}", re <= args.small_re),
                      (f"faint  S/N <= {args.faint_sn}", sn <= args.faint_sn)):
        st = region_stats(a, b, keep, case, rng)
        if st is None:
            print(f"  {lab:<26} too few rows")
            continue
        ma, mb, cen, sd, n = st
        verdict = "DISAGREE" if abs(cen) > 2 * sd else "consistent"
        print(f"  {lab:<26} N={n:>9,}   <R_snc>={ma:+.5f}  <r_sim_self>={mb:+.5f}   "
              f"label/truth - 1 = {cen:+.2f} +- {sd:.2f} %  ({verdict})")

    print("\nREAD IT AS: a resolved disagreement at small size means every 'the flow fails at small\n"
          "size' number in 03k-03t is a statement about the LABEL, not the flow, and the fix is\n"
          "upstream in the response estimator. Agreement instead means the chain (1)(2)(3) in this\n"
          "docstring is inconsistent and must be re-derived before any of it is reused.")
    print("\nSNC_GAP_DONE")


if __name__ == "__main__":
    main()
