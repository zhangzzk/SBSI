"""Is the measured shear response NONLINEAR between |g| = 0.02 and |g| = 0.05?

WHY THIS EXISTS. The V2.2 flow is pinned to a response target built from the half-shear legs at
|g| = 0.05 (`nominal_g = 0.05`, SNC estimator), and is then judged against constgold at |g| = 0.02.
If the response is nonlinear in `g`, the target's LEVEL is systematically offset from the level
constgold demands, and the flow faithfully reproduces the offset target. WORKLOG 2026-08-07c
localized the V2.2 closure deficit to a UNIFORM ~1% shortfall in `R_flow`, flat in size, magnitude
and S/N -- exactly the signature a target-level offset would produce.

WHY THE EXISTING BOUND IS NOT ENOUGH. 2026-08-05u measured the forward-vs-antithetic asymmetry on
constgold at |g| = 0.02 and got `-0.55% +- 0.61`, then EXTRAPOLATED it to the 0.05 leg by `g^2`,
giving `-3.41% +- 3.82`. That excluded a 22% gap, which was the question then. It does NOT exclude a
1% one: the extrapolated error is ~4x the effect now being chased. Every downstream tool
(`compare_size_demand.py`) therefore compares only SHAPES and explicitly refuses to difference the
absolute levels. A uniform offset is precisely the blind spot of that toolset.

WHAT THIS MEASURES, WITHOUT EXTRAPOLATING. The half-shear `g = 0.02` and `g = 0.05` legs cover the
SAME 100 cases with near-identical row counts, so the forward self-response can be formed at both
levels on the SAME objects:

    R_fwd(g) = < (e(+g) - e(0)) . ghat_p > / g

A linear response gives the same number at both. Any difference IS the nonlinearity, measured where
it matters rather than scaled from elsewhere. Note this is a different quantity from 05u's
forward-minus-antithetic asymmetry: that needed a `-g` leg, which the half-shear set does not have.
Both are consequences of the same nonlinearity; this one is measurable at the target's own |g|.

CONSTRUCTION MATCHES THE TARGET EXACTLY, via `sbs_shear.halfshear`, so the number is about the
response and not about the estimator: same SNC lookup and columns, same `ghat` from
`gamma1/2_input_p`, same `1/n_pairs` collapse of all-pairs duplicates, same detected filter, same
true-property box.

WHAT MAY AND MAY NOT BE CONCLUDED. Both legs are FORWARD and from the SAME catalogue family, so this
comparison does NOT cross extraction conventions and does NOT cross catalogues -- the two objections
that block a direct target-vs-constgold comparison. It is therefore a clean measurement of the
response's `g`-dependence. It does NOT by itself prove anything about the V2.2 deficit: relating a
`g = 0.05` target level to a `g = 0.02` constgold demand still crosses conventions. What this can do
is EXCLUDE the nonlinearity as a ~1% carrier, or measure it and say how much it could account for.

FIREWALL. Half-shear only. No constgold is read, no model is involved, nothing is trained, tuned or
corrected. No `m` is reported, so the 16-seed rule does not apply.
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from sbs_shear.halfshear import (LEG_PATHS, case_blocked, load_g0_lookup, load_leg,
                                 self_response_per_object, verify_independent_shear_directions)

# Uniform lift R_flow would need to close the V2.2 deficit (WORKLOG 2026-08-07c). Quoted only to
# size the measurement against the question; nothing here is corrected by it.
NEEDED_PCT = 1.25


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mag-max", type=float, default=25.8)
    ap.add_argument("--re-min", type=float, default=0.5)
    ap.add_argument("--max-case", type=int, default=99)
    args = ap.parse_args()
    t0 = time.time()

    print(f"V2.2 box: true mag < {args.mag_max}, Re > {args.re_min}; cases 0-{args.max_case}")
    g0 = load_g0_lookup()
    print(f"SNC lookup: {len(g0['key']):,} rows ({time.time()-t0:.0f}s)", flush=True)

    got = {}
    for g, path in sorted(LEG_PATHS.items()):
        leg = load_leg(path, args.mag_max, args.re_min, args.max_case)
        conc = verify_independent_shear_directions(leg)
        got[g], st = self_response_per_object(leg, g, g0)
        del leg
        print(f"  |g|={g}: {st['n_rows']:,} rows in box -> {len(got[g]):,} objects, "
              f"SNC matched {st['snc_matched']:.2%}, |g| median {st['gmag_median']:.4f}, "
              f"direction concentration {conc:.4f}  ({time.time()-t0:.0f}s)", flush=True)

    lo_g, hi_g = sorted(LEG_PATHS)
    a, b = got[lo_g], got[hi_g]
    common, ia, ib = np.intersect1d(a["key"].to_numpy(), b["key"].to_numpy(), return_indices=True)
    print(f"\nPAIRED on (case, input_index): {len(common):,} objects "
          f"({len(common)/len(a):.1%} of the {lo_g} leg, {len(common)/len(b):.1%} of the {hi_g})")

    Ra, Rb = a["R"].to_numpy()[ia], b["R"].to_numpy()[ib]
    case = a["case"].to_numpy()[ia]
    m_a, e_a, ncase = case_blocked(Ra, case)
    m_b, e_b, _ = case_blocked(Rb, case)
    d, ed, _ = case_blocked(Rb - Ra, case)          # paired difference, per object

    print(f"\n  R_fwd(|g|={lo_g}) = {m_a:.5f} +- {e_a:.5f}")
    print(f"  R_fwd(|g|={hi_g}) = {m_b:.5f} +- {e_b:.5f}")
    print(f"  PAIRED {hi_g} - {lo_g} = {d:+.5f} +- {ed:.5f}   ({abs(d)/ed:.1f} sigma)"
          f"   = {100*d/m_a:+.2f}% of R_fwd({lo_g})")
    print(f"  (errors blocked on {ncase} cases)")

    print("\nWHAT THIS MEANS FOR THE V2.2 DEFICIT")
    print(f"  The target is built at |g|={hi_g} and the flow is judged at |g|={lo_g}, so a")
    print(f"  nonlinearity of this size transfers to the target's LEVEL. Closing V2.2 needs a")
    print(f"  uniform +{NEEDED_PCT}% on R_flow. Measured here: {100*d/m_a:+.2f}% +- {100*ed/m_a:.2f}.")
    ci = 100 * (d - 2 * ed) / m_a, 100 * (d + 2 * ed) / m_a
    print(f"  95% interval on the transfer: [{ci[0]:+.2f}%, {ci[1]:+.2f}%]")
    if ci[0] <= NEEDED_PCT <= ci[1]:
        print(f"  -> the required {NEEDED_PCT}% lies INSIDE this interval: nonlinearity REMAINS a "
              "viable carrier. Check its SIGN against the deficit's before concluding anything.")
    else:
        print(f"  -> the required {NEEDED_PCT}% lies OUTSIDE this interval: nonlinearity is "
              "EXCLUDED as the carrier at 95%.")
    print("\nDIAG_RESPONSE_NONLINEARITY_DONE", flush=True)


if __name__ == "__main__":
    main()
