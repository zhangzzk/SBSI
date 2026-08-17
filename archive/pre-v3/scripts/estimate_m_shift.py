"""What does the corrected R_blend emulator do to the certified constgold m?

The full acceptance is `validate_constant_with_blend.py` over the 8-seed flow ensemble on GPU. This
is the cheap precursor: the certified m enters as

    1 + m = <R_sim> / (<R_flow> + <R_blend>)

so a change in <R_blend> alone moves m by a known amount WITHOUT needing <R_flow> separately --
eliminate it using the certified numbers:

    <R_flow> + <R_blend_old> = <R_sim> / (1 + m_old)
    1 + m_new = <R_sim> / ( <R_sim>/(1 + m_old) - <R_blend_old> + <R_blend_new> )

Both lookups are per-object sums over neighbours on the SAME constgold field, so the comparison is
restricted to the (case, input_index) rows they share.

APPROXIMATION, stated rather than buried: the pipeline forms m from per-object quantities and the
identity above uses ensemble means, so this predicts the SHIFT well but is not a substitute for the
GPU run. It is meant to say whether the corrected emulator moves m enough to matter, and in which
direction, before spending the full chain.

The distance fix vanishes beyond 3" and R_blend is a sum over ALL neighbours out to 10", most of
which are far, so a large per-pair gain at <1" can still be a small shift here. That is the expected
outcome, not a disappointment.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import pyarrow.feather as pf

CERT = "/home/z/Zekang.Zhang/SBSI/results/blend_lookup_extnbrho_c40-139.feather"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--certified", default=CERT)
    ap.add_argument("--corrected", required=True)
    ap.add_argument("--r-sim", type=float, default=0.4534,
                    help="certified constgold R_sim (fixed-centroid catalogue)")
    ap.add_argument("--m-old", type=float, default=0.00245,
                    help="certified Gold-v1 m with the certified R_blend")
    args = ap.parse_args()

    a = pf.read_table(args.certified, columns=["case", "input_index", "R_blend"]).to_pandas()
    b = pf.read_table(args.corrected, columns=["case", "input_index", "R_blend"]).to_pandas()
    print(f"certified: {len(a):,} rows, cases {a['case'].min()}-{a['case'].max()}")
    print(f"corrected: {len(b):,} rows, cases {b['case'].min()}-{b['case'].max()}")

    m = a.merge(b, on=["case", "input_index"], how="inner", suffixes=("_old", "_new"))
    print(f"shared objects: {len(m):,}")
    if not len(m):
        print("*** no overlap -- cannot compare ***")
        return

    ro = m["R_blend_old"].to_numpy(float)
    rn = m["R_blend_new"].to_numpy(float)
    ok = np.isfinite(ro) & np.isfinite(rn)
    ro, rn = ro[ok], rn[ok]
    mo, mn = ro.mean(), rn.mean()
    print(f"\n<R_blend> certified = {mo:.6f}")
    print(f"<R_blend> corrected = {mn:.6f}   (change {mn-mo:+.6f}, {100*(mn/mo-1):+.3f}%)")
    print(f"per-object |delta| median = {np.median(np.abs(rn-ro)):.6f}")

    denom_old = args.r_sim / (1.0 + args.m_old)
    denom_new = denom_old - mo + mn
    m_new = args.r_sim / denom_new - 1.0
    print(f"\ncertified m = {100*args.m_old:+.3f}%   (R_sim={args.r_sim:.4f}, "
          f"R_flow+R_blend={denom_old:.6f})")
    print(f"implied  m = {100*m_new:+.3f}%   with the corrected R_blend")
    print(f"shift      = {100*(m_new-args.m_old):+.3f} percentage points")
    print("\nNOT a substitute for validate_constant_with_blend.py on the flow ensemble; this uses")
    print("ensemble means where the pipeline uses per-object quantities.")
    print("M_SHIFT_DONE", flush=True)


if __name__ == "__main__":
    main()
