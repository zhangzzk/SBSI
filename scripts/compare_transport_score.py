"""Transport (§5A) and the score route (§5B) side by side, bin by bin, on the same rows.

The two estimators combine the SAME per-object responses differently:

    transport   m + 1 = <r_sim> / (<r_flow> + <r_blend>)     ratio of population means
    score       ghat/g = <I_i r_i / a_i> / <I_i>             information-weighted

so putting them in one table separates a MODEL error (shows up in both) from a
WEIGHTING error (shows up only in the score).  Splits are on shear-independent
covariates only -- true magnitude, true size, `R_blend`, blending -- because cutting on
`I_i` or on the measured shape is cutting on the data.

Needs the `_rflow` dump (`--flow-perobj-only`) and the `_bare`/`_inj` score dumps from the
same rows.  `load_constgold` is deterministic, so the true size is recovered by re-loading
and the alignment is VERIFIED against `r_sim` and `r_input_p` before anything is printed.

Usage:
    python scripts/compare_transport_score.py --base <dir>/perobj_fix2
"""

import argparse
import contextlib
import io
import os
import sys
import types

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True,
                    help="dump prefix; reads <base>_rflow.npz, _bare.npz, _inj.npz")
    ap.add_argument("--mag-cut", type=float, default=26.0)
    ap.add_argument("--size-cut", type=float, default=0.3)
    ap.add_argument("--min-case", type=int, default=40)
    args = ap.parse_args()

    from scripts.eval_score_response import (BLEND_LOOKUP, CROWD_FLUX_LOOKUP, GOLD_CAT,
                                             MEAS_PRIM_LOOKUP, load_constgold)

    rf = np.load(f"{args.base}_rflow.npz")
    sc = np.load(f"{args.base}_bare.npz")
    inj_path = f"{args.base}_inj.npz"
    si = np.load(inj_path) if os.path.exists(inj_path) else None
    if not np.allclose(rf["r_sim"], sc["r_sim"]):
        raise SystemExit("row alignment failed between the rflow and score dumps")
    n = len(rf["r_sim"])
    print(f"aligned: {n:,} rows")

    g = float(sc["g"])
    r_sim, r_flow, r_bl = rf["r_sim"], rf["r_flow"], rf["r_blend"]
    mag_t = sc["r_input_p"]

    if "Re_input_p" in sc:
        re_t = sc["Re_input_p"]
    else:
        a = types.SimpleNamespace(
            catalogue=GOLD_CAT, min_case=args.min_case, max_rows=n,
            no_source_selection=False, crowd_flux_lookup=CROWD_FLUX_LOOKUP,
            meas_prim_lookup=MEAS_PRIM_LOOKUP, blend_lookup=BLEND_LOOKUP)
        with contextlib.redirect_stdout(io.StringIO()):
            df = load_constgold(a)
        if len(df) != n or not np.allclose(df["r_input_p"].to_numpy(float), mag_t):
            raise SystemExit("re-load does not reproduce the dump's rows -- refusing to join")
        re_t = df["Re_input_p"].to_numpy(float)
        print("  true size recovered by re-load, alignment verified against r_input_p")

    sA = 0.5 * (sc["s_plus"] - sc["s_minus"])
    iA = 0.5 * (sc["i_plus"] + sc["i_minus"])
    if si is not None:
        sB = 0.5 * (si["s_plus"] - si["s_minus"])
        iB = 0.5 * (si["i_plus"] + si["i_minus"])

    def row(lab, m):
        k = int(m.sum())
        if k < 500:
            return
        tr = r_sim[m].mean() / (r_flow[m].mean() + r_bl[m].mean()) - 1
        a_ = float(np.sum(sA[m]) / np.sum(iA[m])) / g - 1
        b_ = (float(np.sum(sB[m]) / np.sum(iB[m])) / g - 1) if si is not None else np.nan
        print(f"  {lab:>26} {k:>8,} {r_sim[m].mean():>7.4f} {r_flow[m].mean():>7.4f} "
              f"{r_bl[m].mean():>7.4f} {tr:>+9.2%} {a_:>+9.2%} {b_:>+9.2%}")

    keep = (mag_t < args.mag_cut) & (re_t > args.size_cut)
    print(f"\n{'sample':>28} {'N':>8} {'R_sim':>7} {'R_flow':>7} {'R_bl':>7} "
          f"{'TRANSPORT':>9} {'score bare':>9} {'score inj':>9}")
    row("FULL catalogue", np.ones(n, bool))
    row(f"ACCEPTANCE mag<{args.mag_cut:g}&Re>{args.size_cut:g}", keep)
    row("rejected", ~keep)
    print()
    row(f"true mag < {args.mag_cut:g}", mag_t < args.mag_cut)
    row(f"true Re > {args.size_cut:g}", re_t > args.size_cut)

    print("\n  --- inside acceptance, by R_blend ---")
    q = np.quantile(r_bl[keep][r_bl[keep] > 0.02], [0, .25, .5, .75, 1.])
    row("R_blend<0.02", keep & (r_bl <= 0.02))
    for k in range(4):
        row(f"R_b {q[k]:.3f}-{q[k+1]:.3f}",
            keep & (r_bl > max(q[k], 0.02)) & (r_bl <= q[k + 1]))

    print("\n  --- inside acceptance, by TRUE size ---")
    for lo, hi in ((0.3, 0.4), (0.4, 0.6), (0.6, 0.9), (0.9, 1.5)):
        row(f"true Re {lo}-{hi}", keep & (re_t > lo) & (re_t <= hi))

    print("\n  --- inside acceptance, by TRUE magnitude ---")
    for lo, hi in ((0, 23), (23, 24), (24, 25), (25, 26)):
        row(f"true mag {lo}-{hi}", keep & (mag_t > lo) & (mag_t <= hi))


if __name__ == "__main__":
    main()
