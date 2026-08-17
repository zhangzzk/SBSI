"""How much does the blend/self response label DEPEND on the shear amplitude it was measured at?

The response is even in `g` -- the SNC chord gives `R(g) = R_0 + c g^2` -- so a label measured at a
finite shear is contaminated by `c g^2`. That matters because the two models we compare are calibrated
at different amplitudes (flow #2 at g = 0.05, BlendEMU at g = 0.2) and `m` is evaluated on constgold
at g = 0.02. Moving flow #2 to g = 0.2 buys a 4x noise reduction and pays 16x the curvature term;
**this script measures the size of that payment instead of assuming it is small.**

METHOD. Build the SAME pair set at two shear amplitudes and compare. With `R(g1)` and `R(g2)`:

    c   = (R2 - R1) / (g2^2 - g1^2)
    R_0 = R1 - c g1^2                      (the g -> 0 limit, which is what we actually want)

then report the fractional contamination `c g^2 / R_0` at the amplitudes in play (0.02 / 0.05 / 0.2).

PAIRED, NOT UNPAIRED, WHERE POSSIBLE. Both legs render the SAME galaxies (`gals_info` is identical
across shears -- 699,568 rows, verified), so where the two pair sets share `(case, input_index)` the
difference can be taken per primary. That cancels scene-to-scene variance, which otherwise swamps a
difference this small. The script reports the matched fraction and falls back to an unpaired
comparison, clearly labelled, when the case ranges do not overlap.

WHAT THE ERROR BAR MEANS. The lever arm `g2^2 - g1^2` is tiny between 0.02 and 0.05 (0.0021), so `c`
is amplified by ~1/0.0021 = 476x relative to the response difference. A small difference therefore
implies a large `c`. The error is propagated through that amplification rather than quoted on the
difference, and if `c` is consistent with zero the script says so rather than reporting a number.

FIREWALL: half-shear legs only; constgold is never opened.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import pyarrow.feather as pf

KEY = ["case", "input_index"]


def per_primary(path, label):
    """Per-primary SUMMED blend response and 1/k-weighted self response.

    `R_blend` enters `m` as a SUM over each primary's neighbours, so that is the aggregation used
    here. The self response is a per-PRIMARY quantity repeated across the k rows, so it is averaged,
    not summed -- the same 1/k convention the trainer uses.
    """
    df = pf.read_table(path, columns=KEY + ["blend_truth", "self_truth", "k"]).to_pandas()
    n_pairs = len(df)
    g = df.groupby(KEY, sort=True)
    out = pd.DataFrame({
        "blend_sum": g["blend_truth"].sum(),
        "self_mean": g["self_truth"].mean(),
        "k": g["k"].first(),
    }).reset_index()
    print(f"  {label:>10}: {n_pairs:,} pairs -> {len(out):,} primaries, <k> = {out['k'].mean():.2f}")
    return out


def solve(r1, s1, g1, r2, s2, g2, name, paired_sem=None):
    """Solve R_0 and c from two amplitudes; report the contamination at each amplitude in play."""
    dg2 = g2 ** 2 - g1 ** 2
    c = (r2 - r1) / dg2
    r0 = r1 - c * g1 ** 2
    # Error on the DIFFERENCE -> error on c, amplified by 1/dg2. Paired sem when available.
    sd = paired_sem if paired_sem is not None else float(np.hypot(s1, s2))
    sc = sd / abs(dg2)
    print(f"\n  --- {name} ---")
    print(f"    R(g={g1})           = {r1:+.6f} +- {s1:.6f}")
    print(f"    R(g={g2})           = {r2:+.6f} +- {s2:.6f}")
    print(f"    difference          = {r2 - r1:+.6f} +- {sd:.6f}"
          f"   ({abs(r2 - r1) / sd:.1f} sigma)" if sd > 0 else "")
    print(f"    lever arm g2^2-g1^2 = {dg2:.6f}  (amplifies the difference by {1/abs(dg2):.0f}x)")
    print(f"    c                   = {c:+.4f} +- {sc:.4f}")
    print(f"    R_0 (g -> 0 limit)  = {r0:+.6f}")
    if abs(c) < 2 * sc:
        print(f"    ** c is consistent with ZERO at 2 sigma. The numbers below are UPPER BOUNDS")
        print(f"       built from the 2-sigma limit |c| < {2*sc:.4f}, not measurements. **")
        cuse = 2 * sc
    else:
        cuse = abs(c)
    print(f"    contamination c*g^2 / R_0, at the amplitudes in play:")
    for gg, what in ((0.02, "constgold eval"), (0.05, "flow #2 today"), (0.2, "BlendEMU / proposed")):
        frac = 100 * cuse * gg ** 2 / abs(r0) if r0 else np.nan
        print(f"      g = {gg:<5} ({what:<20}) : {frac:7.3f}%")
    return c, r0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairset", nargs=2, required=True, metavar=("LOW", "HIGH"))
    ap.add_argument("--shear", nargs=2, type=float, required=True, metavar=("G_LOW", "G_HIGH"))
    args = ap.parse_args()
    g1, g2 = args.shear
    if not g2 > g1 > 0:
        raise SystemExit("REFUSING: need 0 < G_LOW < G_HIGH")

    print(f"pair sets:\n  g={g1}: {args.pairset[0]}\n  g={g2}: {args.pairset[1]}\n")
    a = per_primary(args.pairset[0], f"g={g1}")
    b = per_primary(args.pairset[1], f"g={g2}")

    ca, cb = set(a["case"].unique()), set(b["case"].unique())
    print(f"\ncases: g={g1} -> {min(ca)}..{max(ca)} ({len(ca)});  "
          f"g={g2} -> {min(cb)}..{max(cb)} ({len(cb)});  shared: {len(ca & cb)}")

    def stats(df, col):
        v = df[col].to_numpy(float)
        v = v[np.isfinite(v)]
        return float(v.mean()), float(v.std(ddof=1) / np.sqrt(v.size))

    for col, name in (("blend_sum", "BLEND response (summed per primary)"),
                      ("self_mean", "SELF response (1/k-weighted per primary)")):
        m1, s1 = stats(a, col)
        m2, s2 = stats(b, col)
        paired_sem = None
        if ca & cb:
            j = a.merge(b, on=KEY, how="inner", suffixes=("_1", "_2"))
            d = (j[f"{col}_2"] - j[f"{col}_1"]).to_numpy(float)
            d = d[np.isfinite(d)]
            if d.size > 1000:
                paired_sem = float(d.std(ddof=1) / np.sqrt(d.size))
                # Use the PAIRED difference itself: it is the unbiased low-variance estimate.
                m2 = m1 + float(d.mean())
                print(f"\n  [{col}] PAIRED on {d.size:,} shared primaries "
                      f"({100*d.size/min(len(a),len(b)):.1f}% of the smaller set); "
                      f"paired sem {paired_sem:.6f} vs unpaired {np.hypot(s1,s2):.6f} "
                      f"({np.hypot(s1,s2)/paired_sem:.1f}x tighter)")
        else:
            print(f"\n  [{col}] NO SHARED CASES -- unpaired comparison, scene variance NOT cancelled.")
        solve(m1, s1, g1, m2, s2, g2, name, paired_sem)

    print("\nMEASURE_CURVATURE_DONE", flush=True)


if __name__ == "__main__":
    main()
