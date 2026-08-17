"""Is the emulator's per-bin deficit real, or is the RULER broken in those bins?

WHY THIS EXISTS. The error map (2026-08-05c) found the truth response NON-MONOTONIC in separation:
0.0503 close in, a dip to 0.0102 at 1.25-1.5", a second BUMP to 0.0371 at 2.5-3", then decay. The
2.5-4" region carries 57% of the total deficit, so the retrain would be aimed squarely at it. But a
bump where physics expects decay is exactly what a broken estimator looks like, and the ruler's
global 45-degree null being clean does NOT certify each bin: a positive artifact in one bin and a
negative one in another average to zero.

THE TEST. The null is the same projection with the neighbour's shear direction rotated 45 degrees.
Real blend response vanishes under that rotation; an additive artifact does not. So the null must be
consistent with zero IN EVERY BIN, not just overall. Any bin that fails is unusable, and optimising
the emulator against it would be fitting noise.

FIREWALL: reads a ruler npz (half-shear legs only). No constgold, no fitting, no promotion.
"""
from __future__ import annotations

import argparse

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    z = np.load(args.npz)
    if "null" not in z.files:
        raise SystemExit("npz has no per-row `null`; rerun eval_rblend_gap.py after 2026-08-05c")
    t, p, n = z["truth"], z["pred"], z["null"]
    good = np.isfinite(t) & np.isfinite(p) & np.isfinite(n)
    t, p, n, d = t[good], p[good], n[good], z["distance"][good]

    print(f"\n{'='*100}\nPER-BIN NULL TEST  {args.label}\n{'='*100}")
    print(f"N={len(t):,}    global null = {n.mean():+.5f} +- {n.std(ddof=1)/np.sqrt(len(n)):.5f}"
          f"  ({abs(n.mean())/(n.std(ddof=1)/np.sqrt(len(n))):.1f} sigma)")
    print("A clean GLOBAL null does not certify the bins -- opposite-sign artifacts cancel in the mean.")
    print(f"\n  {'separation':>18}{'truth':>9}{'emu':>9}{'NULL':>10}{'+-':>8}{'sig':>7}"
          f"{'verdict':>12}{'N':>12}")
    edges = [0, .5, .75, 1, 1.25, 1.5, 2, 2.5, 3, 4, 5, 7, 10]
    bad = []
    for i in range(len(edges) - 1):
        m = (d >= edges[i]) & (d < edges[i + 1])
        if m.sum() < 500:
            continue
        nn = n[m]
        sem = nn.std(ddof=1) / np.sqrt(m.sum())
        sig = abs(nn.mean()) / sem
        ok = sig < 3.0
        if not ok:
            bad.append((edges[i], edges[i + 1], nn.mean(), sig))
        print(f"  [{edges[i]:>7.2f},{edges[i+1]:>7.2f}){t[m].mean():>9.4f}{p[m].mean():>9.4f}"
              f"{nn.mean():>+10.5f}{sem:>8.5f}{sig:>6.1f}s"
              f"{'ok' if ok else 'FAILS':>12}{int(m.sum()):>12,}")

    print()
    if bad:
        print("BINS THAT FAIL THE NULL -- the ruler is not trustworthy there, and any emulator")
        print("deficit measured in them is not evidence of an emulator fault:")
        for lo, hi, v, s in bad:
            print(f'   {lo}-{hi}"  null={v:+.5f} ({s:.1f} sigma)')
    else:
        print("Every bin passes. The separation structure in truth -- including the 2.5-4 arcsec bump")
        print("-- is a property of the response, not an artifact of the estimator.")
    print("\nRBLEND_NULL_BINS_DONE", flush=True)


if __name__ == "__main__":
    main()
