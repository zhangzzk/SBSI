"""Why does the 45-degree null fail only when the PRIMARY box is imposed?

The unrestricted summed ruler has a clean null (-0.0021 +- 0.0028). Restricted to the emulator's
inference box it reads -0.0172 +- 0.0057 (3.0 sigma) -- an artifact worth 28% of the signal. There is
no mechanism for it: the box cuts on the PRIMARY's true magnitude and size, which cannot correlate
with the NEIGHBOUR's shear direction, so the null should be untouched. Either it is a fluctuation or
the masking is wrong, and those need separating before the box-restricted ruler is used again.

THREE CHECKS, cheapest first:
  (1) STRUCTURAL -- are `tmag`/`tre` really primary-level? The summed script treats them as constant
      within a primary. If they are not, the box mask removes individual NEIGHBOUR rows instead of
      whole primaries, which WOULD break the null: dropping neighbours non-randomly leaves the
      surviving directions correlated, and that is exactly an additive artifact.
  (2) CONTINUITY -- scan the null across box edges. A fluctuation wanders; a masking bug turns on
      sharply as soon as the cut starts biting.
  (3) LOCALISATION -- split the box null by separation and by magnitude.

FIREWALL: reads a ruler npz only. No constgold, no fitting, no correction.
"""
from __future__ import annotations

import argparse

import numpy as np


def cluster_stats(pid, npid, vals, mask):
    """Per-primary sums and the sem over primaries (the ruler's own error model)."""
    p = pid[mask]
    s = np.bincount(p, weights=vals[mask], minlength=npid)
    k = np.bincount(p, minlength=npid)
    have = k > 0
    s = s[have]
    return s.mean(), s.std(ddof=1) / np.sqrt(len(s)), len(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    pid = d["pid"].astype(np.int64)
    npid = int(pid.max()) + 1
    truth, null, tmag, tre, dist = d["truth"], d["null"], d["tmag"], d["tre"], d["dist"]
    base = np.isfinite(truth) & np.isfinite(null)

    print(f"\n{'='*100}\n(1) STRUCTURAL: are tmag/tre constant within a primary?\n{'='*100}")
    order = np.argsort(pid, kind="stable")
    ps = pid[order]
    edge = np.r_[True, ps[1:] != ps[:-1]]
    for nm, v in (("tmag", tmag), ("tre", tre)):
        vs = v[order]
        grp_min = np.minimum.reduceat(vs, np.flatnonzero(edge))
        grp_max = np.maximum.reduceat(vs, np.flatnonzero(edge))
        spread = grp_max - grp_min
        nvary = int((spread > 1e-9).sum())
        print(f"  {nm:>6}: primaries where it VARIES across neighbour rows: {nvary:,} of "
              f"{len(spread):,}   max spread = {spread.max():.6g}")
    print("  If either varies, the box mask drops individual NEIGHBOURS, not whole primaries, and")
    print("  the null failure is a masking bug rather than a fluctuation.")

    print(f"\n{'='*100}\n(2) CONTINUITY: null vs box edges (each cut applied alone)\n{'='*100}")
    print(f"  {'cut':>28}{'null':>11}{'+-':>9}{'sig':>7}{'truth':>10}{'primaries':>12}")
    def row(label, m):
        n, ns, npr = cluster_stats(pid, npid, null, m)
        t, _, _ = cluster_stats(pid, npid, truth, m)
        print(f"  {label:>28}{n:>+11.5f}{ns:>9.5f}{abs(n)/ns:>6.1f}s{t:>10.5f}{npr:>12,}")
    row("unrestricted", base)
    for mm in (28.0, 27.0, 26.5, 26.0, 25.72, 25.0):
        row(f"true mag < {mm}", base & (tmag < mm))
    for lo in (0.0, 0.3, 0.5):
        row(f"true Re > {lo}", base & (tre > lo))
    for hi in (10.0, 2.0, 1.5, 1.2):
        row(f"true Re < {hi}", base & (tre < hi))
    row("FULL v21 box", base & (tmag < 25.72) & (tre > 0.5) & (tre < 1.5))

    print(f"\n{'='*100}\n(3) LOCALISATION: inside the full box, null by separation\n{'='*100}")
    box = base & (tmag < 25.72) & (tre > 0.5) & (tre < 1.5)
    print(f"  {'separation':>28}{'null':>11}{'+-':>9}{'sig':>7}{'truth':>10}{'primaries':>12}")
    ed = [0, 1, 2, 3, 4, 5, 7]
    for i in range(len(ed) - 1):
        row(f'{ed[i]}-{ed[i+1]}"', box & (dist >= ed[i]) & (dist < ed[i + 1]))
    print("\nDIAG_BOX_NULL_DONE", flush=True)


if __name__ == "__main__":
    main()
