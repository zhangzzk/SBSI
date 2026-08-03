"""Does the DATA show the angular structure the flow has?  The question the flow test raised.

WHERE THIS SITS.  `check_flow_isotropy.py` found the flow's pass fraction is not flat on
rings of constant `|e|` -- 65-79x its noise at large `|e|`, m=2 dominant near `|e| ~ 0.3-0.45`
and m=4 above.  The azimuthal-average experiment then showed that structure is the ENTIRE
cause of the `<s>_sel` anomaly: force `Pi` to depend on `|e|` alone and `<s>_sel` collapses to
machine zero, exactly as §5B.2 says it must for an isotropic model.

But the same experiment showed the structure is not spurious noise the estimator would be
better off without: removing it makes `d(m)` WORSE, from +0.640% to -0.635% at cut 0.6 and
from +0.447% to +3.446% at cut 0.4.  The estimator is using the angular structure, and using
it roughly correctly.  So `d(m)` is a 1-3% lever on the flow's angular structure -- far above
the 0.3% deliverable -- and the question is no longer "is it there" but "IS IT RIGHT".

THE TEST.  `Pi` is a statement about the simulation: the fraction of galaxies with a given
TRUE shape whose MEASURED shape passes the cut.  The catalogue has both, so the same quantity
can be measured directly from the data with no flow involved, and the two compared harmonic by
harmonic.  m=4 is expected in both -- square pixels and square postage stamps genuinely give
the measurement C4 rather than full SO(2) symmetry.  m=2 is the discriminating one: a square
grid cannot produce it, so if the flow has m=2 and the data do not, the flow has learned an
asymmetry that is not in the simulation, and it is biasing `m` at the percent level.

WHAT THIS CANNOT DO, stated plainly.  The flow's `Pi` substitutes one shape into every row and
marginalises over the rest of the scene, so its population is identical at every angle by
construction.  The data cannot do that -- at fixed `(|e|, phi)` the real rows differ in
magnitude, size and neighbours, and any genuine astrophysical alignment between shape angle
and those properties would show up here as an angular signal that is real but is NOT the
measurement anisotropy we are testing for.  So a data m=2 detection is suggestive, not
conclusive; a data m=2 NULL against a clear flow m=2 is the stronger inference.  The angular
binning is also much coarser here for a reason: the binomial error on a ring of `N` rows is
`~sqrt(2/N)*sqrt(p(1-p))`, so resolving a 2e-3 harmonic needs the whole catalogue per ring.
"""

import argparse
import os
import sys

import numpy as np

SBSI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (SBSI_ROOT, os.path.join(SBSI_ROOT, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from eval_score_response import G0_CAT, load_g0  # noqa: E402

TRUE1, TRUE2 = "e1_input_rot0_p", "e2_input_rot0_p"
MEAS1, MEAS2 = "NGMIX_G1", "NGMIX_G2"


def harmonic(pass_flag, phi, m):
    """`(amplitude, error)` of the `cos(m phi)/sin(m phi)` part of the pass probability.

    Estimated per row rather than per angular bin, so the answer does not depend on a bin
    choice: `c = 2*mean(w*cos(m phi))` is the projection, and its error is the usual
    standard error of that mean.
    """
    w = pass_flag.astype(float)
    n = len(w)
    c, s = 2.0 * np.mean(w * np.cos(m * phi)), 2.0 * np.mean(w * np.sin(m * phi))
    ec = 2.0 * np.std(w * np.cos(m * phi), ddof=1) / np.sqrt(n)
    es = 2.0 * np.std(w * np.sin(m * phi), ddof=1) / np.sqrt(n)
    amp = np.hypot(c, s)
    err = np.hypot(c * ec, s * es) / max(amp, 1e-12)
    return amp, err, c, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--g0-catalogue", default=G0_CAT)
    ap.add_argument("--max-rows", type=int, default=2_000_000)
    ap.add_argument("--cut-abs-ehat", type=float, default=0.6)
    ap.add_argument("--edges", default="0.05,0.15,0.25,0.375,0.525,0.675,0.825,0.95",
                    help="|e_true| ring edges; wider than the flow test's rings because the "
                         "data have finite rows per ring and the harmonics are ~1e-3")
    args = ap.parse_args()

    df = load_g0(args.g0_catalogue, args.max_rows)
    missing = [c for c in (TRUE1, TRUE2, MEAS1, MEAS2) if c not in df.columns]
    if missing:
        raise SystemExit(f"catalogue lacks {missing}; has {list(df.columns)[:40]}")

    e1, e2 = df[TRUE1].to_numpy(float), df[TRUE2].to_numpy(float)
    m1, m2 = df[MEAS1].to_numpy(float), df[MEAS2].to_numpy(float)
    ok = np.isfinite(e1) & np.isfinite(e2) & np.isfinite(m1) & np.isfinite(m2)
    e1, e2, m1, m2 = e1[ok], e2[ok], m1[ok], m2[ok]
    r = np.hypot(e1, e2)
    phi = np.arctan2(e2, e1)
    passed = np.hypot(m1, m2) < args.cut_abs_ehat

    edges = [float(x) for x in args.edges.split(",")]
    print(f"rows={len(r):,} finite of {len(ok):,}   cut |xhat| < {args.cut_abs_ehat}")
    print(f"overall pass fraction {passed.mean():.4f}")
    print("\n  |e| ring          N     <pass>      A2 +/- err   nsig       A4 +/- err   nsig")
    print("  " + "-" * 76)
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (r >= lo) & (r < hi)
        n = int(sel.sum())
        if n < 1000:
            print(f"  {lo:.3f}-{hi:.3f}  {n:9,}   (too few rows)")
            continue
        a2, e2r, _, _ = harmonic(passed[sel], phi[sel], 2)
        a4, e4r, _, _ = harmonic(passed[sel], phi[sel], 4)
        print(f"  {lo:.3f}-{hi:.3f}  {n:9,}   {passed[sel].mean():.4f}   "
              f"{a2:8.2e} +/- {e2r:7.1e} {a2/max(e2r,1e-12):5.1f}   "
              f"{a4:8.2e} +/- {e4r:7.1e} {a4/max(e4r,1e-12):5.1f}")

    print("\n  Compare against the FLOW's amplitudes from check_flow_isotropy.py at the same")
    print("  |e| (cut 0.6): A2 ~ 3.9e-4 (|e|=0.30), 2.2e-3 (0.45), 6.5e-4 (0.60), 1.9e-3")
    print("  (0.75), 4.5e-3 (0.90);  A4 ~ 6.5e-4, 2.2e-4, 4.8e-3, 1.4e-2, 2.4e-2.")
    print("  m=4 present in BOTH is expected -- square pixels and square stamps really do")
    print("  give the measurement C4 symmetry.  m=2 in the flow but NOT in the data would")
    print("  mean the flow learned an asymmetry the simulation does not have.")
    print("  CAVEAT: at fixed (|e|, phi) the real rows differ in magnitude, size and")
    print("  neighbours, so a data m=2 detection could be astrophysical alignment rather")
    print("  than measurement anisotropy.  A data NULL is the stronger inference.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
