"""Does the HALF-SHEAR target reproduce the SHAPE of constgold's demand across true size?

The size tilt (WORKLOG 2026-08-05z) is `R_flow - (R_sim - R_blend)` measured on constgold. The
subtracted term makes the demand MODEL-dependent: CONVENTIONS.md 6b says constgold cannot isolate a
self response at all, so `R_sim - R_blend` is a model-subtracted quantity, not a measurement. If that
demand curve is wrong in SHAPE across size, the tilt belongs to the instrument, not the flow.

This reads the half-shear response target -- built from half-shear legs only, no constgold anywhere
in it -- marginalises its `Rsim` over the flux and crowding axes with the cell counts as weights, and
prints the resulting demand-vs-size curve beside the constgold one.

WHAT MAY BE COMPARED, AND WHAT MAY NOT. The target is forward at |g| = 0.05, constgold is antithetic;
CONVENTIONS.md 6c forbids mixing them. 05u measured that term and it is `-3.41% +- 3.82` at that leg,
consistent with zero, AND flat in resolution. A size-INDEPENDENT offset cannot create a SLOPE, so:

  * the SHAPE across size bins may be compared -- that is the whole point here;
  * the ABSOLUTE levels may NOT, and this script never differences them. It normalises each curve by
    its own count-weighted mean and compares only the normalised shapes.

The two curves also sit on different catalogues, so a bin's galaxy mix is not identical between them
even at the same true size. That limits this to a shape check, which is what it claims to be.

FIREWALL. The build is half-shear only. The constgold numbers below are a labelled COMPARISON --
pasted in with their provenance, per AGENTS.md "Numerical Integrity", which permits a
derived-and-reported comparison but not a silent pasted correction. Nothing here is fitted, corrected
or fed back into any model.
"""
from __future__ import annotations

import argparse

import numpy as np

# Constgold demand `needed = R_sim - R_blend`, per true-size bin, from job 15545111 (16 fiducial
# dom6x6 seeds, `_ho` R_blend). PROVENANCE-STAMPED COMPARISON INPUT -- printed alongside, never
# combined into a corrected number.
CONSTGOLD = [
    (0.30, 0.35, 0.3926), (0.35, 0.40, 0.6745), (0.40, 0.45, 0.7572), (0.45, 0.50, 0.8062),
    (0.50, 0.55, 0.8221), (0.55, 0.60, 0.8306), (0.60, 0.70, 0.8218), (0.70, 0.80, 0.8143),
    (0.80, 1.00, 0.8000), (1.00, 1.20, 0.8013), (1.20, 1.50, 0.7839),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=True)
    R, C = z["Rsim"], z["counts"]
    edges = z["edges_size"]
    # marginalise over flux (axis 0) and crowd (axis 2), weighting by cell counts
    ax = (0, 2)
    w = np.where(np.isfinite(R), C, 0.0)
    num = np.nansum(np.where(np.isfinite(R), R * w, 0.0), axis=ax)
    den = w.sum(axis=ax)
    tgt = np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)

    print(f"\nhalf-shear target: {args.npz.split('/')[-1]}")
    print(f"  domain stamp: {str(z['domain'])[:90]}")
    print(f"  global_R = {float(z['global_R']):.4f}   size bins = {len(edges)-1}")
    print(f"  cells with a finite Rsim: {int(np.isfinite(R).sum()):,} of {R.size:,}")

    n = min(len(tgt), len(CONSTGOLD))
    t = np.array(tgt[:n], float)
    c = np.array([x[2] for x in CONSTGOLD[:n]], float)
    wt = np.array(den[:n], float)
    ok = np.isfinite(t) & (wt > 0)
    if ok.sum() < 3:
        raise SystemExit("too few populated size bins to compare shapes")

    # normalise each curve by its OWN count-weighted mean: levels are not comparable across
    # extraction conventions, shapes are.
    tn = t / np.average(t[ok], weights=wt[ok])
    cn = c / np.average(c[ok], weights=wt[ok])

    print(f"\n{'='*94}\nDEMAND vs TRUE SIZE -- SHAPE ONLY (each curve divided by its own weighted "
          f"mean)\n{'='*94}")
    print(f"  {'Re bin':>16}{'half-shear':>12}{'constgold':>11}{'   |':>4}"
          f"{'HS/mean':>10}{'CG/mean':>10}{'shape diff':>12}{'weight':>14}")
    for i in range(n):
        lo, hi, _ = CONSTGOLD[i]
        if not ok[i]:
            print(f"  [{lo:>5.2f},{hi:>5.2f}){'--':>12}{c[i]:>11.4f}")
            continue
        print(f"  [{lo:>5.2f},{hi:>5.2f}){t[i]:>12.4f}{c[i]:>11.4f}{'   |':>4}"
              f"{tn[i]:>10.4f}{cn[i]:>10.4f}{tn[i]-cn[i]:>+12.4f}{wt[i]:>14,.0f}")

    d = tn[ok] - cn[ok]
    print(f"\n  weighted mean shape difference = {np.average(d, weights=wt[ok]):+.4f} "
          f"(0 by construction, up to weighting)")
    print(f"  spread of the shape difference = {d.std(ddof=1):.4f}")
    # A tilt in the SHAPE difference is what would say the two instruments disagree about size.
    x = np.array([0.5 * (CONSTGOLD[i][0] + CONSTGOLD[i][1]) for i in range(n)])[ok]
    A = np.vstack([np.ones_like(x), x]).T
    W = wt[ok]
    cov = np.linalg.inv(A.T @ (A * W[:, None]))
    p = cov @ (A.T @ (W * d))
    print(f"  slope of (half-shear - constgold) shape across size = {p[1]:+.4f} per arcsec")
    print("\n  READ: a slope near zero means the two instruments agree about how demand varies with")
    print("  size, so the flow's tilt against constgold is NOT an artifact of the model-subtracted")
    print("  demand. A large slope means they disagree and the tilt cannot be attributed to the flow")
    print("  until that is resolved. Absolute levels are NOT compared -- different conventions.")
    print("\nCOMPARE_SIZE_DEMAND_DONE", flush=True)


if __name__ == "__main__":
    main()
