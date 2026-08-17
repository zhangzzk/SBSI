"""Place pin-grid edges on EQUAL RELATIVE RESPONSE CHANGE instead of equal COUNT.

WHY. The fiducial pin grid (`response_target_crowd_rblend_snc_c0-99_6x6x5_dom.npz`) uses quantile
edges, i.e. equal COUNT per bin. Measured from that grid's own contents, equal-count happens to
equalise the ABSOLUTE response step per cell (0.25, 0.23, 0.21, 0.17, 0.15 across true mag) -- but
`m` is a RATIO, so what a cell's single pinned value costs is the RELATIVE response variation inside
it, and that climbs monotonically toward faint (20% -> 37% across the same bins). In size it is far
worse: R rises 0.4375 -> 0.7016 across the FIRST cell alone (+60%, 0.055" wide) and is flat to
within 3% above Re = 0.42, so five of six size bins are spent on a flat region while the one steep
bin sits directly on the Re > 0.3 acceptance cut.

WHAT THIS DOES. Reads a FINE equal-count response profile (built by the certified
`compute_response_target_blend.py`, so the projection/estimator/cut chain is identical to the
fiducial target) and places `--n-bins` edges at equal increments of cumulative |d log R| -- total
variation of the log response. That is the criterion the metric actually cares about, and it
automatically concentrates bins where the response turns fastest.

THE STATISTICAL GUARD, AND WHY IT IS REPORTED NOT HIDDEN. Pure equal-d(log R) edges can produce a
thin cell, which is how the 8x8 grid was rejected (611 per cell against the 6x6 grid's 1,875). So
the placement criterion is a blend

    crit(x) = (1 - alpha) * TV_norm(x) + alpha * count_norm(x)

with alpha raised from 0 in steps until every bin clears `--min-frac` of the sample. alpha = 0 is
pure response-driven; alpha = 1 reproduces the existing equal-count edges exactly. The final alpha
is PRINTED, so "how far we had to retreat toward equal-count to stay statistically safe" is a
reported quantity rather than a silent tuning knob.

SMOOTHING. |d log R| of a noisy profile is dominated by noise (total variation is not a
noise-robust functional), so log R is boxcar-smoothed over `--smooth` fine bins before the cumulative
sum. The raw and smoothed profiles are both printed so the effect is auditable. This changes only
WHERE EDGES GO; it never enters a reported response value.

FIREWALL. Input is the half-shear g=0.05 ruler profile. constgold is not read, and no `m` is
consulted -- the edge choice is made entirely on the ruler, before any acceptance number exists.
"""
from __future__ import annotations

import argparse

import numpy as np


def _profile(path, axis):
    """(edges, R, counts) collapsed onto `axis` from a fine 1-D target npz."""
    z = np.load(path, allow_pickle=True)
    R, C = z["Rsim"].astype(float), np.asarray(z["counts"], float)
    if axis == "flux":
        edges, R, C = z["edges_flux"].astype(float), R[:, 0, 0], C[:, 0, 0]
    else:
        edges, R, C = z["edges_size"].astype(float), R[0, :, 0], C[0, :, 0]
    if R.ndim != 1 or len(R) != len(edges) - 1:
        raise SystemExit(f"REFUSING: {path} is not a 1-D profile on axis={axis} "
                         f"(Rsim {z['Rsim'].shape}, edges {edges.shape}). Build it with the other "
                         "two axes set to 1 bin.")
    return edges, R, C


def _boxcar(y, w):
    if w <= 1:
        return y.copy()
    pad = w // 2
    return np.convolve(np.pad(y, pad, mode="edge"), np.ones(w) / w, mode="valid")[:len(y)]


def place(edges, R, C, n_bins, min_frac, smooth):
    """Edges at equal increments of a TV/count blend; returns (edges, alpha, report rows)."""
    if not np.all(np.isfinite(R)) or np.any(R <= 0):
        raise SystemExit("REFUSING: the response profile has non-finite or non-positive cells, so "
                         "log R is undefined. Rebuild the profile with fewer bins or a higher "
                         "--min-count rather than clipping here.")
    # Both criteria live on the EDGE grid so they are directly comparable. log R is piecewise
    # constant on bins, so its variation happens AT interior edges: edge k accumulates the jumps
    # between bins 0..k-1, giving tv[0] = tv[1] = 0 and tv[-1] = the total variation.
    jumps = np.abs(np.diff(_boxcar(np.log(R), smooth)))       # len n-1, one per interior edge
    tv = np.concatenate([[0.0, 0.0], np.cumsum(jumps)])       # len n+1, on edges
    cc = np.concatenate([[0.0], np.cumsum(C)])                # len n+1, on edges
    tv_n = tv / tv[-1] if tv[-1] > 0 else cc / cc[-1]
    cc_n = cc / cc[-1]

    total = float(C.sum())
    for alpha in np.arange(0.0, 1.0001, 0.05):
        crit = np.maximum.accumulate((1.0 - alpha) * tv_n + alpha * cc_n)
        inner = np.interp(np.linspace(0, 1, n_bins + 1)[1:-1], crit, edges)
        new = np.concatenate([[edges[0]], inner, [edges[-1]]])
        if np.any(np.diff(new) <= 0):
            continue
        frac = np.diff(np.interp(new, edges, cc)) / total
        if frac.min() >= min_frac:
            return new, float(alpha), frac
    raise SystemExit(f"REFUSING: no blend up to alpha=1 satisfies --min-frac {min_frac}. That means "
                     f"{n_bins} bins cannot be filled at this floor; ask for fewer bins.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", required=True, help="fine 1-D target npz from the certified builder")
    ap.add_argument("--axis", required=True, choices=["flux", "size"])
    ap.add_argument("--n-bins", type=int, default=6)
    ap.add_argument("--min-frac", type=float, default=0.04,
                    help="minimum fraction of the sample per bin, on THIS AXIS' MARGINAL. "
                         "WARNING: this does NOT bound the 3-D cell count. The axes are correlated "
                         "-- small galaxies concentrate in particular mag/crowd cells -- so the "
                         "marginal floor over-estimates the joint occupancy by ~10x in practice: "
                         "--min-frac 0.04 produced a 3-D min cell of 956 against a 1,875 floor "
                         "(job 15499440). ALWAYS build the target and assert on `counts`.min() "
                         "rather than trusting this guard.")
    ap.add_argument("--smooth", type=int, default=5, help="boxcar width on log R before the TV sum")
    args = ap.parse_args()

    edges, R, C = _profile(args.profile, args.axis)
    print(f"fine profile: {len(R)} bins on {args.axis}, span [{edges[0]:.4f}, {edges[-1]:.4f}], "
          f"R {R.min():.4f}..{R.max():.4f}, N_eff {C.sum():,.0f}")
    Rs = _boxcar(np.log(R), args.smooth)
    print(f"total variation of log R: raw {np.abs(np.diff(np.log(R))).sum():.4f} -> "
          f"smoothed {np.abs(np.diff(Rs)).sum():.4f} (boxcar {args.smooth}); the raw value is "
          "noise-inflated, which is why edges are placed on the smoothed curve")

    new, alpha, frac = place(edges, R, C, args.n_bins, args.min_frac, args.smooth)
    print(f"\nalpha = {alpha:.2f}  (0 = pure equal-d(logR), 1 = pure equal-count)")
    if alpha > 0:
        print(f"  the count floor BOUND: pure response-driven edges left a bin under "
              f"{args.min_frac:.0%}, so the placement was blended {alpha:.0%} toward equal-count.")

    # the equal-count edges this replaces, from the SAME cumulative-count curve (the alpha = 1
    # case), so the two columns below differ only in the placement criterion, not in the input
    cc = np.concatenate([[0.0], np.cumsum(C)])
    old = np.interp(np.linspace(0, 1, args.n_bins + 1), cc / cc[-1], edges)
    print(f"\n{'bin':>4}  {'NEW edges':>22}  {'width':>7}  {'N frac':>7}   | equal-count edges")
    for i in range(args.n_bins):
        print(f"{i:>4}  [{new[i]:8.4f},{new[i+1]:8.4f}]  {new[i+1]-new[i]:7.4f}  {frac[i]:6.2%}   "
              f"| [{old[i]:.4f},{old[i+1]:.4f}]")
    print("\n--%s-edges %s" % (args.axis, ",".join(f"{v:.6f}" for v in new)))


if __name__ == "__main__":
    main()
