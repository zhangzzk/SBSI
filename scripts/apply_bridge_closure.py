"""Apply the ngmix->constgold estimator bridge to a saved constgold-closure dump.

The V2 flow is trained on `measured_ngmix_g{1,2}` (det_meas half-shear), while constgold's truth
is its own `measured_e{1,2}_{plus,minus}` estimator.  Those two estimators do NOT share a shear
responsivity: `eval_estimator_match` (job 15201137) measured, on isolated true-cut primaries,

    constgold isolated R1 = +0.9091 +- 0.0051     ngmix isolated R_self = +0.8377 +- 0.0078
    ratio cg/ngmix        = 1.0853                (1.000 would mean the estimators matched)

so a model response in ngmix units has to be multiplied by that ratio before it can be divided
into a constgold R_sim.  An 8.5% factor is ~28x the |m| < 0.3% target, so the choice is not a
detail -- it decides the sign of the answer.  `eval_constgold_closure.py --bridge 1.0` leaves it
out; this script puts it back from the dump, which stores per-seed R_flow and the true magnitude,
so no second pass over the catalogue is needed for what is a multiply at the very end.

The bridge is itself measured on sims, with its own caveat: it compares constgold's antithetic
+-g extraction against the half-shear leg's forward 0->+g extraction, and those two extractions
are already known to disagree at the ~20% level on faint self-response even when the underlying
images are identical.  So `1.0853` is an upper-ish bound on a genuine estimator effect, and both
columns below are reported rather than one being declared correct.

Usage:
    python scripts/apply_bridge_closure.py --npz <dump>.npz
"""

import argparse

import numpy as np

MAG_EDGES = np.array([18.0, 24.0, 25.0, 26.0])
PER_MAG_BRIDGE = np.array([1.0546, 1.0847, 1.2364])   # estimator_match 15201137, per mag bin
SCALAR_BRIDGE = 1.0853
ACC_RE_MIN, ACC_MAG_MAX = 0.30, 26.0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", required=True)
    args = ap.parse_args()

    d = np.load(args.npz)
    Rsim, Rblend = d["Rsim"], d["Rblend"]
    Rf_seeds = d["Rflow_seeds"]          # (nseed, N), ngmix units
    mag, size, iso = d["mag"], d["size"], d["iso"]
    n, nseed = len(Rsim), Rf_seeds.shape[0]
    print(f"{args.npz}\n  N={n:,}  seeds={nseed}  dump bridge={float(d['bridge']):.4f}")

    bidx = np.clip(np.digitize(mag, MAG_EDGES) - 1, 0, len(PER_MAG_BRIDGE) - 1)
    bridges = {"none (1.0000)": np.ones(n),
               f"scalar ({SCALAR_BRIDGE})": np.full(n, SCALAR_BRIDGE),
               "per-mag": PER_MAG_BRIDGE[bidx]}

    acc = (size > ACC_RE_MIN) & (mag < ACC_MAG_MAX)
    bands = [("ISOLATED", iso), ("BLENDED", ~iso), ("ALL", np.ones(n, bool))]
    if not acc.all():
        bands += [("ACCEPTED", acc), ("REJECTED", ~acc)]

    print(f"\n  {'band':>9} {'N':>11} " + " ".join(f"{k:>22}" for k in bridges))
    for name, sel in bands:
        if not sel.any():
            continue
        rs = float(np.nanmean(Rsim[sel]))
        cells = []
        for br in bridges.values():
            ms = [(rs / np.nanmean((Rf_seeds[s] + Rblend)[sel] * br[sel]) - 1) * 100
                  for s in range(nseed)]
            rm = float(np.nanmean((Rf_seeds.mean(0) + Rblend)[sel] * br[sel]))
            cells.append(f"{(rs/rm-1)*100:+8.2f}% +-{np.std(ms):5.2f}")
        print(f"  {name:>9} {int(sel.sum()):>11,} " + " ".join(f"{c:>22}" for c in cells))
    print("\n  +- is the SEED-TO-SEED scatter of the 6-model ensemble, not the statistical error;\n"
          "  R_sim is common to every column, so the columns differ ONLY by the bridge.")


if __name__ == "__main__":
    main()
