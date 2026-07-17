"""Is the apparent increase of |m| at more aggressive cuts just statistical NOISE?

m(cut) = R_sim(0.02,cut)/R_sim(0.05,cut) - 1 is dominated by intrinsic shape scatter divided
by the small shear: its error ~ sigma_e/(g sqrt(N)).  Aggressive cut -> fewer galaxies N ->
bigger error.  We test this empirically: split each catalogue into K independent folds, recompute
m per fold (using a COMMON cut threshold), and report the across-fold mean and scatter.

Reading:
  * mean_m consistent with 0 within (scatter/sqrt(K))  -> no real signal.
  * across-fold scatter GROWS as the cut tightens, matching sigma_e/(g sqrt(N)) -> the
    cut-dependence is pure noise (fewer galaxies), not a brightness-dependent bias.
"""
import argparse
import os
import sys

import numpy as np

SBSI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SBSI_ROOT not in sys.path:
    sys.path.insert(0, SBSI_ROOT)

from archive.sim_direct_cut_bias import load  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cat-005", required=True)
    ap.add_argument("--cat-002", required=True)
    ap.add_argument("--max-rows", type=int, default=8_000_000)
    ap.add_argument("--folds", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    print("Loading g=0.05 ..."); p5, snr5, tm5, gm5 = load(args.cat_005, args.max_rows, args.seed)
    print("Loading g=0.02 ..."); p2, snr2, tm2, gm2 = load(args.cat_002, args.max_rows, args.seed)
    rng = np.random.default_rng(args.seed)
    f5 = rng.integers(0, args.folds, size=len(p5))
    f2 = rng.integers(0, args.folds, size=len(p2))

    fracs = [1.0, 0.6, 0.4, 0.2, 0.1]
    for var in ("measured_snr", "true_mag"):
        key5, key2 = (snr5, snr2) if var == "measured_snr" else (tm5, tm2)
        print(f"\n================ CUT VARIABLE: {var}  (K={args.folds} folds) ================")
        print(f"{'keep_f':>7} {'mean_m%':>8} {'fold_sigma%':>11} {'sigma_mean%':>11} "
              f"{'z':>6} {'N05/fold':>10}  verdict")
        for f in fracs:
            if var == "measured_snr":
                thr5 = np.quantile(snr5, 1 - f); k5 = key5 >= thr5
                thr2 = np.quantile(snr2, 1 - f); k2 = key2 >= thr2
            else:
                thr5 = np.quantile(tm5, f); k5 = key5 <= thr5
                thr2 = np.quantile(tm2, f); k2 = key2 <= thr2
            ms = []
            npf = []
            for kf in range(args.folds):
                s5 = k5 & (f5 == kf); s2 = k2 & (f2 == kf)
                if s5.sum() < 100 or s2.sum() < 100:
                    continue
                R5 = (p5[s5] / gm5[s5]).mean()
                R2 = (p2[s2] / gm2[s2]).mean()
                ms.append(R2 / R5 - 1.0)
                npf.append(int(s5.sum()))
            ms = np.array(ms)
            mean = ms.mean(); fsig = ms.std(ddof=1); smean = fsig / np.sqrt(len(ms))
            z = mean / smean if smean > 0 else 0.0
            verdict = "consistent w/ 0 (noise)" if abs(z) < 2 else "**>2sigma**"
            print(f"{f:>7.2f} {mean*100:>+8.2f} {fsig*100:>11.2f} {smean*100:>11.2f} "
                  f"{z:>+6.1f} {int(np.mean(npf)):>10,}  {verdict}")
        print("  -> if fold_sigma GROWS as keep_f shrinks and z stays <2, the cut-trend is noise.")


if __name__ == "__main__":
    main()
