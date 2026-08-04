"""SCORE THE EDGES-ONLY PILOT against the prediction recorded in WORKLOG 2026-08-03m.

WHAT IS COMPARED. Fiducial dom6x6 vs pilot domB6 (mf08 size edges), on IDENTICAL rows, against the
SAME half-shear self-response truth, using the SAME THREE SEEDS (501/502/503) on both sides. The
fiducial dump carries 16 seeds and the pilot 3; taking the common three removes seed count as a
confound, so the only difference left is the size edges of the response pin.

THE PREDICTION BEING SCORED (recorded before the run):
  1. small-size self-response conditional IMPROVES on the fiducial's -3.42 +- 1.10% (Re <= 0.386")
  2. the within-cell ramp in the FIDUCIAL's first pin cell (-24.9 +- 6.6 %/cell) SHRINKS
  3. if both are null, pin cell resolution is exonerated and the suspects become the flow's own
     smoothness prior and NLL dominance

The ramp is measured on the FIDUCIAL cell edges [0.300, 0.3552] for BOTH models. The pilot's own
edges differ, so scoring it on its own cells would compare two different questions; the fiducial cell
is the fixed ruler on which "did the over-smoothing get better" is a well-posed question.

Errors are bootstrapped over CASES (the unit of independence), not rows.

FIREWALL: half-shear self-response only; constgold is not read and no `m` is computed. 3 seeds cannot
report an `m` under the seed convention, which is why the pilot is scored here and not on constgold.
"""
from __future__ import annotations

import argparse

import numpy as np
import pyarrow.feather as pf

FID = "results/halfshear_selfresp.feather"
PILOT = ("/project/ls-gruen/users/zekang.zhang/sbsi_caches/pin_realloc/"
         "halfshear_selfresp_domB6.feather")
SEEDS = [501, 502, 503]
FID_CELL0 = (0.300, 0.35515322)   # fiducial dom6x6 first size cell


def load(path, seeds):
    df = pf.read_table(path, memory_map=True).to_pandas()
    cols = [f"R_flow_s{s}" for s in seeds]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(f"REFUSING: {path} missing {missing}")
    return df, np.mean([df[c].to_numpy(float) for c in cols], axis=0)


def region(y, mod, keep, case, rng, nboot=300):
    ok = keep & np.isfinite(y) & np.isfinite(mod)
    cen = 100.0 * (float(np.mean(mod[ok])) / float(np.mean(y[ok])) - 1.0)
    uc = np.unique(case[ok])
    bs = []
    for _ in range(nboot):
        pick = rng.choice(uc, size=len(uc), replace=True)
        m = np.isin(case, pick) & ok
        bs.append(100.0 * (float(np.mean(mod[m])) / float(np.mean(y[m])) - 1.0))
    return cen, float(np.std(bs)), int(ok.sum())


def ramp(y, mod, keep, re, case, rng, nboot=300, nu=5):
    lo, hi = FID_CELL0
    ok = keep & np.isfinite(y) & np.isfinite(mod) & (re >= lo) & (re < hi)

    def slope(mask):
        u = (re[mask] - lo) / (hi - lo)
        r = 100.0 * (mod[mask] - y[mask]) / np.mean(y[mask])
        e = np.linspace(0, 1, nu + 1)
        idx = np.clip(np.digitize(u, e) - 1, 0, nu - 1)
        vals = np.array([r[idx == b].mean() if (idx == b).any() else np.nan for b in range(nu)])
        uu = 0.5 * (e[:-1] + e[1:])
        g = np.isfinite(vals)
        return (np.polyfit(uu[g], vals[g], 1)[0] if g.sum() > 1 else np.nan), vals

    s, vals = slope(ok)
    uc = np.unique(case[ok])
    bs = []
    for _ in range(nboot):
        pick = rng.choice(uc, size=len(uc), replace=True)
        m = np.isin(case, pick) & ok
        if m.sum() > 100:
            bs.append(slope(m)[0])
    return s, float(np.std(bs)), vals, int(ok.sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fiducial", default=FID)
    ap.add_argument("--pilot", default=PILOT)
    args = ap.parse_args()

    fdf, fmod = load(args.fiducial, SEEDS)
    pdf, pmod = load(args.pilot, SEEDS)
    if len(fdf) != len(pdf):
        raise SystemExit(f"REFUSING: row counts differ ({len(fdf):,} vs {len(pdf):,})")
    for k in ("case", "input_index"):
        if not np.array_equal(fdf[k].to_numpy(), pdf[k].to_numpy()):
            raise SystemExit(f"REFUSING: `{k}` differs between the two dumps; rows are not aligned")
    y = fdf["r_sim_self"].to_numpy(float)
    if not np.allclose(y, pdf["r_sim_self"].to_numpy(float), equal_nan=True):
        raise SystemExit("REFUSING: the truth column differs between the two dumps")
    re = fdf["Re_input_p"].to_numpy(float)
    sn = fdf["SN"].to_numpy(float)
    case = fdf["case"].to_numpy(np.int64)
    print(f"aligned: {len(fdf):,} rows, seeds {SEEDS}, truth <R_self>={np.nanmean(y):.5f}")
    print(f"<R_flow>  fiducial {np.nanmean(fmod):.5f}   pilot {np.nanmean(pmod):.5f}")

    rng = np.random.default_rng(0)
    print("\n=== REGION RESIDUALS (model/truth - 1), bootstrapped over cases ===")
    for lab, keep in (("ALL", np.ones(len(y), bool)),
                      ("small size Re <= 0.386", re <= 0.386),
                      ("faint  S/N <= 13.09", sn <= 13.09)):
        fc, fs, n = region(y, fmod, keep, case, rng)
        pc, ps, _ = region(y, pmod, keep, case, rng)
        d = pc - fc
        ds = np.hypot(fs, ps)
        print(f"  {lab:<24} N={n:>9,}   fiducial {fc:+6.2f} +- {fs:.2f} %   "
              f"pilot {pc:+6.2f} +- {ps:.2f} %   change {d:+6.2f} +- {ds:.2f} "
              f"({'resolved' if abs(d) > 2 * ds else 'not resolved'})")

    print(f"\n=== WITHIN-CELL RAMP in the FIDUCIAL first size cell "
          f"[{FID_CELL0[0]:.3f},{FID_CELL0[1]:.4f}] ===")
    for lab, mod in (("fiducial", fmod), ("pilot   ", pmod)):
        s, sd, vals, n = ramp(y, mod, np.ones(len(y), bool), re, case, rng)
        print(f"  {lab}: N={n:,}  residual across cell: " + " ".join(f"{v:+6.2f}" for v in vals))
        print(f"            slope = {s:+.2f} +- {sd:.2f} %/cell "
              f"({'RAMP' if abs(s) > 2 * sd else 'no ramp'})")

    print("\nNOTE: 3 seeds, so no `m` is quoted here (the convention requires 16). This scores the\n"
          "half-shear SELF-response conditional, which is where the defect was measured.")
    print("\nPILOT_SCORE_DONE")


if __name__ == "__main__":
    main()
