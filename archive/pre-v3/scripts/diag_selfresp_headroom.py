"""IS FLOW #1's SMALL/FAINT FAILURE RECOVERABLE, OR AN INFORMATION LIMIT?

THE QUESTION (owner, 2026-08-03). Figure 5 shows flow #1 alone missing the half-shear SELF-response
at small size (+6.94% at Re ~ 0.31", -6.16% / -6.73% in the next two bins) and at faint S/N (-3.4 /
-5.1 / -4.4% in the three faintest bins), all resolved above the truth s.e. and with NO blend term
anywhere in the comparison. That is a clean flow #1 defect. What figure 5 cannot say is whether it is
FIXABLE -- a model that could be trained away -- or an information limit of the features the flow
conditions on.

THE TEST. Same rows, same truth (`r_sim_self`), same binning. Fit a gradient-boosted regressor on the
features fig 5 itself carries and compare its per-bin residual against the flow's:

    flow #1        16-seed mean R_flow, conditioned on 8 true properties
    headroom fit   trees on (S/N, Re_input_p, nbr_flux_near) -> r_sim_self, HELD OUT BY CASE

Squared-error regression converges to E[truth | features], so the fit is the best any model of these
features can do. The comparison is therefore:

    fit beats flow at small/faint  ->  RECOVERABLE. The flow is leaving information on the table and
                                       a better-trained flow #1 can close it.
    fit ties flow                  ->  an information limit; more flow #1 training cannot help and
                                       the residual needs a different observable or more features.

THE FIT IS DELIBERATELY HANDICAPPED. It sees 3 features where the flow sees 8, and it is held out by
case while the flow trained on a different catalogue entirely. So "the fit beats the flow" is a
CONSERVATIVE conclusion -- a 3-feature model outscoring an 8-feature one is strong evidence of
recoverable headroom, while a tie is weak evidence of a limit and should be read as inconclusive
rather than as proof.

THIS ALSO TESTS AN EARLIER CLAIM OF MINE. WORKLOG 2026-08-03j attributed the ~+10% small-size floor on
constgold to R_blend or additivity, on the grounds that four self-response models converged there. But
all four were fitted to the SAME SNC label, and models fitted to a common label converge whether or
not that label is right -- convergence is not correctness. If the fit here lands near zero at small
size against the RULER's own truth, the label is fine and the constgold floor really is downstream of
the self response. If the fit is ALSO off at small size, the label is suspect and that attribution has
to be withdrawn.

FIREWALL: half-shear self-response only. constgold is not read.
"""
from __future__ import annotations

import argparse

import numpy as np
import pyarrow.feather as pf
from sklearn.ensemble import HistGradientBoostingRegressor

FEATS = ["SN", "Re_input_p", "nbr_flux_near"]


def table(x, ysim, models, label, nb=12):
    ok = np.isfinite(x) & np.isfinite(ysim)
    for v in models.values():
        ok &= np.isfinite(v)
    q = np.quantile(x[ok], np.linspace(0, 1, nb + 1)); q[0] -= 1e-9; q[-1] += 1e-9
    idx = np.where(ok, np.clip(np.digitize(x, q) - 1, 0, nb - 1), -1)
    cx, sv, se = [], [], []
    res = {k: [] for k in models}
    for b in range(nb):
        k = idx == b
        n = int(k.sum())
        sm = float(np.mean(ysim[k]))
        cx.append(float(np.median(x[k]))); sv.append(sm)
        se.append(100.0 * abs(float(np.std(ysim[k], ddof=1)) / np.sqrt(n) / sm))
        for kk, v in models.items():
            res[kk].append(100.0 * (float(np.mean(v[k])) / sm - 1.0))
    print(f"\nvs {label} (equal-count, {nb} bins)")
    print("  centre    : " + " ".join(f"{v:7.2f}" for v in cx))
    print("  R_self    : " + " ".join(f"{v:7.3f}" for v in sv))
    for kk in models:
        print(f"  {kk:<10}: " + " ".join(f"{v:7.2f}" for v in res[kk]))
    print("  truth s.e.: " + " ".join(f"{v:7.2f}" for v in se))
    se = np.array(se)
    for kk in models:
        r = np.array(res[kk])
        print(f"    {kk:<10} rms={np.sqrt(np.mean(r**2)):6.2f}%  |worst|={np.abs(r).max():6.2f}%  "
              f"resolved {int(np.sum(np.abs(r) > se))}/{nb}")
    return {k: np.array(v) for k, v in res.items()}, se


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selfresp", default="results/halfshear_selfresp.feather")
    ap.add_argument("--nbins", type=int, default=12)
    args = ap.parse_args()

    df = pf.read_table(args.selfresp, memory_map=True).to_pandas()
    scols = sorted(c for c in df.columns if c.startswith("R_flow_s"))
    y = df["r_sim_self"].to_numpy(float)
    flow = np.mean([df[c].to_numpy(float) for c in scols], axis=0)
    X = df[FEATS].to_numpy(np.float32)
    case = df["case"].to_numpy(np.int64)
    print(f"{len(df):,} rows x {len(scols)} seeds, <R_self>={np.nanmean(y):.4f}, "
          f"<R_flow>={np.nanmean(flow):.4f}")

    # Held out by CASE (a noise realisation), never by row: rows of the same case share a rendering,
    # so a row-wise split would leak and flatter the fit.
    cs = np.unique(case)
    held = np.isin(case, cs[len(cs) // 2:])
    print(f"cases {cs.min()}-{cs.max()}: fit on {int((~held).sum()):,} rows, "
          f"held out {int(held.sum()):,}")

    ok = np.isfinite(y) & np.isfinite(X).all(1)
    mdl = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06, max_leaf_nodes=63,
                                        min_samples_leaf=200, l2_regularization=1.0,
                                        early_stopping=False, random_state=0)
    mdl.fit(X[(~held) & ok], y[(~held) & ok])
    fit = np.full(len(y), np.nan)
    fit[held] = mdl.predict(X[held])

    yh, fh, mh = y[held], flow[held], fit[held]
    print(f"held-out means: truth {np.nanmean(yh):.4f}  flow {np.nanmean(fh):.4f}  "
          f"fit {np.nanmean(mh):.4f}")

    models = {"flow #1": fh, "fit(3ft)": mh}
    for col, lab in (("Re_input_p", "TRUE size Re [arcsec]"), ("SN", "S/N")):
        table(df[col].to_numpy(float)[held], yh, models, lab, nb=args.nbins)

    # ---- REGION averages -----------------------------------------------------------------------
    # Individual small/faint bins carry a 2-5% truth s.e., so no single bin decides anything. The
    # region average is the statistic with the power, and it is bootstrapped over CASES (the unit of
    # independence) rather than rows.
    ch = case[held]
    ucase = np.unique(ch[np.isfinite(yh)])
    rng = np.random.default_rng(0)
    for col, lab, lo_side in (("Re_input_p", "SMALL size (lowest quartile of Re)", True),
                              ("SN", "FAINT (lowest quartile of S/N)", True)):
        x = df[col].to_numpy(float)[held]
        thr = np.nanquantile(x, 0.25)
        reg = (x <= thr) if lo_side else (x >= thr)
        ok = reg & np.isfinite(yh) & np.isfinite(fh) & np.isfinite(mh)
        print(f"\n{lab}: {int(ok.sum()):,} rows, threshold {thr:.4g}")
        for kk, v in (("flow #1", fh), ("fit(3ft)", mh)):
            boot = []
            for _ in range(200):
                pick = rng.choice(ucase, size=len(ucase), replace=True)
                sel = np.isin(ch, pick) & ok
                boot.append(float(np.mean(v[sel])) / float(np.mean(yh[sel])) - 1.0)
            b = 100.0 * np.array(boot)
            cen = 100.0 * (float(np.mean(v[ok])) / float(np.mean(yh[ok])) - 1.0)
            # ABSOLUTE error too: a relative error on R_self (~0.72) and the same relative error on
            # R_blend (~0.14) are NOT comparable -- only the absolute shift moves `m` equally.
            print(f"  {kk:<10} rel {cen:+6.2f} +- {b.std():.2f} %   "
                  f"absolute {cen/100*float(np.mean(yh[ok])):+.4f} in R_self units")

    print("\nREAD: the fit sees 3 features where the flow sees 8 and is held out by case, so it is\n"
          "handicapped. If it still beats the flow at small/faint, that headroom is RECOVERABLE and\n"
          "a better-trained flow #1 can take it. A tie is inconclusive, not proof of a limit.")
    print("\nSELFRESP_HEADROOM_DONE")


if __name__ == "__main__":
    main()
