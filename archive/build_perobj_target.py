"""BUILD THE PER-OBJECT RESPONSE TARGET that replaces the per-cell mean pin.

WHAT THIS REPLACES AND WHY. The fiducial pin supervises the flow with a per-CELL MEAN response on a
(true mag x true Re x r_blend) grid. 2026-08-03p exhausted five ways of configuring that grid (edge
allocation, cell resolution, mean-head capacity, log-Re input, response weight) and all were null on
the small-size defect, because a mean constraint is blind to structure INSIDE its own cell and the
defect is a -23 to -32 %/cell ramp inside the FIRST size cell. 2026-08-03q then measured the ceiling
of removing the binning: an out-of-fold regressor on true properties reaches +0.29 +- 0.82% at small
size where the flow sits at -4.28% (3.3 sigma). This script builds the object that turns that ceiling
into a trainable target.

THE TARGET IS E[R | true properties], NOT A PER-OBJECT LABEL. Per-object SNC response is far too
noisy to supervise on directly -- two independent estimators of it agree in mean to 0.35% but
correlate only 0.525 (03q). Under a squared-error pull, per-object supervision converges to the
CONDITIONAL MEAN anyway, so the target IS that conditional mean, estimated once here by a
gradient-boosted regressor. This is the same quantity the grid target estimates; the only difference
is that it is resolved per object instead of per cell. It introduces NO new information and no
empirical offset -- it is the existing supervision signal, unbinned.

FEATURE SET = D6, AND WHY NOT D7. 03q measured D6 (six shear-free true properties) at +0.29 +- 0.82%
at small size against D7's +0.31 +- 0.91% -- indistinguishable. D7's extra feature `e_dot_ghat` needs
an applied shear direction, which does not exist on the g=0 TRAINING catalogue; supplying it would
mean evaluating the target twice (at e_dot_ghat = e1_int and e2_int, matching the two directions the
response loss probes) and averaging. D6 makes all of that unnecessary at no measured cost at small
size. D7 does help at FAINT (-2.52 vs -3.17), but faint is not closed by this lever either way
(ceiling 2.7 sigma from zero), so it is not a reason to take on the machinery.

SAME ESTIMATOR AS THE GRID TARGET IT REPLACES. The label is the ruler's per-object SNC response over
cases 0-99 under the same domain cuts (true mag < 26, Re > 0.3) and the same 1/n_pairs weighting that
`compute_response_target_blend.py` uses, so the per-object target sits on exactly the same footing as
the fiducial `..._snc_c0-99_6x6x5_dom.npz` it replaces. Any offset between SNC estimators is therefore
common to both and is not introduced here.

OUTPUT. A joblib with the fitted regressor plus its feature list and provenance, and a sidecar JSON.
The TRAINER evaluates it on its own rows -- there is no (case, input_index) lookup, because every
feature survives into the training frame, so there is no coverage gap to zero-fill.

FIREWALL. Label and fit are the half-shear g=0.05 ruler only. constgold is never read and no `m` is
computed or consulted. The target is chosen by the ruler-side ceiling of 03q, not by any acceptance
number.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_perobj_oracle import FEATS7, load_ruler  # noqa: E402

FEATS6 = [f for f in FEATS7 if f != "e_dot_ghat"]


def region_resid(y, p, keep, case, rng, nboot=200):
    ok = keep & np.isfinite(y) & np.isfinite(p)
    if ok.sum() < 100:
        return np.nan, np.nan, int(ok.sum())
    cen = 100.0 * (float(np.mean(p[ok])) / float(np.mean(y[ok])) - 1.0)
    uc = np.unique(case[ok])
    bs = []
    for _ in range(nboot):
        pick = rng.choice(uc, size=len(uc), replace=True)
        m = np.isin(case, pick) & ok
        if m.sum() > 100:
            bs.append(100.0 * (float(np.mean(p[m])) / float(np.mean(y[m])) - 1.0))
    return cen, float(np.std(bs)), int(ok.sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ruler", required=True)
    ap.add_argument("--snc-lookup", required=True)
    ap.add_argument("--nominal-g", type=float, default=0.05)
    ap.add_argument("--max-case", type=int, default=99)
    ap.add_argument("--primary-mag-max", type=float, default=26.0)
    ap.add_argument("--primary-re-min", type=float, default=0.3)
    ap.add_argument("--n-folds", type=int, default=5,
                    help="grouped K-fold over CASES, used ONLY to report held-out quality; the "
                         "shipped model is refit on everything")
    ap.add_argument("--small-re", type=float, default=0.386)
    ap.add_argument("--out", required=True, help="output .joblib")
    ap.add_argument("--label-source", choices=["ruler", "dump"], default="ruler",
                    help="WHICH SNC ESTIMATOR THE TARGET IS FIT TO. 'ruler' = the ruler's own R_snc, "
                         "whose g=0 partner is `g0_lookup` (historical). 'dump' = the dump's "
                         "`r_sim_self`, whose g=0 partner is the det_meas g0 leg -- the SAME "
                         "estimator every score is graded on. WORKLOG 2026-08-03v/w: the two g=0 "
                         "sources are independent measurements of the same objects (0.094% agree "
                         "exactly; mean |de| 0.108 at Re<=0.386 vs 0.025 at Re>0.75), so the two "
                         "labels disagree by -3.06% at small size and a flow trained on 'ruler' "
                         "reads ~2.7% low against a 'dump' score no matter how well it fits.")
    ap.add_argument("--dump", default="results/halfshear_selfresp.feather",
                    help="dump providing `r_sim_self` when --label-source dump")
    args = ap.parse_args()

    ruler = load_ruler(args.ruler, args.snc_lookup, args.nominal_g, args.max_case,
                       args.primary_mag_max, args.primary_re_min)
    if args.label_source == "dump":
        # Keep the FEATURES from the ruler (true properties, noise-free) but take the LABEL from the
        # dump, so the target is fit to the same quantity the flow is scored against. Rows are the
        # intersection; unmatched rows are DROPPED, never zero-filled.
        import pyarrow.feather as _pf
        d = _pf.read_table(args.dump, memory_map=True).to_pandas()[["case", "input_index",
                                                                    "r_sim_self"]]
        n0 = len(ruler)
        ruler = ruler.drop(columns=["R_snc"]).merge(d, on=["case", "input_index"], how="inner")
        ruler = ruler.rename(columns={"r_sim_self": "R_snc"})
        print(f"label source = DUMP `r_sim_self`: {n0:,} ruler rows -> {len(ruler):,} matched "
              f"({100*len(ruler)/max(n0,1):.2f}%)")
        if len(ruler) < 100_000:
            raise SystemExit("REFUSING: too few rows after matching the dump; the label would be "
                             "fit on a population unlike the one it supervises.")
    y = ruler["R_snc"].to_numpy(float)
    case = ruler["case"].to_numpy(np.int64)
    X = ruler[FEATS6].to_numpy(float)
    print(f"\nlabel: mean {y.mean():.5f}, std {y.std():.4f}, N {len(y):,}, "
          f"cases {case.min()}-{case.max()}")
    print(f"features (D6): {FEATS6}")

    def _mk():
        return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06, max_leaf_nodes=63,
                                             min_samples_leaf=200, random_state=0)

    # ---- held-out quality, reported so the shipped target is not taken on faith ---------------
    ucase = np.unique(case)
    fold = np.array([{c: i % args.n_folds for i, c in enumerate(ucase)}[c] for c in case])
    oof = np.full(len(y), np.nan)
    ok_y = np.isfinite(y)
    for k in range(args.n_folds):
        tr, te = (fold != k) & ok_y, fold == k
        g = _mk()
        g.fit(X[tr], y[tr])
        oof[te] = g.predict(X[te])
    rng = np.random.default_rng(0)
    re = ruler["Re_input_p"].to_numpy(float)
    print(f"\n=== OUT-OF-FOLD quality of the target (grouped {args.n_folds}-fold over cases) ===")
    stats = {}
    for lab, keep in (("ALL", np.ones(len(y), bool)),
                      (f"small size Re <= {args.small_re}", re <= args.small_re)):
        c, s, n = region_resid(y, oof, keep, case, rng)
        stats[lab] = dict(resid_pct=c, err_pct=s, n=int(n))
        print(f"  {lab:<26} N={n:>9,}   target/label - 1 = {c:+.2f} +- {s:.2f} %")
    print("  (these are the ruler's OWN label, so they say the target reproduces the supervision\n"
          "   signal; the flow's ability to REACH it is a separate question the retrain answers)")

    # ---- ship a model refit on everything ----------------------------------------------------
    final = _mk()
    final.fit(X[ok_y], y[ok_y])
    pred_all = final.predict(X)
    # `e_abs` is DERIVED, not a raw catalogue column, so the trainer must be told how to rebuild it
    # on the g=0 catalogue (which carries axis_ratio_input_p but no e_abs). Declaring it here, and
    # deriving it through the shared library helper, is what stops the fit and the deployment from
    # silently using two different quantities.
    payload = {
        "model": final,
        "features": FEATS6,
        "derived": {"e_abs": "intrinsic_e_abs(axis_ratio_input_p)"},
        "provenance": {
            "ruler": args.ruler, "snc_lookup": args.snc_lookup, "nominal_g": args.nominal_g,
            "max_case": args.max_case, "primary_mag_max": args.primary_mag_max,
            "primary_re_min": args.primary_re_min, "n_fit_rows": int(ok_y.sum()),
            "label_mean": float(y[ok_y].mean()), "target_mean": float(pred_all.mean()),
            "target_min": float(pred_all.min()), "target_max": float(pred_all.max()),
            "oof": stats,
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, out)
    with open(out.with_suffix(".json"), "w") as fh:
        json.dump(payload["provenance"] | {"features": FEATS6}, fh, indent=1, sort_keys=True)
    print(f"\ntarget on the fit rows: mean {pred_all.mean():.5f} "
          f"(label {y[ok_y].mean():.5f}), range {pred_all.min():.4f}..{pred_all.max():.4f}")
    print(f"wrote {out}\nwrote {out.with_suffix('.json')}")
    print("\nPEROBJ_TARGET_DONE")


if __name__ == "__main__":
    main()
