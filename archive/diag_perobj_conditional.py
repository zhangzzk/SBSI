"""DOES PER-OBJECT SUPERVISION FIX THE SMALL/FAINT DEFECT? -- scored on the CONDITIONAL.

WHY THIS EXISTS. 2026-08-03p exhausted the pin lever: five independent attempts (edge allocation,
cell resolution, mean-head capacity, log-Re input, response weight) are all null on flow #1's
small/faint self-response defect, and the pin residual SATURATES -- 18x the weight buys 16%. The
stated mechanism is that the pin constrains a per-CELL MEAN while the defect is a ramp INSIDE the
first size cell, and a mean constraint is blind to structure within its own cell.

The remaining candidate is supervision with within-cell resolution: a per-OBJECT target. This script
prices that lever BEFORE any retrain.

WHY IT MUST BE SCORED ON THE CONDITIONAL, NOT THE AGGREGATE. 2026-08-03n already priced per-object
supervision on aggregate `m` and found conditioning worth 1.05 pt -- but it ALSO found flow #1
already matches the D7 oracle's `<R_flow>` (0.7261 vs 0.7258). On the aggregate the lever therefore
looks like a null no matter what it does to small/faint, because the aggregate is dominated by the
bulk where the flow is already right. The defect lives in the tails, so the tails are where the
ceiling has to be read.

THE LOGIC. Per-object supervision under a squared-error pull converges to E[R | features], so a
regressor fit on the per-object SNC label IS what a perfectly per-object-supervised flow would be
pulled toward. Its residual in a region is therefore the CEILING of that lever there:

    oracle near zero at small/faint  -> per-object supervision CAN fix it; a retrain is worth the GPU
    oracle as bad as the flow        -> the lever is dead too, and the defect is not a supervision
                                        problem at all but an estimator/feature problem

THREE ARMS, identical but for the feature list, so the comparison isolates CONDITIONING:

    D2  (true mag, true Re)                      -- exactly the information the pin grid has
    D3  D2 + nbr_flux_near                       -- bridges to the 3-feature fit of 2026-08-03k
    D7  all 7 true properties on both catalogues -- the full per-object ceiling

D2 is the CONTROL. It carries the grid's own information with binning removed, so D2 - flow isolates
"what unbinning alone buys" and D7 - D2 isolates "what extra conditioning buys". If D2 already lands
near zero at small size, the defect is BINNING; if only D7 does, it is CONDITIONING; if neither does,
per-object supervision is not the answer.

WHAT IS FIT ON, AND WHY IT IS THE DUMP'S OWN TRUTH (revised after job 15501436).

The first version fit the arms on the ruler's per-object `R_snc` and scored them against the dump's
`r_sim_self`. A guard compared the two directly and REFUSED: on 2,326,997 overlapping rows their
means agree to 0.351% but they correlate only 0.525. They are two different ESTIMATORS of the same
response -- the dump forms e(g)-e(0) from a both-detected merge against the `g0.0_train` leg, while
the ruler subtracts `g0_lookup`, built from the raw secondaries Shapes catalogues. Different g=0
measurement, so independent measurement noise; the agreeing means say the quantity is the same.

For a mean-based region residual, fitting on a noisy-but-unbiased label is legitimate, so that
refusal was CONSERVATIVE rather than a real blocker. It is not reinstated, because a strictly better
design exists: fit on the DUMP's OWN `r_sim_self` and take only the FEATURES from the ruler. Features
are per-object TRUE properties (plus the applied-shear direction), so they carry no measurement noise
and no estimator ambiguity. Fit and score are then the same quantity by construction, and this
becomes a direct extension of the 3-feature fit of 2026-08-03k, which also fit on `r_sim_self`.

THE SPLIT. The dump is one case range (0-39), so held-out prediction is done by GROUPED K-FOLD over
CASES: each fold fits on the other cases and predicts its own, giving every row an OUT-OF-FOLD
prediction. No case is ever in its own fit set, and the full row set is scored, so the oracle and the
flow are compared on identical N.

LIKE-FOR-LIKE ROWS. The feature merge does not cover the dump completely (the ruler applies its own
SNC-match and finiteness filters). Unmatched rows are DROPPED, never zero-filled, and the flow is
re-scored on the SAME surviving subset, so every arm in the table describes one identical population.

HANDICAP, and which way it points. The oracles are held out by case; the flow is a 16-seed mean
trained on a different catalogue and its exposure to these cases is not controlled here. That favours
the FLOW if anything, so "an oracle beats the flow" stays a conservative reading, while "the oracle
ties the flow" is weak evidence and must be reported as inconclusive rather than as a limit.

FIREWALL. Half-shear self-response only. constgold is never read and no `m` is computed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.feather as pf
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_perobj_oracle import FEATS2, FEATS7, load_ruler  # noqa: E402

DUMP = "results/halfshear_selfresp.feather"
ARMS = [
    ("D2  (mag, Re) = grid info", FEATS2),
    ("D3  D2 + nbr_flux_near", FEATS2 + ["nbr_flux_near"]),
    ("D6  D7 minus e_dot_ghat", [f for f in FEATS7 if f != "e_dot_ghat"]),
    ("D7  all 7 true props", FEATS7),
    ("D8  D7 + nbr_flux_max", FEATS7 + ["nbr_flux_max"]),
]
# D6 EXISTS TO DECIDE A BUILD. A per-object target has to be evaluated on the g=0 TRAINING
# catalogue, where there is no applied shear and `e_dot_ghat` is undefined. It can still be
# supplied -- the response loss probes ghat=(1,0) and (0,1) and averages, so the target would be
# the mean of two evaluations at e_dot_ghat = e1_int and e2_int -- but that is extra machinery and
# an extra assumption. If D6 matches D7 at small size, none of it is needed and the target is a
# plain function of shear-free true properties.


def region(y, mod, keep, case, rng, nboot=300):
    """Percent residual of `mod` against `y` in a region, bootstrapped over CASES."""
    ok = keep & np.isfinite(y) & np.isfinite(mod)
    if ok.sum() < 100:
        return np.nan, np.nan, int(ok.sum())
    cen = 100.0 * (float(np.mean(mod[ok])) / float(np.mean(y[ok])) - 1.0)
    uc = np.unique(case[ok])
    bs = []
    for _ in range(nboot):
        pick = rng.choice(uc, size=len(uc), replace=True)
        m = np.isin(case, pick) & ok
        if m.sum() > 100:
            bs.append(100.0 * (float(np.mean(mod[m])) / float(np.mean(y[m])) - 1.0))
    return cen, float(np.std(bs)), int(ok.sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ruler", required=True)
    ap.add_argument("--snc-lookup", required=True)
    ap.add_argument("--dump", default=DUMP)
    ap.add_argument("--nominal-g", type=float, default=0.05)
    ap.add_argument("--max-case", type=int, default=39,
                    help="ruler cases to read features from; must cover the dump's case range")
    ap.add_argument("--n-folds", type=int, default=5, help="grouped K-fold over CASES")
    ap.add_argument("--primary-mag-max", type=float, default=26.0)
    ap.add_argument("--primary-re-min", type=float, default=0.3)
    ap.add_argument("--small-re", type=float, default=0.386)
    ap.add_argument("--faint-sn", type=float, default=13.09)
    ap.add_argument("--faint-mag", type=float, default=25.0,
                    help="TRUE-magnitude faint cut, the control for the measured-S/N cut")
    ap.add_argument("--crowd", default="/home/z/Zekang.Zhang/SBSI/results/crowd_flux_conc_c0-199.feather",
                    help="supplies `nbr_flux_max`, which the ruler catalogue does NOT carry")
    args = ap.parse_args()

    ruler = load_ruler(args.ruler, args.snc_lookup, args.nominal_g, args.max_case,
                       args.primary_mag_max, args.primary_re_min)

    dump = pf.read_table(args.dump, memory_map=True).to_pandas()
    seed_cols = [c for c in dump.columns if c.startswith("R_flow_s")]
    if not seed_cols:
        raise SystemExit(f"REFUSING: {args.dump} carries no R_flow_s* columns")
    dump["R_flow"] = np.mean([dump[c].to_numpy(float) for c in seed_cols], axis=0)
    print(f"dump: {len(dump):,} rows, cases {dump['case'].min()}-{dump['case'].max()}, "
          f"{len(seed_cols)} seeds")

    # ---- attach the ruler's FEATURES (not its label) to the dump rows ------------------------
    feat_cols = sorted({f for _, fs in ARMS for f in fs})
    # `nbr_flux_max` lives only in the crowd lookup, not in the ruler catalogue. Merge it onto the
    # ruler here rather than widening load_ruler, which the shipped per-object target depends on.
    if "nbr_flux_max" in feat_cols and "nbr_flux_max" not in ruler.columns:
        cr = pf.read_table(args.crowd, columns=["case", "input_index", "nbr_flux_max"]).to_pandas()
        cr = cr.drop_duplicates(["case", "input_index"])
        n0 = len(ruler)
        ruler = ruler.merge(cr, on=["case", "input_index"], how="left")
        if len(ruler) != n0:
            raise SystemExit(f"REFUSING: crowd merge changed the row count ({n0:,} -> {len(ruler):,})")
        frac = float(np.isfinite(ruler["nbr_flux_max"].to_numpy(float)).mean())
        print(f"merged nbr_flux_max from the crowd lookup: {100*frac:.2f}% of ruler rows matched")
        if frac < 0.95:
            raise SystemExit(f"REFUSING: nbr_flux_max covers only {100*frac:.2f}% of ruler rows")
    shared = [c for c in feat_cols if c in dump.columns]
    feats_tbl = ruler[["case", "input_index"] + feat_cols].drop_duplicates(["case", "input_index"])
    feats_tbl = feats_tbl.rename(columns={c: c + "__chk" for c in shared})
    n0 = len(dump)
    dump = dump.merge(feats_tbl, on=["case", "input_index"], how="inner")

    # Some features exist on BOTH sources. They must be the same numbers, or the ruler rows are not
    # the dump rows and every feature taken from the ruler is suspect. Check, then keep the dump's.
    for c in shared:
        a = dump[c].to_numpy(float)
        b = dump[c + "__chk"].to_numpy(float)
        g = np.isfinite(a) & np.isfinite(b)
        bad = ~np.isclose(a[g], b[g], rtol=1e-3, atol=1e-4)
        print(f"  shared feature `{c}`: {100*bad.mean():.3f}% disagree between dump and ruler "
              f"(max |diff| {np.abs(a[g] - b[g]).max():.3g})")
        if bad.mean() > 0.01:
            raise SystemExit(f"REFUSING: `{c}` disagrees on {100*bad.mean():.2f}% of merged rows; "
                             "the two sources are not describing the same objects.")
    dump = dump.drop(columns=[c + "__chk" for c in shared])
    frac = len(dump) / n0
    print(f"\nfeature merge: {len(dump):,}/{n0:,} dump rows kept ({100*frac:.2f}%) -- unmatched rows "
          "are DROPPED, not zero-filled, and every arm below is scored on this same subset")
    if frac < 0.95:
        raise SystemExit(f"REFUSING: features cover only {100*frac:.2f}% of the dump; the scored "
                         "population would no longer be the one the defect was measured on.")

    # ---- FIT ON THE DUMP'S OWN TRUTH, out-of-fold by CASE ------------------------------------
    y = dump["r_sim_self"].to_numpy(float)
    case = dump["case"].to_numpy(np.int64)
    ucase = np.unique(case)
    fold_of_case = {c: i % args.n_folds for i, c in enumerate(ucase)}
    fold = np.array([fold_of_case[c] for c in case])
    print(f"grouped {len(ucase)}-case {args.n_folds}-fold: "
          + ", ".join(f"f{k}={int((fold == k).sum()):,}" for k in range(args.n_folds)))

    ok_y = np.isfinite(y)
    for label, fs in ARMS:
        X = dump[fs].to_numpy(float)
        pred = np.full(len(dump), np.nan)
        for k in range(args.n_folds):
            tr = (fold != k) & ok_y
            te = fold == k
            if not np.intersect1d(np.unique(case[tr]), np.unique(case[te])).size == 0:
                raise SystemExit("REFUSING: a case appears in both the fit and predict side of a "
                                 "fold; the prediction would be in-sample.")
            g = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06, max_leaf_nodes=63,
                                              min_samples_leaf=200, random_state=0)
            g.fit(X[tr], y[tr])
            pred[te] = g.predict(X[te])
        dump[label] = pred
        print(f"fitted {label:<28} on {len(fs)} features, out-of-fold on all {len(dump):,} rows")

    # ---- score --------------------------------------------------------------------------------
    truth = dump["r_sim_self"].to_numpy(float)
    re = dump["Re_input_p"].to_numpy(float)
    sn = dump["SN"].to_numpy(float)
    case = dump["case"].to_numpy(np.int64)
    rng = np.random.default_rng(0)

    # TWO faint definitions ON PURPOSE. `SN` in the dump is measured_flux_auto/measured_fluxerr_auto
    # -- a MEASURED quantity -- while every oracle feature is a TRUE property. Cutting on measured
    # S/N selects on the noise realisation, and a true-property-only model cannot reproduce that
    # conditional mean no matter how many true features it is given, which would show up as an
    # IRREDUCIBLE ceiling (D7 -2.52, D8 -2.45: adding crowding flux buys nothing). The true-mag cut
    # is the control: if the ceiling closes there, the faint "defect" is a measured-selection
    # artefact of how the region is defined, not a missing-information problem.
    mag = dump["r_input_p"].to_numpy(float)
    regions = [("ALL", np.ones(len(dump), bool)),
               (f"small size Re <= {args.small_re}", re <= args.small_re),
               (f"faint  MEASURED S/N <= {args.faint_sn}", sn <= args.faint_sn),
               (f"faint  TRUE mag > {args.faint_mag}", mag > args.faint_mag)]

    print("\n=== RESIDUAL vs the dump's own truth (model/truth - 1), bootstrapped over cases ===")
    print("    the oracles are the CEILING of per-object supervision; the flow is what we have now")
    print("    `closer` compares DISTANCE FROM ZERO, so an oracle that overshoots is not a win\n")
    for lab, keep in regions:
        fc, fs, n = region(truth, dump["R_flow"].to_numpy(float), keep, case, rng)
        print(f"  {lab:<26} N={n:>9,}")
        print(f"      {'flow #1 (' + str(len(seed_cols)) + ' seeds)':<28} {fc:+6.2f} +- {fs:.2f} %")
        for label, _ in ARMS:
            oc, os_, _ = region(truth, dump[label].to_numpy(float), keep, case, rng)
            closer = abs(fc) - abs(oc)          # > 0 => the oracle sits nearer zero than the flow
            ds = np.hypot(fs, os_)
            verdict = ("CLOSER to 0" if closer > 2 * ds else
                       ("further" if closer < -2 * ds else "ties flow"))
            print(f"      {label:<28} {oc:+6.2f} +- {os_:.2f} %   closer by {closer:+6.2f} +- {ds:.2f} "
                  f"({verdict})")
        print()

    print("READ IT AS: an oracle near 0 means per-object supervision CAN reach that region and a\n"
          "retrain is worth the GPU. An oracle as far off as the flow means the lever is dead there\n"
          "and the defect is not a supervision problem. D2 vs flow = what UNBINNING buys;\n"
          "D7 vs D2 = what extra CONDITIONING buys.")
    print("\nPEROBJ_COND_DONE")


if __name__ == "__main__":
    main()
