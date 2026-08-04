"""PRICE THE PER-OBJECT SUPERVISION LEVER, without training a flow.

THE QUESTION. The pin supervises the flow with a per-CELL mean response on a (flux x size x crowd)
grid. The 2026-07-07 alternative -- never built -- is to supervise each galaxy with its OWN
shape-noise-cancelled response, removing binning entirely. Jobs 15489742/15489783 showed that on the
flux x size axes the grid is already fully resolved (36 -> 400 cells moves aggregate `m` by 0.003
pt), so grid RESOLUTION is not the constraint. What a per-object target would add instead is
CONDITIONING: the flow sees 8 true properties, while the grid can only ever see the 2-3 it is binned
on.

SO THE PRICE OF THE LEVER IS: how much response structure lives in the features the grid cannot see?

HOW THIS ANSWERS IT. Per-object supervision under a squared-error pull converges to the conditional
mean E[R | features] -- so a regressor fit on the per-object SNC label IS the thing a perfectly
per-object-supervised flow would be pulled toward. Two arms, identical in every respect but the
feature list:

    D2  gradient-boosted trees on (true mag, true Re)      -- the grid's own information
    D7  the same, on all 7 true properties available on BOTH catalogues

D2 is also a CONTROL: it should land on the fine-grid arm C (+0.840%). If it does not, the regressor
is not a fair stand-in for a grid and D7 cannot be read either. D7 - D2 is the headroom, in the same
units as every other arm.

FEATURES (7). true mag, true Re, sersic n, |e_intrinsic|, e_intrinsic . ghat, nbr_flux_near,
nbr_flux_far. Intrinsic ellipticity is rebuilt from axis_ratio + position_angle on both catalogues
(blendemu `angle2e` convention, as scripts/decomp_constgold_ensemble.py does). `e . ghat` is a
legitimate feature and not shear leakage: ghat is the KNOWN applied direction and the intrinsic shape
is independent of it, but the projected response genuinely depends on their relative angle, which is
information the flow can use and a (mag, Re) grid cannot.

DELIBERATELY EXCLUDED: `distance` and `neighbored`. The ruler is an ALL-PAIRS catalogue where those
are per-PAIR, while constgold carries one row per object; feeding a per-pair quantity to a model
scored per-object is the kind of silent mismatch this project has been bitten by. Labels are
therefore aggregated to one row per (case, input_index) with the builder's own 1/n_pairs weighting,
and every feature used is a per-TARGET quantity. Because 2 of the flow's 8 inputs are dropped, D7 is
a LOWER bound on the headroom -- the safe direction.

FIREWALL. The label and the fit use the half-shear g=0.05 ruler and its g=0 SNC lookup only. Cases
are split fit 0-79 / held-out 80-99 and the held-out score is reported, so the number is not an
in-sample fit. constgold supplies prediction COORDINATES only; its r_sim is never read here.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as pf
import pyarrow.ipc as ipc
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sbs_shear.preprocessing import DEFAULT_SELECTION_CUTS, source_select_selection  # noqa: E402

FEATS2 = ["r_input_p", "Re_input_p"]
FEATS7 = FEATS2 + ["sersic_n_input_p", "e_abs", "e_dot_ghat", "nbr_flux_near", "nbr_flux_far"]
# Carried through but NOT part of FEATS7 -- see the comment in load_ruler.
EXTRA_FEATS = ["nbr_flux_max"]


def _ellip(q, pa_deg):
    """blendemu angle2e: e = (1-q)/(1+q) * (cos 2theta, sin 2theta), theta in degrees."""
    amp = (1.0 - q) / (1.0 + q)
    t = 2.0 * np.deg2rad(pa_deg)
    return amp * np.cos(t), amp * np.sin(t)


def load_ruler(path, snc_path, nominal_g, max_case, mag_max, re_min):
    """One row per (case, input_index): the SNC self-response label + per-target features."""
    snc = pf.read_table(snc_path, columns=["case", "input_index", "ngmix0_g1", "ngmix0_g2"]).to_pandas()
    snc["key"] = snc["case"].to_numpy(np.int64) * 1_000_003 + snc["input_index"].to_numpy(np.int64)
    snc = snc.sort_values("key")
    skey = snc["key"].to_numpy(); s1 = snc["ngmix0_g1"].to_numpy(float); s2 = snc["ngmix0_g2"].to_numpy(float)
    del snc

    need = ["case", "input_index", "measured_ngmix_g1", "measured_ngmix_g2",
            "gamma1_input_p", "gamma2_input_p", "detected", "r_input_p", "Re_input_p",
            "sersic_n_input_p", "axis_ratio_input_p", "position_angle_input_p",
            "nbr_flux_near", "nbr_flux_far", "nbr_flux_max", "distance", "neighbored"]
    parts = []
    with ipc.open_file(path) as r:
        cols = [c for c in need if c in set(r.schema.names)]
        for bi in range(r.num_record_batches):
            b = pa.Table.from_batches([r.get_batch(bi)]).select(cols).to_pandas()
            if max_case is not None:
                b = b[b["case"] <= max_case]
            if len(b) == 0:
                continue
            # DEFAULT_SELECTION_CUTS with the primary domain narrowed, exactly as
            # compute_response_target_blend._selection_cuts does -- so this label population is the
            # SAME one the fiducial pin target is built on, not a re-derived approximation.
            cuts = [list(c) for c in DEFAULT_SELECTION_CUTS]
            cuts[1][1] = float(mag_max)
            cuts[3][0] = float(re_min)
            b = source_select_selection(b, cuts=cuts)
            if len(b) == 0:
                continue
            parts.append(b[b["detected"].astype(bool)].reset_index(drop=True))
    df = pd.concat(parts, ignore_index=True)
    del parts

    g1 = df["gamma1_input_p"].to_numpy(float); g2 = df["gamma2_input_p"].to_numpy(float)
    gmag = np.hypot(g1, g2)
    df = df[gmag > 1e-6].reset_index(drop=True)
    g1, g2 = g1[gmag > 1e-6], g2[gmag > 1e-6]; gmag = gmag[gmag > 1e-6]
    gh1, gh2 = g1 / gmag, g2 / gmag

    key = df["case"].to_numpy(np.int64) * 1_000_003 + df["input_index"].to_numpy(np.int64)
    pos = np.clip(np.searchsorted(skey, key), 0, len(skey) - 1)
    match = skey[pos] == key
    print(f"SNC match after selection: {match.mean():.2%} ({int(match.sum()):,}/{len(match):,})")
    e1 = df["measured_ngmix_g1"].to_numpy(float) - np.where(match, s1[pos], np.nan)
    e2 = df["measured_ngmix_g2"].to_numpy(float) - np.where(match, s2[pos], np.nan)
    df["R_snc"] = (e1 * gh1 + e2 * gh2) / nominal_g

    ie1, ie2 = _ellip(df["axis_ratio_input_p"].to_numpy(float),
                      df["position_angle_input_p"].to_numpy(float))
    df["e_abs"] = np.hypot(ie1, ie2)
    df["e_dot_ghat"] = ie1 * gh1 + ie2 * gh2
    df = df[np.isfinite(df["R_snc"].to_numpy(float))].reset_index(drop=True)

    # One row per (case, target): the mean over that target's pair rows IS the builder's 1/n_pairs
    # weighting, so the label here is exactly what a cell mean is built from -- just unbinned.
    # EXTRA_FEATS are carried through aggregation but are NOT in FEATS7, so every existing caller
    # (and the shipped per-object target) sees an unchanged feature list. They exist so the faint
    # ceiling can be tested against more crowding information (WORKLOG 2026-08-03w): the D2 -> D3
    # step bought 7.4 pt at faint from `nbr_flux_near` alone, which makes crowding the obvious axis
    # to push, and `nbr_flux_max` is a per-OBJECT dominance measure (unlike `distance`, which is
    # per-pair and is still deliberately excluded).
    extra = [c for c in EXTRA_FEATS if c in df.columns]
    agg = {c: "mean" for c in FEATS7 + extra + ["R_snc"]}
    out = df.groupby(["case", "input_index"], as_index=False).agg(agg)
    print(f"ruler: {len(df):,} pair rows -> {len(out):,} unique (case,input_index) targets")
    return out


def load_constgold(path, crowd_path, min_case):
    need = ["case", "input_index", "r_input_p", "Re_input_p", "sersic_n_input_p",
            "axis_ratio_input_p", "position_angle_input_p"]
    parts = []
    with ipc.open_file(path) as r:
        for bi in range(r.num_record_batches):
            b = pa.Table.from_batches([r.get_batch(bi)]).select(need).to_pandas()
            b = b[b["case"] >= min_case]
            if len(b):
                parts.append(b)
    df = pd.concat(parts, ignore_index=True)
    del parts
    # constgold shear is uniform along g1 (applied_g = (+-0.02, 0)) => ghat = (1, 0)
    ie1, ie2 = _ellip(df["axis_ratio_input_p"].to_numpy(float),
                      df["position_angle_input_p"].to_numpy(float))
    df["e_abs"] = np.hypot(ie1, ie2)
    df["e_dot_ghat"] = ie1

    cr = pf.read_table(crowd_path, columns=["case", "input_index", "nbr_flux_near", "nbr_flux_far"]).to_pandas()
    cr = cr[cr["case"] >= min_case]
    n_cr = len(cr)
    cr = cr.drop_duplicates(subset=["case", "input_index"])
    if len(cr) != n_cr:
        raise SystemExit(f"REFUSING: crowd lookup has duplicate keys ({n_cr:,} -> {len(cr):,}).")
    n0 = len(df)
    df = df.merge(cr, on=["case", "input_index"], how="left")
    if len(df) != n0:
        raise SystemExit(f"REFUSING: crowd merge changed the row count ({n0:,} -> {len(df):,}).")
    frac = float(np.isfinite(df["nbr_flux_near"].to_numpy(float)).mean())
    print(f"constgold: {len(df):,} rows (case >= {min_case}), crowd matched {100*frac:.2f}%")
    if frac < 0.99:
        raise SystemExit(f"REFUSING: crowd lookup covers only {100*frac:.1f}% of constgold rows.")
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ruler", required=True)
    ap.add_argument("--snc-lookup", required=True)
    ap.add_argument("--constgold", required=True)
    ap.add_argument("--crowd", required=True)
    ap.add_argument("--nominal-g", type=float, default=0.05)
    ap.add_argument("--max-case", type=int, default=99)
    ap.add_argument("--fit-max-case", type=int, default=79, help="cases above this are held out")
    ap.add_argument("--min-case", type=int, default=40, help="constgold cases to predict on")
    ap.add_argument("--primary-mag-max", type=float, default=26.0)
    ap.add_argument("--primary-re-min", type=float, default=0.3)
    ap.add_argument("--out-prefix", required=True)
    args = ap.parse_args()

    ruler = load_ruler(args.ruler, args.snc_lookup, args.nominal_g, args.max_case,
                       args.primary_mag_max, args.primary_re_min)
    cg = load_constgold(args.constgold, args.crowd, args.min_case)

    fit = ruler["case"].to_numpy() <= args.fit_max_case
    print(f"fit cases <= {args.fit_max_case}: {int(fit.sum()):,}   held out: {int((~fit).sum()):,}")
    y = ruler["R_snc"].to_numpy(float)
    print(f"per-object SNC label: mean {y.mean():.4f}, std {y.std():.4f} "
          f"(std/mean = {y.std()/abs(y.mean()):.1f}x -- this is why the pin bins in the first place)")

    for label, feats in (("D2", FEATS2), ("D7", FEATS7)):
        X = ruler[feats].to_numpy(np.float32)
        mdl = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06, max_leaf_nodes=63,
                                            min_samples_leaf=200, l2_regularization=1.0,
                                            early_stopping=False, random_state=0)
        mdl.fit(X[fit], y[fit])
        # A per-object R^2 is ~0 by construction (the label is one noisy draw), so the honest
        # held-out check is whether the PREDICTED MEAN reproduces the TRUE MEAN out of sample.
        ph, yh = mdl.predict(X[~fit]), y[~fit]
        n = len(yh)
        print(f"\n{label} ({len(feats)} features): held-out <pred> = {ph.mean():.4f} vs "
              f"<truth> = {yh.mean():.4f}  (diff {100*(ph.mean()/yh.mean()-1):+.3f}%, "
              f"truth sem {100*yh.std()/np.sqrt(n)/abs(yh.mean()):.3f}%)")
        v = mdl.predict(cg[feats].to_numpy(np.float32))
        out = f"{args.out_prefix}_{label}.npz"
        np.savez(out, case=cg["case"].to_numpy(np.int64),
                 input_index=cg["input_index"].to_numpy(np.int64), value=v.astype(float))
        print(f"  wrote {out}: {len(v):,} objects, <value>={v.mean():.4f} "
              f"range=[{v.min():.4f},{v.max():.4f}]")

    print("\nPEROBJ_ORACLE_DONE")


if __name__ == "__main__":
    main()
