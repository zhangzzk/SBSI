"""Isolate the multiplicative-bias source via per-object forward-model fidelity.

The bias is forward-model error: the g=0 conditional mean E[measured chi | true props],
applied to a sheared object's S_gamma(intrinsic), over-predicts the actual measured shape.
That can ONLY happen if measured chi depends on a true property Z that is (a) not in the
conditioning and (b) distributed differently in the sheared-at-scene-shape-X population than
in the g=0-at-scene-shape-X population (shear elongates round galaxies; a g=0 galaxy at the
same scene shape is intrinsically elongated, with different size/flux/profile).

This script finds Z by fitting a flexible conditional mean E[chi | feature_set] on g=0 (a
gradient-boosted regressor: nonlinear, handles interactions + NaN, no linear-extrapolation
pathology), then predicting each sheared object from S_gamma(its own intrinsic shape) + its
own other features, and comparing <pred_par> to <actual_par>.  Feature sets are grown
incrementally; whichever drives pred/act -> 1 is the missing conditioning = the bias source.

Calibration uses g=0 + the exact analytic S_gamma only (no sheared shear truth); the sheared
catalogues supply <measured> (the data) and validate pred/act.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc

SBSI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SBSI_ROOT not in sys.path:
    sys.path.insert(0, SBSI_ROOT)

from sbs_shear.measurement_model import add_measurement_target_features  # noqa: E402
from sbs_shear.preprocessing import DEFAULT_SELECTION_CUTS, source_select_selection  # noqa: E402
from sbs_shear.shear_map import apply_shear_to_ellipticity  # noqa: E402

from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402

CD = "/project/ls-gruen/users/zekang.zhang/sbsi_catalogues"

RAW = [
    "detected", "gamma1_input_p", "gamma2_input_p",
    "e1_input_rot0_p", "e2_input_rot0_p",
    "measured_a_image", "measured_b_image", "measured_theta_image",
    "measured_flux_auto", "measured_fluxerr_auto", "measured_mag_auto",
    "r_input_p", "Re_input_p", "sersic_n_input_p", "distance", "neighbored",
    "r_input_s", "Re_input_s", "sersic_n_input_s", "e1_input_rot0_s", "e2_input_rot0_s",
]

# Incremental feature sets (primary shape always present; add one true property at a time).
FEATURE_SETS = {
    "shape":            ["e1s", "e2s"],
    "+size":            ["e1s", "e2s", "logRe"],
    "+flux":            ["e1s", "e2s", "logRe", "mag"],
    "+sersic":          ["e1s", "e2s", "logRe", "mag", "sersic"],
    "+neighbour":       ["e1s", "e2s", "logRe", "mag", "sersic",
                         "neighbored", "distance", "logRe_s", "mag_s", "sersic_s", "e1s_s", "e2s_s"],
}


def _build_features(df, e1s, e2s):
    """Assemble the regressor feature frame given (possibly sheared) primary scene shape."""
    f = pd.DataFrame()
    f["e1s"] = e1s
    f["e2s"] = e2s
    f["logRe"] = np.log10(df["Re_input_p"].to_numpy(float))
    f["mag"] = df["r_input_p"].to_numpy(float)
    f["sersic"] = df["sersic_n_input_p"].to_numpy(float)
    nb = df["neighbored"].astype(bool).to_numpy()
    f["neighbored"] = nb.astype(float)
    # neighbour features: NaN when not neighboured (HGB handles NaN natively, like the flow gate)
    dist = df["distance"].to_numpy(float)
    f["distance"] = np.where(nb, dist, np.nan)
    f["logRe_s"] = np.where(nb, np.log10(np.clip(df["Re_input_s"].to_numpy(float), 1e-6, None)), np.nan)
    f["mag_s"] = np.where(nb, df["r_input_s"].to_numpy(float), np.nan)
    f["sersic_s"] = np.where(nb, df["sersic_n_input_s"].to_numpy(float), np.nan)
    f["e1s_s"] = np.where(nb, df["e1_input_rot0_s"].to_numpy(float), np.nan)
    f["e2s_s"] = np.where(nb, df["e2_input_rot0_s"].to_numpy(float), np.nan)
    return f


def _stream(path, max_rows, snr_min, seed, max_batches=None):
    rng = np.random.default_rng(seed)
    reservoir, raw = None, 0
    with ipc.open_file(path) as reader:
        avail = set(reader.schema.names)
        cols = [c for c in RAW if c in avail]
        n = reader.num_record_batches
        order = (sorted(set(int(x) for x in np.linspace(0, n - 1, max_batches)))
                 if max_batches and max_batches < n else range(n))
        for bi in order:
            b = pa.Table.from_batches([reader.get_batch(bi)]).select(cols).to_pandas()
            raw += len(b)
            b = source_select_selection(b, cuts=DEFAULT_SELECTION_CUTS)
            if len(b) == 0:
                continue
            b = b[b["detected"].astype(bool)].reset_index(drop=True)
            if snr_min is not None:
                snr = b["measured_flux_auto"].to_numpy(float) / b["measured_fluxerr_auto"].to_numpy(float)
                b = b[np.isfinite(snr) & (snr > snr_min)].reset_index(drop=True)
            if len(b) == 0:
                continue
            b = add_measurement_target_features(b)
            keep = np.isfinite(b["measured_e1_image"]) & np.isfinite(b["measured_e2_image"])
            b = b[keep].reset_index(drop=True)
            if len(b) == 0:
                continue
            b["__k"] = rng.random(len(b))
            reservoir = b if reservoir is None else pd.concat([reservoir, b], ignore_index=True)
            if len(reservoir) > 2 * max_rows:
                reservoir = reservoir.nlargest(max_rows, "__k").reset_index(drop=True)
    reservoir = reservoir.nlargest(min(max_rows, len(reservoir)), "__k").reset_index(drop=True)
    return reservoir, raw


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--g0", default=os.path.join(CD, "det_meas_g0.0_train.feather"))
    ap.add_argument("--sheared", nargs="+",
                    default=[f"{CD}/det_meas_g0.05_val.feather:0.05",
                             f"{CD}/det_meas_g0.2_val.feather:0.2"])
    ap.add_argument("--g0-rows", type=int, default=1_500_000)
    ap.add_argument("--sheared-rows", type=int, default=1_000_000)
    ap.add_argument("--max-batches", type=int, default=None)
    ap.add_argument("--snr-min", type=float, default=None)
    ap.add_argument("--shear-threshold", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--max-leaves", type=int, default=63, help="HGB max_leaf_nodes (lower=more regularized)")
    ap.add_argument("--min-leaf", type=int, default=200, help="HGB min_samples_leaf (higher=more regularized)")
    args = ap.parse_args()

    print(f"[g0] {args.g0}")
    g0, g0raw = _stream(args.g0, args.g0_rows, args.snr_min, args.seed, args.max_batches)
    print(f"[g0] rows={len(g0):,} (raw {g0raw:,})")
    y1 = g0["measured_e1_image"].to_numpy(float)
    y2 = g0["measured_e2_image"].to_numpy(float)
    e1g = g0["e1_input_rot0_p"].to_numpy(float)
    e2g = g0["e2_input_rot0_p"].to_numpy(float)
    Xg_full = _build_features(g0, e1g, e2g)

    # Covariate-shift driver: does intrinsic shape magnitude correlate with size/flux/sersic
    # at g=0?  If so, a model conditioned on those exploits a correlation that shear (which
    # shifts shape independently of size) breaks -> transfer failure.
    eabs = np.hypot(e1g, e2g)
    print("[g0] corr(|e_scene|, .): "
          + "  ".join(f"{k}={np.corrcoef(eabs, Xg_full[k])[0,1]:+.3f}"
                      for k in ["logRe", "mag", "sersic"]))

    sheared = []
    for spec in args.sheared:
        path, nominal = spec.rsplit(":", 1)
        sh, _ = _stream(path, args.sheared_rows, args.snr_min, args.seed + 1, args.max_batches)
        g1 = sh["gamma1_input_p"].to_numpy(float); g2 = sh["gamma2_input_p"].to_numpy(float)
        gm = np.hypot(g1, g2); msk = gm > args.shear_threshold
        sh = sh[msk].reset_index(drop=True)
        g1, g2, gm = g1[msk], g2[msk], gm[msk]
        ei1 = sh["e1_input_rot0_p"].to_numpy(float); ei2 = sh["e2_input_rot0_p"].to_numpy(float)
        se1, se2 = apply_shear_to_ellipticity(ei1, ei2, g1, g2)   # scene shape = S_g(intrinsic)
        gh1, gh2 = g1 / gm, g2 / gm
        am1 = sh["measured_e1_image"].to_numpy(float); am2 = sh["measured_e2_image"].to_numpy(float)
        sheared.append(dict(nominal=float(nominal), sh=sh, se1=se1, se2=se2,
                            gh1=gh1, gh2=gh2, act_par=(am1 * gh1 + am2 * gh2)))
        print(f"[sheared g={nominal}] sheared rows={len(sh):,}  <act_par>={(am1*gh1+am2*gh2).mean():+.5f}")

    print(f"\nHGB regularization: max_leaf_nodes={args.max_leaves} min_samples_leaf={args.min_leaf}")
    print(f"{'feature set':14s} " + "  ".join(f"g={d['nominal']}: pred/act (m_impl)" for d in sheared))
    for name, feats in FEATURE_SETS.items():
        m1 = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                                           max_leaf_nodes=args.max_leaves, min_samples_leaf=args.min_leaf)
        m2 = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                                           max_leaf_nodes=args.max_leaves, min_samples_leaf=args.min_leaf)
        m1.fit(Xg_full[feats], y1)
        m2.fit(Xg_full[feats], y2)
        cells = []
        for d in sheared:
            Xs = _build_features(d["sh"], d["se1"], d["se2"])[feats]
            p1 = m1.predict(Xs); p2 = m2.predict(Xs)
            pred_par = (p1 * d["gh1"] + p2 * d["gh2"]).mean()
            act_par = d["act_par"].mean()
            ratio = pred_par / act_par
            cells.append(f"{ratio:6.4f} ({ratio-1:+.4f})")
        print(f"{name:14s} " + "        ".join(cells))
    print("\npred/act = forward responsivity / data responsivity.  ->1 means the feature set")
    print("captures the response; m_impl is the residual forward over/under-prediction (the bias).")
    print("The feature whose addition collapses m_impl toward 0 is the bias source.")


if __name__ == "__main__":
    main()
