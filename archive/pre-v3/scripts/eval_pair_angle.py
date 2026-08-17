"""Two decision-independent checks for Gold-V3 flow #2, in one pass over the half-shear legs.

CHECK 1 -- DOES THE SIM SHEAR POSITIONS, OR ONLY SHAPES?
-------------------------------------------------------
`shifted_feature_frame` (train_joint_forward.py) shears intrinsic SHAPES only; the separation
vector is never moved. If the sim shears the whole scene, part of the true blend response arrives
through geometry, and a loss that shifts only shapes would force the entire response through the
shape channel -- fitting the training bins and generalising wrong (Gold-V3.md "position shear is
unmodeled"). Settled by comparing the primary->neighbour separation for the SAME (case,
input_index) between the g=0 leg and the sheared leg:

    identical  -> shape-only shear; the existing shift machinery already matches the sim.
    different  -> the loss shift must shear the separation vector too.

Reported as both the scalar `distance` and the RA/DEC-derived vector, since `distance` could be a
stored input rather than a recomputed one. A whole-scene shear moves the separation in the
characteristic spin-2 way, so the check also regresses the fractional separation change on
cos 2(phi_pair - phi_gamma); a nonzero slope of order |gamma| is the signature.

CHECK 2 -- IS THERE REAL SPIN-2 SIGNAL IN THE PAIR ANGLE?
--------------------------------------------------------
BlendEMU sees `distance` but never the pair ORIENTATION, so its prediction is flat in the pair
angle by construction. Two distinct questions, and only the second bears on the -41% close-pair
deficit (WORKLOG 2026-07-28i..l):

  (a) Does the TRUTH vary with the pair angle relative to the neighbour's shear direction?
      If yes, the spin-2 channel carries real signal and is worth conditioning flow #2 on.
      If flat, the neighbour-shape channel is the only one that matters and the separation
      vector can stay scalar.

  (b) Does the MEAN residual survive averaging over that angle? An omitted variable costs a
      regression SCATTER, not a biased conditional mean -- E[R|scalars] is exactly the
      angle-average -- so orientation-blindness alone should NOT produce the -41%. If the
      deficit is flat in angle, the missing-orientation story is dead and the live candidate is
      close-pair DETECTION SELECTION: our truth is conditioned on both objects being detected,
      and at sub-arcsecond separation detection itself depends on orientation, so the population
      we average over is not the one the emulator was trained on.

Truth is the per-pair blend response from eval_rblend_gap.py (both-sheared leg projected on the
neighbour's independent shear direction); see that module's docstring for why the neighbour-only
leg is unusable (no ngmix shapes there). FIREWALL: half-shear legs only, constgold never read.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.feather as pf

SBSI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SBSI_ROOT not in sys.path:
    sys.path.insert(0, SBSI_ROOT)

from scripts.eval_rblend_gap import (  # noqa: E402
    BLEND_MODELS, CAT, COND, NGMIX, PAIR_FEATURES, blend_truth)

POS = ["RA_input_p", "DEC_input_p", "RA_input_s", "DEC_input_s"]
GAMMA = ["gamma1_input_p", "gamma2_input_p", "gamma1_input_s", "gamma2_input_s"]


def sep_vector(d):
    """Primary->neighbour separation in arcsec (RA compressed by cos(DEC)), and its position angle."""
    ra_p = d["RA_input_p"].to_numpy(float)
    de_p = d["DEC_input_p"].to_numpy(float)
    ra_s = d["RA_input_s"].to_numpy(float)
    de_s = d["DEC_input_s"].to_numpy(float)
    dx = (ra_s - ra_p) * np.cos(np.deg2rad(0.5 * (de_p + de_s))) * 3600.0
    dy = (de_s - de_p) * 3600.0
    return dx, dy, np.hypot(dx, dy), np.arctan2(dy, dx)


def load(gs_leg, g0_leg, max_case, re_min, mag_max):
    """Both-sheared rows of the sheared leg matched to the unsheared leg, positions kept from BOTH
    legs so the separation can be compared across them."""
    t0 = time.time()
    cols = ["case", "input_index", "detected", "neighbored", "distance"] + PAIR_FEATURES + NGMIX + GAMMA + POS
    cols = list(dict.fromkeys(cols))
    d = pf.read_table(gs_leg, columns=cols).to_pandas()
    if max_case is not None:
        d = d[d["case"] < max_case]
    gp = np.hypot(d["gamma1_input_p"].to_numpy(float), d["gamma2_input_p"].to_numpy(float))
    gs = np.hypot(d["gamma1_input_s"].to_numpy(float), d["gamma2_input_s"].to_numpy(float))
    nb = d[(gp > 1e-6) & (gs > 1e-6)].copy()
    print(f"both-sheared leg: N={len(nb):,}  ({time.time()-t0:.1f}s)", flush=True)

    nb = nb[(nb["Re_input_p"].to_numpy(float) > re_min) & (nb["r_input_p"].to_numpy(float) < mag_max)]
    nb = nb[nb["detected"].astype(bool)].drop_duplicates(["case", "input_index"])

    ref = pf.read_table(g0_leg, columns=["case", "input_index", "detected", "distance"] + NGMIX + POS).to_pandas()
    if max_case is not None:
        ref = ref[ref["case"] < max_case]
    ref = ref[ref["detected"].astype(bool)].drop_duplicates(["case", "input_index"])

    base = nb.merge(ref, on=["case", "input_index"], suffixes=("_g", "_0"))
    print(f"matched both-detected, in-domain: N={len(base):,}  ({time.time()-t0:.1f}s)", flush=True)
    return base


def position_check(base):
    print("\n" + "=" * 78)
    print("CHECK 1 -- does the sim shear POSITIONS?")
    print("=" * 78)
    dd = base["distance_g"].to_numpy(float) - base["distance_0"].to_numpy(float)
    g = np.isfinite(dd)
    print(f"  scalar `distance` sheared-minus-unsheared: mean={dd[g].mean():+.3e}  "
          f"max|d|={np.abs(dd[g]).max():.3e}  frac nonzero={np.mean(np.abs(dd[g]) > 1e-9):.4f}")

    sub_g = base.rename(columns={c + "_g": c for c in POS})
    sub_0 = base.rename(columns={c + "_0": c for c in POS})
    dxg, dyg, rg, phig = sep_vector(sub_g)
    dx0, dy0, r0, phi0 = sep_vector(sub_0)
    dr = rg - r0
    g = np.isfinite(dr) & (r0 > 1e-6)
    print(f"  RA/DEC separation  sheared-minus-unsheared: mean={dr[g].mean():+.3e} arcsec  "
          f"max|d|={np.abs(dr[g]).max():.3e}  frac nonzero={np.mean(np.abs(dr[g]) > 1e-6):.4f}")
    print(f"  |RA/DEC sep| vs stored `distance` (unsheared leg): "
          f"max|diff|={np.nanmax(np.abs(r0 - base['distance_0'].to_numpy(float))):.3e} arcsec")

    # A whole-scene shear stretches the separation as dr/r = |g| cos 2(phi_pair - phi_gamma).
    g1 = base["gamma1_input_s"].to_numpy(float)
    g2 = base["gamma2_input_s"].to_numpy(float)
    gm = np.hypot(g1, g2)
    phig_s = 0.5 * np.arctan2(g2, g1)
    c2 = np.cos(2.0 * (phi0 - phig_s))
    frac = np.where(r0 > 1e-6, dr / np.maximum(r0, 1e-9), np.nan)
    ok = np.isfinite(frac) & np.isfinite(c2) & (gm > 1e-6)
    if ok.sum() > 1000:
        slope = np.polyfit(c2[ok], frac[ok], 1)[0]
        print(f"  regression of fractional separation change on cos2(phi_pair - phi_gamma_s): "
              f"slope={slope:+.5f}   (|gamma_s|={gm[ok].mean():.4f})")
        print(f"  -> {'POSITIONS ARE SHEARED' if abs(slope) > 0.2 * gm[ok].mean() else 'no position shear detected'}")
    return phi0, r0


def angle_table(truth, pred, c2, mask, label):
    print(f"\n[{label}]")
    print(f"  {'cos2(dphi)':>14} {'truth':>9} {'emulator':>9} {'emu/truth-1 %':>14} {'sem':>8} {'N':>10}")
    edges = np.array([-1.0, -0.6, -0.2, 0.2, 0.6, 1.0])
    good = mask & np.isfinite(truth) & np.isfinite(pred) & np.isfinite(c2)
    a, b = truth[good].mean(), pred[good].mean()
    sem = truth[good].std(ddof=1) / np.sqrt(good.sum())
    print(f"  {'ALL':>14} {a:>9.4f} {b:>9.4f} {(b/a-1)*100 if a else np.nan:>+14.2f} "
          f"{sem:>8.4f} {good.sum():>10,}")
    for i in range(len(edges) - 1):
        m = good & (c2 >= edges[i]) & (c2 < edges[i + 1])
        if m.sum() < 500:
            continue
        a, b = truth[m].mean(), pred[m].mean()
        sem = truth[m].std(ddof=1) / np.sqrt(m.sum())
        print(f"  [{edges[i]:>5.2f},{edges[i+1]:>5.2f}) {a:>9.4f} {b:>9.4f} "
              f"{(b/a-1)*100 if a else np.nan:>+14.2f} {sem:>8.4f} {m.sum():>10,}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gs-leg", default=CAT + "det_meas_ngmix_g0.05_val.feather")
    ap.add_argument("--g0-leg", default=CAT + "det_meas_ngmix_g0.0_train.feather")
    ap.add_argument("--max-case", type=int, default=None)
    ap.add_argument("--true-re-min", type=float, default=0.3)
    ap.add_argument("--true-mag-max", type=float, default=26.0)
    ap.add_argument("--tag", default="lsst_r_extnbr_ho")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    base = load(args.gs_leg, args.g0_leg, args.max_case, args.true_re_min, args.true_mag_max)
    phi0, r0 = position_check(base)

    # the merge already suffixed the ngmix columns _g/_0, exactly what blend_truth() expects
    truth = blend_truth(base)
    null = blend_truth(base, rotate45=True)
    ok = np.isfinite(null)
    nm, nsem = null[ok].mean(), null[ok].std(ddof=1) / np.sqrt(ok.sum())
    print(f"\nNULL TEST (45-deg rotated projection): {nm:+.5f} +- {nsem:.5f} ({abs(nm)/nsem:.1f} sigma)")
    if abs(nm) > 3 * nsem:
        print("  *** NULL FAILS -- do not trust the numbers below. ***")

    sys.path.insert(0, "/home/z/Zekang.Zhang/blendemu")
    from blendemu.inference import BlendingPredictor
    model = BlendingPredictor.load(BLEND_MODELS, tag=args.tag, conditions=COND, device="cpu")
    # `distance` was suffixed by the cross-leg merge; rebuild the emulator's feature frame with the
    # sheared-leg separation so this reproduces eval_rblend_gap.py exactly.
    feat = base[[c for c in PAIR_FEATURES if c != "distance"]].copy()
    feat["distance"] = base["distance_g"].to_numpy(float)
    feat = feat[PAIR_FEATURES]
    pred = model.predict_on_pairs(feat, task="response")["response"].to_numpy(float)

    g1 = base["gamma1_input_s"].to_numpy(float)
    g2 = base["gamma2_input_s"].to_numpy(float)
    c2 = np.cos(2.0 * (phi0 - 0.5 * np.arctan2(g2, g1)))
    dist = base["distance_g"].to_numpy(float)

    print("\n" + "=" * 78)
    print("CHECK 2 -- spin-2 structure in the pair angle (emulator is flat here by construction)")
    print("=" * 78)
    angle_table(truth, pred, c2, np.ones(len(truth), bool), "ALL separations")
    angle_table(truth, pred, c2, dist < 1.0, "CLOSE pairs (<1 arcsec) -- where the -41% lives")
    angle_table(truth, pred, c2, (dist >= 1.0) & (dist < 2.0), "1-2 arcsec")

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        np.savez(args.output, truth=truth, pred=pred, cos2dphi=c2, distance=dist, phi_pair=phi0,
                 sep_unsheared=r0, case=base["case"].to_numpy(int))
        print(f"\nsaved {args.output}")
    print("PAIR_ANGLE_DONE", flush=True)


if __name__ == "__main__":
    main()
