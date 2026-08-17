"""Apply blendemu's OWN bright-neighbour rejection to the half-shear ruler, and see if the -41% closes.

WHY THIS EXISTS
---------------
`blendemu/response.py::retrieve_response` builds the emulator's training labels after calling

    remove_detection_w_bright_neighbour(X_WORLD, Y_WORLD, FLUX_AUTO,
                                        ratio_max=5, r_min=0, r_max=3/3600)

which DROPS every primary that has a detected neighbour within 3" more than 5x brighter. The
half-shear ruler in `eval_rblend_gap.py` applies no such cut. So the emulator is scored on a
population its labels were never allowed to contain -- and the cut's 3" radius lines up with where
the deficit lives (-41.5% at <1", -5.4% at 1-2", -1.4% at 2-3", ~0 beyond).

THE TEST: recompute the ruler truth with the same rejection applied. If the close-pair deficit
collapses, the -41% is a POPULATION mismatch, not a model failure, and no amount of retraining on
those labels could ever have fixed it.

Two versions of the cut are reported, because they bound the effect from both sides:
  FULL   -- faithful replication: KDTree over every detected object in the case, reject the target
            if ANY detected neighbour within 3" is >5x brighter. This is what blendemu does.
  PAIRED -- cheap proxy using only the row's own nearest neighbour. Strictly weaker (it can only
            reject on one neighbour), so it bounds the effect from below.

A third block reports truth and emulator binned by the magnitude contrast itself, which shows
whether the emulator's error is concentrated where the training labels were censored.

FIREWALL: half-shear legs only; constgold is never read.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.feather as pf
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.eval_rblend_gap import (  # noqa: E402
    BLEND_MODELS, CAT, COND, NGMIX, PAIR_FEATURES, blend_truth, load_legs)

RATIO_MAX = 5.0
R_REJECT = 3.0                                   # arcsec
CONTRAST_EDGE = -2.5 * np.log10(RATIO_MAX)       # -1.7485 mag
DIST_EDGES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0]
CONTRAST_EDGES = [-8.0, -4.0, -3.0, -2.5, -2.0, -1.7485, -1.5, -1.0, 0.0, 2.0, 4.0, 8.0]


def rejected_by_bright_neighbour(gs_leg, max_case, verbose=True):
    """Faithful replication of blendemu's cut: (case, input_index) of every REJECTED target.

    blendemu runs this on the SExtractor detection list (X_WORLD/Y_WORLD/FLUX_AUTO). The half-shear
    catalogue stores one row per input object with `detected` and `measured_flux_auto`, so the
    detected subset is the same object list; positions come from the input catalogue, which differs
    from the centroid only at the sub-pixel level -- irrelevant at a 3" radius.
    """
    t0 = time.time()
    cols = ["case", "input_index", "RA_input_p", "DEC_input_p", "detected", "measured_flux_auto"]
    d = pf.read_table(gs_leg, columns=cols).to_pandas()
    if max_case is not None:
        d = d[d["case"] < max_case]
    d = d[d["detected"].astype(bool)].drop_duplicates(["case", "input_index"])
    d = d[np.isfinite(d["measured_flux_auto"].to_numpy(float))]
    if verbose:
        print(f"detection field: N={len(d):,} detected objects over {d['case'].nunique()} cases "
              f"({time.time()-t0:.1f}s)", flush=True)

    out = []
    for case, g in d.groupby("case", sort=False):
        ra = g["RA_input_p"].to_numpy(float)
        dec = g["DEC_input_p"].to_numpy(float)
        flux = g["measured_flux_auto"].to_numpy(float)
        idx = g["input_index"].to_numpy(int)
        # local tangent plane in arcsec -- the case footprint is small, so this is exact enough
        x = (ra - ra.mean()) * np.cos(np.deg2rad(dec.mean())) * 3600.0
        y = (dec - dec.mean()) * 3600.0
        tree = cKDTree(np.column_stack([x, y]))
        pairs = tree.query_pairs(R_REJECT, output_type="ndarray")
        if not len(pairs):
            continue
        i, j = pairs[:, 0], pairs[:, 1]
        # a pair rejects whichever member is the fainter one, if the ratio exceeds the threshold
        with np.errstate(divide="ignore", invalid="ignore"):
            rij = flux[j] / flux[i]
        bad = np.concatenate([idx[i[rij > RATIO_MAX]], idx[j[(1.0 / rij) > RATIO_MAX]]])
        if len(bad):
            out.append(pd.DataFrame({"case": case, "input_index": np.unique(bad)}))
    rej = (pd.concat(out, ignore_index=True) if out
           else pd.DataFrame({"case": [], "input_index": []}, dtype=int))
    if verbose:
        print(f"rejected by bright neighbour (<{R_REJECT}\", >{RATIO_MAX}x): N={len(rej):,} "
              f"({100*len(rej)/max(len(d),1):.2f}% of detections)  ({time.time()-t0:.1f}s)",
              flush=True)
    return rej


def binned(truth, pred, key, edges, mask, label, keyname, note=""):
    print(f"\n[{label}]{note}")
    print(f"  {keyname:>16} {'truth':>9} {'emulator':>9} {'diff':>9} {'emu/truth-1 %':>14} "
          f"{'sem':>8} {'N':>11}")
    good = mask & np.isfinite(truth) & np.isfinite(pred)
    if good.sum():
        a, b = truth[good].mean(), pred[good].mean()
        sem = truth[good].std(ddof=1) / np.sqrt(good.sum())
        print(f"  {'OVERALL':>16} {a:>9.4f} {b:>9.4f} {b-a:>+9.4f} "
              f"{(b/a-1)*100 if a else np.nan:>+14.2f} {sem:>8.4f} {good.sum():>11,}")
    rows = []
    for i in range(len(edges) - 1):
        m = good & (key >= edges[i]) & (key < edges[i + 1])
        if m.sum() < 200:
            rows.append(np.nan)
            continue
        a, b = truth[m].mean(), pred[m].mean()
        sem = truth[m].std(ddof=1) / np.sqrt(m.sum())
        rel = (b / a - 1) * 100 if a else np.nan
        rows.append(rel)
        edge = "  <-- rejection edge" if abs(edges[i] - CONTRAST_EDGE) < 1e-3 else ""
        print(f"  [{edges[i]:>6.2f},{edges[i+1]:>6.2f}) {a:>9.4f} {b:>9.4f} {b-a:>+9.4f} "
              f"{rel:>+14.2f} {sem:>8.4f} {m.sum():>11,}{edge}")
    return rows


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

    base = load_legs(args.gs_leg, args.g0_leg, args.max_case, args.true_re_min, args.true_mag_max)
    truth = blend_truth(base)
    null = blend_truth(base, rotate45=True)
    g = np.isfinite(null)
    nm, nsem = null[g].mean(), null[g].std(ddof=1) / np.sqrt(g.sum())
    print(f"\nNULL TEST (45-deg rotated, must be ~0): {nm:+.5f} +- {nsem:.5f} "
          f"({abs(nm)/nsem:.1f} sigma)  N={g.sum():,}")
    if abs(nm) > 3 * nsem:
        print("  *** NULL FAILS -- do not trust the numbers below. ***")

    sys.path.insert(0, "/home/z/Zekang.Zhang/blendemu")
    from blendemu.inference import BlendingPredictor
    model = BlendingPredictor.load(BLEND_MODELS, tag=args.tag, conditions=COND, device="cpu")
    pred = model.predict_on_pairs(base[PAIR_FEATURES].copy(),
                                  task="response")["response"].to_numpy(float)

    dist = base["distance"].to_numpy(float)
    contrast = base["r_input_s"].to_numpy(float) - base["r_input_p"].to_numpy(float)
    keep_all = np.ones(len(base), bool)

    # --- PAIRED proxy: reject on the row's own neighbour only (weaker than blendemu's cut) ---
    keep_paired = contrast > CONTRAST_EDGE

    # --- FULL replication ---
    rej = rejected_by_bright_neighbour(args.gs_leg, args.max_case)
    if len(rej):
        key = pd.MultiIndex.from_arrays([base["case"].to_numpy(int),
                                         base["input_index"].to_numpy(int)])
        rkey = pd.MultiIndex.from_arrays([rej["case"].to_numpy(int),
                                          rej["input_index"].to_numpy(int)])
        keep_full = ~key.isin(rkey)
    else:
        keep_full = keep_all.copy()
    print(f"\nsurviving fraction of the in-domain ruler sample: "
          f"FULL {100*keep_full.mean():.2f}%   PAIRED-proxy {100*keep_paired.mean():.2f}%")
    for lo, hi in ((0, 1), (1, 2), (2, 3)):
        m = (dist >= lo) & (dist < hi)
        print(f"   {lo}-{hi}\": FULL keeps {100*keep_full[m].mean():.2f}%   "
              f"PAIRED keeps {100*keep_paired[m].mean():.2f}%   N={m.sum():,}")

    binned(truth, pred, dist, DIST_EDGES, keep_all,
           "by SEPARATION -- NO CUT (the certified -41.5% baseline)", "distance")
    binned(truth, pred, dist, DIST_EDGES, keep_full,
           "by SEPARATION -- blendemu bright-neighbour cut APPLIED (full replication)", "distance")
    binned(truth, pred, dist, DIST_EDGES, keep_paired,
           "by SEPARATION -- paired-neighbour proxy cut (lower bound on the effect)", "distance")
    binned(truth, pred, contrast, CONTRAST_EDGES, dist < 1.0,
           "by MAG CONTRAST mag_s - mag_p, pairs < 1\" -- NO CUT", "contrast")

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        np.savez(args.output, truth=truth, pred=pred, distance=dist, contrast=contrast,
                 keep_full=keep_full, keep_paired=keep_paired,
                 case=base["case"].to_numpy(int), input_index=base["input_index"].to_numpy(int))
        print(f"\nsaved {args.output}")
    print("CONTRAST_CUT_DONE", flush=True)


if __name__ == "__main__":
    main()
