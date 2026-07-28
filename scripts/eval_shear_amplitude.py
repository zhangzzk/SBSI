"""Same pairs, two mismatches: is the -41.5% close-pair deficit an amplitude or a distance artefact?

WHY THIS EXISTS
---------------
Job 15328417 showed the emulator basically FITS its own training labels at close separation
(-18.9% at <0.5", +6.2% at 0.5-1") while missing the half-shear ruler by -41.5%. So the deficit is
mostly a disagreement between the LABELS and the RULER, not a fit failure. Job 15328418 killed the
bright-neighbour-rejection explanation (it removes 0.7% of close pairs and moves nothing).

Both catalogues are built from the SAME simulation suite (lsst_sims_fs2_25876), the same input
galaxies, and the same ngmix estimator, so they can be compared PAIR BY PAIR -- which removes every
population argument at once. Two concrete mismatches survive, and this script tests both.

(A) SHEAR AMPLITUDE. The two numbers are different finite differences of the same function:

        emulator label : (e(g_s=0.2)  - e(0)) / 0.2      <- blendemu response catalogue
        ruler truth    : (e(g_s=0.05) - e(0)) / 0.05     <- half-shear catalogue

    Equal only if the blend response is LINEAR in the neighbour's shear out to |g|=0.2 -- least
    likely at sub-arcsecond separations, where the profiles overlap and the shape response saturates.

(B) DISTANCE DEFINITION -- a train/inference mismatch in the emulator's most important feature.
    `retrieve_response` measures the pair separation from the primary's DETECTED centroid:

        pri_pos = scat1[['X_WORLD', 'Y_WORLD']].loc[idx_det1]     # detection, unsheared leg
        dst, ind = kdt_in.query(pri_pos, ...)                     # to INPUT secondary positions

    while `_nearest_neighbor_features` -- which builds the half-shear catalogue the ruler and the m
    pipeline both use -- measures it from INPUT positions at both ends:

        all_pos = icat[['RA_input', 'DEC_input']]                 # input, both ends

    A blended primary's detected centroid is pulled toward its neighbour, so the training distance is
    systematically SMALLER than the input separation for exactly the close pairs in question. Query
    the model at input separation d and it returns what it learned for pairs whose DETECTED
    separation was d -- pairs of larger true separation, hence weaker response. That is an
    under-prediction, concentrated at small d and vanishing at large d: the observed signature.

WHAT IT MEASURES
----------------
Joins the response catalogue to the ruler on (case, primary input_index, neighbour sky position),
then per separation bin reports:

  * label vs truth on the SAME ROWS, and their ratio            -> tests (A)
  * detected-frame vs input-frame separation for the same pair  -> measures (B) directly
  * the emulator re-evaluated with the DETECTED distance it was trained on, against the same truth
    -> if the deficit closes, (B) is the cause, and the fix is to make the two definitions agree

The ratios are per-pair, so shape noise cancels far better than in either mean alone. A `matched`
count per bin is printed so thin bins can be discounted.

FIREWALL: reads the emulator's training catalogue and the half-shear legs. constgold is never read.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.eval_rblend_gap import (  # noqa: E402
    BLEND_MODELS, CAT, COND, PAIR_FEATURES, blend_truth, load_legs)

RESP_CAT = "/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876/response_catalogue_train.feather"
DIST_EDGES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
POS_ROUND = 9          # degrees; both catalogues copy RA/DEC from the same input catalogue


def load_response_pairs(max_dist, re_min, mag_max, stride=1, verbose=True):
    """Streamed read of the emulator's training catalogue, close pairs on the deliverable domain."""
    cols = ["case", "input_index", "RA_input_s", "DEC_input_s", "distance", "delta_et1",
            "Re_input_p", "r_input_p"]
    parts = []
    t0 = time.time()
    with ipc.open_file(RESP_CAT) as r:
        nb = r.num_record_batches
        for bi in range(0, nb, stride):
            b = pa.Table.from_batches([r.get_batch(bi)]).select(cols).to_pandas()
            b = b[(b["distance"].to_numpy(float) < max_dist)
                  & np.isfinite(b["delta_et1"].to_numpy(float))
                  & (b["Re_input_p"].to_numpy(float) > re_min)
                  & (b["r_input_p"].to_numpy(float) < mag_max)]
            if len(b):
                parts.append(b[["case", "input_index", "RA_input_s", "DEC_input_s",
                                "distance", "delta_et1"]])
            if verbose and bi % 500 == 0:
                print(f"  batch {bi}/{nb}  ({time.time()-t0:.0f}s)", flush=True)
    d = pd.concat(parts, ignore_index=True)
    if verbose:
        print(f"response catalogue: {len(d):,} in-domain pairs < {max_dist}\" "
              f"({time.time()-t0:.0f}s)", flush=True)
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gs-leg", default=CAT + "det_meas_ngmix_g0.05_val.feather")
    ap.add_argument("--g0-leg", default=CAT + "det_meas_ngmix_g0.0_train.feather")
    ap.add_argument("--max-case", type=int, default=None)
    ap.add_argument("--max-dist", type=float, default=5.0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--label-shear", type=float, default=0.2)
    ap.add_argument("--true-re-min", type=float, default=0.3)
    ap.add_argument("--true-mag-max", type=float, default=26.0)
    ap.add_argument("--tag", default="lsst_r_extnbr_ho")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    # --- ruler side (Dg = 0.05, INPUT-frame distance) ---
    base = load_legs(args.gs_leg, args.g0_leg, args.max_case, args.true_re_min, args.true_mag_max)
    base = base.copy()
    base["truth"] = blend_truth(base)
    null = blend_truth(base, rotate45=True)
    g = np.isfinite(null)
    nm, nsem = null[g].mean(), null[g].std(ddof=1) / np.sqrt(g.sum())
    print(f"\nNULL TEST (45-deg rotated, must be ~0): {nm:+.5f} +- {nsem:.5f} "
          f"({abs(nm)/nsem:.1f} sigma)  N={g.sum():,}")

    # --- label side (Dg = 0.2, DETECTED-frame distance) ---
    resp = load_response_pairs(args.max_dist, args.true_re_min, args.true_mag_max, args.stride)
    resp = resp.rename(columns={"distance": "distance_det"})
    resp["label"] = resp["delta_et1"].to_numpy(float) / args.label_shear

    # --- join on (case, primary, neighbour sky position) ---
    for df in (base, resp):
        df["_ra_s"] = np.round(df["RA_input_s"].to_numpy(float), POS_ROUND)
        df["_dec_s"] = np.round(df["DEC_input_s"].to_numpy(float), POS_ROUND)
    keys = ["case", "input_index", "_ra_s", "_dec_s"]
    left_cols = keys + ["truth"] + [c for c in PAIR_FEATURES if c not in keys]
    m = base[left_cols].merge(resp[keys + ["distance_det", "label"]], on=keys, how="inner")
    m = m.drop_duplicates(keys)
    print(f"\nmatched pairs present in BOTH catalogues: {len(m):,} "
          f"(ruler {len(base):,}, response {len(resp):,})")
    if len(m) < 1000:
        print("*** join failed -- the two catalogues do not share pair keys; "
              "cannot compare per-pair. ***")
        print("SHEAR_AMPLITUDE_DONE", flush=True)
        return

    sys.path.insert(0, "/home/z/Zekang.Zhang/blendemu")
    from blendemu.inference import BlendingPredictor
    model = BlendingPredictor.load(BLEND_MODELS, tag=args.tag, conditions=COND, device="cpu")

    # as the m pipeline queries it: INPUT-frame distance
    pred_in = model.predict_on_pairs(m[PAIR_FEATURES].copy(),
                                     task="response")["response"].to_numpy(float)
    # as the emulator was TRAINED: detected-frame distance, everything else identical
    feat_det = m[PAIR_FEATURES].copy()
    feat_det["distance"] = m["distance_det"].to_numpy(float)
    pred_det = model.predict_on_pairs(feat_det, task="response")["response"].to_numpy(float)

    truth = m["truth"].to_numpy(float)
    label = m["label"].to_numpy(float)
    d_in = m["distance"].to_numpy(float)
    d_det = m["distance_det"].to_numpy(float)
    ok = np.isfinite(truth) & np.isfinite(label)

    # ---------- (B) how different are the two distance definitions? ----------
    print(f"\n[DISTANCE DEFINITION: detected-frame (training) vs input-frame (inference)]")
    print(f"  {'input dist':>16} {'<d_det>':>9} {'<d_in>':>9} {'<d_det-d_in>':>13} "
          f"{'median ratio':>13} {'frac closer':>12} {'matched':>10}")
    for i in range(len(DIST_EDGES) - 1):
        sel = ok & (d_in >= DIST_EDGES[i]) & (d_in < DIST_EDGES[i + 1])
        n = int(sel.sum())
        if n < 200:
            continue
        dd, di = d_det[sel], d_in[sel]
        print(f"  [{DIST_EDGES[i]:>6.2f},{DIST_EDGES[i+1]:>6.2f}) {dd.mean():>9.4f} "
              f"{di.mean():>9.4f} {(dd-di).mean():>+13.4f} {np.median(dd/di):>13.4f} "
              f"{(dd < di).mean():>12.3f} {n:>10,}")

    # ---------- (A) amplitude, same rows ----------
    print(f"\n[SAME PAIRS, two shear amplitudes]  label=(e(0.2)-e(0))/0.2   "
          f"truth=(e(0.05)-e(0))/0.05")
    print(f"  {'input dist':>16} {'truth(.05)':>11} {'label(0.2)':>11} {'label/truth':>16} "
          f"{'sem(t)':>8} {'matched':>10}")
    rows = []
    for i in range(len(DIST_EDGES) - 1):
        sel = ok & (d_in >= DIST_EDGES[i]) & (d_in < DIST_EDGES[i + 1])
        n = int(sel.sum())
        if n < 200:
            continue
        t, l = truth[sel].mean(), label[sel].mean()
        semt = truth[sel].std(ddof=1) / np.sqrt(n)
        seml = label[sel].std(ddof=1) / np.sqrt(n)
        ratio = l / t if t else np.nan
        rsem = abs(ratio) * np.sqrt((seml / l) ** 2 + (semt / t) ** 2) if (l and t) else np.nan
        rows.append((DIST_EDGES[i], DIST_EDGES[i + 1], t, l, ratio, rsem, n))
        print(f"  [{DIST_EDGES[i]:>6.2f},{DIST_EDGES[i+1]:>6.2f}) {t:>11.4f} {l:>11.4f} "
              f"{ratio:>9.3f}+-{rsem:<6.3f} {semt:>8.4f} {n:>10,}")

    # ---------- does using the TRAINED distance definition close the deficit? ----------
    print(f"\n[EMULATOR vs the SAME truth, queried two ways]")
    print(f"  {'input dist':>16} {'truth':>9} {'emu(input d)':>13} {'err %':>8} "
          f"{'emu(det d)':>11} {'err %':>8} {'matched':>10}")
    for i in range(len(DIST_EDGES) - 1):
        sel = ok & (d_in >= DIST_EDGES[i]) & (d_in < DIST_EDGES[i + 1])
        n = int(sel.sum())
        if n < 200:
            continue
        t = truth[sel].mean()
        pi, pd_ = pred_in[sel].mean(), pred_det[sel].mean()
        print(f"  [{DIST_EDGES[i]:>6.2f},{DIST_EDGES[i+1]:>6.2f}) {t:>9.4f} {pi:>13.4f} "
              f"{(pi/t-1)*100 if t else np.nan:>+8.2f} {pd_:>11.4f} "
              f"{(pd_/t-1)*100 if t else np.nan:>+8.2f} {n:>10,}")
    t = truth[ok].mean()
    print(f"  {'ALL MATCHED':>16} {t:>9.4f} {pred_in[ok].mean():>13.4f} "
          f"{(pred_in[ok].mean()/t-1)*100:>+8.2f} {pred_det[ok].mean():>11.4f} "
          f"{(pred_det[ok].mean()/t-1)*100:>+8.2f} {int(ok.sum()):>10,}")

    print("\nREAD THIS AS:")
    print("  label/truth ~ 1            -> response linear in shear; amplitude is NOT the cause")
    print("  d_det << d_in at small d   -> the training distance really is a different variable")
    print("  emu(det d) beats emu(input d) -> (B) is the cause; fix the definition, do not retrain")

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        np.savez(args.output, truth=truth, label=label, pred_in=pred_in, pred_det=pred_det,
                 d_in=d_in, d_det=d_det, rows=np.array(rows, dtype=float))
        print(f"\nsaved {args.output}")
    print("SHEAR_AMPLITUDE_DONE", flush=True)


if __name__ == "__main__":
    main()
