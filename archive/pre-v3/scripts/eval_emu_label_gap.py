"""Does the emulator FIT its own training labels at <1", or are the labels themselves the problem?

WHY THIS EXISTS
---------------
The certified R_blend emulator under-predicts the half-shear ruler by -41.5% for pairs closer than
1", -5.4% at 1-2", -1.4% at 2-3" (WORKLOG 2026-07-28j/l). Three fixes have failed: restricting
training to our domain (-40.3%), close-pair loss weighting (saturates at -37%), and the pair-angle
hypothesis (deficit survives angle-averaging; the sim does not shear positions at all).

Every one of those attacked the MODEL. None asked whether the model's own labels agree with the
ruler. This asks that. Two outcomes, and they point opposite ways:

  * emulator ~= its own labels at <1", but the labels sit BELOW the ruler
        -> the two datasets measure different things. A ruler/definition mismatch, not a fit
           failure. Retraining can never fix it; matching the definition can.
  * emulator UNDER-predicts its own labels at <1"
        -> genuine representational limit, and the feature set is the thing to change.

TWO CANDIDATE MISMATCHES, both visible from the label-building code (blendemu/response.py):

1. BRIGHT-NEIGHBOUR REJECTION. `retrieve_response` calls
       remove_detection_w_bright_neighbour(..., ratio_max=5, r_min=0, r_max=3/3600)
   which DROPS every primary that has a detected neighbour within 3" more than 5x brighter
   (i.e. mag_s < mag_p - 1.7485). The half-shear ruler applies no such cut. That removes exactly
   the close, high-contrast pairs which carry the largest blending response -- and its 3" radius
   matches where the deficit lives (-41.5% / -5.4% / -1.4% / ~0 beyond 3").

2. SHEAR AMPLITUDE. The label is a forward difference over Dg = 0.2 (`response.shear_values =
   [0.0, 0.2]`, divided by 0.2 in train_emulator). The ruler uses Dg = 0.05. Equal only if the
   blend response is linear in shear -- least likely exactly where the profiles overlap.

WHAT THIS SCRIPT MEASURES
-------------------------
PART 1  label vs emulator prediction, binned by separation, on the emulator's own training
        catalogue under the certified regression cuts + our deliverable domain. Split by
        case>=40 (trained on) vs case<40 (held out) so a fit failure is not confused with overfit.
PART 2  the neighbour-minus-primary magnitude contrast distribution at <3". If hypothesis 1 is
        right, the response catalogue must be almost empty below mag_s - mag_p = -1.7485, while the
        ruler catalogue is not. This is a sharp, falsifiable prediction of a truncation edge.

Reads are streamed batch-by-batch (the catalogue is 27 GB); only running sums are kept.

FIREWALL NOTE: this reads the emulator's own training catalogue, never constgold. The case<40 split
is reported for diagnosis only and tunes nothing.
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

BE = "/home/z/Zekang.Zhang/blendemu"
RESP_CAT = "/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876/response_catalogue_train.feather"
BLEND_MODELS = os.path.join(BE, "models")
COND = dict(pixel_size=0.2, zero_point=30.0, psf_fwhm=0.73, moffat_beta=2.224, pixel_rms=0.312)

PAIR_FEATURES = ["Re_input_p", "r_input_p", "sersic_n_input_p",
                 "Re_input_s", "r_input_s", "sersic_n_input_s", "distance"]
NEED = ["case", "delta_et1"] + PAIR_FEATURES

# certified regression cuts, in blendemu's cut order [r_s, r_p, Re_s, Re_p, distance]
REG_CUTS = [[13, 29], [18, 28], [0.0, 10.0], [0.1, 1.5], [0, 10]]
CUT_ORDER = ["r_input_s", "r_input_p", "Re_input_s", "Re_input_p", "distance"]

# blendemu's own rejection threshold, expressed as a magnitude contrast
CONTRAST_EDGE = -2.5 * np.log10(5.0)          # = -1.7485 ; mag_s - mag_p below this was DROPPED

DIST_EDGES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0]
CONTRAST_EDGES = [-8.0, -4.0, -3.0, -2.5, -2.0, -1.7485, -1.5, -1.0, 0.0, 2.0, 4.0, 8.0]


class BinAcc:
    """Running (sum, sum of squares, count) per bin -- constant memory over a streamed read."""

    def __init__(self, edges, nseries):
        self.edges = np.asarray(edges, float)
        self.n = len(self.edges) - 1
        self.count = np.zeros(self.n)
        self.s = np.zeros((nseries, self.n))
        self.ss = np.zeros((nseries, self.n))

    def add(self, key, series):
        idx = np.digitize(key, self.edges) - 1
        ok = (idx >= 0) & (idx < self.n)
        idx = idx[ok]
        if not idx.size:
            return
        self.count += np.bincount(idx, minlength=self.n)
        for j, v in enumerate(series):
            v = np.asarray(v, float)[ok]
            self.s[j] += np.bincount(idx, weights=v, minlength=self.n)
            self.ss[j] += np.bincount(idx, weights=v * v, minlength=self.n)

    def mean(self, j):
        with np.errstate(invalid="ignore", divide="ignore"):
            return self.s[j] / self.count

    def sem(self, j):
        with np.errstate(invalid="ignore", divide="ignore"):
            var = self.ss[j] / self.count - (self.s[j] / self.count) ** 2
            return np.sqrt(np.maximum(var, 0.0) / self.count)


def apply_cuts(df, cuts=REG_CUTS, order=CUT_ORDER):
    """blendemu's source_select_reg, inlined so the cut order is auditable here."""
    m = np.ones(len(df), bool)
    for col, (lo, hi) in zip(order, cuts):
        v = df[col].to_numpy(float)
        m &= (v >= lo) & (v <= hi)
    return df[m]


def part1(args, predictor):
    """Label vs emulator prediction on the emulator's OWN training catalogue."""
    print("\n" + "=" * 78)
    print("PART 1: emulator vs its OWN training labels  (label = delta_et1 / %.2f)" % args.shear)
    print("=" * 78, flush=True)

    accs = {k: BinAcc(DIST_EDGES, 2) for k in ("train", "heldout")}
    close = {k: BinAcc(CONTRAST_EDGES, 2) for k in ("train", "heldout")}
    n_raw = n_kept = 0
    t0 = time.time()

    scene_features = [name for name in predictor.bst_reg.feature_names
                      if name.startswith("scene_logflux_")]
    if scene_features:
        print(f"additional raw pair features: {scene_features}")
    with ipc.open_file(args.cat) as r:
        nb = r.num_record_batches
        missing = set(scene_features) - set(r.schema.names)
        if missing:
            raise KeyError(f"catalogue lacks model scene features: {sorted(missing)}")
        for bi in range(0, nb, args.stride):
            b = pa.Table.from_batches([r.get_batch(bi)]).select(
                [*NEED, *scene_features]).to_pandas()
            n_raw += len(b)
            b = b[np.isfinite(b["delta_et1"].to_numpy(float))]
            b = apply_cuts(b)
            # deliverable domain: PRIMARY only; neighbours stay full-population
            b = b[(b["Re_input_p"].to_numpy(float) > args.true_re_min)
                  & (b["r_input_p"].to_numpy(float) < args.true_mag_max)]
            if not len(b):
                continue
            n_kept += len(b)

            label = b["delta_et1"].to_numpy(float) / args.shear
            pred = predictor.predict_on_pairs(
                b[[*PAIR_FEATURES, *scene_features]].copy(),
                task="response")["response"].to_numpy(float)
            dist = b["distance"].to_numpy(float)
            case = b["case"].to_numpy(int)
            contrast = b["r_input_s"].to_numpy(float) - b["r_input_p"].to_numpy(float)

            for key, sel in (("train", case >= args.heldout_min),
                             ("heldout", case < args.heldout_min)):
                if not sel.any():
                    continue
                accs[key].add(dist[sel], (label[sel], pred[sel]))
                c = sel & (dist < args.close_max)
                if c.any():
                    close[key].add(contrast[c], (label[c], pred[c]))

            if (bi // max(args.stride, 1)) % 200 == 0:
                print(f"  batch {bi}/{nb}  kept={n_kept:,}  ({time.time()-t0:.0f}s)", flush=True)

    print(f"\nstreamed {n_raw:,} rows (stride={args.stride}); {n_kept:,} pass cuts + domain "
          f"({time.time()-t0:.0f}s)")

    for key, title in (("train", f"cases >= {args.heldout_min}  (TRAINED ON)"),
                       ("heldout", f"cases < {args.heldout_min}  (held out)")):
        a = accs[key]
        print(f"\n[by PAIR SEPARATION -- {title}]")
        print(f"  {'distance':>16} {'label':>9} {'emulator':>9} {'diff':>9} "
              f"{'emu/label-1 %':>14} {'sem':>8} {'N':>12}")
        lab, pre, sem = a.mean(0), a.mean(1), a.sem(0)
        for i in range(a.n):
            if a.count[i] < 200:
                continue
            rel = (pre[i] / lab[i] - 1) * 100 if lab[i] else np.nan
            print(f"  [{a.edges[i]:>6.2f},{a.edges[i+1]:>6.2f}) {lab[i]:>9.4f} {pre[i]:>9.4f} "
                  f"{pre[i]-lab[i]:>+9.4f} {rel:>+14.2f} {sem[i]:>8.4f} {int(a.count[i]):>12,}")

    for key, title in (("train", f"cases >= {args.heldout_min}"),
                       ("heldout", f"cases < {args.heldout_min}")):
        c = close[key]
        if c.count.sum() < 200:
            continue
        print(f"\n[by MAG CONTRAST mag_s - mag_p, pairs < {args.close_max}\" -- {title}]")
        print(f"  {'contrast':>16} {'label':>9} {'emulator':>9} {'emu/label-1 %':>14} {'N':>12}")
        lab, pre = c.mean(0), c.mean(1)
        for i in range(c.n):
            if c.count[i] < 200:
                continue
            rel = (pre[i] / lab[i] - 1) * 100 if lab[i] else np.nan
            edge = "  <-- blendemu rejection edge" if abs(c.edges[i] - CONTRAST_EDGE) < 1e-3 else ""
            print(f"  [{c.edges[i]:>6.2f},{c.edges[i+1]:>6.2f}) {lab[i]:>9.4f} {pre[i]:>9.4f} "
                  f"{rel:>+14.2f} {int(c.count[i]):>12,}{edge}")

    return accs, close


def part2(args):
    """Truncation test: is the response catalogue missing the bright-neighbour pairs?"""
    print("\n" + "=" * 78)
    print("PART 2: mag-contrast distribution -- is the training population truncated?")
    print(f"        blendemu drops primaries with a detected neighbour <3\" and >5x brighter,")
    print(f"        i.e. mag_s - mag_p < {CONTRAST_EDGE:+.4f}. Prediction: the response catalogue")
    print("        is nearly EMPTY below that edge at <3\"; the ruler catalogue is not.")
    print("=" * 78, flush=True)

    hist = np.zeros(len(CONTRAST_EDGES) - 1)
    tot = 0
    with ipc.open_file(args.cat) as r:
        for bi in range(0, r.num_record_batches, args.stride):
            b = pa.Table.from_batches([r.get_batch(bi)]).select(
                ["distance", "r_input_p", "r_input_s", "Re_input_p"]).to_pandas()
            b = b[(b["distance"].to_numpy(float) < 3.0)
                  & (b["Re_input_p"].to_numpy(float) > args.true_re_min)
                  & (b["r_input_p"].to_numpy(float) < args.true_mag_max)]
            if not len(b):
                continue
            con = b["r_input_s"].to_numpy(float) - b["r_input_p"].to_numpy(float)
            hist += np.histogram(con, bins=CONTRAST_EDGES)[0]
            tot += len(b)

    print(f"\n  response catalogue, pairs < 3\", in-domain primary: N={tot:,}")
    print(f"  {'contrast bin':>18} {'N':>12} {'frac %':>9}")
    for i in range(len(hist)):
        edge = "  <-- rejection edge" if abs(CONTRAST_EDGES[i] - CONTRAST_EDGE) < 1e-3 else ""
        print(f"  [{CONTRAST_EDGES[i]:>7.3f},{CONTRAST_EDGES[i+1]:>7.3f}) {int(hist[i]):>12,} "
              f"{100*hist[i]/max(tot,1):>9.3f}{edge}")
    below = hist[np.array(CONTRAST_EDGES[:-1]) < CONTRAST_EDGE - 1e-9].sum()
    print(f"\n  fraction with mag_s - mag_p < {CONTRAST_EDGE:+.4f} (should be ~0 if the cut bites): "
          f"{100*below/max(tot,1):.3f}%   N={int(below):,}")
    return hist, tot


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="lsst_r_extnbr_ho")
    ap.add_argument("--cat", default=RESP_CAT,
                    help="response catalogue to read. Point at the corrected copy from "
                         "fix_response_distance.py to bin the labels by INPUT-frame separation, "
                         "which is the frame the ruler is binned in -- the original binning is in "
                         "the detected frame, so the two tables were not on the same x-axis.")
    ap.add_argument("--shear", type=float, default=0.2, help="label finite-difference step")
    ap.add_argument("--stride", type=int, default=1, help="read every Nth record batch")
    ap.add_argument("--heldout-min", type=int, default=40)
    ap.add_argument("--true-re-min", type=float, default=0.3)
    ap.add_argument("--true-mag-max", type=float, default=26.0)
    ap.add_argument("--close-max", type=float, default=1.0)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    sys.path.insert(0, BE)
    from blendemu.inference import BlendingPredictor
    predictor = BlendingPredictor.load(BLEND_MODELS, tag=args.tag, conditions=COND, device="cpu")
    print(f"emulator tag={args.tag}   label step Dg={args.shear}   ruler step Dg=0.05")
    print("RULER reference (half-shear, in-domain): <1\" 0.0518   1-2\" 0.0261")

    accs, close = part1(args, predictor)
    hist, tot = part2(args)

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        np.savez(args.output,
                 dist_edges=DIST_EDGES, contrast_edges=CONTRAST_EDGES,
                 train_count=accs["train"].count, train_label=accs["train"].mean(0),
                 train_pred=accs["train"].mean(1),
                 heldout_count=accs["heldout"].count, heldout_label=accs["heldout"].mean(0),
                 heldout_pred=accs["heldout"].mean(1),
                 contrast_hist=hist, contrast_total=tot)
        print(f"\nsaved {args.output}")
    print("EMU_LABEL_GAP_DONE", flush=True)


if __name__ == "__main__":
    main()
