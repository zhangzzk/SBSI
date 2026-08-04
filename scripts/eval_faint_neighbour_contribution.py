"""Do the neighbours the fiducial emulator DROPS actually blend? Measured on the per-pair ruler.

THE DISAGREEMENT. The fiducial emulator's stored regression config counts a neighbour only if that
neighbour has `18 <= mag < 26` and `0.3 <= Re <= 1.5` (r_max 10", k 20). Everything fainter or smaller
contributes exactly ZERO to its summed `R_blend`. Flow #2 trained on neighbours down to mag 29 and
Re 0.01, so its unrestricted sum includes them. On constgold that alone is most of a 3.4x gap in
`<R_blend>` (0.468 vs 0.136) -- but a gap is not an answer, because neither model's constgold `m` may
be used to decide which is right (the R_blend firewall: constgold is evaluation-only).

The ruler CAN decide it. The half-shear pair set carries a per-pair blend-response LABEL for every
pair, including the ones the emulator drops:

    truth = ((e1_both - e1_0)*g1hat_s + (e2_both - e2_0)*g2hat_s) / |g_s|

so the question "does a mag-27 neighbour blend?" is directly measurable. If the mean truth for dropped
pairs is consistent with zero, the emulator's cut is harmless and flow #2's unrestricted sum is adding
noise (or worse, extrapolation). If it is significantly non-zero, the emulator is missing real signal
and its `<R_blend>` is biased LOW by the amount reported here.

WHAT IS REPORTED
  * mean per-pair truth inside vs outside the emulator's neighbour selection, with cluster-robust
    errors (clustered on the primary, since a primary's pairs share its noise realisation);
  * the same split by neighbour magnitude, so any threshold behaviour is visible rather than assumed;
  * the SUMMED contribution per primary from each group -- the quantity that actually enters
    `R_blend`, which is what a mean per pair does not tell you on its own;
  * flow #2's own prediction on both groups, so its extrapolation onto the dropped pairs can be
    compared against their measured truth instead of trusted.

The per-pair label scatter is std ~3.9 in response units, so single pairs say nothing; only these
aggregates do. Errors are cluster-robust because a primary contributes several correlated pairs.

FIREWALL: reads the half-shear pair set (training-domain data) only. No constgold quantity is read and
nothing is fitted -- the flow checkpoint is scored, not trained.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

SBSI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SBSI_ROOT not in sys.path:
    sys.path.insert(0, SBSI_ROOT)

from sbs_shear.blend_flow import (  # noqa: E402
    RAW, build_features, load_blend_flow, response_from_contexts, shifted_shape_columns)

EMU_MAG = (18.0, 26.0)
EMU_RE = (0.3, 1.5)


def cluster_sem(vals, pid, mask):
    """SEM of the mean of `vals[mask]`, clustered on `pid` (one cluster per primary)."""
    v = np.asarray(vals)[mask]
    n = v.size
    if n < 2:
        return np.nan
    resid = v - v.mean()
    _, inv = np.unique(np.asarray(pid)[mask], return_inverse=True)
    per_primary = np.bincount(inv, weights=resid)
    return float(np.sqrt((per_primary ** 2).sum()) / n)


def line(name, truth, pred, pid, mask, n_prim_all):
    n = int(mask.sum())
    if n < 100:
        print(f"  {name:<34}{n:>12,}   (too few pairs to report)")
        return
    mt, st = float(truth[mask].mean()), cluster_sem(truth, pid, mask)
    summed = truth[mask].sum() / n_prim_all
    txt = (f"  {name:<34}{n:>12,}{mt:>11.4f}{st:>9.4f}{mt/max(st,1e-12):>8.1f}"
           f"{summed:>12.4f}")
    if pred is not None:
        mp = float(pred[mask].mean())
        txt += f"{mp:>11.4f}{pred[mask].sum()/n_prim_all:>12.4f}"
    print(txt)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairset", required=True)
    ap.add_argument("--checkpoint", default=None,
                    help="optional flow #2 checkpoint; if given, its prediction is scored too")
    ap.add_argument("--max-rows", type=int, default=0, help="0 = all")
    ap.add_argument("--batch-size", type=int, default=500000)
    args = ap.parse_args()

    cols = list(dict.fromkeys(RAW + ["blend_truth", "pid"]))
    d = pd.read_feather(args.pairset, columns=cols)
    if args.max_rows and len(d) > args.max_rows:
        # A subsample is fine HERE (unlike an absolute constgold m): every quantity below is a mean
        # or a per-primary sum whose error is reported, so subsampling widens the error honestly
        # instead of shifting the centre. Still off by default.
        d = d.iloc[: args.max_rows]
        print(f"NOTE: subsampled to the first {args.max_rows:,} rows")

    truth = d["blend_truth"].to_numpy(float)
    pid = d["pid"].to_numpy()
    mag_s = d["r_input_s"].to_numpy(float)
    re_s = d["Re_input_s"].to_numpy(float)
    n_prim = pd.unique(pid).size
    print(f"pair set : {args.pairset}")
    print(f"  {len(d):,} pairs over {n_prim:,} primaries, <k> = {len(d)/n_prim:.3f}")
    print(f"  per-pair label scatter std = {truth.std():.3f} (single pairs are uninformative)")

    inside = ((mag_s >= EMU_MAG[0]) & (mag_s < EMU_MAG[1])
              & (re_s >= EMU_RE[0]) & (re_s <= EMU_RE[1]))
    print(f"\n  emulator KEEPS {100*inside.mean():.2f}% of these pairs "
          f"(mag_s {EMU_MAG[0]}-{EMU_MAG[1]}, Re_s {EMU_RE[0]}-{EMU_RE[1]}); "
          f"it assigns the other {100*(~inside).mean():.2f}% a response of exactly zero")

    pred = None
    if args.checkpoint:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, std, yscale, meta = load_blend_flow(args.checkpoint, device=dev)
        delta = float(meta["delta"])
        derived = bool(meta.get("derived", True))
        i0, i1 = meta["shape_indices"]
        X = std.transform(build_features(d, derived=derived))
        sh = shifted_shape_columns(d, delta)
        m0, s0 = std.mean[i0], std.scale[i0]
        m1, s1 = std.mean[i1], std.scale[i1]
        pred = np.empty(len(d), dtype=np.float64)
        with torch.no_grad():
            for s in range(0, len(d), args.batch_size):
                e = min(s + args.batch_size, len(d))
                c0 = torch.as_tensor(X[s:e]).to(dev)
                ctxs = {}
                for k in ("e1+", "e1-", "e2+", "e2-"):
                    cc = c0.clone()
                    cc[:, i0] = torch.as_tensor(((sh[k][0][s:e] - m0) / s0).astype(np.float32)).to(dev)
                    cc[:, i1] = torch.as_tensor(((sh[k][1][s:e] - m1) / s1).astype(np.float32)).to(dev)
                    ctxs[k] = cc
                pred[s:e] = response_from_contexts(
                    model, c0, ctxs, (yscale[0], yscale[1]), delta).cpu().numpy()
        print(f"  flow #2 : {os.path.basename(args.checkpoint)} scored on all pairs")

    hdr = (f"\n{'group':<36}{'pairs':>12}{'<truth>':>11}{'sem':>9}{'sig':>8}"
           f"{'sum/prim':>12}")
    if pred is not None:
        hdr += f"{'<flow>':>11}{'flowsum':>12}"
    print("=" * (len(hdr) + 6))
    print("THE EMULATOR'S NEIGHBOUR SELECTION" + hdr)
    print("=" * (len(hdr) + 6))
    line("KEPT by the emulator", truth, pred, pid, inside, n_prim)
    line("DROPPED by the emulator", truth, pred, pid, ~inside, n_prim)
    line("all pairs", truth, pred, pid, np.ones(len(d), bool), n_prim)

    print("\nWHY EACH PAIR IS DROPPED (groups overlap)")
    line("  neighbour fainter than 26", truth, pred, pid, mag_s >= EMU_MAG[1], n_prim)
    line("  neighbour smaller than 0.3\"", truth, pred, pid, re_s < EMU_RE[0], n_prim)

    print("\nBY NEIGHBOUR MAGNITUDE (is there a threshold, or does it fade?)")
    edges = [18, 22, 24, 25, 26, 27, 28, 29, 99]
    for lo, hi in zip(edges[:-1], edges[1:]):
        line(f"  {lo} <= mag_s < {hi}", truth, pred, pid,
             (mag_s >= lo) & (mag_s < hi), n_prim)

    print("\nBY NEIGHBOUR SIZE")
    redges = [0.0, 0.1, 0.2, 0.3, 0.5, 1.0, 1.5, 99.0]
    for lo, hi in zip(redges[:-1], redges[1:]):
        line(f"  {lo} <= Re_s < {hi}", truth, pred, pid,
             (re_s >= lo) & (re_s < hi), n_prim)

    tot = truth.sum() / n_prim
    drop = truth[~inside].sum() / n_prim
    sem_drop = cluster_sem(truth, pid, ~inside) * int((~inside).sum()) / n_prim
    print("\n" + "=" * 100)
    print("VERDICT")
    print("=" * 100)
    print(f"  summed truth per primary, ALL neighbours          : {tot:.4f}")
    print(f"  summed truth per primary, emulator's neighbours   : {tot-drop:.4f}")
    print(f"  summed truth per primary, the DROPPED neighbours  : {drop:.4f} +- {sem_drop:.4f}")
    if abs(drop) < 2 * sem_drop:
        print("\n  The dropped neighbours carry no detectable response. The emulator's cut is")
        print("  harmless, and flow #2's unrestricted sum is adding unmeasured extrapolation to")
        print("  `R_blend`. Sum flow #2 over the emulator's selection for a like-for-like `m`.")
    else:
        print(f"\n  The dropped neighbours DO blend, at {abs(drop)/max(sem_drop,1e-12):.1f} sigma.")
        print("  The emulator's summed `R_blend` is therefore biased LOW by roughly the amount")
        print("  above, and flow #2's wider sum is capturing real signal -- but check the `<flow>`")
        print("  column against `<truth>` on the DROPPED row before trusting its size.")
    print("\n  Either way this is decided on the RULER, where the labels live. Do not settle it on")
    print("  constgold `m` -- that is the firewall.")
    print("\nFAINT_NEIGHBOUR_CONTRIBUTION_DONE", flush=True)


if __name__ == "__main__":
    main()
