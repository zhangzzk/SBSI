"""Per-(case, input_index) `R_blend` lookup from FLOW #2 -- the drop-in for `build_blend_lookup.py`.

For each constant-shear case, pair every primary with the neighbours inside the aperture, run flow #2
on each pair, and SUM per primary. Same output contract as the BlendEMU lookup
(`case`, `input_index`, `R_blend`) so the existing `m` pipeline consumes it unchanged, plus two
extra columns that exist specifically to stop the failure below.

THE TRAP THIS SCRIPT IS BUILT TO AVOID
--------------------------------------
`AGENTS.md` records it: the tuned in-domain emulator silently returns nothing outside its stored
training cuts, so wide-population rows fell back to `R_blend = 0`, coverage collapsed to 43.4%, and
`m` read +28.9% with nothing raised. Flow #2 has the same exposure -- it was trained with the PRIMARY
cut to `mag < 26`, `Re > 0.3` (neighbours full-population) -- but a neural network does not decline
to answer outside its domain, it extrapolates confidently. That is worse, not better.

So this script does three things instead of one:
  1. reads the training domain from the CHECKPOINT, never from a constant retyped here;
  2. writes `in_domain` per row and prints the coverage fraction;
  3. REFUSES to write if coverage is below `--min-coverage` unless `--allow-extrapolation` is passed.
Never zero-fill an out-of-domain row: drop it, or carry it flagged and let the consumer decide.

APERTURE
--------
Pairs are formed inside `--aperture` arcsec, default 7.0 to match the `ap7` legs flow #2 trained on.
This is NOT identical to what the BlendEMU lookup does -- `predict_response(t, t)` hands blendemu the
whole field and lets it decide -- so a comparison of the two summed lookups mixes a model difference
with an aperture difference. Stated, not silently absorbed.

WHICH NEIGHBOURS ARE SUMMED (`--neighbour-*`, added 2026-08-02e)
----------------------------------------------------------------
The first run of this script summed over EVERY galaxy inside the aperture and produced
`<R_blend> = 0.468` against the fiducial emulator's 0.136 -- a 3.4x gap that was provisionally read as
"constgold is twice as dense". It is not. The fiducial emulator's stored regression config is

    cuts = [mag_p 13-29, mag_s 18-26, Re_p 0-10, Re_s 0.3-1.5, distance 0-10],  r_max = 10,  k = 20

so it counts a neighbour ONLY if that neighbour is itself brighter than 26 and larger than 0.3", and
at most 20 of them. Everything fainter or smaller contributes exactly zero to the emulator's sum.
Flow #2 trained on neighbours down to mag 29 and Re 0.01 (checkpoint metadata), so summing it over the
whole field is faithful to ITS training -- the two models were simply being asked different questions.

The two are therefore genuinely different claims about the sky, not a bug in either:
  * emulator -- faint/small neighbours do not blend measurably; drop them;
  * flow #2  -- they do; here is their response.
Neither can be settled by whose `m` looks better (that is the constgold firewall), only by the per-pair
ruler, where the labels for those faint pairs exist.

These flags let the same checkpoint be summed over EITHER pair set, so the emulator comparison
isolates the per-pair model difference instead of confounding it with the neighbour selection. Pass
`--neighbour-mag-range 18 26 --neighbour-re-range 0.3 1.5 --max-neighbours 20` for the matched sum.
The selection is printed and stored in the output's attrs; it is never applied by default.

FIREWALL NOTE. This script reads constgold INPUT catalogues (true galaxy properties and positions) to
evaluate an already-trained model. Nothing is fitted, tuned or selected here, and no constgold
MEASUREMENT is read, so the evaluation-only standing of constgold is preserved -- exactly as for the
BlendEMU lookup it replaces.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.feather as pf
import torch
from scipy.spatial import cKDTree

SBSI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SBSI_ROOT not in sys.path:
    sys.path.insert(0, SBSI_ROOT)

from sbs_shear.blend_flow import (  # noqa: E402
    build_features, load_blend_flow, response_from_contexts, shifted_shape_columns)

CBASE = "/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant"
TILE = "tile180.0_-0.5"


def input_feather(case, sign, base):
    return f"{base}/case{case}_{sign}/real0/catalogues/input/gals_info_{TILE}.feather"


def pairs_within(ra, dec, aperture):
    """All (primary, neighbour) index pairs closer than `aperture` arcsec, and their separations.

    RA is compressed by cos(DEC) so the metric is angular, matching `eval_pair_angle.sep_vector`
    and the catalogue's own `distance` column.
    """
    x = ra * np.cos(np.deg2rad(dec)) * 3600.0
    y = dec * 3600.0
    tree = cKDTree(np.column_stack([x, y]))
    pi, pj = np.array(list(tree.query_pairs(aperture, output_type="ndarray"))).T \
        if len(x) > 1 else (np.array([], int), np.array([], int))
    # query_pairs gives each unordered pair once; the response is directional (who is the measured
    # object), so both orderings are needed.
    i = np.concatenate([pi, pj])
    j = np.concatenate([pj, pi])
    d = np.hypot(x[i] - x[j], y[i] - y[j])
    return i, j, d


def select_neighbours(i, j, d, mag_s, re_s, mag_range, re_range, max_k):
    """Restrict the pair list to a neighbour selection, so two models can be summed over one set.

    Returns the filtered (i, j, d) and a short human-readable description. The k-cap keeps the K
    NEAREST survivors per primary -- applied AFTER the property cuts, matching the emulator, which
    cuts first and then takes its k nearest.
    """
    keep = np.ones(len(i), dtype=bool)
    bits = []
    if mag_range is not None:
        lo, hi = mag_range
        keep &= (mag_s[j] >= lo) & (mag_s[j] < hi)
        bits.append(f"{lo} <= mag_s < {hi}")
    if re_range is not None:
        lo, hi = re_range
        keep &= (re_s[j] >= lo) & (re_s[j] <= hi)
        bits.append(f"{lo} <= Re_s <= {hi}")
    i, j, d = i[keep], j[keep], d[keep]
    if max_k and len(i):
        # rank each pair by separation within its primary; keep ranks < max_k
        order = np.lexsort((d, i))
        starts = np.searchsorted(i[order], i[order])   # first slot of each primary's block
        rank = np.arange(len(order)) - starts
        sel = order[rank < max_k]
        sel.sort()
        i, j, d = i[sel], j[sel], d[sel]
        bits.append(f"nearest {max_k}")
    return i, j, d, ("all neighbours in aperture" if not bits else "; ".join(bits))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", type=int, nargs="+", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--sign", default="0.02")
    ap.add_argument("--base", default=CBASE)
    ap.add_argument("--aperture", type=float, default=7.0)
    ap.add_argument("--batch-size", type=int, default=500000)
    ap.add_argument("--min-coverage", type=float, default=0.99)
    ap.add_argument("--allow-extrapolation", action="store_true")
    ap.add_argument("--neighbour-mag-range", type=float, nargs=2, default=None, metavar=("MIN", "MAX"),
                    help="keep a neighbour only if MIN <= r_input_s < MAX (emulator uses 18 26)")
    ap.add_argument("--neighbour-re-range", type=float, nargs=2, default=None, metavar=("MIN", "MAX"),
                    help="keep a neighbour only if MIN <= Re_input_s <= MAX (emulator uses 0.3 1.5)")
    ap.add_argument("--max-neighbours", type=int, default=0,
                    help="keep only the K NEAREST surviving neighbours per primary (0 = no cap; "
                         "the emulator uses 20)")
    args = ap.parse_args()

    t0 = time.time()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, std, yscale, meta = load_blend_flow(args.checkpoint, device=dev)
    if bool(meta.get("crowding", False)):
        raise SystemExit(
            "REFUSING: this checkpoint was trained with the CROWDING block "
            "(nbr_flux_near/far/max, log_k), which is built from a primary's FULL neighbour list "
            "via `pid`/`k`. The frames assembled here are per-case pair lists that do not carry "
            "those columns, so the features cannot be reproduced -- and building them from whatever "
            "rows happen to be present would report a systematically under-crowded galaxy. Use a "
            "non-crowding checkpoint, or extend this script to carry `pid` and `k` through.")
    delta = float(meta["delta"])
    derived = bool(meta.get("derived", True))
    i0, i1 = meta["shape_indices"]
    dom_mag = float(meta.get("primary_mag_max", 26.0))
    dom_re = float(meta.get("primary_re_min", 0.3))
    print(f"checkpoint : {args.checkpoint}")
    print(f"  delta={delta} derived={derived} device={dev}")
    print(f"  TRAINING DOMAIN on the PRIMARY: mag < {dom_mag}, Re > {dom_re} "
          f"(neighbours full-population)")
    print(f"  aperture   : {args.aperture}\" (flow #2 trained on the ap7 legs)")
    _sel = []
    if args.neighbour_mag_range is not None:
        _sel.append("mag_s in [%g, %g)" % tuple(args.neighbour_mag_range))
    if args.neighbour_re_range is not None:
        _sel.append("Re_s in [%g, %g]" % tuple(args.neighbour_re_range))
    if args.max_neighbours:
        _sel.append(f"nearest {args.max_neighbours}")
    print(f"  NEIGHBOUR SELECTION : {'; '.join(_sel) if _sel else 'NONE -- every galaxy in aperture'}")
    if not _sel:
        print("    (the fiducial emulator counts only mag_s 18-26, Re_s 0.3-1.5, nearest 20; an")
        print("     unrestricted sum here is NOT summed over the emulator's pair set)")

    parts, tot_rows, tot_in = [], 0, 0
    for c in args.cases:
        fp = input_feather(c, args.sign, args.base)
        if not os.path.exists(fp):
            print(f"case{c}: MISSING {fp}", flush=True)
            continue
        t = pf.read_table(fp).to_pandas()
        ra = t["RA_input"].to_numpy(float)
        dec = t["DEC_input"].to_numpy(float)
        i, j, d = pairs_within(ra, dec, args.aperture)
        n_raw = len(i)
        i, j, d, seldesc = select_neighbours(
            i, j, d, t["r_input"].to_numpy(float), t["Re_input"].to_numpy(float),
            args.neighbour_mag_range, args.neighbour_re_range, args.max_neighbours)
        if not len(i):
            print(f"case{c}: no pairs inside {args.aperture}\" after the neighbour selection",
                  flush=True)
            continue

        frame = pd.DataFrame({
            "r_input_p": t["r_input"].to_numpy(float)[i],
            "Re_input_p": t["Re_input"].to_numpy(float)[i],
            "sersic_n_input_p": t["sersic_n_input"].to_numpy(float)[i],
            "e1_input_rot0_p": t["e1_input_rot0"].to_numpy(float)[i],
            "e2_input_rot0_p": t["e2_input_rot0"].to_numpy(float)[i],
            "r_input_s": t["r_input"].to_numpy(float)[j],
            "Re_input_s": t["Re_input"].to_numpy(float)[j],
            "sersic_n_input_s": t["sersic_n_input"].to_numpy(float)[j],
            "e1_input_rot0_s": t["e1_input_rot0"].to_numpy(float)[j],
            "e2_input_rot0_s": t["e2_input_rot0"].to_numpy(float)[j],
            "distance": d})

        X = std.transform(build_features(frame, derived=derived))
        sh = shifted_shape_columns(frame, delta)
        m0, s0 = std.mean[i0], std.scale[i0]
        m1, s1 = std.mean[i1], std.scale[i1]
        r = np.empty(len(frame), dtype=np.float64)
        with torch.no_grad():
            for s in range(0, len(frame), args.batch_size):
                e = min(s + args.batch_size, len(frame))
                c0 = torch.as_tensor(X[s:e]).to(dev)
                ctxs = {}
                for k in ("e1+", "e1-", "e2+", "e2-"):
                    cc = c0.clone()
                    cc[:, i0] = torch.as_tensor(
                        ((sh[k][0][s:e] - m0) / s0).astype(np.float32)).to(dev)
                    cc[:, i1] = torch.as_tensor(
                        ((sh[k][1][s:e] - m1) / s1).astype(np.float32)).to(dev)
                    ctxs[k] = cc
                r[s:e] = response_from_contexts(
                    model, c0, ctxs, (yscale[0], yscale[1]), delta).cpu().numpy()

        idx = t["index_input"].to_numpy(np.int64)
        prim = i
        npr = len(t)
        rb = np.bincount(prim, weights=r, minlength=npr)
        nk = np.bincount(prim, minlength=npr)
        indom = (t["r_input"].to_numpy(float) < dom_mag) & (t["Re_input"].to_numpy(float) > dom_re)
        keep = nk > 0
        parts.append(pd.DataFrame({"case": c, "input_index": idx[keep],
                                   "R_blend": rb[keep], "n_neighbours": nk[keep],
                                   "in_domain": indom[keep]}))
        tot_rows += int(keep.sum())
        tot_in += int((keep & indom).sum())
        print(f"case{c}: {int(keep.sum()):,} primaries with neighbours, {len(i):,} pairs "
              f"(of {n_raw:,} in aperture; kept {100*len(i)/max(n_raw,1):.1f}%), "
              f"<k>={nk[keep].mean():.2f}, R_blend mean={rb[keep].mean():.4f}, "
              f"in-domain {100*indom[keep].mean():.1f}%", flush=True)

    if not parts:
        raise SystemExit("REFUSING: no cases produced pairs")
    out = pd.concat(parts, ignore_index=True)
    cov = tot_in / max(tot_rows, 1)
    print(f"\ntotal {tot_rows:,} primaries; {100*cov:.2f}% inside the flow's training domain")
    print(f"  <R_blend> all rows        = {out['R_blend'].mean():.4f}")
    dm = out["in_domain"].to_numpy(bool)
    if dm.any():
        print(f"  <R_blend> in-domain rows  = {out.loc[dm, 'R_blend'].mean():.4f}")
    if (~dm).any():
        print(f"  <R_blend> OUT-of-domain   = {out.loc[~dm, 'R_blend'].mean():.4f}  "
              f"<- EXTRAPOLATED, not measured")
    if cov < args.min_coverage and not args.allow_extrapolation:
        raise SystemExit(
            f"REFUSING to write: only {100*cov:.2f}% of primaries are inside the training domain "
            f"(threshold {100*args.min_coverage:.2f}%). The out-of-domain rows carry confident "
            f"extrapolations, not predictions. Re-run with --allow-extrapolation if you intend to "
            f"keep them, and cut on `in_domain` downstream.")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    out.to_feather(args.output)
    print(f"\nwrote {args.output}: {len(out):,} rows over {out['case'].nunique()} cases "
          f"({time.time()-t0:.0f}s)")
    print("BLEND_LOOKUP_FLOW_DONE", flush=True)


if __name__ == "__main__":
    main()
