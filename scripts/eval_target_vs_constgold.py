"""Is the FLOW wrong, or is its TARGET wrong? The fork in the road for the in-domain m.

The per-cell excess that produces the in-domain m is

    dR_b = <R_flow>_b + <R_blend>_b - <r_sim>_b        (measured on constgold)

and it is LARGE and sign-alternating (+0.27, -0.12, ... in the small-size isolated cells) while its
population-weighted sum is a tiny -0.0044. So m is a small residue of a big cancellation. That is
worth fixing, but only after establishing WHICH of two independent defects produces it, because they
have opposite remedies:

    dR_b  =  [ <R_flow>_b - Rsim_tgt_b ]        (A) PIN RESIDUAL: the flow misses its target
           + [ Rsim_tgt_b + <R_blend>_b - <r_sim>_b ]   (B) TARGET DEFECT: the target is not the
                                                            quantity constgold needs

(A) is fixed by training: finer cells, more pin weight, better optimisation. (B) is NOT -- training
harder just chases a wrong number. (B) is nonzero if the half-shear target measures a different
response than constgold does in that cell, or if the emulator's R_blend does not complete the
identity Rsim_self + R_blend = r_sim_total there.

This decomposition has never been made. It is the cheapest possible measurement -- one pass over the
dumps plus the target npz -- and it decides where every remaining GPU hour should go.

FIREWALL: reads constgold per-object dumps for DIAGNOSIS. Trains nothing and selects nothing.
"""
from __future__ import annotations

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
import pyarrow.feather as pf

CONSTCAT = ("/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/"
            "constant_response_catalogue_train.feather")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-dir",
                    default="/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v2_domain_dumps")
    ap.add_argument("--tag", default="ablate_s2c_lt500_dom6x6")
    ap.add_argument("--target",
                    default="/home/z/Zekang.Zhang/SBSI/results/"
                            "response_target_crowd_rblend_snc_c0-99_6x6x5_dom.npz")
    ap.add_argument("--catalogue", default=CONSTCAT)
    ap.add_argument("--true-mag-max", type=float, default=26.0)
    ap.add_argument("--true-re-min", type=float, default=0.3)
    args = ap.parse_args()

    z = np.load(args.target, allow_pickle=True)
    ef, es, ec = z["edges_flux"], z["edges_size"], z["edges_crowd"]
    nf, ns, nb = len(ef) - 1, len(es) - 1, len(ec) - 1
    tgt = np.asarray(z["Rsim"], float).reshape(-1)
    cnt_tr = np.asarray(z["counts"], float).reshape(-1)
    print(f"target {os.path.basename(args.target)}  grid {nf}x{ns}x{nb}  crowd={z['crowd_col'].item()}")

    truth = pf.read_table(args.catalogue,
                          columns=["case", "input_index", "r_input_p", "Re_input_p"],
                          memory_map=True).to_pandas().drop_duplicates(["case", "input_index"])
    sel = ((truth["r_input_p"].to_numpy(float) < args.true_mag_max)
           & (truth["Re_input_p"].to_numpy(float) > args.true_re_min))
    keep = truth[sel][["case", "input_index", "Re_input_p"]]

    paths = sorted(glob.glob(os.path.join(args.dump_dir, f"{args.tag}_perobj_s*.feather")))
    if not paths:
        raise SystemExit(f"no dumps for {args.tag}")
    ncell = nf * ns * nb
    S = np.zeros((3, ncell))
    N = None
    for p in paths:
        d = pf.read_table(p, memory_map=True).to_pandas().merge(
            keep, on=["case", "input_index"], how="inner")
        fi = np.clip(np.digitize(d["r_input_p"].to_numpy(float), ef) - 1, 0, nf - 1)
        si = np.clip(np.digitize(d["Re_input_p"].to_numpy(float), es) - 1, 0, ns - 1)
        di = np.clip(np.digitize(d["R_blend"].to_numpy(float), ec) - 1, 0, nb - 1)
        cid = ((fi * ns + si) * nb + di).astype(np.int64)
        for k, col in enumerate(("r_sim", "R_flow", "R_blend")):
            S[k] += np.bincount(cid, weights=d[col].to_numpy(float), minlength=ncell)
        if N is None:
            N = np.bincount(cid, minlength=ncell).astype(float)
        del d, cid
    S /= len(paths)
    seeds = [int(re.search(r"_s(\d+)", os.path.basename(p)).group(1)) for p in paths]
    print(f"{len(paths)} dumps, seeds {seeds}")

    occ = N > 0
    rs = np.where(occ, S[0] / np.maximum(N, 1), np.nan)
    rf = np.where(occ, S[1] / np.maximum(N, 1), np.nan)
    rb = np.where(occ, S[2] / np.maximum(N, 1), np.nan)
    dR = rf + rb - rs                     # total excess (drives m)
    A = rf - tgt                          # pin residual
    B = tgt + rb - rs                     # target defect
    w = np.where(occ, N, 0.0) / N[occ].sum()
    denom = np.nansum(w * (rf + rb))

    def contrib(x):
        return -100.0 * np.nansum(w * x) / denom

    print("\n" + "=" * 78)
    print("EXACT SPLIT OF THE IN-DOMAIN m INTO ITS TWO INDEPENDENT CAUSES")
    print("=" * 78)
    print(f"  m from the TOTAL excess dR                = {contrib(dR):+.3f}%   <- the published m")
    print(f"    (A) m from the PIN RESIDUAL  R_flow - target      = {contrib(A):+.3f}%")
    print(f"    (B) m from the TARGET DEFECT target + R_blend - r_sim = {contrib(B):+.3f}%")
    print(f"  check: (A)+(B) = {contrib(A) + contrib(B):+.3f}%  vs total {contrib(dR):+.3f}%")
    print("=" * 78)
    a, b, tot = contrib(A), contrib(B), contrib(dR)
    # A "dominant cause" verdict is the wrong frame when both terms are individually much larger than
    # their sum: then m is a RESIDUE of a cancellation, and fixing either side alone makes m worse.
    if min(abs(a), abs(b)) > 2.0 * abs(tot):
        print(f"\nCANCELLATION, not a dominant cause: |A|={abs(a):.2f}% and |B|={abs(b):.2f}% are each "
              f"more than 2x the surviving |m|={abs(tot):.2f}%.")
        print("  => m is small only because the two errors have opposite signs. Fixing ONE alone moves\n"
              "     m to roughly the OTHER's value. Any single-knob change is expected to make m worse,\n"
              "     which is the signature every previous one-lever attempt produced. Both the pin\n"
              "     residual and the target/R_blend identity have to be fixed together.")
    elif abs(b) > abs(a):
        print("\nDOMINANT CAUSE: (B) the target is not what constgold needs")
        print("  => More/finer training against this target CANNOT fix it. Either the half-shear\n"
              "     target or the R_blend that completes the identity is wrong in these cells.")
    else:
        print("\nDOMINANT CAUSE: (A) the flow missing its target")
        print("  => The target is broadly right and the flow is not reaching it: pin weight, cell\n"
              "     resolution and optimisation are the levers.")

    # per-axis view, so we can see where each cause lives
    shp = (nf, ns, nb)
    for name, axis, edges in (("TRUE MAG", 0, ef), ("TRUE SIZE", 1, es), ("CROWD r_blend", 2, ec)):
        print(f"\n--- by {name} " + "-" * (60 - len(name)))
        print(f"{'bin':>20} {'w':>7} {'<r_sim>':>9} {'target':>8} {'<R_flow>':>9} {'<R_b>':>8} "
              f"{'(A) pin':>9} {'(B) tgt':>9} {'dm_A':>8} {'dm_B':>8}")
        oth = tuple(i for i in range(3) if i != axis)
        Wc = np.where(occ, N, 0.0).reshape(shp)
        for i in range(shp[axis]):
            sl = [slice(None)] * 3
            sl[axis] = i
            ww = Wc[tuple(sl)]
            tot = ww.sum()
            if tot <= 0:
                continue
            def av(x):
                return np.nansum(np.nan_to_num(x.reshape(shp)[tuple(sl)]) * ww) / tot
            wf = tot / Wc.sum()
            print(f"[{edges[i]:>8.4f},{edges[i+1]:>8.4f}) {wf:>7.3f} {av(rs):>9.4f} {av(tgt):>8.4f} "
                  f"{av(rf):>9.4f} {av(rb):>8.4f} {av(A):>+9.4f} {av(B):>+9.4f} "
                  f"{-100 * wf * av(A) / denom:>+7.3f}% {-100 * wf * av(B) / denom:>+7.3f}%")

    # the cells that matter most, with both causes side by side
    print("\n--- 12 cells with the largest |contribution to m| " + "-" * 24)
    print(f"{'(mag,size,crowd)':>18} {'w':>7} {'<r_sim>':>9} {'target':>8} {'<R_flow>':>9} "
          f"{'(A) pin':>9} {'(B) tgt':>9} {'dm tot':>8}")
    order = np.argsort(np.nan_to_num(np.abs(w * dR)))[::-1][:12]
    for b in order:
        fi, si, di = b // (ns * nb), (b // nb) % ns, b % nb
        print(f"{f'({fi},{si},{di})':>18} {w[b]:>7.4f} {rs[b]:>9.4f} {tgt[b]:>8.4f} {rf[b]:>9.4f} "
              f"{A[b]:>+9.4f} {B[b]:>+9.4f} {-100 * w[b] * dR[b] / denom:>+7.3f}%")
    print(f"\ncells occupied on constgold: {int(occ.sum())}/{ncell}; "
          f"training-empty but constgold-occupied: {int((occ & (cnt_tr <= 0)).sum())}")


if __name__ == "__main__":
    main()
