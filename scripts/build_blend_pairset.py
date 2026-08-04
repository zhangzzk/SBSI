"""Build the training set for FLOW #2 -- the blend-response flow (`Gold-V3.md`).

WHAT THIS IS
------------
One row per (primary, annotated neighbour) PAIR, carrying everything flow #2 needs:

  * the primary's true properties INCLUDING its oriented intrinsic shape,
  * the neighbour's true properties INCLUDING its oriented intrinsic shape,
  * the scalar separation,
  * the primary's g=0 measured ngmix shape          -> the NLL target,
  * the per-pair blending response `blend_truth`    -> the BLEND response target,
  * the per-pair self response `self_truth`         -> the SELF response target (added 2026-08-02i),
  * the 45-degree rotated projections `*_null`      -> the validation channels,
  * `k`, the primary's annotated-neighbour count    -> the 1/k weight for the self channel.

`truth` is the same quantity `scripts/eval_rblend_gap.py` scores BlendEMU against, imported from
there rather than re-derived:

    truth = ((e1_both - e1_0) * ghat1_s + (e2_both - e2_0) * ghat2_s) / |g_s|

It is an UNBIASED per-pair estimator of that pair's blending response. The primary's own shear
direction is independent of the neighbour's (measured mean cos = -0.0000, sd = 0.7071), so the
self-response term averages to zero; the other neighbours enter only through cos 2(theta_i - theta_j),
which also has zero mean. Nothing is subtracted -- the contamination is variance, not bias.

THE SELF LABEL, AND WHY IT IS FREE (added 2026-08-02i, for the MERGED flow)
---------------------------------------------------------------------------
Merging flows #1 and #2 into one `p(ehat, thetahat | primary truth, neighbour truth)` needs BOTH
responses supervised in one model. The blend label above projects the measured-shape shift on the
NEIGHBOUR's shear direction. The SAME rows carry the PRIMARY's shear direction independently, so
projecting the IDENTICAL difference on that instead gives the self response:

    self_truth = ((e1_both - e1_0) * ghat1_p + (e2_both - e2_0) * ghat2_p) / |g_p|

No new leg, no new join, no new read -- one extra projection of a difference already computed. The
decorrelation argument is symmetric: it was invoked above to say the SELF term averages out of the
blend label, and the identical independence makes the BLEND term average out of the self label. Both
45-degree nulls are written and both are tested below; if either fails the projection argument has
broken and neither label is usable.

Two asymmetries that are NOT symmetric and must be handled downstream:

  1. SCALE. The self response is ~0.86 and the blend response ~0.14, a factor ~6. Per-pair label
     scatter is the same for both (it is the same `de`, just projected differently), so the self
     label has ~6x the signal-to-noise. A single response weight cannot serve both; the trainer must
     carry one weight per channel.
  2. THE SAMPLING UNIT. A primary with k annotated neighbours contributes k rows. For the BLEND label
     that is correct -- each row is a distinct pair and each label is unbiased for that pair. For the
     SELF label all k rows carry the SAME quantity (the primary's own response, which does not depend
     on which neighbour the row names), so an unweighted average over rows over-weights crowded
     primaries. Since crowding and self-response are correlated (blending suppresses the self
     response), that is a genuine bias, not just a variance inefficiency. `k` is written per row so
     the self channel can be weighted by 1/k; the blend channel must NOT be.

WHY THE SHAPES ARE `rot0` AND WHY gamma IS DROPPED
--------------------------------------------------
`e1/e2_input_rot0_{p,s}` are the INTRINSIC (pre-shear) shapes. On the g=0 leg the rendered shape IS
the intrinsic shape, which is the leg whose measured ngmix we use as the NLL target, so the
conditioning and the target refer to the same image. This is exactly flow #1's convention
(`build_shifted_context`: "g=0 rows: e1/e2_input_rot0_p IS the intrinsic shape; gamma=0"). The
`gamma*_input_*` columns are therefore NOT written -- the trainer sets them to zero anyway, and
carrying them invites double-application of the shear.

WHY ALL NEIGHBOURS, NOT JUST THE NEAREST
----------------------------------------
`m` uses `R_blend` SUMMED over a primary's neighbours (`build_blend_lookup.py`:
`groupby(...)['response'].sum()`), so the model has to be accurate per pair across the whole
aperture, not just for the nearest one. The ap7 legs annotate every neighbour inside 7", giving
~4.2 rows per primary. `pid` labels the primary so the sum can be reconstructed and so errors can be
formed over PRIMARIES (which handles the within-primary correlation) rather than over pairs.

KNOWN LIMITATION, RECORDED NOT HIDDEN
-------------------------------------
A primary with k annotated neighbours contributes k rows that all share the SAME g=0 measured shape,
while each row conditions on only ONE neighbour. That is harmless for the RESPONSE label (each
`truth_j` is unbiased for its own pair) but it does contaminate the DENSITY: the measured shape
carries blending from neighbours the row does not name. So the NLL term here is a representation
learner and a regulariser for the mean head, and this catalogue does NOT certify the density for the
`I = Var(s)` merge in `Gold-V3.md`. The response is protected because it is a DIFFERENCE of the mean
head under a shift of THIS neighbour's shape, and the unnamed neighbours' contribution does not
depend on that shape.

FIREWALL: half-shear legs only. constgold is never opened, so nothing built here can tune on the
acceptance set.
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

SBSI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
for _p in (SBSI_ROOT, SCRIPTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from eval_rblend_gap import CAT, GAMMA, NGMIX, blend_truth  # noqa: E402

# Everything the flow conditions on. Shapes are intrinsic (`rot0`); see the module docstring.
PRIMARY = ["r_input_p", "Re_input_p", "sersic_n_input_p", "e1_input_rot0_p", "e2_input_rot0_p"]
NEIGHBOUR = ["r_input_s", "Re_input_s", "sersic_n_input_s", "e1_input_rot0_s", "e2_input_rot0_s"]
GEOMETRY = ["distance"]
FEATURES = PRIMARY + NEIGHBOUR + GEOMETRY


def self_truth(base, rotate45=False):
    """Per-pair SELF response: the same measured-shape shift, projected on the PRIMARY's shear.

    Deliberately mirrors `blend_truth` line for line rather than sharing a helper, so the two labels
    are visibly the same construction differing only in which shear direction they project on.
    `rotate45=True` is the null: the spin-2 orthogonal direction, which carries no self signal.
    """
    g1 = base["gamma1_input_p"].to_numpy(float)
    g2 = base["gamma2_input_p"].to_numpy(float)
    gm = np.hypot(g1, g2)
    h1, h2 = g1 / gm, g2 / gm
    if rotate45:
        h1, h2 = -h2, h1
    de1 = base["measured_ngmix_g1_g"].to_numpy(float) - base["measured_ngmix_g1_0"].to_numpy(float)
    de2 = base["measured_ngmix_g2_g"].to_numpy(float) - base["measured_ngmix_g2_0"].to_numpy(float)
    return (de1 * h1 + de2 * h2) / gm


def stream(path, cols, max_case, keep_fn, label):
    """Read a leg batch by batch, filtering each batch before it is concatenated.

    The ap7 legs are 130 GB; materialising one and then cutting it needs memory we do not have and
    would spend most of it on rows the domain cut discards.
    """
    t0, parts, nseen = time.time(), [], 0
    with pa.memory_map(path, "rb") as src:
        rd = ipc.open_file(src)
        for i in range(rd.num_record_batches):
            b = pa.Table.from_batches([rd.get_batch(i)]).select(cols).to_pandas()
            nseen += len(b)
            if max_case is not None:
                b = b[b["case"] < max_case]
            if len(b):
                b = keep_fn(b)
                if len(b):
                    parts.append(b)
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cols)
    print(f"  {label}: {len(out):,} kept of {nseen:,} scanned  ({time.time()-t0:.0f}s)", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gs-leg", default=CAT + "det_meas_ngmix_ap7_g0.05_val.feather")
    ap.add_argument("--g0-leg", default=CAT + "det_meas_ngmix_ap7_g0.0_train.feather")
    ap.add_argument("--max-case", type=int, default=None)
    ap.add_argument("--true-re-min", type=float, default=0.3)
    ap.add_argument("--true-mag-max", type=float, default=26.0)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    t0 = time.time()
    gcols = list(dict.fromkeys(["case", "input_index", "detected"] + FEATURES + NGMIX + GAMMA))

    def keep_g(b):
        gp = np.hypot(b["gamma1_input_p"].to_numpy(float), b["gamma2_input_p"].to_numpy(float))
        gs = np.hypot(b["gamma1_input_s"].to_numpy(float), b["gamma2_input_s"].to_numpy(float))
        return b[(gp > 1e-6) & (gs > 1e-6)                       # the BOTH-sheared leg
                 & b["detected"].astype(bool).to_numpy()
                 & (b["Re_input_p"].to_numpy(float) > args.true_re_min)
                 & (b["r_input_p"].to_numpy(float) < args.true_mag_max)
                 # ngmix ran only where the primary is sheared; NaN rows carry no label
                 & np.isfinite(b["measured_ngmix_g1"].to_numpy(float))]

    zcols = ["case", "input_index", "detected", "Re_input_p", "r_input_p"] + NGMIX

    def keep_0(b):
        # true properties are identical per (case, input_index) across legs, so the same domain cut
        # is safe here and keeps the reference leg small
        return b[b["detected"].astype(bool).to_numpy()
                 & (b["Re_input_p"].to_numpy(float) > args.true_re_min)
                 & (b["r_input_p"].to_numpy(float) < args.true_mag_max)
                 & np.isfinite(b["measured_ngmix_g1"].to_numpy(float))]

    print(f"\nsheared leg : {args.gs_leg}\nreference   : {args.g0_leg}", flush=True)
    nb = stream(args.gs_leg, gcols, args.max_case, keep_g, "both-sheared, in-domain, detected")
    nprim = nb[["case", "input_index"]].drop_duplicates().shape[0]
    print(f"  -> {len(nb)/max(nprim,1):.3f} neighbour rows per primary", flush=True)

    ref = stream(args.g0_leg, zcols, args.max_case, keep_0, "g=0 reference, in-domain, detected")
    ref = ref.drop_duplicates(["case", "input_index"])[["case", "input_index"] + NGMIX]

    base = nb.merge(ref, on=["case", "input_index"], suffixes=("_g", "_0"))
    print(f"\nmatched both-detected pairs: {len(base):,}  cases={base['case'].nunique()}", flush=True)
    if not len(base):
        raise SystemExit("REFUSING: no matched pairs")

    truth = blend_truth(base)
    null = blend_truth(base, rotate45=True)
    struth = self_truth(base)
    snull = self_truth(base, rotate45=True)

    # NULL TESTS -- the same guard eval_rblend_gap applies, now on BOTH channels. If a 45-degree
    # projection is not consistent with zero, the OTHER galaxy's response is not averaging out of
    # that label and it is unusable. The two nulls are independent checks of the same assumption.
    for nm_arr, lbl in ((null, "blend"), (snull, "self ")):
        g = np.isfinite(nm_arr)
        nm, nsem = nm_arr[g].mean(), nm_arr[g].std(ddof=1) / np.sqrt(g.sum())
        print(f"\nNULL TEST {lbl} (45-deg rotated projection, must be ~0): {nm:+.5f} +- {nsem:.5f} "
              f"({abs(nm)/nsem:.1f} sigma)  N={g.sum():,}")
        if abs(nm) > 3 * nsem:
            print(f"  *** {lbl} NULL FAILS -- do NOT train on this set; "
                  "the projection argument has broken. ***")

    key = base["case"].to_numpy(np.int64) * 1_000_003 + base["input_index"].to_numpy(np.int64)
    _, pid = np.unique(key, return_inverse=True)

    out = pd.DataFrame({c: base[c].to_numpy(np.float32) for c in FEATURES})
    out["measured_ngmix_g1"] = base["measured_ngmix_g1_0"].to_numpy(np.float32)
    out["measured_ngmix_g2"] = base["measured_ngmix_g2_0"].to_numpy(np.float32)
    out["blend_truth"] = truth.astype(np.float32)
    out["blend_null"] = null.astype(np.float32)
    out["self_truth"] = struth.astype(np.float32)
    out["self_null"] = snull.astype(np.float32)
    out["case"] = base["case"].to_numpy(np.int32)
    out["input_index"] = base["input_index"].to_numpy(np.int32)
    out["pid"] = pid.astype(np.int64)

    fin = np.isfinite(out[FEATURES + ["blend_truth", "self_truth",
                                      "measured_ngmix_g1"]].to_numpy(np.float64)).all(1)
    print(f"rows with all features and both labels finite: {int(fin.sum()):,} of {len(out):,} "
          f"({100*fin.mean():.2f}%)")
    out = out[fin].reset_index(drop=True)

    # Rows this primary actually contributes, counted AFTER the finite filter. Written last on
    # purpose: the self label is a PER-PRIMARY quantity repeated across those rows, so it is averaged
    # with weight 1/k while the blend label (per-pair) is not, and that only equalises primaries if k
    # counts the rows that survive. Counting before the filter would leave primaries that lost rows
    # systematically under-weighted.
    kpid = out["pid"].to_numpy(np.int64)
    out["k"] = np.bincount(kpid)[kpid].astype(np.int32)

    print(f"\n{'column':>22}{'min':>12}{'median':>12}{'max':>12}")
    for c in FEATURES + ["blend_truth", "self_truth", "measured_ngmix_g1"]:
        v = out[c].to_numpy(float)
        print(f"{c:>22}{np.min(v):>12.4f}{np.median(v):>12.4f}{np.max(v):>12.4f}")

    kk = out["k"].to_numpy(float)
    print(f"\nneighbour count k: mean {kk.mean():.3f}  median {np.median(kk):.0f}  max {kk.max():.0f}")
    for name in ("blend_truth", "self_truth"):
        v = out[name].to_numpy(float)
        print(f"\n{name}: mean {v.mean():+.5f}  std {v.std():.4f}  "
              f"sem {v.std()/np.sqrt(len(v)):.6f} "
              f"({100*v.std()/np.sqrt(len(v))/abs(v.mean()):.2f}% of the mean)")
    # The self response is a PER-PRIMARY quantity; the row average over-weights crowded primaries and
    # crowding suppresses the self response, so the two differ by a real bias, not just by noise.
    w = 1.0 / kk
    s = out["self_truth"].to_numpy(float)
    print(f"self_truth, 1/k-weighted (the per-PRIMARY mean, which is the physical one): "
          f"{np.average(s, weights=w):+.5f}   unweighted-minus-weighted: "
          f"{s.mean() - np.average(s, weights=w):+.5f}")

    print("\n  -> both labels are dominated by measurement noise, not by signal. That is expected and")
    print("     unbiased; it sets how finely the response can be resolved, and it is the reason the")
    print("     response supervision is PER PAIR (a regression on noisy unbiased labels converges to")
    print("     the conditional mean) rather than a binned cell target.")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    out.to_feather(args.output)
    print(f"\nsaved {args.output}  ({len(out):,} rows, {os.path.getsize(args.output)/1e9:.2f} GB, "
          f"{time.time()-t0:.0f}s)")
    print("BLEND_PAIRSET_DONE", flush=True)


if __name__ == "__main__":
    main()
