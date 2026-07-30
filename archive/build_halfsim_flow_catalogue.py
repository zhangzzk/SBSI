#!/usr/bin/env python -B
"""Rebuild the joint-flow SHAPE (FLOW) training catalogue INCLUDING isolated objects.

WHY.  The unified forward model's shape-response stream was trained on
``self_response_catalogue_train_cases0_99`` which is 100% pairs / 0% isolated (it is a pair-
scene product -- every row has an explicit ``input_index_sec``).  The joint R_flow therefore
extrapolates on the empty-neighbour-set and inflates the isolated-cell |m| in eval B
(cont.57/58: eval B worst 19.3%, every mag_dN x ngh0 cell degraded).  This is a DATA-SOURCE
choice, not a model or sim limitation: the trainer + architecture already support the empty
set (neighbor_padded -> zeros+mask), and the two shear legs contain isolated objects.

WHAT.  ``self_response_catalogue`` == the ngmix half-sims filtered to pairs, with
    measured_e1_m == measured_ngmix_g1 @ g=0        (VERIFIED corr 1.0000, max|d|=0 on 223k)
    delta_et1     == measured_ngmix_g1@0.05 - @0
This rebuilds the SAME product from the two ngmix legs WITHOUT the pairs filter, so the FLOW
stream sees isolated (neighbored=False) objects.  Convention is identical to the certified
r_sim (ngmix), so the harvested joint R_flow stays directly comparable in
eval_selection_robustness.py.  ADDITIVE / experimental; NEVER wired into certified m.

  leg m (g=0)    : det_meas_ngmix_g0.0_train.feather   -> measured_e{1,2}_m, all base columns
  leg p (g=0.05) : det_meas_ngmix_g0.05_val.feather    -> measured_e{1,2}_p
  delta_et{1,2}  = e_p - e_m   (GAMMA_SELF = 0.05, matches train_forward_prototype.GAMMA_SELF)

Rows kept: matched on (case, input_index), DETECTED at both legs (finite ngmix at both) so the
shape target + response label are defined -- exactly the FLOW stream the trainer filters to,
now with isolated objects included.  Detection itself is the DET head's job (separate stream).
"""
import argparse
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.feather as pf

CAT = "/project/ls-gruen/users/zekang.zhang/sbsi_catalogues/"
LEG_M = CAT + "det_meas_ngmix_g0.0_train.feather"     # g=0     -> _m
LEG_P = CAT + "det_meas_ngmix_g0.05_val.feather"      # g=0.05  -> _p

# Base columns carried from the g=0 leg = the self_response schema minus the derived shape cols.
BASE = [
    "input_index", "input_index_sec", "neighbored",
    "RA_input_p", "RA_input_s", "DEC_input_p", "DEC_input_s",
    "redshift_input_p", "redshift_input_s", "Re_input_p", "Re_input_s",
    "axis_ratio_input_p", "axis_ratio_input_s",
    "position_angle_input_p", "position_angle_input_s",
    "sersic_n_input_p", "sersic_n_input_s", "r_input_p", "r_input_s",
    "polarization_angle", "distance", "shear_component_convention", "case",
    "e1_input_rot0_p", "e2_input_rot0_p", "e1_input_rot0_s", "e2_input_rot0_s",
]
SHP = ["measured_ngmix_g1", "measured_ngmix_g2"]

t0 = time.time()
def log(*a): print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)


def read_leg(path, cols, max_case, tag):
    """Stream a ngmix leg, filter case<max_case (files are case-ascending -> early stop)."""
    parts = []
    scanned = 0
    with ipc.open_file(path) as r:
        avail = set(r.schema.names)
        use = [c for c in cols if c in avail]
        miss = [c for c in cols if c not in avail]
        if miss:
            log(f"[{tag}] WARNING missing cols (skipped): {miss}")
        for bi in range(r.num_record_batches):
            b = pa.Table.from_batches([r.get_batch(bi)]).select(use).to_pandas()
            scanned += len(b)
            cvals = b["case"].astype(int)
            b = b[cvals < max_case]
            if len(b) == 0:
                if int(cvals.min()) >= max_case:   # case-ascending -> nothing more below max_case
                    break
                continue
            parts.append(b)
    df = pd.concat(parts, ignore_index=True)
    log(f"[{tag}] scanned={scanned:,} kept(case<{max_case})={len(df):,}")
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-case", type=int, default=100, help="keep cases [0, max_case)")
    ap.add_argument("--gamma-self", type=float, default=0.05)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    log("reading g=0 leg (m) base+shape ...")
    m = read_leg(LEG_M, BASE + SHP, args.max_case, "m")
    log("reading g=0.05 leg (p) shape + applied shear direction ...")
    # gamma{1,2}_input_p give the PER-OBJECT applied-shear direction at the g=0.05 leg (|gamma|=0.05
    # but the ANGLE varies per object).  self_response's delta_et is the two-leg measured-shape
    # difference PROJECTED onto that spin-2 direction (verified corr 1.0000, max|d|=0 vs ground truth);
    # a raw sky-basis difference would average the response to ~0.  So project, do not subtract raw.
    p = read_leg(LEG_P, ["case", "input_index", "gamma1_input_p", "gamma2_input_p"] + SHP, args.max_case, "p")

    m = m.rename(columns={"measured_ngmix_g1": "measured_e1_m", "measured_ngmix_g2": "measured_e2_m"})
    p = p.rename(columns={"measured_ngmix_g1": "measured_e1_p", "measured_ngmix_g2": "measured_e2_p"})

    log("merging legs on (case, input_index) ...")
    n_m = len(m)
    df = m.merge(p, on=["case", "input_index"], how="inner")
    log(f"matched {len(df):,}/{n_m:,} g=0 rows ({100*len(df)/max(n_m,1):.1f}%)")
    del m, p

    # spin-2 projection of the two-leg measured-shape difference onto the applied-shear direction:
    #   delta_et1 (parallel/tangential -> the response),  delta_et2 (cross -> ~0)
    de1 = (df["measured_e1_p"] - df["measured_e1_m"]).to_numpy(float)
    de2 = (df["measured_e2_p"] - df["measured_e2_m"]).to_numpy(float)
    g1 = df["gamma1_input_p"].to_numpy(float)
    g2 = df["gamma2_input_p"].to_numpy(float)
    gmag = np.hypot(g1, g2)
    with np.errstate(invalid="ignore", divide="ignore"):
        gh1 = np.where(gmag > 1e-8, g1 / gmag, np.nan)
        gh2 = np.where(gmag > 1e-8, g2 / gmag, np.nan)
    df["delta_et1"] = de1 * gh1 + de2 * gh2          # parallel (response)
    df["delta_et2"] = -de1 * gh2 + de2 * gh1         # cross (~0)
    df = df.drop(columns=["gamma1_input_p", "gamma2_input_p"])

    # Keep only rows with a defined shape target AND response label (= detected at both legs).
    fin = np.isfinite(df[["measured_e1_m", "measured_e2_m", "delta_et1", "delta_et2"]].to_numpy(float)).all(axis=1)
    log(f"finite(shape+response) at both legs: {int(fin.sum()):,}/{len(df):,} ({100*fin.mean():.1f}%)")
    df = df.loc[fin].reset_index(drop=True)

    nb = df["neighbored"].astype(bool)
    iso_frac = float((~nb).mean())
    log(f"FINAL rows={len(df):,}  neighbored={float(nb.mean()):.4f}  ISOLATED={iso_frac:.4f}  "
        f"(self_response_catalogue was 0.0000 -- this is the fix)")
    log(f"cases: [{int(df['case'].min())},{int(df['case'].max())}]  n_distinct={df['case'].nunique()}")
    log(f"delta_et1 mean={df['delta_et1'].mean():+.4f}  R_self=delta/g mean={df['delta_et1'].mean()/args.gamma_self:+.4f}")
    log(f"  iso   R_self mean={df.loc[~nb,'delta_et1'].mean()/args.gamma_self:+.4f}  "
        f"blend R_self mean={df.loc[nb,'delta_et1'].mean()/args.gamma_self:+.4f}")

    pf.write_feather(df, args.out)
    log(f"wrote {args.out}  ({len(df.columns)} cols)")
    print("BUILD_HALFSIM_FLOW_DONE", flush=True)


if __name__ == "__main__":
    main()
