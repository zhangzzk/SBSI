#!/usr/bin/env python -B
"""DECISION GATE (read-only, tunes nothing): is the small-size R_flow failure a forward-difference
truncation bias that Richardson extrapolation can cancel WITHOUT reading constgold?

R_flow's response target is a one-sided finite difference at g=0.05; the acceptance truth (constgold)
is a central difference. Forward-diff bias is O(g)*R'' -> a ~constant ABSOLUTE offset that becomes a
huge MULTIPLICATIVE bias where the response R->0 (small, near-PSF galaxies) -> the -63% size0.2-0.3 cliff.

Richardson from two legitimate forward legs (g=0.02, g=0.05) cancels the linear term:
    R_true ~= (5/3) R(g=0.02) - (2/3) R(g=0.05)
using ONLY sheared sims (constgold is never used to build a target -> non-circular; avoids the
retracted sz*cg train-on-validation).

This compares, per SIZE bin (flux+blend marginalized), the g=0.05 target, the g=0.02 target, the
Richardson combo, and the constgold central-diff truth. GATE PASS if Richardson shrinks the
small-size offset (and its multiplicative proxy) substantially vs the g=0.05 target.
"""
import argparse
import numpy as np
import pandas as pd
import pyarrow as pa, pyarrow.ipc as ipc

CG = "/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/constant_response_catalogue_train.feather"


def marg(grid, iso_only):
    """Marginalize a (nf, ns, nblend) response grid over flux (+blend unless iso_only) -> R(size)."""
    R = grid["Rsim"].astype(float)
    C = grid["raw_counts"].astype(float) if "raw_counts" in grid.files else np.ones_like(R)
    if iso_only:
        R = R[:, :, :1]; C = C[:, :, :1]           # blend bin 0 = isolated
    valid = np.isfinite(R)
    Rw = np.where(valid, R, 0.0); Cw = np.where(valid, C, 0.0)
    num = (Rw * Cw).sum(axis=(0, 2))
    den = Cw.sum(axis=(0, 2))
    return np.where(den > 0, num / np.maximum(den, 1e-9), np.nan)


def constgold_R_by_size(es, max_case):
    NEED = ["measured_e1_plus", "measured_e2_plus", "measured_e1_minus", "measured_e2_minus",
            "applied_g1", "applied_g2", "neighbored", "case", "Re_input_p", "detected"]
    parts = []
    with ipc.open_file(CG) as r:
        avail = set(r.schema.names); cols = [c for c in NEED if c in avail]
        for bi in range(r.num_record_batches):
            b = pa.Table.from_batches([r.get_batch(bi)]).select(cols).to_pandas()
            if int(b["case"].min()) > max_case:
                break
            b = b[b["case"] <= max_case]
            if "detected" in b.columns:
                b = b[b["detected"].astype(bool)]
            if len(b):
                parts.append(b)
    df = pd.concat(parts, ignore_index=True)
    g1 = df.applied_g1.to_numpy(float); g2 = df.applied_g2.to_numpy(float)
    gmag = np.hypot(g1, g2); ok = gmag > 1e-6
    df = df[ok]; g1, g2, gmag = g1[ok], g2[ok], gmag[ok]
    gh1, gh2 = g1 / gmag, g2 / gmag
    de1 = df.measured_e1_plus.to_numpy(float) - df.measured_e1_minus.to_numpy(float)
    de2 = df.measured_e2_plus.to_numpy(float) - df.measured_e2_minus.to_numpy(float)
    R = (de1 * gh1 + de2 * gh2) / (2.0 * gmag)
    size = df.Re_input_p.to_numpy(float); ngh = df.neighbored.to_numpy().astype(int)
    Ria, Raa = [], []
    for si in range(len(es) - 1):
        lo, hi = es[si], es[si + 1]
        mi = (size >= lo) & (size < hi) & (ngh == 0) & np.isfinite(R)
        ma = (size >= lo) & (size < hi) & np.isfinite(R)
        Ria.append(R[mi].mean() if mi.sum() else np.nan)
        Raa.append(R[ma].mean() if ma.sum() else np.nan)
    print(f"  constgold |g|~{np.median(gmag):.4f}  rows(detected, c<= {max_case})={len(df):,}", flush=True)
    return np.array(Ria), np.array(Raa)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--g02", required=True)
    ap.add_argument("--g05", required=True)
    ap.add_argument("--max-case", type=int, default=19)
    args = ap.parse_args()
    g02 = np.load(args.g02); g05 = np.load(args.g05)
    es = g05["edges_size"].astype(float)
    assert np.allclose(es, g02["edges_size"].astype(float)), "size edges must match between grids"

    for iso, tag in [(True, "ISOLATED (blend 0)"), (False, "ALL (iso+blended)")]:
        R02 = marg(g02, iso); R05 = marg(g05, iso)
        RICH = (5.0 / 3.0) * R02 - (2.0 / 3.0) * R05
        Ri, Ra = constgold_R_by_size(es, args.max_case)
        Rtrue = Ri if iso else Ra
        print(f"\n================= {tag} =================")
        print(f"{'size':>16s} {'R05(tgt)':>9s} {'R02':>9s} {'Rich':>9s} {'Rtrue(cg)':>10s} "
              f"{'off05':>8s} {'offRich':>8s} {'m05%':>8s} {'mRich%':>8s}")
        for si in range(len(es) - 1):
            lo, hi = es[si], es[si + 1]
            o5 = R05[si] - Rtrue[si]; orc = RICH[si] - Rtrue[si]
            # multiplicative-bias proxy: m = Rtrue/Rmodel - 1  (what a size-cut sample would show)
            m5 = 100.0 * (Rtrue[si] / R05[si] - 1.0) if abs(R05[si]) > 1e-6 else np.nan
            mr = 100.0 * (Rtrue[si] / RICH[si] - 1.0) if abs(RICH[si]) > 1e-6 else np.nan
            print(f"  [{lo:.3f},{hi:.3f}) {R05[si]:>+9.4f} {R02[si]:>+9.4f} {RICH[si]:>+9.4f} "
                  f"{Rtrue[si]:>+10.4f} {o5:>+8.4f} {orc:>+8.4f} {m5:>+8.1f} {mr:>+8.1f}")
        # gate verdict on the small-size bins (where the -63% cliff lives)
        small = slice(0, min(2, len(es) - 1))
        a5 = np.nanmean(np.abs(R05[small] - Rtrue[small]))
        ar = np.nanmean(np.abs(RICH[small] - Rtrue[small]))
        print(f"  small-size mean |offset|: g05={a5:.4f}  Richardson={ar:.4f}  "
              f"-> {'PASS (Richardson shrinks offset %.0f%%)' % (100*(1-ar/max(a5,1e-9))) if ar < 0.6*a5 else 'FAIL (Richardson does not help)'}")
    print("\nGATE_DONE", flush=True)


if __name__ == "__main__":
    main()
