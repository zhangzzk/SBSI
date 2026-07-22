#!/usr/bin/env python
"""DIAGNOSTIC (read-only): does the METRIC-consistent response target (constgold antithetic ±g
central diff, the SAME sample+scheme as the acceptance r_sim) differ BY SIZE from the flow's actual
training target (variable-shear g0.05 SNC forward-diff grid)?  If constgold large-size Rsim ~1.0
(matching r_sim) while the variable grid is ~0.916, the flow was calibrated to a size-inconsistent
target -> rebuild the target from constgold + retrain.  Bins on the SAME flux/size quantile edges as
the variable grid.  Tunes NOTHING.
"""
import numpy as np, pyarrow as pa, pyarrow.ipc as ipc

CAT = "/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/constant_response_catalogue_train.feather"
GRID = "/home/z/Zekang.Zhang/SBSI/results/response_target_isoblend_snc_c0-99_6x6x5.npz"
NEED = ["measured_e1_plus", "measured_e2_plus", "measured_e1_minus", "measured_e2_minus",
        "applied_g1", "applied_g2", "neighbored", "distance", "case", "input_index",
        "r_input_p", "Re_input_p", "detected"]

def main():
    g = np.load(GRID)
    ef, es = g["edges_flux"], g["edges_size"]
    Rg = g["Rsim"]; Cg = g["raw_counts"].astype(float)
    print("variable-shear grid (flow's TRAINING target) Rsim by size (iso blend0 & all-blend):")
    for si in range(len(es) - 1):
        iso = (Rg[:, si, 0] * Cg[:, si, 0]).sum() / max(Cg[:, si, 0].sum(), 1)
        allb = (Rg[:, si, :] * Cg[:, si, :]).sum() / max(Cg[:, si, :].sum(), 1)
        print(f"  size[{es[si]:.3f},{es[si+1]:.3f}): iso={iso:+.4f}  all={allb:+.4f}")

    print("\nloading constgold ±g catalogue (cases 0-99) ...", flush=True)
    parts = []
    with ipc.open_file(CAT) as r:
        avail = set(r.schema.names); cols = [c for c in NEED if c in avail]
        for bi in range(r.num_record_batches):
            b = pa.Table.from_batches([r.get_batch(bi)]).select(cols).to_pandas()
            b = b[b["case"] <= 99]
            if "detected" in b.columns:
                b = b[b["detected"].astype(bool)]
            if len(b):
                parts.append(b)
    import pandas as pd
    df = pd.concat(parts, ignore_index=True)
    print(f"  constgold rows (c0-99, detected)={len(df):,}  has_detected={'detected' in df.columns}", flush=True)

    g1 = df.applied_g1.to_numpy(float); g2 = df.applied_g2.to_numpy(float)
    gmag = np.hypot(g1, g2); ok = gmag > 1e-6
    df = df[ok]; g1, g2, gmag = g1[ok], g2[ok], gmag[ok]
    gh1, gh2 = g1 / gmag, g2 / gmag
    de1 = df.measured_e1_plus.to_numpy(float) - df.measured_e1_minus.to_numpy(float)
    de2 = df.measured_e2_plus.to_numpy(float) - df.measured_e2_minus.to_numpy(float)
    R = (de1 * gh1 + de2 * gh2) / (2.0 * gmag)             # antithetic central diff = metric r_sim scheme
    size = df.Re_input_p.to_numpy(float); mag = df.r_input_p.to_numpy(float)
    ngh = df.neighbored.to_numpy().astype(int)
    print(f"  |g|~{np.median(gmag):.4f}  GLOBAL constgold R={np.average(R):+.4f}  "
          f"(cf variable grid global_R={float(g['global_R']):+.4f})")

    print("\nconstgold (METRIC-consistent) response by size, ON THE GRID's size edges:")
    print(f"{'size':>16s} {'n_iso':>9s} {'R_iso':>8s} {'n_all':>10s} {'R_all':>8s} "
          f"{'grid_iso':>9s} {'iso-grid':>9s}")
    for si in range(len(es) - 1):
        lo, hi = es[si], es[si + 1]
        mi = (size >= lo) & (size < hi) & (ngh == 0) & np.isfinite(R)
        ma = (size >= lo) & (size < hi) & np.isfinite(R)
        ri = R[mi].mean() if mi.sum() else np.nan
        ra = R[ma].mean() if ma.sum() else np.nan
        gi = (Rg[:, si, 0] * Cg[:, si, 0]).sum() / max(Cg[:, si, 0].sum(), 1)
        print(f"  [{lo:.3f},{hi:.3f}) {int(mi.sum()):>9,d} {ri:>+8.4f} {int(ma.sum()):>10,d} "
              f"{ra:>+8.4f} {gi:>+9.4f} {ri-gi:>+9.4f}")

    print("\nfiner LARGE-size read (iso), constgold central-diff vs variable grid iso(size5)=%.4f:" %
          ((Rg[:, 5, 0] * Cg[:, 5, 0]).sum() / max(Cg[:, 5, 0].sum(), 1)))
    for lo, hi in [(0.5, 0.75), (0.75, 1.0), (1.0, 1.5), (0.5, 1.5)]:
        mi = (size >= lo) & (size < hi) & (ngh == 0) & np.isfinite(R)
        print(f"  iso [{lo:.2f},{hi:.2f}): n={int(mi.sum()):>8,d}  R_constgold={R[mi].mean():+.4f}")
    print("\nINTERPRETATION: if constgold R_iso at large size ~1.0 (>> grid iso ~0.916), the flow's "
          "variable-shear target is size-inconsistent with the metric -> rebuild target from constgold "
          "(central ±g) + retrain to remove the +6..8% large-size selection bias.")

if __name__ == "__main__":
    main()
