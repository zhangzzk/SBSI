#!/usr/bin/env python
"""DIAGNOSTIC (read-only): is the acceptance metric's dump r_sim ITSELF biased for large galaxies?

The two-shear SNC measurement (diag_response_nonlinearity) says the true isolated-large response is
~0.91 (linear in g; R(0)extrap ~0.90-0.92), matching the grid target (0.916) AND the dump's own
R_flow (~0.889).  Only the dump r_sim (acceptance 'truth') sits at ~1.007.  Head-to-head on the SAME
(case,input_index) objects (overlap cases 40-99): compare dump r_sim vs a CLEAN SNC forward-diff
response.  If <r_sim> >> <R_snc> on identical objects, the ACCEPTANCE METRIC's truth is biased high
for large galaxies -> the 'size-cut bias' is partly a test artifact, not an estimator error.
Tunes NOTHING.
"""
import sys, numpy as np, pyarrow.feather as pf
sys.path.insert(0, "/home/z/Zekang.Zhang/SBSI/scripts")
from eval_selection_robustness import load, DUMP

CAT = "/project/ls-gruen/users/zekang.zhang/sbsi_catalogues/"
LOOK = "/home/z/Zekang.Zhang/SBSI/results/g0_lookup_c0-99.feather"
SH = ["measured_ngmix_g1", "measured_ngmix_g2"]
G0 = ["ngmix0_g1", "ngmix0_g2"]

def snc_response(path, look, cmin=40, cmax=99):
    cols = ["case", "input_index", SH[0], SH[1], "gamma1_input_p", "gamma2_input_p", "detected"]
    df = pf.read_table(path, columns=cols, memory_map=True).to_pandas()
    df = df[(df.case >= cmin) & (df.case <= cmax) & df.detected.astype(bool)]
    df = df.merge(look, on=["case", "input_index"], how="inner")
    g1 = df.gamma1_input_p.to_numpy(float); g2 = df.gamma2_input_p.to_numpy(float)
    gmag = np.hypot(g1, g2); ok = gmag > 1e-6
    df = df[ok]; g1, g2, gmag = g1[ok], g2[ok], gmag[ok]
    gh1, gh2 = g1 / gmag, g2 / gmag
    de1 = df[SH[0]].to_numpy(float) - df[G0[0]].to_numpy(float)
    de2 = df[SH[1]].to_numpy(float) - df[G0[1]].to_numpy(float)
    r = (de1 * gh1 + de2 * gh2) / gmag
    return df.assign(R_snc=r)[["case", "input_index", "R_snc"]]

def main():
    print("loading dump (c40-139) ...", flush=True)
    d = load(DUMP)
    d = d[(d.case >= 40) & (d.case <= 99)]                 # SNC-lookup overlap
    print(f"  dump c40-99 rows={len(d):,}", flush=True)
    look = pf.read_table(LOOK, memory_map=True).to_pandas()
    print("computing clean SNC forward-diff response (g0.05_val, c40-99) ...", flush=True)
    snc = snc_response(CAT + "det_meas_ngmix_g0.05_val.feather", look)
    print(f"  snc rows={len(snc):,}", flush=True)
    m = d.merge(snc, on=["case", "input_index"], how="inner")
    print(f"  matched rows={len(m):,}", flush=True)

    rsim = m.r_sim.to_numpy(float); rflow = m.R_flow.to_numpy(float)
    rsnc = m.R_snc.to_numpy(float); size = m.Re_input_p.to_numpy(float)
    ngh = m.neighbored.to_numpy().astype(int)
    fin = np.isfinite(rsim) & np.isfinite(rsnc) & np.isfinite(size)

    for tag, sel in [("ALL", fin), ("ISOLATED", fin & (ngh == 0)), ("BLENDED", fin & (ngh == 1))]:
        print(f"\n=== {tag}: dump r_sim vs clean SNC response vs dump R_flow (SAME objects, c40-99) ===")
        print(f"{'size':>14s} {'n':>10s} {'<r_sim>':>9s} {'<R_snc>':>9s} {'<R_flow>':>9s} "
              f"{'rsim-Rsnc':>10s} {'Rflow-Rsnc':>11s}")
        for lo, hi in [(0.0, 0.3), (0.3, 0.5), (0.5, 0.75), (0.75, 1.0), (1.0, 1.5),
                       (0.5, 1.5), (0.0, 5.0)]:
            mk = sel & (size >= lo) & (size < hi)
            n = int(mk.sum())
            if n < 2000:
                continue
            a = rsim[mk].mean(); b = rsnc[mk].mean(); c = rflow[mk].mean()
            print(f"  [{lo:.2f},{hi:.2f}) {n:>10,d} {a:>+9.4f} {b:>+9.4f} {c:>+9.4f} "
                  f"{a-b:>+10.4f} {c-b:>+11.4f}")
    print("\nINTERPRETATION: if <r_sim> is systematically ABOVE <R_snc> for large size (rsim-Rsnc grows "
          "with size) while <R_flow> ~ <R_snc>, the ACCEPTANCE METRIC's truth (dump r_sim) is biased "
          "high for large galaxies -> the certified R_flow is correct and the 'size-cut bias' is a test "
          "artifact. If <r_sim> ~ <R_snc> ~ <R_flow>, no metric bias (the earlier 1.007 was case-range).")

if __name__ == "__main__":
    main()
