#!/usr/bin/env python
"""DIAGNOSTIC (read-only): is the shear response AMPLITUDE-DEPENDENT (nonlinear) for large galaxies?

The response TARGET grid is a FORWARD difference at g=0.05:  R_fwd(0.05) = <(e(0.05)-e(0))*ghat>/0.05
(compute_response_target_blend, --nominal-g 0.05, SNC baseline = g0_lookup).  The acceptance metric's
per-object r_sim and the flow's induced response are CENTRAL secants -> the true g->0 slope R(0).
If e(g) curves (ellipticity saturates, R''<0), R_fwd(0.05) UNDER-estimates R(0), most for high-response
LARGE galaxies -> the grid target is biased LOW there and the flow faithfully reproduces the bias
(diagnosed: grid iso size5 = 0.916 vs dump r_sim ~1.007).  Measure R_fwd at g=0.02 AND g=0.05 and
extrapolate to g=0:  R(0) = (5*R_fwd(0.02) - 2*R_fwd(0.05))/3  (linear-in-g forward-diff model).
Tunes NOTHING.
"""
import numpy as np, pyarrow.feather as pf

CAT = "/project/ls-gruen/users/zekang.zhang/sbsi_catalogues/"
LOOK = "/home/z/Zekang.Zhang/SBSI/results/g0_lookup_c0-99.feather"
SHCOLS = ["measured_ngmix_g1", "measured_ngmix_g2"]
G0COLS = ["ngmix0_g1", "ngmix0_g2"]

def load_R(path, look, max_case=99):
    cols = ["case", "input_index", SHCOLS[0], SHCOLS[1],
            "gamma1_input_p", "gamma2_input_p", "Re_input_p", "r_input_p", "neighbored", "detected"]
    df = pf.read_table(path, columns=cols, memory_map=True).to_pandas()
    df = df[(df.case <= max_case) & df.detected.astype(bool)]
    df = df.merge(look, on=["case", "input_index"], how="inner")
    g1 = df.gamma1_input_p.to_numpy(float); g2 = df.gamma2_input_p.to_numpy(float)
    gmag = np.hypot(g1, g2); ok = gmag > 1e-6
    df = df[ok]; g1, g2, gmag = g1[ok], g2[ok], gmag[ok]
    gh1, gh2 = g1 / gmag, g2 / gmag
    de1 = df[SHCOLS[0]].to_numpy(float) - df[G0COLS[0]].to_numpy(float)
    de2 = df[SHCOLS[1]].to_numpy(float) - df[G0COLS[1]].to_numpy(float)
    proj = de1 * gh1 + de2 * gh2
    R = proj / gmag                                    # per-object forward-diff response at this |g|
    return dict(R=R, size=df.Re_input_p.to_numpy(float), mag=df.r_input_p.to_numpy(float),
                ngh=df.neighbored.astype(int).to_numpy(), gmag=float(np.median(gmag)),
                case=df.case.to_numpy(), n=len(R))

def binmean(d, mask):
    m = mask & np.isfinite(d["R"]) & np.isfinite(d["size"])
    return int(m.sum()), (d["R"][m].mean() if m.sum() else np.nan)

def main():
    look = pf.read_table(LOOK, memory_map=True).to_pandas()
    print("loading g=0.02 (test) ...", flush=True)
    d2 = load_R(CAT + "det_meas_ngmix_g0.02_test.feather", look)
    print(f"  g0.02: n={d2['n']:,}  |g|~{d2['gmag']:.4f}  cases {d2['case'].min()}-{d2['case'].max()}", flush=True)
    print("loading g=0.05 (val) ...", flush=True)
    d5 = load_R(CAT + "det_meas_ngmix_g0.05_val.feather", look)
    print(f"  g0.05: n={d5['n']:,}  |g|~{d5['gmag']:.4f}  cases {d5['case'].min()}-{d5['case'].max()}", flush=True)

    SIZE_BINS = [(0.0, 0.3), (0.3, 0.412), (0.412, 0.592), (0.592, 0.75),
                 (0.75, 1.0), (1.0, 1.5), (0.5, 1.5), (0.592, 1.5)]
    for tag, sel2, sel5 in [("ALL", np.ones(d2["n"], bool), np.ones(d5["n"], bool)),
                            ("ISOLATED", d2["ngh"] == 0, d5["ngh"] == 0)]:
        print(f"\n=== {tag}: forward-diff response vs shear amplitude, by TRUE size ===")
        print(f"{'size':>14s} {'n02':>9s} {'R_fwd(.02)':>10s} {'n05':>9s} {'R_fwd(.05)':>10s} "
              f"{'R(0)extrap':>11s} {'curv%(.05)':>10s}")
        for lo, hi in SIZE_BINS:
            n2, r2 = binmean(d2, sel2 & (d2["size"] >= lo) & (d2["size"] < hi))
            n5, r5 = binmean(d5, sel5 & (d5["size"] >= lo) & (d5["size"] < hi))
            r0 = (5 * r2 - 2 * r5) / 3.0
            curv = 100 * (r0 - r5) / r0 if np.isfinite(r0) and r0 != 0 else np.nan
            print(f"  [{lo:.3f},{hi:.2f}) {n2:>9,d} {r2:>+10.4f} {n5:>9,d} {r5:>+10.4f} "
                  f"{r0:>+11.4f} {curv:>+9.2f}")
    print("\nINTERPRETATION: if R_fwd(.02) > R_fwd(.05) and R(0)extrap climbs toward ~1.0 for large size "
          "while R_fwd(.05) ~ 0.916 (the grid value), the FORWARD-diff-at-0.05 target is biased LOW by "
          "shear nonlinearity -> rebuild the grid with the g->0 slope (or central) and retrain. "
          "curv% is the fractional bias of R_fwd(.05) vs the extrapolated true slope.")

if __name__ == "__main__":
    main()
