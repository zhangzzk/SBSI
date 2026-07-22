#!/usr/bin/env python
"""
cont.107 sizing of lever (B): how much of the GLOBAL and blended-cut bias does R_blend miscalibration
account for, and how much could an HONEST (out-of-sample) fix recover?

(1) DIAGNOSTIC: bin by signed R_blend magnitude (the pipeline's clean axis, not the 3" neighbored flag).
    Per bin: closure m=<r_sim>/(<R_flow>+<R_blend>)-1 and additive deficit D=<r_sim>-<R_flow>-<R_blend>
    (D<0 = R_model OVER-predicts = R_blend over-compensates). GLOBAL m = sum(D)/sum(R_flow+R_blend).
(2) HONEST FIX (non-circular): fit deficit(R_blend-bin) on EVEN cases, APPLY R_blend+D_bin to ODD cases,
    remeasure GLOBAL / blended / ~blended m on ODD only. This is the out-of-sample test the retracted
    scene-R_blend skipped (fitting deficit on the SAME constgold = circular; see project_rblend_firewall).
Firewall: r_sim read for validation only; the fit is a deficit(R_blend) 1-D recalibration tested O-O-S.
Certified dump s501.
"""
import numpy as np, pyarrow.feather as feather, os

DDIR = "/project/ls-gruen/users/zekang.zhang/sbsi_dumps"
CERT = f"{DDIR}/fig2_perobj_s501_fixresp.feather"
OUT = "/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/rblend_bias_sizing.txt"
NB = 20      # R_blend quantile bins for the diagnostic + fit

def main():
    lines = []
    def emit(s): print(s, flush=True); lines.append(s)
    t = feather.read_table(CERT, columns=["case", "r_sim", "R_flow", "R_blend", "neighbored"])
    case = np.asarray(t["case"]).astype("i8")
    rs = np.asarray(t["r_sim"]).astype("f8")
    rf = np.asarray(t["R_flow"]).astype("f8")
    rb = np.asarray(t["R_blend"]).astype("f8")
    nbr = np.asarray(t["neighbored"]).astype(bool)
    g = np.isfinite(rs) & np.isfinite(rf) & np.isfinite(rb)
    case, rs, rf, rb, nbr = case[g], rs[g], rf[g], rb[g], nbr[g]
    N = len(rs)

    def m_of(mask):
        den = (rf[mask] + rb[mask]).sum()
        return (rs[mask].sum() / den - 1) * 100 if den != 0 else np.nan

    emit(f"N={N:,}   <R_blend>={rb.mean():.4f}")
    emit(f"UNCORRECTED closure:  GLOBAL {m_of(np.ones(N,bool)):+.2f}%   "
         f"blended(nbr) {m_of(nbr):+.2f}%   ~neighbored {m_of(~nbr):+.2f}%")
    emit("")

    # (1) diagnostic bins by signed R_blend
    edges = np.unique(np.quantile(rb, np.linspace(0, 1, NB + 1)))
    binid = np.clip(np.digitize(rb, edges[1:-1]), 0, len(edges) - 2)
    nbin = len(edges) - 1
    emit("=== (1) closure & additive deficit binned by R_blend magnitude ===")
    emit(f"   {'bin':>3}{'n':>11}{'<R_blend>':>11}{'<r_sim>':>9}{'<R_flow>':>9}{'m%':>8}"
         f"{'deficit D':>11}{'D*n/N (glob contrib)':>22}")
    Dsum = 0.0
    for b in range(nbin):
        mb = binid == b
        if mb.sum() == 0: continue
        RB, RS, RF = rb[mb].mean(), rs[mb].mean(), rf[mb].mean()
        D = RS - RF - RB
        contrib = D * mb.mean()
        Dsum += contrib
        emit(f"   {b:>3}{mb.sum():>11,}{RB:>+11.4f}{RS:>9.4f}{RF:>9.4f}{m_of(mb):>+7.1f}%"
             f"{D:>+11.4f}{contrib:>+22.5f}")
    emit(f"   sum(D*n/N)={Dsum:+.5f}  (=GLOBAL numerator; GLOBAL m = {Dsum/(rf+rb).mean()*100:+.2f}%)")
    emit("")

    # (2) honest out-of-sample deficit(R_blend) correction: fit on EVEN cases, apply to ODD
    even = (case % 2) == 0
    odd = ~even
    fe = np.quantile(rb[even], np.linspace(0, 1, NB + 1))
    fe = np.unique(fe)
    be_fit = np.clip(np.digitize(rb, fe[1:-1]), 0, len(fe) - 2)
    Dbin = np.zeros(len(fe) - 1)
    for b in range(len(fe) - 1):
        me = even & (be_fit == b)
        Dbin[b] = (rs[me] - rf[me] - rb[me]).mean() if me.sum() > 0 else 0.0
    rb_corr = rb + Dbin[be_fit]                      # corrected R_blend (adds fitted deficit)

    def m_corr(mask):
        den = (rf[mask] + rb_corr[mask]).sum()
        return (rs[mask].sum() / den - 1) * 100 if den != 0 else np.nan

    emit("=== (2) HONEST out-of-sample fix: deficit(R_blend) fit on EVEN cases, measured on ODD ===")
    emit(f"   ODD-set closure BEFORE:  GLOBAL {m_of(odd):+.2f}%   blended {m_of(odd&nbr):+.2f}%   "
         f"~neighbored {m_of(odd&~nbr):+.2f}%")
    emit(f"   ODD-set closure AFTER :  GLOBAL {m_corr(odd):+.2f}%   blended {m_corr(odd&nbr):+.2f}%   "
         f"~neighbored {m_corr(odd&~nbr):+.2f}%")
    emit(f"   (in-sample EVEN AFTER, circular ref: GLOBAL {m_corr(even):+.2f}%)")
    emit("")
    emit("m<0 = R_model over-predicts (R_blend over-compensates); deficit D<0 same. The O-O-S AFTER row is")
    emit("the honest recoverable gain from a deficit(R_blend) recalibration; in/out-of-sample gap = overfit.")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(lines) + "\n")
    emit(f"\nwrote {OUT}\nRBLEND_SIZING_DONE")

if __name__ == "__main__":
    main()
