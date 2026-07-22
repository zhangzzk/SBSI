#!/usr/bin/env python
"""
cont.107 CORRECTION: my cont.105 "isolated over-prediction is an R_blend problem" used the WRONG isolated
definition (~neighbored = the 3" geometric flag) instead of the pipeline's def (R_blend<eps = truly
blend-free, where R_sim should EQUAL R_flow). This recheck:
  1) cross-tabs neighbored vs R_blend<eps and prints <R_blend> for each set (is ~neighbored contaminated?),
  2) for TRULY-isolated (R_blend<eps): closure = r_sim/(R_flow+R_blend)-1 ~ r_sim/R_flow-1 = PURE FLOW test,
     per magnitude -> does the FLOW alone over/under-predict genuinely blend-free objects? (R_blend not used)
  3) same for the ~neighbored set, to show the contamination that produced my wrong claim.
Firewall: r_sim read for validation only. Certified dump s501.
"""
import numpy as np, pyarrow.feather as feather, os

DDIR = "/project/ls-gruen/users/zekang.zhang/sbsi_dumps"
CERT = f"{DDIR}/fig2_perobj_s501_fixresp.feather"
OUT = "/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/isolated_recheck.txt"
EPS = 0.02   # pipeline blend-eps: R_blend<EPS = truly isolated

def main():
    lines = []
    def emit(s): print(s, flush=True); lines.append(s)
    t = feather.read_table(CERT, columns=["r_input_p", "r_sim", "R_flow", "R_blend", "neighbored"])
    mag = np.asarray(t["r_input_p"]).astype("f8")
    rs = np.asarray(t["r_sim"]).astype("f8")
    rf = np.asarray(t["R_flow"]).astype("f8")
    rb = np.asarray(t["R_blend"]).astype("f8")
    nbr = np.asarray(t["neighbored"]).astype(bool)
    N = len(mag)
    good = np.isfinite(rs) & np.isfinite(rf) & np.isfinite(rb)
    iso_rb = rb < EPS            # pipeline "truly isolated"
    emit(f"N={N:,}  good={good.mean():.1%}  <R_blend>_all={rb[good].mean():.4f}")
    emit("")
    emit("=== (1) neighbored flag  vs  R_blend<eps (are they the same 'isolated'?) ===")
    for lab, m in [("~neighbored (my def)", ~nbr & good), ("neighbored", nbr & good),
                   ("R_blend<eps (pipeline)", iso_rb & good), ("R_blend>=eps", ~iso_rb & good)]:
        emit(f"  {lab:26s}: n={m.sum():>11,} ({m.mean():5.1%})  <R_blend>={rb[m].mean():.4f}  "
             f"frac(R_blend<eps)={np.mean(rb[m] < EPS):.3f}")
    emit("")
    emit("=== (2) PURE-FLOW test on TRULY-isolated (R_blend<eps): closure ~ r_sim/R_flow-1, R_blend≈0 ===")
    emit("   (any bias here is 100% R_FLOW; R_blend is not involved)")
    MAG = [(22, 24), (24, 25), (25, 26), (26, 27)]
    emit(f"   {'mag':10s}{'n':>12}{'<R_blend>':>11}{'<r_sim>':>10}{'<R_flow>':>10}{'m=rs/(rf+rb)-1':>16}{'rs/rf-1':>10}")
    for ma, mb in MAG:
        m = good & iso_rb & (mag >= ma) & (mag < mb)
        if m.sum() < 1000:
            emit(f"   {f'{ma}-{mb}':10s}  (n<1000)"); continue
        RS, RF, RB = rs[m].mean(), rf[m].mean(), rb[m].mean()
        clo = RS / (RF + RB) - 1
        pure = RS / RF - 1
        emit(f"   {f'{ma}-{mb}':10s}{m.sum():>12,}{RB:>11.4f}{RS:>10.4f}{RF:>10.4f}{clo:>+15.1%}{pure:>+9.1%}")
    emit("")
    emit("=== (3) SAME per mag on my OLD ~neighbored set (shows R_blend contamination) ===")
    emit(f"   {'mag':10s}{'n':>12}{'<R_blend>':>11}{'<r_sim>':>10}{'<R_flow>':>10}{'m=rs/(rf+rb)-1':>16}{'rs/rf-1':>10}")
    for ma, mb in MAG:
        m = good & ~nbr & (mag >= ma) & (mag < mb)
        if m.sum() < 1000:
            emit(f"   {f'{ma}-{mb}':10s}  (n<1000)"); continue
        RS, RF, RB = rs[m].mean(), rf[m].mean(), rb[m].mean()
        clo = RS / (RF + RB) - 1
        pure = RS / RF - 1
        emit(f"   {f'{ma}-{mb}':10s}{m.sum():>12,}{RB:>11.4f}{RS:>10.4f}{RF:>10.4f}{clo:>+15.1%}{pure:>+9.1%}")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(lines) + "\n")
    emit(f"\nwrote {OUT}\nISO_RECHECK_DONE")

if __name__ == "__main__":
    main()
