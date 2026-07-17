"""Diagnose the 12% gap between the population forward model (ratio 0.88) and the truth
undetected census. xi(theta)~1 (Poisson field), so the gap is a DOMAIN mismatch. Read the saved
per-pair characterization and, over DETECTED primaries, show where R_undet_hard lives in
separation theta and secondary magnitude — in particular how much sits below the MC inner cutoff
THETA_MIN=0.30" and outside the neighbour magnitude range the MC samples.
"""
import numpy as np, pandas as pd, pyarrow.feather as pf

P = pf.read_table("/home/z/Zekang.Zhang/SBSI/results/probblend_char.feather",
                  columns=["dist", "rs", "resp", "det_s", "det_p"]).to_pandas()
dp = P[P.det_p.to_numpy()]
undet = dp[~dp.det_s.to_numpy()]
tot_undet = undet.resp.sum()
tot_full = dp.resp.sum()
print(f"detected-primary pairs: {len(dp):,}; undetected-secondary pairs: {len(undet):,}")
print(f"sum R_undet={tot_undet:.1f}  (={tot_undet/tot_full:.1%} of full)  min theta={undet.dist.min():.3f}\"")

print("\ncumulative R_undet fraction BELOW theta:")
for th in [0.10, 0.20, 0.30, 0.50, 1.0, 2.0, 3.0, 5.0, 10.0]:
    f = undet[undet.dist < th].resp.sum() / tot_undet
    print(f"  theta<{th:5.2f}\": {f:6.1%}")

print("\nR_undet by separation shell:")
edges = [0, 0.3, 0.5, 1, 1.5, 2, 3, 5, 10]
for lo, hi in zip(edges[:-1], edges[1:]):
    s = undet[(undet.dist >= lo) & (undet.dist < hi)]
    print(f"  {lo:4.1f}-{hi:4.1f}\": sum={s.resp.sum():8.1f} ({s.resp.sum()/tot_undet:+6.1%})  n={len(s):,}")

print("\nR_undet by secondary magnitude:")
medg = [18, 24, 25, 26, 26.5, 27, 27.5, 28, 28.5, 29.5]
for lo, hi in zip(medg[:-1], medg[1:]):
    s = undet[(undet.rs >= lo) & (undet.rs < hi)]
    if len(s) == 0:
        continue
    print(f"  r_s {lo:4.1f}-{hi:4.1f}: sum={s.resp.sum():8.1f} ({s.resp.sum()/tot_undet:+6.1%})  n={len(s):,}")
print(f"\n  frac of R_undet with r_s>28 (outside MC [18,28] draw): "
      f"{undet[undet.rs > 28].resp.sum()/tot_undet:.1%}")
print(f"  frac of R_undet with theta<0.30 (below MC THETA_MIN):   "
      f"{undet[undet.dist < 0.30].resp.sum()/tot_undet:.1%}")
print("GAP_DIAG_DONE")
