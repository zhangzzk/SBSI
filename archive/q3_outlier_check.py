"""Is the q3 (moderate-R_blend) +9% residual driven by a few extreme r_sim values?

m_blend(bin) = <r_sim> / <R_flow+R_blend> - 1, all means. If a handful of pathological per-object
r_sim inflate <r_sim>, trimming/winsorizing them collapses the bias -> outlier artifact. If the
TRIMMED mean stays put, the +9% is a genuine population-level deficit.

Reads the existing dump (case,input_index,r_input_p,r_sim,R_flow[scalar,ignore],R_blend,neighbored,
distance). Reconstructs the SAME R_blend quantile bins as the validator (blend_eps=0.02, n_blend=4).
For q3: distribution of r_sim, sensitivity of <r_sim> to trimming, per-case contribution, and the
resulting m_blend using the run's proper binned R_flow (q3 R_flow=0.1754 from cg_dump log).
"""
import sys, numpy as np, pyarrow.feather as pf

path = "/home/z/Zekang.Zhang/SBSI/results/constgold_perobj_raw.feather"
eps, n_blend = 0.02, 4
RFLOW_Q = {1: 0.2761, 2: 0.2281, 3: 0.1754, 4: 0.0651}   # proper per-bin R_flow from cg_dump_14984541 log

d = pf.read_table(path, columns=["case", "r_sim", "R_blend"]).to_pandas()
rb = d["R_blend"].to_numpy(float); rs = d["r_sim"].to_numpy(float); case = d["case"].to_numpy(np.int64)
hi = rb >= eps
qe = np.quantile(rb[hi], np.linspace(0, 1, n_blend + 1)); qe[0] -= 1e-9; qe[-1] += 1e-9
bi = np.where(~hi, 0, 1 + np.clip(np.digitize(rb, qe) - 1, 0, n_blend - 1))


def trim_mean(x, frac):
    lo, hi_ = np.quantile(x, [frac, 1 - frac])
    return x[(x >= lo) & (x <= hi_)].mean()


for q in (1, 2, 3, 4):
    m = bi == q
    x = rs[m]; Rf = RFLOW_Q[q]; Rb = rb[m].mean(); denom = Rf + Rb
    mean = x.mean()
    print(f"\n===== R_blend q{q}  (N={m.sum():,})  R_flow={Rf:.4f} R_bl={Rb:.4f} =====")
    print(f"  <r_sim>          = {mean:.4f}   -> m_blend = {mean/denom-1:+.2%}")
    print(f"  median r_sim     = {np.median(x):.4f}")
    for f in (0.0001, 0.001, 0.01):
        tm = trim_mean(x, f)
        print(f"  trimmed {f:6.2%} mean = {tm:.4f}   -> m_blend = {tm/denom-1:+.2%}")
    p = np.percentile(x, [0.01, 0.1, 1, 50, 99, 99.9, 99.99])
    print(f"  r_sim pctiles [.01 .1 1 50 99 99.9 99.99] = "
          + " ".join(f"{v:.2f}" for v in p) + f"   min={x.min():.2f} max={x.max():.2f}")
    # concentration: how much of the (mean-denom) EXCESS sits in the top-k |deviation| objects?
    excess_total = (x - denom).sum()
    dev = np.abs(x - mean)
    order = np.argsort(dev)[::-1]
    for k_frac in (1e-5, 1e-4, 1e-3, 1e-2):
        k = max(1, int(len(x) * k_frac))
        share = (x[order[:k]] - denom).sum() / excess_total if excess_total != 0 else np.nan
        print(f"  top {k_frac:7.3%} most-extreme ({k:,} obj) carry {share:+6.1%} of the total excess sum")

# per-case m_blend in q3: is one case an outlier?
q = 3; m = bi == q; Rf = RFLOW_Q[q]
print(f"\n===== per-case m_blend in q3 (Rf={Rf}) =====")
cs = np.unique(case[m]); rows = []
for c in cs:
    mm = m & (case == c)
    if mm.sum() < 2000:
        continue
    x = rs[mm]; denom = Rf + rb[mm].mean()
    rows.append((c, mm.sum(), x.mean(), x.mean() / denom - 1))
rows.sort(key=lambda r: r[3])
mvals = np.array([r[3] for r in rows])
print(f"  {len(rows)} cases: m_blend min={mvals.min():+.1%} med={np.median(mvals):+.1%} "
      f"max={mvals.max():+.1%} std={mvals.std():.1%}")
print("  5 lowest:", " ".join(f"c{r[0]}:{r[3]:+.1%}" for r in rows[:5]))
print("  5 highest:", " ".join(f"c{r[0]}:{r[3]:+.1%}" for r in rows[-5:]))
print("Q3_OUTLIER_DONE")
