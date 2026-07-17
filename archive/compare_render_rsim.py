"""Is the +1.6% (cases 0-39) vs +0.15% (40-79) a STATISTICAL fluctuation or a SYSTEMATIC render offset?

Compute per-case mean R_sim (the measured coherent response; the deficit is ~ R_sim/(R_flow+R_blend)-1,
denominator ~0.462 nearly constant, so per-case R_sim carries the bias). Compare the 40 case-means of each
set. If the distributions OVERLAP and the difference is within the per-case scatter/sqrt(N), it's
statistical (bootstrap under-estimates). If cleanly OFFSET, it's a real systematic in one render.
  0-39 : per-object r_sim from the cg_dump (constgold_perobj_raw.feather).
  40-79: reconstruct r_sim = ((e1p-e1m)*gh1+(e2p-e2m)*gh2)/(2g) from the constant catalogue.
"""
import numpy as np, pandas as pd, pyarrow.feather as pf
from scipy import stats

DUMP = "/home/z/Zekang.Zhang/SBSI/results/constgold_perobj_raw.feather"
CAT = "/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/constant_response_catalogue_train.feather"
DENOM = 0.462   # R_flow+R_blend (same for both sets) -> convert r_sim to m

# --- 0-39: per-case mean r_sim from dump ---
d = pf.read_table(DUMP, columns=["case", "r_sim"]).to_pandas()
a = d.groupby("case").r_sim.agg(["mean", "size"]).reset_index()
a.columns = ["case", "rsim", "n"]

# --- 40-79: reconstruct r_sim from the constant catalogue ---
t = pf.read_table(CAT, columns=["case", "applied_g1", "applied_g2", "measured_e1_plus", "measured_e2_plus",
                                 "measured_e1_minus", "measured_e2_minus"]).to_pandas()
gmag = np.hypot(t.applied_g1, t.applied_g2)
g = float(np.median(gmag)); gh1 = t.applied_g1 / gmag; gh2 = t.applied_g2 / gmag
t["r_sim"] = ((t.measured_e1_plus - t.measured_e1_minus) * gh1
              + (t.measured_e2_plus - t.measured_e2_minus) * gh2) / (2 * g)
b = t.groupby("case").r_sim.agg(["mean", "size"]).reset_index()
b.columns = ["case", "rsim", "n"]

def summ(x, lab):
    m, s = x.mean(), x.std(ddof=1); se = s / np.sqrt(len(x))
    print(f"  {lab:>8}: N={len(x)}  <r_sim>={m:.4f}  per-case std={s:.4f}  SE={se:.4f}  "
          f"m={(m/DENOM-1):+.2%} +/- {se/DENOM:.2%}")
    print(f"           per-case r_sim  min={x.min():.4f} q25={np.percentile(x,25):.4f} "
          f"med={np.median(x):.4f} q75={np.percentile(x,75):.4f} max={x.max():.4f}")
    return m, s, se

print("=== per-case R_sim distributions ===")
ma, sa, sea = summ(a.rsim, "0-39")
mb, sb, seb = summ(b.rsim, "40-79")
diff = ma - mb; diff_se = np.hypot(sea, seb)
tt = stats.ttest_ind(a.rsim, b.rsim, equal_var=False)
ks = stats.ks_2samp(a.rsim, b.rsim)
print(f"\n=== comparison ===")
print(f"  <r_sim> diff (0-39 minus 40-79) = {diff:+.4f} +/- {diff_se:.4f}  ({diff/diff_se:+.1f} sigma)")
print(f"  in bias units: delta_m = {diff/DENOM:+.2%}")
print(f"  Welch t-test: t={tt.statistic:+.2f} p={tt.pvalue:.2e}   KS: D={ks.statistic:.3f} p={ks.pvalue:.2e}")
print(f"  pooled per-case std = {np.hypot(sa,sb)/np.sqrt(2):.4f} -> a single 40-case set has SE~{np.hypot(sa,sb)/np.sqrt(2)/np.sqrt(40):.4f} "
      f"(m ~{np.hypot(sa,sb)/np.sqrt(2)/np.sqrt(40)/DENOM:.2%})")
print("\nVERDICT: p>0.05 / heavy overlap => STATISTICAL (bias not pinned at 0.3%, sample-limited).")
print("         p<<0.05 / clean offset  => SYSTEMATIC render difference (hunt the cause).")
# show the sorted per-case values so overlap is visible
print("\n0-39  case r_sim sorted:", np.round(np.sort(a.rsim.values), 3))
print("40-79 case r_sim sorted:", np.round(np.sort(b.rsim.values), 3))
print("COMPARE_RENDER_DONE")
