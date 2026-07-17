"""Preliminary (offline, no GPU) look at the multiplicity hypothesis for the q3 deficit.

Merges the just-built multiplicity lookup (case,input_index,n_pairs,rb_max,rb_top2) onto the per-object
dump (case,input_index,r_sim,R_blend). Two things:

(A) STRUCTURE: per R_blend quantile, what IS the bin? <n_pairs>, dominance = <|rb_max|/R_blend>.
    Multiplicity story predicts q3 = many-moderate (high n_pairs, low dominance); q4 = one-dominant
    (low n_pairs, high dominance).

(B) SIGNAL: within q3, split by n_pairs (low/high) and dominance. Report <r_sim> and <R_blend>.
    Caveat: proper per-cell R_flow needs the GPU job (cg_mult); here R_flow is held at the bin value
    (q3=0.1754) as a first look. If R_flow actually DROPS with n_pairs (crowd suppression), the true
    high-n deficit is even larger than shown, so a rising trend here is a conservative lower bound.
"""
import numpy as np, pyarrow.feather as pf, pandas as pd

DUMP = "/home/z/Zekang.Zhang/SBSI/results/constgold_perobj_raw.feather"
MULT = "/home/z/Zekang.Zhang/SBSI/results/blend_multiplicity_extnbrho_c0-39.feather"
eps, n_blend = 0.02, 4
RFLOW_Q = {1: 0.2761, 2: 0.2281, 3: 0.1754, 4: 0.0651}

d = pf.read_table(DUMP, columns=["case", "input_index", "r_sim", "R_blend"]).to_pandas()
ml = pf.read_table(MULT, columns=["case", "input_index", "n_pairs", "rb_max"]).to_pandas()
d = d.merge(ml, on=["case", "input_index"], how="left")
rb = d["R_blend"].to_numpy(float); rs = d["r_sim"].to_numpy(float)
npair = d["n_pairs"].fillna(0).to_numpy(float); rbmax = d["rb_max"].fillna(0.0).to_numpy(float)
dom = np.where(np.abs(rb) > 1e-9, np.abs(rbmax) / np.maximum(np.abs(rb), 1e-9), 0.0)
matched = d["n_pairs"].notna().to_numpy()

hi = rb >= eps
qe = np.quantile(rb[hi], np.linspace(0, 1, n_blend + 1)); qe[0] -= 1e-9; qe[-1] += 1e-9
bi = np.where(~hi, 0, 1 + np.clip(np.digitize(rb, qe) - 1, 0, n_blend - 1))

print(f"matched multiplicity for {matched.mean():.1%} of dump rows "
      f"(blended-row match {matched[hi].mean():.1%})\n")

print("(A) STRUCTURE per R_blend quantile:")
print(f"  {'bin':>5} {'N':>10} {'<R_bl>':>7} {'<n_pairs>':>9} {'median_np':>9} {'<dominance>':>11} {'frac 1-dom(>=.7)':>16}")
for q in range(1, n_blend + 1):
    m = (bi == q) & matched
    print(f"  {'q'+str(q):>5} {m.sum():>10,} {rb[m].mean():>7.3f} {npair[m].mean():>9.2f} "
          f"{np.median(npair[m]):>9.0f} {dom[m].mean():>11.2f} {np.mean(dom[m] >= 0.7):>16.1%}")

print("\n(B) SIGNAL within each blended quantile: split by n_pairs (<=med / >med) and dominance:")
print(f"  {'cell':>16} {'N':>10} {'<r_sim>':>8} {'<R_bl>':>7} {'<np>':>6} {'approx m*':>9}")
for q in range(1, n_blend + 1):
    base = (bi == q) & matched
    Rf = RFLOW_Q[q]; nmed = np.median(npair[base])
    for lab, m in [(f"q{q} all", base),
                   (f"q{q} n<={nmed:.0f}", base & (npair <= nmed)),
                   (f"q{q} n>{nmed:.0f}", base & (npair > nmed)),
                   (f"q{q} 1-dom>=.7", base & (dom >= 0.7)),
                   (f"q{q} many<.7", base & (dom < 0.7))]:
        if m.sum() < 3000:
            continue
        mr = rs[m].mean(); mb = rb[m].mean()
        print(f"  {lab:>16} {m.sum():>10,} {mr:>8.4f} {mb:>7.3f} {npair[m].mean():>6.1f} "
              f"{mr/(Rf+mb)-1:>+9.1%}")
    print()
print("* approx m uses the FIXED bin R_flow; real per-cell R_flow (cg_mult) will refine, "
      "and if R_flow drops with n_pairs the high-n deficit is UNDER-stated here.")
print("MULT_PRELIM_DONE")
