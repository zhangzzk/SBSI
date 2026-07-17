"""Offline analysis of the constgold per-object dump (results/constgold_perobj_raw.feather).
Decisive flow-vs-blend discriminator: bin BLEND-FREE (isolated, R_blend<eps) targets by TARGET
magnitude r_input_p and check m_bare = <r_sim>/<R_flow> - 1.  With no blend contamination this
isolates the FLOW self-response calibration vs magnitude.
  - faint bins m_bare > 0  => flow UNDER-predicts self-response for faint targets (OOD)  => flow is
    the culprit; the target-mag covariate shift (constgold 50% faint vs 33% train) then drives the
    global +1.7%.
  - all bins m_bare ~ 0    => flow is fine; the deficit is a genuine train-vs-constant blend-truth
    mismatch, and it lives in the (blended) R_blend term after all.
Also: full-sample m_blend and the ABSOLUTE deficit attribution by magnitude (who carries +1.7%)."""
import sys, numpy as np, pandas as pd, pyarrow.feather as pf

path = sys.argv[1] if len(sys.argv) > 1 else "/home/z/Zekang.Zhang/SBSI/results/constgold_perobj_raw.feather"
eps = 0.02
d = pf.read_table(path).to_pandas()
rp = d["r_input_p"].to_numpy(float)
rs = d["r_sim"].to_numpy(float); rf = d["R_flow"].to_numpy(float); rb = d["R_blend"].to_numpy(float)
iso = rb < eps
EDGES = np.array([18, 23, 24, 24.5, 25, 25.5, 26, 26.5, 27, 27.5, 28.1])
lab = [f"{EDGES[i]:.1f}-{EDGES[i+1]:.1f}" for i in range(len(EDGES) - 1)]

print(f"loaded {len(d):,} objects; isolated (R_blend<{eps}) frac={iso.mean():.3f}")
print(f"GLOBAL: <r_sim>={rs.mean():.4f} <R_flow>={rf.mean():.4f} <R_blend>={rb.mean():.4f} "
      f"R_total={rf.mean()+rb.mean():.4f}  m_blend={rs.mean()/(rf.mean()+rb.mean())-1:+.2%}")


def table(name, mask):
    print(f"\n### {name}  (N={mask.sum():,}) ###")
    print(f"  {'r_p bin':>11} {'N':>9} {'<r_sim>':>8} {'<R_flow>':>8} {'<R_bl>':>7} {'m_bare':>8} {'m_blend':>8}")
    for i in range(len(lab)):
        m = mask & (rp >= EDGES[i]) & (rp < EDGES[i + 1])
        if m.sum() < 2000:
            continue
        srs, srf, srb = rs[m].mean(), rf[m].mean(), rb[m].mean()
        mb = srs / srf - 1 if abs(srf) > 1e-9 else np.nan
        mbl = srs / (srf + srb) - 1 if abs(srf + srb) > 1e-9 else np.nan
        print(f"  {lab[i]:>11} {m.sum():>9,} {srs:>8.4f} {srf:>8.4f} {srb:>7.3f} {mb:>+8.1%} {mbl:>+8.1%}")


table("ISOLATED (blend-free) -> pure FLOW self-response vs target mag", iso)
table("ALL targets -> R_flow+R_blend vs target mag", np.ones(len(d), bool))

# absolute deficit attribution: who carries the global R_sim - R_total ?
print("\n### absolute deficit attribution by target mag (deficit = sum(r_sim - R_flow - R_blend)) ###")
defi = rs - rf - rb
tot = defi.sum()
print(f"  total deficit sum={tot:.1f}  (global m from totals={rs.sum()/(rf.sum()+rb.sum())-1:+.2%})")
print(f"  {'r_p bin':>11} {'N':>9} {'deficit_sum':>12} {'share':>7} {'iso_share':>9}")
for i in range(len(lab)):
    m = (rp >= EDGES[i]) & (rp < EDGES[i + 1])
    if m.sum() < 2000:
        continue
    ds = defi[m].sum(); isod = defi[m & iso].sum()
    print(f"  {lab[i]:>11} {m.sum():>9,} {ds:>12.1f} {ds/tot:>+7.0%} {isod/tot:>+9.0%}")
print("PEROBJ_ANALYSIS_DONE")
