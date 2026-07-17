"""Build a 'g=0.02 R_blend' lookup for cases 40-59 to directly test whether using the weak-shear blend
response reduces the constgold bias.

The half-shear 0.02 render only gives primary-secondary pairs (half pairing) and its per-object sum is
noisy. So instead of using the noisy 0.02 sum directly, we scale the emulator's CLEAN full-pairing R_blend
by the measured 0.02/0.2 ratio k, computed in bins of emulator R_blend (denoised in aggregate):
    R_blend_002[obj] = R_blend_emu[obj] * k(bin),   k(bin) = <sum delta_et/0.02>_bin / <sum delta_et/0.2>_bin
Both sums use the SAME half-shear pairs (reg cuts), so k is the pure shear-magnitude+selection ratio at
full-mean (matching how the emulator averages). Unmatched objects (isolated/r>26) keep k=1.
"""
import numpy as np, pandas as pd, pyarrow.feather as pf

R002 = "/home/z/Zekang.Zhang/SBSI/results/resp002_c40-59.feather"
R02 = "/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876/response_catalogue_train.feather"
EMU = "/home/z/Zekang.Zhang/SBSI/results/blend_lookup_extnbrho_c40-79.feather"
OUT = "/home/z/Zekang.Zhang/SBSI/results/blend_lookup_002_c40-59.feather"
CASES = list(range(40, 60))
REG = dict(rs=(13, 29), rp=(18, 28), Res=(0.0, 10.0), Rep=(0.1, 1.5), dist=(0, 10))

def halfsum(path, gamma, from_train, col):
    cols = ["case", "input_index", "r_input_p", "r_input_s", "Re_input_p", "Re_input_s", "distance", "delta_et1"]
    t = pf.read_table(path, columns=cols).to_pandas()
    if from_train:
        t = t[t.case.isin(CASES)]
    m = ((t.r_input_s > REG["rs"][0]) & (t.r_input_s < REG["rs"][1]) & (t.r_input_p > REG["rp"][0]) & (t.r_input_p < REG["rp"][1])
         & (t.Re_input_s > REG["Res"][0]) & (t.Re_input_s < REG["Res"][1]) & (t.Re_input_p > REG["Rep"][0]) & (t.Re_input_p < REG["Rep"][1])
         & (t.distance > REG["dist"][0]) & (t.distance < REG["dist"][1]) & np.isfinite(t.delta_et1))
    t = t[m].copy(); t[col] = t.delta_et1.to_numpy(float) / gamma
    return t.groupby(["case", "input_index"]).agg(**{col: (col, "sum")}).reset_index()

print("summing 0.02 and 0.2 half-shear ...", flush=True)
a = halfsum(R002, 0.02, False, "r002")
b = halfsum(R02, 0.2, True, "r02")
emu = pf.read_table(EMU).to_pandas()[["case", "input_index", "R_blend"]]
emu = emu[emu.case.isin(CASES)].rename(columns={"R_blend": "remu"})
g = emu.merge(a, on=["case", "input_index"], how="left").merge(b, on=["case", "input_index"], how="left")

# k(bin) from matched objects (have both r002 & r02), binned by emulator R_blend
ok = g.r002.notna() & g.r02.notna() & (g.r02.abs() > 1e-6)
gm = g[ok].copy()
qe = np.quantile(gm.remu, np.linspace(0, 1, 11)); qe[0] -= 1e-9; qe[-1] += 1e-9
gm["bin"] = np.clip(np.digitize(gm.remu, qe) - 1, 0, 9)
kbin = gm.groupby("bin").apply(lambda d: d.r002.mean() / d.r02.mean() if abs(d.r02.mean()) > 1e-9 else 1.0)
print("k by emulator-R_blend decile:", np.round(kbin.values, 3))
# assign k per object (matched -> its bin's k; unmatched -> 1)
g["bin"] = np.clip(np.digitize(g.remu, qe) - 1, 0, 9)
g["k"] = g["bin"].map(kbin).fillna(1.0)
g.loc[~ok & g.k.isna(), "k"] = 1.0
g["R_blend"] = g.remu * g.k
print(f"<R_blend> emulator={g.remu.mean():.4f} -> 0.02-scaled={g.R_blend.mean():.4f} "
      f"(ratio {g.R_blend.mean()/g.remu.mean():.3f})")
g[["case", "input_index", "R_blend"]].to_feather(OUT)
print(f"wrote {OUT}: {len(g):,} rows")
print("BUILD_RBLEND002_DONE")
