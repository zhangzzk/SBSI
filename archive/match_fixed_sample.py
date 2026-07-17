"""SELECTION confirmation: compare the blend response on a FIXED (shear-independent) sample.

Each catalogue contains only pairs DETECTED + cross-matched at that shear, so the g=0.02 and g=0.2
samples differ (shear-dependent detection = selection). Match the pairs present in BOTH (same case,
same primary input_index, same secondary input position) -> a fixed sample. Then:
  full sample:    <resp>(0.02) vs (0.2)  -> the ~11% g-dependence (selection + tiny nonlinearity)
  matched sample: <resp>(0.02) vs (0.2)  -> if it COLLAPSES to ~1, the g-dependence was SELECTION
                                             (the sample moved, not the per-pair response).
Robust 5%-trimmed means (0.02 tails). Also reports how many pairs drop in/out between shears.
"""
import numpy as np, pandas as pd, pyarrow.feather as pf
from scipy import stats

R002 = "/home/z/Zekang.Zhang/SBSI/results/resp002_c40-59.feather"
R02 = "/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876/response_catalogue_train.feather"
CASES = list(range(40, 60)); TRIM = 0.05
REG = dict(rs=(13, 29), rp=(18, 28), Res=(0.0, 10.0), Rep=(0.1, 1.5), dist=(0, 10))

def load(path, gamma, from_train):
    cols = ["case", "input_index", "RA_input_s", "DEC_input_s", "r_input_p", "r_input_s",
            "Re_input_p", "Re_input_s", "distance", "delta_et1"]
    t = pf.read_table(path, columns=cols).to_pandas()
    if from_train:
        t = t[t.case.isin(CASES)]
    m = ((t.r_input_s > REG["rs"][0]) & (t.r_input_s < REG["rs"][1]) & (t.r_input_p > REG["rp"][0]) & (t.r_input_p < REG["rp"][1])
         & (t.Re_input_s > REG["Res"][0]) & (t.Re_input_s < REG["Res"][1]) & (t.Re_input_p > REG["Rep"][0]) & (t.Re_input_p < REG["Rep"][1])
         & (t.distance > REG["dist"][0]) & (t.distance < REG["dist"][1]) & np.isfinite(t.delta_et1))
    t = t[m].copy()
    t["resp"] = t.delta_et1.to_numpy(float) / gamma
    t["key"] = (t.case.astype(np.int64).astype(str) + "_" + t.input_index.astype(np.int64).astype(str)
                + "_" + np.round(t.RA_input_s.to_numpy(), 6).astype(str)
                + "_" + np.round(t.DEC_input_s.to_numpy(), 6).astype(str))
    return t

tm = lambda x: stats.trim_mean(x, TRIM)
print("loading 0.02 ...", flush=True); a = load(R002, 0.02, False)
print("loading 0.20 ...", flush=True); b = load(R02, 0.2, True)
ka, kb = set(a.key), set(b.key)
inter = ka & kb
print(f"\n0.02 pairs={len(a):,}  0.20 pairs={len(b):,}  matched(in both)={len(inter):,}")
print(f"  only in 0.02: {len(ka-kb):,} ({len(ka-kb)/len(ka):.1%})   only in 0.20: {len(kb-ka):,} ({len(kb-ka)/len(kb):.1%})")

am = a[a.key.isin(inter)]; bm = b[b.key.isin(inter)]
print(f"\n=== {TRIM:.0%}-TRIMMED mean response ===")
print(f"  {'sample':>16} {'g=0.02':>9} {'g=0.20':>9} {'ratio':>7}")
print(f"  {'FULL':>16} {tm(a.resp):>9.4f} {tm(b.resp):>9.4f} {tm(a.resp)/tm(b.resp):>7.3f}")
print(f"  {'MATCHED (fixed)':>16} {tm(am.resp):>9.4f} {tm(bm.resp):>9.4f} {tm(am.resp)/tm(bm.resp):>7.3f}")
print("\n=== by separation (MATCHED fixed sample) ===")
print(f"  {'sep':>8} {'g=0.02':>9} {'g=0.20':>9} {'ratio':>7} {'N':>9}")
for lo, hi in [(0, 1), (1, 2), (2, 3), (3, 5), (5, 10)]:
    xa = am.resp[(am.distance >= lo) & (am.distance < hi)].to_numpy()
    xb = bm.resp[(bm.distance >= lo) & (bm.distance < hi)].to_numpy()
    if len(xa) < 3000 or len(xb) < 3000:
        continue
    print(f"  {lo}-{hi}\"".rjust(8) + f" {tm(xa):>9.4f} {tm(xb):>9.4f} {tm(xa)/tm(xb) if abs(tm(xb))>1e-6 else np.nan:>7.3f} {len(xa):>9,}")
print("\nMATCHED ratio ~1 => g-dependence was SELECTION. Still ~1.1 => not (pure) selection.")
print("MATCH_FIXED_DONE")
