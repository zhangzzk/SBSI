"""Robust nonlinearity test: the plain mean of response=delta_et/g at g=0.02 is outlier-dominated
(heavy tails from /0.02). A TRIMMED mean discards the tails and tracks the bulk shift, which is where a
shear-magnitude nonlinearity lives. Compare trimmed-mean response at g=0.02 vs g=0.2 on the same fields.
  trimmed(0.02) > trimmed(0.2) => blend response is LARGER at weak shear => emulator (0.2) under-predicts
     constgold => nonlinearity IS the deficit driver (principled fix: retrain at 0.02).
  trimmed(0.02) <= trimmed(0.2) => wrong sign => deficit is the coherent cross-term, not shear-mag.
"""
import numpy as np, pandas as pd, pyarrow.feather as pf
from scipy import stats

R002 = "/home/z/Zekang.Zhang/SBSI/results/resp002_c40-59.feather"
R02 = "/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876/response_catalogue_train.feather"
CASES = list(range(40, 60)); TRIM = 0.05
REG = dict(rs=(13, 29), rp=(18, 28), Res=(0.0, 10.0), Rep=(0.1, 1.5), dist=(0, 10))

def load(path, gamma, from_train):
    cols = ["case", "r_input_p", "r_input_s", "Re_input_p", "Re_input_s", "distance", "delta_et1"]
    t = pf.read_table(path, columns=cols).to_pandas()
    if from_train:
        t = t[t.case.isin(CASES)]
    m = ((t.r_input_s > REG["rs"][0]) & (t.r_input_s < REG["rs"][1]) & (t.r_input_p > REG["rp"][0]) & (t.r_input_p < REG["rp"][1])
         & (t.Re_input_s > REG["Res"][0]) & (t.Re_input_s < REG["Res"][1]) & (t.Re_input_p > REG["Rep"][0]) & (t.Re_input_p < REG["Rep"][1])
         & (t.distance > REG["dist"][0]) & (t.distance < REG["dist"][1]) & np.isfinite(t.delta_et1))
    t = t[m].copy(); t["resp"] = t.delta_et1.to_numpy(float) / gamma
    return t

def tmean(x):
    return stats.trim_mean(x, TRIM)

def tmean_caseboot(t, seed=0):
    rng = np.random.default_rng(seed); cases = t.case.unique()
    per = {c: t.resp[t.case == c].to_numpy() for c in cases}
    vals = [tmean(np.concatenate([per[c] for c in rng.choice(cases, len(cases), replace=True)])) for _ in range(100)]
    return tmean(t.resp.to_numpy()), np.std(vals)

print("loading ...", flush=True)
a = load(R002, 0.02, False); b = load(R02, 0.2, True)
print(f"g=0.02: {len(a):,} pairs   g=0.20: {len(b):,} pairs   (trim={TRIM:.0%} each tail)")
ma, ea = tmean_caseboot(a); mb, eb = tmean_caseboot(b)
print(f"\n=== {TRIM:.0%}-TRIMMED mean response (robust to tails) ===")
print(f"  g=0.02 : {ma:.4f} +/- {ea:.4f}")
print(f"  g=0.20 : {mb:.4f} +/- {eb:.4f}")
print(f"  diff (0.02-0.20) = {ma-mb:+.4f} +/- {np.hypot(ea,eb):.4f}   ratio = {ma/mb:.3f}")
print(f"\n=== by separation ({TRIM:.0%}-trimmed) ===")
print(f"  {'sep':>10} {'g=0.02':>9} {'g=0.20':>9} {'ratio':>7}")
for lo, hi in [(0, 1), (1, 2), (2, 3), (3, 5), (5, 10)]:
    xa = a.resp[(a.distance >= lo) & (a.distance < hi)].to_numpy()
    xb = b.resp[(b.distance >= lo) & (b.distance < hi)].to_numpy()
    if len(xa) < 5000 or len(xb) < 5000:
        continue
    ta, tb = tmean(xa), tmean(xb)
    print(f"  {lo}-{hi}\"".rjust(10) + f" {ta:>9.4f} {tb:>9.4f} {ta/tb if abs(tb)>1e-6 else np.nan:>7.2f}")
print("ROBUST_NONLIN_DONE")
