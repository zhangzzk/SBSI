"""Noise-pairing check: does the response render share the noise realization between the 0.0 and sheared
images (so delta_et measurement noise cancels)? This sets how many cases a g=0.02/0.05 retrain needs.

The primary's INTRINSIC shape cancels in delta_et = e(g)-e(0) regardless (same galaxy/seed). What remains
at LARGE separation (blend response ~0) is the differential MEASUREMENT noise. So std(delta_et) for far,
high-S/N pairs = the per-pair measurement noise floor. It is g-independent (it's on the shape change), so
response noise = std(delta_et)/g. Compare g=0.02 vs g=0.2 far-pair scatter:
  small (<<0.01) & equal at both g -> strongly noise-PAIRED -> delta_et clean -> g=0.02 retrain FEASIBLE.
  ~0.03-0.05                       -> UNPAIRED (~sqrt2 * sigma_meas) -> 0.02 hard (robust loss + many cases).
Then estimate cases needed for a target per-leaf precision.
"""
import numpy as np, pyarrow.feather as pf

R002 = "/home/z/Zekang.Zhang/SBSI/results/resp002_c40-59.feather"
R02 = "/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876/response_catalogue_train.feather"
CASES = list(range(40, 60))

def stats(path, gamma, from_train, label):
    cols = ["case", "r_input_p", "r_input_s", "distance", "delta_et1", "S/N_p"]
    t = pf.read_table(path, columns=cols).to_pandas()
    if from_train:
        t = t[t.case.isin(CASES)]
    far = (t.distance > 7) & (t.distance < 10) & np.isfinite(t.delta_et1)   # blend response ~0 here
    hi = far & (t["S/N_p"] > 20)                                             # high-S/N subset
    d_far = t.delta_et1[far].to_numpy(float); d_hi = t.delta_et1[hi].to_numpy(float)
    # robust scatter (MAD-based std, immune to a few outliers)
    def rstd(x):
        return 1.4826 * np.median(np.abs(x - np.median(x)))
    print(f"\n=== {label} (gamma={gamma}) ===")
    print(f"  far pairs (7-10\"): N={far.sum():,}   <delta_et>={np.mean(d_far):+.5f}")
    print(f"    std(delta_et)      = {np.std(d_far):.5f}   robustStd = {rstd(d_far):.5f}")
    print(f"    -> response noise/pair = robustStd/gamma = {rstd(d_far)/gamma:.4f}")
    print(f"  far & S/N>20: N={hi.sum():,}   std(delta_et)={np.std(d_hi):.5f}  robustStd={rstd(d_hi):.5f}")
    return rstd(d_far)

s002 = stats(R002, 0.02, False, "g=0.02")
s02 = stats(R02, 0.2, True, "g=0.20")
print(f"\n=== VERDICT ===")
print(f"  robustStd(delta_et) far:  g=0.02={s002:.5f}   g=0.20={s02:.5f}   ratio={s002/s02:.2f}")
print(f"  (g-independent => ~equal expected; magnitude sets feasibility)")
# cases estimate: g=0.2 model uses ~200 cases. For same per-leaf precision on the response,
# need N_pairs ~ (noise/signal)^2 * const. response signal ~0.03; noise/pair(0.02)=robustStd/0.02.
sig = 0.03
for g, s in [(0.02, s002), (0.05, s002 * 1.0)]:  # 0.05 noise ~ same delta_et scatter / 0.05
    npair_per_leaf = (s / g / sig) ** 2 * 100      # to reach ~sig/10 per leaf of ~100-signal-unit region
    print(f"  g={g}: response noise/pair={s/g:.3f} -> ~{(s/g)**2/(s02/0.2)**2:.0f}x the pairs of g=0.2 "
          f"for equal precision -> ~{200*(s/g)**2/(s02/0.2)**2:.0f} cases (L2). Robust loss/correction-only cuts this a lot.")
print("NOISE_PAIRING_DONE")
