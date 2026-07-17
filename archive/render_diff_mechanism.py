"""Why does 40-79 have ~3% lower R_sim than 0-39? SELECTION (different magnitude mix from a detection
change) or RENDERING (response itself differs at fixed magnitude)?

Compare R_sim vs target magnitude AND the magnitude distribution between the two sets.
  same R_sim(mag), different mag distribution (40-79 fainter/more objects) -> SELECTION (detection changed).
  different R_sim(mag) at fixed mag                                         -> RENDERING/measurement change.
0-39: cg_dump (case,r_input_p,r_sim). 40-79: catalogue r_sim reconstructed + input gals_info for magnitude.
"""
import glob, numpy as np, pandas as pd, pyarrow.feather as pf

DUMP = "/home/z/Zekang.Zhang/SBSI/results/constgold_perobj_raw.feather"
CAT = "/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/constant_response_catalogue_train.feather"
CBASE = "/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant"
TILE = "tile180.0_-0.5"
EDGES = np.array([18, 23, 24, 24.5, 25, 25.5, 26, 26.5, 27, 27.5, 28.1])

# 0-39
d = pf.read_table(DUMP, columns=["case", "r_input_p", "r_sim"]).to_pandas()
d = d.rename(columns={"r_input_p": "r"})
print(f"0-39: {len(d):,} objects, {len(d)/d.case.nunique():,.0f}/case")

# 40-79: reconstruct r_sim + attach magnitude from per-case input gals_info
t = pf.read_table(CAT, columns=["case", "input_index", "applied_g1", "applied_g2",
                                 "measured_e1_plus", "measured_e2_plus", "measured_e1_minus", "measured_e2_minus"]).to_pandas()
gmag = np.hypot(t.applied_g1, t.applied_g2); g = float(np.median(gmag))
t["r_sim"] = ((t.measured_e1_plus - t.measured_e1_minus) * (t.applied_g1 / gmag)
              + (t.measured_e2_plus - t.measured_e2_minus) * (t.applied_g2 / gmag)) / (2 * g)
mags = []
for c in range(40, 80):
    fp = f"{CBASE}/case{c}_0.02/real0/catalogues/input/gals_info_{TILE}.feather"
    gi = pf.read_table(fp, columns=["index_input", "r_input"]).to_pandas()
    gi["case"] = c; mags.append(gi.rename(columns={"index_input": "input_index", "r_input": "r"}))
mags = pd.concat(mags, ignore_index=True)
t = t.merge(mags, on=["case", "input_index"], how="left")
print(f"40-79: {len(t):,} objects, {len(t)/t.case.nunique():,.0f}/case, mag-matched {t.r.notna().mean():.1%}")

print(f"\n{'mag bin':>11} {'--- 0-39 ---':>20} {'--- 40-79 ---':>20} {'R_sim diff':>10}")
print(f"{'':>11} {'<r_sim>':>9} {'frac':>10} {'<r_sim>':>9} {'frac':>10}")
for i in range(len(EDGES) - 1):
    lo, hi = EDGES[i], EDGES[i + 1]
    da = d[(d.r >= lo) & (d.r < hi)]; db = t[(t.r >= lo) & (t.r < hi)]
    if len(da) < 2000 or len(db) < 2000:
        continue
    fa, fb = len(da) / len(d), len(db) / len(t)
    print(f"  {lo:.1f}-{hi:.1f}".rjust(11) + f" {da.r_sim.mean():>9.4f} {fa:>9.1%} "
          f"{db.r_sim.mean():>9.4f} {fb:>9.1%} {da.r_sim.mean()-db.r_sim.mean():>+10.4f}")
print(f"\n  overall  <r_sim> 0-39={d.r_sim.mean():.4f}  40-79={t.r_sim.mean():.4f}")
print(f"  <mag> 0-39={d.r.mean():.3f}  40-79={t.r.mean():.3f}  (faint frac r>26: 0-39={np.mean(d.r>26):.1%} 40-79={np.mean(t.r>26):.1%})")
# reweight 40-79 R_sim to the 0-39 magnitude distribution -> isolates SELECTION
db_by = t.groupby(np.clip(np.digitize(t.r, EDGES) - 1, 0, len(EDGES) - 2)).r_sim.mean()
da_frac = d.groupby(np.clip(np.digitize(d.r, EDGES) - 1, 0, len(EDGES) - 2)).size() / len(d)
common = sorted(set(db_by.index) & set(da_frac.index))
rw = sum(db_by[i] * da_frac[i] for i in common) / sum(da_frac[i] for i in common)
print(f"\n  40-79 R_sim REWEIGHTED to 0-39 mag distribution = {rw:.4f}  (vs 0-39 {d.r_sim.mean():.4f})")
print("  if reweighted ~ 0-39 -> pure SELECTION (mag mix). if still lower -> RENDERING at fixed mag.")
print("RENDER_MECH_DONE")
