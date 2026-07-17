"""Mirror of audit_blend_truth for the SELF-response: compare the model R_flow to the half-shear
self-response TRUTH on the gold cases 0-39, binned by the production R_blend quantile (the q3 regime).

self_response_catalogue_train_cases0_99.feather (gold 0-39 included, ho-heldout) holds the true
per-object self-response delta_et1_self measured when the galaxy is sheared by gamma_self with its
neighbour present. Truth R_self = delta_et1/gamma_self. Compare to the per-bin model R_flow measured by
cg_dump (q1=0.2761 q2=0.2281 q3=0.1754 q4=0.0651; ISO~0.474).
  q3 R_self_truth >> R_flow (~0.211 vs 0.175) => the crowd-flux flow OVER-SUPPRESSES self-response
     for moderate-blend objects (branch A).
  q3 R_self_truth ~ R_flow (~0.175)          => flow is fine; the q3 deficit is the emulator R_blend
     (branch B) or coherent anisotropy (branch C) -- see audit_blend_truth.
Sanity: ISO (R_blend<eps) R_self_truth ~ R_flow ~ R_sim (the isolated self-response is already validated).
"""
import os, sys, numpy as np, pandas as pd, pyarrow.feather as pf, yaml
CFG = "/home/z/Zekang.Zhang/blendemu/configs/fs2_lsst_r_extnbr_ho.yaml"
SELF = "/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876/self_response_catalogue_train_cases0_99.feather"
BLEND_LOOKUP = "/home/z/Zekang.Zhang/SBSI/results/blend_lookup_extnbrho_c0-39.feather"
RFLOW_BIN = {0: 0.4740, 1: 0.2761, 2: 0.2281, 3: 0.1754, 4: 0.0651}   # model R_flow per R_blend bin (cg_dump)

cfg = yaml.safe_load(open(CFG))
gamma = float(np.diff(cfg["simulation"]["self_response"]["shear_values"])[0])   # values[1]-values[0]=0.05
print(f"gamma(self_response) = {gamma}", flush=True)

cols = ["case", "input_index", "distance", "delta_et1", "r_input_p"]
sch = [f.name for f in pf.read_table(SELF, columns=["case"]).schema]  # cheap: confirm 'case' exists
print("reading gold self-response rows ...", flush=True)
t = pf.read_table(SELF, columns=cols).to_pandas()
t = t[(t["case"] < 40) & np.isfinite(t["delta_et1"])].reset_index(drop=True)
print(f"gold self rows: {len(t):,}", flush=True)
# one row per primary: the nearest-neighbour row (dominant contamination); delta_et1_self is the
# primary's own-shape response, ~constant across its neighbour rows.
t = t.sort_values(["case", "input_index", "distance"]).drop_duplicates(["case", "input_index"], keep="first")
t["R_self_truth"] = t["delta_et1"].to_numpy(float) / gamma
print(f"unique primaries: {len(t):,}", flush=True)

bl = pf.read_table(BLEND_LOOKUP).to_pandas()[["case", "input_index", "R_blend"]]
g = t.merge(bl, on=["case", "input_index"], how="left")
rb = g["R_blend"].fillna(0.0).to_numpy(float)
hi = rb >= 0.02
qe = np.quantile(rb[hi], np.linspace(0, 1, 5)); qe[0] -= 1e-9; qe[-1] += 1e-9
bi = np.where(~hi, 0, 1 + np.clip(np.digitize(rb, qe) - 1, 0, 3))

print(f"\n=== SELF-response TRUTH vs model R_flow, by R_blend quantile ===")
print(f"  {'bin':>6} {'<R_bl>':>8} {'R_flow(model)':>13} {'R_self_truth':>13} {'truth-model':>12} {'N':>9}")
for c in range(5):
    b = bi == c
    if b.sum() < 2000:
        continue
    rst = float(np.mean(g["R_self_truth"].to_numpy()[b]))
    rfm = RFLOW_BIN[c]
    tag = "ISO" if c == 0 else f"q{c}"
    print(f"  {tag:>6} {rb[b].mean():>8.4f} {rfm:>13.4f} {rst:>13.4f} {rst-rfm:>+12.4f} {int(b.sum()):>9,}")

print(f"\n=== by target mag r_p (all) ===")
rp = g["r_input_p"].to_numpy(float); mrst = g["R_self_truth"].to_numpy(float)
for lo, hi_ in [(18, 24), (24, 24.5), (24.5, 25), (25, 25.5), (25.5, 26), (26, 26.5), (26.5, 27), (27, 28.1)]:
    b = (rp >= lo) & (rp < hi_)
    if b.sum() < 2000:
        continue
    print(f"  r_p {lo}-{hi_}: <R_self_truth>={mrst[b].mean():+.4f}  N={int(b.sum()):,}")
print("AUDIT_SELF_DONE")
