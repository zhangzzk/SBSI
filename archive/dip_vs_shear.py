"""PRINCIPLED test of the R_blend dip: is the intermediate-separation dip a LARGE-SHEAR (g=0.2) artifact?

The emulator was trained on the half-shear response at g=0.2; constgold is g=0.02. If the blend response
is nonlinear in shear, the g=0.2 response (and its dip) differs from the g=0.02 one the weak-lensing
constgold actually needs. We build the isotropic blend response at BOTH shears on the SAME half-base
fields and compare response-vs-separation:
  g=0.2  : from the existing half-base response_catalogue_train (delta_et1/0.2).
  g=0.02 : built here via response.retrieve_response(shear_cases=['0.0','0.02']) (delta_et1/0.02).
If the deep ~1" dip is present at g=0.2 but shallow/absent at g=0.02, the dip is a large-shear artifact
=> the emulator carries a spurious dip and under-predicts the true weak-shear response near it. Principled
fix = retrain the emulator at constgold's shear (g=0.02).  [No fitting to constgold.]
"""
import os, sys, numpy as np, pandas as pd, pyarrow.feather as pf
sys.path.insert(0, "/home/z/Zekang.Zhang/blendemu")
from blendemu import response

HALF = "/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876"
RESP02 = f"{HALF}/response_catalogue_train.feather"   # g=0.2 truth (cases 0-199)
CASES = list(range(40, 60))                            # 20 held-out-ish fields for stats
RP = (24.0, 26.0); RS = (18.0, 27.0)                   # q3-relevant slice (matches dip test)
DEDGES = np.arange(0.0, 6.01, 0.4)

def curve(df, gamma, label):
    d = df[(df.r_input_p >= RP[0]) & (df.r_input_p < RP[1]) & (df.r_input_s > RS[0]) & (df.r_input_s < RS[1])
           & (df.distance > 0) & (df.distance < 6) & np.isfinite(df.delta_et1)]
    r = d.delta_et1.to_numpy(float) / gamma
    dist = d.distance.to_numpy(float)
    print(f"\n=== {label}  (gamma={gamma}, N={len(d):,}) ===")
    print(f"  {'sep bin':>10} {'<response>':>10} {'N':>9}")
    out = []
    for i in range(len(DEDGES) - 1):
        b = (dist >= DEDGES[i]) & (dist < DEDGES[i + 1])
        if b.sum() < 300:
            out.append(np.nan); continue
        m = r[b].mean(); out.append(m)
        print(f"  {DEDGES[i]:.1f}-{DEDGES[i+1]:.1f}\"".rjust(10) + f" {m:>10.4f} {int(b.sum()):>9,}")
    return np.array(out)

# ---- g=0.02: build the blend response from the 0.0 vs 0.02 renders ----
print(f"building g=0.02 blend response for cases {CASES[0]}-{CASES[-1]} ...", flush=True)
frames = []
for c in CASES:
    try:
        df = response.retrieve_response(case=c, r_max=10, r_min=0, k=20, real="real0",
                                        tile_name="tile180.0_-0.5", data_path=HALF,
                                        shear_cases=["0.0", "0.02"])
        frames.append(df); print(f"  case{c}: {len(df):,} pairs", flush=True)
    except Exception as e:
        print(f"  case{c}: FAILED {e}", flush=True)
d002 = pd.concat(frames, ignore_index=True)
d002.to_feather("/home/z/Zekang.Zhang/SBSI/results/resp002_c40-59.feather")
print(f"saved g=0.02 response: {len(d002):,} pairs", flush=True)

# ---- g=0.2: same cases from the existing catalogue ----
d02 = pf.read_table(RESP02, columns=["case", "r_input_p", "r_input_s", "distance", "delta_et1"]).to_pandas()
d02 = d02[d02.case.isin(CASES)].reset_index(drop=True)

c002 = curve(d002, 0.02, "g=0.02 (constgold shear)")
c02 = curve(d02, 0.2, "g=0.20 (emulator training shear)")

print("\n=== DIP COMPARISON (response vs separation) ===")
print(f"  {'sep':>10} {'g=0.02':>9} {'g=0.20':>9} {'ratio 02/002':>13}")
for i in range(len(DEDGES) - 1):
    if np.isnan(c002[i]) or np.isnan(c02[i]):
        continue
    print(f"  {DEDGES[i]:.1f}-{DEDGES[i+1]:.1f}\"".rjust(10)
          + f" {c002[i]:>9.4f} {c02[i]:>9.4f} {c02[i]/c002[i] if abs(c002[i])>1e-6 else np.nan:>13.2f}")
print("\nIf g=0.20 dips deeply at ~1\" but g=0.02 does NOT -> dip is a large-shear artifact; retrain at g=0.02.")
print("DIP_VS_SHEAR_DONE")
