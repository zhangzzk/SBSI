"""Test the user's lead: a DIP in R_blend at intermediate angular separation present in the ISOTROPIC
measurements (half-shear truth + emulator) but NOT in the COHERENT (constant) response. If so, the
isotropic average sits below the coherent -> the +0.039 q3 gap.

ISOTROPIC side: response_catalogue_train cases 0-39 (leakage-free), per-pair truth delta_et1/gamma(0.2)
and emulator pred, vs FINE pair separation, in a q3-relevant magnitude slice.
COHERENT side: constgold single-dominant objects (one neighbour carries the blend), coherent blend
= r_sim - R_self(target mag), vs the nearest-neighbour distance. If the isotropic curve dips at mid-sep
and the coherent one does not, the mechanism is confirmed.
"""
import os, sys, numpy as np, pandas as pd, pyarrow.feather as pf, yaml
sys.path.insert(0, "/home/z/Zekang.Zhang/blendemu")
from blendemu.inference import BlendingPredictor

RESP = "/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876/response_catalogue_train.feather"
DUMP = "/home/z/Zekang.Zhang/SBSI/results/constgold_perobj_raw.feather"
MULT = "/home/z/Zekang.Zhang/SBSI/results/blend_multiplicity_extnbrho_c0-39.feather"
COND = dict(pixel_size=0.2, zero_point=30.0, psf_fwhm=0.73, moffat_beta=2.224, pixel_rms=0.312)
GAMMA = 0.2
# R_self truth by target mag (self audit, job 15004817)
RSELF_MAG = {(24, 24.5): 0.5699, (24.5, 25): 0.3638, (25, 25.5): 0.2315, (25.5, 26): 0.1222}
RP_LO, RP_HI = 24.0, 26.0                       # q3-relevant target magnitude slice
DEDGES = np.arange(0.0, 6.01, 0.4)

def rself_of(rp):
    out = np.full(len(rp), np.nan)
    for (lo, hi), v in RSELF_MAG.items():
        out[(rp >= lo) & (rp < hi)] = v
    return out

# ---------- ISOTROPIC: half-shear per-pair truth + emulator, vs fine separation ----------
cols = ["case", "input_index", "r_input_p", "r_input_s", "Re_input_p", "Re_input_s",
        "sersic_n_input_p", "sersic_n_input_s", "distance", "delta_et1", "delta_et2"]
print("reading response_catalogue_train gold rows ...", flush=True)
t = pf.read_table(RESP, columns=cols).to_pandas()
t = t[(t.case < 40) & np.isfinite(t.delta_et1)
      & (t.r_input_p >= RP_LO) & (t.r_input_p < RP_HI) & (t.r_input_s > 18) & (t.r_input_s < 27)
      & (t.distance > 0) & (t.distance < 6)].reset_index(drop=True)
print(f"isotropic pairs (r_p[{RP_LO},{RP_HI}], r_s[18,27], d<6): {len(t):,}", flush=True)
pred = BlendingPredictor.load("/home/z/Zekang.Zhang/blendemu/models", tag="lsst_r_extnbr_ho",
                              conditions=COND, device="cpu")
t["pred"] = pred.predict_on_pairs(t, task="response", warn_extrapolation=False)["response"].to_numpy(float)
t["truth"] = t.delta_et1.to_numpy(float) / GAMMA
t["truth2"] = t.delta_et2.to_numpy(float) / GAMMA

print("\n=== ISOTROPIC (half-shear) response vs separation ===")
print(f"  {'sep bin':>10} {'<truth_et1>':>11} {'<pred>':>9} {'<truth_et2>':>11} {'N':>9}")
for i in range(len(DEDGES) - 1):
    b = (t.distance >= DEDGES[i]) & (t.distance < DEDGES[i + 1])
    if b.sum() < 500:
        continue
    print(f"  {DEDGES[i]:.1f}-{DEDGES[i+1]:.1f}\"".rjust(10)
          + f" {t.truth[b].mean():>11.4f} {t.pred[b].mean():>9.4f} {t.truth2[b].mean():>11.4f} {int(b.sum()):>9,}")

# ---------- COHERENT: constgold single-dominant objects, coherent blend vs separation ----------
d = pf.read_table(DUMP, columns=["case", "input_index", "r_input_p", "r_sim", "R_blend", "distance"]).to_pandas()
ml = pf.read_table(MULT, columns=["case", "input_index", "rb_max"]).to_pandas()
d = d.merge(ml, on=["case", "input_index"], how="left")
dom = np.where(np.abs(d.R_blend) > 1e-9, np.abs(d.rb_max) / np.maximum(np.abs(d.R_blend), 1e-9), 0.0)
rp = d.r_input_p.to_numpy(float)
sel = (dom >= 0.7) & (d.R_blend.to_numpy() >= 0.02) & (rp >= RP_LO) & (rp < RP_HI) \
      & (d.distance.to_numpy() > 0) & (d.distance.to_numpy() < 6)
d = d[sel].reset_index(drop=True)
coh = d.r_sim.to_numpy(float) - rself_of(d.r_input_p.to_numpy(float))
dist = d.distance.to_numpy(float); rbe = d.R_blend.to_numpy(float)
print(f"\n=== COHERENT (constgold single-dominant, dom>=0.7) blend vs separation: {len(d):,} objects ===")
print(f"  {'sep bin':>10} {'<coh_blend>':>11} {'<R_bl_emu>':>10} {'N':>9}")
for i in range(len(DEDGES) - 1):
    b = (dist >= DEDGES[i]) & (dist < DEDGES[i + 1])
    if b.sum() < 500:
        continue
    print(f"  {DEDGES[i]:.1f}-{DEDGES[i+1]:.1f}\"".rjust(10)
          + f" {np.nanmean(coh[b]):>11.4f} {rbe[b].mean():>10.4f} {int(b.sum()):>9,}")
print("\nIf ISOTROPIC dips at mid-sep (e.g. 1-3\") but COHERENT stays high -> the dip is the mechanism.")
print("AUDIT_SEP_DIP_DONE")
