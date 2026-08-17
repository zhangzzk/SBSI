"""Mean emulator R_blend over the SIM's selected objects, per cut.

R_blend is a fixed per-object number with no leg or cut dependence, so R_blend(cut) is purely a
REWEIGHTING. In the table the reweighting uses the FLOW's own drawn measured mag/size. Here it uses
the SIM's real measured MAG_AUTO / FLUX_RADIUS, leg by leg, averaged over legs exactly as
model_selected() does. The difference between the two isolates how much of the blend term's
cut-response is the flow's measured-quantity distribution rather than the emulator.
"""
import os, sys, json
import numpy as np
import pyarrow.feather as pf

ROOT = "/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot"
sys.path.insert(0, ROOT)
from sbs_shear.preprocessing import DEFAULT_SELECTION_CUTS, source_select_selection

NEED = ["case", "input_index", "neighbored", "distance", "polarization_angle",
        "Re_input_p", "Re_input_s", "axis_ratio_input_p", "axis_ratio_input_s",
        "position_angle_input_p", "position_angle_input_s", "r_input_p", "r_input_s",
        "redshift_input_p", "redshift_input_s", "sersic_n_input_p", "sersic_n_input_s",
        "applied_g1", "applied_g2", "shear_magnitude", "shear_angle",
        "measured_e1_plus", "measured_e2_plus", "measured_e1_minus", "measured_e2_minus",
        "et_plus", "S/N_plus", "S/N_minus",
        "measured_mag_auto_plus", "measured_mag_auto_minus",
        "measured_flux_radius_plus", "measured_flux_radius_minus"]
CG = ("/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/"
      "constant_response_catalogue_train.feather")

df = pf.read_table(CG, columns=NEED, memory_map=True).to_pandas()
df = df[df["case"] >= 40].reset_index(drop=True)
df = source_select_selection(df, cuts=DEFAULT_SELECTION_CUTS).reset_index(drop=True)
dom = (df["r_input_p"].to_numpy(float) < 26.0) & (df["Re_input_p"].to_numpy(float) > 0.3)
df = df[dom].reset_index(drop=True)
print("population after selection+domain:", len(df), flush=True)

lk = pf.read_table(os.path.join(ROOT, "results/blend_lookup_indomtuned_c40-139.feather"),
                   memory_map=True).to_pandas()
j = df[["case", "input_index"]].merge(lk, on=["case", "input_index"], how="left")
rb = j["R_blend"].to_numpy(float)
print("blend matched %.4f%%  <R_blend>=%.6f" % (100*np.isfinite(rb).mean(), np.nanmean(rb)), flush=True)

magp = df["measured_mag_auto_plus"].to_numpy(float)
magm = df["measured_mag_auto_minus"].to_numpy(float)
szp  = df["measured_flux_radius_plus"].to_numpy(float)
szm  = df["measured_flux_radius_minus"].to_numpy(float)
snp  = df["S/N_plus"].to_numpy(float); snm = df["S/N_minus"].to_numpy(float)
c2 = np.cos(2*df.shear_angle.to_numpy(float)); s2 = np.sin(2*df.shear_angle.to_numpy(float))
me1p, me2p = df.measured_e1_plus.to_numpy(float), df.measured_e2_plus.to_numpy(float)
me1m, me2m = df.measured_e1_minus.to_numpy(float), df.measured_e2_minus.to_numpy(float)
i1 = np.zeros(len(df))  # only need finiteness of the measured shapes for the mask

fin = (np.isfinite(snp) & np.isfinite(snm) & np.isfinite(magp) & np.isfinite(magm)
       & np.isfinite(szp) & np.isfinite(szm)
       & np.isfinite(me1p) & np.isfinite(me2p) & np.isfinite(me1m) & np.isfinite(me2m)
       & np.isfinite(rb))
print("finite rows:", int(fin.sum()), flush=True)

PX = 0.2
sn_a, sn_b = -0.4, -1.0   # placeholder; proxy rows recomputed below only if defaults known
cuts = {}
for c in (26.0, 25.5, 25.0):
    cuts["mag<%g" % c] = (magp < c, magm < c)
for a in (0.30, 0.40, 0.60, 0.70):
    cuts['R>%.2f"' % a] = (szp > a/PX, szm > a/PX)
cuts['mag<26 & R>0.30"'] = ((magp < 26.0) & (szp > 0.30/PX), (magm < 26.0) & (szm > 0.30/PX))
cuts['mag<25 & R>0.40"'] = ((magp < 25.0) & (szp > 0.40/PX), (magm < 25.0) & (szm > 0.40/PX))
for t in (8.0, 9.0, 10.0):
    cuts["S/N>%g (real)" % t] = (snp > t, snm > t)

out = {"n": int(fin.sum()), "nocut": float(np.mean(rb[fin]))}
print("\n%-22s %10s %10s %10s" % ("cut", "<Rb>_sim", "dRb/Rb %", "keep%"))
print("%-22s %10.6f %10s %10s" % ("__nocut__", out["nocut"], "-", "-"))
for k, (pp, pm) in cuts.items():
    a = fin & pp; b = fin & pm
    v = 0.5*(rb[a].mean() + rb[b].mean())
    kp = 0.5*(a.sum() + b.sum())/fin.sum()
    out[k] = [float(v), float(kp)]
    print("%-22s %10.6f %10.4f %10.4f" % (k, v, 100*(v/out["nocut"]-1), 100*kp))

json.dump(out, open(os.path.join(ROOT, ".scratch/sim_rblend.json"), "w"), indent=1)
print("\nwrote .scratch/sim_rblend.json")
