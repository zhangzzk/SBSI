"""Branch discriminator (offline): is the q3 deficit a crowd-flux R_flow OVER-SUPPRESSION (flow fix)
or a genuine R_blend under-prediction (emulator fix)?

Uses the blend-FREE self-response vs target-mag curve measured on the ISOLATED population by cg_dump
(job 14984541). For each blended object we look up R_flow_iso(r_p) = what its self-response would be
UN-crowded, and compare two closures within each R_blend quantile:
    m_crowd = <r_sim> / (R_flow_crowd_bin + <R_blend>) - 1     (framework: crowd-conditioned flow)
    m_iso   = <r_sim> / (<R_flow_iso(r_p)> + <R_blend>) - 1     (un-suppressed flow)
If m_iso ~ 0 while m_crowd = +9% -> crowd conditioning over-suppresses R_flow (branch A).
If m_iso still ~ +9%             -> R_blend genuinely too low (branch B).
"""
import numpy as np, pyarrow.feather as pf

DUMP = "/home/z/Zekang.Zhang/SBSI/results/constgold_perobj_raw.feather"
eps, n_blend = 0.02, 4
# blend-free self-response at bin CENTRE (mag), from cg_dump ISOLATED table (job 14984541)
ISO_MAG = np.array([21.0, 24.25, 24.75, 25.25, 25.75, 26.25, 26.75, 27.55])
ISO_RF  = np.array([1.0728, 0.6330, 0.4409, 0.2958, 0.1675, 0.0709, 0.0130, -0.0232])
RFLOW_CROWD = {1: 0.2761, 2: 0.2281, 3: 0.1754, 4: 0.0651}   # crowd-conditioned per-bin (framework)

d = pf.read_table(DUMP, columns=["r_input_p", "r_sim", "R_blend"]).to_pandas()
rp = d["r_input_p"].to_numpy(float); rs = d["r_sim"].to_numpy(float); rb = d["R_blend"].to_numpy(float)
rflow_iso = np.interp(rp, ISO_MAG, ISO_RF)                     # per-object un-crowded self-response

hi = rb >= eps
qe = np.quantile(rb[hi], np.linspace(0, 1, n_blend + 1)); qe[0] -= 1e-9; qe[-1] += 1e-9
bi = np.where(~hi, 0, 1 + np.clip(np.digitize(rb, qe) - 1, 0, n_blend - 1))

print(f"{'bin':>5} {'N':>10} {'<r_sim>':>8} {'Rf_crowd':>8} {'<Rf_iso>':>8} {'<R_bl>':>7} "
      f"{'m_crowd':>8} {'m_iso':>8}")
for q in range(1, n_blend + 1):
    m = bi == q
    mr = rs[m].mean(); mb = rb[m].mean(); rfi = rflow_iso[m].mean(); rfc = RFLOW_CROWD[q]
    print(f"{'q'+str(q):>5} {m.sum():>10,} {mr:>8.4f} {rfc:>8.4f} {rfi:>8.4f} {mb:>7.3f} "
          f"{mr/(rfc+mb)-1:>+8.1%} {mr/(rfi+mb)-1:>+8.1%}")

# also the ISO bin (R_blend<eps): pure self-response check, crowd vs iso should agree there
m = bi == 0
print(f"{'ISO':>5} {m.sum():>10,} {rs[m].mean():>8.4f} {'--':>8} {rflow_iso[m].mean():>8.4f} "
      f"{rb[m].mean():>7.3f} {'--':>8} {rs[m].mean()/(rflow_iso[m].mean()+rb[m].mean())-1:>+8.1%}")
print("\nNOTE: <Rf_iso> is the blend-FREE self-response at these objects' target mags; if it is much")
print("higher than Rf_crowd AND m_iso~0, the crowd-flux conditioning is over-suppressing R_flow.")
print("FLOW_VS_BLEND_DONE")
