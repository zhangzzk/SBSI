#!/usr/bin/env python
"""cont.108: MAG-resolved component resolution on the HALF-SHEAR sims (NOT constgold r_sim) — the
clean separation the owner asked for.  Same footing as the size analysis, organised by MAG:
per mag bin, R_self and R_blend each get LABEL vs PREDICTION vs RESID, cleanly separated (R_self by
projection onto ghat_p, R_blend the SINGLE-neighbour per-pair response) — no summing/multiplicity
confound (that only exists on constgold).  Isolates whether a mag-cut problem lives in R_self, in
R_blend, or in neither (=> the constgold mag-bias is a deployment/summing artifact, not a component).

Weighting: R_self aggregated by the full-leg label weights W_self (pred weighted by the same size
mix); R_blend aggregated by blended-cell counts.  Reads three cont.108 npz.
"""
import numpy as np, os
D = "/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk"
L = np.load(f"{D}/halfshear_component_labels.npz")
F = np.load(f"{D}/flow_selfresp_grid.npz")
B = np.load(f"{D}/rblend_pairpred_grid.npz")
G = float(L["G"]); ME = L["MAG_EDGES"]; SE = L["SIZE_EDGES"]
nmag, nsz = len(ME) - 1, len(SE) - 1

# R_self label (full leg) and flow prediction
S_self = L["S_self"].sum(0); W = L["W_self"].sum(0)
self_lab = np.where(W > 0, S_self / W / G, np.nan)
self_prd = F["Rgrid"]
# R_blend single-pair label & per-pair emulator prediction (over blended cells)
bl_lab = np.where(B["Ncell"] > 0, B["Slab"] / B["Ncell"], np.nan)
bl_prd = np.where(B["Ncell"] > 0, B["Sprd"] / B["Ncell"], np.nan)
Nb = B["Ncell"].astype(float)

def wmean(cellvals, wts, mm):
    v = cellvals[mm, :]; w = wts[mm, :]
    ok = np.isfinite(v) & (w > 0)
    return (v[ok] * w[ok]).sum() / w[ok].sum() if w[ok].sum() > 0 else np.nan

lines = []
def emit(s): print(s, flush=True); lines.append(s)
emit("# MAG-RESOLVED component resolution on HALF-SHEAR sims (clean, no constgold summing).")
emit("# R_self: label=<e_snc.ghat_p>/g vs flow R_model.  R_blend: single-pair label vs per-pair emulator.")
emit("")
emit(f"{'mag bin':10s}| {'R_self_lab':>10s}{'R_self_prd':>10s}{'resid':>8s}{'rel%':>7s} "
     f"| {'R_bl_lab':>9s}{'R_bl_prd':>9s}{'resid':>8s} "
     f"| {'Rtot_lab':>9s}{'Rtot_prd':>9s}{'m_clean%':>9s}")
WINS = [("18-24", [0]), ("24-25", [1]), ("25-26", [2]), ("26-28", [3]),
        ("24-26", [1, 2]), ("18-28(all)", [0, 1, 2, 3])]
for name, mm in WINS:
    mm = np.array(mm)
    sl = wmean(self_lab, W, mm); sp = wmean(self_prd, W, mm)
    bl = wmean(bl_lab, Nb, mm); bp = wmean(bl_prd, Nb, mm)
    # R_total on the clean single-pair footing, blend weighted by blend-fraction per mag bin
    #   blend fraction f = (blended count) / (all-self count) per mag bin
    fblend = Nb[mm, :].sum() / np.maximum(W[mm, :].sum(), 1e-9)
    tot_lab = sl + fblend * bl
    tot_prd = sp + fblend * bp
    mclean = (tot_lab / tot_prd - 1) * 100 if tot_prd != 0 else np.nan
    emit(f"{name:10s}| {sl:+10.4f}{sp:+10.4f}{sp-sl:+8.4f}{100*(sp-sl)/abs(sl):>+6.1f}% "
         f"| {bl:+9.4f}{bp:+9.4f}{bp-bl:+8.4f} "
         f"| {tot_lab:+9.4f}{tot_prd:+9.4f}{mclean:>+8.2f}%")
emit("")
emit("resid = pred-label (per component).  m_clean = R_total_lab/R_total_pred-1 on the SINGLE-PAIR")
emit("half-shear footing (blend weighted by per-mag blend fraction) — the mag-cut bias with the")
emit("constgold summing REMOVED.  If |resid_self|,|resid_blend| and m_clean are all small across mag")
emit("=> there is NO clean mag-cut problem in the components; the constgold mag-bias is the summed-")
emit("R_blend/aperture deployment artifact, not a per-component (self or per-pair) failure.")
OUT = f"{D}/mag_resolved_components.txt"
open(OUT, "w").write("\n".join(lines) + "\n")
emit(f"\nwrote {OUT}")
