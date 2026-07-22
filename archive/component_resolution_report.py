#!/usr/bin/env python
"""cont.108 REPORT: combine the half-shear LABEL decomposition (halfshear_component_labels.npz)
with the flow R_self PREDICTION (flow_selfresp_grid.npz) into the clean per-property resolution
map the owner asked for:  for R_self, LABEL vs PREDICTION vs RESIDUAL(pred-label) per (mag x size)
cell + deliverable windows.  This localises where the flow mis-resolves its OWN self-response
target -- firewall-clean (no constgold), the honest R_self resolution.

R_blend prediction is Stage 2 (needs per-pair emulator to match the single-neighbour label;
the summed-neighbour blend_lookup is not apples-to-apples) -- the R_blend LABEL is reported here
as the ground-truth trend the emulator must reproduce.
"""
import numpy as np, os
D = "/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk"
LAB = f"{D}/halfshear_component_labels.npz"
PRED = f"{D}/flow_selfresp_grid.npz"
OUT = f"{D}/component_resolution_report.txt"
G = 0.05


def main():
    lines = []
    def emit(s): print(s, flush=True); lines.append(s)
    L = np.load(LAB)
    S_self, W_self = L["S_self"], L["W_self"]
    S_bl, W_bl = L["S_bl"], L["W_bl"]
    ME, SE = L["MAG_EDGES"], L["SIZE_EDGES"]
    nmag, nsz = len(ME) - 1, len(SE) - 1

    def R_lab(S, W, mm, ss):
        s = S[:, mm][:, :, ss].sum(); w = W[:, mm][:, :, ss].sum()
        return (s / w / G) if w > 0 else np.nan

    self_lab = np.array([[R_lab(S_self, W_self, [a], [b]) for b in range(nsz)] for a in range(nmag)])
    bl_lab = np.array([[R_lab(S_bl, W_bl, [a], [b]) for b in range(nsz)] for a in range(nmag)])

    have_pred = os.path.exists(PRED)
    if have_pred:
        P = np.load(PRED)
        self_pred = P["Rgrid"]
        assert np.allclose(P["MAG_EDGES"], ME) and np.allclose(P["SIZE_EDGES"], SE), "grid mismatch"
    else:
        self_pred = np.full_like(self_lab, np.nan)
        emit("WARNING: flow_selfresp_grid.npz not found -> prediction columns blank")

    szlab = "  ".join(f"[{SE[j]:.2f},{SE[j+1]:.2f})" for j in range(nsz))
    emit(f"# COMPONENT RESOLUTION (half-shear crowd leg, own training data). size bins: {szlab}")
    emit(f"# R_self: LABEL=<e_snc.ghat_p>/g (== flow target) ; PRED=flow induced R_model ; RESID=pred-label")
    emit("")
    emit("=== R_self  LABEL / PRED / RESID  per (mag x size) ===")
    for a in range(nmag):
        emit(f"mag[{ME[a]:.0f},{ME[a+1]:.0f}):")
        emit("  label: " + "".join(f"{self_lab[a,b]:+8.3f}" for b in range(nsz)))
        emit("  pred : " + "".join(f"{self_pred[a,b]:+8.3f}" for b in range(nsz)))
        emit("  resid: " + "".join(f"{self_pred[a,b]-self_lab[a,b]:+8.3f}" for b in range(nsz)))
    emit("")

    # windows: aggregate label (weighted) and pred (count-weighted via Ngrid if present)
    def agg_lab(S, W, mm, ss):
        s = S[:, mm][:, :, ss].sum(); w = W[:, mm][:, :, ss].sum()
        return (s / w / G) if w > 0 else np.nan
    Ngrid = P["Ngrid"] if have_pred else np.ones((nmag, nsz))
    def agg_pred(mm, ss):
        num = 0.0; den = 0.0
        for a in mm:
            for b in ss:
                if np.isfinite(self_pred[a, b]):
                    num += self_pred[a, b] * Ngrid[a, b]; den += Ngrid[a, b]
        return num / den if den > 0 else np.nan

    def mm_of(lo, hi): return [a for a in range(nmag) if ME[a] >= lo - 1e-6 and ME[a+1] <= hi + 1e-6]
    def ss_of(lo, hi): return [b for b in range(nsz) if SE[b] >= lo - 1e-6 and SE[b+1] <= hi + 1e-6]
    ALLM, ALLS = list(range(nmag)), list(range(nsz))
    WINS = [("GLOBAL", ALLM, ALLS), ("mag24-25", mm_of(24, 25), ALLS), ("mag25-26", mm_of(25, 26), ALLS),
            ("mag24-26", mm_of(24, 26), ALLS), ("size_gt0.3", ALLM, ss_of(0.3, 1.5)),
            ("mag24-26&sz>0.3", mm_of(24, 26), ss_of(0.3, 1.5))]
    emit("=== DELIVERABLE WINDOWS: R_self label/pred/resid  +  R_blend label (ground truth) ===")
    emit(f"  {'window':18s}{'R_self_lab':>12s}{'R_self_pred':>12s}{'R_self_resid':>13s}"
         f"{'R_blend_lab':>13s}{'R_tot_lab':>11s}")
    for name, mm, ss in WINS:
        if not mm or not ss:
            continue
        sl = agg_lab(S_self, W_self, mm, ss)
        sp = agg_pred(mm, ss)
        bl = agg_lab(S_bl, W_bl, mm, ss)
        emit(f"  {name:18s}{sl:+12.4f}{sp:+12.4f}{sp-sl:+13.4f}{bl:+13.4f}{sl+bl:+11.4f}")
    emit("")
    emit("READING: R_self_resid = the flow's non-closure to its OWN self-response target in property")
    emit("space (pred>label = flow OVER-resolves R_self). This is firewall-clean and is the dominant")
    emit("term (R_self ~6-7x R_blend on the gentle windows). R_blend_lab is the single-neighbour")
    emit("ground truth the emulator (Stage 2, per-pair) must reproduce.")
    open(OUT, "w").write("\n".join(lines) + "\n")
    emit(f"\nwrote {OUT}\nREPORT_DONE")


if __name__ == "__main__":
    main()
