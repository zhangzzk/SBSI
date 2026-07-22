#!/usr/bin/env python
"""cont.110: NON-CIRCULAR faint/far R_blend debias derivation.

The per-pair emulator's OWN held-out truth says its bias is tiny in ABSOLUTE (response-weighted)
terms (+0.48-0.65%, domain-insensitive; job 14981379) -> that is WHY constgold GLOBAL is unbiased,
and why the emulator SELF-debias (build_emu_correction.py) failed to transfer. So here we test the
per-pair emulator against the *coherent deployment truth* = the half-shear tagged-neighbour label,
RESOLVED into the faint (r_s) x far (distance) tail that cont.108's mag x size grid never reached.

  LABEL = <e_snc . ghat_s>/|g_s|   (measured primary response to the ONE tagged neighbour's shear;
          the actual coherent single-pair response the deployment must reproduce)
  PRED  = BlendingPredictor.predict_on_pairs(task='response')  (emulator per-pair, same tagged pair)
  RESID = pred - label ;  DELTA = label - pred  (additive correction to ADD to the emulator)

Reported globally, per neighbour-mag r_s, per (r_s x distance), and cross-tabbed by TARGET mag r_p
(the covariate whose train<->constgold shift broke the self-debias).  Absolute AND relative bias so
we can connect to the GLOBAL cancellation.  Saves DELTA(r_s, distance) for later per-pair application.

Firewall: crowd g0.05 half-shear leg + emulator ONLY.  constgold NEVER read.  CPU (xgboost).
"""
import sys, os, time, numpy as np, pandas as pd, pyarrow as pa, pyarrow.ipc as ipc, pyarrow.feather as pf
SBSI = "/home/z/Zekang.Zhang/SBSI"
sys.path.insert(0, SBSI); sys.path.insert(0, "/home/z/Zekang.Zhang/blendemu")
from sbs_shear.preprocessing import DEFAULT_SELECTION_CUTS, source_select_selection  # noqa: E402
from blendemu.inference import BlendingPredictor  # noqa: E402

CAT = "/project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_crowd_g0.05_val_full.feather"
SNC = "results/g0_lookup_c0-99.feather"
BLEND_MODELS = "/home/z/Zekang.Zhang/blendemu/models"
COND = dict(pixel_size=0.2, zero_point=30.0, psf_fwhm=0.73, moffat_beta=2.224, pixel_rms=0.312)
OUT = "/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/rblend_faintfar_halfshear.txt"
G = 0.05
KEYMUL = 1_000_003
PAIR_FEATS = ["Re_input_p", "r_input_p", "sersic_n_input_p",
              "Re_input_s", "r_input_s", "sersic_n_input_s", "distance"]
# match build_emu_correction.py edges so DELTA is directly comparable to the failed self-debias table
RS_EDGES = np.array([13., 24., 25., 25.5, 26., 26.5, 27., 27.5, 28., 28.5, 29.01])
D_EDGES = np.array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10.01])
RP_EDGES = np.array([18., 23., 24., 24.5, 25., 25.5, 26., 27.5])
t0 = time.time()
def log(*a): print(f"[{time.time()-t0:6.1f}s]", *a, flush=True)


def main():
    lk = pf.read_table(SNC, columns=["case", "input_index", "ngmix0_g1", "ngmix0_g2"]).to_pandas()
    lkey = lk["case"].to_numpy(np.int64) * KEYMUL + lk["input_index"].to_numpy(np.int64)
    order = np.argsort(lkey); lkey = lkey[order]
    e0_1 = lk["ngmix0_g1"].to_numpy(float)[order]; e0_2 = lk["ngmix0_g2"].to_numpy(float)[order]
    log(f"SNC rows={len(lkey):,}; loading emulator (tag=lsst_r, response)")
    pred = BlendingPredictor.load(BLEND_MODELS, tag="lsst_r", conditions=COND, device="cpu")

    COLS = sorted(set(["detected", "case", "input_index", "neighbored", "distance",
                       "gamma1_input_s", "gamma2_input_s", "measured_ngmix_g1", "measured_ngmix_g2"]
                      + PAIR_FEATS))
    parts = []; nread = 0
    with ipc.open_file(CAT) as r:
        cols = [c for c in COLS if c in set(r.schema.names)]
        for bi in range(r.num_record_batches):
            b = pa.Table.from_batches([r.get_batch(bi)]).select(cols).to_pandas()
            nread += len(b)
            b = b[b["detected"].astype(bool)]
            b = source_select_selection(b, cuts=DEFAULT_SELECTION_CUTS)
            if len(b) == 0:
                continue
            gs = np.hypot(b["gamma1_input_s"], b["gamma2_input_s"])
            b = b[b["neighbored"].astype(bool) & (gs.to_numpy() > 1e-6)]
            if len(b) == 0:
                continue
            parts.append(b.reset_index(drop=True))
            if bi % 40 == 0:
                log(f"batch {bi} read={nread:,} kept_blended={sum(len(p) for p in parts):,}")
    df = pd.concat(parts, ignore_index=True); del parts
    log(f"blended+sheared-nbr rows={len(df):,}")

    # LABEL: SNC-cancelled measured shape projected onto ghat_s (single tagged neighbour)
    e1 = df["measured_ngmix_g1"].to_numpy(float); e2 = df["measured_ngmix_g2"].to_numpy(float)
    key = df["case"].to_numpy(np.int64) * KEYMUL + df["input_index"].to_numpy(np.int64)
    pos = np.clip(np.searchsorted(lkey, key), 0, len(lkey) - 1); match = lkey[pos] == key
    e1 = e1 - np.where(match, e0_1[pos], np.nan); e2 = e2 - np.where(match, e0_2[pos], np.nan)
    g1s = df["gamma1_input_s"].to_numpy(float); g2s = df["gamma2_input_s"].to_numpy(float)
    gs = np.hypot(g1s, g2s)
    lab = (g1s * e1 + g2s * e2) / gs / G

    # PRED: emulator per-pair response
    reg = pred.predict_on_pairs(df[PAIR_FEATS].copy(), task="response", warn_extrapolation=False)
    rbp = reg["response"].to_numpy(float)
    log(f"emulator done. GLOBAL label<>={np.nanmean(lab):+.4f}  pred<>={np.nanmean(rbp):+.4f}")

    rs = df["r_input_s"].to_numpy(float); dd = df["distance"].to_numpy(float)
    rp = df["r_input_p"].to_numpy(float); case = df["case"].to_numpy(np.int64)
    fin = np.isfinite(lab) & np.isfinite(rbp) & match & np.isfinite(rs) & np.isfinite(dd)
    lab, rbp, rs, dd, rp, case = lab[fin], rbp[fin], rs[fin], dd[fin], rp[fin], case[fin]
    N = len(lab)
    log(f"finite rows={N:,}")

    lines = []
    def emit(s): print(s, flush=True); lines.append(s)
    rng = np.random.default_rng(11)

    def wmean_boot(mask, arr):
        """<arr> over mask with case-bootstrap std; returns (mean, n, std)."""
        if mask.sum() < 50:
            return np.nan, int(mask.sum()), np.nan
        cs = case[mask]; a = arr[mask]; uc = np.unique(cs)
        boot = np.empty(200)
        for k in range(200):
            pick = rng.choice(uc, len(uc)); sel = np.isin(cs, pick)
            boot[k] = a[sel].mean() if sel.any() else np.nan
        return a.mean(), int(mask.sum()), np.nanstd(boot)

    emit(f"# cont.110 faint/far R_blend: half-shear per-pair LABEL vs emulator PRED (n={N:,})")
    emit(f"# LABEL=<e_snc.ghat_s>/g (coherent single-pair truth)  PRED=emulator per-pair  RESID=pred-label")

    # ---- GLOBAL (absolute, response-weighted; the quantity that governs constgold m) ----
    gl, gn, ge = wmean_boot(np.ones(N, bool), lab)
    gp, _, gpe = wmean_boot(np.ones(N, bool), rbp)
    gr, _, gre = wmean_boot(np.ones(N, bool), rbp - lab)
    rel = gr / gl if abs(gl) > 1e-9 else np.nan
    emit(f"\n=== GLOBAL ===")
    emit(f"  <label>={gl:+.4f}+/-{ge:.4f}  <pred>={gp:+.4f}+/-{gpe:.4f}  "
         f"<resid=pred-label>={gr:+.4f}+/-{gre:.4f}  (rel {100*rel:+.1f}%)")

    # ---- per neighbour-mag r_s (distance-collapsed) ----
    ri = np.clip(np.digitize(rs, RS_EDGES) - 1, 0, len(RS_EDGES) - 2)
    di = np.clip(np.digitize(dd, D_EDGES) - 1, 0, len(D_EDGES) - 2)
    pri = np.clip(np.digitize(rp, RP_EDGES) - 1, 0, len(RP_EDGES) - 2)
    nr, nd, npr = len(RS_EDGES) - 1, len(D_EDGES) - 1, len(RP_EDGES) - 1
    emit(f"\n=== per NEIGHBOUR-mag r_s (distance-collapsed) ===")
    emit(f"  {'r_s bin':16s}{'label':>9s}{'pred':>9s}{'resid':>9s}{'err':>8s}{'rel%':>8s}{'N':>10s}")
    for i in range(nr):
        m = ri == i
        vl, n, _ = wmean_boot(m, lab); vp, _, _ = wmean_boot(m, rbp); vr, _, er = wmean_boot(m, rbp - lab)
        rl = 100 * vr / vl if abs(vl) > 1e-9 else np.nan
        emit(f"  [{RS_EDGES[i]:5.1f},{RS_EDGES[i+1]:5.1f})  {vl:+9.4f}{vp:+9.4f}{vr:+9.4f}{er:>8.4f}{rl:>8.1f}{n:>10,}")

    # ---- DELTA(r_s x distance) = <label - pred> (additive correction; the non-circular debias table)
    D = np.full((nr, nd), np.nan); C = np.zeros((nr, nd), np.int64)
    Slab = np.zeros((nr, nd)); Sprd = np.zeros((nr, nd))
    cellr = ri * nd + di
    np.add.at(Slab.reshape(-1), cellr, lab)
    np.add.at(Sprd.reshape(-1), cellr, rbp)
    np.add.at(C.reshape(-1), cellr, 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        D = np.where(C > 0, (Slab - Sprd) / np.maximum(C, 1), np.nan)
    emit(f"\n=== RESID = pred-label by (r_s x distance) [blank=<50 pairs] ===")
    emit("  r_s\\dist  " + "".join(f"{D_EDGES[j]:.0f}-{D_EDGES[j+1]:.0f}".rjust(8) for j in range(nd)))
    for i in range(nr):
        row = "".join((f"{-D[i,j]:+8.3f}" if C[i, j] >= 50 else "     .  ") for j in range(nd))
        emit(f"  [{RS_EDGES[i]:4.1f},{RS_EDGES[i+1]:4.1f}){row}")

    # ---- cross-tab by TARGET mag r_p (the covariate that broke the self-debias) ----
    emit(f"\n=== RESID = pred-label by (r_s x TARGET-mag r_p)  [tests covariate-shift dependence] ===")
    emit("  r_s\\r_p  " + "".join(f"[{RP_EDGES[j]:.1f},{RP_EDGES[j+1]:.1f})".rjust(11) for j in range(npr)))
    for i in range(nr):
        cells = []
        for j in range(npr):
            m = (ri == i) & (pri == j)
            if m.sum() >= 50:
                cells.append(f"{(rbp[m]-lab[m]).mean():+11.3f}")
            else:
                cells.append("     .     ")
        emit(f"  [{RS_EDGES[i]:4.1f},{RS_EDGES[i+1]:4.1f}){''.join(cells)}")

    # ---- absolute-contribution view: how much summed R_blend does each r_s carry? ----
    emit(f"\n=== absolute R_blend budget by r_s: <label>*frac  (what the sum actually accumulates) ===")
    emit(f"  {'r_s bin':16s}{'frac_pairs':>11s}{'<label>':>9s}{'<pred>':>9s}{'contrib_resid':>14s}")
    for i in range(nr):
        m = ri == i
        frac = m.mean()
        vl = lab[m].mean() if m.sum() else np.nan
        vp = rbp[m].mean() if m.sum() else np.nan
        emit(f"  [{RS_EDGES[i]:5.1f},{RS_EDGES[i+1]:5.1f})  {frac:>11.4f}{vl:+9.4f}{vp:+9.4f}{frac*(vp-vl):>+14.5f}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez(OUT.replace(".txt", ".npz"), Delta=D, Slab=Slab, Sprd=Sprd, C=C,
             RS_EDGES=RS_EDGES, D_EDGES=D_EDGES, G=G,
             global_label=gl, global_pred=gp, global_resid=gr)
    open(OUT, "w").write("\n".join(lines) + "\n")
    emit(f"\nDELTA(r_s,distance) saved. RESID>0 => emulator OVER-predicts the coherent single-pair response.")
    emit(f"wrote {OUT}\nRBLEND_FAINTFAR_DONE")


if __name__ == "__main__":
    main()
