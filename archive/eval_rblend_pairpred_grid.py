#!/usr/bin/env python
"""cont.108 Stage 2: R_blend LABEL vs per-pair PREDICTION on the SAME blended rows of the crowd
g0.05 half-shear leg, apples-to-apples (both SINGLE tagged-neighbour -> no multiplicity mismatch).

LABEL      = <e_snc . ghat_s>/|g_s|   (measured primary response to the tagged neighbour's shear)
PREDICTION = BlendingPredictor.predict_on_pairs(task='response')  = emulator delta_et_primary/gamma
             for the SAME tagged (primary, secondary) pair (per-pair, NOT summed over neighbours).

Both are the response of the primary shape to ONE neighbour's shear -> directly comparable, unlike
the certified summed blend_lookup (cont.107 "over-compensation" was partly this summing artifact).
Binned into the deliverable (mag x size) grid + windows.  Firewall: measures truth on the half-shear
training leg + emulator; constgold NEVER read.  CPU (xgboost).
"""
import sys, os, time, argparse, numpy as np, pandas as pd, pyarrow as pa, pyarrow.ipc as ipc, pyarrow.feather as pf
SBSI = "/home/z/Zekang.Zhang/SBSI"
sys.path.insert(0, SBSI); sys.path.insert(0, "/home/z/Zekang.Zhang/blendemu")
from sbs_shear.preprocessing import DEFAULT_SELECTION_CUTS, source_select_selection  # noqa: E402
from blendemu.inference import BlendingPredictor  # noqa: E402

CAT = "/project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_crowd_g0.05_val_full.feather"
SNC = "results/g0_lookup_c0-99.feather"
BLEND_MODELS = "/home/z/Zekang.Zhang/blendemu/models"
COND = dict(pixel_size=0.2, zero_point=30.0, psf_fwhm=0.73, moffat_beta=2.224, pixel_rms=0.312)
OUT = "/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/rblend_pairpred_grid.txt"
G = 0.05
KEYMUL = 1_000_003
MAG_EDGES = np.array([18., 24., 25., 26., 28.])
SIZE_EDGES = np.array([0.1, 0.24, 0.30, 0.38, 0.50, 1.50])
PAIR_FEATS = ["Re_input_p", "r_input_p", "sersic_n_input_p",
              "Re_input_s", "r_input_s", "sersic_n_input_s", "distance"]
t0 = time.time()
def log(*a): print(f"[{time.time()-t0:6.1f}s]", *a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-case", type=int, default=99)
    ap.add_argument("--max-rows", type=int, default=0)
    args = ap.parse_args()

    lk = pf.read_table(SNC, columns=["case", "input_index", "ngmix0_g1", "ngmix0_g2"]).to_pandas()
    lkey = lk["case"].to_numpy(np.int64) * KEYMUL + lk["input_index"].to_numpy(np.int64)
    order = np.argsort(lkey); lkey = lkey[order]
    e0_1 = lk["ngmix0_g1"].to_numpy(float)[order]; e0_2 = lk["ngmix0_g2"].to_numpy(float)[order]
    log(f"SNC rows={len(lkey):,}; loading emulator (tag=lsst_r, response)")
    pred = BlendingPredictor.load(BLEND_MODELS, tag="lsst_r", conditions=COND, device="cpu")

    COLS = (["detected", "case", "input_index", "neighbored", "distance",
             "gamma1_input_s", "gamma2_input_s", "measured_ngmix_g1", "measured_ngmix_g2",
             "Re_input_p", "r_input_p"] + PAIR_FEATS)
    COLS = sorted(set(COLS))
    parts = []; nread = 0
    with ipc.open_file(CAT) as r:
        cols = [c for c in COLS if c in set(r.schema.names)]
        for bi in range(r.num_record_batches):
            if args.max_rows and nread >= args.max_rows:
                break
            b = pa.Table.from_batches([r.get_batch(bi)]).select(cols).to_pandas()
            nread += len(b)
            b = b[b["case"] <= args.max_case]
            if len(b) == 0:
                continue
            b = b[b["detected"].astype(bool)]
            b = source_select_selection(b, cuts=DEFAULT_SELECTION_CUTS)
            if len(b) == 0:
                continue
            gs = np.hypot(b["gamma1_input_s"], b["gamma2_input_s"])
            b = b[b["neighbored"].astype(bool) & (gs.to_numpy() > 1e-6)]   # blended + sheared neighbour
            if len(b) == 0:
                continue
            parts.append(b.reset_index(drop=True))
            if bi % 40 == 0:
                log(f"batch {bi} read={nread:,} kept_blended={sum(len(p) for p in parts):,}")
    df = pd.concat(parts, ignore_index=True); del parts
    log(f"blended rows={len(df):,}")

    # LABEL: SNC-cancelled measured shape projected onto ghat_s
    e1 = df["measured_ngmix_g1"].to_numpy(float); e2 = df["measured_ngmix_g2"].to_numpy(float)
    key = df["case"].to_numpy(np.int64) * KEYMUL + df["input_index"].to_numpy(np.int64)
    pos = np.clip(np.searchsorted(lkey, key), 0, len(lkey) - 1); match = lkey[pos] == key
    e1 = e1 - np.where(match, e0_1[pos], np.nan); e2 = e2 - np.where(match, e0_2[pos], np.nan)
    g1s = df["gamma1_input_s"].to_numpy(float); g2s = df["gamma2_input_s"].to_numpy(float)
    gs = np.hypot(g1s, g2s)
    lab = (g1s * e1 + g2s * e2) / gs / G                    # R_blend label per obj (single neighbour)

    # PREDICTION: per-pair emulator response for the tagged pair
    reg = pred.predict_on_pairs(df[PAIR_FEATS].copy(), task="response", warn_extrapolation=False)
    rbp = reg["response"].to_numpy(float)                   # delta_et/gamma per pair
    log(f"per-pair emulator done. label<>=  {np.nanmean(lab):+.4f}   pred<>= {np.nanmean(rbp):+.4f}")

    mag = df["r_input_p"].to_numpy(float); size = df["Re_input_p"].to_numpy(float)
    nmag, nsz = len(MAG_EDGES) - 1, len(SIZE_EDGES) - 1
    mi = np.digitize(mag, MAG_EDGES) - 1; si = np.digitize(size, SIZE_EDGES) - 1
    fin = np.isfinite(lab) & np.isfinite(rbp) & match & (mi >= 0) & (mi < nmag) & (si >= 0) & (si < nsz)
    mi, si, lab, rbp = mi[fin], si[fin], lab[fin], rbp[fin]
    case = df["case"].to_numpy(np.int64)[fin]

    lines = []
    def emit(s): print(s, flush=True); lines.append(s)
    ncase = args.max_case + 1
    rng = np.random.default_rng(7)

    def cell(mm, ss, arr):
        m = np.isin(mi, mm) & np.isin(si, ss)
        if m.sum() < 50:
            return np.nan, np.nan
        v = arr[m].mean()
        # case bootstrap
        cs = case[m]; boot = np.empty(300)
        uc = np.unique(cs)
        for k in range(300):
            pick = rng.choice(uc, len(uc)); sel = np.isin(cs, pick)
            boot[k] = arr[m][sel].mean() if sel.any() else np.nan
        return v, np.nanstd(boot)

    ALLM, ALLS = list(range(nmag)), list(range(nsz))
    emit(f"# R_blend LABEL vs per-pair PRED (crowd g0.05 leg, blended+sheared-nbr, n={len(lab):,})")
    emit(f"# LABEL=<e_snc.ghat_s>/g  PRED=emulator per-pair response  RESID=pred-label")
    emit(f"\n=== (mag x size): R_blend label / pred / resid ===")
    emit("  size bins: " + "  ".join(f"[{SIZE_EDGES[j]:.2f},{SIZE_EDGES[j+1]:.2f})" for j in range(nsz)))
    for a in range(nmag):
        rl = [cell([a], [b], lab)[0] for b in range(nsz)]
        rp = [cell([a], [b], rbp)[0] for b in range(nsz)]
        emit(f"mag[{MAG_EDGES[a]:.0f},{MAG_EDGES[a+1]:.0f}):")
        emit("  label: " + "".join(f"{rl[b]:+8.3f}" for b in range(nsz)))
        emit("  pred : " + "".join(f"{rp[b]:+8.3f}" for b in range(nsz)))
        emit("  resid: " + "".join(f"{rp[b]-rl[b]:+8.3f}" for b in range(nsz)))
    emit(f"\n=== DELIVERABLE WINDOWS: R_blend label / pred / resid ===")
    def mm_of(lo, hi): return [a for a in range(nmag) if MAG_EDGES[a] >= lo-1e-6 and MAG_EDGES[a+1] <= hi+1e-6]
    def ss_of(lo, hi): return [b for b in range(nsz) if SIZE_EDGES[b] >= lo-1e-6 and SIZE_EDGES[b+1] <= hi+1e-6]
    WINS = [("GLOBAL", ALLM, ALLS), ("mag24-25", mm_of(24, 25), ALLS), ("mag25-26", mm_of(25, 26), ALLS),
            ("mag24-26", mm_of(24, 26), ALLS), ("size_gt0.3", ALLM, ss_of(0.3, 1.5)),
            ("mag24-26&sz>0.3", mm_of(24, 26), ss_of(0.3, 1.5))]
    emit(f"  {'window':18s}{'label':>10s}{'pred':>10s}{'resid':>10s}{'err_lab':>9s}")
    for name, mm, ss in WINS:
        if not mm or not ss:
            continue
        vl, el = cell(mm, ss, lab); vp, _ = cell(mm, ss, rbp)
        emit(f"  {name:18s}{vl:+10.4f}{vp:+10.4f}{vp-vl:+10.4f}{el:>9.4f}")
    emit("\nRESID>0 = emulator per-pair OVER-predicts the true single-neighbour blend response.")
    # per-cell sums for consistent downstream marginalisation (mag-resolved combine)
    Slab = np.zeros((nmag, nsz)); Sprd = np.zeros((nmag, nsz)); Ncell = np.zeros((nmag, nsz), np.int64)
    np.add.at(Slab, (mi, si), lab); np.add.at(Sprd, (mi, si), rbp); np.add.at(Ncell, (mi, si), 1)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez(OUT.replace(".txt", ".npz"), Slab=Slab, Sprd=Sprd, Ncell=Ncell,
             MAG_EDGES=MAG_EDGES, SIZE_EDGES=SIZE_EDGES, G=G)
    open(OUT, "w").write("\n".join(lines) + "\n")
    emit(f"\nwrote {OUT}\nRBLEND_PAIRPRED_DONE")


if __name__ == "__main__":
    main()
