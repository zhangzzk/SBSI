#!/usr/bin/env python
"""Owner-directed CLEAN DIAGNOSE (cont.108): resolve the two response COMPONENTS separately in
their OWN training data (the half-shear crowd g0.05 leg -- the exact leg the certified R_self snc
target was built from), as a function of properties, under the deliverable cuts.

R_total = R_self + R_blend (linear).  On the half-shear leg the PRIMARY is sheared at |g_p|=0.05
(per-case dir) and its tagged SECONDARY independently at |g_s|=0.05 in a DECORRELATED random dir
(<ghat_p.ghat_s>~=0).  So projecting the (SNC-cancelled) measured shape onto each direction
ISOLATES one component in expectation:

    R_self  = <e_snc . ghat_p>/|g_p|                       (== the certified snc R_self target)
    R_blend = <e_snc . ghat_s>/|g_s|   over blended (g_s>0) (response to the tagged neighbour's shear)

e_snc = measured_ngmix - ngmix0  (per (case,input_index) g=0 shape, from the certified snc lookup)
cancels intrinsic shape -> low-noise labels.  Per-(case,target) 1/n_pairs weighting mirrors the
target builder exactly.  This is the GROUND-TRUTH decomposition the R_total model must match; it is
firewall-clean (training-data truth, constgold NEVER read).

NOTE the R_blend LABEL here is the SINGLE tagged-neighbour response; the certified emulator R_blend
SUMS all neighbours -> a multiplicity mismatch that must be handled before comparing R_blend pred vs
label (flagged, Stage 2).  R_self has no such issue and is the dominant term (~7x R_blend).

Outputs a per-(mag x size) map + deliverable-window aggregates + marginals, with case-cluster
bootstrap errors.  CPU streaming, memory-safe (per-case-per-cell accumulation).
"""
import sys, time, argparse, numpy as np, pyarrow as pa, pyarrow.ipc as ipc, pyarrow.feather as pf
sys.path.insert(0, "/home/z/Zekang.Zhang/SBSI")
from sbs_shear.preprocessing import DEFAULT_SELECTION_CUTS, source_select_selection

CAT = "/project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_crowd_g0.05_val_full.feather"
SNC = "results/g0_lookup_c0-99.feather"
OUT = "/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/halfshear_component_labels.txt"
G = 0.05
KEYMUL = 1_000_003
MAG_EDGES = np.array([18., 24., 25., 26., 28.])
SIZE_EDGES = np.array([0.1, 0.24, 0.30, 0.38, 0.50, 1.50])
t0 = time.time()
def log(*a): print(f"[{time.time()-t0:6.1f}s]", *a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-case", type=int, default=99)
    ap.add_argument("--max-rows", type=int, default=0)  # 0 = all
    args = ap.parse_args()

    # SNC g=0 lookup (per case,input_index measured shape) -> cancels intrinsic shape
    lk = pf.read_table(SNC, columns=["case", "input_index", "ngmix0_g1", "ngmix0_g2"]).to_pandas()
    lkey = lk["case"].to_numpy(np.int64) * KEYMUL + lk["input_index"].to_numpy(np.int64)
    order = np.argsort(lkey)
    lkey = lkey[order]; e0_1 = lk["ngmix0_g1"].to_numpy(float)[order]; e0_2 = lk["ngmix0_g2"].to_numpy(float)[order]
    log(f"SNC lookup rows={len(lkey):,}")

    nmag, nsz, ncase = len(MAG_EDGES) - 1, len(SIZE_EDGES) - 1, args.max_case + 1
    # per (case, magbin, sizebin) weighted sums.  proj already = e_snc.ghat ; R=<proj>/g
    S_self = np.zeros((ncase, nmag, nsz)); W_self = np.zeros((ncase, nmag, nsz))
    S_bl = np.zeros((ncase, nmag, nsz)); W_bl = np.zeros((ncase, nmag, nsz))
    RAW = np.zeros((nmag, nsz), dtype=np.int64)
    COLS = ["detected", "case", "input_index", "r_input_p", "Re_input_p", "distance", "neighbored",
            "gamma1_input_p", "gamma2_input_p", "gamma1_input_s", "gamma2_input_s",
            "measured_ngmix_g1", "measured_ngmix_g2"]
    nread = 0
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
            e1 = b["measured_ngmix_g1"].to_numpy(float); e2 = b["measured_ngmix_g2"].to_numpy(float)
            case = b["case"].to_numpy(np.int64); ii = b["input_index"].to_numpy(np.int64)
            key = case * KEYMUL + ii
            pos = np.clip(np.searchsorted(lkey, key), 0, len(lkey) - 1)
            match = lkey[pos] == key
            e1 = e1 - np.where(match, e0_1[pos], np.nan)
            e2 = e2 - np.where(match, e0_2[pos], np.nan)
            # per-(case,target) 1/n_pairs weight (target builder convention)
            _, inv, npairs = np.unique(key, return_inverse=True, return_counts=True)
            w = (1.0 / npairs[inv])
            g1p = b["gamma1_input_p"].to_numpy(float); g2p = b["gamma2_input_p"].to_numpy(float)
            g1s = b["gamma1_input_s"].to_numpy(float); g2s = b["gamma2_input_s"].to_numpy(float)
            gp = np.hypot(g1p, g2p); gs = np.hypot(g1s, g2s)
            proj_p = (g1p * e1 + g2p * e2) / np.where(gp > 1e-6, gp, 1.0)   # e_snc . ghat_p
            proj_s = (g1s * e1 + g2s * e2) / np.where(gs > 1e-6, gs, 1.0)   # e_snc . ghat_s
            mag = b["r_input_p"].to_numpy(float); size = b["Re_input_p"].to_numpy(float)
            mi = np.digitize(mag, MAG_EDGES) - 1
            si = np.digitize(size, SIZE_EDGES) - 1
            incell = (mi >= 0) & (mi < nmag) & (si >= 0) & (si < nsz)
            has_p = np.isfinite(proj_p) & (gp > 1e-6) & match
            has_s = np.isfinite(proj_s) & (gs > 1e-6) & b["neighbored"].to_numpy().astype(bool) & match
            # accumulate
            mp = incell & has_p
            np.add.at(S_self, (case[mp], mi[mp], si[mp]), (proj_p[mp] * w[mp]))
            np.add.at(W_self, (case[mp], mi[mp], si[mp]), w[mp])
            np.add.at(RAW, (mi[mp], si[mp]), 1)
            ms = incell & has_s
            np.add.at(S_bl, (case[ms], mi[ms], si[ms]), (proj_s[ms] * w[ms]))
            np.add.at(W_bl, (case[ms], mi[ms], si[ms]), w[ms])
            if bi % 40 == 0:
                log(f"batch {bi}/{r.num_record_batches} read={nread:,}")
    log(f"done streaming; total read={nread:,}")

    # ---- estimators with case-cluster bootstrap ----
    rng = np.random.default_rng(12345)
    NB = 500
    cases = np.arange(ncase)

    def R_cell(S, W, magsel, szsel):
        # sum over selected cells, per case
        s = S[:, magsel][:, :, szsel].sum(axis=(1, 2))   # (ncase,)
        w = W[:, magsel][:, :, szsel].sum(axis=(1, 2))
        wt = w.sum()
        if wt <= 0:
            return np.nan, np.nan, 0.0
        val = s.sum() / wt / G
        boot = np.empty(NB)
        for k in range(NB):
            idx = rng.integers(0, ncase, ncase)
            ws = w[idx].sum()
            boot[k] = (s[idx].sum() / ws / G) if ws > 0 else np.nan
        return val, np.nanstd(boot), wt

    lines = []
    def emit(s): print(s, flush=True); lines.append(s)

    emit(f"# half-shear component labels  (crowd g0.05 leg, SNC, cases<= {args.max_case})  g={G}")
    emit(f"# R_self=<e_snc.ghat_p>/g  R_blend=<e_snc.ghat_s>/g(blended)  R_total=R_self+R_blend")
    allmag = np.arange(nmag); allsz = np.arange(nsz)
    rs, ers, _ = R_cell(S_self, W_self, allmag, allsz)
    rb, erb, _ = R_cell(S_bl, W_bl, allmag, allsz)
    emit(f"\nGLOBAL: R_self={rs:+.4f}+/-{ers:.4f}  R_blend={rb:+.4f}+/-{erb:.4f}  "
         f"R_total={rs+rb:+.4f}   (cross-check R_self vs snc target global 0.2812)")

    emit(f"\n=== (mag x size) MAP: R_self / R_blend / R_total ===")
    emit("  size bins: " + "  ".join(f"[{SIZE_EDGES[j]:.2f},{SIZE_EDGES[j+1]:.2f})" for j in range(nsz)))
    for a in range(nmag):
        emit(f"mag[{MAG_EDGES[a]:.0f},{MAG_EDGES[a+1]:.0f}):")
        r_self = [R_cell(S_self, W_self, [a], [b]) for b in range(nsz)]
        r_bl = [R_cell(S_bl, W_bl, [a], [b]) for b in range(nsz)]
        emit("  R_self : " + "".join(f"{r_self[b][0]:+8.3f}({r_self[b][1]:.3f})" for b in range(nsz)))
        emit("  R_blend: " + "".join(f"{r_bl[b][0]:+8.3f}({r_bl[b][1]:.3f})" for b in range(nsz)))
        emit("  R_total: " + "".join(f"{r_self[b][0]+r_bl[b][0]:+8.3f}        " for b in range(nsz)))

    emit(f"\n=== MARGINALS ===")
    emit("  by mag:")
    for a in range(nmag):
        rs, ers, w = R_cell(S_self, W_self, [a], allsz)
        rb, erb, _ = R_cell(S_bl, W_bl, [a], allsz)
        emit(f"    mag[{MAG_EDGES[a]:.0f},{MAG_EDGES[a+1]:.0f}): R_self={rs:+.4f}+/-{ers:.4f}  "
             f"R_blend={rb:+.4f}+/-{erb:.4f}  R_total={rs+rb:+.4f}  (Neff~{w:,.0f})")
    emit("  by size:")
    for b in range(nsz):
        rs, ers, _ = R_cell(S_self, W_self, allmag, [b])
        rb, erb, _ = R_cell(S_bl, W_bl, allmag, [b])
        emit(f"    Re[{SIZE_EDGES[b]:.2f},{SIZE_EDGES[b+1]:.2f}): R_self={rs:+.4f}+/-{ers:.4f}  "
             f"R_blend={rb:+.4f}+/-{erb:.4f}  R_total={rs+rb:+.4f}")

    emit(f"\n=== DELIVERABLE WINDOWS (mag x size aggregates) ===")
    def magmask(lo, hi): return [a for a in range(nmag) if MAG_EDGES[a] >= lo - 1e-6 and MAG_EDGES[a+1] <= hi + 1e-6]
    def szmask(lo, hi): return [b for b in range(nsz) if SIZE_EDGES[b] >= lo - 1e-6 and SIZE_EDGES[b+1] <= hi + 1e-6]
    ALLM = list(range(nmag)); ALLS = list(range(nsz))
    WINS = [("mag24-25", magmask(24, 25), ALLS), ("mag25-26", magmask(25, 26), ALLS),
            ("mag24-26", magmask(24, 26), ALLS), ("size_gt0.3", ALLM, szmask(0.3, 1.5)),
            ("mag24-26&sz>0.3", magmask(24, 26), szmask(0.3, 1.5))]
    for name, mm, ss in WINS:
        if len(mm) == 0 or len(ss) == 0:
            emit(f"  {name:18s}: (empty)"); continue
        rs, ers, _ = R_cell(S_self, W_self, mm, ss)
        rb, erb, _ = R_cell(S_bl, W_bl, mm, ss)
        emit(f"  {name:18s}: R_self={rs:+.4f}+/-{ers:.4f}  R_blend={rb:+.4f}+/-{erb:.4f}  R_total={rs+rb:+.4f}")

    import os
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    # also stash the per-case-per-cell sums so predictions can be joined on the exact grid later
    np.savez(OUT.replace(".txt", ".npz"), S_self=S_self, W_self=W_self, S_bl=S_bl, W_bl=W_bl,
             RAW=RAW, MAG_EDGES=MAG_EDGES, SIZE_EDGES=SIZE_EDGES, G=G)
    open(OUT, "w").write("\n".join(lines) + "\n")
    emit(f"\nwrote {OUT}\nLABELS_DONE")


if __name__ == "__main__":
    main()
