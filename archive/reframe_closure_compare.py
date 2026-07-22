#!/usr/bin/env python
"""
cont.101: per-cut matched closure for the reframed-loss models vs certified vs szfine (single-seed s501).
Each dump carries its OWN r_sim (constgold matched truth, seed-independent), R_blend (emulator, model-
independent) and R_flow (model-specific). Per-cut closure m = <r_sim>/<R_flow+R_blend> - 1. Re joined
from the +0.02 leg for size cuts. Per-cut closure is SEED-STABLE (16-seed run: seedSD 0.5-1%), so a
single seed ranks the loss variants on the gentle cuts; the winner is then ensembled before any verdict.
Firewall: r_sim read for validation only.
"""
import os, numpy as np, pyarrow.feather as feather

DDIR = "/project/ls-gruen/users/zekang.zhang/sbsi_dumps"
LEGP = "/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/constant_shear_catalogue_0.02_train.feather"
OUT = "/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/reframe_closure_compare.txt"
KEYMUL = np.uint64(800003)
# (label, dump filename) — evaluated if present
VARIANTS = [
    ("certified",       "fig2_perobj_s501_fixresp.feather"),
    ("szfine(cnt)",     "fig2_perobj_s501_szfine.feather"),
    ("p075",            "fig2_perobj_s501_szfine_p075.feather"),
    ("sqrt(0.5)",       "fig2_perobj_s501_szfine_sqrtw.feather"),
    ("sqrt+anc2k",      "fig2_perobj_s501_szfine_sqrtw_anc2k.feather"),
    ("sqrt+anc8k",      "fig2_perobj_s501_szfine_sqrtw_anc8k.feather"),
    ("eqw(0)",          "fig2_perobj_s501_szfine_eqw.feather"),
    ("eqw+anc8k",       "fig2_perobj_s501_szfine_eqw_anc8k.feather"),
    ("cnt+cap256",      "fig2_perobj_s501_szfine_cap256.feather"),
    ("sqrt+rel",        "fig2_perobj_s501_szfine_sqrtw_rel.feather"),
]

def load_re(key0):
    tl = feather.read_table(LEGP, columns=["case", "input_index", "Re_input_p"])
    lk = np.asarray(tl["case"]).astype(np.uint64) * KEYMUL + np.asarray(tl["input_index"]).astype(np.uint64)
    lre = np.asarray(tl["Re_input_p"]).astype("f8")
    o = np.argsort(lk); lks, lrs = lk[o], lre[o]
    j = np.clip(np.searchsorted(lks, key0), 0, len(lks) - 1)
    hit = lks[j] == key0
    re = np.full(len(key0), np.nan); re[hit] = lrs[j[hit]]
    return re

def main():
    lines = []
    def emit(s):
        print(s, flush=True); lines.append(s)

    present = [(lab, f) for lab, f in VARIANTS if os.path.exists(f"{DDIR}/{f}")]
    emit(f"variants present: {[l for l,_ in present]}")
    if not present:
        emit("no dumps found yet"); open(OUT, "w").write("\n".join(lines) + "\n"); return

    # base geometry from the first present dump (all share the same constgold object set / order)
    t0 = feather.read_table(f"{DDIR}/{present[0][1]}", columns=["case", "input_index", "r_input_p", "neighbored"])
    key0 = np.asarray(t0["case"]).astype(np.uint64) * KEYMUL + np.asarray(t0["input_index"]).astype(np.uint64)
    mag = np.asarray(t0["r_input_p"]).astype("f8")
    nbr = np.asarray(t0["neighbored"]).astype(bool)
    N = len(key0)
    re = load_re(key0)
    emit(f"N={N:,}  Re join {np.isfinite(re).mean():.1%}\n")

    CUTS = [
        ("GLOBAL",               np.ones(N, bool)),
        ("mag24-25(win)",        (mag >= 24) & (mag < 25)),
        ("mag25-26(win)",        (mag >= 25) & (mag < 26)),
        ("mag24-26(win)",        (mag >= 24) & (mag < 26)),
        ("size_gt0.3",           re > 0.3),
        ("size_gt0.5",           re > 0.5),
        ("mag24-26&Re>0.3",      (mag >= 24) & (mag < 26) & (re > 0.3)),
        ("mag26-27(tail)",       (mag >= 26) & (mag < 27)),
        ("isolated",             ~nbr),
        ("blended",              nbr),
    ]

    # per-variant per-cut closure
    res = {}
    for lab, f in present:
        t = feather.read_table(f"{DDIR}/{f}", columns=["case", "input_index", "r_sim", "R_flow", "R_blend"])
        k = np.asarray(t["case"]).astype(np.uint64) * KEYMUL + np.asarray(t["input_index"]).astype(np.uint64)
        rs = np.asarray(t["r_sim"]).astype("f8"); rf = np.asarray(t["R_flow"]).astype("f8"); rb = np.asarray(t["R_blend"]).astype("f8")
        if len(k) != N or not np.array_equal(k, key0):   # align if row order differs
            o = np.argsort(key0); ks = key0[o]
            idx = np.clip(np.searchsorted(ks, k), 0, N - 1); h = ks[idx] == k
            RS = np.full(N, np.nan); RF = np.full(N, np.nan); RB = np.full(N, np.nan)
            RS[o[idx[h]]] = rs[h]; RF[o[idx[h]]] = rf[h]; RB[o[idx[h]]] = rb[h]
            rs, rf, rb = RS, RF, RB
        good = np.isfinite(rs) & np.isfinite(rf) & np.isfinite(rb)
        mcut = {}
        for clab, cm in CUTS:
            m = good & cm
            den = (rf[m] + rb[m]).sum()
            mcut[clab] = (rs[m].sum() / den - 1) * 100 if den != 0 else np.nan
        res[lab] = mcut
        emit(f"loaded {lab}: <R_flow>={np.nanmean(rf):.4f}")

    labs = [l for l, _ in present]
    emit("")
    hdr = f"{'cut':22s}" + "".join(f"{l:>16s}" for l in labs)
    emit(hdr); emit("-" * len(hdr))
    for clab, _ in CUTS:
        emit(f"{clab:22s}" + "".join(f"{res[l][clab]:>+15.2f}%" for l in labs))
    emit("")
    emit("m% = <r_sim>/<R_flow+R_blend> - 1 (matched, selection-free). SINGLE SEED (s501); per-cut is")
    emit("seed-stable so this RANKS the loss variants, but GLOBAL/small margins need the ensemble.")
    emit("Goal-1 target |m|<0.3% on the gentle cuts (acc, mag_lt24/25, size_gt0.3/0.5).")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(lines) + "\n")
    emit(f"\nwrote {OUT}\nREFRAME_COMPARE_DONE")

if __name__ == "__main__":
    main()
