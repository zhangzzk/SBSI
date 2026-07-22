#!/usr/bin/env python -B
"""Is the size/faint TARGET bias fixable by FINER NON-CIRCULAR supervision, or a sim floor?

The de-risk (cont.84) showed measured cuts dissolve to the true-property TARGET, and TARGET is large
on the SIZE axis + FAINT tail because the certified R_flow is supervised on a COARSE 3-size-bin target
(response_target_*_6x3x5, size edges [0.10,0.24,0.41,1.5]) that smears the steep size response. A FINER
NON-CIRCULAR target already exists (response_target_isoblend_snc_c0-99_6x6x5, snc leg-based, constgold
NEVER read) and resolves a steep non-monotonic profile ([-0.12,-0.19,+0.01,+0.49,+0.54,+0.46]).

Decisive question: does that finer non-circular (leg-based) target REPRODUCE the constgold r_sim
response across size?  If yes -> supervising the flow on it would drive R_flow -> r_sim -> TARGET~0
(the lever exists, fold it into the flow retrain).  If the target UNDER-supplies vs r_sim
(coherent-response deficit) -> a sim/leg floor, sub-percent on size cuts needs new sims.

Method (no new build): map every dump object to its (r_input_p x Re_input_p x blend) bin in the fine
target, look up R_snc, and per FINE size bin compare -- for ISOLATED objects (R_blend~0, the clean
R_flow test) and overall:
    <r_sim>  (constgold TRUTH)  vs  <R_flow>  (certified, coarse-supervised)  vs  <R_snc>  (fine non-circ)
and the implied m: current = <r_sim>/<R_flow(+R_blend)>-1  vs  target = <r_sim>/<R_snc(+R_blend)>-1.
Leave-one-case jackknife SEM. DIAGNOSTIC ONLY (constgold r_sim used as validation truth; nothing fit).
"""
import argparse
import numpy as np, pandas as pd
import pyarrow as pa, pyarrow.ipc as ipc, pyarrow.feather as pf

DUMP = "/project/ls-gruen/users/zekang.zhang/sbsi_dumps/fig2_perobj_s501_fixresp.feather"
CB = "/project/ls-gruen/users/zekang.zhang/sbsi_caches/"
SIZE = CB + "true_size_lookup_c40-139.feather"
TARGET = "/home/z/Zekang.Zhang/SBSI/results/response_target_isoblend_snc_c0-99_6x6x5.npz"


def m_and_sem(rsim, denom, cases):
    """sum-ratio m = sum(r_sim)/sum(denom)-1 with leave-one-case jackknife SEM."""
    good = np.isfinite(rsim) & np.isfinite(denom)
    rsim, denom, cases = rsim[good], denom[good], cases[good]
    if len(rsim) == 0 or denom.sum() == 0:
        return np.nan, np.nan, len(rsim)
    m = rsim.sum() / denom.sum() - 1.0
    uc = np.unique(cases)
    if uc.size < 3:
        return m, np.nan, len(rsim)
    Srs, Sdn = rsim.sum(), denom.sum()
    # fast jackknife: subtract each case's contribution
    cs_r = np.bincount(np.searchsorted(uc, cases), weights=rsim, minlength=uc.size)
    cs_d = np.bincount(np.searchsorted(uc, cases), weights=denom, minlength=uc.size)
    jm = (Srs - cs_r) / (Sdn - cs_d) - 1.0
    sem = np.sqrt((uc.size - 1) / uc.size * ((jm - jm.mean()) ** 2).sum())
    return m, sem, len(rsim)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default=DUMP, help="per-object dump feather (default: certified fixresp)")
    a = ap.parse_args()
    dump_path = a.dump
    print(f"DUMP={dump_path}", flush=True)
    cols = ["case", "input_index", "r_input_p", "r_sim", "R_flow", "R_blend", "neighbored", "distance"]
    parts = {c: [] for c in cols}
    with pa.memory_map(dump_path, "r") as src:
        r = ipc.open_file(src)
        for i in range(r.num_record_batches):
            b = r.get_batch(i)
            for c in cols:
                parts[c].append(b.column(c).to_numpy(zero_copy_only=False))
    df = pd.DataFrame({c: np.concatenate(parts[c]) for c in cols})
    print(f"dump rows={len(df):,}", flush=True)
    sz = pf.read_table(SIZE, memory_map=True).to_pandas()[["case", "input_index", "Re_input_p"]]
    df = df.merge(sz, on=["case", "input_index"], how="left"); del sz
    df = df[np.isfinite(df["Re_input_p"].to_numpy(float))].reset_index(drop=True)
    print(f"with true size: {len(df):,}", flush=True)

    z = np.load(TARGET)
    ef, es, ed = z["edges_flux"], z["edges_size"], z["edges_dist"]   # flux==r_input_p, size, dist
    R = z["Rsim"]; cnt = z["counts"] if "counts" in z else z["cnt"]
    nfl, nsz, nbl = R.shape
    print(f"fine target grid {R.shape}  size edges={np.round(es,4).tolist()}", flush=True)

    flux = df["r_input_p"].to_numpy(float); size = df["Re_input_p"].to_numpy(float)
    dist = df["distance"].to_numpy(float); nb = df["neighbored"].to_numpy()
    fi = np.clip(np.digitize(flux, ef) - 1, 0, nfl - 1)
    si = np.clip(np.digitize(size, es) - 1, 0, nsz - 1)
    iso = (nb == 0) | (~np.isfinite(dist))
    bi = np.where(iso, 0, np.clip(np.digitize(dist, ed) - 1, 0, nbl - 1))   # bin0=isolated, else distance
    Rsnc = R[fi, si, bi]                                                    # per-object fine non-circ target

    rsim = df["r_sim"].to_numpy(float); rflow = df["R_flow"].to_numpy(float); rbl = df["R_blend"].to_numpy(float)
    case = df["case"].to_numpy(np.int64)
    df["_si"] = si

    def block(title, mask):
        print("\n" + "=" * 132); print(title + f"   (n={int(mask.sum()):,})"); print("=" * 132)
        print(f"  {'size bin':16s} {'n':>10s} | {'<r_sim>':>9s} {'<R_flow>':>9s} {'<R_snc>':>9s} {'<R_blend>':>9s} |"
              f" {'m_cur%':>9s} {'sem':>5s} | {'m_tgt%':>9s} {'sem':>5s}  (m_cur=r_sim/(Rflow+Rbl); m_tgt=r_sim/(Rsnc+Rbl))")
        for b in range(nsz):
            sel = mask & (si == b)
            n = int(sel.sum())
            lo, hi = es[b], es[b + 1]
            if n < 200:
                print(f"  [{lo:.3f},{hi:.3f})   n={n:>8,}  (sparse)"); continue
            rs, rf, rn, rbb = rsim[sel], rflow[sel], Rsnc[sel], rbl[sel]
            cs = case[sel]
            mc, sc, _ = m_and_sem(rs, rf + rbb, cs)
            mt, st, _ = m_and_sem(rs, rn + rbb, cs)
            print(f"  [{lo:.3f},{hi:.3f})   {n:>10,} | {np.nanmean(rs):>+9.4f} {np.nanmean(rf):>+9.4f} "
                  f"{np.nanmean(rn):>+9.4f} {np.nanmean(rbb):>+9.4f} | {100*mc:>+8.2f} {100*sc:>4.2f} | "
                  f"{100*mt:>+8.2f} {100*st:>4.2f}")

    # ISOLATED = clean R_flow test (R_blend~0); then bright/faint iso; then ALL
    block("ISOLATED objects (R_blend~0 => m ~ r_sim/R_flow-1; the clean R_flow size test)", iso)
    block("ISOLATED & BRIGHT (r_input_p < 23)", iso & (flux < 23))
    block("ISOLATED & FAINT (r_input_p >= 25)", iso & (flux >= 25))
    block("ALL objects (denominator R_flow+R_blend, emulator R_blend)", np.ones(len(df), bool))

    print("\nREADING: m_tgt ~ 0 where m_cur is large  => finer NON-CIRCULAR target reproduces constgold r_sim")
    print("         => size TARGET is FIXABLE by better (non-circular) supervision -> fold into the flow retrain.")
    print("         m_tgt still large (R_snc < r_sim, under-supply) => leg/sim FLOOR -> size cuts need new sims.")
    print("\nDIAG_SIZE_RESPONSE_DONE")


if __name__ == "__main__":
    main()
