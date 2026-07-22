#!/usr/bin/env python -B
"""Localize the STABLE large-size (size_gt1.0) negative m bias: missing feature vs sim floor?

m on a phi-selection = R_blend OOS fit error (phi contains size). The size_gt1.0 residual is a
stable -1.4..-2.0% (z~6-9), immune to R_flow grid/seeds and R_blend GB capacity. Two hypotheses:

  (A) phi-COVERAGE: R_blend over-adds blend response to large DOMINANT primaries because phi lacks
      a primary-vs-neighbour dominance descriptor. => the bias lives in BLENDED large galaxies and
      correlates with neighbour geometry; an engineered feature can fix it (a lever remains).
  (B) SIM / R_flow floor: ISOLATED large galaxies are equally biased (no neighbours to mis-model),
      so it's R_flow shape-response error or shape-noise-limited statistics in the rare large tail.
      => no R_blend feature helps; conclude sim-limited for this extreme cut.

Reads the certified dump, swaps in the _prod R_flow + iteration-2 size-aware R_blend (both as the
eval does), joins true size, and reports m = sum(r_sim)/sum(R_flow+R_blend)-1 with leave-one-case
jackknife SEM, split by isolation / distance / neighbour-flux / primary brightness. Diagnostic only.
"""
import numpy as np, pandas as pd
import pyarrow as pa, pyarrow.ipc as ipc, pyarrow.feather as pf

DUMP = "/project/ls-gruen/users/zekang.zhang/sbsi_dumps/fig2_perobj_s501_fixresp.feather"
D = "/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/"
CB = "/project/ls-gruen/users/zekang.zhang/sbsi_caches/"
RFLOW = D + "rflow_joint_prod_ens3_c40-139.npz"
RBLEND = D + "rblend_scene_jointrflow_prod_szphi_c40-139.npz"      # iteration-2 winner
SIZE = CB + "true_size_lookup_c40-139.feather"
CROWD = CB + "crowd_flux_conc_c0-199.feather"


def load_override(df, npz, col):
    z = np.load(npz)
    ov = pd.DataFrame({"case": z["case"].astype(np.int64), "input_index": z["input_index"].astype(np.int64),
                       "_v": z["value"].astype(float)})
    m = df.merge(ov, on=["case", "input_index"], how="left")
    df[col] = np.where(m["_v"].notna(), m["_v"], df[col])
    return df


def m_and_sem(sub):
    """sum-ratio m with leave-one-case jackknife SEM."""
    rs = sub["r_sim"].to_numpy(float); den = (sub["R_flow"] + sub["R_blend"]).to_numpy(float)
    Srs, Sden = rs.sum(), den.sum()
    m = Srs / Sden - 1.0
    cases = sub["case"].to_numpy(np.int64)
    uc = np.unique(cases)
    if uc.size < 3:
        return m, np.nan, len(sub)
    jm = np.empty(uc.size)
    for i, c in enumerate(uc):
        keep = cases != c
        jm[i] = rs[keep].sum() / den[keep].sum() - 1.0
    sem = np.sqrt((uc.size - 1) / uc.size * ((jm - jm.mean()) ** 2).sum())
    return m, sem, len(sub)


def row(tag, sub):
    m, sem, n = m_and_sem(sub)
    z = m / sem if (sem and np.isfinite(sem) and sem > 0) else np.nan
    if n == 0:
        print(f"  {tag:34s} n=         0  (empty)"); return
    rs = sub["r_sim"].mean(); rf = sub["R_flow"].mean(); rb = sub["R_blend"].mean()
    print(f"  {tag:34s} n={n:>10,}  m={m*100:+7.2f}%  sem={sem*100:5.2f}%  z={z:6.1f}  "
          f"|  <r_sim>={rs:+.4f} <R_flow>={rf:+.4f} <R_blend>={rb:+.4f} denom={rf+rb:+.4f}")


def main():
    cols = ["case", "input_index", "r_input_p", "r_sim", "R_flow", "R_blend", "neighbored", "distance"]
    parts = {c: [] for c in cols}
    with pa.memory_map(DUMP, "r") as src:
        r = ipc.open_file(src)
        for i in range(r.num_record_batches):
            b = r.get_batch(i)
            for c in cols:
                parts[c].append(b.column(c).to_numpy(zero_copy_only=False))
    df = pd.DataFrame({c: np.concatenate(parts[c]) for c in cols})
    print(f"dump rows={len(df):,}")
    df = load_override(df, RFLOW, "R_flow")
    df = load_override(df, RBLEND, "R_blend")
    sz = pf.read_table(SIZE, memory_map=True).to_pandas()[["case", "input_index", "Re_input_p"]]
    df = df.merge(sz, on=["case", "input_index"], how="left"); del sz
    cf = pf.read_table(CROWD, memory_map=True).to_pandas()[["case", "input_index", "nbr_flux_max"]]
    df = df.merge(cf, on=["case", "input_index"], how="left"); del cf
    df["nbr_flux_max"] = df["nbr_flux_max"].fillna(0.0)

    for lo in [1.0, 0.5]:
        big = df[df["Re_input_p"] > lo].copy()
        print(f"\n================ size > {lo}  (n={len(big):,}) ================")
        row(f"ALL size>{lo}", big)
        iso = big[(big["neighbored"] == 0) | (~np.isfinite(big["distance"]))]
        bl = big[(big["neighbored"] == 1) & (np.isfinite(big["distance"]))]
        print("  -- by isolation --")
        row("isolated (no neighbour)", iso)
        row("blended (has neighbour)", bl)
        if len(bl) > 0:
            print("  -- blended, by separation --")
            for a, b in [(0, 1.5), (1.5, 3.0), (3.0, 6.0), (6.0, 1e9)]:
                row(f"blended dist[{a},{b})", bl[(bl["distance"] >= a) & (bl["distance"] < b)])
            print("  -- blended, by brightest-neighbour flux --")
            q = bl["nbr_flux_max"].quantile([0.5, 0.9]).to_numpy()
            row(f"blended nbrflux<med({q[0]:.1f})", bl[bl["nbr_flux_max"] < q[0]])
            row(f"blended med..90th", bl[(bl["nbr_flux_max"] >= q[0]) & (bl["nbr_flux_max"] < q[1])])
            row(f"blended >90th({q[1]:.1f})", bl[bl["nbr_flux_max"] >= q[1]])
        print("  -- by primary brightness (r_input_p) --")
        for a, b in [(0, 23), (23, 24), (24, 25), (25, 99)]:
            row(f"r_input_p[{a},{b})", big[(big["r_input_p"] >= a) & (big["r_input_p"] < b)])
    print("\nDIAG_LARGESIZE_DONE")


if __name__ == "__main__":
    main()
