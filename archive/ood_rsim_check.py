"""FLOW-FREE cross-check of mechanism B: does the coherent-blend deficit concentrate on galaxies
with a BRIGHT out-of-domain neighbour (which the emulator scores as 0)?

Superposition => the self-response R_self is context-independent, so per OOD bin:
    implied_blend = <R_sim> - R_flow      (R_flow ~ const across bins, approx)
    deficit       = implied_blend - <R_blend_emulator>
If the deficit is large in the BRIGHT-ood bins and ~0 in the FAINT-ood bins (faint neighbours
carry ~0 true blend, so a faint-ood cut is a crowding-SELECTION control), mechanism B is confirmed
with NO flow / NO GPU. The target r-mag per bin is printed to expose the property confound.
"""
import argparse, os, sys
import numpy as np
import pandas as pd
import pyarrow.feather as pf

sys.path.insert(0, "/home/z/Zekang.Zhang/SBSI")
from scripts.validate_constant_with_blend import load  # noqa

RFLOW_GLOBAL = 0.297  # global flow self-response (crowdflux), for the implied-blend reference


def binstats(name, ov, r_sim, rb, rmag, n_neigh):
    pos = ov > 1e-6
    print(f"\n--- by {name} (has-ood frac={np.mean(pos):.3f}) ---")
    print(f"{'bin':>10} {'<R_sim>':>8} {'<R_bl>':>7} {'impl_bl':>8} {'deficit':>8} {'<r_p>':>6} {'<Nnb>':>6} {'N':>9}")
    if pos.sum() > 5000:
        qe = np.quantile(ov[pos], np.linspace(0, 1, 4)); qe[0] -= 1e-9; qe[-1] += 1e-9
        b = np.where(~pos, 0, 1 + np.clip(np.digitize(ov, qe) - 1, 0, 2))
    else:
        b = np.zeros(len(ov), int)
    for c in range(4):
        m = b == c
        if m.sum() < 3000:
            continue
        Rs = r_sim[m].mean(); Rb = rb[m].mean(); impl = Rs - RFLOW_GLOBAL
        tag = "no-ood" if c == 0 else f"ood q{c}"
        print(f"{tag:>10} {Rs:>8.4f} {Rb:>7.4f} {impl:>8.4f} {impl - Rb:>+8.4f} {rmag[m].mean():>6.2f} "
              f"{n_neigh[m].mean():>6.2f} {int(m.sum()):>9,}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalogue", default="/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/constant_response_catalogue_train.feather")
    ap.add_argument("--blend-lookup", default="results/blend_lookup_const28_c0-39.feather")
    ap.add_argument("--ood-lookup", default="results/ood_split_c0-39.feather")
    ap.add_argument("--max-rows", type=int, default=12000000)
    args = ap.parse_args()

    df = load(args.catalogue, args.max_rows)
    g = float(np.median(np.hypot(df["applied_g1"], df["applied_g2"])))
    gh1 = df["applied_g1"].to_numpy(float) / np.hypot(df["applied_g1"], df["applied_g2"])
    gh2 = df["applied_g2"].to_numpy(float) / np.hypot(df["applied_g1"], df["applied_g2"])
    e1p, e2p = df["measured_e1_plus"].to_numpy(float), df["measured_e2_plus"].to_numpy(float)
    e1m, e2m = df["measured_e1_minus"].to_numpy(float), df["measured_e2_minus"].to_numpy(float)
    r_sim = ((e1p - e1m) * gh1 + (e2p - e2m) * gh2) / (2 * g)

    rbdf = pf.read_table(args.blend_lookup).to_pandas()[["case", "input_index", "R_blend"]]
    rb = df[["case", "input_index"]].merge(rbdf, on=["case", "input_index"], how="left")["R_blend"].fillna(0.0).to_numpy(float)
    od = pf.read_table(args.ood_lookup).to_pandas()
    om = df[["case", "input_index"]].merge(od, on=["case", "input_index"], how="left")
    rmag = df["r_input_p"].to_numpy(float)
    n_neigh = df["neighbored"].astype(float).to_numpy() if "neighbored" in df.columns else np.zeros(len(df))

    print(f"N={len(df):,}  |g|={g:.4f}  global <R_sim>={r_sim.mean():.4f}  <R_blend_emu>={rb.mean():.4f}  "
          f"implied global blend={r_sim.mean()-RFLOW_GLOBAL:.4f} (emu {rb.mean():.4f}, deficit {r_sim.mean()-RFLOW_GLOBAL-rb.mean():+.4f})")
    for col in ("ood_flux_bright", "ood_flux_faint"):
        if col in om.columns:
            binstats(col, om[col].fillna(0.0).to_numpy(float), r_sim, rb, rmag, n_neigh)


if __name__ == "__main__":
    main()
