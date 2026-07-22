#!/usr/bin/env python
"""cont.108 PREDICTION side of the clean diagnose: the certified flow's induced SELF-response
R_model (== R_flow / R_self prediction) on the SAME crowd leg + SAME (mag x size) grid as the
half-shear LABEL (halfshear_component_labels.py).  Differencing pred - label per cell gives the
flow's R_self non-closure to its OWN training target, in property space, under the deliverable cuts.

R_model uses the validated certified forward path (proj_perobj / bundle.sample, CRN-seeded central
secant, trace/2) reused verbatim from eval_self_response_bins.py -> reproduces the certified R_flow.
Population = crowd g0.05 leg cases<=99, detected+selected, intrinsic shape rot0 (shear-independent),
gamma zeroed (R_model applies its own +/-delta).  Firewall: no constgold, no |m| fit.
"""
import sys, os, time, argparse, numpy as np, pandas as pd, pyarrow as pa, pyarrow.ipc as ipc, pyarrow.feather as pf
SBSI = "/home/z/Zekang.Zhang/SBSI"
sys.path.insert(0, SBSI)
import torch  # noqa: E402
from sbs_shear.measurement_model import load_measurement_model, raw_columns_for_measurement_targets  # noqa: E402
from sbs_shear.preprocessing import (DEFAULT_SELECTION_CUTS, raw_columns_for_selection_features,  # noqa: E402
                                     rescale, source_select_selection)
from scripts.eval_self_response_bins import proj_perobj, _seed_flow  # reuse validated forward path

CAT = "/project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_crowd_g0.05_val_full.feather"
CROWDLK = "results/crowd_flux_conc_c0-199.feather"   # supplies nbr_flux_max (not native in the leg)
OUT = "/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/flow_selfresp_grid.npz"
MAG_EDGES = np.array([18., 24., 25., 26., 28.])
SIZE_EDGES = np.array([0.1, 0.24, 0.30, 0.38, 0.50, 1.50])
t0 = time.time()
def log(*a): print(f"[{time.time()-t0:6.1f}s]", *a, flush=True)


def load_pop(bundle, max_rows, max_case, seed):
    rng = np.random.default_rng(seed)
    cond = bundle.condition_preprocessor.feature_names
    tgt = bundle.target_transform.target_names
    with ipc.open_file(CAT) as reader:
        avail = set(reader.schema.names)
        needed = set()
        needed |= raw_columns_for_selection_features(cond, available_columns=avail)
        needed |= raw_columns_for_measurement_targets(tgt)
        needed |= {"detected", "case", "input_index", "e1_input_rot0_p", "e2_input_rot0_p",
                   "gamma1_input_p", "gamma2_input_p", "r_input_p", "Re_input_p"}
        needed.discard("nbr_flux_max")  # not in leg; merged from CROWDLK below
        cols = sorted(c for c in needed if c in avail)
        reservoir = None; raw = 0
        for bi in range(reader.num_record_batches):
            b = pa.Table.from_batches([reader.get_batch(bi)]).select(cols).to_pandas()
            raw += len(b)
            b = b[b["case"] <= max_case]
            if len(b) == 0:
                continue
            b = source_select_selection(b, cuts=DEFAULT_SELECTION_CUTS)
            if len(b) == 0:
                continue
            b = b[b["detected"].astype(bool)].reset_index(drop=True)
            if len(b) == 0:
                continue
            b = b.copy(); b["__key"] = rng.random(len(b))
            reservoir = b if reservoir is None else pd.concat([reservoir, b], ignore_index=True)
            if len(reservoir) > 2 * max_rows:
                reservoir = reservoir.nlargest(max_rows, "__key").reset_index(drop=True)
    if len(reservoir) > max_rows:
        reservoir = reservoir.nlargest(max_rows, "__key").reset_index(drop=True)
    reservoir = reservoir.drop(columns="__key").reset_index(drop=True)
    reservoir["gamma1_input_p"] = 0.0
    reservoir["gamma2_input_p"] = 0.0
    if "nbr_flux_max" not in reservoir.columns:
        clk = pf.read_table(CROWDLK, columns=["case", "input_index", "nbr_flux_max"]).to_pandas()
        reservoir = reservoir.merge(clk, on=["case", "input_index"], how="left")
        miss = reservoir["nbr_flux_max"].isna().mean()
        reservoir["nbr_flux_max"] = reservoir["nbr_flux_max"].fillna(0.0)
        log(f"merged nbr_flux_max from {CROWDLK}: missing={miss:.2%} (filled 0)")
    log(f"raw scanned={raw:,}  kept(detected+sel, case<= {max_case})={len(reservoir):,}")
    return reservoir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--measurement-model",
                    default="models/measurement_flow_g0_ngmix_meas_szfl_noz_lam450_fixresp_s501.pt")
    ap.add_argument("--max-rows", type=int, default=2_000_000)
    ap.add_argument("--max-case", type=int, default=99)
    ap.add_argument("--n-samples", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=65536)
    ap.add_argument("--delta", type=float, default=0.02)
    ap.add_argument("--flow-seed", type=int, default=12345)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"device={device}  model={os.path.basename(args.measurement_model)}")
    bundle = load_measurement_model(args.measurement_model, device=device)

    df = load_pop(bundle, args.max_rows, args.max_case, args.seed)
    rk = dict(pixel_rms=0.312, pixel_size=0.2, zero_mag=30.0, psf_fwhm=0.73, moffat_beta=2.224)
    intr = (df["e1_input_rot0_p"].to_numpy(float).copy(), df["e2_input_rot0_p"].to_numpy(float).copy())
    N = len(df); gh1 = np.ones(N); gh0 = np.zeros(N); d = args.delta
    _seed_flow(args.flow_seed); pp1 = proj_perobj(bundle, df, +d, gh1, gh0, intr, rk, args.n_samples, args.batch_size)
    _seed_flow(args.flow_seed); pm1 = proj_perobj(bundle, df, -d, gh1, gh0, intr, rk, args.n_samples, args.batch_size)
    _seed_flow(args.flow_seed); pp2 = proj_perobj(bundle, df, +d, gh0, gh1, intr, rk, args.n_samples, args.batch_size)
    _seed_flow(args.flow_seed); pm2 = proj_perobj(bundle, df, -d, gh0, gh1, intr, rk, args.n_samples, args.batch_size)
    rmodel = 0.5 * ((pp1 - pm1) + (pp2 - pm2)) / (2 * d)
    log(f"<R_model> global = {np.mean(rmodel):.4f}  (certified R_flow ~0.293)")

    mag = df["r_input_p"].to_numpy(float); size = df["Re_input_p"].to_numpy(float)
    fin = np.isfinite(mag) & np.isfinite(size) & np.isfinite(rmodel)
    mag, size, rm = mag[fin], size[fin], rmodel[fin]
    nmag, nsz = len(MAG_EDGES) - 1, len(SIZE_EDGES) - 1
    mi = np.digitize(mag, MAG_EDGES) - 1; si = np.digitize(size, SIZE_EDGES) - 1
    inc = (mi >= 0) & (mi < nmag) & (si >= 0) & (si < nsz)
    mi, si, rm = mi[inc], si[inc], rm[inc]
    Rgrid = np.full((nmag, nsz), np.nan); Ngrid = np.zeros((nmag, nsz), np.int64)
    for a in range(nmag):
        for b in range(nsz):
            m = (mi == a) & (si == b)
            Ngrid[a, b] = int(m.sum())
            if m.any():
                Rgrid[a, b] = float(np.mean(rm[m]))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez(OUT, Rgrid=Rgrid, Ngrid=Ngrid, MAG_EDGES=MAG_EDGES, SIZE_EDGES=SIZE_EDGES,
             global_R=float(np.mean(rmodel)), model=os.path.basename(args.measurement_model))
    log(f"wrote {OUT}")
    print("R_self_pred grid (mag x size):")
    for a in range(nmag):
        print(f"  mag[{MAG_EDGES[a]:.0f},{MAG_EDGES[a+1]:.0f}): " +
              "".join(f"{Rgrid[a,b]:+8.3f}" for b in range(nsz)))
    print("FLOW_GRID_DONE", flush=True)


if __name__ == "__main__":
    main()
