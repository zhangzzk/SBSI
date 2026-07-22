#!/usr/bin/env python
"""cont.109 de-risk before any retrain: is the flow's R_self size-transition smoothing a FLOW
representational limit (-> finer target / capacity could help) or the TRUE<->MEASURED conditioning
gap (the flow conditions on measured_flux_radius (noisy); the transition is sharp in TRUE size Re, so
binning the response in true-size space convolves it with the true->measured scatter -> unavoidable
smoothing that finer targets CANNOT fix)?

Per object on the g0.05 half-shear leg (SAME rows): LABEL R_self=<e_snc.ghat_p>/g (true response) and
PRED R_model (flow induced self-response).  Bin BOTH by TRUE size (Re_input_p) and by MEASURED size
(measured_flux_radius).  If pred matches label SHARPLY in MEASURED-size bins but is smoothed in
TRUE-size bins -> it's the conditioning gap (fundamental); if pred is smoothed even in measured-size
bins -> a real flow limit (retrain-addressable).  Firewall: half-shear only, constgold never read.
"""
import sys, os, time, argparse, numpy as np, pandas as pd, pyarrow as pa, pyarrow.ipc as ipc, pyarrow.feather as pf
SBSI = "/home/z/Zekang.Zhang/SBSI"; sys.path.insert(0, SBSI)
import torch  # noqa
from sbs_shear.measurement_model import load_measurement_model, raw_columns_for_measurement_targets  # noqa
from sbs_shear.preprocessing import (DEFAULT_SELECTION_CUTS, raw_columns_for_selection_features,  # noqa
                                     source_select_selection)
from scripts.eval_self_response_bins import proj_perobj, _seed_flow  # noqa

CAT = "/project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_crowd_g0.05_val_full.feather"
CROWDLK = "results/crowd_flux_conc_c0-199.feather"
SNC = "results/g0_lookup_c0-99.feather"
OUT = "/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/selfresp_true_vs_measured_size.txt"
G = 0.05; KEYMUL = 1_000_003
t0 = time.time()
def log(*a): print(f"[{time.time()-t0:6.1f}s]", *a, flush=True)


def load_pop(bundle, max_rows, max_case, seed):
    rng = np.random.default_rng(seed)
    cond = bundle.condition_preprocessor.feature_names
    tgt = bundle.target_transform.target_names
    with ipc.open_file(CAT) as reader:
        avail = set(reader.schema.names); needed = set()
        needed |= raw_columns_for_selection_features(cond, available_columns=avail)
        needed |= raw_columns_for_measurement_targets(tgt)
        needed |= {"detected", "case", "input_index", "e1_input_rot0_p", "e2_input_rot0_p",
                   "gamma1_input_p", "gamma2_input_p", "gamma1_input_s", "gamma2_input_s",
                   "r_input_p", "Re_input_p", "measured_flux_radius", "measured_mag_auto",
                   "measured_ngmix_g1", "measured_ngmix_g2", "neighbored"}
        needed.discard("nbr_flux_max")
        cols = sorted(c for c in needed if c in avail)
        reservoir = None; raw = 0
        for bi in range(reader.num_record_batches):
            b = pa.Table.from_batches([reader.get_batch(bi)]).select(cols).to_pandas()
            raw += len(b)
            b = b[b["case"] <= max_case]
            if len(b) == 0:
                continue
            b = source_select_selection(b, cuts=DEFAULT_SELECTION_CUTS)
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
    clk = pf.read_table(CROWDLK, columns=["case", "input_index", "nbr_flux_max"]).to_pandas()
    reservoir = reservoir.merge(clk, on=["case", "input_index"], how="left")
    reservoir["nbr_flux_max"] = reservoir["nbr_flux_max"].fillna(0.0)
    log(f"raw={raw:,} kept={len(reservoir):,}")
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
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"device={device}")
    bundle = load_measurement_model(args.measurement_model, device=device)
    df = load_pop(bundle, args.max_rows, args.max_case, 7)

    # PRED: flow induced self-response (gamma zeroed, perturb intrinsic rot0 shape)
    dpred = df.copy(); dpred["gamma1_input_p"] = 0.0; dpred["gamma2_input_p"] = 0.0
    intr = (dpred["e1_input_rot0_p"].to_numpy(float).copy(), dpred["e2_input_rot0_p"].to_numpy(float).copy())
    rk = dict(pixel_rms=0.312, pixel_size=0.2, zero_mag=30.0, psf_fwhm=0.73, moffat_beta=2.224)
    N = len(df); gh1 = np.ones(N); gh0 = np.zeros(N); d = args.delta
    _seed_flow(args.flow_seed); pp1 = proj_perobj(bundle, dpred, +d, gh1, gh0, intr, rk, args.n_samples, args.batch_size)
    _seed_flow(args.flow_seed); pm1 = proj_perobj(bundle, dpred, -d, gh1, gh0, intr, rk, args.n_samples, args.batch_size)
    _seed_flow(args.flow_seed); pp2 = proj_perobj(bundle, dpred, +d, gh0, gh1, intr, rk, args.n_samples, args.batch_size)
    _seed_flow(args.flow_seed); pm2 = proj_perobj(bundle, dpred, -d, gh0, gh1, intr, rk, args.n_samples, args.batch_size)
    pred = 0.5 * ((pp1 - pm1) + (pp2 - pm2)) / (2 * d)
    log(f"<R_model>={np.nanmean(pred):.4f}")

    # LABEL: SNC-cancelled measured shape projected onto ghat_p (true response, same rows)
    lk = pf.read_table(SNC, columns=["case", "input_index", "ngmix0_g1", "ngmix0_g2"]).to_pandas()
    lkey = lk["case"].to_numpy(np.int64) * KEYMUL + lk["input_index"].to_numpy(np.int64)
    order = np.argsort(lkey); lkey = lkey[order]
    e0_1 = lk["ngmix0_g1"].to_numpy(float)[order]; e0_2 = lk["ngmix0_g2"].to_numpy(float)[order]
    e1 = df["measured_ngmix_g1"].to_numpy(float); e2 = df["measured_ngmix_g2"].to_numpy(float)
    key = df["case"].to_numpy(np.int64) * KEYMUL + df["input_index"].to_numpy(np.int64)
    pos = np.clip(np.searchsorted(lkey, key), 0, len(lkey) - 1); match = lkey[pos] == key
    e1 = e1 - np.where(match, e0_1[pos], np.nan); e2 = e2 - np.where(match, e0_2[pos], np.nan)
    g1p = df["gamma1_input_p"].to_numpy(float); g2p = df["gamma2_input_p"].to_numpy(float)
    gp = np.hypot(g1p, g2p)
    lab = (g1p * e1 + g2p * e2) / np.where(gp > 1e-6, gp, 1.0) / G

    truesz = df["Re_input_p"].to_numpy(float)
    meassz = df["measured_flux_radius"].to_numpy(float)
    mag = df["r_input_p"].to_numpy(float)
    fin = np.isfinite(lab) & np.isfinite(pred) & match & np.isfinite(truesz) & np.isfinite(meassz)

    lines = []
    def emit(s): print(s, flush=True); lines.append(s)

    def table(sizevar, name, sel):
        m = fin & sel
        sv = sizevar[m]; la = lab[m]; pr = pred[m]
        edges = np.quantile(sv, np.linspace(0, 1, 13))
        edges = np.unique(edges)
        bi = np.clip(np.digitize(sv, edges[1:-1]), 0, len(edges) - 2)
        emit(f"\n--- R_self by {name} (12 quantile bins), sel={sel.sum() if sel is not True else 'all'} ---")
        emit(f"  {'bin':>3}{'<'+name+'>':>12}{'n':>10}{'label':>9}{'pred':>9}{'resid':>9}")
        for b in range(len(edges) - 1):
            mm = bi == b
            if mm.sum() < 200:
                continue
            emit(f"  {b:>3}{sv[mm].mean():>12.3f}{mm.sum():>10,}{la[mm].mean():>+9.3f}"
                 f"{pr[mm].mean():>+9.3f}{(pr[mm]-la[mm]).mean():>+9.3f}")

    emit("# R_self LABEL vs flow PRED binned by TRUE size vs MEASURED size (half-shear, same rows)")
    emit("# If resid is SHARP-then-flat in MEASURED bins but wavy in TRUE bins -> true<->measured gap.")
    for magcut, lab_c in [(None, "ALL mag"), (24.0, "mag<24 (sharpest transition)"), (25.0, "mag<25")]:
        sel = np.ones(len(lab), bool) if magcut is None else (mag < magcut)
        emit(f"\n================= {lab_c} =================")
        table(truesz, "true_Re", sel)
        table(meassz, "meas_flux_radius", sel)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(lines) + "\n")
    emit(f"\nwrote {OUT}\nTRUE_VS_MEAS_DONE")


if __name__ == "__main__":
    main()
