#!/usr/bin/env python
"""cont.110g: re-bin the R_self LABEL and the flow R_self PREDICTION by MEASURED mag/size,
side-by-side with the TRUE-binned version, on the SAME 2M half-shear objects.

The flow conditions on measured_mag_auto + measured_flux_radius (confirmed).  So its R_self
prediction is a function of MEASURED size/mag; the prior grid binned it by TRUE size -> smearing.
This test bins BOTH the physical label (<e_snc.ghat_p>/g) and the flow pred by measured props:

  * if pred MATCHES label in MEASURED bins (resid ~0) while it missed in TRUE bins
    -> the flow is a well-calibrated MEASURED-conditioned estimator; the true-cut miss is the
       irreducible true<->measured mismatch (errors-in-variables), not model capacity.
  * if pred STILL misses in MEASURED bins -> genuine capacity / population miscalibration.

Firewall: half-shear crowd g0.05 leg only; constgold NEVER read; no |m| fit.  Reuses the certified
forward path (proj_perobj / _seed_flow) verbatim -> reproduces the certified R_flow.
"""
import sys, os, time, argparse, numpy as np, pandas as pd, pyarrow as pa, pyarrow.ipc as ipc, pyarrow.feather as pf
SBSI = "/home/z/Zekang.Zhang/SBSI"
sys.path.insert(0, SBSI)
import torch  # noqa: E402
from sbs_shear.measurement_model import load_measurement_model, raw_columns_for_measurement_targets  # noqa: E402
from sbs_shear.preprocessing import (DEFAULT_SELECTION_CUTS, raw_columns_for_selection_features,  # noqa: E402
                                     source_select_selection)
from scripts.eval_self_response_bins import proj_perobj, _seed_flow  # certified forward path

CAT = "/project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_crowd_g0.05_val_full.feather"
SNC = "results/g0_lookup_c0-99.feather"
CROWDLK = "results/crowd_flux_conc_c0-199.feather"
OUT = "/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/component_measbinned.npz"
MAG_EDGES = np.array([18., 24., 25., 26., 28.])
SIZE_EDGES = np.array([0.1, 0.24, 0.30, 0.38, 0.50, 1.50])   # TRUE size (arcsec)
G = 0.05
KEYMUL = 1_000_003
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
        needed |= {"detected", "case", "input_index", "neighbored",
                   "e1_input_rot0_p", "e2_input_rot0_p", "gamma1_input_p", "gamma2_input_p",
                   "r_input_p", "Re_input_p", "measured_mag_auto", "measured_flux_radius",
                   "measured_ngmix_g1", "measured_ngmix_g2"}
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
    if "nbr_flux_max" not in reservoir.columns:
        clk = pf.read_table(CROWDLK, columns=["case", "input_index", "nbr_flux_max"]).to_pandas()
        reservoir = reservoir.merge(clk, on=["case", "input_index"], how="left")
        reservoir["nbr_flux_max"] = reservoir["nbr_flux_max"].fillna(0.0)
    log(f"raw scanned={raw:,}  kept(detected+sel, case<= {max_case})={len(reservoir):,}")
    return reservoir


def snc_project(df):
    """per-object label projection proj_p = e_snc . ghat_p, per-target 1/npairs weight, mask has_p."""
    lk = pf.read_table(SNC, columns=["case", "input_index", "ngmix0_g1", "ngmix0_g2"]).to_pandas()
    lkey = lk["case"].to_numpy(np.int64) * KEYMUL + lk["input_index"].to_numpy(np.int64)
    order = np.argsort(lkey); lkey = lkey[order]
    e0_1 = lk["ngmix0_g1"].to_numpy(float)[order]; e0_2 = lk["ngmix0_g2"].to_numpy(float)[order]
    case = df["case"].to_numpy(np.int64); ii = df["input_index"].to_numpy(np.int64)
    key = case * KEYMUL + ii
    pos = np.clip(np.searchsorted(lkey, key), 0, len(lkey) - 1)
    match = lkey[pos] == key
    e1 = df["measured_ngmix_g1"].to_numpy(float) - np.where(match, e0_1[pos], np.nan)
    e2 = df["measured_ngmix_g2"].to_numpy(float) - np.where(match, e0_2[pos], np.nan)
    _, inv, npairs = np.unique(key, return_inverse=True, return_counts=True)
    w = 1.0 / npairs[inv]
    g1p = df["gamma1_input_p"].to_numpy(float); g2p = df["gamma2_input_p"].to_numpy(float)
    gp = np.hypot(g1p, g2p)
    proj_p = (g1p * e1 + g2p * e2) / np.where(gp > 1e-6, gp, 1.0)
    has_p = np.isfinite(proj_p) & (gp > 1e-6) & match
    return proj_p, w, has_p


def cell_label(proj_p, w, has_p, bmi, bsi, mm, ss):
    m = has_p & np.isin(bmi, mm) & np.isin(bsi, ss)
    ww = w[m].sum()
    return (proj_p[m] * w[m]).sum() / ww / G if ww > 0 else np.nan, int(m.sum())


def cell_pred(rmodel, finm, bmi, bsi, mm, ss):
    m = finm & np.isin(bmi, mm) & np.isin(bsi, ss)
    return float(np.mean(rmodel[m])) if m.any() else np.nan, int(m.sum())


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
    bundle = load_measurement_model(args.measurement_model, device=device)
    log(f"device={device}  cond={list(bundle.condition_preprocessor.feature_names)}")

    df = load_pop(bundle, args.max_rows, args.max_case, args.seed)
    # ---- LABEL projection (uses ORIGINAL gamma_p direction + actual measured shape) ----
    proj_p, w, has_p = snc_project(df)
    # ---- FLOW R_self prediction (certified forward path; gamma zeroed, model applies +/-delta) ----
    df["gamma1_input_p"] = 0.0; df["gamma2_input_p"] = 0.0
    rk = dict(pixel_rms=0.312, pixel_size=0.2, zero_mag=30.0, psf_fwhm=0.73, moffat_beta=2.224)
    intr = (df["e1_input_rot0_p"].to_numpy(float).copy(), df["e2_input_rot0_p"].to_numpy(float).copy())
    N = len(df); gh1 = np.ones(N); gh0 = np.zeros(N); d = args.delta
    _seed_flow(args.flow_seed); pp1 = proj_perobj(bundle, df, +d, gh1, gh0, intr, rk, args.n_samples, args.batch_size)
    _seed_flow(args.flow_seed); pm1 = proj_perobj(bundle, df, -d, gh1, gh0, intr, rk, args.n_samples, args.batch_size)
    _seed_flow(args.flow_seed); pp2 = proj_perobj(bundle, df, +d, gh0, gh1, intr, rk, args.n_samples, args.batch_size)
    _seed_flow(args.flow_seed); pm2 = proj_perobj(bundle, df, -d, gh0, gh1, intr, rk, args.n_samples, args.batch_size)
    rmodel = 0.5 * ((pp1 - pm1) + (pp2 - pm2)) / (2 * d)
    log(f"<R_model> global = {np.mean(rmodel):.4f}  (certified R_flow ~0.293)")

    tmag = df["r_input_p"].to_numpy(float); tsz = df["Re_input_p"].to_numpy(float)
    mmag = df["measured_mag_auto"].to_numpy(float); msz = df["measured_flux_radius"].to_numpy(float)
    finm = np.isfinite(rmodel) & np.isfinite(tmag) & np.isfinite(tsz) & np.isfinite(mmag) & np.isfinite(msz)
    nmag, nsz = len(MAG_EDGES) - 1, len(SIZE_EDGES) - 1

    # TRUE bin indices
    tmi = np.digitize(tmag, MAG_EDGES) - 1; tsi = np.digitize(tsz, SIZE_EDGES) - 1
    # MEASURED bin indices.  mag: same edges (same scale, noisy).  size: quantile-matched to true-size fracs.
    base = finm & (tmi >= 0) & (tmi < nmag) & (tsi >= 0) & (tsi < nsz)
    fr = [(tsz[base] < e).mean() for e in SIZE_EDGES[1:-1]]        # pop fraction below each interior true edge
    msz_edges = np.concatenate([[-np.inf], np.quantile(msz[base], fr), [np.inf]])
    log(f"measured-size quantile edges (px) at true fracs {np.round(fr,3).tolist()}: "
        f"{np.round(msz_edges[1:-1],3).tolist()}")
    mmi = np.digitize(mmag, MAG_EDGES) - 1; msi = np.digitize(msz, msz_edges) - 1

    ALLM, ALLS = list(range(nmag)), list(range(nsz))

    def marg_table(tag, bmi, bsi, edges_mag_lab, edges_sz_lab):
        print(f"\n===== {tag} =====")
        print(f"{'bin':22s}| {'Rself_lab':>9s}{'Rself_prd':>9s}{'resid':>8s} | {'m_self%':>8s} | {'n':>11s}")
        print("-" * 74)
        sl, nl = cell_label(proj_p, w, has_p, bmi, bsi, ALLM, ALLS)
        sp, npn = cell_pred(rmodel, finm & base, bmi, bsi, ALLM, ALLS)
        ms = 100 * (sl / sp - 1) if sp else np.nan
        print(f"{'GLOBAL':22s}| {sl:>+9.3f}{sp:>+9.3f}{sp-sl:>+8.3f} | {ms:>+8.2f} | {npn:>11,}")
        print("- by MAG (over all sizes) " + "-" * 47)
        for a in range(nmag):
            sl, _ = cell_label(proj_p, w, has_p, bmi, bsi, [a], ALLS)
            sp, npn = cell_pred(rmodel, finm & base, bmi, bsi, [a], ALLS)
            ms = 100 * (sl / sp - 1) if sp else np.nan
            print(f"  mag[{edges_mag_lab[a]:.0f},{edges_mag_lab[a+1]:.0f}){'':10s}| "
                  f"{sl:>+9.3f}{sp:>+9.3f}{sp-sl:>+8.3f} | {ms:>+8.2f} | {npn:>11,}")
        print("- by SIZE (over all mags) " + "-" * 47)
        for bb in range(nsz):
            sl, _ = cell_label(proj_p, w, has_p, bmi, bsi, ALLM, [bb])
            sp, npn = cell_pred(rmodel, finm & base, bmi, bsi, ALLM, [bb])
            ms = 100 * (sl / sp - 1) if sp else np.nan
            lo, hi = edges_sz_lab[bb], edges_sz_lab[bb + 1]
            print(f"  sz[{lo:.3g},{hi:.3g}){'':8s}| {sl:>+9.3f}{sp:>+9.3f}{sp-sl:>+8.3f} | {ms:>+8.2f} | {npn:>11,}")

    print("\nR_self label = <e_snc.ghat_p>/g (physical, TRUE-property function).  "
          "R_self pred = flow induced self-response (conditions on MEASURED mag/size).")
    print("resid = pred - label.  Compare the SIZE marginals between the two blocks:")
    print("  TRUE-binned: sharp label step, smeared pred (prior result).")
    print("  MEASURED-binned: if resid collapses -> flow is a good measured-conditioned estimator (EiV).")
    marg_table("TRUE-BINNED (sanity: reproduces prior grid)", tmi, tsi, MAG_EDGES, SIZE_EDGES)
    marg_table("MEASURED-BINNED (the test)", mmi, msi, MAG_EDGES, msz_edges)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez(OUT, rmodel=rmodel, proj_p=proj_p, w=w, has_p=has_p, finm=finm,
             tmag=tmag, tsz=tsz, mmag=mmag, msz=msz, case=df["case"].to_numpy(np.int64),
             MAG_EDGES=MAG_EDGES, SIZE_EDGES=SIZE_EDGES, msz_edges=msz_edges, G=G)
    log(f"wrote {OUT}")
    print("MEASBINNED_DONE", flush=True)


if __name__ == "__main__":
    main()
