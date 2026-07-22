"""CLOSING TEST (owner-requested, points 1+2): the certified pipeline's REAL-DATA (ensemble-metric)
multiplicative bias on the gentle ACCEPTANCE ensemble, with selection INCLUDED.

Point 2: the honest object is the response of the ENSEMBLE-AVERAGE measured shape of the detected+
selected sample, d<measured shape>_{det&sel}/dg = R_total (contains shear response AND selection
response) -- NOT the matched per-object constgold r_sim (which CANCELS selection). Point 1: evaluate
it on a fairly-covering sample after GENTLE cuts (mag<25 & Re>0.3), not on pathological tail sub-cuts.

Estimator (single-shear leg, g=0.05, per-case shear direction):
    m_ens(C) = R_total(C) / < R_flow + R_blend >_C  - 1
    R_total(C) = <g_meas_par>_{det&C} / g           (measured shape proj on shear dir; ensemble truth)
    R_flow     = certified flow SELF-response (trace/2 central secant, the SAME validated path as the
                 certified harvest: eval_self_response_bins.proj_perobj / response_ratio_diagnostic)
    R_blend    = validated emulator R_blend, TRANSFERRED per (mag-decile x neighbored) cell from the
                 certified constgold dump (fig2_perobj; R_blend is a weakly-shear-dependent neighbour-
                 leakage response, so the per-cell ensemble transfer is a small correction on a correction)

R_flow is a pure self-response (synthetic shear on intrinsic shape, no population re-selection) => it
models R_shape only, NOT the selection response => m_ens exposes the selection bias the certified
pipeline omits on real data. On constgold that bias is invisible (matched detections cancel it).

Firewall: constgold is read ONLY for the R_blend transfer table (its r_sim is NEVER used); the leg is
the validation. No model is trained. GPU inference only.
"""
from __future__ import annotations
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as pf
import pyarrow.ipc as ipc

SBSI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SBSI_ROOT not in sys.path:
    sys.path.insert(0, SBSI_ROOT)

import torch  # noqa: E402
from sbs_shear.measurement_model import (  # noqa: E402
    load_measurement_model, raw_columns_for_measurement_targets)
from sbs_shear.preprocessing import (  # noqa: E402
    DEFAULT_SELECTION_CUTS, raw_columns_for_selection_features, source_select_selection)
from scripts.eval_self_response_bins import proj_perobj, _seed_flow  # noqa: E402

CATDIR = "/project/ls-gruen/users/zekang.zhang/sbsi_catalogues"
LEG = f"{CATDIR}/det_meas_ngmix_g0.05_val.feather"
CROWD = "/project/ls-gruen/users/zekang.zhang/sbsi_caches/crowd_flux_conc_c0-199.feather"
CONSTDUMP = "/project/ls-gruen/users/zekang.zhang/sbsi_dumps/fig2_perobj_s501_fixresp.feather"
EXTRA = ["detected", "case", "input_index", "gamma1_input_p", "gamma2_input_p",
         "e1_input_rot0_p", "e2_input_rot0_p", "measured_ngmix_g1", "measured_ngmix_g2",
         "measured_mag_auto", "measured_flux_radius", "r_input_p", "Re_input_p",
         "neighbored", "distance"]
t0 = time.time()


def load_leg(bundle, max_rows, seed):
    cond = bundle.condition_preprocessor.feature_names
    tgt = bundle.target_transform.target_names
    with ipc.open_file(LEG) as r:
        avail = set(r.schema.names)
        need = set(EXTRA)
        need |= raw_columns_for_selection_features(cond, available_columns=avail)
        need |= raw_columns_for_measurement_targets(tgt)
        need = [c for c in need if c in avail]
        rng = np.random.default_rng(seed)
        res = None; raw = 0
        for bi in range(r.num_record_batches):
            b = pa.Table.from_batches([r.get_batch(bi)]).select(need).to_pandas()
            raw += len(b)
            b = source_select_selection(b, cuts=DEFAULT_SELECTION_CUTS)
            if len(b) == 0:
                continue
            b = b[b["detected"].astype(bool)].reset_index(drop=True)
            if len(b) == 0:
                continue
            b["__k"] = rng.random(len(b))
            res = b if res is None else pd.concat([res, b], ignore_index=True)
            if len(res) > 2 * max_rows:
                res = res.nlargest(max_rows, "__k").reset_index(drop=True)
    if len(res) > max_rows:
        res = res.nlargest(max_rows, "__k").reset_index(drop=True)
    res = res.drop(columns="__k").reset_index(drop=True)
    print(f"  leg raw={raw:,} kept(det, subsample)={len(res):,} in {time.time()-t0:.0f}s", flush=True)
    return res


def join_crowd(df):
    cr = pf.read_table(CROWD, columns=["case", "input_index", "nbr_flux_near",
                                        "nbr_flux_far", "nbr_flux_max"]).to_pandas()
    n0 = len(df)
    df = df.merge(cr, on=["case", "input_index"], how="left")
    miss = int(df["nbr_flux_near"].isna().sum())
    for c in ("nbr_flux_near", "nbr_flux_far", "nbr_flux_max"):
        df[c] = df[c].fillna(0.0)
    print(f"  crowd join: {n0:,} rows, {miss:,} unmatched (->0)", flush=True)
    return df


def rblend_transfer_table(nmag=10):
    """<R_blend> per (true-mag decile x neighbored) from the certified constgold dump.
    Returns (edges, table[nmag,2]) ; r_sim is NEVER read."""
    cols = ["r_input_p", "neighbored", "R_blend"]
    parts = []
    with ipc.open_file(CONSTDUMP) as r:
        keep = [c for c in cols if c in set(r.schema.names)]
        for bi in range(r.num_record_batches):
            parts.append(pa.Table.from_batches([r.get_batch(bi)]).select(keep).to_pandas())
    d = pd.concat(parts, ignore_index=True)
    mag = d["r_input_p"].to_numpy(float); nb = d["neighbored"].to_numpy(bool)
    rb = d["R_blend"].to_numpy(float)
    ed = np.quantile(mag[np.isfinite(mag)], np.linspace(0, 1, nmag + 1)); ed[0] -= 1e-4; ed[-1] += 1e-4
    bi_ = np.clip(np.digitize(mag, ed[1:-1]), 0, nmag - 1)
    tab = np.full((nmag, 2), np.nan)
    for a in range(nmag):
        for k, sel in enumerate((~nb, nb)):
            m = (bi_ == a) & sel & np.isfinite(rb)
            if m.sum() > 100:
                tab[a, k] = float(np.mean(rb[m]))
    glob = float(np.nanmean(rb[np.isfinite(rb)]))
    tab = np.where(np.isfinite(tab), tab, glob)
    print(f"  R_blend transfer table from constgold: global<R_blend>={glob:.4f} (r_sim NOT read)", flush=True)
    return ed, tab


def assign_rblend(df, ed, tab):
    nmag = tab.shape[0]
    mag = df["r_input_p"].to_numpy(float); nb = df["neighbored"].to_numpy(bool)
    bi_ = np.clip(np.digitize(mag, ed[1:-1]), 0, nmag - 1)
    return np.where(nb, tab[bi_, 1], tab[bi_, 0])


def compute_rflow(bundle, df, n_samples, batch_size, flow_seed, delta):
    rk = dict(pixel_rms=0.312, pixel_size=0.2, zero_mag=30.0, psf_fwhm=0.73, moffat_beta=2.224)
    intr = (df["e1_input_rot0_p"].to_numpy(float).copy(), df["e2_input_rot0_p"].to_numpy(float).copy())
    N = len(df); one = np.ones(N); zero = np.zeros(N); d = delta
    _seed_flow(flow_seed); pp1 = proj_perobj(bundle, df, +d, one, zero, intr, rk, n_samples, batch_size)
    _seed_flow(flow_seed); pm1 = proj_perobj(bundle, df, -d, one, zero, intr, rk, n_samples, batch_size)
    _seed_flow(flow_seed); pp2 = proj_perobj(bundle, df, +d, zero, one, intr, rk, n_samples, batch_size)
    _seed_flow(flow_seed); pm2 = proj_perobj(bundle, df, -d, zero, one, intr, rk, n_samples, batch_size)
    return 0.5 * ((pp1 - pm1) + (pp2 - pm2)) / (2 * d)


def cuts_of(df):
    mag = df["r_input_p"].to_numpy(float); size = df["Re_input_p"].to_numpy(float)
    nb = df["neighbored"].to_numpy(bool)
    mm = df["measured_mag_auto"].to_numpy(float); ms = df["measured_flux_radius"].to_numpy(float)
    allrows = np.ones(len(mag), bool)
    C = [("GLOBAL", allrows)]
    # gentle ACCEPTANCE ensembles (point 1)
    C.append(("acc_true_mag<25&Re>0.3", (mag < 25) & (size > 0.3)))
    C.append(("acc_meas_mag<25&frad>0.3", np.isfinite(mm) & np.isfinite(ms) & (mm < 25) & (ms > 0.3 / 0.2)))
    C.append(("mag_lt24", mag < 24))
    C.append(("mag_lt25", mag < 25))
    C.append(("size_gt0.3", size > 0.3))
    C.append(("meas_mag_lt24.5", np.isfinite(mm) & (mm < 24.5)))
    # a couple of tail diagnostics for contrast
    C.append(("mag26-27(tail)", (mag >= 26) & (mag < 27)))
    C.append(("meas_frad_q1(tail)", np.isfinite(ms) & (ms < np.nanquantile(ms, 0.25))))
    return C


def eval_cut(df, sel, rflow, rblend, gpar, g, picks):
    fin = np.isfinite(gpar) & np.isfinite(rflow) & np.isfinite(rblend)
    s = sel & fin
    n = int(s.sum())
    if n < 3000:
        return None
    case = df["case"].to_numpy()
    uc, inv = np.unique(case, return_inverse=True)
    w = s.astype(np.float64)
    num = np.bincount(inv, weights=np.where(np.isfinite(gpar), gpar, 0) * w, minlength=len(uc))
    den = np.bincount(inv, weights=(rflow + rblend) * w, minlength=len(uc))
    cnt = np.bincount(inv, weights=w, minlength=len(uc))
    Rtot = num.sum() / cnt.sum() / g
    Rhat = den.sum() / cnt.sum()
    m = Rtot / Rhat - 1
    bn = num[picks].sum(1); bd = den[picks].sum(1); bc = cnt[picks].sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        bm = (bn / bc / g) / (bd / bc) - 1
    return dict(n=n, Rtotal=Rtot, Rhat=Rhat, Rflow=float((rflow*w).sum()/cnt.sum()),
                Rblend=float((rblend*w).sum()/cnt.sum()), m=m, sem=float(np.nanstd(bm)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--measurement-model",
                    default="models/measurement_flow_g0_ngmix_meas_szfl_noz_lam450_fixresp_s501.pt")
    ap.add_argument("--nominal-g", type=float, default=0.05)
    ap.add_argument("--max-rows", type=int, default=5_000_000)
    ap.add_argument("--n-samples", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=65536)
    ap.add_argument("--delta", type=float, default=0.02)
    ap.add_argument("--flow-seed", type=int, default=12345)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--out-prefix",
                    default="/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/ensemble_bias_leg")
    args = ap.parse_args()
    g = args.nominal_g
    lines = []

    def emit(s=""):
        print(s, flush=True); lines.append(s)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    emit("=" * 100)
    emit("CLOSING TEST: certified pipeline REAL-DATA (ensemble-metric) bias on the gentle acceptance sample")
    emit("m_ens = R_total / <R_flow + R_blend> - 1   (R_total = measured ensemble response, incl. selection)")
    emit("=" * 100)
    emit(f"device={device}  model={os.path.basename(args.measurement_model)}")
    bundle = load_measurement_model(args.measurement_model, device=device)

    df = load_leg(bundle, args.max_rows, args.seed)
    df = join_crowd(df)
    ed, tab = rblend_transfer_table()
    rblend = assign_rblend(df, ed, tab)

    # per-case shear projection for R_total (measured shape)
    g1 = df["gamma1_input_p"].to_numpy(float); g2 = df["gamma2_input_p"].to_numpy(float)
    gmag = np.hypot(g1, g2)
    with np.errstate(invalid="ignore", divide="ignore"):
        u1 = np.where(gmag > 1e-6, g1 / gmag, np.nan); u2 = np.where(gmag > 1e-6, g2 / gmag, np.nan)
    gpar = df["measured_ngmix_g1"].to_numpy(float) * u1 + df["measured_ngmix_g2"].to_numpy(float) * u2

    emit(f"\n[R_flow] evaluating certified self-response on {len(df):,} leg objects (GPU)...")
    rflow = compute_rflow(bundle, df, args.n_samples, args.batch_size, args.flow_seed, args.delta)
    emit(f"  <R_flow>_global = {np.nanmean(rflow):+.4f}   <R_blend>_global = {np.nanmean(rblend):+.4f}")

    rng = np.random.default_rng(args.seed)
    uc = np.unique(df["case"].to_numpy()); picks = rng.integers(0, len(uc), size=(args.n_boot, len(uc)))
    emit(f"[boot] {len(uc)} cases x {args.n_boot}\n")

    emit(f"{'cut':>26} {'n':>10} {'R_total':>9} {'R_flow':>8} {'R_blend':>8} {'Rhat':>8} {'m_ens%':>9} {'sem%':>7} {'z':>6}")
    emit("-" * 100)
    rows = []
    for label, sel in cuts_of(df):
        r = eval_cut(df, sel, rflow, rblend, gpar, g, picks)
        if r is None:
            emit(f"{label:>26} {'(thin)':>10}"); continue
        z = abs(r["m"]) / r["sem"] if r["sem"] > 0 else float("nan")
        emit(f"{label:>26} {r['n']:>10,} {r['Rtotal']:>+9.4f} {r['Rflow']:>+8.4f} {r['Rblend']:>+8.4f} "
             f"{r['Rhat']:>+8.4f} {100*r['m']:>+9.2f} {100*r['sem']:>7.2f} {z:>6.1f}")
        rows.append((label, r))

    emit("\n" + "=" * 100)
    emit("READING: on the gentle ACCEPTANCE ensembles (acc_*), |m_ens| is the certified pipeline's real-data")
    emit("  bias WITH selection. Sub-percent there => certified already handles realistic selection on the")
    emit("  usable sample. Several-% there => the selection response (#35's target) is needed even on the bulk.")
    emit("  (#35's ceiling = it can drive m_ens toward the R_flow non-closure; it cannot beat R_total in the tail.)")

    txt = args.out_prefix + ".txt"; npz = args.out_prefix + ".npz"
    with open(txt, "w") as f:
        f.write("\n".join(lines) + "\n")
    np.savez(npz, labels=np.array([x[0] for x in rows]),
             Rtotal=np.array([x[1]["Rtotal"] for x in rows]),
             Rflow=np.array([x[1]["Rflow"] for x in rows]),
             Rblend=np.array([x[1]["Rblend"] for x in rows]),
             m=np.array([x[1]["m"] for x in rows]), sem=np.array([x[1]["sem"] for x in rows]),
             n=np.array([x[1]["n"] for x in rows]))
    emit(f"\nwrote {txt}\nwrote {npz}\nENSEMBLE_BIAS_DONE")


if __name__ == "__main__":
    main()
