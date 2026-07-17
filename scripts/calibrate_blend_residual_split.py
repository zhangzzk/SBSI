"""Case-split calibration test for residual coherent-blend response.

The current constant-gold validation uses

    R_total = R_flow(self) + R_blend(emulator)

and reaches global m ~ +0.75%, with a large q3 residual when binned by
emulator R_blend.  This script asks whether that residual is stable enough to
calibrate on one set of simulation cases and transfer to held-out cases:

    Delta_b = <R_sim>_fit,b - <R_flow>_fit,b - <R_blend>_fit,b
    R_total_corrected = R_flow + R_blend + Delta_b

where b is the R_blend bin.  This is not a same-row all-case subtraction: bin
edges and Delta_b are learned from fit cases only, then evaluated on separate
held-out cases.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pyarrow.feather as pf

SBSI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SBSI_ROOT not in sys.path:
    sys.path.insert(0, SBSI_ROOT)

from sbs_shear.measurement_model import load_measurement_model  # noqa: E402
from scripts.response_ratio_diagnostic import model_mean_proj  # noqa: E402
from scripts.validate_constant_with_blend import CBASE, load  # noqa: E402


def merge_lookup(df, path, columns, fill=0.0):
    if not path or not os.path.exists(path):
        for c in columns:
            df[c] = fill
        return df
    tab = pf.read_table(path).to_pandas()[["case", "input_index", *columns]]
    merged = df[["case", "input_index"]].merge(tab, on=["case", "input_index"], how="left")
    for c in columns:
        df[c] = merged[c].fillna(fill).to_numpy(float)
    print(f"{os.path.basename(path)}: matched {np.mean(merged[columns[0]].notna()):.1%}")
    return df


def response_arrays(df):
    gmag = np.hypot(df["applied_g1"].to_numpy(float), df["applied_g2"].to_numpy(float))
    g = float(np.median(gmag))
    gh1 = df["applied_g1"].to_numpy(float) / gmag
    gh2 = df["applied_g2"].to_numpy(float) / gmag
    e1p = df["measured_e1_plus"].to_numpy(float)
    e2p = df["measured_e2_plus"].to_numpy(float)
    e1m = df["measured_e1_minus"].to_numpy(float)
    e2m = df["measured_e2_minus"].to_numpy(float)
    r_sim = ((e1p - e1m) * gh1 + (e2p - e2m) * gh2) / (2.0 * g)
    intr = (df["e1_input_rot0_p"].to_numpy(float).copy(), df["e2_input_rot0_p"].to_numpy(float).copy())
    return g, gh1, gh2, r_sim, intr


def rblend_bins(rb, edges, eps, n_blend):
    hi = rb >= eps
    return np.where(~hi, 0, 1 + np.clip(np.digitize(rb, edges) - 1, 0, n_blend - 1)).astype(np.int64)


def fit_edges(rb_fit, eps, n_blend):
    hi = rb_fit >= eps
    if hi.sum() < n_blend:
        raise ValueError("not enough high-R_blend rows for quantile bins")
    edges = np.quantile(rb_fit[hi], np.linspace(0, 1, n_blend + 1))
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    return edges


def flow_response(bundle, df, mask, g, gh1, gh2, intr, rk, n_samples, batch_size, seed):
    import torch

    sub = df.loc[mask].reset_index(drop=True)
    intr_sub = (intr[0][mask], intr[1][mask])
    torch.manual_seed(seed)
    mp, _ = model_mean_proj(bundle, sub, +g, gh1[mask], gh2[mask], intr_sub, rk, n_samples, batch_size)
    torch.manual_seed(seed + 1)
    mm, _ = model_mean_proj(bundle, sub, -g, gh1[mask], gh2[mask], intr_sub, rk, n_samples, batch_size)
    return (mp - mm) / (2.0 * g)


def bin_stats(name, df, base_mask, binid, rb, r_sim, bundle, g, gh1, gh2, intr, rk, args):
    rows = []
    print(f"\n--- {name} bin stats ---")
    print(f"{'bin':>12} {'R_sim':>8} {'R_flow':>8} {'R_bl':>8} {'m':>9} {'N':>10}")
    for b in range(args.n_blend + 1):
        m = base_mask & (binid == b)
        if m.sum() < args.min_bin_rows:
            raise RuntimeError(f"{name} bin {b} has too few rows: {m.sum()}")
        rf = flow_response(bundle, df, m, g, gh1, gh2, intr, rk, args.n_samples, args.batch_size, args.seed + 100 * b)
        rs = float(np.mean(r_sim[m]))
        rbl = float(np.mean(rb[m]))
        tag = "ISO(Rbl~0)" if b == 0 else f"Rbl q{b}"
        print(f"{tag:>12} {rs:>8.4f} {rf:>8.4f} {rbl:>8.4f} {rs/(rf+rbl)-1:>+9.2%} {int(m.sum()):>10,}")
        rows.append({"bin": b, "label": tag, "N": int(m.sum()), "R_sim": rs, "R_flow": rf, "R_blend": rbl})
    return rows


def assign_rtotal(binid, rb, stats, delta):
    rf = np.zeros(len(binid), dtype=np.float64)
    dd = np.zeros(len(binid), dtype=np.float64)
    for row in stats:
        b = row["bin"]
        m = binid == b
        rf[m] = row["R_flow"]
        dd[m] = delta[b]
    return rf + rb + dd


def bootstrap_m(r_sim, r_total, cases, n_boot, seed):
    uc = np.unique(cases)
    per = {c: (float(r_sim[cases == c].sum()), float(r_total[cases == c].sum()), int((cases == c).sum())) for c in uc}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(uc, size=len(uc), replace=True)
        srs = sum(per[c][0] for c in pick)
        srt = sum(per[c][1] for c in pick)
        sn = sum(per[c][2] for c in pick)
        vals.append((srs / sn) / (srt / sn) - 1.0)
    return float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")


def print_global(name, r_sim, r_total, cases, n_boot, seed):
    rs = float(np.mean(r_sim))
    rt = float(np.mean(r_total))
    m = rs / rt - 1.0
    err = bootstrap_m(r_sim, r_total, cases, n_boot, seed)
    print(f"{name}: R_sim={rs:.5f} R_total={rt:.5f} m={m:+.3%} +/- {err:.3%}")
    return m, err


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--measurement-model", default="models/measurement_flow_g0_ngmix_crowdflux_lam300_v1.pt")
    ap.add_argument("--catalogue", default=CBASE + "constant_response_catalogue_train.feather")
    ap.add_argument("--blend-lookup", default="results/blend_lookup_extnbrho_c0-39.feather")
    ap.add_argument("--crowd-flux-lookup", default="results/crowd_flux_c0-39.feather")
    ap.add_argument("--fit-max-case", type=int, default=19)
    ap.add_argument("--eval-min-case", type=int, default=20)
    ap.add_argument("--max-rows", type=int, default=12_000_000)
    ap.add_argument("--n-blend", type=int, default=4)
    ap.add_argument("--blend-eps", type=float, default=0.02)
    ap.add_argument("--target-bin-m", type=float, default=0.0)
    ap.add_argument("--shrink", type=float, default=1.0)
    ap.add_argument("--min-bin-rows", type=int, default=5000)
    ap.add_argument("--n-samples", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=65536)
    ap.add_argument("--n-boot", type=int, default=300)
    ap.add_argument("--seed", type=int, default=31)
    ap.add_argument("--device", default=None)
    for k, v in dict(pixel_rms=0.312, pixel_size=0.2, zero_mag=30.0, psf_fwhm=0.73, moffat_beta=2.224).items():
        ap.add_argument(f"--{k.replace('_','-')}", type=float, default=v)
    args = ap.parse_args()

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    bundle = load_measurement_model(args.measurement_model, device=device)
    rk = dict(pixel_rms=args.pixel_rms, pixel_size=args.pixel_size, zero_mag=args.zero_mag,
              psf_fwhm=args.psf_fwhm, moffat_beta=args.moffat_beta)
    print(f"device={device} model={args.measurement_model}")
    print(f"fit cases <= {args.fit_max_case}; eval cases >= {args.eval_min_case}")

    df = load(args.catalogue, args.max_rows)
    df = merge_lookup(df, args.blend_lookup, ["R_blend"], fill=0.0)
    df["r_blend"] = df["R_blend"].to_numpy(float)
    df = merge_lookup(df, args.crowd_flux_lookup, ["nbr_flux_near", "nbr_flux_far"], fill=0.0)

    g, gh1, gh2, r_sim, intr = response_arrays(df)
    rb = df["R_blend"].to_numpy(float)
    cases = df["case"].to_numpy(np.int64)
    fit_mask = cases <= args.fit_max_case
    eval_mask = cases >= args.eval_min_case
    print(f"N total={len(df):,}; fit={fit_mask.sum():,}; eval={eval_mask.sum():,}; |g|={g:.4f}")

    edges = fit_edges(rb[fit_mask], args.blend_eps, args.n_blend)
    binid = rblend_bins(rb, edges, args.blend_eps, args.n_blend)
    print(f"R_blend edges from fit: {np.array2string(edges, precision=5)}")

    fit_stats = bin_stats("FIT", df, fit_mask, binid, rb, r_sim, bundle, g, gh1, gh2, intr, rk, args)
    eval_stats = bin_stats("EVAL", df, eval_mask, binid, rb, r_sim, bundle, g, gh1, gh2, intr, rk, args)

    delta = np.zeros(args.n_blend + 1, dtype=np.float64)
    print("\n--- fitted Delta_R_blend by bin ---")
    print(f"{'bin':>12} {'Delta':>10} {'fit_m_before':>13} {'fit_m_after':>12}")
    for row in fit_stats:
        b = row["bin"]
        before = row["R_sim"] / (row["R_flow"] + row["R_blend"]) - 1.0
        target_total = row["R_sim"] / (1.0 + args.target_bin_m)
        raw_delta = target_total - row["R_flow"] - row["R_blend"]
        delta[b] = args.shrink * raw_delta
        after = row["R_sim"] / (row["R_flow"] + row["R_blend"] + delta[b]) - 1.0
        print(f"{row['label']:>12} {delta[b]:>+10.5f} {before:>+13.2%} {after:>+12.2%}")

    rtot_fit_unc = assign_rtotal(binid, rb, fit_stats, np.zeros_like(delta))
    rtot_fit_cor = assign_rtotal(binid, rb, fit_stats, delta)
    rtot_eval_unc = assign_rtotal(binid, rb, eval_stats, np.zeros_like(delta))
    rtot_eval_cor = assign_rtotal(binid, rb, eval_stats, delta)

    print("\n--- global m ---")
    print_global("FIT uncorrected ", r_sim[fit_mask], rtot_fit_unc[fit_mask], cases[fit_mask], args.n_boot, args.seed)
    print_global("FIT corrected   ", r_sim[fit_mask], rtot_fit_cor[fit_mask], cases[fit_mask], args.n_boot, args.seed + 1)
    print_global("EVAL uncorrected", r_sim[eval_mask], rtot_eval_unc[eval_mask], cases[eval_mask], args.n_boot, args.seed + 2)
    print_global("EVAL corrected  ", r_sim[eval_mask], rtot_eval_cor[eval_mask], cases[eval_mask], args.n_boot, args.seed + 3)

    print("\n--- EVAL bins after applying FIT deltas ---")
    print(f"{'bin':>12} {'m_before':>10} {'m_after':>10} {'R_sim':>8} {'Rtot_before':>12} {'Rtot_after':>11} {'N':>10}")
    for row in eval_stats:
        b = row["bin"]
        before_total = row["R_flow"] + row["R_blend"]
        after_total = before_total + delta[b]
        print(
            f"{row['label']:>12} {row['R_sim']/before_total-1:>+10.2%} "
            f"{row['R_sim']/after_total-1:>+10.2%} {row['R_sim']:>8.4f} "
            f"{before_total:>12.4f} {after_total:>11.4f} {row['N']:>10,}"
        )

    print("\nCALIBRATE_BLEND_RESIDUAL_SPLIT_DONE")


if __name__ == "__main__":
    main()
