"""Fit a g=0-derived additive mean correction and validate it on held-out gold.

This is a post-model calibration of the flow's zero-shear conditional mean, not a
gold subtraction:

  delta_mu(x) = E_g0[measured_ngmix - flow_mean | property bin]

The correction is learned from unsheared g=0 rows and then applied to held-out
constant-gold rows.  It should remove the flow additive residual while preserving
the real measured-ngmix additive offset.
"""
import argparse
import gc
import os
import sys

import numpy as np

SBSI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SBSI_ROOT not in sys.path:
    sys.path.insert(0, SBSI_ROOT)

from sbs_shear.measurement_model import load_measurement_model  # noqa
from scripts.diagnose_additive_origin import (  # noqa
    attach_lookups,
    flow_mean,
    load_g0,
    load_gold,
)


def _qedges(x, n):
    x = np.asarray(x, float)
    good = np.isfinite(x)
    if good.sum() < n:
        raise ValueError("not enough finite rows for quantile edges")
    e = np.quantile(x[good], np.linspace(0, 1, n + 1))
    e[0] -= 1e-9
    e[-1] += 1e-9
    return e


def _crowd_edges(x, n_pos, eps):
    x = np.asarray(x, float)
    pos = np.isfinite(x) & (x >= eps)
    if pos.sum() < n_pos:
        return None
    e = np.quantile(x[pos], np.linspace(0, 1, n_pos + 1))
    e[0] -= 1e-9
    e[-1] += 1e-9
    return e


def _bin_ids(df, mag_edges, re_edges, crowd_edges, crowd_col, crowd_eps):
    mag = df["r_input_p"].to_numpy(float)
    re = df["Re_input_p"].to_numpy(float)
    crowd = df[crowd_col].to_numpy(float) if crowd_col in df.columns else np.zeros(len(df))
    im = np.clip(np.digitize(mag, mag_edges) - 1, 0, len(mag_edges) - 2)
    ir = np.clip(np.digitize(re, re_edges) - 1, 0, len(re_edges) - 2)
    if crowd_edges is None:
        ic = np.zeros(len(df), dtype=np.int64)
        nc = 1
    else:
        hi = np.isfinite(crowd) & (crowd >= crowd_eps)
        ic = np.where(hi, 1 + np.clip(np.digitize(crowd, crowd_edges) - 1, 0, len(crowd_edges) - 2), 0)
        nc = len(crowd_edges)
    return im.astype(np.int64), ir.astype(np.int64), ic.astype(np.int64), nc


def fit_correction(df, args):
    mag_edges = np.asarray(args.mag_edges, dtype=float)
    re_edges = _qedges(df["Re_input_p"].to_numpy(float), args.n_size)
    crowd_edges = _crowd_edges(df[args.crowd_col].to_numpy(float), args.n_crowd_pos, args.crowd_eps)
    im, ir, ic, nc = _bin_ids(df, mag_edges, re_edges, crowd_edges, args.crowd_col, args.crowd_eps)
    nm = len(mag_edges) - 1
    nr = len(re_edges) - 1
    resid = np.column_stack([
        df["measured_c1"].to_numpy(float) - df["flow_c1"].to_numpy(float),
        df["measured_c2"].to_numpy(float) - df["flow_c2"].to_numpy(float),
    ])
    global_resid = np.nanmean(resid, axis=0)

    sums = np.zeros((nm, nr, nc, 2), dtype=np.float64)
    counts = np.zeros((nm, nr, nc), dtype=np.float64)
    sums_mr = np.zeros((nm, nr, 2), dtype=np.float64)
    counts_mr = np.zeros((nm, nr), dtype=np.float64)
    good = np.isfinite(resid).all(axis=1)
    for k in np.where(good)[0]:
        sums[im[k], ir[k], ic[k]] += resid[k]
        counts[im[k], ir[k], ic[k]] += 1.0
        sums_mr[im[k], ir[k]] += resid[k]
        counts_mr[im[k], ir[k]] += 1.0

    cell_mean = np.divide(sums, counts[..., None], out=np.zeros_like(sums), where=counts[..., None] > 0)
    mr_mean = np.divide(sums_mr, counts_mr[..., None], out=np.zeros_like(sums_mr), where=counts_mr[..., None] > 0)
    corr = np.zeros_like(cell_mean)
    source = np.full(counts.shape, "global", dtype="<U16")
    for a in range(nm):
        for b in range(nr):
            for c in range(nc):
                if counts[a, b, c] >= args.min_count:
                    corr[a, b, c] = cell_mean[a, b, c]
                    source[a, b, c] = "cell"
                elif counts_mr[a, b] >= args.min_count:
                    corr[a, b, c] = mr_mean[a, b]
                    source[a, b, c] = "mag_size"
                else:
                    corr[a, b, c] = global_resid
    return {
        "mag_edges": mag_edges,
        "re_edges": re_edges,
        "crowd_edges": crowd_edges if crowd_edges is not None else np.array([], dtype=float),
        "crowd_col": args.crowd_col,
        "crowd_eps": float(args.crowd_eps),
        "corr": corr,
        "counts": counts,
        "cell_mean": cell_mean,
        "global_resid": global_resid,
        "source": source,
    }


def apply_correction(df, table):
    crowd_edges = table["crowd_edges"]
    if len(crowd_edges) == 0:
        crowd_edges = None
    im, ir, ic, _ = _bin_ids(
        df,
        table["mag_edges"],
        table["re_edges"],
        crowd_edges,
        str(table["crowd_col"]),
        float(table["crowd_eps"]),
    )
    corr = table["corr"][im, ir, ic]
    out = df.copy()
    out["corr_c1"] = corr[:, 0]
    out["corr_c2"] = corr[:, 1]
    out["flow_corr_c1"] = out["flow_c1"] + out["corr_c1"]
    out["flow_corr_c2"] = out["flow_c2"] + out["corr_c2"]
    out["resid_corr_c1"] = out["measured_c1"] - out["flow_corr_c1"]
    out["resid_corr_c2"] = out["measured_c2"] - out["flow_corr_c2"]
    gh1 = out["gh1"].to_numpy(float)
    gh2 = out["gh2"].to_numpy(float)
    out["flow_corr_par"] = out["flow_corr_c1"] * gh1 + out["flow_corr_c2"] * gh2
    out["flow_corr_cross"] = -out["flow_corr_c1"] * gh2 + out["flow_corr_c2"] * gh1
    out["resid_corr_par"] = out["measured_c1"] * gh1 + out["measured_c2"] * gh2 - out["flow_corr_par"]
    out["resid_corr_cross"] = -out["measured_c1"] * gh2 + out["measured_c2"] * gh1 - out["flow_corr_cross"]
    return out


def _mean(df, col):
    return float(np.nanmean(df[col].to_numpy(float))) if len(df) else float("nan")


def print_before_after(name, df):
    print(f"\n=== {name}: additive correction validation ===")
    print(f"N={len(df):,}")
    print("fixed frame:")
    print(f"  measured      c1={_mean(df,'measured_c1'):+.5f}  c2={_mean(df,'measured_c2'):+.5f}")
    print(f"  flow before   c1={_mean(df,'flow_c1'):+.5f}  c2={_mean(df,'flow_c2'):+.5f}")
    print(f"  flow after    c1={_mean(df,'flow_corr_c1'):+.5f}  c2={_mean(df,'flow_corr_c2'):+.5f}")
    print(f"  residual bef  c1={_mean(df,'measured_c1')-_mean(df,'flow_c1'):+.5f}  "
          f"c2={_mean(df,'measured_c2')-_mean(df,'flow_c2'):+.5f}")
    print(f"  residual aft  c1={_mean(df,'resid_corr_c1'):+.5f}  c2={_mean(df,'resid_corr_c2'):+.5f}")
    print("shear frame:")
    meas_par = df["measured_c1"].to_numpy(float) * df["gh1"].to_numpy(float) + df["measured_c2"].to_numpy(float) * df["gh2"].to_numpy(float)
    meas_cross = -df["measured_c1"].to_numpy(float) * df["gh2"].to_numpy(float) + df["measured_c2"].to_numpy(float) * df["gh1"].to_numpy(float)
    flow_par = df["flow_c1"].to_numpy(float) * df["gh1"].to_numpy(float) + df["flow_c2"].to_numpy(float) * df["gh2"].to_numpy(float)
    flow_cross = -df["flow_c1"].to_numpy(float) * df["gh2"].to_numpy(float) + df["flow_c2"].to_numpy(float) * df["gh1"].to_numpy(float)
    print(f"  measured      par={np.nanmean(meas_par):+.5f}  cross={np.nanmean(meas_cross):+.5f}")
    print(f"  flow before   par={np.nanmean(flow_par):+.5f}  cross={np.nanmean(flow_cross):+.5f}")
    print(f"  flow after    par={_mean(df,'flow_corr_par'):+.5f}  cross={_mean(df,'flow_corr_cross'):+.5f}")
    print(f"  residual bef  par={np.nanmean(meas_par-flow_par):+.5f}  cross={np.nanmean(meas_cross-flow_cross):+.5f}")
    print(f"  residual aft  par={_mean(df,'resid_corr_par'):+.5f}  cross={_mean(df,'resid_corr_cross'):+.5f}")


def print_binned(name, df, col, edges=None, labels=None, masks=None):
    if masks is None:
        x = df[col].to_numpy(float)
        masks = [(x >= edges[i]) & (x < edges[i + 1]) for i in range(len(edges) - 1)]
        labels = [f"[{edges[i]:.3g},{edges[i+1]:.3g})" for i in range(len(edges) - 1)]
    print(f"\n--- {name}: {col} before/after residual c2 ---")
    print(f"{'bin':>14} {'N':>9} {'resid_before':>14} {'resid_after':>13}")
    for lab, m in zip(labels, masks):
        sub = df[m]
        if len(sub) < 500:
            continue
        before = _mean(sub, "measured_c2") - _mean(sub, "flow_c2")
        after = _mean(sub, "resid_corr_c2")
        print(f"{lab:>14} {len(sub):>9,} {before:>+14.5f} {after:>+13.5f}")


def save_table(path, table):
    serial = dict(table)
    serial["source"] = table["source"].astype(str)
    np.savez(path, **serial)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--measurement-model", default="models/measurement_flow_g0_ngmix_crowdflux_lam300_v1.pt")
    ap.add_argument("--g0-catalogue", default="/project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_crowd_g0.0_train_c0-39.feather")
    ap.add_argument("--gold-catalogue", default="/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/constant_response_catalogue_train.feather")
    ap.add_argument("--blend-lookup", default="results/blend_lookup_extnbrho_c0-39.feather")
    ap.add_argument("--crowd-flux-lookup", default="results/crowd_flux_c0-39.feather")
    ap.add_argument("--ood-lookup", default="results/ood_split_c0-39.feather")
    ap.add_argument("--nn-lookup", default="results/nn_dist_const_c0-39.feather")
    ap.add_argument("--output", default="results/additive_correction_g0_crowdflux_4x3x5.npz")
    ap.add_argument("--fit-max-case", type=int, default=19)
    ap.add_argument("--eval-min-case", type=int, default=20)
    ap.add_argument("--max-rows", type=int, default=12_000_000)
    ap.add_argument("--model-rows", type=int, default=1_500_000)
    ap.add_argument("--n-samples", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=65536)
    ap.add_argument("--mag-edges", type=float, nargs="+", default=[18, 24, 25, 26, 28])
    ap.add_argument("--n-size", type=int, default=3)
    ap.add_argument("--crowd-col", default="nbr_flux_near")
    ap.add_argument("--n-crowd-pos", type=int, default=4)
    ap.add_argument("--crowd-eps", type=float, default=1e-6)
    ap.add_argument("--min-count", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    bundle = load_measurement_model(args.measurement_model, device=device)
    print(f"device={device} model={args.measurement_model}")
    print(f"fit: g0 cases <= {args.fit_max_case}; validate: cases >= {args.eval_min_case}")

    g0_all = attach_lookups(load_g0(args), args, overwrite=False)
    g0_fit = g0_all[g0_all["case"] <= args.fit_max_case].reset_index(drop=True)
    g0_eval = g0_all[g0_all["case"] >= args.eval_min_case].reset_index(drop=True)
    g0_fit = flow_mean(bundle, g0_fit, args)
    table = fit_correction(g0_fit, args)
    save_table(args.output, table)

    print("\n=== fitted correction surface ===")
    print(f"global residual c1={table['global_resid'][0]:+.5f} c2={table['global_resid'][1]:+.5f}")
    print(f"cell counts: min={np.nanmin(table['counts']):.0f} median={np.nanmedian(table['counts']):.0f} max={np.nanmax(table['counts']):.0f}")
    print(f"correction c2 range: {np.nanmin(table['corr'][..., 1]):+.5f} .. {np.nanmax(table['corr'][..., 1]):+.5f}")

    g0_eval = flow_mean(bundle, g0_eval, args)
    g0_eval = apply_correction(g0_eval, table)
    print_before_after("G0 held-out", g0_eval)
    print_binned("G0 held-out", g0_eval, "r_input_p", edges=np.asarray(args.mag_edges, float))
    crowd = g0_eval[args.crowd_col].to_numpy(float)
    pos = crowd >= args.crowd_eps
    if pos.any():
        ce = table["crowd_edges"]
        masks = [~pos] + [pos & (crowd >= ce[i]) & (crowd < ce[i + 1]) for i in range(len(ce) - 1)]
        labels = ["zero"] + [f"q{i+1}" for i in range(len(ce) - 1)]
        print_binned("G0 held-out", g0_eval, args.crowd_col, labels=labels, masks=masks)

    del g0_all, g0_fit, g0_eval
    gc.collect()

    gold = attach_lookups(load_gold(args), args, overwrite=True)
    gold = gold[gold["case"] >= args.eval_min_case].reset_index(drop=True)
    gold = flow_mean(bundle, gold, args)
    gold = apply_correction(gold, table)
    print_before_after("GOLD held-out constant", gold)
    print_binned("GOLD held-out", gold, "r_input_p", edges=np.asarray(args.mag_edges, float))
    crowd = gold[args.crowd_col].to_numpy(float)
    pos = crowd >= args.crowd_eps
    if pos.any():
        ce = table["crowd_edges"]
        masks = [~pos] + [pos & (crowd >= ce[i]) & (crowd < ce[i + 1]) for i in range(len(ce) - 1)]
        labels = ["zero"] + [f"q{i+1}" for i in range(len(ce) - 1)]
        print_binned("GOLD held-out", gold, args.crowd_col, labels=labels, masks=masks)

    print("\nFIT_ADD_C_CORRECTION_DONE")


if __name__ == "__main__":
    main()
