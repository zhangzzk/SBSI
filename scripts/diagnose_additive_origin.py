"""Decompose the gold-constant additive bias into sim/catalogue and flow pieces.

For the same selected rows, report:

  intrinsic mean shape
  measured ngmix additive mean
  flow-predicted zero-shear mean
  residual measured - flow

The goal is to distinguish a real ngmix/simulation/catalogue additive offset from
an additive calibration residual caused by the learned flow mean.  The diagnostic
also bins the same quantities by magnitude, size, blend response, OOD neighbour
flux, crowd flux, and truth nearest-neighbour distance.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as pf
import pyarrow.ipc as ipc

SBSI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SBSI_ROOT not in sys.path:
    sys.path.insert(0, SBSI_ROOT)

from sbs_shear.coordinates import ellipticity_from_axis_ratio_angle  # noqa
from sbs_shear.measurement_model import load_measurement_model  # noqa
from sbs_shear.preprocessing import DEFAULT_SELECTION_CUTS, rescale, source_select_selection  # noqa
from scripts.response_ratio_diagnostic import _shape_target_indices  # noqa

KEYMUL = 1_000_003
RK = dict(pixel_rms=0.312, pixel_size=0.2, zero_mag=30.0, psf_fwhm=0.73, moffat_beta=2.224)
CONST_CAT = "/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/constant_response_catalogue_train.feather"
G0_CAT = "/project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_crowd_g0.0_train_c0-39.feather"


def _read_batches(path, columns, max_rows):
    parts = []
    n = 0
    with ipc.open_file(path) as reader:
        avail = set(reader.schema.names)
        cols = [c for c in columns if c in avail]
        for bi in range(reader.num_record_batches):
            batch = pa.Table.from_batches([reader.get_batch(bi)]).select(cols).to_pandas()
            parts.append(batch)
            n += len(batch)
            if max_rows and n >= max_rows:
                break
    if not parts:
        return pd.DataFrame(columns=list(columns))
    return pd.concat(parts, ignore_index=True)


def _key(df):
    return df["case"].to_numpy(np.int64) * KEYMUL + df["input_index"].to_numpy(np.int64)


def _lookup(path, columns):
    if not path or not os.path.exists(path):
        return None, None
    tab = pf.read_table(path).to_pandas()
    need = ["case", "input_index"] + list(columns)
    tab = tab[[c for c in need if c in tab.columns]]
    key = _key(tab)
    order = np.argsort(key)
    vals = {c: tab[c].to_numpy(float)[order] for c in columns if c in tab.columns}
    return key[order], vals


def _merge_lookup(df, keys, vals, fill=0.0):
    if keys is None or vals is None:
        for name in vals or {}:
            df[name] = fill
        return
    k = _key(df)
    pos = np.clip(np.searchsorted(keys, k), 0, len(keys) - 1)
    match = keys[pos] == k
    for name, arr in vals.items():
        df[name] = np.where(match, arr[pos], fill)


def _add_intrinsic(df):
    if "e1_input_rot0_p" in df.columns and "e2_input_rot0_p" in df.columns:
        return df
    e1, e2 = ellipticity_from_axis_ratio_angle(
        df["axis_ratio_input_p"].to_numpy(float),
        df["position_angle_input_p"].to_numpy(float),
    )
    df["e1_input_rot0_p"] = e1
    df["e2_input_rot0_p"] = e2
    return df


def load_gold(args):
    cols = [
        "measured_e1_plus", "measured_e2_plus", "measured_e1_minus", "measured_e2_minus",
        "applied_g1", "applied_g2", "case", "input_index", "neighbored", "distance",
        "Re_input_p", "Re_input_s", "axis_ratio_input_p", "axis_ratio_input_s",
        "position_angle_input_p", "position_angle_input_s", "r_input_p", "r_input_s",
        "redshift_input_p", "redshift_input_s", "sersic_n_input_p", "sersic_n_input_s",
    ]
    df = _read_batches(args.gold_catalogue, cols, args.max_rows)
    _add_intrinsic(df)
    df["gamma1_input_p"] = 0.0
    df["gamma2_input_p"] = 0.0
    df = source_select_selection(df, cuts=DEFAULT_SELECTION_CUTS).reset_index(drop=True)
    e1p = df["measured_e1_plus"].to_numpy(float)
    e1m = df["measured_e1_minus"].to_numpy(float)
    e2p = df["measured_e2_plus"].to_numpy(float)
    e2m = df["measured_e2_minus"].to_numpy(float)
    df["measured_c1"] = 0.5 * (e1p + e1m)
    df["measured_c2"] = 0.5 * (e2p + e2m)
    gm = np.hypot(df["applied_g1"].to_numpy(float), df["applied_g2"].to_numpy(float))
    good = gm > 0
    gh1 = np.zeros(len(df))
    gh2 = np.zeros(len(df))
    gh1[good] = df["applied_g1"].to_numpy(float)[good] / gm[good]
    gh2[good] = df["applied_g2"].to_numpy(float)[good] / gm[good]
    df["gh1"] = gh1
    df["gh2"] = gh2
    return df


def load_g0(args):
    cols = [
        "measured_ngmix_g1", "measured_ngmix_g2", "case", "input_index", "neighbored", "distance",
        "Re_input_p", "Re_input_s", "axis_ratio_input_p", "axis_ratio_input_s",
        "position_angle_input_p", "position_angle_input_s", "r_input_p", "r_input_s",
        "redshift_input_p", "redshift_input_s", "sersic_n_input_p", "sersic_n_input_s",
        "e1_input_rot0_p", "e2_input_rot0_p", "gamma1_input_p", "gamma2_input_p",
        "nbr_flux_near", "nbr_flux_far", "r_blend",
    ]
    df = _read_batches(args.g0_catalogue, cols, args.max_rows)
    _add_intrinsic(df)
    df["gamma1_input_p"] = 0.0
    df["gamma2_input_p"] = 0.0
    df = source_select_selection(df, cuts=DEFAULT_SELECTION_CUTS).reset_index(drop=True)
    df = df[np.isfinite(df["measured_ngmix_g1"]) & np.isfinite(df["measured_ngmix_g2"])].reset_index(drop=True)
    df["measured_c1"] = df["measured_ngmix_g1"].to_numpy(float)
    df["measured_c2"] = df["measured_ngmix_g2"].to_numpy(float)
    df["gh1"] = 1.0
    df["gh2"] = 0.0
    return df


def attach_lookups(df, args, overwrite=True):
    fk, fv = _lookup(args.crowd_flux_lookup, ["nbr_flux_near", "nbr_flux_far"])
    if fk is not None and overwrite:
        _merge_lookup(df, fk, fv)
    for name in ("nbr_flux_near", "nbr_flux_far"):
        if name not in df.columns:
            df[name] = 0.0
        df[name] = df[name].fillna(0.0)

    bk, bv = _lookup(args.blend_lookup, ["R_blend"])
    if bk is not None and overwrite:
        _merge_lookup(df, bk, bv)
        df["r_blend"] = df["R_blend"].to_numpy(float)
    elif "r_blend" not in df.columns:
        df["r_blend"] = 0.0
    df["r_blend"] = df["r_blend"].fillna(0.0)

    ok, ov = _lookup(args.ood_lookup, ["ood_flux", "ood_flux_bright", "ood_flux_faint"])
    if ok is not None and overwrite:
        _merge_lookup(df, ok, ov)
    for name in ("ood_flux", "ood_flux_bright", "ood_flux_faint"):
        if name not in df.columns:
            df[name] = 0.0
        df[name] = df[name].fillna(0.0)

    nk, nv = _lookup(args.nn_lookup, ["nn_dist_any", "nn_dist_bright"])
    if nk is not None and overwrite:
        _merge_lookup(df, nk, nv, fill=np.nan)
    for name in ("nn_dist_any", "nn_dist_bright"):
        if name not in df.columns:
            df[name] = np.nan
    return df


def flow_mean(bundle, df, args):
    if len(df) > args.model_rows:
        df = df.sample(n=args.model_rows, random_state=args.seed).reset_index(drop=True)
    frame = df.copy()
    frame["gamma1_input_p"] = 0.0
    frame["gamma2_input_p"] = 0.0
    frame = rescale(frame, **RK)
    draws = bundle.sample(frame, n_samples=args.n_samples, batch_size=args.batch_size)
    mean = bundle.target_transform.inverse_transform_array(draws.mean(axis=1))
    i1, i2 = _shape_target_indices(bundle.target_transform.target_names)
    df = df.copy()
    df["flow_c1"] = mean[:, i1]
    df["flow_c2"] = mean[:, i2]
    return df


def add_projections(df):
    c1 = df["measured_c1"].to_numpy(float)
    c2 = df["measured_c2"].to_numpy(float)
    f1 = df["flow_c1"].to_numpy(float)
    f2 = df["flow_c2"].to_numpy(float)
    i1 = df["e1_input_rot0_p"].to_numpy(float)
    i2 = df["e2_input_rot0_p"].to_numpy(float)
    gh1 = df["gh1"].to_numpy(float)
    gh2 = df["gh2"].to_numpy(float)
    for prefix, x1, x2 in (("meas", c1, c2), ("flow", f1, f2), ("intr", i1, i2)):
        df[f"{prefix}_par"] = x1 * gh1 + x2 * gh2
        df[f"{prefix}_cross"] = -x1 * gh2 + x2 * gh1
    df["resid_c1"] = df["measured_c1"] - df["flow_c1"]
    df["resid_c2"] = df["measured_c2"] - df["flow_c2"]
    df["resid_par"] = df["meas_par"] - df["flow_par"]
    df["resid_cross"] = df["meas_cross"] - df["flow_cross"]
    df["meas_minus_intr_c1"] = df["measured_c1"] - df["e1_input_rot0_p"]
    df["meas_minus_intr_c2"] = df["measured_c2"] - df["e2_input_rot0_p"]
    df["flow_minus_intr_c1"] = df["flow_c1"] - df["e1_input_rot0_p"]
    df["flow_minus_intr_c2"] = df["flow_c2"] - df["e2_input_rot0_p"]
    return df


def _mean(df, col):
    x = df[col].to_numpy(float)
    return float(np.nanmean(x)) if len(x) else float("nan")


def print_summary(name, df):
    print(f"\n=== {name} same-row additive decomposition ===")
    print(f"N={len(df):,}")
    print("fixed-frame:")
    print(f"  intrinsic     c1={_mean(df,'e1_input_rot0_p'):+.5f}  c2={_mean(df,'e2_input_rot0_p'):+.5f}")
    print(f"  measured/ngmix c1={_mean(df,'measured_c1'):+.5f}  c2={_mean(df,'measured_c2'):+.5f}")
    print(f"  flow          c1={_mean(df,'flow_c1'):+.5f}  c2={_mean(df,'flow_c2'):+.5f}")
    print(f"  measured-flow c1={_mean(df,'resid_c1'):+.5f}  c2={_mean(df,'resid_c2'):+.5f}")
    print(f"  measured-intr c1={_mean(df,'meas_minus_intr_c1'):+.5f}  c2={_mean(df,'meas_minus_intr_c2'):+.5f}")
    print(f"  flow-intr     c1={_mean(df,'flow_minus_intr_c1'):+.5f}  c2={_mean(df,'flow_minus_intr_c2'):+.5f}")
    print("shear frame:")
    print(f"  measured      par={_mean(df,'meas_par'):+.5f}  cross={_mean(df,'meas_cross'):+.5f}")
    print(f"  flow          par={_mean(df,'flow_par'):+.5f}  cross={_mean(df,'flow_cross'):+.5f}")
    print(f"  measured-flow par={_mean(df,'resid_par'):+.5f}  cross={_mean(df,'resid_cross'):+.5f}")


def _quantile_bins(x, n, min_pos=None):
    x = np.asarray(x, float)
    good = np.isfinite(x)
    if min_pos is not None:
        good &= x >= min_pos
    if good.sum() < n:
        return None
    edges = np.quantile(x[good], np.linspace(0, 1, n + 1))
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    return edges


def print_bins(name, df, spec):
    col, labels, masks = spec
    print(f"\n--- {name}: {col} ---")
    header = (
        f"{'bin':>16} {'N':>9} {'intr2':>8} {'meas2':>8} {'flow2':>8} "
        f"{'res2':>8} {'meas-intr2':>11} {'flow-intr2':>11} "
        f"{'meas_cross':>10} {'flow_cross':>10} {'res_cross':>10}"
    )
    print(header)
    for label, mask in zip(labels, masks):
        sub = df[mask]
        if len(sub) < 500:
            continue
        print(
            f"{label:>16} {len(sub):>9,} "
            f"{_mean(sub,'e2_input_rot0_p'):>+8.5f} {_mean(sub,'measured_c2'):>+8.5f} "
            f"{_mean(sub,'flow_c2'):>+8.5f} {_mean(sub,'resid_c2'):>+8.5f} "
            f"{_mean(sub,'meas_minus_intr_c2'):>+11.5f} {_mean(sub,'flow_minus_intr_c2'):>+11.5f} "
            f"{_mean(sub,'meas_cross'):>+10.5f} {_mean(sub,'flow_cross'):>+10.5f} "
            f"{_mean(sub,'resid_cross'):>+10.5f}"
        )


def specs(df):
    out = []
    r = df["r_input_p"].to_numpy(float)
    bins = [(18, 24), (24, 25), (25, 26), (26, 28)]
    out.append(("r-mag", [f"[{a},{b})" for a, b in bins], [(r >= a) & (r < b) for a, b in bins]))
    re = df["Re_input_p"].to_numpy(float)
    e = _quantile_bins(re, 4)
    if e is not None:
        out.append(("Re_input_p", [f"q{i+1}" for i in range(4)], [(re >= e[i]) & (re < e[i + 1]) for i in range(4)]))
    rb = df["r_blend"].to_numpy(float)
    hi = rb >= 0.02
    if hi.any():
        e = _quantile_bins(rb[hi], 4)
        masks = [~hi]
        labels = ["Rbl<0.02"]
        masks += [hi & (rb >= e[i]) & (rb < e[i + 1]) for i in range(4)]
        labels += [f"Rbl q{i+1}" for i in range(4)]
        out.append(("R_blend", labels, masks))
    for col in ("ood_flux_bright", "ood_flux_faint", "nbr_flux_near", "nbr_flux_far"):
        x = df[col].to_numpy(float)
        pos = x > 1e-6
        if pos.any():
            e = _quantile_bins(x[pos], 3)
            if e is not None:
                masks = [~pos] + [pos & (x >= e[i]) & (x < e[i + 1]) for i in range(3)]
                labels = ["zero"] + [f"q{i+1}" for i in range(3)]
                out.append((col, labels, masks))
    for col in ("nn_dist_any", "nn_dist_bright"):
        if col not in df.columns:
            continue
        x = df[col].to_numpy(float)
        finite = np.isfinite(x)
        if finite.sum():
            cuts = [0.0, 2.0, 3.0, 5.0, 7.0]
            out.append((col, [f">{c:g}\"" for c in cuts], [finite & (x > c) for c in cuts]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--measurement-model", default="models/measurement_flow_g0_ngmix_crowdflux_lam300_v1.pt")
    ap.add_argument("--gold-catalogue", default=CONST_CAT)
    ap.add_argument("--g0-catalogue", default=G0_CAT)
    ap.add_argument("--blend-lookup", default="results/blend_lookup_extnbrho_c0-39.feather")
    ap.add_argument("--crowd-flux-lookup", default="results/crowd_flux_c0-39.feather")
    ap.add_argument("--ood-lookup", default="results/ood_split_c0-39.feather")
    ap.add_argument("--nn-lookup", default="results/nn_dist_const_c0-39.feather")
    ap.add_argument("--max-rows", type=int, default=12_000_000)
    ap.add_argument("--model-rows", type=int, default=1_500_000)
    ap.add_argument("--n-samples", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=65536)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    bundle = load_measurement_model(args.measurement_model, device=device)
    print(f"device={device} model={args.measurement_model}")

    gold = attach_lookups(load_gold(args), args, overwrite=True)
    gold = add_projections(flow_mean(bundle, gold, args))
    print_summary("GOLD constant antithetic", gold)
    for spec in specs(gold):
        print_bins("GOLD", gold, spec)
    del gold

    import gc
    gc.collect()

    g0 = attach_lookups(load_g0(args), args, overwrite=False)
    g0 = add_projections(flow_mean(bundle, g0, args))
    print_summary("G0 measured catalogue", g0)
    for spec in specs(g0):
        print_bins("G0", g0, spec)

    print("\nDIAG_ADD_C_ORIGIN_DONE")


if __name__ == "__main__":
    main()
