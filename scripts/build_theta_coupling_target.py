"""Build the R_theta ORIENTATION-COUPLING target for the joint forward flow (owner: pin the
coupling directly, 2026-07-23).

The measured-property shear response the flow's mean head carries in dims 2 (measured_mag_auto)
and 3 (measured_log_flux_radius): its SELECTION-driving part is a spin-2 orientation coupling

    d(log size)/dg = b_size(cell) * (e_int . ghat)          (+ a coherent even part central-diff kills)
    d(mag)/dg      = b_mag(cell)  * (e_int . ghat) ~ 0       (flux ~ conserved)

measured DIRECTLY from the matched half-shear legs (g0 unsheared vs g05 primary-sheared, neighbour
FIXED so no R_blend leak). Per flow-grid cell (flux=r_input_p x size=Re_input_p x blend) we fit the
slope b = cov(d<prop>/dg, e_int.ghat)/var(e_int.ghat). Low-count cells fall back to the global slope.

The trainer (train_joint_forward.py) then pins, per object:
    d(log size)/d_gamma1 -> b_size*e1_int ,  d(log size)/d_gamma2 -> b_size*e2_int   (dim 3)
    d(mag)/d_gamma1       -> b_mag*e1_int  ,  ...                                     (dim 2)
so the mean head reproduces the size selection response R_sel. Grid edges are COPIED from the shape
target npz so the per-object binid is identical. Firewall: half-shear only; constgold untouched.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc

SBSI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SBSI_ROOT not in sys.path:
    sys.path.insert(0, SBSI_ROOT)
from sbs_shear.coordinates import ellipticity_from_axis_ratio_angle  # noqa: E402
from sbs_shear.preprocessing import DEFAULT_SELECTION_CUTS, source_select_selection  # noqa: E402

CATDIR = "/project/ls-gruen/users/zekang.zhang/sbsi_catalogues"
BASE = ["axis_ratio_input_p", "position_angle_input_p", "e1_input_rot0_p", "e2_input_rot0_p",
        "Re_input_p", "r_input_p", "distance", "neighbored", "detected", "case", "input_index",
        "measured_flux_radius", "measured_mag_auto", "r_blend"]
G05_EXTRA = ["gamma1_input_p", "gamma2_input_p", "gamma1_input_s", "gamma2_input_s"]


def _stream(path, cols, max_case, true_cut):
    parts = []
    with ipc.open_file(path) as r:
        use = [c for c in cols if c in r.schema.names]
        for bi in range(r.num_record_batches):
            b = pa.Table.from_batches([r.get_batch(bi)]).select(use).to_pandas()
            if max_case is not None:
                if int(b["case"].min()) > max_case:
                    break
                b = b[b["case"] <= max_case]
            if len(b) == 0:
                continue
            b = source_select_selection(b, cuts=DEFAULT_SELECTION_CUTS)
            b = b[b["detected"].astype(bool)]
            if true_cut is not None:
                b = b[(b["Re_input_p"].to_numpy(float) > true_cut[0])
                      & (b["r_input_p"].to_numpy(float) < true_cut[1])]
            if len(b):
                parts.append(b)
    return pd.concat(parts, ignore_index=True)


def _eint(df):
    """Intrinsic (rot0) sky-basis ellipticity, matching train_joint_forward.intrinsic_shape."""
    if "e1_input_rot0_p" in df.columns and df["e1_input_rot0_p"].notna().any():
        return df["e1_input_rot0_p"].to_numpy(float), df["e2_input_rot0_p"].to_numpy(float), "rot0"
    e1, e2 = ellipticity_from_axis_ratio_angle(df["axis_ratio_input_p"].to_numpy(float),
                                               df["position_angle_input_p"].to_numpy(float))
    return np.asarray(e1, float), np.asarray(e2, float), "axisratio"


def _slope(y, x):
    xc = x - x.mean()
    vv = float(np.dot(xc, xc))
    return float(np.dot(y - y.mean(), xc) / vv) if vv > 0 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--g0", default=CATDIR + "/det_meas_crowd_conc_g0.0_train_full.feather")
    ap.add_argument("--g05", default=CATDIR + "/det_meas_crowd_g0.05_val_full.feather")
    ap.add_argument("--grid-npz", default="results/response_target_isoblend_RAWfine_c0-99_6x9x5.npz",
                    help="shape target npz whose (flux,size,dist) edges define the identical flow grid")
    ap.add_argument("--max-case", type=int, default=99)
    ap.add_argument("--true-re-min", type=float, default=0.3)
    ap.add_argument("--true-mag-max", type=float, default=26.0)
    ap.add_argument("--min-count", type=int, default=300, help="cells below this fall back to the global slope")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    g = 0.05
    true_cut = (args.true_re_min, args.true_mag_max)

    G = np.load(args.grid_npz)
    _cc = G["crowd_col"] if "crowd_col" in G.files else ""
    crowd_col = _cc.item() if hasattr(_cc, "item") and getattr(_cc, "shape", None) == () else str(_cc)
    ef, es = G["edges_flux"], G["edges_size"]
    nf, ns = len(ef) - 1, len(es) - 1
    if crowd_col:                                     # r_blend / nbr_flux crowd-axis grid (matches trainer)
        ec = G["edges_crowd"]; nblend = len(ec) - 1; ndist = None; ed = None
        print(f"grid from {args.grid_npz}: flux{nf} x size{ns} x CROWD[{crowd_col}]{nblend}", flush=True)
    else:
        ed = G["edges_dist"]; ndist = int(G["n_dist"]); nblend = ndist + 1
        print(f"grid from {args.grid_npz}: flux{nf} x size{ns} x blend{nblend} (ndist={ndist})", flush=True)

    G0 = _stream(args.g0, BASE, args.max_case, true_cut)
    G5 = _stream(args.g05, BASE + G05_EXTRA, args.max_case, true_cut)
    gs = np.hypot(G5["gamma1_input_s"].to_numpy(float), G5["gamma2_input_s"].to_numpy(float))
    G5 = G5[gs < 1e-6].copy()                                          # neighbour FIXED
    gp = np.hypot(G5["gamma1_input_p"].to_numpy(float), G5["gamma2_input_p"].to_numpy(float))
    G5 = G5[gp > 1e-6].copy()
    gp = np.hypot(G5["gamma1_input_p"].to_numpy(float), G5["gamma2_input_p"].to_numpy(float))
    G5["gh1"] = G5["gamma1_input_p"].to_numpy(float) / gp
    G5["gh2"] = G5["gamma2_input_p"].to_numpy(float) / gp
    G0 = G0.drop_duplicates(["case", "input_index"])
    G5 = G5.drop_duplicates(["case", "input_index"])
    keep0 = ["case", "input_index", "measured_flux_radius", "measured_mag_auto"]
    m = G5.merge(G0[keep0], on=["case", "input_index"], suffixes=("_g5", "_g0"))
    print(f"matched {len(m):,} (g0 {len(G0):,}, g05 neighbour-fixed {len(G5):,})", flush=True)

    e1i, e2i, econv = _eint(m)
    gh1 = m["gh1"].to_numpy(float); gh2 = m["gh2"].to_numpy(float)
    x = e1i * gh1 + e2i * gh2                                          # e_int . ghat
    # consistency check: rot0 vs axis-ratio ellipticity
    e1a, e2a = ellipticity_from_axis_ratio_angle(m["axis_ratio_input_p"].to_numpy(float),
                                                 m["position_angle_input_p"].to_numpy(float))
    print(f"e_int convention = {econv};  corr(rot0_e1, axisratio_e1) = "
          f"{np.corrcoef(e1i, np.asarray(e1a, float))[0, 1]:.4f}", flush=True)

    fr0 = m["measured_flux_radius_g0"].to_numpy(float); fr5 = m["measured_flux_radius_g5"].to_numpy(float)
    ok = (fr0 > 0) & (fr5 > 0) & np.isfinite(fr0) & np.isfinite(fr5)
    dlogsize = np.where(ok, (np.log(np.where(ok, fr5, 1.0)) - np.log(np.where(ok, fr0, 1.0))) / g, np.nan)
    dmag = (m["measured_mag_auto_g5"].to_numpy(float) - m["measured_mag_auto_g0"].to_numpy(float)) / g

    fin = ok & np.isfinite(dmag) & np.isfinite(x)
    b_size_glob = _slope(dlogsize[fin], x[fin]); b_mag_glob = _slope(dmag[fin], x[fin])
    print(f"GLOBAL coupling: b_size(log) = {b_size_glob:+.4f}   b_mag = {b_mag_glob:+.4f}   "
          f"(N={int(fin.sum()):,})", flush=True)

    flux = m["r_input_p"].to_numpy(float); size = m["Re_input_p"].to_numpy(float)
    fi = np.clip(np.digitize(flux, ef) - 1, 0, nf - 1)
    si = np.clip(np.digitize(size, es) - 1, 0, ns - 1)
    if crowd_col:                                     # quantile bins of r_blend; isolated/NaN -> bin 0
        rb = m[crowd_col].to_numpy(float)
        bi = np.clip(np.digitize(rb, ec) - 1, 0, nblend - 1)
        bi = np.where(np.isfinite(rb), bi, 0)
    else:
        dist = m["distance"].to_numpy(float); nbg = m["neighbored"].astype(bool).to_numpy()
        bi = np.where(nbg, 1 + np.clip(np.digitize(dist, ed) - 1, 0, ndist - 1), 0)

    b_size = np.full((nf, ns, nblend), b_size_glob)
    b_mag = np.full((nf, ns, nblend), b_mag_glob)
    cnt = np.zeros((nf, ns, nblend), np.int64)
    for a in range(nf):
        for b in range(ns):
            for c in range(nblend):
                cell = fin & (fi == a) & (si == b) & (bi == c)
                cnt[a, b, c] = int(cell.sum())
                if cnt[a, b, c] >= args.min_count:
                    bs = _slope(dlogsize[cell], x[cell])
                    if np.isfinite(bs):
                        b_size[a, b, c] = bs
                    # b_mag stays GLOBAL: mag response is orientation-independent ~0 (flux conserved),
                    # so per-cell fits are pure noise (some cells hit spurious O(1) slopes). Pin ~0.

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    save_kw = dict(edges_flux=ef, edges_size=es, coupling_size=b_size, coupling_mag=b_mag, counts=cnt,
                   b_size_global=b_size_glob, b_mag_global=b_mag_glob, nominal_g=g,
                   e_convention=econv, response_target=os.path.basename(args.grid_npz))
    if crowd_col:
        save_kw.update(edges_crowd=ec, crowd_col=crowd_col)
    else:
        save_kw.update(edges_dist=ed, n_dist=ndist)
    np.savez(args.output, **save_kw)
    filled = int((cnt >= args.min_count).sum())
    print(f"b_size(log) per-cell: {np.nanmin(b_size):+.3f}..{np.nanmax(b_size):+.3f}  "
          f"(median {np.nanmedian(b_size):+.3f}); {filled}/{nf * ns * nblend} cells fit, rest=global", flush=True)
    print(f"wrote {args.output}", flush=True)
    print("THETA_COUPLING_TARGET_DONE", flush=True)


if __name__ == "__main__":
    main()
