"""Isolate the SELECTION/detection contribution to the multiplicative shear bias.

Hypothesis (user): the g=0.05/0.2 detected sample is a different selection of true-property
space than the g=0 sample the flow is trained on, because detection is a measured-space
threshold that depends (through shape) on shear. The shape-only g=0 forward model would then
be evaluated on a shear-shifted selection it never saw.

Decisive control: detection completeness is ~1 for BRIGHT galaxies (selection is then
shear-INDEPENDENT) and <1 for FAINT ones (selection active). Comparing the forward->data
response ratio at FIXED true size (which removes the shape<->size covariate shift) between
bright and faint cells triangulates three hypotheses:

  * ratio ~ 1 in bright cells, != 1 in faint cells  -> SELECTION is a real contributor.
  * ratio != 1 even in bright, fixed-size cells      -> residual is intrinsic to the shape
        likelihood at matched truth (insufficient conditioning / rendering-level), NOT selection.
  * (covariate shift is already controlled by binning in true size.)

Forward responsivity R_fwd = (OLS slope of measured chi on scene distortion at g=0) * rho_S,
with rho_S the analytic Mobius scene-distortion response to shear. Data responsivity
R_data = <chi_parallel>_sheared / g. Uses only g=0 truth + the analytic map + the known
per-object shear DIRECTION (not magnitude); no sheared-shear magnitude enters the forward.
"""

from __future__ import annotations

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

from sbs_shear.measurement_model import add_measurement_target_features  # noqa: E402
from sbs_shear.preprocessing import DEFAULT_SELECTION_CUTS, source_select_selection  # noqa: E402
from sbs_shear.shear_map import apply_shear_to_ellipticity  # noqa: E402

CD = "/project/ls-gruen/users/zekang.zhang/sbsi_catalogues"
RAW = ["detected", "gamma1_input_p", "gamma2_input_p", "e1_input_rot0_p", "e2_input_rot0_p",
       "measured_a_image", "measured_b_image", "measured_theta_image",
       "r_input_p", "Re_input_p", "distance", "neighbored"]


def eps_to_distortion(e1, e2):
    d = 1.0 + e1 * e1 + e2 * e2
    return 2.0 * e1 / d, 2.0 * e2 / d


def slope_on_distortion(e1, e2, y1, y2):
    x1, x2 = eps_to_distortion(e1, e2)
    if len(e1) < 50:
        return np.nan
    return 0.5 * (np.polyfit(x1, y1, 1)[0] + np.polyfit(x2, y2, 1)[0])


def rho_S(e1, e2, ds=0.01):
    """Scene-distortion response to applied shear, from the analytic Mobius (g=0)."""
    p1, _ = eps_to_distortion(*apply_shear_to_ellipticity(e1, e2, ds, 0.0))
    m1, _ = eps_to_distortion(*apply_shear_to_ellipticity(e1, e2, -ds, 0.0))
    return (p1.mean() - m1.mean()) / (2 * ds)


def _stream(path, max_rows, seed, keep_undetected):
    """Stream a catalogue; apply source cuts; keep the detected flag (and, if requested,
    undetected rows for the completeness calculation). measured shape is computed only where
    finite."""
    rng = np.random.default_rng(seed)
    reservoir, raw = None, 0
    with ipc.open_file(path) as reader:
        avail = set(reader.schema.names)
        cols = [c for c in RAW if c in avail]
        for bi in range(reader.num_record_batches):
            b = pa.Table.from_batches([reader.get_batch(bi)]).select(cols).to_pandas()
            raw += len(b)
            b = source_select_selection(b, cuts=DEFAULT_SELECTION_CUTS)
            if len(b) == 0:
                continue
            if not keep_undetected:
                b = b[b["detected"].astype(bool)].reset_index(drop=True)
                if len(b) == 0:
                    continue
            b = add_measurement_target_features(b)  # measured_e1/e2_image where possible
            b["__k"] = rng.random(len(b))
            reservoir = b if reservoir is None else pd.concat([reservoir, b], ignore_index=True)
            if len(reservoir) > 2 * max_rows:
                reservoir = reservoir.nlargest(max_rows, "__k").reset_index(drop=True)
    reservoir = reservoir.nlargest(min(max_rows, len(reservoir)), "__k").reset_index(drop=True)
    return reservoir, raw


def cell_response(g0_cell, sheared_cells):
    """Forward (g=0) and data responsivities for one (size x mag) cell."""
    e1 = g0_cell["e1_input_rot0_p"].to_numpy(float)
    e2 = g0_cell["e2_input_rot0_p"].to_numpy(float)
    det = g0_cell["detected"].astype(bool).to_numpy()
    e1d, e2d = e1[det], e2[det]
    y1 = g0_cell["measured_e1_image"].to_numpy(float)[det]
    y2 = g0_cell["measured_e2_image"].to_numpy(float)[det]
    good = np.isfinite(y1) & np.isfinite(y2)
    M = slope_on_distortion(e1d[good], e2d[good], y1[good], y2[good])
    rS = rho_S(e1d[good], e2d[good]) if good.sum() > 50 else np.nan
    R_fwd = M * rS
    out = {"R_fwd": R_fwd, "det_frac_g0": det.mean(), "n_g0_det": int(good.sum())}
    for nominal, sh in sheared_cells.items():
        g1 = sh["gamma1_input_p"].to_numpy(float); g2 = sh["gamma2_input_p"].to_numpy(float)
        gm = np.hypot(g1, g2); det_s = sh["detected"].astype(bool).to_numpy()
        msk = det_s & (gm > 0.01)
        e1m = sh["measured_e1_image"].to_numpy(float); e2m = sh["measured_e2_image"].to_numpy(float)
        msk &= np.isfinite(e1m) & np.isfinite(e2m)
        if msk.sum() < 50:
            out[nominal] = (np.nan, det_s.mean(), 0); continue
        gh1 = g1[msk] / gm[msk]; gh2 = g2[msk] / gm[msk]
        chi_par = (e1m[msk] * gh1 + e2m[msk] * gh2).mean()
        R_data = chi_par / nominal
        out[nominal] = (R_fwd / R_data if R_data else np.nan, det_s.mean(), int(msk.sum()))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--g0", default=os.path.join(CD, "det_meas_g0.0_train.feather"))
    ap.add_argument("--sheared", nargs="+",
                    default=[f"{CD}/det_meas_g0.05_val.feather:0.05",
                             f"{CD}/det_meas_g0.2_val.feather:0.2"])
    ap.add_argument("--g0-rows", type=int, default=4_000_000)
    ap.add_argument("--sheared-rows", type=int, default=3_000_000)
    ap.add_argument("--n-size", type=int, default=3)
    ap.add_argument("--n-mag", type=int, default=2)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    g0, g0raw = _stream(args.g0, args.g0_rows, args.seed, keep_undetected=True)
    print(f"[g0] rows={len(g0):,} (raw {g0raw:,})  overall det_frac={g0['detected'].mean():.3f}")
    shes = {}
    for spec in args.sheared:
        path, nom = spec.rsplit(":", 1); nom = float(nom)
        sh, r = _stream(path, args.sheared_rows, args.seed + 1, keep_undetected=True)
        shes[nom] = sh
        print(f"[g={nom}] rows={len(sh):,} (raw {r:,})  det_frac={sh['detected'].mean():.3f}")

    # Bin edges from the DETECTED g=0 sample (the flow's training support).
    det0 = g0[g0["detected"].astype(bool)]
    size0 = np.log10(det0["Re_input_p"].to_numpy(float))
    mag0 = det0["r_input_p"].to_numpy(float)
    s_edges = np.quantile(size0, np.linspace(0, 1, args.n_size + 1)); s_edges[0] -= 1e-6; s_edges[-1] += 1e-6
    m_edges = np.quantile(mag0, np.linspace(0, 1, args.n_mag + 1)); m_edges[0] -= 1e-6; m_edges[-1] += 1e-6
    print(f"\ntrue size log10(Re) edges: {np.round(s_edges,3)}")
    print(f"true mag r_input_p edges : {np.round(m_edges,3)}  (smaller mag = brighter)")
    print("\nratio = R_forward / R_data  (->1 = g=0 forward transfers).  detf = detection fraction.")
    print("If bright cells (low mag bin) give ratio~1 but faint cells don't -> SELECTION drives m.")
    print("If even bright, fixed-size cells give ratio!=1 -> residual is intrinsic, not selection.\n")

    def assign(df):
        s = np.log10(df["Re_input_p"].to_numpy(float)); m = df["r_input_p"].to_numpy(float)
        return np.clip(np.digitize(s, s_edges) - 1, 0, args.n_size - 1), \
               np.clip(np.digitize(m, m_edges) - 1, 0, args.n_mag - 1)

    g0_si, g0_mi = assign(g0)
    sh_idx = {nom: assign(sh) for nom, sh in shes.items()}
    noms = sorted(shes)
    header = f"{'sizebin':>8s} {'magbin':>7s} {'detf_g0':>8s}"
    for nom in noms:
        header += f" | g={nom}: {'detf':>5s} {'ratio':>6s}"
    print(header)
    for si in range(args.n_size):
        for mi in range(args.n_mag):
            g0c = g0[(g0_si == si) & (g0_mi == mi)]
            shc = {nom: shes[nom][(sh_idx[nom][0] == si) & (sh_idx[nom][1] == mi)] for nom in noms}
            r = cell_response(g0c, shc)
            tag = "BRIGHT" if mi == 0 else ("FAINT" if mi == args.n_mag - 1 else f"mid{mi}")
            line = f"{si:>8d} {tag:>7s} {r['det_frac_g0']:>8.3f}"
            for nom in noms:
                ratio, detf, n = r[nom]
                line += f" | {detf:>10.3f} {ratio:>6.3f}"
            print(line)
    print("\n(For reference, the full-sample flow-MLE m is +0.030; first-moment full-sample"
          " ratio was ~0.82 with size conditioning.)")


if __name__ == "__main__":
    main()
