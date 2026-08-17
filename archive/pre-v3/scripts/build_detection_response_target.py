"""Blend-aware selection-response target b_sim(flux x size x blend) for the response-aware
selection classifier.

The selection-induced shape shift (per unit shear) in a true (flux x size x blend) cell:
  s_par      = S_gamma(e_intrinsic) . ghat                 (true scene shape along shear axis)
  b_sim(cell)= [<s_par>_detected,cell - <s_par>_parent,cell] / g

measured on the PARENT sample (detected + undetected) of a sheared catalogue.  This is the
target the response-aware selection loss supervises the classifier's induced b_model toward.
Also reports whether the selection response depends on blend status (the user's blend-aware ask).

blend bin 0 = ISOLATED (neighbored False); 1..n_dist = BLENDED by neighbour distance.
"""
import argparse
import os
import sys

import numpy as np

SBSI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SBSI_ROOT not in sys.path:
    sys.path.insert(0, SBSI_ROOT)

from sbs_shear.shear_map import apply_shear_to_ellipticity  # noqa
from sbs_shear.sim_stream import stream_reservoir  # noqa

PARENT_COLS = {"e1_input_rot0_p", "e2_input_rot0_p", "gamma1_input_p", "gamma2_input_p",
               "detected", "r_input_p", "Re_input_p", "distance", "neighbored",
               "measured_ngmix_g1"}


def load_parent(cat, max_rows, seed):
    """Parent sample = source-selected, BOTH detected and undetected (sheared catalogue)."""
    return stream_reservoir(cat, PARENT_COLS, max_rows, seed=seed, detected=None, shear_threshold=1e-6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalogue", required=True)
    ap.add_argument("--nominal-g", type=float, default=0.05)
    ap.add_argument("--n-flux", type=int, default=4)
    ap.add_argument("--n-size", type=int, default=2)
    ap.add_argument("--n-dist", type=int, default=3)
    ap.add_argument("--max-rows", type=int, default=20_000_000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    df = load_parent(args.catalogue, args.max_rows, args.seed)
    g1 = df["gamma1_input_p"].to_numpy(float); g2 = df["gamma2_input_p"].to_numpy(float)
    gm = np.hypot(g1, g2); gh1, gh2 = g1 / gm, g2 / gm
    sc1, sc2 = apply_shear_to_ellipticity(df["e1_input_rot0_p"].to_numpy(float),
                                          df["e2_input_rot0_p"].to_numpy(float), g1, g2)
    s_par = sc1 * gh1 + sc2 * gh2
    # "selected" = DETECTED. (ngmix-convergence was DROPPED as a selection channel: the ~50%
    # "non-converged" are the `--targets secondaries` bookkeeping split (1st half of the input
    # list is never fit), NOT a real S/N-driven selection -- see WORKLOG 2026-07-01.)
    det = df["detected"].astype(bool).to_numpy()
    flux = df["r_input_p"].to_numpy(float); size = df["Re_input_p"].to_numpy(float)
    nbf = df["neighbored"].astype(bool).to_numpy(); dist = df["distance"].to_numpy(float)
    g = args.nominal_g

    ef = np.quantile(flux, np.linspace(0, 1, args.n_flux + 1)); ef[0] -= 1e-6; ef[-1] += 1e-6
    es = np.quantile(size, np.linspace(0, 1, args.n_size + 1)); es[0] -= 1e-6; es[-1] += 1e-6
    db = dist[nbf & np.isfinite(dist)]
    ed = np.quantile(db, np.linspace(0, 1, args.n_dist + 1)); ed[0] -= 1e-6; ed[-1] += 1e-6
    fi = np.clip(np.digitize(flux, ef) - 1, 0, args.n_flux - 1)
    si = np.clip(np.digitize(size, es) - 1, 0, args.n_size - 1)
    di = np.where(nbf, 1 + np.clip(np.digitize(dist, ed) - 1, 0, args.n_dist - 1), 0)
    nblend = args.n_dist + 1

    bgrid = np.full((args.n_flux, args.n_size, nblend), np.nan)
    cnt = np.zeros((args.n_flux, args.n_size, nblend), dtype=np.int64)
    gb_par = float(np.mean(s_par)); gb_det = float(np.mean(s_par[det]))
    global_b = (gb_det - gb_par) / g
    for a in range(args.n_flux):
        for b in range(args.n_size):
            for c in range(nblend):
                m = (fi == a) & (si == b) & (di == c)
                cnt[a, b, c] = m.sum()
                if m.sum() > 500 and det[m].sum() > 100:
                    bgrid[a, b, c] = (float(np.mean(s_par[m & det])) - float(np.mean(s_par[m]))) / g
    valid = np.isfinite(bgrid)                      # cells with a real per-cell measurement
    bgrid = np.where(valid, bgrid, global_b)        # fallback-fill the rest (NOT supervised in training)

    np.savez(args.output, edges_flux=ef, edges_size=es, edges_dist=ed, b_sim=bgrid,
             counts=cnt, valid=valid, nominal_g=g, global_b=global_b, n_dist=args.n_dist)
    print(f"  real cells (supervised): {int(valid.sum())}/{valid.size}")
    print(f"selection-response target b_sim(flux x size x blend), g={g}, N_parent={len(df):,}")
    print(f"GLOBAL b_sim/g = {global_b*100:+.3f}%   (sim selection response; classifier currently +1.9%)")
    print("blend-dependence of selection response (count-weighted b_sim/g per blend bin):")
    for c in range(nblend):
        tag = "ISOLATED" if c == 0 else f"blend d{c}"
        w = cnt[:, :, c]
        bb = np.nansum(bgrid[:, :, c] * w) / max(w.sum(), 1)
        print(f"  {tag:>10}: b_sim/g = {bb*100:+.3f}%  (N={w.sum():,})")
    print(f"b_sim/g spans {np.nanmin(bgrid)*100:+.2f}%..{np.nanmax(bgrid)*100:+.2f}%")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
