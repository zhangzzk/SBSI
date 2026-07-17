"""Population-controlled selection bias of a MEASURED-flux cut.

The single-cut test was confounded: a measured-SNR cut and a true-flux cut select different
populations (different true-shape responsivity).  Here we FIX the population: bin detected
galaxies by TRUE flux x TRUE size, then apply the measured-SNR cut WITHIN each bin.  Within a
fixed true-property bin the per-galaxy response is constant, so any residual shift of the mean
true scene-shape along the shear axis is PURELY the shear-correlated selection (the cut drops
shear-elongated, low-surface-brightness galaxies).

  base(bin)   = <s_par>_{bin, all detected}
  m_sel_net   = <s_par>_{all SNR-cut gals} / (cut-count-weighted mean of base(bin)) - 1

m_sel_net != 0  -> genuine selection bias of the measured-flux cut, population-controlled.
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


def load(cat, max_rows, seed):
    """Detected, source-selected sheared sample; returns (s_par, snr, true_mag, true_size)."""
    need = {"e1_input_rot0_p", "e2_input_rot0_p", "gamma1_input_p", "gamma2_input_p",
            "measured_flux_auto", "measured_fluxerr_auto", "r_input_p", "Re_input_p",
            "detected", "distance", "neighbored"}
    res = stream_reservoir(cat, need, max_rows, seed=seed, detected=True, shear_threshold=1e-6)
    g1 = res["gamma1_input_p"].to_numpy(float); g2 = res["gamma2_input_p"].to_numpy(float)
    gm = np.hypot(g1, g2); gh1, gh2 = g1 / gm, g2 / gm
    sc1, sc2 = apply_shear_to_ellipticity(res["e1_input_rot0_p"].to_numpy(float),
                                          res["e2_input_rot0_p"].to_numpy(float), g1, g2)
    s_par = sc1 * gh1 + sc2 * gh2
    snr = res["measured_flux_auto"].to_numpy(float) / res["measured_fluxerr_auto"].to_numpy(float)
    tmag = res["r_input_p"].to_numpy(float)        # NB: r_input_p is a MAGNITUDE (brighter = smaller)
    tsize = res["Re_input_p"].to_numpy(float)
    fin = np.isfinite(s_par) & np.isfinite(snr) & np.isfinite(tmag) & np.isfinite(tsize)
    print(f"  {os.path.basename(cat)}: kept={fin.sum():,}")
    return s_par[fin], snr[fin], tmag[fin], tsize[fin]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalogue", required=True)
    ap.add_argument("--nominal-g", type=float, default=0.05)
    ap.add_argument("--n-flux", type=int, default=6)
    ap.add_argument("--n-size", type=int, default=3)
    ap.add_argument("--max-rows", type=int, default=20_000_000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    s_par, snr, tmag, tsize = load(args.catalogue, args.max_rows, args.seed)
    ef = np.quantile(tmag, np.linspace(0, 1, args.n_flux + 1)); ef[0] -= 1e-9; ef[-1] += 1e-9
    es = np.quantile(tsize, np.linspace(0, 1, args.n_size + 1)); es[0] -= 1e-9; es[-1] += 1e-9
    fi = np.clip(np.digitize(tmag, ef) - 1, 0, args.n_flux - 1)
    si = np.clip(np.digitize(tsize, es) - 1, 0, args.n_size - 1)
    binid = fi * args.n_size + si
    nbin = args.n_flux * args.n_size
    print(f"binned by TRUE flux({args.n_flux}) x size({args.n_size}); g={args.nominal_g}\n")
    print(f"{'keep_f':>7} {'m_sel_net%':>10} {'+/-%':>6}   (measured-SNR cut WITHIN true flux x size bins)")
    for f in [0.8, 0.6, 0.4, 0.2, 0.1]:
        num = 0.0; den = 0.0; nkept = 0
        var = 0.0
        for b in range(nbin):
            m = binid == b
            if m.sum() < 500:
                continue
            base_b = float(np.mean(s_par[m]))
            thr = np.quantile(snr[m], 1 - f)
            cut = m & (snr >= thr)
            nb = int(cut.sum())
            if nb < 100:
                continue
            num += float(np.sum(s_par[cut]))
            den += nb * base_b
            nkept += nb
            var += nb * float(np.var(s_par[cut]))
        msel = num / den - 1.0
        sem = (np.sqrt(var) / nkept) / abs(den / nkept)
        print(f"{f:>7.2f} {msel*100:>+10.2f} {sem*100:>6.2f}")
    print("\nm_sel_net = shear-correlated selection bias of the measured-flux cut, population-controlled.")
    print("NONZERO here = a real selection bias that cutting on TRUE flux would not have.")


if __name__ == "__main__":
    main()
