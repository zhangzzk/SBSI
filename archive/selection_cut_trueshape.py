"""Selection bias of a MEASURED-flux cut vs a TRUE-flux cut -- robust single-catalogue method.

No pairing, no second shear (the 0.02/0.05 renders use independent shear directions, so a
paired cross-shear test is invalid).  Within ONE sheared catalogue we measure how a cut shifts
the kept sample's mean TRUE scene-shape projected on the per-object shear direction:

  s_par      = S_gamma(e_intrinsic) . ghat            (true sheared shape along the shear axis)
  base       = <s_par> over all detected              (~ response in true-shape units)
  b(cut)     = <s_par>_cut - base                     (selection-induced shape shift of the cut)
  m_sel(cut) = b(cut) / base                          (selection bias as a fractional/mult. bias)

Using the TRUE shape needs no response calibration.  A measured-flux/SNR cut is shear-
correlated (shear-elongated galaxies have lower surface brightness -> dropped) so b!=0; a
TRUE-flux cut is shear-independent so b~0 (the control).  m_sel is the multiplicative shear
bias that an uncorrected measured-flux cut would inject.
"""
import argparse
import os
import sys

import numpy as np

SBSI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SBSI_ROOT not in sys.path:
    sys.path.insert(0, SBSI_ROOT)

from sbs_shear.shear_map import apply_shear_to_ellipticity  # noqa: E402
from sbs_shear.sim_stream import stream_reservoir  # noqa: E402


def load(cat, max_rows, seed):
    """Detected, source-selected sheared sample; returns (s_par, snr, true_mag)."""
    need = {"e1_input_rot0_p", "e2_input_rot0_p", "gamma1_input_p", "gamma2_input_p",
            "measured_flux_auto", "measured_fluxerr_auto", "r_input_p", "Re_input_p",
            "detected", "distance", "neighbored"}
    res = stream_reservoir(cat, need, max_rows, seed=seed, detected=True, shear_threshold=1e-6)
    g1 = res["gamma1_input_p"].to_numpy(float); g2 = res["gamma2_input_p"].to_numpy(float)
    gm = np.hypot(g1, g2); gh1, gh2 = g1 / gm, g2 / gm
    e1i = res["e1_input_rot0_p"].to_numpy(float); e2i = res["e2_input_rot0_p"].to_numpy(float)
    sc1, sc2 = apply_shear_to_ellipticity(e1i, e2i, g1, g2)
    s_par = sc1 * gh1 + sc2 * gh2
    snr = res["measured_flux_auto"].to_numpy(float) / res["measured_fluxerr_auto"].to_numpy(float)
    tmag = res["r_input_p"].to_numpy(float)        # NB: r_input_p is a MAGNITUDE (brighter = smaller)
    fin = np.isfinite(s_par) & np.isfinite(snr) & np.isfinite(tmag)
    print(f"  {os.path.basename(cat)}: detected kept={fin.sum():,}  <|g|>={gm[fin].mean():.4f}")
    return s_par[fin], snr[fin], tmag[fin]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalogue", required=True)
    ap.add_argument("--nominal-g", type=float, default=0.05)
    ap.add_argument("--max-rows", type=int, default=20_000_000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    s_par, snr, tmag = load(args.catalogue, args.max_rows, args.seed)
    base = float(np.mean(s_par)); g = args.nominal_g
    sem0 = float(np.std(s_par) / np.sqrt(len(s_par)))
    print(f"\nbaseline <s_par>_detected = {base:.5f} +/- {sem0:.5f}  (true-shape response ~ base/g = {base/g:.3f})")
    print(f"{'keep_f':>7} | {'MEASURED-SNR cut':>28} | {'TRUE-flux cut (control)':>28}")
    print(f"{'':>7} | {'b':>9} {'m_sel%':>8} {'+/-%':>6} | {'b':>9} {'m_sel%':>8} {'+/-%':>6}")
    for f in [0.8, 0.6, 0.4, 0.2, 0.1]:
        # measured-SNR cut: keep brightest fraction f
        thrS = np.quantile(snr, 1 - f); kS = snr >= thrS
        bS = float(np.mean(s_par[kS])) - base
        eS = float(np.std(s_par[kS]) / np.sqrt(kS.sum()))
        mS = bS / base; emS = abs(mS) * np.sqrt((eS / bS) ** 2 + (sem0 / base) ** 2) if bS != 0 else 0
        # true-flux cut: keep brightest fraction f (lower r_input_p)
        thrT = np.quantile(tmag, f); kT = tmag <= thrT
        bT = float(np.mean(s_par[kT])) - base
        eT = float(np.std(s_par[kT]) / np.sqrt(kT.sum()))
        mT = bT / base; emT = abs(mT) * np.sqrt((eT / bT) ** 2 + (sem0 / base) ** 2) if bT != 0 else 0
        print(f"{f:>7.2f} | {bS:>+9.5f} {mS*100:>+8.2f} {emS*100:>6.2f} | "
              f"{bT:>+9.5f} {mT*100:>+8.2f} {emT*100:>6.2f}")
    print("\nm_sel = selection-induced multiplicative shear bias of the cut.")
    print("MEASURED-SNR cut should be NONZERO (shear-correlated); TRUE-flux cut ~0 (control).")


if __name__ == "__main__":
    main()
