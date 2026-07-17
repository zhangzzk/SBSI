"""Does cutting on MEASURED flux induce a selection bias that cutting on TRUE flux does not?

A shear-correlated selection (cut on the noisy, sheared measured flux) shifts the kept
sample's mean shape proportionally to shear -> it is a MULTIPLICATIVE bias and therefore
CANCELS in the ratio R(0.02)/R(0.05) that earlier diagnostics used.  To expose it we use the
galaxies PAIRED across the two renders (input_index, shear-independent) and a metacalibration
-style selection response: hold the selected SET fixed vs let it move with shear.

  R_gal = [<e.ghat>_{S,0.05} - <e.ghat>_{S,0.02}] / 0.03     (S fixed = passing at g=0.02)
                                                              -> per-galaxy response, no selection
  R_tot = [<e.ghat>_{pass@0.05,0.05} - <e.ghat>_{pass@0.02,0.02}] / 0.03
                                                              -> set re-selects each shear: + selection
  m_sel = R_tot/R_gal - 1                                    (the selection bias of the cut)

Pairing cancels intrinsic-shape noise in R_gal (same galaxies both shears) -> tight.
CONTROL: a TRUE-flux cut has an identical set at both shears -> R_tot==R_gal -> m_sel~0.
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

from sbs_shear.measurement_model import (  # noqa: E402
    add_measurement_target_features, raw_columns_for_measurement_targets)
from sbs_shear.preprocessing import DEFAULT_SELECTION_CUTS, source_select_selection  # noqa: E402


def load(cat, max_rows, seed):
    rng = np.random.default_rng(seed)
    need = set(raw_columns_for_measurement_targets(["measured_e1_image", "measured_e2_image"]))
    need |= {"input_index", "detected", "gamma1_input_p", "gamma2_input_p",
             "measured_flux_auto", "measured_fluxerr_auto", "r_input_p",
             "Re_input_p", "distance", "neighbored"}
    with ipc.open_file(cat) as r:
        avail = set(r.schema.names)
        cols = sorted(c for c in need if c in avail)
        res = None
        for bi in range(r.num_record_batches):
            b = pa.Table.from_batches([r.get_batch(bi)]).select(cols).to_pandas()
            b = source_select_selection(b, cuts=DEFAULT_SELECTION_CUTS)
            if len(b) == 0:
                continue
            b = b[b["detected"].astype(bool)].reset_index(drop=True)
            gm = np.hypot(b["gamma1_input_p"].to_numpy(float), b["gamma2_input_p"].to_numpy(float))
            b = b[gm > 1e-6].reset_index(drop=True)
            if len(b) == 0:
                continue
            b = b.copy(); b["__k"] = rng.random(len(b))
            res = b if res is None else pd.concat([res, b], ignore_index=True)
            if len(res) > 2 * max_rows:
                res = res.nlargest(max_rows, "__k").reset_index(drop=True)
    if len(res) > max_rows:
        res = res.nlargest(max_rows, "__k").reset_index(drop=True)
    meas = add_measurement_target_features(res.copy())
    g1 = res["gamma1_input_p"].to_numpy(float); g2 = res["gamma2_input_p"].to_numpy(float)
    gm = np.hypot(g1, g2); gh1, gh2 = g1 / gm, g2 / gm
    eproj = meas["measured_e1_image"].to_numpy(float) * gh1 + meas["measured_e2_image"].to_numpy(float) * gh2
    snr = res["measured_flux_auto"].to_numpy(float) / res["measured_fluxerr_auto"].to_numpy(float)
    out = pd.DataFrame(dict(input_index=res["input_index"].to_numpy(), eproj=eproj, snr=snr,
                            tmag=res["r_input_p"].to_numpy(float), gh1=gh1, gh2=gh2))
    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    out = out.drop_duplicates("input_index").reset_index(drop=True)
    print(f"  {os.path.basename(cat)}: detected+unique={len(out):,}")
    return out


def resp(e5, e2):
    return (e5.mean() - e2.mean()) / 0.03, (e5.mean(), e2.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cat-005", required=True)
    ap.add_argument("--cat-002", required=True)
    ap.add_argument("--max-rows", type=int, default=15_000_000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    print("Loading g=0.05 ..."); d5 = load(args.cat_005, args.max_rows, args.seed)
    print("Loading g=0.02 ..."); d2 = load(args.cat_002, args.max_rows, args.seed)
    m = d5.merge(d2, on="input_index", suffixes=("_5", "_2"))
    cosang = (m["gh1_5"] * m["gh1_2"] + m["gh2_5"] * m["gh2_2"]).to_numpy()
    print(f"\nPaired galaxies: {len(m):,}   mean cos(ghat05,ghat02)={cosang.mean():.4f} "
          f"(must be ~1 for the projection to be paired)")
    e5 = m["eproj_5"].to_numpy(float); e2 = m["eproj_2"].to_numpy(float)
    snr5 = m["snr_5"].to_numpy(float); snr2 = m["snr_2"].to_numpy(float)
    tmag = m["tmag_5"].to_numpy(float)

    def block(name, pass5, pass2):
        # R_tot: set re-selects per shear (selection included)
        Rtot, _ = resp(e5[pass5], e2[pass2])
        # R_gal: set FIXED = passing at g=0.02 (paired, no selection response)
        S = pass2
        Rgal, _ = resp(e5[S], e2[S])
        msel = Rtot / Rgal - 1.0
        # bootstrap error on m_sel over the fixed set
        n = int(S.sum())
        sem = np.std(e5[S] - e2[S]) / np.sqrt(max(n, 1)) / 0.03 / abs(Rgal)
        print(f"  {name:>22}  N(@0.02)={int(pass2.sum()):>9,}  Rtot={Rtot:.4f}  Rgal={Rgal:.4f}  "
              f"m_sel={msel*100:+6.2f}% (+/-~{sem*100:.2f})")

    print("\n--- MEASURED-SNR cut (shear-correlated boundary) ---")
    for X in [5, 7, 10, 15, 20]:
        block(f"SNR>{X}", snr5 > X, snr2 > X)
    print("\n--- CONTROL: TRUE-mag cut (shear-independent boundary; expect m_sel~0) ---")
    for q in [0.8, 0.5, 0.2]:
        thr = np.quantile(tmag, q)   # keep brightest q (lower r_input_p)
        keep = tmag <= thr
        block(f"true_mag keep {q:.0%}", keep, keep)


if __name__ == "__main__":
    main()
