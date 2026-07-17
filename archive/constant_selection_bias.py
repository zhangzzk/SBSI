"""[QUARANTINED 2026-06-30 — DO NOT USE the by-population numbers.]

KNOWN BUGS / LIMITATIONS (review 2026-06-30), do not quote results from this script:
  1. R_gal baseline is INCONSISTENT — the main table averages `response` over the S/N_plus-cut
     population while the tercile block uses the uncut tercile, and neither is the population
     basis of R_total; this biases R_sel = R_total - R_gal at the sub-effect level.
  2. The constant-shear render that feeds this (r_max=10) made ~99.99% of objects `neighbored`
     (all-blended), so it is NOT a representative population.
  3. `et`/`response` here are a metacal/distortion-convention responsivity (~0.46, ~2x the
     epsilon-convention 0.23); only used as the ratio m_sel internally, but do not compare to
     the 0.23-convention quantities elsewhere.
The trusted measured-flux-cut selection bias is `selection_cut_binned.py` (population-controlled,
-1.5..-3.9%). Revisit this script only after re-rendering the constant set with r_max~3 and
fixing R_gal to a single consistent population basis.

Measured-flux-cut SELECTION bias, measured on the constant-shear (antithetic) set.

The constant_response_catalogue pairs each galaxy at +g and -g along a FIXED axis, so the
intrinsic shape cancels in the difference -> a clean metacalibration-style selection response.

For a measured-S/N cut X:
  R_gal   = <response_i>                         over the cut sample (per-galaxy response; no selection)
  R_total = [<et_plus>_{S/N+>X} - <et_minus>_{S/N->X}] / (2g)   (set re-selects per sign -> + selection)
  R_sel   = R_total - R_gal                      (the selection response of the cut)
  m_sel   = R_total/R_gal - 1                    (the multiplicative bias if you DON'T correct)

Correcting = calibrate with R_total (= R_gal + R_sel) instead of R_gal -> m -> 0.  Validates,
on the gold antithetic set, the -1.5..-3.9% bias the population-controlled test found.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc

BASE = "/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/"


def load(cat, max_rows):
    need = ["et_plus", "et_minus", "response", "S/N_plus", "S/N_minus",
            "applied_g1", "applied_g2", "r_input_p", "Re_input_p"]
    with ipc.open_file(cat) as r:
        avail = set(r.schema.names); cols = [c for c in need if c in avail]
        parts = []; n = 0
        for bi in range(r.num_record_batches):
            b = pa.Table.from_batches([r.get_batch(bi)]).select(cols).to_pandas()
            parts.append(b); n += len(b)
            if n >= max_rows:
                break
    return pd.concat(parts, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalogue", default=BASE + "constant_response_catalogue_train.feather")
    ap.add_argument("--max-rows", type=int, default=12_000_000)
    args = ap.parse_args()

    print("*** QUARANTINED: R_gal baseline is inconsistent and the constant set is all-blended; "
          "by-population m_sel numbers are UNRELIABLE. Use selection_cut_binned.py instead. ***")
    df = load(args.catalogue, args.max_rows)
    g = float(np.median(np.hypot(df["applied_g1"], df["applied_g2"])))
    etp = df["et_plus"].to_numpy(float); etm = df["et_minus"].to_numpy(float)
    resp = df["response"].to_numpy(float)
    snp = df["S/N_plus"].to_numpy(float); snm = df["S/N_minus"].to_numpy(float)
    flux = df["r_input_p"].to_numpy(float)
    fin = np.isfinite(etp) & np.isfinite(etm) & np.isfinite(resp) & np.isfinite(snp) & np.isfinite(snm)
    etp, etm, resp, snp, snm, flux = etp[fin], etm[fin], resp[fin], snp[fin], snm[fin], flux[fin]
    print(f"constant set: N={len(etp):,}  |g|={g:.4f}  R_gal(no cut)=<response>={resp.mean():.4f}")

    print(f"\n{'S/N cut':>8} {'keep_f':>7} {'R_gal':>8} {'R_total':>8} {'R_sel':>8} {'m_sel%':>8} {'+/-%':>6}")
    def row(label, kp, km, kgal):
        Rg = resp[kgal].mean()
        Rt = (etp[kp].mean() - etm[km].mean()) / (2 * g)
        msel = Rt / Rg - 1.0
        # error: dominant from the two cut means
        ep = etp[kp].std() / np.sqrt(kp.sum()); em = etm[km].std() / np.sqrt(km.sum())
        eR = np.sqrt(ep**2 + em**2) / (2 * g)
        sem = abs(msel + 1) * (eR / abs(Rt))
        print(f"{label:>8} {kp.mean():>7.2f} {Rg:>8.4f} {Rt:>8.4f} {Rt-Rg:>+8.4f} {msel*100:>+8.2f} {sem*100:>6.2f}")

    row("none", np.ones(len(etp), bool), np.ones(len(etp), bool), np.ones(len(etp), bool))
    for X in [5, 7, 10, 15, 20, 30]:
        kp = snp > X; km = snm > X
        row(f">{X}", kp, km, kp)
    print("\nm_sel = uncorrected multiplicative bias of the measured-S/N cut (clean antithetic).")
    print("Correction: calibrate with R_total (=R_gal+R_sel) instead of R_gal -> m -> 0.")
    # by true-flux tercile (bright/mid/faint), S/N>10
    print("\nby TRUE-flux tercile at S/N>10 (selection bias varies with brightness):")
    ef = np.quantile(flux, [0, 1/3, 2/3, 1.0]); ef[0] -= 1e-6; ef[-1] += 1e-6
    for t, lab in enumerate(["bright", "mid", "faint"]):
        mt = (flux >= ef[t]) & (flux < ef[t+1])
        kp = mt & (snp > 10); km = mt & (snm > 10)
        if kp.sum() > 5000 and km.sum() > 5000:
            Rg = resp[mt].mean(); Rt = (etp[kp].mean() - etm[km].mean())/(2*g)
            print(f"  {lab:>6}: m_sel = {(Rt/Rg-1)*100:+.2f}%  (N_cut={int(kp.sum()):,})")


if __name__ == "__main__":
    main()
