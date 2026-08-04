"""Calibrate the V2.1 true-property S/N proxy, and report the population the V2.1 domain leaves.

TWO JOBS, both prerequisites for retraining anything on V2.1.

(1) CALIBRATE `sbs_shear.domain.sn_true`. The proxy is the standard CCD noise equation on TRUE
    flux and TRUE size (see that module's docstring):

        S/N = flux / sqrt(sky_var * N_pix + flux / gain)

    with two constants that are FITTED here against the catalogue's own measured S/N
    (`measured_flux_auto / measured_fluxerr_auto`), so that "S/N > 10" means a real S/N of 10
    rather than 10 in arbitrary units. Both are physically interpretable -- `sky_var` comes out at
    ~5.4 x PIXEL_RMS^2, which is the Kron-vs-half-light APERTURE AREA RATIO, and `gain` sets the
    source shot-noise term. Fitted and reported together with their residual scatter: the case
    AGENTS.md "Numerical Integrity" explicitly allows.

    THE SLOPE IS THE GATE, NOT A FORMALITY. We regress log10(S/N_meas) on log10(S/N_true) with a
    FREE slope. If the proxy's shape is right the slope is 1 and only the constants matter; if it
    is not, no choice of constants can rescue it and the FORM must change. That check earned its
    keep: the sky-limited version of this proxy (no `flux / gain` term) returned slope 0.826 with
    the residual sliding monotonically from +0.12 dex at the faint end to -0.08 at the bright end.
    Adding the source term moved the slope to 1.004.

    FAR-NEIGHBOUR ONLY, because a close neighbour contaminates `flux_auto` and its Kron aperture,
    which would fold blending into constants meant to describe a SINGLE galaxy. Note there is no
    "isolated" subsample to use instead: this catalogue is one row per object with its nearest
    neighbour annotated, so `neighbored` is True for 100.0% of rows and `~neighbored` selects
    essentially nothing (22 rows in 262k -- an earlier version of this script did exactly that and
    had to be fixed). The stand-in is `distance > --iso-distance` (default 3", ~6 PSF half-light
    radii), which keeps ~25% of the catalogue.

    SELECTION CAVEAT, stated because it bounds what this fit can mean: only DETECTED rows have a
    measured S/N, so the faint end is a detected subsample and the fit there is biased high. The
    calibration is therefore taken above `--sn-fit-min` and the per-bin table shows where it holds.

(2) REPORT THE POPULATION. Retraining on a domain that keeps too few rows is a wasted GPU day, and
    a domain that overlaps V2 only partly cannot be compared to it row-for-row. So: survival of
    each cut separately and together, the V2 (mag<26, Re>0.3) comparison, and the overlap.

FIREWALL: reads the g=0 TRAINING catalogue only. constgold is never opened here.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sbs_shear import domain as D  # noqa: E402
from sbs_shear.paths import catalogue  # noqa: E402

COLS = ["case", "detected", "distance", "r_input_p", "Re_input_p",
        "measured_flux_auto", "measured_fluxerr_auto"]

# V2 domain, for the side-by-side comparison only.
V2_MAG_MAX, V2_RE_MIN = 26.0, 0.3

_LN10 = np.log(10.0)


def fit_sn_constants(flux, n_pix, sn_meas, iters=50):
    """Fit (sky_var, gain) of S/N = flux / sqrt(sky_var*N_pix + flux/gain) to measured S/N.

    Fitted in log10(S/N), which is the right metric: the scatter is multiplicative (a fixed
    FRACTIONAL error on S/N), so a linear-space fit would let the few brightest objects dominate.

    Pure numpy on purpose -- no scipy. `sims1` cannot import scipy on the LOGIN node
    (GLIBCXX_3.4.30, via sklearn), and this script must be runnable there for smoke tests.

    Two stages:
      * INITIALISE by an exact linear least squares. 1/(S/N)^2 = sky_var*(N_pix/flux^2) +
        (1/gain)*(1/flux) is LINEAR in the two unknowns, so the starting point costs one lstsq
        and needs no guess.
      * REFINE by damped Gauss-Newton in log10(S/N), with the parameters carried as logs so they
        cannot go negative.
    """
    y = np.log10(sn_meas)
    # --- linear initialisation in 1/(S/N)^2 space ---
    A = np.c_[n_pix / flux ** 2, 1.0 / flux]
    coef, *_ = np.linalg.lstsq(A, 1.0 / sn_meas ** 2, rcond=None)
    a, b = float(coef[0]), float(coef[1])          # a = sky_var, b = 1/gain
    if not (a > 0 and b > 0):                      # degenerate start -> fall back to sky-only
        a = max(a, np.median(1.0 / sn_meas ** 2 * flux ** 2 / n_pix))
        b = max(b, 1e-6)
    u, v = np.log(a), np.log(b)

    # --- Gauss-Newton in log10(S/N) ---
    prev = np.inf
    for _ in range(iters):
        a, b = np.exp(u), np.exp(v)
        var = a * n_pix + b * flux
        r = np.log10(flux) - 0.5 * np.log10(var) - y
        cost = float(np.mean(r ** 2))
        if not np.isfinite(cost) or abs(prev - cost) < 1e-14:
            break
        prev = cost
        # d r / d(log a), d r / d(log b)
        j = np.c_[-0.5 / _LN10 * (a * n_pix) / var, -0.5 / _LN10 * (b * flux) / var]
        step, *_ = np.linalg.lstsq(j, -r, rcond=None)
        step = np.clip(step, -0.5, 0.5)            # damping: keep it in the trust region
        u += float(step[0]); v += float(step[1])
    a, b = float(np.exp(u)), float(np.exp(v))
    return a, 1.0 / b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalogue",
                    default=catalogue("det_meas_crowd_conc_g0.0_train_full.feather"))
    ap.add_argument("--max-batches", type=int, default=0,
                    help="0 = whole catalogue. Small values are for smoke tests ONLY: the "
                         "constants are inherited by everything downstream, so they are taken "
                         "from the full read.")
    ap.add_argument("--sn-fit-min", type=float, default=8.0,
                    help="measured-S/N floor for the calibration fit -- keeps it out of the "
                         "detection-incomplete regime where only lucky-noise rows survive.")
    ap.add_argument("--sn-fit-max", type=float, default=200.0)
    ap.add_argument("--iso-distance", type=float, default=3.0,
                    help="arcsec; keep only rows whose nearest neighbour is at least this far, so "
                         "the constants describe a single galaxy rather than a blend.")
    ap.add_argument("--fit-per-batch-cap", type=int, default=20000,
                    help="rows kept per batch for the fit; the population counts stay exact.")
    args = ap.parse_args()

    print(f"### V2.1 DOMAIN CALIBRATION ###   {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"catalogue: {args.catalogue}")
    print(f"PSF: FWHM={D.PSF_FWHM}\"  beta={D.MOFFAT_BETA}  ->  Re_psf={D.PSF_RE:.4f}\" "
          f"({D.PSF_RE / D.PIXEL_SIZE:.3f} px)")
    print(f"target domain: Re > {D.V21_RE_MIN}\" ({D.V21_RE_MIN / D.PIXEL_SIZE:g} px, "
          f"R = {float(D.resolution(D.V21_RE_MIN)):.3f}),  sn_true > {D.V21_SN_MIN}")

    # Pass 1 collects only the calibration sample. The population counts need the fitted constants,
    # which do not exist yet, so they all happen in pass 2.
    fit_flux, fit_npix, fit_sn = [], [], []
    n_raw = n_fin = 0
    rng = np.random.default_rng(0)

    with ipc.open_file(args.catalogue) as reader:
        have = set(reader.schema.names)
        missing = [c for c in COLS if c not in have]
        if missing:
            raise RuntimeError(f"catalogue is missing required columns: {missing}")
        nb = reader.num_record_batches
        if args.max_batches:
            nb = min(nb, args.max_batches)
            print(f"!! SMOKE TEST: {nb} of {reader.num_record_batches} batches -- the constants "
                  f"printed below MUST NOT be adopted !!")
        for bi in range(nb):
            b = pa.Table.from_batches([reader.get_batch(bi)]).select(COLS).to_pandas()
            n_raw += len(b)
            mag = b["r_input_p"].to_numpy(float)
            re = b["Re_input_p"].to_numpy(float)
            fin = np.isfinite(mag) & np.isfinite(re)
            n_fin += int(fin.sum())

            det = b["detected"].to_numpy(bool)
            dist = b["distance"].to_numpy(float)
            f = b["measured_flux_auto"].to_numpy(float)
            fe = b["measured_fluxerr_auto"].to_numpy(float)
            good = (fin & det & np.isfinite(dist) & (dist > args.iso_distance)
                    & np.isfinite(f) & np.isfinite(fe) & (f > 0) & (fe > 0))
            if not good.any():
                continue
            snm = np.where(good, f / np.where(fe > 0, fe, np.nan), np.nan)
            use = good & (snm > args.sn_fit_min) & (snm < args.sn_fit_max)
            idx = np.where(use)[0]
            if idx.size == 0:
                continue
            if idx.size > args.fit_per_batch_cap:
                idx = rng.choice(idx, args.fit_per_batch_cap, replace=False)
            fit_flux.append(10.0 ** (-0.4 * (mag[idx] - D.ZERO_MAG)))
            fit_npix.append(D.aperture_pixels(re[idx]))
            fit_sn.append(snm[idx])
            if (bi + 1) % 100 == 0:
                print(f"  ... {bi + 1}/{nb} batches, {n_raw:,} rows", flush=True)

    flux = np.concatenate(fit_flux)
    n_pix = np.concatenate(fit_npix)
    sn_meas = np.concatenate(fit_sn)
    print(f"\nrows read: {n_raw:,}   finite true props: {n_fin:,}")
    print(f"calibration sample (nbr > {args.iso_distance:g}\", detected, {args.sn_fit_min:g} < "
          f"S/N_meas < {args.sn_fit_max:g}): {sn_meas.size:,}")

    # --- (1) the fit ---
    sky_var, gain = fit_sn_constants(flux, n_pix, sn_meas)
    sn_pred = flux / np.sqrt(sky_var * n_pix + flux / gain)
    lx, ly = np.log10(sn_pred), np.log10(sn_meas)
    slope, offset = np.polyfit(lx, ly, 1)
    resid = ly - lx
    print("\n--- CALIBRATION ---")
    print(f"  SN_SKY_VAR = {sky_var:.5f}   (= {sky_var / D.PIXEL_RMS ** 2:.3f} x PIXEL_RMS^2 "
          f"-> Kron/half-light aperture AREA ratio)")
    print(f"  SN_GAIN    = {gain:.4f}")
    print(f"  free-slope check: log10(S/N_meas) = {slope:.4f} * log10(S/N_true) + {offset:.4f}")
    print(f"  residual log10(S/N_meas / S/N_true): mean {np.mean(resid):+.4f}, "
          f"sd {np.std(resid, ddof=1):.4f} dex ({100 * (10 ** np.std(resid, ddof=1) - 1):.1f}%)")
    if abs(slope - 1.0) > 0.05:
        print(f"  !! SLOPE IS OFF BY {abs(slope - 1) * 100:.1f}% -- the proxy's SHAPE is wrong and "
              f"no choice of constants fixes it. Do NOT adopt these; change the FORM in "
              f"sbs_shear/domain.py (this is exactly how the missing source-noise term was found).")
    else:
        print(f"  slope is within {abs(slope - 1) * 100:.1f}% of 1 -> the form holds; adopt these "
              f"two constants in sbs_shear/domain.py with this run's job id and date.")

    print("\n  residual by S/N_true  [0 = proxy is right; the bin containing 10 is the one that "
          "sets the cut]")
    qs = np.quantile(lx, np.linspace(0, 1, 9))
    print(f"    {'S/N_true range':>22} {'N':>10} {'median resid':>14} {'sd':>8}")
    for lo, hi in zip(qs[:-1], qs[1:]):
        m = (lx >= lo) & (lx < hi)
        if m.sum() < 100:
            continue
        print(f"    {10 ** lo:9.1f} - {10 ** hi:<9.1f} {int(m.sum()):>10,} "
              f"{np.median(resid[m]):>14.4f} {np.std(resid[m], ddof=1):>8.4f}")

    # --- (2) the population ---
    print("\n--- POPULATION (second pass with the fitted constants) ---")
    n_fin = n_re = n_sn = n_v21 = n_v2 = n_both = 0
    with ipc.open_file(args.catalogue) as reader:
        nb = (min(reader.num_record_batches, args.max_batches) if args.max_batches
              else reader.num_record_batches)
        for bi in range(nb):
            b = pa.Table.from_batches([reader.get_batch(bi)]).select(
                ["r_input_p", "Re_input_p"]).to_pandas()
            mag = b["r_input_p"].to_numpy(float)
            re = b["Re_input_p"].to_numpy(float)
            fin = np.isfinite(mag) & np.isfinite(re)
            n_fin += int(fin.sum())
            re_ok = fin & (re > D.V21_RE_MIN)
            sn_ok = fin & (D.sn_true(mag, re, sky_var=sky_var, gain=gain) > D.V21_SN_MIN)
            v21 = re_ok & sn_ok
            v2 = fin & (mag < V2_MAG_MAX) & (re > V2_RE_MIN)
            n_re += int(re_ok.sum()); n_sn += int(sn_ok.sum())
            n_v21 += int(v21.sum()); n_v2 += int(v2.sum()); n_both += int((v21 & v2).sum())

    def pct(a):
        return f"{a:,} ({100.0 * a / max(n_fin, 1):.2f}%)"

    print(f"  finite true props         {pct(n_fin)}")
    print(f"  Re > {D.V21_RE_MIN}\" only           {pct(n_re)}")
    print(f"  sn_true > {D.V21_SN_MIN:g} only         {pct(n_sn)}")
    print(f"  V2.1 domain (both)        {pct(n_v21)}")
    print(f"  V2 domain (mag<26,Re>.3)  {pct(n_v2)}")
    print(f"  in BOTH V2 and V2.1       {pct(n_both)}")
    if n_v21:
        print(f"  V2.1 is {100.0 * n_v21 / max(n_v2, 1):.1f}% the size of V2; "
              f"{100.0 * n_both / n_v21:.2f}% of V2.1 lies inside V2")

    print("\n  the S/N cut as a magnitude limit:")
    for re in (0.5, 0.7, 1.0, 1.5):
        print(f"    Re = {re:.1f}\" (R = {float(D.resolution(re)):.3f})  ->  mag < "
              f"{float(D.sn_limiting_mag(re, sky_var=sky_var, gain=gain)):.3f}")

    print(f"\nADOPT: SN_SKY_VAR = {sky_var:.5f}   SN_GAIN = {gain:.4f}")
    print("DOMAIN_V21_DONE", flush=True)


if __name__ == "__main__":
    main()
