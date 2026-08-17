"""PHASE 0c -- compare TRUE-size and MEASURED-size cuts at MATCHED KEEP FRACTION, not at matched
threshold.

WHY THIS IS OWED. Every size statement in this project has carried an unstated caveat: a true cut and
a measured cut written with the same number are not the same cut. `Re_input_p > 0.6"` is INTRINSIC
(pre-PSF) and keeps ~37% of the in-domain population; `measured_flux_radius > 0.60"` is POST-PSF and
keeps ~97%, because the Moffat PSF (FWHM 0.73", beta 2.224) floors measured size at R50 = 0.527" so
almost nothing can land below 0.60" whatever its true size. Reading those two rows against each other
compares a severe cut with a near-no-op and calls the difference a selection effect. Matching on KEEP
FRACTION removes that confound.

WHAT IS COMPARED. For each reference cut, the threshold on the OTHER variable that retains the same
fraction is solved for by quantile, and `m` is reported at both. A difference that survives matched
keep fraction is a real true-vs-measured selection effect; one that does not was a keep-fraction
artefact.

WHY THIS IS CHEAP NOW. It needs no GPU. The near-domain selection table samples the flow's OWN
measured size, which requires scoring; here every cut is FROZEN -- read off the catalogue -- so the
existing per-object dumps already contain everything (`r_sim`, per-seed `R_flow`, tuned `R_blend`).
That is the `[F]` row type of the near-domain table.

  m = r_sim / (R_flow + R_blend) - 1,  the same parameter-free estimator, sign per AGENTS.md.

SEEDS AND ERRORS. The reported quantity is `m`, a bias on the SHAPE response, so the e-response
standard binds: all 16 dom6x6 seeds. The error is formed PER SEED and then spread across seeds --
building the ratio from ensemble means and propagating the numerator alone is a recorded bug that
returns the same error on every row regardless of how aggressive the cut is.

MEASURED SIZE IS THE LEG AVERAGE, `0.5 (plus + minus)`, in arcsec via `PX = 0.2`. Averaging the legs
makes the mask FROZEN -- identical in both legs -- so the selection boundary contributes nothing and
what is compared is purely the response of the retained population. Same construction as the
newcomer/dropout rows.

FIREWALL: nothing is trained, fit or tuned; constgold is evaluation-only. Diagnostic read of dumps.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from plotting.plot_fid_flow_figures import (  # noqa: E402
    CONST_CAT, _read_key_table, load_dumps)

PX = 0.2
PSF_R50 = 0.5268   # Moffat FWHM 0.73", beta 2.224 -- the floor on measured flux_radius


def m_of(mask, rsim, rf, rb):
    """(m_mean, m_err, keep_frac, N) with the ratio formed INSIDE each seed."""
    n = int(mask.sum())
    if n < 1000:
        return np.nan, np.nan, mask.mean(), n
    st = float(np.mean(rsim[mask]))
    bl = float(np.mean(rb[mask]))
    ms = np.array([st / (float(np.mean(rf[s, mask])) + bl) - 1.0 for s in range(rf.shape[0])])
    return float(ms.mean()) * 100, float(ms.std(ddof=1) / np.sqrt(len(ms))) * 100, \
        float(mask.mean()), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--true-cuts", type=float, nargs="*", default=[0.5, 0.6, 0.7])
    ap.add_argument("--meas-cuts", type=float, nargs="*", default=[0.60, 0.70, 0.80])
    ap.add_argument("--save-npz", default="results/keepfrac_matched.npz")
    args = ap.parse_args()

    seeds, ref, flows = load_dumps()
    if not seeds:
        raise SystemExit("REFUSING: no per-object dumps found")
    key = ["case", "input_index"]
    cases = set(ref["case"].unique().tolist())
    ref = ref.merge(_read_key_table(
        CONST_CAT, key + ["Re_input_p", "measured_flux_radius_plus", "measured_flux_radius_minus"],
        cases), on=key, how="left")

    rsim = ref["r_sim"].to_numpy(float)
    rb = ref["R_blend"].to_numpy(float)
    rf = flows.astype(np.float64)
    tsz = ref["Re_input_p"].to_numpy(float)
    msz = 0.5 * (ref["measured_flux_radius_plus"].to_numpy(float)
                 + ref["measured_flux_radius_minus"].to_numpy(float)) * PX

    fin = np.isfinite(tsz) & np.isfinite(msz) & np.isfinite(rsim) & np.isfinite(rb)
    print(f"\nrows with finite true AND measured size: {int(fin.sum()):,} of {len(ref):,}")
    print(f"  true  Re : min {np.nanmin(tsz[fin]):.3f}  med {np.nanmedian(tsz[fin]):.3f}  "
          f"max {np.nanmax(tsz[fin]):.3f} arcsec  (INTRINSIC, pre-PSF)")
    print(f"  meas size: min {np.nanmin(msz[fin]):.3f}  med {np.nanmedian(msz[fin]):.3f}  "
          f"max {np.nanmax(msz[fin]):.3f} arcsec  (POST-PSF; PSF R50 floor = {PSF_R50:.3f})")
    print(f"  fraction with measured size below the PSF floor: "
          f"{100*np.mean(msz[fin] < PSF_R50):.2f}%  <- if ~0, cuts under {PSF_R50:.2f}\" are no-ops")

    base = m_of(fin, rsim, rf, rb)
    print(f"\n{'='*116}\nREFERENCE: no size cut\n{'='*116}")
    print(f"  m = {base[0]:+.3f} +- {base[1]:.3f}%   keep 100.00%   N={base[3]:,}")

    rows = []
    print(f"\n{'='*116}")
    print("MATCHED KEEP FRACTION -- each reference cut, then the OTHER variable's threshold that "
          "keeps the same fraction")
    print(f"{'='*116}")
    print(f"  {'reference cut':>26}{'thr':>8}{'keep%':>9}{'m%':>10}{'+-':>8}   "
          f"{'matched cut':>26}{'thr':>8}{'keep%':>9}{'m%':>10}{'+-':>8}{'dm%':>9}")

    def line(refname, refmask, refthr, othname, othval, higher_is_keep):
        f = float(refmask[fin].mean()) if fin.sum() else np.nan
        # threshold on the OTHER variable retaining the same fraction
        q = np.nanpercentile(othval[fin], 100.0 * (1.0 - f)) if higher_is_keep else np.nan
        omask = fin & (othval > q)
        a = m_of(refmask, rsim, rf, rb)
        b = m_of(omask, rsim, rf, rb)
        print(f"  {refname:>26}{refthr:>8.3f}{100*a[2]:>8.2f}%{a[0]:>10.3f}{a[1]:>8.3f}   "
              f"{othname:>26}{q:>8.3f}{100*b[2]:>8.2f}%{b[0]:>10.3f}{b[1]:>8.3f}"
              f"{b[0]-a[0]:>+9.3f}")
        rows.append((refthr, a[2], a[0], a[1], q, b[2], b[0], b[1]))

    for c in args.true_cuts:
        line(f"TRUE Re > {c:g}\"", fin & (tsz > c), c, "MEASURED size >", msz, True)
    for c in args.meas_cuts:
        line(f"MEASURED size > {c:g}\"", fin & (msz > c), c, "TRUE Re >", tsz, True)

    print(f"\n{'='*116}\nHOW TO READ IT\n{'='*116}")
    print("  The `dm%` column is the matched-keep-fraction cut MINUS the reference cut. It is the")
    print("  honest true-vs-measured comparison: both rows now remove the same fraction of the")
    print("  population, so a surviving difference is a genuine difference between cutting on a")
    print("  true property and cutting on a noisy measurement of it, not a difference in severity.")
    print("\n  Compare against the naive same-NUMBER comparison, which is what every previous size")
    print("  statement used: `TRUE Re > 0.6` against `MEASURED size > 0.60` sets a cut keeping ~37%")
    print("  beside one keeping ~97%. Any difference between those two is dominated by how much")
    print("  they remove, and says nothing about true-vs-measured.")
    print("\n  CAVEAT: matching the keep FRACTION does not match WHICH objects are kept. The two cuts")
    print("  select overlapping but different galaxies -- that is precisely the effect being")
    print("  measured, and it is why this is a comparison, not a correction.")
    np.savez(args.save_npz, rows=np.array(rows, float), base=np.array(base, float),
             seeds=np.array(seeds))
    print(f"\nsaved -> {args.save_npz}")


if __name__ == "__main__":
    main()
