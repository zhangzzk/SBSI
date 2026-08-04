"""Score the BlendEMU per-pair blending response against half-shear TRUTH, on the deliverable domain.

WHY THIS EXISTS
---------------
`eval_selfresp_gap.py` clears R_flow (self-response error -0.05% over 5.9M galaxies, WORKLOG
2026-07-28h). Since in-domain constgold `m = R_sim/(R_flow + R_blend) - 1` is -0.508%, the residual
must live in R_blend or the population transfer. R_blend comes from the BlendEMU emulator
`lsst_r_extnbr_ho`, whose training cuts are a SUPERSET of our domain (primary mag 18-28 vs our <26,
primary Re 0.1-1.5 vs our >0.3). That is defensible for a per-pair CONDITIONAL regression -- unlike
the response pin, there is no population-mean to get wrong -- but its accuracy INSIDE our domain has
never been measured. This measures it.

THE TRUTH
---------
The half-shear catalogue is a 2x2 design over (primary sheared, neighbour sheared):

    neither 30.4% | primary-only 30.4% | neighbour-only 19.6% | BOTH 19.6%

`eval_selfresp_gap.py` uses the primary-only leg (self-response). The obvious choice here would be
the neighbour-only leg -- but **ngmix and galsim shapes were never run on the legs where the primary
is unsheared** (every SExtractor column is populated there; `measured_ngmix_g1/g2` and
`measured_galsim_g1/g2` are 0% finite). Only SExtractor second moments (a/b/theta) survive there, and
those are the superseded estimator, so using them would measure a different quantity than `m`.

So this uses the BOTH-sheared leg, which does have ngmix. It works because the primary's and the
neighbour's shear directions are INDEPENDENT there (measured: mean cos = -0.0000, sd = 0.7071, i.e.
a uniform random relative angle). Projecting the measured-shape shift on the NEIGHBOUR's direction
therefore kills the self-response and keeps the blend response:

    <de . ghat_s> = <R_self * g_p * (ghat_p . ghat_s)> + <R_blend * g_s>
                  =            0                       + R_blend * g_s

    R_blend_truth = ((e1_both - e1_0) * ghat1_s + (e2_both - e2_0) * ghat2_s) / |g_s|

with the g=0 leg as the unsheared reference, matched by (case, input_index). There is exactly one
neighbour row per primary and each neighbour has its own random direction, so this is a per-PAIR
quantity -- the same granularity the emulator predicts at.

NULL TEST: the same projection onto ghat_s rotated by 45 degrees (spin-2 orthogonal) must be
consistent with zero. If it is not, the decorrelation argument above has failed and the central
number must not be trusted.

FIREWALL: half-shear legs only. constgold is never read, so nothing here can tune on the eval set.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.feather as pf
from sbs_shear.paths import CATALOGUES as CAT

BLEND_MODELS = "/home/z/Zekang.Zhang/blendemu/models"
# Rendering conditions of this sim set; the emulator rescales its features with these.
COND = dict(pixel_size=0.2, zero_point=30.0, psf_fwhm=0.73, moffat_beta=2.224, pixel_rms=0.312)

PAIR_FEATURES = ["Re_input_p", "r_input_p", "sersic_n_input_p",
                 "Re_input_s", "r_input_s", "sersic_n_input_s", "distance"]
NGMIX = ["measured_ngmix_g1", "measured_ngmix_g2"]
GAMMA = ["gamma1_input_p", "gamma2_input_p", "gamma1_input_s", "gamma2_input_s"]


def load_legs(gs_leg, g0_leg, max_case, re_min, mag_max, verbose=True, extra_cols=None):
    """BOTH-sheared rows of the sheared leg, matched to the unsheared leg (see module docstring:
    the neighbour-only leg has no ngmix shapes, so the blend response is recovered from the
    both-sheared leg by projecting on the neighbour's independent shear direction).

    `extra_cols` pulls additional columns from the sheared leg -- e.g. the neighbour's sky position,
    needed to key a join against blendemu's response catalogue.
    """
    t0 = time.time()
    cols = ["case", "input_index", "detected", "neighbored"] + PAIR_FEATURES + NGMIX + GAMMA
    for c in (extra_cols or []):
        if c not in cols:
            cols.append(c)
    d = pf.read_table(gs_leg, columns=cols).to_pandas()
    if max_case is not None:
        d = d[d["case"] < max_case]
    gp = np.hypot(d["gamma1_input_p"].to_numpy(float), d["gamma2_input_p"].to_numpy(float))
    gsv = np.hypot(d["gamma1_input_s"].to_numpy(float), d["gamma2_input_s"].to_numpy(float))
    nb = d[(gp > 1e-6) & (gsv > 1e-6)].copy()           # <-- the BOTH-sheared leg
    if verbose:
        print(f"both-sheared leg: N={len(nb):,}  cases={nb['case'].nunique()}  ({time.time()-t0:.1f}s)",
              flush=True)

    # deliverable domain: cut the PRIMARY on true properties; neighbours stay full-population
    nb = nb[(nb["Re_input_p"].to_numpy(float) > re_min)
            & (nb["r_input_p"].to_numpy(float) < mag_max)]
    nb = nb[nb["detected"].astype(bool)].drop_duplicates(["case", "input_index"])

    ref = pf.read_table(g0_leg, columns=["case", "input_index", "detected"] + NGMIX).to_pandas()
    if max_case is not None:
        ref = ref[ref["case"] < max_case]
    ref = ref[ref["detected"].astype(bool)].drop_duplicates(["case", "input_index"])

    base = nb.merge(ref[["case", "input_index"] + NGMIX], on=["case", "input_index"],
                    suffixes=("_g", "_0"))
    if verbose:
        print(f"matched both-detected, in-domain: N={len(base):,}  ({time.time()-t0:.1f}s)", flush=True)
    return base


def blend_truth(base, rotate45=False):
    """Per-pair blending response: primary's measured-shape shift projected on the NEIGHBOUR's shear.

    rotate45=True projects on the spin-2 orthogonal direction instead, which contains NO blend
    signal -- the null test that validates the decorrelation argument.
    """
    g1 = base["gamma1_input_s"].to_numpy(float)
    g2 = base["gamma2_input_s"].to_numpy(float)
    gm = np.hypot(g1, g2)
    h1, h2 = g1 / gm, g2 / gm
    if rotate45:
        h1, h2 = -h2, h1
    de1 = base["measured_ngmix_g1_g"].to_numpy(float) - base["measured_ngmix_g1_0"].to_numpy(float)
    de2 = base["measured_ngmix_g2_g"].to_numpy(float) - base["measured_ngmix_g2_0"].to_numpy(float)
    return (de1 * h1 + de2 * h2) / gm


def table(truth, pred, key, edges, label, name):
    print(f"\n[{label}]")
    print(f"  {name:>16} {'truth':>9} {'emulator':>9} {'diff':>9} {'emu/truth-1 %':>14} "
          f"{'sem':>8} {'N':>10}")
    good = np.isfinite(truth) & np.isfinite(pred)
    a, b = truth[good].mean(), pred[good].mean()
    sem = truth[good].std(ddof=1) / np.sqrt(good.sum())
    print(f"  {'OVERALL':>16} {a:>9.4f} {b:>9.4f} {b-a:>+9.4f} "
          f"{(b/a-1)*100 if a else np.nan:>+14.2f} {sem:>8.4f} {good.sum():>10,}")
    for i in range(len(edges) - 1):
        m = good & (key >= edges[i]) & (key < edges[i + 1])
        if m.sum() < 200:
            continue
        a, b = truth[m].mean(), pred[m].mean()
        sem = truth[m].std(ddof=1) / np.sqrt(m.sum())
        print(f"  [{edges[i]:>6.2f},{edges[i+1]:>6.2f}) {a:>9.4f} {b:>9.4f} {b-a:>+9.4f} "
              f"{(b/a-1)*100 if a else np.nan:>+14.2f} {sem:>8.4f} {m.sum():>10,}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gs-leg", default=CAT + "det_meas_ngmix_g0.05_val.feather")
    ap.add_argument("--g0-leg", default=CAT + "det_meas_ngmix_g0.0_train.feather")
    ap.add_argument("--max-case", type=int, default=None, help="cap cases (default: all)")
    ap.add_argument("--true-re-min", type=float, default=0.3)
    ap.add_argument("--true-mag-max", type=float, default=26.0)
    ap.add_argument("--tag", default="lsst_r_extnbr_ho", help="BlendEMU emulator tag")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    base = load_legs(args.gs_leg, args.g0_leg, args.max_case,
                     args.true_re_min, args.true_mag_max)
    truth = blend_truth(base)
    null = blend_truth(base, rotate45=True)
    g = np.isfinite(null)
    nm, nsem = null[g].mean(), null[g].std(ddof=1) / np.sqrt(g.sum())
    print(f"\nNULL TEST (45-deg rotated projection, must be ~0): {nm:+.5f} +- {nsem:.5f} "
          f"({abs(nm)/nsem:.1f} sigma)  N={g.sum():,}")
    if abs(nm) > 3 * nsem:
        print("  *** NULL FAILS -- the self-response is NOT averaging out; do not trust the numbers "
              "below. ***")

    sys.path.insert(0, "/home/z/Zekang.Zhang/blendemu")
    from blendemu.inference import BlendingPredictor
    pred_model = BlendingPredictor.load(BLEND_MODELS, tag=args.tag, conditions=COND, device="cpu")
    out = pred_model.predict_on_pairs(base[PAIR_FEATURES].copy(), task="response")
    pred = out["response"].to_numpy(float)

    print(f"\nemulator tag={args.tag}   <truth>={np.nanmean(truth):+.4f}   <emulator>={np.nanmean(pred):+.4f}")
    dist = base["distance"].to_numpy(float)
    size = base["Re_input_p"].to_numpy(float)
    mag = base["r_input_p"].to_numpy(float)
    table(truth, pred, dist, [0, 1, 2, 3, 4, 5, 7, 10], "by PAIR SEPARATION (arcsec)", "distance")
    table(truth, pred, size, [0.30, 0.38, 0.50, 0.75, 1.50], "by PRIMARY TRUE SIZE", "Re_input_p")
    table(truth, pred, mag, [18, 22, 23, 24, 25, 26], "by PRIMARY TRUE MAG", "r_input_p")

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        np.savez(args.output, truth=truth, pred=pred, distance=dist, size=size, mag=mag,
                 case=base["case"].to_numpy(int), input_index=base["input_index"].to_numpy(int))
        print(f"\nsaved {args.output}")
    print("RBLEND_GAP_DONE", flush=True)


if __name__ == "__main__":
    main()
