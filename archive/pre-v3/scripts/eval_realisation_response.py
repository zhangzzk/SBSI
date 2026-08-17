"""NON-CIRCULAR acceptance harness for the REALISATION-DEPENDENCE of the modelled shear response.

WHAT IS BEING TESTED
--------------------
`sbs_shear/measurement_model.py:650-652`

    def sample(self, context, n_samples=1, qmc=False):
        s = self.flow.sample(self._flow_ctx(context), n_samples=n_samples, qmc=qmc)
        return s + self._mu(context)[:, None, :]

adds the SAME mean-head shift to every draw of an object, and `_flow_ctx` drops the shape features
from the residual flow.  Consequence: within one object every draw carries an identical shear
response, fixed by its TRUE properties.  The model cannot know that a particular measurement
realisation came out faint or small.  In the sim it very much does: a galaxy whose measurement landed
near the PSF size barely responds to shear.

This script measures that gap on the HALF-SHEAR legs, i.e. on the data the flow was trained on and
NOT on constgold.  Nothing here trains, fits, tunes or selects anything; it scores existing
checkpoints against the det_meas half-shear truth.  The FIREWALL is therefore intact by construction:
no constgold file is opened, no emulator is consulted in the primary scope.

ESTIMATOR
---------
Matched both-detected pairs (g=0 leg <-> g=0.05 leg), true-property acceptance (Re>0.3, mag<26),
per-object random shear direction ghat, gmed = median |gamma|.

  sim, per object      r_sim(i)     = ((e_g - e_0) . ghat) / gmed          (measured ngmix shapes)
  model, per DRAW      r_mod(i,j)   = ((xhat_g(i,j) - xhat_0(i,j)) . ghat) / gmed
                                      with COMMON RANDOM NUMBERS across the two legs, so the flow's
                                      sampling noise cancels in the difference instead of being
                                      amplified by 1/gmed ~ 20x.  FORWARD extraction (0 -> +gmed) on
                                      both sides, matching the sim -- antithetic would reintroduce
                                      the recorded 0.49-vs-0.60 extraction gap.

Both are then averaged inside bins of MEASURED magnitude and MEASURED size.  Two binning modes, and
the difference between them is the entire point:

  MODE A  "identical objects"  -- both sides binned by the SIM's g=0-leg measured mag / size, and the
      model response is averaged over its draws first.  This is the constgold-style diagnostic
      (WORKLOG 2026-07-31i section C).  A true-property model is STRUCTURALLY unable to pass it,
      because the bin conditions on a noise realisation the model never sees.  Reported because it
      is the number the earlier constgold diagnosis quoted, so it is the before/after continuity
      check -- NOT as an acceptance gate.

  MODE B  "self-binned"  -- each catalogue is binned by ITS OWN g=0-leg measured quantity: the sim by
      its measured mag/size, the model by the mag/size IT DREW (dims 2,3 of the same draw whose
      shape response is being accumulated).  This is the acceptance gate.  It is what a real analysis
      does (you only ever see your own catalogue), and it is the one a realisation-aware head can
      actually pass: if the response is allowed to depend on the drawn (mag, log size), then draws
      that land small must carry the collapsed response the sim has there.

  Binning is always on the g=0 LEG, never the sheared leg: a shear-dependent bin label would fold the
  moving-boundary/migration term into what is meant to be a pure response-VALUE test.  The sheared-leg
  variant is reported alongside as a diagnostic so the size of that term is visible.

CONTROLS (printed every run, before any conclusion)
--------------------------------------------------
  * leg noise sharing -- corr(e_0, e_g) and the leg-difference scatter of measured mag / log size
    against the single-leg scatter.  If the two legs did NOT share pixel noise, binning on the g=0
    leg would select a noise realisation that the g=0.05 leg does not have, and the sim-side estimator
    would return the object-average response instead of the realisation-conditional one -- i.e. the
    whole measurement would be vacuous.  This must be checked, not assumed.
  * 45-degree null -- r_sim recomputed with ghat replaced by its spin-2 orthogonal direction
    (gh1,gh2) -> (-gh2,gh1).  Must be consistent with zero in every bin.
  * leg-0 selection contamination -- <e_0 . ghat>/gmed per bin.  ghat is drawn independently of the
    pixel noise so this is zero in expectation whatever the binning, but if it is NOT small the bin
    labels are correlated with the shear direction and the estimator is contaminated.
  * per-object draw scatter of r_mod -- EXACTLY 0 for the current architecture (that is the defect);
    strictly > 0 for any realisation-aware head.  This is the engage / no-engage diagnostic.

SCOPES  (both are valid model tests here -- see the R_blend note)
------
  ISOLATED (default): no brighter true neighbour within --iso-radius (7").
  ALL (--all-too): every matched both-detected in-domain row.  This is the population closest to the
      one the constgold diagnosis used.

  WHY NO R_blend IS ADDED ON EITHER SCOPE.  Each galaxy in the half-shear sim carries its OWN random
  shear direction, and the estimator projects on the PRIMARY's direction ghat_p.  A neighbour's
  contribution enters along its own ghat_s, which is uncorrelated with ghat_p, so it averages to zero
  -- that is the whole design of this ruler (see scripts/dump_halfshear_selfresp.py).  The response
  measured here is therefore the SELF response on both scopes, and R_flow alone is the right model
  side.  Measured confirmation on the ALL scope, job 15422516: no-cut R_sim = +0.7202 vs R_model =
  +0.7181, i.e. -0.30%, with no blend term added anywhere.  --blend-lookup exists for a differently
  constructed catalogue; on THESE legs using it would DOUBLE-COUNT and it should be left off.

FIREWALL / INTEGRITY
--------------------
No constgold path is read.  No offset, scale factor or pasted constant is applied to any number
below.  The constgold reference values from WORKLOG 2026-07-31i are printed once, in a block that
says so, purely so the reader can compare; they enter no computation.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.feather as pf
import torch

SBSI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
for _p in (SBSI_ROOT, SCRIPTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sbs_shear.measurement_model import load_measurement_model  # noqa: E402
from sbs_shear.preprocessing import rescale  # noqa: E402
from sbs_shear.shear_map import apply_shear_to_ellipticity  # noqa: E402
from eval_selection_response import build_base, CAT, RK  # noqa: E402
from eval_selfresp_gap import CROWD, NN  # noqa: E402

PIX = 0.2  # arcsec / pixel; measured_flux_radius is in pixels, all size bins are in ARCSEC

# constgold reference numbers, WORKLOG 2026-07-31i.  QUOTED ONLY -- never used in a computation here.
CG_REF = (
    "REFERENCE (constgold, WORKLOG 2026-07-31i; quoted for comparison, used in NO computation here):\n"
    "    measured size < 0.60\":  R_sim 0.0743   R_model 0.3982   -> model/sim-1 = +436%  (5.4x)\n"
    "    measured mag 26.00-26.25: model/sim-1 = +35.07%\n"
    "    measured mag 26.25-26.50: model/sim-1 = +46.23%\n"
    "    measured mag below 26   : model/sim-1 within -3.6% .. +2.5%\n"
    "    NOTE the two setups differ: constgold is antithetic +/-0.02 with a summed R_blend added to\n"
    "    the model side, this is forward 0 -> 0.05 with the neighbour term nulled by the ghat_p\n"
    "    projection instead.  Expect the SHAPE of the signature to reproduce, not the digits."
)

DEF_SIZE_EDGES = [0.50, 0.60, 0.70, 0.80, 0.90, 1.00, 1.20, 1.50]      # arcsec, measured
DEF_MAG_EDGES = [23.0, 24.0, 24.5, 25.0, 25.25, 25.5, 25.75, 26.0, 26.25, 26.5, 27.0]


# --------------------------------------------------------------------------------------- binning
def edge_labels(edges, unit=""):
    lab = [f"<{edges[0]:g}{unit}"]
    for a, b in zip(edges[:-1], edges[1:]):
        lab.append(f"{a:g}-{b:g}{unit}")
    lab.append(f">{edges[-1]:g}{unit}")
    return lab


def digitize(values, edges):
    """0..nb-1 bin index; nb (= trash) for non-finite. nb = len(edges)+1 real bins."""
    nb = len(edges) + 1
    v = np.asarray(values, dtype=np.float64)
    idx = np.digitize(v, np.asarray(edges, dtype=np.float64), right=False)
    idx = np.where(np.isfinite(v), idx, nb).astype(np.int64)
    return idx, nb


def _acc(idx, w, nb):
    """bincount into nb real bins + 1 trash bin, returning (sum, count) over the real bins."""
    s = np.bincount(idx.ravel(), weights=np.asarray(w, dtype=np.float64).ravel(), minlength=nb + 1)
    c = np.bincount(idx.ravel(), minlength=nb + 1)
    return s[:nb], c[:nb]


# ------------------------------------------------------------------------------------ model side
@torch.no_grad()
def model_binned(bundles, base, gh1, gh2, gmed, sel, simA, edges, n_samples, batch_size,
                 flow_seed, chunk, qmc=False, t0=None, rblend=None):
    """Accumulate the model's per-draw response into MODE A and MODE B bins, per checkpoint.

    simA: dict with the SIM's g=0-leg bin indices ('sz','mg') for every row of `base` (mode A labels).
    edges: dict with 'sz' (log-arcsec edges) and 'mg' (mag edges) plus their bin counts.

    Returns per-bundle dicts of {A_sz,A_mg,B_sz,B_mg,B_szS,B_mgS} -> (sum, count) and the no-cut
    accumulators, plus the mean per-object across-draw scatter of r_mod (0 for a true-property model).
    """
    idx_all = np.where(sel)[0]
    intr1 = base["e1_input_rot0_p"].to_numpy(float)
    intr2 = base["e2_input_rot0_p"].to_numpy(float)
    nbz, nbm = edges["nsz"], edges["nmg"]

    keys = [("A_sz", nbz), ("A_mg", nbm), ("B_sz", nbz), ("B_mg", nbm),
            ("B_szS", nbz), ("B_mgS", nbm)]
    accs = [{k: [np.zeros(n), np.zeros(n)] for k, n in keys} for _ in bundles]
    nocutA = [[0.0, 0] for _ in bundles]     # per-object mean response, no binning
    nocutB = [[0.0, 0] for _ in bundles]     # per-draw response, no binning
    scat = [[0.0, 0] for _ in bundles]       # mean over objects of std_j(r_mod(i,j))
    logpix = float(np.log(PIX))

    def build_frame(ci, s):
        fr = base.iloc[ci].reset_index(drop=True).copy()
        e1s, e2s = apply_shear_to_ellipticity(intr1[ci], intr2[ci], s * gh1[ci], s * gh2[ci])
        fr["e1_input_rot0_p"] = e1s
        fr["e2_input_rot0_p"] = e2s
        return rescale(fr, **RK)

    for cs in range(0, len(idx_all), chunk):
        ci = idx_all[cs:cs + chunk]
        g1c = gh1[ci][:, None]
        g2c = gh2[ci][:, None]
        fr0 = build_frame(ci, 0.0)
        frg = build_frame(ci, +gmed)
        seed = flow_seed + cs                     # identical for both legs of this chunk -> CRN
        aidx_sz = simA["sz"][ci]
        aidx_mg = simA["mg"][ci]
        for k, bundle in enumerate(bundles):
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            d0 = bundle.sample(fr0, n_samples=n_samples, batch_size=batch_size, qmc=qmc)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            dg = bundle.sample(frg, n_samples=n_samples, batch_size=batch_size, qmc=qmc)
            p0 = d0[:, :, 0] * g1c + d0[:, :, 1] * g2c
            pg = dg[:, :, 0] * g1c + dg[:, :, 1] * g2c
            r = (pg - p0) / gmed                                        # (n, ns) per-draw response
            fin = np.isfinite(r)
            r0 = np.where(fin, r, 0.0)

            # ---- MODE A: average the draws first, then bin by the SIM's g=0 measured label
            cnt = fin.sum(axis=1)
            robj = np.where(cnt > 0, r0.sum(axis=1) / np.maximum(cnt, 1), np.nan)
            ok = cnt > 0
            # across-draw scatter (engage / no-engage diagnostic). Computed as the centred rms, NOT
            # <r^2>-<r>^2: for the current architecture the two legs' flow residuals cancel EXACTLY
            # under CRN so the true value is 0, and the uncentred form would return sqrt of a
            # catastrophic cancellation (~1e-4) and look spuriously engaged.
            if n_samples > 1:
                dev = np.where(fin, r0 - np.nan_to_num(robj)[:, None], 0.0)
                sd = np.sqrt((dev ** 2).sum(axis=1) / np.maximum(cnt, 1))
                scat[k][0] += float(np.nansum(sd[ok]))
                scat[k][1] += int(ok.sum())
            if rblend is not None:
                # per-object scalar; adds no draw dependence, only the level. Applied AFTER the
                # scatter diagnostic on purpose.
                rbc = rblend[ci][:, None]
                r0 = np.where(fin, r0 + rbc, 0.0)
                robj = np.where(ok, robj + rblend[ci], np.nan)
            nocutA[k][0] += float(np.nansum(robj[ok]))
            nocutA[k][1] += int(ok.sum())
            for key, idxs in (("A_sz", aidx_sz), ("A_mg", aidx_mg)):
                lab = np.where(ok, idxs, edges["n" + key[2:]])
                s_, c_ = _acc(lab, np.where(ok, robj, 0.0), edges["n" + key[2:]])
                accs[k][key][0] += s_
                accs[k][key][1] += c_

            # ---- MODE B: bin every DRAW by the mag / log-size that same draw produced
            nocutB[k][0] += float(r0.sum())
            nocutB[k][1] += int(fin.sum())
            for tag, dd in (("", d0), ("S", dg)):     # "" = g=0 leg (default), "S" = sheared leg
                magd = dd[:, :, 2]
                lszd = dd[:, :, 3] + logpix           # ln(flux_radius px) -> ln(arcsec)
                iz, _ = digitize(np.where(fin, lszd, np.nan), edges["sz"])
                im, _ = digitize(np.where(fin, magd, np.nan), edges["mg"])
                s_, c_ = _acc(iz, r0, nbz)
                accs[k]["B_sz" + tag][0] += s_
                accs[k]["B_sz" + tag][1] += c_
                s_, c_ = _acc(im, r0, nbm)
                accs[k]["B_mg" + tag][0] += s_
                accs[k]["B_mg" + tag][1] += c_
        if t0 is not None:
            print(f"    chunk {cs//chunk + 1}/{int(np.ceil(len(idx_all)/chunk))} "
                  f"({len(ci):,} rows)  ({time.time()-t0:.1f}s)", flush=True)
    return accs, nocutA, nocutB, scat


# -------------------------------------------------------------------------------------- printing
def report_axis(name, labels, sim, mods, nocut_sim, nocut_mod, out, prefix, seed_names,
                sim_null=None, sim_cont=None, extra_note=""):
    """sim = (R, sem, N); mods = list over ckpts of (R_b array, count array); nocut_* likewise."""
    Rs, es, Ns = sim
    S = len(mods)
    nb = len(labels)
    Rm = np.full((S, nb), np.nan)
    Cm = np.zeros((S, nb))
    for k, (rb, cb) in enumerate(mods):
        Rm[k] = rb
        Cm[k] = cb
    Rm_mean = np.nanmean(Rm, axis=0)
    Rm_sd = np.nanstd(Rm, axis=0, ddof=1) if S > 1 else np.zeros(nb)
    Rm_sem = Rm_sd / np.sqrt(S) if S > 1 else np.zeros(nb)

    # per-seed ratios, then the spread across seeds (never ratios of ensemble means)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = Rm / Rs[None, :] - 1.0                     # model/sim - 1
        mval = Rs[None, :] / Rm - 1.0                      # m = sim/model - 1
        rho_sim = Rs / nocut_sim[0]
        rho_mod = Rm / np.array(nocut_mod)[:, None]
        rho_rel = rho_sim[None, :] / rho_mod - 1.0         # seed level offset cancels here
    def ms(a):
        m = np.nanmean(a, axis=0)
        s = (np.nanstd(a, axis=0, ddof=1) / np.sqrt(S)) if S > 1 else np.zeros(nb)
        return m, s
    ratio_m, ratio_s = ms(ratio)
    m_m, m_s = ms(mval)
    rho_m, rho_s = ms(rho_rel)

    ftot = Ns.sum()
    fmod = Cm.mean(axis=0)
    fmod = fmod / max(fmod.sum(), 1)

    print(f"\n  --- {name} ---{extra_note}")
    hdr = (f"  {'bin':>14} {'N_sim':>10} {'f_sim':>7} {'f_mod':>7} | {'R_sim':>9} {'+-':>7} "
           f"{'R_model':>9} {'+-seed':>7} | {'mod/sim-1':>10} {'+-':>7} | {'m=sim/mod-1':>12} {'+-':>7} "
           f"| {'rho_sim/rho_mod-1':>18} {'+-':>7}")
    print(hdr)
    for b in range(nb):
        if Ns[b] == 0:
            continue
        print(f"  {labels[b]:>14} {int(Ns[b]):>10,} {Ns[b]/ftot:>7.4f} {fmod[b]:>7.4f} | "
              f"{Rs[b]:>+9.4f} {es[b]:>7.4f} {Rm_mean[b]:>+9.4f} {Rm_sem[b]:>7.4f} | "
              f"{ratio_m[b]*100:>+9.2f}% {ratio_s[b]*100:>6.2f}% | "
              f"{m_m[b]*100:>+11.2f}% {m_s[b]*100:>6.2f}% | "
              f"{rho_m[b]*100:>+17.2f}% {rho_s[b]*100:>6.2f}%")
    print(f"  {'ALL':>14} {int(ftot):>10,} {1.0:>7.4f} {1.0:>7.4f} | "
          f"{nocut_sim[0]:>+9.4f} {nocut_sim[1]:>7.4f} {np.mean(nocut_mod):>+9.4f} "
          f"{(np.std(nocut_mod, ddof=1)/np.sqrt(S) if S>1 else 0.0):>7.4f} | "
          f"{(np.mean(np.array(nocut_mod)/nocut_sim[0]-1))*100:>+9.2f}% {'':>6} | "
          f"{(np.mean(nocut_sim[0]/np.array(nocut_mod)-1))*100:>+11.2f}% {'':>6} | {'--':>17} {'':>6}")

    if sim_null is not None or sim_cont is not None:
        print(f"  {'CONTROLS':>14} {'':>10} {'':>7} {'':>7} | "
              f"{'45deg null':>9} {'+-':>7} {'<e0.ghat>/g':>12} {'+-':>7}")
        for b in range(nb):
            if Ns[b] == 0:
                continue
            n_, ne_ = (sim_null[0][b], sim_null[1][b]) if sim_null is not None else (np.nan, np.nan)
            c_, ce_ = (sim_cont[0][b], sim_cont[1][b]) if sim_cont is not None else (np.nan, np.nan)
            print(f"  {labels[b]:>14} {'':>10} {'':>7} {'':>7} | {n_:>+9.4f} {ne_:>7.4f} "
                  f"{c_:>+12.4f} {ce_:>7.4f}")

    out[prefix + "_label"] = np.array(labels)
    out[prefix + "_N"] = Ns
    out[prefix + "_R_sim"] = Rs
    out[prefix + "_R_sim_err"] = es
    out[prefix + "_R_model_seed"] = Rm
    out[prefix + "_frac_model"] = fmod
    out[prefix + "_ratio"] = ratio_m
    out[prefix + "_ratio_err"] = ratio_s
    out[prefix + "_m"] = m_m
    out[prefix + "_m_err"] = m_s
    out[prefix + "_rhorel"] = rho_m
    out[prefix + "_rhorel_err"] = rho_s
    out[prefix + "_seeds"] = np.array(seed_names)
    if sim_null is not None:
        out[prefix + "_null"] = sim_null[0]
        out[prefix + "_null_err"] = sim_null[1]
    if sim_cont is not None:
        out[prefix + "_cont"] = sim_cont[0]
        out[prefix + "_cont_err"] = sim_cont[1]
    return dict(ratio=ratio_m, ratio_err=ratio_s, R_sim=Rs, R_model=Rm_mean, N=Ns, labels=labels)


def sim_binned(idx, nb, vals):
    """(mean, sem, count) of `vals` in each of nb bins given the precomputed bin index."""
    s = np.bincount(idx, weights=vals, minlength=nb + 1)[:nb]
    q = np.bincount(idx, weights=vals ** 2, minlength=nb + 1)[:nb]
    c = np.bincount(idx, minlength=nb + 1)[:nb].astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        mu = s / c
        var = np.maximum(q / c - mu ** 2, 0.0)
        sem = np.sqrt(var / np.maximum(c, 1))
    mu[c == 0] = np.nan
    sem[c == 0] = np.nan
    return mu, sem, c


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", nargs="+", default=None)
    ap.add_argument("--ckpt-glob", default=None)
    ap.add_argument("--g0-leg", default=CAT + "det_meas_ngmix_g0.0_train.feather")
    ap.add_argument("--gS-leg", default=CAT + "det_meas_ngmix_g0.05_val.feather")
    ap.add_argument("--max-case", type=int, default=39)
    ap.add_argument("--true-re-min", type=float, default=0.3)
    ap.add_argument("--true-mag-max", type=float, default=26.0)
    ap.add_argument("--iso-radius", type=float, default=7.0)
    ap.add_argument("--size-edges", type=float, nargs="+", default=DEF_SIZE_EDGES,
                    help="MEASURED size bin edges in ARCSEC (flux_radius px * 0.2)")
    ap.add_argument("--mag-edges", type=float, nargs="+", default=DEF_MAG_EDGES,
                    help="MEASURED mag_auto bin edges")
    ap.add_argument("--n-samples", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=16384)
    ap.add_argument("--chunk", type=int, default=100_000)
    ap.add_argument("--flow-seed", type=int, default=12345)
    ap.add_argument("--qmc", action="store_true")
    ap.add_argument("--all-too", action="store_true",
                    help="also run the ALL-objects scope (DIAGNOSTIC: R_blend missing unless "
                         "--blend-lookup is given, so the level is expected to be low)")
    ap.add_argument("--blend-lookup", default=None,
                    help="optional per-(case,input_index) feather with an R_blend column, added to "
                         "the model response. Only meaningful for the ALL scope.")
    ap.add_argument("--tf32", action="store_true")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    ckpts = list(args.ckpt or [])
    if args.ckpt_glob:
        ckpts += sorted(glob.glob(args.ckpt_glob))
    if not ckpts:
        ap.error("give --ckpt and/or --ckpt-glob")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.tf32:
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8:
            print("!! --tf32 is a NO-OP on this device (needs compute capability >= 8.0)", flush=True)
    t0 = time.time()
    print("=" * 110)
    print("REALISATION-RESPONSE HARNESS  (half-shear legs; NO constgold is read anywhere in this run)")
    print("=" * 110)
    print(f"device={device}  ckpts={len(ckpts)}  n_samples={args.n_samples}  qmc={args.qmc}")
    for c in ckpts:
        print("   ", os.path.basename(c))
    print()
    print(CG_REF)
    print()

    ru = build_base(args.g0_leg, args.gS_leg, args.max_case, args.true_re_min, args.true_mag_max,
                    args.iso_radius, CROWD, NN, t0)
    base, gh1, gh2, gmed, iso = ru["base"], ru["gh1"], ru["gh2"], ru["gmed"], ru["iso"]

    e1_0 = base["measured_ngmix_g1_0"].to_numpy(float)
    e2_0 = base["measured_ngmix_g2_0"].to_numpy(float)
    e1_g = base["measured_ngmix_g1_g"].to_numpy(float)
    e2_g = base["measured_ngmix_g2_g"].to_numpy(float)
    mag0 = base["measured_mag_auto_0"].to_numpy(float)
    magg = base["measured_mag_auto_g"].to_numpy(float)
    rad0 = base["measured_flux_radius_0"].to_numpy(float) * PIX      # arcsec
    radg = base["measured_flux_radius_g"].to_numpy(float) * PIX

    p0 = e1_0 * gh1 + e2_0 * gh2
    pg = e1_g * gh1 + e2_g * gh2
    # spin-2 orthogonal (45 deg) direction: (gh1,gh2) -> (-gh2,gh1)
    q0 = -e1_0 * gh2 + e2_0 * gh1
    qg = -e1_g * gh2 + e2_g * gh1
    r_sim = (pg - p0) / gmed
    r_null = (qg - q0) / gmed
    r_cont = p0 / gmed

    good = np.isfinite(r_sim) & np.isfinite(mag0) & np.isfinite(rad0) & (rad0 > 0)
    print(f"\nrows with finite response AND finite g=0 measured mag/size: {int(good.sum()):,} / "
          f"{len(base):,}  ({good.mean():.4%})")

    # ------------------------------------------------------------------ CONTROL 1: leg noise sharing
    m = good & np.isfinite(magg) & np.isfinite(radg) & (radg > 0)
    c1 = float(np.corrcoef(e1_0[m], e1_g[m])[0, 1])
    c2 = float(np.corrcoef(e2_0[m], e2_g[m])[0, 1])
    dmag = magg[m] - mag0[m]
    dlsz = np.log(radg[m]) - np.log(rad0[m])
    print("\n  --- CONTROL 1: do the two legs share pixel noise? ---")
    print(f"    corr(e1_0, e1_g) = {c1:+.5f}     corr(e2_0, e2_g) = {c2:+.5f}")
    print(f"    std(mag_g - mag_0)      = {np.std(dmag):.5f}   vs single-leg std(mag_0) = {np.std(mag0[m]):.5f}"
          f"   ratio = {np.std(dmag)/np.std(mag0[m]):.5f}")
    print(f"    std(dlog size)          = {np.std(dlsz):.5f}   vs single-leg std        = {np.std(np.log(rad0[m])):.5f}"
          f"   ratio = {np.std(dlsz)/np.std(np.log(rad0[m])):.5f}")
    print(f"    std(e1_g - e1_0)        = {np.std(e1_g[m]-e1_0[m]):.5f}   vs single-leg std        = {np.std(e1_0[m]):.5f}")
    print("    INTERPRETATION: corr ~ 1 and a leg-difference scatter far below the single-leg scatter")
    print("    => the legs share the noise realisation => binning on the g=0 leg IS realisation-")
    print("    conditional and the sim-side estimator measures what this harness claims. corr ~ 0")
    print("    => it does NOT, and every bin below would collapse to the object-average response.")

    scopes = [("ISOLATED (no brighter true nbr within %.0f\")" % args.iso_radius, iso & good)]
    if args.all_too:
        scopes.append(("ALL matched both-detected in-domain rows%s"
                       % (" [+R_blend from --blend-lookup -- DOUBLE COUNTS on half-shear legs]"
                          if args.blend_lookup else ""), good))

    rblend = np.zeros(len(base))
    if args.blend_lookup:
        bl = pf.read_table(args.blend_lookup).to_pandas()
        j = base[["case", "input_index"]].merge(bl, on=["case", "input_index"], how="left")
        rblend = j["R_blend"].to_numpy(float)
        print(f"\n  blend lookup {os.path.basename(args.blend_lookup)}: match "
              f"{np.isfinite(rblend).mean():.2%}  <R_blend>={np.nanmean(rblend):.4f}")
        rblend = np.nan_to_num(rblend, nan=0.0)

    bundles = [load_measurement_model(c, device=device) for c in ckpts]
    names = [bundle.target_transform.target_names for bundle in bundles]
    for nm in names:
        if list(nm[:2]) != ["measured_ngmix_g1", "measured_ngmix_g2"] or len(nm) < 4:
            raise SystemExit(f"unexpected target order {nm}; this harness needs (g1,g2,mag,logR)")
    seed_names = [os.path.basename(c) for c in ckpts]

    edges = dict(sz=np.log(np.asarray(args.size_edges, float)),
                 mg=np.asarray(args.mag_edges, float))
    edges["nsz"] = len(args.size_edges) + 1
    edges["nmg"] = len(args.mag_edges) + 1
    sz_labels = edge_labels(args.size_edges, '"')
    mg_labels = edge_labels(args.mag_edges)

    out = dict(gmed=gmed, ckpts=np.array(seed_names), size_edges=np.asarray(args.size_edges, float),
               mag_edges=np.asarray(args.mag_edges, float), n_samples=args.n_samples,
               pixel_size=PIX, leg_corr_e1=c1, leg_corr_e2=c2,
               g0_leg=args.g0_leg, gS_leg=args.gS_leg, max_case=args.max_case)

    summary = {}
    nocut_ref = {}
    for scope_name, sel in scopes:
        print("\n" + "=" * 110)
        print(f"SCOPE: {scope_name}    N = {int(sel.sum()):,}")
        print("=" * 110, flush=True)
        tag = "iso" if sel is scopes[0][1] else "all"

        # --------------------------------------------------------------- sim side (once, no seeds)
        isz, nbz = digitize(np.where(sel, np.log(np.maximum(rad0, 1e-12)), np.nan), edges["sz"])
        img, nbm = digitize(np.where(sel, mag0, np.nan), edges["mg"])
        iszS, _ = digitize(np.where(sel & np.isfinite(radg) & (radg > 0),
                                    np.log(np.maximum(radg, 1e-12)), np.nan), edges["sz"])
        imgS, _ = digitize(np.where(sel, magg, np.nan), edges["mg"])
        rs = np.where(sel, r_sim, 0.0)
        rn = np.where(sel, r_null, 0.0)
        rc = np.where(sel, r_cont, 0.0)
        sim_sz = sim_binned(isz, nbz, rs)
        sim_mg = sim_binned(img, nbm, rs)
        sim_szS = sim_binned(iszS, nbz, rs)
        sim_mgS = sim_binned(imgS, nbm, rs)
        null_sz = sim_binned(isz, nbz, rn)
        null_mg = sim_binned(img, nbm, rn)
        cont_sz = sim_binned(isz, nbz, rc)
        cont_mg = sim_binned(img, nbm, rc)
        nsel = int(sel.sum())
        R_sim_all = float(r_sim[sel].mean())
        R_sim_all_e = float(r_sim[sel].std() / np.sqrt(nsel))
        nocut_ref[tag] = R_sim_all
        print(f"  sim no-cut R_sim = {R_sim_all:+.4f} +- {R_sim_all_e:.4f}   "
              f"(per-object scatter std = {r_sim[sel].std():.4f})")

        # --------------------------------------------------------------------------- model side
        simA = dict(sz=isz, mg=img)
        accs, nocutA, nocutB, scat = model_binned(
            bundles, base, gh1, gh2, gmed, sel, simA, edges, args.n_samples, args.batch_size,
            args.flow_seed, args.chunk, qmc=args.qmc, t0=t0,
            rblend=(rblend if args.blend_lookup else None))

        def mods(key):
            o = []
            for k in range(len(bundles)):
                s_, c_ = accs[k][key]
                with np.errstate(divide="ignore", invalid="ignore"):
                    rb = s_ / c_
                rb[c_ == 0] = np.nan
                o.append((rb, c_))
            return o

        ncA = [a[0] / max(a[1], 1) for a in nocutA]
        ncB = [b[0] / max(b[1], 1) for b in nocutB]
        dscat = [s[0] / max(s[1], 1) for s in scat]
        print(f"\n  model no-cut  <R_model>(mode A, per object) = {np.mean(ncA):+.4f}  "
              f"(seed sd {np.std(ncA, ddof=1) if len(ncA)>1 else 0.0:.4f})")
        print(f"  model no-cut  <R_model>(mode B, per draw)   = {np.mean(ncB):+.4f}")
        print(f"  CONTROL 4: mean per-object ACROSS-DRAW scatter of r_model = "
              f"{np.mean(dscat):.3e}   <- EXACTLY 0 for a true-property-only response head "
              f"(the defect); > 0 for a realisation-aware head")
        if args.blend_lookup:
            print(f"  (per-object R_blend added to every draw of that object, mean over this scope "
                  f"{float(np.mean(rblend[sel])):+.4f}; it is a scalar with no draw dependence, so it "
                  f"cannot carry realisation structure -- only the level)")

        summary[tag] = {}
        summary[tag]["A_sz"] = report_axis(
            "MODE A  size axis  (both sides binned by the SIM's g=0 measured size; model draws "
            "averaged first)", sz_labels, sim_sz, mods("A_sz"),
            (R_sim_all, R_sim_all_e), ncA, out, f"{tag}_A_sz", seed_names,
            sim_null=(null_sz[0], null_sz[1]), sim_cont=(cont_sz[0], cont_sz[1]),
            extra_note="  [DIAGNOSTIC: a true-property model cannot pass this by construction]")
        summary[tag]["A_mg"] = report_axis(
            "MODE A  mag axis   (both sides binned by the SIM's g=0 measured mag)",
            mg_labels, sim_mg, mods("A_mg"), (R_sim_all, R_sim_all_e), ncA, out,
            f"{tag}_A_mg", seed_names, sim_null=(null_mg[0], null_mg[1]),
            sim_cont=(cont_mg[0], cont_mg[1]),
            extra_note="  [DIAGNOSTIC]")
        summary[tag]["B_sz"] = report_axis(
            "MODE B  size axis  (ACCEPTANCE GATE: sim binned by its own g=0 measured size, model "
            "binned by the size IT DREW)", sz_labels, sim_sz, mods("B_sz"),
            (R_sim_all, R_sim_all_e), ncB, out, f"{tag}_B_sz", seed_names)
        summary[tag]["B_mg"] = report_axis(
            "MODE B  mag axis   (ACCEPTANCE GATE)", mg_labels, sim_mg, mods("B_mg"),
            (R_sim_all, R_sim_all_e), ncB, out, f"{tag}_B_mg", seed_names)
        summary[tag]["BS_sz"] = report_axis(
            "MODE B  size axis, SHEARED-leg binning (diagnostic: difference vs the g=0-leg table "
            "above is the migration/boundary term)", sz_labels, sim_szS, mods("B_szS"),
            (R_sim_all, R_sim_all_e), ncB, out, f"{tag}_BS_sz", seed_names)
        summary[tag]["BS_mg"] = report_axis(
            "MODE B  mag axis, SHEARED-leg binning (diagnostic)", mg_labels, sim_mgS,
            mods("B_mgS"), (R_sim_all, R_sim_all_e), ncB, out, f"{tag}_BS_mg", seed_names)

        out[f"{tag}_N"] = nsel
        out[f"{tag}_R_sim_nocut"] = R_sim_all
        out[f"{tag}_R_sim_nocut_err"] = R_sim_all_e
        out[f"{tag}_R_model_nocut_A"] = np.array(ncA)
        out[f"{tag}_R_model_nocut_B"] = np.array(ncB)
        out[f"{tag}_draw_scatter"] = np.array(dscat)

    # ---------------------------------------------------------------------------- headline block
    print("\n" + "=" * 110)
    print("HEADLINE -- does the harness reproduce the known signature on INDEPENDENT (half-shear) data?")
    print("=" * 110)
    for tag in summary:
        for mode in ("A", "B"):
            s = summary[tag][f"{mode}_sz"]
            lab, N, Rs_, Rm_, ra = s["labels"], s["N"], s["R_sim"], s["R_model"], s["ratio"]
            small = [b for b in range(len(lab)) if N[b] > 0 and (b <= 1)]
            if small:
                num = float(np.nansum(Rs_[small] * N[small])) / max(float(N[small].sum()), 1)
                den = float(np.nansum(Rm_[small] * N[small])) / max(float(N[small].sum()), 1)
                # A ratio is NOT reported here: the sim response in this bin is consistent with zero,
                # so model/sim is an arbitrarily large number set by the denominator's noise. The
                # honest scale-free statement is the ABSOLUTE gap in units of the population mean.
                print(f"  [{tag}] MODE {mode}  measured size < {args.size_edges[1]:g}\" "
                      f"(N={int(N[small].sum()):,}, {N[small].sum()/N.sum():.2%}): "
                      f"R_sim={num:+.4f}  R_model={den:+.4f}  gap={den-num:+.4f} "
                      f"= {(den-num)/nocut_ref[tag]*100:+.1f}% of the population mean R_sim="
                      f"{nocut_ref[tag]:.4f}")
            sm = summary[tag][f"{mode}_mg"]
            for want in ("26-26.25", "26.25-26.5"):
                if want in sm["labels"]:
                    b = list(sm["labels"]).index(want)
                    if sm["N"][b] > 0:
                        print(f"  [{tag}] MODE {mode}  measured mag {want}: "
                              f"R_sim={sm['R_sim'][b]:+.4f}  R_model={sm['R_model'][b]:+.4f}  "
                              f"model/sim-1 = {sm['ratio'][b]*100:+.2f}%")

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        np.savez(args.output, **out)
        print(f"\nsaved {args.output}")
    print("REALISATION_RESPONSE_DONE", flush=True)


if __name__ == "__main__":
    main()
