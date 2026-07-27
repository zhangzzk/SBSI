"""SELECTION bias on constgold (+/-0.02), measured with the UNSHEARED INTRINSIC shape.

THE ESTIMATOR (owner's spec)
----------------------------
1. PAIR the detections between the +0.02 and the -0.02 leg on (case, input_index). constgold
   renders the same galaxies with the same noise seed at opposite shear, so a paired object is
   the SAME galaxy in both legs; its true properties are byte-identical across legs (verified).
2. SELECT separately in each leg, on that leg's OWN measured observable -- so the cut boundary
   MOVES with shear. This is the whole point: pass_plus and pass_minus are different sets.
3. Average the UNSHEARED INTRINSIC ellipticity of whatever survived, per leg.
4. R_sel = ( <e_int>_plus - <e_int>_minus ) / (2 * 0.02)

Why the intrinsic shape is the right numerator: e_int is a fixed property of the galaxy,
IDENTICAL in both legs. It carries no shear response at all. So every bit of R_sel comes from
WHICH galaxies the cut let through -- it is a pure moving-boundary selection term, with the shape
response and measurement noise algebraically removed rather than subtracted. Two consequences:

  * NO CUT (or any cut on a TRUE, non-shearing property) gives R_sel == 0 EXACTLY, not
    approximately, because pass_plus and pass_minus are then the same set. That is a built-in null
    test, and it is why no "R(cut)/R(nocut)-1" ratio is needed -- the raw number already IS the bias.
  * R_sel is in units of shear response, so dividing by the total measured-shape response R_total
    (computed in-run on the same sample, the certified constgold ~0.45) turns it straight into a
    multiplicative bias in percent.

constgold's shear is exactly (+0.02, 0) and (-0.02, 0), so ghat = (1,0) and 2g = 0.04 exactly.

MEASURED COLUMNS. The constgold per-leg catalogues shipped without SExtractor photometry
(measured_e1/e2 and S/N only), so a moving measured cut was impossible. `build_constgold_measured.py`
restores measured_mag_auto + measured_flux_radius per leg from the raw Shapes catalogues using the
builder's own detection<->input matching (row-exact; row counts match the catalogues exactly).

MODEL (Gold-V2, 8-seed ensemble). The same estimator, but the measured observables come from the
flow instead of the sim. The flow is P(measured | TRUE props, blending) and is never conditioned on
g: its shear response arises ONLY by perturbing the truth. So for each object we shear its INTRINSIC
ellipticity by -g and by +g, push both through the flow, draw measured (shape, mag, log-size), apply
the SAME cut to the draws, and weight that object's intrinsic e by the fraction of draws that pass:

    <e_int>_leg = sum_i e_int,i * n_pass(i,leg) / sum_i n_pass(i,leg)
    R_model     = ( <e_int>_plus - <e_int>_minus ) / (2g)

Common random numbers (same torch seed for both legs) so the sampling noise cancels in the
difference. Ensemble = mean over the 8 seeds; error bar = seed-to-seed spread.

FIREWALL: nothing is trained or fitted here. constgold is EVAL-ONLY -- the flow was trained on the
det_meas half-shear legs and has never seen it.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.ipc as ipc
import pyarrow.feather as pf
import torch

SBSI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SBSI_ROOT not in sys.path:
    sys.path.insert(0, SBSI_ROOT)

from sbs_shear.measurement_model import load_measurement_model  # noqa: E402
from sbs_shear.preprocessing import (  # noqa: E402
    rescale, source_select_selection, DEFAULT_SELECTION_CUTS,
)
from sbs_shear.shear_map import apply_shear_to_ellipticity  # noqa: E402

CDIR = "/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/"
PLUS = CDIR + "constant_shear_catalogue_0.02_train.feather"
MINUS = CDIR + "constant_shear_catalogue_-0.02_train.feather"
MEASURED = "/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/constgold_measured_c0-139.feather"
CROWD = "/home/z/Zekang.Zhang/SBSI/results/crowd_flux_conc_c0-199.feather"
RK = dict(pixel_rms=0.312, pixel_size=0.2, zero_mag=30.0, psf_fwhm=0.73, moffat_beta=2.224)

# true-property columns needed for flow conditioning + rescale()'s derived features, plus the
# measured shape (only used for the R_total normalisation, never for the selection numerator)
TRUE_COLS = ["case", "input_index", "neighbored", "distance", "polarization_angle",
             "Re_input_p", "Re_input_s", "r_input_p", "r_input_s",
             "sersic_n_input_p", "sersic_n_input_s",
             "axis_ratio_input_p", "position_angle_input_p",
             "applied_g1", "applied_g2", "measured_e1", "measured_e2"]


def ellip_from_axis_ratio_angle(q, pa_deg):
    """Unsheared intrinsic spin-2 ellipticity from axis ratio + position angle (canonical basis)."""
    q = np.asarray(q, dtype=float)
    e = (1.0 - q) / (1.0 + q)
    phi = np.asarray(pa_deg, dtype=float) / 180.0 * np.pi
    return e * np.cos(2.0 * phi), e * np.sin(2.0 * phi)


def read_leg(path, min_case, max_case, cols, t0, label):
    """Per-leg frame, batch-filtered by case (the catalogues are case-ordered)."""
    parts = []
    with ipc.open_file(pa.memory_map(path)) as r:
        avail = set(r.schema.names)
        use = [c for c in cols if c in avail]
        for bi in range(r.num_record_batches):
            b = pa.Table.from_batches([r.get_batch(bi)]).select(use).to_pandas()
            c = b["case"].to_numpy()
            if c.min() > max_case:
                break
            b = b[(c >= min_case) & (c <= max_case)]
            if len(b):
                parts.append(b)
    df = pd.concat(parts, ignore_index=True)
    print(f"  {label}: {len(df):,} rows ({time.time()-t0:.0f}s)", flush=True)
    return df


def read_case_range(path, min_case, max_case, columns=None):
    """Read a (case, input_index, ...) lookup, filtering on case in Arrow to keep peak RAM down."""
    tb = pf.read_table(path, columns=columns)
    mask = pc.and_(pc.greater_equal(tb["case"], min_case), pc.less_equal(tb["case"], max_case))
    return tb.filter(mask).to_pandas()


def build_base(min_case, max_case, measured, crowd, t0):
    """Both-detected +/-0.02 pairs carrying per-leg measured mag/size, intrinsic shape and ghat."""
    T = read_leg(PLUS, min_case, max_case, TRUE_COLS, t0, "+leg (true props + measured shape)")
    M = read_leg(MINUS, min_case, max_case,
                 ["case", "input_index", "measured_e1", "measured_e2"], t0, "-leg (measured shape)")

    m = read_case_range(measured, min_case, max_case)
    mcols = ["case", "input_index", "measured_mag_auto", "measured_flux_radius"]
    mp = m[m["shear_case"] > 0][mcols]
    mm = m[m["shear_case"] < 0][mcols]
    del m
    print(f"  measured lookup: +leg {len(mp):,}  -leg {len(mm):,} ({time.time()-t0:.0f}s)", flush=True)

    base = T.merge(M, on=["case", "input_index"], how="inner", suffixes=("_plus", "_minus"))
    base = base.merge(mp, on=["case", "input_index"], how="inner")
    base = base.merge(mm, on=["case", "input_index"], how="inner",
                      suffixes=("_plus", "_minus"))
    print(f"  BOTH-DETECTED pairs: {len(base):,}  ({len(base)/max(len(T),1):.1%} of +leg) "
          f"({time.time()-t0:.0f}s)", flush=True)

    # flow-domain cut (same DEFAULT_SELECTION_CUTS as the certified constgold evaluation)
    base = source_select_selection(base, cuts=DEFAULT_SELECTION_CUTS).reset_index(drop=True)
    print(f"  after flow-domain cut: {len(base):,} ({time.time()-t0:.0f}s)", flush=True)

    # neighbour-flux conditioners (the flow's only crowding inputs)
    cf = read_case_range(crowd, min_case, max_case)
    j = base[["case", "input_index"]].merge(cf, on=["case", "input_index"], how="left")
    del cf
    for c in ("nbr_flux_near", "nbr_flux_far", "nbr_flux_max"):
        base[c] = j[c].fillna(0.0).to_numpy(float)
    print(f"  nbr_flux matched {j['nbr_flux_near'].notna().mean():.1%} ({time.time()-t0:.0f}s)",
          flush=True)

    # UNSHEARED intrinsic shape (identical in both legs by construction) + shear direction
    e1i, e2i = ellip_from_axis_ratio_angle(base["axis_ratio_input_p"], base["position_angle_input_p"])
    base["e1_input_rot0_p"] = e1i
    base["e2_input_rot0_p"] = e2i
    base["e1_input_rot0_s"] = 0.0
    base["e2_input_rot0_s"] = 0.0
    g1 = base["applied_g1"].to_numpy(float)
    g2 = base["applied_g2"].to_numpy(float)
    gmag = np.hypot(g1, g2)
    g = float(np.median(gmag))
    gh1, gh2 = g1 / gmag, g2 / gmag                    # +leg direction, used for BOTH legs
    p_int = e1i * gh1 + e2i * gh2                      # the numerator: same value in both legs

    # total measured-shape response on the same sample -> lets R_sel be quoted as a % bias
    r_tot_i = ((base["measured_e1_plus"].to_numpy(float) - base["measured_e1_minus"].to_numpy(float)) * gh1
               + (base["measured_e2_plus"].to_numpy(float) - base["measured_e2_minus"].to_numpy(float)) * gh2
               ) / (2 * g)

    # the flow applies our own +/-s to the intrinsic shape, so zero any catalogue shear
    base["gamma1_input_p"] = 0.0
    base["gamma2_input_p"] = 0.0
    print(f"  g={g:.4f}  <e_int.ghat>={p_int.mean():+.5f}  ghat=({gh1[0]:.0f},{gh2[0]:.0f})  "
          f"({time.time()-t0:.0f}s)", flush=True)
    return base, p_int, r_tot_i, gh1, gh2, g


def make_cuts(mag_cuts, size_cuts, pixel_size):
    """Cut suite. mag: keep BRIGHTER than thr. size: keep LARGER than thr (arcsec).

    Plus two nulls routed through the SAME machinery as a bookkeeping check: no cut at all, and a
    cut on the TRUE (non-shearing) size. Both must return exactly 0.
    """
    cuts = [dict(name="NULL nocut", kind="null", raw=np.nan, keep_high=False,
                 sim_col="measured_mag_auto", sim_thr=np.inf, dim=2, mod_thr=np.inf),
            dict(name="NULL trueRe>0.3", kind="null", raw=np.nan, keep_high=True,
                 sim_col="Re_input_p", sim_thr=0.3, dim=None, mod_thr=None)]
    cuts += [dict(name=f"mag<{c:g}", kind="mag", raw=float(c), keep_high=False,
                  sim_col="measured_mag_auto", sim_thr=float(c), dim=2, mod_thr=float(c))
             for c in mag_cuts]
    cuts += [dict(name=f"size>{c:g}", kind="size", raw=float(c), keep_high=True,
                  sim_col="measured_flux_radius", sim_thr=float(c) / pixel_size,   # arcsec -> px
                  dim=3, mod_thr=float(np.log(float(c) / pixel_size)))             # -> log px
             for c in size_cuts]
    return cuts


def _leg_masks(base, sel, cut):
    """Per-leg pass masks. A TRUE-property column has no _plus/_minus suffix -> identical masks."""
    col = cut["sim_col"]
    if col + "_plus" in base.columns:
        xp = base[col + "_plus"].to_numpy(float)
        xm = base[col + "_minus"].to_numpy(float)
    else:
        xp = xm = base[col].to_numpy(float)
    thr = cut["sim_thr"]
    if cut["keep_high"]:
        return sel & (xp > thr), sel & (xm > thr)
    return sel & (xp < thr), sel & (xm < thr)


def sim_response(base, p_int, sel, cut, g, cases, n_boot, rng):
    """R_sel from the sim + a per-case bootstrap error.

    The two legs share the SAME p_int per object, so R is driven purely by the pass masks. The
    bootstrap resamples CASES (the independent sim units) to capture sample variance correctly.
    """
    pp, pm = _leg_masks(base, sel, cut)
    fin = np.isfinite(p_int)
    pp = pp & fin
    pm = pm & fin
    np_, nm_ = int(pp.sum()), int(pm.sum())
    if np_ < 10 or nm_ < 10:
        return np.nan, np.nan, np.nan
    R = (p_int[pp].mean() - p_int[pm].mean()) / (2 * g)
    frac = 0.5 * (np_ + nm_) / max(int((sel & fin).sum()), 1)
    if n_boot <= 0:
        return R, np.nan, frac

    uc, inv = np.unique(cases, return_inverse=True)
    sp = np.bincount(inv[pp], weights=p_int[pp], minlength=len(uc))
    cp = np.bincount(inv[pp], minlength=len(uc))
    sm = np.bincount(inv[pm], weights=p_int[pm], minlength=len(uc))
    cm = np.bincount(inv[pm], minlength=len(uc))
    bs = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uc), len(uc))
        Cp, Cm = cp[pick].sum(), cm[pick].sum()
        if Cp > 0 and Cm > 0:
            bs.append((sp[pick].sum() / Cp - sm[pick].sum() / Cm) / (2 * g))
    return R, (float(np.std(bs)) if bs else np.nan), frac


def prepare_legs(base, gh1, gh2, g, idx):
    """Rescale the -g and +g truth-perturbed frames ONCE, shared across all checkpoints.

    The shear perturbs the TRUE ellipticity (the model never sees g), so this is the entire
    g-dependence of the model prediction.
    """
    intr1 = base["e1_input_rot0_p"].to_numpy(float)
    intr2 = base["e2_input_rot0_p"].to_numpy(float)
    legs = {}
    for leg, s in ((0, -g), (1, +g)):
        fr = base.iloc[idx].reset_index(drop=True).copy()
        e1s, e2s = apply_shear_to_ellipticity(intr1[idx], intr2[idx], s * gh1[idx], s * gh2[idx])
        fr["e1_input_rot0_p"] = e1s
        fr["e2_input_rot0_p"] = e2s
        legs[leg] = rescale(fr, **RK)
    return legs


@torch.no_grad()
def model_response(bundle, legs, w, cuts, n_samples, batch_size, flow_seed, chunk):
    """R_sel from the flow: weight each object's intrinsic e by its pass-fraction per leg."""
    mcuts = [c for c in cuts if c["dim"] is not None]
    acc = {c["name"]: {0: [0.0, 0.0], 1: [0.0, 0.0]} for c in mcuts}
    n = len(w)
    for cs in range(0, n, chunk):
        sl = slice(cs, min(cs + chunk, n))
        wc = w[sl][:, None]
        seed = flow_seed + cs
        for leg in (0, 1):
            torch.manual_seed(seed)                       # common random numbers across legs
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            d = bundle.sample(legs[leg].iloc[sl], n_samples=n_samples, batch_size=batch_size)
            mag, logsz = d[:, :, 2], d[:, :, 3]
            fin = np.isfinite(mag) & np.isfinite(logsz)
            for c in mcuts:
                xv = logsz if c["dim"] == 3 else mag
                pm = fin & ((xv > c["mod_thr"]) if c["keep_high"] else (xv < c["mod_thr"]))
                acc[c["name"]][leg][0] += float((pm * wc).sum())
                acc[c["name"]][leg][1] += float(pm.sum())
    return acc


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt-glob",
                    default="/project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation/"
                            "measurement_flow_g0_ngmix_ablate_s2c_coupling_lt500_s50*_swaavg.pt")
    ap.add_argument("--ckpt", nargs="+", default=None)
    ap.add_argument("--measured", default=MEASURED)
    ap.add_argument("--crowd", default=CROWD)
    ap.add_argument("--min-case", type=int, default=40, help="held-out constgold split")
    ap.add_argument("--max-case", type=int, default=139)
    ap.add_argument("--mag-cuts", type=float, nargs="+",
                    default=[24.5, 24.75, 25.0, 25.25, 25.5, 25.75, 26.0, 26.25, 26.5])
    ap.add_argument("--size-cuts", type=float, nargs="+",
                    default=[0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70],
                    help="measured half-light radius, ARCSEC; keep larger than thr")
    ap.add_argument("--pixel-size", type=float, default=0.2)
    ap.add_argument("--n-samples", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=16384)
    ap.add_argument("--chunk", type=int, default=250_000)
    ap.add_argument("--flow-seed", type=int, default=12345)
    ap.add_argument("--model-max-rows", type=int, default=1_500_000,
                    help="random subsample of the paired base used for the (expensive) flow legs")
    ap.add_argument("--n-boot", type=int, default=300)
    ap.add_argument("--subsample-seed", type=int, default=7)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    ckpts = list(args.ckpt or [])
    if args.ckpt_glob:
        ckpts += sorted(glob.glob(args.ckpt_glob))
    if not ckpts:
        ap.error("no checkpoints found")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    print(f"device={device}  ckpts={len(ckpts)}  n_samples={args.n_samples}", flush=True)
    for c in ckpts:
        print("   ", os.path.basename(c))

    base, p_int, r_tot_i, gh1, gh2, g = build_base(args.min_case, args.max_case, args.measured,
                                                   args.crowd, t0)
    cases = base["case"].to_numpy()
    cuts = make_cuts(args.mag_cuts, args.size_cuts, args.pixel_size)
    nbrd = base["neighbored"].astype(bool).to_numpy()
    scopes = [("ALL", np.ones(len(base), bool)), ("ISOLATED", ~nbrd)]

    rng = np.random.default_rng(0)
    sub_rng = np.random.default_rng(args.subsample_seed)
    saved = {}
    r_tot_by_scope = {}

    for sname, sel in scopes:
        fin = sel & np.isfinite(p_int) & np.isfinite(r_tot_i)
        R_total = float(np.mean(r_tot_i[fin]))
        r_tot_by_scope[sname] = R_total
        print(f"\n================ {sname}  N={int(sel.sum()):,} ================", flush=True)
        print(f"  R_total (measured-shape response, no cut) = {R_total:.4f}"
              f"   -> R_sel/R_total gives the bias in %", flush=True)

        # ---- MODEL: 8-seed ensemble on a subsample ----
        idx_all = np.where(sel)[0]
        if len(idx_all) > args.model_max_rows:
            idx = np.sort(sub_rng.choice(idx_all, args.model_max_rows, replace=False))
        else:
            idx = idx_all
        print(f"  model subsample: {len(idx):,} objects; rescaling both legs...", flush=True)
        legs = prepare_legs(base, gh1, gh2, g, idx)
        w = p_int[idx]
        print(f"  legs ready ({time.time()-t0:.0f}s)", flush=True)

        per_ckpt = []
        for c in ckpts:
            bundle = load_measurement_model(c, device=device)
            acc = model_response(bundle, legs, w, cuts, args.n_samples, args.batch_size,
                                 args.flow_seed, args.chunk)
            per_ckpt.append(acc)
            print(f"    [{os.path.basename(c)}] done ({time.time()-t0:.0f}s)", flush=True)
            del bundle
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        del legs

        def ens(name):
            Rs, Fs = [], []
            for a in per_ckpt:
                m0 = a[name][0][0] / a[name][0][1] if a[name][0][1] else np.nan   # minus leg
                mg = a[name][1][0] / a[name][1][1] if a[name][1][1] else np.nan   # plus leg
                Rs.append((mg - m0) / (2 * g))
                Fs.append(0.5 * (a[name][0][1] + a[name][1][1]) / (len(w) * args.n_samples))
            return float(np.mean(Rs)), float(np.std(Rs)), float(np.mean(Fs))

        # ---- SIM (full sample) + table ----
        sub_mask = np.zeros(len(base), bool)
        sub_mask[idx] = True
        print(f"\n  {'cut':>15} {'keep%':>6} | {'R_sel_sim':>10} {'+-':>8} | {'R_sel_mod':>10} "
              f"{'+-':>8} | {'sim[sub]':>9} | {'bias_sim%':>9} {'bias_mod%':>9}", flush=True)
        rows = []
        for c in cuts:
            R_s, R_se, frac = sim_response(base, p_int, sel, c, g, cases, args.n_boot, rng)
            R_ss, _, _ = sim_response(base, p_int, sub_mask & sel, c, g, cases, 0, rng)
            if c["dim"] is None:
                R_m, R_msd, fracm = 0.0, 0.0, np.nan     # true-property cut: model null too
            else:
                R_m, R_msd, fracm = ens(c["name"])
            print(f"  {c['name']:>15} {frac*100:>6.1f} | {R_s:+10.5f} {R_se:>8.5f} | "
                  f"{R_m:+10.5f} {R_msd:>8.5f} | {R_ss:+9.5f} | "
                  f"{R_s/R_total*100:>+9.3f} {R_m/R_total*100:>+9.3f}", flush=True)
            rows.append(dict(cut=c["name"], kind=c["kind"], raw=c["raw"], frac=frac, fracM=fracm,
                             R_sim=R_s, R_sim_err=R_se, R_sim_sub=R_ss,
                             R_model=R_m, R_model_sd=R_msd))
        saved[sname] = rows

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        out = dict(g=g, two_g=2 * g, pixel_size=args.pixel_size,
                   ckpts=np.array([os.path.basename(c) for c in ckpts]),
                   scope_names=np.array([s[0] for s in scopes]),
                   min_case=args.min_case, max_case=args.max_case,
                   n_samples=args.n_samples, model_max_rows=args.model_max_rows)
        for si, (sname, _) in enumerate(scopes):
            rows = saved[sname]
            out[f"s{si}_cut"] = np.array([r["cut"] for r in rows])
            out[f"s{si}_kind"] = np.array([r["kind"] for r in rows])
            out[f"s{si}_R_total"] = np.array(r_tot_by_scope[sname])
            for k in ("raw", "frac", "fracM", "R_sim", "R_sim_err", "R_sim_sub",
                      "R_model", "R_model_sd"):
                out[f"s{si}_{k}"] = np.array([r[k] for r in rows], dtype=float)
        np.savez(args.output, **out)
        print(f"\nsaved {args.output}")
    print("SELECTION_INTRINSIC_DONE", flush=True)


if __name__ == "__main__":
    main()
