"""constgold three-column selection table PLUS a model-m column: can the flow predict it?

Owner's request: take the 30s constgold table and add what the FLOW predicts, using the M2 proxy S/N
(mag + log size + |e|^2) built from the flow's own outputs.

THE ASYMMETRY, STATED UP FRONT -- it is forced by the data, not a choice. constgold stores per-leg
`S/N_plus`/`S/N_minus` but NO per-leg measured magnitude or size (checked: the only per-leg columns
are S/N, et, ex, measured_e1/e2). So:
    SIM   cuts on the REAL stored SExtractor S/N
    MODEL cuts on the M2 PROXY built from its sampled mag, size and |e|
These are different variables, so thresholds are matched by KEEP-FRACTION (quantile), not by value.
The proxy-vs-real gap measured on the half-shear legs (RMS 0.27 pts for M2, job 15365211) is
therefore INSIDE this comparison and cannot be separated here. Read the model column as
"flow + proxy", not "flow".

MODEL COLUMN DEFINITION. The flow predicts MEASURED shapes, so its natural counterpart is sim
column (3): the same shift m,
        m_model = R_model(cut) / R_model(no cut) - 1,
        R_model = ( <proj>_+g[pass_+] - <proj>_-g[pass_-] ) / (2 g)
with the cut applied per leg to the flow's own sampled draws so the boundary MOVES, exactly as it
does in the sim. Compare it against column (3), NOT against columns (1)/(2).

Common Random Numbers: both legs are reseeded to the same value, so flow sampling noise cancels in
the difference instead of being amplified by 1/(2g) ~ 25x.

FIREWALL: constgold is EVALUATION only -- nothing is trained, fitted, or selected on it. The proxy
coefficients were fitted on the HALF-SHEAR legs, not here; only their scale-free part is used since
thresholds are set by quantile.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pyarrow.feather as pf
import torch

SBSI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (SBSI_ROOT, os.path.join(SBSI_ROOT, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sbs_shear.measurement_model import load_measurement_model  # noqa: E402
from sbs_shear.preprocessing import (  # noqa: E402
    DEFAULT_SELECTION_CUTS, rescale, source_select_selection,
)
from sbs_shear.coordinates import ellipticity_from_axis_ratio_angle  # noqa: E402
from sbs_shear.shear_map import apply_shear_to_ellipticity  # noqa: E402

# Columns needed for BOTH the flow conditioning and the sim's per-leg columns. They are read in ONE
# pass and kept in ONE frame: `validate_constant_with_blend.load` cannot be reused here because it
# (a) stops after --max-rows record batches, which on this case-ordered catalogue returns only cases
# 0-39 and made the case>=40 filter return an EMPTY frame, and (b) applies source_select_selection,
# which DROPS rows -- so pairing it with a separately-read S/N table by position was misaligned.
NEED = ["case", "input_index", "neighbored", "distance", "polarization_angle",
        "Re_input_p", "Re_input_s", "axis_ratio_input_p", "axis_ratio_input_s",
        "position_angle_input_p", "position_angle_input_s", "r_input_p", "r_input_s",
        "redshift_input_p", "redshift_input_s", "sersic_n_input_p", "sersic_n_input_s",
        "applied_g1", "applied_g2", "shear_magnitude", "shear_angle",
        "measured_e1_plus", "measured_e2_plus", "measured_e1_minus", "measured_e2_minus",
        "et_plus", "S/N_plus", "S/N_minus"]

CG = ("/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/"
      "constant_response_catalogue_train.feather")
CROWD = "/home/z/Zekang.Zhang/SBSI/results/crowd_flux_conc_c0-199.feather"
RK = dict(pixel_rms=0.312, pixel_size=0.2, zero_mag=30.0, psf_fwhm=0.73, moffat_beta=2.224)


def leg_avg(ap, am, g, np_, nm_):
    if np_ == 0 or nm_ == 0:
        return np.nan
    return (ap - am) / (2.0 * g)


@torch.no_grad()
def model_selected(bundle, df, g, gh1, gh2, intr, keep_fracs, sn_ab, n_samples, batch_size,
                   seed, device, sign=1.0, chunk=200_000):
    """Per-leg sampled draws -> proxy S/N -> moving-boundary selected means.

    Returns {keepfrac: R_model} plus '__nocut__'. Thresholds are set from the +g leg's proxy
    distribution at the requested quantiles, then applied to BOTH legs (one threshold, two legs),
    mirroring how the sim applies one S/N threshold to both legs.
    """
    a_, b_, d_ = sn_ab
    names = bundle.target_transform.target_names
    i1, i2 = names.index("measured_ngmix_g1"), names.index("measured_ngmix_g2")
    imag = names.index("measured_mag_auto")
    ilr = names.index("measured_log_flux_radius")
    ln10 = np.log(10.0)
    n = len(df)

    def draws_for(s):
        outs = []
        for lo in range(0, n, chunk):
            hi = min(lo + chunk, n)
            fr = df.iloc[lo:hi].reset_index(drop=True).copy()
            e1s, e2s = apply_shear_to_ellipticity(intr[0][lo:hi], intr[1][lo:hi],
                                                  s * gh1[lo:hi], s * gh2[lo:hi])
            fr["e1_input_rot0_p"] = e1s
            fr["e2_input_rot0_p"] = e2s
            fr = rescale(fr, **RK)
            torch.manual_seed(seed)                     # CRN: same latents in both legs
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            d = bundle.sample(fr, n_samples=n_samples, batch_size=batch_size)
            # `sign` is the SAME global projection sign the sim columns use (locked to the
            # stored et convention). Applied to the PROJECTION only -- never to gh1/gh2, which
            # carry the physical shear direction used to shear the intrinsic shape.
            proj = sign * (d[:, :, i1] * gh1[lo:hi, None] + d[:, :, i2] * gh2[lo:hi, None])
            u = (a_ * d[:, :, imag] + b_ * (d[:, :, ilr] / ln10)
                 + d_ * (d[:, :, i1] ** 2 + d[:, :, i2] ** 2))
            outs.append((proj.astype(np.float32), u.astype(np.float32)))  # f32: (N, n_samples) x4 arrays
        return (np.concatenate([o[0] for o in outs]), np.concatenate([o[1] for o in outs]))

    pj_p, u_p = draws_for(+g)
    pj_m, u_m = draws_for(-g)
    fin_p, fin_m = np.isfinite(pj_p), np.isfinite(pj_m)

    out = {}
    out["__nocut__"] = leg_avg(float(pj_p[fin_p].mean()), float(pj_m[fin_m].mean()), g,
                               int(fin_p.sum()), int(fin_m.sum()))
    good = u_p[fin_p]
    for kf in keep_fracs:
        thr = float(np.quantile(good, 1.0 - kf))
        pp = fin_p & (u_p > thr)
        pm = fin_m & (u_m > thr)
        out[kf] = leg_avg(float(pj_p[pp].mean()) if pp.any() else np.nan,
                          float(pj_m[pm].mean()) if pm.any() else np.nan, g,
                          int(pp.sum()), int(pm.sum()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--cat", default=CG)
    ap.add_argument("--crowd", default=CROWD)
    ap.add_argument("--min-case", type=int, default=40)
    ap.add_argument("--max-rows", type=int, default=4_000_000, help="subsample cap AFTER cuts (0 = keep all)")
    ap.add_argument("--keep-fracs", type=float, nargs="+", default=[0.95, 0.85, 0.70, 0.50, 0.30])
    ap.add_argument("--sn-a", type=float, default=-0.3676)
    ap.add_argument("--sn-b", type=float, default=-0.7736)
    ap.add_argument("--sn-d", type=float, default=0.0, help="|e|^2 coefficient (M2)")
    ap.add_argument("--n-samples", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=16384)
    ap.add_argument("--flow-seed", type=int, default=12345)
    ap.add_argument("--dom-mag-max", type=float, default=26.0,
                    help="flow training domain: primary true mag upper limit")
    ap.add_argument("--dom-re-min", type=float, default=0.3,
                    help="flow training domain: primary true Re lower limit")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    df = pf.read_table(args.cat, columns=NEED, memory_map=True).to_pandas()
    n_raw = len(df)
    df = df[df["case"] >= args.min_case].reset_index(drop=True)
    print(f"constgold: {n_raw:,} rows -> {len(df):,} with case>={args.min_case} "
          f"({time.time()-t0:.0f}s)", flush=True)
    if len(df) == 0:
        raise SystemExit("REFUSING: no rows after the case cut -- check --min-case.")
    e1i, e2i = ellipticity_from_axis_ratio_angle(df["axis_ratio_input_p"].to_numpy(float),
                                                df["position_angle_input_p"].to_numpy(float))
    df["e1_input_rot0_p"] = e1i
    df["e2_input_rot0_p"] = e2i
    df["gamma1_input_p"] = 0.0
    df["gamma2_input_p"] = 0.0
    df = source_select_selection(df, cuts=DEFAULT_SELECTION_CUTS).reset_index(drop=True)
    print(f"  after source_select_selection: {len(df):,}", flush=True)
    # DOMAIN CUT -- mandatory, and getting it wrong is what broke run 15365425. The dom6x6 flow was
    # TRAINED with primary_mag_max=26.0 and primary_re_min=0.3 (read from its own metadata), so
    # scoring it on the full constgold population evaluates it out of domain: R_model(no cut) came
    # out +0.173 against an expected ~0.29, which inflated every model-m entry by ~4x. Both the sim
    # and the model columns must live on the SAME in-domain population.
    dom = (df["r_input_p"].to_numpy(float) < args.dom_mag_max) & \
          (df["Re_input_p"].to_numpy(float) > args.dom_re_min)
    df = df[dom].reset_index(drop=True)
    print(f"  after DOMAIN cut (mag<{args.dom_mag_max}, Re>{args.dom_re_min}): {len(df):,}",
          flush=True)
    # Subsample AFTER all cuts, keeping every column aligned in the one frame. The sim columns have
    # errors ~0.002%, so a few million rows is far more than the comparison needs, and it bounds the
    # (N, n_samples) draw arrays.
    if args.max_rows and len(df) > args.max_rows:
        sel = np.random.default_rng(0).choice(len(df), size=args.max_rows, replace=False)
        sel.sort()
        df = df.iloc[sel].reset_index(drop=True)
        print(f"  subsampled to {len(df):,} rows (seed 0)", flush=True)

    cf = pf.read_table(args.crowd).to_pandas()
    fcols = [c for c in ("nbr_flux_near", "nbr_flux_far", "nbr_flux_max") if c in cf.columns]
    mm = df[["case", "input_index"]].merge(cf[["case", "input_index", *fcols]],
                                           on=["case", "input_index"], how="left")
    for c in fcols:
        df[c] = mm[c].to_numpy(float)
    print(f"crowd-flux: matched {np.mean(mm[fcols[0]].notna()):.1%}  cols={fcols}", flush=True)

    # Per-leg sim quantities come from the SAME frame -- no positional pairing, so no misalignment.
    t = df
    g = float(np.median(t.shear_magnitude.to_numpy(float)))
    sa = t.shear_angle.to_numpy(float)
    c2, s2 = np.cos(2 * sa), np.sin(2 * sa)
    me1p, me2p = t.measured_e1_plus.to_numpy(float), t.measured_e2_plus.to_numpy(float)
    me1m, me2m = t.measured_e1_minus.to_numpy(float), t.measured_e2_minus.to_numpy(float)
    myp, st = me1p * c2 + me2p * s2, t.et_plus.to_numpy(float)
    ok = np.isfinite(myp) & np.isfinite(st)
    sign = 1.0 if np.nanmean((myp * st)[ok]) > 0 else -1.0
    proj = lambda a, b: sign * (a * c2 + b * s2)

    i1 = df["e1_input_rot0_p"].to_numpy(float)
    i2 = df["e2_input_rot0_p"].to_numpy(float)
    gh1, gh2 = np.cos(2 * sa), np.sin(2 * sa)          # unit shear direction per case
    e1p, e2p = apply_shear_to_ellipticity(i1, i2, g * gh1, g * gh2)
    e1m, e2m = apply_shear_to_ellipticity(i1, i2, -g * gh1, -g * gh2)
    kinds = [("unsheared", proj(i1, i2), proj(i1, i2)),
             ("sheared", proj(e1p, e2p), proj(e1m, e2m)),
             ("measured", proj(me1p, me2p), proj(me1m, me2m))]

    snp, snm = t["S/N_plus"].to_numpy(float), t["S/N_minus"].to_numpy(float)
    fin = np.isfinite(snp) & np.isfinite(snm)
    for _, a, b in kinds:
        fin &= np.isfinite(a) & np.isfinite(b)
    print(f"finite: {int(fin.sum()):,}   g={g:.4f}", flush=True)

    R0 = {k: leg_avg(float(a[fin].mean()), float(b[fin].mean()), g, 1, 1) for k, a, b in kinds}
    print("  sim R(no cut): " + "  ".join(f"{k}={R0[k]:+.5f}" for k in R0))

    per = []
    for ck in args.ckpt:
        bundle = load_measurement_model(ck, device=device)
        per.append(model_selected(bundle, df, g, gh1, gh2, (i1, i2), args.keep_fracs,
                                  (args.sn_a, args.sn_b, args.sn_d), args.n_samples,
                                  args.batch_size, args.flow_seed, device, sign=sign))
        print(f"  scored {os.path.basename(ck)} ({time.time()-t0:.0f}s)", flush=True)
    keys = ["__nocut__"] + list(args.keep_fracs)
    mod = {k: float(np.mean([p[k] for p in per])) for k in keys}
    sem = {k: (float(np.std([p[k] for p in per], ddof=1) / np.sqrt(len(per)))
               if len(per) > 1 else np.nan) for k in keys}
    print(f"  model R(no cut) = {mod['__nocut__']:+.5f}")

    print("\n" + "=" * 112)
    print("CONSTGOLD SELECTION: sim three columns + MODEL m   (sim cuts REAL S/N, model cuts PROXY;")
    print("                     thresholds matched by KEEP-FRACTION)")
    print("=" * 112)
    print(f"  {'keep':>6} {'S/N thr':>8} | {'(1) pure sel':>13} {'(2) sheared':>12} "
          f"{'(3) measured':>13} | {'(4) MODEL m':>16}")
    for kf in args.keep_fracs:
        thr = float(np.quantile(snp[fin], 1.0 - kf))
        pp, pm = fin & (snp > thr), fin & (snm > thr)
        cells = []
        for k, a, b in kinds:
            R = leg_avg(float(a[pp].mean()), float(b[pm].mean()), g, int(pp.sum()), int(pm.sum()))
            # The unsheared column has R(no cut) == 0 BY CONSTRUCTION, so its own ratio is 0/0.
            # Normalise it by the sheared column instead -- the same convention as the 30s table.
            den = R0[k] if R0[k] else R0['sheared']
            cells.append(R / den if k == 'unsheared' else R / den - 1.0)
        mm_ = mod[kf] / mod["__nocut__"] - 1.0
        e = sem[kf] / abs(mod["__nocut__"]) * 100.0 if np.isfinite(sem[kf]) else np.nan
        print(f"  {kf:>6.2f} {thr:>8.2f} | {100*cells[0]:>+12.3f}% {100*cells[1]:>+11.3f}% "
              f"{100*cells[2]:>+12.3f}% | {100*mm_:>+10.3f} +- {e:<.3f}")
    print("\n  Column (4) is the flow's counterpart of column (3) -- both are MEASURED-shape shift m.")
    print("  Compare (4) vs (3). Columns (1)/(2) are intrinsic-shape references, not model targets.")
    print("  The proxy-vs-real gap (RMS 0.27 pts, job 15365211) is INSIDE column (4).")
    print("CONSTGOLD_MODEL_DONE", flush=True)


if __name__ == "__main__":
    main()
