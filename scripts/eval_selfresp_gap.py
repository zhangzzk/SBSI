"""Self-response gap eval for V1-format measurement-flow checkpoints (ablation ladder).

Metric (comparable across every ablation step because the readout is identical):

  * TRUTH  R_hs : measured half-shear self-response on the g0.05 leg, matched to the g0.0
                  leg by (case, input_index), both-detected, true-property acceptance cut.
                  R_hs = ((e1_g - e1_0)*ghat1 + (e2_g - e2_0)*ghat2) / g   with
                  e = measured_ngmix_g1/g2, ghat = applied-shear direction, g = |gamma|.
  * MODEL  R_flow : the flow mean-head self-response, read out EXACTLY as in training
                  (sbs_shear.train_measurement_model_swa.build_shifted_context + model._mu):
                  shear the intrinsic primary shape (e1/e2_input_rot0_p) by +/-delta via the
                  analytic S_delta map, hold every other conditioner fixed, and take the
                  induced move of the mean head's shape dims, converted to physical units
                  with the target-standardizer scales. Ensemble = mean R_flow over ckpts.

Reports  flow / R_hs - 1  OVERALL and resolved by TRUE size Re_input_p in bins
[0.30, 0.38, 0.50, 0.75, 1.50], on the ISOLATED acceptance set (no brighter true neighbour
within 7"; the isolation lookup covers cases 0-39, hence --max-case default 39).

The script loads its conditioning columns from each ckpt's OWN metadata, so it evaluates
V1 (measured-conditioned) and Step-1 (true-conditioned) checkpoints unchanged -- the
measured->true swap only changes which raw columns build_shifted_context reads.

FIREWALL: nothing is trained or fit here; this only scores existing checkpoints against the
det_meas half-shear truth (the flow's own training sim family; no constgold, no R_blend).
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
import pyarrow.ipc as ipc
import pyarrow.feather as pf
import torch

SBSI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SBSI_ROOT not in sys.path:
    sys.path.insert(0, SBSI_ROOT)

from sbs_shear.measurement_model import load_measurement_model  # noqa: E402
from sbs_shear.preprocessing import (  # noqa: E402
    DEFAULT_SELECTION_CUTS,
    rescale,
    source_select_selection,
)
from sbs_shear.shear_map import apply_shear_to_ellipticity  # noqa: E402

CAT = "/project/ls-gruen/users/zekang.zhang/sbsi_catalogues/"
# per-(case,input_index) neighbour-flux lookup (nbr_flux_near/far/max), cases 0-199
CROWD = "/home/z/Zekang.Zhang/SBSI/results/crowd_flux_conc_c0-199.feather"
# per-(case,input_index) nearest-brighter-neighbour distance, cases 0-39
NN = "/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/nn_dist_c0-39.feather"

NGMIX = ["measured_ngmix_g1", "measured_ngmix_g2"]
GAMMA = ["gamma1_input_p", "gamma2_input_p"]
# raw columns build_shifted_context / rescale / the V1+S1 conditioning touch on the g leg
FLOW_COLS = [
    "case", "input_index", "detected", "neighbored", "distance", "polarization_angle",
    "shear_component_convention",
    "Re_input_p", "Re_input_s", "r_input_p", "r_input_s",
    "sersic_n_input_p", "sersic_n_input_s", "redshift_input_p", "redshift_input_s",
    "e1_input_rot0_p", "e2_input_rot0_p", "e1_input_rot0_s", "e2_input_rot0_s",
    "axis_ratio_input_p", "position_angle_input_p", "axis_ratio_input_s", "position_angle_input_s",
    "measured_mag_auto", "measured_flux_radius",
]
SIZE_EDGES = np.array([0.30, 0.38, 0.50, 0.75, 1.50])


def read_leg(path, cols, max_case):
    parts = []
    with ipc.open_file(path) as r:
        av = set(r.schema.names)
        use = [c for c in cols if c in av]
        for bi in range(r.num_record_batches):
            b = pa.Table.from_batches([r.get_batch(bi)]).select(use).to_pandas()
            if max_case is not None:
                if int(b["case"].min()) > max_case:
                    break
                b = b[b["case"] <= max_case]
            if len(b):
                parts.append(b)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cols)


def domain_cut(df, re_min=0.3, mag_max=26.0):
    """Flow training domain: base selection cuts + true-property cut + detected."""
    df = source_select_selection(df, cuts=DEFAULT_SELECTION_CUTS)
    det = df["detected"].astype(bool).to_numpy() if "detected" in df.columns else np.ones(len(df), bool)
    m = det & (df["Re_input_p"].to_numpy(float) > re_min) & (df["r_input_p"].to_numpy(float) < mag_max)
    return df[m].reset_index(drop=True)


def build_shifted_context(frame, gdir, delta, preprocessor, condition_features, rescale_kwargs):
    """Standardized conditioning context after applying S_delta to the intrinsic primary shape.

    Mirrors sbs_shear.scripts.train_measurement_model_swa.build_shifted_context exactly, so
    R_flow read out here equals the response the flow was trained on. delta=0 reproduces the
    unshifted context through the identical rescale path.
    """
    f = frame.copy()
    i1 = f["e1_input_rot0_p"].to_numpy(float)
    i2 = f["e2_input_rot0_p"].to_numpy(float)
    s1, s2 = apply_shear_to_ellipticity(i1, i2, delta * gdir[0], delta * gdir[1])
    f["e1_input_rot0_p"] = s1
    f["e2_input_rot0_p"] = s2
    for c in ("gamma1_input_p", "gamma2_input_p"):
        if c in f.columns:
            f[c] = 0.0  # shear folded into rot0; avoid double-applying
    f = rescale(f, **rescale_kwargs)
    missing = [c for c in condition_features if c not in f.columns]
    if missing:
        raise KeyError(f"conditioning columns missing after rescale: {missing}")
    return preprocessor.transform_frame(f)


@torch.no_grad()
def model_selfresp(base, bundle, delta, difference, device, chunk=400_000):
    """Per-object flow self-response R_flow in ngmix units (mean-head finite difference)."""
    model = bundle.model.to(device)
    model.eval()
    pp = bundle.condition_preprocessor
    cond = list(bundle.metadata.get("condition_features", pp.feature_names))
    sc0 = float(bundle.target_transform.scales[0])
    sc1 = float(bundle.target_transform.scales[1])
    rk = dict(pixel_rms=0.312, pixel_size=0.2, zero_mag=30.0, psf_fwhm=0.73, moffat_beta=2.224)

    def mu(ctx_np):
        t = torch.as_tensor(ctx_np, dtype=torch.float32, device=device)
        return model._mu(t)

    n = len(base)
    Rf = np.empty(n, dtype=np.float64)
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        fr = base.iloc[s:e].reset_index(drop=True)
        mu0 = mu(build_shifted_context(fr, (1.0, 0.0), 0.0, pp, cond, rk))
        mu_e1p = mu(build_shifted_context(fr, (1.0, 0.0), +delta, pp, cond, rk))
        mu_e2p = mu(build_shifted_context(fr, (0.0, 1.0), +delta, pp, cond, rk))
        if difference == "central":
            mu_e1m = mu(build_shifted_context(fr, (1.0, 0.0), -delta, pp, cond, rk))
            mu_e2m = mu(build_shifted_context(fr, (0.0, 1.0), -delta, pp, cond, rk))
            r_i = 0.25 * ((mu_e1p[:, 0] - mu_e1m[:, 0]) * sc0
                          + (mu_e2p[:, 1] - mu_e2m[:, 1]) * sc1) / delta
        else:
            r_i = 0.5 * ((mu_e1p[:, 0] - mu0[:, 0]) * sc0
                         + (mu_e2p[:, 1] - mu0[:, 1]) * sc1) / delta
        Rf[s:e] = r_i.detach().cpu().numpy()
    return Rf


def load_ruler(g0_leg, gS_leg, max_case, re_min, mag_max, iso_radius, crowd, nn, t0=None, verbose=True):
    """Build the SHARED ruler used to score EVERY model (V1 ladder or V2) identically:
    matched both-detected g0.05<->g0.0 acceptance base, truth R_hs, isolation mask, true size.
    The only thing that varies across models is the model's own response readout, never this.
    Returns dict(base, R_hs, iso, size, gmed)."""
    tick = (lambda: (time.time() - t0)) if t0 is not None else (lambda: 0.0)
    g0 = domain_cut(read_leg(g0_leg,
                             ["case", "input_index", "detected", "Re_input_p", "r_input_p",
                              "neighbored", "distance"] + NGMIX, max_case),
                    re_min, mag_max).drop_duplicates(["case", "input_index"])
    gS = domain_cut(read_leg(gS_leg, FLOW_COLS + NGMIX + GAMMA, max_case), re_min, mag_max)
    gp = np.hypot(gS["gamma1_input_p"].to_numpy(float), gS["gamma2_input_p"].to_numpy(float))
    gS = gS[gp > 1e-6].reset_index(drop=True).drop_duplicates(["case", "input_index"])
    base = gS.merge(g0[["case", "input_index"] + NGMIX], on=["case", "input_index"], suffixes=("_g", "_0"))
    if verbose:
        print(f"matched both-detected true-cut: N={len(base):,}  cases={base['case'].nunique()}  "
              f"({tick():.1f}s)", flush=True)

    gp = np.hypot(base["gamma1_input_p"].to_numpy(float), base["gamma2_input_p"].to_numpy(float))
    gmed = float(np.median(gp))
    gh1 = base["gamma1_input_p"].to_numpy(float) / gp
    gh2 = base["gamma2_input_p"].to_numpy(float) / gp
    de1 = base["measured_ngmix_g1_g"].to_numpy(float) - base["measured_ngmix_g1_0"].to_numpy(float)
    de2 = base["measured_ngmix_g2_g"].to_numpy(float) - base["measured_ngmix_g2_0"].to_numpy(float)
    R_hs = (de1 * gh1 + de2 * gh2) / gmed
    if verbose:
        print(f"g_med={gmed:.4f}  <R_hs@0.05>={np.nanmean(R_hs[np.isfinite(R_hs)]):+.4f}", flush=True)

    keys = base[["case", "input_index"]]
    nbf = pf.read_table(crowd).to_pandas()[["case", "input_index", "nbr_flux_near",
                                            "nbr_flux_far", "nbr_flux_max"]]
    base = base.merge(nbf, on=["case", "input_index"], how="left")
    nnl = pf.read_table(nn).to_pandas()[["case", "input_index", "nn_dist_bright"]]
    j = keys.merge(nnl, on=["case", "input_index"], how="left")
    nnb = j["nn_dist_bright"].to_numpy(float)
    iso = (~np.isfinite(nnb)) | (nnb > iso_radius)
    if verbose:
        print(f"nbr_flux matched {base['nbr_flux_near'].notna().mean():.1%}  "
              f"nn matched {np.isfinite(nnb).mean():.1%}  isolated frac={iso.mean():.1%}", flush=True)
    size = base["Re_input_p"].to_numpy(float)
    return dict(base=base, R_hs=R_hs, iso=iso, size=size, gmed=gmed)


def print_size_table(R_hs, Rf, size, sel, good, label, size_edges=SIZE_EDGES):
    """flow/R_hs-1 table: OVERALL + resolved by true size. Identical format for every model."""
    print(f"\n[{label}]  N={int((sel & good).sum()):,}")
    print(f"  {'size-bin':>16} {'R_hs':>9} {'R_flow':>9} {'flow/R_hs-1 %':>14} {'N':>10}")
    s = sel & good
    a = float(np.mean(R_hs[s])); b = float(np.mean(Rf[s]))
    print(f"  {'OVERALL':>16} {a:+9.4f} {b:+9.4f} "
          f"{(b/a-1)*100 if a else np.nan:+14.2f} {int(s.sum()):>10,}")
    for i in range(len(size_edges) - 1):
        lo, hi = size_edges[i], size_edges[i + 1]
        m = sel & good & (size >= lo) & (size < hi)
        if m.sum() < 30:
            print(f"  [{lo:.2f},{hi:.2f}){'':>6} {'-':>9} {'-':>9} {'(N<30)':>14} {int(m.sum()):>10,}")
            continue
        a = float(np.mean(R_hs[m])); b = float(np.mean(Rf[m]))
        print(f"  [{lo:.2f},{hi:.2f}){'':>6} {a:+9.4f} {b:+9.4f} "
              f"{(b/a-1)*100 if a else np.nan:+14.2f} {int(m.sum()):>10,}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", nargs="+", default=None,
                    help="one or more V1-format ckpt .pt paths (ensemble = mean R_flow over them)")
    ap.add_argument("--ckpt-glob", default=None,
                    help="glob for ckpts (alternative to --ckpt), e.g. '.../ablate_s0_v1repro_s*_swaavg.pt'")
    ap.add_argument("--g0-leg", default=CAT + "det_meas_ngmix_g0.0_train.feather")
    ap.add_argument("--gS-leg", default=CAT + "det_meas_ngmix_g0.05_val.feather")
    ap.add_argument("--max-case", type=int, default=39,
                    help="cap cases (nn-isolation lookup covers 0-39; raise only with a wider nn lookup)")
    ap.add_argument("--true-re-min", type=float, default=0.3)
    ap.add_argument("--true-mag-max", type=float, default=26.0)
    ap.add_argument("--iso-radius", type=float, default=7.0,
                    help="isolated = no brighter true neighbour within this radius (arcsec)")
    ap.add_argument("--response-difference", choices=["forward", "central"], default="forward",
                    help="flow finite-difference stencil for R_flow (forward matches the g0->g0.05 ruler)")
    ap.add_argument("--response-delta", type=float, default=0.05,
                    help="finite shift for R_flow; 0.05 matches the g0.05 half-shear ruler")
    ap.add_argument("--crowd", default=CROWD)
    ap.add_argument("--nn", default=NN)
    ap.add_argument("--output", default=None, help="optional .npz to save per-object arrays")
    args = ap.parse_args()

    ckpts = list(args.ckpt or [])
    if args.ckpt_glob:
        ckpts += sorted(glob.glob(args.ckpt_glob))
    if not ckpts:
        ap.error("give --ckpt and/or --ckpt-glob")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    print(f"device={device}  ckpts={len(ckpts)}  diff={args.response_difference}  delta={args.response_delta}",
          flush=True)
    for c in ckpts:
        print("   ", os.path.basename(c))

    # ---- SHARED ruler (base + truth R_hs + isolation), identical to the V2 sibling ----
    ru = load_ruler(args.g0_leg, args.gS_leg, args.max_case, args.true_re_min, args.true_mag_max,
                    args.iso_radius, args.crowd, args.nn, t0=t0)
    base, R_hs, iso, size, gmed = ru["base"], ru["R_hs"], ru["iso"], ru["size"], ru["gmed"]

    # ---- model: ensemble R_flow (V1 ConditionalMeanFlow mean-head finite diff) ----
    Rf_seeds = []
    for c in ckpts:
        bundle = load_measurement_model(c, device=device)
        Rf = model_selfresp(base, bundle, args.response_delta, args.response_difference, device)
        Rf_seeds.append(Rf)
        cf = bundle.metadata.get("condition_features", "?")
        print(f"  {os.path.basename(c)}  <R_flow>={np.nanmean(Rf[np.isfinite(Rf)]):+.4f}  "
              f"cond={cf}  ({time.time()-t0:.1f}s)", flush=True)
    Rf_ens = np.nanmean(np.stack(Rf_seeds, 0), axis=0)
    print(f"<R_flow ENS>={np.nanmean(Rf_ens[np.isfinite(Rf_ens)]):+.4f}  (N seeds={len(Rf_seeds)})", flush=True)

    good = np.isfinite(R_hs) & np.isfinite(Rf_ens)
    print_size_table(R_hs, Rf_ens, size, iso, good,
                     "ISOLATED acceptance set (nn_bright>%.0f\")" % args.iso_radius)
    print_size_table(R_hs, Rf_ens, size, np.ones(len(base), bool), good, "ALL objects (diagnostic)")

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        np.savez(args.output, R_hs=R_hs, R_flow=Rf_ens, R_flow_seeds=np.stack(Rf_seeds, 0),
                 size=size, iso=iso, good=good, gmed=gmed,
                 case=base["case"].to_numpy(int), input_index=base["input_index"].to_numpy(int),
                 size_edges=SIZE_EDGES, delta=args.response_delta, difference=args.response_difference,
                 nseeds=len(Rf_seeds))
        print(f"\nsaved {args.output}")
    print("SELFRESP_GAP_DONE", flush=True)


if __name__ == "__main__":
    main()
