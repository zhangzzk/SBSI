"""fig2/fig5 (V2) with the HALF-SHEAR target: does the V2 flow reproduce the self-response it was TRAINED on?

Owner steer 2026-07-23: use the half-shear det_meas self-response as the truth curve, NOT constgold.
Rationale: the V2 flow learns from det_meas half-shear (primary-only sheared; the random-direction field
averages the neighbour-blend response to ~0), so the measured half-shear forward-difference self-response is
EXACTLY the flow's training target. Comparing R_flow against it isolates pure flow-quality from the
neighbour-blend physics (which constgold folds in and R_blend must supply). Both sides are ngmix units on the
SAME sim family -> NO bridge, NO R_blend: a clean apples-to-apples R_self test.

TARGET (measured):  R_hs = [(e_g - e_0).ghat]/g  per object, matched on (case,input_index), both-detected,
                    e = measured_ngmix (compare_selfresp_sims.halfshear convention).
MODEL (predicted):  R_flow = flow mean-head central-difference self-response (primary_only shear), sw_th400
                    seed421, evaluated on the SAME det_meas rows (true-property + inline _s neighbour cond).

Domain = flow training domain: source_select_selection(DEFAULT_SELECTION_CUTS) + true cut Re>0.3, r<26, detected.
Saves per-object arrays -> plotting/plot_v2_halfshear_figs.py makes fig2 (mag/size/nbr_flux, all objects) and
fig5 (mag/size, 7"-isolated). FIREWALL: nothing trains; det_meas is the training sim (no constgold here).
"""
from __future__ import annotations

import pathlib as _pathlib, sys as _sys  # noqa: E402  -- make `sbs_shear` importable
_sys.path.insert(0, str(next(p for p in _pathlib.Path(__file__).resolve().parents
                             if (p / 'sbs_shear').is_dir())))
from sbs_shear import paths  # noqa: E402
import argparse, os, sys, time
import numpy as np, pandas as pd
import pyarrow as pa, pyarrow.ipc as ipc, pyarrow.feather as pf
import torch

SBSI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SBSI_ROOT not in sys.path:
    sys.path.insert(0, SBSI_ROOT)
from sbs_shear.preprocessing import DEFAULT_SELECTION_CUTS, source_select_selection  # noqa: E402
from scripts.eval_constgold_closure import load_model, run_model_on  # noqa: E402

CAT = f"{paths.CATALOGUE_DIR}/"
CROWD = f"{paths.RESULTS_DIR}/crowd_flux_conc_c0-199.feather"
NN = f"{paths.CACHE_DIR}/derisk/nn_dist_c0-39.feather"
CKPT = f"{paths.CACHE_DIR}/forward_proto/forward_sw_th400_seed421_joint.pt"

# columns the flow's build_contexts / intrinsic_shape / shifted_feature_frame / neighbor_padded touch
FLOW_COLS = ["case", "input_index", "neighbored", "distance", "polarization_angle",
             "shear_component_convention",
             "Re_input_p", "Re_input_s", "r_input_p", "r_input_s",
             "sersic_n_input_p", "sersic_n_input_s", "redshift_input_p", "redshift_input_s",
             "e1_input_rot0_p", "e2_input_rot0_p", "e1_input_rot0_s", "e2_input_rot0_s",
             "axis_ratio_input_p", "position_angle_input_p",
             "axis_ratio_input_s", "position_angle_input_s", "detected"]
NGMIX = ["measured_ngmix_g1", "measured_ngmix_g2"]
GAMMA = ["gamma1_input_p", "gamma2_input_p"]

MAG_EDGES = np.array([18.0, 24.0, 25.0, 26.0])
SIZE_EDGES = np.array([0.30, 0.38, 0.50, 1.50])


def read_leg(path, cols, max_case):
    parts = []
    with ipc.open_file(path) as r:
        av = set(r.schema.names); use = [c for c in cols if c in av]
        for bi in range(r.num_record_batches):
            b = pa.Table.from_batches([r.get_batch(bi)]).select(use).to_pandas()
            if max_case is not None:
                if int(b["case"].min()) > max_case:
                    break
                b = b[b["case"] <= max_case]
            if len(b):
                parts.append(b)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cols)


def domain_cut(df):
    """flow training domain: base selection + true cut (Re>0.3, r_input_p<26) + detected."""
    df = source_select_selection(df, cuts=DEFAULT_SELECTION_CUTS)
    det = df["detected"].astype(bool).to_numpy() if "detected" in df.columns else np.ones(len(df), bool)
    m = det & (df["Re_input_p"].to_numpy(float) > 0.3) & (df["r_input_p"].to_numpy(float) < 26.0)
    return df[m].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-case", type=int, default=19)          # g0.02_test only spans 0-19
    ap.add_argument("--gtag", default="g0.02")                   # forward-diff leg
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--output", default="/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/halfshear_flowfig_th400.npz")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    print(f"device={device}  ckpt={os.path.basename(args.ckpt)}  leg={args.gtag}  max_case={args.max_case}", flush=True)

    # ---- g0 (zero) and g (sheared) legs, both-detected matched, flow domain ----
    g0 = domain_cut(read_leg(CAT + "det_meas_ngmix_g0.0_train.feather",
                             ["case", "input_index", "detected", "Re_input_p", "r_input_p",
                              "neighbored", "distance"] + NGMIX, args.max_case))
    gS = domain_cut(read_leg(CAT + f"det_meas_ngmix_{args.gtag}_test.feather",
                             FLOW_COLS + NGMIX + GAMMA, args.max_case))
    gp = np.hypot(gS["gamma1_input_p"].to_numpy(float), gS["gamma2_input_p"].to_numpy(float))
    gS = gS[gp > 1e-6].reset_index(drop=True)
    g0 = g0.drop_duplicates(["case", "input_index"])
    gS = gS.drop_duplicates(["case", "input_index"])
    base = gS.merge(g0[["case", "input_index"] + NGMIX], on=["case", "input_index"], suffixes=("_g", "_0"))
    print(f"matched both-detected true-cut: N={len(base):,}  cases={base['case'].nunique()}  ({time.time()-t0:.1f}s)", flush=True)

    # ---- target: measured half-shear self-response R_hs (forward diff, ngmix) ----
    gp = np.hypot(base["gamma1_input_p"].to_numpy(float), base["gamma2_input_p"].to_numpy(float))
    gmed = float(np.median(gp)); gh1 = base["gamma1_input_p"].to_numpy(float) / gp; gh2 = base["gamma2_input_p"].to_numpy(float) / gp
    de1 = base["measured_ngmix_g1_g"].to_numpy(float) - base["measured_ngmix_g1_0"].to_numpy(float)
    de2 = base["measured_ngmix_g2_g"].to_numpy(float) - base["measured_ngmix_g2_0"].to_numpy(float)
    R_hs = (de1 * gh1 + de2 * gh2) / gmed
    print(f"g_med={gmed:.4f}  <R_hs>={np.nanmean(R_hs[np.isfinite(R_hs)]):+.4f}", flush=True)

    # ---- model: flow R_flow (primary-only self-response, ngmix units) on the SAME rows ----
    model, pp, ns, sc, delta = load_model(args.ckpt, device)
    Rf, Rd = run_model_on(base, model, pp, ns, sc, delta, device)
    print(f"flow delta={delta}  <R_flow>={np.nanmean(Rf[np.isfinite(Rf)]):+.4f}  <dP/dg>={np.nanmean(Rd[np.isfinite(Rd)]):+.4f}  ({time.time()-t0:.1f}s)", flush=True)

    mag = base["r_input_p"].to_numpy(float); size = base["Re_input_p"].to_numpy(float)
    case = base["case"].to_numpy(int); iidx = base["input_index"].to_numpy(int)

    # ---- joins: neighbour flux + nn_dist_bright (7" isolation) ----
    nbf = pf.read_table(CROWD).to_pandas()[["case", "input_index", "nbr_flux_near"]]
    nnl = pf.read_table(NN).to_pandas()[["case", "input_index", "nn_dist_bright"]]
    j = base[["case", "input_index"]].merge(nbf, on=["case", "input_index"], how="left") \
                                     .merge(nnl, on=["case", "input_index"], how="left")
    nbr_flux = j["nbr_flux_near"].to_numpy(float)
    nnb = j["nn_dist_bright"].to_numpy(float)
    iso7 = (~np.isfinite(nnb)) | (nnb > 7.0)      # no brighter true neighbour within 7"
    print(f"nbr_flux matched {np.isfinite(nbr_flux).mean():.1%}  nn matched {np.isfinite(nnb).mean() or 0:.1%}  "
          f"iso7 frac={iso7.mean():.1%}", flush=True)

    good = np.isfinite(R_hs) & np.isfinite(Rf)
    print(f"finite both: {good.sum():,}/{len(good):,}", flush=True)

    # ---- quick per-mag / per-size table (all + iso7) ----
    def tab(sel, label):
        print(f"\n[{label}] N={int(sel.sum()):,}")
        print(f"  {'bin':16s} {'R_hs':>8} {'R_flow':>8} {'ratio-1%':>9}")
        for tag, edges, v in [("mag", MAG_EDGES, mag), ("size", SIZE_EDGES, size)]:
            for i in range(len(edges) - 1):
                s = sel & (v >= edges[i]) & (v < edges[i + 1]) & good
                if s.sum() < 30:
                    continue
                a = float(np.mean(R_hs[s])); b = float(np.mean(Rf[s]))
                print(f"  {tag}[{edges[i]:.2f},{edges[i+1]:.2f}) {a:+8.4f} {b:+8.4f} {(b/a-1)*100 if a else np.nan:+9.1f}")
    tab(np.ones(len(base), bool), "ALL objects")
    tab(iso7, "ISOLATED (nn_bright>7)")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.savez(args.output, R_hs=R_hs, R_flow=Rf, Rdet=Rd, mag=mag, size=size,
             nbr_flux=nbr_flux, nn_bright=nnb, iso7=iso7, good=good,
             case=case, input_index=iidx, gmed=gmed, delta=delta,
             ckpt=os.path.basename(args.ckpt), gtag=args.gtag)
    print(f"\nsaved {args.output}   ({time.time()-t0:.1f}s)")
    print("HALFSHEAR_FLOWFIG_DONE", flush=True)


if __name__ == "__main__":
    main()
