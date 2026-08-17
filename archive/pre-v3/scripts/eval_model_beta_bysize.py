"""MODEL (detection classifier) predicted detection response vs BETA, faceted by SIZE and by MAG -- to
overlay on the constgold SIM (scripts/eval_detection_beta_bysize.py) for the sim-vs-model plot (cont.165).

Runs the cont.160 shear-response-regularized classifier P(detect | true props + blend) on the det_meas
parent and forms its induced detection response with the SAME centered-finite-difference-through-the-
shear-map estimator used in training/eval_detection_response.py, binned on the SAME (primary x beta)
grids the sim run saved (--edges-npz), for both the SIZE facet (fixed mag band) and the MAG facet
(pool size). beta computed identically to the sim (isolated -> beta=0). FIREWALL: read-only.
"""
from __future__ import annotations
import argparse
import os
import sys

import numpy as np
import torch

SBSI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SBSI_ROOT not in sys.path:
    sys.path.insert(0, SBSI_ROOT)
from sbs_shear.selection_model import load_selection_model                 # noqa: E402
from sbs_shear.preprocessing import rescale, raw_columns_for_selection_features  # noqa: E402
from sbs_shear.sim_stream import stream_reservoir                          # noqa: E402
from scripts.train_detection_response import shifted_ctx, RAW_EXTRA        # noqa: E402
from scripts.eval_detection_response import bin_shift_np                   # noqa: E402
from scripts.eval_detection_severity import PSF_SIZE                       # noqa: E402


def beta_frame(frame):
    Re = frame["Re_input_p"].to_numpy(float)
    rp = frame["r_input_p"].to_numpy(float)
    rs = frame["r_input_s"].to_numpy(float)
    dist = frame["distance"].to_numpy(float)
    post = np.sqrt(Re ** 2 + PSF_SIZE ** 2)
    with np.errstate(over="ignore", invalid="ignore"):
        K = np.exp(-0.5 * (dist / post) ** 2)
        beta = (10.0 ** (-0.4 * rs) * K) / (10.0 ** (-0.4 * rp) + 10.0 ** (-0.4 * rs) * K)
    nb = frame["neighbored"].astype(bool).to_numpy() if "neighbored" in frame.columns \
        else np.isfinite(dist)
    beta = np.where(np.isfinite(beta) & nb, beta, 0.0)
    return beta, Re, rp


def facet_model(name, prim, pedges, pmask, beta, bedges, wk, shapes, delta, labels):
    w0, w1, w2 = wk
    se1, se2, se1_0, se2_0 = shapes
    n_p = len(pedges) - 1
    nbeta_pos = len(bedges) - 1
    nbb = nbeta_pos + 1
    pi = np.clip(np.digitize(prim, pedges) - 1, 0, n_p - 1)
    bi = np.where(beta <= 0, 0, 1 + np.clip(np.digitize(beta, bedges) - 1, 0, nbeta_pos - 1))
    binid = np.where(pmask, pi * nbb + bi, n_p * nbb)
    nb_tot = n_p * nbb + 1
    b1d, cnt = bin_shift_np(binid, nb_tot, w1, se1)
    b2d, _ = bin_shift_np(binid, nb_tot, w2, se2)
    b1_0, _ = bin_shift_np(binid, nb_tot, w0, se1_0)
    b2_0, _ = bin_shift_np(binid, nb_tot, w0, se2_0)
    bmodel = 0.5 * ((b1d - b1_0) + (b2d - b2_0)) / delta
    out = {}
    for pj, lab in enumerate(labels):
        sl = slice(pj * nbb, (pj + 1) * nbb)
        out[f"{name}__{lab}__db_model"] = bmodel[sl]
        out[f"{name}__{lab}__N"] = cnt[sl].astype(float)
        bmed = []
        for b in range(nbb):
            m = binid == pj * nbb + b
            bmed.append(float(np.nanmedian(beta[m])) if m.sum() > 50 else np.nan)
        out[f"{name}__{lab}__beta_med"] = np.array(bmed)
        print(f"  {lab}: " + "  ".join(
            f"{100*bmodel[pj*nbb+b]:+.2f}%" if cnt[pj*nbb+b] > 200 else "  --  "
            for b in range(nbb)))
    return out


def main():
    ap = argparse.ArgumentParser()
    base = "/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/"
    ap.add_argument("--model", default=base + "det_response_mlp_lam300_s7.pt")
    ap.add_argument("--edges-npz", default=base + "detection_beta_bysize_v2.npz")
    ap.add_argument("--catalogue",
                    default="/project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_g0.0_train.feather")
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--max-rows", type=int, default=8_000_000)
    ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    bundle = load_selection_model(args.model, device="cpu")
    pre = bundle.preprocessor
    feats = list(pre.feature_names)
    rk = dict(pixel_rms=0.312, pixel_size=0.2, zero_mag=30.0, psf_fwhm=0.73, moffat_beta=2.224)
    cols = (raw_columns_for_selection_features(feats, available_columns=None) | set(RAW_EXTRA)
            | {"Re_input_p", "r_input_p", "r_input_s", "distance", "neighbored", "detected"})
    frame = stream_reservoir(args.catalogue, cols, args.max_rows, seed=args.seed,
                             detected=None, shear_threshold=0.0)
    frame = rescale(frame, **rk)
    beta, Re, mag = beta_frame(frame)
    print(f"parent rows={len(frame):,}", flush=True)

    def pdet(ctx):
        with torch.no_grad():
            lg = bundle.model(torch.as_tensor(ctx)).reshape(-1) / bundle.temperature
        return torch.sigmoid(lg).numpy().astype(float)

    delta = args.delta
    ctx0, _ = shifted_ctx(frame, (1.0, 0.0), 0.0, pre, rk)
    ce1, se1 = shifted_ctx(frame, (1.0, 0.0), delta, pre, rk)
    ce2, se2 = shifted_ctx(frame, (0.0, 1.0), delta, pre, rk)
    se1_0 = frame["e1_input_rot0_p"].to_numpy(float)
    se2_0 = frame["e2_input_rot0_p"].to_numpy(float)
    wk = (pdet(ctx0), pdet(ce1), pdet(ce2))
    shapes = (se1.astype(float), se2.astype(float), se1_0, se2_0)

    z = np.load(args.edges_npz, allow_pickle=True)
    out = {"model": os.path.basename(args.model)}
    # SIZE facet: primary=Re, restricted to the sim's mag band
    slabels = [str(s) for s in z["SIZE__labels"]]
    smask = (mag >= float(z["mag_lo"])) & (mag < float(z["mag_hi"]))
    print("SIZE facet (b_model):")
    out.update(facet_model("SIZE", Re, z["SIZE__pedges"], smask, beta, z["SIZE__beta_edges"],
                           wk, shapes, delta, slabels))
    out["SIZE__labels"] = z["SIZE__labels"]; out["SIZE__beta_edges"] = z["SIZE__beta_edges"]
    out["SIZE__pedges"] = z["SIZE__pedges"]
    # MAG facet: primary=mag, over the mag-edge range
    mlabels = [str(s) for s in z["MAG__labels"]]
    medges = z["MAG__pedges"]
    mmask = (mag >= float(medges[0])) & (mag < float(medges[-1]))
    print("MAG facet (b_model):")
    out.update(facet_model("MAG", mag, medges, mmask, beta, z["MAG__beta_edges"],
                           wk, shapes, delta, mlabels))
    out["MAG__labels"] = z["MAG__labels"]; out["MAG__beta_edges"] = z["MAG__beta_edges"]
    out["MAG__pedges"] = z["MAG__pedges"]

    if args.output:
        np.savez(args.output, **out)
        print(f"saved {args.output}", flush=True)
    print("MODEL_BETA_BYSIZE_DONE", flush=True)


if __name__ == "__main__":
    main()
