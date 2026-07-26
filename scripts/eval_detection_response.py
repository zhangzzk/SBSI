"""Evaluate a trained detection classifier's induced shear-response b_model against the sim target
b_sim, globally and per blend bin. Uses the SAME centered finite-difference-through-the-shear-map
estimator as training (scripts/train_detection_response.epoch), so the numbers are directly the
quantity the response loss supervises.

  b_model(cell) = 0.5*[ (<s>_{P,e1} - <s>_{e1,0}) + (<s>_{P,e2} - <s>_{e2,0}) ] / delta
  <s>_{P} = P(detect | S_delta(scene))-weighted mean of the sheared shape; <s>_{,0} the plain
  P(detect|scene)-weighted mean at delta=0 (baseline subtraction).

Report links to Stage-3 step 1: the sim's true-shape detection selection is ~-1.9% (constgold
sheared-intrinsic, cont.159) / -2.0% (half-shear b_true) -> a BCE-only classifier gets ~+1.9%
(wrong sign); the response-aware one should track ~-2% incl. the blend split.
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
import torch

SBSI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SBSI_ROOT not in sys.path:
    sys.path.insert(0, SBSI_ROOT)
from sbs_shear.selection_model import load_selection_model            # noqa: E402
from sbs_shear.preprocessing import rescale                            # noqa: E402
from sbs_shear.sim_stream import stream_reservoir                      # noqa: E402
from scripts.train_detection_response import (                         # noqa: E402
    shifted_ctx, RAW_EXTRA, load_parent as _lp)
from sbs_shear.preprocessing import raw_columns_for_selection_features  # noqa: E402


def bin_ids(fr, ef, es, ed, nf, ns, nbl):
    fl = fr["r_input_p"].to_numpy(float); sz = fr["Re_input_p"].to_numpy(float)
    fi = np.clip(np.digitize(fl, ef) - 1, 0, nf - 1)
    si = np.clip(np.digitize(sz, es) - 1, 0, ns - 1)
    nbf = fr["neighbored"].astype(bool).to_numpy()
    dist = fr["distance"].to_numpy(float)
    di = np.where(nbf, 1 + np.clip(np.digitize(dist, ed) - 1, 0, nbl - 2), 0)
    return (fi * ns + si) * nbl + di, di


def bin_shift_np(binid, nb, w, s):
    cnt = np.bincount(binid, weights=np.ones_like(s), minlength=nb)
    sw = np.bincount(binid, weights=w * s, minlength=nb)
    sn = np.bincount(binid, weights=w, minlength=nb)
    su = np.bincount(binid, weights=s, minlength=nb)
    return sw / np.clip(sn, 1e-6, None) - su / np.clip(cnt, 1.0, None), cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--response-target-npz", required=True)
    ap.add_argument("--catalogue",
                    default="/project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_g0.0_train.feather")
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--max-rows", type=int, default=4_000_000)
    ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--target-column", default="detected")
    ap.add_argument("--output", default=None, help="npz: per-blend b_sim/b_model + globals, for plotting")
    args = ap.parse_args()

    bundle = load_selection_model(args.model, device="cpu")
    pre = bundle.preprocessor
    feats = list(pre.feature_names)
    rk = dict(pixel_rms=0.312, pixel_size=0.2, zero_mag=30.0, psf_fwhm=0.73, moffat_beta=2.224)

    cols = (raw_columns_for_selection_features(feats, available_columns=None)
            | {args.target_column} | set(RAW_EXTRA))
    frame = stream_reservoir(args.catalogue, cols, args.max_rows, seed=args.seed,
                             detected=None, shear_threshold=0.0)
    avail = set(frame.columns)
    frame = rescale(frame, **rk)
    y = frame[args.target_column].astype(bool).to_numpy()
    print(f"eval parent rows={len(frame):,}  det_frac={y.mean():.4f}  model={os.path.basename(args.model)}")

    tt = np.load(args.response_target_npz)
    ef, es, ed, b_sim = tt["edges_flux"], tt["edges_size"], tt["edges_dist"], tt["b_sim"]
    valid = tt["valid"] if "valid" in tt.files else np.ones_like(b_sim, bool)
    counts = tt["counts"]
    nf, ns, nbl = b_sim.shape
    nb = nf * ns * nbl
    delta = args.delta

    binid, di = bin_ids(frame, ef, es, ed, nf, ns, nbl)

    def pdet(ctx):
        # ctx is ALREADY the preprocessed (28-dim) tensor from shifted_ctx (= pre.transform_frame),
        # so call the raw model directly -- logits_from_raw_tensor would transform a second time.
        with torch.no_grad():
            lg = bundle.model(torch.as_tensor(ctx)).reshape(-1) / bundle.temperature
        return torch.sigmoid(lg).numpy().astype(float)

    ctx0, _ = shifted_ctx(frame, (1.0, 0.0), 0.0, pre, rk)
    ce1, se1 = shifted_ctx(frame, (1.0, 0.0), delta, pre, rk)
    ce2, se2 = shifted_ctx(frame, (0.0, 1.0), delta, pre, rk)
    se1_0 = frame["e1_input_rot0_p"].to_numpy(float)
    se2_0 = frame["e2_input_rot0_p"].to_numpy(float)
    w0 = pdet(ctx0); w1 = pdet(ce1); w2 = pdet(ce2)

    b1d, cnt = bin_shift_np(binid, nb, w1, se1.astype(float))
    b2d, _ = bin_shift_np(binid, nb, w2, se2.astype(float))
    b1_0, _ = bin_shift_np(binid, nb, w0, se1_0)
    b2_0, _ = bin_shift_np(binid, nb, w0, se2_0)
    bmodel = (0.5 * ((b1d - b1_0) + (b2d - b2_0)) / delta).reshape(nf, ns, nbl)

    bsim_g = float(tt["global_b"])
    # UN-BINNED global b_model = the OBSERVABLE detection bias on the detected-catalogue mean shape,
    # directly comparable to the target-builder's global_b (= [<s>_det - <s>_parent]/g).  Includes the
    # cross-cell reweighting that the per-cell (binned) numbers remove -- so this, not the per-cell
    # average, is the apples-to-apples match to b_sim/g = -2.01% / constgold -1.86%.
    # selection shift = P(det)-weighted mean MINUS the PLAIN (unweighted) mean of the SAME shape
    # (this cancels the shape's own shear response, leaving only the detection selection); then
    # centered by the delta=0 baseline, exactly like the per-cell _bin_shift but over all rows.
    def sel_shift(w, s):
        return np.sum(w * s) / np.sum(w) - np.mean(s)
    b1g = sel_shift(w1, se1.astype(float)) - sel_shift(w0, se1_0)
    b2g = sel_shift(w2, se2.astype(float)) - sel_shift(w0, se2_0)
    gmodel_obs = 0.5 * (b1g + b2g) / delta
    # count-weighted per-cell global (what the training log's <b_model/g> tracks)
    wv = (counts * valid).reshape(-1)
    gmodel_cell = float(np.nansum(bmodel.reshape(-1) * wv) / max(wv.sum(), 1))
    print(f"\nGLOBAL OBSERVABLE (un-binned, ~ R_detect):  b_sim/g = {bsim_g*100:+.3f}%   "
          f"b_model/g = {gmodel_obs*100:+.3f}%")
    print(f"per-cell count-weighted (training-log metric): b_model/g = {gmodel_cell*100:+.3f}%")
    print("per blend bin (count-weighted b/g over real cells):")
    print(f"  {'blend':>10} {'b_sim/g':>9} {'b_model/g':>10} {'N':>14}")
    blend_bs, blend_bm, blend_lab = [], [], []
    for c in range(nbl):
        tag = "ISOLATED" if c == 0 else f"blend d{c}"
        w = (counts[:, :, c] * valid[:, :, c])
        bs = np.nansum(b_sim[:, :, c] * w) / max(w.sum(), 1)
        bm = np.nansum(bmodel[:, :, c] * w) / max(w.sum(), 1)
        blend_bs.append(bs); blend_bm.append(bm); blend_lab.append(tag)
        print(f"  {tag:>10} {bs*100:>+8.3f}% {bm*100:>+9.3f}% {int(counts[:,:,c].sum()):>14,}")
    if args.output:
        np.savez(args.output, global_b_sim=bsim_g, global_b_model_obs=gmodel_obs,
                 global_b_model_cell=gmodel_cell, blend_b_sim=np.array(blend_bs),
                 blend_b_model=np.array(blend_bm), blend_labels=np.array(blend_lab),
                 model=os.path.basename(args.model))
        print(f"saved {args.output}")
    print("EVAL_DET_RESPONSE_DONE")


if __name__ == "__main__":
    main()
