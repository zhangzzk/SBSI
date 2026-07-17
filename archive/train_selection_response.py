"""Response-aware selection classifier (Direction 2): fix the wrong-sign selection response.

The g=0 classifier P(s=1|x) has an induced selection response b_model/g=+1.9% while the sim's
is b_true/g=-1.8% (detection DROPS for shear-elongated galaxies).  We add a response-aware
term (the selection analog of the measurement flow's response loss): shear the true scene by
the analytic map S_delta per batch, and push the classifier's induced selection-shape shift
b_model -> b_sim, resolved per (true flux x size x blend) bin (blend-aware).

  b_model(bin) = 0.5*[ (<s>_{P,e1} - <s>_{e1}) + (<s>_{P,e2} - <s>_{e2}) ] / delta
     where <s>_{P} = P(s=1|S_delta(scene))-weighted mean of the sheared shape, <s> the plain
     mean, in each bin.  Supervise toward b_sim(bin) from compute_selection_target_blend.py.

Loss = BCE(detected) + lam * sum_bin cnt*(b_model - b_sim)^2 / sum cnt.
Trained on the PARENT sample (detected+undetected) of the g=0 catalogue.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

SBSI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SBSI_ROOT not in sys.path:
    sys.path.insert(0, SBSI_ROOT)

from sbs_shear.selection_model import (  # noqa: E402
    SelectionMLP, TabularPreprocessor, save_selection_model, load_selection_model)
from sbs_shear.preprocessing import raw_columns_for_selection_features, rescale  # noqa: E402
from sbs_shear.shear_map import apply_shear_to_ellipticity  # noqa: E402
from sbs_shear.sim_stream import stream_reservoir  # noqa: E402

RAW_EXTRA = ["e1_input_rot0_p", "e2_input_rot0_p", "gamma1_input_p", "gamma2_input_p",
             "r_input_p", "Re_input_p", "neighbored", "distance", "measured_ngmix_g1"]


def load_parent(args, feature_names):
    """Parent sample (detected+undetected) from the g=0 catalogue; no shear filter."""
    cols = lambda avail: (raw_columns_for_selection_features(feature_names, available_columns=avail)  # noqa: E731
                          | {args.target_column} | {c for c in RAW_EXTRA if c in avail})
    res = stream_reservoir(args.catalogue, cols, args.max_rows, seed=args.seed,
                           detected=None, shear_threshold=0.0, max_read_batches=args.max_read_batches)
    print(f"  parent rows={len(res):,}  det_frac={res[args.target_column].astype(float).mean():.4f}")
    return res


def shifted_ctx(frame, gdir, delta, pre, rescale_kwargs):
    """Standardized selection features after shearing the primary scene by S_delta along gdir,
    plus the sheared projected shape component (e1 if gdir=(1,0), e2 if (0,1))."""
    f = frame.copy()
    s1, s2 = apply_shear_to_ellipticity(f["e1_input_rot0_p"].to_numpy(float),
                                        f["e2_input_rot0_p"].to_numpy(float),
                                        delta * gdir[0], delta * gdir[1])
    f["e1_input_rot0_p"] = s1
    f["e2_input_rot0_p"] = s2
    for c in ("gamma1_input_p", "gamma2_input_p"):
        if c in f.columns:
            f[c] = 0.0
    f = rescale(f, **rescale_kwargs)
    ctx = pre.transform_frame(f).astype(np.float32)
    s_proj = (s1 if gdir[0] > 0 else s2).astype(np.float32)
    return ctx, s_proj


def make_loader(frame, y, pre, delta, rescale_kwargs, bin_id_fn, batch_size, shuffle, num_workers):
    ctx0, _ = shifted_ctx(frame, (1.0, 0.0), 0.0, pre, rescale_kwargs)
    ce1, se1 = shifted_ctx(frame, (1.0, 0.0), delta, pre, rescale_kwargs)
    ce2, se2 = shifted_ctx(frame, (0.0, 1.0), delta, pre, rescale_kwargs)
    # delta=0 shape projections (= intrinsic e1,e2 since S_0=identity) for the baseline
    # subtraction that removes the static selection-shape correlation b_model(0).
    se1_0 = frame["e1_input_rot0_p"].to_numpy(np.float32)
    se2_0 = frame["e2_input_rot0_p"].to_numpy(np.float32)
    binid = bin_id_fn(frame).astype(np.int64)
    ds = torch.utils.data.TensorDataset(
        torch.as_tensor(y, dtype=torch.float32),
        torch.as_tensor(ctx0), torch.as_tensor(ce1), torch.as_tensor(ce2),
        torch.as_tensor(se1), torch.as_tensor(se2),
        torch.as_tensor(se1_0), torch.as_tensor(se2_0), torch.as_tensor(binid))
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                                       num_workers=num_workers, pin_memory=True)


def _bin_shift(binid, nb, w, s, device):
    """Per-bin selection shift  b = <w*s>/<w> - <s>  via scatter-add."""
    z = lambda: torch.zeros(nb, device=device)  # noqa: E731
    cnt = z().index_add_(0, binid, torch.ones_like(s))
    sw = z().index_add_(0, binid, w * s)
    sn = z().index_add_(0, binid, w)
    su = z().index_add_(0, binid, s)
    return sw / sn.clamp_min(1e-6) - su / cnt.clamp_min(1.0), cnt


def epoch(model, loader, device, delta, bt, valid, lam, optimizer=None, min_bin=200):
    """BCE + per-cell selection-response loss. b_model is a CENTERED (baseline-subtracted)
    estimator [b(delta) - b(0)]/delta so the static (delta=0) selection-shape correlation is
    removed instead of being amplified by 1/delta."""
    training = optimizer is not None
    model.train(training)
    nb = bt.numel()
    tb = tr = n = 0.0
    rmean_sum = rmean_n = 0.0
    for y, ctx0, ce1, ce2, se1, se2, se1_0, se2_0, binid in loader:
        y = y.to(device); ctx0 = ctx0.to(device); ce1 = ce1.to(device); ce2 = ce2.to(device)
        se1 = se1.to(device); se2 = se2.to(device)
        se1_0 = se1_0.to(device); se2_0 = se2_0.to(device); binid = binid.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        logit0 = model(ctx0).reshape(-1)
        bce = F.binary_cross_entropy_with_logits(logit0, y)
        w0 = torch.sigmoid(logit0)                       # reuse the BCE forward for the delta=0 baseline
        w1 = torch.sigmoid(model(ce1).reshape(-1))
        w2 = torch.sigmoid(model(ce2).reshape(-1))
        b1d, cnt = _bin_shift(binid, nb, w1, se1, device)
        b2d, _ = _bin_shift(binid, nb, w2, se2, device)
        b1_0, _ = _bin_shift(binid, nb, w0, se1_0, device)   # static baseline (delta=0)
        b2_0, _ = _bin_shift(binid, nb, w0, se2_0, device)
        bmodel = 0.5 * ((b1d - b1_0) + (b2d - b2_0)) / delta
        present = (cnt > min_bin) & valid                    # skip thin cells whose target is a global fallback
        denom = (cnt * present).sum().clamp_min(1.0)
        resp = (((bmodel - bt) ** 2 * cnt) * present).sum() / denom
        loss = bce + lam * resp
        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        bs = len(y)
        tb += float(bce) * bs; tr += float(resp) * bs; n += bs
        rmean_sum += float((bmodel * cnt * present).sum()); rmean_n += float(denom)
    return tb / n, tr / n, rmean_sum / max(rmean_n, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalogue", default="/project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_g0.0_train.feather")
    ap.add_argument("--reference-model",
                    default=os.path.join(SBSI_ROOT, "models/selection_mlp_g0_shearfree_v1.pt"),
                    help="existing classifier: copy its feature set + architecture for consistency")
    ap.add_argument("--response-target-npz", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--target-column", default="detected")
    ap.add_argument("--response-weight", type=float, default=300.0)
    ap.add_argument("--response-delta", type=float, default=0.05)
    ap.add_argument("--max-rows", type=int, default=8_000_000)
    ap.add_argument("--max-read-batches", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=32768)
    ap.add_argument("--lr", type=float, default=1.0e-3)
    ap.add_argument("--weight-decay", type=float, default=1.0e-4)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--validation-size", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default=None)
    ap.add_argument("--pixel-rms", type=float, default=0.312)
    ap.add_argument("--pixel-size", type=float, default=0.2)
    ap.add_argument("--zero-mag", type=float, default=30.0)
    ap.add_argument("--psf-fwhm", type=float, default=0.73)
    ap.add_argument("--moffat-beta", type=float, default=2.224)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    rk = dict(pixel_rms=args.pixel_rms, pixel_size=args.pixel_size, zero_mag=args.zero_mag,
              psf_fwhm=args.psf_fwhm, moffat_beta=args.moffat_beta)

    ref = load_selection_model(args.reference_model, device="cpu")
    feature_names = list(ref.preprocessor.feature_names)
    _ckpt = torch.load(args.reference_model, map_location="cpu", weights_only=False)
    cfg = dict(_ckpt.get("model_config", {}))
    hidden = int(cfg.get("hidden_dim", 256)); n_layers = int(cfg.get("n_layers", 4))
    act = cfg.get("activation", "silu")
    print(f"feature set ({len(feature_names)}): {feature_names}")
    print(f"arch: hidden={hidden} n_layers={n_layers} act={act}  response lam={args.response_weight} delta={args.response_delta}")

    frame = load_parent(args, feature_names)
    frame = rescale(frame, **rk)
    pre = TabularPreprocessor.fit(frame, feature_names)
    # label = DETECTED. (ngmix-convergence dropped as a selection channel: the ~50%
    # "non-converged" are the `--targets secondaries` bookkeeping split, not physics.)
    y = frame[args.target_column].astype(bool).to_numpy()
    print(f"  selection = detected; sel_frac={y.mean():.4f}")
    y = y.astype(np.float32)

    tt = np.load(args.response_target_npz)
    ef, es, ed, b_sim = tt["edges_flux"], tt["edges_size"], tt["edges_dist"], tt["b_sim"]
    nf, ns, nbl = b_sim.shape
    bin_targets = torch.as_tensor(np.asarray(b_sim).reshape(-1), dtype=torch.float32, device=device)
    # cells whose target was a global fallback (too few rows) must NOT be supervised toward
    # that fallback; the builder flags real cells in `valid` (all-True if absent for back-compat).
    valid_np = tt["valid"].reshape(-1) if "valid" in tt.files else np.ones(b_sim.size, bool)
    valid = torch.as_tensor(valid_np.astype(bool), device=device)
    print(f"target {nf}x{ns}x{nbl} bins ({int(valid_np.sum())}/{valid_np.size} real, rest fallback), "
          f"b_sim/g {np.nanmin(b_sim)*100:+.2f}%..{np.nanmax(b_sim)*100:+.2f}%, global {float(tt['global_b'])*100:+.3f}%")

    def bin_id_fn(fr):
        fl = fr["r_input_p"].to_numpy(float); sz = fr["Re_input_p"].to_numpy(float)
        fi = np.clip(np.digitize(fl, ef) - 1, 0, nf - 1)
        si = np.clip(np.digitize(sz, es) - 1, 0, ns - 1)
        nbf = fr["neighbored"].astype(bool).to_numpy()
        dist = fr["distance"].to_numpy(float)
        di = np.where(nbf, 1 + np.clip(np.digitize(dist, ed) - 1, 0, nbl - 2), 0)
        return (fi * ns + si) * nbl + di

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(frame))
    nval = int(args.validation_size * len(frame))
    vidx, tidx = perm[:nval], perm[nval:]
    tr_df = frame.iloc[tidx].reset_index(drop=True); va_df = frame.iloc[vidx].reset_index(drop=True)
    print("  building shifted selection contexts (train/val) ...")
    tl = make_loader(tr_df, y[tidx], pre, args.response_delta, rk, bin_id_fn, args.batch_size, True, args.num_workers)
    vl = make_loader(va_df, y[vidx], pre, args.response_delta, rk, bin_id_fn, args.batch_size, False, args.num_workers)

    model = SelectionMLP(input_dim=pre.output_dim, hidden_dim=hidden, n_layers=n_layers, activation=act).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best = np.inf; best_state = None; t0 = time.time()
    print("\n--- training response-aware selection classifier ---")
    for ep in range(1, args.epochs + 1):
        tb, trsp, _ = epoch(model, tl, device, args.response_delta, bin_targets, valid, args.response_weight, optimizer=opt)
        vb, vrsp, vR = epoch(model, vl, device, args.response_delta, bin_targets, valid, args.response_weight)
        sel = vb + args.response_weight * vrsp
        if sel < best:
            best = sel; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if ep % 2 == 0 or ep == 1:
            print(f"epoch {ep:03d}: bce={tb:.4f}/{vb:.4f}  resp={vrsp:.3e}  "
                  f"<b_model/g>(val)={vR*100:+.3f}% (target global {float(tt['global_b'])*100:+.3f}%)", flush=True)
    if best_state:
        model.load_state_dict(best_state)
    model_config = {"input_dim": pre.output_dim, "hidden_dim": hidden, "n_layers": n_layers,
                    "dropout": 0.0, "activation": act}
    save_selection_model(args.output, model, pre, model_config, temperature=1.0,
                         metadata={"model_config": model_config, "feature_names": feature_names,
                                   "response_aware": True, "response_target": os.path.basename(args.response_target_npz)})
    print(f"\nSaved: {args.output}  (train time {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
