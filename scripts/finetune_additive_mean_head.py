"""Fine-tune the measurement flow mean head to absorb g=0 additive residuals.

This modifies a measurement-flow checkpoint directly.  It is not a sidecar
gold subtraction and not a separate c model:

  1. load the existing ConditionalMeanFlow,
  2. freeze the residual flow,
  3. optimize only mean_net so the full predicted zero-shear mean
     (mean_net + frozen residual-flow mean) matches measured ngmix means in
     g=0 property bins.

The fit rows are unsheared g=0 only; held-out g=0 and gold transfer are checked
by diagnose_additive_origin.py after saving the new checkpoint.
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time

import numpy as np

SBSI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SBSI_ROOT not in sys.path:
    sys.path.insert(0, SBSI_ROOT)

from sbs_shear.measurement_model import (  # noqa: E402
    ConditionalMeanFlow,
    TargetStandardizer,
    build_flow,
    save_measurement_model,
)
from sbs_shear.preprocessing import rescale  # noqa: E402
from sbs_shear.selection_model import TabularPreprocessor  # noqa: E402
from scripts.diagnose_additive_origin import RK, load_g0  # noqa: E402
from scripts.response_ratio_diagnostic import _shape_target_indices  # noqa: E402


def _qedges(x, n):
    x = np.asarray(x, float)
    good = np.isfinite(x)
    if good.sum() < n:
        raise ValueError("not enough finite rows for quantile edges")
    e = np.quantile(x[good], np.linspace(0, 1, n + 1))
    e[0] -= 1e-9
    e[-1] += 1e-9
    return e


def _crowd_edges(x, n_pos, eps):
    x = np.asarray(x, float)
    pos = np.isfinite(x) & (x >= eps)
    if pos.sum() < n_pos:
        return np.array([], dtype=float)
    e = np.quantile(x[pos], np.linspace(0, 1, n_pos + 1))
    e[0] -= 1e-9
    e[-1] += 1e-9
    return e


def _bin_ids(df, mag_edges, re_edges, crowd_edges, crowd_col, crowd_eps):
    mag = df["r_input_p"].to_numpy(float)
    re = df["Re_input_p"].to_numpy(float)
    crowd = df[crowd_col].to_numpy(float) if crowd_col in df.columns else np.zeros(len(df))
    im = np.clip(np.digitize(mag, mag_edges) - 1, 0, len(mag_edges) - 2)
    ir = np.clip(np.digitize(re, re_edges) - 1, 0, len(re_edges) - 2)
    if len(crowd_edges) == 0:
        ic = np.zeros(len(df), dtype=np.int64)
        nc = 1
    else:
        pos = np.isfinite(crowd) & (crowd >= crowd_eps)
        ic = np.where(pos, 1 + np.clip(np.digitize(crowd, crowd_edges) - 1, 0, len(crowd_edges) - 2), 0)
        nc = len(crowd_edges)
    nm = len(mag_edges) - 1
    nr = len(re_edges) - 1
    return ((im.astype(np.int64) * nr + ir.astype(np.int64)) * nc + ic.astype(np.int64)), nm, nr, nc


def make_bins(train_df, args):
    mag_edges = np.asarray(args.mag_edges, dtype=float)
    re_edges = _qedges(train_df["Re_input_p"].to_numpy(float), args.n_size)
    crowd_edges = _crowd_edges(train_df[args.crowd_col].to_numpy(float), args.n_crowd_pos, args.crowd_eps)
    train_bin, nm, nr, nc = _bin_ids(train_df, mag_edges, re_edges, crowd_edges, args.crowd_col, args.crowd_eps)
    return {
        "mag_edges": mag_edges,
        "re_edges": re_edges,
        "crowd_edges": crowd_edges,
        "crowd_col": args.crowd_col,
        "crowd_eps": float(args.crowd_eps),
        "n_bins": int(nm * nr * nc),
        "train_bin": train_bin,
    }


def prepare_frame(df, condition_preprocessor, target_transform):
    frame = df.copy()
    frame["gamma1_input_p"] = 0.0
    frame["gamma2_input_p"] = 0.0
    frame = rescale(frame, **RK)
    context = condition_preprocessor.transform_frame(frame)
    target = target_transform.transform_frame(frame)
    return frame, context, target


def bin_targets(df, binid, target_names, shape_idx, n_bins, min_count):
    vals = df[target_names].to_numpy(dtype=np.float32, copy=False)[:, list(shape_idx)]
    sums = np.zeros((n_bins, 2), dtype=np.float64)
    counts = np.zeros(n_bins, dtype=np.float64)
    good = np.isfinite(vals).all(axis=1)
    np.add.at(sums, binid[good], vals[good])
    np.add.at(counts, binid[good], 1.0)
    global_mean = np.nanmean(vals, axis=0)
    means = np.divide(sums, counts[:, None], out=np.tile(global_mean, (n_bins, 1)), where=counts[:, None] > 0)
    active = counts >= min_count
    return means.astype(np.float32), counts.astype(np.float32), active


def freeze_except_mean(model):
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.mean_net.parameters():
        p.requires_grad_(True)
    return [p for p in model.mean_net.parameters() if p.requires_grad]


def residual_mean(model, context, n_samples, batch_size, device):
    import torch

    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(context), batch_size):
            ctx = torch.as_tensor(context[start:start + batch_size], dtype=torch.float32, device=device)
            r = model.flow.sample(model._flow_ctx(ctx), n_samples=n_samples).mean(dim=1)
            out.append(r.cpu().numpy())
    return np.concatenate(out).astype(np.float32) if out else np.empty((0, model.target_dim), dtype=np.float32)


def predicted_shape_raw(model, context, resid_std, shape_idx, target_transform, batch_size, device):
    import torch

    model.eval()
    scale = torch.as_tensor(target_transform.scales[list(shape_idx)], dtype=torch.float32, device=device)
    mean = torch.as_tensor(target_transform.means[list(shape_idx)], dtype=torch.float32, device=device)
    parts = []
    with torch.no_grad():
        for start in range(0, len(context), batch_size):
            ctx = torch.as_tensor(context[start:start + batch_size], dtype=torch.float32, device=device)
            rr = torch.as_tensor(resid_std[start:start + batch_size][:, list(shape_idx)], dtype=torch.float32, device=device)
            mu = model._mu(ctx)[:, list(shape_idx)]
            parts.append(((mu + rr) * scale + mean).cpu().numpy())
    return np.concatenate(parts) if parts else np.empty((0, 2), dtype=np.float32)


def summarize(name, measured_raw, pred_before, pred_after, binid, target_bin, active):
    print(f"\n=== {name} mean-head additive summary ===")
    for label, pred in (("before", pred_before), ("after", pred_after)):
        res = measured_raw - pred
        print(
            f"{label:>7}: flow c1={np.nanmean(pred[:,0]):+.5f} c2={np.nanmean(pred[:,1]):+.5f}  "
            f"meas-flow c1={np.nanmean(res[:,0]):+.5f} c2={np.nanmean(res[:,1]):+.5f}"
        )
    sums = np.zeros_like(target_bin, dtype=np.float64)
    counts = np.zeros(len(target_bin), dtype=np.float64)
    np.add.at(sums, binid, pred_after)
    np.add.at(counts, binid, 1.0)
    pred_bin = np.divide(sums, counts[:, None], out=np.zeros_like(sums), where=counts[:, None] > 0)
    diff = pred_bin - target_bin
    ok = active & (counts > 0)
    if ok.any():
        rms = np.sqrt(np.mean(diff[ok] ** 2, axis=0))
        mx = np.max(np.abs(diff[ok]), axis=0)
        print(f"  active-bin after diff RMS c1={rms[0]:.5f} c2={rms[1]:.5f}; "
              f"max |diff| c1={mx[0]:.5f} c2={mx[1]:.5f}")


def train(args):
    import torch

    torch.manual_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"device={device}")
    print(f"input={args.input_model}")

    try:
        checkpoint = torch.load(args.input_model, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(args.input_model, map_location=device)
    condition_preprocessor = TabularPreprocessor.from_state(checkpoint["condition_preprocessor"])
    target_transform = TargetStandardizer.from_state(checkpoint["target_transform"])
    model_config = dict(checkpoint["model_config"])
    metadata = dict(checkpoint.get("metadata", {}))
    model = build_flow(model_config).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    if not isinstance(model, ConditionalMeanFlow):
        raise TypeError("fine-tune requires a ConditionalMeanFlow checkpoint")
    shape_idx = _shape_target_indices(target_transform.target_names)
    trainable = freeze_except_mean(model)
    print(f"trainable mean-head params={sum(p.numel() for p in trainable):,}")
    print(f"targets={target_transform.target_names}; shape_idx={shape_idx}")
    print(f"condition features={len(condition_preprocessor.feature_names)}")

    raw = load_g0(args)
    raw = raw[raw["case"] <= args.fit_max_case].reset_index(drop=True)
    if args.train_rows and len(raw) > args.train_rows:
        raw = raw.sample(n=args.train_rows, random_state=args.seed).reset_index(drop=True)
    print(f"fit rows={len(raw):,} from g0 cases <= {args.fit_max_case}")
    train_frame, train_context, train_target = prepare_frame(raw, condition_preprocessor, target_transform)
    bins = make_bins(train_frame, args)
    train_bin = bins.pop("train_bin")
    target_bin, target_counts, active = bin_targets(
        train_frame, train_bin, target_transform.target_names, shape_idx, bins["n_bins"], args.min_count
    )
    print(f"bins={bins['n_bins']} active={active.sum()} min/median/max count="
          f"{target_counts[target_counts>0].min():.0f}/"
          f"{np.median(target_counts[target_counts>0]):.0f}/{target_counts.max():.0f}")

    t0 = time.time()
    print(f"precomputing frozen residual-flow mean with {args.residual_samples} samples...")
    train_resid = residual_mean(model, train_context, args.residual_samples, args.batch_size, device)
    print(f"  residual mean time={time.time() - t0:.1f}s")

    measured_shape = train_frame[target_transform.target_names].to_numpy(np.float32)[:, list(shape_idx)]
    pred0 = predicted_shape_raw(model, train_context, train_resid, shape_idx, target_transform, args.batch_size, device)

    ds = torch.utils.data.TensorDataset(
        torch.as_tensor(train_target, dtype=torch.float32),
        torch.as_tensor(train_context, dtype=torch.float32),
        torch.as_tensor(train_resid, dtype=torch.float32),
        torch.as_tensor(train_bin, dtype=torch.long),
    )
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    target_bin_t = torch.as_tensor(target_bin, dtype=torch.float32, device=device)
    active_t = torch.as_tensor(active, dtype=torch.bool, device=device)
    scale_t = torch.as_tensor(target_transform.scales[list(shape_idx)], dtype=torch.float32, device=device)
    mean_t = torch.as_tensor(target_transform.means[list(shape_idx)], dtype=torch.float32, device=device)
    n_bins = int(bins["n_bins"])
    history = {"train_nll": [], "train_mean_loss": [], "train_total": []}
    best = np.inf
    best_state = None
    wait = 0

    print("\n--- Fine-tuning mean head ---")
    for epoch in range(1, args.epochs + 1):
        model.train(True)
        total_nll = 0.0
        total_mean = 0.0
        total_loss = 0.0
        total_w = 0.0
        for target, context, resid, binid in loader:
            target = target.to(device, non_blocking=True)
            context = context.to(device, non_blocking=True)
            resid = resid.to(device, non_blocking=True)
            binid = binid.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            lp = model.log_prob(target, context)
            nll = -lp.mean()
            mu_shape = model._mu(context)[:, list(shape_idx)]
            pred_raw = (mu_shape + resid[:, list(shape_idx)]) * scale_t + mean_t
            sums = torch.zeros((n_bins, 2), dtype=torch.float32, device=device).index_add_(0, binid, pred_raw)
            counts = torch.zeros(n_bins, dtype=torch.float32, device=device).index_add_(0, binid, torch.ones_like(binid, dtype=torch.float32))
            pred_bin = sums / counts[:, None].clamp_min(1.0)
            used = active_t & (counts >= args.batch_min_count)
            if used.any():
                diff2 = (pred_bin[used] - target_bin_t[used]) ** 2
                mean_loss = (diff2 * counts[used, None]).sum() / (2.0 * counts[used].sum().clamp_min(1.0))
            else:
                mean_loss = pred_raw.sum() * 0.0
            loss = args.nll_weight * nll + args.mean_weight * mean_loss
            loss.backward()
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
            opt.step()
            bw = float(len(target))
            total_nll += float(nll.detach().cpu()) * bw
            total_mean += float(mean_loss.detach().cpu()) * bw
            total_loss += float(loss.detach().cpu()) * bw
            total_w += bw
        nll_v = total_nll / max(total_w, 1.0)
        mean_v = total_mean / max(total_w, 1.0)
        loss_v = total_loss / max(total_w, 1.0)
        history["train_nll"].append(nll_v)
        history["train_mean_loss"].append(mean_v)
        history["train_total"].append(loss_v)
        print(f"  epoch {epoch:03d}: nll={nll_v:.6f} mean_loss={mean_v:.3e} total={loss_v:.6f}")
        if loss_v < best:
            best = loss_v
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= args.patience:
                print(f"  Early stopping after {epoch} epochs")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)
    pred1 = predicted_shape_raw(model, train_context, train_resid, shape_idx, target_transform, args.batch_size, device)
    summarize("G0 fit cases", measured_shape, pred0, pred1, train_bin, target_bin, active)

    del ds, loader, train_resid, train_context, train_target
    gc.collect()

    if args.eval_rows and args.eval_rows > 0:
        raw_eval = load_g0(args)
        raw_eval = raw_eval[raw_eval["case"] >= args.eval_min_case].reset_index(drop=True)
        if len(raw_eval) > args.eval_rows:
            raw_eval = raw_eval.sample(n=args.eval_rows, random_state=args.seed + 1).reset_index(drop=True)
        eval_frame, eval_context, _ = prepare_frame(raw_eval, condition_preprocessor, target_transform)
        eval_bin, _, _, _ = _bin_ids(
            eval_frame, bins["mag_edges"], bins["re_edges"], bins["crowd_edges"],
            bins["crowd_col"], bins["crowd_eps"]
        )
        eval_target_bin, _, eval_active = bin_targets(
            eval_frame, eval_bin, target_transform.target_names, shape_idx, bins["n_bins"], args.min_count
        )
        eval_resid = residual_mean(model, eval_context, args.residual_samples, args.batch_size, device)
        eval_meas = eval_frame[target_transform.target_names].to_numpy(np.float32)[:, list(shape_idx)]
        eval_pred0 = pred0[:0]
        # The residual flow is frozen but the initial mean head is gone; for heldout before/after
        # use the original checkpoint reloaded cheaply.
        old_model = build_flow(model_config).to(device)
        old_model.load_state_dict(checkpoint["state_dict"])
        old_model.eval()
        eval_pred0 = predicted_shape_raw(old_model, eval_context, eval_resid, shape_idx, target_transform, args.batch_size, device)
        eval_pred1 = predicted_shape_raw(model, eval_context, eval_resid, shape_idx, target_transform, args.batch_size, device)
        summarize("G0 heldout cases", eval_meas, eval_pred0, eval_pred1, eval_bin, eval_target_bin, eval_active)
        del old_model, eval_resid

    metadata["additive_mean_head_finetune"] = {
        "input_model": args.input_model,
        "g0_catalogue": args.g0_catalogue,
        "fit_max_case": int(args.fit_max_case),
        "mean_weight": float(args.mean_weight),
        "nll_weight": float(args.nll_weight),
        "residual_samples": int(args.residual_samples),
        "bins": {k: v for k, v in bins.items() if k != "train_bin"},
        "history": history,
        "best_total": float(best),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    save_measurement_model(args.output, model, condition_preprocessor, target_transform, model_config, metadata=metadata)
    stem = os.path.splitext(os.path.abspath(args.output))[0]
    np.savez(f"{stem}_meanfix_curve.npz", **history)
    print(f"\nSaved mean-head-tuned measurement model: {args.output}")
    print("FINETUNE_ADD_C_MEAN_HEAD_DONE")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-model", default="models/measurement_flow_g0_ngmix_crowdflux_lam300_v1.pt")
    ap.add_argument("--output", default="models/measurement_flow_g0_ngmix_crowdflux_lam300_meanfix_v1.pt")
    ap.add_argument("--g0-catalogue", default="/project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_crowd_g0.0_train_c0-39.feather")
    ap.add_argument("--fit-max-case", type=int, default=19)
    ap.add_argument("--eval-min-case", type=int, default=20)
    ap.add_argument("--max-rows", type=int, default=12_000_000)
    ap.add_argument("--train-rows", type=int, default=1_500_000)
    ap.add_argument("--eval-rows", type=int, default=750_000)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=65536)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--residual-samples", type=int, default=128)
    ap.add_argument("--mean-weight", type=float, default=50000.0)
    ap.add_argument("--nll-weight", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2.0e-4)
    ap.add_argument("--weight-decay", type=float, default=1.0e-6)
    ap.add_argument("--max-grad-norm", type=float, default=5.0)
    ap.add_argument("--mag-edges", type=float, nargs="+", default=[18, 24, 25, 26, 28])
    ap.add_argument("--n-size", type=int, default=3)
    ap.add_argument("--crowd-col", default="nbr_flux_near")
    ap.add_argument("--n-crowd-pos", type=int, default=4)
    ap.add_argument("--crowd-eps", type=float, default=1e-6)
    ap.add_argument("--min-count", type=int, default=1000)
    ap.add_argument("--batch-min-count", type=int, default=128)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--device", default=None)
    return ap.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    train(args)


if __name__ == "__main__":
    main()
