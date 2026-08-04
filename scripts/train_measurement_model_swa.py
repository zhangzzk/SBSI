"""Train the selected-object measurement likelihood.

This model learns the normalized density

    p_meas(xhat | truth, neighbour, shear, s=1)

on rows that satisfy the current source cuts and the selected-object target
column.  The Bernoulli selection probability remains in the separate selection
classifier and should be multiplied in only when constructing p_cat.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import deque

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
import torch
from sklearn.model_selection import train_test_split

SBSI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SBSI_ROOT not in sys.path:
    sys.path.insert(0, SBSI_ROOT)

from sbs_shear.measurement_model import (  # noqa: E402
    DEFAULT_MEASUREMENT_CONDITION_FEATURES,
    DEFAULT_MEASUREMENT_TARGETS,
    MEASUREMENT_CONDITION_FEATURE_SETS,
    ConditionalAffineFlow,
    ConditionalMeanFlow,
    build_flow,
    TargetStandardizer,
    add_measurement_target_features,
    raw_columns_for_measurement_targets,
    save_measurement_model,
)
from sbs_shear.preprocessing import (  # noqa: E402
    DEFAULT_SELECTION_CUTS,
    SHEAR_FEATURES,
    apply_structure_measurement_noise,
    raw_columns_for_selection_features,
    rescale,
    source_select_selection,
)
from sbs_shear.selection_model import TabularPreprocessor  # noqa: E402
from sbs_shear.shear_map import apply_shear_to_ellipticity  # noqa: E402


def _add_legacy_missing_shear(df, features):
    added = []
    for name in features:
        if name in df.columns:
            continue
        if name in SHEAR_FEATURES:
            df[name] = 0.0
            added.append(name)
    if added:
        print("  Added missing shear columns as zeros for legacy zero-shear catalogue:")
        print(f"    {', '.join(added)}")
        print("  This is useful for smoke tests, but not for validating shear derivatives.")
    missing = [name for name in features if name not in df.columns]
    if missing:
        raise KeyError(f"Missing required condition feature columns: {missing}")
    return df


def _finite_target_mask(frame, targets):
    values = frame[list(targets)].to_numpy(dtype=np.float32, copy=False)
    return np.isfinite(values).all(axis=1)


def _append_to_priority_sample(reservoir, batch, max_rows, rng):
    if max_rows is None or max_rows <= 0:
        # Unbounded: keep the parts and concatenate ONCE in _finalize_priority_sample.
        # Folding `pd.concat` per record batch re-materialised the whole reservoir every
        # time -- O(N*B/2) row copies, ~1.3 TB moved over the 27.8M-row / 480-batch
        # production load (`--max-rows 0`), against a single ~5.6 GB pass. Concatenating
        # the parts in order is value-identical to folding them one at a time.
        if reservoir is None:
            reservoir = []
        reservoir.append(batch)
        return reservoir

    batch = batch.copy()
    batch["__sample_key"] = rng.random(len(batch))
    if reservoir is None:
        reservoir = batch
    else:
        reservoir = pd.concat([reservoir, batch], ignore_index=True)

    if len(reservoir) > 2 * max_rows:
        reservoir = reservoir.nlargest(max_rows, "__sample_key").reset_index(drop=True)
    return reservoir


def _priority_sample_rows(reservoir):
    """Row count of a reservoir in either form (list of parts, or a single frame)."""
    if reservoir is None:
        return 0
    if isinstance(reservoir, list):
        return sum(len(p) for p in reservoir)
    return len(reservoir)


def _finalize_priority_sample(reservoir, max_rows):
    if reservoir is None or (isinstance(reservoir, list) and not reservoir):
        raise RuntimeError("No selected finite measured rows were loaded from the catalogue")
    if isinstance(reservoir, list):
        reservoir = pd.concat(reservoir, ignore_index=True)
    if max_rows is not None and max_rows > 0 and len(reservoir) > max_rows:
        reservoir = reservoir.nlargest(max_rows, "__sample_key").reset_index(drop=True)
    if "__sample_key" in reservoir.columns:
        reservoir = reservoir.drop(columns="__sample_key")
    return reservoir.reset_index(drop=True)


def load_measurement_data(args, condition_features, target_features):
    rng = np.random.default_rng(args.seed)
    t0 = time.time()

    print(f"Loading measurement sample: {args.catalogue}")
    print(f"  Selection target column: {args.target_column}")
    print(f"  Requested max selected rows: {args.max_rows:,}" if args.max_rows else "  Requested max selected rows: all")

    reservoir = None
    raw_rows = 0
    source_cut_rows = 0
    selected_rows = 0
    finite_rows = 0
    batches_seen = 0

    with ipc.open_file(args.catalogue) as reader:
        available = set(reader.schema.names)
        condition_raw = raw_columns_for_selection_features(condition_features, available_columns=available)
        target_raw = raw_columns_for_measurement_targets(target_features)
        extra_columns = {args.target_column}
        for c in ("input_index", "case"):   # per-(case,target) all-pairs weighting
            if c in available:
                extra_columns.add(c)
        if args.shear_case is not None and "shear_case" in available:
            extra_columns.add("shear_case")
        if getattr(args, "response_weight", 0.0) > 0:
            # raw shape + shear columns the response loss needs to apply S_delta in main(),
            # and true flux/size for the property-resolved response bins.
            for c in ("e1_input_rot0_p", "e2_input_rot0_p", "gamma1_input_p", "gamma2_input_p",
                      "r_input_p", "Re_input_p", "neighbored", "distance",
                      "r_blend", "nbr_flux_near", "nbr_flux_far"):   # crowding cols for the response-bin axis
                if c in available:
                    extra_columns.add(c)
        requested_columns = sorted(condition_raw | target_raw | extra_columns)
        missing_non_shear = [
            name for name in requested_columns
            if name not in available and name not in SHEAR_FEATURES
        ]
        if missing_non_shear:
            raise KeyError(f"Missing required catalogue columns: {missing_non_shear}")
        read_columns = [name for name in requested_columns if name in available]

        print(f"  File record batches: {reader.num_record_batches:,}")
        print(f"  Reading columns: {len(read_columns):,}")
        print(f"  Condition features: {len(condition_features):,}")
        print(f"  Target features: {len(target_features):,}")

        for batch_index in range(reader.num_record_batches):
            if args.max_read_batches is not None and batch_index >= args.max_read_batches:
                break
            table = pa.Table.from_batches([reader.get_batch(batch_index)]).select(read_columns)
            batch = table.to_pandas()
            raw_rows += len(batch)
            if args.shear_case is not None and "shear_case" in batch.columns:
                batch = batch[np.isclose(batch["shear_case"].astype(float), args.shear_case)]
                if len(batch) == 0:
                    continue

            if args.max_cases is not None and "case" in batch.columns:
                batch = batch[batch["case"].astype(int) < args.max_cases]
                if len(batch) == 0:
                    continue

            batch = source_select_selection(batch, cuts=DEFAULT_SELECTION_CUTS)
            source_cut_rows += len(batch)
            if len(batch) == 0:
                continue

            selected = batch[args.target_column].astype(bool).to_numpy()
            batch = batch.loc[selected].reset_index(drop=True)
            selected_rows += len(batch)
            if len(batch) == 0:
                continue

            batch = rescale(
                batch,
                pixel_rms=args.pixel_rms,
                pixel_size=args.pixel_size,
                zero_mag=args.zero_mag,
                psf_fwhm=args.psf_fwhm,
                moffat_beta=args.moffat_beta,
            )
            batch = _add_legacy_missing_shear(batch, condition_features)
            batch = add_measurement_target_features(batch)
            finite = _finite_target_mask(batch, target_features)
            batch = batch.loc[finite].reset_index(drop=True)
            finite_rows += len(batch)
            if len(batch) == 0:
                continue

            if args.noise_photoz or args.noise_sersic_frac:
                # emulate survey measurement error on the TRUE structure conditioners so the flow
                # learns the noisy channel it will be fed at deployment (realistic-structure study).
                batch = apply_structure_measurement_noise(
                    batch, photoz_sigma=args.noise_photoz, sersic_frac=args.noise_sersic_frac, rng=rng)

            keep_columns = list(dict.fromkeys([*condition_features, *target_features, args.target_column]))
            for c in ("input_index", "case"):   # survive to the dataset for all-pairs weighting
                if c in batch.columns and c not in keep_columns:
                    keep_columns.append(c)
            if getattr(args, "response_weight", 0.0) > 0:
                # keep the raw columns so main() can re-apply the analytic shear map S_delta
                # to the intrinsic shape and recompute the conditioning for the response loss.
                keep_columns += [c for c in read_columns if c not in keep_columns]
            reservoir = _append_to_priority_sample(reservoir, batch[keep_columns], args.max_rows, rng)
            batches_seen += 1
            if args.progress_every and batches_seen % args.progress_every == 0:
                kept = _priority_sample_rows(reservoir)
                print(
                    f"  batches={batches_seen:,}, raw={raw_rows:,}, "
                    f"source_cut={source_cut_rows:,}, selected={selected_rows:,}, "
                    f"finite={finite_rows:,}, reservoir={kept:,}"
                )

    dataset = _finalize_priority_sample(reservoir, args.max_rows)
    print(f"  Raw rows scanned: {raw_rows:,}")
    print(f"  Rows after source cuts: {source_cut_rows:,}")
    print(f"  Selected rows before measured-target cuts: {selected_rows:,}")
    print(f"  Selected rows with finite targets: {finite_rows:,}")
    print(f"  Rows used: {len(dataset):,}")
    print(f"  Load/sample time: {time.time() - t0:.1f}s")
    return dataset


def split_data(frame, seed, validation_size):
    if not (0.0 < validation_size < 1.0):
        raise ValueError("validation_size must be in (0, 1)")
    train_df, val_df = train_test_split(frame, test_size=validation_size, random_state=seed)
    print(f"  Split: train={len(train_df):,}, val={len(val_df):,}")
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


class GPUBatches:
    """Hold the full (small) dataset resident on the GPU and yield batches by slicing -- no
    DataLoader, no worker subprocesses, no per-batch host->device copies. This training is tiny
    per batch, so the CPU data pipeline (9 tensors/batch copied to GPU through workers) starves the
    GPU (0% util) and makes wall-time CPU/IO-bound and sensitive to shared-node contention. Moving
    the ~2GB dataset onto the GPU once removes that bottleneck. Drop-in for a DataLoader over a
    TensorDataset: yields the SAME tensor tuples (already on device, so downstream .to(device) is a
    no-op) in the SAME batch_size, reshuffled per epoch when shuffle=True -- identical training math."""
    def __init__(self, tensors, batch_size, shuffle, device):
        self.tensors = [t.to(device) for t in tensors]
        self.n = int(self.tensors[0].shape[0])
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.device = device

    def __iter__(self):
        if self.shuffle:
            perm = torch.randperm(self.n, device=self.device)
            for i in range(0, self.n, self.batch_size):
                idx = perm[i:i + self.batch_size]
                yield tuple(t[idx] for t in self.tensors)
        else:
            for i in range(0, self.n, self.batch_size):
                sl = slice(i, i + self.batch_size)
                yield tuple(t[sl] for t in self.tensors)

    def __len__(self):
        return (self.n + self.batch_size - 1) // self.batch_size


def make_loader(targets, context, batch_size, shuffle=False, num_workers=0, pin_memory=False,
                weights=None, gpu_resident=False, device=None):
    tensors = [
        torch.as_tensor(targets, dtype=torch.float32),
        torch.as_tensor(context, dtype=torch.float32),
    ]
    if weights is not None:
        tensors.append(torch.as_tensor(np.asarray(weights).reshape(-1), dtype=torch.float32))
    if gpu_resident and device is not None and device.type == "cuda":
        return GPUBatches(tensors, batch_size, shuffle, device)
    dataset = torch.utils.data.TensorDataset(*tensors)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def compute_decorrelation_weights(frame, nbins=40, clip=10.0):
    """Per-row importance weights that make scene-shape magnitude |e| and size independent
    in the g=0 training set: w_i = P(|e|)P(size)/P(|e|,size), estimated on quantile bins.

    The isolated bias source is the g=0 shape<->size correlation (+0.42): the flow learns the
    shape->shape response only on the correlated manifold and extrapolates badly into the
    (sheared-shape, same-size) region the sheared population occupies. Training on the
    weighted (decorrelated) distribution forces the flow to learn the response across the full
    (shape x size) product support. Uses only g=0 truth -- no sheared-shear information."""
    a = np.hypot(frame["e1_input_p"].to_numpy(float), frame["e2_input_p"].to_numpy(float))
    b = frame["Re_input_p_scaled"].to_numpy(float)
    good = np.isfinite(a) & np.isfinite(b)
    ea = np.quantile(a[good], np.linspace(0, 1, nbins + 1)); ea[0] -= 1e-9; ea[-1] += 1e-9
    eb = np.quantile(b[good], np.linspace(0, 1, nbins + 1)); eb[0] -= 1e-9; eb[-1] += 1e-9
    ia = np.clip(np.digitize(a, ea) - 1, 0, nbins - 1)
    ib = np.clip(np.digitize(b, eb) - 1, 0, nbins - 1)
    joint = np.zeros((nbins, nbins)); np.add.at(joint, (ia, ib), good.astype(float))
    pa = joint.sum(1); pb = joint.sum(0); N = max(joint.sum(), 1.0)
    wij = (pa[ia] * pb[ib]) / (N * np.maximum(joint[ia, ib], 1.0))
    w = np.where(good, wij, 1.0)
    w = np.clip(w, 1.0 / clip, clip)
    w *= len(w) / w.sum()
    return w


def per_target_weights(frame):
    """1/n_pairs within (case, target): ALL-PAIRS de-duplication so each independent
    (case, galaxy) measurement contributes weight 1 -- NOT proportional to its neighbour
    count -- while keeping every case (noise realisation) as a separate sample.
    Nearest-pair catalogue -> n_pairs=1 -> all ones. Returns None if input_index absent."""
    if "input_index" not in frame.columns:
        return None
    ii = frame["input_index"].to_numpy(np.int64)
    key = (frame["case"].to_numpy(np.int64) * 1_000_003 + ii) if "case" in frame.columns else ii
    _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    return (1.0 / cnt[inv]).astype(np.float64)


def epoch_nll(model, loader, device, optimizer=None, max_grad_norm=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    n_total = 0.0
    for batch in loader:
        target = batch[0].to(device, non_blocking=True)
        context = batch[1].to(device, non_blocking=True)
        weight = batch[2].to(device, non_blocking=True) if len(batch) > 2 else None
        if training:
            optimizer.zero_grad(set_to_none=True)
        lp = model.log_prob(target, context)
        if weight is not None:
            wsum = weight.sum().clamp_min(1e-8)
            loss = -(weight * lp).sum() / wsum
            bw = float(wsum.detach().cpu())
        else:
            loss = -lp.mean()
            bw = len(target)
        if training:
            loss.backward()
            if max_grad_norm is not None and max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
        total_loss += float(loss.detach().cpu()) * bw
        n_total += bw
    return total_loss / max(n_total, 1e-8)


def build_shifted_context(frame, gdir, delta, preprocessor, condition_features, rescale_kwargs):
    """Standardized conditioning context after applying the analytic shear map S_delta to
    the intrinsic shape (g=0 rows: e1/e2_input_rot0_p IS the intrinsic shape; gamma=0).

    delta=0 reproduces the unshifted context through the identical rescale path, so the
    model's induced response R_model = (mu(shifted) - mu(unshifted)) is a clean finite
    difference. Mirrors model_mean_proj in response_ratio_diagnostic.py."""
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
    f = _add_legacy_missing_shear(f, condition_features)
    return preprocessor.transform_frame(f)


def epoch_response(model, loader, device, target_scales01, delta, bin_targets, lam,
                   response_difference="forward",
                   response_error="absolute", rel_floor=0.05,
                   optimizer=None, max_grad_norm=None):
    """NLL + PROPERTY-RESOLVED response loss. Pulls the model's induced first-moment
    response R_model(bin) -> R_sim(bin) in bins of true flux x size (bin_targets is a
    (n_bins,) tensor; n_bins=1 reduces to the old global response loss). R_model is the
    trace/2 responsivity (response of measured_e1 to an e1-shift and measured_e2 to an
    e2-shift, averaged). Only the explicit mean head carries the shape response, so
    mu(shifted)-mu(unshifted) is the full induced response, differentiable in params; a
    NONLINEAR (MLP) head lets that response vary per bin so the model resolves it."""
    training = optimizer is not None
    model.train(training)
    sc0, sc1 = float(target_scales01[0]), float(target_scales01[1])
    bt = bin_targets.to(device)
    n_bins = bt.numel()
    tot_nll = tot_resp = n_tot = rmodel_sum = rmodel_n = 0.0
    for batch in loader:
        if len(batch) == 7:
            target, context, weight, ctx0, ce1, ce2, binid = batch
            cm1 = cm2 = None
        elif len(batch) == 9:
            target, context, weight, ctx0, ce1, ce2, cm1, cm2, binid = batch
        else:
            raise ValueError(f"unexpected response batch length {len(batch)}")
        target = target.to(device, non_blocking=True)
        context = context.to(device, non_blocking=True)
        weight = weight.to(device, non_blocking=True)
        ctx0 = ctx0.to(device, non_blocking=True)
        ce1 = ce1.to(device, non_blocking=True)
        ce2 = ce2.to(device, non_blocking=True)
        if cm1 is not None:
            cm1 = cm1.to(device, non_blocking=True)
            cm2 = cm2.to(device, non_blocking=True)
        binid = binid.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        lp = model.log_prob(target, context)
        wsum = weight.sum().clamp_min(1e-8)
        nll = -(weight * lp).sum() / wsum
        mu0 = model._mu(ctx0)
        mu1 = model._mu(ce1)
        mu2 = model._mu(ce2)
        # per-galaxy physical-unit response (additive mean cancels in the difference)
        if response_difference == "central":
            if cm1 is None or cm2 is None:
                raise ValueError("central response difference requires negative-shift contexts")
            mum1 = model._mu(cm1)
            mum2 = model._mu(cm2)
            r_i = 0.25 * ((mu1[:, 0] - mum1[:, 0]) * sc0 + (mu2[:, 1] - mum2[:, 1]) * sc1) / delta
        else:
            r_i = 0.5 * ((mu1[:, 0] - mu0[:, 0]) * sc0 + (mu2[:, 1] - mu0[:, 1]) * sc1) / delta
        # per-bin WEIGHTED mean response via scatter (weight = all-pairs 1/n_pairs so each
        # (case,target) counts once, not proportional to neighbour count); absent bins no penalty
        sum_b = torch.zeros(n_bins, device=device).index_add_(0, binid, weight * r_i)
        cnt_b = torch.zeros(n_bins, device=device).index_add_(0, binid, weight)
        mean_b = torch.where(cnt_b > 0, sum_b / cnt_b.clamp_min(1e-8), bt)
        if response_error == "relative":
            # Penalize the FRACTIONAL response error (R_model/R_sim - 1)^2, i.e. the per-bin
            # multiplicative bias m itself, rather than absolute (R_model - R_sim)^2. Absolute
            # error is dominated by the high-response isolated/bright cells and tolerates large
            # RELATIVE errors in the small-response crowded/faint tail, so it drives global m~0
            # only for the TRAINING population weighting and leaves a crowding tilt that survives
            # any population reweighting (constant-gold, survey depth). The relative form pulls
            # m -> 0 uniformly per bin. Floor guards small/negative target bins.
            denom = bt.abs().clamp_min(rel_floor)
            resp = (((mean_b - bt) / denom) ** 2 * cnt_b).sum() / cnt_b.sum().clamp_min(1.0)
        else:
            resp = ((mean_b - bt) ** 2 * cnt_b).sum() / cnt_b.sum().clamp_min(1.0)
        loss = nll + lam * resp
        if training:
            loss.backward()
            if max_grad_norm and max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
        bw = float(wsum.detach().cpu())
        tot_nll += float(nll.detach().cpu()) * bw
        tot_resp += float(resp.detach().cpu()) * bw
        rmodel_sum += float(r_i.sum().detach().cpu())
        rmodel_n += r_i.numel()
        n_tot += bw
    n = max(n_tot, 1e-8)
    return tot_nll / n, tot_resp / n, rmodel_sum / max(rmodel_n, 1)


@torch.no_grad()
def summarize_log_prob(model, loader, device):
    model.eval()
    values = []
    for batch in loader:
        target, context = batch[0], batch[1]
        lp = model.log_prob(target.to(device, non_blocking=True), context.to(device, non_blocking=True))
        values.append(lp.cpu().numpy())
    if not values:
        return {"mean_log_prob": np.nan, "std_log_prob": np.nan}
    arr = np.concatenate(values)
    return {
        "mean_log_prob": float(np.mean(arr)),
        "std_log_prob": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalogue",
        default="/project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_g0.0_train.feather",
        help="SBSI g=0 detection+measured catalogue (built by "
        "build_detection_measurement_catalogue.py from the live blendemu run)",
    )
    parser.add_argument("--output", default=os.path.join(SBSI_ROOT, "models/measurement_flow_g0_shearfree_v1.pt"))
    parser.add_argument("--target-column", default="detected")
    parser.add_argument("--selection-name", default="sextractor_detected")
    parser.add_argument(
        "--feature-set",
        default="v6_primary_frame",
        choices=sorted(MEASUREMENT_CONDITION_FEATURE_SETS),
        help="Condition feature set. 'g0_shearfree' drops applied-shear inputs "
        "for the refined g=0 forward model; pair with --shear-case 0.0. "
        "Ignored if --condition-features is given explicitly.",
    )
    parser.add_argument(
        "--shear-case",
        type=float,
        default=None,
        help="If set, keep only rows whose catalogue 'shear_case' matches "
        "(e.g. 0.0 for the shear-free forward model).",
    )
    parser.add_argument("--condition-features", nargs="+", default=None)
    parser.add_argument("--target-features", nargs="+", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--gpu-resident", action="store_true",
                        help="keep the full dataset resident on the GPU and batch by slicing (no DataLoader/"
                             "workers/per-batch host->device copies). Removes the CPU data-pipeline bottleneck "
                             "(0%% GPU util, contention-sensitive) for this small dataset; identical training math.")
    parser.add_argument("--max-rows", type=int, default=2_000_000)
    parser.add_argument("--max-cases", type=int, default=None,
                        help="keep only rows with case < this (fast prototype on a case subset)")
    parser.add_argument("--noise-photoz", type=float, default=0.0,
                        help="photo-z scatter sigma=this*(1+z) added to redshift_input_p (realistic-structure study)")
    parser.add_argument("--noise-sersic-frac", type=float, default=0.0,
                        help="fractional scatter sigma=this*|n| added to sersic_n_input_p (realistic-structure study)")
    parser.add_argument("--max-read-batches", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--flow-type", default="affine",
                        choices=["affine", "spline", "mean_affine", "mean_spline"],
                        help="Conditional flow family. 'mean_*' adds an explicit conditional-"
                        "mean head (fixes the measured-shape response under-fit that drives "
                        "the multiplicative shear bias).")
    parser.add_argument("--mean-hidden", type=int, default=0,
                        help="Mean-head hidden width for mean_* flows (0 = linear head).")
    parser.add_argument("--flow-blind-features", nargs="*", default=None,
                        help="For mean_* flows: condition features the RESIDUAL flow is blind "
                        "to (forced into the explicit mean head). E.g. e1_input_p e2_input_p "
                        "makes the shape->shape response unshrinkable.")
    parser.add_argument("--freeze-mean-ols", action="store_true",
                        help="For mean_* flows with a linear head: fit the mean head by OLS on "
                        "the g=0 standardized (context, target) and FREEZE it, so the conditional-"
                        "mean response equals the data response M_data by construction (and ML "
                        "training cannot shrink it). Pair with --flow-blind-features on the shape "
                        "inputs. Legitimate g=0 self-calibration; uses no sheared-shear truth.")
    parser.add_argument("--decorrelate-shape-size", action="store_true",
                        help="Importance-weight the g=0 training set so scene-shape |e| and size "
                        "are independent, removing the +0.42 shape<->size correlation that is the "
                        "isolated covariate-shift source. The flow then learns the shape response "
                        "across the full (shape x size) support the sheared population occupies. "
                        "Legitimate g=0 self-calibration; uses no sheared-shear truth.")
    parser.add_argument("--decorrelate-nbins", type=int, default=40)
    parser.add_argument("--decorrelate-clip", type=float, default=10.0)
    # --- Response-aware (Sobolev) training (SBI_shear_response.md) ---
    parser.add_argument("--response-weight", type=float, default=0.0,
                        help="lambda for the response loss L = L_NLL + lambda*||R_model - R_sim||^2. "
                        "0 = off (plain NLL). Requires a mean_* flow with a TRAINABLE mean head.")
    parser.add_argument("--response-target", type=float, default=0.2313,
                        help="R_sim: the simulation's measured first-moment shear response "
                        "<e_meas.ghat>/g at the calibration shear (g=0.05 -> 0.2313). The model's "
                        "induced response is pulled to this. This is the response anchor "
                        "SBI_shear_response.md sanctions; tested on held-out shears (0.2, 0.02).")
    parser.add_argument("--response-delta", type=float, default=0.05,
                        help="finite shift delta for the model's induced response (secant over [0,delta]); "
                        "match to the calibration shear so model & sim secants are comparable.")
    parser.add_argument("--response-difference", choices=["forward", "central"], default="forward",
                        help="Finite-difference stencil for response loss. 'forward' preserves the "
                             "historical [0,delta] secant. 'central' trains the symmetric "
                             "[-delta,+delta]/(2 delta) response used by constant-gold validation.")
    parser.add_argument("--response-error", choices=["absolute", "relative"], default="absolute",
                        help="Per-bin response-loss metric. 'absolute' penalizes (R_model-R_sim)^2 "
                             "(historical; dominated by high-response cells, tolerates large relative "
                             "error in the small-response crowded/faint tail). 'relative' penalizes "
                             "((R_model-R_sim)/R_sim)^2 = the per-bin multiplicative bias m, driving "
                             "m->0 uniformly and making the calibration robust to population reweighting.")
    parser.add_argument("--response-rel-floor", type=float, default=0.05,
                        help="Floor on |R_sim| in the relative response-loss denominator; guards "
                             "small/negative target bins from blowing up the fractional error.")
    parser.add_argument("--response-target-npz", default=None,
                        help="PROPERTY-RESOLVED target from compute_response_target.py "
                        "(edges_flux, edges_size, Rsim[nf,ns]). Supervises R_model(bin)->R_sim(bin) "
                        "in bins of true flux(r_input_p) x size(Re_input_p). Pair with --mean-hidden>0 "
                        "(a linear head cannot resolve a per-bin response). Overrides --response-target.")
    parser.add_argument("--num-bins", type=int, default=8, help="RQ spline bins (spline only).")
    parser.add_argument("--tail-bound", type=float, default=5.0, help="RQ spline tail bound (spline only).")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--condition-layers", type=int, default=3)
    parser.add_argument("--n-flows", type=int, default=8)
    parser.add_argument("--activation", default="silu", choices=["silu", "gelu", "tanh"])
    parser.add_argument("--scale-limit", type=float, default=3.0)
    parser.add_argument("--lr", type=float, default=7.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--max-grad-norm", type=float, default=5.0)
    parser.add_argument("--validation-size", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--swa-last-k", type=int, default=8,
                        help="Stochastic Weight Averaging window: average the last K end-of-epoch "
                             "model snapshots (equal weights) into a second '_swaavg' checkpoint, "
                             "in addition to the best-val checkpoint. A last-K window (not a fixed "
                             "epoch fraction) is robust to early stopping. Set 0/1 to effectively "
                             "disable averaging (K=1 saves the final epoch's weights).")
    parser.add_argument("--seed", type=int, default=421)
    parser.add_argument("--pixel-rms", type=float, default=0.312)
    parser.add_argument("--pixel-size", type=float, default=0.2)
    parser.add_argument("--zero-mag", type=float, default=30.0)
    parser.add_argument("--psf-fwhm", type=float, default=0.73)
    parser.add_argument("--moffat-beta", type=float, default=2.224)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    pin_memory = device.type == "cuda"
    print(f"Using device: {device}")

    condition_features = list(
        args.condition_features or MEASUREMENT_CONDITION_FEATURE_SETS[args.feature_set]
    )
    target_features = list(args.target_features or DEFAULT_MEASUREMENT_TARGETS)
    print(f"Condition feature set: {args.feature_set} ({len(condition_features)} features)")
    if args.shear_case is not None:
        print(f"Restricting to shear_case == {args.shear_case}")
    frame = load_measurement_data(args, condition_features, target_features)
    train_df, val_df = split_data(frame, args.seed, args.validation_size)

    condition_preprocessor = TabularPreprocessor.fit(train_df, condition_features, add_missing_indicators=True)
    target_transform = TargetStandardizer.fit(train_df, target_features)

    train_w = val_w = None
    if args.decorrelate_shape_size:
        train_w = compute_decorrelation_weights(train_df, args.decorrelate_nbins, args.decorrelate_clip)
        val_w = compute_decorrelation_weights(val_df, args.decorrelate_nbins, args.decorrelate_clip)
        eff = train_w.sum() ** 2 / np.sum(train_w ** 2)
        print(f"Decorrelating shape<->size: train w in "
              f"[{train_w.min():.3f}, {train_w.max():.3f}] mean {train_w.mean():.3f}; "
              f"effective N = {eff:,.0f} / {len(train_w):,}")
    # ALL-PAIRS de-duplication weight (multiplies any decorrelation weight). For a nearest-pair
    # catalogue this is all-ones (no-op); for all-pairs it makes every (case,galaxy) count once.
    ptw_tr, ptw_va = per_target_weights(train_df), per_target_weights(val_df)
    if ptw_tr is not None and float(np.max(1.0 / ptw_tr)) > 1.0:
        train_w = ptw_tr if train_w is None else train_w * ptw_tr
        val_w = ptw_va if val_w is None else val_w * ptw_va
        eff = train_w.sum() ** 2 / np.sum(train_w ** 2)
        print(f"All-pairs per-(case,target) weighting: ~{1.0/ptw_tr.mean():.1f} pairs/target; "
              f"effective N = {eff:,.0f} / {len(train_w):,}")

    train_loader = make_loader(
        target_transform.transform_frame(train_df),
        condition_preprocessor.transform_frame(train_df),
        args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        weights=train_w,
        gpu_resident=args.gpu_resident,
        device=device,
    )
    val_loader = make_loader(
        target_transform.transform_frame(val_df),
        condition_preprocessor.transform_frame(val_df),
        args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        gpu_resident=args.gpu_resident,
        device=device,
        weights=val_w,
    )

    model_config = {
        "flow_type": args.flow_type,
        "target_dim": target_transform.dim,
        "context_dim": condition_preprocessor.output_dim,
        "hidden_dim": args.hidden_dim,
        "n_layers": args.condition_layers,
        "n_flows": args.n_flows,
        "scale_limit": args.scale_limit,
        "activation": args.activation,
        "num_bins": args.num_bins,
        "tail_bound": args.tail_bound,
        "mean_hidden": args.mean_hidden,
    }
    if args.flow_blind_features:
        n_feat = len(condition_features)
        drop = []
        for name in args.flow_blind_features:
            if name not in condition_features:
                raise KeyError(f"--flow-blind-features {name!r} not in condition features")
            j = condition_features.index(name)
            drop += [j, j + n_feat]  # standardized column + its missing-indicator column
        model_config["flow_drop_indices"] = sorted(drop)
        print(f"Residual flow blind to {args.flow_blind_features} -> drop context idx {sorted(drop)}")
    model = build_flow(model_config).to(device)

    if args.freeze_mean_ols:
        if not isinstance(model, ConditionalMeanFlow):
            raise ValueError("--freeze-mean-ols requires a mean_* flow_type")
        ctx_std = condition_preprocessor.transform_frame(train_df)
        tgt_std = target_transform.transform_frame(train_df)
        model.set_ols_mean_and_freeze(ctx_std, tgt_std)
        model_config["freeze_mean_ols"] = True
        n_frozen = sum(p.numel() for p in model.mean_net.parameters())
        print(f"Froze linear mean head at OLS conditional mean ({n_frozen} params); "
              f"M_model == M_data by construction.")

    # --- Response-aware setup (SBI_shear_response.md): build shifted-shape contexts ---
    response_on = args.response_weight > 0
    resp_train_loader = resp_val_loader = None
    target_scales01 = None
    if response_on:
        model_config["response_difference"] = args.response_difference
        model_config["response_error"] = args.response_error
        if not isinstance(model, ConditionalMeanFlow):
            raise ValueError("--response-weight requires a mean_* flow (ConditionalMeanFlow).")
        if not any(p.requires_grad for p in model.mean_net.parameters()):
            raise ValueError("--response-weight needs a TRAINABLE mean head; "
                             "do not combine with --freeze-mean-ols.")
        tnames = list(target_transform.target_names)
        if len(tnames) < 2:
            raise ValueError(f"response loss needs 2 shape-component targets (e1/e2-like); got {tnames}")
        # First two targets are the (e1-like, e2-like) shape components: SExtractor
        # measured_e1/e2_image OR ngmix measured_ngmix_g1/g2. The response loss shears
        # component-0 by an e1-shift and component-1 by an e2-shift (build_shifted_context).
        scales = np.asarray(target_transform.scales, dtype=float)
        target_scales01 = (scales[0], scales[1])
        rescale_kwargs = dict(pixel_rms=args.pixel_rms, pixel_size=args.pixel_size,
                              zero_mag=args.zero_mag, psf_fwhm=args.psf_fwhm,
                              moffat_beta=args.moffat_beta)
        d = args.response_delta
        # property-resolved target (bins of true flux x size) or single global scalar
        if args.response_target_npz:
            tt = np.load(args.response_target_npz)
            ef, es, Rsim = tt["edges_flux"], tt["edges_size"], tt["Rsim"]
            bin_targets = torch.as_tensor(np.asarray(Rsim).reshape(-1), dtype=torch.float32)
            ccol = tt["crowd_col"].item() if "crowd_col" in tt.files else ""
            if ccol and np.asarray(Rsim).ndim == 3:
                ec = tt["edges_crowd"]; nf, ns, nb = Rsim.shape  # 3rd axis = crowding quantile bins

                def _bin_id(frame):
                    flux = frame["r_input_p"].to_numpy(float)
                    size = frame["Re_input_p"].to_numpy(float)
                    fi = np.clip(np.digitize(flux, ef) - 1, 0, nf - 1)
                    si = np.clip(np.digitize(size, es) - 1, 0, ns - 1)
                    cr = frame[ccol].to_numpy(float)
                    di = np.clip(np.digitize(cr, ec) - 1, 0, nb - 1)
                    return ((fi * ns + si) * nb + di).astype(np.int64)
                print(f"\nResponse-aware training ON (PROPERTY+CROWD[{ccol}]-RESOLVED): lambda={args.response_weight}, "
                      f"{nf}x{ns}x{nb} (flux x size x {ccol}) bins, "
                      f"R_sim {np.nanmin(Rsim):.3f}..{np.nanmax(Rsim):.3f}, delta={d}")
            elif "edges_dist" in tt.files and np.asarray(Rsim).ndim == 3:
                ed = tt["edges_dist"]; nf, ns, nb = Rsim.shape  # blend bin 0=isolated, 1..nb-1=by distance

                def _bin_id(frame):
                    flux = frame["r_input_p"].to_numpy(float)
                    size = frame["Re_input_p"].to_numpy(float)
                    fi = np.clip(np.digitize(flux, ef) - 1, 0, nf - 1)
                    si = np.clip(np.digitize(size, es) - 1, 0, ns - 1)
                    nbf = (frame["neighbored"].astype(bool).to_numpy() if "neighbored" in frame.columns
                           else np.zeros(len(frame), bool))
                    dist = (frame["distance"].to_numpy(float) if "distance" in frame.columns
                            else np.full(len(frame), np.inf))
                    di = np.where(nbf, 1 + np.clip(np.digitize(dist, ed) - 1, 0, nb - 2), 0)
                    return ((fi * ns + si) * nb + di).astype(np.int64)
                print(f"\nResponse-aware training ON (PROPERTY+BLEND-RESOLVED): lambda={args.response_weight}, "
                      f"{nf}x{ns}x{nb} (flux x size x blend) bins, "
                      f"R_sim {np.nanmin(Rsim):.3f}..{np.nanmax(Rsim):.3f}, delta={d}")
            else:
                nf, ns = Rsim.shape

                def _bin_id(frame):
                    flux = frame["r_input_p"].to_numpy(float)
                    size = frame["Re_input_p"].to_numpy(float)
                    fi = np.clip(np.digitize(flux, ef) - 1, 0, nf - 1)
                    si = np.clip(np.digitize(size, es) - 1, 0, ns - 1)
                    return (fi * ns + si).astype(np.int64)
                print(f"\nResponse-aware training ON (PROPERTY-RESOLVED): lambda={args.response_weight}, "
                      f"{nf}x{ns} bins, R_sim {Rsim.min():.3f}..{Rsim.max():.3f}, delta={d}")
        else:
            bin_targets = torch.as_tensor([args.response_target], dtype=torch.float32)

            def _bin_id(frame):
                return np.zeros(len(frame), dtype=np.int64)
            print(f"\nResponse-aware training ON (global): lambda={args.response_weight}, "
                  f"R_sim_target={args.response_target}, delta={d}")
        print(f"  building shifted conditioning contexts (analytic S_delta along e1 and e2; "
              f"difference={args.response_difference})...")

        def _resp_loader(frame, w, shuffle):
            tgt = target_transform.transform_frame(frame)
            c0 = build_shifted_context(frame, (1.0, 0.0), 0.0, condition_preprocessor,
                                       condition_features, rescale_kwargs)
            ce1 = build_shifted_context(frame, (1.0, 0.0), d, condition_preprocessor,
                                        condition_features, rescale_kwargs)
            ce2 = build_shifted_context(frame, (0.0, 1.0), d, condition_preprocessor,
                                        condition_features, rescale_kwargs)
            tensors = [
                torch.as_tensor(tgt, dtype=torch.float32),
                torch.as_tensor(c0, dtype=torch.float32),    # NLL context == unshifted resp context
            ]
            ww = (np.ones(len(frame), dtype=np.float32) if w is None
                  else np.asarray(w, dtype=np.float32).reshape(-1))
            bid = _bin_id(frame)
            tensors += [
                torch.as_tensor(ww, dtype=torch.float32),
                torch.as_tensor(c0, dtype=torch.float32),    # ctx0 (response baseline)
                torch.as_tensor(ce1, dtype=torch.float32),
                torch.as_tensor(ce2, dtype=torch.float32),
            ]
            if args.response_difference == "central":
                cm1 = build_shifted_context(frame, (1.0, 0.0), -d, condition_preprocessor,
                                            condition_features, rescale_kwargs)
                cm2 = build_shifted_context(frame, (0.0, 1.0), -d, condition_preprocessor,
                                            condition_features, rescale_kwargs)
                tensors += [
                    torch.as_tensor(cm1, dtype=torch.float32),
                    torch.as_tensor(cm2, dtype=torch.float32),
                ]
            tensors.append(torch.as_tensor(bid, dtype=torch.long))
            if args.gpu_resident and device.type == "cuda":
                return GPUBatches(tensors, args.batch_size, shuffle, device)
            ds = torch.utils.data.TensorDataset(*tensors)
            return torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                                               num_workers=args.num_workers, pin_memory=pin_memory)

        resp_train_loader = _resp_loader(train_df, train_w, True)
        resp_val_loader = _resp_loader(val_df, val_w, False)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    best_state = None
    best_val = np.inf
    wait = 0
    history = {"train_nll": [], "val_nll": []}
    # --- SWA: rolling window of the last-K end-of-epoch weight snapshots (kept on CPU). ---
    swa_k = max(int(args.swa_last_k), 1)
    swa_snapshots = deque(maxlen=swa_k)
    swa_epochs = deque(maxlen=swa_k)
    t0 = time.time()
    print("\n--- Training measurement flow ---")
    for epoch in range(1, args.epochs + 1):
        if response_on:
            train_nll, train_resp, train_R = epoch_response(
                model, resp_train_loader, device, target_scales01, args.response_delta,
                bin_targets, args.response_weight, args.response_difference,
                args.response_error, args.response_rel_floor,
                optimizer=optimizer, max_grad_norm=args.max_grad_norm)
            val_nll, val_resp, val_R = epoch_response(
                model, resp_val_loader, device, target_scales01, args.response_delta,
                bin_targets, args.response_weight, args.response_difference,
                args.response_error, args.response_rel_floor)
            history["train_nll"].append(train_nll)
            history["val_nll"].append(val_nll)
            history.setdefault("val_R", []).append(val_R)
            history.setdefault("val_resp", []).append(val_resp)
            print(f"  epoch {epoch:03d}: nll={train_nll:.5f}/{val_nll:.5f}  "
                  f"<R_model>(val)={val_R:+.4f} (target mean {float(bin_targets.mean()):.4f})  "
                  f"per-bin resp={val_resp:.2e}")
            selector = val_nll + args.response_weight * val_resp  # early-stop on TOTAL objective
        else:
            train_nll = epoch_nll(
                model,
                train_loader,
                device,
                optimizer=optimizer,
                max_grad_norm=args.max_grad_norm,
            )
            val_nll = epoch_nll(model, val_loader, device)
            history["train_nll"].append(train_nll)
            history["val_nll"].append(val_nll)
            print(f"  epoch {epoch:03d}: train_nll={train_nll:.6f}, val_nll={val_nll:.6f}")
            selector = val_nll

        # SWA: capture this completed epoch's weights on CPU (rolling last-K window). Done for
        # every completed epoch -- including the one that triggers early stopping below -- so the
        # average always covers the K most-recently-trained epochs regardless of when training ends.
        swa_snapshots.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        swa_epochs.append(int(epoch))

        if selector < best_val:
            best_val = selector
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            wait = 0
            # Persist the best checkpoint to disk on every improvement so long runs are
            # always evaluable and never lost if the job is cancelled/killed.
            os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
            save_measurement_model(
                args.output, model, condition_preprocessor, target_transform, model_config,
                metadata={"checkpoint": "best_so_far", "epoch": int(epoch),
                          "best_val_nll": float(best_val), "feature_set": args.feature_set,
                          "condition_features": condition_features, "target_features": target_features},
            )
        else:
            wait += 1
            if wait >= args.patience:
                print(f"  Early stopping after {epoch} epochs")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)

    train_log_prob = summarize_log_prob(model, train_loader, device)
    val_log_prob = summarize_log_prob(model, val_loader, device)
    print("\nLog-probability summary:")
    print(f"  Train mean={train_log_prob['mean_log_prob']:.6f}, std={train_log_prob['std_log_prob']:.6f}")
    print(f"  Val   mean={val_log_prob['mean_log_prob']:.6f}, std={val_log_prob['std_log_prob']:.6f}")

    metadata = {
        "model_family": "conditional_affine_flow",
        "density_name": "measurement_likelihood",
        "selection_name": args.selection_name,
        "target_column": args.target_column,
        "catalogue_path": args.catalogue,
        "feature_set": args.feature_set,
        "shear_case": None if args.shear_case is None else float(args.shear_case),
        "condition_features": condition_features,
        "condition_input_names": condition_preprocessor.output_names,
        "target_features": target_features,
        "target_raw_columns": sorted(raw_columns_for_measurement_targets(target_features)),
        "history": history,
        "best_val_nll": float(best_val),
        "train_log_prob": train_log_prob,
        "val_log_prob": val_log_prob,
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(val_df)),
        "seed": int(args.seed),
        "max_rows": None if args.max_rows is None else int(args.max_rows),
        "max_read_batches": args.max_read_batches,
        "decorrelate_shape_size": bool(args.decorrelate_shape_size),
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    # (a) best-val checkpoint -- the '_swabase' model (best_state is loaded into `model` above).
    save_measurement_model(
        args.output,
        model,
        condition_preprocessor,
        target_transform,
        model_config,
        metadata=metadata,
    )
    print(f"\nSaved measurement model (best-val): {args.output}")

    # (b) SWA-averaged checkpoint -- the '_swaavg' model. Average the float tensors of the last-K
    # end-of-epoch snapshots elementwise (equal weights); copy non-float entries (e.g. long buffers
    # like keep_indices) from the newest snapshot. The model contains NO BatchNorm / running-stats
    # layers (plain Linear+activation MLPs; the only buffers are the constant coupling `mask` and
    # the long `keep_indices`), so a straight weight average is well-defined and needs no BN recompute.
    swa_used_epochs = list(swa_epochs)
    if swa_snapshots:
        snaps = list(swa_snapshots)
        newest = snaps[-1]
        swa_state = {}
        for key, ref in newest.items():
            if torch.is_floating_point(ref):
                stacked = torch.stack([s[key].to(torch.float64) for s in snaps], dim=0)
                swa_state[key] = stacked.mean(dim=0).to(ref.dtype)
            else:
                swa_state[key] = ref.clone()
        model.load_state_dict(swa_state)
        model.to(device)

        base_dir, base_fn = os.path.split(os.path.abspath(args.output))
        if "swabase" in base_fn:
            swa_fn = base_fn.replace("swabase", "swaavg")
        else:
            stem_fn, ext_fn = os.path.splitext(base_fn)
            swa_fn = f"{stem_fn}_swaavg{ext_fn}"
        swa_output = os.path.join(base_dir, swa_fn)

        swa_metadata = dict(metadata)
        swa_metadata["checkpoint"] = "swa_average"
        swa_metadata["swa_last_k"] = int(swa_k)
        swa_metadata["swa_epochs"] = [int(e) for e in swa_used_epochs]
        save_measurement_model(
            swa_output,
            model,
            condition_preprocessor,
            target_transform,
            model_config,
            metadata=swa_metadata,
        )
        print(f"Saved SWA-averaged model (last-{swa_k}): {swa_output}")
        print(f"  SWA average over epochs: {swa_used_epochs}")

    output_stem = os.path.splitext(os.path.abspath(args.output))[0]
    history["swa_last_k"] = int(swa_k)
    history["swa_epochs"] = np.asarray(swa_used_epochs, dtype=np.int64)
    np.savez(f"{output_stem}_train_curve.npz", **history)

    print(f"Total training time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
