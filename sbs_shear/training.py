"""Training-loop primitives for the measurement-flow trainer.

These functions were consolidated from the pre-V3 trainer forks. Changes here alter
the frozen training recipe and therefore require focused tests and a WORKLOG entry.
Historical byte-identity snapshots remain under ``archive/pre-v3/``.
"""

from __future__ import annotations

import numpy as np
import torch

from .preprocessing import SHEAR_FEATURES, rescale
from .shear_map import apply_shear_to_ellipticity

__all__ = [
    "GPUBatches",
    "make_loader",
    "split_data",
    "_finite_target_mask",
    "_add_legacy_missing_shear",
    "build_shifted_context",
    "compute_decorrelation_weights",
    "per_target_weights",
    "epoch_nll",
    "summarize_log_prob",
]


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


def split_data(frame, seed, validation_size):
    # sklearn is imported lazily so that `import sbs_shear.training` keeps working on the
    # login node, where sims1's sklearn -> scipy chain fails with `GLIBCXX_3.4.30 not found`.
    # Only the trainers (which run on compute nodes) ever reach this call.
    from sklearn.model_selection import train_test_split

    if not (0.0 < validation_size < 1.0):
        raise ValueError("validation_size must be in (0, 1)")
    train_df, val_df = train_test_split(frame, test_size=validation_size, random_state=seed)
    print(f"  Split: train={len(train_df):,}, val={len(val_df):,}")
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def _finite_target_mask(frame, targets):
    values = frame[list(targets)].to_numpy(dtype=np.float32, copy=False)
    return np.isfinite(values).all(axis=1)


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


def summarize_log_prob(model, loader, device):
    """Mean/std log-probability over a loader. Diagnostic only -- never backpropagated.

    `torch.no_grad()` is required, not cosmetic. Without it every batch's log-prob keeps its
    autograd graph alive, so this builds a graph over the ENTIRE train set (4M rows in the real
    runs) purely to take a mean. Newer torch then refuses the `.numpy()` outright with "Can't call
    numpy() on Tensor that requires grad"; the torch in `sims1` allows it and silently pays the
    memory. Numerically this changes nothing -- it only stops gradients being tracked.
    """
    model.eval()
    values = []
    with torch.no_grad():
        for batch in loader:
            target, context = batch[0], batch[1]
            lp = model.log_prob(target.to(device, non_blocking=True),
                                context.to(device, non_blocking=True))
            values.append(lp.cpu().numpy())
    if not values:
        return {"mean_log_prob": np.nan, "std_log_prob": np.nan}
    arr = np.concatenate(values)
    return {
        "mean_log_prob": float(np.mean(arr)),
        "std_log_prob": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
    }
