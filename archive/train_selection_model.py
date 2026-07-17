"""Train the standalone SBS Bernoulli selection model.

The current pilot target is SExtractor detection, encoded by the catalogue
column ``detected``.  The script keeps the API selection-named so later targets
can include measurement success and analysis cuts without changing the model
interface.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
import torch
from sklearn.model_selection import train_test_split

SBS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SBS_ROOT not in sys.path:
    sys.path.insert(0, SBS_ROOT)

from sbs_shear.preprocessing import (  # noqa: E402
    DEFAULT_SELECTION_CUTS,
    SHEAR_FEATURES,
    raw_columns_for_selection_features,
    rescale,
    source_select_selection,
)
from sbs_shear.selection_model import (  # noqa: E402
    DEFAULT_SELECTION_FEATURES,
    SELECTION_FEATURE_SETS,
    FocalLossWithLogits,
    SelectionMLP,
    TabularPreprocessor,
    collect_logits_and_targets,
    fit_temperature,
    save_selection_model,
)


def _stratify_or_none(y):
    labels, counts = np.unique(y, return_counts=True)
    return y if len(labels) == 2 and np.min(counts) >= 2 else None


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
        print("  This is useful for smoke tests, but not for validating dP(s=1)/dgamma.")
    missing = [name for name in features if name not in df.columns]
    if missing:
        raise KeyError(f"Missing required feature columns: {missing}")
    return df


def _append_to_priority_sample(reservoir, batch, max_rows, rng):
    if max_rows is None or max_rows <= 0:
        return pd.concat([reservoir, batch], ignore_index=True) if reservoir is not None else batch

    batch = batch.copy()
    batch["__sample_key"] = rng.random(len(batch))
    if reservoir is None:
        reservoir = batch
    else:
        reservoir = pd.concat([reservoir, batch], ignore_index=True)

    if len(reservoir) > 2 * max_rows:
        reservoir = reservoir.nlargest(max_rows, "__sample_key").reset_index(drop=True)
    return reservoir


def _finalize_priority_sample(reservoir, max_rows):
    if reservoir is None:
        raise RuntimeError("No rows were loaded from the selection catalogue")
    if max_rows is not None and max_rows > 0 and len(reservoir) > max_rows:
        reservoir = reservoir.nlargest(max_rows, "__sample_key").reset_index(drop=True)
    if "__sample_key" in reservoir.columns:
        reservoir = reservoir.drop(columns="__sample_key")
    return reservoir.reset_index(drop=True)


def load_selection_data(args, features):
    rng = np.random.default_rng(args.seed)
    t0 = time.time()

    print(f"Loading selection sample: {args.catalogue}")
    print(f"  Target column: {args.target_column}")
    print(f"  Requested max rows after cuts: {args.max_rows:,}" if args.max_rows else "  Requested max rows after cuts: all")

    reservoir = None
    raw_rows = 0
    selected_rows = 0
    batches_seen = 0

    with ipc.open_file(args.catalogue) as reader:
        available = set(reader.schema.names)
        extra_columns = {args.target_column}
        if args.shear_case is not None and "shear_case" in available:
            extra_columns.add("shear_case")
        requested_columns = sorted(
            raw_columns_for_selection_features(features, available_columns=available)
            | extra_columns
        )
        missing_non_shear = [
            name for name in requested_columns
            if name not in available and name not in SHEAR_FEATURES
        ]
        if missing_non_shear:
            raise KeyError(f"Missing required catalogue columns: {missing_non_shear}")
        read_columns = [name for name in requested_columns if name in available]
        print(f"  File record batches: {reader.num_record_batches:,}")
        print(f"  Reading columns: {len(read_columns):,}")

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
            batch = source_select_selection(batch, cuts=DEFAULT_SELECTION_CUTS)
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
            batch = _add_legacy_missing_shear(batch, features)
            selected_rows += len(batch)
            reservoir = _append_to_priority_sample(reservoir, batch, args.max_rows, rng)
            batches_seen += 1
            if args.progress_every and batches_seen % args.progress_every == 0:
                kept = 0 if reservoir is None else len(reservoir)
                print(
                    f"  batches={batches_seen:,}, raw={raw_rows:,}, "
                    f"selected={selected_rows:,}, reservoir={kept:,}"
                )

    dataset = _finalize_priority_sample(reservoir, args.max_rows)
    y = dataset[args.target_column].astype(np.float32).to_numpy()
    print(f"  Raw rows scanned: {raw_rows:,}")
    print(f"  Rows after cuts before sampling: {selected_rows:,}")
    print(f"  Rows used: {len(dataset):,}")
    print(f"  Selection rate: {y.mean():.4f}")
    print(f"  Load/sample time: {time.time() - t0:.1f}s")
    return dataset, y


def split_data(frame, y, seed, validation_size, calibration_size):
    holdout_size = validation_size + calibration_size
    if not (0.0 < holdout_size < 1.0):
        raise ValueError("validation_size + calibration_size must be in (0, 1)")

    train_df, hold_df, y_train, y_hold = train_test_split(
        frame,
        y,
        test_size=holdout_size,
        random_state=seed,
        stratify=_stratify_or_none(y),
    )
    cal_fraction = calibration_size / holdout_size
    val_df, cal_df, y_val, y_cal = train_test_split(
        hold_df,
        y_hold,
        test_size=cal_fraction,
        random_state=seed + 1,
        stratify=_stratify_or_none(y_hold),
    )
    print(f"  Split: train={len(train_df):,}, val={len(val_df):,}, calibration={len(cal_df):,}")
    return (
        train_df.reset_index(drop=True), y_train,
        val_df.reset_index(drop=True), y_val,
        cal_df.reset_index(drop=True), y_cal,
    )


def make_loader(x, y, batch_size, shuffle=False, num_workers=0, pin_memory=False):
    dataset = torch.utils.data.TensorDataset(
        torch.as_tensor(x, dtype=torch.float32),
        torch.as_tensor(y, dtype=torch.float32),
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def epoch_loss(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    n_total = 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        if training:
            loss.backward()
            optimizer.step()
        total_loss += float(loss.detach().cpu()) * len(yb)
        n_total += len(yb)
    return total_loss / max(n_total, 1)


@torch.no_grad()
def predict_loader(model, loader, device, temperature=1.0):
    model.eval()
    probs = []
    labels = []
    for xb, yb in loader:
        logits = model(xb.to(device, non_blocking=True)) / temperature
        probs.append(torch.sigmoid(logits).cpu().numpy())
        labels.append(yb.numpy())
    return np.concatenate(labels), np.concatenate(probs)


def binary_metrics(y_true, prob):
    y_true = y_true.astype(int)
    prob = np.asarray(prob, dtype=np.float64)
    clipped = np.clip(prob, 1e-7, 1.0 - 1e-7)
    pred = (prob >= 0.5).astype(int)
    metrics = {
        "logloss": float(-np.mean(y_true * np.log(clipped) + (1 - y_true) * np.log(1 - clipped))),
        "brier": float(np.mean((prob - y_true) ** 2)),
        "accuracy": float(np.mean(pred == y_true)),
    }
    recalls = []
    for label in np.unique(y_true):
        mask = y_true == label
        recalls.append(np.mean(pred[mask] == label))
    metrics["balanced_accuracy"] = float(np.mean(recalls)) if recalls else np.nan
    try:
        from sklearn.metrics import roc_auc_score
        metrics["auc"] = float(roc_auc_score(y_true, prob)) if len(np.unique(y_true)) == 2 else np.nan
    except Exception:
        metrics["auc"] = np.nan
    return metrics


def format_metrics(metrics):
    keys = ["logloss", "brier", "accuracy", "balanced_accuracy", "auc"]
    return ", ".join(f"{key}={metrics[key]:.5f}" for key in keys)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalogue", required=True, help="Blendemu/SBS selection catalogue feather file")
    parser.add_argument("--output", default=os.path.join(SBS_ROOT, "models/selection_mlp_detected_v3_coord.pt"))
    parser.add_argument("--target-column", default="detected")
    parser.add_argument("--selection-name", default="sextractor_detected")
    parser.add_argument(
        "--feature-set",
        default="v6_primary_frame",
        choices=sorted(SELECTION_FEATURE_SETS),
        help="Conditioning feature set. 'g0_shearfree' drops applied-shear inputs "
        "for the refined g=0 forward model; pair with --shear-case 0.0.",
    )
    parser.add_argument(
        "--shear-case",
        type=float,
        default=None,
        help="If set, keep only rows whose catalogue 'shear_case' matches this "
        "value (e.g. 0.0 to train the shear-free forward model on g=0 only).",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-rows", type=int, default=2_000_000)
    parser.add_argument("--max-read-batches", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--activation", default="silu", choices=["silu", "gelu", "tanh"])
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument(
        "--loss",
        default="bce",
        choices=["bce", "focal"],
        help="Training loss. BCE is the default because calibrated probabilities are the science target.",
    )
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--focal-alpha", type=float, default=None)
    parser.add_argument("--validation-size", type=float, default=0.15)
    parser.add_argument("--calibration-size", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=321)
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

    features = SELECTION_FEATURE_SETS[args.feature_set]
    print(f"Feature set: {args.feature_set} ({len(features)} features)")
    if args.shear_case is not None:
        print(f"Restricting to shear_case == {args.shear_case}")
    frame, y = load_selection_data(args, features)
    train_df, y_train, val_df, y_val, cal_df, y_cal = split_data(
        frame, y, args.seed, args.validation_size, args.calibration_size
    )

    preprocessor = TabularPreprocessor.fit(train_df, features, add_missing_indicators=True)
    train_loader = make_loader(
        preprocessor.transform_frame(train_df), y_train,
        args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin_memory,
    )
    val_loader = make_loader(
        preprocessor.transform_frame(val_df), y_val,
        args.batch_size, num_workers=args.num_workers, pin_memory=pin_memory,
    )
    cal_loader = make_loader(
        preprocessor.transform_frame(cal_df), y_cal,
        args.batch_size, num_workers=args.num_workers, pin_memory=pin_memory,
    )

    model_config = {
        "input_dim": preprocessor.output_dim,
        "hidden_dim": args.hidden_dim,
        "n_layers": args.n_layers,
        "dropout": args.dropout,
        "activation": args.activation,
    }
    model = SelectionMLP(**model_config).to(device)
    if args.loss == "focal":
        criterion = FocalLossWithLogits(gamma=args.focal_gamma, alpha=args.focal_alpha)
    else:
        criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_state = None
    best_val = np.inf
    wait = 0
    history = {"train_loss": [], "val_loss": []}
    t0 = time.time()
    print("\n--- Training selection model ---")
    for epoch in range(1, args.epochs + 1):
        train_loss = epoch_loss(model, train_loader, criterion, device, optimizer=optimizer)
        val_loss = epoch_loss(model, val_loader, criterion, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(f"  epoch {epoch:03d}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= args.patience:
                print(f"  Early stopping after {epoch} epochs")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)

    cal_logits, cal_targets = collect_logits_and_targets(model, cal_loader, device)
    temperature = fit_temperature(cal_logits, cal_targets)
    print(f"\nCalibration temperature: {temperature:.4f}")

    y_val_eval, val_prob = predict_loader(model, val_loader, device, temperature)
    y_cal_eval, cal_prob = predict_loader(model, cal_loader, device, temperature)
    val_metrics = binary_metrics(y_val_eval, val_prob)
    cal_metrics = binary_metrics(y_cal_eval, cal_prob)
    print(f"  Val: {format_metrics(val_metrics)}")
    print(f"  Cal: {format_metrics(cal_metrics)}")

    metadata = {
        "model_family": "selection_mlp",
        "selection_name": args.selection_name,
        "target_column": args.target_column,
        "catalogue_path": args.catalogue,
        "feature_set": args.feature_set,
        "shear_case": None if args.shear_case is None else float(args.shear_case),
        "features": list(features),
        "input_names": preprocessor.output_names,
        "loss": args.loss,
        "focal_gamma": float(args.focal_gamma),
        "focal_alpha": None if args.focal_alpha is None else float(args.focal_alpha),
        "history": history,
        "best_val_loss": float(best_val),
        "val_metrics": val_metrics,
        "calibration_metrics": cal_metrics,
        "selection_rate": float(y.mean()),
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(val_df)),
        "calibration_rows": int(len(cal_df)),
        "seed": int(args.seed),
        "max_rows": None if args.max_rows is None else int(args.max_rows),
        "max_read_batches": args.max_read_batches,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    save_selection_model(args.output, model, preprocessor, model_config, temperature, metadata)
    output_stem = os.path.splitext(os.path.abspath(args.output))[0]
    np.save(f"{output_stem}_boundary.npy", np.array([[train_df[f].min(), train_df[f].max()] for f in features]))
    np.savez(f"{output_stem}_train_curve.npz", **history)

    print(f"\nSaved selection model: {args.output}")
    print(f"Total training time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
