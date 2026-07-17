"""Train the standalone SBS differentiable detection classifier."""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

SBS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SBS_ROOT not in sys.path:
    sys.path.insert(0, SBS_ROOT)

from sbs_shear.detection_classifier import (  # noqa: E402
    DEFAULT_FEATURES,
    DetectionMLP,
    FocalLossWithLogits,
    TabularPreprocessor,
    collect_logits_and_targets,
    fit_temperature,
    save_detection_classifier,
)
from sbs_shear.preprocessing import DEFAULT_CUTS, SHEAR_FEATURES, rescale, source_select_detection  # noqa: E402


def _stratify_or_none(y):
    labels, counts = np.unique(y, return_counts=True)
    return y if len(labels) == 2 and np.min(counts) >= 2 else None


def _sample_rows(df, max_rows, seed):
    if max_rows is None or max_rows <= 0 or len(df) <= max_rows:
        return df
    y = df["detected"].astype(int).to_numpy()
    sampled, _ = train_test_split(
        df,
        train_size=max_rows,
        random_state=seed,
        stratify=_stratify_or_none(y),
    )
    return sampled.reset_index(drop=True)


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
        print("  This is useful for smoke tests, but not for validating dP/dgamma.")
    missing = [name for name in features if name not in df.columns]
    if missing:
        raise KeyError(f"Missing required feature columns: {missing}")
    return df


def load_data(args, features):
    print(f"Loading: {args.catalogue}")
    dataset = pd.read_feather(args.catalogue)
    print(f"  Raw: {dataset.shape[0]:,} rows")

    dataset = source_select_detection(dataset, cuts=DEFAULT_CUTS)
    print(f"  After cuts: {dataset.shape[0]:,} rows")

    dataset = rescale(
        dataset,
        pixel_rms=args.pixel_rms,
        pixel_size=args.pixel_size,
        zero_mag=args.zero_mag,
        psf_fwhm=args.psf_fwhm,
        moffat_beta=args.moffat_beta,
    )
    dataset = _add_legacy_missing_shear(dataset, features)
    dataset = _sample_rows(dataset, args.max_rows, args.seed)
    if args.max_rows:
        print(f"  Training subset: {dataset.shape[0]:,} rows")

    y = dataset["detected"].astype(np.float32).to_numpy()
    print(f"  Positive rate: {y.mean():.4f}")
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


def make_loader(x, y, batch_size, shuffle=False):
    dataset = torch.utils.data.TensorDataset(
        torch.as_tensor(x, dtype=torch.float32),
        torch.as_tensor(y, dtype=torch.float32),
    )
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def epoch_loss(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    n_total = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
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
        logits = model(xb.to(device)) / temperature
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
    parser.add_argument("--catalogue", required=True, help="Blendemu/SBS detection catalogue feather file")
    parser.add_argument("--output", default=os.path.join(SBS_ROOT, "models/detection_classifier.pt"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--activation", default="silu", choices=["silu", "gelu", "tanh"])
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
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
    print(f"Using device: {device}")

    features = DEFAULT_FEATURES
    frame, y = load_data(args, features)
    train_df, y_train, val_df, y_val, cal_df, y_cal = split_data(
        frame, y, args.seed, args.validation_size, args.calibration_size
    )

    preprocessor = TabularPreprocessor.fit(train_df, features, add_missing_indicators=True)
    train_loader = make_loader(preprocessor.transform_frame(train_df), y_train, args.batch_size, shuffle=True)
    val_loader = make_loader(preprocessor.transform_frame(val_df), y_val, args.batch_size)
    cal_loader = make_loader(preprocessor.transform_frame(cal_df), y_cal, args.batch_size)

    model_config = {
        "input_dim": preprocessor.output_dim,
        "hidden_dim": args.hidden_dim,
        "n_layers": args.n_layers,
        "dropout": args.dropout,
        "activation": args.activation,
    }
    model = DetectionMLP(**model_config).to(device)
    criterion = FocalLossWithLogits(gamma=args.focal_gamma, alpha=args.focal_alpha)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_state = None
    best_val = np.inf
    wait = 0
    history = {"train_loss": [], "val_loss": []}
    t0 = time.time()
    print("\n--- Training classifier ---")
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
        "catalogue_path": args.catalogue,
        "features": list(features),
        "input_names": preprocessor.output_names,
        "history": history,
        "best_val_loss": float(best_val),
        "val_metrics": val_metrics,
        "calibration_metrics": cal_metrics,
        "positive_rate": float(y.mean()),
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(val_df)),
        "calibration_rows": int(len(cal_df)),
        "seed": int(args.seed),
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    save_detection_classifier(args.output, model, preprocessor, model_config, temperature, metadata)
    output_stem = os.path.splitext(os.path.abspath(args.output))[0]
    np.save(f"{output_stem}_boundary.npy", np.array([[train_df[f].min(), train_df[f].max()] for f in features]))
    np.savez(f"{output_stem}_train_curve.npz", **history)

    print(f"\nSaved classifier: {args.output}")
    print(f"Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
