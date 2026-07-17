"""Evaluate SBS selection-model performance on close-neighbour blends."""

from __future__ import annotations

import argparse
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

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
from sbs_shear.coordinates import add_shear_aligned_gradients  # noqa: E402
from sbs_shear.selection_model import load_selection_model  # noqa: E402


GRADIENT_FEATURES = [
    "gamma1_input_p", "gamma2_input_p",
    "gamma1_input_s", "gamma2_input_s",
    "gamma1_sky_p", "gamma2_sky_p",
    "gamma1_sky_s", "gamma2_sky_s",
    "gamma_parallel_p", "gamma_cross_p",
    "gamma_parallel_s", "gamma_cross_s",
    "gamma_parallel_p_blend", "gamma_cross_p_blend",
    "gamma_parallel_s_blend", "gamma_cross_s_blend",
    "gamma_pframe_parallel_p", "gamma_pframe_cross_p",
    "gamma_pframe_parallel_s", "gamma_pframe_cross_s",
    "gamma_pframe_parallel_s_blend", "gamma_pframe_cross_s_blend",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalogue",
        default="/project/ls-gruen/users/zekang.zhang/lsst_selec_emu/sbs_detection_measurement_catalogue_train.feather",
    )
    parser.add_argument(
        "--model",
        default=os.path.join(SBS_ROOT, "models/selection_mlp_detected_v2.pt"),
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(SBS_ROOT, "results/selection_blends_v2"),
    )
    parser.add_argument("--target-column", default="detected")
    parser.add_argument("--radii", type=float, nargs="+", default=[2.0, 3.0])
    parser.add_argument("--distance-bins", type=float, nargs="+", default=None)
    parser.add_argument("--max-rows", type=int, default=1_000_000)
    parser.add_argument("--max-read-batches", type=int, default=None)
    parser.add_argument("--gradient-rows", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=65_536)
    parser.add_argument("--gradient-batch-size", type=int, default=16_384)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--pixel-rms", type=float, default=0.312)
    parser.add_argument("--pixel-size", type=float, default=0.2)
    parser.add_argument("--zero-mag", type=float, default=30.0)
    parser.add_argument("--psf-fwhm", type=float, default=0.73)
    parser.add_argument("--moffat-beta", type=float, default=2.224)
    return parser.parse_args()


def append_to_priority_sample(reservoir, batch, max_rows, rng):
    if max_rows is None or max_rows <= 0:
        return pd.concat([reservoir, batch], ignore_index=True) if reservoir is not None else batch
    batch = batch.copy()
    batch["__sample_key"] = rng.random(len(batch))
    reservoir = pd.concat([reservoir, batch], ignore_index=True) if reservoir is not None else batch
    if len(reservoir) > 2 * max_rows:
        reservoir = reservoir.nlargest(max_rows, "__sample_key").reset_index(drop=True)
    return reservoir


def finalize_priority_sample(reservoir, max_rows):
    if reservoir is None:
        raise RuntimeError("No rows were loaded from the catalogue")
    if max_rows is not None and max_rows > 0 and len(reservoir) > max_rows:
        reservoir = reservoir.nlargest(max_rows, "__sample_key").reset_index(drop=True)
    if "__sample_key" in reservoir.columns:
        reservoir = reservoir.drop(columns="__sample_key")
    return reservoir.reset_index(drop=True)


def add_legacy_missing_shear(frame, features):
    frame = frame.copy()
    for name in features:
        if name not in frame.columns and name in SHEAR_FEATURES:
            frame[name] = 0.0
    missing = [name for name in features if name not in frame.columns]
    if missing:
        raise KeyError(f"Missing required feature columns: {missing}")
    return frame


def load_eval_sample(args, features):
    rng = np.random.default_rng(args.seed)
    reservoir = None
    raw_rows = 0
    selected_rows = 0
    t0 = time.time()

    with ipc.open_file(args.catalogue) as reader:
        available = set(reader.schema.names)
        requested_columns = raw_columns_for_selection_features(features, available_columns=available)
        requested_columns.update([
            args.target_column,
            "case", "shear_case",
            "neighbored", "distance", "polarization_angle",
            "r_input_p", "Re_input_p",
        ])
        missing = [
            name for name in requested_columns
            if name not in available and name not in SHEAR_FEATURES
        ]
        if missing:
            raise KeyError(f"Missing required catalogue columns: {missing}")
        read_columns = sorted(name for name in requested_columns if name in available)

        print(f"File record batches: {reader.num_record_batches:,}")
        print(f"Reading columns: {len(read_columns):,}")
        for batch_index in range(reader.num_record_batches):
            if args.max_read_batches is not None and batch_index >= args.max_read_batches:
                break
            table = pa.Table.from_batches([reader.get_batch(batch_index)]).select(read_columns)
            batch = table.to_pandas()
            raw_rows += len(batch)
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
            batch = add_legacy_missing_shear(batch, features)
            selected_rows += len(batch)
            reservoir = append_to_priority_sample(reservoir, batch, args.max_rows, rng)
            if args.progress_every and (batch_index + 1) % args.progress_every == 0:
                kept = 0 if reservoir is None else len(reservoir)
                print(
                    f"batches={batch_index + 1:,}, raw={raw_rows:,}, "
                    f"selected={selected_rows:,}, reservoir={kept:,}"
                )

    sample = finalize_priority_sample(reservoir, args.max_rows)
    print(f"Raw rows scanned: {raw_rows:,}")
    print(f"Rows after cuts before sampling: {selected_rows:,}")
    print(f"Rows used: {len(sample):,}")
    print(f"Selection rate: {sample[args.target_column].mean():.4f}")
    print(f"Load/sample time: {time.time() - t0:.1f}s")
    return sample


def binary_metrics(y_true, prob):
    y_true = np.asarray(y_true, dtype=int)
    prob = np.asarray(prob, dtype=float)
    clipped = np.clip(prob, 1.0e-7, 1.0 - 1.0e-7)
    pred = prob >= 0.5
    out = {
        "rows": int(len(y_true)),
        "observed_rate": float(np.mean(y_true)) if len(y_true) else np.nan,
        "mean_prob": float(np.mean(prob)) if len(prob) else np.nan,
        "mean_residual": float(np.mean(prob - y_true)) if len(prob) else np.nan,
        "logloss": np.nan,
        "brier": np.nan,
        "accuracy": np.nan,
        "balanced_accuracy": np.nan,
        "auc": np.nan,
    }
    if len(y_true) == 0:
        return out
    out["logloss"] = float(log_loss(y_true, clipped, labels=[0, 1]))
    out["brier"] = float(brier_score_loss(y_true, clipped))
    out["accuracy"] = float(accuracy_score(y_true, pred))
    out["balanced_accuracy"] = float(balanced_accuracy_score(y_true, pred))
    if len(np.unique(y_true)) == 2:
        out["auc"] = float(roc_auc_score(y_true, prob))
    return out


def subset_masks(frame, radii):
    distance = frame["distance"].to_numpy(dtype=float)
    neighbored = frame["neighbored"].astype(bool).to_numpy()
    finite_distance = np.isfinite(distance)
    masks = {
        "all_after_cuts": np.ones(len(frame), dtype=bool),
        "not_neighbored": ~neighbored,
        "neighbored": neighbored & finite_distance,
    }
    for radius in radii:
        label = radius_label(radius)
        masks[f"close_le_{label}arcsec"] = neighbored & finite_distance & (distance <= radius)
    return masks


def radius_label(radius):
    return f"{radius:g}".replace(".", "p")


def summarize_subsets(frame, y_true, prob, radii):
    rows = []
    for name, mask in subset_masks(frame, radii).items():
        row = {"subset": name}
        row.update(binary_metrics(y_true[mask], prob[mask]))
        if len(frame):
            row["fraction_of_eval"] = float(mask.mean())
        rows.append(row)
    return pd.DataFrame(rows)


def distance_bin_table(frame, y_true, prob, bins):
    distance = frame["distance"].to_numpy(dtype=float)
    neighbored = frame["neighbored"].astype(bool).to_numpy()
    y_true = np.asarray(y_true, dtype=float)
    prob = np.asarray(prob, dtype=float)
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = neighbored & np.isfinite(distance) & (distance >= lo) & (distance < hi)
        row = {
            "distance_min": lo,
            "distance_max": hi,
            "distance_center": 0.5 * (lo + hi),
            "rows": int(mask.sum()),
            "observed_rate": np.nan,
            "mean_prob": np.nan,
            "mean_residual": np.nan,
        }
        if mask.any():
            row["observed_rate"] = float(np.mean(y_true[mask]))
            row["mean_prob"] = float(np.mean(prob[mask]))
            row["mean_residual"] = float(np.mean(prob[mask] - y_true[mask]))
        rows.append(row)
    return pd.DataFrame(rows)


def gradient_summary(gradient_frame, subset, radius, prob):
    summary = pd.DataFrame(index=gradient_frame.columns)
    summary["mean"] = gradient_frame.mean(axis=0)
    summary["median"] = gradient_frame.median(axis=0)
    summary["std"] = gradient_frame.std(axis=0)
    summary["mean_abs"] = gradient_frame.abs().mean(axis=0)
    summary["p05"] = gradient_frame.quantile(0.05, axis=0)
    summary["p16"] = gradient_frame.quantile(0.16, axis=0)
    summary["p84"] = gradient_frame.quantile(0.84, axis=0)
    summary["p95"] = gradient_frame.quantile(0.95, axis=0)
    summary.insert(0, "gradient", summary.index)
    summary.insert(0, "mean_prob_gradient_sample", float(np.mean(prob)) if len(prob) else np.nan)
    summary.insert(0, "gradient_rows", int(len(gradient_frame)))
    summary.insert(0, "radius_arcsec", float(radius))
    summary.insert(0, "subset", subset)
    return summary.reset_index(drop=True)


def plot_distance_bins(table, path):
    good = table["rows"] > 0
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(
        table.loc[good, "distance_center"],
        table.loc[good, "observed_rate"],
        marker="o",
        linewidth=2,
        label="observed",
    )
    ax.plot(
        table.loc[good, "distance_center"],
        table.loc[good, "mean_prob"],
        marker="s",
        linewidth=2,
        linestyle="--",
        label="predicted",
    )
    ax.set_xlabel("neighbor distance [arcsec]")
    ax.set_ylabel("selection rate")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_gradient_histograms(gradient_samples, path):
    if not gradient_samples:
        return
    subsets = list(gradient_samples)
    columns = list(next(iter(gradient_samples.values())).columns)
    fig, axes = plt.subplots(len(subsets), len(columns), figsize=(4 * len(columns), 3.2 * len(subsets)), squeeze=False)
    for row, subset in enumerate(subsets):
        gradients = gradient_samples[subset]
        for col, column in enumerate(columns):
            ax = axes[row, col]
            ax.hist(gradients[column], bins=80, histtype="step", linewidth=1.8)
            ax.axvline(0, color="k", linestyle="--", linewidth=1, alpha=0.5)
            ax.set_title(f"{subset}\n{column}")
            ax.set_xlabel("gradient")
            if col == 0:
                ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Loading model: {args.model}")
    bundle = load_selection_model(args.model, device=args.device or ("cuda" if _cuda_available() else "cpu"))
    features = list(bundle.preprocessor.feature_names)
    print(f"Device: {bundle.device}")
    print(f"Features: {len(features)}")

    frame = load_eval_sample(args, features)
    y_true = frame[args.target_column].astype(int).to_numpy()
    prob = bundle.predict_proba(frame, batch_size=args.batch_size)

    metrics = summarize_subsets(frame, y_true, prob, args.radii)
    metrics_path = os.path.join(args.output_dir, "blend_subset_metrics.csv")
    metrics.to_csv(metrics_path, index=False)
    print("\nSubset metrics:")
    print(metrics.to_string(index=False))

    bins = np.asarray(args.distance_bins if args.distance_bins is not None else np.linspace(0.0, max(args.radii), 25))
    distance_bins = distance_bin_table(frame, y_true, prob, bins)
    distance_path = os.path.join(args.output_dir, "blend_distance_bins.csv")
    distance_bins.to_csv(distance_path, index=False)
    plot_distance_bins(distance_bins, os.path.join(args.output_dir, "blend_selection_by_distance.png"))

    gradient_features = [name for name in GRADIENT_FEATURES if name in features]
    gradient_tables = []
    gradient_samples = {}
    rng = np.random.default_rng(args.seed + 13)
    for radius in args.radii:
        label = f"close_le_{radius_label(radius)}arcsec"
        mask = subset_masks(frame, [radius])[label]
        subset = frame.loc[mask].reset_index(drop=True)
        if len(subset) == 0:
            continue
        if args.gradient_rows and len(subset) > args.gradient_rows:
            subset = subset.iloc[rng.choice(len(subset), args.gradient_rows, replace=False)].reset_index(drop=True)
        print(f"\nAutograd subset {label}: {len(subset):,} rows")
        prob_grad, grad_df = bundle.probability_and_gradient(
            subset,
            gradient_features=gradient_features,
            batch_size=args.gradient_batch_size,
        )
        grad_aug = add_shear_aligned_gradients(subset, grad_df)
        gradient_tables.append(gradient_summary(grad_aug, label, radius, prob_grad))
        gradient_samples[label] = grad_aug

    if gradient_tables:
        gradients = pd.concat(gradient_tables, ignore_index=True)
        gradients_path = os.path.join(args.output_dir, "blend_gradient_summary.csv")
        gradients.to_csv(gradients_path, index=False)
        print("\nGradient summary:")
        print(gradients.to_string(index=False))
        plot_gradient_histograms(gradient_samples, os.path.join(args.output_dir, "blend_gradient_histograms.png"))

    print(f"\nWrote blend diagnostics to: {args.output_dir}")


def _cuda_available():
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


if __name__ == "__main__":
    main()
