"""Study whether SBS selection autograd shear gradients are signal or noise."""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

SBS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SBS_ROOT not in sys.path:
    sys.path.insert(0, SBS_ROOT)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from evaluate_selection_blends import (  # noqa: E402
    GRADIENT_FEATURES,
    load_eval_sample,
    radius_label,
    subset_masks,
)
from sbs_shear.coordinates import add_shear_aligned_gradients, perturb_frame_along_sky_shear  # noqa: E402
from sbs_shear.selection_model import load_selection_model  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalogue",
        default="/project/ls-gruen/users/zekang.zhang/lsst_selec_emu/sbs_detection_measurement_catalogue_train.feather",
    )
    parser.add_argument("--model", default=os.path.join(SBS_ROOT, "models/selection_mlp_detected_v2.pt"))
    parser.add_argument("--output-dir", default=os.path.join(SBS_ROOT, "results/selection_gradient_study_v2"))
    parser.add_argument("--target-column", default="detected")
    parser.add_argument("--radii", type=float, nargs="+", default=[1.0, 2.0, 3.0])
    parser.add_argument("--max-rows", type=int, default=1_000_000)
    parser.add_argument("--max-read-batches", type=int, default=None)
    parser.add_argument("--gradient-rows", type=int, default=150_000)
    parser.add_argument("--finite-diff-rows", type=int, default=20_000)
    parser.add_argument("--finite-diff-eps", type=float, default=1.0e-3)
    parser.add_argument("--batch-size", type=int, default=65_536)
    parser.add_argument("--gradient-batch-size", type=int, default=16_384)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=20260501)
    parser.add_argument("--pixel-rms", type=float, default=0.312)
    parser.add_argument("--pixel-size", type=float, default=0.2)
    parser.add_argument("--zero-mag", type=float, default=30.0)
    parser.add_argument("--psf-fwhm", type=float, default=0.73)
    parser.add_argument("--moffat-beta", type=float, default=2.224)
    return parser.parse_args()


def cuda_available():
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def finite_summary(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "sem": np.nan,
            "mean_over_sem": np.nan,
            "std": np.nan,
            "mean_over_std": np.nan,
            "median": np.nan,
            "mean_abs": np.nan,
            "frac_pos": np.nan,
            "skew": np.nan,
            "excess_kurtosis": np.nan,
            "p01": np.nan,
            "p05": np.nan,
            "p16": np.nan,
            "p84": np.nan,
            "p95": np.nan,
            "p99": np.nan,
        }
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    sem = std / np.sqrt(len(arr)) if len(arr) > 1 else np.nan
    return {
        "n": int(len(arr)),
        "mean": float(np.mean(arr)),
        "sem": float(sem),
        "mean_over_sem": float(np.mean(arr) / sem) if sem and np.isfinite(sem) and sem > 0 else np.nan,
        "std": std,
        "mean_over_std": float(np.mean(arr) / std) if std > 0 else np.nan,
        "median": float(np.median(arr)),
        "mean_abs": float(np.mean(np.abs(arr))),
        "frac_pos": float(np.mean(arr > 0)),
        "skew": float(stats.skew(arr, bias=False)) if len(arr) > 2 else np.nan,
        "excess_kurtosis": float(stats.kurtosis(arr, fisher=True, bias=False)) if len(arr) > 3 else np.nan,
        "p01": float(np.quantile(arr, 0.01)),
        "p05": float(np.quantile(arr, 0.05)),
        "p16": float(np.quantile(arr, 0.16)),
        "p84": float(np.quantile(arr, 0.84)),
        "p95": float(np.quantile(arr, 0.95)),
        "p99": float(np.quantile(arr, 0.99)),
    }


def summarize_gradient_frame(aug, subset_name):
    columns = [
        "dPsel_dgamma1_input_p", "dPsel_dgamma2_input_p",
        "dPsel_dgamma1_input_s", "dPsel_dgamma2_input_s",
        "dPsel_dgamma1_sky_p", "dPsel_dgamma2_sky_p",
        "dPsel_dgamma1_sky_s", "dPsel_dgamma2_sky_s",
        "dPsel_dgamma_parallel_p", "dPsel_dgamma_perp_p",
        "dPsel_dgamma_parallel_s", "dPsel_dgamma_perp_s",
        "dPsel_dgamma_cross_p", "dPsel_dgamma_cross_s",
        "dPsel_dgamma_pair_parallel_p", "dPsel_dgamma_pair_cross_p",
        "dPsel_dgamma_pair_parallel_s", "dPsel_dgamma_pair_cross_s",
    ]
    rows = []
    for column in [name for name in columns if name in aug.columns]:
        row = {"subset": subset_name, "quantity": column}
        row.update(finite_summary(aug[column]))
        rows.append(row)
    return pd.DataFrame(rows)


def sample_subset(frame, mask, n_rows, rng):
    subset = frame.loc[mask].reset_index(drop=True)
    if n_rows and len(subset) > n_rows:
        subset = subset.iloc[rng.choice(len(subset), n_rows, replace=False)].reset_index(drop=True)
    return subset


def shear_case_metrics(frame, prob, radii):
    work = frame[["shear_case", "neighbored", "distance"]].copy()
    work["target"] = frame["detected"].astype(int).to_numpy()
    work["prob"] = np.asarray(prob, dtype=float)
    rows = []
    for subset, mask in subset_masks(frame, radii).items():
        grp_frame = work.loc[mask]
        for shear_case, grp in grp_frame.groupby("shear_case", dropna=False):
            rows.append({
                "subset": subset,
                "shear_case": float(shear_case),
                "rows": int(len(grp)),
                "observed_rate": float(grp["target"].mean()),
                "mean_prob": float(grp["prob"].mean()),
                "mean_residual": float((grp["prob"] - grp["target"]).mean()),
            })
    return pd.DataFrame(rows)


def distance_gradient_bins(frame, aug, bins):
    distance = frame["distance"].to_numpy(dtype=float)
    neighbored = frame["neighbored"].astype(bool).to_numpy()
    columns = [
        "dPsel_dgamma1_input_p", "dPsel_dgamma2_input_p",
        "dPsel_dgamma1_sky_p", "dPsel_dgamma2_sky_p",
        "dPsel_dgamma_parallel_p", "dPsel_dgamma_perp_p",
        "dPsel_dgamma_cross_p",
        "dPsel_dgamma_pair_parallel_p", "dPsel_dgamma_pair_cross_p",
        "dPsel_dgamma1_input_s", "dPsel_dgamma2_input_s",
        "dPsel_dgamma1_sky_s", "dPsel_dgamma2_sky_s",
        "dPsel_dgamma_parallel_s", "dPsel_dgamma_perp_s",
        "dPsel_dgamma_cross_s",
        "dPsel_dgamma_pair_parallel_s", "dPsel_dgamma_pair_cross_s",
    ]
    columns = [name for name in columns if name in aug.columns]
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = neighbored & np.isfinite(distance) & (distance >= lo) & (distance < hi)
        row = {
            "distance_min": float(lo),
            "distance_max": float(hi),
            "distance_center": float(0.5 * (lo + hi)),
            "rows": int(mask.sum()),
        }
        for column in columns:
            values = aug.loc[mask, column].to_numpy(dtype=float)
            row[f"{column}_mean"] = float(np.nanmean(values)) if np.isfinite(values).any() else np.nan
            row[f"{column}_mean_abs"] = float(np.nanmean(np.abs(values))) if np.isfinite(values).any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def finite_difference_checks(bundle, frame, grad_aug, eps, batch_size):
    rows = []
    for feature in GRADIENT_FEATURES:
        if feature not in bundle.preprocessor.feature_names:
            continue
        plus = frame.copy()
        minus = frame.copy()
        plus[feature] = plus[feature] + eps
        minus[feature] = minus[feature] - eps
        fd = (bundle.predict_proba(plus, batch_size=batch_size) - bundle.predict_proba(minus, batch_size=batch_size)) / (2.0 * eps)
        ad = grad_aug[f"dPsel_d{feature}"].to_numpy(dtype=float)
        rows.append(compare_arrays(f"component:{feature}", ad, fd))

    for suffix in ("p", "s"):
        pair_parallel = f"gamma_parallel_{suffix}"
        pair_parallel_blend = f"gamma_parallel_{suffix}_blend"
        pframe_parallel = f"gamma_pframe_parallel_{suffix}"
        pframe_parallel_blend = f"gamma_pframe_parallel_{suffix}_blend"
        if pair_parallel in bundle.preprocessor.feature_names:
            plus = frame.copy()
            minus = frame.copy()
            plus[pair_parallel] = plus[pair_parallel] + eps
            minus[pair_parallel] = minus[pair_parallel] - eps
            fd = (
                bundle.predict_proba(plus, batch_size=batch_size)
                - bundle.predict_proba(minus, batch_size=batch_size)
            ) / (2.0 * eps)
            ad = grad_aug[f"dPsel_dgamma_parallel_{suffix}"].to_numpy(dtype=float)
            rows.append(compare_arrays(f"pair_parallel:{suffix}", ad, fd))
            continue
        if pair_parallel_blend in bundle.preprocessor.feature_names:
            plus = frame.copy()
            minus = frame.copy()
            plus[pair_parallel_blend] = plus[pair_parallel_blend] + eps
            minus[pair_parallel_blend] = minus[pair_parallel_blend] - eps
            fd = (
                bundle.predict_proba(plus, batch_size=batch_size)
                - bundle.predict_proba(minus, batch_size=batch_size)
            ) / (2.0 * eps)
            if "neighbored" in frame.columns:
                fd = fd * frame["neighbored"].astype(float).to_numpy()
            ad = grad_aug[f"dPsel_dgamma_parallel_{suffix}"].to_numpy(dtype=float)
            rows.append(compare_arrays(f"pair_parallel:{suffix}", ad, fd))
            continue
        if pframe_parallel in bundle.preprocessor.feature_names:
            plus = frame.copy()
            minus = frame.copy()
            plus[pframe_parallel] = plus[pframe_parallel] + eps
            minus[pframe_parallel] = minus[pframe_parallel] - eps
            fd = (
                bundle.predict_proba(plus, batch_size=batch_size)
                - bundle.predict_proba(minus, batch_size=batch_size)
            ) / (2.0 * eps)
            ad = grad_aug[f"dPsel_dgamma_parallel_{suffix}"].to_numpy(dtype=float)
            rows.append(compare_arrays(f"pframe_parallel:{suffix}", ad, fd))
            continue
        if pframe_parallel_blend in bundle.preprocessor.feature_names:
            plus = frame.copy()
            minus = frame.copy()
            plus[pframe_parallel_blend] = plus[pframe_parallel_blend] + eps
            minus[pframe_parallel_blend] = minus[pframe_parallel_blend] - eps
            fd = (
                bundle.predict_proba(plus, batch_size=batch_size)
                - bundle.predict_proba(minus, batch_size=batch_size)
            ) / (2.0 * eps)
            if "neighbored" in frame.columns:
                fd = fd * frame["neighbored"].astype(float).to_numpy()
            ad = grad_aug[f"dPsel_dgamma_parallel_{suffix}"].to_numpy(dtype=float)
            rows.append(compare_arrays(f"pframe_parallel:{suffix}", ad, fd))
            continue

        g1_name = f"gamma1_sky_{suffix}"
        g2_name = f"gamma2_sky_{suffix}"
        if {g1_name, g2_name}.issubset(frame.columns):
            g1 = frame[g1_name].to_numpy(dtype=float)
            g2 = frame[g2_name].to_numpy(dtype=float)
        else:
            raw1_name = f"gamma1_input_{suffix}"
            raw2_name = f"gamma2_input_{suffix}"
            if not {raw1_name, raw2_name}.issubset(frame.columns):
                continue
            g1 = frame[raw1_name].to_numpy(dtype=float)
            g2 = frame[raw2_name].to_numpy(dtype=float)
        gmag = np.hypot(g1, g2)
        good = np.isfinite(gmag) & (gmag > 1.0e-8)
        if not np.any(good):
            continue
        plus = perturb_frame_along_sky_shear(
            frame.loc[good], suffix, eps, feature_names=bundle.preprocessor.feature_names
        )
        minus = perturb_frame_along_sky_shear(
            frame.loc[good], suffix, -eps, feature_names=bundle.preprocessor.feature_names
        )
        fd = (bundle.predict_proba(plus, batch_size=batch_size) - bundle.predict_proba(minus, batch_size=batch_size)) / (2.0 * eps)
        ad = grad_aug.loc[good, f"dPsel_dgamma_parallel_{suffix}"].to_numpy(dtype=float)
        rows.append(compare_arrays(f"parallel:{suffix}", ad, fd))
    return pd.DataFrame(rows)


def compare_arrays(name, autograd, finite_diff):
    good = np.isfinite(autograd) & np.isfinite(finite_diff)
    autograd = autograd[good]
    finite_diff = finite_diff[good]
    diff = finite_diff - autograd
    corr = np.corrcoef(autograd, finite_diff)[0, 1] if len(autograd) > 1 else np.nan
    return {
        "check": name,
        "rows": int(len(autograd)),
        "autograd_mean": float(np.mean(autograd)) if len(autograd) else np.nan,
        "finite_diff_mean": float(np.mean(finite_diff)) if len(finite_diff) else np.nan,
        "corr": float(corr),
        "rmse": float(np.sqrt(np.mean(diff ** 2))) if len(diff) else np.nan,
        "mae": float(np.mean(np.abs(diff))) if len(diff) else np.nan,
        "max_abs_diff": float(np.max(np.abs(diff))) if len(diff) else np.nan,
    }


def plot_gradient_histograms(aug_by_subset, path):
    quantities = [
        "dPsel_dgamma1_input_p",
        "dPsel_dgamma2_input_p",
        "dPsel_dgamma1_sky_p",
        "dPsel_dgamma2_sky_p",
        "dPsel_dgamma_parallel_p",
        "dPsel_dgamma_perp_p",
        "dPsel_dgamma_cross_p",
        "dPsel_dgamma_pair_parallel_p",
        "dPsel_dgamma_pair_cross_p",
        "dPsel_dgamma1_input_s",
        "dPsel_dgamma2_input_s",
        "dPsel_dgamma1_sky_s",
        "dPsel_dgamma2_sky_s",
        "dPsel_dgamma_parallel_s",
        "dPsel_dgamma_perp_s",
        "dPsel_dgamma_cross_s",
        "dPsel_dgamma_pair_parallel_s",
        "dPsel_dgamma_pair_cross_s",
    ]
    quantities = [name for name in quantities if all(name in aug.columns for aug in aug_by_subset.values())]
    subsets = list(aug_by_subset)
    fig, axes = plt.subplots(len(subsets), len(quantities), figsize=(3.2 * len(quantities), 3.0 * len(subsets)), squeeze=False)
    for row, subset in enumerate(subsets):
        aug = aug_by_subset[subset]
        for col, quantity in enumerate(quantities):
            ax = axes[row, col]
            values = aug[quantity].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            ax.hist(values, bins=80, histtype="step", linewidth=1.5)
            ax.axvline(0.0, color="k", linestyle="--", linewidth=0.8, alpha=0.5)
            ax.set_title(f"{subset}\n{quantity}", fontsize=8)
            if col == 0:
                ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_distance_means(table, path):
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    for ax, suffix in zip(axes, ("p", "s")):
        cross_name = f"dPsel_dgamma_cross_{suffix}_mean"
        if cross_name not in table.columns:
            cross_name = f"dPsel_dgamma_perp_{suffix}_mean"
        for quantity, style in [(f"dPsel_dgamma_parallel_{suffix}_mean", "-"), (cross_name, "--")]:
            if quantity not in table.columns:
                continue
            ax.plot(table["distance_center"], table[quantity], marker="o", linestyle=style, label=quantity)
        ax.axhline(0.0, color="k", linestyle=":", linewidth=1)
        ax.set_ylabel("mean gradient")
        ax.legend(fontsize=8)
    axes[-1].set_xlabel("neighbor distance [arcsec]")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = args.device or ("cuda" if cuda_available() else "cpu")
    print(f"Loading model: {args.model}")
    bundle = load_selection_model(args.model, device=device)
    features = list(bundle.preprocessor.feature_names)
    print(f"Device: {bundle.device}")
    print(f"Features: {len(features)}")

    frame = load_eval_sample(args, features)
    prob = bundle.predict_proba(frame, batch_size=args.batch_size)

    shear_metrics = shear_case_metrics(frame, prob, args.radii)
    shear_metrics.to_csv(os.path.join(args.output_dir, "shear_case_metrics.csv"), index=False)

    rng = np.random.default_rng(args.seed + 101)
    masks = subset_masks(frame, args.radii)
    requested_subsets = ["all_after_cuts", "neighbored"] + [
        f"close_le_{radius_label(radius)}arcsec" for radius in args.radii
    ]
    requested_subsets = [name for name in requested_subsets if name in masks]

    summary_tables = []
    aug_by_subset = {}
    frame_by_subset = {}
    for subset_name in requested_subsets:
        subset = sample_subset(frame, masks[subset_name], args.gradient_rows, rng)
        if len(subset) == 0:
            continue
        print(f"\nAutograd subset {subset_name}: {len(subset):,} rows")
        _, grad_df = bundle.probability_and_gradient(
            subset,
            gradient_features=[name for name in GRADIENT_FEATURES if name in features],
            batch_size=args.gradient_batch_size,
        )
        aug = add_shear_aligned_gradients(subset, grad_df)
        aug_by_subset[subset_name] = aug
        frame_by_subset[subset_name] = subset
        summary_tables.append(summarize_gradient_frame(aug, subset_name))

    gradient_summary = pd.concat(summary_tables, ignore_index=True)
    gradient_summary.to_csv(os.path.join(args.output_dir, "gradient_distribution_summary.csv"), index=False)
    plot_gradient_histograms(aug_by_subset, os.path.join(args.output_dir, "gradient_component_vs_aligned_histograms.png"))

    if "neighbored" in aug_by_subset:
        bins = np.linspace(0.0, max(args.radii), 19)
        distance_table = distance_gradient_bins(frame_by_subset["neighbored"], aug_by_subset["neighbored"], bins)
        distance_table.to_csv(os.path.join(args.output_dir, "gradient_by_distance.csv"), index=False)
        plot_distance_means(distance_table, os.path.join(args.output_dir, "gradient_by_distance.png"))

    fd_subset_name = f"close_le_{radius_label(2.0)}arcsec" if "close_le_2arcsec" in aug_by_subset else requested_subsets[-1]
    fd_frame = frame_by_subset[fd_subset_name]
    fd_aug = aug_by_subset[fd_subset_name]
    if args.finite_diff_rows and len(fd_frame) > args.finite_diff_rows:
        choice = rng.choice(len(fd_frame), args.finite_diff_rows, replace=False)
        fd_frame = fd_frame.iloc[choice].reset_index(drop=True)
        fd_aug = fd_aug.iloc[choice].reset_index(drop=True)
    print(f"\nFinite-difference autograd check on {fd_subset_name}: {len(fd_frame):,} rows")
    fd_checks = finite_difference_checks(
        bundle,
        fd_frame,
        fd_aug,
        eps=args.finite_diff_eps,
        batch_size=args.batch_size,
    )
    fd_checks.insert(0, "subset", fd_subset_name)
    fd_checks.insert(1, "eps", args.finite_diff_eps)
    fd_checks.to_csv(os.path.join(args.output_dir, "finite_difference_autograd_checks.csv"), index=False)

    print("\nGradient distribution summary:")
    print(gradient_summary.to_string(index=False))
    print("\nFinite-difference checks:")
    print(fd_checks.to_string(index=False))
    print(f"\nWrote gradient study to: {args.output_dir}")


if __name__ == "__main__":
    main()
