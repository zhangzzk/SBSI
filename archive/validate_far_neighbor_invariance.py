"""Check whether the selection MLP overuses non-rendered neighbour features."""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SBS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SBS_ROOT not in sys.path:
    sys.path.insert(0, SBS_ROOT)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from evaluate_selection_blends import binary_metrics, load_eval_sample  # noqa: E402
from sbs_shear.selection_model import load_selection_model  # noqa: E402


CATALOGUE_DEFAULT = (
    "/project/ls-gruen/users/zekang.zhang/lsst_selec_emu/"
    "sbs_skycos/sbs_detection_measurement_catalogue_train.feather"
)


SHUFFLE_GROUPS = {
    "distance_only": [
        "distance_scaled_blend",
    ],
    "secondary_properties": [
        "Re_input_s_scaled_blend",
        "r_input_s_scaled_blend",
        "flux_ratio_blend",
        "sersic_n_input_s_blend",
    ],
    "blend_geometry": [
        "pair_pframe_cos2_blend",
        "pair_pframe_sin2_blend",
    ],
    "secondary_shape": [
        "e_pframe_parallel_s_blend",
        "e_pframe_cross_s_blend",
    ],
    "secondary_shear": [
        "gamma_pframe_parallel_s_blend",
        "gamma_pframe_cross_s_blend",
    ],
    "primary_shear_relative_shape": [
        "gamma_pframe_parallel_p",
        "gamma_pframe_cross_p",
    ],
    "all_blend_features": [
        "distance_scaled_blend",
        "pair_pframe_cos2_blend",
        "pair_pframe_sin2_blend",
        "Re_input_s_scaled_blend",
        "r_input_s_scaled_blend",
        "flux_ratio_blend",
        "sersic_n_input_s_blend",
        "e_pframe_parallel_s_blend",
        "e_pframe_cross_s_blend",
        "gamma_pframe_parallel_s_blend",
        "gamma_pframe_cross_s_blend",
    ],
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalogue", default=CATALOGUE_DEFAULT)
    parser.add_argument(
        "--model",
        default=os.path.join(SBS_ROOT, "models/selection_mlp_detected_v6_primary_frame.pt"),
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(SBS_ROOT, "results/selection_far_neighbor_invariance_v6_primary_frame"),
    )
    parser.add_argument("--target-column", default="detected")
    parser.add_argument("--max-rows", type=int, default=1_000_000)
    parser.add_argument("--max-read-batches", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=65_536)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=20260502)
    parser.add_argument("--distance-thresholds", type=float, nargs="+", default=[3.0, 4.0, 5.0, 6.0])
    parser.add_argument("--repeats", type=int, default=5)
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


def shuffled_frame(frame, columns, rng):
    columns = [name for name in columns if name in frame.columns]
    out = frame.copy()
    if not columns or len(frame) == 0:
        return out
    order = rng.permutation(len(frame))
    out.loc[:, columns] = frame.iloc[order][columns].to_numpy()
    return out


def delta_summary(base_prob, shuffled_prob):
    delta = np.asarray(shuffled_prob, dtype=float) - np.asarray(base_prob, dtype=float)
    return {
        "mean_delta_prob": float(np.mean(delta)),
        "mean_abs_delta_prob": float(np.mean(np.abs(delta))),
        "rms_delta_prob": float(np.sqrt(np.mean(delta ** 2))),
        "p50_abs_delta_prob": float(np.quantile(np.abs(delta), 0.50)),
        "p90_abs_delta_prob": float(np.quantile(np.abs(delta), 0.90)),
        "p99_abs_delta_prob": float(np.quantile(np.abs(delta), 0.99)),
        "max_abs_delta_prob": float(np.max(np.abs(delta))),
    }


def subset_masks(frame, thresholds):
    distance = frame["distance"].to_numpy(dtype=float)
    neighbored = frame["neighbored"].astype(bool).to_numpy()
    finite = np.isfinite(distance)
    masks = {
        "all_after_cuts": np.ones(len(frame), dtype=bool),
        "close_neighbored_le_3": neighbored & finite & (distance <= 3.0),
        "not_neighbored": ~neighbored,
    }
    for threshold in thresholds:
        label = f"not_neighbored_gt_{threshold:g}".replace(".", "p")
        masks[label] = ~neighbored & finite & (distance > threshold)
    return masks


def plot_summary(summary, path):
    plot = summary[summary["repeat"] == "mean"].copy()
    if plot.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    for subset, grp in plot.groupby("subset"):
        ax.plot(
            grp["shuffle_group"],
            grp["mean_abs_delta_prob"],
            marker="o",
            linewidth=1.8,
            label=subset,
        )
    ax.set_ylabel("mean |delta P(s=1)| after shuffle")
    ax.set_xlabel("shuffled feature group")
    ax.tick_params(axis="x", rotation=35)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
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
    y_true = frame[args.target_column].astype(int).to_numpy()
    base_prob = bundle.predict_proba(frame, batch_size=args.batch_size)

    masks = subset_masks(frame, args.distance_thresholds)
    rng = np.random.default_rng(args.seed + 99)
    rows = []
    raw_rows = []

    for subset, mask in masks.items():
        if not np.any(mask):
            continue
        subset_frame = frame.loc[mask].reset_index(drop=True)
        subset_y = y_true[mask]
        subset_base = base_prob[mask]
        metric = binary_metrics(subset_y, subset_base)
        print(f"\nSubset {subset}: {len(subset_frame):,} rows")
        print(pd.Series(metric).to_string())
        for group_name, columns in SHUFFLE_GROUPS.items():
            columns = [name for name in columns if name in features]
            if not columns:
                continue
            repeat_rows = []
            for repeat in range(args.repeats):
                shuffled = shuffled_frame(subset_frame, columns, rng)
                shuffled_prob = bundle.predict_proba(shuffled, batch_size=args.batch_size)
                row = {
                    "subset": subset,
                    "shuffle_group": group_name,
                    "repeat": repeat,
                    "rows": int(len(subset_frame)),
                    "columns": ",".join(columns),
                }
                row.update(delta_summary(subset_base, shuffled_prob))
                shuffled_metrics = binary_metrics(subset_y, shuffled_prob)
                row["base_logloss"] = metric["logloss"]
                row["shuffled_logloss"] = shuffled_metrics["logloss"]
                row["delta_logloss"] = shuffled_metrics["logloss"] - metric["logloss"]
                row["base_auc"] = metric["auc"]
                row["shuffled_auc"] = shuffled_metrics["auc"]
                row["delta_auc"] = metric["auc"] - shuffled_metrics["auc"]
                rows.append(row)
                repeat_rows.append(row)
            mean_row = pd.DataFrame(repeat_rows).drop(columns=["repeat"]).mean(numeric_only=True).to_dict()
            mean_row.update({
                "subset": subset,
                "shuffle_group": group_name,
                "repeat": "mean",
                "rows": int(len(subset_frame)),
                "columns": ",".join(columns),
            })
            raw_rows.append(mean_row)

    repeats = pd.DataFrame(rows)
    summary = pd.concat([repeats, pd.DataFrame(raw_rows)], ignore_index=True)
    repeats.to_csv(os.path.join(args.output_dir, "far_neighbor_shuffle_repeats.csv"), index=False)
    summary.to_csv(os.path.join(args.output_dir, "far_neighbor_shuffle_summary.csv"), index=False)
    plot_summary(summary, os.path.join(args.output_dir, "far_neighbor_shuffle_summary.png"))

    readme = [
        "# Far-Neighbor Invariance Check",
        "",
        "Rows with `neighbored=False` have a nearest-neighbour reference frame,",
        "but the nearest neighbour is outside the close-blend radius. Predictions",
        "should be insensitive to shuffling far-neighbour properties if the model",
        "is not overusing non-rendered neighbour information.",
        "",
        f"- Catalogue: `{args.catalogue}`",
        f"- Model: `{args.model}`",
        f"- Sample rows: `{args.max_rows}`",
        f"- Repeats: `{args.repeats}`",
        "",
        "Primary output: `far_neighbor_shuffle_summary.csv`.",
    ]
    with open(os.path.join(args.output_dir, "README.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(readme) + "\n")

    print(f"\nWrote far-neighbour invariance diagnostics to: {args.output_dir}")


if __name__ == "__main__":
    main()
