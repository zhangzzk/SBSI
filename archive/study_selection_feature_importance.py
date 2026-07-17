"""Train a pilot selection MLP and measure permutation feature importance.

The goal is to prune redundant SBS selection-model inputs before the next
production retraining.  The script intentionally keeps the pilot training path
close to ``train_selection_model.py`` so the importance numbers reflect the
actual model family and preprocessing.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

SBS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SBS_ROOT not in sys.path:
    sys.path.insert(0, SBS_ROOT)

from sbs_shear.selection_model import (  # noqa: E402
    DEFAULT_SELECTION_FEATURES,
    FocalLossWithLogits,
    SelectionMLP,
    SelectionModelBundle,
    TabularPreprocessor,
    collect_logits_and_targets,
    fit_temperature,
    save_selection_model,
)
from train_selection_model import (  # noqa: E402
    binary_metrics,
    epoch_loss,
    format_metrics,
    load_selection_data,
    make_loader,
    predict_loader,
    split_data,
)


CATALOGUE_DEFAULT = (
    "/project/ls-gruen/users/zekang.zhang/lsst_selec_emu/"
    "sbs_skycos/sbs_detection_measurement_catalogue_train.feather"
)


FEATURE_GROUPS = {
    "primary_size_flux": [
        "Re_input_p_scaled",
        "r_input_p_scaled",
    ],
    "secondary_size_flux": [
        "Re_input_s_scaled_blend",
        "r_input_s_scaled_blend",
        "flux_ratio_blend",
    ],
    "sersic": [
        "sersic_n_input_p",
        "sersic_n_input_s_blend",
    ],
    "intrinsic_shapes": [
        "e_abs_p",
        "e_pframe_parallel_s_blend",
        "e_pframe_cross_s_blend",
    ],
    "distance_status": [
        "distance_scaled_blend",
        "neighbored",
    ],
    "blend_geometry": [
        "pair_pframe_cos2_blend",
        "pair_pframe_sin2_blend",
    ],
    "secondary_shape": [
        "e_pframe_parallel_s_blend",
        "e_pframe_cross_s_blend",
    ],
    "primary_frame_shear": [
        "gamma_pframe_parallel_p",
        "gamma_pframe_cross_p",
        "gamma_pframe_parallel_s_blend",
        "gamma_pframe_cross_s_blend",
    ],
}


SCIENTIFIC_KEEP_FEATURES = {
    "gamma_pframe_parallel_p",
    "gamma_pframe_cross_p",
    "gamma_pframe_parallel_s_blend",
    "gamma_pframe_cross_s_blend",
}


@dataclass
class TrainedPilot:
    model: SelectionMLP
    preprocessor: TabularPreprocessor
    temperature: float
    model_config: dict
    history: dict
    best_val_loss: float
    val_metrics: dict
    calibration_metrics: dict


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalogue", default=CATALOGUE_DEFAULT)
    parser.add_argument(
        "--output-dir",
        default=os.path.join(SBS_ROOT, "results/selection_feature_importance_v6_primary_frame"),
    )
    parser.add_argument(
        "--pilot-output",
        default=os.path.join(SBS_ROOT, "models/selection_mlp_feature_importance_pilot_v6_primary_frame.pt"),
    )
    parser.add_argument("--target-column", default="detected")
    parser.add_argument("--selection-name", default="sextractor_detected")
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-rows", type=int, default=1_500_000)
    parser.add_argument("--importance-rows", type=int, default=250_000)
    parser.add_argument("--max-read-batches", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--predict-batch-size", type=int, default=65536)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.02)
    parser.add_argument("--activation", default="silu", choices=["silu", "gelu", "tanh"])
    parser.add_argument("--lr", type=float, default=7.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--loss", default="bce", choices=["bce", "focal"])
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--focal-alpha", type=float, default=None)
    parser.add_argument("--validation-size", type=float, default=0.15)
    parser.add_argument("--calibration-size", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--permutation-repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=911)
    parser.add_argument("--pixel-rms", type=float, default=0.312)
    parser.add_argument("--pixel-size", type=float, default=0.2)
    parser.add_argument("--zero-mag", type=float, default=30.0)
    parser.add_argument("--psf-fwhm", type=float, default=0.73)
    parser.add_argument("--moffat-beta", type=float, default=2.224)
    parser.add_argument(
        "--keep-threshold-logloss",
        type=float,
        default=1.0e-4,
        help="Features below this mean logloss importance are marked as prune candidates.",
    )
    parser.add_argument(
        "--keep-threshold-auc",
        type=float,
        default=1.0e-4,
        help="Features below this mean AUC importance are marked as prune candidates.",
    )
    return parser.parse_args()


def train_pilot(args, device, train_df, y_train, val_df, y_val, cal_df, y_cal, features):
    pin_memory = device.type == "cuda"
    preprocessor = TabularPreprocessor.fit(train_df, features, add_missing_indicators=True)
    train_loader = make_loader(
        preprocessor.transform_frame(train_df),
        y_train,
        args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = make_loader(
        preprocessor.transform_frame(val_df),
        y_val,
        args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    cal_loader = make_loader(
        preprocessor.transform_frame(cal_df),
        y_cal,
        args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
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
    print("\n--- Training pilot selection model ---")
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
    print(f"  Pilot training time: {time.time() - t0:.1f}s")

    return TrainedPilot(
        model=model,
        preprocessor=preprocessor,
        temperature=temperature,
        model_config=model_config,
        history=history,
        best_val_loss=float(best_val),
        val_metrics=val_metrics,
        calibration_metrics=cal_metrics,
    )


def choose_importance_sample(val_df, y_val, n_rows, seed):
    if n_rows is None or n_rows <= 0 or n_rows >= len(val_df):
        return val_df.reset_index(drop=True), np.asarray(y_val, dtype=np.float32)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(val_df), size=n_rows, replace=False)
    return val_df.iloc[idx].reset_index(drop=True), np.asarray(y_val, dtype=np.float32)[idx]


def _permuted_frame(frame, columns, rng):
    columns = list(columns)
    permuted = frame.copy()
    order = rng.permutation(len(frame))
    permuted.loc[:, columns] = frame.iloc[order][columns].to_numpy()
    return permuted


def _metric_row(y_true, prob, baseline_metrics):
    metrics = binary_metrics(y_true, prob)
    return {
        "logloss": metrics["logloss"],
        "brier": metrics["brier"],
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "auc": metrics["auc"],
        "delta_logloss": metrics["logloss"] - baseline_metrics["logloss"],
        "delta_brier": metrics["brier"] - baseline_metrics["brier"],
        "delta_accuracy": baseline_metrics["accuracy"] - metrics["accuracy"],
        "delta_balanced_accuracy": baseline_metrics["balanced_accuracy"] - metrics["balanced_accuracy"],
        "delta_auc": baseline_metrics["auc"] - metrics["auc"],
    }


def run_permutation_importance(bundle, frame, y, features, args):
    rng = np.random.default_rng(args.seed + 10_000)
    print("\n--- Permutation importance ---")
    print(f"  Importance rows: {len(frame):,}")
    print(f"  Repeats per feature/group: {args.permutation_repeats}")
    baseline_prob = bundle.predict_proba(frame, batch_size=args.predict_batch_size)
    baseline_metrics = binary_metrics(y, baseline_prob)
    print(f"  Baseline: {format_metrics(baseline_metrics)}")

    rows = []
    for index, feature in enumerate(features, start=1):
        print(f"  feature {index:02d}/{len(features):02d}: {feature}")
        for repeat in range(args.permutation_repeats):
            permuted = _permuted_frame(frame, [feature], rng)
            prob = bundle.predict_proba(permuted, batch_size=args.predict_batch_size)
            row = _metric_row(y, prob, baseline_metrics)
            row.update({"kind": "feature", "name": feature, "repeat": repeat, "columns": feature})
            rows.append(row)

    available_groups = {
        name: [col for col in columns if col in features]
        for name, columns in FEATURE_GROUPS.items()
    }
    available_groups = {name: columns for name, columns in available_groups.items() if columns}
    for index, (name, columns) in enumerate(available_groups.items(), start=1):
        print(f"  group {index:02d}/{len(available_groups):02d}: {name}")
        for repeat in range(args.permutation_repeats):
            permuted = _permuted_frame(frame, columns, rng)
            prob = bundle.predict_proba(permuted, batch_size=args.predict_batch_size)
            row = _metric_row(y, prob, baseline_metrics)
            row.update({"kind": "group", "name": name, "repeat": repeat, "columns": ",".join(columns)})
            rows.append(row)

    return baseline_metrics, pd.DataFrame(rows)


def summarize_importance(raw_importance, args):
    metric_columns = [
        "logloss",
        "brier",
        "accuracy",
        "balanced_accuracy",
        "auc",
        "delta_logloss",
        "delta_brier",
        "delta_accuracy",
        "delta_balanced_accuracy",
        "delta_auc",
    ]
    summary = (
        raw_importance
        .groupby(["kind", "name", "columns"], as_index=False)[metric_columns]
        .agg(["mean", "std"])
    )
    summary.columns = [
        "_".join(part for part in column if part)
        if isinstance(column, tuple) else column
        for column in summary.columns
    ]
    summary = summary.rename(
        columns={
            "kind_": "kind",
            "name_": "name",
            "columns_": "columns",
        }
    )
    summary["prune_candidate"] = False
    feature_mask = summary["kind"] == "feature"
    low_logloss = summary["delta_logloss_mean"] < args.keep_threshold_logloss
    low_auc = summary["delta_auc_mean"] < args.keep_threshold_auc
    scientific_keep = summary["name"].isin(SCIENTIFIC_KEEP_FEATURES)
    summary.loc[feature_mask & low_logloss & low_auc & ~scientific_keep, "prune_candidate"] = True
    summary["scientific_keep"] = summary["name"].isin(SCIENTIFIC_KEEP_FEATURES)
    return summary.sort_values(["kind", "delta_logloss_mean"], ascending=[True, False])


def write_readme(path, args, baseline_metrics, feature_summary, group_summary, actual_importance_rows):
    prune = feature_summary.loc[feature_summary["prune_candidate"], "name"].tolist()
    keep_science = sorted(SCIENTIFIC_KEEP_FEATURES)
    lines = [
        "# Selection Feature Importance V6 Primary Frame",
        "",
        "Pilot permutation-importance study for the primary-major-axis-frame SBS selection MLP.",
        "",
        "## Inputs",
        "",
        f"- Catalogue: `{args.catalogue}`",
        f"- Sampled rows for pilot training: `{args.max_rows}`",
        f"- Requested rows for permutation importance: `{args.importance_rows}`",
        f"- Actual rows for permutation importance: `{actual_importance_rows}`",
        f"- Permutation repeats: `{args.permutation_repeats}`",
        f"- Target column: `{args.target_column}`",
        "",
        "## Baseline Metrics",
        "",
    ]
    for key, value in baseline_metrics.items():
        lines.append(f"- `{key}`: {value:.8g}")
    lines.extend(
        [
            "",
            "## Pruning Rule",
            "",
            "A feature is marked as a prune candidate only when both mean",
            f"`delta_logloss < {args.keep_threshold_logloss:g}` and mean",
            f"`delta_auc < {args.keep_threshold_auc:g}`. The primary-frame shear columns",
            "are retained even if their classifier permutation score is small,",
            "because they are the derivative coordinates for `dP(s=1)/dgamma`.",
            "",
            "Scientific keep features:",
        ]
    )
    lines.extend([f"- `{name}`" for name in keep_science])
    lines.extend(["", "Current prune candidates:"])
    lines.extend([f"- `{name}`" for name in prune] or ["- None"])
    lines.extend(["", "Top individual features by mean `delta_logloss`:"])
    for _, row in feature_summary.head(12).iterrows():
        lines.append(
            f"- `{row['name']}`: delta_logloss={row['delta_logloss_mean']:.6g}, "
            f"delta_auc={row['delta_auc_mean']:.6g}"
        )
    lines.extend(["", "Grouped importances by mean `delta_logloss`:"])
    for _, row in group_summary.iterrows():
        lines.append(
            f"- `{row['name']}`: delta_logloss={row['delta_logloss_mean']:.6g}, "
            f"delta_auc={row['delta_auc_mean']:.6g}"
        )
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def plot_importance(feature_summary, group_summary, output_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping importance plot because matplotlib failed to import: {exc}")
        return

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)

    features = feature_summary.sort_values("delta_logloss_mean", ascending=True)
    axes[0].barh(features["name"], features["delta_logloss_mean"])
    axes[0].axvline(0.0, color="0.3", lw=0.8)
    axes[0].set_title("Individual feature importance")
    axes[0].set_xlabel("mean increase in logloss after permutation")

    groups = group_summary.sort_values("delta_logloss_mean", ascending=True)
    axes[1].barh(groups["name"], groups["delta_logloss_mean"])
    axes[1].axvline(0.0, color="0.3", lw=0.8)
    axes[1].set_title("Grouped feature importance")
    axes[1].set_xlabel("mean increase in logloss after permutation")

    fig.savefig(os.path.join(output_dir, "feature_importance.png"), dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.pilot_output)), exist_ok=True)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")
    print(f"Features ({len(DEFAULT_SELECTION_FEATURES)}):")
    for name in DEFAULT_SELECTION_FEATURES:
        marker = " [scientific keep]" if name in SCIENTIFIC_KEEP_FEATURES else ""
        print(f"  - {name}{marker}")

    frame, y = load_selection_data(args, DEFAULT_SELECTION_FEATURES)
    train_df, y_train, val_df, y_val, cal_df, y_cal = split_data(
        frame, y, args.seed, args.validation_size, args.calibration_size
    )
    pilot = train_pilot(
        args,
        device,
        train_df,
        y_train,
        val_df,
        y_val,
        cal_df,
        y_cal,
        DEFAULT_SELECTION_FEATURES,
    )
    bundle = SelectionModelBundle(
        pilot.model,
        pilot.preprocessor,
        temperature=pilot.temperature,
        metadata={},
        device=device,
    )

    importance_df, y_importance = choose_importance_sample(
        val_df,
        y_val,
        args.importance_rows,
        args.seed + 1,
    )
    baseline_metrics, raw_importance = run_permutation_importance(
        bundle,
        importance_df,
        y_importance,
        DEFAULT_SELECTION_FEATURES,
        args,
    )
    summary = summarize_importance(raw_importance, args)
    feature_summary = summary[summary["kind"] == "feature"].copy()
    group_summary = summary[summary["kind"] == "group"].copy()

    baseline_path = os.path.join(args.output_dir, "baseline_metrics.csv")
    raw_path = os.path.join(args.output_dir, "permutation_importance_repeats.csv")
    feature_path = os.path.join(args.output_dir, "feature_importance.csv")
    group_path = os.path.join(args.output_dir, "group_importance.csv")
    readme_path = os.path.join(args.output_dir, "README.md")

    pd.DataFrame([baseline_metrics]).to_csv(baseline_path, index=False)
    raw_importance.to_csv(raw_path, index=False)
    feature_summary.to_csv(feature_path, index=False)
    group_summary.to_csv(group_path, index=False)
    write_readme(
        readme_path,
        args,
        baseline_metrics,
        feature_summary,
        group_summary,
        actual_importance_rows=len(importance_df),
    )
    plot_importance(feature_summary, group_summary, args.output_dir)

    metadata = {
        "model_family": "selection_mlp_feature_importance_pilot",
        "selection_name": args.selection_name,
        "target_column": args.target_column,
        "catalogue_path": args.catalogue,
        "features": list(DEFAULT_SELECTION_FEATURES),
        "input_names": pilot.preprocessor.output_names,
        "loss": args.loss,
        "history": pilot.history,
        "best_val_loss": pilot.best_val_loss,
        "val_metrics": pilot.val_metrics,
        "calibration_metrics": pilot.calibration_metrics,
        "importance_baseline_metrics": baseline_metrics,
        "importance_rows": int(len(importance_df)),
        "seed": int(args.seed),
        "max_rows": None if args.max_rows is None else int(args.max_rows),
        "max_read_batches": args.max_read_batches,
    }
    save_selection_model(
        args.pilot_output,
        pilot.model,
        pilot.preprocessor,
        pilot.model_config,
        pilot.temperature,
        metadata=metadata,
    )

    print("\n--- Importance summary ---")
    print("Top individual features:")
    print(
        feature_summary[
            ["name", "delta_logloss_mean", "delta_auc_mean", "prune_candidate", "scientific_keep"]
        ].head(20).to_string(index=False)
    )
    print("\nGrouped features:")
    print(
        group_summary[["name", "delta_logloss_mean", "delta_auc_mean"]]
        .sort_values("delta_logloss_mean", ascending=False)
        .to_string(index=False)
    )
    prune = feature_summary.loc[feature_summary["prune_candidate"], "name"].tolist()
    print("\nPrune candidates:")
    print("  " + (", ".join(prune) if prune else "None"))
    print(f"\nSaved pilot model: {args.pilot_output}")
    print(f"Saved importance outputs under: {args.output_dir}")


if __name__ == "__main__":
    main()
