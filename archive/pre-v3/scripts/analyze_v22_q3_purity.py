"""Compare Eq. 17 true blendedness with shell fluxes on constgold q3.

The primary score is five-fold case-held-out prediction of the direct
``R_flow - R_self_truth`` residual.  Identical fixed-capacity histogram
gradient boosting regressors are used for all feature representations.  The
second target, ``R_self_truth``, checks whether conclusions depend on using the
deployed flow in the target.  Constgold remains evaluation-only.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

from scripts.diag_v22_q3_flow_trueflux import case_stats, seeded_stats


BASELINE = [
    "measured_mag_auto", "measured_flux_radius", "sersic_n_input_p",
    "e1_input_rot0_p", "e2_input_rot0_p",
    "nbr_flux_near", "nbr_flux_far", "nbr_flux_max",
]
SHELLS = [
    "logflux_abs_near_0_1", "logflux_abs_mid_1_3", "logflux_abs_far_3_10",
]
PURITY = ["true_blendedness_eq17"]
RBLEND = ["R_blend_model"]
FEATURE_SETS = {
    "rblend_only": RBLEND,
    "purity_only": PURITY,
    "shells_only": SHELLS,
    "v22_inputs": BASELINE,
    "v22_plus_rblend": [*BASELINE, *RBLEND],
    "v22_plus_purity": [*BASELINE, *PURITY],
    "v22_plus_shells": [*BASELINE, *SHELLS],
    "v22_plus_purity_shells": [*BASELINE, *PURITY, *SHELLS],
}
TARGETS = ["flow_minus_self_truth", "R_self_truth"]


def model_pipeline(random_state: int = 7301) -> Pipeline:
    return Pipeline([("regressor", HistGradientBoostingRegressor(
        loss="squared_error", learning_rate=0.05, max_iter=160,
        max_leaf_nodes=15, min_samples_leaf=300, l2_regularization=1.0,
        early_stopping=False, random_state=random_state,
    ))])


def cross_fitted_predictions(frame: pd.DataFrame, target: str,
                             feature_sets: dict[str, list[str]]) -> pd.DataFrame:
    groups = frame["case"].to_numpy(np.int64)
    y = frame[target].to_numpy(float)
    predictions = pd.DataFrame(index=frame.index)
    predictions["constant"] = np.nan
    for name in feature_sets:
        predictions[name] = np.nan
    splitter = GroupKFold(n_splits=5)
    fold_by_case = {}
    for fold, (train, test) in enumerate(splitter.split(frame, y, groups=groups)):
        train_cases = set(groups[train].tolist())
        test_cases = set(groups[test].tolist())
        if train_cases & test_cases:
            raise RuntimeError("case leakage across GroupKFold split")
        predictions.iloc[test, predictions.columns.get_loc("constant")] = float(y[train].mean())
        for case in test_cases:
            fold_by_case[int(case)] = fold
        for name, columns in feature_sets.items():
            model = model_pipeline()
            model.fit(frame.iloc[train][columns].to_numpy(float), y[train])
            predictions.iloc[test, predictions.columns.get_loc(name)] = model.predict(
                frame.iloc[test][columns].to_numpy(float)
            )
    if predictions.isna().any().any():
        raise RuntimeError("cross-fitted prediction coverage is incomplete")
    predictions.insert(0, "fold", frame["case"].map(fold_by_case).to_numpy(np.int8))
    return predictions


def metric_summary(frame: pd.DataFrame, target: str, predictions: pd.DataFrame) -> dict:
    y = frame[target].to_numpy(float)
    case = frame["case"].to_numpy(np.int64)
    metrics = {}
    per_case_mse = {}
    per_case_mean_error = {}
    for name in predictions.columns:
        if name == "fold":
            continue
        pred = predictions[name].to_numpy(float)
        residual = y - pred
        table = pd.DataFrame({"case": case, "sq": residual ** 2, "residual": residual})
        mse_case = table.groupby("case", sort=True)["sq"].mean()
        mean_error_case = table.groupby("case", sort=True)["residual"].mean()
        per_case_mse[name] = mse_case
        per_case_mean_error[name] = mean_error_case
        metrics[name] = {
            "mse": float(np.mean(residual ** 2)),
            "rmse": float(np.sqrt(np.mean(residual ** 2))),
            "mae": float(np.mean(np.abs(residual))),
            "pearson_prediction_truth": float(np.corrcoef(pred, y)[0, 1]),
            "prediction_std": float(np.std(pred, ddof=1)),
            "case_mean_error_rmse": float(np.sqrt(np.mean(mean_error_case.to_numpy() ** 2))),
            "case_mse_values": mse_case.to_numpy(float).tolist(),
            "case_mean_error_values": mean_error_case.to_numpy(float).tolist(),
        }
    comparisons = {}
    pairs = [
        ("purity_only", "constant"), ("shells_only", "constant"),
        ("rblend_only", "constant"), ("v22_inputs", "constant"),
        ("v22_plus_rblend", "v22_inputs"), ("v22_plus_purity", "v22_inputs"),
        ("v22_plus_shells", "v22_inputs"),
        ("v22_plus_purity_shells", "v22_inputs"),
        ("shells_only", "purity_only"),
        ("v22_plus_shells", "v22_plus_purity"),
        ("v22_plus_purity_shells", "v22_plus_shells"),
        ("v22_plus_purity_shells", "v22_plus_purity"),
    ]
    for candidate, reference in pairs:
        if candidate not in per_case_mse or reference not in per_case_mse:
            continue
        values = (per_case_mse[reference] - per_case_mse[candidate]).to_numpy(float)
        reference_mse = metrics[reference]["mse"]
        comparisons[f"{candidate}_versus_{reference}"] = {
            "mean_case_mse_reduction": float(values.mean()),
            "case_sem": float(values.std(ddof=1) / np.sqrt(len(values))),
            "percent_of_reference_mse": float(100.0 * values.mean() / reference_mse),
            "percent_case_sem": float(100.0 * values.std(ddof=1) / np.sqrt(len(values))
                                      / reference_mse),
            "n_cases": int(len(values)), "case_values": values.tolist(),
        }
    return {"metrics": metrics, "comparisons": comparisons}


def purity_profile(frame: pd.DataFrame, flow_columns: list[str]) -> dict:
    blendedness = frame["true_blendedness_eq17"].to_numpy(float)
    edges = np.quantile(blendedness, [0.2, 0.4, 0.6, 0.8])
    if len(np.unique(edges)) != 4:
        raise RuntimeError("true blendedness quintile edges are degenerate")
    classes = np.digitize(blendedness, edges, right=True)
    bins = []
    for index in range(5):
        subset = frame.loc[classes == index]
        residual = subset[flow_columns].subtract(
            subset["R_self_truth"].to_numpy(float), axis=0,
        )
        bins.append({
            "label": f"Q{index + 1}", "n_rows": int(len(subset)),
            "n_cases": int(subset["case"].nunique()),
            "median_true_blendedness": float(subset["true_blendedness_eq17"].median()),
            "median_purity": float(subset["purity_eq17"].median()),
            "flow_minus_self_truth": seeded_stats(
                pd.concat([subset[["case"]], residual], axis=1), flow_columns,
            ),
            "R_self_truth": case_stats(subset, "R_self_truth"),
            "R_self_null": case_stats(subset, "R_self_null"),
        })
    return {"edges": edges.tolist(), "bins": bins}


def correlation_summary(frame: pd.DataFrame) -> dict:
    columns = ["true_blendedness_eq17", "R_blend_model", *SHELLS, *BASELINE[-3:]]
    corr = frame[columns].corr(method="spearman")
    return {
        row: {column: float(corr.loc[row, column]) for column in columns}
        for row in columns
    }


def write_cv_csv(result: dict, path: Path) -> None:
    rows = []
    for target, target_result in result["cross_validation"].items():
        for comparison, stats in target_result["comparisons"].items():
            rows.append({
                "target": target, "comparison": comparison,
                "mse_reduction": stats["mean_case_mse_reduction"],
                "mse_reduction_case_sem": stats["case_sem"],
                "percent_of_reference_mse": stats["percent_of_reference_mse"],
                "percent_case_sem": stats["percent_case_sem"],
            })
    pd.DataFrame(rows).to_csv(path, index=False)


def make_plot(result: dict, prefix: Path) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 8.5,
        "axes.labelsize": 9, "axes.titlesize": 9.5,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.45))
    profile = result["purity_profile"]["bins"]
    x = np.arange(5)
    residual = [item["flow_minus_self_truth"] for item in profile]
    axes[0].errorbar(
        x, 100 * np.asarray([item["mean"] for item in residual]),
        yerr=100 * np.asarray([item["quadrature_sem"] for item in residual]),
        color="#D55E00", marker="o", linewidth=1.3, capsize=2.5,
        label=r"$R_{\rm flow}-R_{\rm self,true}$",
    )
    null = [item["R_self_null"] for item in profile]
    axes[0].errorbar(
        x, 100 * np.asarray([item["mean"] for item in null]),
        yerr=100 * np.asarray([item["case_sem"] for item in null]),
        color="0.45", marker="D", linestyle=":", linewidth=1.0, capsize=2,
        label=r"45$^\circ$ self null",
    )
    axes[0].axhline(0, color="0.25", linestyle="--", linewidth=0.8)
    axes[0].set_xticks(x, [item["label"] for item in profile])
    axes[0].set_xlabel(r"True blendedness $1-\rho$ quintile")
    axes[0].set_ylabel("Response residual (points)")
    axes[0].set_title("Marginal purity profile")
    axes[0].legend(frameon=False, loc="best")

    feature_order = ["rblend_only", "purity_only", "shells_only"]
    feature_labels = [r"$R_{\rm blend}$", "Eq. 17\npurity", "Three\nshells"]
    incremental_order = [
        "v22_plus_rblend", "v22_plus_purity", "v22_plus_shells",
        "v22_plus_purity_shells",
    ]
    incremental_labels = [r"+$R_{\rm blend}$", "+purity", "+shells", "+both"]
    colors = {"flow_minus_self_truth": "#0072B2", "R_self_truth": "#E69F00"}
    target_labels = {
        "flow_minus_self_truth": "Flow residual", "R_self_truth": "True self response",
    }
    for ax, order, labels, reference, title in (
        (axes[1], feature_order, feature_labels, "constant", "Information alone"),
        (axes[2], incremental_order, incremental_labels, "v22_inputs", "Added beyond V2.2 inputs"),
    ):
        positions = np.arange(len(order))
        width = 0.34
        for offset, target in zip((-width / 2, width / 2), TARGETS):
            cv = result["cross_validation"][target]["comparisons"]
            values, errors = [], []
            for name in order:
                stats = cv[f"{name}_versus_{reference}"]
                values.append(stats["percent_of_reference_mse"])
                errors.append(stats["percent_case_sem"])
            ax.bar(
                positions + offset, values, width=width, yerr=errors,
                color=colors[target], alpha=0.88, capsize=2.5,
                label=target_labels[target],
            )
        ax.axhline(0, color="0.25", linewidth=0.8)
        ax.set_xticks(positions, labels)
        ax.set_ylabel("Case-CV MSE reduction (%)")
        ax.set_title(title)
    axes[1].legend(frameon=False, loc="best")
    for panel, ax in zip("ABC", axes):
        ax.text(-0.16, 1.04, panel, transform=ax.transAxes, fontweight="bold", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "How informative is Eq. 17 true blendedness for the V2.2 q3 flow failure?",
        fontsize=11, y=0.995,
    )
    fig.text(
        0.5, 0.008,
        f"{result['n_rows']:,} anchors, {result['n_cases']} cases. Five-fold case-held-out "
        "gradient boosting; fixed model capacity for every representation. Error bars: case SEM. "
        "Positive MSE reduction means more held-out information.",
        ha="center", va="bottom", fontsize=7.4, color="0.30",
    )
    fig.tight_layout(rect=(0, 0.075, 1, 0.95), w_pad=1.3)
    fig.savefig(prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", required=True)
    parser.add_argument("--flow", required=True)
    parser.add_argument("--purity-glob", required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--prediction-output", type=Path)
    args = parser.parse_args()
    outputs = [args.output_prefix.with_suffix(suffix) for suffix in (".json", ".csv", ".pdf", ".png")]
    if args.prediction_output:
        outputs.append(args.prediction_output)
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite outputs: {existing}")

    truth = pd.read_feather(args.truth)
    flow = pd.read_feather(args.flow)
    purity_paths = sorted(glob.glob(args.purity_glob))
    if len(purity_paths) != 10:
        raise RuntimeError(f"expected 10 purity shards, found {purity_paths}")
    purity = pd.concat([pd.read_feather(path) for path in purity_paths], ignore_index=True)
    keys = ["case", "input_index"]
    for name, table in (("truth", truth), ("flow", flow), ("purity", purity)):
        if table.duplicated(keys).any():
            raise RuntimeError(f"duplicate {name} key")
    frame = truth.merge(flow, on=keys, validate="one_to_one").merge(
        purity, on=keys, validate="one_to_one",
    )
    if not (len(frame) == len(truth) == len(flow) == len(purity)):
        raise RuntimeError(
            f"key mismatch truth={len(truth)} flow={len(flow)} purity={len(purity)} both={len(frame)}"
        )
    flow_columns = sorted(
        (column for column in frame if column.startswith("R_flow_s")),
        key=lambda column: int(re.search(r"(\d+)$", column).group(1)),
    )
    if len(flow_columns) != 16 or frame["case"].nunique() != 100:
        raise RuntimeError("expected 16 flow seeds and 100 cases")
    needed = list(dict.fromkeys([
        *BASELINE, *SHELLS, *PURITY, *RBLEND, "purity_eq17",
        "R_self_truth", "R_self_null", *flow_columns,
    ]))
    if not np.isfinite(frame[needed].to_numpy(float)).all():
        raise RuntimeError("non-finite analysis value")
    frame["flow_minus_self_truth"] = frame[flow_columns].mean(axis=1) - frame["R_self_truth"]

    result = {
        "n_rows": len(frame), "n_cases": int(frame["case"].nunique()),
        "n_flow_seeds": len(flow_columns), "evaluation_only": True,
        "purity_definition": (
            "Nourbakhsh et al. 2022 Eq. 17 on unsheared PSF-convolved noiseless SBSI profiles; "
            "anchor pixels >= 0.05 sky RMS"
        ),
        "feature_sets": FEATURE_SETS,
        "cross_validation": {},
        "purity_profile": purity_profile(frame, flow_columns),
        "spearman": correlation_summary(frame),
        "purity_qa": {
            "mean": float(frame["purity_eq17"].mean()),
            "median": float(frame["purity_eq17"].median()),
            "minimum": float(frame["purity_eq17"].min()),
            "maximum": float(frame["purity_eq17"].max()),
            "edge_mask_count": int(frame["mask_touches_stamp_edge"].sum()),
        },
    }
    prediction_table = frame[keys].copy()
    for target in TARGETS:
        predictions = cross_fitted_predictions(frame, target, FEATURE_SETS)
        result["cross_validation"][target] = metric_summary(frame, target, predictions)
        for column in predictions:
            prediction_table[f"{target}__{column}"] = predictions[column].to_numpy()
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    with args.output_prefix.with_suffix(".json").open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    write_cv_csv(result, args.output_prefix.with_suffix(".csv"))
    make_plot(result, args.output_prefix)
    if args.prediction_output:
        args.prediction_output.parent.mkdir(parents=True, exist_ok=True)
        prediction_table.to_feather(args.prediction_output)
    for target in TARGETS:
        print(f"target: {target}")
        comparisons = result["cross_validation"][target]["comparisons"]
        for name in (
            "purity_only_versus_constant", "shells_only_versus_constant",
            "v22_plus_purity_versus_v22_inputs", "v22_plus_shells_versus_v22_inputs",
            "v22_plus_purity_shells_versus_v22_inputs",
            "v22_plus_shells_versus_v22_plus_purity",
        ):
            stats = comparisons[name]
            print(
                f"  {name}: {stats['percent_of_reference_mse']:+.5f} +/- "
                f"{stats['percent_case_sem']:.5f}% MSE"
            )
    print(f"wrote {', '.join(str(path) for path in outputs)}")
    print("V22_Q3_PURITY_ANALYSIS_DONE", flush=True)


if __name__ == "__main__":
    main()
