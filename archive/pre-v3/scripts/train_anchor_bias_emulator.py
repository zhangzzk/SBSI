"""Train and validate a boosted-tree emulator of the V2.2 per-anchor bias.

This is an anchor-only diagnostic.  Cases 400--599 train candidate models,
600--699 tune capacity and select plotted coordinates, and 700--899 are opened
only for final scoring and conditional-bias curves/maps.  Case is the
uncertainty unit and every training case receives equal total weight.

The target is ``truth - V2.2`` (positive means underprediction).  Nothing in
this script changes V2.2 or enters a shear estimator.  The saved model is marked
diagnostic-only and constgold is never read.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from string import ascii_uppercase

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline


KEY = ["case", "input_index"]
TARGET = "bias_truth_minus_model"
POPULATIONS = {
    "all": None,
    "carrier_ratio_gt5_positive": "carrier_ratio_gt5_positive",
}
CORE_PLOT_FEATURES = [
    "scene_prediction", "abs_scene_prediction", "dominant_response",
    "top_abs_fraction", "log1p_dominant_to_runner_up_abs_response",
    "log10_flux_ratio", "log10_size_ratio", "log10_overlap_scale",
    "log10_surface_brightness_ratio",
]
CANDIDATES = {
    "conservative": {
        "max_iter": 160, "max_leaf_nodes": 15,
        "min_samples_leaf": 2000, "l2_regularization": 30.0,
    },
    "medium": {
        "max_iter": 240, "max_leaf_nodes": 31,
        "min_samples_leaf": 1000, "l2_regularization": 10.0,
    },
    "flexible": {
        "max_iter": 320, "max_leaf_nodes": 63,
        "min_samples_leaf": 500, "l2_regularization": 10.0,
    },
}


def finite_stat(values: np.ndarray, hypothesis_test: bool = True) -> dict:
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        raise ValueError("need at least two finite case values")
    sd = float(values.std(ddof=1))
    result = {
        "mean": float(values.mean()),
        "case_sd": sd,
        "case_sem": sd / np.sqrt(values.size),
        "n_cases": int(values.size),
    }
    if hypothesis_test:
        test = stats.ttest_1samp(values, 0.0)
        result.update({"t": float(test.statistic), "p": float(test.pvalue)})
    return result


def case_balanced_weights(cases: np.ndarray) -> np.ndarray:
    cases = np.asarray(cases, np.int64)
    unique, inverse, counts = np.unique(
        cases, return_inverse=True, return_counts=True
    )
    if unique.size < 2:
        raise ValueError("need at least two cases")
    weights = 1.0 / counts[inverse].astype(float)
    weights *= len(weights) / weights.sum()
    return weights


def case_balanced_mean(values: np.ndarray, cases: np.ndarray) -> float:
    frame = pd.DataFrame({"case": cases, "value": values})
    return float(frame.groupby("case", sort=True).value.mean().mean())


def make_model(parameters: dict, random_state: int) -> Pipeline:
    model = HistGradientBoostingRegressor(
        loss="squared_error", learning_rate=0.05, max_bins=255,
        early_stopping=False, random_state=random_state, **parameters,
    )
    return Pipeline([("regressor", model)])


def fit_model(frame: pd.DataFrame, features: list[str], parameters: dict,
              random_state: int) -> Pipeline:
    model = make_model(parameters, random_state)
    x = frame[features].to_numpy(np.float32)
    y = frame[TARGET].to_numpy(float)
    weights = case_balanced_weights(frame.case.to_numpy(np.int64))
    model.fit(x, y, regressor__sample_weight=weights)
    return model


def metric_summary(frame: pd.DataFrame, prediction: np.ndarray,
                   constant: float) -> dict:
    y = frame[TARGET].to_numpy(float)
    prediction = np.asarray(prediction, float)
    if len(prediction) != len(frame) or not np.isfinite(prediction).all():
        raise ValueError("invalid prediction vector")
    work = pd.DataFrame({
        "case": frame.case.to_numpy(np.int64),
        "target": y,
        "prediction": prediction,
    })
    work["model_sq"] = (work.target - work.prediction) ** 2
    work["constant_sq"] = (work.target - constant) ** 2
    work["model_abs"] = np.abs(work.target - work.prediction)
    work["constant_abs"] = np.abs(work.target - constant)
    by_case = work.groupby("case", sort=True).agg(
        model_mse=("model_sq", "mean"),
        constant_mse=("constant_sq", "mean"),
        model_mae=("model_abs", "mean"),
        constant_mae=("constant_abs", "mean"),
        target_mean=("target", "mean"),
        prediction_mean=("prediction", "mean"),
    )
    reduction = by_case.constant_mse - by_case.model_mse
    mean_error = by_case.target_mean - by_case.prediction_mean
    model_mse = float(by_case.model_mse.mean())
    constant_mse = float(by_case.constant_mse.mean())
    corr = float(np.corrcoef(by_case.target_mean, by_case.prediction_mean)[0, 1])
    return {
        "n_rows": int(len(frame)),
        "n_cases": int(frame.case.nunique()),
        "case_balanced_mse": model_mse,
        "constant_case_balanced_mse": constant_mse,
        "mse_skill": float(1.0 - model_mse / constant_mse),
        "case_balanced_mae": float(by_case.model_mae.mean()),
        "constant_case_balanced_mae": float(by_case.constant_mae.mean()),
        "mse_reduction_model_minus_constant": finite_stat(
            reduction.to_numpy(float)
        ),
        "global_target": finite_stat(by_case.target_mean.to_numpy(float)),
        "global_prediction": finite_stat(by_case.prediction_mean.to_numpy(float)),
        "global_target_minus_prediction": finite_stat(mean_error.to_numpy(float)),
        "case_mean_rmse": float(np.sqrt(np.mean(mean_error.to_numpy(float) ** 2))),
        "case_mean_prediction_truth_pearson": corr,
        "row_prediction_sd": float(prediction.std(ddof=1)),
    }


def prediction_calibration(frame: pd.DataFrame, prediction: np.ndarray,
                           edges: np.ndarray) -> dict:
    prediction = np.asarray(prediction, float)
    codes = np.digitize(prediction, edges[1:-1], right=False)
    work = frame[["case", TARGET]].copy()
    work["prediction"] = prediction
    work["bin"] = codes
    bins = []
    for index in range(len(edges) - 1):
        local = work.loc[work.bin == index]
        if local.empty:
            continue
        by_case = local.groupby("case", sort=True)[[TARGET, "prediction"]].mean()
        difference = by_case[TARGET] - by_case.prediction
        bins.append({
            "bin": int(index), "n_rows": int(len(local)),
            "n_cases": int(len(by_case)),
            "prediction": finite_stat(by_case.prediction.to_numpy(float)),
            "target": finite_stat(by_case[TARGET].to_numpy(float)),
            "target_minus_prediction": finite_stat(difference.to_numpy(float)),
        })
    if len(bins) < 5:
        raise RuntimeError("prediction calibration has too few occupied bins")
    low = work.loc[work.bin == bins[0]["bin"]].groupby("case")[
        [TARGET, "prediction"]
    ].mean()
    high = work.loc[work.bin == bins[-1]["bin"]].groupby("case")[
        [TARGET, "prediction"]
    ].mean()
    paired = low.join(high, lsuffix="_low", rsuffix="_high", how="inner")
    observed_span = paired[f"{TARGET}_high"] - paired[f"{TARGET}_low"]
    predicted_span = paired.prediction_high - paired.prediction_low
    predicted_means = np.asarray([item["prediction"]["mean"] for item in bins])
    target_means = np.asarray([item["target"]["mean"] for item in bins])
    slope, intercept = np.polyfit(predicted_means, target_means, 1)
    return {
        "edges": [None if not np.isfinite(value) else float(value) for value in edges],
        "bins": bins,
        "observed_high_minus_low": finite_stat(observed_span.to_numpy(float)),
        "predicted_high_minus_low": finite_stat(predicted_span.to_numpy(float)),
        "bin_mean_spearman": float(stats.spearmanr(predicted_means, target_means).statistic),
        "bin_mean_calibration_slope": float(slope),
        "bin_mean_calibration_intercept": float(intercept),
    }


def within_case_permutation(cases: np.ndarray,
                            rng: np.random.Generator) -> np.ndarray:
    cases = np.asarray(cases, np.int64)
    permutation = np.arange(len(cases), dtype=np.int64)
    for case in np.unique(cases):
        local = np.flatnonzero(cases == case)
        permutation[local] = rng.permutation(local)
    return permutation


def permutation_importance(model: Pipeline, frame: pd.DataFrame,
                           features: list[str], random_state: int) -> list[dict]:
    x = frame[features].to_numpy(np.float32)
    y = frame[TARGET].to_numpy(float)
    cases = frame.case.to_numpy(np.int64)
    base_prediction = model.predict(x)
    base = pd.DataFrame({
        "case": cases, "sq": (y - base_prediction) ** 2,
    }).groupby("case", sort=True).sq.mean()
    rng = np.random.default_rng(random_state)
    work = x.copy()
    result = []
    for index, feature in enumerate(features):
        permutation = within_case_permutation(cases, rng)
        original = work[:, index].copy()
        work[:, index] = original[permutation]
        prediction = model.predict(work)
        work[:, index] = original
        permuted = pd.DataFrame({
            "case": cases, "sq": (y - prediction) ** 2,
        }).groupby("case", sort=True).sq.mean()
        increase = permuted - base
        item = finite_stat(increase.to_numpy(float))
        item.update({
            "feature": feature,
            "percent_of_baseline_mse": float(100.0 * item["mean"] / base.mean()),
        })
        result.append(item)
        print(
            f"importance {index + 1:02d}/{len(features)} {feature}: "
            f"{item['percent_of_baseline_mse']:+.4f}%", flush=True,
        )
    return sorted(result, key=lambda item: item["mean"], reverse=True)


def quantile_edges(values: np.ndarray, n_bins: int) -> np.ndarray:
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        raise RuntimeError("feature has fewer than two finite values")
    inner = np.unique(np.quantile(values, np.arange(1, n_bins) / n_bins))
    if inner.size < min(3, n_bins - 1):
        unique = np.unique(values)
        if unique.size < 2:
            raise RuntimeError("feature is constant")
        if unique.size <= n_bins:
            inner = (unique[:-1] + unique[1:]) / 2.0
        else:
            # A large point mass (usually zero response in a sparse distance
            # shell) can collapse every row-weighted quantile while leaving a
            # continuous nonzero tail.  Keep the fallback bounded by n_bins;
            # using every unique tail value would silently create tens of
            # thousands of nominal "12-bin" curve cells.
            inner = np.unique(np.quantile(
                unique, np.arange(1, n_bins) / n_bins
            ))
    return np.r_[-np.inf, inner, np.inf]


def conditional_curve(development: pd.DataFrame, test: pd.DataFrame,
                      prediction: np.ndarray, feature: str,
                      population: str, n_bins: int) -> list[dict]:
    development = development.loc[np.isfinite(development[feature])]
    if development[feature].nunique() < 2:
        return []
    finite = np.isfinite(test[feature].to_numpy(float))
    test = test.loc[finite].copy()
    prediction = np.asarray(prediction, float)[finite]
    edges = quantile_edges(development[feature].to_numpy(float), n_bins)
    codes = np.digitize(test[feature].to_numpy(float), edges[1:-1])
    work = test[["case", TARGET, feature]].copy()
    work["prediction"] = np.asarray(prediction, float)
    work["bin"] = codes
    rows = []
    for index in range(len(edges) - 1):
        local = work.loc[work.bin == index]
        # Discrete/sparse coordinates can leave an extreme development-defined
        # bin occupied by only one final-test case.  Such a bin has no case SEM,
        # so omit it from the reported curve rather than treating one case as an
        # uncertainty-bearing conditional estimate.
        if local.empty or local.case.nunique() < 2:
            continue
        by_case = local.groupby("case", sort=True)[[TARGET, "prediction"]].mean()
        error = by_case[TARGET] - by_case.prediction
        target = finite_stat(by_case[TARGET].to_numpy(float), hypothesis_test=False)
        predicted = finite_stat(
            by_case.prediction.to_numpy(float), hypothesis_test=False
        )
        calibration = finite_stat(error.to_numpy(float), hypothesis_test=False)
        rows.append({
            "population": population, "feature": feature, "bin": int(index),
            "lower": float(edges[index]), "upper": float(edges[index + 1]),
            "feature_median": float(local[feature].median()),
            "feature_mean": float(local[feature].mean()),
            "n_rows": int(len(local)), "n_cases": int(len(by_case)),
            "target_mean": target["mean"], "target_case_sem": target["case_sem"],
            "prediction_mean": predicted["mean"],
            "prediction_case_sem": predicted["case_sem"],
            "target_minus_prediction_mean": calibration["mean"],
            "target_minus_prediction_case_sem": calibration["case_sem"],
        })
    return rows


def conditional_map(development: pd.DataFrame, test: pd.DataFrame,
                    prediction: np.ndarray, x_feature: str, y_feature: str,
                    population: str, n_bins: int) -> list[dict]:
    development = development.loc[
        np.isfinite(development[x_feature]) & np.isfinite(development[y_feature])
    ]
    finite = (
        np.isfinite(test[x_feature].to_numpy(float))
        & np.isfinite(test[y_feature].to_numpy(float))
    )
    test = test.loc[finite].copy()
    prediction = np.asarray(prediction, float)[finite]
    x_edges = quantile_edges(development[x_feature].to_numpy(float), n_bins)
    y_edges = quantile_edges(development[y_feature].to_numpy(float), n_bins)
    x_code = np.digitize(test[x_feature].to_numpy(float), x_edges[1:-1])
    y_code = np.digitize(test[y_feature].to_numpy(float), y_edges[1:-1])
    work = test[["case", TARGET, x_feature, y_feature]].copy()
    work["prediction"] = np.asarray(prediction, float)
    work["x_bin"] = x_code
    work["y_bin"] = y_code
    rows = []
    for xb in range(len(x_edges) - 1):
        for yb in range(len(y_edges) - 1):
            local = work.loc[(work.x_bin == xb) & (work.y_bin == yb)]
            if local.empty or local.case.nunique() < 2:
                continue
            by_case = local.groupby("case", sort=True)[[TARGET, "prediction"]].mean()
            error = by_case[TARGET] - by_case.prediction
            target = finite_stat(
                by_case[TARGET].to_numpy(float), hypothesis_test=False
            )
            predicted = finite_stat(
                by_case.prediction.to_numpy(float), hypothesis_test=False
            )
            calibration = finite_stat(
                error.to_numpy(float), hypothesis_test=False
            )
            rows.append({
                "population": population, "x_feature": x_feature,
                "y_feature": y_feature, "x_bin": int(xb), "y_bin": int(yb),
                "x_lower": float(x_edges[xb]), "x_upper": float(x_edges[xb + 1]),
                "y_lower": float(y_edges[yb]), "y_upper": float(y_edges[yb + 1]),
                "x_median": float(local[x_feature].median()),
                "y_median": float(local[y_feature].median()),
                "n_rows": int(len(local)), "n_cases": int(len(by_case)),
                "target_mean": target["mean"], "target_case_sem": target["case_sem"],
                "prediction_mean": predicted["mean"],
                "prediction_case_sem": predicted["case_sem"],
                "target_minus_prediction_mean": calibration["mean"],
                "target_minus_prediction_case_sem": calibration["case_sem"],
            })
    return rows


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 8.0,
        "axes.labelsize": 8.5, "axes.titlesize": 8.5,
        "xtick.labelsize": 7.0, "ytick.labelsize": 7.0,
        "legend.fontsize": 7.5, "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.facecolor": "white",
    })


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_calibration(calibration: dict, metrics: dict, stem: str) -> None:
    configure_style()
    bins = calibration["bins"]
    x = np.arange(1, len(bins) + 1)
    observed = np.asarray([item["target"]["mean"] for item in bins])
    observed_sem = np.asarray([item["target"]["case_sem"] for item in bins])
    predicted = np.asarray([item["prediction"]["mean"] for item in bins])
    predicted_sem = np.asarray([item["prediction"]["case_sem"] for item in bins])
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), constrained_layout=True)
    axes[0].errorbar(
        x, observed, yerr=observed_sem, color="#000000", marker="o",
        linewidth=1.1, capsize=2.0, label="Held-out residual",
    )
    axes[0].errorbar(
        x, predicted, yerr=predicted_sem, color="#0072B2", marker="s",
        linestyle="--", linewidth=1.1, capsize=2.0, label="Bias emulator",
    )
    axes[0].axhline(0, color="0.55", linewidth=0.7, linestyle=":")
    axes[0].set_xlabel("Predicted-bias decile")
    axes[0].set_ylabel("truth - V2.2 response")
    axes[0].set_title("Held-out conditional calibration")
    axes[0].legend(frameon=False)
    axes[1].errorbar(
        predicted, observed, yerr=observed_sem, xerr=predicted_sem,
        color="#0072B2", marker="o", linestyle="none", capsize=2.0,
    )
    bound = float(max(np.max(np.abs(predicted)), np.max(np.abs(observed))) * 1.15)
    axes[1].plot([-bound, bound], [-bound, bound], color="0.35", linestyle="--",
                 linewidth=0.9)
    axes[1].set_xlim(-bound, bound)
    axes[1].set_ylim(-bound, bound)
    axes[1].set_xlabel("Bias-emulator prediction")
    axes[1].set_ylabel("Held-out residual")
    axes[1].set_title(
        f"slope={calibration['bin_mean_calibration_slope']:.2f}, "
        f"rho={calibration['bin_mean_spearman']:.2f}\n"
        f"row-MSE skill={metrics['mse_skill']:.3%}"
    )
    for index, axis in enumerate(axes):
        axis.spines[["top", "right"]].set_visible(False)
        axis.text(-0.16, 1.05, ascii_uppercase[index], transform=axis.transAxes,
                  fontweight="bold", fontsize=10, va="top")
    save_figure(fig, stem)


def plot_importance(importance: list[dict], physical: set[str], labels: dict,
                    stem: str, n_show: int = 20) -> None:
    configure_style()
    shown = list(reversed(importance[:n_show]))
    values = 100.0 * np.asarray([item["mean"] for item in shown])
    errors = 100.0 * np.asarray([item["case_sem"] for item in shown])
    names = [labels.get(item["feature"], item["feature"]) for item in shown]
    colors = ["#009E73" if item["feature"] in physical else "#0072B2"
              for item in shown]
    fig, ax = plt.subplots(figsize=(6.8, 5.2), constrained_layout=True)
    y = np.arange(len(shown))
    ax.barh(y, values, xerr=errors, color=colors, alpha=0.85,
            error_kw={"linewidth": 0.7, "capsize": 2})
    ax.axvline(0, color="0.35", linewidth=0.7)
    ax.set_yticks(y, names)
    ax.set_xlabel("Increase in held-out MSE after within-case permutation (x100)")
    ax.set_title("Validation-selected feature importance")
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(0.99, 0.02, "green: physical   blue: response structure",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7)
    save_figure(fig, stem)


def plot_curves(curves: pd.DataFrame, features: list[str], labels: dict,
                population: str, stem: str) -> None:
    configure_style()
    local_all = curves.loc[curves.population == population]
    ncols = 4
    nrows = int(np.ceil(len(features) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(11.0, 2.45 * nrows), constrained_layout=True,
        squeeze=False,
    )
    bound_values = []
    for feature in features:
        local = local_all.loc[local_all.feature == feature]
        bound_values.extend(
            np.abs(local.target_mean).to_list()
            + np.abs(local.prediction_mean).to_list()
            + np.abs(local.target_mean + local.target_case_sem).to_list()
        )
    bound = float(np.nanpercentile(bound_values, 98) * 1.15)
    bound = max(bound, 0.01)
    for panel, (axis, feature) in enumerate(zip(axes.flat, features)):
        local = local_all.loc[local_all.feature == feature].sort_values("bin")
        x = local.feature_median.to_numpy(float)
        axis.errorbar(
            x, local.target_mean, yerr=local.target_case_sem,
            color="#000000", marker="o", linewidth=1.0, markersize=3.3,
            capsize=1.8, label="Held-out residual",
        )
        axis.plot(
            x, local.prediction_mean, color="#0072B2", marker="s",
            markersize=3.0, linestyle="--", linewidth=1.0,
            label="Bias emulator",
        )
        axis.axhline(0, color="0.60", linestyle=":", linewidth=0.6)
        axis.set_ylim(-bound, bound)
        axis.set_xlabel(labels.get(feature, feature))
        axis.set_ylabel("truth - V2.2")
        axis.spines[["top", "right"]].set_visible(False)
        axis.text(-0.14, 1.05, ascii_uppercase[panel], transform=axis.transAxes,
                  fontweight="bold", fontsize=9, va="top")
        if panel == 0:
            axis.legend(frameon=False, loc="best")
    for axis in axes.flat[len(features):]:
        axis.set_visible(False)
    fig.suptitle(
        "Held-out one-dimensional conditional bias — "
        + ("all anchors" if population == "all" else "frozen dominant carrier"),
        fontsize=10,
    )
    save_figure(fig, stem)


def plot_maps(maps: pd.DataFrame, pairs: list[tuple[str, str]], labels: dict,
              population: str, stem: str) -> None:
    configure_style()
    local_all = maps.loc[maps.population == population]
    values = np.r_[
        local_all.target_mean.to_numpy(float),
        local_all.prediction_mean.to_numpy(float),
    ]
    bound = float(np.nanpercentile(np.abs(values), 98))
    bound = max(bound, 0.01)
    norm = TwoSlopeNorm(vmin=-bound, vcenter=0.0, vmax=bound)
    nrows = int(np.ceil(len(pairs) / 2))
    fig, axes = plt.subplots(
        nrows, 4, figsize=(12.0, 3.0 * nrows), constrained_layout=True,
        squeeze=False,
    )
    image = None
    for pair_index, (x_feature, y_feature) in enumerate(pairs):
        row = pair_index // 2
        offset = 2 * (pair_index % 2)
        local = local_all.loc[
            (local_all.x_feature == x_feature)
            & (local_all.y_feature == y_feature)
        ]
        nx = int(local.x_bin.max()) + 1
        ny = int(local.y_bin.max()) + 1
        for col, (column, title) in enumerate((
            ("prediction_mean", "Bias emulator"),
            ("target_mean", "Held-out residual"),
        )):
            matrix = np.full((ny, nx), np.nan)
            for item in local.itertuples():
                matrix[int(item.y_bin), int(item.x_bin)] = getattr(item, column)
            axis = axes[row, offset + col]
            image = axis.imshow(
                matrix, origin="lower", aspect="auto", cmap="RdBu_r", norm=norm,
                interpolation="nearest",
            )
            axis.set_title(title)
            axis.set_xlabel(labels.get(x_feature, x_feature))
            if col == 0:
                axis.set_ylabel(labels.get(y_feature, y_feature))
            else:
                axis.set_yticklabels([])
            xticks = sorted(set([0, max(0, nx // 2), nx - 1]))
            yticks = sorted(set([0, max(0, ny // 2), ny - 1]))
            axis.set_xticks(xticks, [f"Q{value + 1}" for value in xticks])
            axis.set_yticks(yticks, [f"Q{value + 1}" for value in yticks])
            panel = row * 4 + offset + col
            axis.text(-0.14, 1.07, ascii_uppercase[panel],
                      transform=axis.transAxes, fontweight="bold", fontsize=9,
                      va="top")
    for axis in axes.flat[2 * len(pairs):]:
        axis.set_visible(False)
    if image is not None:
        fig.colorbar(image, ax=axes, shrink=0.78, label="truth - V2.2 response")
    fig.suptitle(
        "Held-out two-dimensional conditional bias — "
        + ("all anchors" if population == "all" else "frozen dominant carrier"),
        fontsize=10,
    )
    save_figure(fig, stem)


def json_clean(value):
    if isinstance(value, dict):
        return {key: json_clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_clean(item) for item in value]
    if isinstance(value, tuple):
        return [json_clean(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def markdown(payload: dict) -> str:
    chosen = payload["model_selection"]["chosen_candidate"]
    full = payload["test_metrics"]["full"]
    calibration = payload["test_calibration"]
    lines = [
        "# V2.2 coherent-anchor bias emulator",
        "",
        "The target is `truth - V2.2`; positive means underprediction. Cases "
        "400-599 train, 600-699 tune capacity and select displayed coordinates, "
        "and 700-899 are the final test. Case is the uncertainty unit. Constgold "
        "is not opened and this model is diagnostic-only.",
        "",
        "## Held-out verdict",
        "",
        f"Diagnostic-quality gate: **{'PASS' if payload['model_good'] else 'FAIL'}**.",
        "",
        f"Chosen capacity: `{chosen}`. Full-model test MSE skill over a constant "
        f"is `{full['mse_skill']:.3%}`; its case-paired MSE reduction is "
        f"`{full['mse_reduction_model_minus_constant']['mean']:+.6g} +- "
        f"{full['mse_reduction_model_minus_constant']['case_sem']:.6g}`.",
        "",
        f"Across predicted-bias deciles, the measured top-minus-bottom residual is "
        f"`{calibration['observed_high_minus_low']['mean']:+.5f} +- "
        f"{calibration['observed_high_minus_low']['case_sem']:.5f}`; bin-mean "
        f"Spearman rho is `{calibration['bin_mean_spearman']:.3f}` and calibration "
        f"slope is `{calibration['bin_mean_calibration_slope']:.3f}`.",
        "",
        "Validation gates: " + ", ".join(
            f"`{name}={value}`" for name, value in payload["model_gates"].items()
        ) + ".",
        "",
        "## Feature-set ablation on final test",
        "",
        "| Feature set | Features | MSE skill | Case-mean RMSE | Global prediction | Global target |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, item in payload["test_metrics"].items():
        lines.append(
            f"| `{name}` | {len(payload['feature_sets'][name])} | "
            f"{item['mse_skill']:.3%} | {item['case_mean_rmse']:.5f} | "
            f"{item['global_prediction']['mean']:+.5f} | "
            f"{item['global_target']['mean']:+.5f} |"
        )
    lines.extend([
        "", "## Top validation-selected coordinates", "",
        "Permutation is within rendered case; positive values mean the coordinate "
        "contains out-of-case information beyond correlated alternatives left in "
        "place. Correlated features can split importance.",
        "",
        "| Feature | MSE increase | Percent of baseline | Case SEM |",
        "|---|---:|---:|---:|",
    ])
    for item in payload["permutation_importance"][:15]:
        lines.append(
            f"| `{item['feature']}` | {item['mean']:+.6g} | "
            f"{item['percent_of_baseline_mse']:+.4f}% | "
            f"{item['case_sem']:.3g} |"
        )
    lines.extend([
        "", "## Conditional diagnostics", "",
        "One-dimensional curves are measured-bin conditional means on final-test "
        "anchors, with case SEM on the direct residual and a bias-emulator line. "
        "All feature curves are saved in CSV; the figure coordinates were selected "
        "on the tuning block only.",
        "",
        "The two-dimensional maps use occupied quantile cells, not Cartesian "
        "partial-dependence extrapolation. Map pairs were fixed from tuning-block "
        "permutation importance before the final outcomes were summarized.",
        "",
        "Selected map pairs: " + ", ".join(
            f"`{x} x {y}`" for x, y in payload["selected_map_pairs"]
        ) + ".",
        "",
        "## Interpretation limits", "",
        "- The per-anchor response is heavy-tailed; small row-level MSE skill can "
        "still correspond to well-resolved conditional means.",
        "- Response-structure features are model outputs and identify where the "
        "bias lives, not a physical cause.",
        "- Physical variables are correlated. The 1D curves are marginal "
        "conditional means and the 2D maps condition only on their displayed pair.",
        "- This is not a fitted correction, deployment proposal, or constgold "
        "calibration.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", required=True)
    ap.add_argument("--eda-json", required=True)
    ap.add_argument("--train-max", type=int, default=599)
    ap.add_argument("--tune-max", type=int, default=699)
    ap.add_argument("--seed", type=int, default=7301)
    ap.add_argument("--curve-bins", type=int, default=12)
    ap.add_argument("--map-bins", type=int, default=8)
    ap.add_argument("--output-prefix", required=True)
    args = ap.parse_args()

    suffixes = [
        ".json", ".md", ".joblib", "_test_predictions.feather",
        "_curves.csv", "_maps.csv", "_calibration.pdf", "_calibration.png",
        "_importance.pdf", "_importance.png", "_1d_all.pdf", "_1d_all.png",
        "_1d_carrier.pdf", "_1d_carrier.png", "_2d_all.pdf", "_2d_all.png",
        "_2d_carrier.pdf", "_2d_carrier.png",
    ]
    for suffix in suffixes:
        path = args.output_prefix + suffix
        if os.path.exists(path):
            raise FileExistsError(f"refusing existing output {path}")
    with open(args.eda_json, encoding="utf-8") as handle:
        metadata = json.load(handle)
    feature_sets = metadata["feature_sets"]
    full_features = feature_sets["full"]
    physical_features = set(metadata["physical_features"])
    response_features = set(metadata["response_features"])
    labels = metadata["feature_labels"]
    frame = pd.read_feather(args.features, columns=[
        *KEY, TARGET, "carrier_ratio_gt5_positive",
        "carrier_top_fraction_gt0p7_positive", *full_features,
    ])
    if frame.duplicated(KEY).any():
        raise RuntimeError("duplicate feature-table key")
    values = frame[[TARGET, *full_features]].to_numpy(float)
    if np.isinf(values).any() or frame[TARGET].isna().any():
        raise RuntimeError("infinite feature or non-finite target")
    train = frame.loc[frame.case <= args.train_max].copy()
    tune = frame.loc[frame.case.between(args.train_max + 1, args.tune_max)].copy()
    test = frame.loc[frame.case > args.tune_max].copy()
    for name, local in (("train", train), ("tune", tune), ("test", test)):
        print(
            f"{name}: cases={local.case.min()}-{local.case.max()} "
            f"rows={len(local):,}", flush=True,
        )

    train_constant = case_balanced_mean(
        train[TARGET].to_numpy(float), train.case.to_numpy(np.int64)
    )
    tuning_results = {}
    candidate_models = {}
    for index, (name, parameters) in enumerate(CANDIDATES.items()):
        print(f"fit candidate {name}: {parameters}", flush=True)
        model = fit_model(
            train, full_features, parameters, args.seed + 100 * index
        )
        prediction = model.predict(tune[full_features].to_numpy(np.float32))
        metrics = metric_summary(tune, prediction, train_constant)
        tuning_results[name] = {
            "parameters": parameters, "metrics": metrics,
        }
        candidate_models[name] = model
        print(
            f"candidate {name}: MSE={metrics['case_balanced_mse']:.6g} "
            f"skill={metrics['mse_skill']:.4%}", flush=True,
        )
    chosen_name = min(
        tuning_results,
        key=lambda name: tuning_results[name]["metrics"]["case_balanced_mse"],
    )
    chosen_parameters = CANDIDATES[chosen_name]
    chosen_tune_model = candidate_models[chosen_name]
    tune_prediction = chosen_tune_model.predict(
        tune[full_features].to_numpy(np.float32)
    )
    importance = permutation_importance(
        chosen_tune_model, tune, full_features, args.seed + 991
    )
    importance_rank = [item["feature"] for item in importance]
    selected_physical = [
        feature for feature in importance_rank if feature in physical_features
    ][:3]
    selected_response = next(
        feature for feature in importance_rank if feature in response_features
    )
    if len(selected_physical) != 3:
        raise RuntimeError("could not select three physical map coordinates")
    selected_pairs = [
        (selected_physical[0], selected_physical[1]),
        (selected_physical[0], selected_physical[2]),
        (selected_physical[1], selected_physical[2]),
        *[(selected_response, feature) for feature in selected_physical],
    ]
    selected_curve_features = []
    for feature in [*importance_rank[:12], *CORE_PLOT_FEATURES]:
        if feature in full_features and feature not in selected_curve_features:
            selected_curve_features.append(feature)
        if len(selected_curve_features) == 16:
            break

    development = pd.concat([train, tune], ignore_index=True)
    development_constant = case_balanced_mean(
        development[TARGET].to_numpy(float),
        development.case.to_numpy(np.int64),
    )
    final_models = {}
    test_predictions = {}
    test_metrics = {}
    for index, (set_name, features) in enumerate(feature_sets.items()):
        print(
            f"fit final {set_name}: {len(features)} features, {chosen_name}",
            flush=True,
        )
        model = fit_model(
            development, features, chosen_parameters,
            args.seed + 1000 + 100 * index,
        )
        prediction = model.predict(test[features].to_numpy(np.float32))
        final_models[set_name] = model
        test_predictions[set_name] = prediction
        test_metrics[set_name] = metric_summary(
            test, prediction, development_constant
        )
        print(
            f"test {set_name}: skill={test_metrics[set_name]['mse_skill']:.4%} "
            f"global={test_metrics[set_name]['global_prediction']['mean']:+.5f}/"
            f"{test_metrics[set_name]['global_target']['mean']:+.5f}", flush=True,
        )

    prediction_edges = np.r_[
        -np.inf,
        np.unique(np.quantile(tune_prediction, np.arange(1, 10) / 10.0)),
        np.inf,
    ]
    calibration = prediction_calibration(
        test, test_predictions["full"], prediction_edges
    )
    full_metric = test_metrics["full"]
    mse_reduction = full_metric["mse_reduction_model_minus_constant"]
    global_error = full_metric["global_target_minus_prediction"]
    span = calibration["observed_high_minus_low"]
    gates = {
        "case_paired_mse_reduction_gt2sem": bool(
            mse_reduction["mean"] > 2.0 * mse_reduction["case_sem"]
        ),
        "observed_prediction_decile_span_gt3sem": bool(
            span["mean"] > 3.0 * span["case_sem"]
        ),
        "prediction_decile_rank_correlation_gt0p8": bool(
            calibration["bin_mean_spearman"] > 0.8
        ),
        "global_calibration_within_2_case_sem": bool(
            abs(global_error["mean"]) <= 2.0 * global_error["case_sem"]
        ),
        "positive_test_mse_skill": bool(full_metric["mse_skill"] > 0.0),
    }
    model_good = bool(all(gates.values()))
    print(f"model_good={model_good} gates={gates}", flush=True)

    curve_rows = []
    for population, mask_column in POPULATIONS.items():
        if mask_column is None:
            dev_local = development
            test_local = test
            pred_local = test_predictions["full"]
        else:
            dev_local = development.loc[development[mask_column]].copy()
            mask = test[mask_column].to_numpy(bool)
            test_local = test.loc[mask].copy()
            pred_local = test_predictions["full"][mask]
        for index, feature in enumerate(full_features):
            curve_rows.extend(conditional_curve(
                dev_local, test_local, pred_local, feature, population,
                args.curve_bins,
            ))
            if (index + 1) % 10 == 0:
                print(
                    f"curves {population}: {index + 1}/{len(full_features)}",
                    flush=True,
                )
    curves = pd.DataFrame(curve_rows)

    map_rows = []
    if model_good:
        for population, mask_column in POPULATIONS.items():
            if mask_column is None:
                dev_local = development
                test_local = test
                pred_local = test_predictions["full"]
            else:
                dev_local = development.loc[development[mask_column]].copy()
                mask = test[mask_column].to_numpy(bool)
                test_local = test.loc[mask].copy()
                pred_local = test_predictions["full"][mask]
            for x_feature, y_feature in selected_pairs:
                map_rows.extend(conditional_map(
                    dev_local, test_local, pred_local, x_feature, y_feature,
                    population, args.map_bins,
                ))
    maps = pd.DataFrame(map_rows)

    output_predictions = test[
        [*KEY, TARGET, "carrier_ratio_gt5_positive",
         "carrier_top_fraction_gt0p7_positive"]
    ].copy()
    for name, prediction in test_predictions.items():
        output_predictions[f"predicted_bias_{name}"] = prediction
    output_predictions.to_feather(args.output_prefix + "_test_predictions.feather")
    curves.to_csv(args.output_prefix + "_curves.csv", index=False)
    maps.to_csv(args.output_prefix + "_maps.csv", index=False)

    diagnostic_artifact = {
        "role": "diagnostic-only anchor bias emulator; not a V2.2 correction",
        "target": TARGET,
        "target_sign": "positive means truth exceeds V2.2",
        "models": final_models,
        "feature_sets": feature_sets,
        "chosen_candidate": chosen_name,
        "chosen_parameters": chosen_parameters,
        "case_windows": {"train": [400, args.train_max],
                         "tune": [args.train_max + 1, args.tune_max],
                         "test": [args.tune_max + 1, 899]},
        "constgold_opened": False,
    }
    joblib.dump(diagnostic_artifact, args.output_prefix + ".joblib", compress=3)

    plot_calibration(
        calibration, full_metric, args.output_prefix + "_calibration"
    )
    plot_importance(
        importance, physical_features, labels,
        args.output_prefix + "_importance",
    )
    plot_curves(
        curves, selected_curve_features, labels, "all",
        args.output_prefix + "_1d_all",
    )
    plot_curves(
        curves, selected_curve_features, labels,
        "carrier_ratio_gt5_positive", args.output_prefix + "_1d_carrier",
    )
    if model_good:
        plot_maps(
            maps, selected_pairs, labels, "all",
            args.output_prefix + "_2d_all",
        )
        plot_maps(
            maps, selected_pairs, labels, "carrier_ratio_gt5_positive",
            args.output_prefix + "_2d_carrier",
        )

    payload = {
        "design": (
            "anchor-only boosted-tree conditional-bias diagnostic; capacity and "
            "displayed coordinates selected on tuning cases; final outcomes only "
            "from test cases; equal total training weight per case"
        ),
        "target": TARGET,
        "target_definition": "R_blend_truth - R_blend_lsst_r_extnbr_v22",
        "target_sign": "positive means V2.2 underpredicts",
        "case_windows": {
            "train": [int(train.case.min()), int(train.case.max())],
            "tune": [int(tune.case.min()), int(tune.case.max())],
            "test": [int(test.case.min()), int(test.case.max())],
        },
        "rows": {
            "train": int(len(train)), "tune": int(len(tune)),
            "test": int(len(test)),
        },
        "feature_sets": feature_sets,
        "feature_labels": labels,
        "model_selection": {
            "criterion": "minimum case-balanced row MSE on tuning cases",
            "candidates": tuning_results,
            "chosen_candidate": chosen_name,
            "chosen_parameters": chosen_parameters,
            "training_constant": train_constant,
            "development_constant": development_constant,
        },
        "test_metrics": test_metrics,
        "test_calibration": calibration,
        "model_gates": gates,
        "model_good": model_good,
        "permutation_importance": importance,
        "permutation_definition": (
            "one deterministic within-case permutation per feature on tuning cases"
        ),
        "selected_curve_features": selected_curve_features,
        "selected_physical_map_features": selected_physical,
        "selected_response_map_feature": selected_response,
        "selected_map_pairs": selected_pairs,
        "curve_definition": (
            "development-covariate quantile edges applied to test rows; target and "
            "prediction averaged within case before uncertainty"
        ),
        "map_definition": (
            "occupied observed-distribution quantile cells; not Cartesian partial "
            "dependence; development covariates set edges"
        ),
        "artifacts": {
            "diagnostic_model": os.path.abspath(args.output_prefix + ".joblib"),
            "test_predictions": os.path.abspath(
                args.output_prefix + "_test_predictions.feather"
            ),
            "curves_csv": os.path.abspath(args.output_prefix + "_curves.csv"),
            "maps_csv": os.path.abspath(args.output_prefix + "_maps.csv"),
        },
        "constgold_opened": False,
        "deployable_correction_written": False,
    }
    payload = json_clean(payload)
    with open(args.output_prefix + ".json", "x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    with open(args.output_prefix + ".md", "x", encoding="utf-8") as handle:
        handle.write(markdown(payload))
    print(markdown(payload), flush=True)
    print("ANCHOR_BIAS_EMULATOR_DONE", flush=True)


if __name__ == "__main__":
    main()
